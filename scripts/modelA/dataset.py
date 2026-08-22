"""PyTorch ``Dataset`` for PTB-XL signals + Modèle A 3-class labels.

Key design choices
------------------
* **Patient-independent splits** via PTB-XL's pre-defined ``strat_fold``
  column (folds 1-8 train, 9 val, 10 test).
* **Offline cache** of preprocessed signals (``signals_cache.npy``) and
  HRV/morphology features (``features_cache.npz``). The cache is built
  in parallel via ``multiprocessing.Pool`` with periodic chunk saves so
  long runs can resume without losing progress.
* **Augmentations** (train mode only): random time shift ±50 samples
  and per-lead dropout (one lead masked to zero with probability 0.1).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import (
    MODELS_DIR,
    N_LEADS,
    PTBXL_CSV,
    PTBXL_DIR,
    SAMPLING_RATE_HZ,
    SIGNAL_LENGTH,
    TRAINING,
)
from .feature_extraction import N_FEATURES, extract_features
from .label_mapping import assign_class_from_raw
from .signal_preprocessing import preprocess


logger = logging.getLogger(__name__)


FEATURES_CACHE = MODELS_DIR / "features_cache.npz"
SIGNALS_CACHE = MODELS_DIR / "signals_cache.npy"
CACHE_PROGRESS = MODELS_DIR / "cache_progress.npy"


def load_ptbxl_index(
    csv_path: Path = PTBXL_CSV,
    require_signals: bool = True,
) -> pd.DataFrame:
    """Return the PTB-XL index with the 3-class label attached.

    If ``require_signals`` is True, only rows whose corresponding
    ``records100/...lr.dat`` file is present on disk are kept.
    """
    df = pd.read_csv(csv_path)
    df["label"] = df["scp_codes"].map(assign_class_from_raw)
    if require_signals:
        present = df["filename_lr"].map(
            lambda f: (PTBXL_DIR / f"{f}.dat").exists()
        )
        df = df[present].reset_index(drop=True)
        logger.info("Index filtered to %d records with available signals", len(df))
    return df


@lru_cache(maxsize=4096)
def _read_signal(filename_lr: str) -> np.ndarray:
    """Read ``records100/.../XXXXX_lr`` via wfdb. Returns ``(12, T)`` float32."""
    import wfdb

    path = str(PTBXL_DIR / filename_lr)
    record = wfdb.rdrecord(path)
    sig = np.asarray(record.p_signal, dtype=np.float32)
    if sig.ndim != 2:
        raise ValueError(f"Unexpected signal shape from {filename_lr}: {sig.shape}")
    if sig.shape[1] != N_LEADS:
        if sig.shape[0] == N_LEADS:
            return sig
        raise ValueError(
            f"Expected 12 leads, got shape {sig.shape} from {filename_lr}"
        )
    return sig.T


def _pad_or_crop(signal: np.ndarray, length: int = SIGNAL_LENGTH) -> np.ndarray:
    if signal.shape[1] == length:
        return signal
    if signal.shape[1] > length:
        return signal[:, :length]
    pad = np.zeros((signal.shape[0], length - signal.shape[1]), dtype=signal.dtype)
    return np.concatenate([signal, pad], axis=1)


def _process_one(filename_lr: str) -> tuple[np.ndarray, np.ndarray, bool]:
    """Read + preprocess + feature-extract one record.

    Designed to run in a worker process so the GIL doesn't serialize the
    expensive neurokit2 call. Returns ``(sig[12,1000], feats[12], ok)``.
    """
    try:
        import wfdb

        record = wfdb.rdrecord(str(PTBXL_DIR / filename_lr))
        sig = np.asarray(record.p_signal, dtype=np.float32)
        if sig.ndim != 2:
            return (
                np.zeros((N_LEADS, SIGNAL_LENGTH), dtype=np.float32),
                np.zeros(N_FEATURES, dtype=np.float32),
                False,
            )
        if sig.shape[1] == N_LEADS:
            sig = sig.T
        proc = preprocess(sig)
        if proc.shape[1] != SIGNAL_LENGTH:
            proc = _pad_or_crop(proc, SIGNAL_LENGTH)
        feats = extract_features(proc, SAMPLING_RATE_HZ).as_array()
        return proc, feats, True
    except Exception:
        return (
            np.zeros((N_LEADS, SIGNAL_LENGTH), dtype=np.float32),
            np.zeros(N_FEATURES, dtype=np.float32),
            False,
        )


def _save_partial(
    sigs: np.ndarray, feats: np.ndarray, ids: np.ndarray, progress: np.ndarray
) -> None:
    np.save(SIGNALS_CACHE, sigs)
    np.savez_compressed(FEATURES_CACHE, ecg_ids=ids, features=feats)
    np.save(CACHE_PROGRESS, progress)


def precompute_cache(
    df: pd.DataFrame,
    signals_path: Path = SIGNALS_CACHE,
    features_path: Path = FEATURES_CACHE,
    force: bool = False,
    workers: int | None = None,
    save_every: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (or resume) the signal + feature cache for ``df``.

    Uses ``ProcessPoolExecutor`` to parallelize the slow per-record
    preprocessing + neurokit2 feature extraction. Partial results are
    persisted every ``save_every`` records so an interrupted run resumes
    near the cut-off point.

    Returns
    -------
    (signals, features):
        Float32 arrays of shapes ``(N, 12, 1000)`` and ``(N, 12)``.
    """
    signals_path.parent.mkdir(parents=True, exist_ok=True)
    ids = df["ecg_id"].to_numpy()
    n = len(df)
    filenames = df["filename_lr"].astype(str).tolist()

    fully_cached = (
        not force
        and signals_path.exists()
        and features_path.exists()
        and not CACHE_PROGRESS.exists()
    )
    if fully_cached:
        cached_feats = np.load(features_path)
        if np.array_equal(cached_feats["ecg_ids"], ids):
            sigs = np.load(signals_path)
            if sigs.shape == (n, N_LEADS, SIGNAL_LENGTH):
                logger.info("Loaded cached signals + features for %d records", n)
                return sigs, cached_feats["features"]

    sigs = np.zeros((n, N_LEADS, SIGNAL_LENGTH), dtype=np.float32)
    feats = np.zeros((n, N_FEATURES), dtype=np.float32)
    done = np.zeros(n, dtype=bool)

    if (
        not force
        and signals_path.exists()
        and features_path.exists()
        and CACHE_PROGRESS.exists()
    ):
        cached_feats = np.load(features_path)
        if np.array_equal(cached_feats["ecg_ids"], ids):
            sigs = np.load(signals_path)
            feats = cached_feats["features"]
            done = np.load(CACHE_PROGRESS)
            if done.shape == (n,):
                logger.info(
                    "Resuming cache build — %d/%d already cached",
                    int(done.sum()), n,
                )
            else:
                done = np.zeros(n, dtype=bool)

    todo_idx = np.flatnonzero(~done)
    if todo_idx.size == 0:
        logger.info("Cache already complete (%d records)", n)
        if CACHE_PROGRESS.exists():
            CACHE_PROGRESS.unlink()
        return sigs, feats

    n_workers = workers or max(1, min(8, (os.cpu_count() or 4) - 1))
    logger.info(
        "Precomputing cache for %d records (%d remaining, %d workers)",
        n, todo_idx.size, n_workers,
    )

    last_save = 0
    completed_in_run = 0
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_process_one, filenames[int(i)]): int(i)
                for i in todo_idx
            }
            bar = tqdm(as_completed(futures), total=len(futures),
                       desc="cache", unit="rec", smoothing=0.1)
            for fut in bar:
                i = futures[fut]
                sig, ft, ok = fut.result()
                sigs[i] = sig
                feats[i] = ft
                done[i] = True
                completed_in_run += 1
                if completed_in_run - last_save >= save_every:
                    _save_partial(sigs, feats, ids, done)
                    last_save = completed_in_run
    except KeyboardInterrupt:
        logger.warning("Interrupted — saving partial cache")
        _save_partial(sigs, feats, ids, done)
        raise

    np.save(signals_path, sigs)
    np.savez_compressed(features_path, ecg_ids=ids, features=feats)
    if CACHE_PROGRESS.exists():
        CACHE_PROGRESS.unlink()
    logger.info(
        "Saved cache: %s (%.1f MB) + %s",
        signals_path,
        signals_path.stat().st_size / 1e6,
        features_path,
    )
    return sigs, feats


def precompute_features(
    df: pd.DataFrame, cache_path: Path = FEATURES_CACHE, force: bool = False
) -> np.ndarray:
    """Backwards-compatible wrapper that returns only the feature matrix."""
    _, feats = precompute_cache(df, force=force)
    return feats


class PTBXLDataset(Dataset):
    """PyTorch ``Dataset`` over a subset of PTB-XL.

    Reads from the precomputed signal cache (memmap) so epoch IO is
    near-instant. Augmentations are applied on a copy of the cached signal
    only on the train split.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        signals: np.ndarray,
        features: np.ndarray,
        augment: bool = False,
    ) -> None:
        if len(df) != len(features) or len(df) != len(signals):
            raise ValueError("df / signals / features have different lengths")
        self.df = df.reset_index(drop=True)
        self.signals = signals
        self.features = features.astype(np.float32, copy=False)
        self.labels = self.df["label"].to_numpy(dtype=np.int64)
        self.augment = augment
        self._rng = np.random.default_rng(TRAINING.seed)

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, sig: np.ndarray) -> np.ndarray:
        shift = int(self._rng.integers(-TRAINING.aug_time_shift, TRAINING.aug_time_shift + 1))
        if shift != 0:
            sig = np.roll(sig, shift, axis=-1)
            if shift > 0:
                sig[:, :shift] = 0.0
            else:
                sig[:, shift:] = 0.0
        if self._rng.random() < TRAINING.aug_lead_dropout_p:
            lead = int(self._rng.integers(0, sig.shape[0]))
            sig[lead, :] = 0.0
        return sig

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sig = np.array(self.signals[idx], dtype=np.float32, copy=True)
        if self.augment:
            sig = self._augment(sig)
        return (
            torch.from_numpy(sig),
            torch.from_numpy(self.features[idx]),
            torch.tensor(int(self.labels[idx]), dtype=torch.long),
        )


def build_split_datasets(
    csv_path: Path = PTBXL_CSV,
    train_folds: Sequence[int] = TRAINING.train_folds,
    val_folds: Sequence[int] = TRAINING.val_folds,
    test_folds: Sequence[int] = TRAINING.test_folds,
    require_signals: bool = True,
) -> tuple[PTBXLDataset, PTBXLDataset, PTBXLDataset, pd.DataFrame]:
    """Return ``(train, val, test, full_df)`` ready for DataLoaders.

    Signals + features are cached on first call to ``MODELS_DIR / *cache*``.
    """
    df = load_ptbxl_index(csv_path, require_signals=require_signals)
    signals, features = precompute_cache(df)

    def _subset(folds: Iterable[int]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        mask = df["strat_fold"].isin(list(folds)).to_numpy()
        idx = np.flatnonzero(mask)
        return df.iloc[idx].reset_index(drop=True), signals[idx], features[idx]

    df_train, s_train, f_train = _subset(train_folds)
    df_val, s_val, f_val = _subset(val_folds)
    df_test, s_test, f_test = _subset(test_folds)

    logger.info("Split sizes — train=%d val=%d test=%d",
                len(df_train), len(df_val), len(df_test))
    return (
        PTBXLDataset(df_train, s_train, f_train, augment=True),
        PTBXLDataset(df_val, s_val, f_val, augment=False),
        PTBXLDataset(df_test, s_test, f_test, augment=False),
        df,
    )


def class_weights_from_df(df: pd.DataFrame, n_classes: int = 3) -> torch.Tensor:
    """Return inverse-frequency class weights (normalized to mean 1)."""
    counts = np.bincount(df["label"].to_numpy(), minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = counts.sum() / (n_classes * counts)
    return torch.tensor(inv, dtype=torch.float32)
