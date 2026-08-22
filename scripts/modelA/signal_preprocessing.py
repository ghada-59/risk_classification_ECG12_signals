"""ECG signal preprocessing for Modèle A.

Pipeline:
    raw 12×T ECG (mV)
    → bandpass Butterworth (0.5–40 Hz, order 4)
    → notch 50 Hz (powerline, Europe)
    → per-lead z-score normalization
    → clip ±5σ to absorb residual artefacts

All operations are pure numpy/scipy and vectorized across leads, so a
12×1000 sample window processes in well under 5 ms on CPU.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from .config import PREPROCESSING, SAMPLING_RATE_HZ


logger = logging.getLogger(__name__)


def _bandpass_coeffs(
    low: float, high: float, fs: int, order: int
) -> tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    return butter(order, [low / nyq, high / nyq], btype="bandpass")


def _notch_coeffs(freq: float, q: float, fs: int) -> tuple[np.ndarray, np.ndarray]:
    return iirnotch(freq, q, fs)


_BP_B, _BP_A = _bandpass_coeffs(
    PREPROCESSING.bandpass_low_hz,
    PREPROCESSING.bandpass_high_hz,
    SAMPLING_RATE_HZ,
    PREPROCESSING.bandpass_order,
)
_NOTCH_B, _NOTCH_A = _notch_coeffs(
    PREPROCESSING.notch_freq_hz, PREPROCESSING.notch_quality, SAMPLING_RATE_HZ
)


def bandpass_filter(signal: np.ndarray) -> np.ndarray:
    """Zero-phase Butterworth bandpass.

    Parameters
    ----------
    signal:
        Array of shape ``(n_leads, n_samples)`` or ``(n_samples,)``.
    """
    return filtfilt(_BP_B, _BP_A, signal, axis=-1).astype(np.float32, copy=False)


def notch_filter(signal: np.ndarray) -> np.ndarray:
    """Zero-phase 50 Hz powerline notch."""
    return filtfilt(_NOTCH_B, _NOTCH_A, signal, axis=-1).astype(np.float32, copy=False)


def zscore_normalize(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-lead z-score on ``(n_leads, n_samples)``."""
    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True)
    return ((signal - mean) / (std + eps)).astype(np.float32, copy=False)


def clip_outliers(signal: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Clip to ±``sigma`` after z-score (sigma=5 by default)."""
    s = PREPROCESSING.clip_sigma if sigma is None else sigma
    return np.clip(signal, -s, s).astype(np.float32, copy=False)


def preprocess(signal: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline.

    Parameters
    ----------
    signal:
        Either ``(n_leads, n_samples)`` or ``(n_samples, n_leads)`` —
        we auto-transpose to ``(n_leads, n_samples)`` based on shape.

    Returns
    -------
    np.ndarray
        ``float32`` of shape ``(n_leads, n_samples)``.
    """
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D signal, got shape {arr.shape}")
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    arr = bandpass_filter(arr)
    arr = notch_filter(arr)
    arr = zscore_normalize(arr)
    arr = clip_outliers(arr)
    return arr


def _self_test() -> None:
    rng = np.random.default_rng(0)
    sig = rng.normal(size=(12, 1000)).astype(np.float32)
    sig += 0.3 * np.sin(2 * np.pi * 50 * np.arange(1000) / SAMPLING_RATE_HZ)
    sig += np.linspace(0, 2, 1000)
    out = preprocess(sig)
    assert out.shape == (12, 1000)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert np.abs(out).max() <= PREPROCESSING.clip_sigma + 1e-3
    print(
        f"preprocess OK  shape={out.shape}  mean={out.mean():.3f}  "
        f"std={out.std():.3f}  max|x|={np.abs(out).max():.3f}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _self_test()
