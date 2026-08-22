"""Single-window inference for Modèle A + latency benchmark.

The ``ECGPredictor`` is the API used by the FastAPI server: it loads the
best checkpoint once, runs the full preprocess → features → model
pipeline on a ``(12, 1000)`` signal, and returns the prediction.

Run as a script for a latency benchmark:

    python -m scripts.modelA.inference --benchmark 100
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .config import (
    CLASS_NAMES,
    INFERENCE,
    MODEL,
    N_LEADS,
    SAMPLING_RATE_HZ,
    SIGNAL_LENGTH,
)
from .feature_extraction import extract_features
from .model import ModelA
from .signal_preprocessing import preprocess


logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    label_id: int
    label: str
    probabilities: dict[str, float]
    risk_score: float
    timestamp: str
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


class ECGPredictor:
    """Stateless wrapper around the trained Modèle A.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``modelA_best.pt`` produced by ``train.py``.
    device:
        ``"cpu"`` or ``"cuda"`` (default cpu).
    """

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path or INFERENCE.checkpoint_path)
        self.device = device or INFERENCE.device
        self.model = ModelA(MODEL).to(self.device)
        self._loaded = False
        self._load()

    def _load(self) -> None:
        if not self.checkpoint_path.exists():
            logger.warning(
                "Checkpoint missing at %s — using random weights "
                "(training has not run yet).",
                self.checkpoint_path,
            )
            self.model.eval()
            return
        ckpt = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self._loaded = True
        logger.info(
            "Loaded checkpoint epoch=%s val_auc=%.4f",
            ckpt.get("epoch", "?"), ckpt.get("val_auc", float("nan")),
        )

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _prepare(self, signal: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        arr = np.asarray(signal, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D signal, got shape {arr.shape}")
        if arr.shape[0] != N_LEADS and arr.shape[1] == N_LEADS:
            arr = arr.T
        if arr.shape[0] != N_LEADS:
            raise ValueError(f"Need {N_LEADS} leads, got {arr.shape[0]}")
        proc = preprocess(arr)
        if proc.shape[1] != SIGNAL_LENGTH:
            if proc.shape[1] > SIGNAL_LENGTH:
                proc = proc[:, :SIGNAL_LENGTH]
            else:
                pad = np.zeros(
                    (N_LEADS, SIGNAL_LENGTH - proc.shape[1]), dtype=np.float32
                )
                proc = np.concatenate([proc, pad], axis=1)
        feats = extract_features(proc, SAMPLING_RATE_HZ).as_array()
        return (
            torch.from_numpy(proc).unsqueeze(0).to(self.device),
            torch.from_numpy(feats).unsqueeze(0).to(self.device),
        )

    @torch.no_grad()
    def predict(self, signal: np.ndarray) -> Prediction:
        """Run inference on a single ``(12, 1000)`` window."""
        t0 = time.perf_counter()
        sig_t, feat_t = self._prepare(signal)
        logits = self.model(sig_t, feat_t)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        risk = self.model.risk_score(probs).item()
        label_id = int(torch.argmax(probs).item())
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return Prediction(
            label_id=label_id,
            label=CLASS_NAMES[label_id],
            probabilities={n: round(float(probs[i]), 6) for i, n in enumerate(CLASS_NAMES)},
            risk_score=round(risk, 6),
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            latency_ms=round(elapsed_ms, 3),
        )

    @torch.no_grad()
    def predict_from_features(self, features: np.ndarray) -> Prediction:
        """Run inference from a pre-computed 12-dim feature vector.

        The signal branch receives a zero tensor so only the auxiliary
        feature branch carries meaningful information.
        """
        feat = np.asarray(features, dtype=np.float32).flatten()
        if feat.shape != (12,):
            raise ValueError(f"Expected 12 features, got shape {feat.shape}")
        t0 = time.perf_counter()
        zero_signal = np.zeros((1, N_LEADS, SIGNAL_LENGTH), dtype=np.float32)
        sig_t = torch.from_numpy(zero_signal).to(self.device)
        feat_t = torch.from_numpy(feat).unsqueeze(0).to(self.device)
        logits = self.model(sig_t, feat_t)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        risk = self.model.risk_score(probs).item()
        label_id = int(torch.argmax(probs).item())
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return Prediction(
            label_id=label_id,
            label=CLASS_NAMES[label_id],
            probabilities={n: round(float(probs[i]), 6) for i, n in enumerate(CLASS_NAMES)},
            risk_score=round(risk, 6),
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            latency_ms=round(elapsed_ms, 3),
        )


def benchmark(n: int = 100, device: str = "cpu") -> dict:
    """Measure inference latency on synthetic data — reports p50/p95/p99."""
    predictor = ECGPredictor(device=device)
    rng = np.random.default_rng(0)
    latencies: list[float] = []
    for _ in range(3):
        predictor.predict(rng.normal(size=(12, 1000)).astype(np.float32))
    for _ in range(n):
        sig = rng.normal(size=(12, 1000)).astype(np.float32)
        pred = predictor.predict(sig)
        latencies.append(pred.latency_ms)
    arr = np.array(latencies)
    return {
        "n": int(n),
        "device": device,
        "checkpoint_loaded": predictor.loaded,
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "mean_ms": round(float(arr.mean()), 3),
        "max_ms": round(float(arr.max()), 3),
        "target_ms": INFERENCE.latency_target_ms,
        "passes_target": bool(np.percentile(arr, 95) < INFERENCE.latency_target_ms),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Modèle A inference / benchmark.")
    parser.add_argument("--benchmark", type=int, default=0,
                        help="If > 0, run N synthetic inferences and report latency.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None,
                        help="Optional JSON output path for benchmark results.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    if args.benchmark > 0:
        result = benchmark(args.benchmark, device=args.device)
        import json
        out = json.dumps(result, indent=2)
        print(out)
        if args.output:
            Path(args.output).write_text(out, encoding="utf-8")
        return 0

    predictor = ECGPredictor(device=args.device)
    rng = np.random.default_rng(0)
    sig = rng.normal(size=(12, 1000)).astype(np.float32)
    pred = predictor.predict(sig)
    print(pred.to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
