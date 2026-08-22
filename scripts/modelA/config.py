"""Configuration for Modèle A (paths, hyper-parameters, SCP mapping)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PTBXL_DIR = DATA_DIR / "ptbxl"
PTBXL_CSV = DATA_DIR / "ptbxl_database.csv"

MODELS_DIR = PROJECT_ROOT / "models" / "modelA"
REPORTS_DIR = PROJECT_ROOT / "reports" / "modelA"
VIS_DIR = PROJECT_ROOT / "visualizations" / "modelA"

for _p in (MODELS_DIR, REPORTS_DIR, VIS_DIR, PTBXL_DIR):
    _p.mkdir(parents=True, exist_ok=True)


SAMPLING_RATE_HZ = 100
N_LEADS = 12
SIGNAL_LENGTH = 1000
LEAD_NAMES = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)

CLASS_NAMES = ("normal", "suspect", "critique")
N_CLASSES = 3
CRITICAL_CLASS = 2
SUSPECT_CLASS = 1
NORMAL_CLASS = 0


@dataclass(frozen=True)
class PreprocessingConfig:
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    bandpass_order: int = 4
    notch_freq_hz: float = 50.0
    notch_quality: float = 30.0
    clip_sigma: float = 5.0


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = N_LEADS
    cnn_channels: tuple[int, ...] = (64, 128, 256)
    cnn_kernels: tuple[int, ...] = (7, 5, 3)
    cnn_pool: int = 2
    lstm_hidden: int = 128
    lstm_layers: int = 1
    lstm_bidirectional: bool = True
    aux_in: int = 12
    aux_hidden: int = 32
    fc_hidden: int = 128
    dropout: float = 0.3
    n_classes: int = N_CLASSES


@dataclass(frozen=True)
class TrainingConfig:
    train_folds: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    val_folds: tuple[int, ...] = (9,)
    test_folds: tuple[int, ...] = (10,)
    batch_size: int = 64
    num_workers: int = 0
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    focal_gamma: float = 2.0
    early_stop_patience: int = 6
    grad_clip: float = 1.0
    seed: int = 42
    aug_time_shift: int = 50
    aug_lead_dropout_p: float = 0.1


@dataclass(frozen=True)
class InferenceConfig:
    checkpoint_path: Path = MODELS_DIR / "modelA_best.pt"
    device: str = "cpu"
    latency_target_ms: int = 3000


PREPROCESSING = PreprocessingConfig()
MODEL = ModelConfig()
TRAINING = TrainingConfig()
INFERENCE = InferenceConfig()
