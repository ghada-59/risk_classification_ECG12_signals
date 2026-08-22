"""Dataset loading utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import DatasetConfig


logger = logging.getLogger(__name__)


def load_dataset(cfg: DatasetConfig) -> pd.DataFrame:
    """Load a dataset described by ``cfg`` as a DataFrame.

    Datetime parsing is delegated to the cleaner so that the raw load
    stage stays cheap and reusable.
    """
    if not Path(cfg.path).exists():
        raise FileNotFoundError(f"Dataset file not found: {cfg.path}")

    logger.info("Loading %s from %s", cfg.name, cfg.path)
    df = pd.read_csv(cfg.path, low_memory=False)
    logger.info("Loaded %s: %d rows x %d cols", cfg.name, *df.shape)
    return df


def file_size_mb(path: Path) -> float:
    return round(Path(path).stat().st_size / (1024 * 1024), 2)
