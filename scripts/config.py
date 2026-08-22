"""Central configuration for the EDA pipeline.

Each dataset is described by a `DatasetConfig` entry. Adding a new dataset
is as simple as appending a new entry to ``DATASETS``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
REPORTS_DIR = PROJECT_ROOT / "reports"
VIS_DIR = PROJECT_ROOT / "visualizations"
PROCESSED_DIR = PROJECT_ROOT / "processed_data"


@dataclass(frozen=True)
class DatasetConfig:
    """Declarative description of a dataset.

    Attributes:
        name: Folder-safe short name (used to build output paths).
        path: Absolute path to the CSV file.
        description: Short human-readable description.
        list_columns: Columns whose string values look like Python lists
            (``"['A', 'B']"``) and need to be parsed.
        dict_columns: Columns whose string values look like Python dicts.
        datetime_columns: Columns to parse as datetimes.
        boolean_columns: Explicit boolean columns (auto-detected if empty).
        id_columns: Columns identifying a row (excluded from numeric stats).
        label_columns: Candidate label/target columns (used for class balance).
        max_categories: Cap on categories shown in bar charts.
    """

    name: str
    path: Path
    description: str
    list_columns: Sequence[str] = field(default_factory=tuple)
    dict_columns: Sequence[str] = field(default_factory=tuple)
    datetime_columns: Sequence[str] = field(default_factory=tuple)
    boolean_columns: Sequence[str] = field(default_factory=tuple)
    id_columns: Sequence[str] = field(default_factory=tuple)
    label_columns: Sequence[str] = field(default_factory=tuple)
    max_categories: int = 25


DATASETS: tuple[DatasetConfig, ...] = (
    DatasetConfig(
        name="ptbxl_database",
        path=DATA_DIR / "ptbxl_database.csv",
        description=(
            "PTB-XL ECG database metadata: demographics, recording info, "
            "SCP-ECG diagnostic codes, signal quality annotations and "
            "stratification folds for ~22k 12-lead ECG recordings."
        ),
        dict_columns=("scp_codes",),
        datetime_columns=("recording_date",),
        id_columns=("ecg_id", "patient_id", "filename_lr", "filename_hr"),
        label_columns=("scp_codes", "heart_axis", "sex"),
    ),
)


def dataset_output_dirs(name: str) -> dict[str, Path]:
    """Return (and create) the per-dataset output directories."""
    dirs = {
        "vis": VIS_DIR / name,
        "report": REPORTS_DIR / name,
        "processed": PROCESSED_DIR,
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs
