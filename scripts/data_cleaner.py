"""Dataset cleaning and normalization.

Cleaning is intentionally conservative: we never drop rows by default
because the EDA needs to *characterize* quality issues, not hide them.
The cleaned frame keeps the same row count; only types and obvious
formatting issues are fixed.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import DatasetConfig


logger = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    """Summary of what the cleaner changed."""

    n_rows: int = 0
    n_cols: int = 0
    duplicate_rows: int = 0
    duplicate_ids: dict[str, int] = field(default_factory=dict)
    parsed_list_columns: list[str] = field(default_factory=list)
    parsed_dict_columns: list[str] = field(default_factory=list)
    parsed_datetime_columns: list[str] = field(default_factory=list)
    coerced_boolean_columns: list[str] = field(default_factory=list)
    columns_all_null: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)


def _safe_literal_eval(value: Any) -> Any:
    """Best-effort ``ast.literal_eval`` that never raises."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def clean_dataset(
    df: pd.DataFrame, cfg: DatasetConfig
) -> tuple[pd.DataFrame, CleaningReport]:
    """Return a cleaned copy of ``df`` plus a report of what was changed."""
    report = CleaningReport(n_rows=len(df), n_cols=df.shape[1])
    cleaned = df.copy()

    # Strip whitespace from object columns (handles stray leading commas etc.).
    for col in cleaned.select_dtypes(include="object").columns:
        cleaned[col] = cleaned[col].astype(str).where(cleaned[col].notna(), np.nan)
        mask = cleaned[col].notna()
        cleaned.loc[mask, col] = cleaned.loc[mask, col].str.strip()
        cleaned[col] = cleaned[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    for col in cfg.list_columns:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].map(_safe_literal_eval)
            report.parsed_list_columns.append(col)

    for col in cfg.dict_columns:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].map(_safe_literal_eval)
            report.parsed_dict_columns.append(col)

    for col in cfg.datetime_columns:
        if col in cleaned.columns:
            cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")
            report.parsed_datetime_columns.append(col)

    for col in cfg.boolean_columns:
        if col in cleaned.columns and cleaned[col].dtype != bool:
            cleaned[col] = cleaned[col].astype("boolean")
            report.coerced_boolean_columns.append(col)

    unhashable_cols = list(cfg.list_columns) + list(cfg.dict_columns)
    dup_subset = [c for c in cleaned.columns if c not in unhashable_cols]
    report.duplicate_rows = (
        int(cleaned.duplicated(subset=dup_subset).sum()) if dup_subset else 0
    )
    for id_col in cfg.id_columns:
        if id_col in cleaned.columns and id_col not in unhashable_cols:
            dup = int(cleaned[id_col].duplicated().sum())
            if dup:
                report.duplicate_ids[id_col] = dup

    report.columns_all_null = [c for c in cleaned.columns if cleaned[c].isna().all()]
    constant_cols: list[str] = []
    for c in cleaned.columns:
        if c in report.columns_all_null or c in unhashable_cols:
            continue
        try:
            if cleaned[c].nunique(dropna=True) <= 1:
                constant_cols.append(c)
        except TypeError:
            continue
    report.constant_columns = constant_cols

    logger.info(
        "Cleaned %s: %d duplicate rows, %d all-null cols, %d constant cols",
        cfg.name,
        report.duplicate_rows,
        len(report.columns_all_null),
        len(report.constant_columns),
    )
    return cleaned, report


def save_processed(df: pd.DataFrame, cfg: DatasetConfig, processed_dir) -> str:
    """Persist the cleaned dataset to ``processed_data/``.

    List/dict columns are stringified before writing so the CSV stays
    portable. Returns the output path as a string.
    """
    out = df.copy()
    list_like = list(cfg.list_columns) + list(cfg.dict_columns)
    for col in list_like:
        if col in out.columns:
            out[col] = out[col].map(
                lambda v: "" if isinstance(v, float) and np.isnan(v) else repr(v)
            )
    out_path = processed_dir / f"{cfg.name}_cleaned.csv"
    out.to_csv(out_path, index=False)
    logger.info("Saved cleaned dataset to %s", out_path)
    return str(out_path)
