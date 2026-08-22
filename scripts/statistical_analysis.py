"""Statistical summaries for an EDA."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DatasetConfig


logger = logging.getLogger(__name__)


@dataclass
class StatisticalSummary:
    overview: dict = field(default_factory=dict)
    missing: pd.DataFrame = field(default_factory=pd.DataFrame)
    dtypes: pd.DataFrame = field(default_factory=pd.DataFrame)
    numeric_describe: pd.DataFrame = field(default_factory=pd.DataFrame)
    categorical_describe: pd.DataFrame = field(default_factory=pd.DataFrame)
    datetime_describe: pd.DataFrame = field(default_factory=pd.DataFrame)
    skewness: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    kurtosis: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    outlier_counts: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlation: pd.DataFrame = field(default_factory=pd.DataFrame)


def _iqr_outlier_count(series: pd.Series) -> int:
    s = series.dropna()
    if len(s) < 4:
        return 0
    q1, q3 = np.nanpercentile(s, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return 0
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < low) | (s > high)).sum())


def compute_statistics(df: pd.DataFrame, cfg: DatasetConfig) -> StatisticalSummary:
    summary = StatisticalSummary()

    unhashable_cols = list(cfg.list_columns) + list(cfg.dict_columns)
    dup_subset = [c for c in df.columns if c not in unhashable_cols]
    duplicate_rows = (
        int(df.duplicated(subset=dup_subset).sum()) if dup_subset else 0
    )

    summary.overview = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "memory_MB": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
        "total_cells": int(df.size),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": round(float(df.isna().mean().mean()) * 100, 2),
        "duplicate_rows": duplicate_rows,
    }

    miss = df.isna().sum()
    summary.missing = (
        pd.DataFrame(
            {
                "missing_count": miss,
                "missing_pct": (miss / len(df) * 100).round(2),
                "dtype": df.dtypes.astype(str),
            }
        )
        .sort_values("missing_count", ascending=False)
    )

    summary.dtypes = (
        df.dtypes.astype(str)
        .value_counts()
        .rename_axis("dtype")
        .reset_index(name="n_columns")
    )

    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in cfg.id_columns
    ]
    if numeric_cols:
        summary.numeric_describe = df[numeric_cols].describe().T.round(4)
        summary.skewness = df[numeric_cols].skew(numeric_only=True).round(4)
        summary.kurtosis = df[numeric_cols].kurt(numeric_only=True).round(4)
        summary.outlier_counts = pd.DataFrame(
            {
                "outliers_iqr": {c: _iqr_outlier_count(df[c]) for c in numeric_cols},
                "n_non_null": {c: int(df[c].notna().sum()) for c in numeric_cols},
            }
        )
        summary.outlier_counts["outlier_pct"] = (
            summary.outlier_counts["outliers_iqr"]
            / summary.outlier_counts["n_non_null"].replace(0, np.nan)
            * 100
        ).round(2)

        if len(numeric_cols) >= 2:
            summary.correlation = df[numeric_cols].corr(numeric_only=True).round(3)

    cat_cols = [
        c
        for c in df.select_dtypes(include=["object", "boolean", "bool", "category"]).columns
        if c not in cfg.id_columns and c not in unhashable_cols
    ]
    if cat_cols:
        rows = []
        for c in cat_cols:
            s = df[c]
            try:
                top = s.mode(dropna=True)
                top_val = top.iloc[0] if not top.empty else np.nan
                top_freq = int((s == top_val).sum()) if not top.empty else 0
            except (TypeError, ValueError):
                top_val, top_freq = np.nan, 0
            try:
                n_unique = int(s.nunique(dropna=True))
            except TypeError:
                n_unique = int(s.astype(str).nunique(dropna=True))
            rows.append(
                {
                    "column": c,
                    "n_unique": n_unique,
                    "top": str(top_val)[:80],
                    "top_freq": top_freq,
                    "top_pct": round(top_freq / len(df) * 100, 2),
                }
            )
        summary.categorical_describe = pd.DataFrame(rows).set_index("column")

    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
    if len(dt_cols):
        summary.datetime_describe = pd.DataFrame(
            {
                "min": {c: df[c].min() for c in dt_cols},
                "max": {c: df[c].max() for c in dt_cols},
                "range_days": {
                    c: (df[c].max() - df[c].min()).days
                    if df[c].notna().any()
                    else np.nan
                    for c in dt_cols
                },
                "n_non_null": {c: int(df[c].notna().sum()) for c in dt_cols},
            }
        )

    logger.info("Computed statistical summary for %s", cfg.name)
    return summary
