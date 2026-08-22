"""Feature-level analyses (labels, multi-label columns, derived features)."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import DatasetConfig


logger = logging.getLogger(__name__)


@dataclass
class FeatureAnalysis:
    label_distributions: dict[str, pd.Series] = field(default_factory=dict)
    multilabel_counts: dict[str, pd.Series] = field(default_factory=dict)
    cardinalities: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))
    high_cardinality: list[str] = field(default_factory=list)
    derived: dict[str, Any] = field(default_factory=dict)


def _flatten_items(value: Any) -> list[Any]:
    """Extract labels from list/dict/tuple-like cells."""
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, dict):
        return list(value.keys())
    if isinstance(value, (list, tuple, set)):
        items = []
        for v in value:
            if isinstance(v, (list, tuple)) and v:
                items.append(v[0])
            else:
                items.append(v)
        return items
    return [value]


def analyze_features(df: pd.DataFrame, cfg: DatasetConfig) -> FeatureAnalysis:
    fa = FeatureAnalysis()

    cards = {}
    for col in df.columns:
        try:
            cards[col] = int(df[col].nunique(dropna=True))
        except TypeError:
            cards[col] = int(df[col].astype(str).nunique(dropna=True))
    fa.cardinalities = pd.Series(cards).sort_values(ascending=False)
    fa.high_cardinality = [
        c
        for c, n in fa.cardinalities.items()
        if n > 50 and c not in cfg.id_columns and df[c].dtype == object
    ]

    for col in cfg.label_columns:
        if col not in df.columns:
            continue
        series = df[col]
        if col in cfg.list_columns or col in cfg.dict_columns:
            counter: Counter = Counter()
            for v in series.dropna():
                counter.update(_flatten_items(v))
            if counter:
                fa.multilabel_counts[col] = (
                    pd.Series(counter).sort_values(ascending=False)
                )
        else:
            vc = series.value_counts(dropna=False).head(cfg.max_categories)
            fa.label_distributions[col] = vc

    if {"age", "sex"}.issubset(df.columns):
        fa.derived["age_by_sex"] = (
            df.groupby("sex")["age"].describe().round(2)
        )
    if "scp_codes" in df.columns and df["scp_codes"].notna().any():
        n_codes = df["scp_codes"].map(
            lambda v: len(v) if isinstance(v, dict) else 0
        )
        fa.derived["scp_codes_per_record"] = n_codes.describe().round(2)

    list_cols = [c for c in cfg.list_columns if c in df.columns]
    for col in list_cols:
        sizes = df[col].map(lambda v: len(v) if isinstance(v, (list, tuple)) else 0)
        if sizes.sum() > 0:
            fa.derived[f"{col}_size_describe"] = sizes.describe().round(2)

    logger.info("Feature analysis for %s complete", cfg.name)
    return fa
