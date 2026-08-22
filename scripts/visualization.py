"""Visualization helpers for the EDA pipeline.

All plots are written to ``visualizations/<dataset_name>/`` as PNGs.
Each function returns the list of files it wrote so the report
generator can embed them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .config import DatasetConfig  # noqa: E402
from .feature_analysis import FeatureAnalysis  # noqa: E402
from .statistical_analysis import StatisticalSummary  # noqa: E402


logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="talk")
PALETTE = "viridis"
FIG_DPI = 110


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _truncate_labels(labels: Iterable[str], n: int = 30) -> list[str]:
    return [str(lab)[:n] + ("..." if len(str(lab)) > n else "") for lab in labels]


def plot_missing_values(
    df: pd.DataFrame, cfg: DatasetConfig, out_dir: Path
) -> list[str]:
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        return []
    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(miss))))
    sns.barplot(x=miss.values * 100, y=miss.index, ax=ax, palette=PALETTE, hue=miss.index, legend=False)
    ax.set_xlabel("Missing (%)")
    ax.set_ylabel("Column")
    ax.set_title(f"Missing values per column — {cfg.name}")
    return [_save(fig, out_dir / "missing_values.png")]


def plot_dtype_distribution(
    df: pd.DataFrame, cfg: DatasetConfig, out_dir: Path
) -> list[str]:
    counts = df.dtypes.astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette=PALETTE, hue=counts.index, legend=False)
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {v}", va="center")
    ax.set_xlabel("Number of columns")
    ax.set_ylabel("dtype")
    ax.set_title(f"Column dtypes — {cfg.name}")
    return [_save(fig, out_dir / "dtype_distribution.png")]


def plot_numeric_distributions(
    df: pd.DataFrame, cfg: DatasetConfig, out_dir: Path, max_cols: int = 12
) -> list[str]:
    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in cfg.id_columns and df[c].nunique(dropna=True) > 1
    ]
    if not numeric_cols:
        return []
    numeric_cols = numeric_cols[:max_cols]
    n = len(numeric_cols)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, col in zip(axes, numeric_cols):
        data = df[col].dropna()
        if data.empty:
            ax.set_visible(False)
            continue
        sns.histplot(data, kde=True, ax=ax, color="#3b528b", bins=40)
        ax.set_title(col, fontsize=12)
        ax.set_xlabel("")
    for ax in axes[len(numeric_cols):]:
        ax.set_visible(False)
    fig.suptitle(f"Numeric distributions — {cfg.name}", y=1.02)
    return [_save(fig, out_dir / "numeric_distributions.png")]


def plot_outlier_boxplots(
    df: pd.DataFrame, cfg: DatasetConfig, out_dir: Path, max_cols: int = 10
) -> list[str]:
    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in cfg.id_columns and df[c].nunique(dropna=True) > 1
    ]
    if not numeric_cols:
        return []
    numeric_cols = numeric_cols[:max_cols]
    fig, ax = plt.subplots(figsize=(min(20, 2 + 1.4 * len(numeric_cols)), 6))
    data = df[numeric_cols].apply(
        lambda s: (s - s.mean()) / (s.std() if s.std() else 1)
    )
    sns.boxplot(data=data, ax=ax, palette=PALETTE)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(_truncate_labels(numeric_cols, 18), rotation=30, ha="right")
    ax.set_ylabel("z-score")
    ax.set_title(f"Outliers (standardized) — {cfg.name}")
    return [_save(fig, out_dir / "outliers_boxplot.png")]


def plot_correlation_heatmap(
    summary: StatisticalSummary, cfg: DatasetConfig, out_dir: Path
) -> list[str]:
    corr = summary.correlation
    if corr is None or corr.empty:
        return []
    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(corr)), max(6, 0.6 * len(corr))))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.7},
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title(f"Numeric correlation heatmap — {cfg.name}")
    return [_save(fig, out_dir / "correlation_heatmap.png")]


def plot_class_balance(
    fa: FeatureAnalysis, cfg: DatasetConfig, out_dir: Path
) -> list[str]:
    paths: list[str] = []

    for col, vc in fa.label_distributions.items():
        if vc.empty:
            continue
        vc = vc.head(cfg.max_categories)
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(vc))))
        labels = _truncate_labels(vc.index.astype(str), 40)
        sns.barplot(x=vc.values, y=labels, ax=ax, palette=PALETTE, hue=labels, legend=False)
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
        ax.set_title(f"Class balance — {col}")
        paths.append(_save(fig, out_dir / f"class_balance_{col}.png"))

    for col, counts in fa.multilabel_counts.items():
        top = counts.head(cfg.max_categories)
        fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(top))))
        labels = _truncate_labels(top.index.astype(str), 40)
        sns.barplot(x=top.values, y=labels, ax=ax, palette=PALETTE, hue=labels, legend=False)
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
        ax.set_title(f"Top {len(top)} labels — {col}")
        paths.append(_save(fig, out_dir / f"multilabel_top_{col}.png"))

    return paths


def plot_timeseries(
    df: pd.DataFrame, cfg: DatasetConfig, out_dir: Path
) -> list[str]:
    paths: list[str] = []
    dt_cols = [c for c in cfg.datetime_columns if c in df.columns]
    for col in dt_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        monthly = s.dt.to_period("M").value_counts().sort_index()
        monthly.index = monthly.index.to_timestamp()
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(monthly.index, monthly.values, color="#21918c", linewidth=1.5)
        ax.fill_between(monthly.index, monthly.values, color="#21918c", alpha=0.25)
        ax.set_title(f"Records per month — {col}")
        ax.set_ylabel("count")
        ax.set_xlabel("date")
        paths.append(_save(fig, out_dir / f"timeseries_{col}.png"))

        yearly = s.dt.year.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.barplot(x=yearly.index.astype(int), y=yearly.values, ax=ax, palette=PALETTE, hue=yearly.index.astype(int), legend=False)
        ax.set_title(f"Records per year — {col}")
        ax.set_ylabel("count")
        ax.set_xlabel("year")
        ax.tick_params(axis="x", rotation=45)
        paths.append(_save(fig, out_dir / f"timeseries_year_{col}.png"))
    return paths


def plot_age_sex_pyramid(df: pd.DataFrame, out_dir: Path) -> list[str]:
    if not {"age", "sex"}.issubset(df.columns):
        return []
    sub = df[["age", "sex"]].dropna()
    if sub.empty:
        return []
    sub = sub[(sub["age"] >= 0) & (sub["age"] <= 120)]
    bins = list(range(0, 110, 10))
    labels = [f"{b}-{b+9}" for b in bins[:-1]]
    sub["age_bin"] = pd.cut(sub["age"], bins=bins, labels=labels, right=False)
    pivot = sub.groupby(["age_bin", "sex"], observed=True).size().unstack(fill_value=0)
    pivot.columns = [{0: "male", 1: "female"}.get(c, str(c)) for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(10, 6))
    if "male" in pivot.columns:
        ax.barh(pivot.index.astype(str), -pivot["male"], color="#2c728e", label="male")
    if "female" in pivot.columns:
        ax.barh(pivot.index.astype(str), pivot["female"], color="#dc4869", label="female")
    ax.axvline(0, color="black", linewidth=0.8)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(abs(t))) for t in ticks])
    ax.set_xlabel("count")
    ax.set_ylabel("age group")
    ax.set_title("Age × Sex pyramid")
    ax.legend(loc="lower right")
    return [_save(fig, out_dir / "age_sex_pyramid.png")]


def generate_all_visualizations(
    df: pd.DataFrame,
    cfg: DatasetConfig,
    summary: StatisticalSummary,
    fa: FeatureAnalysis,
    out_dir: Path,
) -> list[str]:
    """Run every applicable plot for this dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    paths += plot_missing_values(df, cfg, out_dir)
    paths += plot_dtype_distribution(df, cfg, out_dir)
    paths += plot_numeric_distributions(df, cfg, out_dir)
    paths += plot_outlier_boxplots(df, cfg, out_dir)
    paths += plot_correlation_heatmap(summary, cfg, out_dir)
    paths += plot_class_balance(fa, cfg, out_dir)
    paths += plot_timeseries(df, cfg, out_dir)
    paths += plot_age_sex_pyramid(df, out_dir)
    logger.info("Wrote %d figures for %s", len(paths), cfg.name)
    return paths
