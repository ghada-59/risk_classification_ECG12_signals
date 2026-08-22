"""Markdown EDA report writer."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DatasetConfig
from .data_cleaner import CleaningReport
from .data_loader import file_size_mb
from .feature_analysis import FeatureAnalysis
from .statistical_analysis import StatisticalSummary


logger = logging.getLogger(__name__)


def _md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_(no data)_"
    return df.head(max_rows).to_markdown()


def _md_series(s: pd.Series, name: str, max_rows: int = 30) -> str:
    if s is None or s.empty:
        return "_(no data)_"
    return s.head(max_rows).to_frame(name).to_markdown()


def _insights(
    df: pd.DataFrame,
    cfg: DatasetConfig,
    summary: StatisticalSummary,
    fa: FeatureAnalysis,
    cleaning: CleaningReport,
) -> list[str]:
    """Build a short list of bullet-point insights tailored to the data."""
    insights: list[str] = []

    miss = summary.overview.get("missing_pct", 0.0)
    insights.append(
        f"Overall missingness is **{miss:.2f}%** across "
        f"{summary.overview['n_rows']:,} rows × {summary.overview['n_columns']} columns."
    )

    if not summary.missing.empty:
        top_missing = summary.missing.head(3)
        worst = ", ".join(
            f"`{c}` ({r.missing_pct:.1f}%)" for c, r in top_missing.iterrows()
            if r.missing_pct > 0
        )
        if worst:
            insights.append(f"Most incomplete columns: {worst}.")

    if cleaning.duplicate_rows:
        insights.append(f"Found **{cleaning.duplicate_rows:,}** fully duplicated rows.")
    if cleaning.duplicate_ids:
        dup_str = ", ".join(f"`{k}` ({v:,})" for k, v in cleaning.duplicate_ids.items())
        insights.append(f"Duplicate identifier values detected: {dup_str}.")
    if cleaning.columns_all_null:
        insights.append(
            "Entirely null columns: "
            + ", ".join(f"`{c}`" for c in cleaning.columns_all_null)
        )
    if cleaning.constant_columns:
        insights.append(
            "Constant (single-value) columns: "
            + ", ".join(f"`{c}`" for c in cleaning.constant_columns)
        )

    if not summary.outlier_counts.empty:
        worst_out = summary.outlier_counts.sort_values(
            "outlier_pct", ascending=False
        ).head(3)
        worst = ", ".join(
            f"`{c}` ({r.outlier_pct:.1f}%)"
            for c, r in worst_out.iterrows()
            if not np.isnan(r.outlier_pct) and r.outlier_pct > 0
        )
        if worst:
            insights.append(f"Highest IQR-outlier rates: {worst}.")

    if not summary.correlation.empty:
        corr = summary.correlation.copy()
        arr = corr.to_numpy(copy=True)
        np.fill_diagonal(arr, np.nan)
        corr = pd.DataFrame(arr, index=corr.index, columns=corr.columns)
        stacked = corr.abs().stack().sort_values(ascending=False)
        if not stacked.empty:
            (a, b), val = stacked.index[0], stacked.iloc[0]
            insights.append(
                f"Strongest numeric correlation: `{a}` ↔ `{b}` "
                f"(|r| = {val:.2f})."
            )

    for col, vc in fa.label_distributions.items():
        if vc.empty:
            continue
        top, top_n = vc.index[0], vc.iloc[0]
        share = top_n / vc.sum() * 100
        insights.append(
            f"`{col}` is dominated by **{top}** ({share:.1f}% of records)."
        )

    for col, counts in fa.multilabel_counts.items():
        total = int(counts.sum())
        unique = int(counts.shape[0])
        top_label = counts.index[0]
        top_share = counts.iloc[0] / total * 100
        insights.append(
            f"`{col}` contains **{unique}** distinct labels across "
            f"**{total:,}** mentions; most frequent is **{top_label}** ({top_share:.1f}%)."
        )

    if not summary.datetime_describe.empty:
        for c, row in summary.datetime_describe.iterrows():
            insights.append(
                f"`{c}` spans {row['min'].date()} → {row['max'].date()} "
                f"({int(row['range_days'])} days, {int(row['n_non_null']):,} non-null)."
            )

    if "scp_codes_per_record" in fa.derived:
        d = fa.derived["scp_codes_per_record"]
        insights.append(
            f"Each ECG carries on average {d['mean']:.2f} SCP codes "
            f"(min {int(d['min'])}, max {int(d['max'])})."
        )

    return insights


def write_report(
    df: pd.DataFrame,
    cfg: DatasetConfig,
    summary: StatisticalSummary,
    fa: FeatureAnalysis,
    cleaning: CleaningReport,
    figures: list[str],
    out_dir: Path,
    vis_dir: Path,
) -> str:
    out_path = out_dir / f"{cfg.name}_eda_report.md"
    rel = lambda p: Path(p).resolve().relative_to(out_dir.resolve().parent.parent).as_posix()  # noqa: E731

    parts: list[str] = []
    parts.append(f"# EDA Report — `{cfg.name}`\n")
    parts.append(f"_{cfg.description}_\n")

    parts.append("## 1. Dataset overview\n")
    ov = summary.overview
    parts.append(
        f"- **Source file:** `{cfg.path.as_posix()}` "
        f"({file_size_mb(cfg.path)} MB)"
    )
    parts.append(f"- **Rows × columns:** {ov['n_rows']:,} × {ov['n_columns']}")
    parts.append(f"- **Memory footprint:** {ov['memory_MB']} MB")
    parts.append(
        f"- **Missing cells:** {ov['missing_cells']:,} "
        f"({ov['missing_pct']:.2f}%)"
    )
    parts.append(f"- **Duplicated rows:** {ov['duplicate_rows']:,}\n")

    parts.append("### Column types\n")
    parts.append(_md_table(summary.dtypes))
    parts.append("")

    parts.append("## 2. Data quality\n")
    parts.append("### Missing values (top 30)\n")
    parts.append(_md_table(summary.missing))
    parts.append("")
    parts.append("### Cleaning actions\n")
    parts.append(
        "| Action | Result |\n|---|---|\n"
        f"| Parsed list columns | {', '.join(cleaning.parsed_list_columns) or '—'} |\n"
        f"| Parsed dict columns | {', '.join(cleaning.parsed_dict_columns) or '—'} |\n"
        f"| Parsed datetime columns | {', '.join(cleaning.parsed_datetime_columns) or '—'} |\n"
        f"| Coerced boolean columns | {', '.join(cleaning.coerced_boolean_columns) or '—'} |\n"
        f"| Fully-null columns | {', '.join(cleaning.columns_all_null) or '—'} |\n"
        f"| Constant columns | {', '.join(cleaning.constant_columns) or '—'} |\n"
        f"| Duplicate rows | {cleaning.duplicate_rows:,} |\n"
        f"| Duplicate identifiers | "
        f"{', '.join(f'{k}: {v:,}' for k, v in cleaning.duplicate_ids.items()) or '—'} |\n"
    )

    if not summary.numeric_describe.empty:
        parts.append("## 3. Statistical summary\n")
        parts.append("### Numeric columns\n")
        parts.append(_md_table(summary.numeric_describe))
        parts.append("")
        parts.append("### Skewness & kurtosis\n")
        sk = pd.DataFrame({"skew": summary.skewness, "kurtosis": summary.kurtosis})
        parts.append(_md_table(sk))
        parts.append("")
        parts.append("### Outliers (IQR rule)\n")
        parts.append(_md_table(summary.outlier_counts))
        parts.append("")

    if not summary.categorical_describe.empty:
        parts.append("### Categorical columns\n")
        parts.append(_md_table(summary.categorical_describe))
        parts.append("")

    if not summary.datetime_describe.empty:
        parts.append("### Datetime columns\n")
        parts.append(_md_table(summary.datetime_describe))
        parts.append("")

    parts.append("## 4. Feature analysis\n")
    parts.append("### Cardinalities (top 20)\n")
    parts.append(_md_series(fa.cardinalities, "n_unique"))
    parts.append("")
    if fa.high_cardinality:
        parts.append(
            "**High-cardinality text columns:** "
            + ", ".join(f"`{c}`" for c in fa.high_cardinality)
        )
        parts.append("")

    for col, vc in fa.label_distributions.items():
        parts.append(f"### Label distribution — `{col}`\n")
        parts.append(_md_series(vc, "count"))
        parts.append("")

    for col, counts in fa.multilabel_counts.items():
        parts.append(f"### Top labels in `{col}`\n")
        parts.append(_md_series(counts, "count"))
        parts.append("")

    for name, value in fa.derived.items():
        parts.append(f"### Derived feature — `{name}`\n")
        if isinstance(value, (pd.Series, pd.DataFrame)):
            parts.append(_md_table(value if isinstance(value, pd.DataFrame) else value.to_frame()))
        else:
            parts.append(f"```\n{value}\n```")
        parts.append("")

    parts.append("## 5. Visualizations\n")
    if figures:
        for fig_path in figures:
            title = Path(fig_path).stem.replace("_", " ").title()
            parts.append(f"### {title}\n")
            parts.append(f"![{title}](../../{rel(fig_path)})\n")
    else:
        parts.append("_No figures generated._\n")

    parts.append("## 6. Key insights\n")
    for line in _insights(df, cfg, summary, fa, cleaning):
        parts.append(f"- {line}")
    parts.append("")

    out_path.write_text("\n".join(parts), encoding="utf-8")
    logger.info("Wrote report %s", out_path)
    return str(out_path)
