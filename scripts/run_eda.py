"""Main entrypoint: run the full EDA pipeline for every configured dataset.

Usage
-----
    python -m scripts.run_eda                 # all datasets
    python -m scripts.run_eda ptbxl_database  # one dataset by name
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import DATASETS, DatasetConfig, dataset_output_dirs
from .data_cleaner import clean_dataset, save_processed
from .data_loader import load_dataset
from .feature_analysis import analyze_features
from .report_generator import write_report
from .statistical_analysis import compute_statistics
from .visualization import generate_all_visualizations


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("eda")


def run_one(cfg: DatasetConfig) -> dict[str, str]:
    t0 = time.time()
    dirs = dataset_output_dirs(cfg.name)
    logger.info("=" * 70)
    logger.info("Starting EDA for %s", cfg.name)
    logger.info("=" * 70)

    raw = load_dataset(cfg)
    cleaned, cleaning = clean_dataset(raw, cfg)
    processed_path = save_processed(cleaned, cfg, dirs["processed"])
    summary = compute_statistics(cleaned, cfg)
    fa = analyze_features(cleaned, cfg)
    figures = generate_all_visualizations(
        cleaned, cfg, summary, fa, dirs["vis"]
    )
    report_path = write_report(
        cleaned, cfg, summary, fa, cleaning, figures,
        out_dir=dirs["report"], vis_dir=dirs["vis"],
    )

    elapsed = time.time() - t0
    logger.info("Finished %s in %.1fs", cfg.name, elapsed)
    return {
        "dataset": cfg.name,
        "report": report_path,
        "processed": processed_path,
        "n_figures": str(len(figures)),
        "elapsed_s": f"{elapsed:.1f}",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run EDA pipeline.")
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Optional list of dataset names to run (default: all).",
    )
    args = parser.parse_args(argv)

    selected = (
        [c for c in DATASETS if c.name in args.datasets]
        if args.datasets
        else list(DATASETS)
    )
    if not selected:
        logger.error("No matching datasets. Known: %s", [c.name for c in DATASETS])
        return 2

    results = [run_one(cfg) for cfg in selected]

    logger.info("\nSUMMARY")
    for r in results:
        logger.info(
            "  %s  | report=%s | processed=%s | figures=%s | %ss",
            r["dataset"],
            Path(r["report"]).name,
            Path(r["processed"]).name,
            r["n_figures"],
            r["elapsed_s"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
