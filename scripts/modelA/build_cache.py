"""One-shot builder for the signals + features cache.

Usage
-----
    python -m scripts.modelA.build_cache             # full dataset
    python -m scripts.modelA.build_cache --workers 8

Safe to re-run: the build is resumable. Interruptions only lose progress
since the last chunk save (default every 2000 records).
"""

from __future__ import annotations

import argparse
import logging
import sys

from .dataset import load_ptbxl_index, precompute_cache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Modèle A signal + feature cache.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel worker processes (default ≈ CPU-1).")
    parser.add_argument("--save-every", type=int, default=2000,
                        help="Save partial cache every N completed records.")
    parser.add_argument("--force", action="store_true",
                        help="Discard existing cache and rebuild from scratch.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    df = load_ptbxl_index()
    sigs, feats = precompute_cache(
        df,
        workers=args.workers,
        save_every=args.save_every,
        force=args.force,
    )
    print(
        f"Cache ready: signals={sigs.shape}  features={feats.shape}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
