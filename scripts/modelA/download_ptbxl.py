"""Download PTB-XL 100 Hz WFDB files from PhysioNet.

The repository already ships with ``data/ptbxl_database.csv`` (the index),
but the actual signal files (``.hea`` + ``.dat``) live on PhysioNet. This
script reads the index, derives the list of WFDB files to fetch, and
downloads them in parallel into ``data/ptbxl/``.

Usage
-----
    python -m scripts.modelA.download_ptbxl                # all 21,799 records
    python -m scripts.modelA.download_ptbxl --limit 1000   # subset for testing
    python -m scripts.modelA.download_ptbxl --workers 32   # tune concurrency
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from .config import PTBXL_CSV, PTBXL_DIR


S3_BASE_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3"
PHYSIONET_BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3"
BASE_URL = S3_BASE_URL
DEFAULT_WORKERS = 24
CHUNK_SIZE = 1024 * 64
HTTP_TIMEOUT = 60
MAX_RETRIES = 5

_thread_local = threading.local()


def _get_session() -> requests.Session:
    """Per-thread requests.Session with built-in retry/keep-alive."""
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        retry = Retry(
            total=4,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=retry)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _thread_local.session = sess
    return sess


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("download_ptbxl")


def _download_file(
    url: str, dest: Path, retries: int = MAX_RETRIES
) -> tuple[Path, bool, str]:
    """Download ``url`` to ``dest`` with retries. Returns (dest, ok, msg)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest, True, "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    session = _get_session()
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=HTTP_TIMEOUT) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
            tmp.replace(dest)
            return dest, True, "downloaded"
        except (requests.RequestException, OSError) as exc:
            if attempt == retries:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
                return dest, False, f"failed after {retries}: {exc}"
            time.sleep(min(30, 2 ** attempt))
    return dest, False, "unreachable"


def _build_task_list(filenames_lr: list[str]) -> list[tuple[str, Path]]:
    """Return ``(url, dest)`` pairs for every WFDB header + data file."""
    tasks: list[tuple[str, Path]] = []
    for rel in filenames_lr:
        for ext in (".hea", ".dat"):
            url = f"{BASE_URL}/{rel}{ext}"
            dest = PTBXL_DIR / f"{rel}{ext}"
            tasks.append((url, dest))
    return tasks


def download_records(
    filenames_lr: list[str], workers: int = DEFAULT_WORKERS
) -> tuple[int, int]:
    """Download every header/data file. Returns (n_ok, n_failed)."""
    tasks = _build_task_list(filenames_lr)
    n_ok = 0
    n_failed = 0
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_file, url, dest): url for url, dest in tasks}
        bar = tqdm(as_completed(futures), total=len(futures), unit="file", desc="PTB-XL 100Hz")
        for fut in bar:
            _, ok, msg = fut.result()
            if ok:
                n_ok += 1
            else:
                n_failed += 1
                failures.append(f"{futures[fut]} :: {msg}")
        bar.close()
    if failures:
        log_path = PTBXL_DIR / "download_failures.log"
        log_path.write_text("\n".join(failures), encoding="utf-8")
        logger.warning("%d failures written to %s", n_failed, log_path)
    return n_ok, n_failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download PTB-XL 100Hz signals.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only download the first N records (useful for smoke tests).",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Parallel HTTP workers (default {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--source", choices=("s3", "physionet"), default="s3",
        help="Mirror to use: 's3' (fast, default) or 'physionet' (rate-limited).",
    )
    args = parser.parse_args(argv)

    global BASE_URL
    BASE_URL = S3_BASE_URL if args.source == "s3" else PHYSIONET_BASE_URL
    logger.info("Using mirror: %s", BASE_URL)

    if not PTBXL_CSV.exists():
        logger.error("Index CSV missing: %s", PTBXL_CSV)
        return 2

    PTBXL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PTBXL_CSV, usecols=["filename_lr"])
    filenames = df["filename_lr"].dropna().astype(str).tolist()
    if args.limit:
        filenames = filenames[: args.limit]
    logger.info("Will fetch WFDB files for %d records (= %d HTTP files)",
                len(filenames), 2 * len(filenames))

    t0 = time.time()
    n_ok, n_failed = download_records(filenames, workers=args.workers)
    elapsed = time.time() - t0
    logger.info("Done in %.1fs — ok=%d failed=%d", elapsed, n_ok, n_failed)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
