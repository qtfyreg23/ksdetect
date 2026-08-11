#!/usr/bin/env python
"""
download_datasets.py — one-time (per dataset) download of the 5 wide-scan
datasets to local disk, via HF's save_to_disk (Arrow format). Run this on
the AutoDL server; needs network access to huggingface.co, but only THIS
script does — core/data/loaders.py never touches the network afterward.

Usage:
    python scripts/download_datasets.py                 # download all 5
    python scripts/download_datasets.py --only gsm8k coqa  # download a subset
    python scripts/download_datasets.py --force          # re-download even
                                                            # if already present

Idempotent: skips a dataset whose target directory already exists and looks
complete (has a dataset_info.json), unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import datasets as hf_datasets

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.data.paths import DATASET_DIRS


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("download_datasets")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    log_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, "download_log.txt"))
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)
    return logger


def _is_already_downloaded(target_dir: str) -> bool:
    if not os.path.isdir(target_dir):
        return False
    # save_to_disk for a DatasetDict writes a dataset_dict.json at the top
    # level; for a single Dataset it writes dataset_info.json directly.
    return (
        os.path.exists(os.path.join(target_dir, "dataset_dict.json"))
        or os.path.exists(os.path.join(target_dir, "dataset_info.json"))
    )


def download_truthfulqa(target_dir: str, logger: logging.Logger):
    ds = hf_datasets.load_dataset("truthful_qa", "generation")
    ds.save_to_disk(target_dir)
    logger.info(f"truthfulqa: saved splits {list(ds.keys())}, "
                 f"sizes {[len(ds[s]) for s in ds]}")


def download_triviaqa(target_dir: str, logger: logging.Logger):
    ds = hf_datasets.load_dataset("trivia_qa", "rc.nocontext")
    ds.save_to_disk(target_dir)
    logger.info(f"triviaqa: saved splits {list(ds.keys())}, "
                 f"sizes {[len(ds[s]) for s in ds]}")


def download_coqa(target_dir: str, logger: logging.Logger):
    # Legacy script-based HF loader; needs trust_remote_code=True to run the
    # dataset's loading script. This is the ONLY point in the whole codebase
    # that executes remote dataset code — it happens once, here, at download
    # time, not on every experiment run.
    ds = hf_datasets.load_dataset("coqa", trust_remote_code=True)
    ds.save_to_disk(target_dir)
    logger.info(f"coqa: saved splits {list(ds.keys())}, "
                 f"sizes {[len(ds[s]) for s in ds]}")


def download_medquad(target_dir: str, logger: logging.Logger):
    ds = hf_datasets.load_dataset("lavita/MedQuAD")
    ds.save_to_disk(target_dir)
    logger.info(f"medquad: saved splits {list(ds.keys())}, "
                 f"sizes {[len(ds[s]) for s in ds]}")


def download_gsm8k(target_dir: str, logger: logging.Logger):
    ds = hf_datasets.load_dataset("gsm8k", "main")
    ds.save_to_disk(target_dir)
    logger.info(f"gsm8k: saved splits {list(ds.keys())}, "
                 f"sizes {[len(ds[s]) for s in ds]}")


DOWNLOADERS = {
    "truthfulqa": download_truthfulqa,
    "triviaqa": download_triviaqa,
    "coqa": download_coqa,
    "medquad": download_medquad,
    "gsm8k": download_gsm8k,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=list(DOWNLOADERS.keys()),
                         default=None, help="Download only these datasets")
    parser.add_argument("--force", action="store_true",
                         help="Re-download even if already present locally")
    args = parser.parse_args()

    logger = setup_logging()
    targets = args.only if args.only else list(DOWNLOADERS.keys())

    logger.info(f"=== download_datasets: targets={targets} force={args.force} ===")

    results = {}
    for name in targets:
        target_dir = DATASET_DIRS[name]
        if _is_already_downloaded(target_dir) and not args.force:
            logger.info(f"{name}: already present at {target_dir}, skipping "
                         f"(use --force to re-download)")
            results[name] = {"status": "skipped_already_present", "path": target_dir}
            continue

        logger.info(f"{name}: downloading to {target_dir} ...")
        t0 = time.time()
        try:
            os.makedirs(os.path.dirname(target_dir), exist_ok=True)
            DOWNLOADERS[name](target_dir, logger)
            elapsed = time.time() - t0
            logger.info(f"{name}: done in {elapsed:.1f}s")
            results[name] = {"status": "downloaded", "path": target_dir,
                              "elapsed_seconds": elapsed}
        except Exception as e:
            logger.error(f"{name}: FAILED — {type(e).__name__}: {e}")
            results[name] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}

    logger.info("=== Summary ===")
    for name, res in results.items():
        logger.info(f"{name}: {res['status']}")

    n_failed = sum(1 for r in results.values() if r["status"] == "failed")
    if n_failed:
        logger.error(f"{n_failed} dataset(s) failed to download. "
                       f"Fix and re-run with --only <name> before proceeding "
                       f"(project plan §2.4 — don't proceed with gaps).")
        sys.exit(1)

    logger.info("All requested datasets are present locally. "
                 "Next: run experiments/exp_01_data_check/run.py")


if __name__ == "__main__":
    main()
