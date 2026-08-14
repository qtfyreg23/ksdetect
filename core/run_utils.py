"""
core.run_utils — small shared helpers for experiment run.py scripts that
don't fit neatly into extraction/labeling/stats/viz.

Currently just the config-hash guard (see config_hash / check_or_write_config_hash
below). This was originally written inline inside
experiments/exp_02_coarse_scan/run.py, then about to be copy-pasted a
second and third time into experiments/exp_03_posthoc_pooling_test's two
scripts — per project plan §2.5 ("已经实现过的功能应先找到原实现并复用"),
the third duplication is exactly the trigger to hoist it here instead.

IMPORTANT: config_hash()'s exact hashing behavior (which keys it excludes,
JSON serialization with sort_keys=True) is unchanged from the original
inline version in exp_02's run.py, so any config_hash.txt files already
written by that earlier version remain valid/compatible after exp_02 was
switched to import from here — this was verified by construction (the
function body is byte-for-byte identical), not just by intent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os


def config_hash(cfg: dict, exclude_keys: tuple = ("progress_log_every_n_batches",)) -> str:
    """
    Hash of everything in `cfg` except purely-cosmetic keys (like how often
    to log progress) that don't affect what gets written to disk — if the
    MEANINGFUL part of a config changes between runs sharing the same
    output_dir, resuming would silently mix incompatible data
    (project plan §2.3).
    """
    relevant = {k: v for k, v in cfg.items() if k not in exclude_keys}
    blob = json.dumps(relevant, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def check_or_write_config_hash(
    output_dir: str,
    cfg: dict,
    logger: logging.Logger,
    exclude_keys: tuple = ("progress_log_every_n_batches",),
) -> None:
    """
    On first use of `output_dir`, writes config_hash.txt. On subsequent
    uses, verifies the current config still hashes to the same value —
    raises RuntimeError if not, rather than silently proceeding and mixing
    old/new results in the same output_dir (project plan §2.3). Call this
    once near the start of any resumable experiment script.
    """
    os.makedirs(output_dir, exist_ok=True)
    hash_path = os.path.join(output_dir, "config_hash.txt")
    current_hash = config_hash(cfg, exclude_keys=exclude_keys)
    if os.path.exists(hash_path):
        with open(hash_path, "r") as f:
            previous_hash = f.read().strip()
        if previous_hash != current_hash:
            raise RuntimeError(
                f"config_hash mismatch in {output_dir}: previous run used "
                f"hash {previous_hash}, current config hashes to "
                f"{current_hash}. This output_dir has results from a "
                f"DIFFERENT config — resuming would silently mix old and "
                f"new settings (project plan §2.3). Either revert the "
                f"config, or use a fresh output_dir/activation_dir for "
                f"this new config."
            )
        logger.info(f"config_hash matches previous run ({current_hash}); resuming.")
    else:
        with open(hash_path, "w") as f:
            f.write(current_hash)
        logger.info(f"Wrote new config_hash: {current_hash}")
