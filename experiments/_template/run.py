"""
Template experiment runner. Copy this file into a new
experiments/exp_YYYYMMDD_short_name/run.py and fill in the TODO sections.

This file intentionally does NOT contain any CV/feature-selection/labeling
logic of its own — it only orchestrates calls into core.*. If you find
yourself about to write a for-loop that does cross-validation, STOP and use
core.stats.repeated_cv_auc / matched_repeated_cv instead (project plan §2.3,
§2.5).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import yaml

# Make `core` importable regardless of where this script is invoked from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.extraction import load_model_and_tokenizer, ActivationCollector, extract_batch, save_activation_shard
from core.labeling import hard_label_from_greedy, soft_label_from_samples
from core.stats import matched_repeated_cv, ci_lower_bound_exceeds
from core.viz import line_with_ci, save_figure


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    return logger


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Refuse to run with template placeholders left in — this is the
    # automated version of project plan §2.1's "确认最新要求已经完整对齐".
    for key, value in cfg.items():
        if isinstance(value, str) and value.startswith("REPLACE_ME"):
            raise ValueError(
                f"config.yaml field {key!r} still has a template placeholder "
                f"({value!r}) — fill it in before running."
            )
    return cfg


def main(config_path: str):
    cfg = load_config(config_path)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    logger = setup_logging(os.path.join(os.path.dirname(cfg["output_dir"]), "run_log.txt"))

    logger.info(f"Starting experiment: {cfg['experiment_name']}")
    logger.info(f"Description: {cfg['description']}")
    logger.info(f"Full config: {json.dumps(cfg, ensure_ascii=False)}")

    t_start = time.time()

    # ------------------------------------------------------------------
    # TODO 1: Load dataset. Must produce a list of examples, each with at
    # least: example_id, question/prompt text, reference answer(s).
    # Use n_examples from config to allow a small-scale pilot run first
    # (project plan §2.2).
    # ------------------------------------------------------------------
    # examples = load_my_dataset(cfg["dataset"], cfg["split"], cfg["n_examples"])
    raise NotImplementedError(
        "TODO 1: implement dataset loading for this experiment, "
        "then delete this line."
    )

    # ------------------------------------------------------------------
    # TODO 2: Load model + tokenizer via core.extraction (do not reimplement).
    # ------------------------------------------------------------------
    # model, tokenizer = load_model_and_tokenizer(
    #     model_path=cfg["model_path"], device=cfg["device"], dtype=cfg["dtype"],
    # )

    # ------------------------------------------------------------------
    # TODO 3: Build the ActivationCollector for the requested modules/layers,
    # then loop over batches calling extract_batch() + save_activation_shard().
    # Report progress per project plan §2.4 (完成/运行中/出错/待运行数量,
    # 当前速度, 预计耗时) — log this every N batches, not just at the end.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # TODO 4: Build labels via core.labeling (hard_label_from_greedy or
    # soft_label_from_samples per cfg["label_type"]).
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # TODO 5: Run core.stats.matched_repeated_cv across the modules/layers/
    # k_values of interest, using ci_lower_bound_exceeds() for any
    # threshold-crossing decisions (never a bare point-estimate comparison).
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # TODO 6: Plot via core.viz (English-only, apply_style already called
    # inside the helpers) and save_figure() into cfg["output_dir"].
    # ------------------------------------------------------------------

    elapsed = time.time() - t_start
    logger.info(f"Experiment finished in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # TODO 7: Write results/summary.json and a results/run_report.md that
    # follows the reporting template in README.md Appendix-equivalent
    # section ("每次运行后的汇报清单") — this is what gets sent back for
    # review, so make sure every field there is filled in, not left blank.
    # ------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
