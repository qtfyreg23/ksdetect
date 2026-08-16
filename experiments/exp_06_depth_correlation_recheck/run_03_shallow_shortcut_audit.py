"""
run_03_shallow_shortcut_audit.py — exp_06, step 3.

Generalizes exp_03's MedQuad-only length-shortcut check
(run_03_medquad_confound_check.py, docs/decisions.md D21) to every dataset
that appears to "saturate" at shallow layers in the depth-correlation
recheck — before treating shallow saturation as evidence that a task
"doesn't need deep integration," rule out that the shallow-layer signal is
actually just picking up a cheap surface shortcut.

Features tested (cheap, question-text-only, no model/GPU needed):
  - word_length, char_length (same as exp_03's check)
  - gsm8k only: n_numbers_in_question, max_number_in_question — math
    problems have an obvious additional cheap shortcut candidate (surface
    numeric content) that other datasets don't have a clean analogue for.

Datasets to audit are read from config.yaml's `shallow_audit_datasets`
list — set this AFTER looking at run_02's depth-correlation output, not
before (this script doesn't decide for itself which datasets look
shallow-saturating; that's a judgment call made from run_02's numbers).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.data import load_generated_labels
from core.stats import repeated_cv_auc


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("exp_06_shortcut_audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)
    return logger


def extract_numbers(text: str) -> list[float]:
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    out = []
    for m in matches:
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def build_features(question: str, dataset: str) -> dict:
    feats = {
        "word_length": len(question.split()),
        "char_length": len(question),
    }
    if dataset == "gsm8k":
        nums = extract_numbers(question)
        feats["n_numbers_in_question"] = len(nums)
        feats["max_number_in_question"] = max(nums) if nums else 0.0
    return feats


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(os.path.join(output_dir, "run_log_shortcut_audit.txt"))
    logger.info("=== exp_06 run_03: shallow-layer surface-shortcut audit ===")

    datasets = cfg.get("shallow_audit_datasets", ["medquad", "gsm8k"])
    logger.info(f"Auditing datasets: {datasets} (set via config.yaml's "
                 f"shallow_audit_datasets — pick these based on run_02's output)")

    all_results = {}
    for dataset in datasets:
        logger.info(f"\n--- {dataset} ---")
        records = load_generated_labels(cfg["coarse_scan_output_dir"], dataset)
        example_ids = sorted(records.keys())

        feature_dicts = [build_features(records[eid]["question"], dataset) for eid in example_ids]
        feature_names = sorted(feature_dicts[0].keys())
        X_individual = {
            name: np.array([[fd[name]] for fd in feature_dicts], dtype=np.float32)
            for name in feature_names
        }
        X_all = np.array([[fd[name] for name in feature_names] for fd in feature_dicts],
                           dtype=np.float32)
        y = np.array([records[eid]["hard_label"] for eid in example_ids], dtype=int)

        logger.info(f"n_examples={len(y)}, n_positive(hard_label=1)={y.sum()} ({100*y.mean():.1f}%)")

        dataset_results = {}
        for name in feature_names:
            run = repeated_cv_auc(X_individual[name], y, k=None, n_splits=cfg["n_splits"],
                                    n_seeds=cfg["n_seeds"], base_seed=cfg["base_seed"])
            dataset_results[name] = {k: v for k, v in run.items() if k != "pooled_aucs"}
            logger.info(f"  {name}: mean AUC = {run['mean']:.3f} "
                         f"(CI [{run['ci_lower']:.3f}, {run['ci_upper']:.3f}])")

        run_all = repeated_cv_auc(X_all, y, k=None, n_splits=cfg["n_splits"],
                                    n_seeds=cfg["n_seeds"], base_seed=cfg["base_seed"])
        dataset_results["all_surface_features_combined"] = {
            k: v for k, v in run_all.items() if k != "pooled_aucs"
        }
        logger.info(f"  ALL combined: mean AUC = {run_all['mean']:.3f} "
                     f"(CI [{run_all['ci_lower']:.3f}, {run_all['ci_upper']:.3f}])")

        all_results[dataset] = dataset_results

    with open(os.path.join(output_dir, "shallow_shortcut_audit.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logger.info("\n=== How to read this ===")
    logger.info("Compare 'all_surface_features_combined' CI-upper against the shallow-layer "
                 "(e.g. layer 0) pregen AUC from run_02's corrected_depth_results.csv for the "
                 "same dataset. If they're close, the shallow-layer signal may substantially be "
                 "a surface shortcut rather than 'the task doesn't need deep integration'. If the "
                 "surface-feature AUC is near chance (~0.5), that explanation is ruled out for "
                 "these specific cheap features (does not rule out subtler shortcuts not tested here).")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                         default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()
    main(args.config)
