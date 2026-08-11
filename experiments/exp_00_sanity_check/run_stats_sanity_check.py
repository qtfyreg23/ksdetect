"""
run_stats_sanity_check.py — Part 1 of exp_00_sanity_check.

Runs entirely on synthetic data (no GPU / model required), and verifies:
  (a) a constructed strong signal is detected with high AUC and CI-lower
      clears a threshold via ci_lower_bound_exceeds(),
  (b) pure noise gives ~0.5 AUC with the CI covering 0.5,
  (c) selected-k features significantly beat random-k features on the
      strong-signal data (paired Wilcoxon) — this is the "necessity of
      selection" check the project plan calls for before trusting any
      sparsity result,
  (d) the deliberately-leaky selection path produces an inflated AUC on
      PURE NOISE data that the leakage-safe path does not — proving this
      sanity-check pipeline is capable of catching the exact bug recorded
      in docs/known_issues.md #3, and
  (e) core.labeling.is_correct behaves correctly on a handful of hand-picked
      cases (word-set overlap, not substring).

This script is the automated version of project plan §4.7's exit criteria
for the statistics half of the infrastructure (the extraction half is
covered separately by run_extraction_smoke_test.py, which needs the real
model on the AutoDL server).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.stats import (
    matched_repeated_cv,
    repeated_cv_auc,
    ci_lower_bound_exceeds,
    paired_wilcoxon,
)
from core.stats.nested_cv import cross_validated_auc, cross_validated_auc_LEAKY_FOR_TESTING_ONLY
from core.labeling import is_correct
from core.viz import bar_with_errorbars, save_figure


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("exp_00_stats")
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


def make_strong_signal_data(rng, n_samples, n_informative, n_noise, effect_size):
    y = rng.randint(0, 2, size=n_samples)
    informative = rng.randn(n_samples, n_informative) + (y[:, None] * effect_size)
    noise = rng.randn(n_samples, n_noise)
    X = np.concatenate([informative, noise], axis=1)
    return X, y


def make_pure_noise_data(rng, n_samples, n_features):
    y = rng.randint(0, 2, size=n_samples)
    X = rng.randn(n_samples, n_features)
    return X, y


def run_labeling_checks(logger) -> dict:
    cases = [
        # (prediction, references, expected)
        ("The answer is Paris.", ["Paris"], True),
        ("I believe it was New York City.", ["New York"], True),
        ("The capital is Berlin.", ["Paris"], False),
        ("42", ["42"], True),
        ("The result is forty-two.", ["42"], False),  # intentionally NOT
        # normalized (numeric-word equivalence is out of scope by design,
        # see correctness.py docstring) — this case documents that choice.
        ("Marie Curie", ["Marie Curie", "Curie"], True),
        ("", ["something"], False),
    ]
    results = []
    all_passed = True
    for pred, refs, expected in cases:
        actual = is_correct(pred, refs)
        passed = actual == expected
        all_passed = all_passed and passed
        results.append({
            "prediction": pred, "references": refs,
            "expected": expected, "actual": actual, "passed": passed,
        })
        logger.info(f"labeling check: pred={pred!r} refs={refs} "
                     f"expected={expected} actual={actual} "
                     f"{'PASS' if passed else 'FAIL'}")
    return {"all_passed": all_passed, "cases": results}


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logging(os.path.join(out_dir, "..", "run_log_stats.txt"))
    t_start = time.time()
    logger.info("=== exp_00_sanity_check / Part 1: stats sanity check ===")
    logger.info(f"Config: {json.dumps(cfg, ensure_ascii=False)}")

    rng = np.random.RandomState(cfg["random_state"])
    report = {}

    # (a) + (c) strong signal: high AUC, and selected >> random
    logger.info("--- (a)/(c) strong-signal scenario ---")
    X_strong, y_strong = make_strong_signal_data(
        rng, cfg["n_samples"], cfg["n_informative_features"],
        cfg["n_noise_features"], cfg["informative_signal_strength"],
    )
    k_check = cfg["k_values"][1]  # middle k value, e.g. 32
    matched = matched_repeated_cv(
        feature_sets={"selected": X_strong, "random": X_strong},
        y=y_strong, k=k_check,
        selection_method="l1",   # will be overridden per-arm below
        n_splits=cfg["n_splits"], n_seeds=cfg["n_seeds"], base_seed=cfg["base_seed"],
    )
    # matched_repeated_cv applies ONE selection_method to all named arms; to
    # compare two different selection methods ("l1" vs "random") on
    # identical folds we call it twice, using the SAME base_seed so the
    # underlying StratifiedKFold splits line up fold-for-fold (both calls
    # derive folds solely from y and the seed).
    selected_run = repeated_cv_auc(
        X_strong, y_strong, k=k_check, selection_method="l1",
        n_splits=cfg["n_splits"], n_seeds=cfg["n_seeds"], base_seed=cfg["base_seed"],
    )
    random_run = repeated_cv_auc(
        X_strong, y_strong, k=k_check, selection_method="random",
        n_splits=cfg["n_splits"], n_seeds=cfg["n_seeds"], base_seed=cfg["base_seed"],
    )
    strong_signal_check = ci_lower_bound_exceeds(
        selected_run["pooled_aucs"], threshold=0.75,
    )
    selected_vs_random = paired_wilcoxon(
        selected_run["pooled_aucs"], random_run["pooled_aucs"],
    )
    logger.info(f"selected(k={k_check}) mean AUC = {selected_run['mean']:.4f} "
                 f"(CI [{selected_run['ci_lower']:.4f}, {selected_run['ci_upper']:.4f}])")
    logger.info(f"random(k={k_check})   mean AUC = {random_run['mean']:.4f} "
                 f"(CI [{random_run['ci_lower']:.4f}, {random_run['ci_upper']:.4f}])")
    logger.info(f"strong-signal CI-lower > 0.75 threshold: {strong_signal_check['passed']}")
    logger.info(f"selected vs random paired Wilcoxon p = {selected_vs_random['p_value']:.6f}")

    report["strong_signal"] = {
        "selected": {k: v for k, v in selected_run.items() if k != "pooled_aucs"},
        "random": {k: v for k, v in random_run.items() if k != "pooled_aucs"},
        "ci_threshold_check": strong_signal_check,
        "selected_vs_random_wilcoxon": selected_vs_random,
        "PASS": bool(strong_signal_check["passed"] and selected_vs_random["p_value"] < 0.05
                       and selected_run["mean"] > random_run["mean"]),
    }

    # (b) pure noise: ~0.5 AUC, CI covers 0.5
    logger.info("--- (b) pure-noise scenario ---")
    X_noise, y_noise = make_pure_noise_data(
        rng, cfg["n_samples"], cfg["n_noise_features"],
    )
    noise_run = repeated_cv_auc(
        X_noise, y_noise, k=k_check, selection_method="l1",
        n_splits=cfg["n_splits"], n_seeds=cfg["n_seeds"], base_seed=cfg["base_seed"],
    )
    noise_covers_half = noise_run["ci_lower"] <= 0.5 <= noise_run["ci_upper"]
    logger.info(f"pure-noise mean AUC = {noise_run['mean']:.4f} "
                 f"(CI [{noise_run['ci_lower']:.4f}, {noise_run['ci_upper']:.4f}]); "
                 f"covers 0.5: {noise_covers_half}")
    report["pure_noise"] = {
        "summary": {k: v for k, v in noise_run.items() if k != "pooled_aucs"},
        "ci_covers_0.5": bool(noise_covers_half),
        "PASS": bool(noise_covers_half),
    }

    # (d) leakage detection
    logger.info("--- (d) leakage detection (regression test) ---")
    proper_aucs, leaky_aucs = [], []
    for seed_offset in range(cfg["n_seeds"]):
        seed = cfg["base_seed"] + seed_offset
        proper_aucs.append(cross_validated_auc(
            X_noise, y_noise, k=k_check, selection_method="l1",
            n_splits=cfg["n_splits"], random_state=seed,
        ))
        leaky_aucs.append(cross_validated_auc_LEAKY_FOR_TESTING_ONLY(
            X_noise, y_noise, k=k_check, selection_method="l1",
            n_splits=cfg["n_splits"], random_state=seed,
        ))
    proper_aucs = np.concatenate(proper_aucs)
    leaky_aucs = np.concatenate(leaky_aucs)
    leak_diff = paired_wilcoxon(leaky_aucs, proper_aucs)
    leak_detected = bool(leaky_aucs.mean() > proper_aucs.mean() and leak_diff["p_value"] < 0.05)
    logger.info(f"proper mean AUC = {proper_aucs.mean():.4f}, "
                 f"leaky mean AUC = {leaky_aucs.mean():.4f}, "
                 f"paired Wilcoxon p = {leak_diff['p_value']:.6f}")
    logger.info(f"leakage successfully detected by this pipeline: {leak_detected}")
    report["leakage_detection"] = {
        "proper_mean_auc": float(proper_aucs.mean()),
        "leaky_mean_auc": float(leaky_aucs.mean()),
        "wilcoxon": leak_diff,
        "PASS": leak_detected,
    }

    # (e) labeling checks
    logger.info("--- (e) labeling correctness checks ---")
    labeling_report = run_labeling_checks(logger)
    report["labeling"] = labeling_report
    report["labeling"]["PASS"] = labeling_report["all_passed"]

    # --- figure: proper vs leaky, and selected vs random ---
    fig, ax = bar_with_errorbars(
        labels=["Proper (safe)", "Leaky (bug fixture)"],
        means=[float(proper_aucs.mean()), float(leaky_aucs.mean())],
        errors=[float(proper_aucs.std(ddof=1)), float(leaky_aucs.std(ddof=1))],
        title="Sanity Check: Leakage Inflates AUC on Pure Noise",
        ylabel="AUC (pooled folds x seeds)",
    )
    save_figure(fig, os.path.join(out_dir, "fig_leakage_check.pdf"))

    fig2, ax2 = bar_with_errorbars(
        labels=["Selected (L1)", "Random"],
        means=[float(selected_run["mean"]), float(random_run["mean"])],
        errors=[float(selected_run["std"]), float(random_run["std"])],
        title=f"Sanity Check: Selected vs Random Features (k={k_check})",
        ylabel="AUC (pooled folds x seeds)",
    )
    save_figure(fig2, os.path.join(out_dir, "fig_selected_vs_random.pdf"))

    elapsed = time.time() - t_start
    overall_pass = all(report[k]["PASS"] for k in
                         ["strong_signal", "pure_noise", "leakage_detection", "labeling"])
    report["overall_PASS"] = overall_pass
    report["elapsed_seconds"] = elapsed

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"=== OVERALL PASS: {overall_pass} (elapsed {elapsed:.1f}s) ===")
    logger.info(f"Results written to {out_dir}/summary.json and figures.")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                         default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()
    result = main(args.config)
    sys.exit(0 if result["overall_PASS"] else 1)
