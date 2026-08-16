"""
run_02_depth_correlation_analysis.py — exp_06, step 2.

For each dataset:
  - pregen: always from exp_02 (never needed correction).
  - posthoc: from whichever source config.yaml's posthoc_source_by_dataset
    maps it to (see that file's header comment for why each dataset maps
    where it does).

Computes per (dataset, module, layer) AUC via repeated_cv_auc (hard label,
k=None — full representation, matching the original Stage-1 wide scan's
methodology so results are comparable), then the layer-vs-AUC correlation
per (dataset, stage), and — if exp_02b's original wide_scan_results.csv is
found at the configured path — compares against the ORIGINAL (uncorrected
posthoc) correlation values, to show explicitly what changed.

Does NOT force a clean verdict — per the lesson from exp_05 (D23), this
script reports the raw numbers and correlations; interpretation of
whether the "two-cluster" pattern survived happens in discussion, not in
an automated label here.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.extraction import load_activation
from core.data import load_generated_labels
from core.stats import repeated_cv_auc


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("exp_06_depth_analysis")
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


def resolve_posthoc_activation_dir(cfg: dict, experiment: str) -> str:
    return {
        "exp03": cfg["exp03_activation_dir"],
        "exp05": cfg["exp05_activation_dir"],
        "exp06": cfg["activation_dir"],
    }[experiment]


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(os.path.join(output_dir, "run_log_depth_analysis.txt"))
    logger.info("=== exp_06 run_02: corrected depth-correlation analysis ===")

    modules_to_test = [m for m in cfg["modules"] if m != "embedding"]
    rows = []

    for dataset, source in cfg["posthoc_source_by_dataset"].items():
        records = load_generated_labels(cfg["coarse_scan_output_dir"], dataset)
        posthoc_dir = resolve_posthoc_activation_dir(cfg, source["experiment"])
        posthoc_stage = source["stage"]
        logger.info(f"--- {dataset}: posthoc source = {source['experiment']}/{posthoc_stage} ---")

        for module in modules_to_test:
            for layer in cfg["layers_coarse"]:
                try:
                    X_pregen, ids_pregen = load_activation(
                        cfg["coarse_scan_activation_dir"], dataset, "pregen", module, layer)
                    X_posthoc, ids_posthoc = load_activation(
                        posthoc_dir, dataset, posthoc_stage, module, layer)
                except FileNotFoundError as e:
                    logger.error(f"Skipping {dataset}/{module}/layer={layer}: {e}")
                    continue

                common = sorted(set(ids_pregen) & set(ids_posthoc))
                idx_p = {eid: i for i, eid in enumerate(ids_pregen)}
                idx_h = {eid: i for i, eid in enumerate(ids_posthoc)}
                X_pregen_a = X_pregen[[idx_p[eid] for eid in common]]
                X_posthoc_a = X_posthoc[[idx_h[eid] for eid in common]]
                y = np.array([records[eid]["hard_label"] for eid in common], dtype=int)

                run_pregen = repeated_cv_auc(X_pregen_a, y, k=None, n_splits=cfg["n_splits"],
                                                n_seeds=cfg["n_seeds"], base_seed=cfg["base_seed"])
                run_posthoc = repeated_cv_auc(X_posthoc_a, y, k=None, n_splits=cfg["n_splits"],
                                                 n_seeds=cfg["n_seeds"], base_seed=cfg["base_seed"])

                rows.append({
                    "dataset": dataset, "module": module, "layer": layer,
                    "n_examples": len(common),
                    "mean_pregen": run_pregen["mean"], "ci_lower_pregen": run_pregen["ci_lower"],
                    "mean_posthoc_corrected": run_posthoc["mean"],
                    "ci_lower_posthoc_corrected": run_posthoc["ci_lower"],
                    "posthoc_source": f"{source['experiment']}/{posthoc_stage}",
                })
                logger.info(f"{dataset}/{module}/layer={layer}: pregen={run_pregen['mean']:.3f} "
                             f"posthoc_corrected={run_posthoc['mean']:.3f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "corrected_depth_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Wrote {len(df)} rows to {csv_path}")

    # --- Correlation per (dataset, stage), averaged over modules first ---
    corr_rows = []
    for dataset, g in df.groupby("dataset"):
        for stage, col in [("pregen", "mean_pregen"), ("posthoc_corrected", "mean_posthoc_corrected")]:
            layer_avg = g.groupby("layer")[col].mean().reset_index()
            corr = layer_avg["layer"].corr(layer_avg[col]) if len(layer_avg) >= 3 else float("nan")
            corr_rows.append({"dataset": dataset, "stage": stage, "corr_layer_vs_auc": corr})
    corr_df = pd.DataFrame(corr_rows)

    # --- Compare against OLD (uncorrected) correlations, if available ---
    old_path = cfg.get("old_wide_scan_csv")
    if old_path and os.path.exists(old_path):
        old_df = pd.read_csv(old_path)
        old_df = old_df[(old_df.label_type == "hard") & (old_df.k.isna())
                          & ~((old_df.module == "embedding") & (old_df.stage == "pregen"))]
        old_corr_rows = []
        for (dataset, stage), g in old_df.groupby(["dataset", "stage"]):
            layer_avg = g.groupby("layer")["mean"].mean().reset_index()
            corr = layer_avg["layer"].corr(layer_avg["mean"]) if len(layer_avg) >= 3 else float("nan")
            old_corr_rows.append({"dataset": dataset, "stage": stage, "old_corr_layer_vs_auc": corr})
        old_corr_df = pd.DataFrame(old_corr_rows)
        old_corr_df["stage"] = old_corr_df["stage"].replace({"posthoc": "posthoc_corrected"})
        corr_df = corr_df.merge(old_corr_df, on=["dataset", "stage"], how="left")
        logger.info("Old wide_scan_results.csv found — comparison included.")
    else:
        logger.warning(f"old_wide_scan_csv not found at {old_path!r} — "
                         f"reporting NEW correlations only, no before/after comparison.")

    corr_csv_path = os.path.join(output_dir, "depth_correlation_comparison.csv")
    corr_df.to_csv(corr_csv_path, index=False)
    logger.info("\n=== Layer-depth vs AUC correlation, per dataset x stage ===")
    logger.info("\n" + corr_df.to_string(index=False))
    logger.info(f"\nWrote correlation comparison to {corr_csv_path}")

    logger.info(
        "\nReminder (per docs/decisions.md D23): these are AGGREGATE "
        "(module-averaged) correlations. Check corrected_depth_results.csv "
        "for the per-module, per-layer numbers before drawing conclusions "
        "about how clean/noisy the pattern is — do not rely on the "
        "correlation number alone."
    )

    return {"csv_path": csv_path, "corr_csv_path": corr_csv_path}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                         default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()
    main(args.config)
