"""
run_03_optimal_window_check.py — exp_05, step 3 (addendum).

The raw means in run_02's results show something the binary H1/H2 verdict
function wasn't designed to catch: a NON-MONOTONIC pattern where
front50/front100 nearly MATCH pregen (gap ~ -0.01 to +0.02, straddling
zero) while the whole-answer "full" pooling shows a real, consistent gap
(~0.02-0.06). This script directly tests statistical significance of
front50/front100 vs pregen (not just eyeballing means), reusing the
activations already extracted by run_01 — no new GPU work.

If front50/front100 are NOT significantly different from pregen, the
honest conclusion is: MedQuad's "posthoc < pregen" phenomenon is ALSO
substantially a pooling-window artifact (like the other 4 datasets), just
needing a properly BOUNDED window rather than either "last token" or
"whole (very long) answer" — not evidence of a genuine representational
masking effect (H2).
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
from core.stats import matched_repeated_cv, paired_wilcoxon


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("exp_05_optimal_window")
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


def align_by_ids(arrays_and_ids: dict):
    id_sets = [set(ids) for _, ids in arrays_and_ids.values()]
    common = sorted(set.intersection(*id_sets))
    out = {}
    for name, (X, ids) in arrays_and_ids.items():
        idx = {eid: i for i, eid in enumerate(ids)}
        rows = [idx[eid] for eid in common]
        out[name] = X[rows]
    return out, common


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output_dir"]
    logger = setup_logging(os.path.join(output_dir, "run_log_optimal_window.txt"))
    logger.info("=== exp_05 run_03: is front50/front100 statistically indistinguishable from pregen? ===")

    modules_to_test = [m for m in cfg["modules"] if m != "embedding"]
    rows = []

    for dataset in cfg["datasets"]:
        records = load_generated_labels(cfg["coarse_scan_output_dir"], dataset)
        for module in modules_to_test:
            for layer in cfg["layers_coarse"]:
                raw = {
                    "pregen": load_activation(cfg["coarse_scan_activation_dir"], dataset, "pregen", module, layer),
                    "front50": load_activation(cfg["activation_dir"], dataset, "posthoc_front50", module, layer),
                    "front100": load_activation(cfg["activation_dir"], dataset, "posthoc_front100", module, layer),
                    "full": load_activation(cfg["exp03_activation_dir"], dataset, "posthoc_answermean", module, layer),
                }
                aligned, common_ids = align_by_ids(raw)
                y = np.array([records[eid]["hard_label"] for eid in common_ids], dtype=int)

                result = matched_repeated_cv(
                    feature_sets=aligned, y=y, k=None,
                    n_splits=cfg["n_splits"], n_seeds=cfg["n_seeds"], base_seed=cfg["base_seed"],
                )
                pooled = result["pooled_aucs"]
                summ = result["summary"]

                p_front50 = paired_wilcoxon(pooled["front50"], pooled["pregen"])
                p_front100 = paired_wilcoxon(pooled["front100"], pooled["pregen"])
                p_full = paired_wilcoxon(pooled["full"], pooled["pregen"])

                rows.append({
                    "dataset": dataset, "module": module, "layer": layer,
                    "mean_pregen": summ["pregen"]["mean"],
                    "mean_front50": summ["front50"]["mean"],
                    "mean_front100": summ["front100"]["mean"],
                    "mean_full": summ["full"]["mean"],
                    "p_front50_vs_pregen": p_front50["p_value"],
                    "p_front100_vs_pregen": p_front100["p_value"],
                    "p_full_vs_pregen": p_full["p_value"],
                })
                logger.info(
                    f"{dataset}/{module}/layer={layer}: pregen={summ['pregen']['mean']:.3f} "
                    f"front50={summ['front50']['mean']:.3f} (p={p_front50['p_value']:.4f}) "
                    f"front100={summ['front100']['mean']:.3f} (p={p_front100['p_value']:.4f}) "
                    f"full={summ['full']['mean']:.3f} (p={p_full['p_value']:.4f})"
                )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "optimal_window_check.csv")
    df.to_csv(csv_path, index=False)

    alpha = cfg["significance_alpha"]
    logger.info(f"\n=== Summary (alpha={alpha}) ===")
    for dataset, g in df.groupby("dataset"):
        n_front50_sig = (g["p_front50_vs_pregen"] < alpha).sum()
        n_front100_sig = (g["p_front100_vs_pregen"] < alpha).sum()
        n_full_sig = (g["p_full_vs_pregen"] < alpha).sum()
        logger.info(f"{dataset}: front50 significantly != pregen in {n_front50_sig}/{len(g)} cells; "
                     f"front100 in {n_front100_sig}/{len(g)}; full in {n_full_sig}/{len(g)}")

    logger.info(f"\nWrote {len(df)} rows to {csv_path}")
    return {"csv_path": csv_path}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                         default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()
    main(args.config)
