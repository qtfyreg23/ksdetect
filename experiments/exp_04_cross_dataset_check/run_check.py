"""
run_check.py — exp_04_cross_dataset_check.

Cheap, CPU-only, no-new-extraction directional check: does a dataset's
"how stylistically similar are correct and incorrect answers" predict how
much signal posthoc loses relative to pregen, ACROSS datasets (not just
within MedQuad)? Supportive evidence here would elevate explanation H2
from "a MedQuad quirk" to "a general, predictable mechanism" — which is
what would make it a real paper-level contribution rather than a one-off
footnote (see the discussion this experiment follows from).

Style-homogeneity proxy (crude, deliberately simple — a more refined
version could use embedding-space similarity or an LLM-judged style
score, but that adds cost/complexity this quick check doesn't need):
Cohen's d effect size between the answer-length distributions of
CORRECT (hard_label=0) vs INCORRECT (hard_label=1) greedy answers.
  - |d| near 0: correct and incorrect answers are similar lengths —
    length alone can't tell them apart — consistent with "stylistically
    homogeneous" in the crude sense this proxy can measure.
  - |d| large: correct and incorrect answers differ systematically in
    length — some surface distinguishability exists.

This is NOT the same "style" as content/tone (which would need a richer
measure), but it's cheap, uses data already on disk, and gives a first
directional read.
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

from core.data import load_generated_labels


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("exp_04_check")
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


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Cohen's d (pooled std) between two samples."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return float("nan")
    return (a.mean() - b.mean()) / pooled_std


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(os.path.join(output_dir, "run_log.txt"))
    logger.info("=== exp_04: cross-dataset style-homogeneity vs posthoc-gap check ===")

    # --- Step 1: style-homogeneity proxy per dataset ---
    style_rows = []
    for dataset in cfg["datasets"]:
        records = load_generated_labels(cfg["coarse_scan_output_dir"], dataset)
        lengths_correct = np.array([
            len(r["greedy_answer"].split()) for r in records.values() if r["hard_label"] == 0
        ])
        lengths_incorrect = np.array([
            len(r["greedy_answer"].split()) for r in records.values() if r["hard_label"] == 1
        ])
        d = cohens_d(lengths_correct, lengths_incorrect)
        style_rows.append({
            "dataset": dataset,
            "n_correct": len(lengths_correct), "n_incorrect": len(lengths_incorrect),
            "mean_len_correct": float(lengths_correct.mean()) if len(lengths_correct) else float("nan"),
            "mean_len_incorrect": float(lengths_incorrect.mean()) if len(lengths_incorrect) else float("nan"),
            "abs_cohens_d": abs(d),
        })
        logger.info(f"{dataset}: n_correct={len(lengths_correct)} n_incorrect={len(lengths_incorrect)} "
                     f"mean_len_correct={style_rows[-1]['mean_len_correct']:.1f} "
                     f"mean_len_incorrect={style_rows[-1]['mean_len_incorrect']:.1f} "
                     f"|Cohen's d|={abs(d):.3f}")

    style_df = pd.DataFrame(style_rows)

    # --- Step 2: known pregen-vs-posthoc_answermean gap per dataset, from exp_03 ---
    disc_df = pd.read_csv(cfg["exp03_discrimination_csv"])
    disc_df["gap"] = disc_df["mean_pregen"] - disc_df["mean_posthoc_answermean"]
    gap_by_dataset = disc_df.groupby("dataset")["gap"].mean().reset_index()
    gap_by_dataset.columns = ["dataset", "mean_gap"]
    logger.info("\nMean pregen-vs-posthoc_answermean gap per dataset (from exp_03):")
    for _, row in gap_by_dataset.iterrows():
        logger.info(f"  {row['dataset']}: {row['mean_gap']:+.4f}")

    # --- Step 3: correlate ---
    merged = style_df.merge(gap_by_dataset, on="dataset")
    logger.info("\n=== Merged table ===")
    logger.info("\n" + merged[["dataset", "abs_cohens_d", "mean_gap"]].to_string(index=False))

    if len(merged) >= 3:
        corr = merged["abs_cohens_d"].corr(merged["mean_gap"])
    else:
        corr = float("nan")

    logger.info(f"\nCorrelation(|Cohen's d| [style DIFFERENCE], mean_gap) = {corr:.3f}")
    logger.info("Interpretation guide (directional only, n=5, NOT a confirmatory statistical test):")
    logger.info("  - H2 predicts: smaller |Cohen's d| (more style-homogeneous) -> BIGGER gap.")
    logger.info("    I.e. a NEGATIVE correlation between |Cohen's d| and mean_gap supports H2")
    logger.info("    generalizing beyond MedQuad.")
    logger.info("  - A near-zero or positive correlation, OR a pattern where MedQuad is simply")
    logger.info("    an outlier unrelated to its |Cohen's d| value, would suggest H2 (at least in")
    logger.info("    this crude length-based operationalization) does NOT generalize cleanly, and")
    logger.info("    MedQuad's effect may have a more idiosyncratic explanation instead.")

    merged.to_csv(os.path.join(output_dir, "style_vs_gap.csv"), index=False)
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "merged_table": merged.to_dict(orient="records"),
            "correlation_abs_cohens_d_vs_gap": None if np.isnan(corr) else float(corr),
        }, f, indent=2)

    logger.info(f"\nResults written to {output_dir}/style_vs_gap.csv and summary.json")
    return {"correlation": corr, "merged": merged}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                         default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()
    main(args.config)
