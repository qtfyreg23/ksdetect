"""
run_02_compare.py — exp_05, step 2: the H1-vs-H2 discrimination test.

For each (dataset, module, layer) — module="embedding" excluded, same
reasoning as exp_03 — loads SIX matched feature sets on the identical set
of examples:
  - "pregen"              (exp_02, reused)
  - "posthoc_last"         (exp_02, reused)
  - "posthoc_full"         (exp_03's full rescan, reused — whole-answer
                             answer_mean pooling)
  - "posthoc_front20"      (exp_05, new)
  - "posthoc_front50"      (exp_05, new)
  - "posthoc_front100"     (exp_05, new)

Runs matched_repeated_cv (same folds across all six) at k=None, then:
  - paired test: front20 vs posthoc_full
  - paired test: front20 vs pregen

Decision rule (fixed in advance, mirrors exp_03's discipline):
  - H1_supported (dilution): front20 is significantly WORSE than
    posthoc_full (p < alpha AND mean_front20 meaningfully lower) — i.e.
    more averaging genuinely recovers more signal, consistent with
    informative content being spread out / diluted by naive full-answer
    averaging.
  - H2_supported (content/style masking): front20 is NOT significantly
    different from posthoc_full (p >= alpha, or the mean difference is
    small relative to the pregen-vs-full gap) — i.e. the degradation is
    already fully present within the first 20 tokens, more context doesn't
    help, consistent with a masking effect that isn't about WHERE you look
    but about the fact that the model has started speaking at all.
  - ambiguous: doesn't cleanly fit either pattern (e.g. non-monotonic
    across window sizes) — reported as such rather than forced into H1/H2.
"""

from __future__ import annotations

import gc
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
from core.run_utils import check_or_write_config_hash


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("exp_05_compare")
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


def align_by_ids(arrays_and_ids: dict) -> tuple[dict, list]:
    """arrays_and_ids: {name: (X, ids)}. Returns ({name: X_reindexed}, common_ids)."""
    id_sets = [set(ids) for _, ids in arrays_and_ids.values()]
    common = sorted(set.intersection(*id_sets))
    out = {}
    for name, (X, ids) in arrays_and_ids.items():
        idx = {eid: i for i, eid in enumerate(ids)}
        rows = [idx[eid] for eid in common]
        out[name] = X[rows]
    return out, common


def verdict(mean_full, mean_front20, p_front20_vs_full, mean_pregen, alpha):
    total_gap = mean_pregen - mean_full
    if abs(total_gap) < 1e-6:
        return "no_gap_to_explain"
    front20_gap = mean_pregen - mean_front20
    # H1: front20 recovers meaningfully LESS of the pregen signal than full does
    # (i.e. front20 is significantly worse than full).
    if p_front20_vs_full < alpha and mean_front20 < mean_full - 0.01:
        return "H1_supported"
    # H2: front20 is statistically indistinguishable from full (or even
    # better) — degradation already complete within the first 20 tokens.
    if p_front20_vs_full >= alpha or mean_front20 >= mean_full - 0.005:
        return "H2_supported"
    return "ambiguous"


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(os.path.join(output_dir, "run_log_compare.txt"))
    logger.info("=== exp_05 run_02: H1 (dilution) vs H2 (style masking) ===")
    check_or_write_config_hash(output_dir, cfg, logger)

    modules_to_test = [m for m in cfg["modules"] if m != "embedding"]
    ks = cfg["front_window_k_values"]

    jsonl_path = os.path.join(output_dir, "h1_h2_results.jsonl")
    done_keys = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_keys.add(json.loads(line)["_key"])
        logger.info(f"Resume: found {len(done_keys)} already-computed cells")

    result_file = open(jsonl_path, "a", encoding="utf-8")
    n_written, n_skipped, n_errored = 0, 0, 0

    for dataset in cfg["datasets"]:
        records = load_generated_labels(cfg["coarse_scan_output_dir"], dataset)

        for module in modules_to_test:
            for layer in cfg["layers_coarse"]:
                key = f"{dataset}|{module}|{layer}"
                if key in done_keys:
                    n_skipped += 1
                    continue

                try:
                    raw = {}
                    raw["pregen"] = load_activation(
                        cfg["coarse_scan_activation_dir"], dataset, "pregen", module, layer)
                    raw["posthoc_last"] = load_activation(
                        cfg["coarse_scan_activation_dir"], dataset, "posthoc", module, layer)
                    raw["posthoc_full"] = load_activation(
                        cfg["exp03_activation_dir"], dataset, "posthoc_answermean", module, layer)
                    for k in ks:
                        raw[f"posthoc_front{k}"] = load_activation(
                            cfg["activation_dir"], dataset, f"posthoc_front{k}", module, layer)
                except FileNotFoundError as e:
                    logger.error(f"Skipping {dataset}/{module}/layer={layer}: {e}")
                    n_errored += 1
                    continue

                aligned, common_ids = align_by_ids(raw)
                y = np.array([records[eid]["hard_label"] for eid in common_ids], dtype=int)

                try:
                    result = matched_repeated_cv(
                        feature_sets=aligned, y=y, k=None,
                        n_splits=cfg["n_splits"], n_seeds=cfg["n_seeds"],
                        base_seed=cfg["base_seed"],
                    )
                    pooled = result["pooled_aucs"]
                    summ = result["summary"]

                    p_front20_vs_full = paired_wilcoxon(
                        pooled["posthoc_front20"], pooled["posthoc_full"])
                    p_front20_vs_pregen = paired_wilcoxon(
                        pooled["posthoc_front20"], pooled["pregen"])

                    v = verdict(
                        summ["posthoc_full"]["mean"], summ["posthoc_front20"]["mean"],
                        p_front20_vs_full["p_value"], summ["pregen"]["mean"],
                        cfg["significance_alpha"],
                    )

                    row = {
                        "_key": key, "dataset": dataset, "module": module, "layer": layer,
                        "n_examples": len(common_ids),
                        "mean_pregen": summ["pregen"]["mean"],
                        "mean_posthoc_last": summ["posthoc_last"]["mean"],
                        "mean_posthoc_full": summ["posthoc_full"]["mean"],
                        "mean_posthoc_front20": summ["posthoc_front20"]["mean"],
                        "mean_posthoc_front50": summ["posthoc_front50"]["mean"],
                        "mean_posthoc_front100": summ["posthoc_front100"]["mean"],
                        "p_front20_vs_full": p_front20_vs_full["p_value"],
                        "p_front20_vs_pregen": p_front20_vs_pregen["p_value"],
                        "verdict": v,
                    }
                    result_file.write(json.dumps(row) + "\n")
                    result_file.flush()
                    n_written += 1
                    logger.info(
                        f"{dataset}/{module}/layer={layer}: pregen={row['mean_pregen']:.3f} "
                        f"front20={row['mean_posthoc_front20']:.3f} "
                        f"front50={row['mean_posthoc_front50']:.3f} "
                        f"front100={row['mean_posthoc_front100']:.3f} "
                        f"full={row['mean_posthoc_full']:.3f} "
                        f"p(front20 vs full)={p_front20_vs_full['p_value']:.4f} -> {v}"
                    )
                except Exception as e:
                    n_errored += 1
                    logger.error(f"{dataset}/{module}/layer={layer} FAILED: "
                                  f"{type(e).__name__}: {e}")
                finally:
                    aligned.clear()
                    raw.clear()
                    gc.collect()

    result_file.close()

    all_rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_rows.append(json.loads(line))
    df = pd.DataFrame(all_rows).drop(columns=["_key"])
    csv_path = os.path.join(output_dir, "h1_h2_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"=== Run finished: written={n_written} skipped={n_skipped} "
                 f"errored={n_errored} === Wrote {len(df)} rows to {csv_path}")

    logger.info("\n=== VERDICT SUMMARY ===")
    counts = df["verdict"].value_counts()
    for v, c in counts.items():
        logger.info(f"  {v}: {c} / {len(df)} ({100*c/len(df):.1f}%)")
    logger.info("\n=== Per-dataset breakdown ===")
    for dataset, g in df.groupby("dataset"):
        logger.info(f"  {dataset}: {g['verdict'].value_counts().to_dict()}")

    summary_path = os.path.join(output_dir, "verdict_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Verdict counts:\n" + counts.to_string())
        f.write("\n\nPer-dataset breakdown:\n")
        for dataset, g in df.groupby("dataset"):
            f.write(f"\n{dataset}:\n{g['verdict'].value_counts().to_string()}\n")
        f.write("\n\nWindow-size trend (mean AUC across all cells per dataset):\n")
        for dataset, g in df.groupby("dataset"):
            f.write(f"\n{dataset}: pregen={g['mean_pregen'].mean():.3f} "
                     f"front20={g['mean_posthoc_front20'].mean():.3f} "
                     f"front50={g['mean_posthoc_front50'].mean():.3f} "
                     f"front100={g['mean_posthoc_front100'].mean():.3f} "
                     f"full={g['mean_posthoc_full'].mean():.3f}\n")
    logger.info(f"Verdict summary written to {summary_path}")

    return {"csv_path": csv_path, "summary_path": summary_path, "n_rows": len(df),
            "n_errored": n_errored}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                         default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()
    result = main(args.config)
    sys.exit(1 if result["n_errored"] > 0 else 0)
