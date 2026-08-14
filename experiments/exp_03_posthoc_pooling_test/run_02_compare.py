"""
run_02_compare.py — Stage 3, step 2: the actual discrimination test.

For each (dataset, module, layer) — module="embedding" excluded, see note
below — loads three MATCHED feature sets on the identical set of examples:
  - "pregen": from exp_02_coarse_scan (existing, reused)
  - "posthoc_last": from exp_02_coarse_scan (existing, reused, "last token"
     pooling — this is what showed up as weaker than pregen in Stage 1)
  - "posthoc_answermean": from exp_03's run_01 (new, "answer_mean" pooling)

Runs core.stats.matched_repeated_cv (same folds across all three, required
for a valid paired test) at k=None (full representation — this test is
about POOLING STRATEGY, not sparsity), then two paired Wilcoxon tests:
  - posthoc_answermean vs pregen
  - posthoc_answermean vs posthoc_last

Applies the decision rule fixed in advance (config.yaml's description /
docs/decisions.md D17):
  - A_supported: posthoc_answermean's mean >= pregen's mean, OR the paired
    test posthoc_answermean-vs-pregen is not significant (p >= alpha).
    -> the original "posthoc < pregen" gap looks like a pooling artifact.
  - B_supported: posthoc_answermean is STILL significantly below pregen,
    AND does not show a significant improvement over posthoc_last either.
    -> the gap looks like a real representational effect, not a pooling
    artifact.
  - mixed: posthoc_answermean is still significantly below pregen, BUT is
    significantly better than posthoc_last. -> partial artifact, partial
    real effect; both are probably contributing.

NOTE on module="embedding": excluded from this analysis. Pregen+embedding
is a KNOWN degenerate cell (constant vector, AUC exactly 0.5, see the
Stage-1 discussion) because of the fixed chat-template suffix token at the
"last" pooling position — comparing against a degenerate, contentless
baseline would not test the actual phenomenon (which is about REAL
representational content), so it's skipped here rather than reported
alongside the other four modules.
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
    logger = logging.getLogger("exp_03_compare")
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


def align_by_ids(X_a, ids_a, X_b, ids_b, X_c, ids_c):
    """Reindex all three arrays to the SAME sorted-intersection example_id
    order, so matched_repeated_cv's fold-sharing assumption (same y, same
    row order) actually holds across all three feature sets."""
    common = sorted(set(ids_a) & set(ids_b) & set(ids_c))
    idx_a = {eid: i for i, eid in enumerate(ids_a)}
    idx_b = {eid: i for i, eid in enumerate(ids_b)}
    idx_c = {eid: i for i, eid in enumerate(ids_c)}
    rows_a = [idx_a[eid] for eid in common]
    rows_b = [idx_b[eid] for eid in common]
    rows_c = [idx_c[eid] for eid in common]
    return X_a[rows_a], X_b[rows_b], X_c[rows_c], common


def verdict(mean_pregen, mean_posthoc_last, mean_answermean, p_vs_pregen, p_vs_last, alpha):
    if mean_answermean >= mean_pregen or p_vs_pregen >= alpha:
        return "A_supported"
    if mean_answermean > mean_posthoc_last and p_vs_last < alpha:
        return "mixed"
    return "B_supported"


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(os.path.join(output_dir, "run_log_compare.txt"))
    logger.info("=== exp_03 run_02: pregen vs posthoc_last vs posthoc_answermean ===")

    check_or_write_config_hash(output_dir, cfg, logger)

    modules_to_test = [m for m in cfg["modules"] if m != "embedding"]

    # --- Resumability: results are written incrementally to a JSONL file,
    # one line per (dataset, module, layer) cell, immediately after that
    # cell finishes — NOT batched up and written only at the very end. If
    # the process dies partway (e.g. OOM on a high-dim module), whatever
    # was already written survives, and restarting skips those cells
    # instead of recomputing them (see docs/known_issues.md #10). This was
    # missing from the first version of this script — added after a real
    # ~25-minute run was lost to an OOM kill with nothing to show for it.
    jsonl_path = os.path.join(output_dir, "discrimination_results.jsonl")
    done_keys = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_keys.add(json.loads(line)["_key"])
        logger.info(f"Resume: found {len(done_keys)} already-computed cells in {jsonl_path}")

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
                    activations = {}
                    activations["X_pregen"], ids_pregen = load_activation(
                        cfg["coarse_scan_activation_dir"], dataset, "pregen", module, layer,
                    )
                    activations["X_last"], ids_last = load_activation(
                        cfg["coarse_scan_activation_dir"], dataset, "posthoc", module, layer,
                    )
                    activations["X_mean"], ids_mean = load_activation(
                        cfg["activation_dir"], dataset, "posthoc_answermean", module, layer,
                    )
                except FileNotFoundError as e:
                    logger.error(f"Skipping {dataset}/{module}/layer={layer} "
                                  f"(activations missing): {e}")
                    n_errored += 1
                    continue

                activations["X_pregen"], activations["X_last"], activations["X_mean"], common_ids = align_by_ids(
                    activations["X_pregen"], ids_pregen,
                    activations["X_last"], ids_last,
                    activations["X_mean"], ids_mean,
                )
                y = np.array([records[eid]["hard_label"] for eid in common_ids], dtype=int)

                try:
                    result = matched_repeated_cv(
                        feature_sets={"pregen": activations["X_pregen"],
                                       "posthoc_last": activations["X_last"],
                                       "posthoc_answermean": activations["X_mean"]},
                        y=y, k=None, n_splits=cfg["n_splits"], n_seeds=cfg["n_seeds"],
                        base_seed=cfg["base_seed"],
                    )
                    pooled = result["pooled_aucs"]
                    summary = result["summary"]

                    p_vs_pregen = paired_wilcoxon(pooled["posthoc_answermean"], pooled["pregen"])
                    p_vs_last = paired_wilcoxon(pooled["posthoc_answermean"], pooled["posthoc_last"])

                    v = verdict(
                        summary["pregen"]["mean"], summary["posthoc_last"]["mean"],
                        summary["posthoc_answermean"]["mean"],
                        p_vs_pregen["p_value"], p_vs_last["p_value"],
                        cfg["significance_alpha"],
                    )

                    row = {
                        "_key": key,
                        "dataset": dataset, "module": module, "layer": layer,
                        "n_examples": len(common_ids),
                        "mean_pregen": summary["pregen"]["mean"],
                        "mean_posthoc_last": summary["posthoc_last"]["mean"],
                        "mean_posthoc_answermean": summary["posthoc_answermean"]["mean"],
                        "ci_lower_answermean": summary["posthoc_answermean"]["ci_lower"],
                        "ci_upper_answermean": summary["posthoc_answermean"]["ci_upper"],
                        "p_answermean_vs_pregen": p_vs_pregen["p_value"],
                        "p_answermean_vs_posthoc_last": p_vs_last["p_value"],
                        "verdict": v,
                    }
                    result_file.write(json.dumps(row) + "\n")
                    result_file.flush()
                    n_written += 1
                    logger.info(
                        f"{dataset}/{module}/layer={layer}: pregen={row['mean_pregen']:.3f} "
                        f"posthoc_last={row['mean_posthoc_last']:.3f} "
                        f"posthoc_answermean={row['mean_posthoc_answermean']:.3f} "
                        f"p_vs_pregen={p_vs_pregen['p_value']:.4f} "
                        f"p_vs_last={p_vs_last['p_value']:.4f} -> {v}"
                    )
                except Exception as e:
                    n_errored += 1
                    logger.error(f"{dataset}/{module}/layer={layer} FAILED: "
                                  f"{type(e).__name__}: {e}")
                finally:
                    # Explicit cleanup after EVERY cell, success or failure —
                    # this is the CPU/RAM analogue of the GPU
                    # torch.cuda.empty_cache() fix from the extraction stage
                    # (docs/known_issues.md #8/#9). High-dimensional modules
                    # (ffn_neuron, 14336-dim) hold large intermediate arrays
                    # during sklearn's L-BFGS fit; on a memory-constrained
                    # instance (this project's CPU analysis machine has only
                    # 2GB RAM) explicitly clearing the activations dict and
                    # forcing a collection between cells measurably reduces
                    # peak usage compared to relying on Python's default GC
                    # timing. (Note: `del locals()[name]` was tried first
                    # and does NOT reliably free bindings in CPython —
                    # using an explicit dict + .clear() instead, which does.)
                    activations.clear()
                    gc.collect()

    result_file.close()

    # --- Compile CSV + summary from the (now-complete, or resumed-complete) JSONL ---
    all_rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_rows.append(json.loads(line))
    df = pd.DataFrame(all_rows).drop(columns=["_key"])
    csv_path = os.path.join(output_dir, "discrimination_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"=== Run finished: written={n_written} skipped(resumed)={n_skipped} "
                 f"errored={n_errored} === Wrote {len(df)} total rows to {csv_path}")

    logger.info("\n=== VERDICT SUMMARY (cell counts) ===")
    counts = df["verdict"].value_counts()
    for v, c in counts.items():
        logger.info(f"  {v}: {c} / {len(df)} cells ({100*c/len(df):.1f}%)")

    logger.info("\n=== Per-dataset verdict breakdown ===")
    for dataset, g in df.groupby("dataset"):
        logger.info(f"  {dataset}: {g['verdict'].value_counts().to_dict()}")

    summary_path = os.path.join(output_dir, "verdict_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Verdict counts (all dataset x module x layer cells):\n")
        f.write(counts.to_string())
        f.write("\n\nPer-dataset breakdown:\n")
        for dataset, g in df.groupby("dataset"):
            f.write(f"\n{dataset}:\n{g['verdict'].value_counts().to_string()}\n")
    logger.info(f"Verdict summary written to {summary_path}")

    if n_errored > 0:
        logger.warning(f"{n_errored} cell(s) errored — check the ERROR lines above/in the log "
                         f"and re-run this script (resume will only retry missing cells).")

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
