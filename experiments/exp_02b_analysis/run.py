"""
run.py — exp_02b_analysis.

For every (dataset, stage, module, layer) cell:
  1. Load activations via core.extraction.load_activation.
  2. Load labels via core.data.load_generated_labels, align to the
     activation's example_ids order (this alignment step is why we always
     load ids alongside X, never assume ordering matches across separately
     loaded arrays).
  3. For each label_type ("hard" | "soft_binarized") build y.
  4. For each k in k_values, run core.stats.repeated_cv_auc(X, y, k=k, ...).
  5. Append one result row to results/wide_scan_results.jsonl (resumable:
     a cell is already-done if a row with the same key already exists in
     that file).

At the end, compiles wide_scan_results.jsonl into a pandas DataFrame
(results/wide_scan_results.csv) and — for label_type="hard", k=None
(full representation) — produces one heatmap per (dataset, stage) with
rows=module, columns=layer (core.viz.heatmap), as the "current state of
the wide scan" visual for discussion. This step draws NO conclusions about
which region is "interesting" — that is Stage 2, a separate discussion.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.extraction import load_activation
from core.data import load_generated_labels
from core.stats import repeated_cv_auc_multi_k
from core.viz import heatmap, save_figure


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("exp_02b_analysis")
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


def cell_key(dataset, stage, module, layer, label_type, k) -> str:
    return f"{dataset}|{stage}|{module}|{layer}|{label_type}|{k}"


def build_y(records: dict, example_ids: list[str], label_type: str, threshold: float) -> np.ndarray:
    """
    records: example_id -> label record (from load_generated_labels).
    example_ids: the ORDER the activations are in — y[i] must correspond to
    the example at example_ids[i], not iteration order over `records`.
    """
    if label_type == "hard":
        return np.array([records[eid]["hard_label"] for eid in example_ids], dtype=int)
    elif label_type == "soft_binarized":
        return np.array(
            [int(records[eid]["soft_label"] > threshold) for eid in example_ids], dtype=int,
        )
    else:
        raise ValueError(f"Unknown label_type: {label_type!r}")


def layers_for_module(module: str, layers_coarse: list[int]) -> list[int]:
    """
    core.extraction.hooks.ActivationCollector only ever records the
    embedding module ONCE per forward pass, under layer index -1 (see
    hooks.py's SUPPORTED_MODULES / _make_output_hook usage) — embedding
    activations don't have a "depth" the way attention/ffn/residual do.
    Iterating cfg["layers"] (e.g. [0,4,8,...,31]) for module="embedding"
    would look for shards that were never written at any of those layer
    indices and silently waste time on FileNotFoundError for 9/9 of them.
    This function is the single place that encodes "embedding only exists
    at layer -1" so the cell grid doesn't generate cells that can never
    exist.
    """
    if module == "embedding":
        return [-1]
    return layers_coarse


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(os.path.join(output_dir, "..", "run_log.txt"))
    logger.info("=== exp_02b_analysis ===")
    logger.info(f"Config: {json.dumps(cfg, ensure_ascii=False)}")

    results_path = os.path.join(output_dir, "wide_scan_results.jsonl")
    done_keys = set()
    if cfg["resume"] and os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    done_keys.add(row["_key"])
        logger.info(f"Resume: found {len(done_keys)} already-computed cells in {results_path}")

    # Build the full list of (dataset, stage, module, layer) cells, then for
    # each cell iterate label_types x k_values — grouping this way means we
    # load activations/labels ONCE per (dataset, stage, module, layer), not
    # once per (label_type, k) combination.
    cells = [
        (d, s, m, l)
        for d in cfg["datasets"]
        for s in cfg["stages"]
        for m in cfg["modules"]
        for l in layers_for_module(m, cfg["layers"])
    ]
    n_cells_total = len(cells)
    n_subcells_total = n_cells_total * len(cfg["label_types"]) * len(cfg["k_values"])
    logger.info(f"{n_cells_total} (dataset,stage,module,layer) cells x "
                 f"{len(cfg['label_types'])} label_types x {len(cfg['k_values'])} k_values "
                 f"= {n_subcells_total} total sub-cells")

    n_written, n_skipped, n_errored = 0, 0, 0
    t_start = time.time()

    # Cache labels per dataset (labels don't depend on stage — pregen vs
    # posthoc only changes which activations we read, not the labels) so we
    # don't re-read the JSONL shards for every stage/module/layer of the
    # same dataset.
    labels_cache: dict[str, dict] = {}

    result_file = open(results_path, "a", encoding="utf-8")

    for cell_idx, (dataset, stage, module, layer) in enumerate(cells):
        if dataset not in labels_cache:
            labels_cache[dataset] = load_generated_labels(
                cfg["coarse_scan_output_dir"], dataset,
            )
        records = labels_cache[dataset]

        try:
            X, example_ids = load_activation(
                cfg["activation_dir"], dataset, stage, module, layer,
            )
        except FileNotFoundError as e:
            logger.error(f"Skipping cell (activations missing): {e}")
            n_errored += len(cfg["label_types"]) * len(cfg["k_values"])
            continue

        for label_type in cfg["label_types"]:
            y = build_y(records, example_ids, label_type,
                         cfg["soft_label_binarize_threshold"])

            keys_for_this_label = {
                k: cell_key(dataset, stage, module, layer, label_type, k)
                for k in cfg["k_values"]
            }
            if all(key in done_keys for key in keys_for_this_label.values()):
                n_skipped += len(cfg["k_values"])
                continue

            def _heartbeat(seed_idx, n_seeds_total, seed_elapsed):
                logger.info(
                    f"  ... {dataset}/{stage}/{module}/layer={layer}/{label_type}: "
                    f"seed {seed_idx}/{n_seeds_total} done in {seed_elapsed:.1f}s "
                    f"(dim={X.shape[1]}, still running — this is a heartbeat, not "
                    f"a new cell)"
                )

            try:
                # ONE call computes ALL k_values together, sharing the
                # per-fold feature ranking across them (docs/known_issues.md
                # #9) instead of the old per-k loop that recomputed it.
                multi_k_result = repeated_cv_auc_multi_k(
                    X, y, k_values=cfg["k_values"],
                    selection_method=cfg["selection_method"],
                    n_splits=cfg["n_splits"], n_seeds=cfg["n_seeds"],
                    base_seed=cfg["base_seed"], on_seed_done=_heartbeat,
                )
            except Exception as e:
                n_errored += len(cfg["k_values"])
                logger.error(
                    f"{dataset}/{stage}/{module}/layer={layer}/{label_type} "
                    f"(all k values) FAILED: {type(e).__name__}: {e}"
                )
                continue

            for k in cfg["k_values"]:
                key = keys_for_this_label[k]
                if key in done_keys:
                    n_skipped += 1
                    continue
                run = multi_k_result[k]
                row = {
                    "_key": key,
                    "dataset": dataset, "stage": stage, "module": module,
                    "layer": layer, "label_type": label_type, "k": k,
                    "n_examples": X.shape[0], "dim": X.shape[1],
                    "mean": run["mean"], "std": run["std"],
                    "ci_lower": run["ci_lower"], "ci_upper": run["ci_upper"],
                    "n_seeds": run["n_seeds"], "n_splits": run["n_splits"],
                }
                result_file.write(json.dumps(row) + "\n")
                result_file.flush()
                n_written += 1

        if (cell_idx + 1) % cfg["progress_log_every_n_cells"] == 0 or cell_idx == n_cells_total - 1:
            elapsed = time.time() - t_start
            done_subcells = n_written + n_skipped
            rate = done_subcells / elapsed if elapsed > 0 else 0.0
            remaining = n_subcells_total - done_subcells - n_errored
            eta_min = (remaining / rate / 60) if rate > 0 else float("nan")
            logger.info(
                f"progress: cells {cell_idx+1}/{n_cells_total} | "
                f"subcells written={n_written} skipped={n_skipped} errored={n_errored} "
                f"pending~={remaining} | rate={rate:.2f} subcells/s | ETA={eta_min:.1f} min"
            )

    result_file.close()
    total_elapsed = time.time() - t_start
    logger.info(f"=== Grid computation finished in {total_elapsed/60:.1f} min "
                 f"(written={n_written}, skipped={n_skipped}, errored={n_errored}) ===")

    # --- Compile CSV ---
    all_rows = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_rows.append(json.loads(line))
    df = pd.DataFrame(all_rows).drop(columns=["_key"])
    csv_path = os.path.join(output_dir, "wide_scan_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Wrote {len(df)} rows to {csv_path}")

    # --- Heatmaps: label_type="hard", k=None (full representation), one per (dataset, stage) ---
    fig_paths = []
    subset = df[(df["label_type"] == "hard") & (df["k"].isna())]
    for (dataset, stage), group in subset.groupby(["dataset", "stage"]):
        modules_present = sorted(group["module"].unique())
        layers_present = sorted(group["layer"].unique())
        matrix = np.full((len(modules_present), len(layers_present)), np.nan)
        for _, row in group.iterrows():
            i = modules_present.index(row["module"])
            j = layers_present.index(row["layer"])
            matrix[i, j] = row["mean"]

        fig, ax = heatmap(
            matrix,
            row_labels=modules_present,
            col_labels=[str(l) for l in layers_present],
            title=f"{dataset} ({stage}) - Full Representation AUC",
            cbar_label="Mean AUC (pooled folds x seeds)",
        )
        fig_path = os.path.join(output_dir, f"fig_heatmap_{dataset}_{stage}.pdf")
        save_figure(fig, fig_path)
        fig_paths.append(fig_path)

    logger.info(f"Wrote {len(fig_paths)} heatmap figures.")
    logger.info("=== exp_02b_analysis DONE. No conclusions drawn — "
                 "see wide_scan_results.csv and the heatmaps for Stage 2 discussion. ===")

    return {"n_written": n_written, "n_skipped": n_skipped, "n_errored": n_errored,
            "csv_path": csv_path, "fig_paths": fig_paths}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                         default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()
    main(args.config)
