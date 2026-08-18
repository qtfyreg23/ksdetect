"""
run_01_template_shortcut_probe.py — exp_07, step 1 (only step for now).

Generalizes exp_06/run_03_shallow_shortcut_audit.py's pattern (same CPU-only,
repeated_cv_auc-based approach) to test the H_A vs H_C split from
docs/decisions.md D26: is MedQuad's real (non-length) shallow-layer signal
explainable by coarse question-type / template-keyword bucketing?

Three feature sets tested, each independently and combined:
  1. keyword_buckets (from config.yaml) — pure text-substring features,
     works regardless of what core.data's schema does or doesn't carry.
  2. question_type (raw MedQuAD categorical field, D8) — ONLY included if
     it can actually be recovered for these examples; see
     _try_load_question_types() below. If it can't be recovered, this
     script does NOT fail — it logs exactly what happened and proceeds
     with keyword_buckets alone, and the summary JSON records
     "question_type_available": false so this isn't silently missed later.
  3. length (word_length, char_length) — re-included from exp_06/run_03
     as a fixed reference point in the SAME run (already known ~0.56-0.59),
     so the new features' AUCs can be read against a familiar anchor
     without cross-referencing a different results file.

Does not compute an automated H_A/H_C verdict label (per the D23 lesson —
automated binary verdicts on this kind of question have been wrong before).
Reports raw AUCs; the H_A vs H_C call is made by a human reading the output
against the reference_real_signal_auc_range in config.yaml.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.data import load_generated_labels
from core.stats import repeated_cv_auc


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("exp_07_shortcut_probe")
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


def _try_load_question_types(example_ids: list[str], n_lookup: int, logger: logging.Logger) -> dict | None:
    """
    Best-effort attempt to recover the raw MedQuAD `question_type` field
    (docs/decisions.md D8) for the given example_ids, via
    core.data.load_dataset_examples("medquad", ...). This field is NOT
    guaranteed to survive core.data's unified 5-key schema
    (example_id/question/references/task_type/dataset, per
    experiments/exp_01_data_check/run.py's REQUIRED_KEYS) — we have not
    verified core/data/loaders.py's actual current implementation (see
    the .gitignore issue noted in docs/known_issues.md #11), so this
    function is written defensively: it tries, logs exactly what it finds,
    and returns None (not an exception) if the field isn't there, so the
    rest of the script can proceed with keyword_buckets alone.

    Returns: {example_id: question_type_string} if recoverable for at
    least SOME of the requested ids, else None.
    """
    try:
        from core.data import load_dataset_examples
    except ImportError as e:
        logger.warning(f"Could not import load_dataset_examples: {e}. "
                         f"Skipping question_type feature entirely.")
        return None

    try:
        raw_examples = load_dataset_examples("medquad", n_examples=n_lookup)
    except Exception as e:
        logger.warning(f"load_dataset_examples('medquad', n_examples={n_lookup}) "
                         f"failed: {type(e).__name__}: {e}. Skipping question_type feature.")
        return None

    if not raw_examples:
        logger.warning("load_dataset_examples('medquad', ...) returned 0 examples. "
                         "Skipping question_type feature.")
        return None

    sample = raw_examples[0]
    if "question_type" not in sample:
        logger.warning(
            f"Raw MedQuad examples do NOT carry a 'question_type' key "
            f"(keys present: {sorted(sample.keys())}). This means "
            f"core/data/loaders.py::load_medquad currently drops this "
            f"field when normalizing to the unified schema. Skipping "
            f"question_type feature — if you want it, load_medquad needs "
            f"a small patch to pass question_type through as an extra "
            f"key (this does not violate the REQUIRED_KEYS check in "
            f"exp_01, which only checks for missing keys, not extra ones)."
        )
        return None

    by_id = {ex["example_id"]: ex.get("question_type") for ex in raw_examples}
    requested_set = set(example_ids)
    found = {eid: by_id[eid] for eid in example_ids if eid in by_id and by_id[eid] is not None}
    missing = requested_set - set(found.keys())
    logger.info(f"question_type recovered for {len(found)}/{len(requested_set)} requested examples "
                 f"(missing: {len(missing)}).")
    if len(found) < len(requested_set):
        logger.warning(
            f"{len(missing)} example_ids from the labeled set were NOT found in "
            f"load_dataset_examples('medquad', n_examples={n_lookup})'s output — "
            f"either n_examples_for_question_type_lookup is too small (raise it in "
            f"config.yaml) or the loader's ordering/sampling differs between calls. "
            f"Proceeding with the {len(found)} examples that DID match; results "
            f"below are computed only on that subset for the question_type feature."
        )
    if not found:
        return None
    return found


def build_keyword_features(question: str, keyword_buckets: dict[str, list[str]]) -> dict:
    q_lower = question.lower()
    return {
        f"kw_{bucket_name}": float(any(kw in q_lower for kw in keywords))
        for bucket_name, keywords in keyword_buckets.items()
    }


def build_length_features(question: str) -> dict:
    return {"word_length": len(question.split()), "char_length": len(question)}


def one_hot(values: list[str]) -> tuple[np.ndarray, list[str]]:
    categories = sorted(set(values))
    idx = {c: i for i, c in enumerate(categories)}
    X = np.zeros((len(values), len(categories)), dtype=np.float32)
    for row, v in enumerate(values):
        X[row, idx[v]] = 1.0
    return X, categories


def run_feature_set(name: str, X: np.ndarray, y: np.ndarray, cfg: dict, logger: logging.Logger) -> dict:
    run = repeated_cv_auc(X, y, k=None, n_splits=cfg["n_splits"],
                            n_seeds=cfg["n_seeds"], base_seed=cfg["base_seed"])
    result = {k: v for k, v in run.items() if k != "pooled_aucs"}
    logger.info(f"  {name} (dim={X.shape[1]}): mean AUC = {run['mean']:.3f} "
                 f"(CI [{run['ci_lower']:.3f}, {run['ci_upper']:.3f}])")
    return result


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(os.path.join(output_dir, "run_log_shortcut_probe.txt"))
    logger.info("=== exp_07 run_01: MedQuad shallow-signal shortcut probe (H_A vs H_C) ===")
    logger.info(f"Reference real shallow-layer signal range (from D26 pt.3, for comparison "
                 f"only, not used in computation): {cfg['reference_real_signal_auc_range']}")

    records = load_generated_labels(cfg["coarse_scan_output_dir"], "medquad")
    example_ids = sorted(records.keys())
    questions = [records[eid]["question"] for eid in example_ids]
    y = np.array([records[eid]["hard_label"] for eid in example_ids], dtype=int)
    logger.info(f"n_examples={len(y)}, n_positive(hard_label=1)={y.sum()} ({100*y.mean():.1f}%)")

    results = {}

    # --- 1. length (reference anchor, re-run in this same script for
    #        convenience — matches exp_06/run_03's numbers, this is not a
    #        new claim, just avoiding a cross-file lookup) ---
    length_feats = [build_length_features(q) for q in questions]
    length_names = sorted(length_feats[0].keys())
    X_length = np.array([[fd[n] for n in length_names] for fd in length_feats], dtype=np.float32)
    logger.info("--- length features (reference anchor, expect ~0.54-0.59 per D26/exp_06) ---")
    results["length_combined"] = run_feature_set("length_combined", X_length, y, cfg, logger)

    # --- 2. keyword template buckets (H_A operationalized, schema-independent) ---
    kw_feats = [build_keyword_features(q, cfg["keyword_buckets"]) for q in questions]
    kw_names = sorted(kw_feats[0].keys())
    X_kw = np.array([[fd[n] for n in kw_names] for fd in kw_feats], dtype=np.float32)
    logger.info(f"--- keyword template buckets ({len(kw_names)} buckets: {kw_names}) ---")
    results["keyword_buckets_combined"] = run_feature_set("keyword_buckets_combined", X_kw, y, cfg, logger)
    for name, col in zip(kw_names, X_kw.T):
        results[f"keyword_{name}_only"] = run_feature_set(
            f"keyword_{name}_only", col.reshape(-1, 1), y, cfg, logger)

    # --- 3. question_type (H_A operationalized via the raw dataset field,
    #        best-effort — see _try_load_question_types docstring) ---
    qtype_by_id = _try_load_question_types(
        example_ids, cfg["n_examples_for_question_type_lookup"], logger)
    question_type_available = qtype_by_id is not None
    if question_type_available:
        matched_ids = [eid for eid in example_ids if eid in qtype_by_id]
        X_qtype, categories = one_hot([qtype_by_id[eid] for eid in matched_ids])
        y_matched = np.array([records[eid]["hard_label"] for eid in matched_ids], dtype=int)
        logger.info(f"--- question_type one-hot ({len(categories)} categories: {categories}) ---")
        results["question_type_onehot"] = run_feature_set(
            "question_type_onehot", X_qtype, y_matched, cfg, logger)
        results["question_type_onehot"]["n_examples_matched"] = len(matched_ids)
        results["question_type_onehot"]["categories"] = categories
    else:
        logger.info("--- question_type: NOT AVAILABLE, skipped (see warning above) ---")

    # --- 4. keyword buckets + length combined (best cheap-feature ceiling) ---
    X_combined = np.concatenate([X_kw, X_length], axis=1)
    logger.info("--- keyword buckets + length, combined ---")
    results["keyword_and_length_combined"] = run_feature_set(
        "keyword_and_length_combined", X_combined, y, cfg, logger)

    summary = {
        "n_examples": len(y),
        "n_positive_hard_label": int(y.sum()),
        "question_type_available": question_type_available,
        "reference_real_signal_auc_range": cfg["reference_real_signal_auc_range"],
        "results": results,
    }
    summary_path = os.path.join(output_dir, "shortcut_probe_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote {summary_path}")

    logger.info("\n=== How to read this (per docs/decisions.md D26) ===")
    logger.info(
        "H_A (coarse template/category shortcut) is SUPPORTED if "
        "question_type_onehot or keyword_buckets_combined's CI reaches "
        "close to the reference real-signal range above. H_C (finer-grained, "
        "not reducible to these cheap features) is SUPPORTED if all of the "
        "above stay well below that range, closer to the length-only "
        "ceiling (~0.54-0.59). This call is made by you reading the numbers "
        "above, not by this script — no automated verdict label is written, "
        "per the D23 lesson about automated H1/H2-style verdicts."
    )

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                         default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()
    main(args.config)