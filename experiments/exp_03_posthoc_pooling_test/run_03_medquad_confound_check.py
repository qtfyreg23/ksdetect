"""
run_03_medquad_confound_check.py — cheap, CPU-only, no-new-extraction audit
of an alternative explanation for MedQuad's persistent pregen>posthoc gap.

Explanation B (kept from exp_03): a real representational masking effect.
Explanation C (new, being tested here): pregen's apparent signal is mostly a
SURFACE-LEVEL SHORTCUT (e.g. question length, which correlates with how
"standard" vs "obscure" a condition's write-up is) rather than genuine
knowledge-sufficiency signal — and posthoc's generated boilerplate simply
doesn't preserve that shortcut as cleanly.

Test: build a trivial classifier using ONLY question length (word count) as
a single feature, run it through the SAME repeated_cv_auc machinery used
everywhere else in this project, and compare its AUC against pregen's real
~0.83-0.86. If the length-only shortcut gets anywhere close to that, it's a
red flag that pregen's real signal may substantially be doing something
similarly shallow. If it's near chance (~0.5), this particular shortcut
explanation is ruled out (though more subtle shortcuts — e.g. disease-name
corpus frequency — would need separate, more expensive checks not covered
here).

This does NOT need a GPU or the model — it only reads already-generated
labels (question text + hard_label).
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
from core.stats import repeated_cv_auc, ci_lower_bound_exceeds


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("exp_03_confound_check")
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


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(os.path.join(output_dir, "run_log_confound_check.txt"))
    logger.info("=== run_03: MedQuad surface-shortcut confound check ===")

    records = load_generated_labels(cfg["coarse_scan_output_dir"], "medquad")
    example_ids = sorted(records.keys())

    # Two cheap surface features, tried both separately and together:
    #   - question length in words
    #   - question length in characters
    # (deliberately NOT more sophisticated features like disease-name
    # corpus frequency — that would need external data and is out of scope
    # for this quick audit; if this check does NOT clear the shortcut
    # explanation, that more expensive check becomes worth doing.)
    word_lens = np.array([[len(records[eid]["question"].split())] for eid in example_ids],
                           dtype=np.float32)
    char_lens = np.array([[len(records[eid]["question"])] for eid in example_ids],
                           dtype=np.float32)
    both = np.concatenate([word_lens, char_lens], axis=1)
    y = np.array([records[eid]["hard_label"] for eid in example_ids], dtype=int)

    logger.info(f"n_examples={len(y)}, n_positive(hard_label=1, i.e. INCORRECT)={y.sum()} "
                 f"({100*y.mean():.1f}%)")

    results = {}
    for name, X in [("word_length_only", word_lens),
                      ("char_length_only", char_lens),
                      ("word_and_char_length", both)]:
        run = repeated_cv_auc(X, y, k=None, n_splits=cfg["n_splits"],
                                n_seeds=cfg["n_seeds"], base_seed=cfg["base_seed"])
        results[name] = {k: v for k, v in run.items() if k != "pooled_aucs"}
        logger.info(f"{name}: mean AUC = {run['mean']:.3f} "
                     f"(CI [{run['ci_lower']:.3f}, {run['ci_upper']:.3f}])")

    # Reference point: pregen's real (attention, layer=0) AUC was ~0.85 per
    # the exp_03 discrimination results — compare against that as the bar.
    reference_pregen_auc = cfg.get("reference_pregen_auc", 0.85)
    max_shortcut_auc = max(r["ci_upper"] for r in results.values())

    logger.info(f"\n=== Verdict ===")
    logger.info(f"Best surface-shortcut CI-upper = {max_shortcut_auc:.3f} vs "
                 f"reference pregen AUC ~= {reference_pregen_auc:.3f}")
    if max_shortcut_auc < 0.6:
        verdict = "shortcut_explanation_NOT_supported"
        logger.info("-> Surface length features carry little signal (CI upper < 0.6). "
                     "The simple 'question length' shortcut explanation (C) does NOT "
                     "account for pregen's real signal. Explanation B (real "
                     "representational effect) remains the leading account, though more "
                     "subtle shortcuts (e.g. disease-name frequency) are still untested.")
    elif max_shortcut_auc < reference_pregen_auc - 0.15:
        verdict = "shortcut_explanation_partially_supported"
        logger.info("-> Surface length features carry SOME signal, but well short of "
                     "pregen's real AUC. A shortcut may be contributing but doesn't "
                     "fully explain pregen's signal.")
    else:
        verdict = "shortcut_explanation_supported"
        logger.info("-> Surface length features alone get close to pregen's real AUC. "
                     "This is a real red flag — pregen's signal for MedQuad may be "
                     "substantially a shallow shortcut, not deep knowledge-sufficiency "
                     "signal. Reconsider before treating this as a 'fluency masking' "
                     "mechanism story.")

    with open(os.path.join(output_dir, "confound_check_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"results": results, "verdict": verdict,
                     "reference_pregen_auc": reference_pregen_auc}, f, indent=2)

    return {"verdict": verdict, "results": results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                         default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()
    main(args.config)
