"""
run.py — exp_01_data_check.

Run on the AutoDL server (needs internet access to huggingface.co; does NOT
need a GPU). This is the "verify before scaling" step for core/data/loaders.py,
mirroring what exp_00_sanity_check did for core/stats and core/extraction.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.data import DATASET_REGISTRY, load_dataset_examples
from core.labeling import gsm8k_is_correct


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("exp_01_data_check")
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


REQUIRED_KEYS = {"example_id", "question", "references", "task_type", "dataset"}


def validate_example(ex: dict) -> list[str]:
    """Returns a list of problem descriptions; empty list means the example is well-formed."""
    problems = []
    missing = REQUIRED_KEYS - set(ex.keys())
    if missing:
        problems.append(f"missing keys: {missing}")
        return problems
    if not isinstance(ex["example_id"], str) or not ex["example_id"]:
        problems.append("example_id is not a non-empty string")
    if not isinstance(ex["question"], str) or not ex["question"].strip():
        problems.append("question is not a non-empty string")
    if not isinstance(ex["references"], list) or len(ex["references"]) == 0:
        problems.append("references is not a non-empty list")
    else:
        for r in ex["references"]:
            if not isinstance(r, str) or not r.strip():
                problems.append(f"a reference is not a non-empty string: {r!r}")
                break
    return problems


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logging(os.path.join(out_dir, "..", "run_log.txt"))
    logger.info(f"=== exp_01_data_check ===")
    logger.info(f"Config: {json.dumps(cfg, ensure_ascii=False)}")

    report = {"datasets": {}}
    overall_pass = True

    for name in cfg["datasets"]:
        logger.info(f"--- Loading {name} (n={cfg['n_check_per_dataset']}) ---")
        t0 = time.time()
        try:
            examples = load_dataset_examples(name, n_examples=cfg["n_check_per_dataset"])
        except Exception as e:
            logger.error(f"FAILED to load {name}: {type(e).__name__}: {e}")
            report["datasets"][name] = {"PASS": False, "error": f"{type(e).__name__}: {e}"}
            overall_pass = False
            continue
        elapsed = time.time() - t0

        problems_by_example = {}
        for ex in examples:
            problems = validate_example(ex)
            if problems:
                problems_by_example[ex.get("example_id", "<unknown>")] = problems

        ds_pass = len(problems_by_example) == 0 and len(examples) > 0
        overall_pass = overall_pass and ds_pass

        logger.info(f"{name}: loaded {len(examples)} examples in {elapsed:.1f}s, "
                     f"{'PASS' if ds_pass else 'FAIL'}")
        if examples:
            sample = examples[0]
            logger.info(f"{name} sample example_id={sample['example_id']!r}")
            logger.info(f"{name} sample question (first 200 chars): "
                         f"{sample['question'][:200]!r}")
            logger.info(f"{name} sample references: {sample['references'][:3]!r}"
                         f"{' ...' if len(sample['references']) > 3 else ''}")
        if problems_by_example:
            logger.warning(f"{name} schema problems: {problems_by_example}")

        report["datasets"][name] = {
            "PASS": ds_pass,
            "n_loaded": len(examples),
            "elapsed_seconds": elapsed,
            "sample_example_id": examples[0]["example_id"] if examples else None,
            "sample_question_preview": examples[0]["question"][:200] if examples else None,
            "sample_references": examples[0]["references"][:3] if examples else None,
            "schema_problems": problems_by_example,
        }

    # GSM8K numeric-match sanity check on real data
    logger.info("--- GSM8K numeric-match sanity check ---")
    gsm8k_check_pass = True
    if "gsm8k" in report["datasets"] and report["datasets"]["gsm8k"]["PASS"]:
        gsm8k_examples = load_dataset_examples("gsm8k", n_examples=3)
        for ex in gsm8k_examples:
            raw_answer_field = ex["references"][0]
            # Feed the model's OWN reasoning text back in as a fake
            # "prediction" that should obviously match — this checks that
            # gsm8k_is_correct's parsing works on real GSM8K answer text,
            # not that the model gets it right (there's no model here).
            is_match = gsm8k_is_correct(raw_answer_field, raw_answer_field)
            logger.info(f"gsm8k self-match check ({ex['example_id']}): {is_match}")
            if not is_match:
                gsm8k_check_pass = False
    else:
        gsm8k_check_pass = False
        logger.warning("Skipping GSM8K numeric-match check because gsm8k loading failed above.")

    report["gsm8k_numeric_match_check_PASS"] = gsm8k_check_pass
    overall_pass = overall_pass and gsm8k_check_pass

    report["overall_PASS"] = overall_pass
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"=== OVERALL PASS: {overall_pass} ===")
    logger.info("Send the full console output (or run_log.txt) back for review "
                 "before building the Stage-1 coarse-scan extraction pipeline.")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                         default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()
    result = main(args.config)
    sys.exit(0 if result["overall_PASS"] else 1)
