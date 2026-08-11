"""
run_extraction_smoke_test.py — Part 2 of exp_00_sanity_check.

REQUIRES the real model at MODEL_PATH and a CUDA GPU. Run this on the
AutoDL server, not in any offline/CPU-only environment.

Loads the model once, runs a handful of prompts through extract_batch()
with all four module types active, and checks:
  - no exceptions during model load (confirms local_files_only load works,
    confirms MODEL_PATH is correct),
  - every requested module/layer produced a tensor of shape (batch, hidden)
    with hidden == model.config.hidden_size (or intermediate_size for
    ffn_neuron),
  - no NaN/Inf values in any extracted vector,
  - basic sanity: the SAME prompt run twice gives IDENTICAL activations
    (confirms no stray randomness / dropout-in-eval-mode bug).

This script intentionally does NOT touch core.stats or core.labeling — it
only exercises core.extraction, so a failure here points unambiguously at
extraction code / environment / model path issues.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.extraction import (
    load_model_and_tokenizer,
    ActivationCollector,
    extract_batch,
    SUPPORTED_MODULES,
)


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("exp_00_extraction")
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

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logging(os.path.join(out_dir, "..", "run_log_extraction.txt"))
    t_start = time.time()
    logger.info("=== exp_00_sanity_check / Part 2: extraction smoke test ===")

    if not torch.cuda.is_available():
        logger.error("CUDA is not available in this environment. This script "
                       "must run on the AutoDL server with a GPU, not locally.")
        sys.exit(1)

    logger.info(f"Loading model from {cfg['model_path']} ...")
    model, tokenizer = load_model_and_tokenizer(
        model_path=cfg["model_path"], device=cfg["device"], dtype=cfg["dtype"],
    )
    logger.info(f"Model loaded. hidden_size={model.config.hidden_size}, "
                 f"intermediate_size={model.config.intermediate_size}, "
                 f"num_hidden_layers={model.config.num_hidden_layers}")

    prompts = cfg["smoke_test_prompts"]
    layers_to_check = [0, model.config.num_hidden_layers // 2,
                        model.config.num_hidden_layers - 1]
    collector = ActivationCollector(model, modules=SUPPORTED_MODULES, layers=layers_to_check)

    all_pass = True
    report = {"checks": []}

    logger.info(f"Running extract_batch on {len(prompts)} prompts, "
                 f"layers={layers_to_check}, modules={SUPPORTED_MODULES}")
    result_1 = extract_batch(model, tokenizer, collector, prompts,
                               device=cfg["device"], pooling="last")

    expected_dims = {
        "embedding": model.config.hidden_size,
        "attention": model.config.hidden_size,
        "ffn_module": model.config.hidden_size,
        "ffn_neuron": model.config.intermediate_size,
        "residual": model.config.hidden_size,
    }

    for module_name, layer_dict in result_1.items():
        for layer_idx, arr in layer_dict.items():
            check = {
                "module": module_name, "layer": layer_idx,
                "shape": list(arr.shape),
            }
            shape_ok = (arr.shape == (len(prompts), expected_dims[module_name]))
            nan_ok = not np.isnan(arr).any() and not np.isinf(arr).any()
            check["shape_ok"] = bool(shape_ok)
            check["no_nan_inf"] = bool(nan_ok)
            check["PASS"] = bool(shape_ok and nan_ok)
            all_pass = all_pass and check["PASS"]
            report["checks"].append(check)
            logger.info(f"{module_name} layer={layer_idx}: shape={arr.shape} "
                         f"expected=({len(prompts)}, {expected_dims[module_name]}) "
                         f"{'PASS' if check['PASS'] else 'FAIL'}")

    logger.info("Re-running the same prompts to check determinism ...")
    result_2 = extract_batch(model, tokenizer, collector, prompts,
                               device=cfg["device"], pooling="last")
    determinism_ok = True
    for module_name in result_1:
        for layer_idx in result_1[module_name]:
            same = np.allclose(result_1[module_name][layer_idx],
                                 result_2[module_name][layer_idx], atol=1e-4)
            determinism_ok = determinism_ok and same
    report["determinism_check_PASS"] = bool(determinism_ok)
    all_pass = all_pass and determinism_ok
    logger.info(f"Determinism check: {'PASS' if determinism_ok else 'FAIL'}")

    report["overall_PASS"] = bool(all_pass)
    report["elapsed_seconds"] = time.time() - t_start

    with open(os.path.join(out_dir, "summary_extraction.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"=== EXTRACTION SMOKE TEST: {'PASS' if all_pass else 'FAIL'} "
                 f"(elapsed {report['elapsed_seconds']:.1f}s) ===")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                         default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()
    result = main(args.config)
    sys.exit(0 if result["overall_PASS"] else 1)
