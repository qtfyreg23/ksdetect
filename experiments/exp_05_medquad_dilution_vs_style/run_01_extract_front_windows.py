"""
run_01_extract_front_windows.py — exp_05, step 1.

For each dataset x each K in front_window_k_values: re-extracts posthoc
activations pooled over ONLY the first K tokens of the already-generated
greedy answer (no new generation). Structurally near-identical to exp_03's
run_01_extract_answer_mean.py, generalized to loop over multiple window
sizes and save each under its own stage name ("posthoc_front{K}").
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.data import load_generated_labels
from core.extraction import (
    load_model_and_tokenizer,
    ActivationCollector,
    extract_batch,
    save_activation_shard,
    compute_prompt_token_lengths,
)
from core.run_utils import check_or_write_config_hash


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("exp_05_extract")
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


def stage_name(k: int) -> str:
    return f"posthoc_front{k}"


def done_marker_path(activation_dir: str, dataset: str, stage: str, shard_id: int) -> str:
    d = os.path.join(activation_dir, dataset, stage, "_done_markers")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"shard_{shard_id:05d}.done")


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(os.path.join(output_dir, "run_log_extract.txt"))
    logger.info("=== exp_05 run_01: re-extract posthoc(front-K windows) ===")
    logger.info(f"Config: {json.dumps(cfg, ensure_ascii=False)}")

    check_or_write_config_hash(output_dir, cfg, logger)

    logger.info(f"Loading model from {cfg['model_path']} ...")
    model, tokenizer = load_model_and_tokenizer(
        model_path=cfg["model_path"], device=cfg["device"], dtype=cfg["dtype"],
    )
    logger.info("Model loaded.")

    collector = ActivationCollector(model, modules=cfg["modules"], layers=cfg["layers_coarse"])

    summary = {"datasets": {}}
    t_run_start = time.time()

    for dataset in cfg["datasets"]:
        logger.info(f"--- Dataset: {dataset} ---")
        records = load_generated_labels(cfg["coarse_scan_output_dir"], dataset)
        example_ids_sorted = sorted(records.keys())
        n_total = len(example_ids_sorted)

        batch_size = cfg["batch_size"]
        n_batches = (n_total + batch_size - 1) // batch_size
        dataset_stats = {}

        for k in cfg["front_window_k_values"]:
            stage = stage_name(k)
            logger.info(f"  -- window K={k} (stage={stage}) --")
            completed, skipped_resumed, errored = 0, 0, 0
            failed_batches = []
            t_stage_start = time.time()

            for shard_id in range(n_batches):
                start = shard_id * batch_size
                end = min(start + batch_size, n_total)
                batch_ids = example_ids_sorted[start:end]
                batch_records = [records[eid] for eid in batch_ids]

                marker = done_marker_path(cfg["activation_dir"], dataset, stage, shard_id)
                if cfg["resume"] and os.path.exists(marker):
                    skipped_resumed += 1
                    continue

                try:
                    questions = [r["question"] for r in batch_records]
                    greedy_answers = [r["greedy_answer"] for r in batch_records]

                    chat_prompts = [
                        tokenizer.apply_chat_template(
                            [{"role": "user", "content": q}], tokenize=False,
                            add_generation_prompt=True,
                        )
                        for q in questions
                    ]
                    posthoc_texts = [p + a for p, a in zip(chat_prompts, greedy_answers)]

                    prompt_lengths = compute_prompt_token_lengths(
                        tokenizer, chat_prompts, max_length=cfg["max_input_length"],
                    )
                    total_lengths = compute_prompt_token_lengths(
                        tokenizer, posthoc_texts, max_length=cfg["max_input_length"],
                    )
                    answer_token_counts = [
                        max(0, total - prompt)
                        for total, prompt in zip(total_lengths, prompt_lengths)
                    ]

                    acts = extract_batch(
                        model, tokenizer, collector, posthoc_texts,
                        device=cfg["device"], pooling="answer_mean",
                        max_length=cfg["max_input_length"],
                        answer_token_counts=answer_token_counts,
                        answer_window_k=k,
                    )

                    save_activation_shard(
                        cfg["activation_dir"], dataset, stage, shard_id, acts, batch_ids,
                    )
                    with open(marker, "w") as f:
                        f.write("done")
                    completed += 1

                except Exception as e:
                    errored += 1
                    failed_batches.append(shard_id)
                    logger.error(
                        f"{dataset}/{stage} shard {shard_id} (ids={batch_ids}) FAILED: "
                        f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if (shard_id + 1) % cfg["progress_log_every_n_batches"] == 0 or shard_id == n_batches - 1:
                    elapsed = time.time() - t_stage_start
                    done_so_far = completed + skipped_resumed
                    remaining = n_batches - done_so_far - errored
                    rate = done_so_far / elapsed if elapsed > 0 else 0.0
                    eta_min = (remaining / rate / 60) if rate > 0 else float("nan")
                    logger.info(
                        f"{dataset}/{stage} progress: completed={completed} "
                        f"resumed_skipped={skipped_resumed} errored={errored} "
                        f"pending={remaining}/{n_batches} | rate={rate:.3f} batches/s | "
                        f"ETA={eta_min:.1f} min"
                    )

            stage_elapsed = time.time() - t_stage_start
            logger.info(f"{dataset}/{stage} DONE: completed={completed} "
                         f"resumed_skipped={skipped_resumed} errored={errored} "
                         f"failed_batches={failed_batches} elapsed={stage_elapsed/60:.1f} min")
            dataset_stats[stage] = {
                "n_examples": n_total, "n_batches": n_batches, "completed": completed,
                "resumed_skipped": skipped_resumed, "errored": errored,
                "failed_batches": failed_batches, "elapsed_seconds": stage_elapsed,
            }

        summary["datasets"][dataset] = dataset_stats

    summary["total_elapsed_seconds"] = time.time() - t_run_start
    summary["any_failures"] = any(
        stage_stats["errored"] > 0
        for ds_stats in summary["datasets"].values()
        for stage_stats in ds_stats.values()
    )

    with open(os.path.join(output_dir, "summary_extract.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"=== run_01 FINISHED in {summary['total_elapsed_seconds']/60:.1f} min ===")
    if summary["any_failures"]:
        logger.warning("Some batches FAILED — see failed_batches in summary_extract.json. "
                         "Fix and re-run this script (resume will only redo missing shards).")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                         default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()
    result = main(args.config)
    sys.exit(1 if result["any_failures"] else 0)
