"""
run.py — exp_02_coarse_scan.

Pipeline per batch of examples, within each dataset:
  1. Build chat-formatted prompts (tokenizer.apply_chat_template).
  2. Greedy-generate 1 completion per prompt -> hard label + posthoc text.
  3. Temperature-sample n_samples_for_soft_label completions per prompt ->
     soft label.
  4. Extract PREGEN activations (prompt only) at layers_coarse, all modules.
  5. Extract POSTHOC activations (prompt + greedy answer) at layers_coarse,
     all modules.
  6. Save activation shards (core.extraction.save_activation_shard) and a
     labels shard (JSONL) for this batch.

Resumable: a batch is considered DONE iff its labels shard file exists
(written last, after all activation shards for that batch succeeded) — see
config.yaml's `resume` field and docs/decisions.md D13.

A batch that raises an exception is logged with its full traceback and
recorded in results/summary.json's `failed_batches`; the run continues with
the next batch rather than aborting (project plan §2.4). Failed batches
must be investigated and re-run (this script does NOT auto-retry — do not
delete/mask the failure record, re-run this same script after fixing the
cause, since resume will only re-attempt batches that are missing their
labels shard).
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

from core.data import load_dataset_examples
from core.extraction import (
    load_model_and_tokenizer,
    ActivationCollector,
    extract_batch,
    save_activation_shard,
    generate_greedy_batch,
    generate_samples_batch,
    DEFAULT_MAX_NEW_TOKENS,
)
from core.labeling import (
    is_correct,
    hard_label_from_greedy,
    soft_label_from_samples,
    gsm8k_is_correct,
)
from core.run_utils import check_or_write_config_hash


def _gsm8k_correctness(prediction: str, references: list[str]) -> bool:
    """Adapter so gsm8k_is_correct fits the (prediction, references) -> bool
    signature the rest of core.labeling uses; references[0] holds the raw
    GSM8K answer field (see core/data/loaders.py::load_gsm8k)."""
    return gsm8k_is_correct(prediction, references[0])


CORRECTNESS_FN_BY_DATASET = {
    "truthfulqa": is_correct,
    "triviaqa": is_correct,
    "coqa": is_correct,
    "medquad": is_correct,
    "gsm8k": _gsm8k_correctness,
}


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("exp_02_coarse_scan")
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



def labels_shard_path(output_dir: str, dataset: str, shard_id: int) -> str:
    d = os.path.join(output_dir, "labels", dataset)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"shard_{shard_id:05d}.jsonl")


def process_batch(
    batch_examples: list[dict],
    dataset_name: str,
    model, tokenizer, collector_pregen, collector_posthoc,
    cfg: dict,
    device: str,
) -> list[dict]:
    """Runs steps 1-5 for one batch; returns a list of per-example label
    records (step 6's content). Raises on any failure — caller is
    responsible for catching, logging, and recording the failed batch."""

    questions = [ex["question"] for ex in batch_examples]
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
        )
        for q in questions
    ]

    max_new_tokens = cfg["max_new_tokens_override"].get(
        dataset_name, DEFAULT_MAX_NEW_TOKENS[dataset_name]
    )

    greedy_answers = generate_greedy_batch(
        model, tokenizer, chat_prompts, max_new_tokens=max_new_tokens, device=device,
    )
    sampled_answers = generate_samples_batch(
        model, tokenizer, chat_prompts, max_new_tokens=max_new_tokens,
        n_samples=cfg["n_samples_for_soft_label"],
        temperature=cfg["sampling_temperature"], device=device,
    )

    correctness_fn = CORRECTNESS_FN_BY_DATASET[dataset_name]

    pregen_acts = extract_batch(
        model, tokenizer, collector_pregen, chat_prompts,
        device=device, pooling=cfg["pooling_pregen"],
    )

    posthoc_texts = [p + a for p, a in zip(chat_prompts, greedy_answers)]
    posthoc_acts = extract_batch(
        model, tokenizer, collector_posthoc, posthoc_texts,
        device=device, pooling=cfg["pooling_posthoc"],
    )

    records = []
    for i, ex in enumerate(batch_examples):
        hard_label = hard_label_from_greedy(
            greedy_answers[i], ex["references"], correctness_fn=correctness_fn,
        )
        soft = soft_label_from_samples(
            sampled_answers[i], ex["references"], correctness_fn=correctness_fn,
        )
        records.append({
            "example_id": ex["example_id"],
            "dataset": dataset_name,
            "task_type": ex["task_type"],
            "question": ex["question"],
            "references": ex["references"],
            "greedy_answer": greedy_answers[i],
            "hard_label": hard_label,
            "soft_label": soft["soft_label"],
            "soft_label_n_samples": soft["n_samples"],
            "soft_label_n_incorrect": soft["n_incorrect"],
        })

    return records, pregen_acts, posthoc_acts


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output_dir"]
    activation_dir = cfg["activation_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(os.path.join(output_dir, "..", "run_log.txt"))
    logger.info("=== exp_02_coarse_scan ===")
    logger.info(f"Config: {json.dumps(cfg, ensure_ascii=False)}")

    check_or_write_config_hash(output_dir, cfg, logger)

    logger.info(f"Loading model from {cfg['model_path']} ...")
    model, tokenizer = load_model_and_tokenizer(
        model_path=cfg["model_path"], device=cfg["device"], dtype=cfg["dtype"],
    )
    logger.info("Model loaded.")

    collector_pregen = ActivationCollector(model, modules=cfg["modules"], layers=cfg["layers_coarse"])
    collector_posthoc = ActivationCollector(model, modules=cfg["modules"], layers=cfg["layers_coarse"])

    overall_summary = {"datasets": {}}
    t_run_start = time.time()

    for dataset_name in cfg["datasets"]:
        logger.info(f"--- Dataset: {dataset_name} ---")
        examples = load_dataset_examples(dataset_name, n_examples=cfg["n_examples_per_dataset"])
        n_total = len(examples)
        if n_total < cfg["n_examples_per_dataset"]:
            logger.info(f"{dataset_name}: requested {cfg['n_examples_per_dataset']} examples "
                         f"but only {n_total} are available — using all {n_total} "
                         f"(not silently ignored, logging explicitly per project plan).")

        batch_size = cfg["batch_size"]
        n_batches = (n_total + batch_size - 1) // batch_size
        completed, skipped_resumed, errored = 0, 0, 0
        failed_batches = []
        t_dataset_start = time.time()

        for shard_id in range(n_batches):
            start = shard_id * batch_size
            end = min(start + batch_size, n_total)
            batch_examples = examples[start:end]

            lpath = labels_shard_path(output_dir, dataset_name, shard_id)
            if cfg["resume"] and os.path.exists(lpath):
                skipped_resumed += 1
                continue

            try:
                records, pregen_acts, posthoc_acts = process_batch(
                    batch_examples, dataset_name,
                    model, tokenizer, collector_pregen, collector_posthoc,
                    cfg, device=cfg["device"],
                )

                example_ids = [ex["example_id"] for ex in batch_examples]
                save_activation_shard(
                    activation_dir, dataset_name, "pregen", shard_id,
                    pregen_acts, example_ids,
                )
                save_activation_shard(
                    activation_dir, dataset_name, "posthoc", shard_id,
                    posthoc_acts, example_ids,
                )

                with open(lpath, "w", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                completed += 1

            except Exception as e:
                errored += 1
                failed_batches.append(shard_id)
                # Diagnostic breadcrumb for OOM/memory-related failures:
                # record the longest prompt length in this batch (character
                # count, cheap proxy for token count) so a pattern like
                # "failures cluster on unusually long CoQA stories" is
                # visible from the log without re-running anything — see
                # docs/known_issues.md #8.
                max_question_chars = max(len(ex["question"]) for ex in batch_examples)
                example_ids_in_batch = [ex["example_id"] for ex in batch_examples]
                logger.error(
                    f"{dataset_name} shard {shard_id} FAILED "
                    f"(example_ids={example_ids_in_batch}, "
                    f"max_question_chars={max_question_chars}): "
                    f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                )
                # Defensive cleanup after ANY failure, not just OOM — a
                # failed generate() call can leave fragmented/unreleased
                # CUDA memory behind, which then makes the NEXT batch more
                # likely to fail too (see docs/known_issues.md #8).
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            # Proactive defragmentation after every successful batch too —
            # over a run of thousands of generate() calls with varying
            # sequence lengths, CUDA memory fragmentation accumulates even
            # without any single failure; this trades a small amount of
            # per-batch overhead for not OOMing near the end of a long run
            # (docs/known_issues.md #8).
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (shard_id + 1) % cfg["progress_log_every_n_batches"] == 0 or shard_id == n_batches - 1:
                elapsed = time.time() - t_dataset_start
                done_so_far = completed + skipped_resumed
                remaining = n_batches - done_so_far - errored
                rate = done_so_far / elapsed if elapsed > 0 else 0.0
                eta_seconds = remaining / rate if rate > 0 else float("nan")
                logger.info(
                    f"{dataset_name} progress: completed={completed} "
                    f"resumed_skipped={skipped_resumed} errored={errored} "
                    f"pending={remaining} / total_batches={n_batches} | "
                    f"rate={rate:.3f} batches/s | "
                    f"ETA={eta_seconds/60:.1f} min"
                )

        dataset_elapsed = time.time() - t_dataset_start
        logger.info(
            f"{dataset_name} DONE: completed={completed} resumed_skipped={skipped_resumed} "
            f"errored={errored} failed_batches={failed_batches} "
            f"elapsed={dataset_elapsed/60:.1f} min"
        )
        overall_summary["datasets"][dataset_name] = {
            "n_examples": n_total,
            "n_batches": n_batches,
            "completed": completed,
            "resumed_skipped": skipped_resumed,
            "errored": errored,
            "failed_batches": failed_batches,
            "elapsed_seconds": dataset_elapsed,
        }

    total_elapsed = time.time() - t_run_start
    overall_summary["total_elapsed_seconds"] = total_elapsed
    overall_summary["any_failures"] = any(
        d["errored"] > 0 for d in overall_summary["datasets"].values()
    )

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    logger.info(f"=== exp_02_coarse_scan FINISHED in {total_elapsed/60:.1f} min ===")
    if overall_summary["any_failures"]:
        logger.warning(
            "Some batches FAILED — see failed_batches per dataset in "
            "summary.json and the ERROR lines in run_log.txt. Investigate "
            "and fix the cause, then re-run this script (resume=true will "
            "only re-attempt the missing/failed shards)."
        )
    else:
        logger.info("No batch failures. Ready for exp_02b analysis.")

    return overall_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                         default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()
    result = main(args.config)
    sys.exit(1 if result["any_failures"] else 0)
