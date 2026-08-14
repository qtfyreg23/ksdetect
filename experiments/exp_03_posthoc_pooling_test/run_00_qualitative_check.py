"""
run_00_qualitative_check.py — free, CPU-only, no-model pre-check before
committing to the GPU re-extraction in run_01.

Reads the ALREADY-GENERATED greedy_answer text from exp_02_coarse_scan's
saved labels (no new generation), for TruthfulQA and MedQuad, and reports:
  - length distribution of answers (in words), split by hard_label
    (correct vs incorrect)
  - most common LAST WORD across answers (a crude proxy for "does the
    answer usually end in a generic/low-information token")
  - a handful of example (answer, hard_label) pairs to eyeball directly

This does NOT decide between explanation A and B on its own — it's a cheap
sanity check for whether explanation A ("posthoc last-token pooling is a
poor summary because answers end generically") has any qualitative
plausibility at all before spending GPU time on run_01. If most answers end
in clearly content-specific words (not "the.", "is.", etc.), that's a weak
signal against a STRONG version of explanation A — but the real test is
still run_01 + run_02, not this script.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.data import load_generated_labels


def analyze_dataset(labels_dir: str, dataset: str, n_examples_to_print: int = 5):
    records = load_generated_labels(labels_dir, dataset)
    print(f"\n{'='*70}\n{dataset}  (n={len(records)})\n{'='*70}")

    lengths_by_label = {0: [], 1: []}
    last_words = Counter()
    last_words_by_label = {0: Counter(), 1: Counter()}

    for rec in records.values():
        answer = rec["greedy_answer"].strip()
        words = answer.split()
        n_words = len(words)
        label = rec["hard_label"]
        lengths_by_label[label].append(n_words)

        last_word = words[-1].lower().rstrip(".,!?;:\"'") if words else "<empty>"
        last_words[last_word] += 1
        last_words_by_label[label][last_word] += 1

    for label, name in [(0, "CORRECT (hard_label=0)"), (1, "INCORRECT (hard_label=1)")]:
        lens = lengths_by_label[label]
        if lens:
            print(f"\n{name}: n={len(lens)}, answer length in words: "
                  f"mean={sum(lens)/len(lens):.1f}, min={min(lens)}, max={max(lens)}")
        else:
            print(f"\n{name}: n=0 (no examples)")

    print(f"\nTop 15 most common LAST WORDS across all answers "
          f"(out of {len(records)} total):")
    for word, count in last_words.most_common(15):
        pct = 100 * count / len(records)
        print(f"  {word!r:20s} {count:5d} ({pct:.1f}%)")

    top_last_word_pct = last_words.most_common(1)[0][1] / len(records) * 100 if last_words else 0
    print(f"\n--- Crude signal for explanation A: single most common last "
          f"word covers {top_last_word_pct:.1f}% of all answers. ---")
    print("(High % here means answers often end the same way — weakly "
          "supportive of explanation A. Low % means endings are varied — "
          "weakly against a STRONG version of explanation A. Either way, "
          "this is NOT the actual test; see run_01/run_02.)")

    print(f"\n{n_examples_to_print} example answers (random-ish, first N by dict order):")
    for i, (eid, rec) in enumerate(records.items()):
        if i >= n_examples_to_print:
            break
        print(f"  [{eid}] hard_label={rec['hard_label']} "
              f"answer={rec['greedy_answer'][:150]!r}")


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for dataset in cfg["datasets"]:
        analyze_dataset(cfg["coarse_scan_output_dir"], dataset)

    print(f"\n{'='*70}")
    print("Qualitative pre-check done. This is informal — proceed to "
          "run_01_extract_answer_mean.py for the actual discrimination test.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                         default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()
    main(args.config)
