"""
generate.py — batched text generation, used for both hard labels (greedy)
and soft labels (temperature sampling).

Rule: any experiment needing model.generate() calls these functions, not
its own generate() call — decoding settings (max_new_tokens per dataset,
left-padding assumptions, stripping newly-generated tokens correctly) are
easy to get subtly wrong and expensive to debug per-experiment.

max_new_tokens defaults mirror LAFaCT's Appendix E, Table 10 generation
settings (same paper this project benchmarks against in Stage 6), so
generation length is comparable dataset-for-dataset even though we don't
replicate LAFaCT's exact temperature/top_p/labeling protocol elsewhere.
"""

from __future__ import annotations

import torch

# From LAFaCT Table 10 (Appendix E): GSM8K/MedQuad=256, TruthfulQA=128,
# CoQA/TriviaQA=64. Not verified against our own labeling protocol's needs —
# if generations are getting cut off (check for missing "#### N" on GSM8K,
# or answers that look truncated), this is the first thing to revisit.
DEFAULT_MAX_NEW_TOKENS = {
    "truthfulqa": 128,
    "triviaqa": 64,
    "coqa": 64,
    "medquad": 256,
    "gsm8k": 256,
}


def _decode_new_tokens(tokenizer, input_ids: torch.Tensor, output_ids: torch.Tensor) -> list[str]:
    """
    Given left-padded `input_ids` (batch, prompt_len) and `output_ids`
    (batch, prompt_len + generated_len) from model.generate(), returns the
    NEWLY generated text only (decoded, special tokens skipped, stripped).
    Relies on left-padding (see model_loader.py) so prompt_len is the same
    for every row in the batch — this is exactly why padding_side="left"
    was set there; do not call this with a right-padded tokenizer.
    """
    prompt_len = input_ids.shape[1]
    new_token_ids = output_ids[:, prompt_len:]
    texts = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
    return [t.strip() for t in texts]


def generate_greedy_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    device: str = "cuda",
    max_input_length: int = 2048,
) -> list[str]:
    """
    One greedy (deterministic, do_sample=False) completion per prompt.
    Used for: hard labels, and as the "posthoc" text (prompt + this answer)
    for posthoc activation extraction.
    """
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_input_length,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
        )

    return _decode_new_tokens(tokenizer, inputs["input_ids"], output_ids)


def generate_samples_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    n_samples: int,
    temperature: float,
    device: str = "cuda",
    max_input_length: int = 2048,
) -> list[list[str]]:
    """
    n_samples temperature-sampled completions PER prompt, used for soft
    labels (core.labeling.soft_label_from_samples).

    Returns a list of length len(prompts), each element a list of
    n_samples completion strings for that prompt.

    Memory note: this generates len(prompts) * n_samples sequences in one
    batch via num_return_sequences — with the conservative batch_size this
    project starts with (e.g. 4 prompts), n_samples=10 means an effective
    generation batch of 40 sequences. If this OOMs before the prompt-level
    batch_size is reduced further, that is the parameter to adjust first
    (config.yaml's `batch_size`), not n_samples (which is a label-quality
    parameter, not a memory-tuning one) — flag this back rather than
    silently lowering n_samples to make it fit.
    """
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_input_length,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=n_samples,
            pad_token_id=tokenizer.pad_token_id,
        )

    # output_ids has shape (len(prompts) * n_samples, seq_len); rows for the
    # same prompt are contiguous (HF's num_return_sequences ordering).
    prompt_len = inputs["input_ids"].shape[1]
    new_token_ids = output_ids[:, prompt_len:]
    all_texts = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
    all_texts = [t.strip() for t in all_texts]

    grouped = []
    for i in range(len(prompts)):
        grouped.append(all_texts[i * n_samples : (i + 1) * n_samples])
    return grouped
