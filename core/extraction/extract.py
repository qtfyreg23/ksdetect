"""
extract.py — batch-level activation extraction and sharded on-disk storage.

Storage layout (see README.md "目录结构说明" for the full convention):

  data/activations/{dataset}/{stage}/{module}/layer_{L:03d}/shard_{shard_id:05d}.npy
      -> np.ndarray of shape (batch_size, hidden_dim), float32

  data/activations/{dataset}/{stage}/example_ids/shard_{shard_id:05d}.npy
      -> np.ndarray of shape (batch_size,), the example_id for each row in
         that shard, in the SAME row order as every module/layer shard file
         for that shard_id. This is what lets you join activations back to
         labels/metadata later.

`stage` is one of "pregen" (prompt/question only) or "posthoc" (prompt +
generated answer).

Pooling strategy (`pooling` argument):
  - "last": take the activation at the final sequence position. Because
    model_loader.load_model_and_tokenizer() sets padding_side="left", the
    final position (-1) is the same real (non-pad) token index for every row
    in a batch, so this is a plain index — no attention_mask math needed.
  - "mean": mean-pool over all non-pad positions, using attention_mask to
    exclude padding. Pools over the ENTIRE sequence (prompt + answer, for
    posthoc text) — for a posthoc summary that is NOT diluted by the prompt
    tokens, use "answer_mean" instead.
  - "answer_mean": mean-pool over only the LAST `answer_token_counts[i]`
    real tokens of each row — i.e. only the generated-answer span, not the
    prompt. Requires the `answer_token_counts` argument. Added specifically
    for the Stage-3 discrimination experiment in
    experiments/exp_03_posthoc_pooling_test (see docs/decisions.md D17) to
    test whether "posthoc last-token pooling" was under-selling posthoc's
    real signal — this was flagged as a deferred simplification back in
    D11 and is now being resolved for a targeted comparison, not rolled out
    to the full Stage-1 grid.
    Optional `answer_window_k` argument (only meaningful with
    pooling="answer_mean"): if given, restricts the pooling window to the
    FIRST `answer_window_k` tokens of the answer span (not the whole
    thing) — added for the exp_05 dilution-vs-style discrimination
    experiment (docs/decisions.md D22), to test whether signal loss is a
    function of averaging over a lot of the answer (dilution) or happens
    even in a short leading window (content/style masking). Left as None
    (pool over the WHOLE answer span) by default — unchanged behavior for
    every prior caller.
"""

from __future__ import annotations

import os

import numpy as np
import torch


def compute_prompt_token_lengths(tokenizer, prompts: list[str], max_length: int = 2048) -> list[int]:
    """
    Returns the UNPADDED token length of each prompt string, tokenized
    individually (not as a padded batch) so the count is exact per example.
    Used to figure out, for a posthoc text built as `prompt + answer`, how
    many of its tokens belong to the answer (= total tokens for the full
    posthoc text - this prompt length) — see "answer_mean" pooling above.

    Deliberately a plain Python loop (not a batched tokenizer call): batch
    tokenization with padding would make every row's length equal to the
    batch max, destroying the per-example length information this function
    exists to compute. This is fine cost-wise — tokenization itself is cheap
    relative to a forward pass.
    """
    lengths = []
    for p in prompts:
        ids = tokenizer(p, truncation=True, max_length=max_length)["input_ids"]
        lengths.append(len(ids))
    return lengths


def extract_batch(
    model,
    tokenizer,
    collector,
    texts: list[str],
    device: str = "cuda",
    pooling: str = "last",
    max_length: int = 2048,
    answer_token_counts: list[int] | None = None,
    answer_window_k: int | None = None,
) -> dict[str, dict[int, np.ndarray]]:
    """
    Tokenizes `texts` (already-assembled prompt or prompt+answer strings),
    runs one forward pass with `collector` active, pools each captured
    (batch, seq, dim) tensor down to (batch, dim) per the `pooling` strategy,
    and returns collector-shaped output:
        {module_name: {layer_idx: np.ndarray of shape (batch, dim)}}
    (embedding module uses layer_idx == -1, matching hooks.py's convention.)

    answer_token_counts: required (and ONLY used) when pooling="answer_mean".
    Must be the same length as `texts`; answer_token_counts[i] is the number
    of trailing real tokens in texts[i] that belong to the answer (see
    compute_prompt_token_lengths() for how to derive this for a
    prompt+answer posthoc string). Silently clipped to each row's actual
    real-token count if larger (logged via a printed warning, since this
    function has no logger of its own — callers running at scale should
    watch stdout/redirect it) — this should not normally happen unless the
    original generation was truncated at max_new_tokens in a way that
    doesn't match this function's own max_length truncation.

    answer_window_k: optional, only used with pooling="answer_mean". If
    given, pools over just the FIRST answer_window_k tokens of each row's
    answer span (clipped to that row's actual answer length if shorter),
    instead of the whole answer span.
    """
    if pooling not in ("last", "mean", "answer_mean"):
        raise ValueError(f"pooling must be 'last', 'mean', or 'answer_mean', got {pooling!r}")
    if pooling == "answer_mean" and answer_token_counts is None:
        raise ValueError("pooling='answer_mean' requires answer_token_counts")
    if answer_window_k is not None and pooling != "answer_mean":
        raise ValueError("answer_window_k is only meaningful with pooling='answer_mean'")
    if answer_token_counts is not None and len(answer_token_counts) != len(texts):
        raise ValueError(
            f"answer_token_counts length ({len(answer_token_counts)}) must "
            f"match texts length ({len(texts)})"
        )

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        with collector:
            model(**inputs)
        buffer = {m: dict(layers) for m, layers in collector.buffer.items()}

    attention_mask = inputs["attention_mask"].to(dtype=torch.float32, device="cpu")
    seq_len = attention_mask.shape[1]

    if pooling == "answer_mean":
        # Left-padded: real tokens occupy the RIGHTMOST `real_len` positions
        # of each row, ending at position seq_len-1. The answer span STARTS
        # at position (seq_len - answer_token_counts[i]). Without
        # answer_window_k, the window runs to the end of the sequence (the
        # whole answer); with answer_window_k, the window is truncated to
        # only the first answer_window_k tokens of that span (verified with
        # a standalone numpy check before being wired in here — see
        # docs/decisions.md D22).
        real_lens = attention_mask.sum(dim=1).long()
        answer_mask = torch.zeros_like(attention_mask)
        for i, n_answer in enumerate(answer_token_counts):
            real_len = int(real_lens[i].item())
            n_answer_clipped = min(n_answer, real_len)
            if n_answer_clipped != n_answer:
                print(
                    f"[extract_batch] WARNING: answer_token_counts[{i}]="
                    f"{n_answer} exceeds real token count {real_len} for "
                    f"this row; clipped to {n_answer_clipped}. This should "
                    f"be rare — investigate if it happens often (possible "
                    f"truncation mismatch between generation and re-extraction)."
                )
            answer_start = seq_len - n_answer_clipped
            if answer_window_k is not None:
                window_len = min(answer_window_k, n_answer_clipped)
                window_end = answer_start + window_len
            else:
                window_end = seq_len
            if window_end > answer_start:
                answer_mask[i, answer_start:window_end] = 1.0
        mask_for_pooling = answer_mask
    else:
        mask_for_pooling = attention_mask

    pooled: dict[str, dict[int, np.ndarray]] = {m: {} for m in buffer}
    for module_name, layer_dict in buffer.items():
        for layer_idx, tensor in layer_dict.items():
            # tensor: (batch, seq, dim)
            if pooling == "last":
                pooled_tensor = tensor[:, -1, :]
            else:  # "mean" or "answer_mean" — same math, different mask
                mask = mask_for_pooling.unsqueeze(-1)  # (batch, seq, 1)
                summed = (tensor * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1.0)
                pooled_tensor = summed / counts
            pooled[module_name][layer_idx] = pooled_tensor.numpy().astype(np.float32)

    return pooled


def save_activation_shard(
    output_dir: str,
    dataset: str,
    stage: str,
    shard_id: int,
    activations: dict[str, dict[int, np.ndarray]],
    example_ids: list[str],
) -> None:
    """
    Writes one shard's worth of pooled activations (as returned by
    extract_batch) to disk following the layout documented at the top of
    this file. Creates directories as needed. Overwrites any existing shard
    file with the same (dataset, stage, module, layer, shard_id) — callers
    are responsible for choosing shard_ids that do not collide with a
    previous, still-wanted run (see project plan §2.3: "新设置与旧设置必须
    清楚区分" — use a run-specific dataset/stage naming or a fresh
    output_dir per experiment run, do not silently overwrite a prior run's
    shards from a DIFFERENT config).
    """
    base = os.path.join(output_dir, dataset, stage)

    ids_dir = os.path.join(base, "example_ids")
    os.makedirs(ids_dir, exist_ok=True)
    np.save(
        os.path.join(ids_dir, f"shard_{shard_id:05d}.npy"),
        np.array(example_ids, dtype=object),
    )

    for module_name, layer_dict in activations.items():
        for layer_idx, arr in layer_dict.items():
            layer_dir = os.path.join(base, module_name, f"layer_{layer_idx:03d}")
            os.makedirs(layer_dir, exist_ok=True)
            np.save(
                os.path.join(layer_dir, f"shard_{shard_id:05d}.npy"),
                arr,
            )


def load_activation(
    activation_dir: str,
    dataset: str,
    stage: str,
    module: str,
    layer: int,
) -> tuple[np.ndarray, list[str]]:
    """
    Reads back ALL shards for one (dataset, stage, module, layer) cell,
    concatenated in shard-index order, alongside the matching example_ids
    (read from the same shard set's example_ids/ directory, same order).

    Returns (X, example_ids) where X has shape (n_examples, dim) and
    example_ids[i] corresponds to X[i].

    Raises FileNotFoundError with a clear message if the directory doesn't
    exist (e.g. wrong module/layer/stage combination, or extraction wasn't
    run for that combination) rather than returning an empty array — an
    empty result silently propagating into a CV call produces a confusing
    downstream error instead of a clear one here.
    """
    layer_dir = os.path.join(activation_dir, dataset, stage, module, f"layer_{layer:03d}")
    ids_dir = os.path.join(activation_dir, dataset, stage, "example_ids")
    if not os.path.isdir(layer_dir):
        raise FileNotFoundError(
            f"No activation shards found at {layer_dir}. Check that "
            f"extraction was run for dataset={dataset!r}, stage={stage!r}, "
            f"module={module!r}, layer={layer!r}."
        )

    shard_files = sorted(f for f in os.listdir(layer_dir) if f.endswith(".npy"))
    if not shard_files:
        raise FileNotFoundError(f"{layer_dir} exists but has no shard files.")

    X_parts, ids_parts = [], []
    for shard_file in shard_files:
        shard_id = shard_file  # same filename used for both activation and ids shards
        X_parts.append(np.load(os.path.join(layer_dir, shard_file)))
        ids_arr = np.load(os.path.join(ids_dir, shard_id), allow_pickle=True)
        ids_parts.append(list(ids_arr))

    X = np.concatenate(X_parts, axis=0)
    example_ids = [eid for part in ids_parts for eid in part]
    if X.shape[0] != len(example_ids):
        raise ValueError(
            f"Mismatch between activation rows ({X.shape[0]}) and example_ids "
            f"({len(example_ids)}) for {layer_dir} — shard files may be "
            f"corrupted or partially written; re-run extraction for this cell."
        )
    return X, example_ids
