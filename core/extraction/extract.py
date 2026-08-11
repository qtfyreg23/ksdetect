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
    exclude padding. Use this for "posthoc" extraction when you want a
    summary over the whole generated answer rather than just its last token
    (the project plan's later token-localization work, if it happens, will
    likely replace this with a critical-token-weighted pool — that change
    belongs here, not in an experiment script).
"""

from __future__ import annotations

import os

import numpy as np
import torch


def extract_batch(
    model,
    tokenizer,
    collector,
    texts: list[str],
    device: str = "cuda",
    pooling: str = "last",
    max_length: int = 2048,
) -> dict[str, dict[int, np.ndarray]]:
    """
    Tokenizes `texts` (already-assembled prompt or prompt+answer strings),
    runs one forward pass with `collector` active, pools each captured
    (batch, seq, dim) tensor down to (batch, dim) per the `pooling` strategy,
    and returns collector-shaped output:
        {module_name: {layer_idx: np.ndarray of shape (batch, dim)}}
    (embedding module uses layer_idx == -1, matching hooks.py's convention.)
    """
    if pooling not in ("last", "mean"):
        raise ValueError(f"pooling must be 'last' or 'mean', got {pooling!r}")

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

    pooled: dict[str, dict[int, np.ndarray]] = {m: {} for m in buffer}
    for module_name, layer_dict in buffer.items():
        for layer_idx, tensor in layer_dict.items():
            # tensor: (batch, seq, dim), attention_mask: (batch, seq)
            if pooling == "last":
                pooled_tensor = tensor[:, -1, :]
            else:  # mean
                mask = attention_mask.unsqueeze(-1)  # (batch, seq, 1)
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
