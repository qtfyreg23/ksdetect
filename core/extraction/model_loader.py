"""
model_loader.py — single point of model/tokenizer loading.

Rule: MODEL_PATH is the only place the model path is hardcoded. Every other
piece of code (experiments configs included) should reference the model via
this constant or via config.yaml's `model_path` field, which should be set
to this same value — do not hardcode the path a second time anywhere else.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fixed local path on the AutoDL instance (per project requirements). We load
# from local disk, not the HF Hub, to avoid the endpoint/network configuration
# issues recorded in docs/known_issues.md #4.
MODEL_PATH = "/autodl-fs/data/Llama-3.1-8B-Instruct"


def load_model_and_tokenizer(
    model_path: str = MODEL_PATH,
    device: str = "cuda",
    dtype: str = "bfloat16",
    output_hidden_states: bool = True,
    output_attentions: bool = False,
):
    """
    Loads the model and tokenizer from a LOCAL path only (local_files_only=True)
    so that a missing/misconfigured network never silently falls back to a
    slow or failing Hub download.

    dtype: "bfloat16" (default, matches typical A-series AutoDL GPUs) or
    "float16" / "float32" — pass the string, not a torch.dtype, so this is
    trivially settable from config.yaml.

    Returns (model, tokenizer). Caller is responsible for model.eval() usage
    context (already set here) and for moving new tensors to `device`.
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype {dtype!r}, expected one of {list(dtype_map)}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        # Llama tokenizers commonly lack a pad token; use eos as pad for
        # batched extraction (padding side set to left for causal-LM decode
        # alignment, see NOTE below).
        tokenizer.pad_token = tokenizer.eos_token

    # IMPORTANT: left-padding is required for correct "last token" activation
    # extraction in a batch — with right-padding, the last non-pad token is
    # at a different index per example, which the extraction code in
    # extract.py assumes is handled by left-padding + attention_mask instead.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype_map[dtype],
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
    )
    model.to(device)
    model.eval()

    return model, tokenizer
