"""
hooks.py — forward-hook registration for the four signal sources defined in
the project plan §5.1: FFN, Attention, Residual, Embedding.

Structural assumptions (verified against transformers' LlamaForCausalLM /
LlamaModel implementation as of transformers==4.44.2, pinned in
environment.yml — if the installed transformers version differs, RE-VERIFY
these assumptions against that version's source before trusting extracted
activations, per project plan §2.2):

  model.model.embed_tokens          -> token embedding layer
  model.model.layers[i]             -> i-th LlamaDecoderLayer
  model.model.layers[i].self_attn   -> attention sub-module; forward output
                                        is a tuple whose [0] element is the
                                        post-o_proj attention output,
                                        shape (batch, seq, hidden_size)
  model.model.layers[i].mlp         -> LlamaMLP; forward OUTPUT (after
                                        down_proj) has shape
                                        (batch, seq, hidden_size) — this is
                                        the "FFN module-level" signal used in
                                        Exp1.5-style comparisons.
  model.model.layers[i].mlp.down_proj -> forward INPUT to this linear layer
                                        has shape
                                        (batch, seq, intermediate_size)
                                        (14336 for this 8B model) — this is
                                        the "FFN neuron-level" signal used in
                                        Exp2-style sparsity scans.
  model.model.layers[i]              -> the decoder layer's own forward
                                        OUTPUT's [0] element, shape
                                        (batch, seq, hidden_size), is the
                                        "residual stream" signal AFTER layer
                                        i.

If a future transformers version changes any of these (e.g. output no longer
a tuple), core/extraction/hooks.py is the ONLY file that needs updating —
that is the point of centralizing this here.
"""

from __future__ import annotations

import torch
import torch.nn as nn

SUPPORTED_MODULES = ["embedding", "attention", "ffn_module", "ffn_neuron", "residual"]


class ActivationCollector:
    """
    Registers forward hooks on a subset of SUPPORTED_MODULES across all (or a
    subset of) decoder layers, and buffers the raw (batch, seq, dim) tensors
    captured during the next forward pass. Call `.collect()` once per
    forward pass, read `.buffer`, then call `.clear_buffer()` before the next
    pass (or call `.collect()` again inside a fresh `with` block — it clears
    automatically on __enter__).

    Usage:
        collector = ActivationCollector(model, modules=["ffn_module", "residual"],
                                          layers=list(range(model.config.num_hidden_layers)))
        with collector:
            model(**batch_inputs)
        # collector.buffer["ffn_module"][layer_idx] -> tensor (batch, seq, hidden)
    """

    def __init__(
        self,
        model,
        modules: list[str],
        layers: list[int] | None = None,
    ):
        unknown = set(modules) - set(SUPPORTED_MODULES)
        if unknown:
            raise ValueError(f"Unsupported module(s) requested: {unknown}. "
                              f"Supported: {SUPPORTED_MODULES}")

        self.model = model
        self.modules = modules
        n_layers = model.config.num_hidden_layers
        self.layers = layers if layers is not None else list(range(n_layers))

        self.buffer: dict[str, dict[int, torch.Tensor]] = {m: {} for m in modules}
        self._handles = []

    def _make_output_hook(self, module_name: str, layer_idx: int, index_into_output=None):
        def hook(module, inputs, output):
            tensor = output[index_into_output] if index_into_output is not None else output
            # Detach + move to CPU float32 immediately to avoid holding onto
            # the autograd graph / GPU memory across the whole batch loop.
            self.buffer[module_name][layer_idx] = tensor.detach().to(
                dtype=torch.float32, device="cpu"
            )
        return hook

    def _make_input_hook(self, module_name: str, layer_idx: int):
        def hook(module, inputs):
            tensor = inputs[0]
            self.buffer[module_name][layer_idx] = tensor.detach().to(
                dtype=torch.float32, device="cpu"
            )
        return hook

    def register(self):
        base_layers = self.model.model.layers

        if "embedding" in self.modules:
            emb = self.model.model.embed_tokens
            h = emb.register_forward_hook(
                self._make_output_hook("embedding", -1, index_into_output=None)
            )
            self._handles.append(h)

        for layer_idx in self.layers:
            layer = base_layers[layer_idx]

            if "attention" in self.modules:
                h = layer.self_attn.register_forward_hook(
                    self._make_output_hook("attention", layer_idx, index_into_output=0)
                )
                self._handles.append(h)

            if "ffn_module" in self.modules:
                h = layer.mlp.register_forward_hook(
                    self._make_output_hook("ffn_module", layer_idx, index_into_output=None)
                )
                self._handles.append(h)

            if "ffn_neuron" in self.modules:
                h = layer.mlp.down_proj.register_forward_pre_hook(
                    self._make_input_hook("ffn_neuron", layer_idx)
                )
                self._handles.append(h)

            if "residual" in self.modules:
                h = layer.register_forward_hook(
                    self._make_output_hook("residual", layer_idx, index_into_output=0)
                )
                self._handles.append(h)

        return self

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def clear_buffer(self):
        self.buffer = {m: {} for m in self.modules}

    def __enter__(self):
        self.clear_buffer()
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
        return False
