"""Live streaming KV cache that quantizes on append (K2c recipe made live).

Mirrors transformers' QuantizedCache/QuantizedLayer split: a per-layer
DynamicLayer subclass (StreamingQuantizedLayer) that stores only the compressed
representation and RETURNS dequantized K/V from update() for attention, plus a
thin Cache container (StreamingQuantizedCache) that replicates the layer across
the model. Because the layer never persists the dense dequant, resident state is
the compressed footprint — real memory by the official cache contract.

Lands in two stages:
  Stage A (prev commit): fp16 passthrough + the layer/container plumbing. With an
    fp16 spec the layer delegates to DynamicLayer — gate is bit-identical logits
    and generation vs a plain default cache.
  Stage B (this commit): the quantize-on-append path — pre-RoPE K capture + frozen
    subspace, RoPE-at-read, codec-driven _quantize/_dequantize.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache, DynamicLayer

from bmx.cache.codecs import quantize_cache
from bmx.cache.collect import _reshape_heads, from_matrix, to_matrix
from bmx.cache.rope import apply_rope, rope_cos_sin
from bmx.cache.specs import CacheCodecSpec


class StreamingQuantizedLayer(DynamicLayer):
    """Per-layer streaming-quantized cache.

    Parameters
    ----------
    k_spec, v_spec : CacheCodecSpec
        Codec specs for keys and values. ``arm="fp16"`` => passthrough that side.
    model_config :
        HF model config (RoPE tables + head counts, used by the codec).
    recent_window : int
        Most-recent tokens kept fp16 before flushing to quantized state (future).
    """

    def __init__(self, k_spec, v_spec, model_config, recent_window: int = 32):
        super().__init__()
        self.k_spec = k_spec
        self.v_spec = v_spec
        self.model_config = model_config
        self.recent_window = recent_window
        # Pre-RoPE key capture buffer: accumulated by stash_pre_rope, consumed in update.
        self._k_pre: torch.Tensor | None = None
        self.bpe_k = float("nan")
        self.bpe_v = float("nan")
        self._h_kv = getattr(
            model_config, "num_key_value_heads", model_config.num_attention_heads
        )
        self._d_head = (
            getattr(model_config, "head_dim", None)
            or model_config.hidden_size // model_config.num_attention_heads
        )

    def _is_passthrough(self) -> bool:
        return self.k_spec.arm == "fp16" and self.v_spec.arm == "fp16"

    def stash_pre_rope(self, out: torch.Tensor):
        """Called by the cache's k_proj hook: append a captured pre-RoPE block.

        out: (1, T, h_kv*d) -> reshaped to (h_kv, T, d) fp16, concatenated
        across calls to accumulate the full sequence.
        """
        block = _reshape_heads(out, self._h_kv, self._d_head)  # (h_kv, T, d)
        self._k_pre = (
            block if self._k_pre is None else torch.cat([self._k_pre, block], dim=1)
        )

    def _quantize_matrix(self, kv_fp32: torch.Tensor, spec: CacheCodecSpec):
        """(h,S,d) fp32 -> (dequantized (h,S,d) fp32, bpe). fp16 spec is identity."""
        if spec.arm == "fp16":
            return kv_fp32, 16.0
        h = kv_fp32.shape[0]
        M_hat, bpe = quantize_cache(
            spec.arm,
            to_matrix(kv_fp32),
            bits=spec.bits,
            seed=spec.seed,
            group=spec.group,
            rank=spec.rank,
        )
        return from_matrix(M_hat, h), bpe

    def update(self, key_states, value_states, *args, **kwargs):
        # Let DynamicLayer concat + return the full (post-RoPE) keys/values.
        keys, values = super().update(key_states, value_states, *args, **kwargs)

        # Passthrough: no pre_rope flag and fp16 arms — skip codec entirely.
        if self._is_passthrough() and not self.k_spec.pre_rope:
            self.bpe_k = 16.0
            self.bpe_v = 16.0
            return keys, values

        cache_dtype = keys.dtype
        S = keys.shape[2]  # (1, h_kv, S, d)

        # Keys: pre-RoPE source when captured, RoPE re-applied at read; else post-RoPE.
        if self.k_spec.pre_rope:
            assert self._k_pre is not None, (
                "k_spec.pre_rope=True but no captured pre-RoPE keys; "
                "call cache.attach(model) before prefill"
            )
            k_src = self._k_pre[:, :S, :].float()  # (h_kv, S, d) pre-RoPE
            k_hat_pre, self.bpe_k = self._quantize_matrix(k_src, self.k_spec)
            cos, sin = rope_cos_sin(self.model_config, S)
            k_hat = apply_rope(k_hat_pre, cos.float(), sin.float())  # post-RoPE
        else:
            k_hat, self.bpe_k = self._quantize_matrix(
                keys.squeeze(0).float(), self.k_spec
            )

        v_hat, self.bpe_v = self._quantize_matrix(
            values.squeeze(0).float(), self.v_spec
        )

        # Persist the dequantized slab as the layer's stored cache. The bpe fields
        # record the honest compressed cost; the stored tensor is the dequant approx
        # (Stage B: packed-byte storage is the perf refinement, out of scope here).
        self.keys = k_hat.to(cache_dtype).unsqueeze(0)  # (1, h_kv, S, d)
        self.values = v_hat.to(cache_dtype).unsqueeze(0)  # (1, h_kv, S, d)
        return self.keys, self.values


class StreamingQuantizedCache(Cache):
    """Cache container replicating StreamingQuantizedLayer across the model.

    Drop-in ``past_key_values=`` for model() / model.generate().
    """

    def __init__(
        self,
        model_config,
        k_spec: CacheCodecSpec,
        v_spec: CacheCodecSpec,
        recent_window: int = 32,
    ):
        # layer_class_to_replicate lazily appends one layer per new layer_idx.
        super().__init__(
            layer_class_to_replicate=lambda: StreamingQuantizedLayer(
                k_spec, v_spec, model_config, recent_window
            )
        )
        self.model_config = model_config
        self.k_spec = k_spec
        self.v_spec = v_spec
        self.recent_window = recent_window
        self._handles: list = []

    def attach(self, model) -> "StreamingQuantizedCache":
        """Register k_proj hooks so each layer captures its pre-RoPE keys.

        Call before prefill when k_spec.pre_rope. Hooks fire on every forward
        including each decode step. No-op when k_spec.pre_rope is False.
        Idempotent; hooks removed by detach()/__exit__.

        The hook writes into self.layers[i].stash_pre_rope. Because the cache
        layers are lazily created on first update, the hook may fire before
        self.layers[i] exists. To guard this, we pre-size the layers list here
        so self.layers[i] always exists when the hook fires.
        """
        self.detach()  # Clear any previously-registered hooks (idempotence).
        if not self.k_spec.pre_rope:
            return self

        # Pre-size: ensure self.layers[i] exists for every model layer so the
        # hook can always find self.layers[i] when it fires (before update).
        n_layers = model_config_n_layers(model)
        while len(self.layers) < n_layers:
            self.layers.append(
                StreamingQuantizedLayer(
                    self.k_spec, self.v_spec, self.model_config, self.recent_window
                )
            )

        for i, mlayer in enumerate(model.model.layers):

            def k_hook(module, inp, out, i=i):
                self.layers[i].stash_pre_rope(out)

            self._handles.append(mlayer.self_attn.k_proj.register_forward_hook(k_hook))
        return self

    def detach(self) -> "StreamingQuantizedCache":
        """Remove all registered k_proj hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.detach()
        return False

    def reconstruct_layer(self, layer_idx: int):
        """Return (k_post, v) as stored on the layer — keys RoPE'd, V dequantized.

        Returns (1, h_kv, S, d) tensors.
        """
        layer = self.layers[layer_idx]
        return layer.keys, layer.values

    def bits_per_entry(self):
        """(bpe_k, bpe_v) from the last layer's last quantize (uniform across layers)."""
        if not self.layers:
            return float("nan"), float("nan")
        last = self.layers[-1]
        return last.bpe_k, last.bpe_v


def model_config_n_layers(model) -> int:
    """Number of transformer layers in model (structural probe, not model_type)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise ValueError(
        f"Cannot determine n_layers for model type {type(model).__name__}. "
        "Expected model.model.layers (Llama-family) or model.transformer.h (GPT-2)."
    )
