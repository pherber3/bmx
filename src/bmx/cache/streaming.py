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

from bmx.cache.codecs import S_DIVISIBILITY_ARMS, quantize_kv_layout
from bmx.cache.collect import _reshape_heads
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
        # Precomputed per-instance constants (avoid re-evaluating each update step).
        self._passthrough = k_spec.arm == "fp16" and v_spec.arm == "fp16"
        self._g = k_spec.group if k_spec.arm in S_DIVISIBILITY_ARMS else 1
        # RoPE cos/sin cache: keyed by S_q; avoids constructing nn.Module each step.
        self._rope_cache: dict[int, tuple] = {}

    def _is_passthrough(self) -> bool:
        return self._passthrough

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
        return quantize_kv_layout(kv_fp32, spec)

    def _group_size(self) -> int:
        """Binding group size for S-alignment.

        Only rtn_channel and lowrank_rtn_channel assert S % group == 0.
        All other arms have no S-divisibility constraint, so g=1 (quantize
        everything except the fp16 window without any alignment restriction).
        """
        return self._g

    def update(self, key_states, value_states, *args, **kwargs):
        # Let DynamicLayer concat + return the full (post-RoPE) keys/values.
        keys, values = super().update(key_states, value_states, *args, **kwargs)

        # Passthrough: no pre_rope flag and fp16 arms — skip codec entirely.
        if self._passthrough and not self.k_spec.pre_rope:
            self.bpe_k = 16.0
            self.bpe_v = 16.0
            return keys, values

        cache_dtype = keys.dtype
        S = keys.shape[2]  # (1, h_kv, S, d)
        W = self.recent_window
        g = self._g

        # Compute the quantized-prefix length: largest multiple of g that leaves
        # at least W recent tokens in the fp16 window.
        S_q = ((S - W) // g) * g if S > W else 0

        if S_q <= 0:
            # Nothing to quantize yet — whole cache stays fp16 this call.
            self.keys = keys  # (1, h_kv, S, d) already from DynamicLayer
            self.values = values
            self.bpe_k = 16.0
            self.bpe_v = 16.0
            return self.keys, self.values

        # --- Quantize the prefix [:S_q] ---

        # Keys prefix: use pre-RoPE source when captured, then re-apply RoPE.
        if self.k_spec.pre_rope:
            assert self._k_pre is not None, (
                "k_spec.pre_rope=True but no captured pre-RoPE keys; "
                "call cache.attach(model) before prefill"
            )
            k_pre_prefix = self._k_pre[:, :S_q, :].float()  # (h_kv, S_q, d)
            k_hat_pre, codec_bpe_k = self._quantize_matrix(k_pre_prefix, self.k_spec)
            if S_q not in self._rope_cache:
                self._rope_cache[S_q] = rope_cos_sin(self.model_config, S_q)
            cos, sin = self._rope_cache[S_q]
            k_prefix = apply_rope(k_hat_pre, cos.float(), sin.float())  # post-RoPE
        else:
            k_prefix_fp32 = keys.squeeze(0)[..., :S_q, :].float()  # (h_kv, S_q, d)
            k_prefix, codec_bpe_k = self._quantize_matrix(k_prefix_fp32, self.k_spec)

        # Values prefix.
        v_prefix_fp32 = values.squeeze(0)[..., :S_q, :].float()  # (h_kv, S_q, d)
        v_prefix, codec_bpe_v = self._quantize_matrix(v_prefix_fp32, self.v_spec)

        # --- fp16 tail [S_q:] — already post-RoPE from DynamicLayer ---
        # For keys: `keys` (from super().update) is already post-RoPE; just slice.
        k_tail = keys.squeeze(0)[..., S_q:, :]  # (h_kv, S-S_q, d) fp16
        v_tail = values.squeeze(0)[..., S_q:, :]  # (h_kv, S-S_q, d) fp16

        # --- Reassemble: concat along seq dim (dim=-2) ---
        k_hat = torch.cat([k_prefix.to(cache_dtype), k_tail], dim=-2)  # (h_kv, S, d)
        v_hat = torch.cat([v_prefix.to(cache_dtype), v_tail], dim=-2)  # (h_kv, S, d)

        # Blended bpe: quantized prefix costs codec_bpe; fp16 tail costs 16.
        self.bpe_k = (S_q * codec_bpe_k + (S - S_q) * 16.0) / S
        self.bpe_v = (S_q * codec_bpe_v + (S - S_q) * 16.0) / S

        # Persist the dequantized + fp16-tail slab as the layer's stored cache.
        self.keys = k_hat.unsqueeze(0)  # (1, h_kv, S, d)
        self.values = v_hat.unsqueeze(0)  # (1, h_kv, S, d)
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

    def memory_report(
        self, seq_len: int, h_kv: int | None = None, d_head: int | None = None
    ) -> dict:
        """Honest KV footprint: dense fp16 baseline vs packed (bpe-derived) bytes.

        packed_bytes uses the honest bits_per_entry() (ALL metadata counted by the
        codec) — the real deployable cache size. Raw fp16-slab bytes would understate
        the win because Stage-B stores the dequant for the model to read; the bpe is
        the deployable number. Process-level peak memory (the literal 5x) is the
        fused-kernel/paged-store VM measurement.
        """
        cfg = self.model_config
        h_kv = h_kv or getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        d = d_head or (
            getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        )
        n_layer = cfg.num_hidden_layers
        entries_per_side = n_layer * h_kv * seq_len * d  # K (and V) entries
        fp16_bytes = 2 * entries_per_side * 2  # 2 sides, 2 bytes/entry
        bpe_k, bpe_v = self.bits_per_entry()
        # nan (passthrough) => treat as 16 bpe (no compression).
        bpe_k = 16.0 if bpe_k != bpe_k else bpe_k
        bpe_v = 16.0 if bpe_v != bpe_v else bpe_v
        packed_bits = entries_per_side * (bpe_k + bpe_v)
        packed_bytes = packed_bits / 8.0
        return {
            "fp16_bytes": float(fp16_bytes),
            "packed_bytes": float(packed_bytes),
            "compression": fp16_bytes / max(packed_bytes, 1e-9),
        }


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
