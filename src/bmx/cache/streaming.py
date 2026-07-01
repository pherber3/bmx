"""Live streaming KV cache that quantizes on append (K2c recipe made live).

Mirrors transformers' QuantizedCache/QuantizedLayer split: a per-layer
DynamicLayer subclass (StreamingQuantizedLayer) that stores only the compressed
representation and RETURNS dequantized K/V from update() for attention, plus a
thin Cache container (StreamingQuantizedCache) that replicates the layer across
the model. Because the layer never persists the dense dequant, resident state is
the compressed footprint — real memory by the official cache contract.

Write-once semantics:
  Each token's K/V is quantized EXACTLY ONCE at write time from its pristine fp16
  source, and the dequantised result is frozen in _q_prefix_k/_q_prefix_v.
  Re-quantising a dequantised value is the bug (turboquant_mse is non-idempotent:
  per-token norm rescale compounds => V norm explodes over decode steps).

Frozen subspace:
  For lowrank_rtn_channel K, the channel subspace V is fitted at the FIRST flush
  and reused for all subsequent blocks (_frozen_svd). Per-block Us is computed as
  M_block @ V_frozen (projection onto the frozen subspace).

Memory pruning:
  After committing a pre-RoPE block to _q_prefix_k, the corresponding columns of
  _k_pre are no longer needed (write-once!). We prune _k_pre to keep only the
  un-flushed tail, tracking the offset (_k_pre_offset) so indexing stays correct.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache, DynamicLayer

from bmx.cache.codecs import S_DIVISIBILITY_ARMS, quantize_cache, quantize_kv_layout
from bmx.cache.collect import from_matrix, reshape_heads, to_matrix
from bmx.cache.hf_compat import (
    model_config_n_layers,
    resolve_decoder_layers,
    resolve_text_config,
)
from bmx.cache.rope import apply_rope, rope_cos_sin
from bmx.cache.specs import CacheCodecSpec
from bmx.decomp.lrs import truncated_svd


def compute_flush_schedule(S: int, W: int, g: int) -> int:
    """Largest multiple of g that leaves >= W recent tokens fp16, else 0.

    Single source of truth for the committed-block boundary; both
    StreamingQuantizedLayer and PackedStreamingLayer call this so their schedules
    cannot drift (bit-for-bit parity depends on it).
    """
    return ((S - W) // g) * g if S > W else 0


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
        # _k_pre_offset tracks the absolute sequence position of _k_pre[:, 0, :].
        # After commits, _k_pre is pruned to remove already-committed positions.
        self._k_pre: torch.Tensor | None = None
        self._k_pre_offset: int = 0  # absolute position of _k_pre[0] along seq dim

        # Write-once prefix state.
        # _q_prefix_k/v: frozen dequantized prefix (h, committed_S_q, d); fp16.
        # _committed_S_q: monotonically growing count of quantized tokens.
        self._q_prefix_k: torch.Tensor | None = None
        self._q_prefix_v: torch.Tensor | None = None
        self._committed_S_q: int = 0

        # Frozen subspace: (Us, V) from truncated_svd at first flush.
        # Only used for lowrank_rtn_channel K with pre_rope.
        # V is the (C, rank) channel subspace — frozen across all blocks.
        self._frozen_svd: tuple[torch.Tensor, torch.Tensor] | None = None

        # Honest bpe accounting: track total quantized bits for K and V so blended
        # bpe stays correct as the prefix grows. (entries are recomputed from
        # tail_len / total_entries each step — no separate counter needed.)
        self._quant_bits_k: float = 0.0
        self._quant_bits_v: float = 0.0

        self.bpe_k = float("nan")
        self.bpe_v = float("nan")
        tc = resolve_text_config(model_config)
        self._h_kv = getattr(tc, "num_key_value_heads", tc.num_attention_heads)
        self._d_head = (
            getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
        )
        # Precomputed per-instance constants (avoid re-evaluating each update step).
        self._passthrough = k_spec.arm == "fp16" and v_spec.arm == "fp16"
        self._g = k_spec.group if k_spec.arm in S_DIVISIBILITY_ARMS else 1
        # PAGE: fixed flush-block size (paged-KV layout). MUST match
        # PackedStreamingLayer._page exactly so the two caches flush identical
        # uniform PAGE-token blocks -> bit-for-bit parity (the shared-schedule
        # contract). Multiple of _g; default 128.
        self._page = max(self._g, (128 // self._g) * self._g) if self._g > 1 else 128
        # RoPE cos/sin: one growing (max_S, d_head) table, extended per flush block
        # (covered length is self._rope_cos.shape[0]).
        self._rope_cos: torch.Tensor | None = None
        self._rope_sin: torch.Tensor | None = None

    def stash_pre_rope(self, out: torch.Tensor):
        """Called by the cache's k_proj hook: append a captured pre-RoPE block.

        out: (1, T, h_kv*d) -> reshaped to (h_kv, T, d) fp16, concatenated
        across calls to accumulate the full sequence.
        """
        block = reshape_heads(out, self._h_kv, self._d_head)  # (h_kv, T, d)
        self._k_pre = (
            block if self._k_pre is None else torch.cat([self._k_pre, block], dim=1)
        )

    def _quantize_k_block_pre_rope(
        self,
        k_block_pre: torch.Tensor,
        committed: int,
        new_committed: int,
    ) -> tuple[torch.Tensor, float]:
        """Quantize a pre-RoPE key block and apply RoPE at its TRUE positions.

        Parameters
        ----------
        k_block_pre : (h_kv, block_len, d) fp32  — pristine pre-RoPE source
        committed    : absolute start position of this block in the sequence
        new_committed: absolute end position (exclusive) of this block

        Returns (k_block_post_rope, codec_bpe) — (h_kv, block_len, d) fp32
        """
        spec = self.k_spec
        h = k_block_pre.shape[0]

        if spec.arm == "fp16":
            # fp16 arm: no quantization; just apply RoPE at the correct positions.
            k_hat_pre = k_block_pre
            codec_bpe = 16.0
        elif spec.arm == "lowrank_rtn_channel":
            # Frozen subspace across blocks: fit once at first flush, project thereafter.
            M = to_matrix(k_block_pre)  # (block_len, h*d) fp32
            if self._frozen_svd is None:
                # First flush: fit the SVD and freeze V.
                rank = spec.rank
                Us, V = truncated_svd(M, rank)  # Us:(block_len, r), V:(C, r)
                self._frozen_svd = (Us, V)
            else:
                # Later flushes: project onto the frozen subspace.
                # S-divisibility (lowrank_rtn_channel requires S % group == 0, group >= rank)
                # guarantees block_len >= rank; assert here to document the invariant.
                _, V_frozen = self._frozen_svd
                assert M.shape[0] >= V_frozen.shape[1], (
                    f"block_len={M.shape[0]} < rank={V_frozen.shape[1]}; "
                    "S-divisibility by group (>= rank) must hold"
                )
                # Us_block = M @ V_frozen  (project block rows onto frozen channel subspace)
                Us = M @ V_frozen  # (block_len, rank)
            M_hat, codec_bpe = quantize_cache(
                spec.arm,
                M,
                bits=spec.bits,
                group=spec.group,
                rank=spec.rank,
                svd_factors=(Us, self._frozen_svd[1]),
            )
            k_hat_pre = from_matrix(M_hat, h)  # (h_kv, block_len, d)
        else:
            # General path (rtn_channel, rtn_token, rotate_rtn_token, turboquant_*).
            k_hat_pre, codec_bpe = quantize_kv_layout(k_block_pre, spec)

        # Extend the growing RoPE table to cover [covered, new_committed), then
        # slice this block's positions [committed, new_committed).
        covered = 0 if self._rope_cos is None else self._rope_cos.shape[0]
        if new_committed > covered:
            new_cos, new_sin = rope_cos_sin(
                self.model_config,
                new_committed - covered,
                start=covered,
                device=k_block_pre.device,
            )
            if self._rope_cos is None:
                self._rope_cos, self._rope_sin = new_cos, new_sin
            else:
                self._rope_cos = torch.cat([self._rope_cos, new_cos], dim=0)
                self._rope_sin = torch.cat([self._rope_sin, new_sin], dim=0)
        cos = self._rope_cos[committed:new_committed].float()  # (block_len, d)
        sin = self._rope_sin[committed:new_committed].float()  # (block_len, d)
        k_block_post = apply_rope(k_hat_pre.float(), cos, sin)  # (h_kv, block_len, d)

        return k_block_post, codec_bpe

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

        # Compute the new committed length: largest multiple of PAGE that leaves
        # at least W recent tokens in the fp16 window. Flushing on the PAGE grid
        # (not _g) makes every committed block exactly PAGE tokens — the uniform
        # paged layout, identical to PackedStreamingLayer (shared-schedule parity).
        new_S_q = compute_flush_schedule(S, W, self._page)

        if new_S_q <= self._committed_S_q:
            # No new block to flush — prefix is unchanged. Just reassemble.
            k_tail = keys.squeeze(0)[
                ..., self._committed_S_q :, :
            ]  # (h_kv, tail_len, d)
            v_tail = values.squeeze(0)[..., self._committed_S_q :, :]
            if self._q_prefix_k is not None:
                k_hat = torch.cat([self._q_prefix_k, k_tail.to(cache_dtype)], dim=-2)
                v_hat = torch.cat([self._q_prefix_v, v_tail.to(cache_dtype)], dim=-2)
                self.keys = k_hat.unsqueeze(0)
                self.values = v_hat.unsqueeze(0)
            else:
                self.keys = keys
                self.values = values
            # Recompute blended bpe from accumulated counts.
            tail_len = S - self._committed_S_q
            total_entries = S * self._h_kv * self._d_head
            if total_entries > 0:
                self.bpe_k = (
                    self._quant_bits_k + tail_len * self._h_kv * self._d_head * 16.0
                ) / total_entries
                self.bpe_v = (
                    self._quant_bits_v + tail_len * self._h_kv * self._d_head * 16.0
                ) / total_entries
            return self.keys, self.values

        # --- New region [_committed_S_q : new_S_q] is ready to flush. ---
        # Emit it as uniform PAGE-token blocks (matching PackedStreamingLayer): each
        # page quantized ONCE from pristine source and appended to the frozen prefix.
        for pg0 in range(self._committed_S_q, new_S_q, self._page):
            block_start = pg0
            block_end = pg0 + self._page
            block_len = self._page

            # --- Quantize K page ---
            if self.k_spec.pre_rope:
                assert self._k_pre is not None, (
                    "k_spec.pre_rope=True but no captured pre-RoPE keys; "
                    "call cache.attach(model) before prefill"
                )
                local_start = block_start - self._k_pre_offset
                local_end = block_end - self._k_pre_offset
                k_block_pre = self._k_pre[
                    :, local_start:local_end, :
                ].float()  # (h_kv, PAGE, d)
                k_block_post, codec_bpe_k = self._quantize_k_block_pre_rope(
                    k_block_pre, block_start, block_end
                )
                k_block_post = k_block_post.to(cache_dtype)
            else:
                # Post-RoPE keys: the page is already RoPE'd at its correct positions
                # inside `keys`; pristine because it was in the fp16 tail until now.
                k_block_fp32 = keys.squeeze(0)[..., block_start:block_end, :].float()
                k_block_post_raw, codec_bpe_k = quantize_kv_layout(
                    k_block_fp32, self.k_spec
                )
                k_block_post = k_block_post_raw.to(cache_dtype)

            # --- Quantize V page (pristine fp16 in the tail until now) ---
            v_block_fp32 = values.squeeze(0)[..., block_start:block_end, :].float()
            v_block_raw, codec_bpe_v = quantize_kv_layout(v_block_fp32, self.v_spec)
            v_block = v_block_raw.to(cache_dtype)

            # --- Append page to frozen prefix ---
            if self._q_prefix_k is None:
                self._q_prefix_k = k_block_post
                self._q_prefix_v = v_block
            else:
                self._q_prefix_k = torch.cat([self._q_prefix_k, k_block_post], dim=-2)
                self._q_prefix_v = torch.cat([self._q_prefix_v, v_block], dim=-2)

            # --- Accumulate honest bits ---
            block_entries = block_len * self._h_kv * self._d_head
            self._quant_bits_k += codec_bpe_k * block_entries
            self._quant_bits_v += codec_bpe_v * block_entries

        # --- Update committed counter ---
        self._committed_S_q = new_S_q

        # --- Prune _k_pre to free already-committed positions ---
        # After committing up to new_S_q, positions [_k_pre_offset, new_S_q) are
        # no longer needed. Prune _k_pre to start at new_S_q.
        if self._k_pre is not None and self.k_spec.pre_rope:
            prune_local_end = new_S_q - self._k_pre_offset
            if prune_local_end > 0 and prune_local_end <= self._k_pre.shape[1]:
                self._k_pre = self._k_pre[:, prune_local_end:, :].contiguous()
                self._k_pre_offset = new_S_q
            elif prune_local_end >= self._k_pre.shape[1]:
                # All pre-RoPE data committed; keep empty (None) to signal no tail.
                self._k_pre = None
                self._k_pre_offset = new_S_q

        # --- fp16 tail [new_S_q:S] (pristine, from DynamicLayer) ---
        k_tail = keys.squeeze(0)[..., new_S_q:, :]  # (h_kv, tail_len, d) fp16
        v_tail = values.squeeze(0)[..., new_S_q:, :]  # (h_kv, tail_len, d) fp16

        # --- Reassemble: frozen prefix + fp16 tail ---
        k_hat = torch.cat([self._q_prefix_k, k_tail.to(cache_dtype)], dim=-2)
        v_hat = torch.cat([self._q_prefix_v, v_tail.to(cache_dtype)], dim=-2)

        # --- Blended bpe: quantized prefix costs codec_bpe; fp16 tail costs 16 ---
        tail_len = S - new_S_q
        total_entries = S * self._h_kv * self._d_head
        self.bpe_k = (
            self._quant_bits_k + tail_len * self._h_kv * self._d_head * 16.0
        ) / total_entries
        self.bpe_v = (
            self._quant_bits_v + tail_len * self._h_kv * self._d_head * 16.0
        ) / total_entries

        # Persist the reassembled slab as the layer's stored cache.
        # NOTE: self.keys/self.values is what DynamicLayer uses as the base for
        # the next step's cat. The tail (fp16) region is pristine, so next step
        # DynamicLayer appends new_token to this slab and the new tail stays pristine.
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

        for i, mlayer in enumerate(resolve_decoder_layers(model)):

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
        cfg = resolve_text_config(self.model_config)
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
