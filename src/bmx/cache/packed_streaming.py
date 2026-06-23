"""Packed streaming KV cache: resident packed codes, chunked dequant-attention.

Sibling of StreamingQuantizedCache. Stores per-block PACKED codes (the bpe
footprint) + the frozen subspace + the fp16 recent window — never the dense
dequant prefix or a reassembled dense slab. Attention is routed through
chunked_dequant_attention via the transformers AttentionInterface registry, so
the dense K/V is never materialized. Bit-for-bit parity with
StreamingQuantizedCache is the correctness gate.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache, DynamicLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from bmx.cache.chunked_attention import chunked_dequant_attention
from bmx.cache.codecs import S_DIVISIBILITY_ARMS, quantize_packed
from bmx.cache.collect import _reshape_heads, to_matrix
from bmx.cache.rope import rope_cos_sin
from bmx.cache.specs import CacheCodecSpec
from bmx.cache.streaming import (
    compute_flush_schedule,
    model_config_n_layers,
    resolve_decoder_layers,
    resolve_text_config,
)
from bmx.decomp.lrs import truncated_svd

_ATTN_NAME = "chunked_dequant"


def chunked_attention_forward(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    """Registered attention fn: route through packed chunked dequant-attention.

    query: (1, n_q_heads, n_q, d). Reads packed state off module._packed_layer.
    Returns (attn_output (1, n_q, n_q_heads*d), attn_weights=None) per HF contract.

    The dense key/value tensors passed by HF are ignored — attention is computed
    entirely from the packed blocks stored on module._packed_layer.

    Causal masking: when n_q > 1 (prefill), SDPA applies is_causal=True internally.
    We match this by passing a computed causal attention_mask to chunked_dequant_attention.
    During decode (n_q == 1), no mask is needed — the single query attends to all history.
    """
    layer = module._packed_layer
    n_q = query.shape[2]  # query is (1, n_q_heads, n_q, d)
    q = query.squeeze(0)  # (n_q_heads, n_q, d)
    out = layer.attend(q, scaling, is_causal=(n_q > 1))  # (n_q_heads, n_q, d)
    n_q_heads, n_q, d = out.shape
    attn_output = out.transpose(0, 1).reshape(1, n_q, n_q_heads * d)
    return attn_output.to(query.dtype), None


ALL_ATTENTION_FUNCTIONS.register(_ATTN_NAME, chunked_attention_forward)


class PackedStreamingLayer(DynamicLayer):
    """Per-layer packed streaming cache.

    Stores compressed codes (packed dicts) for flushed blocks, plus the frozen
    low-rank subspace and the fp16 recent window. Attention is routed through
    chunked_dequant_attention, so the dense K/V slab is never materialized.

    The block schedule exactly mirrors StreamingQuantizedLayer via the shared
    compute_flush_schedule — this is the parity invariant.
    """

    def __init__(
        self,
        k_spec: CacheCodecSpec,
        v_spec: CacheCodecSpec,
        model_config,
        recent_window: int = 32,
    ):
        super().__init__()
        self.k_spec = k_spec
        self.v_spec = v_spec
        self.model_config = model_config
        self.recent_window = recent_window

        # Pre-RoPE key buffer (mirrors StreamingQuantizedLayer._k_pre).
        self._k_pre: torch.Tensor | None = None
        self._k_pre_offset: int = 0

        # Committed block count (how many tokens are packed).
        self._committed_S_q: int = 0

        # Absolute position of self.keys[..., 0, :] after slab pruning (Fix 3).
        # Zero until the first flush; updated each time a block is committed.
        self._fp16_offset: int = 0

        # Packed block lists: list of (packed_dict, start, end).
        self._k_blocks: list[tuple[dict, int, int]] = []
        self._v_blocks: list[tuple[dict, int, int]] = []

        # Frozen subspace for lowrank_rtn_channel K (same I1 fix as streaming.py).
        self._frozen_svd: tuple[torch.Tensor, torch.Tensor] | None = None

        # Growing RoPE cos/sin tables, extended on each flush.
        self._rope_cos: torch.Tensor | None = None
        self._rope_sin: torch.Tensor | None = None

        # Head geometry (needed for reshape helpers and n_q_groups).
        tc = resolve_text_config(model_config)
        self._h_kv = getattr(tc, "num_key_value_heads", tc.num_attention_heads)
        self._d_head = (
            getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
        )
        # Group-alignment constant (same as StreamingQuantizedLayer._g).
        self._g = k_spec.group if k_spec.arm in S_DIVISIBILITY_ARMS else 1

    def stash_pre_rope(self, out: torch.Tensor) -> None:
        """Called by the k_proj hook: append captured pre-RoPE keys.

        out: (1, S, h_kv*d) -> reshaped to (h_kv, S, d) fp16, concatenated.
        """
        block = _reshape_heads(out, self._h_kv, self._d_head)  # (h_kv, S, d)
        self._k_pre = (
            block if self._k_pre is None else torch.cat([self._k_pre, block], dim=1)
        )

    def _extend_rope(self, new_committed: int, device: torch.device) -> None:
        """Extend the growing RoPE table to cover [0, new_committed)."""
        covered = 0 if self._rope_cos is None else self._rope_cos.shape[0]
        if new_committed > covered:
            nc, ns = rope_cos_sin(
                self.model_config, new_committed - covered, start=covered, device=device
            )
            if self._rope_cos is None:
                self._rope_cos, self._rope_sin = nc, ns
            else:
                self._rope_cos = torch.cat([self._rope_cos, nc], dim=0)
                self._rope_sin = torch.cat([self._rope_sin, ns], dim=0)

    def _pack_k_block(
        self,
        k_block_pre: torch.Tensor,
        block_start: int,
        block_end: int,
    ) -> dict:
        """Quantize a pre-RoPE K block to packed form.

        k_block_pre: (h_kv, block_len, d) fp32.
        Mirrors the frozen-subspace logic in StreamingQuantizedLayer exactly.
        Returns a packed dict; RoPE is applied at READ (chunked_dequant_attention).
        """
        M = to_matrix(k_block_pre)  # (block_len, h_kv*d)
        spec = self.k_spec

        if spec.arm == "lowrank_rtn_channel":
            if self._frozen_svd is None:
                # First flush: fit the SVD and freeze V (I1 fix, mirrors streaming.py).
                Us, V = truncated_svd(M, spec.rank)
                self._frozen_svd = (Us, V)
            else:
                # Later flushes: project onto frozen subspace.
                _, V_frozen = self._frozen_svd
                Us = M @ V_frozen  # (block_len, rank)
            packed, _ = quantize_packed(
                spec.arm,
                M,
                bits=spec.bits,
                group=spec.group,
                rank=spec.rank,
                svd_factors=(Us, self._frozen_svd[1]),
                seed=spec.seed,
            )
        else:
            packed, _ = quantize_packed(
                spec.arm,
                M,
                bits=spec.bits,
                group=spec.group,
                rank=spec.rank,
                seed=spec.seed,
            )

        # Extend RoPE table to cover this block (needed later in attend()).
        self._extend_rope(block_end, k_block_pre.device)
        return packed

    def _pack_v_block(self, v_block: torch.Tensor) -> dict:
        """Quantize a V block to packed form.

        v_block: (h_kv, block_len, d) fp32.
        """
        M = to_matrix(v_block)  # (block_len, h_kv*d)
        spec = self.v_spec
        packed, _ = quantize_packed(
            spec.arm,
            M,
            bits=spec.bits,
            group=spec.group,
            rank=spec.rank,
            seed=spec.seed,
        )
        return packed

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new KV tokens, flush to packed codes on schedule.

        Returns (keys, values) for HF bookkeeping; attention routes through
        the registered chunked_attention_forward fn instead.
        """
        # Let DynamicLayer concatenate the running slab (post-RoPE, pruned after Fix 3).
        keys, values = super().update(key_states, value_states, *args, **kwargs)

        # Total sequence length = absolute start of slab + current slab length.
        # After Fix 3 pruning, _fp16_offset is the absolute position of slab[...,0,:].
        S = self._fp16_offset + keys.shape[2]  # total tokens in sequence
        W = self.recent_window
        new_S_q = compute_flush_schedule(S, W, self._g)

        if new_S_q > self._committed_S_q:
            block_start = self._committed_S_q
            block_end = new_S_q

            # --- Pack K block ---
            if self.k_spec.pre_rope:
                assert self._k_pre is not None, (
                    "k_spec.pre_rope=True but no pre-RoPE keys captured; "
                    "call cache.attach(model) before prefill"
                )
                local_start = block_start - self._k_pre_offset
                local_end = block_end - self._k_pre_offset
                k_block_pre = self._k_pre[:, local_start:local_end, :].float()
                kpacked = self._pack_k_block(k_block_pre, block_start, block_end)
            else:
                # Post-RoPE: block was pristine fp16 tail until now (same as streaming.py).
                # After slab pruning, keys is the local slab starting at _fp16_offset.
                local_start = block_start - self._fp16_offset
                local_end = block_end - self._fp16_offset
                k_block_fp32 = keys.squeeze(0)[..., local_start:local_end, :].float()
                M = to_matrix(k_block_fp32)
                kpacked, _ = quantize_packed(
                    self.k_spec.arm,
                    M,
                    bits=self.k_spec.bits,
                    group=self.k_spec.group,
                    rank=self.k_spec.rank,
                    seed=self.k_spec.seed,
                )

            # --- Pack V block ---
            # After slab pruning, values is the local slab starting at _fp16_offset.
            local_start = block_start - self._fp16_offset
            local_end = block_end - self._fp16_offset
            v_block_fp32 = values.squeeze(0)[..., local_start:local_end, :].float()
            vpacked = self._pack_v_block(v_block_fp32)

            self._k_blocks.append((kpacked, block_start, block_end))
            self._v_blocks.append((vpacked, block_start, block_end))
            self._committed_S_q = block_end

            # --- C3: Prune _k_pre to free committed positions ---
            if self.k_spec.pre_rope and self._k_pre is not None:
                prune_local_end = block_end - self._k_pre_offset
                if prune_local_end >= self._k_pre.shape[1]:
                    self._k_pre = None
                    self._k_pre_offset = block_end
                elif prune_local_end > 0:
                    self._k_pre = self._k_pre[:, prune_local_end:, :].contiguous()
                    self._k_pre_offset = block_end

            # --- Fix 3: Prune fp16 slab to tail-only ---
            # Committed region lives solely as packed codes in _k_blocks/_v_blocks.
            # After pruning, keys/values hold only [block_end:] in ABSOLUTE terms,
            # which is [local_prune:] in the slab that starts at _fp16_offset.
            # _fp16_offset tracks the absolute position of self.keys[..., 0, :].
            # attend() computes total_seq_len = _committed_S_q + self.keys.shape[2],
            # and get_seq_length() returns the same value.
            local_prune = block_end - self._fp16_offset
            keys = keys[..., local_prune:, :].contiguous()
            values = values[..., local_prune:, :].contiguous()
            self._fp16_offset = block_end

        # Store pruned (or full, if no flush this step) slab.
        self.keys = keys
        self.values = values
        return keys, values

    def get_seq_length(self) -> int:
        """Total sequence length = committed tokens + resident fp16 slab length."""
        if not self.is_initialized or self.keys is None or self.keys.numel() == 0:
            return 0
        return self._committed_S_q + self.keys.shape[-2]

    def attend(
        self, q: torch.Tensor, scaling: float, is_causal: bool = False
    ) -> torch.Tensor:
        """Run chunked dequant-attention for this layer.

        q: (n_q_heads, n_q, d) — already sliced from HF's query tensor.
        is_causal: apply causal masking (True during prefill when n_q > 1, matching
            SDPA's is_causal=True; False during decode when n_q == 1).
        Returns (n_q_heads, n_q, d).
        """
        tail_start = self._committed_S_q
        # After Fix 3 (slab pruning), self.keys holds only the tail starting at
        # absolute position _fp16_offset. Slicing [tail_start:] on the pruned slab
        # would overshoot; use (tail_start - _fp16_offset) as the local index.
        local_tail_start = tail_start - self._fp16_offset
        k_tail = self.keys.squeeze(0)[..., local_tail_start:, :]  # (h_kv, tail_len, d)
        v_tail = self.values.squeeze(0)[
            ..., local_tail_start:, :
        ]  # (h_kv, tail_len, d)
        n_q_heads = q.shape[0]
        n_q = q.shape[1]
        n_q_groups = n_q_heads // self._h_kv

        if is_causal and n_q > 1:
            # Prefill: causal attention to match SDPA's is_causal=True behaviour.
            # Use absolute-position arithmetic in _attend_causal so both committed
            # blocks and the tail are masked correctly.
            # total_seq_len = committed + current fp16 slab length; query[0] is at
            # absolute position (total_seq_len - n_q).
            total_seq_len = self._committed_S_q + self.keys.shape[2]
            query_abs_start = total_seq_len - n_q
            return self._attend_causal(
                q,
                k_tail,
                v_tail,
                scaling,
                n_q_groups,
                query_abs_start,
            )

        return chunked_dequant_attention(
            q,
            self._k_blocks,
            self._v_blocks,
            k_arm=self.k_spec.arm,
            v_arm=self.v_spec.arm,
            group=self.k_spec.group,
            seed=self.k_spec.seed,
            k_pre_rope=self.k_spec.pre_rope,
            rope_cos=self._rope_cos,
            rope_sin=self._rope_sin,
            k_tail=k_tail,
            v_tail=v_tail,
            tail_start=tail_start,
            n_q_groups=n_q_groups,
            scale=scaling,
        )

    def _attend_causal(
        self,
        q: torch.Tensor,
        k_tail: torch.Tensor,
        v_tail: torch.Tensor,
        scaling: float,
        n_q_groups: int,
        query_abs_start: int = 0,
    ) -> torch.Tensor:
        """Causal attention for prefill (n_q > 1) matching SDPA's is_causal=True.

        Absolute-position rule: query at absolute position (query_abs_start + qi)
        must not attend key at absolute position j where j > query_abs_start + qi.
        This applies uniformly to committed blocks and the tail.

        Returns (n_q_heads, n_q, d).
        """
        from bmx.cache.chunked_attention import (
            _dequant_block,
            online_softmax_update,
        )
        from bmx.cache.rope import apply_rope

        n_q_heads, n_q, d = q.shape
        h_kv = n_q_heads // n_q_groups
        acc = torch.zeros(n_q_heads, n_q, d, dtype=q.dtype, device=q.device)
        m = torch.full(
            (n_q_heads, n_q, 1), float("-inf"), dtype=q.dtype, device=q.device
        )
        lse = torch.zeros(n_q_heads, n_q, 1, dtype=q.dtype, device=q.device)

        # Absolute positions of each query: shape (n_q,)
        q_abs = torch.arange(query_abs_start, query_abs_start + n_q, device=q.device)

        # Committed blocks: apply absolute-position causal mask so query qi
        # cannot see key at block position j where (start + j) > q_abs[qi].
        for (kpacked, start, end), (vpacked, _vs, _ve) in zip(
            self._k_blocks, self._v_blocks
        ):
            K_kv = _dequant_block(
                kpacked, self.k_spec.arm, self.k_spec.group, self.k_spec.seed, h_kv
            ).to(q.dtype)
            if self.k_spec.pre_rope:
                K_kv = apply_rope(
                    K_kv,
                    self._rope_cos[start:end].to(q.dtype),
                    self._rope_sin[start:end].to(q.dtype),
                )
            V_kv = _dequant_block(
                vpacked, self.v_spec.arm, self.v_spec.group, self.v_spec.seed, h_kv
            ).to(q.dtype)
            K_exp = K_kv.repeat_interleave(n_q_groups, dim=0)
            V_exp = V_kv.repeat_interleave(n_q_groups, dim=0)
            s = (q @ K_exp.transpose(-1, -2)) * scaling  # (n_q_heads, n_q, blk)
            # key_abs[j] = start + j; mask where key_abs[j] > q_abs[qi].
            key_abs = torch.arange(start, end, device=q.device)  # (blk,)
            # causal_mask[qi, j] = True means MASK OUT (future key).
            causal_mask = key_abs.unsqueeze(0) > q_abs.unsqueeze(1)  # (n_q, blk)
            s = s.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
            acc, m, lse = online_softmax_update(acc, m, lse, s, V_exp)

        # Tail: apply absolute-position causal mask.
        # key at tail-position j has absolute position (tail_start + j);
        # mask out where (tail_start + j) > q_abs[qi].
        if k_tail is not None and k_tail.shape[1] > 0:
            tail_start = self._committed_S_q
            tail_len = k_tail.shape[1]
            K_exp = k_tail.to(q.dtype).repeat_interleave(n_q_groups, dim=0)
            V_exp = v_tail.to(q.dtype).repeat_interleave(n_q_groups, dim=0)
            s = (q @ K_exp.transpose(-1, -2)) * scaling  # (n_q_heads, n_q, tail_len)
            key_abs = torch.arange(
                tail_start, tail_start + tail_len, device=q.device
            )  # (tail_len,)
            # causal_mask[qi, j] = True means MASK OUT (future key).
            causal_mask = key_abs.unsqueeze(0) > q_abs.unsqueeze(1)  # (n_q, tail_len)
            s = s.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
            acc, m, lse = online_softmax_update(acc, m, lse, s, V_exp)

        return acc / lse


class PackedStreamingCache(Cache):
    """Cache container replicating PackedStreamingLayer across the model.

    Drop-in ``past_key_values=`` for model() / model.generate(). Registers a
    custom attention fn via the transformers AttentionInterface so attention
    routes through chunked_dequant_attention rather than materializing dense K/V.

    Use as a context manager or call attach()/detach() manually:

        cache = PackedStreamingCache(model.config, k_spec=k_spec, v_spec=v_spec)
        cache.attach(model)
        out = model.generate(..., past_key_values=cache)
        cache.detach()
    """

    def __init__(
        self,
        model_config,
        k_spec: CacheCodecSpec,
        v_spec: CacheCodecSpec,
        recent_window: int = 32,
    ):
        super().__init__(
            layer_class_to_replicate=lambda: PackedStreamingLayer(
                k_spec, v_spec, model_config, recent_window
            )
        )
        self.model_config = model_config
        self.k_spec = k_spec
        self.v_spec = v_spec
        self.recent_window = recent_window
        self._handles: list = []
        self._saved_impl: str | None = None
        self._model = None

    def attach(self, model) -> "PackedStreamingCache":
        """Register the chunked-dequant attention fn and k_proj hooks.

        Sets model.config._attn_implementation = "chunked_dequant" so HF routes
        every attention call to chunked_attention_forward, which reads packed state
        off module._packed_layer. Saves and restores the prior implementation on
        detach().
        """
        self.detach()  # Clear any previously-registered hooks (idempotence).
        self._model = model
        self._saved_impl = model.config._attn_implementation
        model.config._attn_implementation = _ATTN_NAME

        # Pre-size layers so hooks can find self.layers[i] before the first update.
        n_layers = model_config_n_layers(model)
        while len(self.layers) < n_layers:
            self.layers.append(
                PackedStreamingLayer(
                    self.k_spec, self.v_spec, self.model_config, self.recent_window
                )
            )

        for i, mlayer in enumerate(resolve_decoder_layers(model)):
            # Back-reference so chunked_attention_forward can find this layer's state.
            mlayer.self_attn._packed_layer = self.layers[i]

            if self.k_spec.pre_rope:

                def k_hook(module, inp, out, i=i):
                    self.layers[i].stash_pre_rope(out)

                self._handles.append(
                    mlayer.self_attn.k_proj.register_forward_hook(k_hook)
                )
        return self

    def detach(self) -> "PackedStreamingCache":
        """Remove all hooks and restore the saved attention implementation."""
        for h in self._handles:
            h.remove()
        self._handles = []
        # Fix 4: remove the _packed_layer back-reference so the model's attention
        # modules do not hold a circular reference to this cache after detach.
        if self._model is not None:
            for mlayer in resolve_decoder_layers(self._model):
                if hasattr(mlayer.self_attn, "_packed_layer"):
                    del mlayer.self_attn._packed_layer
        if self._model is not None and self._saved_impl is not None:
            self._model.config._attn_implementation = self._saved_impl
        self._model = None
        self._saved_impl = None
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.detach()
        return False
