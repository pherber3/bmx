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
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from bmx.cache.chunked_attention import chunked_dequant_attention
from bmx.cache.codecs import S_DIVISIBILITY_ARMS, quantize_packed
from bmx.cache.triton_dequant_attention import (
    TRITON_AVAILABLE,
    build_kv_stacked_packed,
    fused_decode_attention_packed,
    triton_decode_attention,
)
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

    Masking (prefill, n_q > 1): the model supplies the exact attention_mask for the
    cached-prefill case (a 4D (b,1,q,kv) tensor that is NOT equivalent to a plain
    is_causal=True bottom-right mask). We thread it into the prefill SDPA path and
    let it do the masking — mirroring stock sdpa_attention_forward, which uses
    is_causal only when attention_mask is None. Using is_causal=True instead of the
    model's mask was a real bug: it produced wrong prefill logits at scale.
    During decode (n_q == 1), no mask is needed — the single query attends to all
    history.
    """
    assert hasattr(module, "_packed_layer"), (
        "PackedStreamingCache.attach(model) must be called before forward"
    )
    layer = module._packed_layer
    n_q = query.shape[2]  # query is (1, n_q_heads, n_q, d)
    q = query.squeeze(0)  # (n_q_heads, n_q, d)
    out = layer.attend(
        q, scaling, is_causal=(n_q > 1), attention_mask=attention_mask
    )  # (n_q_heads, n_q, d)
    n_q_heads, n_q, d = out.shape
    attn_output = out.transpose(0, 1).reshape(1, n_q, n_q_heads * d)
    return attn_output.to(query.dtype), None


ALL_ATTENTION_FUNCTIONS.register(_ATTN_NAME, chunked_attention_forward)
# Register the mask builder too: without this, transformers skips mask creation for
# our custom impl and passes attention_mask=None — which silently falls back to
# is_causal=True in the prefill SDPA path, WRONG for the cached two-block prefill
# (n_q < n_kv). sdpa_mask builds the same 4D causal mask (with correct q/kv offsets)
# the stock 'sdpa' impl receives, so our prefill matches dense bit-for-bit.
ALL_MASK_ATTENTION_FUNCTIONS.register(_ATTN_NAME, sdpa_mask)


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
        # Also tracks the absolute position of self.keys[..., 0, :] after slab
        # pruning (Fix 3) — these two quantities are always equal.
        self._committed_S_q: int = 0

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
            # Cast once at grow-time to the cache compute dtype (fp16), so the
            # decode loop doesn't re-cast the slice every block (deferred opt #2).
            nc, ns = nc.to(torch.float16), ns.to(torch.float16)
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

        # Total sequence length = committed tokens + current slab length.
        # _committed_S_q is the absolute position of slab[..., 0, :] after pruning.
        S = self._committed_S_q + keys.shape[2]  # total tokens in sequence
        W = self.recent_window
        new_S_q = compute_flush_schedule(S, W, self._g)

        if new_S_q > self._committed_S_q:
            block_start = self._committed_S_q
            block_end = new_S_q
            block_len = block_end - block_start
            # block_start == old _committed_S_q == the slab's absolute start, so the
            # newly-committed block is the slab's leading [: block_len] in BOTH K and
            # V, and the slab prune below trims exactly that. (Invariant of the flush
            # schedule: each flush commits the front of the un-flushed slab.)

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
                k_block_fp32 = keys.squeeze(0)[..., :block_len, :].float()
                M = to_matrix(k_block_fp32)
                kpacked, _ = quantize_packed(
                    self.k_spec.arm,
                    M,
                    bits=self.k_spec.bits,
                    group=self.k_spec.group,
                    rank=self.k_spec.rank,
                    seed=self.k_spec.seed,
                )

            # --- Pack V block (leading [: block_len] of the slab, same as K) ---
            v_block_fp32 = values.squeeze(0)[..., :block_len, :].float()
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
            # The slab started at block_start (== old _committed_S_q), so the committed
            # front is the slab's leading [: block_len]; drop it, keeping only the tail.
            # _committed_S_q now tracks the absolute position of self.keys[..., 0, :];
            # attend()/get_seq_length() recover total length as _committed_S_q + slab len.
            keys = keys[..., block_len:, :].contiguous()
            values = values[..., block_len:, :].contiguous()

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
        self,
        q: torch.Tensor,
        scaling: float,
        is_causal: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run chunked dequant-attention for this layer.

        q: (n_q_heads, n_q, d) — already sliced from HF's query tensor.
        is_causal: True during prefill (n_q > 1), False during decode (n_q == 1).
        attention_mask: the model's 4D (b,1,q,kv) mask for the prefill SDPA path;
            when provided it (not is_causal) governs masking, matching stock SDPA.
        Returns (n_q_heads, n_q, d).
        """
        # After Fix 3 (slab pruning), self.keys holds only the tail: the slab starts
        # at absolute position _committed_S_q, so the tail begins at index 0 of the slab.
        k_tail = self.keys.squeeze(0)  # (h_kv, tail_len, d)
        v_tail = self.values.squeeze(0)  # (h_kv, tail_len, d)
        n_q_heads = q.shape[0]
        n_q = q.shape[1]
        n_q_groups = n_q_heads // self._h_kv

        # query_abs_start is the PREFILL GATE for chunked_dequant_attention: set
        # (not-None) iff this is a prefill (n_q > 1), which makes that fn delegate to
        # the dense flash-SDPA path. Its integer value is not used for masking — the
        # model's attn_mask (built via the registered sdpa_mask) handles causality.
        # (Computed as total_seq_len - n_q = the absolute position of query[0], kept
        # as a meaningful value in case a future path needs it.)
        query_abs_start = None
        if is_causal and n_q > 1:
            total_seq_len = self._committed_S_q + self.keys.shape[2]
            query_abs_start = total_seq_len - n_q

        # Dispatch: decode (n_q==1, query_abs_start is None) → Triton kernel when
        # TRITON_AVAILABLE, else chunked PyTorch.  Prefill (n_q>1) always uses
        # chunked (it delegates to flash-SDPA inside chunked_dequant_attention).
        #
        # FAIL-LOUD RULE: TRITON_AVAILABLE is a CAPABILITY check (Triton+CUDA
        # present).  When True, the kernel call is UNCONDITIONAL — no try/except
        # that would silently fall back on a kernel error.  A kernel error must
        # propagate so correctness regressions are never hidden.
        #
        # The k2b (lowrank_rtn_channel K) + pre_rope=True path now applies RoPE to
        # the lowrank-reconstructed K IN-KERNEL (verified vs the chunked reference on
        # GH200), so the full k2b recipe runs on the Triton kernel — no fallback.
        #
        # q.is_cuda is part of the capability check: TRITON_AVAILABLE means Triton+CUDA
        # are INSTALLED, but the model may still run on CPU (e.g. a CPU model on a CUDA
        # box). The Triton kernel needs CUDA tensors — a CPU q means use the chunked
        # path. (A CPU pointer to a Triton kernel raises "cannot be accessed".)
        is_decode = query_abs_start is None  # n_q==1

        # FUSED PACKED fast path (the deployment kernel): single-launch split-KV
        # decode that dequants int8 RTN codes IN-KERNEL (packed-resident, no dense
        # copy) — ~3000x vs chunked, compression preserved. Applies when K and V
        # are plain rtn_token (the packed-stack layout build_kv_stacked_packed
        # assumes) and K is post-RoPE (this kernel has no in-kernel RoPE — k2b /
        # pre-RoPE stay on the per-block triton path below). The fp16 recent-window
        # tail is folded in via the online-softmax merge. Stacks are built per call
        # here for correctness; a production engine maintains them incrementally
        # (the kernel consumes exactly the paged-KV block-table layout).
        # The fused kernel assumes a UNIFORM stored-block length (it pads the row
        # dim to the next power of 2 internally, so blk need not be pow2). The
        # geometric flush schedule normally emits equal-length blocks; on the rare
        # mixed-length tail we fall back to the per-block triton path below.
        blocks = self._k_blocks
        uniform_blk = bool(blocks) and len({e - s for _, s, e in blocks}) == 1

        fused_packed_ok = (
            is_decode
            and q.is_cuda
            and TRITON_AVAILABLE
            and self.k_spec.arm == "rtn_token"
            and self.v_spec.arm == "rtn_token"
            and not self.k_spec.pre_rope
            and uniform_blk
        )
        if fused_packed_ok:
            blk = self._k_blocks[0][2] - self._k_blocks[0][1]  # block length
            n_blocks = len(self._k_blocks)
            seq_len_packed = n_blocks * blk
            k_codes, v_codes, k_scales, v_scales = build_kv_stacked_packed(
                self._k_blocks,
                self._v_blocks,
                max_blocks=n_blocks,
                h_kv=self._h_kv,
                blk_size=blk,
                d=q.shape[2],
                group=self.k_spec.group,
                v_group=self.v_spec.group,
                device=q.device,
            )
            return fused_decode_attention_packed(
                q,
                k_codes,
                v_codes,
                k_scales,
                v_scales,
                seq_len_packed,
                n_q_groups=n_q_groups,
                scale=scaling,
                k_group=self.k_spec.group,
                v_group=self.v_spec.group,
                k_tail=k_tail,
                v_tail=v_tail,
            )

        if TRITON_AVAILABLE and is_decode and q.is_cuda:
            return triton_decode_attention(
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
                n_q_groups=n_q_groups,
                scale=scaling,
                v_group=self.v_spec.group,
                v_seed=self.v_spec.seed,
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
            n_q_groups=n_q_groups,
            scale=scaling,
            query_abs_start=query_abs_start,
            v_group=self.v_spec.group,
            v_seed=self.v_spec.seed,
            attn_mask=attention_mask,
        )


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
