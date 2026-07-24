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
  For the lowrank K arms (_FROZEN_SUBSPACE_ARMS), the channel subspace V is
  fitted at the FIRST flush and reused for all subsequent blocks (_frozen_svd).
  Per-block Us is computed as M_block @ V_frozen (projection onto the frozen
  subspace).

Memory pruning:
  After committing a pre-RoPE block to _q_prefix_k, the corresponding columns of
  _k_pre are no longer needed (write-once!). We prune _k_pre to keep only the
  un-flushed tail, tracking the offset (_k_pre_offset) so indexing stays correct.

Batched flush:
  Batchable codecs quantize many pages per codec call (super-block capped),
  bit-identical to the per-page loop — see _flush_batchable for the full
  license and tests/test_streaming_batched_flush.py for the bitwise A/B gate.
"""

from __future__ import annotations

import dataclasses
import weakref

import torch
from transformers.cache_utils import Cache, DynamicLayer

from bmx.cache.codecs import S_DIVISIBILITY_ARMS, quantize_cache, quantize_kv_layout
from bmx.cache.collect import from_matrix, reshape_heads, to_matrix
from bmx.cache.hf_compat import (
    model_config_n_layers,
    resolve_decoder_layers,
    resolve_qk_capture_modules,
    resolve_text_config,
)
from bmx.cache.rope import apply_rope, rope_cos_sin
from bmx.cache.specs import CacheCodecSpec
from bmx.cache.spectral import (
    SpectralPack,
    int8_decoder_roundtrip,
    load_packs,
    skeptic_charge,
    spectral_quantize,
)
from bmx.decomp.lrs import truncated_svd
from bmx.quant.hadamard import is_power_of_2

# Lowrank K arms that share the frozen pre-RoPE subspace path: V fitted once at
# the first flush, later blocks projected onto it (never refit per block). Both
# accept svd_factors=(Us, V) in quantize_cache; they differ only in how the
# post-lowrank residual is coded (per-channel RTN vs turboquant-MSE).
_FROZEN_SUBSPACE_ARMS = frozenset({"lowrank_rtn_channel", "lowrank_turboquant"})


def _flush_batchable(
    spec: CacheCodecSpec,
    h_kv: int,
    d_head: int,
    page: int,
    pack: SpectralPack | None = None,
) -> bool:
    """True iff N stacked PAGE-token pages can be quantized in ONE codec call
    bit-identically to N per-page calls (the batched-flush license).

    Group/window structure never spans a page boundary for the licensed arms —
    rtn_token groups run along C (per-row); rtn_channel groups run along S but
    stay page-aligned when PAGE % group == 0; turboquant norms are per-(row,
    head); the Hadamard rotation is per-row butterflies (fwht). That makes the
    elementwise/per-group arms bit-identical by construction; the per-row NORM
    reductions (turboquant_*) and spectral's enc/dec matmuls are pinned by the
    CPU A/B oracle (tests/test_streaming_batched_flush.py) and re-pinned on
    CUDA by the same test's cuda parametrization at the GH200 gate — reduction
    split config and BLAS kernel selection are shape-dependent there, so CPU
    equality alone does not prove it. GEMM-bearing codecs (lowrank_* factor
    products, turboquant_prod's QJL sketch, the random-orthogonal rotation
    fallback when the rotated dim is not a power of 2) stay on the per-page
    loop: for the lowrank_* arms even a passing CUDA A/B would not license
    batching, because PackedStreamingLayer packs the same pages per page at
    (PAGE, C) shapes (the GH200 streaming-vs-packed parity gate), and their
    factor_bits accounting is S-dependent.

    spectral is the licensed exception despite its enc/dec matmuls: it is
    pack-gated and never routes through PackedStreamingCache, so no packed
    twin — hence no GH200 packed-parity constraint — exists for it; its only
    bitwise gate is the CPU streaming-vs-offline replay, and the batched A/B
    test pins the enc/dec row-batch invariance there. Its RTN group windows
    run along S and stay page-aligned (PAGE % pack.group == 0, checked below),
    so per-group scales and mse_scale iterations are identical either way.
    Whole-span is also the call granularity the offline G1 gauntlet
    (k4_frontier/k4_spectra) actually measured — one spectral_quantize per
    matrix — so the per-page loop was the deviation, not the batch.
    """
    C = h_kv * d_head
    if spec.arm in ("fp16", "rtn_token"):
        return True
    if spec.arm == "rtn_channel":
        # Groups run along S: page-aligned iff group divides PAGE (guaranteed
        # when rtn_channel is the K arm by _page's construction; a V group that
        # doesn't divide PAGE keeps the per-page loop — and its S % group
        # assert — exactly as before).
        return page % spec.group == 0
    if spec.arm in ("rotate_rtn_token", "turboquant_mse"):
        return is_power_of_2(C)  # fwht path only; non-pow2 C rotates via a GEMM
    if spec.arm == "turboquant_mse_perhead":
        return is_power_of_2(d_head)  # per-head fwht over d_head
    if spec.arm == "spectral":
        # Alignment is checked against the pack's own group (sidecar-loaded) —
        # that is what spectral_quantize asserts along S, not spec.group.
        return pack is not None and page % pack.group == 0
    return False


# Super-block cap for batched flush spans, in pages: batchable codecs quantize
# at most this many pages per codec call. Caps the transient fp32
# materialization of the flush span (an uncapped whole-span flush regressed
# peak memory exactly where the 128k margin is thin) while keeping call counts
# tiny — 64 pages ≈ 8k tokens holds a 31.5k prefill to ~4 codec calls per side.
_SUPER_PAGES = 64


def _flush_spans(
    start: int, end: int, page: int, batchable: bool
) -> list[tuple[int, int]]:
    """Codec-call spans covering [start, end) on the PAGE grid: super-block
    spans (<= _SUPER_PAGES pages each) for batchable codecs; one span per page
    for loop-kept codecs, so every matmul runs at the reference (PAGE, C) shape.
    """
    step = page * _SUPER_PAGES if batchable else page
    return [(s, min(s + step, end)) for s in range(start, end, step)]


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

    def __init__(
        self,
        k_spec,
        v_spec,
        model_config,
        recent_window: int = 32,
        pack: SpectralPack | None = None,
    ):
        super().__init__()
        self.k_spec = k_spec
        self.v_spec = v_spec
        self.model_config = model_config
        self.recent_window = recent_window
        # Corpus-fit spectral pack for this layer (k_spec.arm == "spectral" only);
        # handed in once at cache-init time (loaded once at the cache level).
        self._pack = pack
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
        # Only used for _FROZEN_SUBSPACE_ARMS K with pre_rope.
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
        # Batched-flush license per side (full rationale: _flush_batchable).
        self._k_flush_batchable = _flush_batchable(
            k_spec, self._h_kv, self._d_head, self._page, pack=pack
        )
        self._v_flush_batchable = _flush_batchable(
            v_spec, self._h_kv, self._d_head, self._page
        )
        # RoPE cos/sin: one growing (max_S, d_head) table, extended per flush block
        # (covered length is self._rope_cos.shape[0]).
        self._rope_cos: torch.Tensor | None = None
        self._rope_sin: torch.Tensor | None = None

    def stash_pre_rope(self, out: torch.Tensor):
        """Called by the cache's QK-capture hook: append a captured pre-RoPE block.

        out: (1, T, h_kv*d) (k_proj) or (1, T, h_kv, d) (k_norm) -> reshaped
        to (h_kv, T, d) fp16, concatenated across calls to accumulate the
        full sequence.
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
        elif spec.arm in _FROZEN_SUBSPACE_ARMS:
            # Frozen subspace across blocks: fit once at first flush, project thereafter.
            M = to_matrix(k_block_pre)  # (block_len, h*d) fp32
            if self._frozen_svd is None:
                # First flush: fit the SVD and freeze V.
                rank = spec.rank
                Us, V = truncated_svd(M, rank)  # Us:(block_len, r), V:(C, r)
                self._frozen_svd = (Us, V)
            else:
                # Later flushes: project onto the frozen subspace.
                # block_len is a full PAGE (>= 128 >= rank); for
                # lowrank_rtn_channel S-divisibility (S % group == 0, group >=
                # rank) also guarantees it. Assert to document the invariant.
                _, V_frozen = self._frozen_svd
                assert M.shape[0] >= V_frozen.shape[1], (
                    f"block_len={M.shape[0]} < rank={V_frozen.shape[1]}; "
                    "flush blocks must be at least rank tokens long"
                )
                # Us_block = M @ V_frozen  (project block rows onto frozen channel subspace)
                Us = M @ V_frozen  # (block_len, rank)
            M_hat, codec_bpe = quantize_cache(
                spec.arm,
                M,
                bits=spec.bits,
                seed=spec.seed,  # lowrank_turboquant's residual rotation is seeded
                group=spec.group,
                rank=spec.rank,
                svd_factors=(Us, self._frozen_svd[1]),
            )
            k_hat_pre = from_matrix(M_hat, h)  # (h_kv, block_len, d)
        elif spec.arm == "spectral":
            # Corpus-fit spectral basis (frozen pack, loaded once at cache init):
            # no per-block fitting at all, unlike the frozen-subspace arms above.
            # codec_bpe here is the model-level payload+scale bpe; the per-sequence
            # skeptic pack charge is added once in bits_per_entry(), not per block.
            M = to_matrix(k_block_pre)  # (block_len, h*d) fp32
            if self._pack.enc.device != k_block_pre.device:
                # One-time device placement (packs load on CPU; move once, not per block).
                self._pack = dataclasses.replace(
                    self._pack,
                    enc=self._pack.enc.to(k_block_pre.device),
                    dec=self._pack.dec.to(k_block_pre.device),
                    lam=self._pack.lam.to(k_block_pre.device),
                    bits=self._pack.bits.to(k_block_pre.device),
                )
            M_hat, codec_bpe = spectral_quantize(M, self._pack, mse_scale=True)
            k_hat_pre = from_matrix(M_hat, h)  # (h_kv, block_len, d)
        else:
            # General path (rtn_channel, rtn_token, rotate_rtn_token, turboquant_*).
            k_hat_pre, codec_bpe = quantize_kv_layout(k_block_pre, spec)

        # Extend the growing RoPE table to cover [covered, new_committed), then
        # slice this block's positions [committed, new_committed). Exact for the
        # static rope_scaling types this repo targets (default, llama3 — pure
        # outer products, position-independent); a "dynamic" NTK config would
        # recompute inv_freq from max position, making one big extension differ
        # from page-sized ones — batching would need a guard there.
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

    def _quantize_k_flush(self, keys, start: int, end: int):
        """Quantize K tokens [start, end) from their pristine source.

        Pristine source: the pre-RoPE capture buffer when k_spec.pre_rope (RoPE
        applied at true positions after quantization), else the post-RoPE fp16
        region inside `keys` — already RoPE'd at its correct positions, pristine
        because it was in the fp16 tail until now. Called once with the whole
        flush span (batchable codecs) or once per PAGE (GEMM-bearing codecs) —
        one body for both paths so they cannot drift.

        Returns (k_post (h_kv, end-start, d) fp32, codec_bpe).
        """
        if self.k_spec.pre_rope:
            lo = start - self._k_pre_offset
            k_block_pre = self._k_pre[:, lo : lo + (end - start), :].float()
            return self._quantize_k_block_pre_rope(k_block_pre, start, end)
        k_block_fp32 = keys.squeeze(0)[..., start:end, :].float()
        return quantize_kv_layout(k_block_fp32, self.k_spec)

    def _quantize_v_flush(self, values, start: int, end: int):
        """Quantize V tokens [start, end) (pristine fp16 in the tail until now)."""
        v_block_fp32 = values.squeeze(0)[..., start:end, :].float()
        return quantize_kv_layout(v_block_fp32, self.v_spec)

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
            # No new block to flush — prefix is unchanged. `keys`/`values` (from
            # super().update() above) already equal [prefix | tail | new_token]:
            # the previous step stored exactly [prefix | tail] as self.keys, and
            # DynamicLayer's update() concatenated the new token onto that same
            # storage. Slicing the tail back out and re-catting it onto the
            # prefix would just reconstruct this identical tensor, so we assign
            # it directly (bit-identical by construction; cache_dtype is
            # unchanged here so no dtype cast is even needed).
            self.keys, self.values = keys, values
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
        # Span granularity per side comes from _flush_spans — super-block spans for
        # batchable codecs, per-page spans for the loop-kept GEMM-bearing codecs
        # (license: _flush_batchable). Blocks collect into ONE torch.cat per side
        # below (the per-page cat was an O(S²/PAGE) prefix re-copy at prefill).
        committed = self._committed_S_q
        page_entries = self._page * self._h_kv * self._d_head
        if self.k_spec.pre_rope:
            assert self._k_pre is not None, (
                "k_spec.pre_rope=True but no captured pre-RoPE keys; "
                "call cache.attach(model) before prefill"
            )

        # --- Quantize K spans ---
        new_k_blocks: list[torch.Tensor] = []
        for s0, s1 in _flush_spans(
            committed, new_S_q, self._page, self._k_flush_batchable
        ):
            k_block, codec_bpe_k = self._quantize_k_flush(keys, s0, s1)
            new_k_blocks.append(k_block.to(cache_dtype))
            # Batchable codecs report an S/data-independent codec_bpe — the exact
            # float every per-page call returned — so accumulating page-by-page
            # reproduces the per-page loop's float sum bit-for-bit (pinned by the
            # bitwise oracle; do NOT collapse into a multiply).
            for _ in range((s1 - s0) // self._page):
                self._quant_bits_k += codec_bpe_k * page_entries

        # --- Quantize V spans (pristine fp16 in the tail until now) ---
        new_v_blocks: list[torch.Tensor] = []
        for s0, s1 in _flush_spans(
            committed, new_S_q, self._page, self._v_flush_batchable
        ):
            v_block, codec_bpe_v = self._quantize_v_flush(values, s0, s1)
            new_v_blocks.append(v_block.to(cache_dtype))
            for _ in range((s1 - s0) // self._page):
                self._quant_bits_v += codec_bpe_v * page_entries

        # --- Append to frozen prefix: ONE cat per side per update ---
        k_parts = (
            [self._q_prefix_k] if self._q_prefix_k is not None else []
        ) + new_k_blocks
        v_parts = (
            [self._q_prefix_v] if self._q_prefix_v is not None else []
        ) + new_v_blocks
        self._q_prefix_k = (
            k_parts[0] if len(k_parts) == 1 else torch.cat(k_parts, dim=-2)
        )
        self._q_prefix_v = (
            v_parts[0] if len(v_parts) == 1 else torch.cat(v_parts, dim=-2)
        )

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
        # Spectral K arm: pre-RoPE only, and the corpus pack file is loaded ONCE
        # here (never per-layer, never per-call) then handed out by layer_idx.
        assert k_spec.dec_quant in ("fp32", "int8"), (
            f"dec_quant must be 'fp32' or 'int8'; got {k_spec.dec_quant!r}"
        )
        self._packs: dict[int, SpectralPack] = {}
        if k_spec.arm == "spectral":
            assert k_spec.pre_rope, (
                "spectral quantizes pre-RoPE keys; set pre_rope=True"
            )
            assert k_spec.pack_path, "spectral requires pack_path"
            self._packs = load_packs(k_spec.pack_path, k_spec.budget)
            if k_spec.dec_quant == "int8":
                # Lever 2 (gated on a later VM quality measurement): roundtrip
                # each layer pack's decoder through int8 ONCE, here, at init --
                # never refit, never re-applied per call. dataclasses.replace
                # keeps every other pack field (enc, lam, bits, tiers, ...)
                # untouched; only dec's stored precision changes.
                self._packs = {
                    i: dataclasses.replace(
                        pack, dec=int8_decoder_roundtrip(pack.dec, pack.bits)
                    )
                    for i, pack in self._packs.items()
                }

        # layer_class_to_replicate lazily appends one layer per new layer_idx, always
        # in order (transformers' Cache.update appends up to layer_idx while
        # len(self.layers) <= layer_idx) -- so len(self.layers) at call time IS the
        # new layer's index. Used to hand each layer its own pack. This closure only
        # ever EXECUTES from inside a later update() call (never during __init__
        # itself) OR from attach()'s pre-size loop below, by which point
        # self.k_spec/_pack_for_layer are fully set up below. Stored as
        # self._make_layer so attach() reuses the exact same construction path
        # instead of re-spelling the constructor call.
        # weakref, not self: this closure is stored on self, so a strong capture
        # is a self->closure->self cycle — the cache (holding the full dequantized
        # fp16 K/V) then survives until a gen-2 gc pass instead of dying by
        # refcount; cycle-trapped caches accumulated to an 88 GiB CUDA OOM across
        # 31.5k-token LongBench shards. The proxy is only dereferenced while the
        # cache is alive (update()/attach() call through self).
        wself = weakref.proxy(self)

        def _make_layer():
            return StreamingQuantizedLayer(
                k_spec,
                v_spec,
                model_config,
                recent_window,
                pack=wself._pack_for_layer(len(wself.layers)),
            )

        self._make_layer = _make_layer
        super().__init__(layer_class_to_replicate=_make_layer)
        self.model_config = model_config
        self.k_spec = k_spec
        self.v_spec = v_spec
        self.recent_window = recent_window
        self._handles: list = []
        # dec_quant=="int8" charges the skeptic decoder term at 8 bits/entry
        # (the pack's dec was already roundtripped to int8 precision above);
        # bits_per_entry()'s spectral branch reads this.
        self._dec_bits = 8.0 if k_spec.dec_quant == "int8" else 16.0

    def _pack_for_layer(self, i: int) -> "SpectralPack | None":
        """Look up layer i's spectral pack, asserting it's present when spectral."""
        if self.k_spec.arm != "spectral":
            return None
        assert i in self._packs, f"spectral pack file missing layer {i}"
        return self._packs[i]

    def attach(self, model) -> "StreamingQuantizedCache":
        """Register pre-RoPE capture hooks (resolved QK-capture modules).

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
        # Reuses the same _make_layer closure __init__ hands to
        # layer_class_to_replicate, so there is one construction path, not two.
        n_layers = model_config_n_layers(model)
        while len(self.layers) < n_layers:
            self.layers.append(self._make_layer())

        decoder_layers = resolve_decoder_layers(model)
        if not hasattr(decoder_layers[0], "self_attn") or not hasattr(
            decoder_layers[0].self_attn, "k_proj"
        ):
            raise ValueError(
                f"unsupported architecture {model.config.model_type!r} for pre-RoPE "
                "streaming: attach() hooks the resolved QK-capture modules (Llama k_proj / "
                "Qwen3 k_norm); GPT-2-style "
                "fused c_attn is not supported"
            )

        for i, mlayer in enumerate(decoder_layers):

            def k_hook(module, inp, out, i=i):
                self.layers[i].stash_pre_rope(out)

            _, k_mod = resolve_qk_capture_modules(mlayer.self_attn)
            self._handles.append(k_mod.register_forward_hook(k_hook))
        return self

    def detach(self) -> "StreamingQuantizedCache":
        """Remove all registered capture hooks."""
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
        """(bpe_k, bpe_v) — blended payload bpe averaged across layers.

        The across-layer mean, not layers[-1]: allocated spectral packs give
        each layer its own budget (mean-preserving), so any single layer's bpe
        misstates the cache (layer 31 draws the floor budget and under-reported
        the whole cache by ~1 bit until this was caught). For uniform packs and
        every non-spectral arm the layers are identical, so the mean reproduces
        the old single-layer value exactly.

        For the spectral K arm, bpe_k is the layer's blended block-payload bpe
        (quantized-prefix codec_bpe + fp16-tail 16.0, both already blended in
        StreamingQuantizedLayer.update — see bpe_k there) PLUS the per-sequence
        skeptic pack charge, skeptic-v2 (used-columns; see spectral.py's
        "Accounting modes" docstring):

            bpe_k = (blended block payload bpe)
                    + skeptic_charge(C, S, tiers, c_used=mean_c_used, dec_bits=self._dec_bits)

        where mean_c_used = mean over layers of layer._pack.c_used (the number
        of decoder columns actually carrying nonzero bits — see
        SpectralPack.c_used). skeptic_charge is linear in c_used, so the
        across-layer mean of per-layer v2 charges equals the charge evaluated
        at mean_c_used; this is exact for allocated packs too, where each
        layer's own budget gives it a different c_used (range 139-423 zero
        dirs at b2.5). skeptic-v1 (the expression every parquet before
        2026-07-23 measured) was the same call with c_used left at its
        default (None -> C, the full decoder charged regardless of how many
        columns are actually read at decode):

            bpe_k = (blended block payload bpe) + skeptic_charge(C, S, tiers)

        where S = last.get_seq_length() (the full committed sequence length, fp16
        tail included — NOT just the quantized-committed count) and
        C = h_kv * d_head. This charge amortizes the corpus pack's decoder matrix
        + tier map over the WHOLE sequence once here, rather than re-charging it
        per flushed block (skeptic_charge's own S-amortization already does the
        1/S scaling; calling it once per block would double-count blocks that
        preceded S growing).
        """
        if not self.layers:
            return float("nan"), float("nan")
        last = self.layers[-1]
        bpe_k = sum(layer.bpe_k for layer in self.layers) / len(self.layers)
        bpe_v = sum(layer.bpe_v for layer in self.layers) / len(self.layers)
        # Only charge once something has actually flushed through the spectral
        # path (_committed_S_q > 0); an all-fp16 cache (nothing quantized yet)
        # must report bpe_k == 16.0, not 16.0 + a charge for an unused pack.
        # The charge is layer-independent (same C/S/tiers) but NOT c_used-
        # independent (each layer's pack can have a different c_used under
        # per-layer allocation), so the mean_c_used passed to skeptic_charge
        # must be computed BEFORE this single call — linearity in c_used
        # (see spectral.skeptic_charge) makes that equal to averaging
        # per-layer charged values.
        if self.k_spec.arm == "spectral" and last._committed_S_q > 0:
            S = last.get_seq_length()
            C = last._h_kv * last._d_head
            mean_c_used = sum(layer._pack.c_used for layer in self.layers) / len(
                self.layers
            )
            bpe_k = bpe_k + skeptic_charge(
                C, S, last._pack.tiers, c_used=mean_c_used, dec_bits=self._dec_bits
            )
        return bpe_k, bpe_v

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
