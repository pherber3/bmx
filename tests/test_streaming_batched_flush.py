"""Batched prefill flush: bitwise A/B against the pre-batching per-page loop.

StreamingQuantizedLayer.update()'s flush section used to quantize committed
pages one PAGE(=128)-token page at a time — a Python loop of small codec calls
plus an O(S²/PAGE) per-page prefix re-cat (measured: ~7,800 codec calls per
31.5k-token LongBench sample). The batched flush processes all newly-flushable
pages of one update() in ONE codec call per side (for the batchable, GEMM-free
arms) and reassembles the prefix with ONE torch.cat — and must be BITWISE
identical to the per-page loop for every arm (the non-negotiable gate: the
streaming-vs-packed parity tests depend on it).

``_reference_update`` below is the per-page implementation copied VERBATIM
from streaming.py as of 562d696 — the oracle each A/B test replays through a
second cache via monkeypatch, so the comparison is old-code-verbatim vs new,
end-to-end (all layers, multi-page and 1-token-append flushes, no-flush steps).
"""

import pytest
import torch
from transformers.cache_utils import DynamicLayer

from bmx.cache.codecs import quantize_kv_layout
from bmx.cache.specs import CacheCodecSpec
from bmx.cache.streaming import (
    StreamingQuantizedCache,
    StreamingQuantizedLayer,
    compute_flush_schedule,
)
from tests.factories import ids, tiny_llama
from tests.test_streaming_spectral import _fit_tiny_packs

# ---------------------------------------------------------------------------
# Reference oracle: the per-page flush loop, copied VERBATIM from
# StreamingQuantizedLayer.update() at 562d696. Only mechanical change:
# ``super().update`` is spelled ``DynamicLayer.update(self, ...)`` (this is a
# module-level function; single inheritance makes the two calls identical).
# ---------------------------------------------------------------------------


def _reference_update(self, key_states, value_states, *args, **kwargs):
    # Let DynamicLayer concat + return the full (post-RoPE) keys/values.
    keys, values = DynamicLayer.update(self, key_states, value_states, *args, **kwargs)

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
    if self._k_pre is not None and self.k_spec.pre_rope:
        prune_local_end = new_S_q - self._k_pre_offset
        if prune_local_end > 0 and prune_local_end <= self._k_pre.shape[1]:
            self._k_pre = self._k_pre[:, prune_local_end:, :].contiguous()
            self._k_pre_offset = new_S_q
        elif prune_local_end >= self._k_pre.shape[1]:
            self._k_pre = None
            self._k_pre_offset = new_S_q

    # --- fp16 tail [new_S_q:S] (pristine, from DynamicLayer) ---
    k_tail = keys.squeeze(0)[..., new_S_q:, :]
    v_tail = values.squeeze(0)[..., new_S_q:, :]

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

    self.keys = k_hat.unsqueeze(0)
    self.values = v_hat.unsqueeze(0)
    return self.keys, self.values


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_chunked(model, k_spec, v_spec, chunks, recent_window=8):
    """Feed `chunks` through a fresh StreamingQuantizedCache; return (cache, logits)."""
    cache = StreamingQuantizedCache(
        model.config, k_spec=k_spec, v_spec=v_spec, recent_window=recent_window
    )
    cache.attach(model)
    logits = []
    try:
        with torch.no_grad():
            for chunk in chunks:
                out = model(chunk, past_key_values=cache, use_cache=True)
                logits.append(out.logits.clone())
    finally:
        cache.detach()
    return cache, logits


# The bitwise license is proven by construction for elementwise/per-group arms,
# but the per-row NORM reductions (turboquant_*) and spectral's enc/dec matmuls
# are only pinned empirically — and reduction split config / BLAS kernel
# selection are shape-dependent on CUDA, so a CPU pass does not cover it. The
# cuda parametrization makes the GH200 suite run re-pin the license there.
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _assert_layer_state_equal(la, lb):
    """Bitwise equality of everything the flush writes: committed prefixes,
    reassembled slab, and the bpe accounting (exact float equality)."""
    assert la._committed_S_q == lb._committed_S_q
    assert torch.equal(la._q_prefix_k, lb._q_prefix_k), "committed K prefix differs"
    assert torch.equal(la._q_prefix_v, lb._q_prefix_v), "committed V prefix differs"
    assert torch.equal(la.keys, lb.keys), "reassembled keys differ"
    assert torch.equal(la.values, lb.values), "reassembled values differ"
    assert la.bpe_k == lb.bpe_k, f"bpe_k {la.bpe_k!r} != {lb.bpe_k!r}"
    assert la.bpe_v == lb.bpe_v, f"bpe_v {la.bpe_v!r} != {lb.bpe_v!r}"
    assert la._quant_bits_k == lb._quant_bits_k
    assert la._quant_bits_v == lb._quant_bits_v


def _flush_schedule_chunks(input_ids):
    """The A/B feeding schedule (recent_window=8, PAGE=128, 393 tokens total):
    280-token prefill (2 pages flush in ONE update), a 110-token chunk (no
    flush), then 1-token appends where S=392 flushes the 3rd page (a 1-token-
    append flush) and S=393 is a no-flush decode step. >= 3 pages total."""
    return [
        input_ids[:, :280],
        input_ids[:, 280:390],
        input_ids[:, 390:391],
        input_ids[:, 391:392],
        input_ids[:, 392:393],
    ]


# Representative set covering every codec route through the flush.
# k_batched/v_batched pin _flush_batchable's verdict per side: GEMM-free codecs
# batch; GEMM-bearing codecs (lowrank_*) keep the per-page loop; spectral is
# licensed whole-span (no packed twin) and tested separately with its pack.
ARM_CONFIGS = [
    pytest.param(  # passthrough sanity (pre_rope so the flush machinery runs)
        CacheCodecSpec(arm="fp16", pre_rope=True),
        CacheCodecSpec(arm="fp16"),
        True,
        True,
        id="fp16_prerope",
    ),
    pytest.param(  # rtn_channel groups run along S — the page-aligned-groups case
        CacheCodecSpec(arm="rtn_channel", bits=2, group=16, pre_rope=True),
        CacheCodecSpec(arm="rtn_token", bits=2, group=16),
        True,
        True,
        id="rtn_channel_k_prerope",
    ),
    pytest.param(
        CacheCodecSpec(arm="rtn_channel", bits=2, group=16),
        CacheCodecSpec(arm="rtn_channel", bits=2, group=16),
        True,
        True,
        id="rtn_channel_postrope_both",
    ),
    pytest.param(  # seeded Hadamard rotation (pow2 C -> fwht, row-independent)
        CacheCodecSpec(arm="rotate_rtn_token", bits=3, group=16),
        CacheCodecSpec(arm="rtn_token", bits=2, group=16),
        True,
        True,
        id="rotate_rtn_token_k",
    ),
    pytest.param(  # the b3 pair
        CacheCodecSpec(arm="turboquant_mse", bits=3),
        CacheCodecSpec(arm="turboquant_mse", bits=2),
        True,
        True,
        id="turboquant_b3",
    ),
    pytest.param(  # the k2b K path: frozen-subspace lowrank stays per-page
        CacheCodecSpec(
            arm="lowrank_rtn_channel", bits=3, rank=4, group=16, pre_rope=True
        ),
        CacheCodecSpec(arm="turboquant_mse", bits=2),
        False,
        True,
        id="k2b",
    ),
    pytest.param(  # the k2b_ph deployment pair (per-head Hadamard V)
        CacheCodecSpec(
            arm="lowrank_rtn_channel", bits=3, rank=4, group=16, pre_rope=True
        ),
        CacheCodecSpec(arm="turboquant_mse_perhead", bits=2),
        False,
        True,
        id="k2b_ph",
    ),
    pytest.param(  # k2t: turboquant residual on the frozen-subspace path
        CacheCodecSpec(arm="lowrank_turboquant", bits=3, rank=4, pre_rope=True),
        CacheCodecSpec(arm="rtn_token", bits=2, group=16),
        False,
        True,
        id="k2t",
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("k_spec,v_spec,k_batched,v_batched", ARM_CONFIGS)
def test_batched_flush_bitwise_matches_reference(
    k_spec, v_spec, k_batched, v_batched, device, monkeypatch
):
    """The gate: batched flush must be bitwise identical to the per-page loop —
    committed prefixes, reassembled keys/values, and exact bpe floats — across
    a multi-page flush, a multi-token no-flush step, and 1-token appends."""
    model = tiny_llama().to(device)
    input_ids = ids(vocab=97, seq=393, seed=29).to(device)
    chunks = _flush_schedule_chunks(input_ids)

    prod_cache, prod_logits = _run_chunked(model, k_spec, v_spec, chunks)
    # Non-vacuity: 3 pages committed; and the batching license engaged per side
    # exactly as designed (GEMM-free arms batch, GEMM-bearing arms loop).
    assert prod_cache.layers[0]._committed_S_q == 384
    assert prod_cache.layers[0]._k_flush_batchable is k_batched
    assert prod_cache.layers[0]._v_flush_batchable is v_batched

    with monkeypatch.context() as m:
        m.setattr(StreamingQuantizedLayer, "update", _reference_update)
        ref_cache, ref_logits = _run_chunked(model, k_spec, v_spec, chunks)

    for la, lb in zip(prod_cache.layers, ref_cache.layers, strict=True):
        _assert_layer_state_equal(la, lb)
    for pa, pb in zip(prod_logits, ref_logits, strict=True):
        assert torch.equal(pa, pb)


@pytest.mark.parametrize("device", DEVICES)
def test_batched_flush_bitwise_matches_reference_spectral(
    tmp_path, device, monkeypatch
):
    """Spectral K (pre_rope, tiny fitted pack) through the same A/B, BATCHED:
    spectral is pack-gated and never routes through PackedStreamingCache, so no
    packed-parity constraint binds it — it is licensed whole-span (matching the
    call granularity the offline G1 gauntlet measured). Its RTN group windows
    are page-aligned (PAGE % pack.group == 0) by construction; this test pins
    the remaining piece — enc/dec matmul row-batch invariance — bitwise against
    the per-page oracle, on every available device (the cuda run at the GH200
    gate is the one that actually covers BLAS kernel-selection variance)."""
    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path, budget=2.5, group=8)
    model = model.to(device)
    k_spec = CacheCodecSpec(
        arm="spectral", pre_rope=True, group=8, pack_path=path, budget=2.5
    )
    v_spec = CacheCodecSpec(arm="turboquant_mse", bits=2)

    input_ids = ids(vocab=97, seq=393, seed=29).to(device)
    chunks = _flush_schedule_chunks(input_ids)

    prod_cache, prod_logits = _run_chunked(model, k_spec, v_spec, chunks)
    assert prod_cache.layers[0]._committed_S_q == 384
    assert prod_cache.layers[0]._k_flush_batchable is True
    assert prod_cache.layers[0]._v_flush_batchable is True

    with monkeypatch.context() as m:
        m.setattr(StreamingQuantizedLayer, "update", _reference_update)
        ref_cache, ref_logits = _run_chunked(model, k_spec, v_spec, chunks)

    for la, lb in zip(prod_cache.layers, ref_cache.layers, strict=True):
        _assert_layer_state_equal(la, lb)
    for pa, pb in zip(prod_logits, ref_logits, strict=True):
        assert torch.equal(pa, pb)


def test_batched_flush_multi_update_page_crossing(monkeypatch):
    """Multi-update streaming: 1-token appends across a page boundary must equal
    the verbatim reference bitwise; and the chunked schedule must agree with a
    single-big-forward run where the schedule coincides (same committed count;
    values to the existing write-once bar — cross-chunking equality was never
    bitwise because projection GEMM shapes change with S, mirroring
    test_write_once_v_stable_token_by_token's rel < 0.05)."""
    model = tiny_llama()
    k_spec = CacheCodecSpec(
        arm="lowrank_rtn_channel", bits=3, rank=4, group=16, pre_rope=True
    )
    v_spec = CacheCodecSpec(arm="turboquant_mse", bits=2)
    input_ids = ids(vocab=97, seq=266, seed=3)
    # 260-token prefill commits page 1 (S_q=128); the append at S=264 commits
    # page 2 from a 1-token update; S=265/266 are no-flush steps.
    chunks = [input_ids[:, :260]] + [input_ids[:, t : t + 1] for t in range(260, 266)]

    prod_cache, _ = _run_chunked(model, k_spec, v_spec, chunks)
    assert prod_cache.layers[0]._committed_S_q == 256  # page 2 flushed mid-decode

    with monkeypatch.context() as m:
        m.setattr(StreamingQuantizedLayer, "update", _reference_update)
        ref_cache, _ = _run_chunked(model, k_spec, v_spec, chunks)
    for la, lb in zip(prod_cache.layers, ref_cache.layers, strict=True):
        _assert_layer_state_equal(la, lb)

    # Single big forward over the same tokens: the schedule lands on the same
    # committed count, and the committed V prefix matches to the write-once bar.
    big_cache, _ = _run_chunked(model, k_spec, v_spec, [input_ids])
    l0p, l0b = prod_cache.layers[0], big_cache.layers[0]
    assert l0b._committed_S_q == l0p._committed_S_q == 256
    rel = (
        l0p._q_prefix_v.float() - l0b._q_prefix_v.float()
    ).norm() / l0b._q_prefix_v.float().norm().clamp_min(1e-6)
    assert rel < 0.05, f"chunked vs big-forward committed V rel={rel:.3f}"


def test_batched_flush_one_codec_call_per_side_per_update(monkeypatch):
    """Requirement pin: a multi-page flush inside ONE update() makes exactly ONE
    batched codec call per side per layer (turboquant_mse both sides — the b3
    pair, fully batchable), each spanning the whole flushed region."""
    import bmx.cache.streaming as streaming_mod

    model = tiny_llama()
    calls: list[tuple[str, int]] = []
    real = streaming_mod.quantize_kv_layout

    def counting(kv_fp, spec):
        calls.append((spec.arm, kv_fp.shape[1]))  # (arm, S of this call)
        return real(kv_fp, spec)

    monkeypatch.setattr(streaming_mod, "quantize_kv_layout", counting)
    cache = StreamingQuantizedCache(
        model.config,
        k_spec=CacheCodecSpec(arm="turboquant_mse", bits=3),
        v_spec=CacheCodecSpec(arm="turboquant_mse", bits=2),
        recent_window=8,
    )
    cache.attach(model)
    with torch.no_grad():
        model(ids(vocab=97, seq=280, seed=31), past_key_values=cache, use_cache=True)
    cache.detach()

    n_layers = len(cache.layers)
    assert cache.layers[0]._committed_S_q == 256  # two pages flushed (non-vacuous)
    # ONE K call + ONE V call per layer for the whole 2-page flush (the per-page
    # loop would have made 2 sides x 2 pages = 4 per layer).
    assert len(calls) == 2 * n_layers, (
        f"expected {2 * n_layers} codec calls (one per side per layer), "
        f"got {len(calls)}: {calls}"
    )
    assert all(s == 256 for _, s in calls), f"call did not span the flush: {calls}"
