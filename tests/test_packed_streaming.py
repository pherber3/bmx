"""PackedStreamingCache: parity with StreamingQuantizedCache (bit-for-bit)."""

import pytest
import torch

from bmx.cache.codecs import dequant_packed, quantize_kv_layout, quantize_packed
from bmx.cache.collect import from_matrix, to_matrix
from bmx.cache.packed_streaming import PackedStreamingCache, PackedStreamingLayer
from bmx.cache.specs import CacheCodecSpec
from bmx.cache.streaming import StreamingQuantizedCache
from factories import ids, tiny_llama, tiny_llama_d32


def _k2b():
    return (
        CacheCodecSpec(
            arm="lowrank_rtn_channel", bits=3, rank=4, group=16, pre_rope=True
        ),
        CacheCodecSpec(arm="turboquant_mse", bits=2),
    )


def test_packed_generate_matches_streaming():
    model = tiny_llama()
    input_ids = ids(vocab=97, seq=12, seed=5)
    k_spec, v_spec = _k2b()

    ref_cache = StreamingQuantizedCache(model.config, k_spec=k_spec, v_spec=v_spec)
    ref_cache.attach(model)
    with torch.no_grad():
        ref = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,
            use_cache=True,
            past_key_values=ref_cache,
        )
    ref_cache.detach()

    packed = PackedStreamingCache(model.config, k_spec=k_spec, v_spec=v_spec)
    packed.attach(model)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,
            use_cache=True,
            past_key_values=packed,
        )
    packed.detach()

    assert torch.equal(out, ref)


def test_packed_generate_matches_streaming_long_prefill():
    # seq=200 > PAGE(128)+recent_window(32) so a 128-token page flushes during prefill,
    # exercising the committed-blocks causal path + slab prune the seq=12 test misses.
    model = tiny_llama()
    input_ids = ids(vocab=97, seq=200, seed=11)
    k_spec, v_spec = _k2b()
    ref_cache = StreamingQuantizedCache(model.config, k_spec=k_spec, v_spec=v_spec)
    ref_cache.attach(model)
    with torch.no_grad():
        ref = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,
            use_cache=True,
            past_key_values=ref_cache,
        )
    ref_cache.detach()
    packed = PackedStreamingCache(model.config, k_spec=k_spec, v_spec=v_spec)
    packed.attach(model)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,
            use_cache=True,
            past_key_values=packed,
        )
    # Verify resident slab is bounded (tail-only, not full S) — committed tokens live
    # only as packed pages. After PAGE flushing the slab is the un-committed tail,
    # bounded by recent_window + PAGE (the most that can accumulate before the next
    # page boundary).
    layer0 = packed.layers[0]
    total_seq = 200 + 20  # prefill + new tokens
    slab_len = layer0.keys.shape[2]
    assert slab_len < total_seq, (
        f"Slab not pruned: keys.shape[2]={slab_len} >= total_seq={total_seq}"
    )
    assert slab_len <= layer0.recent_window + layer0._page + 1, (
        f"Slab too large: {slab_len} > recent_window({layer0.recent_window})"
        f" + PAGE({layer0._page}) + 1"
    )
    packed.detach()
    assert torch.equal(out, ref)


def _last_logit_two_block_prefill(model, input_ids, n_prefill, k_spec, v_spec, Cls):
    """Run the cached TWO-block prefill ([0:n_prefill] then [n_prefill:L]) through a
    cache and return the last-position logit (what seeds decoding)."""
    cache = Cls(model.config, k_spec=k_spec, v_spec=v_spec)
    cache.attach(model)
    with cache, torch.no_grad():
        model(
            input_ids[:, :n_prefill],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        out = model(
            input_ids[:, n_prefill:],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
    cache.detach()
    return out.logits[0, -1].float()


def test_packed_two_block_prefill_logits_match_streaming():
    """The cached two-block prefill path — the regime plain model.generate does NOT
    exercise, and where the attention-mask bug lived.

    generate_through_cache (and any cached prefill) runs TWO forwards: [0:n_prefill]
    then [n_prefill:L]. The second has n_q < n_kv with a nonzero query offset — the
    cached-prefill case where is_causal=True (bottom-right) is NOT the model's mask.
    If the custom attention impl doesn't get the real mask (no sdpa_mask registered,
    or mask not threaded into the prefill SDPA), the prefill logits diverge.

    This asserts the LAST-position logit (which seeds decoding) matches dense.
    Token-equality is too weak — at tiny scale the bug shifts logits by ~0.02 without
    flipping the argmax; the divergence only flips tokens at real-model magnitudes.
    The numerical check catches it at tiny scale: with the fix the logits are
    bit-identical (diff 0.0); with the bug the diff is ~0.02.
    """
    model = tiny_llama()
    input_ids = ids(vocab=97, seq=60, seed=9)  # > recent_window so prefill flushes
    k_spec, v_spec = _k2b()
    n_prefill = 16
    dense = _last_logit_two_block_prefill(
        model, input_ids, n_prefill, k_spec, v_spec, StreamingQuantizedCache
    )
    packed = _last_logit_two_block_prefill(
        model, input_ids, n_prefill, k_spec, v_spec, PackedStreamingCache
    )
    max_abs = (dense - packed).abs().max().item()
    assert max_abs < 1e-3, (
        f"two-block prefill logits diverged: max_abs={max_abs} "
        "(packed prefill not using the model's causal mask — see sdpa_mask "
        "registration in packed_streaming.py)"
    )


def _rtn_specs():
    """Plain RTN K and V (post-RoPE K) — the fused-packed kernel's supported arm."""
    return (
        CacheCodecSpec(arm="rtn_token", bits=4, group=8, pre_rope=False),
        CacheCodecSpec(arm="rtn_token", bits=4, group=8),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused-packed decode path is taken only when q.is_cuda + Triton present",
)
def test_fused_packed_generate_matches_streaming_cuda():
    """The DEPLOYMENT path: PackedStreamingCache decode routes through the fused
    split-KV kernel that dequants int8 RTN codes IN-KERNEL (no dense copy). On CUDA
    with rtn_token + post-RoPE K, attend() takes fused_decode_attention_packed.

    Compares greedy generate vs StreamingQuantizedCache (the reference). The fused
    kernel's tl.dot uses tf32 tensor cores -> ~1e-3 logit drift, so token-equality
    can occasionally differ; assert decode-logit closeness instead (the meaningful
    quality gate). seq > recent_window so blocks flush and the committed-packed +
    fp16-tail merge path is exercised.
    """
    model = tiny_llama().cuda()
    input_ids = ids(vocab=97, seq=60, seed=11).cuda()
    k_spec, v_spec = _rtn_specs()

    def _decode_logits(Cls):
        cache = Cls(model.config, k_spec=k_spec, v_spec=v_spec)
        cache.attach(model)
        with cache, torch.no_grad():
            model(input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            # one decode step — this is where the fused-packed path runs (n_q==1)
            step = ids(vocab=97, seq=1, seed=12).cuda()
            out = model(step, past_key_values=cache, use_cache=True)
        cache.detach()
        return out.logits[0, -1].float()

    ref = _decode_logits(StreamingQuantizedCache)
    fused = _decode_logits(PackedStreamingCache)
    max_abs = (ref - fused).abs().max().item()
    assert max_abs < 5e-2, (
        f"fused-packed decode logits diverged from streaming: max_abs={max_abs} "
        "(fused_decode_attention_packed in packed_streaming.attend)"
    )


def test_perhead_v_codec_bit_parity_streaming_vs_packed():
    """turboquant_mse_perhead V: reference path (quantize_kv_layout) and packed path
    (quantize_packed + dequant_packed + from_matrix) must produce bit-identical
    dequantized tensors and equal bpe after CHANGE 1 threads h_heads through
    quantize_cache/quantize_kv_layout.

    Before the fix, quantize_kv_layout defaulted h_heads=0 → h=1 (full-C Hadamard),
    while _pack_v_block passed h_heads=h_kv (per-head Hadamard) — a silent divergence.
    This test is the exact-parity gate that the 5e-2 logit test could not provide.
    """
    torch.manual_seed(42)
    h_kv, S, d = 2, 128, 32
    v_spec = CacheCodecSpec(arm="turboquant_mse_perhead", bits=2)

    # V tensor: (h_kv, S, d) in fp16 (cache storage dtype), cast to fp32 for codecs
    V_fp16 = torch.randn(h_kv, S, d, dtype=torch.float16)
    V_fp32 = V_fp16.float()

    # --- Reference path (streaming.py / ppl_eval.py): quantize_kv_layout ---
    V_ref_hat, ref_bpe = quantize_kv_layout(V_fp32, v_spec)

    # --- Packed path (_pack_v_block in packed_streaming.py) ---
    M = to_matrix(V_fp32)  # (S, h_kv*d)
    packed, pack_bpe = quantize_packed(
        "turboquant_mse_perhead",
        M,
        bits=v_spec.bits,
        seed=v_spec.seed,
        h_heads=h_kv,
    )
    M_hat = dequant_packed("turboquant_mse_perhead", packed, seed=v_spec.seed)
    V_pack_hat = from_matrix(M_hat, h_kv)

    assert torch.equal(V_pack_hat, V_ref_hat), (
        f"turboquant_mse_perhead reference vs packed paths diverged: "
        f"max_abs={(V_pack_hat - V_ref_hat).abs().max():.2e}. "
        "h_heads is not being threaded through quantize_cache → quantize_kv_layout."
    )
    assert ref_bpe == pack_bpe, f"bpe mismatch: reference={ref_bpe}, packed={pack_bpe}"


def _k2b_perhead():
    """The REAL recipe with the per-head Hadamard V (the fused-k2b kernel's arm).

    rank=16 and the d32 model (head_dim=32) so the fused-k2b dims gate (d>=16,
    rank>=16, d pow2) passes and attend() actually takes fused_decode_attention_k2b.
    """
    return (
        CacheCodecSpec(
            arm="lowrank_rtn_channel", bits=3, rank=16, group=16, pre_rope=True
        ),
        CacheCodecSpec(arm="turboquant_mse_perhead", bits=2),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused-k2b decode path is taken only when q.is_cuda + Triton present",
)
@pytest.mark.parametrize("pack_v", [False, True], ids=["v_idx", "v_idx_packed"])
def test_fused_k2b_generate_matches_streaming_cuda(pack_v):
    """The REAL-recipe deployment path: PackedStreamingCache decode routes through
    the fused k2b kernel (in-kernel lowrank-K + RoPE + per-head turboquant-V with an
    in-kernel d-point Hadamard unrotate). On CUDA with lowrank_rtn_channel K +
    turboquant_mse_perhead V, attend() takes fused_decode_attention_k2b.

    Compares decode-logit closeness vs StreamingQuantizedCache (same per-head codec,
    chunked path). seq > recent_window so blocks flush and the committed-packed +
    fp16-tail merge is exercised. tf32 tensor cores -> ~1e-3 logit drift.

    pack_v parametrization (W5-2, GH200 acceptance prep): pack_v=True exercises
    the full flag-gated path end-to-end (build_kv_stacked_k2b pack_v=True ->
    _repoint_k2b_blocks indices_packed re-point -> V_PACKED in-kernel unpack).
    """
    model = tiny_llama_d32().cuda()  # head_dim=32 so the fused-k2b dims gate passes
    input_ids = ids(vocab=97, seq=60, seed=13).cuda()
    k_spec, v_spec = _k2b_perhead()

    def _decode_logits(Cls, **cache_kwargs):
        cache = Cls(model.config, k_spec=k_spec, v_spec=v_spec, **cache_kwargs)
        cache.attach(model)
        with cache, torch.no_grad():
            model(input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            step = ids(vocab=97, seq=1, seed=14).cuda()
            out = model(step, past_key_values=cache, use_cache=True)
        cache.detach()
        return out.logits[0, -1].float()

    ref = _decode_logits(StreamingQuantizedCache)
    fused = _decode_logits(PackedStreamingCache, pack_v=pack_v)
    max_abs = (ref - fused).abs().max().item()
    # Codec is now identical on both sides (both use per-head Hadamard after CHANGE 1).
    # Residual diff is fused-kernel tf32 tensor-core math vs chunked fp32 reference,
    # so atol=0 exact parity is not expected here; 5e-2 covers the tf32 drift.
    assert max_abs < 5e-2, (
        f"fused-k2b decode logits diverged from streaming (pack_v={pack_v}): "
        f"max_abs={max_abs} (fused_decode_attention_k2b in packed_streaming.attend)"
    )


def test_uniform_blk_incremental_matches_recomputation():
    """desk review F3: PackedStreamingLayer maintains self._blk_len/_uniform_blk
    incrementally at the append site in update() instead of recomputing
    `len({e - s for _, s, e in self._k_blocks}) == 1` every attend() call. This
    pins the invariant: after EVERY update() call (i.e. every model forward step,
    whether or not it flushes a page), the incremental fields must equal what the
    old set-comprehension would compute from scratch.

    seq=300 (> 2 * PAGE(128) + recent_window) so multiple pages flush across the
    prefill + a run of decode steps, exercising the append path repeatedly.
    """
    model = tiny_llama()
    input_ids = ids(vocab=97, seq=300, seed=21)
    k_spec, v_spec = _rtn_specs()  # post-RoPE plain RTN — uniform PAGE blocks

    cache = PackedStreamingCache(model.config, k_spec=k_spec, v_spec=v_spec)
    cache.attach(model)

    calls = []
    orig_update = PackedStreamingLayer.update

    def _checked_update(self, *args, **kwargs):
        result = orig_update(self, *args, **kwargs)
        blocks = self._k_blocks
        expect_uniform = bool(blocks) and len({e - s for _, s, e in blocks}) == 1
        expect_len = (blocks[-1][2] - blocks[-1][1]) if blocks else None
        calls.append((self._uniform_blk, expect_uniform, self._blk_len, expect_len))
        assert self._uniform_blk == expect_uniform, (
            f"_uniform_blk={self._uniform_blk} != recomputed {expect_uniform} "
            f"after update() with {len(blocks)} blocks"
        )
        assert self._blk_len == expect_len, (
            f"_blk_len={self._blk_len} != recomputed {expect_len} "
            f"after update() with {len(blocks)} blocks"
        )
        return result

    PackedStreamingLayer.update = _checked_update
    try:
        with torch.no_grad():
            model.generate(
                input_ids,
                max_new_tokens=25,
                do_sample=False,
                use_cache=True,
                past_key_values=cache,
            )
    finally:
        PackedStreamingLayer.update = orig_update
        cache.detach()

    # Sanity: the invariant was actually exercised across multiple flushed pages
    # (>= 2 committed blocks by the end), not vacuously true on an empty list.
    assert len(calls) > 0
    assert cache.layers[0]._blk_len is not None
    assert len(cache.layers[0]._k_blocks) >= 2


# ---------------------------------------------------------------------------
# W5-1: single-storage pages — block dicts become views into _PagedStacks.
#
# _PagedStacks.view() is device-agnostic and directly callable on CPU (it has no
# CUDA dependency — only the fused Triton kernels it feeds require CUDA), so these
# tests drive the k2b stacking + re-point path directly instead of going through
# attend()'s CUDA-gated fused dispatch (q.is_cuda is always False on this box).
# ---------------------------------------------------------------------------


def _k2b_cpu_blocks(n_blocks, h_kv=2, blk=32, d=16, rank=16, group=16, seed=0):
    """n_blocks of (lowrank_rtn_channel K, turboquant_mse_perhead V) packed dicts,
    the k2b_ph fused-kernel arm pair, built directly via quantize_packed (no RoPE —
    this exercises the stacking/re-point plumbing, not RoPE correctness, which is
    already covered elsewhere)."""
    torch.manual_seed(seed)
    k_blocks, v_blocks = [], []
    for i in range(n_blocks):
        s, e = i * blk, (i + 1) * blk
        kM = to_matrix(torch.randn(h_kv, blk, d))
        vM = to_matrix(torch.randn(h_kv, blk, d))
        kp, _ = quantize_packed(
            "lowrank_rtn_channel", kM, bits=3, group=group, rank=rank, seed=seed
        )
        vp, _ = quantize_packed(
            "turboquant_mse_perhead", vM, bits=2, seed=seed, h_heads=h_kv
        )
        k_blocks.append((kp, s, e))
        v_blocks.append((vp, s, e))
    return k_blocks, v_blocks


def _chunked_decode(q, k_blocks, v_blocks, h_kv, group, seed, d):
    """Minimal decode-mode chunked_dequant_attention call for the k2b arm pair."""
    from bmx.cache.chunked_attention import chunked_dequant_attention

    n_q_heads = q.shape[0]
    k_tail = torch.zeros(h_kv, 0, d)
    v_tail = torch.zeros(h_kv, 0, d)
    return chunked_dequant_attention(
        q,
        k_blocks,
        v_blocks,
        k_arm="lowrank_rtn_channel",
        v_arm="turboquant_mse_perhead",
        group=group,
        seed=seed,
        k_pre_rope=False,
        rope_cos=None,
        rope_sin=None,
        k_tail=k_tail,
        v_tail=v_tail,
        n_q_groups=n_q_heads // h_kv,
        scale=1.0 / (d**0.5),
        is_prefill=False,
        v_group=None,
        v_seed=seed,
    )


def test_k2b_repoint_storage_identity():
    """After _repoint_k2b_blocks, the two re-pointable fields (K's res_Q_int, V's
    indices) share storage with the stack buffer — no duplicate allocation survives
    for those fields. The four cast fields (Us/V/res_scale/norms) are intentionally
    NOT re-pointed (fp32->fp16 cast in the builder — a view cannot span two dtypes),
    so they must still be the ORIGINAL block tensors, not stack-buffer views.
    """
    from bmx.cache.packed_streaming import PackedStreamingLayer, _PagedStacks
    from bmx.cache.triton_dequant_attention import build_kv_stacked_k2b

    h_kv, blk, d, rank, group, n = 2, 32, 16, 16, 16, 3
    k_blocks, v_blocks = _k2b_cpu_blocks(n, h_kv, blk, d, rank, group, seed=1)
    orig_res_Q_int = [kp["res_Q_int"] for kp, _, _ in k_blocks]
    orig_indices = [vp["indices"] for vp, _, _ in v_blocks]
    orig_Us = [kp["Us"] for kp, _, _ in k_blocks]

    layer = PackedStreamingLayer.__new__(PackedStreamingLayer)
    layer._k_blocks = k_blocks
    layer._v_blocks = v_blocks
    layer._k2b_repointed = 0
    layer._k2b_repoint_version = -1

    stacks = _PagedStacks(build_kv_stacked_k2b, dict(h_kv=h_kv, blk_size=blk, d=d))
    stacks.view(k_blocks, v_blocks, torch.device("cpu"))
    layer._repoint_k2b_blocks(stacks)

    res_int_buf = stacks._buf["res_int"]
    v_idx_buf = stacks._buf["v_idx"]
    for i in range(n):
        kp, _, _ = k_blocks[i]
        vp, _, _ = v_blocks[i]
        assert kp["res_Q_int"].untyped_storage().data_ptr() == (
            res_int_buf.untyped_storage().data_ptr()
        ), f"block {i}: res_Q_int not re-pointed into the stack buffer"
        assert vp["indices"].untyped_storage().data_ptr() == (
            v_idx_buf.untyped_storage().data_ptr()
        ), f"block {i}: indices not re-pointed into the stack buffer"
        assert torch.equal(kp["res_Q_int"], orig_res_Q_int[i]), (
            "re-pointed res_Q_int value changed"
        )
        assert torch.equal(vp["indices"], orig_indices[i]), (
            "re-pointed indices value changed"
        )
        # Cast fields (Us/V/res_scale/norms) stay the ORIGINAL block tensors.
        assert kp["Us"] is orig_Us[i], (
            "Us was re-pointed despite the builder casting it fp32->fp16"
        )
        assert kp["Us"].untyped_storage().data_ptr() != (
            stacks._buf["us"].untyped_storage().data_ptr()
        )


def test_k2b_repoint_parity_before_after_and_after_grow():
    """chunked_dequant_attention output over the k2b blocks must be torch.equal:
    (a) before any re-point, (b) after re-pointing the first stacking, and (c)
    after a forced _grow (which reallocates the buffer, invalidating the earlier
    views) followed by a re-point refresh. This is the referee for W5-1: the
    consumer (chunked_dequant_attention) must see IDENTICAL values whether it
    reads the original block tensors or the re-pointed stack-buffer views.
    """
    from bmx.cache.packed_streaming import PackedStreamingLayer, _PagedStacks
    from bmx.cache.triton_dequant_attention import build_kv_stacked_k2b

    h_kv, blk, d, rank, group, n = 2, 32, 16, 16, 16, 2
    n_q_heads = h_kv * 2  # n_q_groups = 2
    k_blocks, v_blocks = _k2b_cpu_blocks(n, h_kv, blk, d, rank, group, seed=2)
    q = torch.randn(n_q_heads, 1, d)

    out_before = _chunked_decode(q, k_blocks, v_blocks, h_kv, group, 2, d)

    layer = PackedStreamingLayer.__new__(PackedStreamingLayer)
    layer._k_blocks = k_blocks
    layer._v_blocks = v_blocks
    layer._k2b_repointed = 0
    layer._k2b_repoint_version = -1

    stacks = _PagedStacks(build_kv_stacked_k2b, dict(h_kv=h_kv, blk_size=blk, d=d))
    stacks.view(k_blocks[:1], v_blocks[:1], torch.device("cpu"))
    layer._repoint_k2b_blocks(stacks)
    out_after_repoint = _chunked_decode(q, k_blocks, v_blocks, h_kv, group, 2, d)
    assert torch.equal(out_before, out_after_repoint), (
        "chunked output changed after re-pointing block 0's res_Q_int/indices"
    )

    # Force a _grow: append the remaining blocks with a tiny starting capacity so
    # `view` must grow past the block-0-only allocation.
    version_before_grow = stacks.version
    stacks.view(k_blocks, v_blocks, torch.device("cpu"))
    assert stacks.version > version_before_grow, (
        "test setup did not actually exercise a _grow — cap/n_stacked did not "
        "cross the capacity boundary"
    )
    layer._repoint_k2b_blocks(stacks)  # must refresh stale views after the grow
    out_after_grow = _chunked_decode(q, k_blocks, v_blocks, h_kv, group, 2, d)
    assert torch.equal(out_before, out_after_grow), (
        "chunked output changed after a _grow + re-point refresh — stale view "
        "(pointing at a freed pre-grow buffer) or refresh bug"
    )

    # And the post-grow views must actually point at the NEW buffer, not the old one.
    res_int_buf = stacks._buf["res_int"]
    for kp, _, _ in k_blocks:
        assert kp["res_Q_int"].untyped_storage().data_ptr() == (
            res_int_buf.untyped_storage().data_ptr()
        ), "block still points at a pre-grow (stale) buffer"


def test_k2b_repoint_restack_on_shrink_is_forbidden():
    """The `n < self._n_stacked` restack branch would rebuild pages FROM the block
    list while freeing the buffer those very blocks may be re-pointed into — an
    unsound rebuild-from-views-of-the-buffer-being-replaced hazard. It is
    unreachable through PackedStreamingLayer (pages are append-only — see the F3
    invariant note), so _PagedStacks.view() now asserts instead of silently
    producing a wrong restack. This test pins that it fails loudly rather than
    reachably corrupting state, should some future caller violate the invariant.
    """
    from bmx.cache.packed_streaming import _PagedStacks
    from bmx.cache.triton_dequant_attention import build_kv_stacked_k2b

    h_kv, blk, d, rank, group, n = 2, 32, 16, 16, 16, 3
    k_blocks, v_blocks = _k2b_cpu_blocks(n, h_kv, blk, d, rank, group, seed=3)

    stacks = _PagedStacks(build_kv_stacked_k2b, dict(h_kv=h_kv, blk_size=blk, d=d))
    stacks.view(k_blocks, v_blocks, torch.device("cpu"))
    with pytest.raises(AssertionError, match="restack-on-shrink"):
        stacks.view(k_blocks[:1], v_blocks[:1], torch.device("cpu"))


def test_rtn_token_blocks_never_repointed():
    """rtn_token/rtn_token has no re-pointable field: build_kv_stacked_packed's
    from_matrix head-split (S, h_kv*d) <-> (h_kv, S, d) permute+reshape is not a
    free view for h_kv > 1 (reshape after that permute silently falls back to a
    copy — verified: the resulting tensor's storage differs from the stack
    buffer's). PackedStreamingLayer therefore never re-points that config; this
    test pins Q_int/scale as untouched, original, non-buffer-aliased tensors after
    building the packed stack, documenting the config is (and must remain) fully
    duplicated rather than shipping a copy disguised as a view.
    """
    from bmx.cache.triton_dequant_attention import build_kv_stacked_packed

    h_kv, blk, d, group, n = 2, 16, 8, 8, 2
    torch.manual_seed(4)
    k_blocks, v_blocks = [], []
    for i in range(n):
        s, e = i * blk, (i + 1) * blk
        kM = to_matrix(torch.randn(h_kv, blk, d))
        kp, _ = quantize_packed("rtn_token", kM, bits=4, group=group, seed=4)
        k_blocks.append((kp, s, e))
        v_blocks.append((kp, s, e))
    orig_Q_int = [kp["Q_int"] for kp, _, _ in k_blocks]

    k_codes, v_codes, k_scales, v_scales = build_kv_stacked_packed(
        k_blocks,
        v_blocks,
        max_blocks=n,
        h_kv=h_kv,
        blk_size=blk,
        d=d,
        group=group,
        v_group=group,
        device="cpu",
    )

    for i in range(n):
        kp, _, _ = k_blocks[i]
        assert kp["Q_int"] is orig_Q_int[i], (
            "rtn_token block Q_int must stay the original tensor (no re-point "
            "exists for this config — see _repoint_k2b_blocks docstring)"
        )
        # Confirm (rather than assume) the layout genuinely can't view-alias: the
        # would-be inverse (permute back to (S, h_kv*d)) does not share storage
        # with the stack buffer.
        back = k_codes[i].permute(1, 0, 2).reshape(blk, h_kv * d)
        assert back.untyped_storage().data_ptr() != (
            k_codes.untyped_storage().data_ptr()
        ), (
            "unexpected: from_matrix's permuted layout turned out to be reshape-"
            "contiguous here — re-examine whether rtn_token IS re-pointable "
            "(this would contradict the documented h_kv>1 analysis)"
        )


# ---------------------------------------------------------------------------
# W5-2: flag-gated 2-bit packing for stacked V indices (4 codes/byte).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [2, 4])
def test_pack_unpack_codes_roundtrip(bits):
    """pack_codes/unpack_codes must be an exact inverse pair over random codes."""
    from bmx.cache.triton_dequant_attention import pack_codes, unpack_codes

    torch.manual_seed(0)
    per_byte = 8 // bits
    shape = (3, 5, per_byte * 8)  # C divisible by per_byte
    codes = torch.randint(0, 2**bits, shape, dtype=torch.int16)

    packed = pack_codes(codes, bits)
    assert packed.dtype == torch.uint8
    assert packed.shape == (*shape[:-1], shape[-1] // per_byte)

    unpacked = unpack_codes(packed, bits, shape[-1])
    assert unpacked.dtype == torch.int16
    assert torch.equal(unpacked, codes)


def test_pack_codes_rejects_odd_shapes_and_bits():
    """Loud assertion on unsupported bit-widths / non-divisible C."""
    from bmx.cache.triton_dequant_attention import pack_codes, unpack_codes

    codes = torch.randint(0, 8, (2, 8), dtype=torch.int16)
    with pytest.raises(AssertionError):
        pack_codes(codes, bits=3)  # 8 % 3 != 0
    with pytest.raises(AssertionError):
        pack_codes(codes, bits=8)  # not < 8
    codes_bad_c = torch.randint(0, 4, (2, 6), dtype=torch.int16)  # per_byte=4, 6%4!=0
    with pytest.raises(AssertionError):
        pack_codes(codes_bad_c, bits=2)

    packed = pack_codes(torch.randint(0, 4, (2, 8), dtype=torch.int16), bits=2)
    with pytest.raises(AssertionError):
        unpack_codes(packed, bits=3, C=8)
    with pytest.raises(AssertionError):
        unpack_codes(packed, bits=2, C=6)  # C not divisible by per_byte


def test_build_kv_stacked_k2b_pack_v_false_matches_today():
    """Regression pin: pack_v=False (the default) must be byte-identical to the
    pre-W5-2 builder output — same field names/dtypes/values."""
    from bmx.cache.triton_dequant_attention import build_kv_stacked_k2b

    h_kv, blk, d, rank, group, n = 2, 32, 16, 16, 16, 2
    k_blocks, v_blocks = _k2b_cpu_blocks(n, h_kv, blk, d, rank, group, seed=10)

    built = build_kv_stacked_k2b(
        k_blocks, v_blocks, max_blocks=n, h_kv=h_kv, blk_size=blk, d=d, device="cpu"
    )
    assert set(built.keys()) == {
        "us",
        "vfac",
        "res_int",
        "res_scale",
        "v_idx",
        "v_norm",
        "rank",
        "k_group",
    }
    assert built["v_idx"].dtype == torch.int16
    assert built["v_idx"].shape == (n, blk, h_kv * d)
    assert "v_idx_packed" not in built
    assert "pack_v" not in built


def test_build_kv_stacked_k2b_pack_v_true_unpacks_to_same_indices():
    """pack_v=True's v_idx_packed, unpacked, must equal the pack_v=False v_idx."""
    from bmx.cache.triton_dequant_attention import build_kv_stacked_k2b, unpack_codes

    h_kv, blk, d, rank, group, n = 2, 32, 16, 16, 16, 2
    k_blocks, v_blocks = _k2b_cpu_blocks(n, h_kv, blk, d, rank, group, seed=11)
    C = h_kv * d
    vbits = v_blocks[0][0]["bits"]

    built_unpacked = build_kv_stacked_k2b(
        k_blocks, v_blocks, max_blocks=n, h_kv=h_kv, blk_size=blk, d=d, device="cpu"
    )
    built_packed = build_kv_stacked_k2b(
        k_blocks,
        v_blocks,
        max_blocks=n,
        h_kv=h_kv,
        blk_size=blk,
        d=d,
        device="cpu",
        pack_v=True,
    )

    assert built_packed["pack_v"] is True
    assert built_packed["vbits"] == vbits
    assert built_packed["v_idx_packed"].dtype == torch.uint8
    per_byte = 8 // vbits
    assert built_packed["v_idx_packed"].shape == (n, blk, C // per_byte)
    assert "v_idx" not in built_packed

    recovered = unpack_codes(built_packed["v_idx_packed"], vbits, C)
    assert torch.equal(recovered, built_unpacked["v_idx"])

    # Every other field is untouched by pack_v.
    for key in ("us", "vfac", "res_int", "res_scale", "v_norm", "rank", "k_group"):
        a, b = built_unpacked[key], built_packed[key]
        if torch.is_tensor(a):
            assert torch.equal(a, b), f"field {key} differs under pack_v"
        else:
            assert a == b, f"field {key} differs under pack_v"


def test_k2b_repoint_pack_v_true_replaces_indices_with_packed_view():
    """Under pack_v=True, _repoint_k2b_blocks must DELETE the block dict's
    "indices" key and replace it with "indices_packed" (a view into the packed
    uint8 stack buffer) — NOT re-point "indices" itself to a wrong-shape/dtype
    tensor. res_Q_int re-pointing is unaffected by pack_v.
    """
    from bmx.cache.packed_streaming import PackedStreamingLayer, _PagedStacks
    from bmx.cache.triton_dequant_attention import build_kv_stacked_k2b, unpack_codes

    h_kv, blk, d, rank, group, n = 2, 32, 16, 16, 16, 2
    k_blocks, v_blocks = _k2b_cpu_blocks(n, h_kv, blk, d, rank, group, seed=12)
    orig_indices = [vp["indices"] for vp, _, _ in v_blocks]
    vbits = v_blocks[0][0]["bits"]
    C = h_kv * d

    layer = PackedStreamingLayer.__new__(PackedStreamingLayer)
    layer._k_blocks = k_blocks
    layer._v_blocks = v_blocks
    layer._k2b_repointed = 0
    layer._k2b_repoint_version = -1

    stacks = _PagedStacks(
        build_kv_stacked_k2b, dict(h_kv=h_kv, blk_size=blk, d=d, pack_v=True)
    )
    stacks.view(k_blocks, v_blocks, torch.device("cpu"))
    layer._repoint_k2b_blocks(stacks)

    v_idx_packed_buf = stacks._buf["v_idx_packed"]
    for i in range(n):
        vp, _, _ = v_blocks[i]
        assert "indices" not in vp, "indices must be deleted under pack_v=True"
        assert "indices_packed" in vp
        assert vp["indices_packed"].untyped_storage().data_ptr() == (
            v_idx_packed_buf.untyped_storage().data_ptr()
        )
        recovered = unpack_codes(vp["indices_packed"], vbits, C)
        assert torch.equal(recovered, orig_indices[i])

    # res_Q_int re-pointing is unaffected by pack_v.
    res_int_buf = stacks._buf["res_int"]
    for kp, _, _ in k_blocks:
        assert kp["res_Q_int"].untyped_storage().data_ptr() == (
            res_int_buf.untyped_storage().data_ptr()
        )


def test_block_v_indices_helper_transparent_to_pack_v():
    """block_v_indices returns the same int16 indices whether the block dict
    holds "indices" (pack_v=False) or "indices_packed" (pack_v=True, re-pointed)."""
    from bmx.cache.triton_dequant_attention import block_v_indices

    torch.manual_seed(13)
    h_kv, blk, d = 2, 16, 8
    C = h_kv * d
    vbits = 2
    from bmx.cache.codecs import quantize_packed
    from bmx.cache.collect import to_matrix

    vM = to_matrix(torch.randn(h_kv, blk, d))
    vp, _ = quantize_packed(
        "turboquant_mse_perhead", vM, bits=vbits, seed=0, h_heads=h_kv
    )
    direct = block_v_indices(vp, vbits, C)
    assert torch.equal(direct, vp["indices"])

    from bmx.cache.triton_dequant_attention import pack_codes

    vp_packed = {k: v for k, v in vp.items() if k != "indices"}
    vp_packed["indices_packed"] = pack_codes(vp["indices"], vbits)
    via_unpack = block_v_indices(vp_packed, vbits, C)
    assert torch.equal(via_unpack, vp["indices"])


def test_chunked_attention_after_pack_v_repoint_matches_pristine():
    """The chunked-attention fallback path (CPU-reachable; used for prefill and
    any non-fused decode) must produce IDENTICAL output whether it reads V blocks
    that (a) still hold pristine int16 "indices", or (b) have been re-pointed
    under pack_v=True and now hold "indices_packed" only. This is the referee
    for the W5-1/W5-2 interaction: a post-stacking chunked read must still work.
    """
    from bmx.cache.packed_streaming import PackedStreamingLayer, _PagedStacks
    from bmx.cache.triton_dequant_attention import build_kv_stacked_k2b

    h_kv, blk, d, rank, group, n = 2, 32, 16, 16, 16, 2
    n_q_heads = h_kv * 2  # n_q_groups = 2
    k_blocks, v_blocks = _k2b_cpu_blocks(n, h_kv, blk, d, rank, group, seed=14)
    q = torch.randn(n_q_heads, 1, d)

    out_pristine = _chunked_decode(q, k_blocks, v_blocks, h_kv, group, 14, d)

    layer = PackedStreamingLayer.__new__(PackedStreamingLayer)
    layer._k_blocks = k_blocks
    layer._v_blocks = v_blocks
    layer._k2b_repointed = 0
    layer._k2b_repoint_version = -1

    stacks = _PagedStacks(
        build_kv_stacked_k2b, dict(h_kv=h_kv, blk_size=blk, d=d, pack_v=True)
    )
    stacks.view(k_blocks, v_blocks, torch.device("cpu"))
    layer._repoint_k2b_blocks(stacks)

    # Sanity: the re-point actually happened (indices deleted, packed present).
    for vp, _, _ in v_blocks:
        assert "indices" not in vp
        assert "indices_packed" in vp

    out_after_pack_v_repoint = _chunked_decode(
        q, k_blocks, v_blocks, h_kv, group, 14, d
    )
    assert torch.equal(out_pristine, out_after_pack_v_repoint), (
        "chunked_dequant_attention output changed after pack_v=True re-pointing "
        "— block_v_indices access-path wiring in chunked_attention._dequant_block "
        "is not transparent to pack_v"
    )


def test_rope_tables_shared_across_layers_and_caches():
    """2026-07-06 memory fix: RoPE cos/sin are process-shared, not per-layer copies.

    The memory-ledger probe found ~0.5 GiB/cache of identical per-layer tables at
    32k. After _shared_rope, every layer of every cache on the same config must hold
    THE SAME storage, and values must equal a direct rope_cos_sin computation.
    """
    import torch

    from bmx.cache.rope import rope_cos_sin

    model = tiny_llama()
    k_spec, v_spec = _k2b()
    input_ids = ids(vocab=97, seq=200, seed=3)  # >PAGE+window so pages flush

    caches = []
    for _ in range(2):
        cache = PackedStreamingCache(model.config, k_spec=k_spec, v_spec=v_spec)
        cache.attach(model)
        with torch.no_grad():
            model.generate(
                input_ids,
                max_new_tokens=4,
                do_sample=False,
                use_cache=True,
                past_key_values=cache,
            )
        cache.detach()
        caches.append(cache)

    ptrs = set()
    for cache in caches:
        for layer in cache.layers:
            if layer._rope_cos is None:
                continue
            ptrs.add(layer._rope_cos.untyped_storage().data_ptr())
    assert len(ptrs) == 1, f"expected ONE shared rope storage, got {len(ptrs)}"

    any_layer = next(
        layer for c in caches for layer in c.layers if layer._rope_cos is not None
    )
    n = any_layer._rope_cos.shape[0]
    ref_cos, ref_sin = rope_cos_sin(model.config, n, start=0, device="cpu")
    assert torch.equal(any_layer._rope_cos, ref_cos.to(torch.float16))
    assert torch.equal(any_layer._rope_sin, ref_sin.to(torch.float16))
