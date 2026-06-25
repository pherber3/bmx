"""PackedStreamingCache: parity with StreamingQuantizedCache (bit-for-bit)."""

import pytest
import torch

from bmx.cache.codecs import dequant_packed, quantize_kv_layout, quantize_packed
from bmx.cache.collect import from_matrix, to_matrix
from bmx.cache.packed_streaming import PackedStreamingCache
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
def test_fused_k2b_generate_matches_streaming_cuda():
    """The REAL-recipe deployment path: PackedStreamingCache decode routes through
    the fused k2b kernel (in-kernel lowrank-K + RoPE + per-head turboquant-V with an
    in-kernel d-point Hadamard unrotate). On CUDA with lowrank_rtn_channel K +
    turboquant_mse_perhead V, attend() takes fused_decode_attention_k2b.

    Compares decode-logit closeness vs StreamingQuantizedCache (same per-head codec,
    chunked path). seq > recent_window so blocks flush and the committed-packed +
    fp16-tail merge is exercised. tf32 tensor cores -> ~1e-3 logit drift.
    """
    model = tiny_llama_d32().cuda()  # head_dim=32 so the fused-k2b dims gate passes
    input_ids = ids(vocab=97, seq=60, seed=13).cuda()
    k_spec, v_spec = _k2b_perhead()

    def _decode_logits(Cls):
        cache = Cls(model.config, k_spec=k_spec, v_spec=v_spec)
        cache.attach(model)
        with cache, torch.no_grad():
            model(input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            step = ids(vocab=97, seq=1, seed=14).cuda()
            out = model(step, past_key_values=cache, use_cache=True)
        cache.detach()
        return out.logits[0, -1].float()

    ref = _decode_logits(StreamingQuantizedCache)
    fused = _decode_logits(PackedStreamingCache)
    max_abs = (ref - fused).abs().max().item()
    # Codec is now identical on both sides (both use per-head Hadamard after CHANGE 1).
    # Residual diff is fused-kernel tf32 tensor-core math vs chunked fp32 reference,
    # so atol=0 exact parity is not expected here; 5e-2 covers the tf32 drift.
    assert max_abs < 5e-2, (
        f"fused-k2b decode logits diverged from streaming: max_abs={max_abs} "
        "(fused_decode_attention_k2b in packed_streaming.attend)"
    )
