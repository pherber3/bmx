"""PackedStreamingCache: parity with StreamingQuantizedCache (bit-for-bit)."""

import pytest
import torch

from bmx.cache.packed_streaming import PackedStreamingCache
from bmx.cache.specs import CacheCodecSpec
from bmx.cache.streaming import StreamingQuantizedCache
from factories import ids, tiny_llama


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
    # seq=48 > recent_window=32 so a block flushes during prefill, exercising the
    # committed-blocks causal path that the seq=12 test never reaches.
    model = tiny_llama()
    input_ids = ids(vocab=97, seq=48, seed=11)
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
    # Verify resident slab is bounded (tail-only, not full S).
    layer0 = packed.layers[0]
    total_seq = 48 + 20  # prefill + new tokens
    slab_len = layer0.keys.shape[2]
    assert slab_len < total_seq, (
        f"Slab not pruned: keys.shape[2]={slab_len} >= total_seq={total_seq}"
    )
    assert slab_len <= layer0.recent_window + layer0._g + 1, (
        f"Slab too large: {slab_len} > recent_window({layer0.recent_window})"
        f" + g({layer0._g}) + 1"
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
