"""PackedStreamingCache: parity with StreamingQuantizedCache (bit-for-bit)."""

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
