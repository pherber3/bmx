"""StreamingQuantizedCache: plumbing, quality, and memory gates (tiny_llama)."""

import torch

from bmx.cache.specs import CacheCodecSpec
from bmx.cache.streaming import StreamingQuantizedCache
from factories import ids, tiny_llama


def _fp16():
    return CacheCodecSpec(arm="fp16")


def test_fp16_passthrough_bit_identical_prefill():
    # With a no-op codec the streaming cache must reproduce a plain forward exactly.
    model = tiny_llama()
    input_ids = ids(vocab=97, seq=16, seed=3)
    with torch.no_grad():
        ref = model(input_ids, use_cache=True)
    cache = StreamingQuantizedCache(model.config, k_spec=_fp16(), v_spec=_fp16())
    with torch.no_grad():
        out = model(input_ids, past_key_values=cache, use_cache=True)
    assert torch.equal(out.logits, ref.logits)


def test_fp16_passthrough_bit_identical_generate():
    # The real autoregressive loop: greedy generate must match a plain default cache.
    model = tiny_llama()
    input_ids = ids(vocab=97, seq=8, seed=4)
    with torch.no_grad():
        ref = model.generate(
            input_ids, max_new_tokens=10, do_sample=False, use_cache=True
        )
    cache = StreamingQuantizedCache(model.config, k_spec=_fp16(), v_spec=_fp16())
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=False,
            use_cache=True,
            past_key_values=cache,
        )
    assert torch.equal(out, ref)


def test_prerope_key_capture_and_rope_at_read():
    # The cache must (1) capture pre-RoPE keys via its own hook, (2) on read produce
    # post-RoPE keys close to the true post-RoPE keys — confirming RoPE-at-read at the
    # right positions. fp16 K spec => exact match (no quant error), isolating the
    # capture+RoPE plumbing from quantization error.

    model = tiny_llama()
    input_ids = ids(vocab=97, seq=40, seed=7)

    # fp16-but-pre_rope: capture pre-RoPE, apply_rope at read, no quant. Must match
    # the true post-RoPE keys a plain cache stores.
    cache = StreamingQuantizedCache(
        model.config,
        k_spec=CacheCodecSpec(arm="fp16", pre_rope=True),
        v_spec=CacheCodecSpec(arm="fp16"),
    )
    cache.attach(model)
    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True)
    cache.detach()
    k_post, _ = cache.reconstruct_layer(0)

    ref = StreamingQuantizedCache(
        model.config,
        k_spec=CacheCodecSpec(arm="fp16"),
        v_spec=CacheCodecSpec(arm="fp16"),
    )
    with torch.no_grad():
        model(input_ids, past_key_values=ref, use_cache=True)
    k_true = ref.layers[0].keys

    rel = (k_post.float() - k_true.float()).norm() / k_true.float().norm().clamp_min(
        1e-6
    )
    assert rel < 1e-2  # capture + RoPE-at-read reproduces true post-RoPE keys


def test_quantized_prerope_recon_finite_and_compressed():
    # seq=48 so S=48 is divisible by group=16 (lowrank_rtn_channel requires S % group == 0).
    model = tiny_llama()
    input_ids = ids(vocab=97, seq=48, seed=8)
    cache = StreamingQuantizedCache(
        model.config,
        k_spec=CacheCodecSpec(
            arm="lowrank_rtn_channel", bits=3, rank=4, group=16, pre_rope=True
        ),
        v_spec=CacheCodecSpec(arm="rtn_token", bits=2, group=16),
    )
    cache.attach(model)
    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True)
    cache.detach()
    k_post, v = cache.reconstruct_layer(0)
    assert torch.isfinite(k_post).all() and torch.isfinite(v).all()
    bpe_k, bpe_v = cache.bits_per_entry()
    assert bpe_k < 16.0 and bpe_v < 16.0


def test_attach_is_idempotent():
    # Double attach must not double-register hooks (which would double-stash _k_pre).
    model = tiny_llama()
    input_ids = ids(vocab=97, seq=40, seed=7)
    cache = StreamingQuantizedCache(
        model.config,
        k_spec=CacheCodecSpec(arm="fp16", pre_rope=True),
        v_spec=CacheCodecSpec(arm="fp16"),
    )
    cache.attach(model)
    cache.attach(model)  # second attach must not duplicate hooks
    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True)
    cache.detach()
    # If hooks double-fired, _k_pre would be 2x the sequence length. Confirm correct S.
    k_post, _ = cache.reconstruct_layer(0)
    assert k_post.shape[2] == input_ids.shape[1], (
        f"expected S={input_ids.shape[1]}, got {k_post.shape[2]} (double-stash?)"
    )
