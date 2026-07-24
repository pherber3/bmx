"""Second-model-family (Qwen3) compatibility: hf_compat resolution, the qk-norm
pre-RoPE capture point, streaming attach, spectral packs, EOS-list stop.
Everything offline via tiny_qwen3."""

import torch

from bmx.cache.hf_compat import (
    model_config_n_layers,
    resolve_decoder_layers,
    resolve_text_config,
    resolve_vocab_size,
)
from tests.factories import ids, tiny_qwen3


def test_hf_compat_resolves_qwen3():
    m = tiny_qwen3()
    assert model_config_n_layers(m) == 2
    layers = resolve_decoder_layers(m)
    sa = layers[0].self_attn
    assert hasattr(sa, "q_proj") and hasattr(sa, "k_proj")
    assert hasattr(sa, "q_norm") and hasattr(sa, "k_norm")  # the qk-norm family marker
    tc = resolve_text_config(m.config)
    assert tc is m.config  # no multimodal wrapper on Qwen3ForCausalLM
    assert tc.num_key_value_heads == 2
    d = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    assert d == 8  # explicit head_dim; Qwen3Config's default (128) must not leak in
    assert resolve_vocab_size(m.config) == 97


def test_collect_k_pre_is_post_knorm_qwen3():
    """THE capture-point invariant: k == RoPE(k_pre) on a real Qwen3 module
    tree. Qwen3 applies per-head RMSNorm (k_norm) between k_proj and RoPE, so
    k_pre MUST be captured at the k_norm OUTPUT — hooking k_proj (the Llama
    point) breaks this identity and every rope-at-read consumer downstream.
    Mirrors tests/test_cache_rope.py::test_apply_rope_matches_collect_cache."""
    from bmx.cache.collect import collect_cache
    from bmx.cache.rope import apply_rope, rope_cos_sin

    model = tiny_qwen3()
    input_ids = ids(seq=16)
    cache = collect_cache(model, input_ids, n_q_keep=256)
    S = input_ids.shape[1]
    cos, sin = rope_cos_sin(model.config, S)
    for i in range(model.config.num_hidden_layers):
        k_pre = cache[f"layer{i}.k_pre"].float()
        k = cache[f"layer{i}.k"].float()
        rel = (apply_rope(k_pre, cos.float(), sin.float()) - k).norm() / k.norm()
        assert rel < 1e-2, f"layer{i}: rel_fro {rel:.4e} — k_pre not post-k_norm?"


def test_collect_hooks_land_on_qk_norm_qwen3():
    """Structural pin for BOTH capture hooks (the k identity above cannot see
    q): on a qk-norm family the q/k hooks must hang on q_norm/k_norm, not the
    projections — the K4 W-moment statistics must see the query attention
    actually uses (q_norm output), not the raw q_proj output."""
    from bmx.cache.collect import register_hooks

    model = tiny_qwen3()
    store: dict = {}
    handles, n_layer = register_hooks(model, store, 8)
    try:
        sa = model.model.layers[0].self_attn
        assert len(sa.q_norm._forward_hooks) == 1
        assert len(sa.k_norm._forward_hooks) == 1
        assert len(sa.q_proj._forward_hooks) == 0
        assert len(sa.k_proj._forward_hooks) == 0
    finally:
        for h in handles:
            h.remove()
    assert n_layer == 2


def test_streaming_prerope_roundtrip_qwen3():
    """attach() + rope-at-read reproduce the true post-RoPE keys through a
    Qwen3 tree (fp16 K isolates capture+RoPE plumbing from quant error —
    the tests/test_streaming_cache.py::test_prerope_key_capture_and_rope_at_read
    pattern). seq=150 with recent_window=8 crosses the PAGE(128) flush
    threshold, so the committed region definitely derives from the
    hook-captured k_pre, not the pristine fp16 tail."""
    from bmx.cache.specs import CacheCodecSpec
    from bmx.cache.streaming import StreamingQuantizedCache

    model = tiny_qwen3()
    input_ids = ids(seq=150, seed=7)
    cache = StreamingQuantizedCache(
        model.config,
        k_spec=CacheCodecSpec(arm="fp16", pre_rope=True),
        v_spec=CacheCodecSpec(arm="fp16"),
        recent_window=8,
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
    assert rel < 1e-2


def test_packed_attach_hooks_k_norm_qwen3():
    """PackedStreamingCache.attach shares the capture dispatch (structural pin
    only — the packed path is out of replication scope, but a silently-wrong
    hook point must not ship). Mirror the smallest attach fixture in
    tests/test_packed_streaming.py if the constructor kwargs differ."""
    from bmx.cache.packed_streaming import PackedStreamingCache
    from bmx.cache.specs import CacheCodecSpec

    model = tiny_qwen3()
    cache = PackedStreamingCache(
        model.config,
        k_spec=CacheCodecSpec(arm="rtn_token", bits=4, group=8, pre_rope=True),
        v_spec=CacheCodecSpec(arm="rtn_token", bits=4, group=8),
    )
    cache.attach(model)
    try:
        sa = model.model.layers[0].self_attn
        assert len(sa.k_norm._forward_hooks) == 1
        assert len(sa.k_proj._forward_hooks) == 0
    finally:
        cache.detach()
