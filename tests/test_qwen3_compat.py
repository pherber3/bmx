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


class _StubTok:
    """Minimal tokenizer for generate_through_cache: ids in, space-joined out."""

    eos_token_id = None

    def decode(self, t, skip_special_tokens=True):
        return " ".join(map(str, t.tolist() if hasattr(t, "tolist") else t))


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


def test_streaming_k2b_qwen3():
    """The proven k2b arm streams through a Qwen3 module tree: attach() hooks
    fire, pages flush, bpe accounting is real (<16). Mirror the fixture pattern
    of tests/test_streaming_cache.py::test_k2b_pre_rope_streams_token_by_token
    (seq=150, recent_window=8 — read it first, copy its invariant exactly)."""
    from bmx.cache.recipes import spec_pair
    from bmx.cache.streaming import StreamingQuantizedCache

    model = tiny_qwen3()
    k_spec, v_spec = spec_pair("k2b", rank=4, group=8)
    cache = StreamingQuantizedCache(
        model.config, k_spec=k_spec, v_spec=v_spec, recent_window=8
    )
    cache.attach(model)
    with cache:
        with torch.no_grad():
            model(ids(seq=150), past_key_values=cache, use_cache=True)
    bpe_k, bpe_v = cache.bits_per_entry()
    assert bpe_k < 16.0 and bpe_v < 16.0  # at least one page actually flushed


def test_generate_k4_qwen3(tmp_path):
    """k4_b2.5 end-to-end (attach + hooks + spectral flush + greedy decode) on
    Qwen3 — the exact recipe the VM probe runs."""
    from bmx.cache.generate import generate_through_cache
    from bmx.cache.recipes import spec_pair
    from tests.test_streaming_spectral import _fit_tiny_packs

    model = tiny_qwen3()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)  # budget=2.5, group=8

    k_spec, v_spec = spec_pair("k4_b2.5", group=8, pack_path=path)
    out = generate_through_cache(
        model,
        _StubTok(),
        ids(seq=150),
        n_prefill=128,
        k_spec=k_spec,
        v_spec=v_spec,
        max_new_tokens=4,
    )
    assert isinstance(out, str) and out


def test_generate_stops_on_any_eos_in_list():
    """Qwen3's generation_config.eos_token_id is a LIST ([151645, 151643] on
    the real model) — the decode loop must stop on ANY member. Pinned here on
    tiny_qwen3 because Llama-3.1's list was the only case ever exercised.
    Verified stop semantics (generate.py:105-113): the EOS token IS appended
    to new_ids before the break, so the stub-decoded output has length 1 when
    the first decode token is an eos member."""
    from bmx.cache.generate import generate_through_cache
    from bmx.cache.specs import CacheCodecSpec

    model = tiny_qwen3()
    fp16 = CacheCodecSpec(arm="fp16")
    prompt = ids(seq=24)
    # Probe: which token does greedy decode emit SECOND? (the first decode
    # token is emitted before the loop's eos check ever runs — generate.py:103)
    out = generate_through_cache(
        model,
        _StubTok(),
        prompt,
        n_prefill=12,
        k_spec=fp16,
        v_spec=fp16,
        max_new_tokens=4,
        strip=False,
    )
    toks = out.split()
    assert len(toks) == 4  # no eos configured => full budget
    second = int(toks[1])
    # Re-run with an eos LIST containing that token (plus a never-emitted one):
    model.generation_config.eos_token_id = [96, second]
    out2 = generate_through_cache(
        model,
        _StubTok(),
        prompt,
        n_prefill=12,
        k_spec=fp16,
        v_spec=fp16,
        max_new_tokens=4,
        strip=False,
    )
    assert len(out2.split()) == 2  # stopped ON the second token, immediately
