"""PackedStreaming write path — spectral K-branch, pack loading, V containers.

Task 2 of the packed-spectral Phase A plan
(docs/superpowers/plans/2026-07-23-packed-spectral-path.md). The read path
(chunked dequant-attention spectral branch) is Task 3 — these tests exercise
only the write path: pack loading at cache init, the guard move to
construction time, and container discipline on committed pages.
"""

import dataclasses

import pytest
import torch

from bmx.cache.specs import CacheCodecSpec
from bmx.cache.packed_streaming import PackedStreamingCache
from tests.factories import ids, tiny_llama
from tests.test_streaming_spectral import _fit_tiny_packs


def _k4_specs(path, group=8, budget=2.5):
    return (
        CacheCodecSpec(
            arm="spectral", pre_rope=True, group=group, pack_path=path, budget=budget
        ),
        CacheCodecSpec(arm="turboquant_mse", bits=2, seed=0),
    )


def test_packed_spectral_guards(tmp_path):
    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    k, v = _k4_specs(path)
    with pytest.raises(AssertionError, match="pre_rope"):
        PackedStreamingCache(
            model.config,
            k_spec=CacheCodecSpec(arm="spectral", pack_path=path, budget=2.5),
            v_spec=v,
        )
    with pytest.raises(AssertionError, match="pack_path"):
        PackedStreamingCache(
            model.config,
            k_spec=CacheCodecSpec(arm="spectral", pre_rope=True),
            v_spec=v,
        )


def test_packed_spectral_container_discipline(tmp_path):
    """The T4 pin: committed pages hold packed dtypes ONLY — no int16 indices,
    no fp32/fp16 dense codes. seq=300, W=8 flushes two 128-token pages."""
    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    k, v = _k4_specs(path)
    cache = PackedStreamingCache(model.config, k_spec=k, v_spec=v, recent_window=8)
    cache.attach(model)
    with cache:
        with torch.no_grad():
            model(ids(seq=300), past_key_values=cache, use_cache=True)
    layer = cache.layers[0]
    assert len(layer._k_blocks) == 2 and layer._committed_S_q == 256
    for kp, _s, _e in layer._k_blocks:
        for key, t in kp.items():
            if key.endswith("_codes"):
                assert t.dtype in (torch.uint8, torch.int8), (key, t.dtype)
            else:
                assert key.endswith("_scale") and t.dtype == torch.float32
    for vp, _s, _e in layer._v_blocks:
        assert "indices" not in vp and vp["indices_packed"].dtype == torch.uint8
        assert vp["norms"].dtype == torch.float16


def _run_pair(model, path, seq, group=8):
    from bmx.cache.streaming import StreamingQuantizedCache

    k, v = _k4_specs(path, group=group)
    caches = []
    for Cls in (StreamingQuantizedCache, PackedStreamingCache):
        cache = Cls(model.config, k_spec=k, v_spec=v, recent_window=8)
        cache.attach(model)
        with cache:
            with torch.no_grad():
                model(ids(seq=seq), past_key_values=cache, use_cache=True)
        caches.append(cache)
    return caches


def test_committed_blocks_bitwise_match_streaming(tmp_path):
    """BINDING GATE 1: same compute_flush_schedule, same PAGE=128 pages — the
    packed read-path reconstruction of every committed page (dequant -> fp32
    RoPE at true positions -> cache-dtype cast) equals streaming's frozen
    prefix bit-for-bit; V pages likewise.

    Cast to sl._q_prefix_k/_v's own dtype (== cache_dtype == the tiny_llama
    CPU fixture's fp32 model dtype here; fp16 on a real fp16-loaded model) —
    mirrors test_streaming_spectral_committed_block_matches_offline_and_frozen's
    `.to(committed_before.dtype)` convention rather than hardcoding fp16.
    """
    from bmx.cache.chunked_attention import _dequant_block
    from bmx.cache.rope import apply_rope

    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    stream, packed = _run_pair(model, path, seq=300)
    for i, (sl, pl) in enumerate(zip(stream.layers, packed.layers)):
        assert sl._committed_S_q == pl._committed_S_q == 256  # shared schedule
        for (kp, s, e), (vp, _s, _e) in zip(pl._k_blocks, pl._v_blocks):
            K = _dequant_block(kp, "spectral", 8, 0, pl._h_kv, pack=pl._pack)
            K = apply_rope(K, pl._rope_cos[s:e], pl._rope_sin[s:e]).to(
                sl._q_prefix_k.dtype
            )
            assert torch.equal(K, sl._q_prefix_k[:, s:e, :]), f"K layer {i} [{s}:{e})"
            V = _dequant_block(vp, "turboquant_mse", 8, 0, pl._h_kv).to(
                sl._q_prefix_v.dtype
            )
            assert torch.equal(V, sl._q_prefix_v[:, s:e, :]), f"V layer {i} [{s}:{e})"


def test_packed_spectral_decode_matches_oracle(tmp_path):
    """Chunked online-softmax decode vs naive_dense_attention on the same
    committed blocks (the standing oracle gate; mirror test_chunked_attention's
    tolerance conventions)."""
    from bmx.cache.chunked_attention import (
        attention_diff,
        chunked_dequant_attention,
        naive_dense_attention,
    )

    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    k_spec, v_spec = _k4_specs(path)
    cache = PackedStreamingCache(
        model.config, k_spec=k_spec, v_spec=v_spec, recent_window=8
    )
    cache.attach(model)
    with cache:
        with torch.no_grad():
            model(ids(seq=300), past_key_values=cache, use_cache=True)
    layer = cache.layers[0]
    n_q_heads = layer._h_kv * (
        model.config.num_attention_heads // model.config.num_key_value_heads
    )
    d = layer._d_head
    q = torch.randn(n_q_heads, 1, d, dtype=torch.float16)
    scale = 1.0 / (d**0.5)
    k_tail = layer.keys.squeeze(0)
    v_tail = layer.values.squeeze(0)
    common = dict(
        k_arm=layer.k_spec.arm,
        v_arm=layer.v_spec.arm,
        group=layer.k_spec.group,
        seed=layer.k_spec.seed,
        k_pre_rope=layer.k_spec.pre_rope,
        rope_cos=layer._rope_cos,
        rope_sin=layer._rope_sin,
        k_tail=k_tail,
        v_tail=v_tail,
        n_q_groups=n_q_heads // layer._h_kv,
        scale=scale,
        v_group=layer.v_spec.group,
        v_seed=layer.v_spec.seed,
    )
    oracle = naive_dense_attention(
        q, layer._k_blocks, layer._v_blocks, k_pack=layer._pack, **common
    )
    fast = chunked_dequant_attention(
        q, layer._k_blocks, layer._v_blocks, k_pack=layer._pack, **common
    )
    drift = attention_diff(fast, oracle)
    # Measured drift is 1.5-3.1e-5 (online-softmax reassembly, ~1 fp16 ULP);
    # 1e-3 keeps ~30x headroom while actually policing the path.
    assert drift["max_abs"] < 1e-3, drift


def test_packed_spectral_read_path_k_tier_cols_bitwise_identical(tmp_path):
    """FIX 1 license: threading k_tier_cols (the precomputed
    tier_columns(pack.bits)) through chunked_dequant_attention's read path is
    the SAME math as leaving it None (spectral_dequant_packed recomputes it
    internally) — bitwise-identical output either way. Trivial by
    construction (see spectral_dequant_packed's cols_by_tier default), pinned
    here so the threading itself can't silently change values."""
    from bmx.cache.chunked_attention import chunked_dequant_attention
    from bmx.cache.spectral import tier_columns

    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    k_spec, v_spec = _k4_specs(path)
    cache = PackedStreamingCache(
        model.config, k_spec=k_spec, v_spec=v_spec, recent_window=8
    )
    cache.attach(model)
    with cache:
        with torch.no_grad():
            model(ids(seq=300), past_key_values=cache, use_cache=True)
    layer = cache.layers[0]
    n_q_heads = layer._h_kv * (
        model.config.num_attention_heads // model.config.num_key_value_heads
    )
    d = layer._d_head
    q = torch.randn(n_q_heads, 1, d, dtype=torch.float16)
    scale = 1.0 / (d**0.5)
    k_tail = layer.keys.squeeze(0)
    v_tail = layer.values.squeeze(0)
    common = dict(
        k_arm=layer.k_spec.arm,
        v_arm=layer.v_spec.arm,
        group=layer.k_spec.group,
        seed=layer.k_spec.seed,
        k_pre_rope=layer.k_spec.pre_rope,
        rope_cos=layer._rope_cos,
        rope_sin=layer._rope_sin,
        k_tail=k_tail,
        v_tail=v_tail,
        n_q_groups=n_q_heads // layer._h_kv,
        scale=scale,
        v_group=layer.v_spec.group,
        v_seed=layer.v_spec.seed,
    )
    without_cols = chunked_dequant_attention(
        q, layer._k_blocks, layer._v_blocks, k_pack=layer._pack, **common
    )
    with_cols = chunked_dequant_attention(
        q,
        layer._k_blocks,
        layer._v_blocks,
        k_pack=layer._pack,
        k_tier_cols=tier_columns(layer._pack.bits),
        **common,
    )
    assert torch.equal(with_cols, without_cols)
    # And layer._tier_cols itself (the precomputed value the layer actually
    # threads) reproduces the identical result.
    with_layer_cols = chunked_dequant_attention(
        q,
        layer._k_blocks,
        layer._v_blocks,
        k_pack=layer._pack,
        k_tier_cols=layer._tier_cols,
        **common,
    )
    assert torch.equal(with_layer_cols, without_cols)


def test_packed_spectral_cached_decode_never_recomputes_tier_columns(
    tmp_path, monkeypatch
):
    """Structural pin for FIX 1: the hot decode loop must receive
    PackedStreamingLayer._tier_cols already threaded, never recomputing
    tier_columns(pack.bits) per block per decode step. Monkeypatch
    bmx.cache.spectral.tier_columns to raise — a real CUDA decode step
    through the cache path must not hit it (CPU decode also routes through
    chunked_dequant_attention here, since the fused k2b/packed kernels
    require CUDA — see test_packed_dispatch.py's fallback pattern)."""
    import bmx.cache.spectral as spectral_mod

    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    k_spec, v_spec = _k4_specs(path)
    cache = PackedStreamingCache(
        model.config, k_spec=k_spec, v_spec=v_spec, recent_window=8
    )
    cache.attach(model)

    def _raise_if_called(bits):
        raise AssertionError(
            "tier_columns recomputed during cached decode — the hot-path "
            "k_tier_cols threading regressed"
        )

    with cache:
        with torch.no_grad():
            # Prefill: commits at least one page (seq=300 with recent_window=8
            # crosses the flush threshold — matches the fixture used above).
            # _tier_cols is precomputed once at layer construction (before
            # this call), so prefill itself must not hit tier_columns either
            # — but guard only the decode step below to isolate the claim
            # FIX 1 actually makes (the hot per-block read loop).
            model(ids(seq=300), past_key_values=cache, use_cache=True)
            monkeypatch.setattr(spectral_mod, "tier_columns", _raise_if_called)
            model(ids(seq=1), past_key_values=cache, use_cache=True)


def test_packed_spectral_generate_matches_streaming(tmp_path):
    """BINDING GATE 2 (short context): greedy tokens identical, both the
    no-flush (seq=120) and flush-during-prefill (seq=300) variants — mirrors
    test_packed_generate_matches_streaming / _long_prefill."""
    from bmx.cache.streaming import StreamingQuantizedCache

    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    k_spec, v_spec = _k4_specs(path)

    for seq in (12, 300):
        input_ids = ids(vocab=97, seq=seq, seed=5)

        ref_cache = StreamingQuantizedCache(
            model.config, k_spec=k_spec, v_spec=v_spec, recent_window=8
        )
        ref_cache.attach(model)
        with torch.no_grad():
            ref = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                use_cache=True,
                past_key_values=ref_cache,
            )
        ref_cache.detach()

        packed = PackedStreamingCache(
            model.config, k_spec=k_spec, v_spec=v_spec, recent_window=8
        )
        packed.attach(model)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                use_cache=True,
                past_key_values=packed,
            )
        packed.detach()

        assert torch.equal(out, ref), f"seq={seq}"


def test_packed_spectral_two_block_prefill_logits_match_streaming(tmp_path):
    """The cached-two-block-prefill mask-bug class (cf21d06): logit parity on a
    prefill split across two forwards — mirrors
    test_packed_two_block_prefill_logits_match_streaming with the spectral arm."""
    from bmx.cache.streaming import StreamingQuantizedCache

    def _last_logit_two_block_prefill(model, input_ids, n_prefill, k_spec, v_spec, Cls):
        cache = Cls(model.config, k_spec=k_spec, v_spec=v_spec, recent_window=8)
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

    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    k_spec, v_spec = _k4_specs(path)
    input_ids = ids(vocab=97, seq=60, seed=9)  # > recent_window so prefill flushes
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


def test_packed_spectral_dec_quant_int8_matches_streaming(tmp_path):
    """dec_quant='int8' must be applied at pack materialization on the packed
    cache too (mirroring StreamingQuantizedCache.__init__ exactly: once, at
    load, via dataclasses.replace) — the packed cache must not silently ignore
    it. Pin: both caches' layer-0 pack.dec tensors are bitwise identical."""
    from bmx.cache.streaming import StreamingQuantizedCache

    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    k = CacheCodecSpec(
        arm="spectral",
        pre_rope=True,
        group=8,
        pack_path=path,
        budget=2.5,
        dec_quant="int8",
    )
    v = CacheCodecSpec(arm="turboquant_mse", bits=2, seed=0)

    stream = StreamingQuantizedCache(model.config, k_spec=k, v_spec=v)
    packed = PackedStreamingCache(model.config, k_spec=k, v_spec=v)

    stream_pack = stream._pack_for_layer(0)
    packed_pack = packed._packs[0]
    assert torch.equal(stream_pack.dec, packed_pack.dec)
    assert stream_pack.dec.dtype == packed_pack.dec.dtype


def test_packed_bits_per_entry_equals_streaming(tmp_path):
    """Same schedule, same codec calls per page => identical honest accounting,
    including the per-sequence skeptic pack charge. Also un-NaNs the census."""
    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    stream, packed = _run_pair(model, path, seq=300)
    assert packed.bits_per_entry() == stream.bits_per_entry()
    sm = stream.memory_report(seq_len=300)
    pm = packed.memory_report(seq_len=300)
    assert pm == sm  # same bpe in => same honest bytes out


def test_packed_bits_per_entry_equals_streaming_dec_quant_int8(tmp_path):
    """Same equality pin, but with dec_quant='int8' — exercises the
    _dec_bits=8.0 skeptic-v2-int8 charge path on both caches."""
    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    k, v = _k4_specs(path)
    k = dataclasses.replace(k, dec_quant="int8")

    from bmx.cache.streaming import StreamingQuantizedCache

    caches = []
    for Cls in (StreamingQuantizedCache, PackedStreamingCache):
        cache = Cls(model.config, k_spec=k, v_spec=v, recent_window=8)
        cache.attach(model)
        with cache:
            with torch.no_grad():
                model(ids(seq=300), past_key_values=cache, use_cache=True)
        caches.append(cache)
    stream, packed = caches
    assert packed.bits_per_entry() == stream.bits_per_entry()
    sm = stream.memory_report(seq_len=300)
    pm = packed.memory_report(seq_len=300)
    assert pm == sm


# ---------------------------------------------------------------------------
# K4 Lloyd payload-quantizer gate, Task 1 (2026-07-25 design):
# CacheCodecSpec.payload_quant threaded through PackedStreamingCache
# (write path + chunked-attention read path).
# ---------------------------------------------------------------------------


def test_packed_payload_quant_default_inert(tmp_path):
    """payload_quant default ('rtn') must reproduce byte-identical bpe AND
    committed-block bytes vs a spec that omits the field entirely."""
    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    k_implicit, v = _k4_specs(path)
    k_explicit = dataclasses.replace(k_implicit, payload_quant="rtn")

    def _run(k):
        cache = PackedStreamingCache(model.config, k_spec=k, v_spec=v, recent_window=8)
        cache.attach(model)
        with cache:
            with torch.no_grad():
                model(ids(seq=300), past_key_values=cache, use_cache=True)
        return cache

    implicit, explicit = _run(k_implicit), _run(k_explicit)
    assert implicit.bits_per_entry() == explicit.bits_per_entry()
    for (kpi, si, ei), (kpe, se, ee) in zip(
        implicit.layers[0]._k_blocks, explicit.layers[0]._k_blocks
    ):
        assert si == se and ei == ee
        for key, t in kpi.items():
            assert torch.equal(t, kpe[key]), key


def test_packed_payload_quant_lloyd_committed_blocks_bitwise_match_streaming(tmp_path):
    """BINDING: with payload_quant='lloyd', the packed read-path reconstruction
    of every committed page (dequant -> RoPE -> cast) must equal streaming's
    frozen prefix bit-for-bit -- mirrors
    test_committed_blocks_bitwise_match_streaming but for the lloyd quantizer,
    proving the read path (_dequant_block's k_quantizer thread) is wired, not
    just the write path."""
    from bmx.cache.chunked_attention import _dequant_block
    from bmx.cache.rope import apply_rope
    from bmx.cache.streaming import StreamingQuantizedCache

    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    k, v = _k4_specs(path)
    k = dataclasses.replace(k, payload_quant="lloyd")

    caches = []
    for Cls in (StreamingQuantizedCache, PackedStreamingCache):
        cache = Cls(model.config, k_spec=k, v_spec=v, recent_window=8)
        cache.attach(model)
        with cache:
            with torch.no_grad():
                model(ids(seq=300), past_key_values=cache, use_cache=True)
        caches.append(cache)
    stream, packed = caches
    assert (
        stream.bits_per_entry() == packed.bits_per_entry()
    )  # bpe identical by construction
    for sl, pl in zip(stream.layers, packed.layers):
        assert sl._committed_S_q == pl._committed_S_q == 256
        for (kp, s, e), (_vp, _s, _e) in zip(pl._k_blocks, pl._v_blocks):
            K = _dequant_block(
                kp, "spectral", 8, 0, pl._h_kv, pack=pl._pack, k_quantizer="lloyd"
            )
            K = apply_rope(K, pl._rope_cos[s:e], pl._rope_sin[s:e]).to(
                sl._q_prefix_k.dtype
            )
            assert torch.equal(K, sl._q_prefix_k[:, s:e, :])


def test_packed_payload_quant_lloyd_decode_matches_oracle(tmp_path):
    """Chunked online-softmax decode (layer.attend, the real call path) must
    equal naive_dense_attention (the standing oracle) with
    payload_quant='lloyd', proving k_quantizer is wired through the full
    chunked_dequant_attention chain (prefill + decode)."""
    from bmx.cache.chunked_attention import attention_diff, naive_dense_attention

    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    k, v = _k4_specs(path)
    k = dataclasses.replace(k, payload_quant="lloyd")
    cache = PackedStreamingCache(model.config, k_spec=k, v_spec=v, recent_window=8)
    cache.attach(model)
    with cache:
        with torch.no_grad():
            model(ids(seq=300), past_key_values=cache, use_cache=True)
    layer = cache.layers[0]
    d_head = layer._d_head
    n_q_groups = model.config.num_attention_heads // layer._h_kv
    q = torch.randn(model.config.num_attention_heads, 1, d_head)
    scale = d_head**-0.5

    chunked_out = layer.attend(q, scale)  # real call path, k_quantizer wired internally

    k_tail = layer.keys.squeeze(0)
    v_tail = layer.values.squeeze(0)
    oracle_out = naive_dense_attention(
        q,
        layer._k_blocks,
        layer._v_blocks,
        k_arm=k.arm,
        v_arm=v.arm,
        group=k.group,
        seed=k.seed,
        k_pre_rope=k.pre_rope,
        rope_cos=layer._rope_cos,
        rope_sin=layer._rope_sin,
        k_tail=k_tail,
        v_tail=v_tail,
        n_q_groups=n_q_groups,
        scale=scale,
        v_group=v.group,
        v_seed=v.seed,
        k_pack=layer._pack,
        k_tier_cols=layer._tier_cols,
        k_quantizer="lloyd",
    )
    diff = attention_diff(chunked_out, oracle_out)
    assert diff["max_abs"] < 1e-4, diff
