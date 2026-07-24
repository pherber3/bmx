"""PackedStreaming write path — spectral K-branch, pack loading, V containers.

Task 2 of the packed-spectral Phase A plan
(docs/superpowers/plans/2026-07-23-packed-spectral-path.md). The read path
(chunked dequant-attention spectral branch) is Task 3 — these tests exercise
only the write path: pack loading at cache init, the guard move to
construction time, and container discipline on committed pages.
"""

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
    assert drift["max_abs"] < 1e-2, drift


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
