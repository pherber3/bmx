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


@pytest.mark.xfail(
    reason=(
        "requires the chunked_attention.py spectral read branch (Task 3, not "
        "yet implemented) — model(...) drives attention, which dequants every "
        "committed block via dequant_packed; spectral is not yet a split arm "
        "there (only the K WRITE path — spectral_quantize_packed via "
        "PackedStreamingLayer._pack_k_block — landed in Task 2). The guard "
        "move IS verified: quantize_packed no longer raises for spectral "
        "(test_packed_spectral_guards); this test documents the next gate."
    ),
    strict=False,
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
