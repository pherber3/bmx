"""Task 4 dispatch tests: PackedStreamingLayer.attend decode routing.

Two CPU-local tests:
1. fallback_used_on_no_cuda: TRITON_AVAILABLE=False (the local reality) →
   attend decode returns EXACTLY chunked_dequant_attention's result.
2. no_silent_swallow: TRITON_AVAILABLE=True (monkeypatched) + triton_decode_attention
   raises a sentinel → attend decode RAISES (does NOT silently fall back to chunked).
   This is the KEY test: it would FAIL if someone wrapped the dispatch in try/except.

Both tests run fully on CPU (AMD dev box, no CUDA/Triton).
"""

import pytest
import torch

import bmx.cache.packed_streaming as ps_mod
from bmx.cache.packed_streaming import PackedStreamingCache
from bmx.cache.specs import CacheCodecSpec
from factories import ids, tiny_llama


def _k2b_specs():
    return (
        CacheCodecSpec(
            arm="lowrank_rtn_channel", bits=3, rank=4, group=16, pre_rope=True
        ),
        CacheCodecSpec(arm="turboquant_mse", bits=2),
    )


def _rtn_specs():
    """Simpler RTN specs — for the no-swallow test (avoids k2b codepath)."""
    return (
        CacheCodecSpec(arm="rtn_token", bits=4, group=8, pre_rope=False),
        CacheCodecSpec(arm="rtn_token", bits=4, group=8),
    )


def _run_decode_step(model, input_ids, k_spec, v_spec):
    """Run a prefill + one decode step with PackedStreamingCache; return the cache."""
    cache = PackedStreamingCache(model.config, k_spec=k_spec, v_spec=v_spec)
    cache.attach(model)
    with torch.no_grad():
        model.generate(
            input_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
            past_key_values=cache,
        )
    cache.detach()
    return cache


# ---------------------------------------------------------------------------
# Test 1: fallback used on no-CUDA (CPU)
# ---------------------------------------------------------------------------


def test_fallback_used_on_no_cuda(monkeypatch):
    """With TRITON_AVAILABLE=False, attend decode produces EXACTLY the
    chunked_dequant_attention result (the capability-absence fallback path).

    We confirm the dispatch chooses the chunked path by:
      - Running generate through PackedStreamingCache (which calls attend internally).
      - Running the SAME generate through StreamingQuantizedCache (reference).
      - Asserting token equality (existing parity test pattern).

    Force TRITON_AVAILABLE=False via monkeypatch so this exercises the fallback
    path on EVERY machine (AMD/no-CUDA AND the CUDA VM) — the prior version
    asserted no-CUDA and so failed on the VM where Triton is present.
    """
    import bmx.cache.packed_streaming as ps_mod

    # Force the capability-absence path regardless of the host's real CUDA/Triton.
    monkeypatch.setattr(ps_mod, "TRITON_AVAILABLE", False)

    model = tiny_llama()
    input_ids = ids(vocab=97, seq=12, seed=7)
    k_spec, v_spec = _k2b_specs()

    from bmx.cache.streaming import StreamingQuantizedCache

    ref_cache = StreamingQuantizedCache(model.config, k_spec=k_spec, v_spec=v_spec)
    ref_cache.attach(model)
    with torch.no_grad():
        ref_out = model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,
            use_cache=True,
            past_key_values=ref_cache,
        )
    ref_cache.detach()

    packed = PackedStreamingCache(model.config, k_spec=k_spec, v_spec=v_spec)
    packed.attach(model)
    with torch.no_grad():
        packed_out = model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,
            use_cache=True,
            past_key_values=packed,
        )
    packed.detach()

    assert torch.equal(packed_out, ref_out), (
        "PackedStreamingCache (chunked fallback) diverged from StreamingQuantizedCache. "
        "The dispatch did not route to chunked_dequant_attention."
    )


# ---------------------------------------------------------------------------
# Test 2: no-silent-swallow (the KEY test)
# ---------------------------------------------------------------------------


class _SentinelError(RuntimeError):
    """Raised by the fake Triton kernel to test that errors propagate."""


def test_no_silent_swallow(monkeypatch):
    """Monkeypatch TRITON_AVAILABLE=True and triton_decode_attention to raise a
    sentinel error; assert that calling attend decode RAISES that error.

    This test FAILS if attend wraps the dispatch in try/except that falls back
    to chunked on error — exactly the silent-swallow trap Task 4 guards against.

    Patch targets:
      - bmx.cache.packed_streaming.TRITON_AVAILABLE  (the name attend checks)
      - bmx.cache.packed_streaming.triton_decode_attention  (the name attend calls)
    Both are module-level names in packed_streaming, imported at load time.
    """

    def _raise_sentinel(*args, **kwargs):
        raise _SentinelError("fake Triton kernel error — must propagate")

    # Set the capability flag to True so the dispatch enters the Triton branch.
    monkeypatch.setattr(ps_mod, "TRITON_AVAILABLE", True)
    # Replace the kernel with a stub that raises.
    monkeypatch.setattr(ps_mod, "triton_decode_attention", _raise_sentinel)

    model = tiny_llama()
    k_spec, v_spec = _rtn_specs()

    # Prefill only (puts packed blocks in place + prepares the layer for decoding).
    input_ids = ids(vocab=97, seq=12, seed=3)
    cache = PackedStreamingCache(model.config, k_spec=k_spec, v_spec=v_spec)
    cache.attach(model)
    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)

    # Now run ONE decode step — this calls attend with n_q==1 (decode).
    # With TRITON_AVAILABLE=True, attend must call triton_decode_attention, which raises.
    # If it silently falls back to chunked, no error is raised and the test fails.
    decode_ids = ids(vocab=97, seq=1, seed=99)
    with pytest.raises(_SentinelError, match="fake Triton kernel error"):
        with torch.no_grad():
            model(decode_ids, past_key_values=cache, use_cache=True)

    cache.detach()


# ---------------------------------------------------------------------------
# Test 3: k2b+pre_rope=True falls back to chunked even when TRITON_AVAILABLE=True
# ---------------------------------------------------------------------------


def test_k2b_pre_rope_falls_back_to_chunked(monkeypatch):
    """The canonical k2b config (lowrank_rtn_channel + pre_rope=True) must fall
    back to chunked_dequant_attention even when TRITON_AVAILABLE=True.

    The Triton kernel raises NotImplementedError for in-kernel RoPE on lowrank
    keys (capability-not-yet-implemented).  The guard in attend() must divert
    BEFORE the kernel is called, routing to chunked instead.

    This test:
      - Monkeypatches TRITON_AVAILABLE=True (simulates CUDA/Triton present).
      - Leaves triton_decode_attention as a stub that raises _SentinelError.
      - Runs a full prefill + decode with k2b + pre_rope=True.
      - Asserts NO error is raised (the guard diverted to chunked).
      - Asserts the chunked output matches the StreamingQuantizedCache reference
        (confirming the right path ran and produced correct output).

    FAILS before Fix 1 (the kernel would be called and raise NotImplementedError).
    PASSES after Fix 1 (the guard diverts to chunked before the kernel is touched).
    """
    from bmx.cache.streaming import StreamingQuantizedCache

    # Canonical k2b config: lowrank_rtn_channel + pre_rope=True + bits=3, rank=16, group=64
    # This is the config that crashed the VM (the gap that hid this bug).
    k_spec = CacheCodecSpec(
        arm="lowrank_rtn_channel", bits=3, rank=16, group=64, pre_rope=True
    )
    v_spec = CacheCodecSpec(arm="turboquant_mse", bits=2)

    def _raise_sentinel(*args, **kwargs):
        raise _SentinelError(
            "triton_decode_attention called for k2b+pre_rope — guard missing"
        )

    # Patch TRITON_AVAILABLE=True + replace kernel with sentinel that would crash.
    monkeypatch.setattr(ps_mod, "TRITON_AVAILABLE", True)
    monkeypatch.setattr(ps_mod, "triton_decode_attention", _raise_sentinel)

    model = tiny_llama()
    input_ids = ids(vocab=97, seq=12, seed=5)

    # Reference: StreamingQuantizedCache (chunked path, unpatched).
    ref_cache = StreamingQuantizedCache(model.config, k_spec=k_spec, v_spec=v_spec)
    ref_cache.attach(model)
    with torch.no_grad():
        ref_out = model.generate(
            input_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
            past_key_values=ref_cache,
        )
    ref_cache.detach()

    # Under test: PackedStreamingCache with TRITON_AVAILABLE=True but k2b+pre_rope guard.
    # Must NOT raise _SentinelError (guard diverts before sentinel is called).
    packed_cache = PackedStreamingCache(model.config, k_spec=k_spec, v_spec=v_spec)
    packed_cache.attach(model)
    with torch.no_grad():
        packed_out = model.generate(
            input_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
            past_key_values=packed_cache,
        )
    packed_cache.detach()

    # Output must match reference (chunked path ran correctly).
    assert torch.equal(packed_out, ref_out), (
        "k2b+pre_rope fallback output diverged from StreamingQuantizedCache reference. "
        "The guard diverted to chunked but chunked produced wrong output."
    )
