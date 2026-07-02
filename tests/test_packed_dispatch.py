"""Task 4 dispatch tests: PackedStreamingLayer.attend decode routing.

Key tests:
1. fallback_used_on_no_cuda: TRITON_AVAILABLE=False (the local reality) →
   attend decode returns EXACTLY chunked_dequant_attention's result.
2. no_silent_swallow: TRITON_AVAILABLE=True (monkeypatched) + fused_decode_attention_packed
   raises a sentinel → attend decode RAISES (does NOT silently fall back to chunked).
   This is the KEY test: it would FAIL if someone wrapped the dispatch in try/except.
   It needs post-RoPE rtn_token specs on BOTH K and V AND >=1 committed uniform-length
   page (fused_packed_ok's full predicate) — a too-short prefill silently commits zero
   blocks and defeats the test without CUDA to reveal it, hence the reachability probe
   inside the test plus its CPU-runnable companion
   (no_silent_swallow_fixture_reaches_fused_predicate).
3. k2b_pre_rope_falls_back_to_chunked: a k2b config must route to chunked even when
   TRITON_AVAILABLE=True (neither fused predicate matches).

Most tests run fully on CPU (AMD dev box, no CUDA/Triton); test_no_silent_swallow
needs a real CUDA branch entry (q.is_cuda) and is skipped otherwise, but its fixture
is reachability-checked locally by its CPU companion test.

The legacy per-block triton_decode_attention fallback was removed (Wave 2 debloat) —
the fail-loud contract now governs only the two fused entry points
(fused_decode_attention_packed / fused_decode_attention_k2b); everything else falls
through to chunked_dequant_attention.
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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="dispatch enters the fused Triton branch only when q.is_cuda — needs CUDA",
)
def test_no_silent_swallow(monkeypatch):
    """Monkeypatch TRITON_AVAILABLE=True and fused_decode_attention_packed to raise
    a sentinel error; assert that calling attend decode RAISES that error.

    This test FAILS if attend wraps the dispatch in try/except that falls back
    to chunked on error — exactly the silent-swallow trap Task 4 guards against.

    CUDA-gated: the dispatch now also requires q.is_cuda (a CPU model on a CUDA
    box uses chunked), so the model must be on CUDA for the fused branch to be
    taken at all. The RTN specs below are post-RoPE rtn_token on both K and V —
    the arm/pre_rope half of fused_packed_ok — but that predicate ALSO requires
    uniform_blk (>=1 committed page, all the same length): PAGE=128 with the
    default recent_window=32 means compute_flush_schedule(S, 32, 128) needs
    S >= 160 to commit anything at all (12 tokens commits zero blocks, which
    silently makes fused_packed_ok False and defeats this test — see the
    reachability probe below and the companion CPU test). With seq=160, attend
    must call fused_decode_attention_packed.  Patch targets:
      - bmx.cache.packed_streaming.TRITON_AVAILABLE  (the name attend checks)
      - bmx.cache.packed_streaming.fused_decode_attention_packed  (the name attend calls)
    Both are module-level names in packed_streaming, imported at load time.
    """

    def _raise_sentinel(*args, **kwargs):
        raise _SentinelError("fake Triton kernel error — must propagate")

    # Set the capability flag to True so the dispatch enters the fused branch.
    monkeypatch.setattr(ps_mod, "TRITON_AVAILABLE", True)
    # Replace the fused-packed kernel with a stub that raises.
    monkeypatch.setattr(ps_mod, "fused_decode_attention_packed", _raise_sentinel)

    model = tiny_llama().cuda()  # CUDA so q.is_cuda -> the fused branch is taken
    k_spec, v_spec = _rtn_specs()

    # Prefill only (puts packed blocks in place + prepares the layer for decoding).
    # seq=160 > PAGE(128) + recent_window(32) so exactly one 128-token page commits
    # (see docstring above) -- required for uniform_blk / fused_packed_ok to be True.
    input_ids = ids(vocab=97, seq=160, seed=3).cuda()
    cache = PackedStreamingCache(model.config, k_spec=k_spec, v_spec=v_spec)
    cache.attach(model)
    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)

    # Reachability probe: confirm the fixture actually commits >=1 uniform-length
    # page BEFORE trusting the sentinel assertion below. Without this, a fixture
    # that commits zero blocks would make fused_packed_ok False, the sentinel
    # would never fire, and pytest.raises would fail for the WRONG reason (or,
    # if uniform_blk's `bool(blocks) and ...` happened to look satisfied some
    # other way, silently pass without ever exercising the fused branch). This
    # mirrors attend's own `uniform_blk` computation exactly.
    layer = cache.layers[0]
    assert len(layer._k_blocks) >= 1, (
        "fixture committed zero packed blocks -- fused_packed_ok cannot be True; "
        "raise the prefill length past PAGE(128) + recent_window(32)"
    )
    uniform_blk = len({e - s for _, s, e in layer._k_blocks}) == 1
    assert uniform_blk, (
        "committed blocks are not uniform length -- fused_packed_ok cannot be True"
    )

    # Now run ONE decode step — this calls attend with n_q==1 (decode).
    # With TRITON_AVAILABLE=True + q.is_cuda + rtn_token/rtn_token specs + the
    # reachability probe above confirming >=1 uniform committed page,
    # fused_packed_ok is True, so attend must call fused_decode_attention_packed,
    # which raises. If it silently falls back to chunked, no error is raised -> fail.
    decode_ids = ids(vocab=97, seq=1, seed=99).cuda()
    with pytest.raises(_SentinelError, match="fake Triton kernel error"):
        with torch.no_grad():
            model(decode_ids, past_key_values=cache, use_cache=True)

    cache.detach()


def test_no_silent_swallow_fixture_reaches_fused_predicate():
    """CPU-runnable companion to test_no_silent_swallow.

    test_no_silent_swallow is CUDA-skipped on this dev box, so a broken fixture
    (e.g. one that commits zero packed blocks) would silently never be caught
    locally -- it would only explode on the GH200 CUDA re-verify when the
    sentinel-raise assertion fails for the wrong reason. This test builds the
    EXACT SAME prefill fixture (same seq len, specs, seeds) on CPU and asserts
    the block-commit / uniform-length condition directly, giving local evidence
    that the fixture reaches the fused_packed_ok predicate's uniform_blk half
    (the CUDA/TRITON_AVAILABLE/arm/pre_rope half needs no runtime check -- it's
    static from the specs and the .cuda() call).
    """
    model = tiny_llama()
    k_spec, v_spec = _rtn_specs()

    input_ids = ids(vocab=97, seq=160, seed=3)
    cache = PackedStreamingCache(model.config, k_spec=k_spec, v_spec=v_spec)
    cache.attach(model)
    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)

    layer = cache.layers[0]
    assert len(layer._k_blocks) >= 1, (
        "fixture committed zero packed blocks -- fused_packed_ok cannot be True; "
        "raise the prefill length past PAGE(128) + recent_window(32)"
    )
    uniform_blk = len({e - s for _, s, e in layer._k_blocks}) == 1
    assert uniform_blk, (
        "committed blocks are not uniform length -- fused_packed_ok cannot be True"
    )

    cache.detach()


# ---------------------------------------------------------------------------
# Test 3: k2b+pre_rope=True falls back to chunked even when TRITON_AVAILABLE=True
# ---------------------------------------------------------------------------


def test_k2b_pre_rope_falls_back_to_chunked(monkeypatch):
    """A k2b config with full-C turboquant_mse V AND a non-power-of-2 rank
    (lowrank_rtn_channel + pre_rope) must route to chunked_dequant_attention even
    when TRITON_AVAILABLE=True.

    Dispatch (attend): fused_packed_ok is False (K arm != rtn_token); fused_k2b_ok
    is False for TWO independent reasons — V arm is turboquant_mse, not the
    per-head turboquant_mse_perhead the fused k2b kernel needs, AND rank=12 is not
    a power of 2 (k2b_dims_ok requires it, for the in-kernel tl.arange/tl.dot
    preconditions). Any non-fused config now falls straight through to chunked —
    the legacy per-block triton_decode_attention fallback was removed (Wave 2).

    This test:
      - Monkeypatches TRITON_AVAILABLE=True (simulates CUDA/Triton present).
      - Replaces BOTH fused_decode_attention_packed and fused_decode_attention_k2b
        with sentinels that raise if called.
      - Runs a full prefill + decode with this k2b config.
      - Asserts NO error is raised (neither fused kernel was ever reached).
      - Asserts the chunked output matches the StreamingQuantizedCache reference
        (confirming the right path ran and produced correct output).
    """
    from bmx.cache.streaming import StreamingQuantizedCache

    # k2b config with a non-pow2 rank (12) AND non-perhead V — two independent
    # reasons fused_k2b_ok must be False. bits=3, group=64, pre_rope=True is the
    # config family that crashed the VM (the gap that hid this bug).
    k_spec = CacheCodecSpec(
        arm="lowrank_rtn_channel", bits=3, rank=12, group=64, pre_rope=True
    )
    v_spec = CacheCodecSpec(arm="turboquant_mse", bits=2)

    def _raise_sentinel(*args, **kwargs):
        raise _SentinelError(
            "fused kernel called for k2b+pre_rope non-pow2-rank — guard missing"
        )

    # Patch TRITON_AVAILABLE=True + replace BOTH fused kernels with sentinels that
    # would crash if either fused predicate wrongly matched this config.
    monkeypatch.setattr(ps_mod, "TRITON_AVAILABLE", True)
    monkeypatch.setattr(ps_mod, "fused_decode_attention_packed", _raise_sentinel)
    monkeypatch.setattr(ps_mod, "fused_decode_attention_k2b", _raise_sentinel)

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


# ---------------------------------------------------------------------------
# _PagedStacks: incremental append must equal from-scratch build (the I3 fix).
# Fully CPU-testable — the correctness gate for the production stacked-KV buffer.
# ---------------------------------------------------------------------------


def _packed_rtn_blocks(n_blocks, h_kv, blk, d, group, seed=0):
    """Build n_blocks of (rtn_token packed dict, start, end) for K and V."""
    from bmx.cache.codecs import quantize_packed
    from bmx.cache.collect import to_matrix

    torch.manual_seed(seed)
    k_blocks, v_blocks = [], []
    for i in range(n_blocks):
        s, e = i * blk, (i + 1) * blk
        kM = to_matrix(torch.randn(h_kv, blk, d))
        vM = to_matrix(torch.randn(h_kv, blk, d))
        kp, _ = quantize_packed("rtn_token", kM, bits=4, group=group, seed=seed)
        vp, _ = quantize_packed("rtn_token", vM, bits=4, group=group, seed=seed)
        k_blocks.append((kp, s, e))
        v_blocks.append((vp, s, e))
    return k_blocks, v_blocks


def _k2b_perhead_blocks(n_blocks, h_kv, blk, d, rank, group, seed=0):
    """Build n_blocks of (lowrank_rtn_channel K, turboquant_mse_perhead V) blocks."""
    from bmx.cache.codecs import quantize_packed
    from bmx.cache.collect import to_matrix

    torch.manual_seed(seed)
    k_blocks, v_blocks = [], []
    for i in range(n_blocks):
        s, e = i * blk, (i + 1) * blk
        kM = to_matrix(torch.randn(h_kv, blk, d))
        vM = to_matrix(torch.randn(h_kv, blk, d))
        kp, _ = quantize_packed(
            "lowrank_rtn_channel", kM, bits=3, group=group, rank=rank, seed=seed
        )
        vp, _ = quantize_packed(
            "turboquant_mse_perhead", vM, bits=2, seed=seed, h_heads=h_kv
        )
        k_blocks.append((kp, s, e))
        v_blocks.append((vp, s, e))
    return k_blocks, v_blocks


def test_paged_stacks_packed_incremental_equals_rebuild():
    """_PagedStacks.view appended page-by-page must equal build_kv_stacked_packed
    from scratch — bit-identical. This is the I3 production-buffer correctness gate
    (incremental O(page) append vs the O(context) per-call rebuild it replaces)."""
    from bmx.cache.packed_streaming import _PagedStacks
    from bmx.cache.triton_dequant_attention import build_kv_stacked_packed

    h_kv, blk, d, group, n = 2, 16, 8, 8, 5
    k_blocks, v_blocks = _packed_rtn_blocks(n, h_kv, blk, d, group, seed=3)

    buf = _PagedStacks(
        build_kv_stacked_packed,
        dict(h_kv=h_kv, blk_size=blk, d=d, group=group, v_group=group),
    )
    # Append one page at a time (mirrors flush-then-decode), then a multi-page jump.
    for upto in (1, 2, 3, 5):
        inc = buf.view(k_blocks[:upto], v_blocks[:upto], torch.device("cpu"))
        ref = build_kv_stacked_packed(
            k_blocks[:upto],
            v_blocks[:upto],
            max_blocks=upto,
            h_kv=h_kv,
            blk_size=blk,
            d=d,
            group=group,
            v_group=group,
            device="cpu",
        )
        assert len(inc) == len(ref) == 4
        for a, b in zip(inc, ref):
            assert torch.equal(a, b), (
                f"packed incremental != rebuild at n={upto} "
                f"(max_abs={(a.float() - b.float()).abs().max():.2e})"
            )


def test_paged_stacks_k2b_incremental_equals_rebuild():
    """_PagedStacks.view (k2b dict path) appended page-by-page must equal
    build_kv_stacked_k2b from scratch — every tensor field bit-identical and the
    non-tensor meta (rank, k_group) preserved."""
    from bmx.cache.packed_streaming import _PagedStacks
    from bmx.cache.triton_dequant_attention import build_kv_stacked_k2b

    h_kv, blk, d, rank, group, n = 2, 32, 16, 16, 16, 4
    k_blocks, v_blocks = _k2b_perhead_blocks(n, h_kv, blk, d, rank, group, seed=7)

    buf = _PagedStacks(build_kv_stacked_k2b, dict(h_kv=h_kv, blk_size=blk, d=d))
    for upto in (1, 2, 4):
        inc = buf.view(k_blocks[:upto], v_blocks[:upto], torch.device("cpu"))
        ref = build_kv_stacked_k2b(
            k_blocks[:upto],
            v_blocks[:upto],
            max_blocks=upto,
            h_kv=h_kv,
            blk_size=blk,
            d=d,
            device="cpu",
        )
        assert set(inc) == set(ref)
        for key in ref:
            if torch.is_tensor(ref[key]):
                assert torch.equal(inc[key], ref[key]), (
                    f"k2b incremental != rebuild for '{key}' at n={upto}"
                )
            else:
                assert inc[key] == ref[key], f"k2b meta '{key}' mismatch at n={upto}"
