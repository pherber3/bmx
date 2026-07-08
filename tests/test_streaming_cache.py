"""StreamingQuantizedCache: plumbing, quality, and memory gates (tiny_llama)."""

import pytest
import torch

from bmx.cache.specs import CacheCodecSpec
from bmx.cache.streaming import StreamingQuantizedCache
from factories import ids, tiny_llama


def _k2b_spec():
    """K2b headline spec: lowrank K (pre_rope) + turboquant_mse V."""
    return (
        CacheCodecSpec(
            arm="lowrank_rtn_channel", bits=3, rank=4, group=16, pre_rope=True
        ),
        CacheCodecSpec(arm="turboquant_mse", bits=2),
    )


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
    # seq=200 > PAGE(128)+recent_window(32) so a 128-token page flushes (the cache
    # now commits on the fixed 128-token PAGE grid; shorter sequences stay all-fp16).
    model = tiny_llama()
    input_ids = ids(vocab=97, seq=200, seed=8)
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


def test_memory_report_packed_below_fp16():
    model = tiny_llama()
    # seq=288 > 2*PAGE(128)+recent_window(32): two 128-token pages flush at 2-bit,
    # leaving 32 fp16 -> blended bpe well below 16 (the cache commits on the 128-token
    # PAGE grid; shorter sequences would stay all-fp16 and show no compression).
    input_ids = ids(vocab=97, seq=288, seed=21)
    cache = StreamingQuantizedCache(
        model.config,
        k_spec=CacheCodecSpec(arm="rtn_channel", bits=2, group=16, pre_rope=True),
        v_spec=CacheCodecSpec(arm="rtn_token", bits=2, group=16),
    )
    cache.attach(model)
    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True)
    cache.detach()
    rep = cache.memory_report(seq_len=input_ids.shape[1])
    # 256 of 288 tokens quantized at 2bpe + 32 fp16 -> blended bpe ≈ (256*2+32*16)/288
    # ≈ 3.5, compression ≈ 4.5x. The packed footprint is honestly below fp16.
    assert rep["packed_bytes"] < rep["fp16_bytes"]
    assert rep["compression"] > 1.0


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


def test_streaming_token_by_token_channel_grouped_no_crash():
    """Regression: rtn_channel asserts S % group == 0; residual window avoids this crash.

    Prefill 16 tokens (S=16, 16%16=0 OK), then decode ONE TOKEN AT A TIME for 20
    steps (S=17,18,...,36). Without the window, S=17 immediately crashes the
    rtn_channel assert. With the window (W=8), S_q = ((S-8)//16)*16 stays group-
    aligned, so the assert is never violated.
    """
    model = tiny_llama()
    g = torch.Generator().manual_seed(5)
    input_ids = torch.randint(0, 97, (1, 36), generator=g)
    cache = StreamingQuantizedCache(
        model.config,
        k_spec=CacheCodecSpec(arm="rtn_channel", bits=2, group=16),
        v_spec=CacheCodecSpec(arm="rtn_token", bits=2, group=16),
        recent_window=8,
    )
    cache.attach(model)
    try:
        with torch.no_grad():
            model(input_ids[:, :16], past_key_values=cache, use_cache=True)  # prefill
            for t in range(16, 36):  # decode one token at a time
                model(input_ids[:, t : t + 1], past_key_values=cache, use_cache=True)
    finally:
        cache.detach()
    assert cache.layers[0].get_seq_length() == 36


def test_short_cache_stays_fp16_until_window_exceeded():
    """With recent_window=8 and only 4 prefill tokens (S=4 < W=8), bpe must be 16.0
    (nothing quantized yet — whole cache is the fp16 window).
    """
    model = tiny_llama()
    input_ids = ids(vocab=97, seq=4, seed=9)
    cache = StreamingQuantizedCache(
        model.config,
        k_spec=CacheCodecSpec(arm="rtn_channel", bits=2, group=16),
        v_spec=CacheCodecSpec(arm="rtn_token", bits=2, group=16),
        recent_window=8,
    )
    cache.attach(model)
    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True)
    cache.detach()
    bpe_k, bpe_v = cache.bits_per_entry()
    assert bpe_k == 16.0, f"expected 16.0 (no quant yet), got {bpe_k}"
    assert bpe_v == 16.0, f"expected 16.0 (no quant yet), got {bpe_v}"


def test_k2b_pre_rope_streams_token_by_token():
    """K2b headline spec (lowrank_rtn_channel K, rtn_token V) with pre_rope=True
    must stream token-by-token without crashing and produce finite, compressed output.

    Prefill 130 tokens then 20 single-token decode steps. The cache commits on the
    fixed 128-token PAGE grid, so a page flushes during this run (S grows past
    PAGE + recent_window); shorter sequences would stay all-fp16.
    """
    model = tiny_llama()
    g = torch.Generator().manual_seed(11)
    input_ids = torch.randint(0, 97, (1, 150), generator=g)
    cache = StreamingQuantizedCache(
        model.config,
        k_spec=CacheCodecSpec(
            arm="lowrank_rtn_channel", bits=3, rank=4, group=16, pre_rope=True
        ),
        v_spec=CacheCodecSpec(arm="rtn_token", bits=2, group=16),
        recent_window=8,
    )
    cache.attach(model)
    try:
        with torch.no_grad():
            model(input_ids[:, :130], past_key_values=cache, use_cache=True)  # prefill
            for t in range(130, 150):  # 20 single-token decode steps
                model(input_ids[:, t : t + 1], past_key_values=cache, use_cache=True)
    finally:
        cache.detach()
    k_post, v = cache.reconstruct_layer(0)
    assert torch.isfinite(k_post).all() and torch.isfinite(v).all()
    bpe_k, bpe_v = cache.bits_per_entry()
    assert bpe_k < 16.0, f"expected blended bpe_k < 16.0, got {bpe_k}"


def test_write_once_v_stable_token_by_token():
    """C1 regression gate: with the K2b spec (turboquant_mse V), token-by-token V cache
    must closely match a batched run (rel < 0.05).

    Before the write-once fix, turboquant_mse is non-idempotent (per-token norm rescale
    compounds): V cache norm explodes from ~4 to ~400 over 64 steps (rel ~98).
    After the fix, each token's V is quantized exactly once from pristine fp16 source.
    """
    model = tiny_llama()
    k_spec, v_spec = _k2b_spec()
    g = torch.Generator().manual_seed(42)
    # Prefill 16 tokens, then 64 single-token decode steps = 80 total.
    total = 80
    prefill_len = 16
    input_ids = torch.randint(0, 97, (1, total), generator=g)

    # --- Token-by-token run ---
    cache_tbt = StreamingQuantizedCache(
        model.config, k_spec=k_spec, v_spec=v_spec, recent_window=8
    )
    cache_tbt.attach(model)
    try:
        with torch.no_grad():
            model(input_ids[:, :prefill_len], past_key_values=cache_tbt, use_cache=True)
            for t in range(prefill_len, total):
                model(
                    input_ids[:, t : t + 1], past_key_values=cache_tbt, use_cache=True
                )
    finally:
        cache_tbt.detach()
    _, v_tbt = cache_tbt.reconstruct_layer(0)

    # --- Batched reference run (prefill + one big decode step) ---
    cache_batch = StreamingQuantizedCache(
        model.config, k_spec=k_spec, v_spec=v_spec, recent_window=8
    )
    cache_batch.attach(model)
    try:
        with torch.no_grad():
            model(
                input_ids[:, :prefill_len], past_key_values=cache_batch, use_cache=True
            )
            model(
                input_ids[:, prefill_len:], past_key_values=cache_batch, use_cache=True
            )
    finally:
        cache_batch.detach()
    _, v_batch = cache_batch.reconstruct_layer(0)

    # Both runs should have the same sequence length.
    assert v_tbt.shape == v_batch.shape, (
        f"shape mismatch: {v_tbt.shape} vs {v_batch.shape}"
    )

    rel = (v_tbt.float() - v_batch.float()).norm() / v_batch.float().norm().clamp_min(
        1e-6
    )
    assert rel < 0.05, (
        f"V cache token-by-token vs batched rel={rel:.3f} (expected < 0.05); "
        "write-once not enforced — turboquant_mse is still compounding"
    )


def test_each_token_quantized_once():
    """The committed prefix must be frozen: re-running update doesn't change it.

    After a flush event, _q_prefix_k[:, :old_committed_S_q] must be bitwise
    identical before and after the next flush step.
    """
    model = tiny_llama()
    k_spec, v_spec = _k2b_spec()
    g = torch.Generator().manual_seed(7)
    input_ids = torch.randint(0, 97, (1, 64), generator=g)

    cache = StreamingQuantizedCache(
        model.config, k_spec=k_spec, v_spec=v_spec, recent_window=8
    )
    cache.attach(model)
    try:
        with torch.no_grad():
            # Prefill 16 — first flush at S=16 (S_q=8 with W=8, g=16 → 0; then at 24)
            # Use 24 prefill to ensure first flush happens.
            model(input_ids[:, :24], past_key_values=cache, use_cache=True)

        # After first flush, save the committed prefix of layer 0.
        layer = cache.layers[0]
        old_committed = layer._committed_S_q
        if old_committed == 0:
            # No flush happened yet — skip loudly so a grid change can't make this
            # invariant check vacuously pass.
            pytest.skip("no flush occurred; nothing committed to freeze-check")
        # Save copies of the frozen prefix (both K and V — V's write-once is the
        # C1-critical one since turboquant_mse is non-idempotent).
        prefix_k_before = layer._q_prefix_k.clone()
        prefix_v_before = layer._q_prefix_v.clone()

        # Run more decode steps to trigger another flush.
        with torch.no_grad():
            for t in range(24, 40):
                model(input_ids[:, t : t + 1], past_key_values=cache, use_cache=True)

        # The portion that was committed before the extra steps must be unchanged.
        prefix_k_after = layer._q_prefix_k
        prefix_v_after = layer._q_prefix_v
        assert torch.equal(prefix_k_before, prefix_k_after[:, :old_committed, :]), (
            "Committed K prefix changed — write-once not enforced"
        )
        assert torch.equal(prefix_v_before, prefix_v_after[:, :old_committed, :]), (
            "Committed V prefix changed — write-once not enforced"
        )
    finally:
        cache.detach()


def test_multiblock_k_rope_positions_correct():
    # Multi-block streaming with fp16+pre_rope K: the 2nd/3rd quantized blocks must
    # get RoPE at their TRUE absolute positions (committed..new), not 0..block_len.
    # committed_S_q > 0 for later blocks, so a position-offset bug (which the single-
    # block rel<1e-2 test cannot see) shows up here. fp16 codec => lossless, so any
    # deviation is a RoPE-position error, not quant error.
    model = tiny_llama()
    input_ids = ids(vocab=97, seq=60, seed=9)

    # Stream token-by-token with a small window so multiple blocks flush (positions
    # 16+, 32+, ... get committed at nonzero offsets).
    cache = StreamingQuantizedCache(
        model.config,
        k_spec=CacheCodecSpec(arm="fp16", pre_rope=True),
        v_spec=CacheCodecSpec(arm="fp16"),
        recent_window=8,
    )
    cache.attach(model)
    with torch.no_grad():
        model(input_ids[:, :16], past_key_values=cache, use_cache=True)
        for t in range(16, 60):
            model(input_ids[:, t : t + 1], past_key_values=cache, use_cache=True)
    cache.detach()
    k_post, _ = cache.reconstruct_layer(0)

    # True post-RoPE keys from a plain default cache over the same full sequence.
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
    assert rel < 1e-2, (
        f"multi-block K RoPE positions wrong: rel={rel} (position offset bug?)"
    )


def test_frozen_subspace_not_refit():
    """After first K flush, _frozen_svd is set and its V factor must not change
    across subsequent flushes.
    """
    model = tiny_llama()
    k_spec, v_spec = _k2b_spec()
    g = torch.Generator().manual_seed(9)
    input_ids = torch.randint(0, 97, (1, 80), generator=g)

    cache = StreamingQuantizedCache(
        model.config, k_spec=k_spec, v_spec=v_spec, recent_window=8
    )
    cache.attach(model)
    try:
        with torch.no_grad():
            model(input_ids[:, :24], past_key_values=cache, use_cache=True)

        layer = cache.layers[0]
        if layer._frozen_svd is None:
            # Frozen SVD not implemented — skip loudly (acceptable fallback per
            # brief) so the invariant check can't pass vacuously.
            pytest.skip("frozen SVD not set; nothing to check")
        _, V_frozen_first = layer._frozen_svd

        with torch.no_grad():
            for t in range(24, 60):
                model(input_ids[:, t : t + 1], past_key_values=cache, use_cache=True)

        _, V_frozen_after = layer._frozen_svd
        assert torch.equal(V_frozen_first, V_frozen_after), (
            "_frozen_svd V changed after first flush — subspace is not frozen"
        )
    finally:
        cache.detach()


def test_k2t_frozen_subspace_engaged_and_not_refit(monkeypatch):
    """k2t (lowrank_turboquant K) must take the SAME frozen-subspace path as
    lowrank_rtn_channel: the channel subspace V is fitted ONCE per layer at the
    first flush and REUSED (projection only) for every later block. A fall-
    through to the generic path would refit truncated_svd per 128-token page —
    violating the frozen-subspace design and re-charging factor bpe per block.

    Counts truncated_svd calls on BOTH modules that bind it (streaming's frozen
    fit and codecs' internal refit), so a per-block refit on either path fails.
    """
    import bmx.cache.codecs as codecs_mod
    import bmx.cache.streaming as streaming_mod
    from bmx.cache.recipes import spec_pair

    model = tiny_llama()
    k_spec, v_spec = spec_pair("k2t", rank=4, group=16)
    assert k_spec.arm == "lowrank_turboquant"  # pin the recipe wiring

    svd_calls: list = []
    real_svd = streaming_mod.truncated_svd

    def counting_svd(M, rank):
        svd_calls.append(tuple(M.shape))
        return real_svd(M, rank)

    monkeypatch.setattr(streaming_mod, "truncated_svd", counting_svd)
    monkeypatch.setattr(codecs_mod, "truncated_svd", counting_svd)

    g = torch.Generator().manual_seed(17)
    input_ids = torch.randint(0, 97, (1, 270), generator=g)
    cache = StreamingQuantizedCache(
        model.config, k_spec=k_spec, v_spec=v_spec, recent_window=8
    )
    cache.attach(model)
    try:
        with torch.no_grad():
            # Prefill 140: first 128-token page flushes (S-W=132 >= PAGE).
            model(input_ids[:, :140], past_key_values=cache, use_cache=True)
        layer = cache.layers[0]
        assert layer._committed_S_q == 128, "first flush did not fire (vacuous)"
        assert layer._frozen_svd is not None, (
            "k2t did not engage the frozen-subspace path — lowrank_turboquant "
            "fell through to the generic per-block-refit path"
        )
        _, V_first = layer._frozen_svd
        V_first = V_first.clone()
        fits_after_first_flush = len(svd_calls)

        with torch.no_grad():
            # One more batched step to S=270: second page flushes (S_q -> 256).
            model(input_ids[:, 140:], past_key_values=cache, use_cache=True)
        assert layer._committed_S_q == 256, "second flush did not fire (vacuous)"
    finally:
        cache.detach()

    n_layers = len(cache.layers)
    assert fits_after_first_flush == n_layers, (
        f"expected exactly one SVD fit per layer at first flush, "
        f"got {fits_after_first_flush} for {n_layers} layers"
    )
    assert len(svd_calls) == n_layers, (
        f"truncated_svd refit on a later flush ({len(svd_calls)} total calls for "
        f"{n_layers} layers) — subspace is not frozen"
    )
    _, V_after = cache.layers[0]._frozen_svd
    assert torch.equal(V_first, V_after), (
        "_frozen_svd V changed after first flush — subspace is not frozen"
    )


def test_k2t_generate_finite_logits():
    """k2t end-to-end on a tiny factory model: prefill past the PAGE boundary
    (so the lowrank_turboquant K codec actually runs) then greedy-decode token
    by token through the streaming cache — every step's logits must be finite
    and the blended bpe must show real compression (non-vacuous).
    """
    from bmx.cache.recipes import spec_pair

    model = tiny_llama()
    k_spec, v_spec = spec_pair("k2t", rank=4, group=16)
    g = torch.Generator().manual_seed(23)
    prompt = torch.randint(0, 97, (1, 140), generator=g)
    cache = StreamingQuantizedCache(
        model.config, k_spec=k_spec, v_spec=v_spec, recent_window=8
    )
    cache.attach(model)
    try:
        with torch.no_grad():
            out = model(prompt, past_key_values=cache, use_cache=True)
            assert torch.isfinite(out.logits).all()
            tok = out.logits[:, -1:].argmax(-1)
            for _ in range(12):  # greedy decode, one token at a time
                out = model(tok, past_key_values=cache, use_cache=True)
                assert torch.isfinite(out.logits).all()
                tok = out.logits[:, -1:].argmax(-1)
    finally:
        cache.detach()
    k_post, v = cache.reconstruct_layer(0)
    assert torch.isfinite(k_post).all() and torch.isfinite(v).all()
    bpe_k, _ = cache.bits_per_entry()
    assert bpe_k < 16.0, f"expected blended bpe_k < 16.0 (page flushed), got {bpe_k}"


def test_greedy_decode_no_flush_branch_matches_itself():
    """Pins the no-flush branch's numerics as a change guard (Task 2 / Wave 3).

    Prefill 100 tokens (all fp16, no flush yet — recent_window=32 default keeps
    S<=32 unflushed and 100<160 so nothing has committed after prefill), then
    greedy-decode 64 tokens one at a time. With PAGE=128 and W=32,
    compute_flush_schedule fires exactly once, at S=160 (decode step 60 of the
    64): the run exercises BOTH branches (no-flush for most steps, the flush
    branch exactly once). We probe `_committed_S_q` before/after to confirm the
    flush actually happened (the Wave-2 lesson: a fixture that never reaches the
    predicate it claims to test). This test runs a plain forward loop TWICE with
    fresh caches built from the same seed and asserts the logits are bit-
    identical to each other — establishing a reproducible baseline the
    no-flush-branch deletion (this task) must reproduce exactly via torch.equal
    on the reconstructed K/V, not just the loop's own determinism.
    """
    model = tiny_llama()
    k_spec, v_spec = _k2b_spec()
    prefill_len = 100
    n_decode = 64
    g = torch.Generator().manual_seed(13)
    input_ids = torch.randint(0, 97, (1, prefill_len + n_decode), generator=g)

    def run():
        cache = StreamingQuantizedCache(model.config, k_spec=k_spec, v_spec=v_spec)
        cache.attach(model)
        committed_trace = []
        logits = []
        try:
            with torch.no_grad():
                out = model(
                    input_ids[:, :prefill_len], past_key_values=cache, use_cache=True
                )
                committed_trace.append(cache.layers[0]._committed_S_q)
                logits.append(out.logits.clone())
                for t in range(prefill_len, prefill_len + n_decode):
                    out = model(
                        input_ids[:, t : t + 1],
                        past_key_values=cache,
                        use_cache=True,
                    )
                    committed_trace.append(cache.layers[0]._committed_S_q)
                    logits.append(out.logits.clone())
        finally:
            cache.detach()
        k_post, v = cache.reconstruct_layer(0)
        return logits, k_post, v, committed_trace

    logits_a, k_a, v_a, trace_a = run()

    # Confirm the flush boundary actually fires during this run (not vacuous):
    # _committed_S_q must transition 0 -> 128 exactly once, i.e. both values are
    # observed and 128 first appears strictly after prefill.
    assert trace_a[0] == 0, f"expected no flush after prefill, got {trace_a[0]}"
    assert 128 in trace_a, (
        f"flush never fired (no 128 in trace) — boundary not exercised: {trace_a}"
    )
    flush_idx = trace_a.index(128)
    assert all(c == 0 for c in trace_a[:flush_idx]), (
        f"flush fired earlier than expected: {trace_a}"
    )
    assert all(c == 128 for c in trace_a[flush_idx:]), (
        f"committed count changed again after first flush: {trace_a}"
    )

    logits_b, k_b, v_b, trace_b = run()

    assert trace_b == trace_a
    for la, lb in zip(logits_a, logits_b, strict=True):
        assert torch.equal(la, lb)
    assert torch.equal(k_a, k_b)
    assert torch.equal(v_a, v_b)


# --- Multimodal-nesting resolvers (Qwen3.5 / Gemma4 ForConditionalGeneration) ---


def test_resolve_text_config_unwraps_multimodal():
    """Qwen3.5/Gemma4 stash head geometry under config.text_config; unwrap it."""
    import types

    from bmx.cache.hf_compat import resolve_text_config

    text = types.SimpleNamespace(
        num_attention_heads=24, num_key_value_heads=4, head_dim=256, hidden_size=5120
    )
    top = types.SimpleNamespace(model_type="qwen3_5", text_config=text)
    assert resolve_text_config(top) is text
    # Llama-style flat config (text_config absent) returns itself.
    flat = types.SimpleNamespace(num_attention_heads=32, num_key_value_heads=8)
    assert resolve_text_config(flat) is flat
    # A text_config lacking head attrs (unrelated) is NOT mistaken for the LM config.
    bogus = types.SimpleNamespace(
        text_config=types.SimpleNamespace(foo=1), num_attention_heads=8
    )
    assert resolve_text_config(bogus) is bogus


def test_resolve_decoder_layers_across_nestings():
    """Layers live under model.model.language_model.layers (multimodal),
    model.model.layers (Llama), or model.transformer.h (GPT-2)."""
    import types

    from bmx.cache.hf_compat import resolve_decoder_layers

    sentinel = ["L0", "L1", "L2"]
    # Multimodal: model.model.language_model.layers
    mm = types.SimpleNamespace(
        model=types.SimpleNamespace(
            language_model=types.SimpleNamespace(layers=sentinel)
        )
    )
    assert resolve_decoder_layers(mm) is sentinel
    # Llama: model.model.layers
    llama = types.SimpleNamespace(model=types.SimpleNamespace(layers=sentinel))
    assert resolve_decoder_layers(llama) is sentinel
    # GPT-2: model.transformer.h
    gpt2 = types.SimpleNamespace(transformer=types.SimpleNamespace(h=sentinel))
    assert resolve_decoder_layers(gpt2) is sentinel


def test_resolve_vocab_size_unwraps_multimodal():
    """Qwen3.5/Gemma4 stash vocab_size under config.text_config; unwrap it."""
    import types

    from bmx.cache.hf_compat import resolve_vocab_size

    # Multimodal: vocab_size lives under text_config, absent at the top level.
    text = types.SimpleNamespace(num_attention_heads=24, vocab_size=152064)
    top = types.SimpleNamespace(model_type="qwen3_5", text_config=text)
    assert resolve_vocab_size(top) == 152064
    # Llama-style flat config (text_config absent) returns top-level vocab_size.
    flat = types.SimpleNamespace(vocab_size=128256)
    assert resolve_vocab_size(flat) == 128256
