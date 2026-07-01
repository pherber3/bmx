"""Tests for src/bmx/cache/ppl_eval.py — offline, tiny random models only.

Test idiom mirrors tests/test_cache_collect.py: build from config, no downloads.

Identity invariant note
-----------------------
The cached forward with labels=ids[:, n_prefill:] applies transformers' internal
label shift, predicting tokens n_prefill+1..N-1 (n_cont-1 tokens). The full-forward
reference matches this by masking labels[:, :n_prefill+1] = -100. Both sets of NLLs
are computed from identical logits, so the difference is purely floating-point order;
empirically ~0 for Llama and ~3e-5 absolute for GPT-2.
"""

import torch
from factories import ids as _ids
from factories import tiny_gpt2 as _tiny_gpt2
from factories import tiny_llama as _tiny_llama

from bmx.cache.ppl_eval import quantized_prefill_ppl, run_prefill
from bmx.cache.specs import CacheCodecSpec


# ---------------------------------------------------------------------------
# Shared specs
# ---------------------------------------------------------------------------

# group=4 so rtn_channel/rtn_token work with small S (n_prefill=16, S=16)
_FP16_SPEC = CacheCodecSpec(arm="fp16", bits=3, group=4)
_RTN8_SPEC = CacheCodecSpec(arm="rtn_token", bits=8, group=4)
_RTN2_SPEC = CacheCodecSpec(arm="rtn_token", bits=2, group=4)
_N_PREFILL = 16
_N_SEQ = 24  # 8 continuation tokens -> n_eval = 7 (after label shift)


# ---------------------------------------------------------------------------
# Test 1: Identity invariant (load-bearing)
# fp16 spec is a no-op -> cached ppl must equal full-forward continuation ppl
# within 1e-3 relative.
# ---------------------------------------------------------------------------


def test_identity_invariant_gpt2():
    """fp16 k_spec + v_spec: cache surgery is a no-op; ppl == full-forward within 1e-3 rel."""
    model = _tiny_gpt2()
    ids = _ids(seq=_N_SEQ)

    result = quantized_prefill_ppl(model, ids, _N_PREFILL, _FP16_SPEC, _FP16_SPEC)

    # Reference: full-forward masking labels[:, :n_prefill+1] = -100
    # so the comparable set is tokens n_prefill+1..N-1 (same as cached).
    labels_full = ids.clone()
    labels_full[:, : _N_PREFILL + 1] = -100
    with torch.no_grad():
        out_full = model(ids, labels=labels_full)
    ppl_ref = torch.exp(out_full.loss).item()

    ppl_got = result["ppl"]
    rel = abs(ppl_got - ppl_ref) / ppl_ref
    assert rel < 1e-3, (
        f"Identity invariant failed: ppl={ppl_got:.6f}, ref={ppl_ref:.6f}, rel={rel:.2e}"
    )
    assert result["n_eval"] == _N_SEQ - _N_PREFILL - 1, (
        f"n_eval={result['n_eval']}, expected {_N_SEQ - _N_PREFILL - 1}"
    )


def test_identity_invariant_llama():
    """Same identity invariant for Llama."""
    model = _tiny_llama()
    ids = _ids(seq=_N_SEQ)

    result = quantized_prefill_ppl(model, ids, _N_PREFILL, _FP16_SPEC, _FP16_SPEC)

    labels_full = ids.clone()
    labels_full[:, : _N_PREFILL + 1] = -100
    with torch.no_grad():
        out_full = model(ids, labels=labels_full)
    ppl_ref = torch.exp(out_full.loss).item()

    ppl_got = result["ppl"]
    rel = abs(ppl_got - ppl_ref) / ppl_ref
    assert rel < 1e-3, (
        f"Identity invariant (Llama) failed: ppl={ppl_got:.6f}, ref={ppl_ref:.6f}, rel={rel:.2e}"
    )


# ---------------------------------------------------------------------------
# Test 2: High bits ≈ baseline (8-bit rtn_token within 2% rel of fp16)
# ---------------------------------------------------------------------------


def test_high_bits_close_to_baseline():
    """bits=8 rtn_token K+V ppl within 2% rel of fp16 baseline."""
    model = _tiny_gpt2()
    ids = _ids(seq=_N_SEQ)

    result_fp16 = quantized_prefill_ppl(model, ids, _N_PREFILL, _FP16_SPEC, _FP16_SPEC)
    result_8bit = quantized_prefill_ppl(model, ids, _N_PREFILL, _RTN8_SPEC, _RTN8_SPEC)

    ppl_fp16 = result_fp16["ppl"]
    ppl_8bit = result_8bit["ppl"]
    rel = abs(ppl_8bit - ppl_fp16) / ppl_fp16
    assert rel < 0.02, (
        f"8-bit rtn_token not close to fp16: ppl_fp16={ppl_fp16:.4f}, "
        f"ppl_8bit={ppl_8bit:.4f}, rel={rel:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3: Garbage hurts (2-bit mean ppl > fp16 mean ppl over 5 inputs)
# ---------------------------------------------------------------------------


def test_garbage_hurts():
    """Mean bits=2 rtn_token ppl > mean fp16 ppl over 5 random inputs.

    A single tiny random model + input may not show ppl degradation (the model
    outputs near-uniform logits over a random vocabulary). Averaging over 5
    different random inputs makes the signal reliable: the expected ppl increase
    from 2-bit quantization is positive across the distribution.
    """
    model = _tiny_gpt2()

    ppls_fp16 = []
    ppls_2bit = []
    for seed in range(5):
        ids = torch.randint(
            0, 97, (1, _N_SEQ), generator=torch.Generator().manual_seed(seed)
        )
        r0 = quantized_prefill_ppl(model, ids, _N_PREFILL, _FP16_SPEC, _FP16_SPEC)
        r2 = quantized_prefill_ppl(model, ids, _N_PREFILL, _RTN2_SPEC, _RTN2_SPEC)
        ppls_fp16.append(r0["ppl"])
        ppls_2bit.append(r2["ppl"])

    mean_fp16 = sum(ppls_fp16) / len(ppls_fp16)
    mean_2bit = sum(ppls_2bit) / len(ppls_2bit)
    assert mean_2bit > mean_fp16, (
        f"2-bit mean ppl={mean_2bit:.4f} should exceed fp16 mean ppl={mean_fp16:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: pre_rope identity — fp16 arm + pre_rope=True roundtrips on Llama
# ---------------------------------------------------------------------------


def test_pre_rope_identity_llama():
    """fp16 + pre_rope=True: capture k_pre -> apply_rope -> writeback is lossless.
    ppl must be within 1e-2 rel of fp16 baseline (pins capture/rope/writeback chain).
    """
    model = _tiny_llama()
    ids = _ids(seq=_N_SEQ)

    fp16_pre_rope = CacheCodecSpec(arm="fp16", bits=3, group=4, pre_rope=True)
    result_pre = quantized_prefill_ppl(
        model, ids, _N_PREFILL, fp16_pre_rope, _FP16_SPEC
    )
    result_base = quantized_prefill_ppl(model, ids, _N_PREFILL, _FP16_SPEC, _FP16_SPEC)

    ppl_pre = result_pre["ppl"]
    ppl_base = result_base["ppl"]
    rel = abs(ppl_pre - ppl_base) / ppl_base
    assert rel < 1e-2, (
        f"pre_rope identity failed: ppl_pre={ppl_pre:.6f}, "
        f"ppl_base={ppl_base:.6f}, rel={rel:.2e}"
    )


# ---------------------------------------------------------------------------
# Test 5: bpe reporting matches quantize_cache's
# ---------------------------------------------------------------------------


def test_bpe_reporting_matches_quantize_cache():
    """bpe_k and bpe_v reported by quantized_prefill_ppl match quantize_cache."""
    from bmx.cache.codecs import quantize_cache

    model = _tiny_gpt2()
    ids = _ids(seq=_N_SEQ)

    k_spec = CacheCodecSpec(arm="rtn_token", bits=3, group=4)
    v_spec = CacheCodecSpec(arm="rtn_token", bits=8, group=4)

    result = quantized_prefill_ppl(model, ids, _N_PREFILL, k_spec, v_spec)

    # Derive expected bpe from quantize_cache directly
    # S = n_prefill, C = h_kv * d; dummy matrix for bpe only
    cfg = model.config
    h_kv = cfg.n_head  # GPT-2: h_kv == h
    d = cfg.n_embd // cfg.n_head
    S = _N_PREFILL
    C = h_kv * d

    dummy = torch.zeros(S, C)
    _, expected_bpe_k = quantize_cache("rtn_token", dummy, bits=3, group=4)
    _, expected_bpe_v = quantize_cache("rtn_token", dummy, bits=8, group=4)

    assert result["bpe_k"] == expected_bpe_k, (
        f"bpe_k={result['bpe_k']}, expected {expected_bpe_k}"
    )
    assert result["bpe_v"] == expected_bpe_v, (
        f"bpe_v={result['bpe_v']}, expected {expected_bpe_v}"
    )


# ---------------------------------------------------------------------------
# Test 6: PrefillState reuse — identical results, no cross-arm contamination
# ---------------------------------------------------------------------------


def test_prefill_state_reuse_matches_fresh():
    """One run_prefill serving multiple arms must give the exact same ppl as
    per-arm fresh prefills, and a later fp16 call through the shared state must
    equal the plain fp16 baseline (no contamination from earlier surgery)."""
    model = _tiny_llama()
    ids = _ids(seq=_N_SEQ)

    state = run_prefill(model, ids, _N_PREFILL, capture_pre_rope=False)

    # State-reused calls: a quantized arm first, then fp16 through the same state
    result_rtn_state = quantized_prefill_ppl(
        model, ids, _N_PREFILL, _RTN8_SPEC, _RTN8_SPEC, state=state
    )
    result_fp16_state = quantized_prefill_ppl(
        model, ids, _N_PREFILL, _FP16_SPEC, _FP16_SPEC, state=state
    )

    # Fresh references (no state)
    result_rtn_fresh = quantized_prefill_ppl(
        model, ids, _N_PREFILL, _RTN8_SPEC, _RTN8_SPEC
    )
    result_fp16_fresh = quantized_prefill_ppl(
        model, ids, _N_PREFILL, _FP16_SPEC, _FP16_SPEC
    )

    assert result_rtn_state["ppl"] == result_rtn_fresh["ppl"], (
        f"state-reused rtn_token ppl {result_rtn_state['ppl']} != "
        f"fresh ppl {result_rtn_fresh['ppl']}"
    )
    # fp16 evaluated AFTER the rtn surgery through the same state: any
    # contamination of the shared cache would break this equality.
    assert result_fp16_state["ppl"] == result_fp16_fresh["ppl"], (
        f"state-reused fp16 ppl {result_fp16_state['ppl']} != "
        f"plain fp16 baseline {result_fp16_fresh['ppl']}"
    )
