"""Tests for src/bmx/cache/metrics.py — offline, tiny tensors only.

TDD: tests written first, implementation follows.

Metric contracts
----------------
logit_distortion(K, Kq, Q) -> float
    Mean over heads of ||Q Kq^T - Q K^T||_F / ||Q K^T||_F.
    K, Kq: (h_kv, S, d); Q: (h, T, d).
    GQA: repeat_interleave K along head dim by g = h // h_kv.

attn_output_distortion(K, V, Kq, Vq, Q) -> float
    Rel Frobenius error of softmax(Q K^T / sqrt(d)) V, mean over heads.
    Same GQA expansion.  No causal mask.

logit_distortion_causal(K, Kq, Q, cos, sin, q_start) -> float
    The THIRD INSTRUMENT (K4 math review #3(b), task-4 third-instrument
    prescription): true causal per-position logit error
    Delta L_{t,s} = (R_t q_t)^T e_s, e_s = Kq_s - K_s, masked to s <= t,
    with q_t the pre-RoPE probe queries at their TRUE absolute positions
    (q_start + local index) and K/Kq already post-RoPE (apply_rope at read).
    Mean over heads of relative Frobenius error over the causal (t, s) set
    only (mirrors logit_distortion's normalization convention).
"""

import torch

from bmx.cache.metrics import (
    attn_output_distortion,
    logit_distortion,
    logit_distortion_causal,
)
from bmx.cache.rope import apply_rope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand(shape, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(*shape, generator=g)


# ---------------------------------------------------------------------------
# logit_distortion
# ---------------------------------------------------------------------------


def test_logit_distortion_identity_is_zero():
    """Kq == K => distortion is exactly 0.0."""
    h, S, d = 2, 8, 16
    K = _rand((h, S, d), seed=1)
    Q = _rand((h, 4, d), seed=2)

    result = logit_distortion(K, K, Q)

    assert result == 0.0, f"identity should give 0.0, got {result}"


def test_logit_distortion_small_scaling():
    """K scaled by (1+1e-3) gives logit_distortion ≈ 1e-3 (within 1e-5)."""
    h, S, d = 2, 16, 32
    K = _rand((h, S, d), seed=3)
    Q = _rand((h, 8, d), seed=4)
    eps = 1e-3
    Kq = K * (1.0 + eps)

    result = logit_distortion(K, Kq, Q)

    # ||Q * (Kq - K)^T||_F / ||Q K^T||_F
    # = ||Q * (eps * K)^T||_F / ||Q K^T||_F
    # = eps * ||Q K^T||_F / ||Q K^T||_F = eps
    assert abs(result - eps) < 1e-5, (
        f"expected ≈ {eps:.6f}, got {result:.8f} (diff={abs(result - eps):.2e})"
    )


def test_logit_distortion_gqa_equals_manually_expanded():
    """GQA path (h=4, h_kv=2) equals a manually repeat_interleaved computation."""
    h, h_kv, S, d = 4, 2, 12, 16
    g = h // h_kv

    K = _rand((h_kv, S, d), seed=5)
    Kq = _rand((h_kv, S, d), seed=6)
    Q = _rand((h, 6, d), seed=7)

    result_gqa = logit_distortion(K, Kq, Q)

    # Manual expansion: repeat K and Kq to h heads
    K_exp = K.repeat_interleave(g, dim=0)  # (h, S, d)
    Kq_exp = Kq.repeat_interleave(g, dim=0)  # (h, S, d)
    result_manual = logit_distortion(K_exp, Kq_exp, Q)

    assert abs(result_gqa - result_manual) < 1e-6, (
        f"GQA result {result_gqa} != manual {result_manual}"
    )


def test_logit_distortion_gqa_no_head_mismatch_when_equal():
    """h == h_kv (no GQA) still works correctly."""
    h, S, d = 3, 10, 8
    K = _rand((h, S, d), seed=8)
    Kq = _rand((h, S, d), seed=9)
    Q = _rand((h, 5, d), seed=10)

    result = logit_distortion(K, Kq, Q)

    assert isinstance(result, float)
    assert result > 0.0


def test_logit_distortion_fp16_inputs_handled():
    """fp16 inputs do not cause NaN/inf — function casts to fp32 internally."""
    h, S, d = 2, 8, 16
    K = _rand((h, S, d)).half()
    Kq = _rand((h, S, d), seed=11).half()
    Q = _rand((h, 4, d), seed=12).half()

    result = logit_distortion(K, Kq, Q)

    assert isinstance(result, float)
    assert torch.isfinite(torch.tensor(result)), f"non-finite result: {result}"


# ---------------------------------------------------------------------------
# attn_output_distortion
# ---------------------------------------------------------------------------


def test_attn_output_distortion_identity_is_zero():
    """Kq==K and Vq==V => distortion is exactly 0.0."""
    h, S, d = 2, 8, 16
    K = _rand((h, S, d), seed=13)
    V = _rand((h, S, d), seed=14)
    Q = _rand((h, 4, d), seed=15)

    result = attn_output_distortion(K, V, K, V, Q)

    assert result == 0.0, f"identity should give 0.0, got {result}"


def test_attn_output_distortion_positive_under_noise():
    """Random Kq != K gives a finite positive distortion."""
    h, S, d = 2, 12, 16
    K = _rand((h, S, d), seed=16)
    V = _rand((h, S, d), seed=17)
    Kq = _rand((h, S, d), seed=18)
    Vq = _rand((h, S, d), seed=19)
    Q = _rand((h, 6, d), seed=20)

    result = attn_output_distortion(K, V, Kq, Vq, Q)

    assert isinstance(result, float)
    assert torch.isfinite(torch.tensor(result)), f"non-finite result: {result}"
    assert result > 0.0, f"expected positive distortion, got {result}"


def test_attn_output_distortion_gqa_equals_manually_expanded():
    """GQA path (h=4, h_kv=2) equals manually-expanded computation."""
    h, h_kv, S, d = 4, 2, 10, 16
    g = h // h_kv

    K = _rand((h_kv, S, d), seed=21)
    V = _rand((h_kv, S, d), seed=22)
    Kq = _rand((h_kv, S, d), seed=23)
    Vq = _rand((h_kv, S, d), seed=24)
    Q = _rand((h, 5, d), seed=25)

    result_gqa = attn_output_distortion(K, V, Kq, Vq, Q)

    K_exp = K.repeat_interleave(g, dim=0)
    V_exp = V.repeat_interleave(g, dim=0)
    Kq_exp = Kq.repeat_interleave(g, dim=0)
    Vq_exp = Vq.repeat_interleave(g, dim=0)
    result_manual = attn_output_distortion(K_exp, V_exp, Kq_exp, Vq_exp, Q)

    assert abs(result_gqa - result_manual) < 1e-6, (
        f"GQA result {result_gqa} != manual {result_manual}"
    )


def test_attn_output_distortion_isolate_k_only():
    """Pass Vq=V to isolate only K-side distortion; must still be finite."""
    h, S, d = 2, 8, 16
    K = _rand((h, S, d), seed=26)
    V = _rand((h, S, d), seed=27)
    Kq = _rand((h, S, d), seed=28)
    Q = _rand((h, 4, d), seed=29)

    result = attn_output_distortion(K, V, Kq, V, Q)

    assert isinstance(result, float)
    assert torch.isfinite(torch.tensor(result))


def test_attn_output_distortion_fp16_inputs_handled():
    """fp16 inputs do not cause NaN/inf — function casts to fp32 internally."""
    h, S, d = 2, 8, 16
    K = _rand((h, S, d)).half()
    V = _rand((h, S, d), seed=30).half()
    Kq = _rand((h, S, d), seed=31).half()
    Vq = _rand((h, S, d), seed=32).half()
    Q = _rand((h, 4, d), seed=33).half()

    result = attn_output_distortion(K, V, Kq, Vq, Q)

    assert isinstance(result, float)
    assert torch.isfinite(torch.tensor(result))


# ---------------------------------------------------------------------------
# RoPE-composition equivalence pin (correctness pin for logit_distortion_causal)
# ---------------------------------------------------------------------------


def _rope_tables(S: int, d: int):
    """A real-shaped RoPE table: duplicated-half frequency structure."""
    freqs = torch.linspace(0.5, 1.0, d // 2)
    theta = torch.arange(S).unsqueeze(1).float() * torch.cat([freqs, freqs]).unsqueeze(
        0
    )
    return theta.cos(), theta.sin()


def test_rope_composition_absolute_equals_relative():
    """(R_t q)^T (R_s k) == (R_{t-s} q)^T k for the shipped apply_rope
    convention (rotate-half, duplicated-half cos/sin). This is the identity
    logit_distortion_causal's absolute-position form relies on being
    equivalent to the relative-offset form stated in the task brief."""
    d = 16
    S = 32
    cos, sin = _rope_tables(S, d)

    g = torch.Generator().manual_seed(42)
    q = torch.randn(1, 1, d, generator=g)
    k = torch.randn(1, 1, d, generator=g)

    for t, s in [(10, 3), (31, 0), (5, 5), (20, 19)]:
        Rt_q = apply_rope(q, cos[t : t + 1], sin[t : t + 1])
        Rs_k = apply_rope(k, cos[s : s + 1], sin[s : s + 1])
        lhs = (Rt_q * Rs_k).sum()

        m = t - s
        assert m >= 0
        Rm_q = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        rhs = (Rm_q * k).sum()

        assert torch.allclose(lhs, rhs, atol=1e-5), f"t={t} s={s}: {lhs} != {rhs}"


# ---------------------------------------------------------------------------
# logit_distortion_causal — the third instrument
# ---------------------------------------------------------------------------


def test_logit_distortion_causal_identity_is_zero():
    """Kq == K => causal distortion is exactly 0.0."""
    h_kv, S, d, T = 2, 12, 16, 4
    K = _rand((h_kv, S, d), seed=40)
    Q = _rand((h_kv, T, d), seed=41)
    cos, sin = _rope_tables(S, d)

    result = logit_distortion_causal(K, K, Q, cos, sin, q_start=S - T)

    assert result == 0.0, f"identity should give 0.0, got {result}"


def test_logit_distortion_causal_matches_manual_masked_computation():
    """Value-pin: brute-force the masked (t, s) sum with explicit per-pair
    forward RoPE on q at its true absolute position, dotted against the
    (already post-RoPE) key error e_s = Kq_s - K_s, s <= t only."""
    h_kv, S, d, T = 1, 10, 8, 4
    q_start = S - T  # true absolute positions of the probe queries
    K = _rand((h_kv, S, d), seed=42)  # already "post-RoPE" for this pin
    Kq = _rand((h_kv, S, d), seed=43)
    Q = _rand((h_kv, T, d), seed=44)  # pre-RoPE probe queries
    cos, sin = _rope_tables(S, d)

    result = logit_distortion_causal(K, Kq, Q, cos, sin, q_start=q_start)

    # Manual per-head masked computation.
    diffs = []
    refs = []
    for i in range(T):
        t = q_start + i
        q_t = Q[:, i : i + 1, :]  # (h_kv, 1, d)
        Rt_q = apply_rope(q_t, cos[t : t + 1], sin[t : t + 1])  # (h_kv, 1, d)
        for s in range(t + 1):  # causal: s <= t
            k_s = K[:, s, :]  # (h_kv, d)
            kq_s = Kq[:, s, :]
            logit_true = (Rt_q[:, 0, :] * k_s).sum(-1)  # (h_kv,)
            logit_approx = (Rt_q[:, 0, :] * kq_s).sum(-1)
            diffs.append(logit_approx - logit_true)
            refs.append(logit_true)
    diffs = torch.stack(diffs, dim=0)  # (n_pairs, h_kv)
    refs = torch.stack(refs, dim=0)
    per_head = diffs.norm(dim=0) / refs.norm(dim=0).clamp_min(1e-12)
    expected = per_head.mean().item()

    assert abs(result - expected) < 1e-4, f"{result} != {expected}"


def test_logit_distortion_causal_no_future_leakage():
    """Corrupting Kq only at a position s that is never <= any masked
    query's t must leave the causal metric unchanged (the mask must
    actually exclude s > t pairs)."""
    h_kv, S, d = 1, 8, 8
    cos, sin = _rope_tables(S, d)

    K2 = _rand((h_kv, S, d), seed=52)
    Q2 = _rand((h_kv, 1, d), seed=53)  # single probe query at position 0
    Kq2_clean = K2.clone()
    Kq2_dirty = K2.clone()
    Kq2_dirty[:, S - 1, :] += 5.0  # corrupt a position strictly after t=0

    d_clean = logit_distortion_causal(K2, Kq2_clean, Q2, cos, sin, q_start=0)
    d_dirty = logit_distortion_causal(K2, Kq2_dirty, Q2, cos, sin, q_start=0)
    assert d_clean == 0.0
    assert d_dirty == 0.0, (
        "corrupting a key strictly after the only query's position leaked "
        f"into the causal metric: {d_dirty}"
    )

    # Sanity: the metric IS sensitive to corruption within the causal window.
    Kq2_causal = K2.clone()
    Kq2_causal[:, 0, :] += 5.0  # position 0 <= t=0: in-window
    d_causal = logit_distortion_causal(K2, Kq2_causal, Q2, cos, sin, q_start=0)
    assert d_causal > 0.0


def test_logit_distortion_causal_gqa_equals_manually_expanded():
    """GQA path (h=4, h_kv=2) equals a manually repeat_interleaved computation."""
    h, h_kv, S, d, T = 4, 2, 10, 16, 3
    g = h // h_kv
    q_start = S - T

    K = _rand((h_kv, S, d), seed=60)
    Kq = _rand((h_kv, S, d), seed=61)
    Q = _rand((h, T, d), seed=62)
    cos, sin = _rope_tables(S, d)

    result_gqa = logit_distortion_causal(K, Kq, Q, cos, sin, q_start=q_start)

    K_exp = K.repeat_interleave(g, dim=0)
    Kq_exp = Kq.repeat_interleave(g, dim=0)
    result_manual = logit_distortion_causal(K_exp, Kq_exp, Q, cos, sin, q_start=q_start)

    assert abs(result_gqa - result_manual) < 1e-6, (
        f"GQA result {result_gqa} != manual {result_manual}"
    )


def test_logit_distortion_causal_fp16_inputs_handled():
    """fp16 inputs do not cause NaN/inf — function casts to fp32 internally."""
    h_kv, S, d, T = 2, 8, 16, 3
    K = _rand((h_kv, S, d)).half()
    Kq = _rand((h_kv, S, d), seed=70).half()
    Q = _rand((h_kv, T, d), seed=71).half()
    cos, sin = _rope_tables(S, d)

    result = logit_distortion_causal(K, Kq, Q, cos, sin, q_start=S - T)

    assert isinstance(result, float)
    assert torch.isfinite(torch.tensor(result))
