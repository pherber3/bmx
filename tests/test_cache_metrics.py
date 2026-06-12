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
"""

import torch

from bmx.cache.metrics import attn_output_distortion, logit_distortion


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
