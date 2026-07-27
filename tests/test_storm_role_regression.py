"""Storm Task-4 — head-role regression: predictor + response math pins.

Offline tiny-factory tests (no model download, no cache files): each predictor
computation is pinned on a hand-built input of KNOWN value (a planted-rank QK
circuit, a synthetic copying OV with known positive-eigen mass, a planted
trigonometric distance peak), the per-head λ block-sum identity is pinned
against the full fit, and the regression helpers recover R² ≈ 1 on a planted
linear relation.
"""

import math

import numpy as np
import torch

from bmx.cache.spectral import assemble_whitener, fit_spectral_basis
from experiments.storm_role_regression import (
    effective_rank,
    mean_resultant_length,
    ols_r2,
    ov_positive_eigenmass,
    pack_lambda_attribution,
    per_head_weighted_spectra,
    qk_effective_rank,
    regression_report,
    rope_distance_profile,
    spearman,
)

# ---------------------------------------------------------------------------
# Predictor (i): QK-circuit effective rank
# ---------------------------------------------------------------------------


def test_effective_rank_flat_and_peaked():
    assert abs(effective_rank(torch.ones(5)) - 5.0) < 1e-10
    assert abs(effective_rank(torch.tensor([1.0, 0.0, 0.0])) - 1.0) < 1e-10


def test_qk_effective_rank_planted_rank():
    """A hand-built W_QK of known rank: r orthonormal aligned columns give a
    product with exactly r unit singular values ⇒ erank == r, through the
    d_head bottleneck (d_model=16 > d_head=8)."""
    d_model, d = 16, 8
    for r in (1, 4, 8):
        A = torch.zeros(d_model, d, dtype=torch.float64)
        for i in range(r):
            A[i, i] = 1.0
        assert abs(qk_effective_rank(A, A) - r) < 1e-8, f"rank {r}"


def test_qk_effective_rank_scale_invariant():
    """erank depends on the singular-value SHAPE, not the overall scale."""
    g = torch.Generator().manual_seed(0)
    A = torch.randn(16, 8, generator=g)
    B = torch.randn(16, 8, generator=g)
    assert abs(qk_effective_rank(A, B) - qk_effective_rank(3.0 * A, B)) < 1e-6


# ---------------------------------------------------------------------------
# Predictor (iv): OV copying score
# ---------------------------------------------------------------------------


def test_ov_positive_eigenmass_planted():
    """Synthetic copying OV with known positive-eigen mass: W_O W_V is forced
    to diag(2, -1, 1, 0) ⇒ mass = (2+1)/(2+1+1) = 0.75. A symmetric-PSD
    construction gives exactly 1.0 (pure copying)."""
    d_model, d = 16, 4
    WO = torch.zeros(d, d_model)
    WO[:, :d] = torch.eye(d)
    WV = torch.zeros(d_model, d)
    WV[:d, :d] = torch.diag(torch.tensor([2.0, -1.0, 1.0, 0.0]))
    assert abs(ov_positive_eigenmass(WV, WO) - 0.75) < 1e-10

    g = torch.Generator().manual_seed(1)
    B = torch.randn(d_model, d, generator=g)
    # W_O = W_V^T => circuit spectrum = eig(B^T B) all >= 0 => mass 1.0.
    assert abs(ov_positive_eigenmass(B, B.mT) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Predictor (ii): concentration
# ---------------------------------------------------------------------------


def test_mean_resultant_length_extremes():
    v = torch.tensor([[1.0, 2.0, -1.0]])
    aligned = torch.cat([v * s for s in (1.0, 2.0, 5.0)], dim=0)  # same direction
    assert abs(mean_resultant_length(aligned) - 1.0) < 1e-10
    antipodal = torch.cat([v, -v], dim=0)
    assert mean_resultant_length(antipodal) < 1e-10


# ---------------------------------------------------------------------------
# Predictor (iii): trigonometric distance preference
# ---------------------------------------------------------------------------


def test_rope_distance_profile_planted_peak():
    """Planted phase: q̄ = e0 and k̄ = R_{m0} e0 give a(m) = cos((m-m0)·θ0)
    ⇒ argmax at exactly m0 (θ0·S < 2π + margin keeps the peak unique)."""
    S, m0, theta0, theta1 = 40, 17, 0.2, 0.05  # head dim d = 4
    m = torch.arange(S, dtype=torch.float64)
    cos = torch.stack(
        [
            (m * theta0).cos(),
            (m * theta1).cos(),
            (m * theta0).cos(),
            (m * theta1).cos(),
        ],
        dim=1,
    )
    sin = torch.stack(
        [
            (m * theta0).sin(),
            (m * theta1).sin(),
            (m * theta0).sin(),
            (m * theta1).sin(),
        ],
        dim=1,
    )
    q_bar = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    k_bar = torch.tensor(
        [math.cos(m0 * theta0), 0.0, math.sin(m0 * theta0), 0.0],
        dtype=torch.float64,
    )
    a = rope_distance_profile(q_bar, k_bar, cos, sin)
    assert a.shape == (S,)
    assert int(torch.argmax(a)) == m0
    assert abs(float(a[m0]) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Response: per-head λ block-sum identity + pack-tensor attribution
# ---------------------------------------------------------------------------


def test_per_head_spectra_block_sum_and_attribution():
    """The load-bearing granularity claim: with W block-diagonal per head,
    (a) Σ_heads (per-head λ totals) == the FULL fit's Σλ (block-trace
    identity), and (b) attributing the full fit's λ through the orthonormal
    eigenvector block energies reproduces the per-head totals exactly (up to
    the pack's fp32 enc/dec storage)."""
    torch.manual_seed(0)
    h_kv, d, S = 2, 8, 256
    C = h_kv * d
    A = torch.randn(h_kv, d, d, dtype=torch.float64)
    W_blocks = A @ A.mT + torch.eye(d, dtype=torch.float64)
    M = torch.randn(S, C)

    lam_heads = per_head_weighted_spectra(M, W_blocks, ridge=1e-3)  # (h_kv, d)
    assert lam_heads.shape == (h_kv, d)
    # rows descending
    assert (lam_heads[:, :-1] >= lam_heads[:, 1:] - 1e-12).all()

    Wh, Wh_inv = assemble_whitener(W_blocks, ridge=1e-3)
    basis = fit_spectral_basis(M, Wh, Wh_inv)
    full_sum = float(basis.lam64.sum())
    head_sum = float(lam_heads.sum())
    assert abs(full_sum - head_sum) / full_sum < 1e-8

    bits = torch.ones(C, dtype=torch.int64)  # all directions used
    lam_soft, bits_soft, bdi = pack_lambda_attribution(
        basis.lam, bits, basis.dec, Wh, h_kv
    )
    per_head_totals = lam_heads.sum(dim=1)
    rel = ((lam_soft - per_head_totals).abs() / per_head_totals).max()
    assert float(rel) < 1e-4  # fp32 dec/lam storage round-off only
    # attribution fractions partition each direction: bits mass is conserved
    assert abs(float(bits_soft.sum()) - float(bits.sum())) < 1e-6
    assert 1.0 / h_kv - 1e-9 <= bdi <= 1.0


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------


def test_regression_planted_linear_recovers_r2_one():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5 * X[:, 2]
    names = ["x0", "x1", "x2"]
    rep = regression_report(X, y, names)
    assert rep["ols_r2"] > 1.0 - 1e-10
    assert rep["ols_adj_r2"] > 1.0 - 1e-10
    assert rep["rank_ols_r2"] > 0.9  # rank transform loses a little linearity
    assert rep["dominant_by_univariate_r2"] == "x0"
    assert rep["dominant_by_abs_coef"] == "x0"
    # standardized coefficient signs match the planted relation
    assert rep["coef_std"]["x0"] > 0 > rep["coef_std"]["x1"]


def test_regression_pure_noise_r2_near_zero():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 3))
    y = rng.standard_normal(200)
    rep = regression_report(X, y, ["a", "b", "c"])
    assert rep["ols_r2"] < 0.1
    assert abs(rep["ols_adj_r2"]) < 0.1


def test_ols_r2_matches_pearson_sq_univariate():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(100)
    y = 0.7 * x + rng.standard_normal(100)
    r = np.corrcoef(x, y)[0, 1]
    assert abs(ols_r2(x[:, None], y, ["x"])["r2"] - r**2) < 1e-10


def test_spearman_monotone_and_antitone():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(50)
    assert abs(spearman(x, x**3) - 1.0) < 1e-12  # monotone map: ranks equal
    assert abs(spearman(x, -(x**3)) + 1.0) < 1e-12
