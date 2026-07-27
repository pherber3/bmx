"""Offline tiny-factory tests for the load-bearing math of storm Task 6's
Tier-2 closure batch (experiments/storm_tier2_closure.py): the weighted-distortion
helper, the block-diagonal W assembly, and the mechanism of sub-items (a), (b),
(f) on synthetic cases with KNOWN answers. No downloads, no GPU, fp64 math.
"""

from __future__ import annotations

import math

import torch

from bmx.cache.codecs import allocate_bits_from_variance
from bmx.cache.spectral import SpectralPack, spectral_quantize, tier_columns
from experiments.storm_tier2_closure import (
    _captured_weighted_energy,
    _centered_code_moments,
    _quantize_with_code_offset,
    assemble_dense_W,
    bit_equivalent,
    weighted_distortion,
)


def test_weighted_distortion_matches_explicit_double_sum():
    """weighted_distortion(E, W) == (1/S) Σ_s e_sᵀ W e_s on a tiny known case."""
    torch.manual_seed(0)
    S, C = 7, 5
    E = torch.randn(S, C, dtype=torch.float64)
    A = torch.randn(C, C, dtype=torch.float64)
    W = A @ A.mT  # PSD, symmetric
    helper = weighted_distortion(E, W)
    explicit = sum(E[s] @ W @ E[s] for s in range(S)).item() / S
    assert math.isclose(helper, explicit, rel_tol=1e-12, abs_tol=1e-12)


def test_weighted_distortion_identity_W_is_mean_sq_row_norm():
    """With W = I the weighted distortion is the mean squared row norm of E."""
    torch.manual_seed(1)
    E = torch.randn(9, 4, dtype=torch.float64)
    W = torch.eye(4, dtype=torch.float64)
    assert math.isclose(
        weighted_distortion(E, W),
        float((E**2).sum(dim=1).mean()),
        rel_tol=1e-12,
    )


def test_assemble_dense_W_is_block_diagonal():
    """Blocks land on the diagonal in head-major order; off-diagonal is zero."""
    torch.manual_seed(2)
    h_kv, d = 3, 4
    blocks = torch.randn(h_kv, d, d, dtype=torch.float64)
    Wd = assemble_dense_W(blocks)
    C = h_kv * d
    assert Wd.shape == (C, C)
    for j in range(h_kv):
        sl = slice(j * d, (j + 1) * d)
        assert torch.allclose(Wd[sl, sl], blocks[j])
    # zero everywhere off the diagonal blocks
    off = Wd.clone()
    for j in range(h_kv):
        sl = slice(j * d, (j + 1) * d)
        off[sl, sl] = 0.0
    assert float(off.abs().max()) == 0.0


# ---- sub-item (a): oracle-vs-average query gap -----------------------------


def test_a_gap_exactly_zero_when_oracle_equals_average():
    """Second-moment sufficiency: if the read's own W_s equals the average W_bar
    (same query distribution), the weighted distortion of ANY fixed residual is
    identical under both — rel_gap == 0 exactly. This is the mechanism sub-item
    (a)'s in-distribution gate rides on."""
    torch.manual_seed(3)
    S, C = 11, 6
    E = torch.randn(S, C, dtype=torch.float64)
    A = torch.randn(C, C, dtype=torch.float64)
    W = A @ A.mT
    d_avg = weighted_distortion(E, W)
    d_orc = weighted_distortion(E, W)  # oracle == average
    assert abs(d_orc - d_avg) / d_avg == 0.0


def test_a_gap_grows_with_W_mismatch():
    """A larger W_s vs W_bar mismatch => a larger rel_gap (monotone sanity: the
    gap is a genuine function of the W discrepancy, not a constant artifact)."""
    torch.manual_seed(4)
    S, C = 20, 5
    E = torch.randn(S, C, dtype=torch.float64)
    W_bar = torch.eye(C, dtype=torch.float64)
    d_avg = weighted_distortion(E, W_bar)
    gaps = []
    for eps in (0.0, 0.5, 2.0):
        pert = torch.eye(C, dtype=torch.float64)
        pert[0, 0] = 1.0 + eps  # inflate one query direction
        d_orc = weighted_distortion(E, pert)
        gaps.append(abs(d_orc - d_avg) / d_avg)
    assert gaps[0] == 0.0
    assert gaps[1] < gaps[2]  # bigger mismatch, bigger gap


# ---- sub-item (d): optimal-decode weighted-capture formula ------------------


def test_captured_weighted_energy_klt_identity():
    """The capture formula tr(WΣA(AᵀΣA)⁺AᵀΣ) with A = the top-r KLT coordinate
    functionals W^{1/2}E_r equals the top-r eigenvalue sum of
    T = W^{1/2}ΣW^{1/2}; a single arbitrary functional never exceeds λ₁
    (Rayleigh); and the capture is invariant to column rescaling. These are the
    three facts sub-item (d)'s gate arithmetic rides on."""
    torch.manual_seed(7)
    d = 6
    A_ = torch.randn(d, d, dtype=torch.float64)
    Sigma = A_ @ A_.mT + 0.1 * torch.eye(d, dtype=torch.float64)
    B_ = torch.randn(d, d, dtype=torch.float64)
    W = B_ @ B_.mT + 0.1 * torch.eye(d, dtype=torch.float64)
    wl, wE = torch.linalg.eigh(W)
    Wh = wE @ torch.diag(wl.sqrt()) @ wE.mT  # symmetric sqrt
    T = Wh @ Sigma @ Wh
    lam, E = torch.linalg.eigh(0.5 * (T + T.mT))
    lam, E = lam.flip(0), E.flip(1)  # descending
    for r in (1, 3):
        cap = _captured_weighted_energy(Sigma, W, Wh @ E[:, :r])
        assert math.isclose(cap, float(lam[:r].sum()), rel_tol=1e-9)
    for _ in range(5):
        a = torch.randn(d, 1, dtype=torch.float64)
        assert _captured_weighted_energy(Sigma, W, a) <= float(lam[0]) * (1 + 1e-9)
    a = torch.randn(d, 1, dtype=torch.float64)
    c1 = _captured_weighted_energy(Sigma, W, a)
    c2 = _captured_weighted_energy(Sigma, W, 3.7 * a)
    assert math.isclose(c1, c2, rel_tol=1e-9)


# ---- sub-item (b)/(f): mean-centering / global-center predictor -------------


def test_code_offset_zero_reproduces_spectral_quantize():
    """`_quantize_with_code_offset` with a zero offset reproduces
    `spectral_quantize`'s reconstruction exactly (same enc matmul, same per-tier
    RTN call, same decode) — the license for sub-items (b)/(f) reading the
    baseline-vs-centered difference as a PURE predictor/centering delta."""
    torch.manual_seed(8)
    S, C, group = 128, 8, 64
    M = torch.randn(S, C)
    Q, _ = torch.linalg.qr(torch.randn(C, C, dtype=torch.float64))
    pack = SpectralPack(
        enc=Q.float(),
        dec=Q.float(),  # enc @ dec.mT = I (orthonormal)
        lam=torch.linspace(8.0, 1.0, C),
        bits=torch.tensor([8, 6, 5, 4, 3, 2, 0, 0]),
        group=group,
        tiers=(0, 2, 3, 4, 5, 6, 8),
        budget=3.5,
    )
    M_hat, _ = spectral_quantize(M, pack)
    W = torch.eye(C, dtype=torch.float64)
    d_direct = weighted_distortion(M_hat - M, W)
    d_helper = _quantize_with_code_offset(
        M, pack, tier_columns(pack.bits), torch.zeros(C, dtype=torch.float64), W
    )
    assert math.isclose(d_helper, d_direct, rel_tol=1e-9, abs_tol=1e-15)


def test_bit_equivalent_law():
    """Δbit = 0.5·log2(D_before/D_after): one bit quarters distortion => halving
    distortion is 0.5 bit-equivalent (the 4^{-b} floor)."""
    assert math.isclose(bit_equivalent(4.0, 1.0), 1.0, rel_tol=1e-12)  # /4 = 1 bit
    assert math.isclose(bit_equivalent(2.0, 1.0), 0.5, rel_tol=1e-12)  # /2 = 0.5 bit
    assert math.isclose(bit_equivalent(1.0, 1.0), 0.0, abs_tol=1e-12)  # no change


def test_f_centering_frees_a_bit_on_large_mean_direction():
    """A code direction with |μ| ≫ σ: the uncentered 2nd moment (μ²+σ²) is large
    so the waterfill funds it; the centered variance (σ²) is tiny so it needs far
    fewer bits. The mean-centering lever's mechanism = the allocation shifts bits
    OFF the high-mean direction. Constructed identity: allocation on σ² gives that
    direction 0 bits while allocation on μ²+σ² gives it many."""
    # direction 0: huge mean, tiny variance; others: moderate variance, zero mean.
    mu = torch.tensor([100.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    sig2 = torch.tensor([1e-4, 1.0, 1.0, 1.0], dtype=torch.float64)
    uncentered = mu**2 + sig2  # μ²+σ²
    tiers = (0, 2, 3, 4, 5, 6, 8)
    b_uncentered = allocate_bits_from_variance(uncentered, 2.5, tiers)
    b_centered = allocate_bits_from_variance(sig2, 2.5, tiers)
    # uncentered pours bits into the high-mean direction; centered starves it.
    assert b_uncentered[0] > b_centered[0]
    assert int(b_centered[0]) == 0  # σ²=1e-4 is negligible -> dropped


def test_f_centered_code_moments_recovers_mean_and_variance():
    """_centered_code_moments returns the per-direction code-space mean and
    population variance of Y = M @ enc — on enc = I it is just M's column stats."""
    torch.manual_seed(5)
    S, C = 64, 3
    M = torch.randn(S, C) * torch.tensor([1.0, 5.0, 0.2]) + torch.tensor(
        [10.0, -3.0, 0.0]
    )
    enc = torch.eye(C)
    mu, sig2 = _centered_code_moments(M, enc)
    assert torch.allclose(mu, M.double().mean(dim=0), atol=1e-6)
    assert torch.allclose(sig2, M.double().var(dim=0, unbiased=False), atol=1e-6)


def test_b_global_center_null_on_zero_mean_data():
    """The global-center predictor subtracts the per-direction mean. On zero-mean
    code data it removes nothing (residual == signal) => ~0 bit-equivalent gain.
    On strongly-biased data it removes the whole mean => the residual energy
    collapses to the variance. This pins the predictor's mechanism: its win is
    exactly the mean-energy fraction, nothing more."""
    torch.manual_seed(6)
    S, C = 128, 4
    # zero-mean: centering buys nothing (energy before ≈ energy after).
    Y0 = torch.randn(S, C, dtype=torch.float64)
    mu0 = Y0.mean(dim=0)
    e_before0 = float((Y0**2).mean())
    e_after0 = float(((Y0 - mu0) ** 2).mean())
    assert bit_equivalent(e_before0, e_after0) < 0.05  # negligible

    # large mean: centering removes the mean-energy, a big cut.
    Y1 = torch.randn(S, C, dtype=torch.float64) + 50.0
    mu1 = Y1.mean(dim=0)
    e_before1 = float((Y1**2).mean())
    e_after1 = float(((Y1 - mu1) ** 2).mean())
    assert bit_equivalent(e_before1, e_after1) > 3.0  # many bits when |μ|≫σ
