import pytest
import torch

from bmx.decomp.lrs import (
    fit_lrs,
    hard_threshold,
    spikiness_ratio,
    topk_sparse,
    two_step_lrs,
)
from bmx.quant.hadamard import orthogonalize


def _planted(m=64, p=48, r=3, k=20, seed=0, spike=1.0, scale=0.01):
    """L: incoherent rank-r with entries ~scale; S: k spikes of magnitude `spike`.

    Separation spike/scale ~ 100x makes support identification immediate and
    alternation convergence geometric.
    """
    g = torch.Generator().manual_seed(seed)
    U = orthogonalize(torch.randn(m, r, generator=g, dtype=torch.float64))
    V = orthogonalize(torch.randn(p, r, generator=g, dtype=torch.float64))
    s = torch.linspace(1.0, 0.5, r, dtype=torch.float64) * scale * (m * p) ** 0.5
    L = (U * s) @ V.mT
    idx = torch.randperm(m * p, generator=g)[:k]
    S = torch.zeros(m * p, dtype=torch.float64)
    signs = (torch.rand(k, generator=g, dtype=torch.float64) > 0.5).double() * 2 - 1
    S[idx] = spike * signs
    return L, S.view(m, p)


def test_hard_threshold_is_hard_not_soft():
    # Eq. 11.58: T_nu keeps the VALUE; soft-thresholding would shrink by nu.
    x = torch.tensor([0.5, -2.0, 1.1, 1.0])
    out = hard_threshold(x, 1.0)
    assert torch.equal(out, torch.tensor([0.0, -2.0, 1.1, 0.0]))


def test_topk_sparse_exact_count_and_values():
    g = torch.Generator().manual_seed(3)
    W = torch.randn(10, 7, generator=g, dtype=torch.float64)
    S = topk_sparse(W, 5)
    nz = S != 0
    assert nz.sum().item() == 5
    assert torch.equal(S[nz], W[nz])  # kept verbatim
    # agrees with hard_threshold at any nu between the 5th and 6th magnitude
    mags = W.abs().flatten().sort(descending=True).values
    nu = (mags[4] + mags[5]).item() / 2
    assert torch.equal(S, hard_threshold(W, nu))
    assert torch.equal(topk_sparse(W, 0), torch.zeros_like(W))


def test_planted_recovery():
    L, S = _planted()
    W = L + S
    Us, V, S_hat = two_step_lrs(W, r=3, k=20, n_alternations=10)
    # exact support recovery
    assert torch.equal(S_hat != 0, S != 0)
    rec = Us @ V.mT + S_hat
    assert (rec - W).norm() / W.norm() < 1e-5
    assert (Us @ V.mT - L).norm() / L.norm() < 1e-4


def test_fit_lrs_edges_and_param_count():
    L, S = _planted(m=16, p=12, r=2, k=4)
    W = L + S
    full = fit_lrs(W, rank=(12, 0))
    assert full.relative_error(W) < 1e-10  # full rank, no sparsity: exact
    fit = fit_lrs(W, rank=(2, 4))
    assert fit.param_count() == 2 * (16 + 12) + 4
    zero = fit_lrs(W, rank=(0, 0))
    assert abs(zero.relative_error(W) - 1.0) < 1e-12


def test_spikiness_ratio():
    flat = torch.ones(8, 8)
    assert abs(spikiness_ratio(flat) - 1.0) < 1e-6
    spiky = torch.zeros(8, 8)
    spiky[0, 0] = 1.0
    assert abs(spikiness_ratio(spiky) - 8.0) < 1e-6  # max*sqrt(64)/fro = 8


def test_registered_and_rejects_3d():
    from bmx.decomp.base import available_methods

    assert "lrs" in available_methods()
    with pytest.raises(AssertionError):
        fit_lrs(torch.zeros(4, 4, 4), rank=(2, 2))


def test_two_step_rejects_out_of_range_budgets():
    W = torch.randn(
        6, 4, generator=torch.Generator().manual_seed(0), dtype=torch.float64
    )
    with pytest.raises(AssertionError):
        two_step_lrs(W, r=5, k=0)  # r > min(m,p)
    with pytest.raises(AssertionError):
        two_step_lrs(W, r=2, k=25)  # k > numel
    with pytest.raises(AssertionError):
        spikiness_ratio(torch.zeros(3, 3))
