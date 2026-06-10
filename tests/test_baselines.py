import torch

from bmx.decomp.baselines import fit_cp, fit_shared_tucker, fit_slice_svd, fit_tucker


def _T(m=8, p=6, n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, p, n, generator=g, dtype=torch.float64)


def test_slice_svd_exact_at_full_rank_and_params():
    T = _T()
    fit = fit_slice_svd(T, rank=6)
    assert fit.relative_error(T) < 1e-10
    fit2 = fit_slice_svd(T, rank=2)
    assert fit2.param_count() == 5 * 2 * (8 + 6)  # h * r * (m + p)


def test_cp_param_count_and_error_decreases():
    T = _T()
    f_small = fit_cp(T, rank=2, seed=0)
    f_big = fit_cp(T, rank=30, seed=0)
    assert f_small.param_count() == 2 * (8 + 6 + 5)
    assert f_big.relative_error(T) < f_small.relative_error(T)


def test_tucker_exact_at_full_rank_and_params():
    T = _T()
    fit = fit_tucker(T, rank=(8, 6, 5))
    assert fit.relative_error(T) < 1e-8
    f2 = fit_tucker(T, rank=(2, 3, 4))
    assert f2.param_count() == 8 * 2 + 6 * 3 + 5 * 4 + 2 * 3 * 4


def test_shared_tucker_exact_at_full_rank_and_params():
    T = _T()
    fit = fit_shared_tucker(T, rank=(8, 6))
    assert fit.relative_error(T) < 1e-8
    f2 = fit_shared_tucker(T, rank=(3, 2))
    # n1*R1 + n2*R2 + h*R1*R2 : per-slice cores, shared factors
    assert f2.param_count() == 8 * 3 + 6 * 2 + 5 * 3 * 2


def test_all_registered():
    from bmx.decomp.base import available_methods

    for name in ("slice_svd", "cp", "tucker", "shared_tucker"):
        assert name in available_methods()


def test_over_dimension_ranks_raise():
    import pytest

    T = _T()  # (8, 6, 5)
    with pytest.raises(AssertionError):
        fit_slice_svd(T, rank=12)
    with pytest.raises(AssertionError):
        fit_tucker(T, rank=(16, 6, 5))
    with pytest.raises(AssertionError):
        fit_shared_tucker(T, rank=(3, 7))
