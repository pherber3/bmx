import pytest
import torch

from bmx.quant.arms import ARMS, reconstruct_arm, total_bits


def _W(m=32, p=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, p, generator=g, dtype=torch.float64)


def test_total_bits_formula():
    # bulk ints + fp16 group scales + fp16 L factors + (fp16 + index) sparse
    m, p, bits, gs, r, k = 32, 64, 4, 32, 2, 5
    idx_bits = (m * p - 1).bit_length()  # ceil(log2(2048)) = 11
    expected = (
        m * p * bits + (m * p // gs) * 16 + r * (m + p) * 16 + k * (16 + idx_bits)
    )
    assert total_bits(m, p, bits=bits, group_size=gs, r=r, k=k) == expected
    # rotation arms store nothing extra: r=k=0 accounting
    assert total_bits(m, p, bits=4, group_size=32, r=0, k=0) == m * p * 4 + 64 * 16


def test_plain_arm_matches_rtn():
    from bmx.quant.rtn import rtn_quantize

    W = _W()
    rec, r, k = reconstruct_arm("rtn", W, bits=4, group_size=32, r=8, k=10, seed=0)
    assert torch.equal(rec, rtn_quantize(W, 4, 32))
    assert (r, k) == (0, 0)  # plain arm stores no L/S regardless of request


def test_lrs_arm_exact_at_full_rank():
    W = _W()
    rec, r, k = reconstruct_arm("lrs_rtn", W, bits=2, group_size=32, r=32, k=0, seed=0)
    # full-rank L makes R = 0; quantization of zeros is exact
    assert (rec - W).norm() / W.norm() < 1e-10
    assert (r, k) == (32, 0)


def test_rotate_arms_deterministic_in_seed():
    W = _W()
    a1, _, _ = reconstruct_arm("rotate_rtn", W, bits=4, group_size=32, r=0, k=0, seed=7)
    a2, _, _ = reconstruct_arm("rotate_rtn", W, bits=4, group_size=32, r=0, k=0, seed=7)
    a3, _, _ = reconstruct_arm("rotate_rtn", W, bits=4, group_size=32, r=0, k=0, seed=8)
    assert torch.equal(a1, a2)
    assert not torch.equal(a1, a3)


def test_all_arms_run_and_unknown_raises():
    W = _W()
    for arm in ARMS:
        rec, _, _ = reconstruct_arm(arm, W, bits=4, group_size=32, r=4, k=8, seed=0)
        assert rec.shape == W.shape
        assert (rec - W).norm() / W.norm() < 0.5
    with pytest.raises(AssertionError):
        reconstruct_arm("nope", W, bits=4, group_size=32, r=0, k=0, seed=0)


def test_lrs_arms_subset_and_ls_passthrough():
    from bmx.quant.arms import LRS_ARMS, fit_ls

    assert set(LRS_ARMS) <= set(ARMS)
    W = _W()
    ls = fit_ls(W, r=4, k=8)
    for arm in LRS_ARMS:
        direct, *_ = reconstruct_arm(arm, W, bits=4, group_size=32, r=4, k=8, seed=0)
        cached, *_ = reconstruct_arm(
            arm, W, bits=4, group_size=32, r=4, k=8, seed=0, ls=ls
        )
        assert torch.equal(direct, cached)
