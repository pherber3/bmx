import torch

from bmx.quant.rtn import rtn_quantize, rtn_quantize_packed


def _gaussian(S=256, C=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(S, C, generator=g)


def test_default_unchanged():
    """mse_scale defaults False and is bit-identical to the historical behavior."""
    W = _gaussian()
    q_old, s_old = rtn_quantize_packed(W, 3, 64)
    q_new, s_new = rtn_quantize_packed(W, 3, 64, mse_scale=False)
    assert torch.equal(q_old, q_new)
    assert torch.equal(s_old, s_new)


def test_mse_scale_lowers_mse():
    """MSE-optimal step strictly beats max-based step on Gaussian data at 2 and 3 bits."""
    W = _gaussian()
    for bits in (2, 3):
        err_max = (rtn_quantize(W, bits, 64) - W).pow(2).mean()
        err_mse = (rtn_quantize(W, bits, 64, mse_scale=True) - W).pow(2).mean()
        assert err_mse < err_max, f"bits={bits}: {err_mse} !< {err_max}"


def test_mse_scale_deterministic():
    W = _gaussian(seed=3)
    a = rtn_quantize(W, 2, 64, mse_scale=True)
    b = rtn_quantize(W, 2, 64, mse_scale=True)
    assert torch.equal(a, b)
