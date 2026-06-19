"""Pin the faithful-baseline arms against the TurboQuant paper construction.

Guards the K3 head-to-head: K2b is compared against paper-correct TurboQuant/KIVI,
not the degraded '×0.707' variant from the public repo that prompted this work.
Vault refs: 'TurboQuant - Online Vector Quantization', 'Two-Stage Quantization
for Unbiased Inner Products'.
"""

import math

import torch

from bmx.cache.codecs import qjl_reconstruct, quantize_cache


def _M(S=64, C=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(S, C, generator=g)


def test_turboquant_mse_bpe_is_bits_plus_one_norm():
    # TurboQuant_mse stores one fp16 norm per vector over C coords: bpe = bits + 16/C.
    M = _M()
    _, bpe = quantize_cache("turboquant_mse", M, bits=2)
    assert math.isclose(bpe, 2 + 16.0 / M.shape[1], rel_tol=1e-9)


def test_turboquant_prod_bpe_is_two_stage_two_norms():
    # TurboQuant_prod = MSE at (b-1) + 1-bit QJL on residual; two fp16 norms.
    # bpe = (b-1) + 1 + 32/C  (vault: Two-Stage Quantization, the ||r|| overhead).
    M = _M()
    _, bpe = quantize_cache("turboquant_prod", M, bits=3)
    assert math.isclose(bpe, 2 + 1 + 32.0 / M.shape[1], rel_tol=1e-9)


def test_turboquant_prod_unbiased_inner_product():
    # The load-bearing property (vault Theorem 2): QJL stage is unbiased for <y,x>.
    # Averaging the QJL reconstruction over seeds drives bias -> 0.
    torch.manual_seed(0)
    R = _M(S=8, C=256)
    y = torch.randn(8, 256)
    true_ip = (R * y).sum(dim=1)
    ests = torch.stack(
        [(qjl_reconstruct(R, seed=s) * y).sum(dim=1) for s in range(200)]
    )
    mean_ip = ests.mean(dim=0)
    # Unbiased: mean estimate within a few % of truth (Monte-Carlo over 200 seeds).
    rel_err = (mean_ip - true_ip).abs() / true_ip.abs().clamp_min(1e-6)
    assert rel_err.mean() < 0.1


def test_kivi_pairing_is_channel_then_token():
    # KIVI = per-channel K (rtn_channel) / per-token V (rtn_token); both real arms.
    M = _M(S=64, C=128)
    _, bpe_k = quantize_cache("rtn_channel", M, bits=2, group=64)
    _, bpe_v = quantize_cache("rtn_token", M, bits=2, group=64)
    assert math.isclose(bpe_k, 2 + 16.0 / 64, rel_tol=1e-9)
    assert math.isclose(bpe_v, 2 + 16.0 / 64, rel_tol=1e-9)
