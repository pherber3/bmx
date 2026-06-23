"""Codec split: packed quantize/dequant must equal the dequant-returning path."""

import torch

from bmx.quant.rtn import rtn_quantize, rtn_quantize_packed, rtn_dequantize_packed


def test_rtn_packed_roundtrip_matches_dequant_path():
    torch.manual_seed(0)
    W = torch.randn(8, 64, dtype=torch.float64)
    bits, group = 3, 16
    ref = rtn_quantize(W, bits, group)
    Q_int, scale = rtn_quantize_packed(W, bits, group)
    W_hat = rtn_dequantize_packed(Q_int, scale, group)
    assert W_hat.shape == W.shape
    assert torch.equal(W_hat, ref)  # exact: same arithmetic, just split
    # Q_int holds integer levels within the symmetric range.
    qmax = 2 ** (bits - 1) - 1
    assert Q_int.max() <= qmax and Q_int.min() >= -qmax - 1
    assert Q_int.dtype == torch.int8
