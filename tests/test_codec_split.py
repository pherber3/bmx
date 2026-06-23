"""Codec split: packed quantize/dequant must equal the dequant-returning path."""

import pytest
import torch

from bmx.cache.codecs import dequant_packed, quantize_cache, quantize_packed
from bmx.quant.rtn import rtn_dequantize_packed, rtn_quantize, rtn_quantize_packed


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


SPLIT_ARMS = [
    ("rtn_token", dict(bits=3, group=16)),
    ("rtn_channel", dict(bits=3, group=8)),
    ("rotate_rtn_token", dict(bits=3, group=16)),
    ("turboquant_mse", dict(bits=2)),
    ("turboquant_prod", dict(bits=3)),
    ("lowrank_rtn_channel", dict(bits=3, group=8, rank=4)),
]


@pytest.mark.parametrize("arm,kw", SPLIT_ARMS)
def test_quantize_packed_matches_quantize_cache(arm, kw):
    torch.manual_seed(0)
    # S=16 (divisible by group 8), C=16 (power of 2 for hadamard rotate arms).
    M = torch.randn(16, 16, dtype=torch.float64)
    ref_hat, ref_bpe = quantize_cache(arm, M, **kw)
    packed, bpe = quantize_packed(arm, M, **kw)
    hat = dequant_packed(arm, packed, group=kw.get("group", 64), seed=0)
    assert bpe == pytest.approx(ref_bpe)
    assert torch.equal(hat, ref_hat)


def test_waterfill_arm_not_split_raises():
    M = torch.randn(16, 16, dtype=torch.float64)
    with pytest.raises(NotImplementedError):
        quantize_packed("lowrank_waterfill_channel", M, bits=3, group=8, rank=4)
