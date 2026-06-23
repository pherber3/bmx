"""Groupwise symmetric round-to-nearest quantization.

Two-step form (quantize -> packed -> dequant) plus the original one-shot
`rtn_quantize` kept as the composition for the existing dequant-returning callers.
"""

import torch


def rtn_quantize_packed(W: torch.Tensor, bits: int, group_size: int):
    """(..., d) -> (Q_int int8 same shape, scale (..., n_groups, 1)).

    Q_int holds the integer levels; scale is per-group. Dequant is
    `rtn_dequantize_packed(Q_int, scale, group_size)`.
    """
    *lead, d = W.shape
    assert d % group_size == 0, f"dim {d} not divisible by group {group_size}"
    qmax = 2 ** (bits - 1) - 1
    G = W.reshape(*lead, d // group_size, group_size)
    scale = G.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    Q = (G / scale).round().clamp(-qmax - 1, qmax)
    Q_int = Q.to(torch.int8).reshape(W.shape)
    return Q_int, scale


def rtn_dequantize_packed(
    Q_int: torch.Tensor, scale: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Inverse of rtn_quantize_packed: (Q_int, scale) -> dequantized W_hat."""
    *lead, d = Q_int.shape
    G = Q_int.reshape(*lead, d // group_size, group_size).to(scale.dtype)
    return (G * scale).reshape(Q_int.shape)


def rtn_quantize(W: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Groupwise symmetric RTN, returning dequantized values (unchanged API)."""
    Q_int, scale = rtn_quantize_packed(W, bits, group_size)
    return rtn_dequantize_packed(Q_int, scale, group_size)
