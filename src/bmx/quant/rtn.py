"""Groupwise symmetric round-to-nearest quantization.

Two-step form (quantize -> packed -> dequant) plus the original one-shot
`rtn_quantize` kept as the composition for the existing dequant-returning callers.
"""

import torch


def _mse_refine_scale(
    G: torch.Tensor, scale: torch.Tensor, qmax: int, n_iter: int = 10
) -> torch.Tensor:
    """Alternating-minimization refinement of the per-group scale (Lloyd on the
    step size only; codebook stays uniform). Deterministic, monotone in MSE."""
    for _ in range(n_iter):
        Q = (G / scale).round().clamp(-qmax - 1, qmax)
        num = (G * Q).sum(dim=-1, keepdim=True)
        den = (Q * Q).sum(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = (num / den).clamp_min(1e-12)
    return scale


def rtn_quantize_packed(
    W: torch.Tensor, bits: int, group_size: int, mse_scale: bool = False
):
    """(..., d) -> (Q_int int8 same shape, scale (..., n_groups, 1)).

    Q_int holds the integer levels; scale is per-group. Dequant is
    `rtn_dequantize_packed(Q_int, scale, group_size)`.

    Set mse_scale=True to use MSE-optimal step; default False preserves historical
    max-based behavior bit-identically.
    """
    *lead, d = W.shape
    assert d % group_size == 0, f"dim {d} not divisible by group {group_size}"
    assert bits <= 8, f"rtn_quantize_packed: int8 codes require bits <= 8, got {bits}"
    qmax = 2 ** (bits - 1) - 1
    G = W.reshape(*lead, d // group_size, group_size)
    scale = G.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    if mse_scale:
        scale = _mse_refine_scale(G, scale, qmax)
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


def rtn_quantize(
    W: torch.Tensor, bits: int, group_size: int, mse_scale: bool = False
) -> torch.Tensor:
    """Groupwise symmetric RTN, returning dequantized values (unchanged API).

    Set mse_scale=True to use MSE-optimal step; default False preserves historical
    max-based behavior bit-identically.
    """
    Q_int, scale = rtn_quantize_packed(W, bits, group_size, mse_scale=mse_scale)
    return rtn_dequantize_packed(Q_int, scale, group_size)
