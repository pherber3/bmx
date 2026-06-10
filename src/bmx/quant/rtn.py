"""Groupwise symmetric round-to-nearest quantization (returns dequantized values)."""

import torch


def rtn_quantize(W: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    *lead, d = W.shape
    assert d % group_size == 0, f"dim {d} not divisible by group {group_size}"
    qmax = 2 ** (bits - 1) - 1
    G = W.reshape(*lead, d // group_size, group_size)
    scale = G.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    Q = (G / scale).round().clamp(-qmax - 1, qmax)
    return (Q * scale).reshape(W.shape)
