"""Compression arms for the Avenue 1 comparison, plus honest bit accounting.

Each arm maps W -> reconstructed W-hat. The L+S arms fit on CLEAN W in the
ORIGINAL basis (load-bearing — rotation spreads the mass S needs; see
docs/next-avenues-structured-residual.md), then quantize R = W - L - S.
Rotations are generated from a seed, so they cost 0 stored bits.
"""

import torch

from bmx.decomp.lrs import two_step_lrs
from bmx.quant.hadamard import random_orthogonal
from bmx.quant.rtn import rtn_quantize

FP_BITS = 16  # storage precision for L factors, S values, and group scales

ARMS = ("rtn", "rotate_rtn", "lrs_rtn", "lrs_rotate_rtn")
LRS_ARMS = ("lrs_rtn", "lrs_rotate_rtn")  # the arms that consume an (r, k) budget


def total_bits(m: int, p: int, *, bits: int, group_size: int, r: int, k: int) -> int:
    """Total stored bits: bulk ints + group scales + L factors + S (values+indices)."""
    bulk = m * p * bits + (m * p // group_size) * FP_BITS
    idx_bits = (m * p - 1).bit_length()
    return bulk + r * (m + p) * FP_BITS + k * (FP_BITS + idx_bits)


def _rotate_rtn(W: torch.Tensor, bits: int, group_size: int, seed: int) -> torch.Tensor:
    Q = random_orthogonal(W.shape[-1], seed=seed, dtype=W.dtype, device=W.device)
    return rtn_quantize(W @ Q.mT, bits, group_size) @ Q


def fit_ls(
    W: torch.Tensor, r: int, k: int, n_alternations: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense (L, S) from the two-step estimator. Depends only on (W, r, k) —
    sweeps should fit once per budget point and reuse across bit-widths/arms."""
    Us, V, S = two_step_lrs(W, r, k, n_alternations=n_alternations)
    return Us @ V.mT, S


def reconstruct_arm(
    arm: str,
    W: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    r: int,
    k: int,
    seed: int,
    n_alternations: int = 2,
    ls: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, int, int]:
    """Returns (W_hat, r_stored, k_stored); pass r/k stored to total_bits.

    ls: optional precomputed fit_ls(W, r, k) result, reused by sweeps."""
    assert arm in ARMS, f"unknown arm {arm!r}; available: {ARMS}"
    if arm == "rtn":
        return rtn_quantize(W, bits, group_size), 0, 0
    if arm == "rotate_rtn":
        return _rotate_rtn(W, bits, group_size, seed), 0, 0
    L, S = fit_ls(W, r, k, n_alternations=n_alternations) if ls is None else ls
    R = W - L - S
    if arm == "lrs_rtn":
        Rq = rtn_quantize(R, bits, group_size)
    else:  # lrs_rotate_rtn: rotate only the residual, after L+S extraction
        Rq = _rotate_rtn(R, bits, group_size, seed)
    return L + S + Rq, r, k
