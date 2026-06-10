"""Known-answer tensors for solver validation and random factors for bench shapes."""

import torch

from bmx.decomp.ops import bmp, random_bmd_factors

__all__ = ["bm_rank_tensor", "random_bmd_factors"]


def bm_rank_tensor(
    m: int,
    p: int,
    n: int,
    ell: int,
    seed: int,
    dtype: torch.dtype = torch.float64,
    device: "str | torch.device" = "cpu",
):
    """Exact BM-rank<=ell tensor with known generating factors."""
    A, B, C = random_bmd_factors(m, p, n, ell, seed, dtype=dtype, device=device)
    return bmp(A, B, C), (A, B, C)
