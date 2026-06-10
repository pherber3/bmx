"""Known-answer tensors for solver validation and random factors for bench shapes."""

import torch

from bmx.decomp.ops import bmp


def random_bmd_factors(
    m: int,
    p: int,
    n: int,
    ell: int,
    seed: int,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(m, ell, n, generator=g, dtype=dtype)
    B = torch.randn(m, p, ell, generator=g, dtype=dtype)
    C = torch.randn(ell, p, n, generator=g, dtype=dtype)
    return A.to(device), B.to(device), C.to(device)


def bm_rank_tensor(
    m: int,
    p: int,
    n: int,
    ell: int,
    seed: int,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
):
    """Exact BM-rank<=ell tensor with known generating factors."""
    A, B, C = random_bmd_factors(m, p, n, ell, seed, dtype=dtype, device=device)
    return bmp(A, B, C), (A, B, C)
