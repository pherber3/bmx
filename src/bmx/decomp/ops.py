"""Core Bhattacharya-Mesner tensor operations.

Conventions (fixed across the codebase):
    stack tensor  T : (n1, n2, h)   -- slice/stack axis is mode 3
    factor A : (n1, ell, h)         -- per-slice output gains
    factor B : (n1, n2, ell)        -- shared templates
    factor C : (ell, n2, h)         -- per-slice input gains
"""

import torch


def bmp(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """BM product: out[i,j,k] = sum_t A[i,t,k] * B[i,j,t] * C[t,j,k]."""
    m, ell, n = A.shape
    mb, p, lb = B.shape
    lc, pc, nc = C.shape
    assert (m, ell) == (mb, lb) and (ell, p, n) == (lc, pc, nc), (
        f"incompatible BMD factor shapes A={tuple(A.shape)} "
        f"B={tuple(B.shape)} C={tuple(C.shape)}"
    )
    return torch.einsum("itk,ijt,tjk->ijk", A, B, C)


def bmd_param_count(m: int, p: int, n: int, ell: int) -> int:
    """Total factor entries of a rank-ell BMD: A (m,ell,n) + B (m,p,ell) + C (ell,p,n)."""
    return ell * (m * p + m * n + p * n)


def cyclic_transpose(T: torch.Tensor) -> torch.Tensor:
    """X^T in the BM sense: 1-based permute [2,3,1]. Order 3."""
    return T.permute(1, 2, 0)


def cyclic_transpose_inv(T: torch.Tensor) -> torch.Tensor:
    """Inverse of cyclic_transpose (= applying it twice)."""
    return T.permute(2, 0, 1)
