"""Constructive BMD initializations from BM-rank upper bounds.

ss_svd_init  -- Tian-Kilmer Thm 3.3: per-frontal-slice truncated SVD.
mode1_init   -- Tian-Kilmer Thm 3.1: truncated SVD of the mode-1 unfolding.

Both set the template factor B to all-ones, which specializes the BM product
to slice-wise matrix products: bmp(A, 1, C)[:, :, k] = A[:, :, k] @ C[:, :, k].
"""

import torch


def ss_svd_init(T: torch.Tensor, ell: int):
    m, p, n = T.shape
    assert ell <= min(m, p), f"ss_svd_init needs ell <= min(m,p)={min(m, p)}"
    U, S, Vh = torch.linalg.svd(T.permute(2, 0, 1), full_matrices=False)
    A = (U[:, :, :ell] * S[:, None, :ell]).permute(1, 2, 0)  # (m, ell, n)
    C = Vh[:, :ell, :].permute(1, 2, 0)  # (ell, p, n)
    B = torch.ones(m, p, ell, dtype=T.dtype, device=T.device)
    return A, B, C


def mode1_init(T: torch.Tensor, ell: int):
    m, p, n = T.shape
    X = T.permute(0, 2, 1).reshape(m * n, p)  # X[i*n + k, j] = T[i, j, k]
    assert ell <= min(X.shape), f"mode1_init needs ell <= {min(X.shape)}"
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    Us = U[:, :ell] * S[:ell]
    A = Us.reshape(m, n, ell).permute(0, 2, 1)  # (m, ell, n)
    C = Vh[:ell].unsqueeze(-1).expand(ell, p, n).contiguous()  # same V^T every slice
    B = torch.ones(m, p, ell, dtype=T.dtype, device=T.device)
    return A, B, C
