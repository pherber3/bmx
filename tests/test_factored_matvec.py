import torch

from bmx.bench.factored_matvec import (
    dense_from_factors,
    dense_slice_matvec,
    factored_matvec,
)
from bmx.stacks.synthetic import random_bmd_factors


def test_factored_matches_dense():
    A, B, C = random_bmd_factors(16, 12, 4, ell=3, seed=0)
    x = torch.randn(5, 12, dtype=torch.float64)  # batch 5
    W = dense_from_factors(A, B, C)  # (h, m, p)
    assert W.shape == (4, 16, 12)
    y_dense = dense_slice_matvec(W, x)  # (h, b, m)
    y_fact = factored_matvec(A, B, C, x)
    assert y_fact.shape == (4, 5, 16)
    torch.testing.assert_close(y_fact, y_dense)


def test_single_slice_is_diag_template_matvec():
    A, B, C = random_bmd_factors(8, 8, 3, ell=2, seed=1)
    x = torch.randn(1, 8, dtype=torch.float64)
    y = factored_matvec(A, B, C, x)
    k = 1
    manual = sum(A[:, t, k] * (B[:, :, t] @ (C[t, :, k] * x[0])) for t in range(2))
    torch.testing.assert_close(y[k, 0], manual)
