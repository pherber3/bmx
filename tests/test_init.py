import torch

from bmx.decomp.init import mode1_init, ss_svd_init
from bmx.decomp.ops import bmp


def test_ss_svd_init_equals_per_slice_truncated_svd():
    """At zero ALS sweeps, SS-SVD init IS the per-slice truncated SVD baseline."""
    T = torch.randn(8, 6, 5, dtype=torch.float64)
    ell = 3
    A, B, C = ss_svd_init(T, ell)
    rec = bmp(A, B, C)
    for k in range(T.shape[2]):
        U, S, Vh = torch.linalg.svd(T[:, :, k], full_matrices=False)
        trunc = U[:, :ell] @ torch.diag(S[:ell]) @ Vh[:ell, :]
        torch.testing.assert_close(rec[:, :, k], trunc)


def test_ss_svd_exact_when_ell_ge_max_slice_rank():
    T = torch.randn(4, 6, 5, dtype=torch.float64)  # slice rank <= 4
    A, B, C = ss_svd_init(T, ell=4)
    torch.testing.assert_close(bmp(A, B, C), T)


def test_mode1_init_exact_at_unfolding_rank():
    T = torch.randn(3, 4, 5, dtype=torch.float64)  # mode-1 unfolding (15, 4): rank 4
    A, B, C = mode1_init(T, ell=4)
    torch.testing.assert_close(bmp(A, B, C), T)


def test_init_factor_shapes():
    T = torch.randn(8, 6, 5, dtype=torch.float64)
    for init in (ss_svd_init, mode1_init):
        A, B, C = init(T, ell=2)
        assert A.shape == (8, 2, 5)
        assert B.shape == (8, 6, 2)
        assert C.shape == (2, 6, 5)
