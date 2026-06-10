import torch

from bmx.decomp.ops import bmp, cyclic_transpose, cyclic_transpose_inv


def _factors(m=4, p=3, n=5, ell=2, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(m, ell, n, generator=g, dtype=dtype)
    B = torch.randn(m, p, ell, generator=g, dtype=dtype)
    C = torch.randn(ell, p, n, generator=g, dtype=dtype)
    return A, B, C


def test_bmp_shape():
    A, B, C = _factors()
    assert bmp(A, B, C).shape == (4, 3, 5)


def test_bmp_matches_diag_template_slices():
    A, B, C = _factors()
    T = bmp(A, B, C)
    m, p, n = T.shape
    ell = A.shape[1]
    for k in range(n):
        slice_k = sum(
            torch.diag(A[:, t, k]) @ B[:, :, t] @ torch.diag(C[t, :, k])
            for t in range(ell)
        )
        torch.testing.assert_close(T[:, :, k], slice_k)


def test_transpose_identity():
    A, B, C = _factors()
    lhs = cyclic_transpose(bmp(A, B, C))
    rhs = bmp(cyclic_transpose(B), cyclic_transpose(C), cyclic_transpose(A))
    torch.testing.assert_close(lhs, rhs)


def test_cyclic_transpose_order_three():
    T = torch.randn(4, 3, 5, dtype=torch.float64)
    torch.testing.assert_close(
        cyclic_transpose(cyclic_transpose(cyclic_transpose(T))), T
    )
    torch.testing.assert_close(cyclic_transpose_inv(cyclic_transpose(T)), T)
