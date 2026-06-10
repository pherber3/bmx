import torch

from bmx.decomp.ops import bmp
from bmx.stacks.synthetic import bm_rank_tensor, random_bmd_factors


def test_random_factors_shapes_and_determinism():
    A1, B1, C1 = random_bmd_factors(6, 5, 4, ell=2, seed=3)
    A2, B2, C2 = random_bmd_factors(6, 5, 4, ell=2, seed=3)
    assert A1.shape == (6, 2, 4) and B1.shape == (6, 5, 2) and C1.shape == (2, 5, 4)
    torch.testing.assert_close(A1, A2)
    torch.testing.assert_close(B1, B2)
    torch.testing.assert_close(C1, C2)


def test_bm_rank_tensor_is_bmp_of_factors():
    T, (A, B, C) = bm_rank_tensor(6, 5, 4, ell=2, seed=0)
    torch.testing.assert_close(T, bmp(A, B, C))


def test_slices_are_generically_full_rank():
    # The whole point: low BM-rank does NOT mean low slice rank.
    T, _ = bm_rank_tensor(8, 8, 4, ell=2, seed=0)
    for k in range(4):
        assert torch.linalg.matrix_rank(T[:, :, k]).item() == 8
