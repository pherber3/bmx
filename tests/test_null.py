import torch

from bmx.stacks.null import permutation_null


def test_null_preserves_per_slice_spectra_as_multiset():
    T = torch.randn(8, 6, 5, dtype=torch.float64)
    Tn, transform = permutation_null(T, seed=0)
    assert Tn.shape == T.shape
    orig = torch.linalg.svdvals(T.permute(2, 0, 1))
    new = torch.linalg.svdvals(Tn.permute(2, 0, 1))
    # slice k of Tn is a two-sided rotation of slice perm[k] of T
    torch.testing.assert_close(new, orig[transform.perm], rtol=1e-10, atol=1e-10)


def test_null_is_seeded_and_changes_tensor():
    T = torch.randn(8, 6, 5, dtype=torch.float64)
    T1, _ = permutation_null(T, seed=1)
    T2, _ = permutation_null(T, seed=1)
    T3, _ = permutation_null(T, seed=2)
    torch.testing.assert_close(T1, T2)
    assert not torch.allclose(T1, T3)
    assert not torch.allclose(T1, T)
