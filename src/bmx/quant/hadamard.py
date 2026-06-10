"""Data-oblivious rotations: fast Walsh-Hadamard (power-of-2 dims, Sylvester
ordering) and QR random orthogonal (any dim, the Haar reference)."""

import torch


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal FWHT over the last dim (must be a power of 2)."""
    d = x.shape[-1]
    assert d & (d - 1) == 0 and d > 0, f"fwht requires power-of-2 dim, got {d}"
    orig_shape = x.shape
    y = x.reshape(-1, d).clone()
    h = 1
    while h < d:
        y = y.view(-1, d // (2 * h), 2, h)
        pos = y[:, :, 0, :] + y[:, :, 1, :]
        neg = y[:, :, 0, :] - y[:, :, 1, :]
        y = torch.stack((pos, neg), dim=2).view(-1, d)
        h *= 2
    return (y / d**0.5).view(orig_shape)


def randomized_hadamard(x: torch.Tensor, seed: int) -> torch.Tensor:
    """H @ diag(signs) @ x rows — the standard randomized Hadamard rotation."""
    d = x.shape[-1]
    g = torch.Generator().manual_seed(seed)
    signs = (torch.randint(0, 2, (d,), generator=g) * 2 - 1).to(x.dtype)
    return fwht(x * signs)


def orthogonalize(M: torch.Tensor) -> torch.Tensor:
    """QR-orthogonalize the last two dims (batched ok), sign-canonicalized.

    torch QR leaves diag(R)'s signs backend-dependent; forcing diag(R) >= 0
    makes the result a pure function of M (and Haar-distributed for
    Gaussian M).
    """
    Q, R = torch.linalg.qr(M)
    signs = R.diagonal(dim1=-2, dim2=-1).sign()
    signs[signs == 0] = 1.0
    return Q * signs.unsqueeze(-2)


def random_orthogonal(
    d: int, seed: int, dtype=torch.float32, device="cpu"
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(d, d, generator=g, dtype=dtype)
    return orthogonalize(M).to(device)
