"""The permutation null (A3): destroys cross-slice alignment, preserves
per-slice spectra. The per-slice two-sided orthogonal rotations are the
load-bearing part; the slice shuffle alone is absorbed into every method's
slice-mode factor."""

from dataclasses import dataclass

import torch

from bmx.quant.hadamard import orthogonalize


@dataclass
class NullTransform:
    seed: int
    perm: torch.Tensor  # (h,)
    Q: torch.Tensor  # (h, m, m) left rotations
    R: torch.Tensor  # (h, p, p) right rotations


def _random_orthogonal_batch(count: int, dim: int, g, dtype):
    M = torch.randn(count, dim, dim, generator=g, dtype=dtype)
    return orthogonalize(M)


def permutation_null(T: torch.Tensor, seed: int):
    m, p, n = T.shape
    # Rotations are generated on CPU so the transform is a function of the
    # seed alone (identical across CPU/GPU runs), then moved to T's device.
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    Q = _random_orthogonal_batch(n, m, g, T.dtype).to(T.device)
    R = _random_orthogonal_batch(n, p, g, T.dtype).to(T.device)
    X = T[:, :, perm].permute(2, 0, 1)  # (n, m, p)
    Y = Q @ X @ R.mT  # slice k -> Q_k T[:,:,perm_k] R_k^T
    return Y.permute(1, 2, 0).contiguous(), NullTransform(seed, perm, Q, R)
