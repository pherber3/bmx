"""Distribution diagnostics for D1 and the distortion-floor machinery for D3."""

import torch


def kurtosis(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Fisher excess kurtosis along dim (0 for Gaussian)."""
    mu = x.mean(dim=dim, keepdim=True)
    var = x.var(dim=dim, unbiased=False, keepdim=True)
    return ((x - mu) ** 4).mean(dim=dim) / var.squeeze(dim) ** 2 - 3.0


def outlier_mass(W: torch.Tensor, k_sigma: float = 3.0) -> torch.Tensor:
    """Per-channel (last-dim column) fraction of entries beyond k_sigma * global std."""
    thresh = k_sigma * W.std()
    return (W.abs() > thresh).to(torch.float64).mean(dim=0)


def ip_distortion(W: torch.Tensor, Wq: torch.Tensor, X: torch.Tensor) -> float:
    """Relative inner-product distortion ||W X^T - Wq X^T||_F / ||W X^T||_F."""
    ref = W @ X.mT
    return ((Wq @ X.mT - ref).norm() / ref.norm()).item()


def sq_floor(bits: int) -> float:
    """Worst-case MSE rate floor 4^-b (Shannon + Yao, TurboQuant §3.3)."""
    return 4.0 ** (-bits)
