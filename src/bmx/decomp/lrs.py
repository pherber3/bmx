"""Low-rank-plus-sparse two-step estimator (Avenue 1).

Hard-threshold-then-truncated-SVD per Wainwright HDS §11.4.2 (Eq. 11.58 /
Prop. 11.19; direct method due to Agarwal et al. 2012): S = T_nu(W) keeps
large entries VERBATIM (hard threshold — soft-thresholding is a different,
wrong operator here), L = truncated SVD of W - S, optionally alternated.

Operates on a single 2-D weight matrix — unlike the 3-D stack methods — and
is parameterized by budget (r, k) rather than threshold nu, so matched-bit
sweeps are direct. Fit in the ORIGINAL basis: rotation provably spreads the
concentrated mass S needs (see docs/next-avenues-structured-residual.md).
"""

import torch

from bmx.decomp.base import FitResult, register


def hard_threshold(W: torch.Tensor, nu: float) -> torch.Tensor:
    """T_nu(v) = v * 1[|v| > nu] — keeps the value, no shrinkage (Eq. 11.58)."""
    return W * (W.abs() > nu)


def topk_sparse(W: torch.Tensor, k: int) -> torch.Tensor:
    """Hard threshold parameterized by budget: exactly the k largest |entries|."""
    if k == 0:
        return torch.zeros_like(W)
    flat = W.flatten()
    idx = flat.abs().topk(k).indices
    S = torch.zeros_like(flat)
    S[idx] = flat[idx]
    return S.view_as(W)


def truncated_svd(W: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Best rank-r approximation as (U*s, V): W_r = Us @ V.mT."""
    m, p = W.shape
    if r == 0:
        return W.new_zeros(m, 0), W.new_zeros(p, 0)
    U, s, Vh = torch.linalg.svd(W, full_matrices=False)
    return U[:, :r] * s[:r], Vh[:r, :].mT.contiguous()


def two_step_lrs(
    W: torch.Tensor, r: int, k: int, n_alternations: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (Us, V, S) with L = Us @ V.mT; W ≈ L + S."""
    assert W.ndim == 2, f"two_step_lrs operates on a matrix, got ndim={W.ndim}"
    m, p = W.shape
    assert 0 <= r <= min(m, p), f"rank {r} > min(m,p)={min(m, p)}"
    assert 0 <= k <= W.numel(), f"sparse budget {k} > numel={W.numel()}"
    S = topk_sparse(W, k)
    Us, V = truncated_svd(W - S, r)
    for _ in range(n_alternations):
        S = topk_sparse(W - Us @ V.mT, k)
        Us, V = truncated_svd(W - S, r)
    return Us, V, S


def spikiness_ratio(M: torch.Tensor) -> float:
    """alpha_hat = ||M||_max * sqrt(d1*d2) / ||M||_F (Wainwright's spikiness,
    normalized so a flat matrix scores ~1 and a single spike scores sqrt(d1*d2))."""
    assert M.norm() > 0, "spikiness_ratio undefined for an all-zero matrix"
    return (M.abs().max() * M.numel() ** 0.5 / M.norm()).item()


class LRSFit(FitResult):
    def __init__(self, Us: torch.Tensor, V: torch.Tensor, S: torch.Tensor):
        # achieved (not requested) support size: topk can pick exact zeros
        # only on degenerate inputs, and stored-numbers accounting is honest
        k = int((S != 0).sum())
        super().__init__(method="lrs", rank=(Us.shape[1], k), loss_history=[])
        self.Us, self.V, self.S = Us, V, S

    def reconstruct(self) -> torch.Tensor:
        return self.Us @ self.V.mT + self.S

    def param_count(self) -> int:
        # stored NUMBERS only (r*(m+p) factor entries + k sparse values);
        # sparse index bits are storage, not parameters — counted by
        # bmx.quant.arms.total_bits where bit budgets are compared.
        r, k = self.rank
        return r * (self.Us.shape[0] + self.V.shape[0]) + k


@register("lrs")
def fit_lrs(W: torch.Tensor, rank, *, n_alternations: int = 2) -> LRSFit:
    r, k = (int(x) for x in rank)
    Us, V, S = two_step_lrs(W, r, k, n_alternations=n_alternations)  # asserts bounds
    fit = LRSFit(Us, V, S)
    fit.loss_history = [fit.relative_error(W)]
    return fit
