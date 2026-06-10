"""Tian-Kilmer RALS for the BM decomposition.

Each factor update is mp independent least-squares of size (n x ell)
(paper Eqs. 6.5/6.7/6.9), batched as one torch.linalg.lstsq call. The cyclic
transpose identity bmp(A,B,C)^T = bmp(B^T, C^T, A^T) lets a single middle-slot
solver serve all three factor updates.
"""

import torch

from bmx.decomp.base import FitResult, register
from bmx.decomp.init import mode1_init, ss_svd_init
from bmx.decomp.ops import bmp, cyclic_transpose
from bmx.stacks.synthetic import random_bmd_factors


class BMDFit(FitResult):
    def __init__(self, A, B, C, loss_history):
        super().__init__(method="bmd_rals", rank=A.shape[1], loss_history=loss_history)
        self.A, self.B, self.C = A, B, C

    def reconstruct(self) -> torch.Tensor:
        return bmp(self.A, self.B, self.C)

    def param_count(self) -> int:
        m, ell, n = self.A.shape
        p = self.B.shape[1]
        return ell * (m * p + m * n + p * n)


def _solve_middle(T, F1, F3, lam: float):
    """min_F2 ||T - bmp(F1, F2, F3)||_F^2, decoupled over (i, j)."""
    m, p, n = T.shape
    ell = F1.shape[1]
    H = torch.einsum("itk,tjk->ijkt", F1, F3).reshape(m * p, n, ell)
    y = T.reshape(m * p, n, 1)
    if lam == 0.0:
        # CUDA lstsq only supports the full-rank 'gels' driver; rank-deficient
        # blocks would silently produce garbage there. Use lstsq on CPU (rank-
        # revealing driver), tiny-ridge normal equations on accelerators.
        if T.device.type == "cpu":
            sol = torch.linalg.lstsq(H, y).solution
        else:
            eps = torch.finfo(T.dtype).eps * 100
            G = H.mT @ H + eps * torch.eye(ell, dtype=T.dtype, device=T.device)
            sol = torch.linalg.solve(G, H.mT @ y)
    else:
        G = H.mT @ H + lam * torch.eye(ell, dtype=T.dtype, device=T.device)
        sol = torch.linalg.solve(G, H.mT @ y)
    return sol.reshape(m, p, ell)


def _run_als(
    T: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    n_iters: int,
    tol: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
    """Run ALS from given initial factors, returning (A, B, C, loss_history)."""
    norm_T = torch.linalg.norm(T)
    cyc = cyclic_transpose
    Tt = cyc(T).contiguous()
    Ttt = cyc(Tt).contiguous()

    history: list[float] = []
    for _ in range(n_iters):
        B = _solve_middle(T, A, C, lam)
        # C sits in the middle slot of the once-transposed problem.
        Ct = _solve_middle(Tt, cyc(B).contiguous(), cyc(A).contiguous(), lam)
        C = Ct.permute(2, 0, 1).contiguous()
        # A sits in the middle slot of the twice-transposed problem.
        Att = _solve_middle(
            Ttt, cyc(cyc(C)).contiguous(), cyc(cyc(B)).contiguous(), lam
        )
        A = Att.permute(1, 2, 0).contiguous()

        re = (torch.linalg.norm(bmp(A, B, C) - T) / norm_T).item()
        history.append(re)
        if len(history) >= 2 and abs(history[-2] - history[-1]) < tol:
            break

    return A, B, C, history


@register("bmd_rals")
def fit_bmd_rals(
    T: torch.Tensor,
    rank: int,
    *,
    n_iters: int = 200,
    tol: float = 1e-9,
    init: "str | tuple[torch.Tensor, torch.Tensor, torch.Tensor]" = "ss_svd",
    lam: float = 0.0,
    seed: int = 0,
    n_restarts: int = 3,
) -> BMDFit:
    """Fit a BM decomposition via alternating least squares.

    Parameters
    ----------
    T:
        Input tensor of shape (m, p, n).
    rank:
        Target BM rank (ell).
    n_iters:
        Maximum ALS iterations per restart.
    tol:
        Convergence tolerance on relative-error change.
    init:
        Initialization strategy:

        * ``'ss_svd'`` -- per-slice SVD warm start followed by *n_restarts*
          random restarts (recommended).
        * ``'mode1'`` -- mode-1 unfolding SVD warm start, same restart logic.
        * ``'random'`` -- single random start at *seed*; *n_restarts* ignored.
        * A 3-tuple ``(A0, B0, C0)`` of tensors -- used directly as the
          starting point; shapes must be compatible with T and rank.
          *n_restarts* and *seed* are ignored.
    lam:
        Tikhonov regularisation weight (0 = no regularisation).
    seed:
        Random seed used when ``init='random'`` or as a base for random
        restarts.
    n_restarts:
        Number of independent random restarts when ``init='ss_svd'`` or
        ``'mode1'``.  The ss/mode1 warm start plus *n_restarts* random starts
        are all run; the best result is returned.
        Set to 0 to disable random restarts and use only the structured init.
        Ignored when ``init='random'`` or ``init`` is a tuple.
        The returned ``loss_history`` is the winning run's trajectory only.
    """
    ell = int(rank)
    m, p, n = T.shape

    # Explicit-factor init: validate shapes against T and rank, then run once.
    if isinstance(init, tuple):
        A0, B0, C0 = init
        assert A0.shape == (m, ell, n), (
            f"init A shape {tuple(A0.shape)} != expected {(m, ell, n)}"
        )
        assert B0.shape == (m, p, ell), (
            f"init B shape {tuple(B0.shape)} != expected {(m, p, ell)}"
        )
        assert C0.shape == (ell, p, n), (
            f"init C shape {tuple(C0.shape)} != expected {(ell, p, n)}"
        )
        A, B, C, history = _run_als(T, A0, B0, C0, n_iters, tol, lam)
        return BMDFit(A, B, C, history)

    if init == "random":
        # Single-start random init; n_restarts is ignored.
        A, B, C = random_bmd_factors(
            m, p, n, ell, seed, dtype=T.dtype, device=str(T.device)
        )
        A, B, C, history = _run_als(T, A, B, C, n_iters, tol, lam)
        return BMDFit(A, B, C, history)

    # Structured warm start.
    if init == "ss_svd":
        A0, B0, C0 = ss_svd_init(T, ell)
    elif init == "mode1":
        A0, B0, C0 = mode1_init(T, ell)
    else:
        raise ValueError(f"unknown init {init!r}")

    best_A, best_B, best_C, best_hist = _run_als(T, A0, B0, C0, n_iters, tol, lam)
    best_re = best_hist[-1]

    # Random restarts: use seeds offset far from small user/data seeds to avoid
    # colliding with data-generating seeds (which use small ints 0, 1, 2, ...).
    for i in range(n_restarts):
        restart_seed = 100_003 + 7919 * i + seed
        Ar, Br, Cr = random_bmd_factors(
            m, p, n, ell, restart_seed, dtype=T.dtype, device=str(T.device)
        )
        Ar, Br, Cr, hist_r = _run_als(T, Ar, Br, Cr, n_iters, tol, lam)
        if hist_r[-1] < best_re:
            best_re = hist_r[-1]
            best_A, best_B, best_C, best_hist = Ar, Br, Cr, hist_r

    return BMDFit(best_A, best_B, best_C, best_hist)
