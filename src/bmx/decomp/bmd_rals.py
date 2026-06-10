"""Tian-Kilmer RALS for the BM decomposition.

Each factor update is mp independent least-squares of size (n x ell)
(paper Eqs. 6.5/6.7/6.9), batched as one solver call. The cyclic transpose
identity bmp(A,B,C)^T = bmp(B^T, C^T, A^T) lets a single middle-slot solver
serve all three factor updates.
"""

import torch

from bmx.decomp.base import FitResult, register
from bmx.decomp.init import mode1_init, ss_svd_init
from bmx.decomp.ops import (
    bmd_param_count,
    bmp,
    cyclic_transpose,
    cyclic_transpose_inv,
    random_bmd_factors,
)


class BMDFit(FitResult):
    def __init__(self, A, B, C, loss_history, solver: str = "lstsq"):
        super().__init__(method="bmd_rals", rank=A.shape[1], loss_history=loss_history)
        self.A, self.B, self.C = A, B, C
        self.solver = solver

    def reconstruct(self) -> torch.Tensor:
        return bmp(self.A, self.B, self.C)

    def param_count(self) -> int:
        m, ell, n = self.A.shape
        p = self.B.shape[1]
        return bmd_param_count(m, p, n, ell)


def _solve_middle(T, F1, F3, lam: float, auto_ridge: bool):
    """min_F2 ||T - bmp(F1, F2, F3)||_F^2, decoupled over (i, j).

    lam > 0 -> Tikhonov normal equations with the absolute lam (the actual
    regularized objective). lam == 0: rank-revealing lstsq, unless auto_ridge
    (accelerators, where torch's lstsq assumes full rank) -- then a
    scale-relative epsilon ridge per block. Relative, because an absolute
    epsilon vanishes against blocks whose entries are large (G ~ ||H||^2),
    leaving rank-deficient blocks numerically singular.
    """
    m, p, n = T.shape
    ell = F1.shape[1]
    H = torch.einsum("itk,tjk->ijkt", F1, F3).reshape(m * p, n, ell)
    y = T.reshape(m * p, n, 1)
    if lam == 0.0 and not auto_ridge:
        sol = torch.linalg.lstsq(H, y).solution
    else:
        G = H.mT @ H
        eye = torch.eye(ell, dtype=T.dtype, device=T.device)
        if lam > 0.0:
            reg = lam * eye
        else:
            eps = torch.finfo(T.dtype).eps * 100
            scale = G.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1.0)
            reg = (eps * scale)[:, None, None] * eye
        sol = torch.linalg.solve(G + reg, H.mT @ y)
    return sol.reshape(m, p, ell)


def _run_als(
    T: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    n_iters: int,
    tol: float,
    lam: float,
    auto_ridge: bool,
    check_every: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
    """Run ALS from given initial factors, returning (A, B, C, loss_history)."""
    norm_T = torch.linalg.norm(T)
    cyc, cyc_inv = cyclic_transpose, cyclic_transpose_inv
    Tt = cyc(T).contiguous()
    Ttt = cyc(Tt).contiguous()

    history: list[float] = []
    for it in range(n_iters):
        B = _solve_middle(T, A, C, lam, auto_ridge)
        # C sits in the middle slot of the once-transposed problem.
        C = cyc_inv(_solve_middle(Tt, cyc(B), cyc(A), lam, auto_ridge)).contiguous()
        # A sits in the middle slot of the twice-transposed problem.
        A = cyc(
            _solve_middle(Ttt, cyc_inv(C), cyc_inv(B), lam, auto_ridge)
        ).contiguous()

        # The dense error check costs ~10% of a sweep at experiment scale, so
        # it can be sampled; the final sweep always records.
        if (it + 1) % check_every == 0 or it == n_iters - 1:
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
    check_every: int = 1,
) -> BMDFit:
    """Fit a BM decomposition via alternating least squares.

    Parameters
    ----------
    T:
        Input tensor of shape (m, p, n).
    rank:
        Target BM rank (ell).
    n_iters:
        Maximum ALS sweeps per start.
    tol:
        Convergence tolerance on the relative-error change between
        consecutive checks (see *check_every*).
    init:
        Starting point(s):

        * ``'ss_svd'`` -- per-slice SVD warm start plus *n_restarts* random
          restarts (recommended).
        * ``'mode1'`` -- mode-1 unfolding SVD warm start, same restart logic.
        * ``'random'`` -- single random start at *seed*; *n_restarts* ignored.
        * A 3-tuple ``(A0, B0, C0)`` of tensors -- used directly as the only
          starting point; shapes must match T and rank. *n_restarts* and
          *seed* are ignored.
    lam:
        Tikhonov regularisation weight (0 = no regularisation).
    seed:
        Random seed used when ``init='random'`` or as a base for random
        restarts.
    n_restarts:
        Number of independent random restarts added after a structured
        (``'ss_svd'``/``'mode1'``) warm start; the best run wins and the
        returned ``loss_history`` is the winning run's trajectory only.
        Set to 0 to use only the structured init.
    check_every:
        Record the relative error (and test convergence) every this many
        sweeps; the final sweep always records. Values > 1 skip most of the
        dense reconstruction cost during long fits.

    Notes
    -----
    Solver policy, resolved once and recorded as ``fit.solver``: exact lstsq
    is rank-revealing only on CPU; on accelerators torch's 'gels' driver
    assumes full rank and would silently produce garbage on rank-deficient
    blocks, so a scale-relative epsilon-ridge normal-equations solve is used
    there instead (relative, because an absolute epsilon vanishes against
    large-magnitude blocks and singular solves come back).
    """
    ell = int(rank)
    m, p, n = T.shape

    auto_ridge = lam == 0.0 and T.device.type != "cpu"
    if lam > 0:
        solver = f"tikhonov={lam:.3g}"
    elif auto_ridge:
        solver = "auto-ridge(rel)"
    else:
        solver = "lstsq"

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
        candidates = [(A0, B0, C0)]
    elif init == "random":
        candidates = [
            random_bmd_factors(m, p, n, ell, seed, dtype=T.dtype, device=T.device)
        ]
    elif init in ("ss_svd", "mode1"):
        structured = (ss_svd_init if init == "ss_svd" else mode1_init)(T, ell)
        # Restart seeds are offset far from small user/data seeds so a restart
        # can never regenerate a data-generating seed's exact factors.
        candidates = [structured] + [
            random_bmd_factors(
                m, p, n, ell, 100_003 + 7919 * i + seed, dtype=T.dtype, device=T.device
            )
            for i in range(n_restarts)
        ]
    else:
        raise ValueError(f"unknown init {init!r}")

    best: BMDFit | None = None
    for A0, B0, C0 in candidates:
        A, B, C, history = _run_als(
            T, A0, B0, C0, n_iters, tol, lam, auto_ridge, check_every
        )
        if best is None or history[-1] < best.loss_history[-1]:
            best = BMDFit(A, B, C, history, solver=solver)
    return best
