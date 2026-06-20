"""Side-information break-even instrument (Shannon 4^-b accounting).

Scores a matrix for whether fp16 side-information ever pays against spending
the same bits on the bulk quantizer: side info costing Db bits/weight pays iff
the energy fraction eps it removes satisfies eps > 1 - 4^(-Db)
(docs/2026-06-11-lrs-results.md, theoretical postmortem). Used on weight
matrices (experiments/frontier_breakeven.py) and on cache activation matrices
(experiments/k1_cache_census.py).
"""

import math

import torch


def best_margin(
    eps: torch.Tensor, db: torch.Tensor, max_db: float
) -> tuple[float, int, float, float]:
    """Best (saved-bits - cost-bits) over a budget grid.

    eps: energy fraction captured at each grid point, db: side-info cost in
    bits/weight. Returns (margin, grid index, eps, db) at the argmax.
    """
    eps = eps.double().clamp(max=1 - 1e-9)
    saved = torch.log2(1.0 / (1.0 - eps)) / 2.0  # log4(x) = log2(x)/2
    margin = torch.where(db <= max_db, saved - db, saved.new_full((), -torch.inf))
    i = int(margin.argmax())
    return margin[i].item(), i, eps[i].item(), db[i].item()


def breakeven_row(W: torch.Tensor, max_side_bpw: float = 6.0) -> dict:
    """Low-rank and sparse break-even margins (effective bits/weight) for a
    2-D matrix, plus stable rank. Positive margin => that structure pays."""
    m, p = W.shape
    n = m * p
    w2_total = (W.double() ** 2).sum()

    # low-rank side: cumulative spectrum energy vs fp16 factor cost
    s2 = torch.linalg.svdvals(W).double() ** 2
    eps_r = s2.cumsum(0) / w2_total
    r = torch.arange(1, len(s2) + 1, dtype=torch.float64, device=W.device)
    db_r = 16.0 * r * (m + p) / n
    lr_margin, i, lr_eps, lr_db = best_margin(eps_r, db_r, max_side_bpw)

    # sparse side: cumulative top-|entry| energy vs fp16+index cost
    a2 = (W.flatten().double() ** 2).sort(descending=True).values
    idx_bits = (n - 1).bit_length()
    k_grid = torch.unique(
        torch.logspace(
            0, math.log10(n), steps=512, dtype=torch.float64, device=W.device
        ).long()
    )
    eps_k = a2.cumsum(0)[k_grid - 1] / w2_total
    db_k = k_grid.double() * (16 + idx_bits) / n
    sp_margin, j, sp_eps, sp_db = best_margin(eps_k, db_k, max_side_bpw)

    return {
        "lr_margin_bits": lr_margin,
        "lr_best_r": i + 1,
        "lr_eps": lr_eps,
        "lr_db": lr_db,
        "eps_r64": eps_r[min(63, len(eps_r) - 1)].item(),
        "sp_margin_bits": sp_margin,
        "sp_best_k": int(k_grid[j]),
        "sp_eps": sp_eps,
        "sp_db": sp_db,
        "stable_rank": (s2.sum() / s2[0].clamp_min(1e-12)).item(),
    }
