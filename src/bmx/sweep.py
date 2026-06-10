"""Matched-parameter decomposition sweep: the shared engine of a2/a3/c2.

Output rows are keyed by (model, layer, object, method, rank, params) so
distribution-over-layers reporting is the default downstream."""

import time

import pandas as pd

import bmx.decomp  # noqa: F401  (registers all methods)
from bmx.decomp.base import get_method
from bmx.stacks.base import Stack


def decomp_sweep(
    stack: Stack,
    plan: dict[str, list],
    fit_opts: dict[str, dict] | None = None,
    extra_cols: dict | None = None,
) -> pd.DataFrame:
    fit_opts = fit_opts or {}
    rows = []
    for method, ranks in plan.items():
        fn = get_method(method)
        for rank in ranks:
            t0 = time.perf_counter()
            fit = fn(stack.tensor, rank, **fit_opts.get(method, {}))
            seconds = time.perf_counter() - t0
            rows.append(
                {
                    "model": stack.model,
                    "layer": stack.layer,
                    "object": stack.object_name,
                    "method": method,
                    "rank": str(rank),
                    "params": fit.param_count(),
                    "rel_error": fit.relative_error(stack.tensor),
                    "seconds": seconds,
                    "n_iters": len(fit.loss_history),
                }
                | (extra_cols or {})
            )
    return pd.DataFrame(rows)
