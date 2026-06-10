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
    extra_cols = extra_cols or {}
    rows = []
    for method, ranks in plan.items():
        fn = get_method(method)
        for rank in ranks:
            t0 = time.perf_counter()
            fit = fn(stack.tensor, rank, **fit_opts.get(method, {}))
            seconds = time.perf_counter() - t0
            # Every fit records its final relative error in loss_history;
            # reusing it avoids a second dense reconstruction per fit.
            rel_error = (
                fit.loss_history[-1]
                if fit.loss_history
                else fit.relative_error(stack.tensor)
            )
            row = {
                "model": stack.model,
                "layer": stack.layer,
                "object": stack.object_name,
                "method": method,
                "rank": str(fit.rank),  # the fit's own rank, not the plan entry
                "params": fit.param_count(),
                "rel_error": rel_error,
                "seconds": seconds,
                "n_iters": len(fit.loss_history),
                "solver": getattr(fit, "solver", ""),
            }
            collisions = extra_cols.keys() & row.keys()
            assert not collisions, f"extra_cols would override {sorted(collisions)}"
            rows.append(row | extra_cols)
    return pd.DataFrame(rows)
