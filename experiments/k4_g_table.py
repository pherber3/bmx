"""K4 measured-g table (math review 2026-07-24 finding #4): ONE measurement of
the per-tier RTN distortion ratios g_hat(b) on calibration codes, replacing
the 4^{-b} model in the Lagrangian allocator (reported beside the A-gate,
never gated).

Procedure (transcribed from the review): per layer, project the calibration
rows into the shipped eigenbasis (Y = M_fit @ enc, fp32 — the codec path's
own arithmetic); groupwise-RTN every column at each tier (mse_scale=True,
the codec's step policy); per-direction relative distortion
mean((Y-Y_hat)_i^2)/mean(Y_i^2); g_hat(b) = pooled mean over kept directions
(energy > 1e-12 * max) and layers. g_hat(0) = 1 EXACT (a dropped direction's
error is its coordinate). The per-direction p10/p90 spread per tier is the
shared-shape audit (finding #4's fragile leg) — reported only.

The emitted table plugs into `k4_charge_alloc --g-table ...` (and any
`pack_from_basis(g_table=...)` call); `_tier_g` validates grid-convexity at
consumption time — this script also validates at emission time and fails
loudly if the measurement violates the optimality lemma's conditions.
"""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics
from bmx.cache.codecs import _tier_g
from bmx.cache.collect import to_matrix
from bmx.cache.spectral import load_packs
from bmx.quant.rtn import rtn_quantize
from experiments._k4_common import load_layer_keys


@dataclasses.dataclass
class Config:
    pack_path: str  # shipped pack file — enc per layer (budget-independent)
    corpus_cache_paths: tuple[str, ...]  # calibration rows (the fit slices)
    model_label: str = ""
    enc_budget: float = 2.5  # any budget stored in the pack file; enc is shared
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8)
    group: int = 64
    energy_floor: float = 1e-12
    out_root: str = ""


def main(cfg: Config):
    assert cfg.corpus_cache_paths
    run = (
        create_run("k4_g_table", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_g_table", cfg)
    )
    packs = load_packs(cfg.pack_path, cfg.enc_budget)
    per_cache = [load_layer_keys(p) for p in cfg.corpus_cache_paths]
    layers = sorted(per_cache[0].keys())

    rows: list[dict] = []
    pooled: dict[int, list[torch.Tensor]] = {t: [] for t in cfg.tiers if t > 0}
    n_rows_total = 0
    for layer_i in layers:
        M_fit = torch.cat([to_matrix(lk[layer_i]["k_pre"]) for lk in per_cache], dim=0)
        n_rows_total = max(n_rows_total, M_fit.shape[0])
        S = M_fit.shape[0]
        S_use = (S // cfg.group) * cfg.group  # rtn groups need divisibility
        assert S_use > 0, f"layer {layer_i}: too few rows for group={cfg.group}"
        Y = (M_fit[:S_use] @ packs[layer_i].enc).double()
        energy = (Y**2).mean(dim=0)
        keep = energy > cfg.energy_floor * float(energy.max())
        for t in cfg.tiers:
            if t == 0:
                continue
            Y_hat = rtn_quantize(
                Y[:, keep].float().mT, t, cfg.group, mse_scale=True
            ).mT.double()
            r = ((Y[:, keep] - Y_hat) ** 2).mean(dim=0) / energy[keep]
            pooled[t].append(r)
            rows.append(
                dict(
                    model=cfg.model_label or "unknown",
                    layer=layer_i,
                    tier=t,
                    g_hat=float(r.mean()),
                    p10=float(r.quantile(0.10)),
                    p90=float(r.quantile(0.90)),
                    n_dirs=int(keep.sum()),
                )
            )

    table = []
    spread = {}
    for t in cfg.tiers:
        if t == 0:
            table.append(1.0)  # exact — no quantizer model at the drop boundary
            continue
        all_r = torch.cat(pooled[t])
        table.append(float(all_r.mean()))
        spread[str(t)] = [float(all_r.quantile(0.10)), float(all_r.quantile(0.90))]

    # Fail loudly if the measurement violates the lemma's conditions.
    tiers_t = torch.tensor([float(t) for t in cfg.tiers], dtype=torch.float64)
    _tier_g(tiers_t, tuple(table))

    out = dict(
        tiers=list(cfg.tiers),
        g_table=table,
        n_rows=n_rows_total,
        spread_p10_p90_by_tier=spread,
        git_sha=git_sha(),
    )
    (run / "g_table.json").write_text(json.dumps(out, indent=2))
    write_metrics(run, pd.DataFrame(rows))
    print(json.dumps(out, indent=2))
    print(f"-> {run}")
    return run


if __name__ == "__main__":
    main(tyro.cli(Config))
