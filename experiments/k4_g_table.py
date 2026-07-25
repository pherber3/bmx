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

K4 Lloyd-gate design (2026-07-25, `docs/superpowers/specs/2026-07-25-k4-lloyd-
gate-design.md`): `Config.quantizer` ("rtn" default, bit-exact-pinned |
"lloyd") threads through the SAME calibration pipeline above -- the per-tier
quantize call swaps from `rtn_quantize` (uniform step, mse_scale=True) to the
analytic Gaussian Lloyd-Max codebook alternating-minimization quantizer
(mirroring `spectral._lloyd_quantize_packed`'s assign/refit pattern, applied
here to the UNPACKED (dequantized) form since this script measures distortion
directly, not containers). Every row gains a `quantizer` column and an
`analytic_gaussian` column: the CLOSED, deterministic Gaussian-Lloyd
reference distortion for that tier (quantize a large fixed-seed N(0,1) sample
against `gaussian_codebook(bits)` and report its MSE -- unit-variance source,
so the MSE IS the relative distortion, directly comparable to g_hat). A row
is `sampling_limited=True` when `g_hat < analytic_gaussian`: for a truly
Gaussian source this is impossible for a real quantizer against the
Gaussian-OPTIMAL reference and flags undersampled tail bins — but the
lloyd-gate run (2026-07-24) showed the metric-dominant top directions are
sub-Gaussian in the lambda-weighted sense (lambda-weighted Pearson kurtosis
2.58; a top-few/weighted statement, NOT uniform across the tier-8 tranche —
see docs/2026-07-25-k4-lloyd-gate-results.md §2, the authoritative
statement), and a group-adaptive uniform quantizer legitimately beats the
GAUSSIAN-Lloyd reference on such sources. Read the flag as
"sub-Gaussian source OR sampling-limited" — it cannot distinguish the two
without per-direction shape statistics; do NOT auto-pin analytic values
over flagged measurements (the 2026-07-25 lloyd-gate results doc records
the corrected inference). Both new
columns are ADDITIVE -- `quantizer="rtn"` (default) reproduces every
pre-existing column byte-identically (pinned by
`test_k4_g_table_rtn_default_byte_identical_to_prior`).
"""

from __future__ import annotations

import dataclasses
import functools
import json

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics
from bmx.cache.codecs import _tier_g, gaussian_codebook
from bmx.cache.collect import to_matrix
from bmx.cache.spectral import QUANTIZERS, load_packs
from bmx.quant.rtn import rtn_quantize
from experiments._k4_common import load_layer_keys

# The analytic-reference sample: fixed seed, n >= 2e6, fp64 standard normal --
# deterministic and cached per (bits, seed) so repeated tiers/layers/caches
# within (and across) a run never re-draw or re-quantize it.
_ANALYTIC_N_SAMPLES = 2_000_000
_ANALYTIC_SEED = 0


@functools.lru_cache(maxsize=16)
def _analytic_gaussian_distortion(bits: int) -> float:
    """Closed reference: MSE of quantizing a large fixed-seed N(0,1) fp64
    sample against the analytic Gaussian Lloyd-Max codebook for `bits`. The
    source has unit variance, so this MSE already equals a RELATIVE
    distortion -- directly comparable to g_hat (which normalizes by the
    empirical per-direction energy). Deterministic; cached per tier."""
    g = torch.Generator().manual_seed(_ANALYTIC_SEED)
    x = torch.randn(_ANALYTIC_N_SAMPLES, generator=g, dtype=torch.float64)
    cb = gaussian_codebook(bits).double()  # (2**bits,) sorted, unit-variance-fit
    mid = (cb[:-1] + cb[1:]) / 2
    idx = torch.bucketize(x, mid)
    x_hat = cb[idx]
    return float(((x - x_hat) ** 2).mean())


def _lloyd_quantize_unpacked(
    Y: torch.Tensor, bits: int, group_size: int
) -> torch.Tensor:
    """Groupwise dequantized reconstruction of Y against the analytic
    Gaussian Lloyd-Max codebook -- the unpacked-form twin of
    `spectral._lloyd_quantize_packed`/`_lloyd_dequantize_packed` (this script
    measures distortion directly on the dequantized reconstruction, never
    stores containers, so there is no packed-format concern). SAME
    alternating-minimization (assign <-> refit scale), deterministic, fp32,
    3 iterations from a group-std init -- mirrors `rtn_quantize`'s call
    convention exactly: `(..., d) -> (..., d)` dequantized values.
    """
    *lead, d = Y.shape
    assert d % group_size == 0, f"dim {d} not divisible by group {group_size}"
    cb = gaussian_codebook(bits).to(device=Y.device, dtype=Y.dtype)
    mid = (cb[:-1] + cb[1:]) / 2
    G = Y.reshape(*lead, d // group_size, group_size)
    scale = G.std(dim=-1, keepdim=True).clamp_min(1e-12)
    codes = torch.bucketize((G / scale).contiguous(), mid)
    for _ in range(3):
        level = cb[codes.long()]
        num = (G * level).sum(dim=-1, keepdim=True)
        den = (level * level).sum(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = (num / den).clamp_min(1e-12)
        codes = torch.bucketize((G / scale).contiguous(), mid)
    level = cb[codes.long()]
    return (level * scale).reshape(Y.shape)


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
    # K4 Lloyd-gate design: "rtn" (default, bit-exact-pinned) reproduces the
    # historical calibration pipeline exactly. "lloyd" swaps the per-tier
    # quantizer for the analytic Gaussian Lloyd-Max codebook -- same
    # pipeline, same pooling, same grid-convexity validation.
    quantizer: str = "rtn"


def main(cfg: Config):
    assert cfg.corpus_cache_paths
    assert cfg.quantizer in QUANTIZERS, (
        f"quantizer must be one of {QUANTIZERS}; got {cfg.quantizer!r}"
    )
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
            if cfg.quantizer == "lloyd":
                Y_hat = _lloyd_quantize_unpacked(
                    Y[:, keep].float().mT, t, cfg.group
                ).mT.double()
            else:
                Y_hat = rtn_quantize(
                    Y[:, keep].float().mT, t, cfg.group, mse_scale=True
                ).mT.double()
            r = ((Y[:, keep] - Y_hat) ** 2).mean(dim=0) / energy[keep]
            pooled[t].append(r)
            g_hat_t = float(r.mean())
            analytic_t = _analytic_gaussian_distortion(t)
            rows.append(
                dict(
                    model=cfg.model_label or "unknown",
                    layer=layer_i,
                    tier=t,
                    g_hat=g_hat_t,
                    p10=float(r.quantile(0.10)),
                    p90=float(r.quantile(0.90)),
                    n_dirs=int(keep.sum()),
                    quantizer=cfg.quantizer,
                    analytic_gaussian=analytic_t,
                    sampling_limited=bool(g_hat_t < analytic_t),
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

    analytic_gaussian_table = [
        1.0 if t == 0 else _analytic_gaussian_distortion(t) for t in cfg.tiers
    ]
    out = dict(
        tiers=list(cfg.tiers),
        g_table=table,
        quantizer=cfg.quantizer,
        analytic_gaussian_table=analytic_gaussian_table,
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
