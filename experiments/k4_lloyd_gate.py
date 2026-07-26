"""K4 Lloyd payload-quantizer gate (Task 2 RUN): THE pre-registered
kill-or-confirm decision for the K-side Lloyd-Max codebook swap
(`docs/superpowers/specs/2026-07-25-k4-lloyd-gate-design.md`).

Same packs, same allocation, same bpe (identical by construction — a pure
quantizer swap at fixed allocation): per (cache, layer, budget), the SAME
fitted `SpectralPack` is quantized both ways — `spectral_quantize(...,
quantizer="rtn")` (today's uniform-step codec) and `quantizer="lloyd"` (the
analytic Gaussian Lloyd-Max codebook, Task 1) — and scored via `_score_tail`
on the tail region (headline `logit`; no causal, no RoPE — mirrors
`k4_dec_quant.py`'s no-causal structure, simpler than `k4_w_rope_ab.py`
since there is no basis refit here, only a quantizer swap on an existing
pack).

`bpe_model` is asserted identical across arms per (cache, layer, budget) —
not merely assumed (`_assert_bpe_identical`): `spectral_payload_bpe` reads
only `pack.bits`/`pack.group`/`pack.c_used`, never the quantizer, so this
assert is a construction check, not a live risk, but the design doc requires
verifying it, not assuming it.

win = TQ-curve interp at the shared skeptic-v2 deploy bpe point, per
(cache, layer) — the k4_dec_quant/k4_shrinkage convention: turboquant_mse
k_pre curves are keyed PER CACHE (never pooled — the documented bug class,
see `_tq_layer_curve`'s per-cache grouping requirement in those modules'
docstrings).

PRE-REGISTERED GATE (binding): PROMOTE iff heldout win(lloyd)/win(rtn) >=
1.02 at BOTH budgets AND the measured Lloyd g_table is grid-convex
(`bmx.cache.codecs._tier_g`'s assert, required for allocator validity).
Instruments (reported, never gated): the offline certificate — per pack,
predicted relative payload-distortion reduction
sum_i lam_i*(ghat_rtn(b_i) - ghat_lloyd(b_i)) / sum_i lam_i*ghat_rtn(b_i)
over the pack's OWN allocated bits, from the MEASURED g_tables (the
cheap-analytic-instrument agreement check, same pattern as the int8
certificate in k4_dec_quant.py).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics
from bmx.cache.codecs import _tier_g, quantize_cache
from bmx.cache.spectral import (
    SpectralPack,
    load_packs,
    skeptic_charge,
    spectral_quantize,
)
from experiments._k4_common import (
    DEPLOY_S,
    _layer_ctx,
    _log_interp,
    _score_tail,
    _tq_layer_curve,
    load_layer_keys,
    setup_rope,
)

ARMS = ("rtn", "lloyd")

_DEFAULT_TIERS: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8)


@dataclasses.dataclass
class Config:
    pack_path: str
    cache_paths: tuple[str, ...]  # heldout (scored) caches
    model_label: str = "gpt2"
    model_name: str = ""  # HF repo id; empty => stored-basis logit only (gpt2, no RoPE)
    budgets: tuple[float, ...] = (2.2, 2.5)
    tq_bits: tuple[int, ...] = (2, 3, 4)
    seed: int = 0
    out_root: str = ""
    # Offline certificate inputs (spec §"Instruments"): explicit run-dir
    # selection, no globs -- each path is a k4_g_table run's g_table.json
    # (or the run directory containing it), one for the rtn arm and one for
    # the lloyd arm. Empty (default) skips the certificate (still reports
    # measured win ratios; the certificate is an agreement check, never
    # gated, so its absence must not block the gate itself).
    g_table_rtn_path: str = ""
    g_table_lloyd_path: str = ""
    win_factor: float = 1.02


def _load_g_table(path: str) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "g_table.json"
    return json.loads(p.read_text())


def _assert_bpe_identical(df: pd.DataFrame, *, tol: float = 1e-9) -> None:
    """bpe_model must be identical across arms per (cache, layer, budget) —
    identical by construction (spectral_quantize's bpe accounting reads only
    pack.bits/pack.group/pack.c_used, never the quantizer). Verifies rather
    than assumes."""
    piv = df.pivot_table(
        index=["cache", "layer", "budget"], columns="arm", values="bpe_model"
    )
    missing = [a for a in ARMS if a not in piv.columns]
    assert not missing, f"missing arm(s) in frame: {missing}"
    delta = (piv["rtn"] - piv["lloyd"]).abs()
    bad = delta[delta > tol]
    assert bad.empty, (
        f"bpe_model mismatch across quantizer arms at (cache, layer, budget) "
        f"rows (should be identical by construction):\n{bad}"
    )


def _predicted_reduction(
    *,
    bits: torch.Tensor,
    lam: torch.Tensor,
    tiers: tuple[int, ...],
    g_table_rtn: tuple[float, ...],
    g_table_lloyd: tuple[float, ...],
) -> float:
    """Offline certificate (spec §"Instruments"): per pack, predicted relative
    payload-distortion reduction

        sum_i lam_i * (ghat_rtn(b_i) - ghat_lloyd(b_i)) / sum_i lam_i * ghat_rtn(b_i)

    evaluated over the pack's OWN allocated bits b_i (`pack.bits`), using the
    MEASURED per-tier g tables (from the g_table runs), index-aligned with
    `tiers`. `bits`/`lam` are 1-D, index-aligned per direction (same
    convention as `SpectralPack.bits`/`.lam`)."""
    assert bits.shape == lam.shape, (
        f"bits shape {tuple(bits.shape)} != lam shape {tuple(lam.shape)}"
    )
    assert len(g_table_rtn) == len(tiers) and len(g_table_lloyd) == len(tiers), (
        "g tables must be index-aligned with tiers"
    )
    g_rtn_by_tier = dict(zip(tiers, g_table_rtn))
    g_lloyd_by_tier = dict(zip(tiers, g_table_lloyd))
    lam64 = lam.double()
    num = 0.0
    den = 0.0
    for b_i, lam_i in zip(bits.tolist(), lam64.tolist()):
        b_i = int(b_i)
        assert b_i in g_rtn_by_tier, f"bit-width {b_i} not in tiers {tiers}"
        num += lam_i * (g_rtn_by_tier[b_i] - g_lloyd_by_tier[b_i])
        den += lam_i * g_rtn_by_tier[b_i]
    return num / den if den != 0.0 else float("nan")


def _lloyd_verdict(
    df: pd.DataFrame,
    *,
    budgets: tuple[float, ...],
    tiers: tuple[int, ...],
    g_table_rtn: tuple[float, ...] | None,
    g_table_lloyd: tuple[float, ...] | None,
    win_factor: float = 1.02,
    certificate_by_budget: dict[str, dict] | None = None,
) -> dict:
    """THE GATE (pre-registered, binding): PROMOTE iff at BOTH budgets
    heldout win(lloyd)/win(rtn) >= win_factor AND the measured Lloyd g_table
    is grid-convex (`_tier_g`'s assert). Everything else is a diagnostic,
    reported never gated."""
    per_budget: dict[str, dict] = {}
    gate_passes: list[bool] = []
    for budget in budgets:
        b_key = f"{budget:g}"
        sub = df[df.budget == float(budget)]
        if sub.empty:
            continue
        win_rtn = float(sub[sub.arm == "rtn"].win.mean())
        win_lloyd = float(sub[sub.arm == "lloyd"].win.mean())
        win_ratio = win_lloyd / max(win_rtn, 1e-300)
        budget_pass = bool(win_ratio >= win_factor)
        gate_passes.append(budget_pass)
        per_budget[b_key] = dict(
            win_rtn=win_rtn,
            win_lloyd=win_lloyd,
            win_ratio=win_ratio,
            gate_pass=budget_pass,
            n_samples=int((sub.arm == "rtn").sum()),
        )
        if certificate_by_budget is not None and b_key in certificate_by_budget:
            per_budget[b_key]["certificate"] = certificate_by_budget[b_key]

    # Convexity of the MEASURED Lloyd g table: required for allocator
    # validity (the optimality lemma). Catch _tier_g's assert and report a
    # bool rather than letting the whole run fail — the failure mode itself
    # is part of the gate's verdict, not an experiment crash.
    convex = None
    convex_error = None
    if g_table_lloyd is not None:
        tiers_t = torch.tensor([float(t) for t in tiers], dtype=torch.float64)
        try:
            _tier_g(tiers_t, tuple(g_table_lloyd))
            convex = True
        except AssertionError as e:
            convex = False
            convex_error = str(e)

    win_gate_pass = bool(gate_passes) and all(gate_passes)
    gate_pass = bool(win_gate_pass and (convex is True))

    return dict(
        rule=(
            "Pre-registered (spec 2026-07-25): PROMOTE iff heldout "
            "win(lloyd)/win(rtn) >= win_factor at BOTH budgets AND the "
            "measured Lloyd g_table is grid-convex (_tier_g assert). Fail "
            "=> honest negative; certificate/per-tier ratios reported, "
            "never gated."
        ),
        win_factor=win_factor,
        per_budget=per_budget,
        win_gate_pass=win_gate_pass,
        convex=convex,
        convex_error=convex_error,
        gate_pass=gate_pass,
        honest_negative=not gate_pass,
        g_table_rtn=list(g_table_rtn) if g_table_rtn is not None else None,
        g_table_lloyd=list(g_table_lloyd) if g_table_lloyd is not None else None,
        tiers=list(tiers),
    )


def main(cfg: Config):
    assert cfg.cache_paths, "cache_paths must be non-empty"
    assert cfg.budgets, "budgets must be non-empty"

    run = (
        create_run("k4_lloyd_gate", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_lloyd_gate", cfg)
    )

    model_label = cfg.model_label or "unknown"
    rows: list[dict] = []
    tq_rows: list[dict] = []
    headline_col = "logit"
    any_rope_ready = False
    # (cache, budget) -> {layer: pack} — retained for the certificate (needs
    # the actual SpectralPack.bits/.lam, not just the metrics rows).
    packs_by_cache_budget: dict[tuple[str, float], dict[int, SpectralPack]] = {}

    cache_labels = [
        cache_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        for cache_path in cfg.cache_paths
    ]
    for cache_path, cache_label in zip(cfg.cache_paths, cache_labels):
        layer_keys = load_layer_keys(cache_path)
        layers = sorted(layer_keys.keys())
        rope_ready, get_cos_sin = setup_rope(cfg.model_name, layer_keys, layers)
        any_rope_ready = any_rope_ready or rope_ready

        # ---- per (cache, layer) turboquant_mse k_pre curve, ONCE per cache --
        for layer_i in layers:
            ctx = _layer_ctx(
                layer_keys[layer_i], rope_ready=rope_ready, get_cos_sin=get_cos_sin
            )
            for b in cfg.tq_bits:
                M_hat_tq, bpe_tq = quantize_cache(
                    "turboquant_mse", ctx.M_pre, bits=b, seed=cfg.seed
                )
                rf_tq, lg_tq, lg_rope_tq, _lg_causal_tq = _score_tail(
                    M_hat_tq,
                    ctx.h_kv,
                    ctx.tail,
                    ctx.K_post_true,
                    ctx.Q_fp32,
                    ctx.cos_l,
                    ctx.sin_l,
                    rope_ready,
                    ctx.k_pre_t,
                    ctx.M_pre,
                )
                tq_rows.append(
                    dict(
                        model=model_label,
                        cache=cache_label,
                        layer=layer_i,
                        kind="k_pre",
                        arm="turboquant_mse",
                        budget=float("nan"),
                        bpe_model=bpe_tq,
                        bpe_skeptic_deploy=bpe_tq,
                        rel_fro=rf_tq,
                        logit=lg_tq,
                        logit_rope=lg_rope_tq,
                    )
                )

        for budget in cfg.budgets:
            packs = load_packs(cfg.pack_path, budget)
            packs_by_cache_budget[(cache_label, float(budget))] = packs
            for layer_i in layers:
                if layer_i not in packs:
                    continue
                ctx = _layer_ctx(
                    layer_keys[layer_i], rope_ready=rope_ready, get_cos_sin=get_cos_sin
                )
                pack = packs[layer_i]
                assert pack.enc.shape == (ctx.C, ctx.C), (
                    f"pack C mismatch at layer {layer_i}: {pack.enc.shape} vs C={ctx.C}"
                )

                for arm in ARMS:
                    M_hat, bpe_model = spectral_quantize(ctx.M_pre, pack, quantizer=arm)
                    rf, lg, lg_rope, _lg_causal = _score_tail(
                        M_hat,
                        ctx.h_kv,
                        ctx.tail,
                        ctx.K_post_true,
                        ctx.Q_fp32,
                        ctx.cos_l,
                        ctx.sin_l,
                        rope_ready,
                        ctx.k_pre_t,
                        ctx.M_pre,
                    )
                    bpe_skeptic_deploy = bpe_model + skeptic_charge(
                        ctx.C, DEPLOY_S, pack.tiers, c_used=pack.c_used
                    )
                    rows.append(
                        dict(
                            model=model_label,
                            cache=cache_label,
                            layer=layer_i,
                            kind="k_pre",
                            arm=arm,
                            budget=float(budget),
                            bpe_model=bpe_model,
                            bpe_skeptic_deploy=bpe_skeptic_deploy,
                            rel_fro=rf,
                            logit=lg,
                            logit_rope=lg_rope,
                        )
                    )
                    print(
                        f"  cache={cache_label:16s} layer={layer_i:2d} "
                        f"arm={arm:6s} b={budget:g} "
                        f"bpe={bpe_model:.3f} logit={lg:.6g}",
                        flush=True,
                    )

    headline_col = "logit_rope" if any_rope_ready else "logit"
    df = pd.DataFrame(rows)
    tq_df = pd.DataFrame(tq_rows)
    write_metrics(run, df)
    write_metrics(run, tq_df, name="tq_curve")

    # Construction check, not an assumption: bpe_model must match across arms
    # per (cache, layer, budget).
    _assert_bpe_identical(df)

    # ---- win = TQ-curve interp at the shared skeptic-v2 deploy bpe point, --
    # ---- per (cache, layer) -- tq_curves keyed PER CACHE, never pooled -----
    tq_curves = {
        cache: _tq_layer_curve(g, headline_col) for cache, g in tq_df.groupby("cache")
    }
    wins: list[float] = []
    extrapolated_any = False
    for _, r in df.iterrows():
        pts = tq_curves.get(r.cache, {}).get(int(r.layer))
        if not pts:
            wins.append(float("nan"))
            continue
        tq_dist, ex = _log_interp(pts, float(r.bpe_skeptic_deploy))
        extrapolated_any = extrapolated_any or ex
        wins.append(tq_dist / max(float(r[headline_col]), 1e-300))
    df = df.assign(win=wins)
    write_metrics(
        run,
        df[["model", "cache", "layer", "arm", "budget", "win", "bpe_model"]],
        name="win",
    )

    # ---- offline certificate (instrument, never gated) ---------------------
    g_rtn = g_lloyd = None
    g_tiers = _DEFAULT_TIERS
    certificate_by_budget: dict[str, dict] = {}
    if cfg.g_table_rtn_path and cfg.g_table_lloyd_path:
        gt_rtn = _load_g_table(cfg.g_table_rtn_path)
        gt_lloyd = _load_g_table(cfg.g_table_lloyd_path)
        assert tuple(gt_rtn["tiers"]) == tuple(gt_lloyd["tiers"]), (
            f"g_table tier grids differ: {gt_rtn['tiers']} vs {gt_lloyd['tiers']}"
        )
        g_tiers = tuple(gt_rtn["tiers"])
        g_rtn = tuple(gt_rtn["g_table"])
        g_lloyd = tuple(gt_lloyd["g_table"])
        for budget in cfg.budgets:
            b_key = f"{budget:g}"
            preds = []
            for (cache_label, b), packs in packs_by_cache_budget.items():
                if b != float(budget):
                    continue
                for layer_i, pack in packs.items():
                    assert tuple(pack.tiers) == g_tiers, (
                        f"pack tiers {pack.tiers} != g_table tiers {g_tiers}"
                    )
                    preds.append(
                        _predicted_reduction(
                            bits=pack.bits,
                            lam=pack.lam,
                            tiers=g_tiers,
                            g_table_rtn=g_rtn,
                            g_table_lloyd=g_lloyd,
                        )
                    )
            if preds:
                certificate_by_budget[b_key] = dict(
                    predicted_reduction_mean=float(pd.Series(preds).mean()),
                    predicted_reduction_min=float(min(preds)),
                    predicted_reduction_max=float(max(preds)),
                    n_packs=len(preds),
                )

    verdict = _lloyd_verdict(
        df,
        budgets=cfg.budgets,
        tiers=g_tiers,
        g_table_rtn=g_rtn,
        g_table_lloyd=g_lloyd,
        win_factor=cfg.win_factor,
        certificate_by_budget=certificate_by_budget or None,
    )
    verdict["headline_metric"] = headline_col
    verdict["extrapolated"] = bool(extrapolated_any)
    verdict["certificate"] = certificate_by_budget
    verdict["git_sha"] = git_sha()

    (run / "lloyd_verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n" + "=" * 88)
    print("LLOYD GATE VERDICT — K-side Lloyd-Max payload codebook vs uniform RTN")
    print("=" * 88)
    print(json.dumps(verdict, indent=2))
    print(f"-> {run}")
    return run


if __name__ == "__main__":
    main(tyro.cli(Config))
