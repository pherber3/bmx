"""K4 estimation-levers Part 1: eigenvalue-shrinkage allocation-input gate
(finding #7, kill-or-confirm; spec
docs/superpowers/specs/2026-07-25-k4-estimation-levers-design.md Part 1).

Sample eigenvalues at deployment fit aspect ratios (gamma = C/n ~ 0.09-0.13)
are spread vs the population spectrum. `shrink_spectrum` (Task 1) shrinks the
waterfill ALLOCATION INPUT toward its own mean via Ledoit-Wolf (`lw`, needs
the fit rows in the basis's own whitened frame) or OAS (`oas`, spectrum-only)
-- consumed through the existing `pack_from_basis(lam_alloc=...)` hook. Basis,
stored lam, and every accounting expression are UNCHANGED; only the waterfill
input changes.

Per n_fit (subsample of the concatenated fit rows, seeded, BEFORE moment
building -- n_fit=0 means the full/standard fit, byte-identical to
`corpus_fit_bases`), fits one SpectralBasis per layer, then for each arm in
{plain (lam_alloc=None), lw, oas} packs at each budget and scores every
heldout cache with the A-gate/dec-quant win pattern: win = tq_interp(the
per-(cache,layer) turboquant_mse curve, AT THE PLAIN ARM's skeptic-v2 bpe
point) / dist_of_that_arm -- the matched-bpe-point convention from
k4_charge_alloc.py / k4_dec_quant.py, here matching lw/oas against plain's
own bpe per (layer, cache, budget, n_fit) so only the numerator's distortion
(via a DIFFERENT allocation, same basis) changes across arms.

THE GATE (pre-registered, binding, evaluates ONLY full-n lw vs full-n plain):
PROMOTE iff at BOTH budgets heldout win(lw) >= 1.02 * win(plain) AND no
matched-budget bpe_v2 regression > 0.02 (lw's own bpe_v2 must not exceed
plain's own bpe_v2 by more than 0.02 bits). Else HONEST NEGATIVE -- allocator
input stays raw, refit unaffected. oas is reported beside, never gated.

Diagnostics (reported, never gated): n-scaling (mean win by arm x n_fit x
budget -- the mechanism signature is improvement growing as n shrinks);
c_used stability (spread across n_fit, by arm); rho per layer/method/n_fit
(near 0 means "don't shrink" -- itself informative); 0<->2 tier-map deltas
between plain and lw at full n.
"""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics
from bmx.cache.codecs import quantize_cache
from bmx.cache.spectral import (
    SpectralBasis,
    fit_spectral_basis,
    pack_from_basis,
    shrink_spectrum,
    skeptic_charge,
    spectral_quantize,
)
from experiments._k4_common import (
    DEPLOY_S,
    PerCacheMoments,
    _layer_ctx,
    _log_interp,
    _score_tail,
    _tq_layer_curve,
    load_layer_keys,
    per_cache_weighted_moments,
    setup_rope,
)

_DEFAULT_FIT_PATHS = (
    "results/cache/gpt2_1024_off1024.safetensors",
    "results/cache/gpt2_1024_off2048.safetensors",
    "results/cache/gpt2_1024_off3072.safetensors",
    "results/cache/gpt2_1024_off4096.safetensors",
)
_DEFAULT_HELDOUT_PATHS = (
    "results/cache/gpt2_1024.safetensors",
    "results/cache/gpt2_1024_off5120.safetensors",
)


@dataclasses.dataclass
class Config:
    fit_cache_paths: tuple[str, ...] = _DEFAULT_FIT_PATHS
    heldout_cache_paths: tuple[str, ...] = _DEFAULT_HELDOUT_PATHS
    model_label: str = "gpt2"
    model_name: str = ""  # HF repo id for RoPE; empty => no-RoPE tables (gpt2)
    budgets: tuple[float, ...] = (2.2, 2.5)
    n_fits: tuple[int, ...] = (768, 1536, 3072, 0)  # 0 = full fit (standard path)
    methods: tuple[str, ...] = ("lw", "oas")
    tq_bits: tuple[int, ...] = (2, 3, 4)
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8)
    group: int = 64
    ridge: float = 1e-3
    position_stride: int = 8
    seed: int = 0
    out_root: str = ""

    # Pre-registered gate constants (spec Part 1) -- kept here so the verdict
    # JSON records them; not meant to be tuned per run.
    win_factor: float = 1.02
    bpe_guard: float = 0.02


def _subsample_rows(M: torch.Tensor, n_fit: int, seed: int) -> torch.Tensor:
    """n_fit=0 -> M unchanged (byte-identical to the standard fit path).
    Else a seeded random subset of n_fit rows (sorted indices, so row ORDER
    is preserved -- only which rows are kept is randomized)."""
    if n_fit == 0:
        return M
    n_total = M.shape[0]
    assert 0 < n_fit <= n_total, f"n_fit={n_fit} out of (0, {n_total}]"
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n_total, generator=g)[:n_fit].sort().values
    return M[idx]


def _fit_bases_at_n(
    per_cache_layer_keys,
    get_cos_sins,
    rope_ready: bool,
    layers: list[int],
    *,
    ridge: float,
    position_stride: int,
    n_fit: int,
    seed: int,
) -> dict[int, tuple[SpectralBasis, torch.Tensor]]:
    """Per-layer SpectralBasis fit at fit-row count `n_fit` (0 = full,
    byte-identical to `corpus_fit_bases`'s concat -- no subsampling applied),
    alongside the exact M_fit_sub used (needed by the caller to build the
    LW rows in the basis's own frame). Mirrors `corpus_fit_bases`'s body;
    the ONLY difference is the seeded row subsample inserted before
    `fit_spectral_basis`."""
    out: dict[int, tuple[SpectralBasis, torch.Tensor]] = {}
    for layer_i in layers:
        pcm: PerCacheMoments = per_cache_weighted_moments(
            per_cache_layer_keys,
            get_cos_sins,
            rope_ready,
            layer_i,
            w_source="corpus",
            ridge=ridge,
            position_stride=position_stride,
        )
        M_fit = torch.cat(pcm.M_parts, dim=0)
        M_fit_sub = _subsample_rows(M_fit, n_fit, seed + layer_i)
        basis = fit_spectral_basis(M_fit_sub, pcm.Wh, pcm.Wh_inv)
        out[layer_i] = (basis, M_fit_sub)
    return out


def _lw_rows(basis: SpectralBasis, M_fit_sub: torch.Tensor) -> torch.Tensor:
    """The fit rows projected into the basis's own whitened eigenbasis frame:
    rows = M_fit_sub @ enc (fp64). eigvalsh(rows^T rows / n) == basis.lam64
    (pinned by test_lw_rows_frame_pin_matches_basis_lam64) -- the frame
    shrink_spectrum's LW rho must be computed in (Task 1's report: rho is
    NOT invariant to the W-weighting)."""
    return M_fit_sub.double() @ basis.enc.double()


def _arm_lam_alloc(
    arm: str, basis: SpectralBasis, M_fit_sub: torch.Tensor, n_fit_actual: int
) -> tuple[torch.Tensor | None, float]:
    """(lam_alloc, rho) for one arm. "plain" -> (None, 0.0) -- pack_from_basis
    allocates on basis.lam64 unchanged, bit-exact. "lw"/"oas" -> shrunk
    spectrum + the estimated intensity."""
    if arm == "plain":
        return None, 0.0
    if arm == "lw":
        rows = _lw_rows(basis, M_fit_sub)
        shrunk, rho = shrink_spectrum(
            basis.lam64, n=n_fit_actual, method="lw", rows=rows
        )
        return shrunk, rho
    if arm == "oas":
        shrunk, rho = shrink_spectrum(basis.lam64, n=n_fit_actual, method="oas")
        return shrunk, rho
    raise AssertionError(f"unknown arm {arm!r}")


def main(cfg: Config):
    assert cfg.fit_cache_paths and cfg.heldout_cache_paths
    assert set(cfg.methods) <= {"lw", "oas"}, f"unknown methods: {cfg.methods}"

    run = (
        create_run("k4_shrinkage", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_shrinkage", cfg)
    )

    # ---- fit-fleet loading (once; subsampling happens per n_fit below) -----
    per_cache = [load_layer_keys(p) for p in cfg.fit_cache_paths]
    layers = sorted(per_cache[0].keys())
    for lk in per_cache[1:]:
        assert sorted(lk.keys()) == layers, "fit caches disagree on layer set"
    rope_ready = False
    get_cos_sins = []
    for lk in per_cache:
        ready, gcs = setup_rope(cfg.model_name, lk, layers)
        rope_ready = rope_ready or ready
        get_cos_sins.append(gcs)

    arms = ("plain",) + tuple(cfg.methods)

    # ---- per (n_fit, layer): basis + per-arm packs -------------------------
    packs: dict[tuple[int, str, float], dict[int, object]] = {}
    rho_rows: list[dict] = []
    tier_rows: list[dict] = []
    for n_fit in cfg.n_fits:
        fit_at_n = _fit_bases_at_n(
            per_cache,
            get_cos_sins,
            rope_ready,
            layers,
            ridge=cfg.ridge,
            position_stride=cfg.position_stride,
            n_fit=n_fit,
            seed=cfg.seed,
        )
        for layer_i in layers:
            basis, M_fit_sub = fit_at_n[layer_i]
            n_fit_actual = M_fit_sub.shape[0]
            for arm in arms:
                lam_alloc, rho = _arm_lam_alloc(arm, basis, M_fit_sub, n_fit_actual)
                method = "" if arm == "plain" else arm
                for budget in cfg.budgets:
                    pack = pack_from_basis(
                        basis,
                        budget,
                        tiers=cfg.tiers,
                        group=cfg.group,
                        lam_alloc=lam_alloc,
                    )
                    packs[(n_fit, arm, budget)] = packs.get((n_fit, arm, budget), {})
                    packs[(n_fit, arm, budget)][layer_i] = pack
                    rho_rows.append(
                        dict(
                            n_fit=n_fit,
                            layer=layer_i,
                            arm=arm,
                            method=method,
                            budget=float(budget),
                            rho=rho,
                            n_fit_actual=n_fit_actual,
                            c_used=int(pack.c_used),
                        )
                    )
                    if n_fit == 0:  # tier-map deltas: full n only (spec Part 1)
                        for t in cfg.tiers:
                            tier_rows.append(
                                dict(
                                    layer=layer_i,
                                    arm=arm,
                                    budget=float(budget),
                                    tier=t,
                                    n=int((pack.bits == t).sum()),
                                )
                            )

    # ---- scoring on heldout caches ------------------------------------------
    rows: list[dict] = []
    tq_rows: list[dict] = []
    any_rope = False
    for cache_path in cfg.heldout_cache_paths:
        cache_label = cache_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        layer_keys = load_layer_keys(cache_path)
        sc_layers = sorted(layer_keys.keys())
        c_ready, get_cos_sin = setup_rope(cfg.model_name, layer_keys, sc_layers)
        any_rope = any_rope or c_ready
        for layer_i in sc_layers:
            ctx = _layer_ctx(
                layer_keys[layer_i], rope_ready=c_ready, get_cos_sin=get_cos_sin
            )
            base = dict(model=cfg.model_label or "unknown", cache=cache_label)
            for b in cfg.tq_bits:
                M_hat, bpe = quantize_cache(
                    "turboquant_mse", ctx.M_pre, bits=b, seed=cfg.seed
                )
                rf, lg, lg_rope, _lg_causal = _score_tail(
                    M_hat,
                    ctx.h_kv,
                    ctx.tail,
                    ctx.K_post_true,
                    ctx.Q_fp32,
                    ctx.cos_l,
                    ctx.sin_l,
                    c_ready,
                    ctx.k_pre_t,
                    ctx.M_pre,
                )
                tq_rows.append(
                    dict(
                        base,
                        layer=layer_i,
                        arm="turboquant_mse",
                        kind="k_pre",
                        bpe_model=bpe,
                        rel_fro=rf,
                        logit=lg,
                        logit_rope=lg_rope,
                    )
                )
            for (n_fit, arm, budget), by_layer in packs.items():
                if layer_i not in by_layer:
                    continue
                pack = by_layer[layer_i]
                M_hat, bpe_model = spectral_quantize(ctx.M_pre, pack)
                rf, lg, lg_rope, _lg_causal = _score_tail(
                    M_hat,
                    ctx.h_kv,
                    ctx.tail,
                    ctx.K_post_true,
                    ctx.Q_fp32,
                    ctx.cos_l,
                    ctx.sin_l,
                    c_ready,
                    ctx.k_pre_t,
                    ctx.M_pre,
                )
                bpe_v2 = bpe_model + skeptic_charge(
                    ctx.C, DEPLOY_S, cfg.tiers, c_used=float(pack.c_used)
                )
                rows.append(
                    dict(
                        base,
                        layer=layer_i,
                        arm=arm,
                        method="" if arm == "plain" else arm,
                        n_fit=n_fit,
                        budget=float(budget),
                        bpe_model=bpe_model,
                        bpe_v2=bpe_v2,
                        c_used=float(pack.c_used),
                        rel_fro=rf,
                        logit=lg,
                        logit_rope=lg_rope,
                    )
                )
                print(
                    f"  cache={cache_label:16s} layer={layer_i:2d} arm={arm:6s} "
                    f"n_fit={n_fit:5d} b={budget:g} bpe_v2={bpe_v2:.3f} "
                    f"c_used={pack.c_used}",
                    flush=True,
                )

    headline = "logit_rope" if any_rope else "logit"
    df = pd.DataFrame(rows)
    tq_df = pd.DataFrame(tq_rows)
    rho_df = pd.DataFrame(rho_rows)
    tier_df = pd.DataFrame(tier_rows)

    # ---- win at the PLAIN arm's matched bpe_v2 point, per (n_fit, budget, --
    # ---- layer, cache) -------------------------------------------------------
    tq_curves = {
        cache: _tq_layer_curve(g.assign(kind="k_pre"), headline)
        for cache, g in tq_df.groupby("cache")
    }
    plain_bpe = df[df.arm == "plain"].set_index(["n_fit", "budget", "cache", "layer"])[
        "bpe_v2"
    ]

    wins = []
    extrapolated_any = False
    for _, r in df.iterrows():
        key = (r.n_fit, r.budget, r.cache, r.layer)
        if key not in plain_bpe.index:
            wins.append(float("nan"))
            continue
        bpe_at = float(plain_bpe.loc[key])
        pts = tq_curves.get(r.cache, {}).get(int(r.layer))
        if not pts:
            wins.append(float("nan"))
            continue
        tq_dist, ex = _log_interp(pts, bpe_at)
        extrapolated_any = extrapolated_any or ex
        wins.append(tq_dist / max(float(r[headline]), 1e-300))
    df = df.assign(win=wins)

    metrics_df = df.merge(
        rho_df[["n_fit", "layer", "arm", "budget", "rho"]],
        on=["n_fit", "layer", "arm", "budget"],
        how="left",
    )[
        [
            "model",
            "cache",
            "layer",
            "arm",
            "method",
            "budget",
            "n_fit",
            "win",
            "bpe_v2",
            "c_used",
            "rho",
        ]
    ]
    write_metrics(run, metrics_df)
    write_metrics(run, tq_df, name="tq_curve")
    write_metrics(run, rho_df, name="rho_diagnostics")
    if not tier_df.empty:
        write_metrics(run, tier_df, name="tier_map")

    verdict = _shrinkage_verdict(
        metrics_df,
        budgets=cfg.budgets,
        win_factor=cfg.win_factor,
        bpe_guard=cfg.bpe_guard,
        tier_df=tier_df if not tier_df.empty else None,
        extrapolated=extrapolated_any,
        headline_metric=headline,
    )
    (run / "shrinkage_verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n" + "=" * 88)
    print("SHRINKAGE VERDICT — eigenvalue-shrinkage allocation-input gate (finding #7)")
    print("=" * 88)
    print(json.dumps(verdict, indent=2))
    print(f"-> {run}")
    return run


def _shrinkage_verdict(
    metrics_df: pd.DataFrame,
    *,
    budgets: tuple[float, ...],
    win_factor: float,
    bpe_guard: float,
    tier_df: pd.DataFrame | None = None,
    extrapolated: bool = False,
    headline_metric: str = "",
) -> dict:
    """THE GATE (spec Part 1, binding): evaluates ONLY (full-n, lw) vs
    (full-n, plain). PROMOTE iff at BOTH budgets heldout win(lw) >=
    win_factor * win(plain) AND lw's own bpe_v2 does not exceed plain's own
    bpe_v2 by more than bpe_guard (matched-budget bpe regression guard).
    Everything else in the returned dict is a diagnostic, never gated.
    """
    full = metrics_df[metrics_df.n_fit == 0]

    gate: dict[str, dict] = {}
    gate_passes: list[bool] = []
    for budget in budgets:
        b_key = f"{budget:g}"
        sub = full[full.budget == budget]
        plain_win = float(sub[sub.arm == "plain"].win.mean())
        plain_bpe = float(sub[sub.arm == "plain"].bpe_v2.mean())
        lw_sub = sub[sub.arm == "lw"]
        if lw_sub.empty:
            continue
        lw_win = float(lw_sub.win.mean())
        lw_bpe = float(lw_sub.bpe_v2.mean())
        win_ratio = lw_win / max(plain_win, 1e-300)
        bpe_regression = lw_bpe - plain_bpe
        bpe_regression_ok = bool(bpe_regression <= bpe_guard)
        win_ok = bool(win_ratio >= win_factor)
        budget_pass = bool(win_ok and bpe_regression_ok)
        gate_passes.append(budget_pass)
        gate[b_key] = dict(
            plain_win=plain_win,
            lw_win=lw_win,
            win_ratio=win_ratio,
            win_ok=win_ok,
            plain_bpe_v2=plain_bpe,
            lw_bpe_v2=lw_bpe,
            bpe_regression=bpe_regression,
            bpe_regression_ok=bpe_regression_ok,
            gate_pass=budget_pass,
        )

    gate_pass = bool(gate_passes) and all(gate_passes)

    # ---- n-scaling: mean win by arm -> n_fit -> budget ----------------------
    n_scaling: dict[str, dict[str, dict[str, float]]] = {}
    for arm, g_arm in metrics_df.groupby("arm"):
        n_scaling[str(arm)] = {}
        for n_fit, g_n in g_arm.groupby("n_fit"):
            n_scaling[str(arm)][str(int(n_fit))] = {
                str(budget): float(g_b.win.mean())
                for budget, g_b in g_n.groupby("budget")
            }

    # ---- rho summary: mean rho by method -> n_fit ---------------------------
    rho_summary: dict[str, dict[str, float]] = {}
    for arm, g_arm in metrics_df[metrics_df.arm != "plain"].groupby("arm"):
        rho_summary[str(arm)] = {
            str(int(n_fit)): float(g_n.rho.mean())
            for n_fit, g_n in g_arm.groupby("n_fit")
        }

    # ---- c_used stability: mean/std across n_fit, by arm ---------------------
    c_used_stability: dict[str, dict[str, float]] = {}
    for arm, g_arm in metrics_df.groupby("arm"):
        by_n = g_arm.groupby("n_fit").c_used.mean()
        c_used_stability[str(arm)] = dict(
            mean_across_n_fit=float(by_n.mean()),
            std_across_n_fit=float(by_n.std(ddof=0)) if len(by_n) > 1 else 0.0,
        )

    result = dict(
        headline_metric=headline_metric,
        rule=(
            "Pre-registered (spec Part 1): at BOTH budgets, full-n heldout "
            "win(lw) >= win_factor * win(plain) AND lw's own bpe_v2 must not "
            "exceed plain's own bpe_v2 by more than bpe_guard. Evaluates ONLY "
            "(full-n, lw) vs (full-n, plain); oas and all n_fit<full rows are "
            "diagnostics, never gated."
        ),
        win_factor=win_factor,
        bpe_guard=bpe_guard,
        gate=gate,
        gate_pass=gate_pass,
        honest_negative=not gate_pass,
        n_scaling=n_scaling,
        rho_summary=rho_summary,
        c_used_stability=c_used_stability,
        extrapolated=bool(extrapolated),
        git_sha=git_sha(),
    )

    if tier_df is not None and not tier_df.empty:
        tier_map: dict[str, dict[str, float]] = {}
        for t in (0, 2):
            plain_n = tier_df[(tier_df.arm == "plain") & (tier_df.tier == t)].n.mean()
            deltas = {}
            for arm in sorted(set(tier_df.arm) - {"plain"}):
                arm_n = tier_df[(tier_df.arm == arm) & (tier_df.tier == t)].n.mean()
                deltas[arm] = float(arm_n - plain_n)
            tier_map[f"t{t}"] = deltas
        result["tier_map_deltas_full_n"] = tier_map

    return result


if __name__ == "__main__":
    main(tyro.cli(Config))
