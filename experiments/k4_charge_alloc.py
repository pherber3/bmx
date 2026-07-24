"""K4 A-gate: charge-aware (deployment-context-aware) allocation vs the plain
reverse-waterfill — math-review finding #2, pre-registered kill-or-confirm.

Fits plain packs (a budget grid spanning LOW total-charge values, so the plain
frontier can be interpolated at the charge-aware points) and charge-aware
packs (ca_budgets x s_refs) from the SAME corpus bases, scores everything on
heldout caches with the G1 instrument (tail-region distortion vs the
per-(cache, layer) turboquant_mse curve), and emits the pre-registered
verdict. Gate (binding, evaluated at S_ref=4096 points only):

  win_not_worse:  heldout G1 win of the charge-aware pack >= the plain
                  frontier's win interpolated at the SAME skeptic-v2
                  bpe@S_ref (quality not sacrificed), AND
  bits_saved_blended >= 0.4: (plain bpe@S_ref interpolated at the
                  charge-aware win) - (charge-aware bpe@S_ref), halved
                  (K-side only; blended = K/2, the duel convention).

a_gate_pass = the gate passes at ANY ca_budget's S_ref=4096 point; both
budgets fail => honest_negative (recorded; allocator stays as-is). At
S_ref=16384 both quantities are reported, never gated. Diagnostics (reported,
never gated): c_used vs s_ref, per-tier direction counts (the 0<->2 boundary
movement prediction), and every pack's (bpe, win) at each eval_s — the
bpe-vs-S frontier question (does optimizing for 4k hurt 64k?).

Accounting discipline: allocation-only change. payload/skeptic expressions
are the shipped ones (spectral_payload_bpe inside spectral_quantize +
skeptic_charge); c_used simply becomes smaller.
"""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics
from bmx.cache.codecs import quantize_cache
from bmx.cache.spectral import pack_from_basis, skeptic_charge, spectral_quantize
from experiments._k4_common import (
    _layer_ctx,
    _log_interp,
    _score_tail,
    _tq_layer_curve,
    corpus_fit_bases,
    load_layer_keys,
    setup_rope,
)


@dataclasses.dataclass
class Config:
    corpus_cache_paths: tuple[str, ...]
    cache_paths: tuple[str, ...]  # scored (heldout) caches
    model_label: str = ""
    model_name: str = ""  # HF repo id for RoPE; empty => no-RoPE tables (gpt2)
    # Plain grid spans low TOTAL-charge values so the plain frontier brackets
    # the charge-aware points (a CA pack at budget 2.5 sits at ~2.5 total
    # charge; a plain b2.5 pack sits at ~5.9 bpe@4k).
    plain_budgets: tuple[float, ...] = (1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0)
    ca_budgets: tuple[float, ...] = (2.2, 2.5)
    s_refs: tuple[int, ...] = (4096, 16384)
    eval_s: tuple[int, ...] = (4096, 8192, 16384, 32768, 65536)
    tq_bits: tuple[int, ...] = (2, 3, 4)
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8)
    group: int = 64
    ridge: float = 1e-3
    position_stride: int = 8
    g_table: tuple[float, ...] = ()  # empty = None (4^-b); else measured, tier-aligned
    seed: int = 0
    out_root: str = ""

    # Pre-registered gate constants (spec 2026-07-24 §A) — not CLI-tunable in
    # spirit; kept here so the verdict JSON records them.
    gate_s_ref: int = 4096
    gate_blended_bits: float = 0.4


def _arm_list(cfg: Config) -> list[tuple[str, float, int]]:
    """(arm, budget, s_ref) triples; s_ref == -1 is the plain sentinel."""
    arms = [("plain", b, -1) for b in cfg.plain_budgets]
    arms += [("charge_aware", b, s) for b in cfg.ca_budgets for s in cfg.s_refs]
    return arms


def main(cfg: Config):
    assert cfg.corpus_cache_paths and cfg.cache_paths
    gt = tuple(cfg.g_table) if cfg.g_table else None

    run = (
        create_run("k4_charge_alloc", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_charge_alloc", cfg)
    )

    # ---- corpus fit (one basis per layer, w_source="corpus") --------------
    per_cache = [load_layer_keys(p) for p in cfg.corpus_cache_paths]
    layers = sorted(per_cache[0].keys())
    for lk in per_cache[1:]:
        assert sorted(lk.keys()) == layers, "corpus caches disagree on layer set"
    rope_ready = False
    get_cos_sins = []
    for lk in per_cache:
        ready, gcs = setup_rope(cfg.model_name, lk, layers)
        rope_ready = rope_ready or ready
        get_cos_sins.append(gcs)
    fit = corpus_fit_bases(
        per_cache,
        get_cos_sins,
        rope_ready,
        layers,
        w_source="corpus",
        ridge=cfg.ridge,
        position_stride=cfg.position_stride,
    )

    # ---- packs + diagnostics ---------------------------------------------
    arms = _arm_list(cfg)
    packs: dict[tuple[str, float, int], dict[int, object]] = {}
    diag_rows: list[dict] = []
    for arm, budget, s_ref in arms:
        by_layer = {}
        for layer_i in layers:
            pack = pack_from_basis(
                fit.bases[layer_i],
                budget,
                tiers=cfg.tiers,
                group=cfg.group,
                s_ref=None if s_ref < 0 else s_ref,
                g_table=gt,
            )
            by_layer[layer_i] = pack
            row = dict(
                layer=layer_i,
                arm=arm,
                budget=float(budget),
                s_ref=int(s_ref),
                c_used=int(pack.c_used),
                mean_bits=float(pack.bits.double().mean()),
            )
            for t in cfg.tiers:
                row[f"n_t{t}"] = int((pack.bits == t).sum())
            diag_rows.append(row)
        packs[(arm, budget, s_ref)] = by_layer

    # ---- scoring on heldout caches -----------------------------------------
    rows: list[dict] = []
    tq_rows: list[dict] = []
    any_rope = False
    for cache_path in cfg.cache_paths:
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
                rf, lg, lg_rope = _score_tail(
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
                        budget=float("nan"),
                        s_ref=-1,
                        C=ctx.C,
                        bpe_model=bpe,
                        c_used=float("nan"),
                        rel_fro=rf,
                        logit=lg,
                        logit_rope=lg_rope,
                    )
                )
            for (arm, budget, s_ref), by_layer in packs.items():
                if layer_i not in by_layer:
                    continue
                pack = by_layer[layer_i]
                M_hat, bpe_model = spectral_quantize(ctx.M_pre, pack)
                rf, lg, lg_rope = _score_tail(
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
                rows.append(
                    dict(
                        base,
                        layer=layer_i,
                        arm=arm,
                        kind="k_pre",
                        budget=float(budget),
                        s_ref=int(s_ref),
                        C=ctx.C,
                        bpe_model=bpe_model,
                        c_used=float(pack.c_used),
                        rel_fro=rf,
                        logit=lg,
                        logit_rope=lg_rope,
                    )
                )
                print(
                    f"  cache={cache_label:16s} layer={layer_i:2d} arm={arm:12s} "
                    f"b={budget:g} s_ref={s_ref} bpe_model={bpe_model:.3f} "
                    f"c_used={pack.c_used}",
                    flush=True,
                )

    headline = "logit_rope" if any_rope else "logit"
    df = pd.DataFrame(rows)
    tq_df = pd.DataFrame(tq_rows)
    diag_df = pd.DataFrame(diag_rows)
    write_metrics(run, df)
    write_metrics(run, tq_df, name="tq_curve")
    write_metrics(run, diag_df, name="diagnostics")

    # ---- (bpe, win) point for every (arm, budget, s_ref) at every eval S --
    tq_curves = {
        cache: _tq_layer_curve(g.assign(kind="k_pre"), headline)
        for cache, g in tq_df.groupby("cache")
    }

    def point(arm: str, budget: float, s_ref: int, s_eval: int):
        sub = df[(df.arm == arm) & (df.budget == budget) & (df.s_ref == s_ref)]
        bpes, wins, extrap = [], [], False
        for _, r in sub.iterrows():
            pts = tq_curves.get(r.cache, {}).get(int(r.layer))
            if not pts:
                continue
            bpe = float(r.bpe_model) + skeptic_charge(
                int(r.C), s_eval, tuple(cfg.tiers), c_used=float(r.c_used)
            )
            tq_dist, ex = _log_interp(pts, bpe)
            extrap = extrap or ex
            wins.append(tq_dist / max(float(r[headline]), 1e-300))
            bpes.append(bpe)
        assert bpes, f"no scored rows for ({arm}, {budget}, {s_ref})"
        n = float(len(bpes))
        return sum(bpes) / n, sum(wins) / n, extrap

    frontier_rows = []
    for arm, budget, s_ref in arms:
        for s_eval in cfg.eval_s:
            bpe, win, ex = point(arm, budget, s_ref, s_eval)
            frontier_rows.append(
                dict(
                    arm=arm,
                    budget=float(budget),
                    s_ref=int(s_ref),
                    s_eval=int(s_eval),
                    bpe=bpe,
                    win=win,
                    extrapolated=ex,
                )
            )
    write_metrics(run, pd.DataFrame(frontier_rows), name="frontier")

    # ---- pre-registered verdict ------------------------------------------
    verdict = _verdict(cfg, point, diag_df, headline)
    (run / "charge_alloc_verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\n" + "=" * 88)
    print("A-GATE VERDICT — charge-aware allocation (math review #2)")
    print("=" * 88)
    print(json.dumps(verdict, indent=2))
    print(f"-> {run}")
    return run


def _verdict(cfg: Config, point, diag_df: pd.DataFrame, headline: str) -> dict:
    per_point: dict[str, dict] = {}
    gate_passes: list[bool] = []
    for s_ref in cfg.s_refs:
        # Plain frontier AT this S_ref: (bpe, win) per plain budget.
        plain_pts = sorted(point("plain", b, -1, s_ref)[:2] for b in cfg.plain_budgets)
        win_by_bpe = plain_pts  # [(bpe, win)] sorted by bpe
        bpe_by_win = sorted((w, b) for b, w in plain_pts)  # [(win, bpe)]
        frontier_monotone = all(
            win_by_bpe[i][1] <= win_by_bpe[i + 1][1] for i in range(len(win_by_bpe) - 1)
        )
        for budget in cfg.ca_budgets:
            bpe_ca, win_ca, ex_ca = point("charge_aware", budget, s_ref, s_ref)
            win_plain_at_bpe, ex1 = _log_interp(win_by_bpe, bpe_ca)
            bpe_plain_at_win, ex2 = _log_interp(bpe_by_win, win_ca)
            bits_saved_k = bpe_plain_at_win - bpe_ca
            bits_saved_blended = 0.5 * bits_saved_k
            win_not_worse = bool(win_ca >= win_plain_at_bpe)
            entry = dict(
                s_ref=int(s_ref),
                budget=float(budget),
                bpe_ca=bpe_ca,
                win_ca=win_ca,
                win_plain_at_matched_bpe=win_plain_at_bpe,
                win_not_worse=win_not_worse,
                bpe_plain_at_matched_win=bpe_plain_at_win,
                bits_saved_k_side=bits_saved_k,
                bits_saved_blended=bits_saved_blended,
                extrapolated=bool(ex_ca or ex1 or ex2),
                plain_frontier_monotone=bool(frontier_monotone),
            )
            if s_ref == cfg.gate_s_ref:
                entry["gate_pass"] = bool(
                    win_not_worse and bits_saved_blended >= cfg.gate_blended_bits
                )
                gate_passes.append(entry["gate_pass"])
            per_point[f"b{budget:g}_s{s_ref}"] = entry

    # c_used monotonicity diagnostic (mean over layers), per ca budget.
    c_used_diag = {}
    for budget in cfg.ca_budgets:
        by_s = {
            str(s): float(
                diag_df[
                    (diag_df.arm == "charge_aware")
                    & (diag_df.budget == budget)
                    & (diag_df.s_ref == s)
                ].c_used.mean()
            )
            for s in cfg.s_refs
        }
        by_s["plain"] = (
            float(
                diag_df[
                    (diag_df.arm == "plain") & (diag_df.budget == budget)
                ].c_used.mean()
            )
            if budget in cfg.plain_budgets
            else float("nan")
        )
        vals = [by_s[str(s)] for s in sorted(cfg.s_refs)]
        c_used_diag[f"b{budget:g}"] = dict(
            by_s_ref=by_s,
            monotone_decreasing=bool(
                all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))
            ),
        )

    a_gate_pass = bool(gate_passes) and any(gate_passes)
    return dict(
        headline_metric=headline,
        rule=(
            "Pre-registered (spec 2026-07-24 SA): per (budget, s_ref), at "
            "MATCHED skeptic-v2 bpe@S_ref the charge-aware heldout G1 win "
            "must be >= the plain pack's, AND skeptic-v2 bpe@S_ref at "
            "matched win must undercut plain by >= 0.4 blended bits at "
            "S_ref=4096 (blended = K-side/2). Both budgets fail => honest "
            "negative; allocator stays as-is. S_ref=16384 reported, not "
            "gated."
        ),
        gate_s_ref=cfg.gate_s_ref,
        gate_blended_bits=cfg.gate_blended_bits,
        g_table=list(cfg.g_table) if cfg.g_table else None,
        per_point=per_point,
        c_used_diagnostic=c_used_diag,
        a_gate_pass=a_gate_pass,
        honest_negative=not a_gate_pass,
        git_sha=git_sha(),
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
