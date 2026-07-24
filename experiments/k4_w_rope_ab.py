"""K4 B-readout: w_rope A/B — frozen (shipped) vs rotated (causal-corrected)
query moment, math review 2026-07-24 finding #3.

Fits corpus bases BOTH ways on the same caches, scores both on heldout caches
with the G1 instrument, and reports (measurement, not a pass/fail gate):
per-budget heldout G1 win ratio rotated/frozen, and per-(layer, rank)
subspace overlap between the two bases' dec columns.

Pre-registered decision rule (spec 2026-07-24 SB), recorded in the verdict:
|rel_win_delta| < 2% at BOTH budgets -> decision "scoped_negligible" (the
paper scopes the claim: frozen-rotation approximation measured-negligible at
this scale; Llama spot-check queued for the rental). Otherwise ->
"rotated_form_required" (the paper uses the rotated form's numbers; Llama
refit rides the rental as REQUIRED). Either way the sign-flip footnote enters
the methods section.

SUBSTRATE: must run on a RoPE model (qwen3-0.6b locally) — on a no-RoPE
model (gpt2, model_name="") the two variants are mathematically identical
and the readout is a null control.
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
    DEPLOY_S,
    _layer_ctx,
    _log_interp,
    _score_tail,
    _tq_layer_curve,
    corpus_fit_bases,
    load_layer_keys,
    setup_rope,
)
from experiments.k4_corpus_transfer import _rank_overlap

W_ROPE_VARIANTS = ("frozen", "rotated")


@dataclasses.dataclass
class Config:
    corpus_cache_paths: tuple[str, ...]
    cache_paths: tuple[str, ...]  # scored (heldout) caches
    model_label: str = ""
    model_name: str = ""  # HF repo id; MUST be a RoPE model for a live A/B
    budgets: tuple[float, ...] = (2.2, 2.5)
    tq_bits: tuple[int, ...] = (2, 3, 4)
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8)
    group: int = 64
    ridge: float = 1e-3
    position_stride: int = 8
    overlap_ranks: tuple[int, ...] = (8, 16, 32, 64)
    seed: int = 0
    out_root: str = ""


def main(cfg: Config):
    assert cfg.corpus_cache_paths and cfg.cache_paths

    run = (
        create_run("k4_w_rope_ab", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_w_rope_ab", cfg)
    )

    per_cache = [load_layer_keys(p) for p in cfg.corpus_cache_paths]
    layers = sorted(per_cache[0].keys())
    rope_ready = False
    get_cos_sins = []
    for lk in per_cache:
        ready, gcs = setup_rope(cfg.model_name, lk, layers)
        rope_ready = rope_ready or ready
        get_cos_sins.append(gcs)

    fits = {
        variant: corpus_fit_bases(
            per_cache,
            get_cos_sins,
            rope_ready,
            layers,
            w_source="corpus",
            ridge=cfg.ridge,
            position_stride=cfg.position_stride,
            w_rope=variant,
        )
        for variant in W_ROPE_VARIANTS
    }

    # ---- per-(layer, rank) subspace overlap between the two bases --------
    ov_rows = [
        dict(
            layer=layer_i,
            rank=r,
            value=_rank_overlap(
                fits["frozen"].bases[layer_i].dec,
                fits["rotated"].bases[layer_i].dec,
                r,
            ),
        )
        for layer_i in layers
        for r in cfg.overlap_ranks
    ]
    write_metrics(run, pd.DataFrame(ov_rows), name="overlap")

    # ---- score both variants on heldout caches ---------------------------
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
                        kind="k_pre",
                        arm="turboquant_mse",
                        w_rope="",
                        budget=float("nan"),
                        bpe_model=bpe,
                        bpe_skeptic_deploy=bpe,
                        rel_fro=rf,
                        logit=lg,
                        logit_rope=lg_rope,
                    )
                )
            for variant in W_ROPE_VARIANTS:
                for budget in cfg.budgets:
                    pack = pack_from_basis(
                        fits[variant].bases[layer_i],
                        budget,
                        tiers=cfg.tiers,
                        group=cfg.group,
                    )
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
                    bpe_deploy = bpe_model + skeptic_charge(
                        ctx.C, DEPLOY_S, tuple(cfg.tiers), c_used=pack.c_used
                    )
                    rows.append(
                        dict(
                            base,
                            layer=layer_i,
                            kind="k_pre",
                            arm="spectral",
                            w_rope=variant,
                            budget=float(budget),
                            bpe_model=bpe_model,
                            bpe_skeptic_deploy=bpe_deploy,
                            rel_fro=rf,
                            logit=lg,
                            logit_rope=lg_rope,
                        )
                    )
                    print(
                        f"  cache={cache_label:16s} layer={layer_i:2d} "
                        f"w_rope={variant:8s} b={budget:g} "
                        f"bpe={bpe_model:.3f} logit={lg:.5f}",
                        flush=True,
                    )

    headline = "logit_rope" if any_rope else "logit"
    df = pd.DataFrame(rows)
    tq_df = pd.DataFrame(tq_rows)
    write_metrics(run, df)
    write_metrics(run, tq_df, name="tq_curve")

    # ---- verdict: win ratio per budget + the 2% decision rule -------------
    tq_curves = {
        cache: _tq_layer_curve(g, headline) for cache, g in tq_df.groupby("cache")
    }
    per_budget: dict[str, dict] = {}
    deltas: list[float] = []
    for budget in cfg.budgets:
        wins = {v: [] for v in W_ROPE_VARIANTS}
        for variant in W_ROPE_VARIANTS:
            sub = df[(df.w_rope == variant) & (df.budget == float(budget))]
            for _, r in sub.iterrows():
                pts = tq_curves.get(r.cache, {}).get(int(r.layer))
                if not pts:
                    continue
                tq_dist, _ = _log_interp(pts, float(r.bpe_skeptic_deploy))
                wins[variant].append(tq_dist / max(float(r[headline]), 1e-300))
        win_f = float(pd.Series(wins["frozen"]).mean())
        win_r = float(pd.Series(wins["rotated"]).mean())
        delta = win_r / win_f - 1.0
        deltas.append(delta)
        per_budget[f"{budget:g}"] = dict(
            win_frozen=win_f,
            win_rotated=win_r,
            rel_win_delta=delta,
            n_samples=len(wins["frozen"]),
        )
    scoped = all(abs(d) < 0.02 for d in deltas)
    ov_df = pd.DataFrame(ov_rows)
    verdict = dict(
        headline_metric=headline,
        rule=(
            "Pre-registered (spec 2026-07-24 SB): |rel_win_delta| < 2% at "
            "both budgets -> scoped_negligible (frozen-rotation approximation "
            "measured-negligible at this scale; Llama spot-check queued); "
            ">= 2% -> rotated_form_required (paper uses rotated numbers; "
            "Llama refit REQUIRED on the rental). Sign-flip footnote enters "
            "the methods section either way."
        ),
        per_budget=per_budget,
        decision="scoped_negligible" if scoped else "rotated_form_required",
        llama_refit_required=not scoped,
        overlap_mean_by_rank={
            str(r): float(ov_df[ov_df["rank"] == r].value.mean())
            for r in cfg.overlap_ranks
        },
        git_sha=git_sha(),
    )
    (run / "w_rope_verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\n" + "=" * 88)
    print("W_ROPE A/B VERDICT — frozen vs rotated query moment (math review #3)")
    print("=" * 88)
    print(json.dumps(verdict, indent=2))
    print(f"-> {run}")
    return run


if __name__ == "__main__":
    main(tyro.cli(Config))
