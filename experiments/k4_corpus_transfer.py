"""K4 corpus-transfer gate: fit-corpus × eval-corpus win matrix + mechanism
decomposition (spec: docs/superpowers/specs/2026-07-23-k4-corpus-transfer-design.md).

Cells: plain matrix fit ∈ {wiki, code, null} × eval ∈ {wiki, code} (this
task); hybrid basis-A+alloc-B and W-cross Σ_A×W_B (Task 5); per-rank overlap
/ tier-agreement / spectrum / analytic cross-retention diagnostics in
overlap.parquet (Task 6). "null" = token-shuffled wikitext, fit-side only.

Win metric (BINDING — do not "fix" into a matched-bpe constraint): per
(cell, budget, eval cache, layer), win = TQ curve (turboquant_mse on k_pre,
per eval cache + layer) log-interpolated at the pack's OWN
bpe_skeptic_deploy (bpe_model + skeptic_charge(C, DEPLOY_S, tiers,
c_used=pack.c_used)) ÷ the pack's tail distortion. Bits-normalized PER PACK,
so cross-fit win ratios stay fair even when packs' bpe differ slightly.

Verdict (spec §4): D = 1 − win(fit≠eval)/win(fit=eval), computed per eval
cache (win = mean over layers), then mean/min/max across the eval caches.
D < 10% → corpus-insensitive; D > 25% → domain-sensitive; between → reported
as measured. Null-fit ≈ wikitext-fit on the wiki eval side (D < 10%) raises
model_intrinsic_flag — the stronger "basis is model-intrinsic" claim. All
gpt2-scale numbers are MECHANISM verdicts only (yellow flag in the JSON).

Synthesis-order addendum (spec §3b): five FIT-SIDE-ONLY arms — shufcode
(token-permuted code slices), uniwiki/unicode (i.i.d. samples from each fit
slice's unigram histogram), biwiki/bicode (bigram-Markov samples) — join the
same matrix at matched budgets. Verdict gains per_budget[b]["synthesis"]
with the pre-registered order-ladder rules: (a) recipe-confirmed for eval
side E if D(uni_E->E) < 10%; (b) order-2 earns its keep if
D(uni_E->E) - D(bi_E->E) >= 0.5 * D(uni_E->E); no higher orders unless (b)
passes on BOTH sides. uni_* is the recipe estimator, shuf_* its
without-replacement control.
"""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.codecs import quantize_cache
from bmx.cache.spectral import (
    basis_alloc_moment,
    fit_spectral_basis,
    pack_from_basis,
    skeptic_charge,
    spectral_quantize,
)
from bmx.census import subspace_overlap
from bmx.quant.hadamard import orthogonalize
from experiments._k4_common import (
    DEPLOY_S,
    CorpusFit,
    _layer_ctx,
    _log_interp,
    _score_tail,
    _tq_layer_curve,
    corpus_fit_bases,
    load_layer_keys,
    setup_rope,
)

FIT_CORPORA = ("wiki", "code", "null")
EVAL_CORPORA = ("wiki", "code")
# §3b synthesis-order fit arms (fit-side only; labels compose as
# f"uni{eval_side}" / f"bi{eval_side}" in the order-ladder rules).
SYNTH_FIT_CORPORA = ("shufcode", "uniwiki", "unicode", "biwiki", "bicode")
# (basis_corpus, alloc_corpus) — scored on the alloc corpus's eval side (H3).
_HYBRID_CELLS = (("wiki", "code"), ("code", "wiki"))
# (sigma_corpus, w_corpus) — scored on BOTH eval sides (binding decision 2).
_WCROSS_CELLS = (("wiki", "code"), ("code", "wiki"))

_PAIRS = (("wiki", "code"), ("wiki", "null"), ("code", "null"))


def _cache_label(path: str) -> str:
    return path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def _rank_overlap(dec_a: torch.Tensor, dec_b: torch.Tensor, r: int) -> float:
    """Mean squared principal cosine between the top-r reconstruction
    subspaces span(dec[:, :r]) of two fits, in [0, 1]."""
    return subspace_overlap(dec_a[:, :r].double(), orthogonalize(dec_b[:, :r].double()))


def _proxy_distortion(lam64: torch.Tensor, bits: torch.Tensor) -> float:
    """Gaussian rate-distortion proxy Σ_i lam_i · 4^(−bits_i) (same form as
    k4_fit_packs._distortion_curves)."""
    return float((lam64 * torch.pow(4.0, -bits.double())).sum())


def _diagnostics(
    fits: dict[str, CorpusFit], layers: list[int], cfg: Config
) -> pd.DataFrame:
    rows: list[dict] = []

    def emit(**kw):
        base = dict(
            kind="",
            pair="",
            corpus="",
            layer=-1,
            rank=-1,
            budget=float("nan"),
            tier=-1,
            centered=False,
            value=float("nan"),
        )
        base.update(kw)
        rows.append(base)

    C = fits["wiki"].bases[layers[0]].lam64.numel()
    ranks = [r for r in cfg.overlap_ranks if r <= C]
    assert ranks, f"no overlap_ranks <= C={C}"

    # Centered refits (Cov(k) instead of E[kkᵀ]), same whitener — H1 probe.
    centered_bases: dict[str, dict[int, object]] = {}
    for corpus in sorted({c for p in _PAIRS for c in p}):
        fit = fits[corpus]
        centered_bases[corpus] = {}
        for layer_i in layers:
            M = fit.M_fits[layer_i]
            Wh, Wh_inv = fit.whiteners[layer_i]
            centered_bases[corpus][layer_i] = fit_spectral_basis(
                M - M.mean(dim=0, keepdim=True), Wh, Wh_inv
            )

    for a, b in _PAIRS:
        for layer_i in layers:
            for r in ranks:
                emit(
                    kind="overlap",
                    pair=f"{a}-{b}",
                    layer=layer_i,
                    rank=r,
                    centered=False,
                    value=_rank_overlap(
                        fits[a].bases[layer_i].dec, fits[b].bases[layer_i].dec, r
                    ),
                )
                emit(
                    kind="overlap",
                    pair=f"{a}-{b}",
                    layer=layer_i,
                    rank=r,
                    centered=True,
                    value=_rank_overlap(
                        centered_bases[a][layer_i].dec,
                        centered_bases[b][layer_i].dec,
                        r,
                    ),
                )

    # Tier-map agreement, rank-index-aligned (both spectra descending).
    for a, b in _PAIRS:
        for layer_i in layers:
            for budget in cfg.budgets:
                bits_a = pack_from_basis(
                    fits[a].bases[layer_i], budget, tiers=cfg.tiers, group=cfg.group
                ).bits
                bits_b = pack_from_basis(
                    fits[b].bases[layer_i], budget, tiers=cfg.tiers, group=cfg.group
                ).bits
                for tier in cfg.tiers:
                    mask = bits_a == tier
                    if mask.any():
                        emit(
                            kind="tier_agreement",
                            pair=f"{a}-{b}",
                            layer=layer_i,
                            budget=float(budget),
                            tier=int(tier),
                            value=float((bits_b[mask] == tier).float().mean()),
                        )
                za, zb = bits_a == 0, bits_b == 0
                union = int((za | zb).sum())
                emit(
                    kind="zero_jaccard",
                    pair=f"{a}-{b}",
                    layer=layer_i,
                    budget=float(budget),
                    value=float((za & zb).sum()) / union if union else float("nan"),
                )

    # Eigenvalue-spectrum overlay.
    for corpus, fit in fits.items():
        for layer_i in layers:
            lam = fit.bases[layer_i].lam64
            for r in range(lam.numel()):
                emit(
                    kind="spectrum",
                    corpus=corpus,
                    layer=layer_i,
                    rank=r,
                    value=float(lam[r]),
                )

    # Analytic cross-corpus retention (Gate-A machinery pointed across
    # corpora): src basis+alloc under dst's covariance vs dst's own fit.
    # Never INTO null (nothing is evaluated on shuffled text, spec §2).
    for a, b in _PAIRS:
        for src, dst in ((a, b), (b, a)):
            if dst == "null":
                continue
            for layer_i in layers:
                for budget in cfg.budgets:
                    pack_src = pack_from_basis(
                        fits[src].bases[layer_i],
                        budget,
                        tiers=cfg.tiers,
                        group=cfg.group,
                    )
                    lam_dst_given_src = basis_alloc_moment(
                        fits[src].bases[layer_i], fits[dst].M_fits[layer_i]
                    )
                    D_cross = _proxy_distortion(lam_dst_given_src, pack_src.bits)
                    pack_dst = pack_from_basis(
                        fits[dst].bases[layer_i],
                        budget,
                        tiers=cfg.tiers,
                        group=cfg.group,
                    )
                    D_own = _proxy_distortion(
                        fits[dst].bases[layer_i].lam64, pack_dst.bits
                    )
                    emit(
                        kind="xretention",
                        pair=f"{src}->{dst}",
                        layer=layer_i,
                        budget=float(budget),
                        value=D_own / max(D_cross, 1e-300),
                    )

    cols = [
        "kind",
        "pair",
        "corpus",
        "layer",
        "rank",
        "budget",
        "tier",
        "centered",
        "value",
    ]
    return pd.DataFrame(rows)[cols]


@dataclasses.dataclass
class Config:
    wiki_fit_paths: tuple[str, ...]
    code_fit_paths: tuple[str, ...]
    null_fit_paths: tuple[str, ...]
    wiki_eval_paths: tuple[str, ...]
    code_eval_paths: tuple[str, ...]
    # ---- §3b synthesis-order fit arms (fit-side only; all five or none) ----
    shufcode_fit_paths: tuple[str, ...] = ()
    uniwiki_fit_paths: tuple[str, ...] = ()
    unicode_fit_paths: tuple[str, ...] = ()
    biwiki_fit_paths: tuple[str, ...] = ()
    bicode_fit_paths: tuple[str, ...] = ()
    model_label: str = ""
    model_name: str = ""  # HF repo id for RoPE; empty => no-RoPE (gpt2), headline=logit
    budgets: tuple[float, ...] = (2.2, 2.5)
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8)
    group: int = 64
    ridge: float = 1e-3
    position_stride: int = 8
    tq_bits: tuple[int, ...] = (2, 3, 4)
    overlap_ranks: tuple[int, ...] = (8, 16, 32, 64)  # Task 6 diagnostics
    seed: int = 0
    out_root: str = ""


def _load_side(paths: tuple[str, ...], model_name: str):
    """Load caches + per-cache RoPE. Returns (per_cache_layer_keys,
    get_cos_sins, rope_ready, layers)."""
    per_cache = [load_layer_keys(p) for p in paths]
    layers = sorted(per_cache[0].keys())
    for lk in per_cache[1:]:
        assert sorted(lk.keys()) == layers, "caches disagree on layer set"
    rope_ready, get_cos_sins = False, []
    for lk in per_cache:
        ready, gcs = setup_rope(model_name, lk, layers)
        rope_ready = rope_ready or ready
        get_cos_sins.append(gcs)
    return per_cache, get_cos_sins, rope_ready, layers


def _fit_rows(per_cache, layers) -> int:
    return sum(lk[layers[0]]["k_pre"].shape[1] for lk in per_cache)


def main(cfg: Config):
    fit_paths = {
        "wiki": cfg.wiki_fit_paths,
        "code": cfg.code_fit_paths,
        "null": cfg.null_fit_paths,
    }
    synth_paths = {
        "shufcode": cfg.shufcode_fit_paths,
        "uniwiki": cfg.uniwiki_fit_paths,
        "unicode": cfg.unicode_fit_paths,
        "biwiki": cfg.biwiki_fit_paths,
        "bicode": cfg.bicode_fit_paths,
    }
    n_synth = sum(1 for p in synth_paths.values() if p)
    assert n_synth in (0, len(synth_paths)), (
        "§3b synthesis arms are all-or-nothing (the order-ladder rules need "
        f"all five): got {n_synth}/5 non-empty "
        f"{ {k: len(v) for k, v in synth_paths.items()} }"
    )
    if n_synth:
        fit_paths.update(synth_paths)
    for name, paths in fit_paths.items():
        assert paths, f"{name}_fit_paths must be non-empty"
    assert cfg.wiki_eval_paths and cfg.code_eval_paths, "eval paths must be non-empty"

    run = (
        create_run("k4_corpus_transfer", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_corpus_transfer", cfg)
    )
    model_label = cfg.model_label or "unknown"

    # ---- fit side: one CorpusFit per corpus, matched budgets asserted ------
    fits: dict[str, CorpusFit] = {}
    layers = None
    fit_row_counts: dict[str, int] = {}
    for name, paths in fit_paths.items():
        per_cache, get_cos_sins, rope_ready, this_layers = _load_side(
            paths, cfg.model_name
        )
        if layers is None:
            layers = this_layers
        assert this_layers == layers, f"{name} fit caches disagree on layer set"
        fit_row_counts[name] = _fit_rows(per_cache, layers)
        print(f"\n== fitting corpus {name!r} ({len(paths)} caches) ==", flush=True)
        fits[name] = corpus_fit_bases(
            per_cache,
            get_cos_sins,
            rope_ready,
            layers,
            w_source="corpus",
            ridge=cfg.ridge,
            position_stride=cfg.position_stride,
        )
    # Binding decision 1: matched fit-token budgets across corpora.
    assert len({len(p) for p in fit_paths.values()}) == 1, (
        f"matched fit budgets violated: slice counts "
        f"{ {k: len(v) for k, v in fit_paths.items()} }"
    )
    assert len(set(fit_row_counts.values())) == 1, (
        f"matched fit budgets violated: total fit rows {fit_row_counts}"
    )

    # ---- eval side: per-cache layer ctxs (built once, reused by all arms) --
    eval_paths = {"wiki": cfg.wiki_eval_paths, "code": cfg.code_eval_paths}
    # ctxs[(eval_corpus, cache_label)][layer] = _LayerCtx; rope flags per cache
    ctxs: dict[tuple[str, str], dict] = {}
    cache_rope: dict[tuple[str, str], bool] = {}
    any_rope_ready = False
    for eval_c, paths in eval_paths.items():
        for path in paths:
            label = _cache_label(path)
            layer_keys = load_layer_keys(path)
            assert sorted(layer_keys.keys()) == layers, (
                f"eval cache {label} layer set mismatch"
            )
            rope_ready, get_cos_sin = setup_rope(cfg.model_name, layer_keys, layers)
            any_rope_ready = any_rope_ready or rope_ready
            cache_rope[(eval_c, label)] = rope_ready
            ctxs[(eval_c, label)] = {
                layer_i: _layer_ctx(
                    layer_keys[layer_i],
                    rope_ready=rope_ready,
                    get_cos_sin=get_cos_sin,
                )
                for layer_i in layers
            }
    headline_col = "logit_rope" if any_rope_ready else "logit"

    rows: list[dict] = []
    tq_rows: list[dict] = []

    def emit(dest, **kw):
        base = dict(
            model=model_label,
            kind="k_pre",  # _tq_layer_curve filters on kind == "k_pre"
            fit_corpus="",
            w_corpus="",
            alloc_corpus="",
            eval_corpus="",
            cache="",
            layer=-1,
            arm="",
            budget=float("nan"),
            bpe_model=float("nan"),
            bpe_skeptic_deploy=float("nan"),
            c_used=float("nan"),
            rel_fro=float("nan"),
            logit=float("nan"),
            logit_rope=float("nan"),
        )
        base.update(kw)
        dest.append(base)
        print(
            f"  {base['arm']:16s} fit={base['fit_corpus']:4s} "
            f"eval={base['eval_corpus']:4s} cache={base['cache']:28s} "
            f"layer={base['layer']:2d} budget={base['budget']:5.2f} "
            f"logit={base['logit']:.4f}",
            flush=True,
        )

    def score_pack(pack, eval_c, label, layer_i, *, arm, fit_c, w_c, alloc_c, budget):
        ctx = ctxs[(eval_c, label)][layer_i]
        rope_ready = cache_rope[(eval_c, label)]
        assert pack.enc.shape == (ctx.C, ctx.C), (
            f"pack C mismatch at layer {layer_i}: {pack.enc.shape} vs C={ctx.C}"
        )
        M_hat, bpe_model = spectral_quantize(ctx.M_pre, pack)
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
        bpe_deploy = bpe_model + skeptic_charge(
            ctx.C, DEPLOY_S, cfg.tiers, c_used=pack.c_used
        )
        emit(
            rows,
            fit_corpus=fit_c,
            w_corpus=w_c,
            alloc_corpus=alloc_c,
            eval_corpus=eval_c,
            cache=label,
            layer=layer_i,
            arm=arm,
            budget=float(budget),
            bpe_model=bpe_model,
            bpe_skeptic_deploy=bpe_deploy,
            c_used=float(pack.c_used),
            rel_fro=rf,
            logit=lg,
            logit_rope=lg_rope,
        )

    # ---- TQ baseline curves, per (eval cache, layer), computed ONCE --------
    for (eval_c, label), layer_ctxs in ctxs.items():
        rope_ready = cache_rope[(eval_c, label)]
        for layer_i, ctx in layer_ctxs.items():
            for b in cfg.tq_bits:
                M_hat_tq, bpe_tq = quantize_cache(
                    "turboquant_mse", ctx.M_pre, bits=b, seed=cfg.seed
                )
                rf, lg, lg_rope, _lg_causal = _score_tail(
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
                emit(
                    tq_rows,
                    eval_corpus=eval_c,
                    cache=label,
                    layer=layer_i,
                    arm="turboquant_mse",
                    bpe_model=bpe_tq,
                    bpe_skeptic_deploy=bpe_tq,
                    rel_fro=rf,
                    logit=lg,
                    logit_rope=lg_rope,
                )

    # ---- plain matrix: fit ∈ fit_paths (natural + present synth) × every
    #      eval cache ---------------------------------------------------------
    for fit_c in fit_paths:
        for budget in cfg.budgets:
            for layer_i in layers:
                pack = pack_from_basis(
                    fits[fit_c].bases[layer_i],
                    budget,
                    tiers=cfg.tiers,
                    group=cfg.group,
                )
                for eval_c, paths in eval_paths.items():
                    for path in paths:
                        label = _cache_label(path)
                        score_pack(
                            pack,
                            eval_c,
                            label,
                            layer_i,
                            arm="spectral",
                            fit_c=fit_c,
                            w_c=fit_c,
                            alloc_c=fit_c,
                            budget=budget,
                        )

    # ---- hybrid (H3): basis from A, lam measured on B, waterfill rerun -----
    for basis_c, alloc_c in _HYBRID_CELLS:
        for budget in cfg.budgets:
            for layer_i in layers:
                lam_alloc = basis_alloc_moment(
                    fits[basis_c].bases[layer_i], fits[alloc_c].M_fits[layer_i]
                )
                pack = pack_from_basis(
                    fits[basis_c].bases[layer_i],
                    budget,
                    tiers=cfg.tiers,
                    group=cfg.group,
                    lam_alloc=lam_alloc,
                )
                for path in eval_paths[alloc_c]:
                    label = _cache_label(path)
                    score_pack(
                        pack,
                        alloc_c,
                        label,
                        layer_i,
                        arm="spectral_hybrid",
                        fit_c=basis_c,
                        w_c=basis_c,
                        alloc_c=alloc_c,
                        budget=budget,
                    )

    # ---- W-cross (binding decision 2): Σ from A, W (whitener) from B -------
    for sigma_c, w_c in _WCROSS_CELLS:
        for layer_i in layers:
            Wh_b, Wh_inv_b = fits[w_c].whiteners[layer_i]
            basis_x = fit_spectral_basis(fits[sigma_c].M_fits[layer_i], Wh_b, Wh_inv_b)
            for budget in cfg.budgets:
                pack = pack_from_basis(
                    basis_x, budget, tiers=cfg.tiers, group=cfg.group
                )
                for eval_c, paths in eval_paths.items():
                    for path in paths:
                        label = _cache_label(path)
                        score_pack(
                            pack,
                            eval_c,
                            label,
                            layer_i,
                            arm="spectral_wcross",
                            fit_c=sigma_c,
                            w_c=w_c,
                            alloc_c=sigma_c,
                            budget=budget,
                        )

    ov_df = _diagnostics(fits, layers, cfg)
    write_metrics(run, ov_df, name="overlap")
    print(f"overlap.parquet: {len(ov_df)} diagnostic rows", flush=True)

    cols = [
        "model",
        "kind",
        "fit_corpus",
        "w_corpus",
        "alloc_corpus",
        "eval_corpus",
        "cache",
        "layer",
        "arm",
        "budget",
        "bpe_model",
        "bpe_skeptic_deploy",
        "c_used",
        "rel_fro",
        "logit",
        "logit_rope",
    ]
    df = pd.DataFrame(rows)[cols]
    tq_df = pd.DataFrame(tq_rows)[cols]
    write_metrics(run, df)
    write_metrics(run, tq_df, name="tq_curve")

    verdict = _transfer_verdict(df, tq_df, headline_col, cfg)
    (run / "corpus_transfer_verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n" + "=" * 88)
    print("CORPUS-TRANSFER VERDICT (spec §4)")
    print("=" * 88)
    print(json.dumps(verdict, indent=2))
    print(f"\nTotal rows: {len(df)}")
    print(f"-> {run}")
    return run


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def _cell_wins(
    sub: pd.DataFrame, tq_curves: dict, headline_col: str
) -> tuple[dict[str, float], bool]:
    """Per eval cache: mean over layers of tq_dist(at the pack's OWN
    bpe_skeptic_deploy) / pack distortion (bits-normalized per pack —
    binding decision 3)."""
    wins: dict[str, list[float]] = {}
    extrapolated = False
    for _, row in sub.iterrows():
        pts = tq_curves.get((row.eval_corpus, row.cache), {}).get(int(row.layer))
        if not pts:
            continue
        tq_dist, ex = _log_interp(pts, float(row.bpe_skeptic_deploy))
        extrapolated = extrapolated or ex
        dist = max(float(row[headline_col]), 1e-300)
        wins.setdefault(row.cache, []).append(tq_dist / dist)
    return (
        {c: float(pd.Series(v).mean()) for c, v in wins.items()},
        extrapolated,
    )


def _transfer_verdict(
    df: pd.DataFrame, tq_df: pd.DataFrame, headline_col: str, cfg: Config
) -> dict:
    # TQ curves keyed per (eval_corpus, cache) — never pooled across caches
    # (same reasoning as k4_dec_quant._dec_quant_verdict) and never keyed by
    # bare cache basename, which would collide if eval filenames ever repeat
    # across corpora (e.g. both wiki_eval_paths and code_eval_paths pointing
    # at a cache named the same thing).
    tq_curves = {
        key: _tq_layer_curve(g, headline_col)
        for key, g in tq_df.groupby(["eval_corpus", "cache"])
    }
    present = set(df.loc[df.arm == "spectral", "fit_corpus"])
    fit_corpora = [c for c in FIT_CORPORA + SYNTH_FIT_CORPORA if c in present]

    per_budget: dict[str, dict] = {}
    for budget in cfg.budgets:
        cells: dict[str, dict] = {}
        for fit_c in fit_corpora:
            for eval_c in EVAL_CORPORA:
                sub = df[
                    (df.arm == "spectral")
                    & (df.budget == float(budget))
                    & (df.fit_corpus == fit_c)
                    & (df.eval_corpus == eval_c)
                ]
                if sub.empty:
                    continue
                wins, ex = _cell_wins(sub, tq_curves, headline_col)
                cells[f"{fit_c}->{eval_c}"] = dict(
                    win_per_cache=wins,
                    win_mean=float(pd.Series(list(wins.values())).mean()),
                    extrapolated=bool(ex),
                )

        D: dict[str, dict] = {}
        for eval_c in EVAL_CORPORA:
            matched = cells.get(f"{eval_c}->{eval_c}")
            if matched is None:
                continue
            for fit_c in fit_corpora:
                if fit_c == eval_c:
                    continue
                cross = cells.get(f"{fit_c}->{eval_c}")
                if cross is None:
                    continue
                ds = [
                    1.0 - cross["win_per_cache"][c] / matched["win_per_cache"][c]
                    for c in matched["win_per_cache"]
                    if c in cross["win_per_cache"]
                ]
                d_mean = float(pd.Series(ds).mean())
                label = (
                    "insensitive"
                    if d_mean < 0.10
                    else "domain-sensitive"
                    if d_mean > 0.25
                    else "as-measured"
                )
                D[f"{fit_c}->{eval_c}"] = dict(
                    mean=d_mean,
                    min=float(min(ds)),
                    max=float(max(ds)),
                    label=label,
                )

        hybrid: dict[str, dict] = {}
        for basis_c, alloc_c in _HYBRID_CELLS:
            sub = df[
                (df.arm == "spectral_hybrid")
                & (df.budget == float(budget))
                & (df.fit_corpus == basis_c)
                & (df.alloc_corpus == alloc_c)
            ]
            matched = cells.get(f"{alloc_c}->{alloc_c}")
            if sub.empty or matched is None:
                continue
            wins, ex = _cell_wins(sub, tq_curves, headline_col)
            win_mean = float(pd.Series(list(wins.values())).mean())
            recovery = win_mean / matched["win_mean"]
            hybrid[f"basis_{basis_c}_alloc_{alloc_c}"] = dict(
                win_mean=win_mean,
                recovery=recovery,
                h3_pass=bool(recovery >= 0.9),
                extrapolated=bool(ex),
            )

        wcross: dict[str, dict] = {}
        for sigma_c, w_c in _WCROSS_CELLS:
            for eval_c in EVAL_CORPORA:
                sub = df[
                    (df.arm == "spectral_wcross")
                    & (df.budget == float(budget))
                    & (df.fit_corpus == sigma_c)
                    & (df.w_corpus == w_c)
                    & (df.eval_corpus == eval_c)
                ]
                if sub.empty:
                    continue
                wins, ex = _cell_wins(sub, tq_curves, headline_col)
                wcross[f"sigma_{sigma_c}_W_{w_c}->{eval_c}"] = dict(
                    win_mean=float(pd.Series(list(wins.values())).mean()),
                    extrapolated=bool(ex),
                )

        # §3b synthesis-order rules — gates for the RECIPE claim (spec §3b).
        synthesis: dict = {}
        if any(c in fit_corpora for c in SYNTH_FIT_CORPORA):
            rules: dict[str, dict] = {}
            for eval_c in EVAL_CORPORA:
                d_uni = D.get(f"uni{eval_c}->{eval_c}", {}).get("mean")
                d_bi = D.get(f"bi{eval_c}->{eval_c}", {}).get("mean")
                if d_uni is None or d_bi is None:
                    continue
                shuf_cell = "shufcode->code" if eval_c == "code" else "null->wiki"
                rules[eval_c] = dict(
                    D_uni=d_uni,
                    D_bi=d_bi,
                    D_shuf=D.get(shuf_cell, {}).get("mean"),
                    # rule (a): the sampled-unigram recipe transfers on E
                    recipe_confirmed=bool(d_uni < 0.10),
                    # rule (b): bigram closes >= half the unigram gap on E
                    order2_earns_keep=bool((d_uni - d_bi) >= 0.5 * d_uni),
                )
            synthesis = dict(
                rules=rules,
                # "on BOTH eval sides" (note below) is binding: a missing D
                # cell drops that side out of `rules` above, so require full
                # coverage — a single-side pass must never license order-3
                # (2026-07-25 sweep: latent robustness gap, never fired in
                # the shipped pipeline where both cells always exist).
                climb_to_order3=bool(
                    len(rules) == len(EVAL_CORPORA)
                    and all(r["order2_earns_keep"] for r in rules.values())
                ),
                note=(
                    "§3b pre-registered gates for the traffic-histogram RECIPE "
                    "claim: uni_* is the recipe estimator, shuf_* its "
                    "without-replacement control (D_shuf for wiki = the Stage-1 "
                    "null->wiki cell); no higher orders unless order2_earns_keep "
                    "on BOTH eval sides"
                ),
            )

        null_wiki = D.get("null->wiki", {}).get("mean")
        per_budget[f"{budget:g}"] = dict(
            cells=cells,
            D=D,
            model_intrinsic_flag=bool(null_wiki is not None and null_wiki < 0.10),
            hybrid=hybrid,
            wcross=wcross,
            synthesis=synthesis,
        )

    return dict(
        headline_metric=headline_col,
        verdict_rule="D<0.10 insensitive; D>0.25 domain-sensitive; else as-measured",
        gpt2_yellow_flag=(
            "gpt2 scale = mechanism verdict only (corpus-W retention ~0.47-0.52, "
            "docs/2026-07-15-k4-duel-results.md); Llama fit-side replication "
            "pre-registered before any paper claim"
        ),
        per_budget=per_budget,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
