"""K4 Stage 1: the frontier duel (gate G1) — spectral codec vs turboquant_mse.

Per layer, on `k_pre`, tail-region-scored (rows [S//2:], like Task 6, so every
arm — spectral AND baselines — is compared apples-to-apples): fits the K4
spectral codec (weighted × {oracle, heldout, corpus?}), its two ablation
controls (unweighted W=I, and a random-orthogonal-basis load-bearing-
eigenstructure control), and the incumbent baseline family (turboquant_mse,
lowrank_rtn_channel, lowrank_turboquant, plus two experiment-local arms:
k2t_coeffquant for P3 and rtn_channel for the Task-1 step-policy ablation).

G1 verdict: for each (weighted=True, fit_mode, budget) spectral point,
interpolate log(distortion) linearly in bpe against the turboquant_mse k_pre
curve (3 points, b in {2,3,4}) PER LAYER, at both bpe_model and
bpe_skeptic_deploy. win = tq_interp / spectral, on layer means; g1_pass
requires win > 1 at every budget in both accounting modes AND >=90% of
layers beating the per-layer interpolation. See
docs/superpowers/specs/2026-07-12-k4-spectral-codec-design.md §4.

W note (query second moment that defines the whitener): `w_source` selects
"scored" (default, byte-identical to prior behavior) — W from the SCORED
cache's own queries, the upper-bound/circular variant (optimistic: the same
sequence being scored supplies its own W) — or "corpus" — W averaged over
each corpus cache's own queries and cos/sin tables (equal-weight per cache,
same convention as k4_fit_packs.py), the deployment-grade query-heldout
variant.
"""

from __future__ import annotations

import dataclasses
import json
import math

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.codecs import quantize_cache
from bmx.cache.collect import to_matrix
from bmx.cache.rope import apply_rope
from bmx.cache.spectral import (
    SpectralBasis,
    SpectralPack,
    assemble_whitener,
    fit_spectral_basis,
    identity_whitener,
    pack_from_basis,
    query_position_moment,
    skeptic_charge,
    spectral_quantize,
)
from bmx.decomp.lrs import truncated_svd
from bmx.quant.hadamard import random_orthogonal
from bmx.quant.rtn import rtn_quantize
from experiments._k4_common import (
    DEPLOY_S,
    _score_tail,
    bucket_layer_keys,
    corpus_query_moment,
    load_layer_keys,
    setup_rope,
)

_W_SOURCES = {"scored", "corpus"}


# ---------------------------------------------------------------------------
# Experiment-local helpers (verbatim from the task brief)
# ---------------------------------------------------------------------------


def _rtn_channel_arm(M, bits, group, mse_scale):
    from bmx.cache.codecs import scale_bits

    M_hat = rtn_quantize(M.mT, bits, group, mse_scale=mse_scale).mT
    return M_hat, bits + scale_bits(group)


def _k2t_coeffquant_arm(M, rank, coeff_bits, res_bits, group, seed, factors):
    """P3: quantize the low-rank coefficients (Us) instead of storing fp16.
    bpe: residual turboquant payload+norm, Us at coeff_bits (+ its group
    scales), V still fp16 (16·r/S per entry)."""
    from bmx.cache.codecs import norm_bits, quantize_cache

    S, C = M.shape
    Us, V = factors
    Us_q = rtn_quantize(Us.mT, coeff_bits, group, mse_scale=True).mT
    V_st = V.half().float()
    L = Us_q @ V_st.mT
    R_hat, _ = quantize_cache("turboquant_mse", M - L, bits=res_bits, seed=seed)
    bpe = (
        res_bits
        + norm_bits(1, C)  # residual per-row norm
        + coeff_bits * rank / C  # Us payload
        + (16.0 / group) * rank / C  # Us group scales
        + 16.0 * rank / S  # V fp16
    )
    return L + R_hat, bpe


@dataclasses.dataclass
class Config:
    cache_path: str
    corpus_cache_paths: tuple[str, ...] = ()
    model_label: str = ""
    model_name: str = ""  # HF repo id for RoPE; empty => stored-basis logit only
    budgets: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8)
    group: int = 64
    ridge: float = 1e-3
    position_stride: int = 8
    ranks: tuple[int, ...] = (16, 32)
    uniform_rank: int = 16
    uniform_bits: tuple[int, ...] = (2, 3, 4, 5)
    tq_bits: tuple[int, ...] = (2, 3, 4)
    coeffquant_rank: int = 32
    coeffquant_bits: int = 6
    coeffquant_res_bits: int = 2
    k2t_bits: int = 2
    seed: int = 0
    max_layers: int = 0
    w_source: str = "scored"
    out_root: str = ""


def _fit_spectral_randbasis(M_fit: torch.Tensor, seed: int) -> SpectralBasis:
    """W=random-orthogonal-basis control: budget-independent fit half — Q, its
    fit-region variances (uncentered second moment per direction), sorted
    descending. Q is seeded random orthogonal (0 stored bits)."""
    C = M_fit.shape[1]
    Q = random_orthogonal(C, seed, dtype=torch.float64)
    Y = M_fit.double() @ Q
    var = (Y * Y).mean(dim=0)  # uncentered second moment per direction
    order = torch.argsort(var, descending=True)
    var_sorted = var[order]
    Q_sorted = Q[:, order]
    Qf = Q_sorted.float()
    return SpectralBasis(enc=Qf, dec=Qf, lam=var_sorted.float(), lam64=var_sorted)


def _spectral_randbasis_pack(
    M_fit: torch.Tensor, budget: float, seed: int, tiers, group: int
) -> SpectralPack:
    """W=random-orthogonal-basis control: allocation from fit-region variances
    of Y = M_fit @ Q, Q a seeded random orthogonal (0 stored bits)."""
    return pack_from_basis(
        _fit_spectral_randbasis(M_fit, seed), budget, tiers=tiers, group=group
    )


def main(cfg: Config):
    assert cfg.w_source in _W_SOURCES, f"w_source={cfg.w_source!r} not in {_W_SOURCES}"
    if cfg.w_source == "corpus":
        assert cfg.corpus_cache_paths, (
            "w_source='corpus' requires non-empty corpus_cache_paths"
        )

    run = (
        create_run("k4_frontier", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_frontier", cfg)
    )

    from bmx.cache.collect import load_cache

    corpus_caches = [load_cache(p) for p in cfg.corpus_cache_paths]
    # Per-cache layer-keyed view + per-cache RoPE, only needed for w_source
    # "corpus" (mirrors k4_fit_packs.py's per-cache W loop exactly). Reuses
    # corpus_caches (already loaded above) instead of re-reading each file.
    corpus_layer_keys = [bucket_layer_keys(c) for c in corpus_caches]
    corpus_get_cos_sins = []
    corpus_rope_ready = False
    for lk in corpus_layer_keys:
        c_ready, c_get_cos_sin = setup_rope(cfg.model_name, lk, sorted(lk.keys()))
        corpus_rope_ready = corpus_rope_ready or c_ready
        corpus_get_cos_sins.append(c_get_cos_sin)

    layer_keys = load_layer_keys(cfg.cache_path)

    layers = sorted(layer_keys.keys())
    if cfg.max_layers > 0:
        layers = layers[: cfg.max_layers]

    rope_ready, get_cos_sin = setup_rope(cfg.model_name, layer_keys, layers)

    rows: list[dict] = []
    model_label = cfg.model_label or "unknown"
    headline_col = "logit_rope" if rope_ready else "logit"

    def emit(row: dict) -> None:
        rows.append(row)
        budget_str = f"{row['budget']:.2f}" if not math.isnan(row["budget"]) else " n/a"
        print(
            f"  layer={row['layer']:2d} kind={row['kind']:6s} "
            f"arm={row['arm']:20s} fit_mode={row['fit_mode']:8s} "
            f"weighted={row['weighted']!s:5s} budget={budget_str}  "
            f"bpe_model={row['bpe_model']:.3f}  rel_fro={row['rel_fro']:.4f}  "
            f"{headline_col}={row[headline_col]:.4f}",
            flush=True,
        )

    def full_row(**kw) -> dict:
        base = dict(
            model=model_label,
            fit_mode="baseline",
            weighted=False,
            budget=float("nan"),
            mse_scale=False,
            w_source=cfg.w_source,
        )
        base.update(kw)
        return base

    for layer_i in layers:
        kinds_map = layer_keys[layer_i]
        k_pre_t = kinds_map["k_pre"]  # (h_kv, S, d) fp16, pre-RoPE
        k_t = kinds_map["k"]  # (h_kv, S, d) fp16, post-RoPE
        q_t = kinds_map["q"]  # (h, T, d) fp16
        h_kv, S, d = k_pre_t.shape
        C = h_kv * d
        Q_fp32 = q_t.float()

        if rope_ready:
            cos_l, sin_l = get_cos_sin(S)
        else:
            cos_l = torch.ones(S, d)
            sin_l = torch.zeros(S, d)
        K_post_true = apply_rope(k_pre_t.float(), cos_l, sin_l) if rope_ready else None

        M_pre = to_matrix(k_pre_t)  # (S, C) fp32
        M_post = to_matrix(k_t)
        tail = slice(S // 2, S)

        print(f"\n[layer {layer_i}] (h_kv={h_kv}, S={S}, d={d}, C={C})", flush=True)

        def score(M_hat, kind):
            src_t = k_pre_t if kind == "k_pre" else k_t
            M_ref = M_pre if kind == "k_pre" else M_post
            return _score_tail(
                M_hat,
                h_kv,
                tail,
                K_post_true,
                Q_fp32,
                cos_l,
                sin_l,
                rope_ready if kind == "k_pre" else False,
                # kind="k": src_t is k_t (already post-RoPE); _score_tail's
                # k_true_t scores without a second RoPE application
                # (rope_ready=False here).
                src_t,
                M_ref,
            )

        # ---- SVD factors cached per (layer, rank), as in k2d's get_svd -----
        svd_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        def get_svd(rank: int):
            if rank not in svd_cache:
                svd_cache[rank] = truncated_svd(M_pre, rank)
            return svd_cache[rank]

        # ---- Spectral fit machinery: query moment, whitener, fit matrices --
        # "scored" (default) uses THIS cache's own queries; "corpus" pools
        # query_position_moment over the corpus caches' own queries (equal-
        # weight per cache, same convention as k4_fit_packs.py).
        if cfg.w_source == "corpus":
            W_blocks = corpus_query_moment(
                corpus_layer_keys,
                corpus_get_cos_sins,
                corpus_rope_ready,
                layer_i,
                h_kv,
                d,
                cfg.position_stride,
            )
        else:  # "scored"
            W_blocks = query_position_moment(
                q_t.float(), cos_l, sin_l, h_kv, position_stride=cfg.position_stride
            )
        Wh, Wh_inv = assemble_whitener(W_blocks, ridge=cfg.ridge)
        Ih, Ih_inv = identity_whitener(C)

        fit_matrices: dict[str, torch.Tensor] = {
            "oracle": M_pre,
            "heldout": M_pre[: S // 2],
        }
        if corpus_caches:
            others = []
            for other in corpus_caches:
                other_k = other[f"layer{layer_i}.k_pre"]
                assert other_k.shape[0] == h_kv and other_k.shape[2] == d, (
                    f"corpus cache layer{layer_i}.k_pre shape {tuple(other_k.shape)} "
                    f"incompatible with (h_kv={h_kv}, d={d})"
                )
                others.append(to_matrix(other_k))
            fit_matrices["corpus"] = torch.cat(others, dim=0)

        # ==== spectral / spectral_unweighted / spectral_randbasis ==========
        for arm_name, weighted, (basis_h, basis_h_inv) in (
            ("spectral", True, (Wh, Wh_inv)),
            ("spectral_unweighted", False, (Ih, Ih_inv)),
        ):
            for fit_mode, M_fit in fit_matrices.items():
                basis = fit_spectral_basis(M_fit, basis_h, basis_h_inv)
                for budget in cfg.budgets:
                    pack = pack_from_basis(
                        basis, budget, tiers=cfg.tiers, group=cfg.group
                    )
                    M_hat, bpe_model = spectral_quantize(M_pre, pack)
                    rf, lg, lg_rope = score(M_hat, "k_pre")
                    bpe_skeptic = bpe_model + skeptic_charge(C, S, cfg.tiers)
                    bpe_skeptic_deploy = bpe_model + skeptic_charge(
                        C, DEPLOY_S, cfg.tiers
                    )
                    emit(
                        full_row(
                            layer=layer_i,
                            kind="k_pre",
                            arm=arm_name,
                            fit_mode=fit_mode,
                            weighted=weighted,
                            budget=float(budget),
                            bits=-1,
                            rank=-1,
                            bpe_model=bpe_model,
                            bpe_skeptic=bpe_skeptic,
                            bpe_skeptic_deploy=bpe_skeptic_deploy,
                            rel_fro=rf,
                            logit=lg,
                            logit_rope=lg_rope,
                        )
                    )

        # spectral_randbasis: only needs a fit region for the allocation; use
        # the same fit-mode set as the weighted arm (oracle/heldout/corpus?).
        for fit_mode, M_fit in fit_matrices.items():
            randbasis = _fit_spectral_randbasis(M_fit, cfg.seed)
            for budget in cfg.budgets:
                pack = pack_from_basis(
                    randbasis, budget, tiers=cfg.tiers, group=cfg.group
                )
                M_hat, bpe_model = spectral_quantize(M_pre, pack)
                rf, lg, lg_rope = score(M_hat, "k_pre")
                bpe_skeptic = bpe_model + skeptic_charge(C, S, cfg.tiers)
                bpe_skeptic_deploy = bpe_model + skeptic_charge(C, DEPLOY_S, cfg.tiers)
                emit(
                    full_row(
                        layer=layer_i,
                        kind="k_pre",
                        arm="spectral_randbasis",
                        fit_mode=fit_mode,
                        weighted=False,
                        budget=float(budget),
                        bits=-1,
                        rank=-1,
                        bpe_model=bpe_model,
                        bpe_skeptic=bpe_skeptic,
                        bpe_skeptic_deploy=bpe_skeptic_deploy,
                        rel_fro=rf,
                        logit=lg,
                        logit_rope=lg_rope,
                    )
                )

        # ==== baseline arms (fit_mode="baseline", weighted=False, budget=NaN)
        # turboquant_mse @ tq_bits on BOTH kinds
        for b in cfg.tq_bits:
            for kind, M_orig in (("k", M_post), ("k_pre", M_pre)):
                M_hat, bpe = quantize_cache(
                    "turboquant_mse", M_orig, bits=b, seed=cfg.seed
                )
                rf, lg, lg_rope = score(M_hat, kind)
                emit(
                    full_row(
                        layer=layer_i,
                        kind=kind,
                        arm="turboquant_mse",
                        bits=b,
                        rank=-1,
                        bpe_model=bpe,
                        bpe_skeptic=bpe,
                        bpe_skeptic_deploy=bpe,
                        rel_fro=rf,
                        logit=lg,
                        logit_rope=lg_rope,
                    )
                )

        # lowrank_rtn_channel rank=uniform_rank @ uniform_bits, on k_pre
        ur = min(cfg.uniform_rank, S, C)
        for b in cfg.uniform_bits:
            M_hat, bpe = quantize_cache(
                "lowrank_rtn_channel",
                M_pre,
                bits=b,
                rank=ur,
                group=cfg.group,
                seed=cfg.seed,
                svd_factors=get_svd(ur),
            )
            rf, lg, lg_rope = score(M_hat, "k_pre")
            emit(
                full_row(
                    layer=layer_i,
                    kind="k_pre",
                    arm="lowrank_rtn_channel",
                    bits=b,
                    rank=ur,
                    bpe_model=bpe,
                    bpe_skeptic=bpe,
                    bpe_skeptic_deploy=bpe,
                    rel_fro=rf,
                    logit=lg,
                    logit_rope=lg_rope,
                )
            )

        # lowrank_turboquant r in ranks @ k2t_bits, on k_pre
        for rank in cfg.ranks:
            r = min(rank, S, C)
            M_hat, bpe = quantize_cache(
                "lowrank_turboquant",
                M_pre,
                bits=cfg.k2t_bits,
                rank=r,
                seed=cfg.seed,
                svd_factors=get_svd(r),
            )
            rf, lg, lg_rope = score(M_hat, "k_pre")
            emit(
                full_row(
                    layer=layer_i,
                    kind="k_pre",
                    arm="lowrank_turboquant",
                    bits=cfg.k2t_bits,
                    rank=r,
                    bpe_model=bpe,
                    bpe_skeptic=bpe,
                    bpe_skeptic_deploy=bpe,
                    rel_fro=rf,
                    logit=lg,
                    logit_rope=lg_rope,
                )
            )

        # k2t_coeffquant (P3), on k_pre
        cr = min(cfg.coeffquant_rank, S, C)
        M_hat, bpe = _k2t_coeffquant_arm(
            M_pre,
            cr,
            cfg.coeffquant_bits,
            cfg.coeffquant_res_bits,
            cfg.group,
            cfg.seed,
            get_svd(cr),
        )
        rf, lg, lg_rope = score(M_hat, "k_pre")
        emit(
            full_row(
                layer=layer_i,
                kind="k_pre",
                arm="k2t_coeffquant",
                bits=cfg.coeffquant_res_bits,
                rank=cr,
                bpe_model=bpe,
                bpe_skeptic=bpe,
                bpe_skeptic_deploy=bpe,
                rel_fro=rf,
                logit=lg,
                logit_rope=lg_rope,
            )
        )

        # rtn_channel b3 x mse_scale in {False, True}, on k_pre
        for mse_scale in (False, True):
            M_hat, bpe = _rtn_channel_arm(M_pre, 3, cfg.group, mse_scale)
            rf, lg, lg_rope = score(M_hat, "k_pre")
            row = full_row(
                layer=layer_i,
                kind="k_pre",
                arm="rtn_channel",
                bits=3,
                rank=-1,
                bpe_model=bpe,
                bpe_skeptic=bpe,
                bpe_skeptic_deploy=bpe,
                rel_fro=rf,
                logit=lg,
                logit_rope=lg_rope,
            )
            row["mse_scale"] = mse_scale
            emit(row)

    df = pd.DataFrame(rows)
    df = df[
        [
            "model",
            "layer",
            "kind",
            "arm",
            "fit_mode",
            "weighted",
            "budget",
            "w_source",
            "bits",
            "rank",
            "mse_scale",
            "bpe_model",
            "bpe_skeptic",
            "bpe_skeptic_deploy",
            "rel_fro",
            "logit",
            "logit_rope",
        ]
    ]
    write_metrics(run, df)

    verdict = _g1_verdict(df, headline_col, cfg)
    (run / "g1_verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n" + "=" * 88)
    print("G1 VERDICT — spectral vs turboquant_mse frontier duel")
    print("=" * 88)
    print(json.dumps(verdict, indent=2))
    print(f"\nTotal rows: {len(df)}")
    print(f"-> {run}")

    return run


# ---------------------------------------------------------------------------
# G1/P3/P4 verdicts
# ---------------------------------------------------------------------------


def _tq_layer_curve(
    df: pd.DataFrame, headline_col: str
) -> dict[int, list[tuple[float, float]]]:
    """Per-layer turboquant_mse k_pre curve: sorted [(bpe, distortion), ...]."""
    sub = df[(df.arm == "turboquant_mse") & (df.kind == "k_pre")]
    curves: dict[int, list[tuple[float, float]]] = {}
    for layer, g in sub.groupby("layer"):
        pts = sorted(zip(g.bpe_model.tolist(), g[headline_col].tolist()))
        curves[int(layer)] = pts
    return curves


def _log_interp(pts: list[tuple[float, float]], x: float) -> tuple[float, bool]:
    """Interpolate log(y) linearly in x over sorted pts=[(x,y),...].
    Extrapolates log-linearly from the nearest two points when x is outside
    the range; returns (y, extrapolated)."""
    xs = [p[0] for p in pts]
    ys = [math.log(max(p[1], 1e-300)) for p in pts]
    if x <= xs[0]:
        if x == xs[0]:
            return math.exp(ys[0]), False
        x0, x1, y0, y1 = xs[0], xs[1], ys[0], ys[1]
        t = (x - x0) / (x1 - x0)
        return math.exp(y0 + t * (y1 - y0)), True
    if x >= xs[-1]:
        if x == xs[-1]:
            return math.exp(ys[-1]), False
        x0, x1, y0, y1 = xs[-2], xs[-1], ys[-2], ys[-1]
        t = (x - x0) / (x1 - x0)
        return math.exp(y0 + t * (y1 - y0)), True
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return math.exp(ys[i] + t * (ys[i + 1] - ys[i])), False
    raise AssertionError("unreachable")


def _g1_verdict(df: pd.DataFrame, headline_col: str, cfg: Config) -> dict:
    tq_curves = _tq_layer_curve(df, headline_col)
    fit_modes = ["oracle", "heldout"] + (["corpus"] if cfg.corpus_cache_paths else [])

    g1: dict = {}
    for fit_mode in fit_modes:
        entries = []
        pass_all = True
        for budget in cfg.budgets:
            sub = df[
                (df.arm == "spectral")
                & (df.weighted)
                & (df.fit_mode == fit_mode)
                & (df.budget == float(budget))
            ]
            if sub.empty:
                continue
            model_wins, deploy_wins = [], []
            extrapolated = False
            for _, r in sub.iterrows():
                pts = tq_curves.get(int(r.layer))
                if not pts:
                    continue
                tq_model, ex1 = _log_interp(pts, r.bpe_model)
                tq_deploy, ex2 = _log_interp(pts, r.bpe_skeptic_deploy)
                extrapolated = extrapolated or ex1 or ex2
                spectral_dist = max(r[headline_col], 1e-300)
                model_wins.append(tq_model / spectral_dist)
                deploy_wins.append(tq_deploy / spectral_dist)
            if not model_wins:
                continue
            win_model = float(pd.Series(model_wins).mean())
            win_deploy = float(pd.Series(deploy_wins).mean())
            layer_win_frac_model = float((pd.Series(model_wins) > 1).mean())
            layer_win_frac_deploy = float((pd.Series(deploy_wins) > 1).mean())
            entry = dict(
                bpe_model=float(sub.bpe_model.mean()),
                bpe_skeptic_deploy=float(sub.bpe_skeptic_deploy.mean()),
                win_model=win_model,
                win_skeptic_deploy=win_deploy,
                layer_win_fraction_model=layer_win_frac_model,
                layer_win_fraction_deploy=layer_win_frac_deploy,
                extrapolated=bool(extrapolated),
            )
            entries.append(entry)
            if not (
                win_model > 1
                and win_deploy > 1
                and layer_win_frac_model >= 0.9
                and layer_win_frac_deploy >= 0.9
            ):
                pass_all = False
        g1[fit_mode] = {"budgets": entries, "g1_pass": bool(pass_all and entries)}

    g1_pass = bool(g1) and all(v["g1_pass"] for v in g1.values())

    # ---- P3: k2t_coeffquant vs lowrank_turboquant r16@2 -------------------
    cq = df[df.arm == "k2t_coeffquant"]
    lrtq = df[df.arm == "lowrank_turboquant"]
    k2t_rank = min(16, int(lrtq["rank"].max())) if not lrtq.empty else 16
    k2t16 = lrtq[lrtq["rank"] == k2t_rank]
    p3 = {}
    if not cq.empty and not k2t16.empty:
        cq_bpe = float(cq.bpe_model.mean())
        cq_headline = float(cq[headline_col].mean())
        k2t_bpe = float(k2t16.bpe_model.mean())
        k2t_headline = float(k2t16[headline_col].mean())
        p3 = dict(
            coeffquant_bpe=cq_bpe,
            coeffquant_headline=cq_headline,
            k2t_r16_bpe=k2t_bpe,
            k2t_r16_headline=k2t_headline,
            coeffquant_dominates=bool(cq_bpe < k2t_bpe and cq_headline <= k2t_headline),
        )

    # ---- P4: weighted vs unweighted mean ratio at oracle mode -------------
    p4_by_budget = {}
    for budget in cfg.budgets:
        w = df[
            (df.arm == "spectral")
            & (df.fit_mode == "oracle")
            & (df.budget == float(budget))
        ]
        u = df[
            (df.arm == "spectral_unweighted")
            & (df.fit_mode == "oracle")
            & (df.budget == float(budget))
        ]
        if w.empty or u.empty:
            continue
        w_by_layer = w.set_index("layer")[headline_col]
        u_by_layer = u.set_index("layer")[headline_col]
        common = sorted(set(w_by_layer.index) & set(u_by_layer.index))
        if not common:
            continue
        ratios = [u_by_layer.loc[ly] / w_by_layer.loc[ly] for ly in common]
        p4_by_budget[str(budget)] = float(pd.Series(ratios).mean())

    return dict(
        headline_metric=headline_col,
        g1_pass=g1_pass,
        g1_by_fit_mode=g1,
        p3_verdict=p3,
        p4_verdict={"mean_ratio_unweighted_over_weighted": p4_by_budget},
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
