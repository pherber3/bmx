"""Storm Task-5 — prompt-policy pack robustness (plan
`docs/superpowers/plans/2026-07-26-storm-gates.md` Task 5).

Question: does a K4 pack fitted on RAW-policy caches hold its G1 frontier win
when scored on CHAT-policy caches (and vice versa)? Prompt policy is the
largest measured shift axis in this program (the 43.7-pt chat-wrap NIAH
sensitivity, docs/2026-07-06-anchor-forensics-results.md), and the shipped
story fits ONE corpus pack per model — so this is the deployment-robustness
gate for that story.

Substrate note (scope): qwen3-0.6b ONLY — it ships a chat template; gpt2 does
not, so the CHAT policy is undefined there (qwen-only is in-scope per the
plan). Mechanism scale, shipped-instrument defaults: w_source="corpus",
w_rope="frozen" (the shipped mechanism-scale instrument; the locked
Llama-refit config uses w_rope="rotated" — that axis is NOT probed here and
the verdict says so).

Policies (matched token budgets — S=2048 per cache, matched slice offsets):
  RAW  — plain wikitext continuation slices, exactly the shipped collects.
  CHAT — the SAME underlying wikitext text wrapped as ONE user turn via
         tokenizer.apply_chat_template(add_generation_prompt=True,
         tokenize=True, return_dict=True) under the tokenizer's template
         defaults, content-trimmed so the wrapped stream is EXACTLY S tokens
         (bmx.eval.layer_swap.chat_wrap_token_ids is the wrapping contract).
         Collected through the SAME collect_cache path behind its flag-gated
         --chat-wrap knob (default off = byte-exact inert, pinned by
         tests/test_storm_policy_robustness.py).

Fit: offsets (2048, 4096, 6144, 8192) per policy → `corpus_fit_bases` (the
exact k4_fit_packs machinery: pooled per-layer Σ_k on pre-RoPE keys, pooled
corpus-query W) → packs at budgets {2.2, 2.5}. Heldout: offset 0 per policy
(never in any fit corpus).

Cross-score with the SHIPPED frontier instrument (k4_frontier's G1
arithmetic): per layer on k_pre, tail-region logit_rope; per-layer win =
tq_interp(bpe) / spectral_distortion against the SCORED cache's own
turboquant_mse curve (bits 2/3/4, log-interp in bpe); layer-aggregate win =
mean over layers (win_model at payload accounting; win_skeptic_deploy at
DEPLOY_S as companion).

Win retention per (budget, direction):
  raw_to_chat = win(raw pack on chat heldout) / win(chat pack on chat heldout)
  chat_to_raw = win(chat pack on raw heldout) / win(raw pack on raw heldout)

PRE-REGISTERED GATE (verbatim, plan Task 5): cross-policy win retention
>= 0.9 of same-policy ⇒ the shipped-pack story is robust to the largest
measured shift axis. < 0.9 ⇒ FLAG for the paper's deployment section + spec a
refit/shift-detector note (refit is cheap: nc=1 suffices per the rental
record). GATE QUANTITY (fixed here BEFORE running): the model-level-accounting
retention on win_model — the shipped-pack story IS model-level accounting
(the pack ships with the model) — evaluated at BOTH budgets × BOTH
directions; all four must clear 0.9. The deploy-accounting retention is
reported alongside as a companion diagnostic and any disagreement with the
gate quantity is recorded in the verdict.

Shift-magnitude diagnostic (the mechanism number behind the ratio): per
layer, the per-dimension symmetrized log-det (Jensen) gap between the two
policies' second moments on a shared frame —
    jensen_gap_report([A, B])["log_gap"]
      = (1/C)·[logdet((A+B)/2) − (logdet A + logdet B)/2]   (nats/dim)
— the Jensen–Bregman log-det divergence per dimension (0 iff A == B),
computed for Σ_k (pooled fit-corpus pre-RoPE key second moment) and for the
RIDGE-FLOORED W the instrument actually whitens with (assemble_whitener
output squared — the floored object, since near-null query directions are
priced by it, not by the raw moment). A WITHIN-policy yardstick (fit-corpus
split {2048,4096} vs {6144,8192}, same machinery, matched 2-cache subsets on
both sides so finite-sample log-det bias cancels to first order) calibrates
what "large" means against ordinary corpus heterogeneity.

Mechanism scale, no VM, no web. fp32 codec path, fp64 moment/eig math —
exactly the conventions the K4 family already carries.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics
from bmx.cache.codecs import quantize_cache
from bmx.cache.collect import to_matrix
from bmx.cache.spectral import (
    assemble_whitener,
    jensen_gap_report,
    key_second_moment,
    pack_from_basis,
    query_position_moment,
    skeptic_charge,
    spectral_quantize,
)
from experiments._k4_common import (
    DEPLOY_S,
    _layer_ctx,
    _log_interp,
    _score_tail,
    corpus_fit_bases,
    load_layer_keys,
    setup_rope,
)

POLICIES = ("raw", "chat")
BUDGETS = (2.2, 2.5)
GATE_RETENTION = 0.9  # plan-locked


@dataclasses.dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"  # HF repo id (tokenizer + RoPE); "" => no-RoPE
    model_label: str = "qwen3-0.6b"
    seq_len: int = 2048
    n_q_keep: int = 256
    fit_offsets: tuple[int, ...] = (2048, 4096, 6144, 8192)
    heldout_offset: int = 0
    budgets: tuple[float, ...] = BUDGETS
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8)
    group: int = 64
    ridge: float = 1e-3
    position_stride: int = 8
    w_source: str = "corpus"  # shipped k4_fit_packs default
    w_rope: str = "frozen"  # shipped mechanism-scale instrument default
    tq_bits: tuple[int, ...] = (2, 3, 4)
    seed: int = 0
    collect_missing: bool = True  # auto-collect absent caches via collect_cache
    cache_root: str = "results/cache"
    max_layers: int = 0
    out_root: str = ""


# ---------------------------------------------------------------------------
# Retention arithmetic + gate (pure; pinned by the offline tests)
# ---------------------------------------------------------------------------


def retention_ratios(wins: dict[tuple[str, str], float]) -> dict[str, float]:
    """Cross-policy win retention from layer-aggregate G1 wins.

    `wins`: {(pack_policy, score_policy): win}. Each cross-policy win is
    normalized by the SAME-policy win on the SAME scored heldout — the scored
    cache and its tq baseline curve are held fixed; only the pack changes, so
    the ratio isolates the pack-transfer effect.
    """
    return {
        "raw_to_chat": wins[("raw", "chat")] / wins[("chat", "chat")],
        "chat_to_raw": wins[("chat", "raw")] / wins[("raw", "raw")],
    }


def gate_eval(
    retention_by_budget: dict[str, dict[str, float]],
    threshold: float = GATE_RETENTION,
) -> dict:
    """Evaluate the pre-registered gate: EVERY (budget, direction) retention
    must be >= threshold to pass; any miss ⇒ FLAG."""
    flat = {
        f"budget {b} {d}": r
        for b, per_dir in retention_by_budget.items()
        for d, r in per_dir.items()
    }
    assert flat, "no retention entries to gate"
    min_at = min(flat, key=flat.get)  # type: ignore[arg-type]
    gate_pass = all(v >= threshold for v in flat.values())
    return dict(
        threshold=threshold,
        retention=flat,
        min_retention=flat[min_at],
        min_at=min_at,
        gate_pass=gate_pass,
        gate_outcome=(
            "ROBUST — cross-policy win retention >= 0.9 of same-policy at every "
            "(budget, direction): the shipped-pack story survives the largest "
            "measured shift axis"
            if gate_pass
            else "FLAG — cross-policy retention < 0.9: the paper's deployment "
            "section needs the prompt-policy caveat + a refit/shift-detector "
            "note (refit is cheap: nc=1 suffices per the rental record)"
        ),
    )


# ---------------------------------------------------------------------------
# Shift-magnitude diagnostic (pure core; pinned by the offline tests)
# ---------------------------------------------------------------------------


def policy_shift_gap(A: torch.Tensor, B: torch.Tensor) -> float:
    """Per-dimension symmetrized log-det (Jensen) gap between two same-frame
    PSD second moments: jensen_gap_report([A, B])["log_gap"] =
    (1/C)[logdet((A+B)/2) − (logdet A + logdet B)/2] — the Jensen–Bregman
    log-det divergence per dimension (nats/dim; >= 0, and 0 iff A == B).
    Symmetric in (A, B) by construction."""
    return float(jensen_gap_report([A, B])["log_gap"])


def _shift_rows(
    cfg: Config,
    layers: list[int],
    lks_by_policy: dict[str, list[dict]],
    gcs_by_policy: dict[str, list],
    rope_ready_by_policy: dict[str, bool],
) -> list[dict]:
    """Per-layer shift diagnostics. For each policy: per-cache Σ_c (pre-RoPE
    key second moment) and per-cache W_c (query position moment, cfg.w_rope
    convention — the same moments the fit consumes). Pooled/pool-subset
    means then feed `policy_shift_gap`:

      *_cross          — pooled 4-cache raw vs pooled 4-cache chat (headline)
      *_cross_matched  — mean of {rawA vs chatA, rawB vs chatB} (2v2, matched)
      *_within_raw     — rawA vs rawB (2v2)
      *_within_chat    — chatA vs chatB (2v2)

    W enters RIDGE-FLOORED (assemble_whitener → Wh @ Wh): the instrument
    whitens with the floored object, and flooring keeps the log-det finite
    when a query moment has near-null directions.
    """
    n = len(cfg.fit_offsets)
    all_idx = list(range(n))
    a_idx, b_idx = all_idx[: n // 2], all_idx[n // 2 :]
    rows: list[dict] = []
    for layer_i in layers:
        sig: dict[str, list[torch.Tensor]] = {}
        wmo: dict[str, list[torch.Tensor]] = {}
        for policy in POLICIES:
            sig[policy], wmo[policy] = [], []
            for lk, gcs in zip(lks_by_policy[policy], gcs_by_policy[policy]):
                k_pre = lk[layer_i]["k_pre"]
                h_kv, S, d = k_pre.shape
                sig[policy].append(key_second_moment(to_matrix(k_pre)))
                if rope_ready_by_policy[policy]:
                    cos, sin = gcs(S)
                else:
                    cos, sin = torch.ones(S, d), torch.zeros(S, d)
                wmo[policy].append(
                    query_position_moment(
                        lk[layer_i]["q"].float(),
                        cos,
                        sin,
                        h_kv,
                        position_stride=cfg.position_stride,
                        w_rope=cfg.w_rope,
                    )
                )

        def sigma_mean(policy: str, idx: list[int]) -> torch.Tensor:
            sel = [sig[policy][i] for i in idx]
            return sum(sel[1:], start=sel[0]) / len(sel)

        def w_floored(policy: str, idx: list[int]) -> torch.Tensor:
            sel = [wmo[policy][i] for i in idx]
            W = sum(sel[1:], start=sel[0]) / len(sel)
            Wh, _ = assemble_whitener(W, ridge=cfg.ridge)
            return Wh @ Wh

        def matched(fn, pol_a: str, pol_b: str) -> float:
            return 0.5 * (
                policy_shift_gap(fn(pol_a, a_idx), fn(pol_b, a_idx))
                + policy_shift_gap(fn(pol_a, b_idx), fn(pol_b, b_idx))
            )

        rows.append(
            dict(
                model=cfg.model_label,
                layer=layer_i,
                gap_sigma_cross=policy_shift_gap(
                    sigma_mean("raw", all_idx), sigma_mean("chat", all_idx)
                ),
                gap_w_cross=policy_shift_gap(
                    w_floored("raw", all_idx), w_floored("chat", all_idx)
                ),
                gap_sigma_cross_matched=matched(sigma_mean, "raw", "chat"),
                gap_w_cross_matched=matched(w_floored, "raw", "chat"),
                gap_sigma_within_raw=policy_shift_gap(
                    sigma_mean("raw", a_idx), sigma_mean("raw", b_idx)
                ),
                gap_sigma_within_chat=policy_shift_gap(
                    sigma_mean("chat", a_idx), sigma_mean("chat", b_idx)
                ),
                gap_w_within_raw=policy_shift_gap(
                    w_floored("raw", a_idx), w_floored("raw", b_idx)
                ),
                gap_w_within_chat=policy_shift_gap(
                    w_floored("chat", a_idx), w_floored("chat", b_idx)
                ),
            )
        )
        print(
            f"  [shift layer {layer_i}] sigma cross={rows[-1]['gap_sigma_cross']:.4f} "
            f"within=({rows[-1]['gap_sigma_within_raw']:.4f},"
            f"{rows[-1]['gap_sigma_within_chat']:.4f})  "
            f"W cross={rows[-1]['gap_w_cross']:.4f}",
            flush=True,
        )
    return rows


def _shift_summary(df_shift: pd.DataFrame) -> dict:
    def mean(col: str) -> float:
        return float(df_shift[col].mean())

    within_sigma = 0.5 * (mean("gap_sigma_within_raw") + mean("gap_sigma_within_chat"))
    within_w = 0.5 * (mean("gap_w_within_raw") + mean("gap_w_within_chat"))
    return dict(
        definition=(
            "jensen_gap_report([A,B]).log_gap = (1/C)[logdet((A+B)/2) - "
            "(logdet A + logdet B)/2] per layer; Sigma = pooled fit-corpus "
            "pre-RoPE key second moment, W = ridge-floored query moment "
            "(assemble_whitener squared)"
        ),
        units="nats/dim",
        sigma_cross_mean=mean("gap_sigma_cross"),
        sigma_cross_max=float(df_shift.gap_sigma_cross.max()),
        w_cross_mean=mean("gap_w_cross"),
        w_cross_max=float(df_shift.gap_w_cross.max()),
        sigma_cross_matched_mean=mean("gap_sigma_cross_matched"),
        w_cross_matched_mean=mean("gap_w_cross_matched"),
        sigma_within_mean=within_sigma,
        w_within_mean=within_w,
        sigma_cross_over_within=mean("gap_sigma_cross_matched")
        / max(within_sigma, 1e-12),
        w_cross_over_within=mean("gap_w_cross_matched") / max(within_w, 1e-12),
        note=(
            "cross_over_within compares MATCHED 2-cache subsets on both sides "
            "so finite-sample log-det bias cancels to first order; the 4v4 "
            "cross gap is the headline magnitude"
        ),
    )


# ---------------------------------------------------------------------------
# Cache management (auto-collect through the SHIPPED collect path)
# ---------------------------------------------------------------------------


def _cache_path(cfg: Config, policy: str, offset: int) -> Path:
    """Mirror collect_cache._out_path's naming (corpus_label='chat' for the
    CHAT policy) under cfg.cache_root."""
    model_short = (cfg.model_name.split("/")[-1] or cfg.model_label).lower()
    label = "_chat" if policy == "chat" else ""
    suffix = f"_off{offset}" if offset else ""
    return Path(cfg.cache_root) / (
        f"{model_short}_{cfg.seq_len}{label}{suffix}.safetensors"
    )


def _ensure_cache(cfg: Config, policy: str, offset: int) -> Path:
    path = _cache_path(cfg, policy, offset)
    if path.exists():
        return path
    assert cfg.collect_missing, f"cache {path} missing and collect_missing=False"
    from experiments.collect_cache import Config as CollectConfig
    from experiments.collect_cache import main as collect_main

    print(f"[collect] {policy} off{offset} -> {path}", flush=True)
    collect_main(
        CollectConfig(
            model_name=cfg.model_name,
            seq_len=cfg.seq_len,
            n_q_keep=cfg.n_q_keep,
            token_offset=offset,
            out=str(path),
            chat_wrap=(policy == "chat"),
            corpus_label=("chat" if policy == "chat" else ""),
        )
    )
    assert path.exists(), f"collect did not produce {path}"
    return path


# ---------------------------------------------------------------------------
# Frontier cross-scoring
# ---------------------------------------------------------------------------


def _score_heldout(
    cfg: Config,
    score_policy: str,
    heldout_path: Path,
    packs,
    layers: list[int],
) -> list[dict]:
    """Score one heldout cache: its own tq baseline curve per layer + all four
    (pack_policy, budget) spectral cells, exactly the k4_frontier G1
    arithmetic (tail-region logit_rope headline; per-layer log-interp win)."""
    layer_keys = load_layer_keys(str(heldout_path))
    assert sorted(layer_keys.keys())[: len(layers)] == layers, (
        f"heldout {heldout_path} layer set disagrees with fit layers"
    )
    rope_ready, get_cos_sin = setup_rope(cfg.model_name, layer_keys, layers)
    rows: list[dict] = []
    for layer_i in layers:
        ctx = _layer_ctx(
            layer_keys[layer_i], rope_ready=rope_ready, get_cos_sin=get_cos_sin
        )
        assert ctx.S == cfg.seq_len, (
            f"heldout {heldout_path} layer{layer_i} S={ctx.S} != {cfg.seq_len}"
        )

        def score(M_hat):
            rf, lg, lg_rope, _ = _score_tail(
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
            dist = lg_rope if rope_ready else lg
            return rf, lg, lg_rope, dist

        base = dict(
            model=cfg.model_label,
            score_policy=score_policy,
            layer=layer_i,
            S=ctx.S,
            C=ctx.C,
        )
        pts: list[tuple[float, float]] = []
        for b in cfg.tq_bits:
            M_hat, bpe = quantize_cache(
                "turboquant_mse", ctx.M_pre, bits=b, seed=cfg.seed
            )
            rf, lg, lg_rope, dist = score(M_hat)
            pts.append((bpe, max(dist, 1e-300)))
            rows.append(
                dict(
                    **base,
                    pack_policy="",
                    arm="turboquant_mse",
                    budget=float("nan"),
                    bits=b,
                    bpe_model=bpe,
                    bpe_skeptic_deploy=bpe,
                    c_used=float("nan"),
                    rel_fro=rf,
                    logit=lg,
                    logit_rope=lg_rope,
                    dist=dist,
                    tq_interp_model=float("nan"),
                    tq_interp_deploy=float("nan"),
                    win_model=float("nan"),
                    win_deploy=float("nan"),
                    extrapolated=False,
                )
            )
        pts.sort()

        for pack_policy in POLICIES:
            for budget in cfg.budgets:
                pack = packs[pack_policy][budget][layer_i]
                M_hat, bpe_model = spectral_quantize(ctx.M_pre, pack)
                rf, lg, lg_rope, dist = score(M_hat)
                bpe_deploy = bpe_model + skeptic_charge(
                    ctx.C, DEPLOY_S, cfg.tiers, c_used=pack.c_used
                )
                tq_m, ex1 = _log_interp(pts, bpe_model)
                tq_d, ex2 = _log_interp(pts, bpe_deploy)
                dist_c = max(dist, 1e-300)
                rows.append(
                    dict(
                        **base,
                        pack_policy=pack_policy,
                        arm="spectral",
                        budget=float(budget),
                        bits=-1,
                        bpe_model=bpe_model,
                        bpe_skeptic_deploy=bpe_deploy,
                        c_used=float(pack.c_used),
                        rel_fro=rf,
                        logit=lg,
                        logit_rope=lg_rope,
                        dist=dist,
                        tq_interp_model=tq_m,
                        tq_interp_deploy=tq_d,
                        win_model=tq_m / dist_c,
                        win_deploy=tq_d / dist_c,
                        extrapolated=bool(ex1 or ex2),
                    )
                )
                print(
                    f"  [{score_policy}-heldout layer {layer_i}] "
                    f"{pack_policy}-pack b{budget:g}: bpe={bpe_model:.3f} "
                    f"dist={dist:.5f} win_model={tq_m / dist_c:.3f}",
                    flush=True,
                )
    return rows


def aggregate_wins(df: pd.DataFrame, budgets: tuple[float, ...]) -> dict:
    """Layer-aggregate G1 wins per (pack_policy, score_policy, budget) — mean
    of per-layer win ratios, exactly _g1_verdict's aggregation."""
    out: dict = {}
    spec = df[df.arm == "spectral"]
    for pack_policy in POLICIES:
        for score_policy in POLICIES:
            for budget in budgets:
                g = spec[
                    (spec.pack_policy == pack_policy)
                    & (spec.score_policy == score_policy)
                    & (spec.budget == float(budget))
                ]
                assert not g.empty, (
                    f"no rows for ({pack_policy}, {score_policy}, {budget})"
                )
                out[(pack_policy, score_policy, float(budget))] = dict(
                    win_model=float(g.win_model.mean()),
                    win_deploy=float(g.win_deploy.mean()),
                    layer_win_frac_model=float((g.win_model > 1).mean()),
                    layer_win_frac_deploy=float((g.win_deploy > 1).mean()),
                    bpe_model=float(g.bpe_model.mean()),
                    bpe_skeptic_deploy=float(g.bpe_skeptic_deploy.mean()),
                    extrapolated=bool(g.extrapolated.any()),
                    n_layers=int(len(g)),
                )
    return out


def per_layer_retention(df: pd.DataFrame, budgets: tuple[float, ...]) -> dict:
    """Per-layer retention distribution per (budget, direction) — is a miss
    uniform or concentrated in a few layers?"""
    out: dict = {}
    spec = df[df.arm == "spectral"]
    for budget in budgets:
        for direction, (cross_pack, score_policy) in (
            ("raw_to_chat", ("raw", "chat")),
            ("chat_to_raw", ("chat", "raw")),
        ):
            sel = spec[spec.budget == float(budget)]
            cross = sel[
                (sel.pack_policy == cross_pack) & (sel.score_policy == score_policy)
            ].set_index("layer")["win_model"]
            same = sel[
                (sel.pack_policy == score_policy) & (sel.score_policy == score_policy)
            ].set_index("layer")["win_model"]
            ratio = (cross / same).dropna()
            assert len(ratio) > 0
            out[f"budget {budget:g} {direction}"] = dict(
                frac_layers_ge_gate=float((ratio >= GATE_RETENTION).mean()),
                min_layer_retention=float(ratio.min()),
                median_layer_retention=float(ratio.median()),
            )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Config):
    assert cfg.w_source == "corpus", "Task-5 probes the shipped corpus-W fit"
    assert cfg.w_rope in ("frozen", "rotated")
    assert len(cfg.fit_offsets) >= 2, "need >= 2 fit offsets for the within split"
    assert cfg.heldout_offset not in cfg.fit_offsets, (
        "heldout offset must be disjoint from the fit corpus"
    )

    run = (
        create_run("storm_policy_robustness", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("storm_policy_robustness", cfg)
    )

    # ---- caches (auto-collect via the shipped collect path) ---------------
    fit_paths = {
        p: [_ensure_cache(cfg, p, off) for off in cfg.fit_offsets] for p in POLICIES
    }
    heldout_paths = {p: _ensure_cache(cfg, p, cfg.heldout_offset) for p in POLICIES}

    # ---- per-policy corpus fits (the exact k4_fit_packs machinery) --------
    lks_by_policy: dict[str, list[dict]] = {}
    gcs_by_policy: dict[str, list] = {}
    rope_ready_by_policy: dict[str, bool] = {}
    bases_by_policy: dict[str, dict] = {}
    layers: list[int] = []
    for policy in POLICIES:
        lks = [load_layer_keys(str(p)) for p in fit_paths[policy]]
        pol_layers = sorted(lks[0].keys())
        for lk in lks[1:]:
            assert sorted(lk.keys()) == pol_layers, "fit caches disagree on layers"
        if cfg.max_layers > 0:
            pol_layers = pol_layers[: cfg.max_layers]
        if not layers:
            layers = pol_layers
        assert pol_layers == layers, "policies disagree on layer set"
        rope_ready = False
        gcs = []
        for lk in lks:
            ready, g = setup_rope(cfg.model_name, lk, pol_layers)
            rope_ready = rope_ready or ready
            gcs.append(g)
        print(f"\n[fit {policy}] {len(lks)} caches, {len(pol_layers)} layers")
        fit = corpus_fit_bases(
            lks,
            gcs,
            rope_ready,
            pol_layers,
            w_source=cfg.w_source,
            ridge=cfg.ridge,
            position_stride=cfg.position_stride,
            w_rope=cfg.w_rope,
        )
        bases_by_policy[policy] = fit.bases
        lks_by_policy[policy] = lks
        gcs_by_policy[policy] = gcs
        rope_ready_by_policy[policy] = rope_ready
        del fit  # drop M_fits/whiteners; only the bases are needed downstream

    packs = {
        policy: {
            budget: {
                layer_i: pack_from_basis(
                    bases_by_policy[policy][layer_i],
                    budget,
                    tiers=cfg.tiers,
                    group=cfg.group,
                )
                for layer_i in layers
            }
            for budget in cfg.budgets
        }
        for policy in POLICIES
    }

    # ---- frontier cross-score on both heldouts ----------------------------
    rows: list[dict] = []
    for score_policy in POLICIES:
        print(f"\n[score] {score_policy} heldout: {heldout_paths[score_policy]}")
        rows.extend(
            _score_heldout(
                cfg, score_policy, heldout_paths[score_policy], packs, layers
            )
        )
    df = pd.DataFrame(rows)
    write_metrics(run, df)

    # ---- retention + gate --------------------------------------------------
    agg = aggregate_wins(df, cfg.budgets)
    retention_model = {
        f"{b:g}": retention_ratios(
            {
                (pp, sp): agg[(pp, sp, float(b))]["win_model"]
                for pp in POLICIES
                for sp in POLICIES
            }
        )
        for b in cfg.budgets
    }
    retention_deploy = {
        f"{b:g}": retention_ratios(
            {
                (pp, sp): agg[(pp, sp, float(b))]["win_deploy"]
                for pp in POLICIES
                for sp in POLICIES
            }
        )
        for b in cfg.budgets
    }
    gate = gate_eval(retention_model)
    gate_deploy = gate_eval(retention_deploy)

    # ---- shift diagnostic --------------------------------------------------
    print("\n[shift diagnostic]")
    shift_rows = _shift_rows(
        cfg, layers, lks_by_policy, gcs_by_policy, rope_ready_by_policy
    )
    df_shift = pd.DataFrame(shift_rows)
    write_metrics(run, df_shift, name="shift")

    verdict = dict(
        task="storm Task-5 prompt-policy pack robustness",
        model=cfg.model_label,
        substrate_note=(
            "qwen3-0.6b ONLY: gpt2 ships no chat template, so the CHAT policy "
            "is undefined there (qwen-only is in-scope per the plan)"
        ),
        scope_note=(
            "mechanism scale; w_source=corpus, w_rope=frozen (the shipped "
            "mechanism-scale instrument default; the locked Llama-refit config "
            "uses w_rope=rotated — that axis is NOT probed here)"
        ),
        wrapping=(
            "CHAT = same underlying wikitext slice wrapped as ONE user turn "
            "via apply_chat_template(add_generation_prompt=True, tokenize=True, "
            "return_dict=True) under tokenizer template defaults (Qwen3: "
            "thinking-mode default), content-trimmed so the wrapped stream is "
            "exactly S tokens — bmx.eval.layer_swap.chat_wrap_token_ids"
        ),
        fit_offsets=list(cfg.fit_offsets),
        heldout_offset=cfg.heldout_offset,
        n_layers=len(layers),
        gate=gate,
        gate_deploy_companion=gate_deploy,
        deploy_gate_agrees=bool(gate_deploy["gate_pass"] == gate["gate_pass"]),
        per_cell={
            f"{pp}_pack_on_{sp}@b{b:g}": agg[(pp, sp, float(b))]
            for pp in POLICIES
            for sp in POLICIES
            for b in cfg.budgets
        },
        per_layer_retention=per_layer_retention(df, cfg.budgets),
        shift_diagnostic=_shift_summary(df_shift),
        git_sha=git_sha(),
    )
    (run / "verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n" + "=" * 78)
    print("STORM TASK-5 VERDICT — prompt-policy pack robustness")
    print("=" * 78)
    print(json.dumps(verdict, indent=2))
    print(f"\n-> {run}")
    return run


if __name__ == "__main__":
    main(tyro.cli(Config))
