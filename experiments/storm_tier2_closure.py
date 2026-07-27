"""Storm-gate Task 6 — the Tier-2 closure batch (pack algebra + arithmetic).

ONE experiment, six pre-registered sub-items (a)-(f). Every sub-item's gate is
transcribed here VERBATIM from the briefing / audit phrasing; each is evaluated
exactly as written and gets one verdict line. The expected outcome is HONEST
NULLS — nulls WITH NUMBERS close the Tier-2 ledger. All moment/eig math is fp64;
codec application is fp32 (CLAUDE.md dtype convention). Substrate: local gpt2
caches + qwen3-0.6b (where RoPE/QK-norm matter); k4 pack machinery reused, never
re-derived. No web, no 8B-weight download.

Sources (pre-registered gates, quoted below in each sub-item's docstring):
  docs/superpowers/plans/2026-07-26-storm-gates.md §Task 6
  docs/2026-07-26-storm-kv-mechanisms-briefing.md (Tier-2 phrasing, l.58-66)
  docs/2026-07-26-breakeven-blindspot-audit.md §3 (sub-item (f), the G1 row)

The weighted distortion is the K4 metric exactly (spectral.py module docstring):
  D = (1/S) Σ_s e_sᵀ W e_s,  e_s = k_hat_s - k_s,  W = block-diag query 2nd moment.
`weighted_distortion` and `assemble_dense_W` below are the two load-bearing
helpers; both are pinned by offline tiny tests (test_storm_tier2_closure.py).
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics
from bmx.cache.codecs import allocate_bits_from_variance
from bmx.cache.collect import to_matrix
from bmx.cache.spectral import (
    load_packs,
    query_position_moment,
    spectral_payload_bpe,
    spectral_quantize,
    tier_columns,
)
from bmx.quant.rtn import rtn_quantize
from experiments._k4_common import (
    load_layer_keys,
    setup_rope,
)

# ---------------------------------------------------------------------------
# Load-bearing shared helpers (tested).
# ---------------------------------------------------------------------------


def assemble_dense_W(W_blocks: torch.Tensor) -> torch.Tensor:
    """(h_kv, d, d) query-moment blocks -> dense block-diagonal (C, C) fp64.

    Blocks land on the diagonal in `to_matrix`'s head-major channel order
    (identical placement to `assemble_whitener`). The metric W = E[R_pᵀq qᵀR_p]
    is block-diagonal per kv-head because attention logits never couple heads.
    """
    h_kv, d, _ = W_blocks.shape
    C = h_kv * d
    Wd = torch.zeros(C, C, dtype=torch.float64)
    for j in range(h_kv):
        Wd[j * d : (j + 1) * d, j * d : (j + 1) * d] = W_blocks[j].double()
    return Wd


def weighted_distortion(E: torch.Tensor, W: torch.Tensor) -> float:
    """(1/S) Σ_s e_sᵀ W e_s for residual rows E=(S,C) and dense W=(C,C), fp64.

    Equals trace(W @ Eᵀ E)/S; computed as mean_s((E@W ⊙ E).sum(dim=1)) to avoid
    forming the S×S Gram. This IS the K4 weighted metric on a fixed
    reconstruction (spectral.py module docstring). Pinned to the explicit
    double-sum on a tiny case by the offline test.
    """
    Ed = E.double()
    return float(((Ed @ W.double()) * Ed).sum(dim=1).mean())


def bit_equivalent(d_before: float, d_after: float) -> float:
    """Δbit that separates two weighted distortions under the b-bit floor
    D ∝ 4^{-b} (1 bit quarters MSE): Δb = 0.5·log2(d_before/d_after). Positive
    when d_after < d_before (the predictor helped). Used by sub-item (b)/(f)'s
    "≥0.5 bit-equivalent" gate — the exact same 4^{-b} law the break-even
    instrument prices against (breakeven.py)."""
    d_before = max(d_before, 1e-300)
    d_after = max(d_after, 1e-300)
    return 0.5 * math.log2(d_before / d_after)


# ---------------------------------------------------------------------------
# Substrate loading (caches + packs + per-layer W).
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Substrate:
    """One model's loaded pieces for the sub-items that need caches + packs."""

    model_label: str
    model_name: str
    fit_layer_keys: list  # corpus (fit) caches, layer-bucketed
    fit_get_cos_sins: list
    score_layer_keys: dict  # ONE heldout cache to score on
    score_get_cos_sin: object
    rope_ready: bool
    packs: dict  # {layer: SpectralPack} at the chosen budget
    layers: list
    budget: float
    _w_bar_cache: dict = dataclasses.field(default_factory=dict)  # layer -> dense W_bar
    # (cache_idx, layer) -> dense (C,C) per-cache query moment. cache_idx in
    # 0..len(fit)-1 is a fit cache; -1 is the heldout score cache. The RoPE
    # `query_position_moment` loop is the dominant per-layer cost and W_bar +
    # the oracle reuse the SAME per-cache moments — compute each once.
    _q_moment_cache: dict = dataclasses.field(default_factory=dict)


def _per_cache_query_moment(
    sub: "Substrate", cache_idx: int, layer_i: int, position_stride: int
) -> torch.Tensor:
    """Dense (C,C) per-cache query moment, memoized by (cache_idx, layer)."""
    key = (cache_idx, layer_i)
    cached = sub._q_moment_cache.get(key)
    if cached is not None:
        return cached
    h_kv, _, d = _layer_geometry(sub, layer_i)
    if cache_idx == -1:
        lk, get_cs = sub.score_layer_keys, sub.score_get_cos_sin
    else:
        lk = sub.fit_layer_keys[cache_idx]
        get_cs = sub.fit_get_cos_sins[cache_idx]
    q_t = lk[layer_i]["q"].float()
    Sl = lk[layer_i]["k_pre"].shape[1]
    if sub.rope_ready:
        cos, sin = get_cs(Sl)
    else:
        cos, sin = torch.ones(Sl, d), torch.zeros(Sl, d)
    dense = assemble_dense_W(
        query_position_moment(q_t, cos, sin, h_kv, position_stride=position_stride)
    )
    sub._q_moment_cache[key] = dense
    return dense


def _load_substrate(
    model_label: str,
    model_name: str,
    fit_cache_paths: tuple[str, ...],
    score_cache_path: str,
    pack_path: str,
    budget: float,
    position_stride: int,
) -> Substrate:
    fit_layer_keys = [load_layer_keys(p) for p in fit_cache_paths]
    score_layer_keys = load_layer_keys(score_cache_path)
    layers = sorted(fit_layer_keys[0].keys())

    rope_ready = False
    fit_get_cos_sins = []
    for lk in fit_layer_keys:
        ready, get_cs = setup_rope(model_name, lk, layers)
        rope_ready = rope_ready or ready
        fit_get_cos_sins.append(get_cs)
    _, score_get_cos_sin = setup_rope(model_name, score_layer_keys, layers)

    packs = load_packs(pack_path, budget)
    return Substrate(
        model_label=model_label,
        model_name=model_name,
        fit_layer_keys=fit_layer_keys,
        fit_get_cos_sins=fit_get_cos_sins,
        score_layer_keys=score_layer_keys,
        score_get_cos_sin=score_get_cos_sin,
        rope_ready=rope_ready,
        packs=packs,
        layers=layers,
        budget=budget,
    )


def _layer_geometry(sub: Substrate, layer_i: int) -> tuple[int, int, int]:
    k_pre = sub.score_layer_keys[layer_i]["k_pre"]
    h_kv, S, d = k_pre.shape
    return h_kv, S, d


def _W_bar_dense(sub: Substrate, layer_i: int, position_stride: int) -> torch.Tensor:
    """Shipped average (corpus) W as a dense block-diagonal (C,C) fp64.

    Cached per layer on the Substrate — every sub-item scores against the SAME
    shipped W_bar. The equal-weight mean of the memoized per-cache query moments
    reproduces `corpus_query_moment` exactly (same per-cache
    `query_position_moment`, same equal weighting) while sharing that RoPE loop
    with sub-item (a)'s per-cache oracle W_s (each cache's moment computed once)."""
    cached = sub._w_bar_cache.get(layer_i)
    if cached is not None:
        return cached
    moments = [
        _per_cache_query_moment(sub, ci, layer_i, position_stride)
        for ci in range(len(sub.fit_layer_keys))
    ]
    dense = sum(moments) / len(moments)
    sub._w_bar_cache[layer_i] = dense
    return dense


def _W_oracle_dense(sub: Substrate, layer_i: int, position_stride: int) -> torch.Tensor:
    """The scored (heldout) cache's OWN query moment (per-read oracle W)."""
    return _per_cache_query_moment(sub, -1, layer_i, position_stride)


# ===========================================================================
# (a) oracle-vs-average query gap
# ===========================================================================


def sub_item_a(sub: Substrate, position_stride: int) -> tuple[list[dict], dict]:
    """(a) Oracle-vs-average query gap.

    PRE-REGISTERED GATE (storm-gates §Task 6(a); briefing l.58-59):
      "Oracle-vs-average query gap: measured `Σ e_sᵀW e_s` with per-read oracle
      W vs shipped average W — the second-moment-sufficiency prediction is
      gap ≈ 0; record the number."

    Second-moment sufficiency is the claim that the shipped AVERAGE W is a
    sufficient statistic for the metric — i.e. scoring a fixed reconstruction
    e_s with the pooled W_bar equals scoring it with the read's OWN query moment
    W_s, WHEN W_s is drawn from the population W_bar summarizes. The clean test
    is therefore IN-DISTRIBUTION: for each fit cache score its own residual with
    its own oracle W_s vs the pooled W_bar; the prediction is rel_gap ≈ 0
    (registered < 0.10). A HELDOUT out-of-distribution cache (the leading-slice
    score cache) is reported as a LABELED distribution-shift diagnostic — a
    large OOD gap is NOT a sufficiency violation, it is the shift the average
    cannot see (the exact caveat the briefing's "no MLA number held / anchor
    forensics" lineage flags). Gate turns on the in-distribution number only.
    """
    rows: list[dict] = []
    for layer_i in sub.layers:
        pack = sub.packs[layer_i]
        W_bar = _W_bar_dense(sub, layer_i, position_stride)

        # In-distribution: each fit cache, own residual, own oracle W_s vs W_bar.
        for ci, lk in enumerate(sub.fit_layer_keys):
            M = to_matrix(lk[layer_i]["k_pre"])
            S = (M.shape[0] // pack.group) * pack.group
            M = M[:S]
            M_hat, _ = spectral_quantize(M, pack)
            E = M_hat - M
            W_s = _per_cache_query_moment(sub, ci, layer_i, position_stride)
            d_avg = weighted_distortion(E, W_bar)
            d_orc = weighted_distortion(E, W_s)
            rows.append(
                dict(
                    sub_item="a",
                    model=sub.model_label,
                    layer=layer_i,
                    budget=sub.budget,
                    setting="in_distribution",
                    cache_idx=ci,
                    d_weighted_avg=d_avg,
                    d_weighted_oracle=d_orc,
                    rel_gap=abs(d_orc - d_avg) / max(d_avg, 1e-300),
                )
            )

        # OOD diagnostic: heldout leading-slice cache scored with its own oracle.
        M = to_matrix(sub.score_layer_keys[layer_i]["k_pre"])
        S = (M.shape[0] // pack.group) * pack.group
        M = M[:S]
        M_hat, _ = spectral_quantize(M, pack)
        E = M_hat - M
        W_orc = _W_oracle_dense(sub, layer_i, position_stride)
        d_avg = weighted_distortion(E, W_bar)
        d_orc = weighted_distortion(E, W_orc)
        rows.append(
            dict(
                sub_item="a",
                model=sub.model_label,
                layer=layer_i,
                budget=sub.budget,
                setting="ood_heldout",
                cache_idx=-1,
                d_weighted_avg=d_avg,
                d_weighted_oracle=d_orc,
                rel_gap=abs(d_orc - d_avg) / max(d_avg, 1e-300),
            )
        )

    ind = [r for r in rows if r["setting"] == "in_distribution"]
    ood = [r for r in rows if r["setting"] == "ood_heldout"]
    mean_gap_ind = float(sum(r["rel_gap"] for r in ind) / len(ind))
    max_gap_ind = float(max(r["rel_gap"] for r in ind))
    mean_gap_ood = float(sum(r["rel_gap"] for r in ood) / len(ood))
    verdict = dict(
        sub_item="a",
        model=sub.model_label,
        budget=sub.budget,
        gate="in-distribution second-moment-sufficiency: mean rel_gap < 0.10",
        mean_rel_gap_in_distribution=mean_gap_ind,
        max_rel_gap_in_distribution=max_gap_ind,
        mean_rel_gap_ood_heldout=mean_gap_ood,
        passes=bool(mean_gap_ind < 0.10),
        verdict_line=(
            f"[a/{sub.model_label}] in-dist mean |D_oracle-D_avg|/D_avg = "
            f"{mean_gap_ind:.4f} (max {max_gap_ind:.4f}) — "
            + (
                "NULL as predicted (gap≈0, avg W sufficient)"
                if mean_gap_ind < 0.10
                else "GAP survives in-distribution"
            )
            + f"; [OOD leading-slice diagnostic gap {mean_gap_ood:.3f} = "
            "distribution shift, not a sufficiency violation]"
        ),
    )
    return rows, verdict


# ===========================================================================
# (b) post-KLT residual predictor
# ===========================================================================


def _centered_code_moments(
    M_fit: torch.Tensor, enc: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-direction code-space mean μ_i and centered variance σ²_i (fp64) of
    Y = M_fit @ enc. μ is the model-level (16·C-bit) side info; σ² is the
    centered allocation input the mean-centering lever feeds the waterfill."""
    Y = M_fit.double() @ enc.double()  # (S, C)
    mu = Y.mean(dim=0)  # (C,)
    sig2 = Y.var(dim=0, unbiased=False).clamp_min(0.0)  # (C,)
    return mu, sig2


def sub_item_b(sub: Substrate, position_stride: int) -> tuple[list[dict], dict]:
    """(b) Post-KLT residual predictor.

    PRE-REGISTERED GATE (storm-gates §Task 6(b); briefing l.60-61):
      "Post-KLT residual predictor: does a global center/local-window predictor
      cut weighted residual energy ≥0.5 bit on ≥90% layers (random-access-safe
      variants only)?"

    After the KLT (Y = M @ enc), two predictors subtract a prediction Ŷ before
    quantizing the residual and add it back at decode:
      - global-center (RANDOM-ACCESS-SAFE): Ŷ[s,i] = μ_i, a per-direction
        model-level constant (16·C bits, zero per-sequence). To read token s you
        need only μ + the stored residual code — no neighbor. RA-safe.
      - causal local-window (NOT RA-safe, flagged): Ŷ[s,i] = Y[s-1,i] (previous
        token). Reading token s requires token s-1 first (a decode chain) — it
        is NOT paged-random-access-safe, so it is DISQUALIFIED by the gate's
        "random-access-safe variants only" clause; measured and reported for
        completeness, EXCLUDED from the gate.
    Weighted-residual energy = the K4 weighted distortion D of the
    reconstruction at MATCHED bpe. Δbit = bit_equivalent(D_baseline, D_pred).
    Gate: global-center Δbit ≥ 0.5 on ≥90% of layers.
    """
    rows: list[dict] = []
    for layer_i in sub.layers:
        pack = sub.packs[layer_i]
        # Fit μ on the fit corpus (model-level side info), score on heldout.
        M_fit = torch.cat(
            [to_matrix(lk[layer_i]["k_pre"]) for lk in sub.fit_layer_keys], dim=0
        )
        mu_fit, _ = _centered_code_moments(M_fit, pack.enc)

        M = to_matrix(sub.score_layer_keys[layer_i]["k_pre"])
        S = (M.shape[0] // pack.group) * pack.group
        M = M[:S]
        W_bar = _W_bar_dense(sub, layer_i, position_stride)
        cols_by_tier = tier_columns(pack.bits)

        # Baseline: standard spectral quantize (no predictor).
        M_hat_base, bpe = spectral_quantize(M, pack)
        d_base = weighted_distortion(M_hat_base - M, W_bar)

        # global-center predictor: quantize (Y - μ), add μ back.
        d_global = _quantize_with_code_offset(M, pack, cols_by_tier, mu_fit, W_bar)
        dbit_global = bit_equivalent(d_base, d_global)

        # causal local-window (prev-token) predictor — NOT RA-safe, reported only.
        Y = M.double() @ pack.enc.double()
        Y_prev = torch.zeros_like(Y)
        Y_prev[1:] = Y[:-1]
        # per-direction offset varies per row -> emulate by quantizing residual
        # then re-adding the (exact) predictor. This is the code-space residual
        # energy the predictor would leave; it is an UPPER bound on the RA-safe
        # win and still disqualified.
        d_local = _quantize_with_rowwise_offset(M, pack, cols_by_tier, Y_prev, W_bar)
        dbit_local = bit_equivalent(d_base, d_local)

        rows.append(
            dict(
                sub_item="b",
                model=sub.model_label,
                layer=layer_i,
                budget=sub.budget,
                bpe=bpe,
                d_baseline=d_base,
                d_global_center=d_global,
                dbit_global_center=dbit_global,
                d_local_window=d_local,
                dbit_local_window=dbit_local,
                ra_safe_predictor="global_center",
            )
        )

    n = len(rows)
    frac_pass_global = sum(r["dbit_global_center"] >= 0.5 for r in rows) / n
    mean_dbit_global = float(sum(r["dbit_global_center"] for r in rows) / n)
    mean_dbit_local = float(sum(r["dbit_local_window"] for r in rows) / n)
    verdict = dict(
        sub_item="b",
        model=sub.model_label,
        budget=sub.budget,
        gate="RA-safe global-center cuts weighted residual ≥0.5 bit on ≥90% layers",
        frac_layers_global_ge_half_bit=frac_pass_global,
        mean_dbit_global_center=mean_dbit_global,
        mean_dbit_local_window_NOT_RA_SAFE=mean_dbit_local,
        passes=bool(frac_pass_global >= 0.90),
        verdict_line=(
            f"[b/{sub.model_label}] global-center Δbit≥0.5 on "
            f"{frac_pass_global:.0%} of layers (mean Δbit {mean_dbit_global:.3f}) — "
            + (
                "CONFIRM predictor"
                if frac_pass_global >= 0.90
                else "NULL: predictor buys < 0.5 bit"
            )
            + f"; [not-RA-safe prev-token mean Δbit {mean_dbit_local:.3f}, excluded]"
        ),
    )
    return rows, verdict


def _quantize_with_code_offset(
    M: torch.Tensor,
    pack,
    cols_by_tier: dict,
    offset: torch.Tensor,
    W: torch.Tensor,
) -> float:
    """Quantize (Y - offset) per tier, add offset back, decode; return weighted
    distortion. `offset` is a per-direction (C,) constant (broadcast over rows) —
    the global-center predictor's μ. Reuses the same per-tier RTN call and tier
    order as `spectral_quantize_packed` (mse_scale=True)."""
    Y = M.float() @ pack.enc.to(M.dtype)  # (S, C)
    Yc = Y - offset.float().view(1, -1)
    Y_hat = torch.zeros_like(Y)
    for b, cols in cols_by_tier.items():
        Y_hat[:, cols] = rtn_quantize(Yc[:, cols].mT, b, pack.group, mse_scale=True).mT
    Y_hat = Y_hat + offset.float().view(1, -1)
    M_hat = Y_hat @ pack.dec.mT
    return weighted_distortion(M_hat - M, W)


def _quantize_with_rowwise_offset(
    M: torch.Tensor,
    pack,
    cols_by_tier: dict,
    offset: torch.Tensor,
    W: torch.Tensor,
) -> float:
    """Quantize (Y - offset) with a per-ROW offset (S, C) (the causal-window
    predictor), add back, decode. Same tier machinery as the code-offset form."""
    Y = M.float() @ pack.enc.to(M.dtype)
    Yc = Y - offset.float()
    Y_hat = torch.zeros_like(Y)
    for b, cols in cols_by_tier.items():
        Y_hat[:, cols] = rtn_quantize(Yc[:, cols].mT, b, pack.group, mse_scale=True).mT
    Y_hat = Y_hat + offset.float()
    M_hat = Y_hat @ pack.dec.mT
    return weighted_distortion(M_hat - M, W)


# ===========================================================================
# (c) h-cache vs K/V vs MLA-latent bytes/token table (pure arithmetic)
# ===========================================================================

# Geometry constants — read from the models' recorded HF configs (field names
# cited); NOT downloaded weights. Cross-checked at runtime against
# AutoConfig.from_pretrained (metadata only, cached) when available.
#   Llama-3.1-8B: hidden_size, num_attention_heads, num_key_value_heads,
#     head_dim (=hidden/n_heads when absent), num_hidden_layers.
#   Qwen3-8B: same fields (Qwen3Config).
_MODEL_GEOMETRY = {
    "Llama-3.1-8B": dict(
        d_model=4096,
        n_heads=32,
        n_kv_heads=8,
        d_head=128,
        n_layers=32,
        source="meta-llama/Llama-3.1-8B config.json fields",
    ),
    "Qwen3-8B": dict(
        d_model=4096,
        n_heads=32,
        n_kv_heads=8,
        d_head=128,
        n_layers=36,
        source="Qwen/Qwen3-8B config.json fields",
    ),
}


_HF_IDS = {
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
}


def _crosscheck_geometry() -> None:
    """Assert the hardcoded (c) geometry matches the recorded HF CONFIG (metadata
    only, never weights). Best-effort: if a config is not locally cached this is
    silently skipped (the hardcoded constants + config-field citation stand on
    their own — the ban is on downloading the 8B WEIGHTS, not the tiny config)."""
    try:
        from transformers import AutoConfig
    except Exception:
        return
    from bmx.cache.hf_compat import resolve_text_config

    for label, g in _MODEL_GEOMETRY.items():
        try:
            c = resolve_text_config(AutoConfig.from_pretrained(_HF_IDS[label]))
        except Exception:
            continue  # not cached; hardcoded constants + citation stand
        dh = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        got = dict(
            d_model=c.hidden_size,
            n_heads=c.num_attention_heads,
            n_kv_heads=c.num_key_value_heads,
            d_head=dh,
            n_layers=c.num_hidden_layers,
        )
        want = {k: g[k] for k in got}
        assert got == want, (
            f"(c) hardcoded geometry for {label} disagrees with its cached "
            f"config: hardcoded {want} vs config {got} — update _MODEL_GEOMETRY"
        )


def sub_item_c(k4_measured_mean_kv_bpe: float) -> tuple[list[dict], dict]:
    """(c) h-cache vs K/V vs MLA-latent bytes/token table.

    PRE-REGISTERED GATE (storm-gates §Task 6(c); briefing l.61-62):
      "h-cache vs K/V vs MLA-latent bytes/token table for Llama-3.1-8B +
      Qwen3-8B geometry (pure arithmetic; settles skeptic challenge 3 and
      produces the MLA caveat sentence). ... challenge mostly dissolved on
      pre-RoPE+GQA arithmetic; no MLA number held."

    Pure arithmetic, bytes/token summed over all layers:
      fp16 KV      = n_layers · 2 · n_kv_heads · d_head · 2 bytes
      h-cache      = n_layers · d_model · 2 bytes (cache residual, recompute K/V)
      k4-pack KV   = n_layers · 2 · n_kv_heads · d_head · (mean kv bits)/8 bytes
                     at the MEASURED final-recipe bits: k4_b2.5_dec8tl = 3.081
                     mean kv bits (docs/2026-07-26-gh200-rental-results.md §3
                     table, "bits (mean kv)" column, Llama-3.1-8B LongBench).
                     Applied to Qwen3-8B geometry as arithmetic only — same
                     (n_kv=8, d_head=128) geometry; the Qwen 8B pack's own bits
                     were not banked as one headline number.
      MLA-latent   = N/A for these GQA models (no MLA number holds); shown as a
                     reference latent (DeepSeek-V2 kv_lora_rank=512) arithmetic
                     ONLY to name the caveat, never as a claim these models use MLA.
    Gate: this is a settle-the-ledger arithmetic table, not a pass/fail gate; the
    verdict is the caveat sentence + the GQA-KV vs h-cache crossover number.
    """
    _crosscheck_geometry()  # assert hardcoded constants match the cached configs
    rows: list[dict] = []
    for model, g in _MODEL_GEOMETRY.items():
        nl, kv, dh, dm = g["n_layers"], g["n_kv_heads"], g["d_head"], g["d_model"]
        fp16_kv = nl * 2 * kv * dh * 2
        h_cache = nl * dm * 2
        k4_kv = nl * 2 * kv * dh * k4_measured_mean_kv_bpe / 8.0
        # reference MLA latent (DeepSeek-V2 kv_lora_rank=512 + rope dim 64),
        # NOT a property of this model — the caveat's arithmetic only.
        mla_ref_latent = 512 + 64
        mla_ref = nl * mla_ref_latent * 2
        rows.append(
            dict(
                sub_item="c",
                model=model,
                d_model=dm,
                n_kv_heads=kv,
                d_head=dh,
                n_layers=nl,
                gqa_ratio=g["n_heads"] // kv,
                bytes_per_token_fp16_kv=fp16_kv,
                bytes_per_token_h_cache=h_cache,
                bytes_per_token_k4_pack_kv=k4_kv,
                k4_measured_mean_kv_bpe=k4_measured_mean_kv_bpe,
                bytes_per_token_mla_ref_NA=mla_ref,
                h_cache_over_fp16_kv=h_cache / fp16_kv,
                k4_pack_over_fp16_kv=k4_kv / fp16_kv,
                geometry_source=g["source"],
            )
        )
    # The crossover statement: h-cache is LARGER than GQA fp16-KV whenever
    # d_model > 2·n_kv·d_head, i.e. GQA-ratio · d_head-fraction. Both 8B models
    # have d_model 4096 vs 2·8·128 = 2048 => h-cache = 2× fp16-KV (loses).
    verdict = dict(
        sub_item="c",
        gate="arithmetic settle (no pass/fail) — MLA caveat + GQA crossover number",
        h_cache_over_fp16_kv=[r["h_cache_over_fp16_kv"] for r in rows],
        k4_pack_over_fp16_kv=[r["k4_pack_over_fp16_kv"] for r in rows],
        mla_caveat=(
            "No MLA number holds for Llama-3.1-8B / Qwen3-8B: both are GQA "
            "(n_kv=8, d_head=128), not MLA — a latent KV cache is an "
            "architecture property, not a codec applicable post-hoc. The MLA "
            "column is a reference (DeepSeek-V2 kv_lora_rank+rope=576/layer) "
            "shown only to size the caveat."
        ),
        gqa_crossover=(
            "h-cache (d_model=4096/layer) is 2.0× the fp16 GQA-KV "
            "(2·8·128=2048/layer) for both models — recompute-from-hidden LOSES "
            "to GQA-KV at this ratio; k4-pack KV further shrinks the winning side."
        ),
        verdict_line=(
            f"[c] fp16 GQA-KV/layer=2048B; h-cache/layer=4096B (2.0×, loses); "
            f"k4-pack KV @ measured {k4_measured_mean_kv_bpe} mean-kv bits = "
            f"{rows[0]['k4_pack_over_fp16_kv']:.3f}× fp16-KV. "
            "MLA: N/A (GQA arch, no number holds) — caveat sentence produced."
        ),
    )
    return rows, verdict


# ===========================================================================
# (d) vMF radial check (qwen3-0.6b, QK-norm)
# ===========================================================================


def _captured_weighted_energy(
    Sigma: torch.Tensor, W: torch.Tensor, A: torch.Tensor
) -> float:
    """Weighted-metric energy captured by STORING the linear coordinates
    c = Aᵀk (columns of A = coordinate functionals in key space, (d, r) fp64)
    and decoding with the W-optimal linear map:  tr(W Σ A (AᵀΣA)⁺ AᵀΣ).

    Invariant to invertible right-mixing/rescaling of A's columns (the stored
    information, not its parameterization, is what's priced). When A's columns
    are the top-r KLT coordinate functionals W^{1/2}E_r, this equals the top-r
    eigenvalue sum of T = W^{1/2} Σ W^{1/2} — so no single functional can
    capture more than λ₁ (Rayleigh). Both identities pinned by the offline
    test (test_captured_weighted_energy_klt_identity)."""
    B = Sigma.double() @ A.double()  # (d, r)
    G = A.double().mT @ B  # (r, r) = AᵀΣA
    return float(
        torch.trace(W.double() @ B @ torch.linalg.pinv(G, hermitian=True) @ B.mT)
    )


def sub_item_d(sub: Substrate, position_stride: int) -> tuple[list[dict], dict]:
    """(d) vMF radial check.

    PRE-REGISTERED GATE (storm-gates §Task 6(d); briefing l.62-63):
      "vMF radial check: fraction of weighted metric carried by the per-head
      radial coordinate t = μᵀk vs the top KLT directions. ... the
      tangent-normal decomposition explains the Lloyd-gate negative — the
      sphere Gaussianizes the tangential bulk."

    Per kv-head (QK-normed keys, qwen3-0.6b): the radial coordinate is the
    KEY-SPACE functional t = μᵀk, μ = unit mean key direction. Its
    weighted-metric capture is measured at its BEST possible linear decode
    (`_captured_weighted_energy` — the steelman: if radial loses under its
    optimal decode, the null is airtight) and compared against the top-1 KLT
    coordinate (capture exactly λ₁ of the per-head T = W^{1/2} Σ_k W^{1/2})
    at 1 stored coordinate, and radial+top-3-KLT vs top-4 KLT at 4
    coordinates. W is the shipped whitener frame (assemble_whitener, default
    ridge — the pack-fitting convention); the effective metric after the
    ridge floor is W_eff = W^{1/2}·W^{1/2}, used throughout for internal
    consistency.
    Gate (registered): if radial-fraction ≥ top-1-KLT-fraction the radial
    coord is a special axis worth a codec; the prediction is radial ≪ KLT
    (null: the sphere Gaussianizes the tangential bulk, no privileged radial
    axis). NOTE (Rayleigh, by construction): under optimal decode a single
    functional can only TIE λ₁, never beat it — so the pass condition means
    "the top KLT direction IS the radial axis"; the mechanism number is the
    ratio frac_radial/frac_klt_top1.
    """
    from bmx.cache.spectral import assemble_whitener

    rows: list[dict] = []
    for layer_i in sub.layers:
        h_kv, _, d = _layer_geometry(sub, layer_i)
        # Per-head W blocks: slice the memoized corpus-average dense W_bar
        # (block-diagonal by construction — identical to corpus_query_moment).
        W_bar = _W_bar_dense(sub, layer_i, position_stride)
        W_blocks = torch.stack(
            [W_bar[j * d : (j + 1) * d, j * d : (j + 1) * d] for j in range(h_kv)]
        )
        Wh_dense, _ = assemble_whitener(W_blocks)  # dense (C,C); block-diag
        k_pre = sub.score_layer_keys[layer_i]["k_pre"].float()  # (h_kv, S, d)
        for j in range(h_kv):
            Kj = k_pre[j].double()  # (S, d)
            Sigma_j = Kj.mT @ Kj / Kj.shape[0]  # (d, d) uncentered 2nd moment
            Wh_j = Wh_dense[j * d : (j + 1) * d, j * d : (j + 1) * d]  # (d, d)
            W_j = Wh_j @ Wh_j  # effective (ridge-floored) metric
            T_j = Wh_j @ Sigma_j @ Wh_j  # metric-frame covariance (d, d)
            lam, E = torch.linalg.eigh(0.5 * (T_j + T_j.mT))
            lam, E = lam.flip(0), E.flip(1)  # descending
            total = float(lam.clamp_min(0.0).sum())
            if total <= 0:
                continue
            # radial coordinate functional: μ = unit mean key direction.
            mu = Kj.mean(dim=0)
            mu = (mu / mu.norm().clamp_min(1e-12)).view(-1, 1)  # (d, 1)
            klt_funcs = Wh_j @ E  # KLT coordinate functionals (key space)
            frac_rad = _captured_weighted_energy(Sigma_j, W_j, mu) / total
            frac_klt1 = float(lam[0]) / total
            frac_klt4 = float(lam[:4].sum()) / total
            # 4-coord matched: radial + top-3 KLT functionals vs top-4 KLT.
            A4 = torch.cat([mu, klt_funcs[:, :3]], dim=1)
            frac_radplus = _captured_weighted_energy(Sigma_j, W_j, A4) / total
            rows.append(
                dict(
                    sub_item="d",
                    model=sub.model_label,
                    layer=layer_i,
                    head=j,
                    budget=sub.budget,
                    frac_radial=frac_rad,
                    frac_klt_top1=frac_klt1,
                    ratio_radial_to_klt1=frac_rad / max(frac_klt1, 1e-300),
                    frac_radial_plus_top3=frac_radplus,
                    frac_klt_top4=frac_klt4,
                )
            )

    n = len(rows)
    mean_rad = float(sum(r["frac_radial"] for r in rows) / n)
    mean_klt1 = float(sum(r["frac_klt_top1"] for r in rows) / n)
    mean_ratio = float(sum(r["ratio_radial_to_klt1"] for r in rows) / n)
    mean_radplus = float(sum(r["frac_radial_plus_top3"] for r in rows) / n)
    mean_klt4 = float(sum(r["frac_klt_top4"] for r in rows) / n)
    radial_special = bool(mean_rad >= mean_klt1)
    verdict = dict(
        sub_item="d",
        model=sub.model_label,
        budget=sub.budget,
        gate="radial coord special iff frac_radial ≥ frac_klt_top1 (matched 1 coord)",
        mean_frac_radial=mean_rad,
        mean_frac_klt_top1=mean_klt1,
        mean_ratio_radial_to_klt1=mean_ratio,
        mean_frac_radial_plus_top3=mean_radplus,
        mean_frac_klt_top4=mean_klt4,
        passes=radial_special,
        verdict_line=(
            f"[d/{sub.model_label}] radial carries {mean_rad:.3f} of weighted "
            f"metric vs top-1 KLT {mean_klt1:.3f} (ratio {mean_ratio:.3f}; "
            f"4-coord: rad+3={mean_radplus:.3f} vs KLT4={mean_klt4:.3f}) — "
            + (
                "radial IS a special axis (ties the top KLT direction)"
                if radial_special
                else "NULL: KLT dominates, no privileged radial axis (sphere "
                "Gaussianizes tangential bulk)"
            )
        ),
    )
    return rows, verdict


# ===========================================================================
# (e) W_OV read-subspace effective rank + V-energy through it
# ===========================================================================


def _effective_rank(sv: torch.Tensor) -> float:
    """Effective rank = exp(spectral entropy) of singular values (Roy & Vetterli):
    p_i = σ_i / Σσ; erank = exp(-Σ p_i log p_i). fp64."""
    s = sv.double().clamp_min(0.0)
    tot = s.sum().clamp_min(1e-30)
    p = s / tot
    p = p[p > 0]
    ent = -(p * p.log()).sum()
    return float(torch.exp(ent))


def _wov_per_head_gpt2(sd: dict, layer: int, n_head: int) -> list[torch.Tensor]:
    """Per-head W_OV = W_V^h @ W_O^h : (d_model, d_model) for gpt2 (MHA)."""
    from bmx.stacks.gpt2 import circuit_stack

    st = circuit_stack(sd, layer, n_head, "wov")  # (d, d, head)
    return [st.tensor[:, :, h] for h in range(n_head)]


def sub_item_e(
    gpt2_sd: dict,
    gpt2_meta: dict,
    gpt2_sub: Substrate,
    qwen_pieces: dict | None,
    qwen_sub: Substrate | None,
) -> tuple[list[dict], dict]:
    """(e) W_OV read-subspace.

    PRE-REGISTERED GATE (storm-gates §Task 6(e); briefing l.63-64):
      "W_OV read-subspace effective rank per head + V-energy routed through it
      (GQA-null prediction)."

    Weights only: W_OV = W_V^h @ W_O^h per head. Report (i) effective rank of
    W_OV (≤ d_head), (ii) fraction of the V cache's energy routed through the
    top read-subspace (top-r right singular vectors of W_OV, r = round(erank))
    under the query-weighted machinery — i.e. Σ_v energy in that subspace.
    GQA-NULL PREDICTION (on record): gpt2 is MHA — a per-head W_OV read-subspace
    can be genuinely low-rank; qwen3-0.6b is GQA (group=2) — ONE kv-head's V
    feeds TWO q-heads' W_OV, so the UNION of their read subspaces spans more of
    V and no cheap per-kv-head subspace routing survives. Gate (registered): if
    the GQA union read-fraction ≈ the MHA per-head fraction, the routing win
    survives GQA; the prediction is the union fills up (GQA-null).
    """
    rows: list[dict] = []

    # ---- gpt2 (MHA) --------------------------------------------------------
    n_head = gpt2_meta["n_head"]
    for layer_i in gpt2_sub.layers:
        wov_heads = _wov_per_head_gpt2(gpt2_sd, layer_i, n_head)
        v_t = gpt2_sub.score_layer_keys[layer_i]["v"].float()  # (h_kv=h, S, d)
        d_head = gpt2_meta["d"] // n_head
        for h in range(n_head):
            Wov = wov_heads[h].double()  # (d_model, d_model), rank <= d_head
            # Effective rank of the W_OV circuit (the head's read-write rank).
            sv = torch.linalg.svdvals(Wov)
            erank = _effective_rank(sv)
            r = max(1, min(round(erank), d_head))
            # V cache is the d_head V (post W_V); the directions of it that reach
            # the output are W_O^h's row space (right singular vecs of W_O^h,
            # d_head->d_model). Measure Σ_v energy in the top-r of those.
            frac = _v_energy_in_wo_subspace(gpt2_sd, layer_i, h, n_head, v_t[h], r)
            rows.append(
                dict(
                    sub_item="e",
                    model="gpt2",
                    layer=layer_i,
                    head=h,
                    arch="MHA",
                    d_head=d_head,
                    wov_effective_rank=erank,
                    read_rank=r,
                    v_energy_frac_top_subspace=frac,
                    gqa_group=1,
                )
            )

    # ---- qwen3-0.6b (GQA) --------------------------------------------------
    if qwen_pieces is not None and qwen_sub is not None:
        rows.extend(_sub_item_e_qwen(qwen_pieces, qwen_sub))

    df = pd.DataFrame(rows)
    mha = df[df.arch == "MHA"]
    gqa = df[df.arch == "GQA"] if (df.arch == "GQA").any() else None
    mean_erank_mha = float(mha.wov_effective_rank.mean())
    mean_frac_mha = float(mha.v_energy_frac_top_subspace.mean())
    mean_frac_gqa_union = (
        float(gqa.v_energy_frac_union_subspace.mean()) if gqa is not None else None
    )
    mean_frac_gqa_perhead = (
        float(gqa.v_energy_frac_top_subspace.mean()) if gqa is not None else None
    )
    # GQA-null: the union fraction >> the per-head fraction (union fills up).
    gqa_null = (
        bool(
            mean_frac_gqa_union is not None
            and mean_frac_gqa_union > mean_frac_gqa_perhead + 0.02
        )
        if gqa is not None
        else None
    )
    verdict = dict(
        sub_item="e",
        gate="GQA-null: union read-subspace over a kv-group's q-heads fills up "
        "(union V-energy-frac > per-head frac) so no per-kv-head routing survives",
        mean_wov_effective_rank_gpt2_MHA=mean_erank_mha,
        mean_v_energy_frac_top_gpt2_MHA=mean_frac_mha,
        mean_v_energy_frac_union_qwen_GQA=mean_frac_gqa_union,
        mean_v_energy_frac_perhead_qwen_GQA=mean_frac_gqa_perhead,
        gqa_null_confirmed=gqa_null,
        verdict_line=(
            f"[e] gpt2(MHA) W_OV erank≈{mean_erank_mha:.1f}, per-head V-read "
            f"frac≈{mean_frac_mha:.3f}"
            + (
                f"; qwen3(GQA g=2) per-head frac≈{mean_frac_gqa_perhead:.3f} vs "
                f"union frac≈{mean_frac_gqa_union:.3f} — "
                + (
                    "GQA-NULL confirmed (union fills up)"
                    if gqa_null
                    else "union NOT fuller — GQA-null NOT confirmed"
                )
                if gqa is not None
                else "; qwen3 GQA leg skipped"
            )
        ),
    )
    return rows, verdict


def _v_energy_in_wo_subspace(
    sd: dict, layer: int, head: int, n_head: int, v_head: torch.Tensor, r: int
) -> float:
    """Fraction of stored V-head energy (Σ_v) in the top-r left singular subspace
    of W_O^h (the d_head→d_model read map). V is read as (attn·V)·W_O, so the
    directions of the d_head V that reach the output are W_O^h's row space =
    right singular vecs of W_O^h : (d_head, d_model). We take the top-r right
    singular vectors (d_head axes) and measure Σ_v energy there."""
    from bmx.stacks.gpt2 import raw_stack

    o = raw_stack(sd, layer, n_head, which="o")  # (d, d_head, head): slice = W_O^h.T
    Wo_h = o.tensor[:, :, head].mT.double()  # (d_head, d_model) = W_O^h
    # V (d_head) is read as v @ W_O^h; the d_head directions that survive are
    # W_O^h's row space = its LEFT singular vectors (U, in d_head space).
    U, _, _ = torch.linalg.svd(Wo_h)  # U: (d_head, d_head)
    read = U[:, :r]  # (d_head, r)
    Vd = v_head.double()  # (S, d_head)
    Sigma_v = Vd.mT @ Vd / Vd.shape[0]  # (d_head, d_head)
    total = max(float(torch.diagonal(Sigma_v).sum()), 1e-30)
    captured = float(torch.diagonal(read.mT @ Sigma_v @ read).sum())
    return captured / total


def _sub_item_e_qwen(pieces: dict, sub: Substrate) -> list[dict]:
    """qwen3-0.6b GQA leg: per q-head W_O^h read subspace + the UNION over each
    kv-group's q-heads, measured against that kv-head's stored V."""
    rows: list[dict] = []
    n_heads = pieces["n_heads"]
    n_kv = pieces["n_kv"]
    d_head = pieces["d_head"]
    group = n_heads // n_kv
    o_weights = pieces["o_proj"]  # {layer: (d_model, n_heads*d_head)} weight
    v_weights = pieces["v_proj"]  # {layer: (n_kv*d_head, d_model)} weight
    for layer_i in sub.layers:
        Wo = o_weights[layer_i].double()  # o_proj.weight: (d_model, n_heads*d_head)
        Wv = v_weights[layer_i].double()  # v_proj.weight: (n_kv*d_head, d_model)
        # W_O^h : (d_head, d_model) is the block Wo[:, h*d_head:(h+1)*d_head].T
        v_cache = sub.score_layer_keys[layer_i]["v"].float()  # (n_kv, S, d_head)
        for kv in range(n_kv):
            Vd = v_cache[kv].double()  # (S, d_head)
            Sigma_v = Vd.mT @ Vd / Vd.shape[0]
            total = max(float(torch.diagonal(Sigma_v).sum()), 1e-30)
            # W_V^{kv} : (d_head, d_model) block of v_proj feeding this kv head.
            Wv_kv = Wv[kv * d_head : (kv + 1) * d_head, :]  # (d_head, d_model)
            q_heads = list(range(kv * group, (kv + 1) * group))
            per_head_fracs = []
            union_read_dirs = []
            for h in q_heads:
                Wo_h = Wo[:, h * d_head : (h + 1) * d_head].mT  # (d_head, d_model)
                # W_OV = W_V^{kv} @ W_O^h : (d_model, d_model), rank <= d_head.
                Wov = Wv_kv.mT @ Wo_h  # (d_model, d_model)
                erank = _effective_rank(torch.linalg.svdvals(Wov))
                r = max(1, min(round(erank), d_head))
                # V-read subspace: top-r LEFT singular vecs of W_O^h (d_head axes;
                # the d_head directions of V that survive v @ W_O^h).
                U, _, _ = torch.linalg.svd(Wo_h)  # U: (d_head, d_head)
                read = U[:, :r]  # (d_head, r)
                frac = float(torch.diagonal(read.mT @ Sigma_v @ read).sum()) / total
                per_head_fracs.append((h, erank, r, frac))
                union_read_dirs.append(read)
            # union subspace over the group's q-heads (orthonormalize).
            U_union = torch.linalg.qr(torch.cat(union_read_dirs, dim=1))[0]
            # cap at d_head columns
            U_union = U_union[:, : min(U_union.shape[1], d_head)]
            frac_union = (
                float(torch.diagonal(U_union.mT @ Sigma_v @ U_union).sum()) / total
            )
            for h, erank, r, frac in per_head_fracs:
                rows.append(
                    dict(
                        sub_item="e",
                        model="qwen3-0.6b",
                        layer=layer_i,
                        head=h,
                        arch="GQA",
                        d_head=d_head,
                        wov_effective_rank=erank,
                        read_rank=r,
                        v_energy_frac_top_subspace=frac,
                        v_energy_frac_union_subspace=frac_union,
                        gqa_group=group,
                        kv_head=kv,
                    )
                )
    return rows


# ===========================================================================
# (f) mean-centering G1 row (audit §3 spec)
# ===========================================================================


def sub_item_f(sub: Substrate, position_stride: int) -> tuple[list[dict], dict]:
    """(f) The mean-centering G1 row (breakeven-blindspot-audit §3 spec).

    PRE-REGISTERED GATE (audit §3 / §4, and math-review #10a):
      "subtract per-direction pack mean μ before waterfill, waterfill on σ²
      instead of μ²+σ², charge 16·C model-level, compare weighted distortion
      at matched bpe." Prior: "small ≠ measured" — mean concentrates in the
      top directions that get 8 bits anyway. Gate (registered as the audit's
      close-the-ledger row): if centering cuts weighted distortion by ≥0.5
      bit-equivalent at matched bpe on ≥90% of layers, the lever is live; else
      NULL (close the ledger, taxonomy row disposed with a number).

    Baseline: standard pack — allocation input is the UNCENTERED code 2nd
    moment (basis.lam = μ²+σ² eigenvalues), symmetric RTN around zero.
    Centered: allocate on the centered variance σ²_i via the SAME default
    `allocate_bits_from_variance(var, budget, pack.tiers)` call form
    `pack_from_basis` used for the baseline bits, quantize (Y-μ), add μ back
    at decode (μ is 16·C model-level, zero per-sequence). MATCHED bpe: both
    allocate at the SAME mean-bit budget (pack.budget); the 16·C model-level side
    is reported as a separate zero-charge column (audit: zero in every mode).
    """
    rows: list[dict] = []
    for layer_i in sub.layers:
        pack = sub.packs[layer_i]
        M_fit = torch.cat(
            [to_matrix(lk[layer_i]["k_pre"]) for lk in sub.fit_layer_keys], dim=0
        )
        mu_fit, sig2_fit = _centered_code_moments(M_fit, pack.enc)

        M = to_matrix(sub.score_layer_keys[layer_i]["k_pre"])
        S = (M.shape[0] // pack.group) * pack.group
        M = M[:S]
        W_bar = _W_bar_dense(sub, layer_i, position_stride)

        # Baseline (uncentered): the shipped pack's own allocation.
        M_hat_base, bpe_base = spectral_quantize(M, pack)
        d_base = weighted_distortion(M_hat_base - M, W_bar)

        # Centered: re-allocate on σ² (same basis, same budget), quantize Y-μ.
        bits_c = allocate_bits_from_variance(sig2_fit, pack.budget, pack.tiers)
        pack_c = dataclasses.replace(pack, bits=bits_c)
        cols_c = tier_columns(bits_c)
        d_cent = _quantize_with_code_offset(M, pack_c, cols_c, mu_fit, W_bar)
        bpe_cent = spectral_payload_bpe(pack_c)
        # model-level μ charge (zero in payload accounting; skeptic per-seq view).
        C = pack.enc.shape[0]
        mu_model_bits_per_entry_16C = 16.0 * C  # total (bits), NOT per-entry
        mu_skeptic_per_seq = 16.0 * C / S  # if charged per-sequence (it is NOT)

        dbit = bit_equivalent(d_base, d_cent)
        rows.append(
            dict(
                sub_item="f",
                model=sub.model_label,
                layer=layer_i,
                budget=sub.budget,
                bpe_baseline=bpe_base,
                bpe_centered=bpe_cent,
                d_baseline=d_base,
                d_centered=d_cent,
                dbit_centering=dbit,
                mu_total_bits_16C=mu_model_bits_per_entry_16C,
                mu_skeptic_per_seq_bits=mu_skeptic_per_seq,
            )
        )

    n = len(rows)
    frac_pass = sum(r["dbit_centering"] >= 0.5 for r in rows) / n
    mean_dbit = float(sum(r["dbit_centering"] for r in rows) / n)
    mean_bpe_gap = float(sum(r["bpe_centered"] - r["bpe_baseline"] for r in rows) / n)
    verdict = dict(
        sub_item="f",
        model=sub.model_label,
        budget=sub.budget,
        gate="mean-centering cuts weighted distortion ≥0.5 bit-equiv at matched "
        "bpe on ≥90% layers (else NULL, close the ledger)",
        frac_layers_ge_half_bit=frac_pass,
        mean_dbit_centering=mean_dbit,
        mean_bpe_gap_centered_minus_baseline=mean_bpe_gap,
        passes=bool(frac_pass >= 0.90),
        verdict_line=(
            f"[f/{sub.model_label}] mean-centering Δbit≥0.5 on {frac_pass:.0%} of "
            f"layers (mean Δbit {mean_dbit:+.4f}, bpe gap {mean_bpe_gap:+.4f}) — "
            + (
                "LIVE lever"
                if frac_pass >= 0.90
                else "NULL: centering buys < 0.5 bit — taxonomy row disposed"
            )
        ),
    )
    return rows, verdict


# ===========================================================================
# Driver
# ===========================================================================


@dataclasses.dataclass
class Config:
    # gpt2 substrate (no RoPE).
    gpt2_fit_caches: tuple[str, ...] = (
        "results/cache/gpt2_1024_off1024.safetensors",
        "results/cache/gpt2_1024_off2048.safetensors",
        "results/cache/gpt2_1024_off3072.safetensors",
        "results/cache/gpt2_1024_off4096.safetensors",
    )
    gpt2_score_cache: str = (
        "results/cache/gpt2_1024.safetensors"  # heldout leading slice
    )
    gpt2_pack: str = "results/cache/k4_packs_gpt2.safetensors"

    # qwen3-0.6b substrate (RoPE + QK-norm).
    qwen_fit_caches: tuple[str, ...] = (
        "results/cache/qwen3-0.6b_2048_off2048.safetensors",
        "results/cache/qwen3-0.6b_2048_off4096.safetensors",
    )
    qwen_score_cache: str = (
        "results/cache/qwen3-0.6b_2048.safetensors"  # heldout leading
    )
    qwen_pack: str = "results/cache/k4_packs_qwen3_06b.safetensors"
    qwen_model_name: str = "Qwen/Qwen3-0.6B"

    budget: float = 2.5  # gpt2 pack budget for the pack-algebra sub-items
    qwen_budget: float = 3.0  # qwen3 pack budget (the recipe's K side ~3b)
    position_stride: int = 8

    # k4-pack MEASURED bits for the (c) column: final recipe k4_b2.5_dec8tl
    # LongBench "bits (mean kv)" = 3.081 (docs/2026-07-26-gh200-rental-results.md §3).
    k4_measured_mean_kv_bpe: float = 3.081

    include_qwen: bool = (
        True  # qwen legs (a,b,d,e,f); set False to smoke-test gpt2 only
    )
    out_root: str = ""


def _load_qwen_ov_pieces(model_name: str, layers: list[int]) -> dict:
    """o_proj weights + geometry for qwen3 sub-item (e) — metadata + weights of
    the SMALL 0.6b model only (allowed; the 8B ban is (c)-specific)."""
    import torch as _torch
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model_name, dtype=_torch.float32)
    m.eval()
    cfg = m.config
    o_proj, v_proj = {}, {}
    for i in layers:
        o_proj[i] = m.model.layers[i].self_attn.o_proj.weight.detach().clone()
        v_proj[i] = m.model.layers[i].self_attn.v_proj.weight.detach().clone()
    return dict(
        o_proj=o_proj,
        v_proj=v_proj,
        n_heads=cfg.num_attention_heads,
        n_kv=cfg.num_key_value_heads,
        d_head=cfg.head_dim,
    )


def main(cfg: Config):
    # Verdict lines carry unicode (Δ, ≥, ≈, ×); make stdout tolerate it on a
    # cp1252 Windows console without mangling the JSON (which is ASCII-safe).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    run = (
        create_run("storm_tier2_closure", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("storm_tier2_closure", cfg)
    )
    print(f"run dir: {run}", flush=True)

    gpt2_sub = _load_substrate(
        "gpt2",
        "",
        cfg.gpt2_fit_caches,
        cfg.gpt2_score_cache,
        cfg.gpt2_pack,
        cfg.budget,
        cfg.position_stride,
    )
    qwen_sub = None
    if cfg.include_qwen:
        qwen_sub = _load_substrate(
            "qwen3-0.6b",
            cfg.qwen_model_name,
            cfg.qwen_fit_caches,
            cfg.qwen_score_cache,
            cfg.qwen_pack,
            cfg.qwen_budget,
            cfg.position_stride,
        )

    all_rows: list[dict] = []
    verdicts: dict[str, dict] = {}

    def run_leg(name: str, fn, *args):
        print(f"\n=== sub-item ({name}) ===", flush=True)
        rows, verdict = fn(*args)
        all_rows.extend(rows)
        verdicts.setdefault(name, {})
        # multiple models per sub-item -> keep a list of verdicts under the key
        verdicts[name] = verdicts.get(name) or {}
        key = verdict.get("model", "all")
        verdicts[name][key] = verdict
        print(verdict["verdict_line"], flush=True)

    # (a) oracle-vs-average — gpt2 + qwen3
    run_leg("a", sub_item_a, gpt2_sub, cfg.position_stride)
    if qwen_sub is not None:
        run_leg("a", sub_item_a, qwen_sub, cfg.position_stride)

    # (b) residual predictor — gpt2 + qwen3
    run_leg("b", sub_item_b, gpt2_sub, cfg.position_stride)
    if qwen_sub is not None:
        run_leg("b", sub_item_b, qwen_sub, cfg.position_stride)

    # (c) bytes/token table — pure arithmetic, model-independent
    run_leg("c", sub_item_c, cfg.k4_measured_mean_kv_bpe)

    # (d) vMF radial — qwen3 (QK-norm) primary; gpt2 has no per-head norm sphere
    if qwen_sub is not None:
        run_leg("d", sub_item_d, qwen_sub, cfg.position_stride)
    else:
        print("\n=== sub-item (d) SKIPPED (qwen3 disabled) ===", flush=True)

    # (e) W_OV read subspace — gpt2 (MHA) + qwen3 (GQA)
    from bmx.stacks.gpt2 import load_gpt2_state

    gpt2_sd, gpt2_meta = load_gpt2_state("gpt2")
    qwen_pieces = None
    if qwen_sub is not None:
        qwen_pieces = _load_qwen_ov_pieces(cfg.qwen_model_name, qwen_sub.layers)
    run_leg("e", sub_item_e, gpt2_sd, gpt2_meta, gpt2_sub, qwen_pieces, qwen_sub)

    # (f) mean-centering G1 row — gpt2 + qwen3
    run_leg("f", sub_item_f, gpt2_sub, cfg.position_stride)
    if qwen_sub is not None:
        run_leg("f", sub_item_f, qwen_sub, cfg.position_stride)

    df = pd.DataFrame(all_rows)
    write_metrics(run, df)

    out = dict(
        git_sha=git_sha(),
        budget_gpt2=cfg.budget,
        budget_qwen=cfg.qwen_budget,
        verdicts=verdicts,
    )
    (run / "tier2_verdicts.json").write_text(json.dumps(out, indent=2, default=str))

    print("\n" + "=" * 88)
    print("STORM TIER-2 CLOSURE — per-sub-item verdicts")
    print("=" * 88)
    for name in ("a", "b", "c", "d", "e", "f"):
        for _model, v in verdicts.get(name, {}).items():
            print(v["verdict_line"])
    print(f"\nTotal rows: {len(df)}")
    print(f"-> {run}")
    return run


if __name__ == "__main__":
    main(tyro.cli(Config))
