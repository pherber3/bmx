"""Storm Task-4 — head-role regression (plan
`docs/superpowers/plans/2026-07-26-storm-gates.md` Task 4).

Question: do WEIGHT-READABLE per-head role statistics explain the measured
per-layer/per-head query-weighted sensitivity spread in the K4 packs? All head
statistics are derived FROM MODEL WEIGHTS AND LOCAL CACHES DIRECTLY — nothing
is taken from the vault wiki note the briefing flags as compiled-layer /
single-source.

Pre-registered gate (verbatim, plan Task 4):
    R² >= 0.5  ⇒ role is real allocation signal (spec a role-prior lever).
    R² <  0.5  ⇒ honest null: the query-weighted spectrum already subsumes
    function — a paper-strengthening statement either way.

PRIMARY registered regression (declared before running): per model, the
head-level OLS R² of log10(per-head λ-mass total) on the role predictors
below. Secondary (reported, not gated): the per-head top-direction share
regression, the rank (Spearman) versions, and the layer-level aggregate.
qwen3-0.6b (RoPE + QK-norm) is the main event; gpt2 (no RoPE) is included
with the caveat that its role statistics lack the positional-frequency
component entirely (predictor (iii) does not exist without RoPE).

Predictors (per kv-head; per-query-head stats mean-pooled over the GQA group):
  (i)  qk_erank — QK-circuit effective rank: entropy effective rank
       (Roy & Vetterli, exp of the entropy of normalized singular values) of
       the per-head QK circuit W_Q^h (W_K^h)^T through the d_head bottleneck
       (singular values computed exactly via the QR reduction; Qwen3's
       per-head q_norm/k_norm γ scales are folded in as diag(γ_q), diag(γ_k)
       — the weight-readable part of QK-norm; the data-dependent RMS division
       and any projection biases (gpt2) are not representable from weights and
       are documented as out of scope).
  (ii) q_conc / k_conc — pre-RoPE Q/K concentration: mean resultant length of
       the row-normalized per-head query / key vectors from the corpus caches'
       stored q / k_pre (cache-derived, deployment-free; for Qwen3 these are
       post-qk-norm — exactly what attention consumes).
  (iii) pos_peak_log1p (RoPE models only) — positional preference: the
       distance-preference profile a(m) = (R_m q̄_h)·k̄_h from the per-head
       Q/K cache centers through the model's own rotary tables (the mean
       logit as a function of relative distance m; simplest defensible
       weights+centers derivation — documented as such). Predictor is
       log1p(argmax_m a(m)); the local-mass fraction (m < N_LOCAL) is
       recorded as an aux column.
  (iv) ov_pos_mass — OV copying score: fraction of positive real-eigenvalue
       mass of the per-head OV circuit W_V W_O (eigenvalues of the d_head ×
       d_head W_O^h W_V^{kv(h)}; Σ max(Re λ, 0) / Σ |Re λ|).

Response (from the shipped K4 machinery — granularity statement):
  The banked packs store λ/bits PER LAYER over C = h_kv·d directions. Their
  eigendirections are NOT head-aligned (W is per-kv-head block-diagonal but
  Σ_k couples heads; the measured max-block energy of the eigenvectors is
  reported in the verdict — ~0.25, far from one-hot). However the per-head
  λ-MASS TOTAL is EXACTLY recoverable: Wh is block-diagonal, so the head-j
  block trace of T = Wh Σ Wh equals tr(Wh_j Σ_jj Wh_j) — the total λ of the
  per-head-block weighted spectrum. We recompute that per-head spectrum with
  the shipped machinery (corpus_query_moment / assemble_whitener /
  fit_spectral_basis restricted to the head block) on the pack's OWN corpus
  caches, and cross-validate two ways: (a) Σ_heads λ-total == banked pack Σλ
  per layer, (b) the pack-tensor attribution Σ_i λ_i·||E_block_j,i||² (E
  reconstructed as Wh·dec) == the direct per-head totals. Per-head
  TOP-DIRECTION SHARE comes from the per-head-block eigendecomposition (not
  stored in packs — recomputed). Per-head BITS are only softly attributable
  (directions mix heads) and are recorded as an aux column, never gated.

Mechanism scale (gpt2 / qwen3-0.6b), offline, no VM, no web.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics
from bmx.cache.collect import to_matrix
from bmx.cache.hf_compat import resolve_decoder_layers, resolve_text_config
from bmx.cache.rope import _rotate_half
from bmx.cache.spectral import assemble_whitener, fit_spectral_basis, load_packs
from experiments._k4_common import corpus_query_moment, load_layer_keys, setup_rope

GATE_R2 = 0.5  # plan-locked
BITS_BUDGET = 2.5  # the final-recipe operating budget (bits attribution)
POSITION_STRIDE = 8  # matches the banked pack fit (k4_fit_packs default)
RIDGE = 1e-3  # matches the banked pack sidecars
N_LOCAL = 64  # aux local-mass window for the distance profile


# ---------------------------------------------------------------------------
# Predictors (pure; pinned by tests/test_storm_role_regression.py)
# ---------------------------------------------------------------------------


def effective_rank(sv: torch.Tensor) -> float:
    """Entropy effective rank (Roy & Vetterli): exp(H(p)) with
    p_i = σ_i / Σσ over the (non-negative) singular values. Zero singular
    values contribute nothing (0·log 0 = 0). Returns 1.0 <= erank <= len(sv)
    for any nonzero spectrum."""
    sv = sv.double().clamp_min(0.0)
    total = float(sv.sum())
    assert total > 0.0, "effective_rank of an all-zero spectrum is undefined"
    p = sv / total
    nz = p[p > 0]
    return float(torch.exp(-(nz * nz.log()).sum()))


def qk_effective_rank(WQ_h: torch.Tensor, WK_h: torch.Tensor) -> float:
    """Effective rank of the per-head QK circuit W_Q^h (W_K^h)^T through the
    d_head bottleneck. WQ_h/WK_h: (d_model, d_head). The circuit's nonzero
    singular values equal svdvals(R_Q R_K^T) via the reduced QR
    W = U R (U orthonormal): W_Q W_K^T = U_Q (R_Q R_K^T) U_K^T — an exact
    d_head × d_head reduction, no d_model × d_model SVD."""
    A, B = WQ_h.double(), WK_h.double()
    assert A.shape == B.shape and A.dim() == 2, (
        f"WQ_h/WK_h must be matching (d_model, d_head); got "
        f"{tuple(A.shape)} vs {tuple(B.shape)}"
    )
    _, RQ = torch.linalg.qr(A, mode="reduced")
    _, RK = torch.linalg.qr(B, mode="reduced")
    return effective_rank(torch.linalg.svdvals(RQ @ RK.mT))


def ov_positive_eigenmass(WV_j: torch.Tensor, WO_h: torch.Tensor) -> float:
    """Copying score of the per-head OV circuit W_V W_O: the fraction of
    positive real-eigenvalue mass, Σ max(Re λ, 0) / Σ |Re λ|, over the
    eigenvalues of the d_head × d_head W_O^h W_V^{kv(h)} (which carry the
    nonzero spectrum of the d_model × d_model circuit). 1.0 = pure copying
    (all-positive), 0.0 = pure anti-copying, ~0.5 = balanced."""
    WV64, WO64 = WV_j.double(), WO_h.double()
    d = WV64.shape[1]
    assert WO64.shape[0] == d, (
        f"W_V (d_model, d)={tuple(WV64.shape)} and W_O (d, d_model)="
        f"{tuple(WO64.shape)} disagree on d_head"
    )
    ev = torch.linalg.eigvals(WO64 @ WV64).real
    denom = float(ev.abs().sum())
    if denom == 0.0:
        return 0.5  # degenerate zero circuit: no sign preference
    return float(ev.clamp_min(0.0).sum()) / denom


def mean_resultant_length(X: torch.Tensor) -> float:
    """Directional concentration of row vectors: R = ||mean(x_i/||x_i||)||,
    in [0, 1]. 1 = all rows aligned, ~0 = isotropic/antipodal."""
    assert X.dim() == 2, f"X must be (n, d); got {tuple(X.shape)}"
    Xn = X.double() / X.double().norm(dim=1, keepdim=True).clamp_min(1e-30)
    return float(Xn.mean(dim=0).norm())


def rope_distance_profile(
    q_bar: torch.Tensor, k_bar: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Mean-logit distance preference a(m) = (R_m q̄)·k̄ for m in [0, S):
    the trigonometric per-head preference the Q/K centers place on relative
    distance m under the model's own rotary tables (rotate-half convention;
    R_p^T R_s = R_{s-p} makes (R_m q)·k the exact m = p−s ≥ 0 causal form).
    q_bar/k_bar: (d,); cos/sin: (S, d). Returns (S,) fp64."""
    q64, k64 = q_bar.double(), k_bar.double()
    assert q64.shape == k64.shape and q64.dim() == 1
    q_rot = cos.double() * q64 + sin.double() * _rotate_half(q64)  # (S, d)
    return q_rot @ k64


# ---------------------------------------------------------------------------
# Response: per-head λ spectra + pack-tensor attribution
# ---------------------------------------------------------------------------


def per_head_weighted_spectra(
    M_fit: torch.Tensor, W_blocks: torch.Tensor, *, ridge: float = RIDGE
) -> torch.Tensor:
    """Per-kv-head query-weighted key spectra: row j is the descending fp64
    eigenvalue vector of Wh_j Σ_jj Wh_j — the SAME shipped machinery the pack
    fit uses (assemble_whitener + fit_spectral_basis), restricted to head j's
    channel block. Because Wh is block-diagonal, Σ_j row-sums equal the full
    pack's Σλ exactly (block-trace identity; pinned in the test).

    M_fit: (S, C) fp32 pre-RoPE key matrix; W_blocks: (h_kv, d, d) fp64.
    Returns (h_kv, d) fp64."""
    h_kv, d, _ = W_blocks.shape
    S, C = M_fit.shape
    assert C == h_kv * d, f"C={C} != h_kv*d={h_kv * d}"
    lams = []
    for j in range(h_kv):
        Wh_j, Wh_inv_j = assemble_whitener(W_blocks[j : j + 1], ridge=ridge)
        basis_j = fit_spectral_basis(M_fit[:, j * d : (j + 1) * d], Wh_j, Wh_inv_j)
        lams.append(basis_j.lam64)
    return torch.stack(lams)


def pack_lambda_attribution(
    lam: torch.Tensor,
    bits: torch.Tensor,
    dec: torch.Tensor,
    Wh: torch.Tensor,
    h_kv: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Attribute the banked pack's per-direction λ/bits to kv heads via the
    orthonormal eigenvectors E = Wh @ dec (dec = Wh^{-1} E ⇒ this reconstructs
    E exactly up to the pack's fp32 storage; columns re-normalized to absorb
    that round-off). frac[j, i] = ||E[block_j, i]||² sums to 1 over j, so
    λ-attribution Σ_i λ_i frac[j, i] equals the head-j block trace of
    Wh Σ Wh — the direct per-head λ total (exact identity, pinned in the
    test). Bits attribution through the same fractions is SOFT (directions
    mix heads) and is reported as aux only.

    Returns (lam_soft (h_kv,), bits_soft (h_kv,), block_diag_index) where
    block_diag_index = mean over used (bits>0) directions of the max head-block
    energy fraction — 1.0 would mean head-aligned directions, 1/h_kv is the
    fully-mixed baseline."""
    C = lam.shape[0]
    d = C // h_kv
    E = Wh.double() @ dec.double()  # (C, C), ~orthonormal columns
    E = E / E.norm(dim=0, keepdim=True).clamp_min(1e-30)
    frac = (E.reshape(h_kv, d, C) ** 2).sum(dim=1)  # (h_kv, C), cols sum to 1
    lam_soft = frac @ lam.double()
    bits_soft = frac @ bits.double()
    used = bits > 0
    assert bool(used.any()), "pack allocates zero bits everywhere"
    block_diag_index = float(frac[:, used].max(dim=0).values.mean())
    return lam_soft, bits_soft, block_diag_index


# ---------------------------------------------------------------------------
# Regression helpers (pure; numpy only, no scipy)
# ---------------------------------------------------------------------------


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average-tie ranks (pandas rank, method='average')."""
    return pd.Series(x).rank(method="average").to_numpy()


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation: Pearson on average-tie ranks."""
    rx, ry = _ranks(x), _ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def ols_r2(X: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    """OLS with intercept on z-scored predictors. Returns
    {r2, adj_r2, coef_std: {name: standardized coefficient}}. Standardizing
    X changes no R² (affine reparametrization) but makes coefficients
    comparable across predictors."""
    n, p = X.shape
    assert len(names) == p and n > p + 1, f"need n > p+1; got n={n}, p={p}"
    sd = X.std(axis=0)
    assert (sd > 0).all(), f"constant predictor among {names}"
    Xs = (X - X.mean(axis=0)) / sd
    A = np.column_stack([np.ones(n), Xs])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    assert ss_tot > 0, "constant response"
    r2 = 1.0 - float((resid**2).sum()) / ss_tot
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    return dict(
        r2=r2,
        adj_r2=adj,
        coef_std={nm: float(b) for nm, b in zip(names, beta[1:])},
    )


def regression_report(X: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    """Full report for one (predictor-matrix, response) pair: multiple OLS,
    rank-transformed OLS (the Spearman analogue of R²), per-predictor
    univariate R² + Spearman ρ, and the dominant predictor under both the
    univariate-R² and |standardized coefficient| readings."""
    ols = ols_r2(X, y, names)
    Xr = np.column_stack([_ranks(X[:, i]) for i in range(X.shape[1])])
    rank_ols = ols_r2(Xr, _ranks(y), names)
    uni = {}
    for i, nm in enumerate(names):
        uni[nm] = dict(
            r2=ols_r2(X[:, i : i + 1], y, [nm])["r2"],
            spearman=spearman(X[:, i], y),
        )
    dom_uni = max(uni, key=lambda nm: uni[nm]["r2"])
    dom_coef = max(ols["coef_std"], key=lambda nm: abs(ols["coef_std"][nm]))
    return dict(
        n=int(X.shape[0]),
        ols_r2=ols["r2"],
        ols_adj_r2=ols["adj_r2"],
        rank_ols_r2=rank_ols["r2"],
        coef_std=ols["coef_std"],
        univariate=uni,
        dominant_by_univariate_r2=dom_uni,
        dominant_by_abs_coef=dom_coef,
    )


# ---------------------------------------------------------------------------
# Weight extraction (structural dispatch, mirrors collect.register_hooks)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LayerHeadWeights:
    """Per-layer per-head projection slices, all fp64.
    WQ: (h, d_model, d); WK/WV: (h_kv, d_model, d); WO: (h, d, d_model).
    Qwen3 qk-norm γ scales are already folded into WQ/WK columns."""

    WQ: torch.Tensor
    WK: torch.Tensor
    WV: torch.Tensor
    WO: torch.Tensor


def _extract_gpt2_weights(model) -> tuple[list[LayerHeadWeights], int, int, int]:
    cfg = model.config
    h = cfg.n_head
    d = cfg.n_embd // h
    n_embd = h * d
    out = []
    for block in model.transformer.h:
        W = block.attn.c_attn.weight.detach().double()  # (n_embd, 3*n_embd), x@W
        WQ_full = W[:, :n_embd]
        WK_full = W[:, n_embd : 2 * n_embd]
        WV_full = W[:, 2 * n_embd : 3 * n_embd]
        WO_full = block.attn.c_proj.weight.detach().double()  # (n_embd, n_embd)
        WQ = torch.stack([WQ_full[:, i * d : (i + 1) * d] for i in range(h)])
        WK = torch.stack([WK_full[:, i * d : (i + 1) * d] for i in range(h)])
        WV = torch.stack([WV_full[:, i * d : (i + 1) * d] for i in range(h)])
        WO = torch.stack([WO_full[i * d : (i + 1) * d, :] for i in range(h)])
        out.append(LayerHeadWeights(WQ=WQ, WK=WK, WV=WV, WO=WO))
    return out, h, h, d  # gpt2: h_kv == h


def _extract_qkproj_weights(model) -> tuple[list[LayerHeadWeights], int, int, int]:
    cfg = resolve_text_config(model.config)
    h = cfg.num_attention_heads
    h_kv = getattr(cfg, "num_key_value_heads", h)
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // h
    out = []
    for layer in resolve_decoder_layers(model):
        attn = layer.self_attn
        # nn.Linear weight is (out, in); slice output rows per head, transpose
        # to the (d_model, d) column convention.
        wq = attn.q_proj.weight.detach().double()  # (h*d, d_model)
        wk = attn.k_proj.weight.detach().double()  # (h_kv*d, d_model)
        wv = attn.v_proj.weight.detach().double()  # (h_kv*d, d_model)
        wo = attn.o_proj.weight.detach().double()  # (d_model, h*d)
        WQ = torch.stack([wq[i * d : (i + 1) * d, :].mT for i in range(h)])
        WK = torch.stack([wk[i * d : (i + 1) * d, :].mT for i in range(h_kv)])
        WV = torch.stack([wv[i * d : (i + 1) * d, :].mT for i in range(h_kv)])
        WO = torch.stack([wo[:, i * d : (i + 1) * d].mT for i in range(h)])
        if hasattr(attn, "q_norm"):  # Qwen3-style per-head qk-norm γ
            gq = attn.q_norm.weight.detach().double()  # (d,)
            gk = attn.k_norm.weight.detach().double()  # (d,)
            WQ = WQ * gq.view(1, 1, d)
            WK = WK * gk.view(1, 1, d)
        out.append(LayerHeadWeights(WQ=WQ, WK=WK, WV=WV, WO=WO))
    return out, h, h_kv, d


def extract_head_weights(model) -> tuple[list[LayerHeadWeights], int, int, int]:
    """Per-layer per-head W_Q/W_K/W_V/W_O slices + (h, h_kv, d). Structural
    dispatch exactly like collect.register_hooks: GPT-2 packed c_attn Conv1D
    vs Llama/Qwen-style split projections (with qk-norm γ folding)."""
    if hasattr(model, "transformer") and hasattr(model.transformer.h[0].attn, "c_attn"):
        return _extract_gpt2_weights(model)
    return _extract_qkproj_weights(model)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    label: str
    hf_name: str  # weights (and RoPE config when rope_name non-empty)
    rope_name: str  # "" => no-RoPE (gpt2) tables
    pack_path: str  # banked corpus pack (+ .json sidecar)


MODEL_SPECS: dict[str, ModelSpec] = {
    "gpt2": ModelSpec(
        label="gpt2",
        hf_name="gpt2",
        rope_name="",
        pack_path="results/cache/k4_packs_gpt2.safetensors",
    ),
    "qwen3-0.6b": ModelSpec(
        label="qwen3-0.6b",
        hf_name="Qwen/Qwen3-0.6B",
        rope_name="Qwen/Qwen3-0.6B",
        pack_path="results/cache/k4_packs_qwen3_06b.safetensors",
    ),
}


@dataclasses.dataclass
class Config:
    models: tuple[str, ...] = ("gpt2", "qwen3-0.6b")
    budget: float = BITS_BUDGET  # pack budget for the bits attribution
    out_root: str = ""


def _corpus_paths(spec: ModelSpec) -> list[str]:
    sidecar_path = Path(spec.pack_path + ".json")
    assert sidecar_path.exists(), (
        f"pack sidecar {sidecar_path} missing — refit the pack via "
        f"experiments/k4_fit_packs.py for {spec.label}"
    )
    sidecar = json.loads(sidecar_path.read_text())
    paths = sidecar["corpus_cache_paths"]
    for p in paths:
        assert Path(p).exists(), (
            f"corpus cache {p} missing — regenerate via "
            f"`uv run python experiments/collect_cache.py --model-name "
            f"{spec.hf_name} ...` (see the cache filename for seq-len/offset)"
        )
    return paths


def _head_rows_for_model(spec: ModelSpec, budget: float) -> tuple[pd.DataFrame, dict]:
    """All per-(layer, kv_head) predictor + response rows for one model, plus
    the per-model validation stats block."""
    corpus_paths = _corpus_paths(spec)
    assert Path(spec.pack_path).exists(), (
        f"banked pack {spec.pack_path} missing — refit via k4_fit_packs"
    )
    per_cache_lk = [load_layer_keys(p) for p in corpus_paths]
    layers = sorted(per_cache_lk[0].keys())
    rope_ready, get_cos_sin = setup_rope(spec.rope_name, per_cache_lk[0], layers)
    get_cos_sins = [get_cos_sin] * len(per_cache_lk)
    packs = load_packs(spec.pack_path, budget)
    assert sorted(packs.keys()) == layers, (
        f"pack layers {sorted(packs.keys())} != corpus layers {layers}"
    )

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_name, torch_dtype=torch.float32
    ).eval()
    weights, h, h_kv, d = extract_head_weights(model)
    del model
    assert len(weights) == len(layers), (
        f"{spec.label}: {len(weights)} weight layers vs {len(layers)} cache layers"
    )
    grp = h // h_kv

    S_ref = max(lk[layers[0]]["k_pre"].shape[1] for lk in per_cache_lk)
    cos_ref, sin_ref = get_cos_sin(S_ref) if rope_ready else (None, None)

    rows: list[dict] = []
    lam_reldiffs: list[float] = []
    attr_reldiffs: list[float] = []
    block_diag_indices: list[float] = []
    for li in layers:
        lw = weights[li]
        # ---- response: per-head weighted spectra on the pack's own corpus ----
        W_blocks = corpus_query_moment(
            per_cache_lk, get_cos_sins, rope_ready, li, h_kv, d, POSITION_STRIDE
        )
        M_fit = torch.cat([to_matrix(lk[li]["k_pre"]) for lk in per_cache_lk], dim=0)
        lam_heads = per_head_weighted_spectra(M_fit, W_blocks)  # (h_kv, d) fp64
        lam_totals = lam_heads.sum(dim=1)  # (h_kv,)
        top_shares = lam_heads[:, 0] / lam_totals.clamp_min(1e-30)

        # ---- validation vs the banked pack tensors --------------------------
        pack = packs[li]
        Wh, _ = assemble_whitener(W_blocks, ridge=RIDGE)
        lam_soft, bits_soft, bdi = pack_lambda_attribution(
            pack.lam, pack.bits, pack.dec, Wh, h_kv
        )
        pack_lam_sum = float(pack.lam.double().sum())
        perhead_sum = float(lam_totals.sum())
        lam_reldiff = abs(perhead_sum - pack_lam_sum) / max(pack_lam_sum, 1e-30)
        attr_reldiff = float(
            ((lam_soft - lam_totals).abs() / lam_totals.clamp_min(1e-30)).max()
        )
        lam_reldiffs.append(lam_reldiff)
        attr_reldiffs.append(attr_reldiff)
        block_diag_indices.append(bdi)

        # ---- cache-side stats (pooled over corpus caches) -------------------
        q_all = torch.cat([lk[li]["q"].float() for lk in per_cache_lk], dim=1)
        k_all = torch.cat([lk[li]["k_pre"].float() for lk in per_cache_lk], dim=1)
        q_conc_qh = [mean_resultant_length(q_all[qh]) for qh in range(h)]
        q_bar_qh = q_all.double().mean(dim=1)  # (h, d)
        k_bar_kv = k_all.double().mean(dim=1)  # (h_kv, d)

        for j in range(h_kv):
            qh_slice = range(j * grp, (j + 1) * grp)
            erank_qh = [qk_effective_rank(lw.WQ[qh], lw.WK[j]) for qh in qh_slice]
            ov_qh = [ov_positive_eigenmass(lw.WV[j], lw.WO[qh]) for qh in qh_slice]
            row = dict(
                model=spec.label,
                layer=li,
                kv_head=j,
                n_q_heads=grp,
                qk_erank=float(np.mean(erank_qh)),
                q_conc=float(np.mean([q_conc_qh[qh] for qh in qh_slice])),
                k_conc=mean_resultant_length(k_all[j]),
                ov_pos_mass=float(np.mean(ov_qh)),
                lam_total=float(lam_totals[j]),
                log10_lam_total=float(np.log10(max(float(lam_totals[j]), 1e-30))),
                lam_top_share=float(top_shares[j]),
                lam_soft_pack=float(lam_soft[j]),
                bits_soft_pack=float(bits_soft[j]),
            )
            if rope_ready:
                peaks, locals_ = [], []
                for qh in qh_slice:
                    a = rope_distance_profile(
                        q_bar_qh[qh], k_bar_kv[j], cos_ref, sin_ref
                    )
                    peaks.append(float(torch.argmax(a)))
                    denom = float(a.abs().sum())
                    locals_.append(
                        float(a[:N_LOCAL].abs().sum()) / denom if denom > 0 else 0.0
                    )
                row["pos_peak"] = float(np.mean(peaks))
                row["pos_peak_log1p"] = float(np.mean(np.log1p(peaks)))
                row["pos_local_frac"] = float(np.mean(locals_))
            else:
                row["pos_peak"] = float("nan")
                row["pos_peak_log1p"] = float("nan")
                row["pos_local_frac"] = float("nan")
            rows.append(row)
        print(
            f"[{spec.label} layer {li}] lam_total spread "
            f"{float(lam_totals.max() / lam_totals.min().clamp_min(1e-30)):.1f}x  "
            f"pack-sum reldiff {lam_reldiff:.2e}  attr reldiff {attr_reldiff:.2e}",
            flush=True,
        )

    validation = dict(
        pack_lam_sum_reldiff_max=max(lam_reldiffs),
        attribution_reldiff_max=max(attr_reldiffs),
        block_diag_index_mean=float(np.mean(block_diag_indices)),
        block_diag_index_baseline=1.0 / h_kv,
        h=h,
        h_kv=h_kv,
        d=d,
        n_layers=len(layers),
    )
    return pd.DataFrame(rows), validation


def _predictor_names(rope_ready: bool) -> list[str]:
    base = ["qk_erank", "q_conc", "k_conc", "ov_pos_mass"]
    return base + (["pos_peak_log1p"] if rope_ready else [])


def _model_verdict(df: pd.DataFrame, spec: ModelSpec, validation: dict) -> dict:
    """Regressions + the pre-registered gate for one model's head rows."""
    rope_ready = bool(spec.rope_name)
    names = _predictor_names(rope_ready)
    X = df[names].to_numpy(dtype=np.float64)
    y_primary = df["log10_lam_total"].to_numpy(dtype=np.float64)
    y_top = df["lam_top_share"].to_numpy(dtype=np.float64)

    head_primary = regression_report(X, y_primary, names)
    head_top = regression_report(X, y_top, names)

    # Layer-level aggregate: mean head role per layer vs log10 layer λ total.
    agg = df.groupby("layer").agg({**{nm: "mean" for nm in names}, "lam_total": "sum"})
    Xl = agg[names].to_numpy(dtype=np.float64)
    layer_lam = agg["lam_total"].to_numpy(dtype=np.float64)
    yl = np.log10(np.maximum(layer_lam, 1e-30))
    layer_level = regression_report(Xl, yl, names)
    layer_level["spread_ratio"] = float(layer_lam.max() / max(layer_lam.min(), 1e-30))
    head_lam = df["lam_total"].to_numpy(dtype=np.float64)
    head_spread = float(head_lam.max() / max(head_lam.min(), 1e-30))

    r2 = head_primary["ols_r2"]
    gate_pass = bool(r2 >= GATE_R2)
    return dict(
        model=spec.label,
        rope=rope_ready,
        predictors=names,
        primary_response="log10_lam_total (per-head λ-mass total)",
        head_level=head_primary,
        head_level_top_share=head_top,
        layer_level=layer_level,
        head_spread_ratio=head_spread,
        validation=validation,
        gate=dict(
            r2=r2,
            threshold=GATE_R2,
            gate_pass=gate_pass,
            outcome=(
                "CONFIRM: role is real allocation signal (spec a role-prior lever)"
                if gate_pass
                else "honest null: the query-weighted spectrum already subsumes "
                "function"
            ),
        ),
    )


def main(cfg: Config):
    run = (
        create_run("storm_role_regression", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("storm_role_regression", cfg)
    )

    frames: list[pd.DataFrame] = []
    per_model: dict[str, dict] = {}
    for label in cfg.models:
        spec = MODEL_SPECS[label]
        print(f"\n=== {label} ===", flush=True)
        df, validation = _head_rows_for_model(spec, cfg.budget)
        per_model[label] = _model_verdict(df, spec, validation)
        frames.append(df)

    write_metrics(run, pd.concat(frames, ignore_index=True))

    # Layer-level aggregate parquet (the 35×-spread-analog view).
    layer_frames = []
    for label, df in zip(cfg.models, frames):
        names = _predictor_names(bool(MODEL_SPECS[label].rope_name))
        agg = (
            df.groupby("layer")
            .agg({**{nm: "mean" for nm in names}, "lam_total": "sum"})
            .reset_index()
        )
        agg.insert(0, "model", label)
        layer_frames.append(agg)
    write_metrics(run, pd.concat(layer_frames, ignore_index=True), name="layer_metrics")

    # ---- pre-registered gate, evaluated verbatim per model -------------------
    # The gate is a single R² threshold; with two substrates the honest overall
    # statement is per-model, never collapsed through an any/all shortcut: a
    # split verdict is reported as a split.
    passes = {lb: v["gate"]["gate_pass"] for lb, v in per_model.items()}
    if all(passes.values()):
        overall_outcome = "CONFIRM on every model: role is real allocation signal"
    elif not any(passes.values()):
        overall_outcome = (
            "honest null on every model: the query-weighted spectrum already "
            "subsumes function"
        )
    else:
        overall_outcome = "SPLIT: " + "; ".join(
            f"{lb} {'CONFIRM' if p else 'honest null'}" for lb, p in passes.items()
        )
    verdict = dict(
        task="storm Task-4 head-role regression",
        gate_verbatim=(
            "R^2 >= 0.5 => role is real allocation signal (spec a role-prior "
            "lever). R^2 < 0.5 => honest null: the query-weighted spectrum "
            "already subsumes function."
        ),
        primary_regression=(
            "head-level OLS R^2 of log10(per-head lambda-mass total) on the "
            "role predictors, per model (declared in the module docstring "
            "before running)"
        ),
        budget=cfg.budget,
        main_event="qwen3-0.6b (RoPE + QK-norm); gpt2 lacks the positional-"
        "frequency role component (no RoPE)",
        per_model=per_model,
        overall=dict(
            per_model_gate_pass=passes,
            outcome=overall_outcome,
        ),
        git_sha=git_sha(),
    )
    (run / "verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n" + "=" * 78)
    print("STORM TASK-4 VERDICT — head-role regression")
    print("=" * 78)
    for label, v in per_model.items():
        g = v["gate"]
        print(
            f"{label}: head OLS R^2={g['r2']:.3f} (rank {v['head_level']['rank_ols_r2']:.3f}) "
            f"layer R^2={v['layer_level']['ols_r2']:.3f} "
            f"dominant={v['head_level']['dominant_by_univariate_r2']} -> "
            f"{'PASS' if g['gate_pass'] else 'null'}"
        )
    print(json.dumps(verdict["overall"], indent=2))
    print(f"\n-> {run}")
    return run


if __name__ == "__main__":
    main(tyro.cli(Config))
