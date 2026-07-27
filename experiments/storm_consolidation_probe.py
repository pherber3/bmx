"""Storm Task-9 — consolidation probe: superposed / sequence-level storage
(plan `docs/superpowers/plans/2026-07-26-storm-gates.md` Task 9; the skeptic's
Challenge-2 experiment).

Question: on the consolidation-friendly layer, do m << S consolidated
representatives (cluster centroids of the layer's keys, in the query-weighted
metric) at MATCHED TOTAL BITS hold logit distortion / the retrieval signature
vs the per-token K4 pack?

Pre-registered gate (verbatim, plan Task 9):
    holds within the codec's own quality margin  =>  a genuinely new axis
    opens (order-cost accounting next).  Fails  =>  the token-identity
    doctrine gets its first direct positive defense.

Gate operationalization (pre-registered HERE, before any run)
-------------------------------------------------------------
* "the codec's own margin" at a layer = the quality band the codec itself
  spans between its two shipped operating budgets on the same instrument:
      margin_mean(layer)  = |dist_pack@2.2  - dist_pack@2.5|
      margin_worst(layer) = |worst_pack@2.2 - worst_pack@2.5|
* A pure-storage consolidation arm HOLDS at budget b iff BOTH prongs pass:
      mean prong :  dist_cons(b)  <= dist_pack(b)  + margin_mean(layer)
      worst prong:  worst_cons(b) <= worst_pack(b) + margin_worst(layer)
  (the worst prong is the retrieval/needle signature — a mean-only gate could
  pass while a single merged needle token is destroyed), AND m/S <= 0.5 (the
  "m << S" stipulation; at the shipped budgets m/S lands near 0.15).
* The GATE arm is pure-storage consolidation on the CONSOLIDATION-FRIENDLY
  layer; both pre-registered position handlings (`cons_pure`, and `cons_post`
  on RoPE models) are legitimate pure-consolidation implementations, so the
  model CONFIRMS iff EITHER holds at >= 1 budget there. The steepest layer is
  contrast only (the verdict carries the spectrum-dependence, per the task
  brief). `cons_assign` (token identity retained) is a DIAGNOSTIC bridge,
  never the gate arm — see "Storage models".

Layer selection (per the task brief): d_eff(layer) = tr(lam)/lam_max of the
banked pack's query-weighted eigenspectrum (lam = spectrum of
W^{1/2} Sigma_k W^{1/2}, budget-independent). LOWEST d_eff = the
consolidation-friendly diagnostic — the weighted key energy concentrates in
few directions, exactly where m centroids have the best shot at covering the
key distribution in the metric that matters (k-means distortion decays like
m^(-2/D) in effective dimension D). The task brief labels this end "flattest";
operationally the selection IS argmin d_eff and the contrast layer argmax
d_eff, both run so the verdict carries the spectrum-dependence.

Storage models (stated explicitly; every bit the read path needs is charged,
per-sequence, matched DOWNWARD to the pack's model-level payload bpe — the
consolidation arm never receives more bits than the pack it challenges; the
pack's own conventions follow the Task-2 precedent: pack bpe = payload-v2
model-level, the pack ships with the model, while the consolidation charge is
wholly per-sequence, which only handicaps consolidation):

* cons_pure — TRUE consolidation, token identity GONE. Stores m centroid rows
  fp16 over C coords (16*m*C) + one per-centroid COUNT (ceil_log2(S) bits —
  the merged-softmax multiplicity weight) + one per-centroid REPRESENTATIVE
  POSITION (ceil_log2(S) bits, RoPE models only; a no-RoPE read never
  consults position). No per-token data survives.
* cons_post (RoPE models only) — cons_pure with position handled by
  consolidating POST-RoPE keys (position baked into the stored rows): no
  position charge, but clustering must live in post-RoPE space, where the
  metric is the plain (forward-rotated) query second moment.
* cons_assign — the diagnostic bridge: m centroid rows fp16 + a PER-TOKEN
  assignment table (S * ceil_log2(m) bits, position-indexed, so counts AND
  positions are implied — RoPE-at-read rotates each token's centroid at the
  token's TRUE position, and V stays per-token exact). This is per-layer VQ,
  NOT consolidation (token identity retained); it isolates geometry-coverage
  from identity-loss.

Evaluation embedding (why the shipped instruments apply unchanged): the
merged m-term softmax with counts is IDENTICAL to the expanded S-row softmax
(sum_j n_j exp(q.c_j) = sum_s exp(q.c_{a(s)})), and the expanded (S, C) key
matrix (row s = stored centroid of s's cluster) weights each cluster by its
multiplicity in the Frobenius norm exactly as the true logit matrix weights
positions — so `logit_distortion` / `attn_output_distortion` score the
consolidated cache through its expanded read-form with zero convention drift.

Clustering respects the query metric: k-means runs on X = M @ enc (the
banked pack's encoder; enc @ enc^T = W exactly, so Euclidean distance in X IS
the W-metric on keys) — cons_post uses X = M_post @ Wh_post with Wh_post the
symmetric square root of the forward-rotated pooled query moment. The
W-metric Frechet mean under ANY fixed PSD W is the plain arithmetic mean, so
centroids are computed as key-space member means (then fp16-roundtripped —
the stored form).

RoPE position handling — the mechanism's structural difficulty, measured
head-on: cons_pure rotates centroid j at ONE representative position
(rounded mean member position) for every member token; the per-arm
`dist_pos_oracle` column re-reads the SAME centroids rotated at each member's
TRUE position (an uncharged oracle), so (dist - dist_pos_oracle) isolates
exactly what the single-position approximation costs at identical centroids.
cons_post avoids the representative position by paying the post-RoPE
clustering cost instead. Both are reported; whichever bites is flagged in the
verdict.

V-side accounting stance (stated, not hidden): the gate is the K-side
instrument at matched K-side bits; V bits are held out of the accounting on
BOTH sides (pack: exact fp16 V uncharged; cons_pure/cons_post: merged
cluster-mean V uncharged — their V storage is m rows vs the pack's S, so
ignoring V bits is conservative AGAINST consolidation on bits, while the
merged-V quality cost still shows honestly in the softmax-output errors).
`out_err` is the shipped non-causal `attn_output_distortion`; `out_err_causal`
is the true-read-convention causal softmax-output error (forward-rotated
queries at their true positions) — the merged terms change the denominator,
and the causal column is the honest account of that.

fp32 experiment path per repo convention (caches fp16, moment/whitener math
fp64 through the shipped machinery).
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics
from bmx.cache.collect import from_matrix, to_matrix
from bmx.cache.metrics import _expand_kv, attn_output_distortion, logit_distortion
from bmx.cache.rope import apply_rope
from bmx.cache.spectral import assemble_whitener, load_packs, spectral_quantize
from experiments._k4_common import load_layer_keys, setup_rope

BUDGETS = (2.2, 2.5)  # the pre-registered K4 operating band (plan Task 9)
GATE_M_OVER_S = 0.5  # "m << S": a hold with m/S above this is vacuous
PURE_ARMS = ("cons_pure", "cons_post")  # pure-storage (gate-eligible) arms


# ---------------------------------------------------------------------------
# Bit accounting (pre-registered; pinned by test_consolidation_bit_accounting)
# ---------------------------------------------------------------------------


def ceil_log2(n: int) -> int:
    """Bits to name one of n alternatives: ceil(log2 n); 0 when n == 1."""
    assert n >= 1, f"n must be >= 1; got {n}"
    return int(math.ceil(math.log2(n))) if n > 1 else 0


def pure_row_meta_bits(S: int, *, rope: bool) -> int:
    """Per-centroid metadata for the pure model: a count in [1, S]
    (ceil_log2(S) bits) plus, on RoPE models, a representative position in
    [0, S) (ceil_log2(S) bits)."""
    return ceil_log2(S) * (2 if rope else 1)


def pure_bits(m: int, S: int, C: int, *, rope: bool) -> int:
    """Total stored bits for a pure centroid cache: m fp16 rows over C coords
    + per-centroid count (+ representative position on RoPE models)."""
    return m * (16 * C + pure_row_meta_bits(S, rope=rope))


def assign_bits(m: int, S: int, C: int) -> int:
    """Total stored bits for the assignment (VQ) model: m fp16 rows + a
    position-indexed per-token assignment table (S entries of ceil_log2(m)
    bits; counts and positions are implied by the table, so neither is
    charged)."""
    return m * 16 * C + S * ceil_log2(m)


def matched_m_pure(target_bits: float, S: int, C: int, *, rope: bool) -> int:
    """Largest m with pure_bits(m) <= target_bits (floor matching — the
    consolidation arm never receives more bits than the pack), clamped to
    [1, S]."""
    m = int(target_bits // (16 * C + pure_row_meta_bits(S, rope=rope)))
    return max(1, min(m, S))


def matched_m_assign(target_bits: float, S: int, C: int) -> int:
    """Largest m with assign_bits(m) <= target_bits, clamped to [1, S]. Walks
    down from the no-assignment-charge upper bound floor(target/16C) (the
    ceil_log2(m) table charge is <= S*ceil_log2(S) bits ~ one or two centroid
    rows, so the walk is short); the first feasible m from above is the
    largest feasible m."""
    m = max(1, min(int(target_bits // (16 * C)), S))
    while m > 1 and assign_bits(m, S, C) > target_bits:
        m -= 1
    return m


# ---------------------------------------------------------------------------
# Whitened k-means (pure; pinned by the offline tests)
# ---------------------------------------------------------------------------


def _kmeans_pp_init(X: torch.Tensor, m: int, gen: torch.Generator) -> torch.Tensor:
    """k-means++ seeding (deterministic given gen). Degenerate duplicates
    (all remaining distances 0) fall back to uniform choice among unchosen."""
    S = X.shape[0]
    idx = torch.empty(m, dtype=torch.int64)
    idx[0] = torch.randint(S, (1,), generator=gen)
    d2 = ((X - X[idx[0]]) ** 2).sum(dim=1)
    for j in range(1, m):
        if float(d2.sum()) <= 0.0:
            chosen = torch.zeros(S, dtype=torch.bool)
            chosen[idx[:j]] = True
            rest = (~chosen).nonzero(as_tuple=True)[0]
            idx[j] = rest[torch.randint(rest.numel(), (1,), generator=gen)]
        else:
            idx[j] = torch.multinomial(d2, 1, generator=gen)
        d2 = torch.minimum(d2, ((X - X[idx[j]]) ** 2).sum(dim=1))
    return X[idx].clone()


def _lloyd(
    X: torch.Tensor, cent: torch.Tensor, iters: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Lloyd iterations with deterministic empty-cluster repair (the point
    farthest from its assigned centroid claims each empty cluster). Returns
    (assign, centroids, inertia)."""
    S = X.shape[0]
    m = cent.shape[0]
    assign: torch.Tensor | None = None
    for _ in range(iters):
        d2 = torch.cdist(X, cent).pow(2)  # (S, m)
        new_assign = d2.argmin(dim=1)
        counts = torch.bincount(new_assign, minlength=m)
        if (counts == 0).any():
            # Farthest points claim empty clusters; repeated until none remain
            # (a donation from a singleton cluster can empty the donor).
            own_d2 = d2.gather(1, new_assign.view(-1, 1)).squeeze(1)
            far = own_d2.argsort(descending=True, stable=True)
            used = torch.zeros(S, dtype=torch.bool)
            fi = 0
            while (counts == 0).any():
                for e in (counts == 0).nonzero(as_tuple=True)[0].tolist():
                    while used[far[fi]]:
                        fi += 1
                        assert fi < S, "empty-cluster repair exhausted points"
                    p = int(far[fi])
                    used[p] = True
                    new_assign[p] = e
                counts = torch.bincount(new_assign, minlength=m)
        if assign is not None and torch.equal(new_assign, assign):
            break
        assign = new_assign
        cent = torch.zeros_like(cent).index_add_(0, assign, X)
        cent = cent / counts.clamp_min(1).unsqueeze(1).float()
    assert assign is not None
    inertia = float(torch.cdist(X, cent).pow(2).gather(1, assign.view(-1, 1)).sum())
    return assign, cent, inertia


def kmeans_whitened(
    X: torch.Tensor, m: int, *, iters: int = 100, seed: int = 0, n_init: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic k-means on the rows of an already metric-transformed
    (S, D) matrix X: Euclidean distance in X IS the intended quadratic-form
    metric on the original rows when X = M @ T with T @ T^T = W (the pack's
    enc satisfies enc @ enc^T = W exactly; Wh_post is symmetric with
    Wh_post^2 = W_post). Best of n_init seeded k-means++ restarts by inertia.
    m == S short-circuits to the identity clustering (every row its own
    centroid) — the degenerate case the test pins."""
    S, _D = X.shape
    assert 1 <= m <= S, f"need 1 <= m={m} <= S={S}"
    Xf = X.float()
    if m == S:
        return torch.arange(S, dtype=torch.int64), Xf.clone()
    best: tuple[torch.Tensor, torch.Tensor, float] | None = None
    for i in range(n_init):
        gen = torch.Generator().manual_seed(seed + 9973 * i)
        cent0 = _kmeans_pp_init(Xf, m, gen)
        assign, cent, inertia = _lloyd(Xf, cent0, iters)
        if best is None or inertia < best[2]:
            best = (assign, cent, inertia)
    assert best is not None
    return best[0], best[1]


class Consolidation(NamedTuple):
    """One layer's consolidated cache in stored form."""

    assign: torch.Tensor  # (S,) int64 — cluster of each token (evaluation aid)
    counts: torch.Tensor  # (m,) int64, all >= 1 — merged-softmax weights
    centroids: torch.Tensor  # (m, C) fp32, fp16-roundtripped — STORED rows
    rep_pos: torch.Tensor  # (m,) int64 — rounded mean member position


def consolidate(
    M: torch.Tensor,
    X: torch.Tensor,
    m: int,
    *,
    iters: int = 100,
    seed: int = 0,
    n_init: int = 4,
) -> Consolidation:
    """Cluster the rows of the metric-transformed X (same row order as the
    (S, C) source matrix M); centroids are the plain arithmetic means of
    member rows of M (the W-metric Frechet mean under any fixed PSD W),
    stored fp16 (.half().float() — the storage model charges 16 bits/coord).
    rep_pos[j] = round(mean member position) — the single position the pure
    RoPE read rotates centroid j at."""
    S, C = M.shape
    assert X.shape[0] == S, f"X rows {X.shape[0]} != M rows {S}"
    assign, _ = kmeans_whitened(X, m, iters=iters, seed=seed, n_init=n_init)
    counts = torch.bincount(assign, minlength=m)
    assert (counts >= 1).all(), "empty cluster survived k-means repair"
    cent = torch.zeros(m, C, dtype=torch.float32).index_add_(0, assign, M.float())
    cent = (cent / counts.unsqueeze(1).float()).half().float()
    pos_sum = torch.zeros(m, dtype=torch.float64).index_add_(
        0, assign, torch.arange(S, dtype=torch.float64)
    )
    rep_pos = torch.round(pos_sum / counts.double()).long()
    return Consolidation(assign=assign, counts=counts, centroids=cent, rep_pos=rep_pos)


def expanded_key_matrix(cons: Consolidation) -> torch.Tensor:
    """(S, C) expanded read-form: row s = stored centroid of s's cluster.
    Exactly the merged m-term softmax re-expressed per position (see module
    docstring) — the object the shipped instruments consume."""
    return cons.centroids[cons.assign]


def cluster_mean_rows(V_mat: torch.Tensor, cons: Consolidation) -> torch.Tensor:
    """(S, C_v) expanded merged-V read-form: row s = fp16-stored mean of s's
    cluster's V rows (what a pure consolidated cache would store and read)."""
    m = cons.counts.shape[0]
    Cv = V_mat.shape[1]
    vc = torch.zeros(m, Cv, dtype=torch.float32).index_add_(
        0, cons.assign, V_mat.float()
    )
    vc = (vc / cons.counts.unsqueeze(1).float()).half().float()
    return vc[cons.assign]


# ---------------------------------------------------------------------------
# Post-RoPE metric + causal diagnostics
# ---------------------------------------------------------------------------


def post_rope_whitener(
    q_read: torch.Tensor, h_kv: int, *, ridge: float = 1e-3
) -> torch.Tensor:
    """(C, C) fp64 symmetric square root Wh_post of the POST-RoPE query metric
    W_post: per-kv-head pooled second moment of the forward-rotated stored
    queries at their true positions (the queries that actually read a
    post-RoPE key cache), assembled block-diagonally by the shipped
    `assemble_whitener`. k-means on x = k_post @ Wh_post is k-means under
    W_post = Wh_post @ Wh_post."""
    h, T, d = q_read.shape
    assert h % h_kv == 0, f"h={h} not divisible by h_kv={h_kv}"
    grp = h // h_kv
    q64 = q_read.double()
    W = torch.zeros(h_kv, d, d, dtype=torch.float64)
    for j in range(h_kv):
        qj = q64[j * grp : (j + 1) * grp].reshape(-1, d)
        W[j] = qj.mT @ qj / (grp * T)
    Wh, _ = assemble_whitener(W, ridge=ridge)
    return Wh


def causal_worst_and_output(
    K_true_read: torch.Tensor,
    K_hat_read: torch.Tensor,
    V_true: torch.Tensor,
    V_hat: torch.Tensor,
    q_read: torch.Tensor,
) -> tuple[float, float]:
    """(worst_ratio, out_err_causal) under the true read convention (stored
    queries forward-rotated at their true absolute positions [S-T, S), causal
    mask).

    worst_ratio — the retrieval/needle signature: max over causal (head,
    query, source) of |q_t.k_hat_s - q_t.k_s| / max causal |q_t.k_s| (the
    1/sqrt(d) scale cancels in the ratio).

    out_err_causal — the honest merged-softmax account: per-head relative
    Frobenius error of softmax(causal logits) @ V, mean over heads (merged
    terms change the denominator; this column prices exactly that)."""
    q = q_read.float()
    h, T, d = q.shape
    S = K_true_read.shape[1]
    Kt = _expand_kv(K_true_read.float(), h)
    Kh = _expand_kv(K_hat_read.float(), h)
    Vt = _expand_kv(V_true.float(), h)
    Vh = _expand_kv(V_hat.float(), h)
    lt = q @ Kt.transpose(-1, -2) / (d**0.5)  # (h, T, S)
    lh = q @ Kh.transpose(-1, -2) / (d**0.5)
    q_pos = torch.arange(S - T, S).view(T, 1)
    s_pos = torch.arange(S).view(1, S)
    causal = (s_pos <= q_pos).view(1, T, S)
    dl = (lh - lt).abs().masked_fill(~causal, 0.0)
    max_true = float(lt.abs().masked_fill(~causal, 0.0).max())
    worst_ratio = float(dl.max()) / max(max_true, 1e-12)
    at = torch.softmax(lt.masked_fill(~causal, float("-inf")), dim=-1)
    ah = torch.softmax(lh.masked_fill(~causal, float("-inf")), dim=-1)
    ot = at @ Vt  # (h, T, d_v)
    oh = ah @ Vh
    num = (oh - ot).flatten(1).norm(dim=-1)
    den = ot.flatten(1).norm(dim=-1).clamp_min(1e-12)
    return worst_ratio, float((num / den).mean())


def d_eff(lam: torch.Tensor) -> float:
    """tr(lam)/lam_max of a query-weighted eigenspectrum — the task brief's
    consolidation diagnostic (lowest = consolidation-friendly)."""
    lam64 = lam.double().clamp_min(0.0)
    return float(lam64.sum() / lam64.max().clamp_min(1e-30))


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Config:
    # Held-out (offset-0) cache — distinct from the pack-fit corpus caches.
    cache_path: str = "results/cache/gpt2_1024.safetensors"
    # Banked K4 spectral packs (+ .json sidecar) — the per-token champion AND
    # the source of the query-weighted spectrum/enc the probe clusters in.
    pack_path: str = "results/cache/k4_packs_gpt2.safetensors"
    model_label: str = "gpt2"
    model_name: str = ""  # HF repo id for RoPE tables; "" => no-RoPE (gpt2)
    budgets: tuple[float, ...] = BUDGETS
    kmeans_iters: int = 100
    kmeans_restarts: int = 4
    seed: int = 0
    ridge: float = 1e-3  # post-RoPE whitener ridge (RoPE models only)
    out_root: str = ""


class _LayerCtx(NamedTuple):
    k_pre: torch.Tensor  # (h_kv, S, d) fp16
    M_pre: torch.Tensor  # (S, C) fp32
    M_post: torch.Tensor  # (S, C) fp32 (== M_pre for no-RoPE)
    K_true_read: torch.Tensor  # (h_kv, S, d) fp32 post-RoPE (== pre for no-RoPE)
    V: torch.Tensor  # (h_kv, S, d_v) fp32
    V_mat: torch.Tensor  # (S, C_v) fp32
    Q: torch.Tensor  # (h, T, d) fp32 stored (pre-RoPE) probe queries
    q_read: torch.Tensor  # (h, T, d) fp32 forward-rotated at [S-T, S)
    cos: torch.Tensor | None
    sin: torch.Tensor | None
    h_kv: int
    S: int
    C: int


def _build_ctx(kinds: dict[str, torch.Tensor], rope_ready: bool, get_cos_sin):
    k_pre = kinds["k_pre"]
    h_kv, S, d = k_pre.shape
    M_pre = to_matrix(k_pre)
    Q = kinds["q"].float()
    T = Q.shape[1]
    V = kinds["v"].float()
    if rope_ready:
        cos, sin = get_cos_sin(S)
        K_true_read = apply_rope(k_pre.float(), cos, sin)
        q_positions = torch.arange(S - T, S)
        q_read = apply_rope(Q, cos[q_positions], sin[q_positions])
    else:
        cos = sin = None
        K_true_read = k_pre.float()
        q_read = Q
    return _LayerCtx(
        k_pre=k_pre,
        M_pre=M_pre,
        M_post=to_matrix(K_true_read),
        K_true_read=K_true_read,
        V=V,
        V_mat=to_matrix(V),
        Q=Q,
        q_read=q_read,
        cos=cos,
        sin=sin,
        h_kv=h_kv,
        S=S,
        C=h_kv * d,
    )


def _score(
    ctx: _LayerCtx, K_hat_read: torch.Tensor, V_hat: torch.Tensor
) -> dict[str, float]:
    """All four instrument columns for one reconstructed read-form cache."""
    dist = logit_distortion(ctx.K_true_read, K_hat_read, ctx.Q)
    out_err = attn_output_distortion(ctx.K_true_read, ctx.V, K_hat_read, V_hat, ctx.Q)
    worst, out_causal = causal_worst_and_output(
        ctx.K_true_read, K_hat_read, ctx.V, V_hat, ctx.q_read
    )
    return dict(
        dist=dist, worst_ratio=worst, out_err=out_err, out_err_causal=out_causal
    )


def _rotate_rows(
    M_hat: torch.Tensor,
    h_kv: int,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Read-form keys: rotate row s of the (S, C) matrix at positions[s]."""
    return apply_rope(from_matrix(M_hat, h_kv).float(), cos[positions], sin[positions])


def _cons_rows(
    ctx: _LayerCtx,
    cfg: Config,
    rope_ready: bool,
    enc: torch.Tensor,
    Wh_post: torch.Tensor | None,
    bpe_target: float,
    cons_cache: dict,
) -> list[dict]:
    """The consolidation arms for one (layer, budget): cons_pure (+ cons_post
    on RoPE models) + cons_assign, each bit-matched downward to bpe_target."""
    S, C = ctx.S, ctx.C
    target_bits = bpe_target * S * C
    X_pre = ctx.M_pre @ enc.float()
    rows: list[dict] = []

    def _get_cons(space: str, m: int) -> Consolidation:
        key = (space, m)
        if key not in cons_cache:
            M, X = (
                (ctx.M_pre, X_pre)
                if space == "pre"
                else (ctx.M_post, ctx.M_post @ Wh_post.float())
            )
            cons_cache[key] = consolidate(
                M,
                X,
                m,
                iters=cfg.kmeans_iters,
                seed=cfg.seed,
                n_init=cfg.kmeans_restarts,
            )
        return cons_cache[key]

    # --- cons_pure: pre-RoPE clustering, representative-position read -------
    m_pure = matched_m_pure(target_bits, S, C, rope=rope_ready)
    cons = _get_cons("pre", m_pure)
    M_hat = expanded_key_matrix(cons)
    V_hat_merged = from_matrix(cluster_mean_rows(ctx.V_mat, cons), ctx.h_kv)
    if rope_ready:
        pos_rep = cons.rep_pos[cons.assign]
        K_hat = _rotate_rows(M_hat, ctx.h_kv, pos_rep, ctx.cos, ctx.sin)
        # Uncharged position ORACLE: same centroids, each member rotated at its
        # TRUE position — isolates the representative-position cost exactly.
        K_hat_oracle = _rotate_rows(M_hat, ctx.h_kv, torch.arange(S), ctx.cos, ctx.sin)
        oracle = _score(ctx, K_hat_oracle, V_hat_merged)
    else:
        K_hat = from_matrix(M_hat, ctx.h_kv).float()
        oracle = None
    rows.append(
        dict(
            arm="cons_pure",
            mode="pre_rep" if rope_ready else "none",
            m=m_pure,
            m_over_S=m_pure / S,
            bpe=pure_bits(m_pure, S, C, rope=rope_ready) / (S * C),
            **_score(ctx, K_hat, V_hat_merged),
            dist_pos_oracle=(oracle["dist"] if oracle else float("nan")),
            worst_pos_oracle=(oracle["worst_ratio"] if oracle else float("nan")),
        )
    )

    # --- cons_post: post-RoPE clustering, position baked in (RoPE only) -----
    if rope_ready:
        m_post = matched_m_pure(target_bits, S, C, rope=False)  # no position charge
        cons_p = _get_cons("post", m_post)
        K_hat_p = from_matrix(expanded_key_matrix(cons_p), ctx.h_kv).float()
        V_hat_p = from_matrix(cluster_mean_rows(ctx.V_mat, cons_p), ctx.h_kv)
        rows.append(
            dict(
                arm="cons_post",
                mode="post",
                m=m_post,
                m_over_S=m_post / S,
                bpe=pure_bits(m_post, S, C, rope=False) / (S * C),
                **_score(ctx, K_hat_p, V_hat_p),
                dist_pos_oracle=float("nan"),
                worst_pos_oracle=float("nan"),
            )
        )

    # --- cons_assign: VQ diagnostic bridge (token identity retained) --------
    m_asg = matched_m_assign(target_bits, S, C)
    cons_a = _get_cons("pre", m_asg)
    M_hat_a = expanded_key_matrix(cons_a)
    if rope_ready:
        K_hat_a = _rotate_rows(M_hat_a, ctx.h_kv, torch.arange(S), ctx.cos, ctx.sin)
    else:
        K_hat_a = from_matrix(M_hat_a, ctx.h_kv).float()
    rows.append(
        dict(
            arm="cons_assign",
            mode="true_pos" if rope_ready else "none",
            m=m_asg,
            m_over_S=m_asg / S,
            bpe=assign_bits(m_asg, S, C) / (S * C),
            **_score(ctx, K_hat_a, ctx.V),  # V per-token exact in this model
            dist_pos_oracle=float("nan"),
            worst_pos_oracle=float("nan"),
        )
    )
    return rows


def evaluate_gate(df: pd.DataFrame, cfg: Config) -> dict:
    """The pre-registered gate, evaluated verbatim (see module docstring):
    per (layer_class, budget), a pure-storage arm HOLDS iff dist <= dist_pack
    + margin_mean AND worst <= worst_pack + margin_worst AND m/S <= 0.5, with
    margins = the pack's own quality band across the two shipped budgets at
    that layer. Model CONFIRMS iff any pure arm holds at any budget on the
    consolidation-FRIENDLY layer."""
    budgets = sorted(df.budget.unique())
    per_class: dict[str, dict] = {}
    confirmed = False
    for lclass, g in df.groupby("layer_class"):
        pack = g[g.arm == "pack"].set_index("budget")
        assert len(pack) == len(budgets), f"pack rows missing for {lclass}"
        margin_mean = float(abs(pack["dist"].max() - pack["dist"].min()))
        margin_worst = float(abs(pack["worst_ratio"].max() - pack["worst_ratio"].min()))
        per_budget: dict[str, dict] = {}
        for b in budgets:
            p = pack.loc[b]
            arms: dict[str, dict] = {}
            for _, r in g[(g.budget == b) & (g.arm != "pack")].iterrows():
                holds_mean = bool(r["dist"] <= p["dist"] + margin_mean)
                holds_worst = bool(r["worst_ratio"] <= p["worst_ratio"] + margin_worst)
                small_m = bool(r["m_over_S"] <= GATE_M_OVER_S)
                holds = holds_mean and holds_worst and small_m
                arms[r["arm"]] = dict(
                    mode=r["mode"],
                    m=int(r["m"]),
                    m_over_S=float(r["m_over_S"]),
                    bpe=float(r["bpe"]),
                    dist=float(r["dist"]),
                    worst_ratio=float(r["worst_ratio"]),
                    out_err=float(r["out_err"]),
                    out_err_causal=float(r["out_err_causal"]),
                    dist_over_pack=float(r["dist"] / max(p["dist"], 1e-12)),
                    holds_mean=holds_mean,
                    holds_worst=holds_worst,
                    holds=holds,
                )
                if holds and lclass == "friendly" and r["arm"] in PURE_ARMS:
                    confirmed = True
            per_budget[f"{b:g}"] = dict(
                pack=dict(
                    bpe=float(p["bpe"]),
                    dist=float(p["dist"]),
                    worst_ratio=float(p["worst_ratio"]),
                    out_err=float(p["out_err"]),
                    out_err_causal=float(p["out_err_causal"]),
                ),
                arms=arms,
            )
        per_class[str(lclass)] = dict(
            layer=int(g.layer.iloc[0]),
            d_eff=float(g.d_eff.iloc[0]),
            margin_mean=margin_mean,
            margin_worst=margin_worst,
            per_budget=per_budget,
        )
    return dict(
        task="storm Task-9 consolidation probe (superposed storage)",
        model=cfg.model_label,
        gate_arms=list(PURE_ARMS),
        gate_m_over_s=GATE_M_OVER_S,
        per_layer_class=per_class,
        gate_outcome=(
            "CONFIRM — consolidated representatives hold within the codec's "
            "own margin (order-cost accounting next)"
            if confirmed
            else "FAIL — the token-identity doctrine gets its first direct "
            "positive defense"
        ),
        confirmed=bool(confirmed),
        git_sha=git_sha(),
    )


def main(cfg: Config):
    assert Path(cfg.cache_path).exists(), (
        f"cache {cfg.cache_path!r} absent — regenerate via "
        "`uv run python experiments/collect_cache.py --model-name <model> ...`"
    )
    assert Path(cfg.pack_path).exists(), f"pack {cfg.pack_path!r} absent"
    run = (
        create_run("storm_consolidation_probe", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("storm_consolidation_probe", cfg)
    )

    layer_keys = load_layer_keys(cfg.cache_path)
    layers = sorted(layer_keys.keys())
    rope_ready, get_cos_sin = setup_rope(cfg.model_name, layer_keys, layers)

    packs_by_budget = {float(b): load_packs(cfg.pack_path, b) for b in cfg.budgets}
    packs0 = packs_by_budget[float(cfg.budgets[0])]
    assert sorted(packs0.keys()) == layers, (
        f"pack layers {sorted(packs0.keys())} != cache layers {layers}"
    )

    # Layer selection: lam is budget-independent — d_eff from the banked packs.
    d_effs = {li: d_eff(packs0[li].lam) for li in layers}
    friendly = min(d_effs, key=lambda li: (d_effs[li], li))
    steep = max(d_effs, key=lambda li: (d_effs[li], li))
    selected = {"friendly": friendly}
    if steep != friendly:
        selected["steep"] = steep
    print(
        f"d_eff per layer: { {li: round(v, 2) for li, v in d_effs.items()} }\n"
        f"friendly (argmin) = layer {friendly} (d_eff {d_effs[friendly]:.2f}), "
        f"steep (argmax) = layer {steep} (d_eff {d_effs[steep]:.2f})",
        flush=True,
    )

    rows: list[dict] = []
    for lclass, li in selected.items():
        ctx = _build_ctx(layer_keys[li], rope_ready, get_cos_sin)
        enc = packs0[li].enc  # basis is budget-independent
        Wh_post = (
            post_rope_whitener(ctx.q_read, ctx.h_kv, ridge=cfg.ridge)
            if rope_ready
            else None
        )
        cons_cache: dict = {}
        for b in cfg.budgets:
            pack = packs_by_budget[float(b)][li]
            M_hat_pack, bpe_pack = spectral_quantize(ctx.M_pre, pack)
            if rope_ready:
                K_hat_pack = _rotate_rows(
                    M_hat_pack, ctx.h_kv, torch.arange(ctx.S), ctx.cos, ctx.sin
                )
            else:
                K_hat_pack = from_matrix(M_hat_pack, ctx.h_kv).float()
            base = dict(
                model=cfg.model_label,
                layer_class=lclass,
                layer=li,
                d_eff=d_effs[li],
                S=ctx.S,
                C=ctx.C,
                budget=float(b),
                bpe_target=bpe_pack,
            )
            rows.append(
                dict(
                    **base,
                    arm="pack",
                    mode="per_token",
                    m=-1,
                    m_over_S=float("nan"),
                    bpe=bpe_pack,
                    **_score(ctx, K_hat_pack, ctx.V),
                    dist_pos_oracle=float("nan"),
                    worst_pos_oracle=float("nan"),
                )
            )
            for r in _cons_rows(
                ctx, cfg, rope_ready, enc, Wh_post, bpe_pack, cons_cache
            ):
                assert r["bpe"] <= bpe_pack + 1e-9, (
                    f"consolidation over budget: {r['arm']} bpe {r['bpe']:.4f} "
                    f"> pack {bpe_pack:.4f}"
                )
                rows.append(dict(**base, **r))
            got = [r for r in rows if r["budget"] == float(b) and r["layer"] == li]
            print(
                f"[{lclass} layer {li} | budget {b:g}] "
                + "  ".join(
                    f"{r['arm']}: dist {r['dist']:.4f} worst {r['worst_ratio']:.3f}"
                    f" (m={r['m']})"
                    for r in got
                ),
                flush=True,
            )

    df = pd.DataFrame(rows)
    write_metrics(run, df)

    verdict = evaluate_gate(df, cfg)
    verdict["d_eff_all_layers"] = {str(li): d_effs[li] for li in layers}
    (run / "verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n" + "=" * 78)
    print("STORM TASK-9 VERDICT — consolidation probe (superposed storage)")
    print("=" * 78)
    print(
        json.dumps(
            {k: v for k, v in verdict.items() if k != "d_eff_all_layers"}, indent=2
        )
    )
    print(f"\n-> {run}")
    return run


if __name__ == "__main__":
    main(tyro.cli(Config))
