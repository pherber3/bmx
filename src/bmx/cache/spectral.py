"""K4 spectral codec: query-weighted eigenbasis + waterfilled bit allocation.

The metric that scores the task is E[(qᵀ R_p (k - k_hat))²] where R_p is the
per-position RoPE rotation. That equals eᵀ W e with W = E[R_pᵀ q qᵀ R_p] —
block-diagonal per kv-head (attention logits never couple heads). Substituting
u = W^{1/2} k reduces weighted rate-distortion to plain MSE on covariance
W^{1/2} Σ_k W^{1/2}: the optimal basis is that matrix's eigenbasis and bits
waterfill on its eigenvalues (spec §3, theory grounding §8 of
docs/superpowers/specs/2026-07-12-k4-spectral-codec-design.md).

All moment/eig math is fp64; codec application is fp32.

Accounting modes (expressions are the scientific record; never re-derive or
reassociate — every change to these is a NEW NAMED MODE with the prior
expression documented beside it):

- skeptic-v1 (full-C fp16): `16·C/S + tier_bits(tiers, S)`. Charges a full
  C×C fp16 decoder matrix regardless of how many directions actually carry
  nonzero bits. This is the expression every parquet before 2026-07-23
  measured. `skeptic_charge`'s defaults reproduce this bit-exactly forever.
- skeptic-v2 (used-columns): `16·c_used/S + tier_bits(tiers, S)`. Only the
  `c_used` decoder columns whose direction carries nonzero bits are ever
  read at decode (see `test_dropped_decoder_columns_never_read` — the
  license for this: mutating a dropped column provably cannot change the
  reconstruction). Charging the full C columns over-counts by
  `16·(C - c_used)/S`.
- skeptic-v2-int8 (used-columns, int8 decoder): `8·c_used/S +
  16·c_used/(S·C) + tier_bits(tiers, S)`. Same used-columns charge but the
  decoder itself is stored int8 (8 bits/entry) plus one fp16 per-column
  scale amortized over S rows (16·c_used/(S·C)). Generalized by
  `mixed_dec_charge` (K4 local-levers Task 1): `c_int8` of the `c_used`
  columns int8-stored, the rest fp16 — reduces to skeptic-v2 at `c_int8=0`
  and to skeptic-v2-int8 (blanket) at `c_int8=c_used`; the tier-gated case
  (`0 < c_int8 < c_used`, driven by `dec_quant="int8_t{T}"` via
  `dec_quant_threshold`) is the interior of that range.
- payload-v1: `mean(bits) + scale_bits(group)`. Charges the groupwise fp16
  scale for every one of the C directions, including zero-bit (dropped)
  ones that store no payload and so need no scale.
  See `test_dropped_decoder_columns_never_read`.
- payload-v2: `mean(bits) + scale_bits(group)·(c_used/C)`. Prorates the
  scale term by the fraction of directions actually carrying bits, removing
  payload-v1's phantom scales on dropped directions.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file, save_file

from bmx.cache.codecs import (
    allocate_bits_from_variance,
    scale_bits,
    tier_bits,
)
from bmx.cache.rope import _rotate_half
from bmx.cache.triton_dequant_attention import (
    pack_codes,
    pack_signed_codes,
    unpack_codes,
    unpack_signed_codes,
)
from bmx.quant.rtn import rtn_dequantize_packed, rtn_quantize_packed


def key_second_moment(M: torch.Tensor) -> torch.Tensor:
    """Uncentered per-channel second moment MᵀM/S of an (S, C) fp matrix, fp64."""
    assert M.dim() == 2, f"M must be (S, C); got {tuple(M.shape)}"
    Md = M.double()
    return Md.mT @ Md / M.shape[0]


def query_position_moment(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    h_kv: int,
    *,
    position_stride: int = 8,
    w_rope: str = "frozen",
) -> torch.Tensor:
    """W blocks (h_kv, d, d): E over probe queries and sampled key positions of
    (R_pᵀ q)(R_pᵀ q)ᵀ, pooled over each kv-head's GQA query group.

    R_pᵀ q is RoPE at position p with negated sin (inverse rotation). cos/sin
    are (S, d) tables from rope_cos_sin; pass cos=ones/sin=zeros for no-RoPE
    models (gpt2) — then W is the plain pooled query second moment.

    w_rope="frozen" (default, bit-exact): the shipped instrument — inverse
    rotation R_p^T q at uniform-strided absolute positions (the query's own
    rotation frozen at zero). w_rope="rotated" (math review 2026-07-24 #3):
    the causal-attention-corrected moment — FORWARD rotation R_m q (sign of
    sin flipped) at the same strided positions read as relative offsets
    m = p, each weighted triangularly (#pairs at offset m ~ S - m). Even
    plane terms agree between the two up to the offset distribution; the odd
    sin*cos plane term flips sign — identical when sin == 0 (no-RoPE models).
    """
    h, T, d = q.shape
    assert h % h_kv == 0, f"h={h} not divisible by h_kv={h_kv}"
    grp = h // h_kv
    S = cos.shape[0]
    q64 = q.double()
    W = torch.zeros(h_kv, d, d, dtype=torch.float64)
    assert w_rope in ("frozen", "rotated"), f"unknown w_rope {w_rope!r}"
    positions = list(range(0, S, position_stride))
    if w_rope == "frozen":
        for p in positions:
            cp = cos[p].double().view(1, 1, d)
            sp = sin[p].double().view(1, 1, d)
            q_rot = q64 * cp + _rotate_half(q64) * (-sp)  # (h, T, d) = R_pᵀ q
            for j in range(h_kv):
                qj = q_rot[j * grp : (j + 1) * grp].reshape(-1, d)
                W[j] += qj.mT @ qj
        W /= len(positions) * grp * T
    else:
        total = 0.0
        for p in positions:
            cp = cos[p].double().view(1, 1, d)
            sp = sin[p].double().view(1, 1, d)
            q_rot = q64 * cp + _rotate_half(q64) * sp  # (h, T, d) = R_m q (forward)
            wt = float(S - p)  # triangular offset weight
            for j in range(h_kv):
                qj = q_rot[j * grp : (j + 1) * grp].reshape(-1, d)
                W[j] += wt * (qj.mT @ qj)
            total += wt
        W /= total * grp * T
    return W


def assemble_whitener(
    W_blocks: torch.Tensor, *, ridge: float = 1e-3
) -> tuple[torch.Tensor, torch.Tensor]:
    """(h_kv, d, d) fp64 blocks -> dense (C, C) fp64 (W^{1/2}, W^{-1/2}).

    Per-block symmetric eigendecomposition; eigenvalues floored at
    ridge·λ_max(block) so near-null query directions can't explode W^{-1/2}.
    Blocks land on the diagonal in to_matrix's head-major channel order.
    """
    h_kv, d, _ = W_blocks.shape
    C = h_kv * d
    Wh = torch.zeros(C, C, dtype=torch.float64)
    Wh_inv = torch.zeros(C, C, dtype=torch.float64)
    for j in range(h_kv):
        Wj = 0.5 * (W_blocks[j] + W_blocks[j].mT)
        lam, E = torch.linalg.eigh(Wj)
        lam = lam.clamp_min(ridge * lam.max().clamp_min(1e-30))
        sl = slice(j * d, (j + 1) * d)
        Wh[sl, sl] = E @ torch.diag(lam.sqrt()) @ E.mT
        Wh_inv[sl, sl] = E @ torch.diag(lam.rsqrt()) @ E.mT
    return Wh, Wh_inv


def identity_whitener(C: int) -> tuple[torch.Tensor, torch.Tensor]:
    """W = I: the unweighted-KLT ablation path (plain Σ_k eigenbasis)."""
    eye = torch.eye(C, dtype=torch.float64)
    return eye, eye.clone()


@dataclasses.dataclass
class SpectralPack:
    """Corpus-fit spectral codec for one (layer, side): basis + bit allocation.

    Y = M @ enc (encode); M_hat = Y_hat @ dec.mT (decode); enc @ dec.mT = I.
    enc = W^{1/2} E, dec = W^{-1/2} E where E is the eigenbasis of
    W^{1/2} Σ_k W^{1/2} (descending eigenvalues lam). bits waterfills lam.
    Model-level accounting: the pack ships with the model (zero per-sequence
    bits); skeptic-mode per-sequence charge is skeptic_charge(C, S, tiers).
    """

    enc: torch.Tensor  # (C, C) fp32
    dec: torch.Tensor  # (C, C) fp32
    lam: torch.Tensor  # (C,) fp32, descending
    bits: torch.Tensor  # (C,) int64, members of tiers
    group: int
    tiers: tuple[int, ...]
    budget: float
    # RUNTIME-ONLY (never persisted by save_pack_file/load_packs -- the pack
    # FILE format is unchanged; this is set by load_packs_for_spec at
    # materialization only): the tier threshold this pack's `dec` was
    # ACTUALLY int8-roundtripped at, or None if `dec` is still fp-stored.
    # Ground truth for accounting -- see `cache_bits_per_entry`, which must
    # read this rather than re-derive a threshold from the (possibly already
    # roundtripped) pack, since int8_decoder_roundtrip is near-idempotent:
    # re-running the certificate on an already-int8-stored decoder sees
    # ddec ~= 0 and can silently pass a HIGHER tier than the pristine
    # certificate would have allowed (K4 estimation-levers Task 3 fix wave --
    # this was a real bug in the "int8_tl" streaming path: recomputing
    # per_layer_tier_thresholds from the post-roundtrip packs drifted from
    # the map materialization actually applied, under-charging bpe).
    dec_tier: int | None = None

    @property
    def c_used(self) -> int:
        """Number of directions carrying nonzero bits — the decoder columns
        actually read at decode (see skeptic-v2 / payload-v2 above)."""
        return int((self.bits != 0).sum())

    def c_int8(self, tier_threshold: int) -> int:
        """Number of USED decoder columns int8-eligible at `tier_threshold`:
        `count(0 < bits <= tier_threshold)` (K4 local-levers Task 1). At the
        top of the standard tier grid (8) this equals `c_used` (blanket); at
        or below the smallest used tier it is 0. The single home for the
        int8-column count `mixed_dec_charge`/`int8_decoder_certificate_tiered`
        /the streaming charge all price against — see `mixed_dec_charge`."""
        return int(((self.bits > 0) & (self.bits <= tier_threshold)).sum())


@dataclasses.dataclass
class SpectralBasis:
    """Budget-independent part of a spectral fit: basis + spectrum.

    `lam64` retains the fp64 eigenvalues (the exact tensor
    `allocate_bits_from_variance` is called with at pack time); `lam` is the
    fp32-rounded copy stored on the resulting SpectralPack.
    """

    enc: torch.Tensor  # (C, C) fp32
    dec: torch.Tensor  # (C, C) fp32
    lam: torch.Tensor  # (C,) fp32, descending
    lam64: torch.Tensor  # (C,) fp64, descending — private, allocation input


def fit_spectral_basis(
    M_fit: torch.Tensor,
    Wh: torch.Tensor,
    Wh_inv: torch.Tensor,
) -> SpectralBasis:
    """Fit the budget-independent basis + spectrum on M_fit. fp64 internally."""
    Sigma = key_second_moment(M_fit)
    T = Wh @ Sigma @ Wh
    lam, E = torch.linalg.eigh(0.5 * (T + T.mT))
    lam = lam.flip(0).clamp_min(0.0)  # descending
    E = E.flip(1)
    return SpectralBasis(
        enc=(Wh @ E).float(),
        dec=(Wh_inv @ E).float(),
        lam=lam.float(),
        lam64=lam,
    )


def _wishart_logdet_bias(n: int, C: int) -> float:
    """Bartlett/Wishart closed-form bias of E[log det(S)] relative to
    log det(Σ) for a sample covariance S = X^T X / n from n iid Gaussian
    rows in C dims (n >= C required — the standard non-degenerate Wishart
    regime): per-dim bias term

        b(n, C) = (1/C) * sum_{i=1..C} [psi((n-i+1)/2) + log(2/n)]

    via `torch.special.digamma` (no scipy dependency). E[gm(S)] ~
    gm(Sigma)*exp(b(n,C)) to first order (see `jensen_gap_report`'s n_rows
    docstring for the exact/first-order scope caveat).
    """
    assert n >= C, f"Wishart log-det bias requires n >= C; got n={n}, C={C}"
    idx = torch.arange(1, C + 1, dtype=torch.float64)
    psi = torch.special.digamma((n - idx + 1) / 2.0)
    return float((psi + math.log(2.0 / n)).mean())


def jensen_gap_report(
    moments: Sequence[torch.Tensor], *, n_rows: Sequence[int] | None = None
) -> dict:
    """Determinant-Jensen Gate-A anchor (math review 2026-07-24 #6, spec Part
    2): the population functional behind Gate A's measured corpus-basis
    retention ceiling.

    `moments`: a sequence of C×C PSD matrices on a FIXED SHARED FRAME (any fp
    dtype in, fp64 internally) — e.g. per-sequence whitened key moments
    `T_s = Wh @ Σ_s @ Wh` sharing one whitener `Wh` across the population, so
    the theorem's "W cancels" argument applies verbatim. This function is
    otherwise budget-free and knows nothing about RoPE/whitening/allocation —
    it only consumes matrices.

    Per moment: eigvalsh (fp64), PSD-guard the smallest eigenvalue (allows a
    small negative slack for fp round-off, asserts otherwise), clamp at a
    tiny positive floor (counted in `n_clamped`), then
    `gm(M) = exp(mean(log evals)) = det(M)^{1/C}` — the overflow-safe form of
    the geometric mean of the spectrum (never forms det(M) directly, which
    would over/underflow at C ~ hundreds).

    `pooled = mean_s(moments)` (the pooled-fit second moment Σ̄ under the
    theorem's setup) and:

        gm_pool      = gm(pooled)
        mean_gm_seq  = mean_s(gm(moments[s]))
        r_pred       = mean_gm_seq / gm_pool
        log_gap      = log(gm_pool) - mean_s(log(gm(moments[s])))

    By Minkowski's determinant inequality (det^{1/C} concave on PSD), Jensen
    gives `r_pred <= 1`. `log_gap` is the SAME comparison in log-space but is
    NOT `-log(r_pred)` in general (Jensen's gap is different when you take
    the log before vs after averaging the gm(Σ_s) across sequences) — both
    are reported deliberately, do not conflate them or "derive" one from the
    other downstream.

    `n_rows` (Wishart log-det debiasing, follow-up to the original anchor):
    None (default) reproduces the exact prior output bit-for-bit — no new
    keys added. When given (one int per moment: the token-row count that
    sample moment was ESTIMATED from — e.g. the sequence length S the
    per-cache Σ_s came from), a raw sample gm(Σ_s) is a BIASED estimate of
    the true population gm at finite n/C — `gm(Σ̄)` (pooled over many more
    effective rows) is far less biased than any single `gm(Σ_s)`, so `r_pred`
    itself is contaminated by an asymmetric small-sample bias, not just the
    "real" Jensen gap. `_wishart_logdet_bias(n, C)` gives the closed-form
    Bartlett/digamma correction to `E[log det(S)]`; this function applies it
    per moment (`b_s = _wishart_logdet_bias(n_rows[s], C)`) and once to the
    pool (`b_pool = _wishart_logdet_bias(sum(n_rows), C)`, since the pooled
    moment's effective sample size is the SUM of the per-cache row counts —
    consistent with `mean_s(moments)` being algebraically the same object as
    the second moment of the row-concatenated matrix when every cache
    contributes equally, per `per_cache_weighted_moments`' pooling
    convention), and adds three keys:

        bias_factor_seq   = exp(mean_s(b_s))
        bias_factor_pool  = exp(b_pool)
        r_pred_debiased   = mean_s(gm_s * exp(-b_s)) / (gm_pool * exp(-b_pool))

    HONEST SCOPE: `_wishart_logdet_bias` is EXACT only for iid Gaussian rows
    (a real Wishart sample covariance) and requires `n_rows[s] >= C` for
    every moment and `sum(n_rows) >= C` for the pool. Token rows in this
    codebase's caches are autocorrelated (adjacent positions are highly
    correlated), which makes the EFFECTIVE sample size smaller than the raw
    row count — so this correction is FIRST-ORDER and systematically
    UNDER-CORRECTS (the true bias at the smaller effective n is larger than
    what `n_rows[s]` predicts). `r_pred_debiased` is a bias-REDUCED estimate
    of the population ratio, not an exact one — report it as such, never as
    "the corrected R_pred."
    """
    assert len(moments) > 0, "moments must be non-empty"
    if n_rows is not None:
        assert len(n_rows) == len(moments), (
            f"n_rows length {len(n_rows)} != moments length {len(moments)}"
        )
    tiny = 1e-300
    C = None
    gms: list[float] = []
    log_gms: list[float] = []
    n_clamped = 0
    pooled_sum: torch.Tensor | None = None
    for i, M in enumerate(moments):
        M64 = M.double()
        assert M64.dim() == 2 and M64.shape[0] == M64.shape[1], (
            f"moments[{i}] must be square 2-D; got {tuple(M.shape)}"
        )
        if C is None:
            C = M64.shape[0]
        else:
            assert M64.shape[0] == C, (
                f"moments[{i}] has size {M64.shape[0]} != moments[0]'s {C} — "
                "all moments must share one frame"
            )
        M64 = 0.5 * (M64 + M64.mT)
        evals = torch.linalg.eigvalsh(M64)
        evals_max = float(evals.max())
        assert float(evals.min()) > -1e-10 * max(evals_max, tiny), (
            f"moments[{i}] is not PSD (within tolerance): min eigenvalue "
            f"{float(evals.min())!r}, max {evals_max!r}"
        )
        n_clamped += int((evals < tiny).sum())
        evals = evals.clamp_min(tiny)
        log_gm = float(evals.log().mean())
        gms.append(math.exp(log_gm))
        log_gms.append(log_gm)
        pooled_sum = M64 if pooled_sum is None else pooled_sum + M64

    pooled = pooled_sum / len(moments)
    pooled_evals = torch.linalg.eigvalsh(0.5 * (pooled + pooled.mT))
    pooled_evals_max = float(pooled_evals.max())
    assert float(pooled_evals.min()) > -1e-10 * max(pooled_evals_max, tiny), (
        f"pooled moment is not PSD (within tolerance): min eigenvalue "
        f"{float(pooled_evals.min())!r}, max {pooled_evals_max!r}"
    )
    n_clamped += int((pooled_evals < tiny).sum())
    pooled_evals = pooled_evals.clamp_min(tiny)
    log_gm_pool = float(pooled_evals.log().mean())
    gm_pool = math.exp(log_gm_pool)

    mean_gm_seq = float(sum(gms) / len(gms))
    mean_log_gm_seq = float(sum(log_gms) / len(log_gms))

    report = dict(
        gm_pool=gm_pool,
        mean_gm_seq=mean_gm_seq,
        r_pred=mean_gm_seq / gm_pool,
        log_gap=log_gm_pool - mean_log_gm_seq,
        n_seq=len(moments),
        n_clamped=n_clamped,
    )

    if n_rows is not None:
        b_seq_list = [_wishart_logdet_bias(int(n), C) for n in n_rows]
        b_pool = _wishart_logdet_bias(int(sum(n_rows)), C)
        mean_b_seq = sum(b_seq_list) / len(b_seq_list)
        debiased_gms = [gm * math.exp(-b) for gm, b in zip(gms, b_seq_list)]
        mean_debiased_gm_seq = sum(debiased_gms) / len(debiased_gms)
        debiased_gm_pool = gm_pool * math.exp(-b_pool)
        report["bias_factor_seq"] = math.exp(mean_b_seq)
        report["bias_factor_pool"] = math.exp(b_pool)
        report["r_pred_debiased"] = mean_debiased_gm_seq / debiased_gm_pool

    return report


def pack_from_basis(
    basis: SpectralBasis,
    budget: float,
    *,
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8),
    group: int = 64,
    lam_alloc: torch.Tensor | None = None,
    s_ref: int | None = None,
    g_table: tuple[float, ...] | None = None,
) -> SpectralPack:
    """Allocate bits for one budget against an already-fit SpectralBasis.

    `lam_alloc` (fp64, (C,), index-aligned with enc's columns) substitutes the
    waterfill input — the K4 corpus-transfer hybrid path ("basis transfers,
    allocation adapts": basis from corpus A, per-direction variances measured
    on corpus B via basis_alloc_moment). Default None reproduces the prior
    behavior bit-exactly (allocates on basis.lam64).

    s_ref (charge-aware allocation, math review 2026-07-24 #2): when set, the
    allocator prices every used direction's TRUE storage cost under the
    reported accounting -- per-direction fixed charge
    s = 16/group + 16*C/s_ref (fp16 decoder column + group-scale share) --
    via the exact Lagrangian enumeration
    b_i = argmin_{b in T} lam_i*g(b) + kappa_L*(b + s*1[b>0]).
    budget then bounds the mean per-direction TOTAL charge
    (1/C)*sum_i (b_i + s*1[b_i>0]) = payload-v2 bpe + 16*c_used/s_ref
    = skeptic-v2 bpe@s_ref - tier_bits(tiers, s_ref). This is an ALLOCATION
    change only: the bpe accounting expressions stay frozen; c_used simply
    becomes smaller. g_table (finding #4) swaps 4^{-b} for measured per-tier
    ratios (Lagrangian selection, fixed charge 0 unless s_ref is also set).
    Defaults (None, None) reproduce the prior behavior bit-exactly.
    """
    assert 1 not in tiers, "symmetric RTN is undefined at 1 bit (qmax=0)"
    alloc_input = basis.lam64 if lam_alloc is None else lam_alloc
    assert alloc_input.shape == basis.lam64.shape, (
        f"lam_alloc shape {tuple(alloc_input.shape)} != {tuple(basis.lam64.shape)}"
    )
    if s_ref is not None or g_table is not None:
        fixed = 0.0
        if s_ref is not None:
            assert s_ref > 0, f"s_ref must be positive; got {s_ref}"
            C = basis.enc.shape[0]
            # math review #2: s = 16/group + dec_bits*C/S_ref, dec_bits=16
            # (fp16 decoder; the A-gate never invokes the int8 lever).
            fixed = scale_bits(group) + 16.0 * C / float(s_ref)
        bits = allocate_bits_from_variance(
            alloc_input,
            budget,
            tiers,
            selection="lagrange",
            g_table=g_table,
            fixed_charge=fixed,
        )
    else:
        bits = allocate_bits_from_variance(alloc_input, budget, tiers)
    return SpectralPack(
        enc=basis.enc,
        dec=basis.dec,
        lam=basis.lam,
        bits=bits,
        group=group,
        tiers=tuple(tiers),
        budget=float(budget),
    )


def basis_alloc_moment(basis: SpectralBasis, M_alloc: torch.Tensor) -> torch.Tensor:
    """Per-direction second moments of M_alloc's rows in `basis`'s coordinate
    system: diag(encᵀ Σ_alloc enc) = E[(M_alloc @ enc)_i²], fp64 (C,), clamped
    ≥ 0. The waterfill input for the H3 hybrid (pack_from_basis lam_alloc)."""
    Sigma = key_second_moment(M_alloc)
    enc64 = basis.enc.double()
    return torch.einsum("ci,cd,di->i", enc64, Sigma, enc64).clamp_min(0.0)


def shrink_spectrum(
    lam64: torch.Tensor,
    *,
    n: int,
    method: str,
    rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Eigenvalue shrinkage for the waterfill ALLOCATION INPUT (K4
    estimation-levers Part 1, finding #7). Consumed via
    `pack_from_basis(lam_alloc=shrink_spectrum(...)[0])`; changes no basis,
    no stored lam, no accounting expression.

    Sample eigenvalues at deployment aspect ratios (gamma = C/n ~ 0.1-0.75)
    are spread vs the population spectrum (top biased up, bulk/tail biased
    down). Both estimators below are a monotone affine map of the SAME
    spectrum toward its own mean -- same eigenvectors, same basis:

        lambda'_i = (1 - rho) * lambda_hat_i + rho * mu_hat,  mu_hat = mean(lambda_hat)

    Returns (shrunk fp64 spectrum, same shape/order as lam64; rho in [0, 1]).

    method="lw": Ledoit-Wolf (2004) linear shrinkage. Requires `rows`, the
    (n, C) fp matrix the sample moment Sigma_hat = rows^T @ rows / n was
    built from (index-aligned columns with lam64's directions -- callers
    typically pass the eigenbasis-projected rows). Intensity:

        d^2   = ||Sigma_hat - mu_hat*I||_F^2 = sum_i (lambda_hat_i - mu_hat)^2
        bbar2 = min(d^2, (1/n^2) * sum_t ||x_t x_t^T - Sigma_hat||_F^2)
        rho   = bbar2 / d^2                                   (in [0, 1] by the min)

    The b-bar^2 sum is computed WITHOUT materializing per-row outer
    products, via the trace identity (verified against the naive
    double-loop form in tests/test_spectral.py):

        ||x_t x_t^T - Sigma_hat||_F^2 = ||x_t||^4 - 2*x_t^T Sigma_hat x_t + ||Sigma_hat||_F^2
        sum_t x_t^T Sigma_hat x_t     = tr(Sigma_hat * sum_t x_t x_t^T) = n * tr(Sigma_hat^2)
        => sum_t ||x_t x_t^T - Sigma_hat||_F^2 = sum_t ||x_t||^4 - n * tr(Sigma_hat^2)

    method="oas": Oracle-Approximating Shrinkage (Chen, Wiesel, Eldar &
    Hero 2010), closed form from (lam64, n) alone -- no rows needed:

        rho = min(1, [(1 - 2/C)*tr(Sigma_hat^2) + tr(Sigma_hat)^2]
                     / [(n + 1 - 2/C)*(tr(Sigma_hat^2) - tr(Sigma_hat)^2/C)])

    with tr(Sigma_hat) = sum(lambda_hat), tr(Sigma_hat^2) = sum(lambda_hat^2).
    The denominator is <= 0 exactly when Sigma_hat is already proportional
    to I (rank-1 spectrum spread == 0); guarded to rho = 1 (already at
    target) rather than dividing.

    Edge semantics: shrinkage LIFTS zero/clamped eigenvalues toward mu_hat
    -- intended, since the tail bias is downward (sample tail eigenvalues
    are the most under-estimated by the top/tail spreading effect).
    """
    assert method in ("lw", "oas"), f"method must be 'lw' or 'oas'; got {method!r}"
    assert lam64.dtype == torch.float64, f"lam64 must be fp64; got {lam64.dtype}"
    C = lam64.shape[0]
    mu = lam64.mean()
    d2 = float(((lam64 - mu) ** 2).sum())

    if method == "lw":
        assert rows is not None, "method='lw' requires rows (the (n, C) fit matrix)"
        assert rows.shape == (n, C), f"rows shape {tuple(rows.shape)} != (n={n}, C={C})"
        rows64 = rows.double()
        Sigma = rows64.mT @ rows64 / n
        trace_sigma_sq = float(
            (Sigma * Sigma).sum()
        )  # tr(Sigma^2) == ||Sigma||_F^2 (symmetric)
        norm4_sum = float((rows64.pow(2).sum(dim=1) ** 2).sum())
        num_sum = norm4_sum - n * trace_sigma_sq  # sum_t ||x_t x_t^T - Sigma||_F^2
        bbar2 = min(d2, num_sum / (n**2))
        rho = 1.0 if d2 == 0.0 else bbar2 / d2
    else:
        trace_sigma = float(lam64.sum())
        trace_sigma_sq = float((lam64**2).sum())
        num = (1.0 - 2.0 / C) * trace_sigma_sq + trace_sigma**2
        den = (n + 1.0 - 2.0 / C) * (trace_sigma_sq - trace_sigma**2 / C)
        rho = 1.0 if den <= 0.0 else min(1.0, num / den)

    assert 0.0 <= rho <= 1.0, f"rho out of [0, 1]: {rho}"
    shrunk = (1.0 - rho) * lam64 + rho * mu
    assert (shrunk[lam64 > 0] > 0).all(), (
        "shrinkage must preserve positivity where input > 0"
    )
    return shrunk, float(rho)


def fit_spectral_pack(
    M_fit: torch.Tensor,
    Wh: torch.Tensor,
    Wh_inv: torch.Tensor,
    budget: float,
    *,
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8),
    group: int = 64,
    s_ref: int | None = None,
    g_table: tuple[float, ...] | None = None,
) -> SpectralPack:
    """Fit basis + allocation on M_fit (the calibration matrix). fp64 internally."""
    return pack_from_basis(
        fit_spectral_basis(M_fit, Wh, Wh_inv),
        budget,
        tiers=tiers,
        group=group,
        s_ref=s_ref,
        g_table=g_table,
    )


def spectral_payload_bpe(pack: SpectralPack) -> float:
    """Model-level payload bpe for a fitted pack (payload-v2, see module
    docstring): mean(bits) + scale_bits(group)·(c_used/C).

    payload-v1 was `mean(bits) + scale_bits(group)` — a full groupwise-scale
    charge regardless of how many directions actually carry bits. Directions
    with bits == 0 store no payload at all, so they need no scale either;
    payload-v1 over-counts those phantom scales. payload-v2 prorates the
    scale term by the used fraction c_used/C.
    """
    C = pack.bits.shape[0]
    return float(pack.bits.float().mean().item()) + scale_bits(pack.group) * (
        pack.c_used / C
    )


def spectral_payload_v1_bpe(pack: SpectralPack) -> float:
    """Model-level payload bpe, payload-v1 (see module docstring):
    mean(bits) + scale_bits(group) — the full groupwise-scale charge
    regardless of how many directions actually carry bits. This is the
    expression every parquet before 2026-07-23 measured; kept callable so
    `_fullc` companion columns can be joined against those old parquets.

    Identity: payload_v1(pack) == spectral_payload_bpe(pack) +
    scale_bits(group)·(1 − c_used/C).
    """
    return float(pack.bits.float().mean().item()) + scale_bits(pack.group)


def tier_columns(bits: torch.Tensor) -> dict[int, torch.Tensor]:
    """Ascending column indices per nonzero tier, `sorted(set(bits.tolist()))`
    exactly as `quantize_by_bits` iterates — byte-for-byte the same loop.

    This IS the sort-by-tier permutation: scattering into these columns at
    dequant is its inverse. `spectral_quantize_packed`/`spectral_dequant_packed`
    take this as an optional precomputed arg because the read path calls it
    per committed page per decode step per layer; the layer computes it once.
    """
    cols_by_tier: dict[int, torch.Tensor] = {}
    for b in sorted(set(int(x) for x in bits.tolist())):
        if b == 0:
            continue
        cols_by_tier[b] = (bits == b).nonzero(as_tuple=True)[0]
    return cols_by_tier


_SUBBYTE_TIERS = frozenset({2, 4})  # offset-binary via pack_codes (exact width)
_NIBBLE_TIERS = frozenset({3})  # signed nibbles via pack_signed_codes (4-bit container)


def _pack_tier_codes(Q_int: torch.Tensor, b: int) -> torch.Tensor:
    """Container policy for one tier's RTN integer codes (reuse-only, no new
    bit-twiddling): tiers 2/4 offset to unsigned then `pack_codes` (exact
    width); tier 3 signed nibbles via `pack_signed_codes`; tiers 5/6/8 stored
    as the int8 container `rtn_quantize_packed` already produced."""
    if b in _SUBBYTE_TIERS:
        return pack_codes(Q_int.to(torch.int16) + 2 ** (b - 1), b)
    if b in _NIBBLE_TIERS:
        return pack_signed_codes(Q_int, b)
    return Q_int  # int8 container (tiers 5, 6, 8)


def _unpack_tier_codes(t: torch.Tensor, b: int, S: int) -> torch.Tensor:
    """Exact inverse of `_pack_tier_codes`."""
    if b in _SUBBYTE_TIERS:
        return (unpack_codes(t, b, S) - 2 ** (b - 1)).to(torch.int8)
    if b in _NIBBLE_TIERS:
        return unpack_signed_codes(t, b, S)
    return t


def spectral_quantize_packed(
    M: torch.Tensor,
    pack: SpectralPack,
    *,
    mse_scale: bool = True,
    cols_by_tier: dict[int, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], float]:
    """Packed-codec form of `spectral_quantize`: per-tier code containers +
    fp32 group scales instead of a dequantized M_hat. Flat-key dict
    `{f"t{b}_codes": container, f"t{b}_scale": fp32 (n_b, S//group, 1)}`.

    Same enc matmul, same per-tier `rtn_quantize_packed` call, same tier
    order as `quantize_by_bits` — `spectral_dequant_packed` is its exact
    inverse (bitwise-equal to `spectral_quantize`'s M_hat by construction).
    """
    S, C = M.shape
    assert pack.enc.shape == (C, C), f"pack C mismatch: {pack.enc.shape} vs C={C}"
    assert S % pack.group == 0, f"S={S} not divisible by group={pack.group}"
    cols_by_tier = cols_by_tier if cols_by_tier is not None else tier_columns(pack.bits)
    assert cols_by_tier, "pack allocates zero bits everywhere; nothing to store"
    Y = M @ pack.enc.to(M.dtype)
    packed: dict[str, torch.Tensor] = {}
    for b, cols in cols_by_tier.items():
        Q_int, scale = rtn_quantize_packed(
            Y[:, cols].mT, b, pack.group, mse_scale=mse_scale
        )
        packed[f"t{b}_codes"] = _pack_tier_codes(Q_int, b)
        packed[f"t{b}_scale"] = scale
    return packed, spectral_payload_bpe(pack)


def spectral_dequant_packed(
    packed: dict[str, torch.Tensor],
    pack: SpectralPack,
    *,
    cols_by_tier: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Inverse of `spectral_quantize_packed`: fp32 `(S, C)` M_hat, bitwise-equal
    to `spectral_quantize(M, pack)[0]`."""
    cols_by_tier = cols_by_tier if cols_by_tier is not None else tier_columns(pack.bits)
    first_b = next(iter(cols_by_tier))
    S = packed[f"t{first_b}_scale"].shape[1] * pack.group  # scale is (n_b, S//group, 1)
    C = pack.enc.shape[0]
    Y_hat = torch.zeros(S, C, dtype=pack.dec.dtype, device=pack.dec.device)
    for b, cols in cols_by_tier.items():
        Q_int = _unpack_tier_codes(packed[f"t{b}_codes"], b, S)
        Y_hat[:, cols] = rtn_dequantize_packed(
            Q_int, packed[f"t{b}_scale"], pack.group
        ).mT
    return Y_hat @ pack.dec.mT


def spectral_quantize(
    M: torch.Tensor, pack: SpectralPack, *, mse_scale: bool = True
) -> tuple[torch.Tensor, float]:
    """Quantize (S, C) M with a fitted pack. Returns (M_hat, bpe_model).

    bpe_model = spectral_payload_bpe(pack) (payload-v2 model-level accounting —
    the pack itself ships with the model). Add skeptic_charge(C, S, tiers) for
    the per-sequence-charged view.

    Re-expressed as the composition `spectral_dequant_packed ∘
    spectral_quantize_packed` — bitwise-neutral by construction (`rtn_quantize`
    is literally `rtn_dequantize_packed(rtn_quantize_packed(...))`, so the
    per-tier compose is the same enc/dec matmuls, same order, same
    device/dtype as the direct `quantize_by_bits` path it replaces).
    """
    S, C = M.shape
    assert pack.enc.shape == (C, C), f"pack C mismatch: {pack.enc.shape} vs C={C}"
    assert S % pack.group == 0, f"S={S} not divisible by group={pack.group}"
    packed, bpe = spectral_quantize_packed(M, pack, mse_scale=mse_scale)
    M_hat = spectral_dequant_packed(packed, pack)
    return M_hat, bpe


def skeptic_charge(
    C: int,
    S: int,
    tiers: tuple[int, ...],
    *,
    c_used: float | None = None,
    dec_bits: float = 16.0,
) -> float:
    """Per-sequence charge when the pack is NOT granted model-level status.

    skeptic-v1 (defaults, c_used=None -> C, dec_bits=16.0; the expression every
    parquet before 2026-07-23 measured, reproduced bit-exactly forever):
        16·C/S + tier_bits(tiers, S)
    charges a full C×C fp16 decoder matrix.

    skeptic-v2 (c_used given, dec_bits=16.0) charges only the c_used decoder
    columns actually read at decode (see test_dropped_decoder_columns_never_read
    for the license — mutating a dropped column can't change the
    reconstruction):
        16·c_used/S + tier_bits(tiers, S)

    skeptic-v2-int8 (c_used given, dec_bits=8.0) additionally stores the
    decoder int8 (8 bits/entry) plus one fp16 per-column scale amortized over
    S rows:
        8·c_used/S + 16·c_used/(S·C) + tier_bits(tiers, S)
    """
    assert dec_bits in (8.0, 16.0), f"dec_bits must be 8.0 or 16.0; got {dec_bits}"
    if c_used is None:
        c_used = C
    assert 0 < c_used <= C, f"c_used must be in (0, C]; got {c_used}, C={C}"
    int8_scale_term = 16.0 * c_used / (S * C) if dec_bits < 16.0 else 0.0
    return dec_bits * c_used / S + int8_scale_term + tier_bits(tiers, S)


def int8_decoder_roundtrip(
    dec: torch.Tensor, bits_pc: torch.Tensor, *, tier_threshold: int | None = None
) -> torch.Tensor:
    """Simulate int8 storage of a decoder matrix's USED columns (Lever 2).

    Per-column symmetric absmax int8: scale = absmax/127, cast to fp16 (the
    shipped scale dtype) then back to fp32 BEFORE dequant so the roundtrip
    reflects the exact precision an int8-stored decoder would reconstruct at.
    Unused columns (bits_pc == 0 -- never read at decode, see
    test_dropped_decoder_columns_never_read) are returned untouched.
    Deterministic; fp32 in, fp32 out. Refits nothing -- this operates on an
    already-fitted pack's dec tensor.

    `tier_threshold` (K4 local-levers Task 1, the tier-gated int8 promotion):
    None (default) gates only on bits_pc != 0 -- today's blanket behavior,
    reproduced bit-exactly. When given, only columns with
    `0 < bits_pc <= tier_threshold` are roundtripped; used columns above the
    threshold are left fp32-as-loaded (they ship fp16 in the mixed accounting
    -- see mixed_dec_charge). The standard tier grid tops out at 8, so
    `tier_threshold=8` reproduces the blanket roundtrip exactly (every used
    column is <= 8).
    """
    gate = bits_pc != 0
    if tier_threshold is not None:
        gate = gate & (bits_pc <= tier_threshold)
    dec_rt = dec.clone()
    dec_gated = dec[:, gate]
    scale = (dec_gated.abs().amax(dim=0) / 127.0).clamp_min(1e-12).half().float()
    codes = (dec_gated / scale).round().clamp(-127, 127)
    dec_rt[:, gate] = codes * scale
    return dec_rt


def _int8_cert_terms(pack: SpectralPack, ddec: torch.Tensor) -> dict[str, float]:
    """The certificate's shared closed form on a (possibly column-gated) fp64
    decoder perturbation ddec — see `int8_decoder_certificate` for the
    derivation and honest limits."""
    proj = pack.enc.double().mT @ ddec  # (C, C): column i = enc^T ddec[:, i]
    lam = pack.lam.double().clamp_min(0.0)
    added = float((lam * (proj**2).sum(dim=0)).sum())
    payload = float((lam * torch.pow(4.0, -pack.bits.double())).sum())
    return dict(
        added=added,
        payload=payload,
        noise_to_signal=added / max(payload, 1e-300),
        implied_rel_degradation=added / max(payload + added, 1e-300),
    )


def int8_decoder_certificate(pack: SpectralPack) -> dict[str, float]:
    """Exact offline certificate for the int8-decoder distortion (math review
    2026-07-24 #9): the roundtrip perturbation ddec = dec_int8 - dec is
    deterministic per pack, so the added weighted reconstruction distortion is
    a computable NUMBER, not a bound and not a VM measurement.

    Closed form (fp64; derivation): the added K-space error on a row with
    code vector y_hat is ddec @ y_hat; its weighted norm is
    ||W^{1/2} ddec y_hat||^2 = ||enc^T ddec y_hat||^2 (W = enc enc^T exactly,
    since enc = W^{1/2} E with E orthogonal). Taking the expectation with the
    code second moment diag(lam) — exact on the fit corpus, where
    enc^T Sigma_fit enc = diag(lam) by the eigendecomposition:

        added   = sum_i lam_i * ||enc^T ddec[:, i]||^2
        payload = sum_i lam_i * 4^{-bits_i}     (dropped dirs: 4^0 = 1)

    noise_to_signal = added/payload; implied_rel_degradation =
    added/(payload + added) — the same axis as the pre-registered VM gate
    rel_degradation_int8 < 5% (win is inverse distortion, so
    1 - win_int8/win_fp16 = added/(payload + added) under matched bpe).

    Honest limits (what this does NOT capture): E[y_hat y_hat^T] is modeled
    by diag(lam) — the payload-error shift of the code moment and the
    payload x decoder cross-term (both O(g(b)) relative on a ~7e-5 base) are
    not represented; query-distribution interaction beyond the modeled second
    moment is not represented; task-level effects are NOT certified (they
    remain the VM half of the ledger). pack.lam is the fp32-stored spectrum
    (1e-7 relative rounding — three orders below the certificate's margin).
    """
    ddec = int8_decoder_roundtrip(pack.dec, pack.bits).double() - pack.dec.double()
    return _int8_cert_terms(pack, ddec)


def int8_decoder_certificate_tiered(pack: SpectralPack, tier_threshold: int) -> dict:
    """Tier-gated rescue of `int8_decoder_certificate` (K4 math-actions 6b):
    "is int8 dead or just misapplied?" The blanket certificate int8-stores
    EVERY used decoder column; this asks what happens if only the low-tier
    columns (`0 < bits_i <= tier_threshold`) are int8-stored and the rest
    stay fp16.

    Re-expressed through `int8_decoder_roundtrip(..., tier_threshold=)`
    (K4 local-levers Task 1): columns with `bits_i > tier_threshold` are left
    fp32-as-loaded by the gated roundtrip itself, so `ddec` is zero there
    without any manual post-processing — same numbers as the prior
    zero-after-the-fact formulation (columns above T contribute nothing to
    `enc^T ddec` either way). `added`, `payload`, `noise_to_signal`,
    `implied_rel_degradation` are exactly `int8_decoder_certificate`'s
    expressions restricted to the gated ddec — at
    `tier_threshold >= max(pack.bits)` this reproduces the blanket
    certificate bit-for-bit (nothing gets zeroed); at
    `tier_threshold < min(used tier)` `added == 0.0` exactly (everything
    gets zeroed).

    Mixed-decoder accounting (`c_used`/`c_int8` below) is priced separately by
    `mixed_dec_charge`/`effective_dec_bits` — see those docstrings for the
    skeptic-v2 arithmetic generalized to a per-column int8/fp16 mix.
    """
    c_used = pack.c_used
    c_int8 = pack.c_int8(tier_threshold)

    dec_gated = int8_decoder_roundtrip(
        pack.dec, pack.bits, tier_threshold=tier_threshold
    )
    ddec = dec_gated.double() - pack.dec.double()

    return dict(
        tier_threshold=int(tier_threshold),
        c_used=c_used,
        c_int8=c_int8,
        frac_int8=c_int8 / c_used if c_used > 0 else 0.0,
        **_int8_cert_terms(pack, ddec),
    )


def mixed_dec_charge(
    C: int, S: int, tiers: tuple[int, ...], *, c_used: float, c_int8: float
) -> float:
    """Per-sequence decoder+tier-map charge (skeptic-v2 arithmetic) when only
    `c_int8` of the `c_used` used columns are int8-stored and the remaining
    `c_used - c_int8` stay fp16 -- the same skeptic-v2-int8 per-column terms
    (`8·c/S` entry cost + `16·c/(S·C)` amortized fp16 scale), split across
    the int8 and fp16 subsets and summed:

        dec_charge(S) = [8·c_int8/S + 16·c_int8/(S·C)]  (int8 cols + scales)
                      + 16·(c_used - c_int8)/S            (fp16 cols, no scale)
                      + tier_bits(tiers, S)

    Reduces to `skeptic_charge(dec_bits=16.0)` at `c_int8=0` and to
    `skeptic_charge(dec_bits=8.0)` at `c_int8=c_used`
    (checked by `test_mixed_dec_charge_endpoints_match_skeptic_charge`).

    `c_used`/`c_int8` accept float (K4 local-levers Task 1): the streaming
    call site passes across-layer MEANS of per-layer integer counts, and the
    expression is linear in both, so the mean-of-charges equals the
    charge-of-means exactly -- the same license `cache_bits_per_entry`
    already relies on for `mean_c_used`.
    """
    assert 0 <= c_int8 <= c_used <= C, (
        f"need 0<=c_int8<=c_used<=C; got {c_int8},{c_used},{C}"
    )
    int8_term = 8.0 * c_int8 / S + 16.0 * c_int8 / (S * C)
    fp16_term = 16.0 * (c_used - c_int8) / S
    return int8_term + fp16_term + tier_bits(tiers, S)


def effective_dec_bits(C: int, c_used: int, c_int8: int) -> float:
    """The c_used-weighted mix bit rate implied by `mixed_dec_charge`'s
    decoder-column terms alone (excludes tier_bits, which is S-amortized
    metadata, not a per-column decoder rate): 16.0 bits/column for the fp16
    columns, `8 + 16/C` bits/column for the int8 ones (entry + amortized
    per-column scale, matching skeptic_charge's `dec_bits=8.0` convention of
    folding the scale into an equivalent per-entry rate). Independent of S.
    """
    if c_used == 0:
        return 16.0
    int8_rate = 8.0 + 16.0 / C
    return (c_int8 * int8_rate + (c_used - c_int8) * 16.0) / c_used


def save_pack_file(
    path: str | Path,
    bases: dict[int, SpectralBasis],
    budgets: tuple[float, ...],
    *,
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8),
    group: int = 64,
    meta: dict | None = None,
    layer_budgets: dict[float, dict[int, float]] | None = None,
) -> dict[float, dict[int, SpectralPack]]:
    """Save per-layer bases + per-budget bit allocations to one safetensors file.

    Bits are computed via `pack_from_basis` at save time (the fp64 allocation
    input `basis.lam64` is only needed here) so `load_packs` never re-runs the
    allocator — reloaded packs are allocation-frozen artifacts, not re-fit.
    A JSON sidecar at `<path>.json` carries tiers/group/budgets/meta since
    safetensors only stores tensors. Returns the packs it computed, keyed
    [budget][layer], so callers can read stats off the exact saved allocation
    without re-running the waterfill.

    `layer_budgets`, when given, maps each entry of `budgets` (then a target
    MEAN label) to {layer: allocated budget}: layer i's bits are waterfilled
    at layer_budgets[budget][i] but stored under the label `bits_b{budget:g}`,
    so `load_packs` / streaming / recipes select an allocated pack with zero
    changes. The allocation is written into the sidecar itself under the
    reserved key "layer_budgets" (never trusted to the caller's `meta`).
    """
    tensors: dict[str, torch.Tensor] = {}
    packs: dict[float, dict[int, SpectralPack]] = {b: {} for b in budgets}
    for i, basis in bases.items():
        tensors[f"layer{i}.enc"] = basis.enc
        tensors[f"layer{i}.dec"] = basis.dec
        tensors[f"layer{i}.lam"] = basis.lam
        for budget in budgets:
            b_i = budget if layer_budgets is None else layer_budgets[budget][i]
            pack = pack_from_basis(basis, b_i, tiers=tiers, group=group)
            tensors[f"layer{i}.bits_b{budget:g}"] = pack.bits
            packs[budget][i] = pack
    save_file(tensors, str(path))

    sidecar: dict = {"tiers": list(tiers), "group": group, "budgets": list(budgets)}
    if layer_budgets is not None:
        sidecar["layer_budgets"] = {
            f"{budget:g}": {str(layer): b for layer, b in lb.items()}
            for budget, lb in layer_budgets.items()
        }
    if meta:
        collisions = set(meta) & (set(sidecar) | {"layer_budgets"})
        assert not collisions, (
            f"meta keys collide with sidecar keys: {sorted(collisions)}"
        )
        sidecar.update(meta)
    Path(str(path) + ".json").write_text(json.dumps(sidecar))
    return packs


def load_packs(path: str | Path, budget: float) -> dict[int, SpectralPack]:
    """Reconstruct per-layer SpectralPacks for one budget from a saved file.

    Raises KeyError (with the available budgets) if `budget` isn't in the file.
    """
    sidecar = json.loads(Path(str(path) + ".json").read_text())
    budgets = sidecar["budgets"]
    if budget not in budgets:
        raise KeyError(f"budget {budget} not in pack file; available: {budgets}")
    tiers = tuple(sidecar["tiers"])
    group = sidecar["group"]

    tensors = load_file(str(path))
    layers = sorted(
        {int(k.split(".")[0].removeprefix("layer")) for k in tensors if "." in k}
    )
    packs: dict[int, SpectralPack] = {}
    for i in layers:
        packs[i] = SpectralPack(
            enc=tensors[f"layer{i}.enc"],
            dec=tensors[f"layer{i}.dec"],
            lam=tensors[f"layer{i}.lam"],
            bits=tensors[f"layer{i}.bits_b{budget:g}"],
            group=group,
            tiers=tiers,
            budget=float(budget),
        )
    return packs


TIER_THRESHOLD_GRID: tuple[int, ...] = (2, 3, 4, 5, 6, 8)

# Sentinel dec_quant_threshold() returns for "int8_tl": per-layer,
# certificate-derived thresholds (see per_layer_tier_thresholds) rather than
# one shared int -- consumers (load_packs_for_spec, cache_bits_per_entry)
# switch on this value to take the per-layer path.
PER_LAYER_TIER_SENTINEL = -1


def per_layer_tier_thresholds(
    packs: dict[int, "SpectralPack"], *, bar: float = 0.05
) -> dict[int, int]:
    """Certificate-derived per-layer int8 tier threshold (K4 estimation-levers
    Task 3): for each layer, the largest T in the standard tier grid
    `TIER_THRESHOLD_GRID` (2, 3, 4, 5, 6, 8) whose
    `int8_decoder_certificate_tiered(pack, T)["implied_rel_degradation"]`
    stays at or below `bar`.

    The layer-uniform `int8_t5` threshold binds on the single worst layer at
    every T (that layer sets the ceiling for every other layer too); this
    computes the ceiling PER LAYER instead, so layers whose spectrum tolerates
    int8 further can use a higher T while the worst layer is left at whatever
    it can actually certify.

    Deterministic and pack-derived only -- no fit, no refit, no new metadata:
    callers recompute this from the already-loaded pack dict, once.

    Grid is scanned ascending; the returned T is the LAST one that still
    passes (monotonicity of `implied_rel_degradation` in T is validated
    empirically, not assumed -- see test_int8_certificate_tiered_endpoints_
    and_charge's ordering pin), so a layer failing even T=2 gets 0 (no int8
    for that layer -- fp32-as-loaded/fp16-shipped, same meaning as
    dec_quant_threshold's None).
    """
    thresholds: dict[int, int] = {}
    for layer_i, pack in packs.items():
        best = 0
        for t in TIER_THRESHOLD_GRID:
            cert = int8_decoder_certificate_tiered(pack, t)
            if cert["implied_rel_degradation"] <= bar:
                best = t
        thresholds[layer_i] = best
    return thresholds


def dec_quant_threshold(dec_quant: str) -> int | None:
    """Parse a `CacheCodecSpec.dec_quant` string to the tier threshold
    `int8_decoder_roundtrip` expects (K4 local-levers Task 1 -- the one
    parameter the whole dec_quant surface collapses to):

        "fp32"      -> None   (no int8 storage; fp32-as-loaded/fp16-shipped)
        "int8"      -> 8      (blanket -- every used column is <= 8, the top
                                of the standard tier grid, so this reproduces
                                today's blanket roundtrip exactly)
        "int8_t{T}" -> T       (2 <= T <= 8; tier-gated -- only columns with
                                0 < bits <= T are int8-stored)
        "int8_tl"   -> PER_LAYER_TIER_SENTINEL (-1; K4 estimation-levers
                                Task 3) -- per-layer certificate-derived
                                thresholds (per_layer_tier_thresholds), not
                                one shared int. Consumers switch on the
                                sentinel explicitly; it is never used as a
                                real tier_threshold value.

    Anything else asserts with a clear message.
    """
    if dec_quant == "fp32":
        return None
    if dec_quant == "int8":
        return 8
    if dec_quant == "int8_tl":
        return PER_LAYER_TIER_SENTINEL
    if dec_quant.startswith("int8_t"):
        t_str = dec_quant[len("int8_t") :]
        assert t_str.isdigit(), (
            f"dec_quant int8_t suffix must be an integer; got {dec_quant!r}"
        )
        t = int(t_str)
        assert 2 <= t <= 8, (
            f"dec_quant int8_t threshold must be in [2, 8]; got {t} from {dec_quant!r}"
        )
        return t
    raise AssertionError(
        f"dec_quant must be 'fp32', 'int8', 'int8_tl', or 'int8_t{{T}}' "
        f"(2<=T<=8); got {dec_quant!r}"
    )


def load_packs_for_spec(k_spec) -> dict[int, SpectralPack]:
    """Materialize the per-layer pack dict a KV cache should hold for `k_spec`.

    Pack materialization owns the dec_quant decision — this is the
    design-note altitude both StreamingQuantizedCache.__init__ and
    PackedStreamingCache.__init__ delegate to (pure code motion, zero
    numeric change: byte-identical blocks extracted verbatim from both).

    Non-spectral arms (arm != "spectral") return {} — nothing to load.
    Spectral arms load the fitted pack file (asserting pre_rope + pack_path
    are set) and, when `dec_quant_threshold(k_spec.dec_quant)` is not None,
    roundtrip every layer's decoder matrix through the gated int8 roundtrip
    ONCE here (Lever 2 — see int8_decoder_roundtrip) via dataclasses.replace,
    which keeps every other pack field (enc, lam, bits, tiers, ...)
    untouched. `dec_quant="int8"` parses to threshold 8 (blanket -- the same
    result as before this generalization, pinned by
    test_load_packs_for_spec_matches_load_packs_and_owns_dec_quant). Every
    roundtripped pack's `dec_tier` is set to the threshold ACTUALLY applied
    to it (the ground truth `cache_bits_per_entry` reads back — see
    `SpectralPack.dec_tier`'s docstring for why re-deriving a threshold from
    an already-roundtripped decoder is unsafe).

    `dec_quant="int8_tl"` (K4 estimation-levers Task 3): thr is the
    PER_LAYER_TIER_SENTINEL, not a real threshold. `per_layer_tier_thresholds`
    is computed ONCE here from the just-loaded PRISTINE pack dict
    (deterministic, pack-derived -- no extra state), then each layer is
    roundtripped at its OWN T_ℓ with `dec_tier` set to that same T_ℓ; a layer
    whose T_ℓ is 0 (even T=2 fails the certificate bar) is left untouched (no
    int8 for that layer, `dec_tier` stays None -- same meaning as thr=None).

    k_spec: a CacheCodecSpec (arm, pre_rope, pack_path, budget, dec_quant).
    """
    thr = dec_quant_threshold(k_spec.dec_quant)
    if k_spec.arm != "spectral":
        return {}
    assert k_spec.pre_rope, "spectral quantizes pre-RoPE keys; set pre_rope=True"
    assert k_spec.pack_path, "spectral requires pack_path"
    packs = load_packs(k_spec.pack_path, k_spec.budget)
    if thr == PER_LAYER_TIER_SENTINEL:
        # per_layer_tier_thresholds MUST see the pristine (just-loaded, never
        # roundtripped) packs -- int8_decoder_roundtrip is near-idempotent,
        # so deriving the map from an already-roundtripped decoder would
        # silently certify a higher (wrong) tier. This is the ONLY place the
        # map is ever computed; every other consumer (streaming accounting)
        # reads the per-layer dec_tier this loop stamps, never re-derives.
        t_map = per_layer_tier_thresholds(packs)
        packs = {
            i: (
                pack
                if t_map[i] == 0
                else dataclasses.replace(
                    pack,
                    dec=int8_decoder_roundtrip(
                        pack.dec, pack.bits, tier_threshold=t_map[i]
                    ),
                    dec_tier=t_map[i],
                )
            )
            for i, pack in packs.items()
        }
    elif thr is not None:
        # Lever 2 (gated on a later VM quality measurement): roundtrip each
        # layer pack's decoder through the gated int8 roundtrip ONCE, here,
        # at init -- never refit, never re-applied per call.
        packs = {
            i: dataclasses.replace(
                pack,
                dec=int8_decoder_roundtrip(pack.dec, pack.bits, tier_threshold=thr),
                dec_tier=thr,
            )
            for i, pack in packs.items()
        }
    return packs
