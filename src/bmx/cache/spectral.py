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
  scale amortized over S rows (16·c_used/(S·C)).
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
from pathlib import Path

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

    @property
    def c_used(self) -> int:
        """Number of directions carrying nonzero bits — the decoder columns
        actually read at decode (see skeptic-v2 / payload-v2 above)."""
        return int((self.bits != 0).sum())


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


def int8_decoder_roundtrip(dec: torch.Tensor, bits_pc: torch.Tensor) -> torch.Tensor:
    """Simulate int8 storage of a decoder matrix's USED columns (Lever 2).

    Per-column symmetric absmax int8: scale = absmax/127, cast to fp16 (the
    shipped scale dtype) then back to fp32 BEFORE dequant so the roundtrip
    reflects the exact precision an int8-stored decoder would reconstruct at.
    Unused columns (bits_pc == 0 -- never read at decode, see
    test_dropped_decoder_columns_never_read) are returned untouched.
    Deterministic; fp32 in, fp32 out. Refits nothing -- this operates on an
    already-fitted pack's dec tensor.
    """
    used = bits_pc != 0
    dec_rt = dec.clone()
    dec_used = dec[:, used]
    scale = (dec_used.abs().amax(dim=0) / 127.0).clamp_min(1e-12).half().float()
    codes = (dec_used / scale).round().clamp(-127, 127)
    dec_rt[:, used] = codes * scale
    return dec_rt


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


def int8_decoder_certificate_tiered(pack: SpectralPack, tier_threshold: int) -> dict:
    """Tier-gated rescue of `int8_decoder_certificate` (K4 math-actions 6b):
    "is int8 dead or just misapplied?" The blanket certificate int8-stores
    EVERY used decoder column; this asks what happens if only the low-tier
    columns (`0 < bits_i <= tier_threshold`) are int8-stored and the rest
    stay fp16.

    Pure post-processing of the existing `ddec = dec_int8 - dec`: columns
    with `bits_i > tier_threshold` are zeroed in `ddec` BEFORE the same
    `enc^T ddec` projection the blanket certificate uses (those columns stay
    fp16 -> zero added error; no codec change, no new roundtrip). `added`,
    `payload`, `noise_to_signal`, `implied_rel_degradation` are exactly
    `int8_decoder_certificate`'s expressions restricted to the gated ddec —
    at `tier_threshold >= max(pack.bits)` this reproduces the blanket
    certificate bit-for-bit (nothing gets zeroed); at
    `tier_threshold < min(used tier)` `added == 0.0` exactly (everything
    gets zeroed).

    Mixed-decoder accounting (`c_used`/`c_int8` below) is priced separately by
    `mixed_dec_charge`/`effective_dec_bits` — see those docstrings for the
    skeptic-v2 arithmetic generalized to a per-column int8/fp16 mix.
    """
    used = pack.bits != 0
    gate = used & (pack.bits <= tier_threshold)  # int8-eligible columns
    c_used = int(used.sum())
    c_int8 = int(gate.sum())

    ddec_full = int8_decoder_roundtrip(pack.dec, pack.bits).double() - pack.dec.double()
    ddec = ddec_full * gate.to(ddec_full.dtype)  # zero columns above threshold
    proj = pack.enc.double().mT @ ddec
    lam = pack.lam.double().clamp_min(0.0)
    added = float((lam * (proj**2).sum(dim=0)).sum())
    payload = float((lam * torch.pow(4.0, -pack.bits.double())).sum())

    return dict(
        tier_threshold=int(tier_threshold),
        c_used=c_used,
        c_int8=c_int8,
        frac_int8=c_int8 / c_used if c_used > 0 else 0.0,
        added=added,
        payload=payload,
        noise_to_signal=added / max(payload, 1e-300),
        implied_rel_degradation=added / max(payload + added, 1e-300),
    )


def mixed_dec_charge(
    C: int, S: int, tiers: tuple[int, ...], *, c_used: int, c_int8: int
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


def load_packs_for_spec(k_spec) -> dict[int, SpectralPack]:
    """Materialize the per-layer pack dict a KV cache should hold for `k_spec`.

    Pack materialization owns the dec_quant decision — this is the
    design-note altitude both StreamingQuantizedCache.__init__ and
    PackedStreamingCache.__init__ delegate to (pure code motion, zero
    numeric change: byte-identical blocks extracted verbatim from both).

    Non-spectral arms (arm != "spectral") return {} — nothing to load.
    Spectral arms load the fitted pack file (asserting pre_rope + pack_path
    are set) and, when k_spec.dec_quant == "int8", roundtrip every layer's
    decoder matrix through int8 ONCE here (Lever 2 — see
    int8_decoder_roundtrip) via dataclasses.replace, which keeps every other
    pack field (enc, lam, bits, tiers, ...) untouched.

    k_spec: a CacheCodecSpec (arm, pre_rope, pack_path, budget, dec_quant).
    """
    assert k_spec.dec_quant in ("fp32", "int8"), (
        f"dec_quant must be 'fp32' or 'int8'; got {k_spec.dec_quant!r}"
    )
    if k_spec.arm != "spectral":
        return {}
    assert k_spec.pre_rope, "spectral quantizes pre-RoPE keys; set pre_rope=True"
    assert k_spec.pack_path, "spectral requires pack_path"
    packs = load_packs(k_spec.pack_path, k_spec.budget)
    if k_spec.dec_quant == "int8":
        # Lever 2 (gated on a later VM quality measurement): roundtrip each
        # layer pack's decoder through int8 ONCE, here, at init -- never
        # refit, never re-applied per call.
        packs = {
            i: dataclasses.replace(
                pack, dec=int8_decoder_roundtrip(pack.dec, pack.bits)
            )
            for i, pack in packs.items()
        }
    return packs
