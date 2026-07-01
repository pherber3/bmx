"""KV-cache codecs: the arm registry, honest-bpe accounting, and dispatch.

Arms are registered in `_ARM_TABLE` (name -> `_ArmTraits`); `CACHE_ARMS`,
`S_DIVISIBILITY_ARMS`, and `_SPLIT_ARMS` are all derived from that one table,
so adding/removing an arm never requires hand-syncing multiple lists.

Every arm reports an honest bits-per-entry (bpe): ALL metadata is counted
(codebook-free rotations aside, which are seed-generated and cost 0 stored
bits) — scales, per-channel/per-head norms, low-rank factors, and tier maps
all go into the number, never just the payload bit-width.

Streaming-path arms (`_SPLIT_ARMS`) additionally expose a packed split,
`quantize_packed`/`dequant_packed`, used by the token-by-token cache; for
those arms `quantize_cache` is literally `dequant_packed ∘ quantize_packed`.
Non-split (waterfill) arms only support the whole-matrix `quantize_cache`
path.

See `quantize_cache`'s docstring for the per-arm parameter reference.
"""

import dataclasses
import functools
import math

import torch

from bmx.decomp.lrs import truncated_svd
from bmx.quant.hadamard import is_power_of_2, random_orthogonal, randomized_hadamard
from bmx.quant.rtn import rtn_dequantize_packed, rtn_quantize, rtn_quantize_packed

# ---------------------------------------------------------------------------
# Public arm registry
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _ArmTraits:
    s_divisible: bool = False  # codec asserts S % group == 0 (streaming alignment)
    packed: bool = False  # has a quantize_packed/dequant_packed split


_ARM_TABLE: dict[str, _ArmTraits] = {
    "rtn_token": _ArmTraits(packed=True),
    "rtn_channel": _ArmTraits(s_divisible=True, packed=True),
    "rotate_rtn_token": _ArmTraits(packed=True),
    "turboquant_mse": _ArmTraits(packed=True),
    "turboquant_mse_perhead": _ArmTraits(packed=True),
    "turboquant_prod": _ArmTraits(packed=True),
    "lowrank_rtn_channel": _ArmTraits(s_divisible=True, packed=True),
    "lowrank_waterfill_channel": _ArmTraits(s_divisible=True),
    "lowrank_eigwaterfill_channel": _ArmTraits(s_divisible=True),
    "lowrank_randwaterfill_channel": _ArmTraits(s_divisible=True),
    "lowrank_topkwaterfill_channel": _ArmTraits(s_divisible=True),
    "lowrank_blockdiagwaterfill_channel": _ArmTraits(s_divisible=True),
    "lowrank_frozenwaterfill_channel": _ArmTraits(s_divisible=True),
    "lowrank_oraclewaterfill_channel": _ArmTraits(s_divisible=True),
}

CACHE_ARMS = tuple(_ARM_TABLE)
# Arms whose codec asserts S % group == 0 (used by streaming.py for alignment).
S_DIVISIBILITY_ARMS = frozenset(a for a, t in _ARM_TABLE.items() if t.s_divisible)


# ---------------------------------------------------------------------------
# Honest-bpe metadata terms — the audit surface for "ALL metadata counted".
# Every arm's bpe is payload bits + a sum of these named terms; the expressions
# are the scientific record and must not be re-derived or reassociated.
# ---------------------------------------------------------------------------


def scale_bits(group: int) -> float:
    """fp16 groupwise-RTN scale: one fp16 per `group` entries."""
    return 16.0 / group


def norm_bits(h: int, C: int) -> float:
    """fp16 per-row norms, `h` per row of C channels (h=1: one full-row norm)."""
    return 16.0 * h / C


def factor_bits(rank: int, S: int, C: int) -> float:
    """fp16 low-rank factors Us (S×r) + V (C×r), amortized per entry."""
    return 16.0 * rank * (S + C) / (S * C)


def tier_bits(tiers: tuple[int, ...], S: int) -> float:
    """Per-channel tier map: ceil(log2(n_tiers)) bits per channel, amortized."""
    return math.ceil(math.log2(len(tiers))) / S


# ---------------------------------------------------------------------------
# Helper: rotation / unrotation over the channel dim
# ---------------------------------------------------------------------------


def _rotate(M: torch.Tensor, seed: int) -> torch.Tensor:
    """Rotate rows of (S, C) matrix using Hadamard (C power-of-2) or random orthogonal."""
    C = M.shape[-1]
    if is_power_of_2(C):
        return randomized_hadamard(M, seed)
    else:
        Q = random_orthogonal(C, seed, dtype=M.dtype, device=M.device)
        return M @ Q.T


@functools.lru_cache(maxsize=16)
def _hadamard_signs(C: int, seed: int) -> torch.Tensor:
    """Cached ±1 sign vector for the Hadamard rotation (CPU, by (C, seed)).

    Identical values to the inline form used by randomized_hadamard at quantize
    time; cached because _unrotate runs per dequantized block (per decode step, per
    layer) — re-seeding a Generator + randint each call is wasted work at long
    context. Shared/read-only; callers .to(target) for dtype/device.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (C,), generator=g) * 2 - 1


def _unrotate(M_rot: torch.Tensor, seed: int) -> torch.Tensor:
    """Inverse rotation (Hadamard is self-inverse up to same signs; QR needs Q)."""
    C = M_rot.shape[-1]
    if is_power_of_2(C):
        # randomized_hadamard is H @ diag(signs) @ x; inverse is diag(signs) @ H @ x
        # but H^{-1} = H (orthonormal), so: signs * H @ M_rot
        signs = _hadamard_signs(C, seed).to(M_rot)
        from bmx.quant.hadamard import fwht

        return fwht(M_rot) * signs
    else:
        Q = random_orthogonal(C, seed, dtype=M_rot.dtype, device=M_rot.device)
        return M_rot @ Q  # Q^T inverse is Q (orthogonal)


# ---------------------------------------------------------------------------
# Reverse water-filling per-channel bit allocator
# ---------------------------------------------------------------------------


def _round_to_tiers(b: torch.Tensor, tiers_t: torch.Tensor) -> torch.Tensor:
    """Round each continuous bit-rate to the nearest value in tiers_t (1-D sorted)."""
    # |b - tier| argmin over the tier axis
    diffs = (b.unsqueeze(-1) - tiers_t).abs()  # (C, n_tiers)
    idx = diffs.argmin(dim=-1)
    return tiers_t[idx]


def allocate_channel_bits(
    R: torch.Tensor,
    budget_bits: float,
    tiers: tuple[int, ...] = (0, 2, 3, 4),
    *,
    axis: int = 0,
    n_search: int = 40,
) -> torch.Tensor:
    """Reverse-water-filling per-channel bit allocation (Cover-Thomas Thm 13.3.3).

    Per-channel variance var_c (over `axis`); continuous rate
    b_c = max(0, 0.5*log2(var_c / kappa)); kappa bisected so the tier-rounded
    mean lands at-or-just-below budget_bits. Deterministic.

    Returns (C,) int64 bit-widths, each a member of `tiers`.
    """
    assert R.dim() == 2, f"R must be 2-D (S, C); got {tuple(R.shape)}"
    var = R.var(dim=axis, unbiased=False).double().clamp_min(1e-30)  # (C,)
    tiers_t = torch.tensor(sorted(tiers), dtype=torch.float64, device=R.device)

    def rounded_mean(kappa: float) -> tuple[torch.Tensor, float]:
        b_cont = (0.5 * torch.log2(var / kappa)).clamp_min(0.0)
        b_round = _round_to_tiers(b_cont, tiers_t)
        return b_round, b_round.mean().item()

    # Bracket kappa in log space: smaller kappa => more bits.
    lo_k = float(var.min().item()) * 1e-6  # high-bit end
    hi_k = float(var.max().item()) * 1e6  # zero-bit end
    lo = math.log(lo_k)
    hi = math.log(hi_k)

    # Bisect for the smallest kappa whose rounded mean <= budget (monotone:
    # mean is non-increasing in kappa). Keep the best feasible candidate.
    best = _round_to_tiers((0.5 * torch.log2(var / hi_k)).clamp_min(0.0), tiers_t)
    for _ in range(n_search):
        mid = 0.5 * (lo + hi)
        b_round, m = rounded_mean(math.exp(mid))
        if m <= budget_bits + 1e-12:
            best = b_round  # feasible (not over budget); try for more bits
            hi = mid  # decrease kappa
        else:
            lo = mid  # over budget; raise kappa
    return best.to(torch.int64)


# ---------------------------------------------------------------------------
# Gaussian Lloyd-Max codebook (cached)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def gaussian_codebook(
    bits: int,
    n_samples: int = 2**18,
    n_iter: int = 40,
    seed: int = 0,
) -> torch.Tensor:
    """1-D Lloyd-Max codebook for N(0,1), cached by (bits, n_samples, n_iter, seed).

    Returns a sorted tensor of shape (2**bits,).
    """
    n_levels = 2**bits
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, generator=g)

    # Initialize levels at quantiles of the empirical distribution
    quantile_probs = torch.linspace(0, 1, n_levels + 2)[1:-1]  # skip 0 and 1
    levels = torch.quantile(x, quantile_probs)

    for _ in range(n_iter):
        # Compute midpoints (decision boundaries)
        midpoints = (levels[:-1] + levels[1:]) / 2  # (n_levels - 1,)
        # Assign each sample to nearest level via bucketize
        indices = torch.bucketize(x, midpoints)  # 0 .. n_levels-1
        # Recenter each level to the mean of its assigned samples
        new_levels = torch.zeros(n_levels)
        for k in range(n_levels):
            mask = indices == k
            if mask.any():
                new_levels[k] = x[mask].mean()
            else:
                new_levels[k] = levels[k]
        levels = new_levels

    return levels.sort().values


# ---------------------------------------------------------------------------
# QJL public helper
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _qjl_sketch(C: int, seed: int) -> torch.Tensor:
    """Shared (C, C) fp32 Gaussian QJL sketch, cached by (C, seed).

    The returned tensor is shared across callers and must be treated as
    read-only — never mutate it in place.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randn(C, C, generator=g)


def qjl_reconstruct(R: torch.Tensor, seed: int) -> torch.Tensor:
    """Unbiased QJL linear reconstruction for inner-product preservation.

    For each row r in R:
      G = shared Gaussian sketch (C x C), seed-generated (0 stored bits)
      sign_bits = sign(r_unit @ G.T)  -- 1-bit per dimension
      r_hat = ||r||_fp16 * sqrt(pi/2)/C * (sign_bits @ G)

    Returns R_hat of same shape as R.
    """
    S, C = R.shape

    # Per-row norms; fp16 roundtrip for honest accounting
    r_norms = R.norm(dim=1, keepdim=True).clamp_min(1e-12)
    r_norms_stored = r_norms.half().float()

    # Unit vectors
    R_unit = R / r_norms_stored

    # One shared sketch for the whole matrix (cached, read-only)
    G = _qjl_sketch(C, seed).to(R)

    # Signs: (S, C) — sign of projection
    signs = torch.sign(R_unit @ G.T)  # (S, C)

    # Reconstruction: r_hat = ||r|| * sqrt(pi/2)/C * (signs @ G)
    scale = math.sqrt(math.pi / 2) / C
    R_hat = r_norms_stored * scale * (signs @ G)

    return R_hat


# ---------------------------------------------------------------------------
# Arm 7+: lowrank_waterfill_channel family (rotation-parameterized)
# ---------------------------------------------------------------------------


def _klt_basis(X: torch.Tensor) -> torch.Tensor:
    """KLT rotation for the channel covariance of X: eigenvectors of XᵀX, descending
    eigenvalue order. Orthogonal (C×C) — columns are the residual's principal directions."""
    _, ev = torch.linalg.eigh(X.mT @ X)
    return ev.flip(dims=(1,))


def _lowrank_rotwaterfill_channel(
    M: torch.Tensor,
    budget_bits: float,
    group: int,
    rank: int,
    tiers: tuple[int, ...] = (0, 2, 3, 4),
    rotation: str = "klt",
    seed: int = 0,
    charge_rotation: bool = False,
    svd_factors: tuple | None = None,
    topk_k: int = 0,
    prefill_fit_len: int = 0,
    h_kv: int = 0,
) -> tuple[torch.Tensor, float]:
    """Low-rank + rotated per-channel residual at water-filled mixed bit-widths.

    Same as the identity mode, but the residual R = M - L is first rotated
    by an orthogonal Q before per-channel water-filling, then unrotated. Q is either
    the KLT (eigenvectors of R^T R, variance-concentrating) or a seeded random
    orthogonal (variance-spreading control). Q orthogonal => inner products preserved,
    so the rotation is logit-neutral; only the post-rotation quantization distorts.

    Supported rotation modes:
    - "identity": no rotation — water-fill in the original basis (the former
      lowrank_waterfill_channel base arm).
    - "klt": full C×C KLT on residual; charge_rotation=True adds 16*C/S.
    - "random": seeded random orthogonal (0 stored bits; charge_rotation no-op).
    - "topk": honest partial rotation — only top-k eigenvectors (C×k) stored;
      complement stays in original basis. Charge: 16*topk_k/S when charge_rotation.
    - "blockdiag": per-head (d×d) KLT; heads are independent. Charge: 16*d/S.
    - "frozen": full KLT fit on the first prefill_fit_len tokens, applied to all.
      Charge: 16*C/S when charge_rotation.
    - "oracle": full KLT refit on ALL tokens (control; never charged).

    bpe: idealized (rotation free) by default; charge_rotation=True charges the
    stored rotation metadata per mode (except oracle which is never charged).
    """
    S, C = M.shape
    assert rotation in (
        "identity",
        "klt",
        "random",
        "topk",
        "blockdiag",
        "frozen",
        "oracle",
    ), f"unknown rotation {rotation!r}"
    assert rank > 0, f"rotwaterfill requires rank > 0, got {rank}"
    assert rank <= min(S, C), f"rank {rank} > min(S,C)={min(S, C)}"
    assert S % group == 0, f"S={S} not divisible by group={group}"

    if svd_factors is not None:
        Us, V = svd_factors
    else:
        Us, V = truncated_svd(M, rank)
    Us_stored = Us.half().float()
    V_stored = V.half().float()
    L = Us_stored @ V_stored.mT
    R = M - L  # (S, C)

    # When there is only one tier, the allocation is trivially uniform regardless of
    # rotation — skip the rotate/unrotate so the codec is identical to the identity
    # mode (logit-neutral by construction).
    use_rotation = len(set(tiers)) > 1

    def _waterfill_in_basis(
        R_in: torch.Tensor, Q: torch.Tensor | None
    ) -> tuple[torch.Tensor, float]:
        """Water-fill R_in after rotating by Q (None = identity). Returns (R_hat, mean_payload)
        with R_hat in the ORIGINAL basis."""
        R_rot = R_in if Q is None else (R_in @ Q)
        bits_pc = allocate_channel_bits(R_rot, budget_bits, tiers=tiers, axis=0)
        R_rot_hat = torch.zeros_like(R_rot)
        # Each b in the set is present by construction, so cols is never empty.
        for b in sorted(set(int(x) for x in bits_pc.tolist())):
            if b == 0:
                continue
            cols = (bits_pc == b).nonzero(as_tuple=True)[0]
            R_rot_hat[:, cols] = rtn_quantize(R_rot[:, cols].mT, b, group).mT
        R_hat_local = R_rot_hat if Q is None else (R_rot_hat @ Q.mT)
        return R_hat_local, float(bits_pc.float().mean().item())

    if rotation == "identity" or not use_rotation:
        R_hat, mean_payload = _waterfill_in_basis(R, None)
        rot_bits = 0.0
    elif rotation in ("klt", "oracle", "frozen"):
        # All three are a full-C×C KLT differing only in WHICH tokens fit it and the
        # charge rule: klt/oracle fit on all tokens, frozen on the first P; oracle is
        # the never-charged control (it refits on the scored tokens — not deployable).
        P = min(prefill_fit_len if prefill_fit_len > 0 else S, S)
        fit = R[:P] if rotation == "frozen" else R
        Q = _klt_basis(fit)
        R_hat, mean_payload = _waterfill_in_basis(R, Q)
        rot_bits = (16.0 * C / S) if (charge_rotation and rotation != "oracle") else 0.0
    elif rotation == "random":
        Q = random_orthogonal(C, seed, dtype=R.dtype, device=R.device)
        R_hat, mean_payload = _waterfill_in_basis(R, Q)
        rot_bits = 0.0  # seeded rotation costs 0 stored bits
    elif rotation == "topk":
        # Honest partial rotation: only top-k eigen-directions (C×k) stored.
        # The complement stays in the original basis — the complement eigenvectors
        # are data-dependent and NOT recomputable by the decoder, so rotating the
        # full basis and charging only 16*k/S would overstate compression.
        # topk_k=0 defaults to C (full eigenbasis), equivalent to KLT.
        kk = min(topk_k if topk_k > 0 else C, C)
        Qk = _klt_basis(R)[:, :kk]  # (C, k) top-k eigenvectors by eigenvalue
        # Rotate top-k subspace; water-fill those k columns.
        Rk = R @ Qk  # (S, k) projection onto top-k subspace
        Rk_hat, p_k = _waterfill_in_basis(Rk, None)  # already in rotated subspace
        topk_back = Rk_hat @ Qk.mT  # (S, C) contribution from top-k subspace
        # Complement: residual not explained by top-k, water-filled in original basis.
        R_comp = R - Rk @ Qk.mT  # project OUT the top-k subspace (Rk = R @ Qk, reused)
        Rcomp_hat, p_c = _waterfill_in_basis(R_comp, None)
        R_hat = topk_back + Rcomp_hat
        # Blended payload over all C channels (k rotated + C-k complement).
        mean_payload = (p_k * kk + p_c * (C - kk)) / C
        rot_bits = (16.0 * kk / S) if charge_rotation else 0.0
    else:  # blockdiag
        # h_kv=0 defaults to 1 (treat whole channel as one block = full KLT).
        _h = h_kv if h_kv > 0 else 1
        assert C % _h == 0, f"C={C} not divisible by h_kv={_h}"
        d = C // _h
        R_hat = torch.zeros_like(R)
        payloads = []
        for hh in range(_h):
            sl = slice(hh * d, (hh + 1) * d)
            Rh = R[:, sl]
            Rh_hat, ph = _waterfill_in_basis(Rh, _klt_basis(Rh))
            R_hat[:, sl] = Rh_hat
            payloads.append(ph)
        mean_payload = float(sum(payloads) / len(payloads))
        rot_bits = (16.0 * d / S) if charge_rotation else 0.0

    M_hat = L + R_hat
    scale_term = scale_bits(group)
    factor_term = factor_bits(rank, S, C)
    tier_term = tier_bits(tiers, S)
    bpe = mean_payload + scale_term + factor_term + tier_term + rot_bits
    return M_hat, bpe


# ---------------------------------------------------------------------------
# Packed split: quantize_packed / dequant_packed
# ---------------------------------------------------------------------------

# Arms with a quantize_packed/dequant_packed split (the streaming path).
_SPLIT_ARMS = frozenset(a for a, t in _ARM_TABLE.items() if t.packed)

# Waterfill dispatch: arm name -> _lowrank_rotwaterfill_channel's rotation mode.
_WATERFILL_ROTATION = {
    "lowrank_waterfill_channel": "identity",
    "lowrank_eigwaterfill_channel": "klt",
    "lowrank_randwaterfill_channel": "random",
    "lowrank_topkwaterfill_channel": "topk",
    "lowrank_blockdiagwaterfill_channel": "blockdiag",
    "lowrank_frozenwaterfill_channel": "frozen",
    "lowrank_oraclewaterfill_channel": "oracle",
}


# ---------------------------------------------------------------------------
# Per-head turboquant_mse (QuaRot/SpinQuant-style block-diagonal rotation)
#
# The standard turboquant_mse rotates the FULL (S, C) row with one C-wide
# Hadamard, which couples all heads — fine for chunked dequant, but it blocks a
# fused per-head decode kernel (the unrotate would need all C channels, and under
# GQA each query head has its own softmax, so neither an o_proj-fold nor a
# per-head accumulation recovers it cleanly).
#
# The per-head variant rotates each d_head block INDEPENDENTLY (Hadamard over d,
# per-head norms), so V dequant is fully per-head: the fused kernel does a
# d_head-point FWHT in-register, no cross-head traffic, no o_proj surgery. The
# turboquant distortion bound (√3·π/2·4^−b) is dimension-independent in the
# constant and the Beta→Gaussian concentration is excellent at d=128, so per-head
# is quality-equivalent to full-C (see brain consult / QuaRot/SpinQuant precedent).
# ---------------------------------------------------------------------------


def _turboquant_mse_perhead_packed(
    M: torch.Tensor, bits: int, seed: int, h: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head turboquant pack. (S, C=h*d) -> (indices int16 (S,C), norms (S,h)).

    Rotates each of the h d-blocks independently (Hadamard over d); norms are
    per-(row, head) so each head's d-block is self-contained.
    """
    S, C = M.shape
    d = C // h
    Mh = M.reshape(S, h, d)  # (S, h, d) — per-head blocks
    norms = Mh.norm(dim=2).clamp_min(1e-12).half().float()  # (S, h) per-head norm
    Mh_unit = Mh / norms[:, :, None]  # (S, h, d)
    Mh_rot = _rotate(Mh_unit.reshape(S * h, d), seed).reshape(
        S, h, d
    )  # per-d-block rot
    cb = gaussian_codebook(bits).to(M.device)
    sqrt_d = math.sqrt(d)
    mid = (cb[:-1] + cb[1:]) / 2
    indices = torch.bucketize(Mh_rot * sqrt_d, mid).to(torch.int16).reshape(S, C)
    return indices, norms


def _turboquant_mse_perhead_dequant(
    indices: torch.Tensor, norms: torch.Tensor, bits: int, seed: int, h: int
) -> torch.Tensor:
    """Inverse of _turboquant_mse_perhead_packed. (S,C),(S,h) -> (S,C)."""
    S, C = indices.shape
    d = C // h
    cb = gaussian_codebook(bits).to(norms.device)
    sqrt_d = math.sqrt(d)
    Mh_quant = (cb[indices.long()] / sqrt_d).reshape(S, h, d)  # (S, h, d)
    Mh_recon = _unrotate(Mh_quant.reshape(S * h, d), seed).reshape(S, h, d)
    return (Mh_recon * norms[:, :, None]).reshape(S, C)


# Confirmed 2026-07-01 (kill-or-confirm gate): full-C turboquant is exactly the
# h=1 case of the perhead codec above. _turboquant_mse_perhead_packed reshapes
# to (S*h, d) with d = C // h before calling the shared _rotate helper, so at
# h=1, d == C and _rotate dispatches identically (Hadamard iff C is a power of
# 2, else the same random_orthogonal fallback) — there is no pow2(d) assertion
# anywhere in the perhead path. Verified bit-identical via torch.equal for
# C=128 (pow2), C=96 and C=48 (non-pow2). The two dict schemas
# ({indices, norms, bits} vs {indices, norms, bits, h}) are preserved by
# reshaping norms (S,1) <-> (S,h=1) at the call sites in quantize_packed /
# dequant_packed, not in the shared body.
def _turboquant_mse_packed(
    M: torch.Tensor, bits: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(S,C) -> (indices int16 (S,C), norms fp (S,1)). Codebook from bits+seed."""
    indices, norms_h = _turboquant_mse_perhead_packed(M, bits, seed, 1)
    return indices, norms_h.reshape(M.shape[0], 1)


def _turboquant_mse_dequant(
    indices: torch.Tensor, norms: torch.Tensor, bits: int, seed: int, C: int
) -> torch.Tensor:
    """Reconstruct from packed turboquant_mse representation."""
    return _turboquant_mse_perhead_dequant(indices, norms, bits, seed, 1)


def quantize_packed(
    arm: str,
    M: torch.Tensor,
    *,
    bits: int,
    seed: int = 0,
    group: int = 64,
    rank: int = 0,
    svd_factors: tuple | None = None,
    h_heads: int = 0,
) -> tuple[dict, float]:
    """(S,C) fp -> (packed dict, honest bpe). Inverse: dequant_packed.

    Only the streaming-path arms are supported; waterfill arms raise
    NotImplementedError (use quantize_cache directly for those).
    """
    if arm not in _SPLIT_ARMS:
        raise NotImplementedError(
            f"arm {arm!r} not split into packed form (not on the streaming path); "
            f"use quantize_cache. Split arms: {sorted(_SPLIT_ARMS)}"
        )
    S, C = M.shape
    if arm == "rtn_token":
        Q_int, scale = rtn_quantize_packed(M, bits, group)
        return {"Q_int": Q_int, "scale": scale}, bits + scale_bits(group)
    if arm == "rtn_channel":
        Q_int, scale = rtn_quantize_packed(M.mT, bits, group)
        return {"Q_int": Q_int, "scale": scale}, bits + scale_bits(group)
    if arm == "rotate_rtn_token":
        Q_int, scale = rtn_quantize_packed(_rotate(M, seed), bits, group)
        return {"Q_int": Q_int, "scale": scale}, bits + scale_bits(group)
    if arm == "turboquant_mse":
        indices, norms = _turboquant_mse_packed(M, bits, seed)
        return {"indices": indices, "norms": norms, "bits": bits}, bits + norm_bits(
            1, C
        )
    if arm == "turboquant_mse_perhead":
        # Per-head (block-diagonal d_head Hadamard) turboquant — fuses into the
        # per-head decode kernel (no cross-head coupling). norms are per-(row, head).
        # h_heads omitted (0) -> degenerate single head = full-C rotation (same as
        # turboquant_mse); the cache always passes the real h_kv. C % h_heads == 0.
        h = h_heads if h_heads > 0 else 1
        assert C % h == 0, f"C={C} not divisible by h_heads={h}"
        indices, norms = _turboquant_mse_perhead_packed(M, bits, seed, h)
        # bpe: bits/elem + 16 (fp16 norm) per (head's d) channels = bits + 16/(C/h).
        return {"indices": indices, "norms": norms, "bits": bits, "h": h}, (
            bits + norm_bits(h, C)
        )
    if arm == "turboquant_prod":
        assert bits >= 2, f"turboquant_prod requires bits >= 2, got {bits}"
        indices, norms = _turboquant_mse_packed(M, bits - 1, seed)
        M1 = _turboquant_mse_dequant(indices, norms, bits - 1, seed, C)
        R = M - M1
        r_norms = R.norm(dim=1, keepdim=True).clamp_min(1e-12).half().float()
        R_unit = R / r_norms
        G = _qjl_sketch(C, seed).to(R)
        signs = torch.sign(R_unit @ G.T)
        packed = {
            "mse_indices": indices,
            "mse_norms": norms,
            "bits": bits,
            "qjl_signs": signs.to(torch.int8),
            "qjl_norms": r_norms,
        }
        # payload + 1 sign bit + two fp16 norm vectors (mse + qjl) = 2 * norm_bits(1, C)
        return packed, (bits - 1) + 1 + 32.0 / C
    # lowrank_rtn_channel
    assert rank > 0, f"lowrank_rtn_channel requires rank > 0, got {rank}"
    assert rank <= min(S, C), f"rank {rank} > min(S,C)={min(S, C)}"
    assert S % group == 0, f"S={S} not divisible by group={group}"
    if svd_factors is not None:
        Us, V = svd_factors
    else:
        Us, V = truncated_svd(M, rank)
    Us_stored = Us.half().float()
    V_stored = V.half().float()
    L = Us_stored @ V_stored.mT
    R = M - L
    res_Q_int, res_scale = rtn_quantize_packed(R.mT, bits, group)
    bpe = bits + scale_bits(group) + factor_bits(rank, S, C)
    return {
        "Us": Us_stored,
        "V": V_stored,
        "res_Q_int": res_Q_int,
        "res_scale": res_scale,
    }, bpe


def dequant_packed(
    arm: str, packed: dict, *, seed: int = 0, group: int = 64
) -> torch.Tensor:
    """Inverse of quantize_packed -> dequantized (S,C) M_hat."""
    if arm not in _SPLIT_ARMS:
        raise NotImplementedError(
            f"arm {arm!r} not split into packed form (not on the streaming path); "
            f"use quantize_cache. Split arms: {sorted(_SPLIT_ARMS)}"
        )
    if arm == "rtn_token":
        return rtn_dequantize_packed(packed["Q_int"], packed["scale"], group)
    if arm == "rtn_channel":
        return rtn_dequantize_packed(packed["Q_int"], packed["scale"], group).mT
    if arm == "rotate_rtn_token":
        M_rot_hat = rtn_dequantize_packed(packed["Q_int"], packed["scale"], group)
        return _unrotate(M_rot_hat, seed)
    if arm == "turboquant_mse":
        C = packed["indices"].shape[1]
        return _turboquant_mse_dequant(
            packed["indices"], packed["norms"], packed["bits"], seed, C
        )
    if arm == "turboquant_mse_perhead":
        return _turboquant_mse_perhead_dequant(
            packed["indices"], packed["norms"], packed["bits"], seed, packed["h"]
        )
    if arm == "turboquant_prod":
        C = packed["mse_indices"].shape[1]
        M1 = _turboquant_mse_dequant(
            packed["mse_indices"], packed["mse_norms"], packed["bits"] - 1, seed, C
        )
        G = _qjl_sketch(C, seed).to(packed["qjl_norms"])
        signs = packed["qjl_signs"].to(G.dtype)
        scale = math.sqrt(math.pi / 2) / C
        R_hat = packed["qjl_norms"] * scale * (signs @ G)
        return M1 + R_hat
    # lowrank_rtn_channel
    L = packed["Us"] @ packed["V"].mT
    R_hat = rtn_dequantize_packed(packed["res_Q_int"], packed["res_scale"], group).mT
    return L + R_hat


# ---------------------------------------------------------------------------
# Dispatch: quantize_cache
# ---------------------------------------------------------------------------


def quantize_cache(
    arm: str,
    M: torch.Tensor,
    *,
    bits: int,
    seed: int = 0,
    group: int = 64,
    rank: int = 0,
    svd_factors: tuple | None = None,
    tiers: tuple[int, ...] = (0, 2, 3, 4),
    charge_rotation: bool = False,
    topk_k: int = 0,
    prefill_fit_len: int = 0,
    h_kv: int = 0,
    h_heads: int = 0,
) -> tuple[torch.Tensor, float]:
    """Compress (S, C) fp32 matrix M with the specified arm.

    Parameters
    ----------
    arm : str
        One of CACHE_ARMS.
    M : torch.Tensor
        (S, C) fp32 token-by-channel cache matrix.
    bits : int
        Quantization bit width.
    seed : int
        RNG seed for rotation/sketch arms (0 stored bits).
    group : int
        Group size for rtn_token / rtn_channel / rotate_rtn_token / lowrank arms.
    rank : int
        Low-rank components for lowrank_rtn_channel (must be > 0 for that arm).
    svd_factors : tuple | None
        Optional pre-computed (Us, V) from truncated_svd(M, rank), mirroring
        bmx.quant.arms's ``ls`` param precedent.  When provided, the internal
        truncated_svd call is skipped (useful when sweeping bits for a fixed
        (M, rank) — the SVD result depends only on those two, not on bits).
        Used by lowrank_rtn_channel (svd_factors only) and all lowrank_*waterfill_channel arms;
        ignored by the RTN/turboquant arms.
    tiers : tuple[int, ...]
        Allowed bit-widths for per-channel allocation in lowrank_rtn_channel (svd_factors only)
        and all lowrank_*waterfill_channel arms; ignored by the RTN/turboquant arms.
    charge_rotation : bool
        Add the rotation-matrix metadata cost to bpe; arm-dependent (see
        _lowrank_rotwaterfill_channel docstring for per-mode details).
    topk_k : int
        Number of top eigen-directions to rotate; used by lowrank_topkwaterfill_channel.
    prefill_fit_len : int
        Number of prefix tokens to fit the frozen KLT on; used by
        lowrank_frozenwaterfill_channel.
    h_kv : int
        Number of KV heads; used by lowrank_blockdiagwaterfill_channel to reshape
        channels into per-head blocks.
    h_heads : int
        Number of KV heads for per-head split arms (turboquant_mse_perhead); 0 = full-C.
        Ignored by other arms.

    Returns
    -------
    M_hat : torch.Tensor
        Dequantized approximation, same shape as M.
    bpe : float
        Honest bits-per-entry including ALL metadata.
    """
    assert arm in CACHE_ARMS, f"unknown arm {arm!r}; available: {CACHE_ARMS}"

    if arm in _SPLIT_ARMS:
        packed, bpe = quantize_packed(
            arm,
            M,
            bits=bits,
            seed=seed,
            group=group,
            rank=rank,
            svd_factors=svd_factors,
            h_heads=h_heads,
        )
        return dequant_packed(arm, packed, seed=seed, group=group), bpe
    return _lowrank_rotwaterfill_channel(
        M,
        float(bits),
        group,
        rank,
        tiers=tiers,
        rotation=_WATERFILL_ROTATION[arm],
        seed=seed,
        charge_rotation=charge_rotation,
        topk_k=topk_k,
        prefill_fit_len=prefill_fit_len,
        h_kv=h_kv,
        svd_factors=svd_factors,
    )


# ---------------------------------------------------------------------------
# Shared layout helper (used by ppl_eval and streaming)
# ---------------------------------------------------------------------------


def quantize_kv_layout(
    kv_fp: torch.Tensor,
    spec,  # CacheCodecSpec; avoid circular import — duck-typed on .arm/.bits/etc.
) -> tuple[torch.Tensor, float]:
    """Quantize an (h, S, d) tensor using the matrix layout convention.

    Returns ``(kv_hat, bpe)`` where ``kv_hat`` is the dequantized fp32 result
    with the same shape as *kv_fp*.  For ``arm="fp16"``, returns the input
    unchanged with ``bpe=16.0``.  Raises ``AssertionError`` for unknown arms.
    """
    from bmx.cache.collect import (
        from_matrix,
        to_matrix,
    )  # local to avoid top-level cycle

    h = kv_fp.shape[0]
    if spec.arm == "fp16":
        return kv_fp, 16.0

    assert spec.arm in CACHE_ARMS, (
        f"unknown arm {spec.arm!r}; use one of {CACHE_ARMS} or 'fp16'"
    )
    M = to_matrix(kv_fp)  # (S, h*d)
    M_hat, bpe = quantize_cache(
        spec.arm,
        M,
        bits=spec.bits,
        seed=spec.seed,
        group=spec.group,
        rank=spec.rank,
        h_heads=h,
    )
    return from_matrix(M_hat, h), bpe
