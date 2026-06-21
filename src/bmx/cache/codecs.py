"""KV-cache codecs: six compression arms at honestly matched bits.

Public API
----------
CACHE_ARMS : tuple[str, ...]
    All supported arm names.

quantize_cache(arm, M, *, bits, seed=0, group=64, rank=0) -> (M_hat, bpe)
    Compress (S, C) fp32 matrix M with the given arm and return the
    dequantized approximation plus the *honest* bits-per-entry (ALL metadata
    included; seed-generated rotations/sketches cost 0 stored bits).

gaussian_codebook(bits, ...) -> Tensor  [module-level, lru_cache]
    1-D Lloyd-Max codebook for quantizing coordinates of unit-norm vectors.

qjl_reconstruct(R, seed) -> R_hat
    Unbiased QJL linear reconstruction; exported for unit-testing.
"""

import functools
import math

import torch

from bmx.decomp.lrs import truncated_svd
from bmx.quant.hadamard import is_power_of_2, random_orthogonal, randomized_hadamard
from bmx.quant.rtn import rtn_quantize

# ---------------------------------------------------------------------------
# Public arm registry
# ---------------------------------------------------------------------------

CACHE_ARMS = (
    "rtn_token",
    "rtn_channel",
    "rotate_rtn_token",
    "turboquant_mse",
    "turboquant_prod",
    "lowrank_rtn_channel",
    "lowrank_waterfill_channel",
    "lowrank_eigwaterfill_channel",
    "lowrank_randwaterfill_channel",
    "lowrank_topkwaterfill_channel",
    "lowrank_blockdiagwaterfill_channel",
    "lowrank_frozenwaterfill_channel",
    "lowrank_oraclewaterfill_channel",
)

# Arms whose codec asserts S % group == 0 (used by streaming.py for alignment).
S_DIVISIBILITY_ARMS = frozenset(
    {
        "rtn_channel",
        "lowrank_rtn_channel",
        "lowrank_waterfill_channel",
        "lowrank_eigwaterfill_channel",
        "lowrank_randwaterfill_channel",
        "lowrank_topkwaterfill_channel",
        "lowrank_blockdiagwaterfill_channel",
        "lowrank_frozenwaterfill_channel",
        "lowrank_oraclewaterfill_channel",
    }
)


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


def _unrotate(M_rot: torch.Tensor, seed: int) -> torch.Tensor:
    """Inverse rotation (Hadamard is self-inverse up to same signs; QR needs Q)."""
    C = M_rot.shape[-1]
    if is_power_of_2(C):
        # randomized_hadamard is H @ diag(signs) @ x; inverse is diag(signs) @ H @ x
        # but H^{-1} = H (orthonormal), so: signs * H @ M_rot
        g = torch.Generator().manual_seed(seed)
        signs = (torch.randint(0, 2, (C,), generator=g) * 2 - 1).to(M_rot)
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
# Arm 1: rtn_token
# ---------------------------------------------------------------------------


def _rtn_token(M: torch.Tensor, bits: int, group: int) -> tuple[torch.Tensor, float]:
    """Groupwise symmetric RTN along channel dim per token."""
    S, C = M.shape
    assert C % group == 0, f"C={C} not divisible by group={group}"
    M_hat = rtn_quantize(M, bits, group)
    bpe = bits + 16.0 / group
    return M_hat, bpe


# ---------------------------------------------------------------------------
# Arm 2: rtn_channel
# ---------------------------------------------------------------------------


def _rtn_channel(M: torch.Tensor, bits: int, group: int) -> tuple[torch.Tensor, float]:
    """KIVI-style: symmetric RTN along the token dim per channel."""
    S, C = M.shape
    assert S % group == 0, f"S={S} not divisible by group={group}"
    M_hat = rtn_quantize(M.mT, bits, group).mT
    bpe = bits + 16.0 / group
    return M_hat, bpe


# ---------------------------------------------------------------------------
# Arm 3: rotate_rtn_token
# ---------------------------------------------------------------------------


def _rotate_rtn_token(
    M: torch.Tensor, bits: int, group: int, seed: int
) -> tuple[torch.Tensor, float]:
    """QuaRot-style: rotate channels, quantize per-token, rotate back."""
    S, C = M.shape
    assert C % group == 0, f"C={C} not divisible by group={group}"
    M_rot = _rotate(M, seed)
    M_rot_hat = rtn_quantize(M_rot, bits, group)
    M_hat = _unrotate(M_rot_hat, seed)
    bpe = bits + 16.0 / group
    return M_hat, bpe


# ---------------------------------------------------------------------------
# Arm 4: turboquant_mse
# ---------------------------------------------------------------------------


def _turboquant_mse(
    M: torch.Tensor, bits: int, seed: int
) -> tuple[torch.Tensor, float]:
    """Per-token: store ||x|| fp16, rotate normalized vector, quantize with
    Lloyd-Max codebook scaled by 1/sqrt(C), unrotate, rescale."""
    S, C = M.shape

    # Per-token norms; fp16 roundtrip to simulate honest storage
    norms = M.norm(dim=1, keepdim=True).clamp_min(1e-12)
    norms_stored = norms.half().float()

    # Normalize
    M_unit = M / norms_stored

    # Rotate
    M_rot = _rotate(M_unit, seed)

    # Quantize using Lloyd codebook: coords of unit vector ~ N(0, 1/C)
    # so quantize x*sqrt(C) against N(0,1) codebook then divide by sqrt(C)
    cb = gaussian_codebook(bits).to(M.device)
    sqrt_c = math.sqrt(C)
    M_scaled = M_rot * sqrt_c

    # Nearest-codebook assignment: the codebook is sorted, so bucketize on the
    # midpoints between adjacent levels is exact (up to measure-zero fp ties).
    # M_scaled: (S, C), cb: (2^bits,)
    mid = (cb[:-1] + cb[1:]) / 2  # (2^bits - 1,)
    indices = torch.bucketize(M_scaled, mid)  # (S, C), values 0 .. 2^bits-1
    M_quantized = cb[indices] / sqrt_c  # (S, C)

    # Unrotate
    M_recon = _unrotate(M_quantized, seed)

    # Rescale
    M_hat = M_recon * norms_stored

    bpe = bits + 16.0 / C
    return M_hat, bpe


# ---------------------------------------------------------------------------
# QJL public helper (also used by turboquant_prod)
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
# Arm 5: turboquant_prod
# ---------------------------------------------------------------------------


def _turboquant_prod(
    M: torch.Tensor, bits: int, seed: int
) -> tuple[torch.Tensor, float]:
    """Two-stage: turboquant_mse at (bits-1) + 1-bit QJL on residual."""
    assert bits >= 2, f"turboquant_prod requires bits >= 2, got {bits}"
    S, C = M.shape

    # Stage 1: MSE quantization at (bits-1)
    M1, _ = _turboquant_mse(M, bits - 1, seed)

    # Residual
    R = M - M1  # (S, C)

    # Stage 2: QJL reconstruction (unbiased for inner product)
    R_hat = qjl_reconstruct(R, seed)

    M_hat = M1 + R_hat

    # bpe = (b-1) + 1 + 32/C  (two fp16 norms: one from MSE stage, one from QJL)
    bpe = (bits - 1) + 1 + 32.0 / C
    return M_hat, bpe


# ---------------------------------------------------------------------------
# Arm 6: lowrank_rtn_channel
# ---------------------------------------------------------------------------


def _lowrank_rtn_channel(
    M: torch.Tensor,
    bits: int,
    group: int,
    rank: int,
    svd_factors: tuple | None = None,
) -> tuple[torch.Tensor, float]:
    """Low-rank + quantized residual (K1-margin arm)."""
    S, C = M.shape
    assert rank > 0, f"lowrank_rtn_channel requires rank > 0, got {rank}"
    assert rank <= min(S, C), f"rank {rank} > min(S,C)={min(S, C)}"
    assert S % group == 0, f"S={S} not divisible by group={group}"

    # Low-rank approximation — optionally skip truncated_svd when factors are
    # pre-computed (e.g., reused across a bits sweep for the same (M, rank)).
    if svd_factors is not None:
        Us, V = svd_factors
    else:
        Us, V = truncated_svd(M, rank)  # Us: (S, rank), V: (C, rank)

    # fp16 roundtrip for honest stored-precision
    Us_stored = Us.half().float()
    V_stored = V.half().float()
    L = Us_stored @ V_stored.mT  # (S, C)

    # Residual quantized per channel (rtn_channel)
    R = M - L
    R_hat, _ = _rtn_channel(R, bits, group)

    M_hat = L + R_hat

    # bpe = b + 16/group + 16*rank*(S+C)/(S*C)
    bpe = bits + 16.0 / group + 16.0 * rank * (S + C) / (S * C)
    return M_hat, bpe


# ---------------------------------------------------------------------------
# Arm 7: lowrank_waterfill_channel
# ---------------------------------------------------------------------------


def _lowrank_waterfill_channel(
    M: torch.Tensor,
    budget_bits: float,
    group: int,
    rank: int,
    tiers: tuple[int, ...] = (0, 2, 3, 4),
    svd_factors: tuple | None = None,
) -> tuple[torch.Tensor, float]:
    """Low-rank + per-channel residual at water-filled mixed bit-widths.

    Same low-rank path as lowrank_rtn_channel; the residual R = M - L is
    quantized per channel at bit-widths chosen by reverse water-filling over
    per-channel variance (Cover-Thomas Thm 13.3.3). Tier 0 channels are dropped
    (reconstructed from L only).
    """
    S, C = M.shape
    assert rank > 0, f"lowrank_waterfill_channel requires rank > 0, got {rank}"
    assert rank <= min(S, C), f"rank {rank} > min(S,C)={min(S, C)}"
    assert S % group == 0, f"S={S} not divisible by group={group}"

    if svd_factors is not None:
        Us, V = svd_factors
    else:
        Us, V = truncated_svd(M, rank)

    # fp16 roundtrip for honest stored-precision — identical to _lowrank_rtn_channel
    Us_stored = Us.half().float()
    V_stored = V.half().float()
    L = Us_stored @ V_stored.mT  # (S, C)

    R = M - L  # (S, C)

    # Allocate per-channel bits on the residual.
    bits_per_ch = allocate_channel_bits(R, budget_bits, tiers=tiers, axis=0)  # (C,)

    # Quantize each tier-group of channels at its bit-width; tier 0 -> zeros.
    # Each b in the set is present by construction, so cols is never empty.
    R_hat = torch.zeros_like(R)
    for b in sorted(set(int(x) for x in bits_per_ch.tolist())):
        if b == 0:
            continue  # dropped channels stay zero
        cols = (bits_per_ch == b).nonzero(as_tuple=True)[0]
        sub = R[:, cols]  # (S, n_b); quantize per channel along token dim
        sub_hat = rtn_quantize(sub.mT, b, group).mT  # (n_b, S) groups -> back
        R_hat[:, cols] = sub_hat

    M_hat = L + R_hat

    # Honest bpe (per entry; all metadata counted):
    mean_payload = float(bits_per_ch.float().mean().item())
    scale_term = 16.0 / group
    factor_term = 16.0 * rank * (S + C) / (S * C)
    tier_term = math.ceil(math.log2(len(tiers))) / S
    bpe = mean_payload + scale_term + factor_term + tier_term
    return M_hat, bpe


# ---------------------------------------------------------------------------
# Arm 8+9: lowrank_eigwaterfill_channel / lowrank_randwaterfill_channel
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

    Same as _lowrank_waterfill_channel, but the residual R = M - L is first rotated
    by an orthogonal Q before per-channel water-filling, then unrotated. Q is either
    the KLT (eigenvectors of R^T R, variance-concentrating) or a seeded random
    orthogonal (variance-spreading control). Q orthogonal => inner products preserved,
    so the rotation is logit-neutral; only the post-rotation quantization distorts.

    Supported rotation modes:
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
    # rotation — skip the rotate/unrotate so the codec is identical to the base
    # _lowrank_waterfill_channel arm (logit-neutral by construction).
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

    if not use_rotation:
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
        R_comp = R - (R @ Qk) @ Qk.mT  # project OUT the top-k subspace
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
    scale_term = 16.0 / group
    factor_term = 16.0 * rank * (S + C) / (S * C)
    tier_term = math.ceil(math.log2(len(tiers))) / S
    bpe = mean_payload + scale_term + factor_term + tier_term + rot_bits
    return M_hat, bpe


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
        Only used by lowrank_rtn_channel, lowrank_waterfill_channel,
        lowrank_eigwaterfill_channel, and lowrank_randwaterfill_channel; ignored
        by all other arms.
    tiers : tuple[int, ...]
        Allowed bit-widths for per-channel allocation in lowrank_waterfill_channel,
        lowrank_eigwaterfill_channel, and lowrank_randwaterfill_channel.
        Ignored by all other arms.
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

    Returns
    -------
    M_hat : torch.Tensor
        Dequantized approximation, same shape as M.
    bpe : float
        Honest bits-per-entry including ALL metadata.
    """
    assert arm in CACHE_ARMS, f"unknown arm {arm!r}; available: {CACHE_ARMS}"

    if arm == "rtn_token":
        return _rtn_token(M, bits, group)
    elif arm == "rtn_channel":
        return _rtn_channel(M, bits, group)
    elif arm == "rotate_rtn_token":
        return _rotate_rtn_token(M, bits, group, seed)
    elif arm == "turboquant_mse":
        return _turboquant_mse(M, bits, seed)
    elif arm == "turboquant_prod":
        return _turboquant_prod(M, bits, seed)
    elif arm == "lowrank_rtn_channel":
        return _lowrank_rtn_channel(M, bits, group, rank, svd_factors=svd_factors)
    elif arm == "lowrank_waterfill_channel":
        return _lowrank_waterfill_channel(
            M, float(bits), group, rank, tiers=tiers, svd_factors=svd_factors
        )
    elif arm == "lowrank_eigwaterfill_channel":
        return _lowrank_rotwaterfill_channel(
            M,
            float(bits),
            group,
            rank,
            tiers=tiers,
            rotation="klt",
            charge_rotation=charge_rotation,
            svd_factors=svd_factors,
        )
    elif arm == "lowrank_randwaterfill_channel":
        return _lowrank_rotwaterfill_channel(
            M,
            float(bits),
            group,
            rank,
            tiers=tiers,
            rotation="random",
            seed=seed,
            svd_factors=svd_factors,
        )
    elif arm == "lowrank_topkwaterfill_channel":
        return _lowrank_rotwaterfill_channel(
            M,
            float(bits),
            group,
            rank,
            tiers=tiers,
            rotation="topk",
            topk_k=topk_k,
            charge_rotation=charge_rotation,
            svd_factors=svd_factors,
        )
    elif arm == "lowrank_blockdiagwaterfill_channel":
        return _lowrank_rotwaterfill_channel(
            M,
            float(bits),
            group,
            rank,
            tiers=tiers,
            rotation="blockdiag",
            h_kv=h_kv,
            charge_rotation=charge_rotation,
            svd_factors=svd_factors,
        )
    elif arm == "lowrank_frozenwaterfill_channel":
        return _lowrank_rotwaterfill_channel(
            M,
            float(bits),
            group,
            rank,
            tiers=tiers,
            rotation="frozen",
            prefill_fit_len=prefill_fit_len,
            charge_rotation=charge_rotation,
            svd_factors=svd_factors,
        )
    else:  # lowrank_oraclewaterfill_channel — guarded by the CACHE_ARMS assert above
        return _lowrank_rotwaterfill_channel(
            M,
            float(bits),
            group,
            rank,
            tiers=tiers,
            rotation="oracle",
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
    )
    return from_matrix(M_hat, h), bpe
