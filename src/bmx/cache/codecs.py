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
        Q = random_orthogonal(C, seed, dtype=M.dtype)
        return M @ Q.T


def _unrotate(M_rot: torch.Tensor, seed: int) -> torch.Tensor:
    """Inverse rotation (Hadamard is self-inverse up to same signs; QR needs Q)."""
    C = M_rot.shape[-1]
    if is_power_of_2(C):
        # randomized_hadamard is H @ diag(signs) @ x; inverse is diag(signs) @ H @ x
        # but H^{-1} = H (orthonormal), so: signs * H @ M_rot
        g = torch.Generator().manual_seed(seed)
        signs = (torch.randint(0, 2, (C,), generator=g) * 2 - 1).to(M_rot.dtype)
        from bmx.quant.hadamard import fwht

        return fwht(M_rot) * signs
    else:
        Q = random_orthogonal(C, seed, dtype=M_rot.dtype)
        return M_rot @ Q  # Q^T inverse is Q (orthogonal)


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
    cb = gaussian_codebook(bits)
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
    G = _qjl_sketch(C, seed).to(R.dtype)

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
        Only used by lowrank_rtn_channel; ignored by all other arms.

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
    else:  # lowrank_rtn_channel — guarded by the CACHE_ARMS assert above
        return _lowrank_rtn_channel(M, bits, group, rank, svd_factors=svd_factors)
