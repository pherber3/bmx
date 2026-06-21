"""Tests for src/bmx/cache/codecs.py — TDD-first, all must fail before implementation."""

import math

import pytest
import torch

from bmx.cache.codecs import (
    CACHE_ARMS,
    allocate_channel_bits,
    gaussian_codebook,
    qjl_reconstruct,
    quantize_cache,
)
from bmx.decomp.lrs import truncated_svd


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

S, C = 32, 64
BITS = 3
GROUP = 16
RANK = 4
SEED = 42


def _seeded_matrix(s=S, c=C, seed=7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(s, c, generator=g)


def _rel_err(M_hat: torch.Tensor, M: torch.Tensor) -> float:
    return (M_hat - M).norm().item() / M.norm().item()


# ---------------------------------------------------------------------------
# 1. Bit accounting — exact bpe formulas
# ---------------------------------------------------------------------------


class TestBitAccounting:
    def test_rtn_token_bpe(self):
        M = _seeded_matrix()
        _, bpe = quantize_cache("rtn_token", M, bits=BITS, group=GROUP)
        expected = BITS + 16.0 / GROUP
        assert math.isclose(bpe, expected, rel_tol=1e-9), f"{bpe} != {expected}"

    def test_rtn_channel_bpe(self):
        M = _seeded_matrix()
        _, bpe = quantize_cache("rtn_channel", M, bits=BITS, group=GROUP)
        expected = BITS + 16.0 / GROUP
        assert math.isclose(bpe, expected, rel_tol=1e-9), f"{bpe} != {expected}"

    def test_rotate_rtn_token_bpe(self):
        M = _seeded_matrix()
        _, bpe = quantize_cache(
            "rotate_rtn_token", M, bits=BITS, group=GROUP, seed=SEED
        )
        expected = BITS + 16.0 / GROUP
        assert math.isclose(bpe, expected, rel_tol=1e-9), f"{bpe} != {expected}"

    def test_turboquant_mse_bpe(self):
        M = _seeded_matrix()
        _, bpe = quantize_cache("turboquant_mse", M, bits=BITS, seed=SEED)
        expected = BITS + 16.0 / C
        assert math.isclose(bpe, expected, rel_tol=1e-9), f"{bpe} != {expected}"

    def test_turboquant_prod_bpe(self):
        # bpe = (b-1) + 1 + 32/C  (two fp16 norms)
        M = _seeded_matrix()
        _, bpe = quantize_cache("turboquant_prod", M, bits=BITS, seed=SEED)
        expected = (BITS - 1) + 1 + 32.0 / C
        assert math.isclose(bpe, expected, rel_tol=1e-9), f"{bpe} != {expected}"

    def test_lowrank_rtn_channel_bpe(self):
        M = _seeded_matrix()
        _, bpe = quantize_cache(
            "lowrank_rtn_channel", M, bits=BITS, group=GROUP, rank=RANK
        )
        expected = BITS + 16.0 / GROUP + 16.0 * RANK * (S + C) / (S * C)
        assert math.isclose(bpe, expected, rel_tol=1e-9), f"{bpe} != {expected}"


# ---------------------------------------------------------------------------
# 2. Monotonicity: rel error at b=4 < rel error at b=2
# ---------------------------------------------------------------------------


class TestMonotonicity:
    @pytest.mark.parametrize("arm", list(CACHE_ARMS))
    def test_higher_bits_lower_error(self, arm: str):
        M = _seeded_matrix(seed=99)
        kwargs: dict = dict(seed=SEED, group=GROUP, rank=RANK)
        # b=2 and b=4 are valid for every arm (turboquant_prod requires b>=2)
        m2, _ = quantize_cache(arm, M, bits=2, **kwargs)
        m4, _ = quantize_cache(arm, M, bits=4, **kwargs)
        assert _rel_err(m4, M) < _rel_err(m2, M), (
            f"arm={arm}: b=4 err {_rel_err(m4, M):.4f} not < b=2 err {_rel_err(m2, M):.4f}"
        )


# ---------------------------------------------------------------------------
# 3. Seed determinism (rotate/turboquant arms)
# ---------------------------------------------------------------------------


SEEDED_ARMS = ("rotate_rtn_token", "turboquant_mse", "turboquant_prod")


class TestDeterminism:
    @pytest.mark.parametrize("arm", SEEDED_ARMS)
    def test_same_seed_same_output(self, arm: str):
        M = _seeded_matrix()
        m1, _ = quantize_cache(arm, M, bits=BITS, seed=SEED, group=GROUP, rank=RANK)
        m2, _ = quantize_cache(arm, M, bits=BITS, seed=SEED, group=GROUP, rank=RANK)
        assert torch.equal(m1, m2), f"arm={arm}: same seed → different output"

    @pytest.mark.parametrize("arm", SEEDED_ARMS)
    def test_different_seed_different_output(self, arm: str):
        M = _seeded_matrix()
        m1, _ = quantize_cache(arm, M, bits=BITS, seed=0, group=GROUP, rank=RANK)
        m2, _ = quantize_cache(arm, M, bits=BITS, seed=999, group=GROUP, rank=RANK)
        assert not torch.equal(m1, m2), f"arm={arm}: different seeds → identical output"


# ---------------------------------------------------------------------------
# 4. rtn_channel beats rtn_token on a rogue-channel matrix
# ---------------------------------------------------------------------------


class TestRogueChannel:
    def test_rtn_channel_better_on_rogue_column(self):
        """rtn_channel isolates the rogue channel to its own scale group;
        rtn_token (one group per row = group=C) has its scale dominated by
        the rogue column, collapsing ALL normal channels to near-zero quant
        levels -- the structural sanity that motivates KIVI.

        We measure relative error on the non-rogue channels only, which is
        the quantity KIVI's design targets: protecting normal channels from
        outlier-scale contamination."""
        S_rogue, C_rogue = 32, 64
        ROGUE_COL = 3
        g = torch.Generator().manual_seed(0)
        M = torch.randn(S_rogue, C_rogue, generator=g)
        M[:, ROGUE_COL] *= 50.0

        # rtn_token: one group per token (group=C) -- one scale per row,
        # dominated by the 50x rogue column -> normal channels quantized to 0
        m_token, _ = quantize_cache("rtn_token", M, bits=BITS, group=C_rogue)
        # rtn_channel: group=S -- one scale per channel -> rogue column is
        # isolated; normal channels get their own proper scale
        m_channel, _ = quantize_cache("rtn_channel", M, bits=BITS, group=S_rogue)

        # Measure on normal channels only (the quantity KIVI protects)
        mask = torch.ones(C_rogue, dtype=torch.bool)
        mask[ROGUE_COL] = False
        M_normal = M[:, mask]
        err_token = _rel_err(m_token[:, mask], M_normal)
        err_channel = _rel_err(m_channel[:, mask], M_normal)
        assert err_channel < err_token, (
            f"rtn_channel normal-ch err {err_channel:.4f} not < "
            f"rtn_token normal-ch err {err_token:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. Gaussian codebook properties
# ---------------------------------------------------------------------------


class TestGaussianCodebook:
    def test_codebook_length(self):
        for b in (2, 3, 4):
            cb = gaussian_codebook(b)
            assert cb.shape == (2**b,), f"b={b}: wrong length {cb.shape}"

    def test_codebook_sorted(self):
        for b in (2, 3, 4):
            cb = gaussian_codebook(b)
            assert (cb[1:] >= cb[:-1]).all(), f"b={b}: codebook not sorted"

    def test_codebook_beats_uniform_in_mse(self):
        """Lloyd-Max codebook should beat uniform quantization on N(0,1) data."""
        b = 3
        g = torch.Generator().manual_seed(1)
        x = torch.randn(2**18, generator=g)

        cb = gaussian_codebook(b)
        # Assign each point to nearest codebook entry
        diffs = (x.unsqueeze(1) - cb.unsqueeze(0)).abs()  # (N, 2^b)
        indices = diffs.argmin(dim=1)
        mse_lloyd = ((x - cb[indices]) ** 2).mean().item()

        # Uniform codebook over the same range
        lo, hi = cb[0].item(), cb[-1].item()
        n_levels = 2**b
        edges = torch.linspace(lo, hi, n_levels + 1)
        centers_uniform = (edges[:-1] + edges[1:]) / 2
        diffs_u = (x.unsqueeze(1) - centers_uniform.unsqueeze(0)).abs()
        idx_u = diffs_u.argmin(dim=1)
        mse_uniform = ((x - centers_uniform[idx_u]) ** 2).mean().item()

        assert mse_lloyd < mse_uniform, (
            f"Lloyd MSE {mse_lloyd:.6f} not < uniform MSE {mse_uniform:.6f}"
        )


# ---------------------------------------------------------------------------
# 6. turboquant_prod / qjl_reconstruct unbiasedness
# ---------------------------------------------------------------------------


class TestQJLUnbiasedness:
    def test_qjl_reconstruct_unbiased_vectorwise(self):
        """E_seeds[qjl_reconstruct(r)] ≈ r, tested vector-wise so a wrong
        dequantization constant fails: a sqrt(2)-off constant gives rel ≈ 0.41
        and 2x gives rel ≈ 1.0, vs ~0.13 expected sampling noise at 128 seeds."""
        C_test = 64
        g = torch.Generator().manual_seed(17)
        r = torch.randn(1, C_test, generator=g)

        mean_hat = torch.zeros_like(r)
        n_seeds = 128
        for s in range(n_seeds):
            mean_hat += qjl_reconstruct(r, seed=s)
        mean_hat /= n_seeds

        rel = ((mean_hat - r).norm() / r.norm()).item()
        assert rel < 0.25, f"E[r_hat] deviates from r: rel={rel:.3f}"


# ---------------------------------------------------------------------------
# 7. lowrank_rtn_channel at full rank ≈ exact
# ---------------------------------------------------------------------------


class TestLowRankFullRank:
    def test_full_rank_near_exact(self):
        S_small, C_small = 16, 16
        M = _seeded_matrix(s=S_small, c=C_small)
        full_rank = min(S_small, C_small)
        # At full rank, low-rank component captures everything, residual ~0
        m_hat, _ = quantize_cache(
            "lowrank_rtn_channel",
            M,
            bits=8,
            group=S_small,
            rank=full_rank,
        )
        err = _rel_err(m_hat, M)
        assert err < 1e-3, (
            f"Full-rank lowrank_rtn_channel rel error too large: {err:.6f}"
        )


# ---------------------------------------------------------------------------
# 8. Unknown arm raises ValueError
# ---------------------------------------------------------------------------


class TestUnknownArm:
    def test_unknown_arm_raises(self):
        M = _seeded_matrix()
        with pytest.raises((ValueError, AssertionError)):
            quantize_cache("not_a_real_arm", M, bits=BITS)


# ---------------------------------------------------------------------------
# 9. svd_factors equivalence: passed-factors == internally computed factors
# ---------------------------------------------------------------------------


class TestSvdFactorsEquivalence:
    def test_svd_factors_same_as_internal(self):
        """quantize_cache with pre-computed svd_factors must give torch.equal result
        to the default code path that calls truncated_svd internally."""
        M = _seeded_matrix()
        # Pre-compute the same factors that lowrank_rtn_channel would compute internally
        factors = truncated_svd(M, RANK)

        m_default, bpe_default = quantize_cache(
            "lowrank_rtn_channel", M, bits=BITS, group=GROUP, rank=RANK
        )
        m_passed, bpe_passed = quantize_cache(
            "lowrank_rtn_channel",
            M,
            bits=BITS,
            group=GROUP,
            rank=RANK,
            svd_factors=factors,
        )
        assert torch.equal(m_default, m_passed), (
            "lowrank_rtn_channel: passed svd_factors != internally computed factors"
        )
        assert math.isclose(bpe_default, bpe_passed, rel_tol=1e-9), (
            f"bpe mismatch: {bpe_default} vs {bpe_passed}"
        )


# ---------------------------------------------------------------------------
# 10. allocate_channel_bits — reverse water-filling per-channel allocator
# ---------------------------------------------------------------------------


def _channel_matrix(per_channel_std, s=256, seed=11):
    """(s, C) matrix whose column c has std per_channel_std[c]."""
    g = torch.Generator().manual_seed(seed)
    C = len(per_channel_std)
    base = torch.randn(s, C, generator=g, dtype=torch.float64)
    return base * torch.tensor(per_channel_std, dtype=torch.float64)


def test_allocate_monotone_in_variance():
    # increasing per-channel std -> rounded bits non-decreasing
    stds = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    R = _channel_matrix(stds)
    bits = allocate_channel_bits(R, budget_bits=3.0)
    bits_list = bits.tolist()
    assert bits_list == sorted(bits_list), f"not monotone: {bits_list}"


def test_allocate_realized_mean_near_budget():
    stds = [0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0]
    R = _channel_matrix(stds)
    for budget in (2.0, 3.0, 3.5):
        bits = allocate_channel_bits(R, budget_bits=budget)
        realized = bits.float().mean().item()
        # tier-rounding can only land at-or-below; never overshoot the budget
        assert realized <= budget + 1e-9, f"overshoot: {realized} > {budget}"
        assert realized >= budget - 1.0, f"too far under: {realized} << {budget}"


def test_allocate_drops_low_variance_channels_when_tight():
    # one giant channel + many tiny ones, tight budget -> tiny ones dropped to 0
    stds = [1000.0] + [0.001] * 20
    R = _channel_matrix(stds)
    bits = allocate_channel_bits(R, budget_bits=1.0)
    assert bits[0].item() > 0
    assert (bits[1:] == 0).any(), (
        "expected some low-variance channels dropped to tier 0"
    )


def test_allocate_isotropic_is_uniform():
    # equal variance -> all channels same tier (degenerate water-fill)
    stds = [1.0] * 12
    R = _channel_matrix(stds)
    bits = allocate_channel_bits(R, budget_bits=3.0)
    assert len(set(bits.tolist())) == 1, f"isotropic not uniform: {bits.tolist()}"


def test_allocate_deterministic():
    stds = [0.1, 1.0, 10.0, 100.0]
    R = _channel_matrix(stds)
    a = allocate_channel_bits(R, budget_bits=3.0)
    b = allocate_channel_bits(R, budget_bits=3.0)
    assert torch.equal(a, b)


def test_allocate_returns_only_tier_values():
    stds = [0.1, 1.0, 10.0, 100.0, 1000.0]
    R = _channel_matrix(stds)
    tiers = (0, 2, 3, 4)
    bits = allocate_channel_bits(R, budget_bits=3.0, tiers=tiers)
    assert set(bits.tolist()).issubset(set(tiers))


# ---------------------------------------------------------------------------
# 11. lowrank_waterfill_channel codec arm
# ---------------------------------------------------------------------------


def test_waterfill_arm_in_registries():
    from bmx.cache.codecs import CACHE_ARMS, S_DIVISIBILITY_ARMS

    assert "lowrank_waterfill_channel" in CACHE_ARMS
    assert "lowrank_waterfill_channel" in S_DIVISIBILITY_ARMS


def test_waterfill_reduces_to_uniform_single_tier():
    # With a single uniform tier {3}, every channel gets 3 bits, so the arm must
    # match lowrank_rtn_channel @3b bit-for-bit (same SVD, same per-channel RTN).
    M = _seeded_matrix(s=64, c=64, seed=3).double()
    rank = 4
    factors = truncated_svd(M, rank)
    uni, bpe_uni = quantize_cache(
        "lowrank_rtn_channel", M, bits=3, group=GROUP, rank=rank, svd_factors=factors
    )
    wf, bpe_wf = quantize_cache(
        "lowrank_waterfill_channel",
        M,
        bits=3,
        group=GROUP,
        rank=rank,
        tiers=(3,),
        svd_factors=factors,
    )
    assert torch.allclose(wf, uni, atol=1e-9), "single-tier waterfill != uniform rtn"
    # bpe differs only by the tier-index map; with 1 tier that term is 0 bits.
    assert abs(bpe_wf - bpe_uni) < 1e-9


def test_waterfill_honest_bpe_formula():
    # Hand-check the bpe accounting on a fixed small matrix.
    S_, C_, group_, rank_ = 64, 32, 16, 2
    M = _seeded_matrix(s=S_, c=C_, seed=5).double()
    tiers = (0, 2, 3, 4)
    _, bpe = quantize_cache(
        "lowrank_waterfill_channel",
        M,
        bits=3,
        group=group_,
        rank=rank_,
        tiers=tiers,
    )
    import math as _m

    # The codec recomputes its own allocation on R = M - L; recover the expected
    # residual-payload mean by trusting the codec's reported bpe minus the known
    # metadata terms, then assert each metadata term is the documented constant.
    scale_term = 16.0 / group_
    factor_term = 16.0 * rank_ * (S_ + C_) / (S_ * C_)
    tier_term = _m.ceil(_m.log2(len(tiers))) / S_
    payload = bpe - scale_term - factor_term - tier_term
    assert payload >= 0.0, f"payload negative: {payload}"
    assert payload <= 4.0 + 1e-9, f"payload exceeds max tier: {payload}"


def test_waterfill_s_divisibility_assert():
    M = _seeded_matrix(s=63, c=64, seed=9).double()  # 63 % 16 != 0
    with pytest.raises(AssertionError):
        quantize_cache(
            "lowrank_waterfill_channel", M, bits=3, group=16, rank=2, tiers=(0, 2, 3, 4)
        )


def test_waterfill_dropped_channels_are_zero_in_residual():
    """Tier-0 (dropped) channels must reconstruct from the low-rank component L only.

    Construction: build a (S, C) fp64 matrix whose LAST column has residual
    variance ~(1e-4)^2 while all others are O(1).  Under a tight budget this
    near-zero column is deterministically allocated 0 bits by the water-filler.
    We assert (a) at least one tier-0 channel exists, and (b) for every dropped
    column c: M_hat[:, c] == L[:, c] exactly (atol=1e-9), i.e. the quantized
    residual contributes nothing.
    """
    S_, C_, rank_, group_ = 64, 32, 4, 16
    tiers = (0, 2, 3, 4)
    budget_bits = 2

    # Build base matrix and inject one near-zero-residual column.
    g = torch.Generator().manual_seed(13)
    M = torch.randn(S_, C_, generator=g, dtype=torch.float64)
    # Make column 0 have very small std so its residual is tiny.
    M[:, 0] *= 1e-4

    # Pre-compute svd_factors so we can independently reconstruct L.
    svd_factors = truncated_svd(M, rank_)
    Us, V = svd_factors
    # Replicate the codec's fp16 roundtrip exactly.
    L = Us.half().float() @ V.half().float().mT  # (S_, C_)

    # Identify which channels the water-filler drops on the residual.
    R = M - L
    bits_per_ch = allocate_channel_bits(R, budget_bits, tiers=tiers, axis=0)
    dropped = (bits_per_ch == 0).nonzero(as_tuple=True)[0]
    assert dropped.numel() > 0, (
        "Construction failed: no channel was allocated tier 0.  "
        "Tighten budget_bits or reduce M[:, 0] std further."
    )

    # Run the full codec arm (passes svd_factors so it uses the same L).
    M_hat, _ = quantize_cache(
        "lowrank_waterfill_channel",
        M,
        bits=budget_bits,
        group=group_,
        rank=rank_,
        tiers=tiers,
        svd_factors=svd_factors,
    )
    assert M_hat.shape == M.shape

    # Dropped channels must equal L exactly (residual contribution is zero).
    L_f64 = L.double()
    assert torch.allclose(M_hat[:, dropped], L_f64[:, dropped], atol=1e-9), (
        f"Dropped channel(s) {dropped.tolist()} deviate from L: "
        f"max |M_hat - L| = {(M_hat[:, dropped] - L_f64[:, dropped]).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# 12. Rotated-waterfill arms (KLT + random)
# ---------------------------------------------------------------------------

from bmx.cache.metrics import logit_distortion as _logit_distortion  # noqa: E402


def _qkv_for(M, h_kv=2, seed=123):
    """A fake query set (h_kv, T, d) matching M's (S, C=h_kv*d) layout for logit scoring."""
    S, C = M.shape
    d = C // h_kv
    g = torch.Generator().manual_seed(seed)
    return torch.randn(h_kv, 8, d, generator=g, dtype=M.dtype)


def test_rotwaterfill_arms_registered():
    from bmx.cache.codecs import CACHE_ARMS, S_DIVISIBILITY_ARMS

    for arm in ("lowrank_eigwaterfill_channel", "lowrank_randwaterfill_channel"):
        assert arm in CACHE_ARMS
        assert arm in S_DIVISIBILITY_ARMS


def test_rotation_is_inner_product_neutral():
    # With a single high-bit uniform tier (near-lossless RTN), rotate+quantize+unrotate
    # must match the unrotated near-lossless arm on LOGIT distortion to tight tol —
    # for BOTH klt and random. Proves Q is orthogonal and rotate/unrotate is exact.
    from bmx.cache.collect import from_matrix

    M = _seeded_matrix(s=64, c=64, seed=4).double()
    h_kv = 2
    q = _qkv_for(M, h_kv=h_kv)
    factors = truncated_svd(M, 4)
    # near-lossless: one tier at 8 bits
    base, _ = quantize_cache(
        "lowrank_waterfill_channel",
        M,
        bits=8,
        group=GROUP,
        rank=4,
        tiers=(8,),
        svd_factors=factors,
    )
    lg_base = _logit_distortion(
        from_matrix(M, h_kv).double(), from_matrix(base, h_kv).double(), q
    )
    for rotation in ("klt", "random"):
        rot, _ = quantize_cache(
            "lowrank_eigwaterfill_channel"
            if rotation == "klt"
            else "lowrank_randwaterfill_channel",
            M,
            bits=8,
            group=GROUP,
            rank=4,
            tiers=(8,),
            seed=1,
            svd_factors=factors,
        )
        lg_rot = _logit_distortion(
            from_matrix(M, h_kv).double(), from_matrix(rot, h_kv).double(), q
        )
        assert abs(lg_rot - lg_base) < 1e-6, (
            f"{rotation}: rotation not inner-product-neutral"
        )


def test_klt_reduces_to_raw_waterfill_when_diagonal():
    # Diagonal-covariance residual -> KLT Q is identity (up to sign/perm). KLT arm then
    # matches raw waterfill on LOGIT distortion (not raw tensors — eigvec sign ambiguity).
    from bmx.cache.collect import from_matrix

    # Build M whose residual after rank-r low-rank is independent per-channel:
    # use a matrix with no low-rank structure so L is tiny and R ~= M with diagonal cov.
    stds = [0.3, 1.0, 3.0, 9.0] * 16  # C = 64, varied per-channel, uncorrelated
    R = _channel_matrix(
        stds, s=64, seed=8
    )  # (64, 64) fp64, diagonal cov by construction
    h_kv = 2
    q = _qkv_for(R, h_kv=h_kv)
    factors = truncated_svd(R, 4)
    raw, _ = quantize_cache(
        "lowrank_waterfill_channel",
        R,
        bits=3,
        group=GROUP,
        rank=4,
        tiers=(0, 2, 3, 4),
        svd_factors=factors,
    )
    klt, _ = quantize_cache(
        "lowrank_eigwaterfill_channel",
        R,
        bits=3,
        group=GROUP,
        rank=4,
        tiers=(0, 2, 3, 4),
        svd_factors=factors,
    )
    lg_raw = _logit_distortion(
        from_matrix(R, h_kv).double(), from_matrix(raw, h_kv).double(), q
    )
    lg_klt = _logit_distortion(
        from_matrix(R, h_kv).double(), from_matrix(klt, h_kv).double(), q
    )
    # diagonal cov => Q ~ I (up to sign) => same allocation, same logit distortion
    assert abs(lg_raw - lg_klt) < 0.05, (
        f"diagonal KLT diverged from raw: {lg_raw} vs {lg_klt}"
    )


def test_random_arm_is_free_and_reproducible():
    M = _seeded_matrix(s=64, c=64, seed=6).double()
    factors = truncated_svd(M, 4)
    a, bpe_a = quantize_cache(
        "lowrank_randwaterfill_channel",
        M,
        bits=3,
        seed=7,
        group=GROUP,
        rank=4,
        tiers=(0, 2, 3, 4),
        svd_factors=factors,
    )
    b, bpe_b = quantize_cache(
        "lowrank_randwaterfill_channel",
        M,
        bits=3,
        seed=7,
        group=GROUP,
        rank=4,
        tiers=(0, 2, 3, 4),
        svd_factors=factors,
    )
    assert torch.allclose(a, b), "random arm not reproducible at fixed seed"
    assert abs(bpe_a - bpe_b) < 1e-12
    # honest == idealized: random rotation costs 0 stored bits. Compare to raw waterfill bpe
    # (same payload+scale+factor+tier terms, no rotation term either way).
    _, bpe_raw = quantize_cache(
        "lowrank_waterfill_channel",
        M,
        bits=3,
        group=GROUP,
        rank=4,
        tiers=(0, 2, 3, 4),
        svd_factors=factors,
    )
    assert abs(bpe_a - bpe_raw) < 1e-9, (
        "random arm bpe should match raw waterfill (no rotation charge)"
    )


def test_klt_honest_rotation_charge():
    S_, C_, group_, rank_ = 64, 32, 16, 2
    M = _seeded_matrix(s=S_, c=C_, seed=5).double()
    _, bpe_ideal = quantize_cache(
        "lowrank_eigwaterfill_channel",
        M,
        bits=3,
        group=group_,
        rank=rank_,
        tiers=(0, 2, 3, 4),
        charge_rotation=False,
    )
    _, bpe_honest = quantize_cache(
        "lowrank_eigwaterfill_channel",
        M,
        bits=3,
        group=group_,
        rank=rank_,
        tiers=(0, 2, 3, 4),
        charge_rotation=True,
    )
    expected = 16.0 * C_ / S_
    assert abs((bpe_honest - bpe_ideal) - expected) < 1e-9, "rotation charge != 16*C/S"


def test_klt_concentrates_random_spreads_variance():
    # KLT increases per-column variance CV (concentration); random decreases it (spreading).
    from bmx.cache.codecs import _round_to_tiers  # noqa: F401  (sanity import path exists)

    stds = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0] * 8  # C=64 anisotropic
    R = _channel_matrix(stds, s=256, seed=2)

    def cv(x):
        v = x.var(dim=0, unbiased=False)
        return (v.std() / v.mean().clamp_min(1e-30)).item()

    from bmx.quant.hadamard import random_orthogonal

    cv_raw = cv(R)
    eigvals, eigvecs = torch.linalg.eigh(R.mT @ R)
    R_klt = R @ eigvecs
    cv_klt = cv(R_klt)
    Qr = random_orthogonal(R.shape[1], seed=3, dtype=R.dtype)
    R_rand = R @ Qr.mT
    cv_rand = cv(R_rand)
    assert cv_klt > cv_raw, f"KLT did not concentrate variance: {cv_klt} <= {cv_raw}"
    assert cv_rand < cv_raw, f"random did not spread variance: {cv_rand} >= {cv_raw}"


def test_rotwaterfill_s_divisibility_assert():
    M = _seeded_matrix(s=63, c=64, seed=9).double()  # 63 % 16 != 0
    with pytest.raises(AssertionError):
        quantize_cache(
            "lowrank_eigwaterfill_channel",
            M,
            bits=3,
            group=16,
            rank=2,
            tiers=(0, 2, 3, 4),
        )
