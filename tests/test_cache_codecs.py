"""Tests for src/bmx/cache/codecs.py — TDD-first, all must fail before implementation."""

import math

import pytest
import torch

from bmx.cache.codecs import (
    CACHE_ARMS,
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
        if arm == "turboquant_prod":
            # b>=2 required; b=2 and b=4 are both valid
            m2, _ = quantize_cache(arm, M, bits=2, **kwargs)
            m4, _ = quantize_cache(arm, M, bits=4, **kwargs)
        else:
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
