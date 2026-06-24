"""De-risk tests: CPU bit-for-bit verification of the butterfly-FWHT reference.

PURPOSE
-------
Prove the FWHT/unrotate ALGORITHM is correct BEFORE porting it to Triton.
All tests run on CPU with fp64 for maximum strictness (torch.equal or max_abs < 1e-12).

A loose tolerance would defeat the de-risk — any drift here is a bug in the
algorithm, not a hardware/dtype rounding artefact.

WHAT IS VERIFIED
----------------
1. fwht_butterfly_ref == hadamard.fwht  (bit-for-bit, fp64, several dims & shapes)
2. unrotate_ref == codecs._unrotate  (power-of-2 C, fp64, several seeds)
3. Full V-dequant path: cb[indices]/sqrt(C) -> unrotate_ref -> *norms
   == _turboquant_mse_dequant  (bit-for-bit, fp64)

Once these pass, the VM task is ONLY to port fwht_butterfly_ref / unrotate_ref
to tl.ops in the k2b decode kernel — any drift on the VM is a Triton translation
bug, not an algorithm bug.

SCOPE NOTE (non-power-of-2)
----------------------------
The non-power-of-2 _unrotate path (random_orthogonal matmul) is NOT covered
here because k2b head dims are d=128 and C = h_kv * d is always power-of-2 for
the target models:
  - LLaMA-2-7B : h_kv=32, d=128, C=4096=2^12
  - LLaMA-3-8B : h_kv=8,  d=128, C=1024=2^10
  - GPT-2 test  : h_kv=2,  d=8,   C=16=2^4
All fall on the FWHT path — the path this module verifies.
"""

import math

import pytest
import torch

from bmx.cache.codecs import _turboquant_mse_dequant, _unrotate, gaussian_codebook
from bmx.cache.hadamard_kernel_ref import fwht_butterfly_ref, unrotate_ref
from bmx.quant.hadamard import fwht


# ---------------------------------------------------------------------------
# Test 1: fwht_butterfly_ref matches hadamard.fwht  (bit-for-bit, fp64)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [16, 32, 64, 128])
@pytest.mark.parametrize("n_rows", [1, 4, 8])
def test_fwht_butterfly_ref_matches_fwht(d, n_rows):
    """fwht_butterfly_ref(x) == fwht(x) bit-for-bit across dims and batch sizes."""
    torch.manual_seed(d + n_rows * 1000)
    x = torch.randn(n_rows, d, dtype=torch.float64)
    ref = fwht(x)
    got = fwht_butterfly_ref(x)
    assert got.shape == ref.shape, f"shape mismatch: {got.shape} != {ref.shape}"
    max_abs = (got - ref).abs().max().item()
    assert torch.equal(got, ref), (
        f"fwht_butterfly_ref != fwht for d={d}, n_rows={n_rows}. max_abs={max_abs:.2e}"
    )


@pytest.mark.parametrize("d", [16, 32, 64, 128])
def test_fwht_butterfly_ref_matches_fwht_seeds(d):
    """Bit-exact match across 5 random seeds for each dimension."""
    for seed in range(5):
        torch.manual_seed(seed)
        x = torch.randn(3, d, dtype=torch.float64)
        ref = fwht(x)
        got = fwht_butterfly_ref(x)
        assert torch.equal(got, ref), (
            f"fwht_butterfly_ref != fwht for d={d}, seed={seed}. "
            f"max_abs={(got - ref).abs().max().item():.2e}"
        )


def test_fwht_butterfly_ref_3d_input():
    """fwht_butterfly_ref handles 3-D inputs (leading dims flattened correctly)."""
    torch.manual_seed(99)
    x = torch.randn(2, 3, 32, dtype=torch.float64)
    ref = fwht(x)
    got = fwht_butterfly_ref(x)
    assert got.shape == ref.shape
    assert torch.equal(got, ref), (
        f"3-D shape mismatch. max_abs={(got - ref).abs().max().item():.2e}"
    )


def test_fwht_butterfly_ref_orthonormality():
    """fwht_butterfly_ref is orthonormal: H^T H = I  (rows of H are unit orthogonal)."""
    d = 32
    eye = torch.eye(d, dtype=torch.float64)
    H = fwht_butterfly_ref(eye)  # apply to identity columns -> H rows
    gram = H.T @ H
    err = (gram - eye).abs().max().item()
    assert err < 1e-12, f"H^T H - I max_abs = {err:.2e} (expected < 1e-12)"


def test_fwht_butterfly_ref_self_inverse():
    """H applied twice returns the original (up to sign flip which cancels with /sqrt(d) twice)."""
    # For orthonormal FWHT: fwht(fwht(x)) == x (since H^{-1} = H^T = H)
    torch.manual_seed(7)
    x = torch.randn(4, 64, dtype=torch.float64)
    recovered = fwht_butterfly_ref(fwht_butterfly_ref(x))
    err = (recovered - x).abs().max().item()
    assert err < 1e-11, f"fwht(fwht(x)) != x: max_abs={err:.2e}"


# ---------------------------------------------------------------------------
# Test 2: unrotate_ref matches codecs._unrotate  (power-of-2 C, fp64)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("C", [16, 32, 64, 128])
@pytest.mark.parametrize("seed", [0, 1, 42, 123])
def test_unrotate_ref_matches_unrotate(C, seed):
    """unrotate_ref == _unrotate bit-for-bit for power-of-2 C, several seeds."""
    torch.manual_seed(seed + C)
    M_rot = torch.randn(8, C, dtype=torch.float64)
    ref = _unrotate(M_rot, seed)
    got = unrotate_ref(M_rot, seed)
    assert got.shape == ref.shape
    max_abs = (got - ref).abs().max().item()
    assert torch.equal(got, ref), (
        f"unrotate_ref != _unrotate for C={C}, seed={seed}. max_abs={max_abs:.2e}"
    )


@pytest.mark.parametrize("C", [16, 32, 64, 128])
def test_unrotate_ref_is_inverse_of_rotate(C):
    """unrotate_ref undoes the Hadamard rotation: unrotate(rotate(x)) == x."""
    from bmx.cache.codecs import _rotate

    torch.manual_seed(C + 99)
    M = torch.randn(6, C, dtype=torch.float64)
    seed = 7
    M_rot = _rotate(M, seed)
    M_back = unrotate_ref(M_rot, seed)
    err = (M_back - M).abs().max().item()
    assert err < 1e-11, f"unrotate_ref(rotate(x)) != x for C={C}. max_abs={err:.2e}"


# ---------------------------------------------------------------------------
# Test 3: Full V-dequant path (the port-ready pipeline)
#   cb[indices]/sqrt(C) -> unrotate_ref -> *norms  == _turboquant_mse_dequant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("C", [16, 32, 64])
@pytest.mark.parametrize("bits", [2, 3])
@pytest.mark.parametrize("seed", [0, 5])
def test_full_v_dequant_path_matches_turboquant_mse_dequant(C, bits, seed):
    """cb[indices]/sqrt(C) -> unrotate_ref -> *norms matches _turboquant_mse_dequant.

    This is the full V-dequant pipeline the in-kernel FWHT will replace — verifying
    that unrotate_ref can substitute for _unrotate in the real codec path.
    """
    S = 12
    torch.manual_seed(seed + bits * 100 + C)

    # Build fake indices + norms matching what _turboquant_mse_packed would store
    n_levels = 2**bits
    indices = torch.randint(0, n_levels, (S, C)).to(torch.int16)
    norms = torch.rand(S, 1, dtype=torch.float64) + 0.1  # positive norms

    # Reference: the canonical codec dequant
    # Note: _turboquant_mse_dequant works in the dtype of norms (fp64 here)
    ref = _turboquant_mse_dequant(indices, norms, bits, seed, C)

    # Port-ready path: inline the sequence using unrotate_ref.
    # Codebook is kept in fp32 (matching _turboquant_mse_dequant internals — it
    # does NOT cast cb to the norms dtype; the upcast happens at *norms).
    cb = gaussian_codebook(bits)  # fp32 — matches canonical dequant exactly
    sqrt_c = math.sqrt(C)
    M_quant = cb[indices.long()] / sqrt_c  # cb gather + /sqrt(C)  (fp32)
    M_recon = unrotate_ref(M_quant, seed)  # in-kernel FWHT replaces this (fp32)
    got = M_recon * norms  # fp32 * fp64 -> fp64 (same upcast as canonical)

    assert got.shape == ref.shape, f"shape mismatch: {got.shape} != {ref.shape}"
    max_abs = (got - ref).abs().max().item()
    assert torch.equal(got, ref), (
        f"Full V-dequant path != _turboquant_mse_dequant for "
        f"C={C}, bits={bits}, seed={seed}. max_abs={max_abs:.2e}"
    )


def test_full_v_dequant_path_k2b_dims():
    """Full V-dequant with realistic k2b config: C=128 (h_kv=1, d=128), bits=2.

    This is the actual target: d=128, C=128=2^7 on a single head.
    For LLaMA-2-7B the matrix layout makes C=h_kv*d=4096=2^12; verifying
    d=128 (the per-head dim) covers the FWHT behaviour since C is always power-of-2.
    """
    C, bits, seed, S = 128, 2, 0, 64
    torch.manual_seed(42)
    n_levels = 2**bits
    indices = torch.randint(0, n_levels, (S, C)).to(torch.int16)
    norms = torch.rand(S, 1, dtype=torch.float64) + 0.1

    ref = _turboquant_mse_dequant(indices, norms, bits, seed, C)

    cb = gaussian_codebook(bits)  # fp32, matching canonical dequant internals
    M_quant = cb[indices.long()] / math.sqrt(C)
    M_recon = unrotate_ref(M_quant, seed)
    got = M_recon * norms  # fp32 * fp64 -> fp64 (same upcast as canonical)

    max_abs = (got - ref).abs().max().item()
    assert torch.equal(got, ref), f"k2b-dims V-dequant mismatch. max_abs={max_abs:.2e}"


# ---------------------------------------------------------------------------
# Sanity: non-power-of-2 raises clearly (documents the scope boundary)
# ---------------------------------------------------------------------------


def test_fwht_butterfly_ref_rejects_non_power_of_2():
    """fwht_butterfly_ref asserts on non-power-of-2 dims (scope boundary)."""
    x = torch.randn(4, 17, dtype=torch.float64)
    with pytest.raises(AssertionError):
        fwht_butterfly_ref(x)


def test_unrotate_ref_rejects_non_power_of_2():
    """unrotate_ref asserts on non-power-of-2 C (documents the scope boundary)."""
    M = torch.randn(4, 17, dtype=torch.float64)
    with pytest.raises(AssertionError):
        unrotate_ref(M, seed=0)
