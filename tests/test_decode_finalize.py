"""CPU-testable pure functions from the decode finalize/merge path.

Desk review: docs/2026-07-04-triton-decode-desk-review.md.

F2 replaced two per-call `.to(device)` H2D copies (gaussian_codebook, Hadamard
signs) with device-cached getters, mirroring the pre-existing `_hadamard_matrix`
pattern.
"""

import torch

from bmx.cache.codecs import _hadamard_signs, gaussian_codebook
from bmx.cache.triton_dequant_attention import _codebook_dev, _signs_dev


# ---------------------------------------------------------------------------
# F2 — device-cached constants return the SAME object on repeated calls
# ---------------------------------------------------------------------------


def test_codebook_dev_cached_identity():
    """Same (bits, device) args -> the identical tensor object (no re-copy)."""
    a = _codebook_dev(4, "cpu")
    b = _codebook_dev(4, "cpu")
    assert a is b
    assert torch.equal(a, gaussian_codebook(4).to("cpu", torch.float32))


def test_codebook_dev_matches_uncached_construction():
    for bits in (2, 3, 4):
        cached = _codebook_dev(bits, "cpu")
        uncached = gaussian_codebook(bits).to("cpu", torch.float32)
        assert torch.equal(cached, uncached)
        assert cached.dtype == torch.float32


def test_signs_dev_cached_identity():
    a = _signs_dev(32, 7, "cpu")
    b = _signs_dev(32, 7, "cpu")
    assert a is b
    assert torch.equal(a, _hadamard_signs(32, 7).to("cpu", torch.float32))


def test_signs_dev_distinguishes_seed_and_d():
    base = _signs_dev(32, 7, "cpu")
    diff_seed = _signs_dev(32, 8, "cpu")
    diff_d = _signs_dev(16, 7, "cpu")
    assert not torch.equal(base, diff_seed)
    assert base.shape != diff_d.shape
