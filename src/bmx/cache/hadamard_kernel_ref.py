"""CPU reference butterfly-FWHT for the in-kernel Hadamard-unrotate port.

PURPOSE — DE-RISK, NOT PRODUCTION
-----------------------------------
This module provides ``fwht_butterfly_ref`` and ``unrotate_ref``: pure-PyTorch
implementations of the fast Walsh-Hadamard transform and the turboquant_mse V
unrotate step that ARE STRUCTURALLY MIRRORED to what the Triton 3c+ in-kernel
FWHT will look like.

The implementations are intentionally written in the "port-faithful" style —
explicit log2(d) stage loop with a flat index calculation that a Triton kernel
would reproduce with tl.arange / pointer arithmetic — rather than the more
vectorized reshaping used in the production ``bmx.quant.hadamard.fwht``.

Both functions are verified bit-for-bit against the known-good production
implementations on CPU (see tests/test_hadamard_kernel_ref.py).  Once the
tests pass, the VM task is narrowed to a straight translation of
``fwht_butterfly_ref`` / ``unrotate_ref`` to tl.ops — any drift there is a
Triton-translation bug, NOT an algorithm bug.

SCOPE
-----
- Power-of-2 last-dim ONLY.  The production ``_unrotate`` also handles the
  non-power-of-2 path via ``random_orthogonal`` matrix-multiply; that path is
  NOT covered here because k2b head dims are d=128 and C = h_kv * d is always
  power-of-2 for the real target models (e.g., LLaMA-2-7B: h_kv=32, d=128,
  C=4096=2^12).
- CPU only.  These functions are for offline algorithm verification; no
  CUDA / Triton dependency.

TRITON PORT GUIDE (VM task)
----------------------------
``fwht_butterfly_ref`` maps directly to a Triton kernel as follows:

    Python loop variable:  h  (stride = block size at this stage)
    Triton equivalent:     constexpr loop  for h in range(1, d, *= 2)

    For each element index i in [0, d):
        half  = i % (2 * h)       # position within the butterfly group
        group = i // (2 * h)      # which group
        if half < h:
            lo_idx = group * 2 * h + half          # lower butterfly element
            hi_idx = group * 2 * h + half + h      # upper butterfly element
            a = x[lo_idx]; b = x[hi_idx]
            x[lo_idx] = a + b
            x[hi_idx] = a - b
    x /= sqrt(d)   # applied once after all stages

In Triton: x lives in registers (tl.load from SRAM / DRAM); the butterfly
indices are tl.arange arithmetic; the in-place update is register assignment.
"""

from __future__ import annotations

import math

import torch

from bmx.cache.codecs import _hadamard_signs


# ---------------------------------------------------------------------------
# fwht_butterfly_ref
# ---------------------------------------------------------------------------


def fwht_butterfly_ref(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal FWHT — butterfly structure mirroring an in-kernel Triton loop.

    Accepts any shape; operates on the last dimension, which must be a power of 2.
    Mirrors ``bmx.quant.hadamard.fwht`` exactly but uses flat-index arithmetic
    (the form a Triton kernel will use) instead of the reshape/stack idiom.

    Algorithm (Sylvester ordering, log2(d) stages):
        h = 1  (initial butterfly stride)
        while h < d:
            for each pair (lo, hi) where hi = lo + h:
                a, b = x[lo], x[hi]
                x[lo] = a + b
                x[hi] = a - b
            h *= 2
        x /= sqrt(d)

    The flat-index pair selection is:
        For element i in [0, d):  lo = i if (i // h) % 2 == 0  (else hi)
        Equivalently, paired as: for each group of 2h elements starting at
        group*2h, the first h are 'lo' and the next h are 'hi'.

    This is BIT-FOR-BIT identical to fwht() for all power-of-2 dims and any
    floating-point dtype (fp32, fp64, fp16 supported; tests use fp64 for
    strictest tolerance).
    """
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, (
        f"fwht_butterfly_ref requires power-of-2 last dim, got {d}"
    )
    orig_shape = x.shape
    # Flatten all leading dims so we work row-by-row (n_rows, d)
    y = x.reshape(-1, d).clone()

    # Build butterfly pair indices once — reused across all stages and rows.
    # For stride h, the lo indices are those i where (i // h) % 2 == 0.
    # Equivalently: for k in range(d // (2*h)): lo[k*h:(k+1)*h] = k*2h + arange(h)
    h = 1
    while h < d:
        # lo/hi index pair tensors for this butterfly stage (shape: (d//2,))
        # Group indices: d // (2h) groups, each contributing h lo-hi pairs.
        n_groups = d // (2 * h)
        group_idx = torch.arange(n_groups, device=y.device)  # (n_groups,)
        within = torch.arange(h, device=y.device)  # (h,)
        # lo[g, w] = g * 2h + w,  hi[g, w] = g * 2h + w + h
        lo = (group_idx[:, None] * (2 * h) + within[None, :]).reshape(-1)  # (d//2,)
        hi = lo + h

        # Gather both halves for all rows simultaneously
        a = y[:, lo]  # (n_rows, d//2)
        b = y[:, hi]  # (n_rows, d//2)
        y[:, lo] = a + b
        y[:, hi] = a - b
        h *= 2

    y = y / math.sqrt(d)
    return y.view(orig_shape)


# ---------------------------------------------------------------------------
# unrotate_ref
# ---------------------------------------------------------------------------


def unrotate_ref(M_rot: torch.Tensor, seed: int) -> torch.Tensor:
    """Inverse Hadamard rotation for the power-of-2-C path — kernel-port reference.

    Mirrors ``bmx.cache.codecs._unrotate`` power-of-2 branch exactly, using
    ``fwht_butterfly_ref`` instead of the production ``fwht``.

    The randomized Hadamard rotation applied at quantize time is:
        rotate(x) = H @ diag(signs) @ x    (where H is the orthonormal FWHT)
    Its inverse is:
        unrotate(y) = diag(signs) @ H @ y  (since H^{-1} = H for orthonormal FWHT)
    which is: ``fwht(y) * signs``

    Args:
        M_rot: (..., C) tensor — rotated input.  Last dim must be power-of-2.
        seed:  int — same seed used at quantize time.

    Returns:
        (..., C) tensor — unrotated output, same shape and dtype as M_rot.

    Non-power-of-2 path NOT covered (random_orthogonal matmul, out of scope for
    the in-kernel FWHT port; k2b dims are always power-of-2).
    """
    C = M_rot.shape[-1]
    assert C > 0 and (C & (C - 1)) == 0, (
        f"unrotate_ref only covers power-of-2 C (the FWHT path); got C={C}. "
        "For non-power-of-2, use _unrotate (random_orthogonal matmul, not in scope)."
    )
    signs = _hadamard_signs(C, seed).to(M_rot)
    return fwht_butterfly_ref(M_rot) * signs
