"""Triton fused dequant-attention DECODE kernel (k2b recipe).

Stage 3a: Serial-over-blocks online-softmax decode, correctness-first skeleton.

Deliberate staging (do not collapse prematurely):
  3a (this file): DEQUANT stays in PyTorch per block (reuse dequant_packed +
      from_matrix + apply_rope).  ONLY the online-softmax QK^T·V contraction
      runs in Triton.  Isolates the kernel skeleton so it is independently
      bit-exact vs naive_dense_attention before any in-kernel unpacking.
  3b/3c (future): Move RTN unpack into the Triton kernel.

Carry strategy: (acc, m, lse) are Python/PyTorch tensors carried between
per-block Triton kernel launches.  For decode (n_q==1) the blocks are small
relative to the per-launch overhead, but correctness is the goal here;
fusing the KV-block loop is a 3b concern.

GQA: q is (n_q_heads, 1, d), kv is (h_kv, blk, d).
  n_q_groups = n_q_heads // h_kv.
  The kernel avoids repeat_interleave by viewing q as (h_kv, n_q_groups, 1, d)
  and iterating: for each KV head g, launch a sub-kernel over the n_q_groups
  query heads that share it.  In this 3a skeleton we do this in the Python
  loop (one Triton launch per KV head per block), which is correct and simple.
  A single fused launch per block is a 3b optimisation.

Correctness bar: max_abs vs naive_dense_attention < 1e-2 (expect much tighter,
near fp16 rounding ~2e-4).  If it drifts, fix the kernel — do NOT loosen.

HARDWARE NOTE: Triton/CUDA not available on this AMD dev box.  The module
imports cleanly with TRITON_AVAILABLE=False.  The @triton.jit kernel is
verified on VM (Task 6 batch).  Concerns flagged in the report.
"""

from __future__ import annotations

import torch

from bmx.cache.collect import from_matrix
from bmx.cache.rope import apply_rope

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TRITON_AVAILABLE = False
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Capability guard — fail loud; NO silent fallback (fallback is Task 4).
# ---------------------------------------------------------------------------


def _require_triton() -> None:
    """Raise if Triton + CUDA are not available.

    The caller (dispatcher, Task 4) is responsible for routing to
    chunked_dequant_attention when this raises.  This module never falls back
    silently — a silent fallback would hide a missing capability.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "triton_decode_attention requires Triton + CUDA. "
            "TRITON_AVAILABLE=False on this machine (no CUDA or Triton not "
            "installed). Callers must dispatch to chunked_dequant_attention."
        )


# ---------------------------------------------------------------------------
# Triton kernel — online-softmax one block, one query head
# ---------------------------------------------------------------------------
#
# Grid: (G,) where G = n_q_groups.  Program idx = query-head-within-KV-head.
# Each program handles ONE query row against the full (BLK, D) key/value block.
#
# Layout (all pointers are contiguous):
#   q_ptr   : (G, D)     — all query rows for this KV head (n_q==1 squeezed)
#   k_ptr   : (BLK, D)   — one KV-head block of keys (dequanted+RoPE, fp16)
#   v_ptr   : (BLK, D)   — matching values (fp16)
#   acc_ptr : (G, D)     — running weighted-value accumulator (fp16)
#   m_ptr   : (G,)       — running max scalar (fp32)
#   lse_ptr : (G,)       — running log-sum-exp denominator (fp32)
#   scale   : fp32 scalar — 1/sqrt(d)
#   BLK     : constexpr block size (== blk at launch)
#   D       : constexpr head dim
#
# By using grid=(G,) and each program handling one query row, we sidestep
# the tl.dot ≥ 16 constraint on the G dimension.  tl.dot only sees:
#   q_row (1, D) x k.T (D, BLK) -> (1, BLK)  — D≥16, BLK≥16 ✓
#   p     (1, BLK) x v (BLK, D) -> (1, D)    — BLK≥16, D≥16 ✓
# (both D and BLK are expected to be 64/128 in practice)
#
# Online softmax math (base-e, mirrors online_softmax_update in chunked_attention):
#   scores  = (q_row @ k.T) * scale      (1, BLK)
#   m_new   = max(m_old, max(scores))
#   alpha   = exp(m_old - m_new)
#   p       = exp(scores - m_new)        (1, BLK)
#   lse_new = lse_old * alpha + sum(p)
#   acc_new = acc_old * alpha + p @ v    (1, D)
#
# For 3a we do NOT tile over D (one kernel launch loads the full D-slice).


if TRITON_AVAILABLE:

    @triton.jit
    def _online_softmax_block_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        acc_ptr,
        m_ptr,
        lse_ptr,
        scale,
        BLK: tl.constexpr,
        D: tl.constexpr,
    ):
        """Online-softmax step for ONE query head over ONE (BLK, D) K/V block.

        Grid: (n_q_groups,) — each program is one query head within the KV head.
        Correctness-first (3a): one query row at a time, full D loaded, no tiling.

        tl.dot is safe here: all matrix dims are D (≥64) and BLK (≥16 in practice).
        The G (n_q_groups) dimension is NOT a tl.dot dimension — it is the grid.
        """
        g = tl.program_id(0)  # which query head (within this KV head)
        d_idx = tl.arange(0, D)
        b_idx = tl.arange(0, BLK)

        # ------------------------------------------------------------------
        # Load q row: (1, D) for this query head
        # ------------------------------------------------------------------
        q_row = tl.load(q_ptr + g * D + d_idx).to(tl.float32)  # (D,)

        # ------------------------------------------------------------------
        # Load k block: (BLK, D)
        # ------------------------------------------------------------------
        k_offsets = b_idx[:, None] * D + d_idx[None, :]  # (BLK, D)
        k = tl.load(k_ptr + k_offsets).to(tl.float32)  # (BLK, D)

        # ------------------------------------------------------------------
        # scores = (q_row @ k.T) * scale  -> (BLK,)
        # Use tl.dot: (1, D) x (D, BLK) -> (1, BLK), then squeeze to (BLK,).
        # Reshape q_row to (1, D) for tl.dot.
        # ------------------------------------------------------------------
        q_2d = tl.reshape(q_row, (1, D))  # (1, D)
        scores_2d = tl.dot(q_2d, tl.trans(k)) * scale  # (1, BLK)
        scores = tl.reshape(scores_2d, (BLK,))  # (BLK,)

        # ------------------------------------------------------------------
        # Online softmax update (base-e):
        # ------------------------------------------------------------------
        m_old = tl.load(m_ptr + g).to(tl.float32)  # scalar
        lse_old = tl.load(lse_ptr + g).to(tl.float32)  # scalar

        m_new = tl.maximum(m_old, tl.max(scores, axis=0))  # scalar
        alpha = tl.exp(m_old - m_new)  # scalar correction
        p = tl.exp(scores - m_new)  # (BLK,)
        lse_new = lse_old * alpha + tl.sum(p, axis=0)  # scalar

        # ------------------------------------------------------------------
        # Accumulator update: acc_new = acc_old * alpha + p @ v   (D,)
        # ------------------------------------------------------------------
        v_offsets = b_idx[:, None] * D + d_idx[None, :]  # (BLK, D)
        v = tl.load(v_ptr + v_offsets).to(tl.float32)  # (BLK, D)

        acc_old = tl.load(acc_ptr + g * D + d_idx).to(tl.float32)  # (D,)

        # p @ v: (BLK,) x (BLK, D) -> (D,) via tl.dot on (1, BLK) x (BLK, D)
        p_2d = tl.reshape(p, (1, BLK))  # (1, BLK)
        pv_2d = tl.dot(p_2d, v)  # (1, D)
        pv = tl.reshape(pv_2d, (D,))  # (D,)

        acc_new = acc_old * alpha + pv  # (D,)

        # ------------------------------------------------------------------
        # Store updated (acc, m, lse)
        # ------------------------------------------------------------------
        tl.store(acc_ptr + g * D + d_idx, acc_new.to(tl.float16))
        tl.store(m_ptr + g, m_new.to(tl.float32))
        tl.store(lse_ptr + g, lse_new.to(tl.float32))


# ---------------------------------------------------------------------------
# Python-level per-block launcher — carries (acc, m, lse) in PyTorch tensors
# ---------------------------------------------------------------------------


def _online_block_kernel_launch(
    q: torch.Tensor,
    K_kv: torch.Tensor,
    V_kv: torch.Tensor,
    acc: torch.Tensor,
    m: torch.Tensor,
    lse: torch.Tensor,
    n_q_groups: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the Triton online-softmax kernel for one (h_kv, blk, d) K/V block.

    GQA loop: one Triton launch per KV head, grid=(n_q_groups,) so each
    program handles one query head.  Carry (acc, m, lse) flow in PyTorch
    tensors between launches.  Correctness-first for 3a; fusing KV heads
    into one launch is a 3b concern.

    Args:
        q:          (n_q_heads, 1, d) fp16
        K_kv:       (h_kv, blk, d) fp16 — dequanted + RoPE applied
        V_kv:       (h_kv, blk, d) fp16
        acc:        (n_q_heads, 1, d) fp16 — running accumulator (in-out)
        m:          (n_q_heads, 1, 1) fp32 — running max (in-out)
        lse:        (n_q_heads, 1, 1) fp32 — running lse denominator (in-out)
        n_q_groups: n_q_heads // h_kv
        scale:      attention scale (1/sqrt(d))

    Returns updated (acc, m, lse) — same shapes, updated in-place internally.
    """
    _require_triton()
    n_q_heads, n_q, d = q.shape
    h_kv, blk, _d = K_kv.shape
    assert n_q == 1, "decode only"
    assert _d == d

    # Lay out carry buffers as (h_kv, n_q_groups, d/1) contiguous so that
    # [kv_head] slices are contiguous and can be passed directly to Triton.
    # The kernel writes back via stored pointers, so the slice IS the buffer.
    q_v = q.view(h_kv, n_q_groups, d)  # (h_kv, G, d)  squeeze n_q
    # acc: fp16 in/out  — (h_kv, G, d)
    acc_buf = acc.view(h_kv, n_q_groups, d).contiguous()  # (h_kv, G, d) fp16
    # m, lse: fp32 carry — (h_kv, G)
    m_buf = m.view(h_kv, n_q_groups).float().contiguous()  # (h_kv, G) fp32
    lse_buf = lse.view(h_kv, n_q_groups).float().contiguous()  # (h_kv, G) fp32

    for kv in range(h_kv):
        # Each slice [kv] is contiguous (last dims G, d/1 are row-major).
        q_kv = q_v[kv].contiguous()  # (G, d) — query rows for this KV head
        k_kv = K_kv[kv].contiguous()  # (blk, d)
        v_kv = V_kv[kv].contiguous()  # (blk, d)
        acc_kv = acc_buf[kv]  # (G, d) fp16 — Triton writes in-place
        m_kv = m_buf[kv]  # (G,)  fp32 — Triton writes in-place
        lse_kv = lse_buf[kv]  # (G,)  fp32 — Triton writes in-place

        # Grid = (n_q_groups,): each program is one query head within this KV head.
        _online_softmax_block_kernel[(n_q_groups,)](
            q_kv,
            k_kv,
            v_kv,
            acc_kv,
            m_kv,
            lse_kv,
            float(scale),
            BLK=blk,
            D=d,
        )
        # Triton stores in-place via pointer, so acc_kv/m_kv/lse_kv are updated.

    # Reconstruct original-shape carry tensors from updated buffers.
    acc_new = acc_buf.view(n_q_heads, 1, d)
    m_new = m_buf.view(n_q_heads, 1, 1)
    lse_new = lse_buf.view(n_q_heads, 1, 1)
    return acc_new, m_new, lse_new


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def triton_decode_attention(
    q: torch.Tensor,
    k_blocks: list,
    v_blocks: list,
    *,
    k_arm: str,
    v_arm: str,
    group: int,
    seed: int,
    k_pre_rope: bool,
    rope_cos: torch.Tensor | None,
    rope_sin: torch.Tensor | None,
    k_tail: torch.Tensor | None,
    v_tail: torch.Tensor | None,
    n_q_groups: int,
    scale: float,
) -> torch.Tensor:
    """Triton decode attention: online-softmax over packed KV blocks.

    Same call shape as chunked_dequant_attention (minus query_abs_start —
    this is decode-only, n_q==1).  Produces (n_q_heads, n_q, d).

    Stage 3a: dequant stays in PyTorch (dequant_packed + from_matrix +
    apply_rope); only the online-softmax contraction runs in Triton.
    Each block:  PyTorch dequant -> Triton _online_softmax_block_kernel.
    Carry (acc, m, lse) flows between launches in Python/PyTorch tensors.

    GQA: one Triton launch per KV head per block (fusing into one launch
    per block is a 3b optimisation).
    """
    _require_triton()
    from bmx.cache.codecs import dequant_packed

    n_q_heads, n_q, d = q.shape
    assert n_q == 1, (
        "triton_decode_attention is decode-only (n_q==1); "
        "prefill stays on the flash-SDPA path in chunked_dequant_attention."
    )
    h_kv = n_q_heads // n_q_groups

    # Initialise carry tensors on the same device as q.
    acc = torch.zeros(n_q_heads, n_q, d, dtype=q.dtype, device=q.device)
    m = torch.full(
        (n_q_heads, n_q, 1), float("-inf"), dtype=torch.float32, device=q.device
    )
    lse = torch.zeros(n_q_heads, n_q, 1, dtype=torch.float32, device=q.device)

    def _attend(K_kv: torch.Tensor, V_kv: torch.Tensor) -> None:
        nonlocal acc, m, lse
        acc, m, lse = _online_block_kernel_launch(
            q, K_kv, V_kv, acc, m, lse, n_q_groups, scale
        )

    for (kpacked, start, end), (vpacked, _vs, _ve) in zip(k_blocks, v_blocks):
        # Stage 3a: dequant in PyTorch (3b/3c will move this in-kernel).
        K_kv = from_matrix(
            dequant_packed(k_arm, kpacked, seed=seed, group=group), h_kv
        ).to(q.dtype)
        if k_pre_rope:
            K_kv = apply_rope(
                K_kv,
                rope_cos[start:end].to(K_kv.dtype),
                rope_sin[start:end].to(K_kv.dtype),
            )
        V_kv = from_matrix(
            dequant_packed(v_arm, vpacked, seed=seed, group=group), h_kv
        ).to(q.dtype)
        _attend(K_kv, V_kv)

    # fp16 tail window (post-RoPE for K, already in correct dtype).
    if k_tail is not None and k_tail.shape[1] > 0:
        _attend(k_tail.to(q.dtype), v_tail.to(q.dtype))

    # Normalise: acc / lse (lse is the raw sum-of-weights, not log-normalised).
    return acc / lse.to(q.dtype)
