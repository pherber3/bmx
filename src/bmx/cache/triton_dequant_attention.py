"""Triton fused dequant-attention DECODE kernel (k2b recipe).

Provides: Triton online-softmax decode kernel + graphable split-KV path +
dispatch helpers.  Two kernel paths: RTN arms (dense K/V from Python dequant)
and k2b (lowrank_rtn_channel K unpacked in-kernel; turboquant_mse V dequanted
in Python and passed dense).  Imports cleanly with TRITON_AVAILABLE=False
(AMD/no-CUDA dev box); kernel verified on VM.

Usage: call triton_decode_attention(q, k_blocks, v_blocks, k_arm=..., ...).
Design rationale and staged-build ledger:
  docs/superpowers/specs/2026-06-24-triton-decode-kernel-design.md
"""

from __future__ import annotations

import math

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
# Staged-build ledger + correctness invariants (see spec for full rationale)
# ---------------------------------------------------------------------------
#
# Staged build (do not collapse prematurely):
#   3a: Python dequant per block; only online-softmax contraction in Triton.
#       Isolates kernel skeleton for bit-exact verification vs naive_dense_attention.
#   3b: Split-KV decode parallelism + @triton.autotune + do_not_specialize on
#       block-count arg.
#   3c: In-kernel k2b unpack — lowrank K via tl.dot(Us, V_fac.T) + RTN residual;
#       turboquant_mse V dequanted in Python (_unrotate stays Python, deferred 3c+).
#   3c+: Full in-kernel FWHT for V (deferred — subtle-bug magnet; negligible cost
#        vs memory-bandwidth saving on K; revisit only if VM profiling demands it).
#
# v_group / v_seed (3c): K and V may differ in seed/group; both accepted as kwargs
#   (default to K's values for back-compat with 3a/3b RTN-only tests).
#
# GQA carry: (acc, m, lse) Python tensors between per-block Triton launches;
#   fusing the KV-block loop is a 3c+ concern.
#
# Correctness bar: max_abs vs naive_dense_attention < 1e-2 (expect ~2e-4 at fp16).
#   Do NOT loosen — fix the kernel if it drifts.
#
# Split-KV merge invariant (3b — must hold):
#   Each split stores pre-normalization (acc_i, m_i, lse_i), merged as:
#     m = max_i(m_i);  l = sum_i(lse_i * exp(m_i - m));
#     out = sum_i(acc_i * exp(m_i - m)) / l
#   At num_splits=1 this reduces to acc_0 / lse_0 (bit-identical to 3a).
#
# Base-e consistency: ALL kernels and merge use natural exp — do NOT mix base-2.
#   A base-2 merge formula is a silent correctness trap (class of bug flagged in
#   the 3a report).
#
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
# For 3a/3b we do NOT tile over D (one kernel launch loads the full D-slice).
#
# 3b AUTOTUNE NOTE:
#   @triton.autotune wraps the kernel with a configs list; key=["d","n_q_groups"]
#   so Triton specializes on head dim + GQA groups (hardware-characteristic),
#   NOT on seq_len / block count (which changes every decode step).
#   do_not_specialize=["n_blocks_in_split"] prevents per-block-count recompiles
#   (the AWS 10x TTFT regression).  n_blocks_in_split cannot be tl.constexpr
#   because it varies per split; it is used only for loop bounds.


if TRITON_AVAILABLE:
    # ---------------------------------------------------------------------------
    # Autotune configs — a few representative BLOCK/warp/stage combos.
    # Tuned on shape args only (d, n_q_groups).  NOT on seq_len.
    # These configs are intentionally modest; the VM run will tell us the winner.
    # Adding more configs costs compile time, not correctness.
    # ---------------------------------------------------------------------------
    # Import Config directly so Pylance sees the concrete type (not `triton: None`).
    from triton import Config as _TritonConfig

    _AUTOTUNE_CONFIGS = [
        _TritonConfig({"BLK": 64}, num_warps=4, num_stages=2),
        _TritonConfig({"BLK": 64}, num_warps=8, num_stages=2),
        _TritonConfig({"BLK": 128}, num_warps=4, num_stages=2),
        _TritonConfig({"BLK": 128}, num_warps=8, num_stages=2),
    ]

    @triton.autotune(
        configs=_AUTOTUNE_CONFIGS,
        key=["d", "n_q_groups"],
        # This kernel mutates acc/m/lse IN PLACE (read-modify-write carry). Autotune
        # benchmarks each config by calling the kernel repeatedly on the SAME buffers,
        # which would accumulate N times and corrupt the carry on the first (tuning)
        # call for each shape. restore_value clones+restores these between trials so the
        # post-tuning real run starts from the correct pre-call carry. Without this the
        # first call per shape returns garbage (confirmed on GH200: seed-dependent
        # m/lse blow-up that vanished once the autotune cache was warm).
        restore_value=["acc_ptr", "m_ptr", "lse_ptr"],
    )
    @triton.jit(do_not_specialize=["blk"])
    def _online_softmax_block_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        acc_ptr,
        m_ptr,
        lse_ptr,
        scale,
        blk,  # ACTUAL rows in this K/V block — do_not_specialize runtime arg
        d: tl.constexpr,
        n_q_groups: tl.constexpr,  # noqa: ARG001
        BLK: tl.constexpr,  # autotune TILE size — may EXCEED blk (must mask)
    ):
        """Online-softmax step for ONE query head over ONE (blk, D) K/V block.

        Grid: (n_q_groups,) — each program is one query head within the KV head.
        Correctness-first (3a/3b): one query row at a time, full D loaded, no tiling.

        BLK (autotune tile) is INDEPENDENT of blk (the actual block length): autotune
        picks BLK from a config list (64, 128, ...), so BLK may be LARGER than blk.
        Every block-dim load/score MUST be masked by `b_idx < blk`, or the kernel
        reads out-of-bounds rows -> garbage/NaN scores -> the softmax max/denominator
        blow up (confirmed on GH200: BLK=128 on a 64-row block gave m=NaN).
        Masked scores are set to -inf so exp(score-m)=0 — they contribute nothing to
        the running max or the lse denominator (online-softmax max-subtraction;
        Physics of LLM Inference ~line 1931, FlashAttention online-softmax).

        blk is do_not_specialize so Triton does not recompile per decode step / per
        tail block.  tl.dot dims are D (>=64) and BLK (>=64 from configs).
        """
        g = tl.program_id(0)  # which query head (within this KV head)
        d_idx = tl.arange(0, d)
        b_idx = tl.arange(0, BLK)
        blk_mask = b_idx < blk  # (BLK,) True for real rows, False for OOB tile rows

        # ------------------------------------------------------------------
        # Load q row: (1, D) for this query head
        # ------------------------------------------------------------------
        q_row = tl.load(q_ptr + g * d + d_idx).to(tl.float32)  # (D,)

        # ------------------------------------------------------------------
        # Load k block: (BLK, D) — mask OOB tile rows to 0 (other=0.0)
        # ------------------------------------------------------------------
        k_offsets = b_idx[:, None] * d + d_idx[None, :]  # (BLK, D)
        k = tl.load(k_ptr + k_offsets, mask=blk_mask[:, None], other=0.0).to(
            tl.float32
        )  # (BLK, D)

        # ------------------------------------------------------------------
        # scores[b] = (sum_d q[d]*k[b,d]) * scale  -> (BLK,)
        # Decode is M=1 (single query), so this is a GEMV/reduction, NOT a GEMM:
        # use broadcast-multiply + tl.sum, which is numerically identical to tl.dot
        # AND has no >=16 min-dim constraint (tl.dot needs M,N,K>=16 — fails for
        # tiny head_dim d<16 or small BLK; real models have d>=64 but the test model
        # has d=8). Verified bit-exact (1e-7) vs tl.dot at d=8 and d=64.
        # ------------------------------------------------------------------
        scores = tl.sum(q_row[None, :] * k, axis=1) * scale  # (BLK,)
        # Masked (OOB) positions -> -inf so they vanish in the softmax max + denom.
        scores = tl.where(blk_mask, scores, float("-inf"))  # (BLK,)

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
        v_offsets = b_idx[:, None] * d + d_idx[None, :]  # (BLK, D)
        v = tl.load(v_ptr + v_offsets, mask=blk_mask[:, None], other=0.0).to(
            tl.float32
        )  # (BLK, D) — OOB rows 0 (p is also 0 there, so doubly safe)

        acc_old = tl.load(acc_ptr + g * d + d_idx).to(tl.float32)  # (D,)

        # pv[dd] = sum_b p[b]*v[b,dd]  -> (D,)  (GEMV via multiply+sum, not tl.dot;
        # same no-min-dim rationale as the scores above).
        pv = tl.sum(p[:, None] * v, axis=0)  # (D,)

        acc_new = acc_old * alpha + pv  # (D,)

        # ------------------------------------------------------------------
        # Store updated (acc, m, lse)
        # ------------------------------------------------------------------
        tl.store(acc_ptr + g * d + d_idx, acc_new.to(tl.float16))
        tl.store(m_ptr + g, m_new.to(tl.float32))
        tl.store(lse_ptr + g, lse_new.to(tl.float32))


# ---------------------------------------------------------------------------
# Stage 3c: k2b kernel — in-kernel lowrank K unpack + online-softmax.
#
# This kernel handles k_arm="lowrank_rtn_channel" (K) with any pre-dequanted
# V (turboquant_mse V is Python-dequanted before call via _v_dequant_turboquant).
#
# K reconstruction (in-kernel):
#   L        = tl.dot(us_block, vfac.T)          — lowrank component (BLK, D)
#   R_hat    = (res_int * res_scale).T            — RTN residual (BLK, D)
#   K        = L + R_hat                          — full K block (BLK, D)
#
# Layout of packed K inputs (per block, single KV head):
#   us_ptr      : (BLK, RANK) fp16 — Us[:, h, :] for this head & block
#   vfac_ptr    : (D,   RANK) fp16 — V[:, h, :] for this head (full D×RANK)
#   res_ptr     : (D,   BLK)  int8 — RTN codes, column-major within the block
#   res_scale_ptr: (D, BLK//GROUP, 1) fp16 — per-group scale, same layout
#   v_ptr       : (BLK, D)   fp16 — pre-dequanted V (Python applied _unrotate)
#   q_ptr / acc_ptr / m_ptr / lse_ptr: same as _online_softmax_block_kernel
#
# RTN residual memory layout:
#   rtn_quantize_packed(R.mT, bits, group) stores (C=D, S=full_S) int8.
#   For a block [start:end] with blk=end-start tokens, the per-head slice is
#   (D, blk) int8 contiguous after Python pre-slicing in the launcher.
#   res_scale is (D, blk//group) after squeeze.
#
# VM risk #1 (highest): codebook gather — verify tl.load(cb_ptr + idx) round-
#   trips indices correctly (int16 → int32 cast, no wraparound).
# VM risk #2: RTN residual in-kernel — verify res_scale broadcast shape
#   (D, blk//GROUP, 1) → (D, blk).  The kernel repeats scale GROUP times.
# VM risk #3: tl.dot shape — us_block is (BLK, RANK), vfac is (D, RANK);
#   we compute tl.dot(us_block, tl.trans(vfac)) → (BLK, D).  Requires
#   BLK ≥ 16, D ≥ 16, RANK ≥ 16.  If RANK < 16, tl.dot may error; mitigate
#   by tiling or using tl.sum elementwise.  Flag for VM.
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    @triton.jit(do_not_specialize=["n_blocks_in_split"])
    def _k2b_softmax_block_kernel(
        # Packed K inputs (lowrank_rtn_channel, per head, per block)
        us_ptr,  # (BLK, RANK) fp16 — Us block for this KV head
        vfac_ptr,  # (D, RANK) fp16 — V factor (same for all blocks)
        res_ptr,  # (D, BLK) int8 — RTN residual codes
        res_scale_ptr,  # (D, N_GROUPS) fp16 — per-group RTN scales (squeezed)
        # Pre-dequanted V (Python applied _unrotate * norms)
        v_ptr,  # (BLK, D) fp16
        # Query + carry buffers (same layout as _online_softmax_block_kernel)
        q_ptr,
        acc_ptr,
        m_ptr,
        lse_ptr,
        # RoPE cos/sin for THIS block's absolute positions: (blk, d) each. Only read
        # when HAS_ROPE; pass a dummy (e.g. v_ptr) when HAS_ROPE is False.
        cos_ptr,
        sin_ptr,
        scale,
        n_blocks_in_split,  # intentionally unused in kernel body — exists only as a do_not_specialize autotune-specialization guard (prevents recompile when seq_len changes)  # noqa: ARG001
        d: tl.constexpr,
        n_q_groups: tl.constexpr,  # noqa: ARG001
        blk: tl.constexpr,  # actual block size (NOT autotuned — matches packed block)
        rank: tl.constexpr,  # lowrank rank (small, e.g. 16–64)
        group: tl.constexpr,  # RTN group size
        HAS_ROPE: tl.constexpr,  # apply in-kernel RoPE to reconstructed K (pre-RoPE keys)
    ):
        """Online-softmax step with in-kernel lowrank K unpack.

        Grid: (n_q_groups,) — each program is one query head within the KV head.

        K is reconstructed in-kernel:
            L     = tl.dot(us_block, vfac.T)     (BLK, D)
            R_hat = RTN dequant of res_ptr        (BLK, D)
            K     = L + R_hat

        V is passed pre-dequanted (Python _unrotate applied outside kernel).

        VM risk #3 (RANK): tl.dot requires both inner dims ≥ 16.
            BLK × RANK: BLK ≥ 16 ✓ (blk=block_size ≥ 64 in practice).
            RANK × D: RANK ≥ 16 is NOT guaranteed (typical: 16–64).
            If RANK < 16 on the VM, replace the lowrank tl.dot with a
            manual accumulation loop (RANK iters of outer-product-add).
            Flag this in the VM-verify checklist.
        """
        g = tl.program_id(0)  # query head index within this KV head
        d_idx = tl.arange(0, d)
        b_idx = tl.arange(0, blk)
        r_idx = tl.arange(0, rank)
        n_groups = blk // group

        # ------------------------------------------------------------------
        # Load q row: (D,) for this query head
        # ------------------------------------------------------------------
        q_row = tl.load(q_ptr + g * d + d_idx).to(tl.float32)  # (D,)

        # ------------------------------------------------------------------
        # In-kernel K lowrank reconstruction
        # Step 1: Load Us block (BLK, RANK) — row-major (blk × rank)
        # ------------------------------------------------------------------
        us_offsets = b_idx[:, None] * rank + r_idx[None, :]  # (BLK, RANK)
        us_block = tl.load(us_ptr + us_offsets).to(tl.float32)  # (BLK, RANK)

        # Step 2: Load V factor (D, RANK) — stored row-major (d × rank)
        vfac_offsets = d_idx[:, None] * rank + r_idx[None, :]  # (D, RANK)
        vfac = tl.load(vfac_ptr + vfac_offsets).to(tl.float32)  # (D, RANK)

        # Step 3: Lowrank L[b,dd] = sum_r us[b,r]*vfac[dd,r]  → (BLK, D).
        # Broadcast-multiply + tl.sum over RANK instead of tl.dot — no >=16 min-dim
        # constraint (tl.dot needs D,RANK,BLK>=16; the test model has D=8 and RANK can
        # be 16-at-the-boundary). Materializes a (BLK,D,RANK) transient (small for
        # decode blocks). Numerically identical to the dot.
        K_lowrank = tl.sum(
            us_block[:, None, :] * vfac[None, :, :], axis=2
        )  # (BLK, D) fp32

        # ------------------------------------------------------------------
        # Step 4: RTN residual (D, BLK) int8 → dequant → (BLK, D)
        # res_ptr layout: (D, BLK) contiguous → res[d_i, b_j] = res_ptr[d_i*blk + b_j]
        # res_scale layout: (D, N_GROUPS) fp16 → scale[d_i, g_j] = res_scale_ptr[d_i*n_groups + g_j]
        # Dequant: for each (d_i, b_j): Q_int[d_i, b_j] * scale[d_i, b_j // group]
        # ------------------------------------------------------------------
        res_offsets = d_idx[:, None] * blk + b_idx[None, :]  # (D, BLK)
        res_int = tl.load(res_ptr + res_offsets).to(tl.float32)  # (D, BLK) fp32

        # Load scales per (D, BLK): for each (d_i, b_j), use res_scale[d_i, b_j // group].
        # Directly compute load offsets as d_i * n_groups + b_j // group.
        # This maps (D, BLK) → res_scale_ptr with correct group alignment.
        # Each b_j in [0, BLK) maps to group index b_j // group in [0, N_GROUPS).
        res_scale_expanded_offsets = (
            d_idx[:, None] * n_groups + b_idx[None, :] // group
        )  # (D, BLK) int
        res_scale_expanded = tl.load(res_scale_ptr + res_scale_expanded_offsets).to(
            tl.float32
        )  # (D, BLK) fp32 — scale broadcast per group

        # RTN dequant: element-wise multiply codes by their group scale
        res_dequant_d_b = res_int * res_scale_expanded  # (D, BLK) fp32

        # Transpose (D, BLK) → (BLK, D) for K reconstruction
        K_residual = tl.trans(res_dequant_d_b)  # (BLK, D) fp32

        # ------------------------------------------------------------------
        # Full K = lowrank + residual
        # ------------------------------------------------------------------
        k = K_lowrank + K_residual  # (BLK, D) fp32

        # ------------------------------------------------------------------
        # In-kernel RoPE on the reconstructed K (keys stored PRE-RoPE; applied at
        # read). cos_ptr/sin_ptr are this block's (blk, d) tables (sliced by the
        # launcher for the block's absolute positions). rotate_half via gather:
        # rotate_half(x) = cat(-x[d/2:], x[:d/2]) (transformers convention, matches
        # rope._rotate_half). For col j: source col = j+half if j<half else j-half;
        # sign = -1 if j<half else +1. Verified bit-exact (2.4e-7) vs apply_rope.
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # scores[b] = (sum_dd q[dd] * k_rope[b,dd]) * scale  → (BLK,)  (GEMV
        # multiply+sum, not tl.dot — no >=16 min-dim constraint).
        # ------------------------------------------------------------------
        if HAS_ROPE:
            # In-register RoPE via masked column gather — NO HBM scratch, NO barrier,
            # NO 2D tensor slicing (this Triton supports none of tl.cat / tl.join /
            # k[:, :half]). rotate_half(x)=cat(-x[d/2:], x[:d/2]): build rot from k by
            # masking the FULL (BLK, D) tensor — for column j, rot[:,j] picks the
            # opposite half with a sign flip. We form rot via two masked full-width
            # tensors (shift up / shift down) selected by the column's half, all in
            # registers. cos/sin are loaded full-width (BLK, D). Verified vs apply_rope.
            half = d // 2
            cos = tl.load(cos_ptr + b_idx[:, None] * d + d_idx[None, :]).to(
                tl.float32
            )  # (BLK, D)
            sin = tl.load(sin_ptr + b_idx[:, None] * d + d_idx[None, :]).to(
                tl.float32
            )  # (BLK, D)
            # k shifted by +half (cols d/2..d-1 brought to 0..d/2-1) and -half.
            # Build via full-width multiply with a one-hot-ish reduction: rot[:,j] =
            # sum_jj k[:,jj] * P[jj,j] where P is the rotate_half permutation+sign.
            # P[jj,j] = -1 if (jj==j+half and j<half) ; +1 if (jj==j-half and j>=half).
            jj = d_idx  # source col index (over D)
            # For each output col j (d_idx) we need k[:, src] * sign:
            #   j<half  -> src=j+half, sign=-1 ; j>=half -> src=j-half, sign=+1
            # Implement as: rot = sum over a (D, D) permutation applied to k (BLK, D).
            # P_mat[src, j]: (D, D)
            j_is_first = d_idx < half  # over output cols j
            src_for_j = tl.where(
                j_is_first, d_idx + half, d_idx - half
            )  # (D,) src col per j
            sign_for_j = tl.where(j_is_first, -1.0, 1.0)  # (D,)
            # one-hot P[src, j] = (jj==src_for_j[j]) * sign_for_j[j], shape (D, D)
            P = tl.where(
                jj[:, None] == src_for_j[None, :], sign_for_j[None, :], 0.0
            )  # (D_src, D_out)
            rot = tl.sum(k[:, :, None] * P[None, :, :], axis=1)  # (BLK, D_out)
            k = k * cos + rot * sin  # (BLK, D)
        scores = tl.sum(q_row[None, :] * k, axis=1) * scale  # (BLK,)

        # ------------------------------------------------------------------
        # Online softmax update (base-e):
        # ------------------------------------------------------------------
        m_old = tl.load(m_ptr + g).to(tl.float32)
        lse_old = tl.load(lse_ptr + g).to(tl.float32)

        m_new = tl.maximum(m_old, tl.max(scores, axis=0))
        alpha = tl.exp(m_old - m_new)
        p = tl.exp(scores - m_new)  # (BLK,)
        lse_new = lse_old * alpha + tl.sum(p, axis=0)

        # ------------------------------------------------------------------
        # Accumulator update: acc_new = acc_old * alpha + p @ v  (D,)
        # ------------------------------------------------------------------
        v_offsets = b_idx[:, None] * d + d_idx[None, :]  # (BLK, D)
        v = tl.load(v_ptr + v_offsets).to(tl.float32)  # (BLK, D)

        acc_old = tl.load(acc_ptr + g * d + d_idx).to(tl.float32)  # (D,)

        # pv[dd] = sum_b p[b]*v[b,dd]  → (D,)  (GEMV multiply+sum, not tl.dot)
        pv = tl.sum(p[:, None] * v, axis=0)  # (D,)

        acc_new = acc_old * alpha + pv  # (D,)

        # ------------------------------------------------------------------
        # Store updated carry
        # ------------------------------------------------------------------
        tl.store(acc_ptr + g * d + d_idx, acc_new.to(tl.float16))
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
    n_blocks_in_split: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the Triton online-softmax kernel for one (h_kv, blk, d) K/V block.

    GQA loop: one Triton launch per KV head, grid=(n_q_groups,) so each
    program handles one query head.  Carry (acc, m, lse) flow in PyTorch
    tensors between launches.  Correctness-first for 3a; fusing KV heads
    into one launch is a 3b concern.

    Args:
        q:                  (n_q_heads, 1, d) fp16
        K_kv:               (h_kv, blk, d) fp16 — dequanted + RoPE applied
        V_kv:               (h_kv, blk, d) fp16
        acc:                (n_q_heads, 1, d) fp16 — running accumulator (in-out)
        m:                  (n_q_heads, 1, 1) fp32 — running max (in-out)
        lse:                (n_q_heads, 1, 1) fp32 — running lse denominator (in-out)
        n_q_groups:         n_q_heads // h_kv
        scale:              attention scale (1/sqrt(d))
        n_blocks_in_split:  total blocks in this split (passed to kernel as
                            do_not_specialize runtime arg; prevents recompile
                            per seq_len).

    Returns updated (acc, m, lse) — same shapes, updated in-place internally.
    """
    _require_triton()
    n_q_heads, n_q, d = q.shape
    h_kv, _blk, _d = K_kv.shape
    assert n_q == 1, "decode only"
    assert _d == d

    # Lay out carry buffers as (h_kv, n_q_groups, d/1) contiguous so that
    # [kv_head] slices are contiguous and can be passed directly to Triton.
    # The kernel writes back via stored pointers, so the slice IS the buffer.
    q_v = q.view(h_kv, n_q_groups, d)  # (h_kv, G, d)  squeeze n_q
    # acc: fp16 in/out  — (h_kv, G, d)
    acc_buf = acc.view(h_kv, n_q_groups, d)  # (h_kv, G, d) fp16 — zeros, contiguous
    # m, lse: fp32 carry — (h_kv, G)
    m_buf = m.view(h_kv, n_q_groups).float()  # (h_kv, G) fp32 — already contiguous
    lse_buf = lse.view(h_kv, n_q_groups).float()  # (h_kv, G) fp32 — already contiguous

    for kv in range(h_kv):
        # Each slice [kv] is contiguous (last dims G, d/1 are row-major).
        q_kv = q_v[kv].contiguous()  # (G, d) — query rows for this KV head
        k_kv = K_kv[kv].contiguous()  # (blk, d)
        v_kv = V_kv[kv].contiguous()  # (blk, d)
        acc_kv = acc_buf[kv]  # (G, d) fp16 — Triton writes in-place
        m_kv = m_buf[kv]  # (G,)  fp32 — Triton writes in-place
        lse_kv = lse_buf[kv]  # (G,)  fp32 — Triton writes in-place

        # Grid = (n_q_groups,): each program is one query head within this KV head.
        # Note: BLK is NOT passed explicitly — autotune provides it.
        # n_blocks_in_split is a do_not_specialize runtime arg (not constexpr).
        _online_softmax_block_kernel[(n_q_groups,)](
            q_kv,
            k_kv,
            v_kv,
            acc_kv,
            m_kv,
            lse_kv,
            float(scale),
            int(_blk),  # ACTUAL block length — kernel masks the BLK tile down to this
            d=d,
            n_q_groups=n_q_groups,
        )
        # Triton stores in-place via pointer, so acc_kv/m_kv/lse_kv are updated.

    # Reconstruct original-shape carry tensors from updated buffers.
    acc_new = acc_buf.view(n_q_heads, 1, d)
    m_new = m_buf.view(n_q_heads, 1, 1)
    lse_new = lse_buf.view(n_q_heads, 1, 1)
    return acc_new, m_new, lse_new


# ---------------------------------------------------------------------------
# Stage 3c: Python V pre-dequant for turboquant_mse + k2b kernel launcher
# ---------------------------------------------------------------------------


def _v_dequant_turboquant_mse(
    vpacked: dict,
    h_kv: int,
    v_seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequant one turboquant_mse V block → (h_kv, blk, d) dense fp16.

    Delegates to codecs._turboquant_mse_dequant (codebook gather + _unrotate +
    scale by norms), then converts from (S, C) matrix layout to (h_kv, blk, d)
    via from_matrix.  The _unrotate (FWHT) is deliberately kept in Python —
    in-kernel FWHT is a deferred 3c+ optimisation (see staged-build ledger above).
    Note: codebook dtype is fp32 inside _turboquant_mse_dequant; cast to fp16 here.
    """
    # Deferred imports: codecs is not imported at module top (circular-import
    # avoidance — codecs → collect → triton_dequant_attention would cycle).
    from bmx.cache.codecs import _turboquant_mse_dequant

    indices = vpacked["indices"]  # (S, C) int16
    norms = vpacked["norms"]  # (S, 1) fp
    bits = int(vpacked["bits"])
    C = indices.shape[1]

    # Move to target device if needed (mirrors _blocks_cuda pattern).
    if indices.device != device:
        indices = indices.to(device)
        norms = norms.to(device)

    # Delegate to the canonical codec dequant (_unrotate stays Python — deferred
    # in-kernel FWHT; see 3c note in module docstring).
    V_mat = _turboquant_mse_dequant(indices, norms, bits, v_seed, C).to(dtype)

    # Convert from (S=blk, C=d) matrix layout back to (h_kv, blk, d).
    return from_matrix(V_mat, h_kv)


def _k2b_block_kernel_launch(
    q: torch.Tensor,
    kpacked: dict,
    V_kv: torch.Tensor,
    acc: torch.Tensor,
    m: torch.Tensor,
    lse: torch.Tensor,
    n_q_groups: int,
    scale: float,
    k_group: int,
    n_blocks_in_split: int = 1,
    rope_cos_blk: torch.Tensor | None = None,
    rope_sin_blk: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the k2b Triton kernel (in-kernel lowrank K) for one K block.

    K is unpacked in-kernel per head:  L_h = Us @ V_h.T + RTN_residual_h
    V is pre-dequanted (Python _unrotate already applied) and passed dense.

    Memory layout (from quantize_packed("lowrank_rtn_channel")):
        The codec calls quantize_packed on the COMBINED (S, C) matrix where
        C = h_kv * d_head (to_matrix layout).  The resulting tensors are:
          'Us'        : (blk=S, rank)      — shared left singular vectors
          'V'         : (h_kv*d, rank)     — right singular vectors (all heads)
          'res_Q_int' : (h_kv*d, blk) int8 — RTN residual codes (C × S layout)
          'res_scale' : (h_kv*d, n_groups, 1) — per-group RTN scales

        For each KV head kv (0..h_kv-1), we slice the per-head block:
          us_kv       = Us                                      (blk, rank)  [shared]
          vfac_kv     = V[kv*d : (kv+1)*d, :]                 (d, rank)
          res_int_kv  = res_Q_int[kv*d : (kv+1)*d, :]         (d, blk) int8
          res_sc_kv   = res_scale[kv*d : (kv+1)*d, :, 0]      (d, n_groups) fp16

        The per-head K reconstruction in-kernel:
          L_h    = tl.dot(us_kv, vfac_kv.T)         (blk, d)
          R_h    = RTN dequant(res_int_kv, res_sc_kv) (blk, d)
          K_h    = L_h + R_h

    Args:
        q:                  (n_q_heads, 1, d) fp16  — d = d_head
        kpacked:            packed K dict (lowrank_rtn_channel format)
        V_kv:               (h_kv, blk, d) fp16 — pre-dequanted V values
        acc / m / lse:      running carry tensors
        n_q_groups:         n_q_heads // h_kv
        scale:              attention scale (1/sqrt(d_head))
        k_group:            RTN group size for K's residual (S must be divisible)
        n_blocks_in_split:  do_not_specialize arg (prevents recompile per seq_len)

    Returns updated (acc, m, lse).
    """
    _require_triton()
    n_q_heads, n_q, d = q.shape  # d = d_head
    h_kv, blk, _d = V_kv.shape
    assert n_q == 1, "decode only"
    assert _d == d, f"V_kv d_head={_d} != q d_head={d}"

    # Unpack K tensors from the combined (S, C=h_kv*d) matrix layout
    Us = kpacked["Us"]  # (blk, rank) — shared left singular vectors
    Vfac_full = kpacked["V"]  # (h_kv*d, rank) — right singular vectors
    res_Q_int_full = kpacked["res_Q_int"]  # (h_kv*d, blk) int8
    res_scale_full = kpacked["res_scale"]  # (h_kv*d, n_groups, 1)

    # Validate C dim = h_kv * d
    C = Vfac_full.shape[0]
    assert C == h_kv * d, (
        f"Vfac C={C} != h_kv*d={h_kv}*{d}={h_kv * d}. "
        "Expected lowrank_rtn_channel packed on the to_matrix (S, h_kv*d) layout."
    )

    dev = q.device

    def _to_dev(t: torch.Tensor) -> torch.Tensor:
        return t.to(dev) if t.device != dev else t

    # .contiguous() is REQUIRED: the codec produces these via SVD and can return
    # NON-contiguous tensors (e.g. Us arrives with strides (1, rank) — a transposed
    # view). The kernel indexes with row-major offset arithmetic (b_idx*rank + r_idx),
    # so a non-contiguous input is read with the wrong strides -> garbage K
    # reconstruction (confirmed on GH200: Us strides (1,64) -> us[0,1] read wrong).
    # .to(fp16) alone does NOT guarantee contiguity (no-op when already fp16).
    Us = _to_dev(Us).to(torch.float16).contiguous()
    Vfac_full = _to_dev(Vfac_full).to(torch.float16).contiguous()
    res_Q_int_full = _to_dev(res_Q_int_full).contiguous()  # keep int8
    res_scale_full = _to_dev(res_scale_full).contiguous()

    rank = Us.shape[1]
    n_groups = blk // k_group
    assert n_groups > 0, f"blk={blk} < k_group={k_group}"
    assert blk % k_group == 0, f"blk={blk} not divisible by k_group={k_group}"

    # res_scale: (h_kv*d, n_groups, 1) → (h_kv*d, n_groups) fp16
    res_scale_mat = res_scale_full.view(C, n_groups).to(torch.float16)

    # Lay out carry buffers as (h_kv, n_q_groups, d/1) — zeros, already contiguous.
    q_v = q.view(h_kv, n_q_groups, d)  # (h_kv, G, d)
    acc_buf = acc.view(h_kv, n_q_groups, d)  # fp16 — contiguous (from torch.zeros)
    m_buf = m.view(h_kv, n_q_groups).float()  # fp32 — contiguous
    lse_buf = lse.view(h_kv, n_q_groups).float()  # fp32 — contiguous

    # V_kv is (h_kv, blk, d_head) — fresh allocation from _v_dequant, contiguous.
    V_buf = V_kv

    # Us is shared across all KV heads — made contiguous above (the SVD output is
    # non-contiguous; the .contiguous() at the _to_dev cast is load-bearing).
    us_kv = Us

    # In-kernel RoPE setup: cos/sin for this block (contiguous fp16, shared across
    # heads) + an HBM scratch buffer for the rotate_half gather. HAS_ROPE drives the
    # kernel's optional RoPE branch. Dummy pointers (V_buf) are passed when no RoPE.
    has_rope = rope_cos_blk is not None
    if has_rope:
        cos_blk = _to_dev(rope_cos_blk).to(torch.float16).contiguous()  # (blk, d)
        sin_blk = _to_dev(rope_sin_blk).to(torch.float16).contiguous()  # (blk, d)

    for kv in range(h_kv):
        q_kv = q_v[
            kv
        ].contiguous()  # (G, d) query rows for this KV head — view, needs .contiguous()

        # Slice per-head K factors from the combined (h_kv*d, ...) tensors.
        # Head kv occupies channel rows [kv*d : (kv+1)*d].
        lo, hi = kv * d, (kv + 1) * d
        vfac_kv = Vfac_full[lo:hi, :].contiguous()  # (d, rank) fp16
        res_int_kv = res_Q_int_full[lo:hi, :].contiguous()  # (d, blk) int8
        res_sc_kv = res_scale_mat[lo:hi, :].contiguous()  # (d, n_groups) fp16

        # .contiguous() REQUIRED: V_kv comes from from_matrix (a permute/reshape) and
        # is NON-contiguous at h_kv>1 (V_kv[kv] strides (h_kv*d, 1), not (d, 1)). The
        # kernel reads v_ptr + b_idx*d + d_idx assuming row-major (d,1), so a
        # non-contiguous slice reads the WRONG (interleaved) V — m/scores stay right
        # (K-based) but p@v is wrong (confirmed on GH200: head outputs 0.5+ off at h_kv=2).
        v_kv = V_buf[kv].contiguous()  # (blk, d) fp16

        acc_kv = acc_buf[kv]  # (G, d) fp16 — written in-place by Triton
        m_kv = m_buf[kv]  # (G,)  fp32
        lse_kv = lse_buf[kv]  # (G,)  fp32

        # cos/sin: real tensors when has_rope, else dummy (v_kv) — the kernel only
        # dereferences them under the HAS_ROPE constexpr branch.
        cos_arg = cos_blk if has_rope else v_kv
        sin_arg = sin_blk if has_rope else v_kv

        _k2b_softmax_block_kernel[(n_q_groups,)](
            us_kv,
            vfac_kv,
            res_int_kv,
            res_sc_kv,
            v_kv,
            q_kv,
            acc_kv,
            m_kv,
            lse_kv,
            cos_arg,
            sin_arg,
            float(scale),
            int(n_blocks_in_split),
            d=d,
            n_q_groups=n_q_groups,
            blk=blk,
            rank=rank,
            group=k_group,
            HAS_ROPE=has_rope,
        )

    # Reconstruct carry tensors from updated buffers
    acc_new = acc_buf.view(n_q_heads, 1, d)
    m_new = m_buf.view(n_q_heads, 1, 1)
    lse_new = lse_buf.view(n_q_heads, 1, 1)
    return acc_new, m_new, lse_new


# ---------------------------------------------------------------------------
# Split-KV helpers: partition + merge (3b)
# ---------------------------------------------------------------------------


def _partition_blocks(
    k_blocks: list,
    v_blocks: list,
    num_splits: int,
) -> list[tuple[list, list]]:
    """Partition (k_blocks, v_blocks) into num_splits contiguous ranges.

    If len(blocks) < num_splits, some splits get zero blocks.  The caller
    handles empty splits by detecting them before the kernel launch.

    Returns a list of (k_split, v_split) pairs of length num_splits.
    """
    n = len(k_blocks)
    # Ceiling-division chunk sizes so all blocks are covered.
    chunk = math.ceil(n / num_splits) if n > 0 else 1
    splits = []
    for s in range(num_splits):
        lo = s * chunk
        hi = min(lo + chunk, n)
        splits.append((k_blocks[lo:hi], v_blocks[lo:hi]))
    return splits


def _merge_partials(
    partial_accs: list[torch.Tensor],
    partial_ms: list[torch.Tensor],
    partial_lses: list[torch.Tensor],
) -> torch.Tensor:
    """Merge per-split (acc_i, m_i, lse_i) into the final normalized output.

    Implements the standard online-softmax combine across splits (base-e):

        m   = max_i(m_i)                            # global running max
        l   = sum_i(lse_i * exp(m_i - m))           # re-scaled lse sum
        out = sum_i(acc_i * exp(m_i - m)) / l       # re-scaled acc sum, normalized

    CORRECTNESS INVARIANT (num_splits=1):
        m   = m_0
        l   = lse_0 * exp(m_0 - m_0) = lse_0
        out = acc_0 * 1 / lse_0 = acc_0 / lse_0
        => Bit-identical to 3a's final division `acc / lse`.

    AT MULTIPLE SPLITS:
        This is exactly online_softmax_update applied across the split axis,
        giving the same result as if all blocks had been processed serially.

    BASE-E NOTE: partial_lse is the raw unnormalized sum-of-softmax-weights
    (lse in online_softmax_update — not the log of that sum).  The correction
    exp(m_i - m) is base-e.  Do NOT use exp2/log2 here (3a kernel is base-e).

    Args:
        partial_accs:  list of (n_q_heads, 1, d) fp32/fp16 — pre-normalized acc
        partial_ms:    list of (n_q_heads, 1, 1) fp32 — per-split running max
        partial_lses:  list of (n_q_heads, 1, 1) fp32 — per-split lse denominator

    Returns:
        out: (n_q_heads, 1, d) fp16 — merged normalized attention output
    """
    # Stack to (num_splits, n_q_heads, 1, d/1) for vectorized ops.
    # Keep fp32 throughout to avoid fp16 saturation during accumulation.
    accs = torch.stack([a.float() for a in partial_accs], dim=0)  # (S, H, 1, d)
    ms = torch.stack([m.float() for m in partial_ms], dim=0)  # (S, H, 1, 1)
    lses = torch.stack([lse_t.float() for lse_t in partial_lses], dim=0)  # (S, H, 1, 1)

    # Global max across splits — shape (1, H, 1, 1) -> broadcast over S
    m_global = ms.amax(dim=0, keepdim=True)  # (1, H, 1, 1)

    # Rescaling factors per split: exp(m_i - m_global)
    scales = torch.exp(ms - m_global)  # (S, H, 1, 1) — base-e, matches 3a

    # Merged denominator: sum_i(lse_i * exp(m_i - m))
    l_merged = (lses * scales).sum(dim=0)  # (H, 1, 1)

    # Merged numerator: sum_i(acc_i * exp(m_i - m))  — (S, H, 1, d) * (S, H, 1, 1)
    acc_merged = (accs * scales).sum(dim=0)  # (H, 1, d)

    # Normalize and return in fp16 (matches 3a's `acc / lse.to(q.dtype)`)
    return (acc_merged / l_merged).to(torch.float16)


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
    num_splits: int = 1,
    v_group: int | None = None,
    v_seed: int | None = None,
) -> torch.Tensor:
    """Triton decode attention: online-softmax over packed KV blocks.

    Same call shape as chunked_dequant_attention (minus query_abs_start —
    this is decode-only, n_q==1).  Produces (n_q_heads, n_q, d).

    Stage 3a: dequant stays in PyTorch (dequant_packed + from_matrix +
    apply_rope); only the online-softmax contraction runs in Triton.
    Each block:  PyTorch dequant -> Triton _online_softmax_block_kernel.
    Carry (acc, m, lse) flows between launches in Python/PyTorch tensors.

    Stage 3b (num_splits > 1):
    The block list is partitioned into num_splits contiguous ranges.  Each
    split runs the 3a serial-online-softmax independently, producing a partial
    (acc_i, m_i, lse_i) with acc_i NOT yet divided by lse_i.  _merge_partials
    then combines them via the standard online-softmax combine (base-e).

    num_splits=1 (default) is BACK-COMPATIBLE with 3a: the merge with a single
    partial reduces exactly to acc_0 / lse_0 (see _merge_partials docstring).

    Stage 3c (k_arm="lowrank_rtn_channel"):
    K lowrank reconstruction runs IN-KERNEL via _k2b_softmax_block_kernel:
        K = tl.dot(Us, V_fac.T) + RTN_residual_dequant
    V is dequanted in Python (_unrotate applied per block, deferred from kernel)
    and passed dense to the contraction kernel.

    v_group / v_seed: allow K and V to use different quantization params.
    When not provided, they default to group / seed (3a/3b back-compat).

    GQA: one Triton launch per KV head per block (fusing into one launch
    per block is a 3c optimisation).

    Args:
        q:          (n_q_heads, 1, d) fp16 — single decode query token
        k_blocks:   list of (packed_dict, start, end) — packed KV key blocks
        v_blocks:   list of (packed_dict, start, end) — packed KV value blocks
        k_arm:      codec arm name for keys (e.g. "rtn_token", "lowrank_rtn_channel")
        v_arm:      codec arm name for values (e.g. "rtn_token", "turboquant_mse")
        group:      K quantization group size (also V group if v_group not provided)
        seed:       K quantization seed (also V seed if v_seed not provided)
        k_pre_rope: if True, apply RoPE to K at read time (pre-RoPE keys)
        rope_cos:   (S, d) fp16|fp32 — cosine table, sliced [start:end] per block
        rope_sin:   (S, d) fp16|fp32 — sine table
        k_tail:     (h_kv, tail_len, d) fp16|None — unquantized residual window
        v_tail:     (h_kv, tail_len, d) fp16|None
        n_q_groups: n_q_heads // h_kv (GQA group count)
        scale:      attention scale, typically 1/sqrt(d)
        num_splits: number of KV-block splits for decode parallelism (default 1).
                    1 = 3a serial path (back-compatible, bit-identical to 3a).
                    >1 = 3b split path: partition blocks, merge partials.
                    The tail block (k_tail/v_tail) is always processed in split 0.
        v_group:    V quantization group size (defaults to group if None).
                    Allows K and V to use different packed formats.
        v_seed:     V quantization seed (defaults to seed if None).

    Returns:
        (n_q_heads, 1, d) fp16 attention output.
    """
    _require_triton()
    from bmx.cache.codecs import dequant_packed

    n_q_heads, n_q, d = q.shape
    assert n_q == 1, (
        "triton_decode_attention is decode-only (n_q==1); "
        "prefill stays on the flash-SDPA path in chunked_dequant_attention."
    )
    h_kv = n_q_heads // n_q_groups

    # Resolve V params — default to K's for 3a/3b back-compat
    _v_group = v_group if v_group is not None else group
    _v_seed = v_seed if v_seed is not None else seed

    # k2b path flag: K uses lowrank_rtn_channel (in-kernel lowrank unpack)
    _k2b = k_arm == "lowrank_rtn_channel"

    def _init_carry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fresh (acc, m, lse) carry tensors on q's device."""
        acc_ = torch.zeros(n_q_heads, n_q, d, dtype=q.dtype, device=q.device)
        m_ = torch.full(
            (n_q_heads, n_q, 1), float("-inf"), dtype=torch.float32, device=q.device
        )
        lse_ = torch.zeros(n_q_heads, n_q, 1, dtype=torch.float32, device=q.device)
        return acc_, m_, lse_

    def _dequant_block(
        kpacked: dict, vpacked: dict, start: int, end: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize one KV block pair → (K_kv, V_kv) on q's device, fp16.

        For 3a/3b RTN arms: full Python dequant for both K and V.
        For k2b (k_arm=lowrank_rtn_channel): K stays packed (returned as-is);
        only V is dequanted here.  The k2b path calls _k2b_block_kernel_launch
        with kpacked directly (K is unpacked in-kernel).
        """
        # Packed codes are stored CPU-resident in the cache (to save GPU memory), so
        # dequant_packed/from_matrix produce CPU tensors — move to q.device for the
        # kernel (a CPU pointer to a Triton kernel raises "cannot be accessed").
        if not _k2b:
            # 3a/3b RTN path: Python dequant for both K and V
            K_kv = from_matrix(
                dequant_packed(k_arm, kpacked, seed=seed, group=group), h_kv
            ).to(device=q.device, dtype=q.dtype)
            if k_pre_rope:
                # RTN path applies RoPE in PyTorch (not in-kernel); cos/sin may be CPU.
                K_kv = apply_rope(
                    K_kv,
                    rope_cos[start:end].to(q.device),
                    rope_sin[start:end].to(q.device),
                )
            V_kv = from_matrix(
                dequant_packed(v_arm, vpacked, seed=_v_seed, group=_v_group), h_kv
            ).to(device=q.device, dtype=q.dtype)
            return K_kv, V_kv
        else:
            # k2b path: K stays packed (moved to device in _k2b_block_kernel_launch);
            # V is Python-dequanted here.
            if v_arm == "turboquant_mse":
                V_kv = _v_dequant_turboquant_mse(
                    vpacked, h_kv, _v_seed, q.device, q.dtype
                )
            else:
                V_kv = from_matrix(
                    dequant_packed(v_arm, vpacked, seed=_v_seed, group=_v_group), h_kv
                ).to(device=q.device, dtype=q.dtype)
            # K dequant is deferred to _k2b_block_kernel_launch
            return kpacked, V_kv  # type: ignore[return-value]

    def _run_split(
        kb_split: list, vb_split: list, with_tail: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the 3a/3c serial online-softmax over one split's blocks.

        Returns pre-normalization (acc, m, lse) — do NOT divide acc by lse here.
        The caller (_merge_partials) handles the final normalization.

        with_tail: if True, also consume the k_tail/v_tail residual window.
        NOTE: Only one split (split 0) receives the tail (avoids double-counting).
        """
        acc, m, lse = _init_carry()
        n_blocks_in_split = len(kb_split) + (
            1 if with_tail and k_tail is not None and k_tail.shape[1] > 0 else 0
        )

        for (kpacked, start, end), (vpacked, _vs, _ve) in zip(kb_split, vb_split):
            K_or_packed, V_kv = _dequant_block(kpacked, vpacked, start, end)
            if _k2b:
                # K_or_packed is the raw packed dict; K is unpacked in-kernel.
                # When keys are pre-RoPE, pass this block's cos/sin slice so the
                # kernel applies RoPE to the lowrank-reconstructed K in-register
                # (in-kernel rotate_half via HBM-scratch gather — verified bit-exact
                # vs apply_rope). Pre-RoPE subspace design: keys quantized before RoPE,
                # RoPE applied at read (CLAUDE.md: "quantize keys PRE-RoPE").
                rope_cos_blk = rope_cos[start:end] if k_pre_rope else None
                rope_sin_blk = rope_sin[start:end] if k_pre_rope else None
                acc, m, lse = _k2b_block_kernel_launch(
                    q,
                    K_or_packed,
                    V_kv,
                    acc,
                    m,
                    lse,
                    n_q_groups,
                    scale,
                    k_group=group,
                    n_blocks_in_split=n_blocks_in_split,
                    rope_cos_blk=rope_cos_blk,
                    rope_sin_blk=rope_sin_blk,
                )
            else:
                acc, m, lse = _online_block_kernel_launch(
                    q,
                    K_or_packed,
                    V_kv,
                    acc,
                    m,
                    lse,
                    n_q_groups,
                    scale,
                    n_blocks_in_split=n_blocks_in_split,
                )

        if with_tail and k_tail is not None and k_tail.shape[1] > 0:
            assert v_tail is not None, "v_tail must be set when k_tail is set"
            acc, m, lse = _online_block_kernel_launch(
                q,
                k_tail.to(device=q.device, dtype=q.dtype),
                v_tail.to(device=q.device, dtype=q.dtype),
                acc,
                m,
                lse,
                n_q_groups,
                scale,
                n_blocks_in_split=n_blocks_in_split,
            )

        return acc, m, lse

    # ------------------------------------------------------------------
    # num_splits=1 fast path: 3a-compatible serial loop (back-compat)
    # ------------------------------------------------------------------
    if num_splits == 1:
        acc, _m, lse = _run_split(k_blocks, v_blocks, with_tail=True)
        # 3a-identical normalization path (preserves bit-identity for num_splits=1)
        return acc / lse.to(q.dtype)

    # ------------------------------------------------------------------
    # num_splits>1: split-KV parallel path (3b)
    # ------------------------------------------------------------------
    splits = _partition_blocks(k_blocks, v_blocks, num_splits)

    partial_accs = []
    partial_ms = []
    partial_lses = []

    for s_idx, (kb_split, vb_split) in enumerate(splits):
        # Skip empty splits (when n_blocks < num_splits).
        # Empty splits contribute zero to the merge — handled by sentinel m=-inf, lse=0.
        # But that would break the merge (0/0 if ALL splits empty).
        # Instead, only add non-empty splits to the partial lists.
        with_tail = s_idx == 0  # tail always goes to split 0
        has_blocks = len(kb_split) > 0
        has_tail_here = with_tail and k_tail is not None and k_tail.shape[1] > 0
        if not has_blocks and not has_tail_here:
            # Truly empty split — skip to avoid degenerate lse=0
            continue

        acc_i, m_i, lse_i = _run_split(kb_split, vb_split, with_tail=with_tail)
        partial_accs.append(acc_i)
        partial_ms.append(m_i)
        partial_lses.append(lse_i)

    if not partial_accs:
        # Edge case: no blocks and no tail — return zeros (empty context)
        return torch.zeros(n_q_heads, n_q, d, dtype=q.dtype, device=q.device)

    if len(partial_accs) == 1:
        # Only one non-empty split: skip merge overhead, normalize directly.
        return partial_accs[0] / partial_lses[0].to(q.dtype)

    return _merge_partials(partial_accs, partial_ms, partial_lses)


# ---------------------------------------------------------------------------
# Stage 3d: CUDA-graph-safe decode path
#
# THE PROBLEM with the existing decode path for CUDA graphs:
#   - k_blocks is a Python list; its length (seq_len / n_blocks) is a Python int.
#   - A CUDA graph captures a fixed kernel launch grid and fixed kernel args.
#   - If seq_len (or n_blocks) is a Python int kernel arg, every decode step
#     with a different seq_len requires a DIFFERENT compiled kernel specialization.
#     Replaying an old capture at a new seq_len would use the WRONG (old) length.
#
# THE FIX (vLLM pattern, confirmed via DeepWiki):
#   - Pre-stack KV blocks into a device tensor (max_blocks, h_kv, blk, d).
#     The graph captures the pointer to this buffer; in-place updates are visible
#     to replays without re-capture.
#   - Pass seq_len as a DEVICE TENSOR (int32 scalar on CUDA).
#     The kernel reads `seq_len_ptr[0]` at runtime — NOT from the launch args.
#     Updating seq_len_dev.fill_(new_len) in-place between replays causes the
#     replay to use the new length without re-capture.
#   - Launch grid is FIXED = (max_blocks, h_kv, n_q_groups) — always the same shape.
#     Blocks beyond seq_len_dev[0] are masked (kernel exits early).
#
# SCOPE OF 3d (honest):
#   - RTN path (rtn_token arm): K and V are pre-dequanted into k_stacked/v_stacked
#     before capture. In-place updates to the stacked tensors between replays make
#     new KV blocks visible. This is the minimum satisfying the gate.
#   - k2b path (lowrank_rtn_channel): K's packed dict (Us, V, res_Q_int, res_scale)
#     is a heterogeneous Python dict, not a single contiguous device tensor. Stacking
#     these across max_blocks into a graph-capturable device tensor requires a larger
#     refactor (a paged block table for each factor). DEFERRED from 3d.
#   - Tail window: fp16 residual window is handled by the caller (not part of the
#     graphable path — it is always small and can be processed outside the graph).
#   - CUDA graph capture/replay test: CUDA-gated. Locally (AMD/no-CUDA) the test
#     skips loud. The test IS a real capture->update->replay->compare-to-fresh test
#     that would catch a Python-int-seqlen implementation.
#
# KEY CORRECTNESS PROPERTY:
#   A Python-int-seqlen implementation would bake seq_len=S0 into the captured
#   kernel specialization. Replaying at S0+k would either: (a) run the S0 kernel
#   (masking at S0, ignoring the extra blocks) → output matches fresh-at-S0 but
#   NOT fresh-at-S0+k, catching the bug; or (b) recompile per step, defeating the
#   purpose. The test asserts replayed ≈ fresh-at-S0+k, catching case (a).
# ---------------------------------------------------------------------------


if TRITON_AVAILABLE:

    @triton.jit(do_not_specialize=["h_kv"])
    def _graphable_decode_kernel(
        # Query: (h_kv, n_q_groups, d) — n_q=1 squeezed, GQA-expanded view
        q_ptr,
        # Pre-stacked KV: (max_blocks, h_kv, blk_size, d) contiguous fp16
        k_stacked_ptr,
        v_stacked_ptr,
        # Carry buffers: (h_kv, n_q_groups, d/1) contiguous
        acc_ptr,  # fp16
        m_ptr,  # fp32
        lse_ptr,  # fp32
        # Device scalar: actual live sequence length (int32, pointer-read)
        seq_len_ptr,
        # Scale factor (1/sqrt(d), fp32 scalar — runtime, not constexpr)
        scale,
        # h_kv: runtime int (do_not_specialize) — passed explicitly to avoid
        # tl.num_programs(1) which is grid-padding-dependent on some hardware.
        h_kv,
        # Fixed-size launch params (constexpr — do NOT include seq_len here)
        blk_size: tl.constexpr,
        d: tl.constexpr,
        n_q_groups: tl.constexpr,
    ):
        """Graph-safe decode online-softmax kernel.

        Grid: (max_blocks, h_kv, n_q_groups) — always the same shape.
        Each program handles ONE (block_idx, kv_head, q_group) triple.

        seq_len_ptr is a DEVICE TENSOR (int32 pointer) — not a Python int.
        The kernel reads seq_len_ptr[0] at runtime. Updating the device tensor
        in-place between graph replays makes the new length visible without
        re-capture. This is the key graph-safety property.

        Block masking: if block_idx * blk_size >= seq_len, this program exits
        early (contributes nothing to the accumulator). The last block is
        partially masked: only tokens [block_start : seq_len] are active.

        do_not_specialize: h_kv is a runtime int (not constexpr) — passed
        explicitly so the stride computation is deterministic and not
        grid-padding-dependent (avoids tl.num_programs(1) assumptions).
        scale is also a runtime float; neither goes in the autotune key.
        """
        block_idx = tl.program_id(0)
        kv_head = tl.program_id(1)
        g = tl.program_id(2)  # query group within kv_head

        # ------------------------------------------------------------------
        # Read live seq_len from device tensor (THE graph-safety invariant).
        # This pointer-read is what the graph captures — not a baked-in int.
        # ------------------------------------------------------------------
        seq_len = tl.load(seq_len_ptr).to(tl.int32)

        block_start = block_idx * blk_size
        # Mask: skip this block entirely if it starts at or beyond seq_len.
        if block_start >= seq_len:
            return

        # ------------------------------------------------------------------
        # Number of active tokens in this block (handles last partial block).
        # ------------------------------------------------------------------
        active = tl.minimum(blk_size, seq_len - block_start)

        d_idx = tl.arange(0, d)
        b_idx = tl.arange(0, blk_size)

        # ------------------------------------------------------------------
        # Load q row: (d,) for query group g within kv_head.
        # q layout: (h_kv, n_q_groups, d).
        # q_ptr[kv_head, g, :] offset = (kv_head * n_q_groups + g) * d
        # ------------------------------------------------------------------
        q_offset = (kv_head * n_q_groups + g) * d
        q_row = tl.load(q_ptr + q_offset + d_idx).to(tl.float32)  # (d,)

        # ------------------------------------------------------------------
        # Load k block: (blk_size, d).
        # k_stacked layout: (max_blocks, h_kv, blk_size, d).
        # k_stacked[block_idx, kv_head, :, :] offset:
        #   = (block_idx * h_kv + kv_head) * blk_size * d
        # h_kv is passed explicitly (do_not_specialize) — deterministic stride,
        # not grid-padding-dependent (avoids tl.num_programs(1) assumption).
        # ------------------------------------------------------------------
        k_base = (block_idx * h_kv + kv_head) * blk_size * d
        k_offsets = b_idx[:, None] * d + d_idx[None, :]  # (blk_size, d)
        # Mask for partial last block: b_idx < active
        k_mask = b_idx < active  # (blk_size,) bool — last block partial mask
        k = tl.load(
            k_stacked_ptr + k_base + k_offsets,
            mask=k_mask[:, None],
            other=0.0,
        ).to(tl.float32)  # (blk_size, d) fp32

        # ------------------------------------------------------------------
        # scores[b] = (sum_dd q[dd]*k[b,dd]) * scale  -> (blk_size,)  (GEMV
        # multiply+sum, not tl.dot — no >=16 min-dim constraint).
        # ------------------------------------------------------------------
        scores = tl.sum(q_row[None, :] * k, axis=1) * scale  # (blk_size,)

        # Mask inactive tokens to -inf so they don't contribute to softmax.
        scores = tl.where(b_idx < active, scores, float("-inf"))

        # ------------------------------------------------------------------
        # ATOMIC online-softmax update.
        #
        # CONCURRENCY NOTE: Multiple programs write the SAME (kv_head, g) carry
        # slot (since we parallelise over block_idx too). This requires atomic
        # read-modify-write on (acc, m, lse). Triton does not have a built-in
        # atomic online-softmax update. Instead, we use the standard approach:
        # each program stores its LOCAL (acc_local, m_local, lse_local) to
        # per-block scratch buffers, and a REDUCTION PASS in Python merges them
        # (identical to _merge_partials in the 3b path).
        #
        # For the graphable path the REDUCTION is done outside the kernel in
        # Python (not CUDA-graph-captured), which is acceptable for 3d — the
        # graph captures the scatter kernel; the reduce is cheap.
        # The key graph-safe property (device-pointer seq_len) is exercised.
        # ------------------------------------------------------------------
        # Carry slot for this (block_idx, kv_head, g):
        # acc_scratch: (max_blocks, h_kv, n_q_groups, d)
        # m_scratch:   (max_blocks, h_kv, n_q_groups)
        # lse_scratch: (max_blocks, h_kv, n_q_groups)
        # ------------------------------------------------------------------
        # FRESH local carry (no read from shared buffer — avoids race).
        m_local = float("-inf")
        lse_local = 0.0

        # Online softmax for this block's scores (base-e):
        m_new = tl.maximum(m_local, tl.max(scores, axis=0))
        alpha = tl.exp(m_local - m_new)  # = 0 when m_local == -inf
        p = tl.where(b_idx < active, tl.exp(scores - m_new), 0.0)
        lse_new = lse_local * alpha + tl.sum(p, axis=0)

        # acc = p @ v:
        v_base = (block_idx * h_kv + kv_head) * blk_size * d
        v = tl.load(
            v_stacked_ptr + v_base + k_offsets,
            mask=k_mask[:, None],
            other=0.0,
        ).to(tl.float32)  # (blk_size, d)
        # acc[dd] = sum_b p[b]*v[b,dd]  (GEMV multiply+sum, not tl.dot)
        acc_local = tl.sum(p[:, None] * v, axis=0)  # (d,)

        # ------------------------------------------------------------------
        # Store (acc_local, m_new, lse_new) to per-block scratch buffers.
        # Layout: [block_idx * (h_kv * n_q_groups) + kv_head * n_q_groups + g]
        # ------------------------------------------------------------------
        scratch_row = block_idx * h_kv * n_q_groups + kv_head * n_q_groups + g
        # acc_ptr in graphable path points to scratch: (max_blocks*h_kv*G, d)
        tl.store(acc_ptr + scratch_row * d + d_idx, acc_local.to(tl.float16))
        tl.store(m_ptr + scratch_row, m_new.to(tl.float32))
        tl.store(lse_ptr + scratch_row, lse_new.to(tl.float32))


def _graphable_reduce(
    acc_scratch: torch.Tensor,
    m_scratch: torch.Tensor,
    lse_scratch: torch.Tensor,
    max_blocks: int,
    h_kv: int,  # noqa: ARG001 — kept for call-site symmetry; folded into n_q_heads
    n_q_groups: int,  # noqa: ARG001 — kept for call-site symmetry; folded into n_q_heads
    n_q_heads: int,
    d: int,
) -> torch.Tensor:
    """Merge per-block scratch buffers into the final attention output.

    acc_scratch: (max_blocks * h_kv * n_q_groups, d) fp16
    m_scratch:   (max_blocks * h_kv * n_q_groups,) fp32
    lse_scratch: (max_blocks * h_kv * n_q_groups,) fp32

    Reshapes to (max_blocks, n_q_heads, 1, d/1) and applies _merge_partials.
    Returns (n_q_heads, 1, d) fp16.
    """
    # Reshape flat scratch → (max_blocks, n_q_heads, 1, d/1) so each split is
    # a (n_q_heads, 1, d/1) slice — the shape _merge_partials expects.
    # n_q_heads = h_kv * n_q_groups; (h_kv, G) dims are already folded in.
    acc_s = acc_scratch.view(max_blocks, n_q_heads, 1, d)  # (B, H, 1, d)
    m_s = m_scratch.view(max_blocks, n_q_heads, 1, 1)  # (B, H, 1, 1)
    lse_s = lse_scratch.view(max_blocks, n_q_heads, 1, 1)  # (B, H, 1, 1)
    return _merge_partials(
        list(acc_s.unbind(0)), list(m_s.unbind(0)), list(lse_s.unbind(0))
    )


def triton_decode_attention_graphable(
    q: torch.Tensor,
    k_stacked: torch.Tensor,
    v_stacked: torch.Tensor,
    seq_len_dev: torch.Tensor,
    *,
    n_q_groups: int,
    scale: float,
) -> torch.Tensor:
    """CUDA-graph-safe decode attention.

    Drop-in for triton_decode_attention on the RTN path where K/V are
    pre-dequanted and pre-stacked into device tensors.

    GRAPH SAFETY INVARIANT:
        seq_len_dev is a DEVICE int32 TENSOR (not a Python int). The kernel
        reads seq_len_dev[0] at runtime via pointer. A captured CUDA graph
        retains the pointer — replaying after `seq_len_dev.fill_(new_len)`
        uses new_len, not the captured-time value. This is the vLLM pattern.

    DEFERRED (out of scope for 3d):
        - k2b path (lowrank_rtn_channel): K's packed dict factors (Us, V,
          res_Q_int, res_scale) are heterogeneous; stacking into a flat device
          tensor requires a paged-block-table refactor. Deferred post-3d.
        - Tail window (fp16 residual): process outside the captured graph.
        - RoPE application inside the graph: apply RoPE to k_stacked before
          capture if k_pre_rope=True.

    Args:
        q:           (n_q_heads, 1, d) fp16 CUDA — single decode query token.
        k_stacked:   (max_blocks, h_kv, blk_size, d) fp16 CUDA — pre-dequanted
                     keys. Slots beyond live blocks may be zero or stale (masked
                     by seq_len_dev).
        v_stacked:   (max_blocks, h_kv, blk_size, d) fp16 CUDA — pre-dequanted
                     values. Same layout.
        seq_len_dev: torch.Tensor, dtype=int32, shape=() or (1,), CUDA device.
                     Holds the LIVE sequence length. Updated in-place each step.
                     Graph captures the pointer; replay reads the updated value.
        n_q_groups:  n_q_heads // h_kv (GQA group count).
        scale:       attention scale (1/sqrt(d)).

    Returns:
        (n_q_heads, 1, d) fp16 — attention output.
    """
    _require_triton()

    n_q_heads, n_q, d = q.shape
    assert n_q == 1, "graphable path is decode-only (n_q==1)"
    # dtype guard BEFORE device guard so a CPU tensor with wrong dtype gets
    # the dtype error — enabling an offline CPU test for the int32 requirement.
    assert seq_len_dev.dtype == torch.int32, (
        f"seq_len_dev must be int32, got {seq_len_dev.dtype}. "
        "This is the device-pointer that the kernel reads for graph safety."
    )
    assert seq_len_dev.is_cuda, "seq_len_dev must be a CUDA tensor"

    max_blocks, h_kv, blk_size, _d = k_stacked.shape
    assert _d == d, f"k_stacked d={_d} != q d={d}"
    assert n_q_heads == h_kv * n_q_groups, (
        f"n_q_heads={n_q_heads} != h_kv={h_kv} * n_q_groups={n_q_groups}"
    )

    # ------------------------------------------------------------------
    # Allocate per-block scratch buffers.
    # These are fixed-size and can be pre-allocated outside the graph.
    # The graph writes into them; the Python reduce reads from them.
    # Shape: (max_blocks * h_kv * n_q_groups, d/1).
    # ------------------------------------------------------------------
    scratch_rows = max_blocks * h_kv * n_q_groups
    acc_scratch = torch.zeros(scratch_rows, d, dtype=torch.float16, device=q.device)
    m_scratch = torch.full(
        (scratch_rows,), float("-inf"), dtype=torch.float32, device=q.device
    )
    lse_scratch = torch.zeros(scratch_rows, dtype=torch.float32, device=q.device)

    # ------------------------------------------------------------------
    # Query layout: (h_kv, n_q_groups, d) — contiguous for kernel indexing.
    # ------------------------------------------------------------------
    q_kv = q.squeeze(1).view(h_kv, n_q_groups, d).contiguous()

    # ------------------------------------------------------------------
    # Launch kernel: grid = (max_blocks, h_kv, n_q_groups) — FIXED.
    # This is the graph-capturable grid; size never changes between replays.
    # ------------------------------------------------------------------
    grid = (max_blocks, h_kv, n_q_groups)
    _graphable_decode_kernel[grid](
        q_kv,
        k_stacked,
        v_stacked,
        acc_scratch,
        m_scratch,
        lse_scratch,
        seq_len_dev,
        float(scale),
        h_kv,
        blk_size=blk_size,
        d=d,
        n_q_groups=n_q_groups,
    )

    # ------------------------------------------------------------------
    # Reduce scratch → output (outside the graph; cheap).
    # ------------------------------------------------------------------
    return _graphable_reduce(
        acc_scratch,
        m_scratch,
        lse_scratch,
        max_blocks,
        h_kv,
        n_q_groups,
        n_q_heads,
        d,
    )


def build_kv_stacked(
    k_blocks: list,
    v_blocks: list,
    *,
    max_blocks: int,
    h_kv: int,
    blk_size: int,
    d: int,
    k_arm: str,
    v_arm: str,
    group: int,
    seed: int,
    v_group: int | None = None,
    v_seed: int | None = None,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-stack dequanted KV blocks into device tensors for the graphable path.

    RTN path only (lowrank/k2b deferred — see triton_decode_attention_graphable
    docstring). Allocates (max_blocks, h_kv, blk_size, d) fp16 tensors and fills
    slots 0..len(k_blocks)-1 from the block list. Remaining slots are zero.

    Args:
        k_blocks:   list of (packed_dict, start, end) — RTN-quantized keys.
        v_blocks:   list of (packed_dict, start, end) — RTN-quantized values.
        max_blocks: total slots (>= len(k_blocks)); sets the fixed graph grid.
        h_kv:       number of KV heads.
        blk_size:   tokens per block.
        d:          head dim.
        k_arm:      codec arm for keys (must be RTN; k2b deferred).
        v_arm:      codec arm for values.
        group:      K quantization group size.
        seed:       K quantization seed.
        v_group:    V quantization group size (defaults to group).
        v_seed:     V quantization seed (defaults to seed).
        device:     target CUDA device.

    Returns:
        k_stacked: (max_blocks, h_kv, blk_size, d) fp16
        v_stacked: (max_blocks, h_kv, blk_size, d) fp16
    """
    from bmx.cache.codecs import dequant_packed

    if k_arm == "lowrank_rtn_channel":
        raise NotImplementedError(
            "build_kv_stacked: k2b (lowrank_rtn_channel) path is deferred from 3d. "
            "Stack the RTN arm tensors instead, or use triton_decode_attention for k2b."
        )

    _v_group = v_group if v_group is not None else group
    _v_seed = v_seed if v_seed is not None else seed

    k_stacked = torch.zeros(
        max_blocks, h_kv, blk_size, d, dtype=torch.float16, device=device
    )
    v_stacked = torch.zeros(
        max_blocks, h_kv, blk_size, d, dtype=torch.float16, device=device
    )

    for i, ((kpacked, _ks, _ke), (vpacked, _vs, _ve)) in enumerate(
        zip(k_blocks, v_blocks)
    ):
        assert i < max_blocks, (
            f"more blocks ({len(k_blocks)}) than max_blocks ({max_blocks})"
        )
        K_mat = dequant_packed(k_arm, kpacked, seed=seed, group=group)
        V_mat = dequant_packed(v_arm, vpacked, seed=_v_seed, group=_v_group)
        K_kv = from_matrix(K_mat, h_kv).to(torch.float16)  # (h_kv, blk, d)
        V_kv = from_matrix(V_mat, h_kv).to(torch.float16)
        k_stacked[i] = K_kv.to(device)
        v_stacked[i] = V_kv.to(device)

    return k_stacked, v_stacked
