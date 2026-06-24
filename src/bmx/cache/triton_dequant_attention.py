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


# ---------------------------------------------------------------------------
# Phase 3a: FUSED decode kernel — single launch, internal KV-block loop.
#
# THE REWRITE. The per-block Python launch loop (triton_decode_attention,
# _online_block_kernel_launch) is the dominant suboptimality: n_blocks * h_kv
# launches per decode step, carry threaded through PyTorch between launches.
# Baseline (results/k3_triton_decode/...1767dcf): ~1.6x SLOWER than chunked
# PyTorch at every context (launch overhead dominates).
#
# This fused kernel does ONE launch and loops over ALL KV blocks INTERNALLY,
# carrying (m, lse, acc) in fp32 registers, one output write. Design grounded in
# the brain consult (personal-brain both layers + deepwiki flashinfer/vLLM/triton;
# see SDD ledger "VM PHASE 3a — design locked"):
#   - GQA GROUP FUSION: each program handles ONE kv_head and ALL n_q_groups query
#     heads. The KV tile is loaded ONCE per block and reused across the whole group
#     -> 4x less KV HBM traffic (the KV load IS the whole cost at M=1 decode).
#     (vLLM "3D kernel": process all Q heads of a KV head together for cache reuse.)
#   - REGISTER CARRY: acc[G, D], m[G], lse[G] live in fp32 registers across the
#     whole block loop (acc = 4*128 fp32 = 2KB/program, trivial vs 256KB SM regs).
#     fp16 accumulation over 512-2000 blocks would lose precision (FlashAttention).
#   - FIRST-BLOCK -inf: m init -inf, lse/acc init 0. On block 0, alpha =
#     exp(-inf - m_new) = 0 annihilates the garbage init (lse=0*0+sum p,
#     acc=0*0+pv). No special-case — flashinfer relies on exactly this.
#   - 128-bit LDG.E.128 loads are AUTOMATIC from contiguous fp16 D=128 inner axis;
#     eviction_policy="evict_first" makes KV a read-once L2 stream (read exactly
#     once per decode step) so it doesn't evict the reused weight working set.
#   - GEMV (multiply + tl.sum), NOT tl.dot: decode is M=1, bandwidth-bound; tl.dot
#     is useless at M=1 and has a min-dim>=16 constraint.
#
# num_splits=1 first cut (this kernel). Split-KV (grid z-dim + merge kernel) is
# 3a.2 — needed for SM utilization at long context (no-split = 32/132 SMs on GH200).
#
# Correctness bar: max_abs vs naive_dense_attention < 1e-2 (expect ~2-3e-4 at fp16).
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:
    # The kernel iterates ONE stored block (blk_size rows) per loop iter — the unit
    # contiguous in memory for a single head — so there's no BLOCK_N tile to tune.
    # Tune only num_warps (memory-bound tops ~4-8) and num_stages (the software
    # pipeline that overlaps the next block's loads with current compute).
    _FUSED_AUTOTUNE_CONFIGS = [
        _TritonConfig({}, num_warps=2, num_stages=2),
        _TritonConfig({}, num_warps=4, num_stages=2),
        _TritonConfig({}, num_warps=4, num_stages=3),
        _TritonConfig({}, num_warps=8, num_stages=3),
        _TritonConfig({}, num_warps=8, num_stages=4),
    ]

    @triton.autotune(configs=_FUSED_AUTOTUNE_CONFIGS, key=["d", "n_q_groups"])
    @triton.jit(do_not_specialize=["seq_len", "num_splits"])
    def _fused_decode_kernel(
        # Query: (h_kv, n_q_groups, d) — n_q=1 squeezed, GQA-grouped view
        q_ptr,
        # Pre-stacked dense KV: (max_blocks, h_kv, blk_size, d) contiguous fp16
        k_stacked_ptr,
        v_stacked_ptr,
        # Partial outputs (one per split), written by every program:
        #   acc_part: (num_splits, h_kv, G, d) fp32 — pre-normalization weighted V
        #   m_part:   (num_splits, h_kv, G)    fp32 — per-split running max
        #   lse_part: (num_splits, h_kv, G)    fp32 — per-split denominator
        acc_part_ptr,
        m_part_ptr,
        lse_part_ptr,
        # Live sequence length + split count (Python int runtime args;
        # do_not_specialize so the kernel is NOT recompiled per decode step).
        seq_len,
        num_splits,
        scale,  # fp32 1/sqrt(d)
        h_kv: tl.constexpr,  # number of KV heads (stacked stride; fixed model dim)
        blk_size: tl.constexpr,  # tokens per stored block (== build_kv_stacked blk)
        d: tl.constexpr,
        n_q_groups: tl.constexpr,  # G query heads per KV head (group-fused)
        GPAD: tl.constexpr,  # G padded up to >=16 so tl.dot's M dim is legal
        USE_DOT: tl.constexpr,  # tl.dot path (dims>=16) vs broadcast cube (tiny test)
    ):
        """Fused decode online-softmax with split-KV: one program per (kv_head, split).

        Grid: (h_kv, num_splits) — program (kv, s) walks ONLY its contiguous token
        slice of the KV length and accumulates ALL n_q_groups query heads in
        registers, writing a PARTIAL (acc, m, lse) to scratch. A second merge kernel
        combines the num_splits partials per query head (Triton has no global
        barrier -> the merge must be a separate launch; vLLM "3D kernel" pattern).

        Token slice for split s: [s*tokens_per_split, (s+1)*tokens_per_split) ∩
        [0, seq_len). tokens_per_split = ceil(seq_len / num_splits) rounded UP to a
        multiple of blk_size so split boundaries land on stored-block edges (keeps
        loads block-aligned). A split whose slice is empty writes m=-inf, lse=0 (the
        merge skips it via the same alpha=exp(-inf)=0 mechanism).

        Layout (all contiguous):
          q_ptr:        (h_kv, G, d)          — q[kv, g, :] = q_ptr[(kv*G+g)*d + :]
          k/v_stacked:  (max_blocks, h_kv, blk_size, d)
                        token t, head kv -> base = ((t//blk)*h_kv + kv)*blk*d + (t%blk)*d
          acc_part:     (num_splits, h_kv, G, d) — [(s*h_kv+kv)*G+g]*d + dd
          m/lse_part:   (num_splits, h_kv, G)    — (s*h_kv+kv)*G + g

        The K/V load is one stored block (blk_size, d) via a FLAT contiguous run
        (block_base + arange(blk_size*d)) so Triton proves contiguity and emits
        128-bit LDG.E.128; evict_first marks it read-once.
        """
        kv = tl.program_id(0)  # which KV head
        s = tl.program_id(1)  # which split
        d_idx = tl.arange(0, d)  # (d,)
        gp_idx = tl.arange(0, GPAD)  # (GPAD,) padded query-head index
        gp_valid = gp_idx < n_q_groups  # (GPAD,) real rows (rest are pad)
        r_idx = tl.arange(0, blk_size)  # (blk,) row offsets within one stored block
        # Flat element offsets for one contiguous (blk_size, d) stored block: as a
        # 1-D run [0, blk_size*d) this lets Triton's AxisInfoAnalysis prove
        # contiguity -> 128-bit LDG.E.128 loads. Per-row div/mod offset math
        # DEFEATS that proof -> scalar loads. Splits are block-aligned so one
        # stored block is fully contiguous for a single head.
        flat_idx = tl.arange(0, blk_size * d)  # (blk*d,) contiguous run

        # ------------------------------------------------------------------
        # This split's contiguous token range, block-aligned.
        # tokens_per_split = ceil(seq_len/num_splits) rounded UP to a blk_size
        # multiple so each split starts/ends on stored-block boundaries.
        # ------------------------------------------------------------------
        raw = (seq_len + num_splits - 1) // num_splits  # ceil
        tokens_per_split = ((raw + blk_size - 1) // blk_size) * blk_size  # round to blk
        split_start = s * tokens_per_split
        split_end = tl.minimum(split_start + tokens_per_split, seq_len)
        first_block = split_start // blk_size  # this split's first stored block
        last_block = (split_end + blk_size - 1) // blk_size  # exclusive

        head_stride = h_kv * blk_size * d  # advance one stored block (skip h_kv heads)
        kv_head_off = kv * blk_size * d  # this head's offset within a stored block

        # ------------------------------------------------------------------
        # Load q PADDED to GPAD rows (GPAD>=16 so tl.dot's M dim is legal). Pad
        # rows (g>=n_q_groups) load 0 -> their scores are 0, harmless; we mask the
        # pad rows out at the final store. The contraction is tl.dot, NOT a
        # broadcast cube: the (G, blk, d) fp32 cube caused register pressure that
        # capped per-program bandwidth (~3 GB/s vs ~14 GB/s for tl.dot in the
        # micro-probe). tl.dot needs M,N,K>=16; GPAD>=16, blk>=16, d>=16 satisfy it.
        # ------------------------------------------------------------------
        q_off = (kv * n_q_groups + gp_idx)[:, None] * d + d_idx[None, :]  # (GPAD, d)
        q_rows = tl.load(q_off + q_ptr, mask=gp_valid[:, None], other=0.0).to(
            tl.float32
        )  # (GPAD, d)

        # ------------------------------------------------------------------
        # Register carry for the padded group (fp32 across the entire loop).
        # ------------------------------------------------------------------
        m = tl.full((GPAD,), float("-inf"), tl.float32)  # (GPAD,) running max
        lse = tl.zeros((GPAD,), tl.float32)  # (GPAD,) running denom
        acc = tl.zeros((GPAD, d), tl.float32)  # (GPAD, d) running weighted V

        # ------------------------------------------------------------------
        # Internal loop: ONE stored block per iteration (the unit contiguous in
        # memory for a single head). Empty split -> range empty -> 0 iters ->
        # carry stays (m=-inf, lse=0, acc=0): the empty-split partial.
        # ------------------------------------------------------------------
        for blk in range(first_block, last_block):
            block_base = blk * head_stride + kv_head_off  # scalar contiguous base
            row_abs = blk * blk_size + r_idx  # (blk,) absolute token per row
            tile_mask = row_abs < split_end  # (blk,) last block may be partial

            # Flat contiguous load -> reshape to (blk, d). evict_first: read-once.
            k = tl.reshape(
                tl.load(
                    k_stacked_ptr + block_base + flat_idx,
                    eviction_policy="evict_first",
                ).to(tl.float32),
                (blk_size, d),
            )  # (blk, d)

            # scores[g, n] = scale * sum_dd q[g, dd] * k[n, dd]  -> (GPAD, blk).
            # tl.dot when dims allow (M=GPAD>=16, N=blk>=16, K=d>=16); else the
            # broadcast cube (only the tiny offline test model has d<16 — real
            # models are d>=64). USE_DOT is a constexpr so one branch is compiled.
            if USE_DOT:
                scores = tl.dot(q_rows, tl.trans(k)) * scale  # (GPAD, blk)
            else:
                scores = (
                    tl.sum(q_rows[:, None, :] * k[None, :, :], axis=2) * scale
                )  # (GPAD, blk)
            scores = tl.where(tile_mask[None, :], scores, float("-inf"))  # (GPAD, blk)

            # Online-softmax update (base-e):
            m_tile = tl.max(scores, axis=1)  # (GPAD,)
            m_new = tl.maximum(m, m_tile)  # (GPAD,)
            alpha = tl.exp(m - m_new)  # (GPAD,) (0 when m=-inf)
            p = tl.exp(scores - m_new[:, None])  # (GPAD, blk)
            lse = lse * alpha + tl.sum(p, axis=1)  # (GPAD,)

            v = tl.reshape(
                tl.load(
                    v_stacked_ptr + block_base + flat_idx,
                    eviction_policy="evict_first",
                ).to(tl.float32),
                (blk_size, d),
            )  # (blk, d)
            # pv[g, dd] = sum_n p[g, n] * v[n, dd]  -> (GPAD, d).
            if USE_DOT:
                pv = tl.dot(p, v)  # (GPAD, d)
            else:
                pv = tl.sum(p[:, :, None] * v[None, :, :], axis=1)  # (GPAD, d)
            acc = acc * alpha[:, None] + pv  # (GPAD, d)
            m = m_new

        # ------------------------------------------------------------------
        # Store PRE-normalization partials for this (split, kv_head): only the
        # real G rows (pad rows masked out). The merge kernel divides by the
        # combined denominator. Do NOT divide here.
        # ------------------------------------------------------------------
        head_row = s * h_kv + kv  # row index in (num_splits*h_kv, G[, d]) layout
        acc_off = (head_row * n_q_groups + gp_idx)[:, None] * d + d_idx[None, :]
        tl.store(acc_part_ptr + acc_off, acc, mask=gp_valid[:, None])  # fp32
        ml_off = head_row * n_q_groups + gp_idx  # (GPAD,)
        tl.store(m_part_ptr + ml_off, m, mask=gp_valid)
        tl.store(lse_part_ptr + ml_off, lse, mask=gp_valid)

    @triton.jit
    def _fused_merge_kernel(
        # Partials: (num_splits, h_kv, G, d) / (num_splits, h_kv, G)
        acc_part_ptr,
        m_part_ptr,
        lse_part_ptr,
        # Output: (h_kv, G, d) fp16
        out_ptr,
        num_splits,  # runtime int (do_not_specialize via being non-constexpr)
        h_kv: tl.constexpr,
        d: tl.constexpr,
        n_q_groups: tl.constexpr,
    ):
        """Merge num_splits partial (acc, m, lse) into the final normalized output.

        Grid: (h_kv,) — one program per KV head, merges all G query heads.

        Online-softmax combine across splits (base-e):
            m_g   = max_s m_part[s, g]
            l_g   = sum_s lse_part[s, g] * exp(m_part[s, g] - m_g)
            o_g   = sum_s acc_part[s, g] * exp(m_part[s, g] - m_g) / l_g
        Empty splits carry m=-inf -> exp(-inf - m_g)=0, contributing nothing
        (provided some split is non-empty so m_g is finite). With num_splits chosen
        so at least split 0 is non-empty, m_g is always finite.
        """
        kv = tl.program_id(0)
        d_idx = tl.arange(0, d)
        g_idx = tl.arange(0, n_q_groups)

        # First pass: global max across splits, per query head.
        m_global = tl.full((n_q_groups,), float("-inf"), tl.float32)
        for s in range(num_splits):
            head_row = s * h_kv + kv
            m_s = tl.load(m_part_ptr + head_row * n_q_groups + g_idx)  # (G,)
            m_global = tl.maximum(m_global, m_s)

        # Second pass: accumulate rescaled denom + numerator.
        l_acc = tl.zeros((n_q_groups,), tl.float32)  # (G,)
        o_acc = tl.zeros((n_q_groups, d), tl.float32)  # (G, d)
        for s in range(num_splits):
            head_row = s * h_kv + kv
            ml_off = head_row * n_q_groups + g_idx  # (G,)
            m_s = tl.load(m_part_ptr + ml_off)  # (G,)
            lse_s = tl.load(lse_part_ptr + ml_off)  # (G,)
            scale_s = tl.exp(m_s - m_global)  # (G,) 0 for empty/-inf splits
            l_acc += lse_s * scale_s
            acc_off = (head_row * n_q_groups + g_idx)[:, None] * d + d_idx[None, :]
            acc_s = tl.load(acc_part_ptr + acc_off)  # (G, d)
            o_acc += acc_s * scale_s[:, None]

        out = o_acc / l_acc[:, None]  # (G, d)
        out_off = (kv * n_q_groups + g_idx)[:, None] * d + d_idx[None, :]  # (G, d)
        tl.store(out_ptr + out_off, out.to(tl.float16))


def pick_num_splits(
    seq_len: int, blk_size: int, h_kv: int, n_sms: int = 132, occupancy_mult: int = 2
) -> int:
    """Choose num_splits for split-KV decode (brain/vLLM/flashinfer heuristic).

    OVERSUBSCRIBE the SMs: base programs = h_kv; target h_kv*num_splits ≈
    occupancy_mult * n_sms so each SM gets >1 block and the scheduler always has
    another warp to run when one stalls on an HBM load (vLLM occupancy_multiplier=2).
    Confirmed empirically (split sweep, tl.dot kernel): 32 splits (=2*132/8 → pow2)
    is the optimum at 32k AND 128k on GH200 (54% of HBM peak); 16 under-fills, 64
    regresses (merge/over-split overhead). Clamp so each split walks >= 1 stored
    block (min-work floor) and cap at 64. Rounded DOWN to a power of 2.

    At ctx <= ~a few blocks num_splits collapses to 1 (the min-work floor) = the
    no-split fast path — correct, since there's no length to parallelize.
    """
    n_blocks = max(1, (seq_len + blk_size - 1) // blk_size)
    target = max(1, occupancy_mult * n_sms // max(1, h_kv))  # oversubscribe SMs
    target = min(target, n_blocks, 64)  # min-work floor + cap
    # Round DOWN to a power of 2 (stable launch grid; avoids odd split sizes).
    p = 1
    while p * 2 <= target:
        p *= 2
    return p


def fused_decode_attention(
    q: torch.Tensor,
    k_stacked: torch.Tensor,
    v_stacked: torch.Tensor,
    seq_len: int,
    *,
    n_q_groups: int,
    scale: float,
    num_splits: int | None = None,
) -> torch.Tensor:
    """Fused split-KV decode attention over pre-stacked dense KV.

    The decode kernel launches grid=(h_kv, num_splits): each program walks ONLY
    its contiguous token slice, group-fusing all n_q_groups query heads (KV tile
    loaded once per block, reused across the group), carrying (m, lse, acc) in
    fp32 registers, and writes a PARTIAL. A second tiny merge kernel combines the
    partials per query head. This replaces the per-block Python launch loop.

    Split-KV gives the device-level parallelism a no-split grid lacks: grid=(h_kv,)
    is only h_kv programs (8 on GH200's 132 SMs); the split dim fills the machine.

    Args:
        q:           (n_q_heads, 1, d) fp16 CUDA — single decode query token.
        k_stacked:   (max_blocks, h_kv, blk_size, d) fp16 CUDA — pre-dequanted
                     keys (RoPE already applied if pre-RoPE). See build_kv_stacked.
        v_stacked:   (max_blocks, h_kv, blk_size, d) fp16 CUDA — pre-dequanted V.
        seq_len:     live number of KV tokens (<= max_blocks*blk_size). Tokens
                     beyond seq_len in the last stored block are masked out.
        n_q_groups:  n_q_heads // h_kv (GQA group count).
        scale:       attention scale (1/sqrt(d)).
        num_splits:  KV splits for decode parallelism. None -> pick_num_splits
                     (SM-aware heuristic). 1 = no-split (single program per head).

    Returns:
        (n_q_heads, 1, d) fp16 attention output.
    """
    _require_triton()
    n_q_heads, n_q, d = q.shape
    assert n_q == 1, "fused_decode_attention is decode-only (n_q==1)"
    max_blocks, h_kv, blk_size, _d = k_stacked.shape
    assert _d == d, f"k_stacked d={_d} != q d={d}"
    assert n_q_heads == h_kv * n_q_groups, (
        f"n_q_heads={n_q_heads} != h_kv={h_kv} * n_q_groups={n_q_groups}"
    )
    assert seq_len <= max_blocks * blk_size, (
        f"seq_len={seq_len} > capacity max_blocks*blk_size={max_blocks * blk_size}"
    )

    if num_splits is None:
        num_splits = pick_num_splits(seq_len, blk_size, h_kv)
    num_splits = max(1, int(num_splits))

    # tl.dot needs M,N,K>=16: pad the GQA group dim up to the next power of 2 >=16,
    # and use the dot path only when blk_size>=16 and d>=16 too (the tiny offline
    # test model has d=8 -> falls back to the broadcast cube; real models are d>=64).
    gpad = 16
    while gpad < n_q_groups:
        gpad *= 2
    use_dot = blk_size >= 16 and d >= 16

    # Query laid out (h_kv, G, d) contiguous so q[kv, g, :] is a clean offset.
    q_kv = q.squeeze(1).view(h_kv, n_q_groups, d).contiguous()

    # Partial buffers, one per split (pre-normalization). fp32 to avoid loss.
    acc_part = torch.empty(
        num_splits, h_kv, n_q_groups, d, dtype=torch.float32, device=q.device
    )
    m_part = torch.empty(
        num_splits, h_kv, n_q_groups, dtype=torch.float32, device=q.device
    )
    lse_part = torch.empty(
        num_splits, h_kv, n_q_groups, dtype=torch.float32, device=q.device
    )

    _fused_decode_kernel[(h_kv, num_splits)](
        q_kv,
        k_stacked,
        v_stacked,
        acc_part,
        m_part,
        lse_part,
        int(seq_len),
        int(num_splits),
        float(scale),
        h_kv=h_kv,
        blk_size=blk_size,
        d=d,
        n_q_groups=n_q_groups,
        GPAD=gpad,
        USE_DOT=use_dot,
    )

    out = torch.empty(h_kv, n_q_groups, d, dtype=torch.float16, device=q.device)
    _fused_merge_kernel[(h_kv,)](
        acc_part,
        m_part,
        lse_part,
        out,
        int(num_splits),
        h_kv=h_kv,
        d=d,
        n_q_groups=n_q_groups,
    )
    return out.view(n_q_heads, 1, d)


if TRITON_AVAILABLE:

    @triton.autotune(configs=_FUSED_AUTOTUNE_CONFIGS, key=["d", "n_q_groups"])
    @triton.jit(do_not_specialize=["seq_len", "num_splits"])
    def _fused_decode_packed_kernel(
        # Query: (h_kv, n_q_groups, d) — n_q=1 squeezed, GQA-grouped view
        q_ptr,
        # Pre-stacked PACKED RTN codes + per-group scales (NO dense copy):
        #   k_codes/v_codes:   (max_blocks, h_kv, blk_size, d)         int8
        #   k_scales/v_scales: (max_blocks, h_kv, blk_size, d//group)  fp16
        k_codes_ptr,
        v_codes_ptr,
        k_scales_ptr,
        v_scales_ptr,
        # Partial outputs (same as the dense kernel).
        acc_part_ptr,
        m_part_ptr,
        lse_part_ptr,
        seq_len,
        num_splits,
        scale,  # fp32 1/sqrt(d)
        h_kv: tl.constexpr,
        blk_size: tl.constexpr,
        d: tl.constexpr,
        n_q_groups: tl.constexpr,
        k_group: tl.constexpr,  # RTN group size for K (scale along d)
        v_group: tl.constexpr,  # RTN group size for V
        BLK_POW2: tl.constexpr,  # next pow2 >= blk_size (legal tl.arange row dim)
        GPAD: tl.constexpr,  # G padded up to >=16 so tl.dot's M dim is legal
        USE_DOT: tl.constexpr,  # tl.dot path (dims>=16) vs broadcast cube (tiny test)
    ):
        """Split-KV decode online-softmax, dequanting int8 RTN codes IN-KERNEL.

        Identical skeleton + grid to _fused_decode_kernel, but each block's K/V is
        loaded as int8 codes (blk, d) + per-group fp16 scale (blk, d//group) and
        dequanted in-register: deq[r,dd] = code[r,dd] * scale[r, dd//group], via the
        reshape-broadcast idiom. Resident storage stays PACKED (the compression is
        preserved); int8 is half the bytes of fp16 so the bandwidth ceiling is ~2x.
        """
        kv = tl.program_id(0)
        s = tl.program_id(1)
        d_idx = tl.arange(0, d)
        gp_idx = tl.arange(0, GPAD)
        gp_valid = gp_idx < n_q_groups
        n_kg: tl.constexpr = d // k_group  # K scale groups along d
        n_vg: tl.constexpr = d // v_group  # V scale groups along d
        # 2D loads (BLK_POW2 rows x d cols): each axis is a power of 2 (BLK_POW2 =
        # next pow2 of the real block length blk_size; d is pow2 for real models),
        # so tl.arange is legal even when blk_size is NOT a power of 2 (the geometric
        # flush schedule emits arbitrary block lengths). Rows >= blk_size are masked
        # out. This replaces the flat tl.arange(blk*d) which required blk*d be pow2.
        r_idx = tl.arange(0, BLK_POW2)  # (BLK_POW2,) padded row index
        row_real = r_idx < blk_size  # (BLK_POW2,) real rows of this block
        kg_idx = tl.arange(0, n_kg)
        vg_idx = tl.arange(0, n_vg)

        raw = (seq_len + num_splits - 1) // num_splits
        tokens_per_split = ((raw + blk_size - 1) // blk_size) * blk_size
        split_start = s * tokens_per_split
        split_end = tl.minimum(split_start + tokens_per_split, seq_len)
        first_block = split_start // blk_size
        last_block = (split_end + blk_size - 1) // blk_size

        head_stride = h_kv * blk_size * d  # advance one stored block (codes)
        kv_head_off = kv * blk_size * d  # this head within a stored block (codes)
        sc_head_stride_k = h_kv * blk_size * n_kg  # scale strides (smaller inner dim)
        sc_kv_off_k = kv * blk_size * n_kg
        sc_head_stride_v = h_kv * blk_size * n_vg
        sc_kv_off_v = kv * blk_size * n_vg

        # 2D row-major offsets within one stored block (one head):
        code_off = r_idx[:, None] * d + d_idx[None, :]  # (BLK_POW2, d) codes
        k_sc_off = r_idx[:, None] * n_kg + kg_idx[None, :]  # (BLK_POW2, n_kg)
        v_sc_off = r_idx[:, None] * n_vg + vg_idx[None, :]  # (BLK_POW2, n_vg)

        q_off = (kv * n_q_groups + gp_idx)[:, None] * d + d_idx[None, :]  # (GPAD, d)
        q_rows = tl.load(q_off + q_ptr, mask=gp_valid[:, None], other=0.0).to(
            tl.float32
        )  # (GPAD, d)

        m = tl.full((GPAD,), float("-inf"), tl.float32)
        lse = tl.zeros((GPAD,), tl.float32)
        acc = tl.zeros((GPAD, d), tl.float32)

        for blk in range(first_block, last_block):
            code_base = blk * head_stride + kv_head_off  # scalar
            row_abs = blk * blk_size + r_idx
            tile_mask = (row_abs < split_end) & row_real  # (BLK_POW2,)

            # --- K: 2D int8 code load (rows padded+masked) + per-group scale ---
            k_code = tl.load(
                k_codes_ptr + code_base + code_off,
                mask=row_real[:, None],
                other=0,
                eviction_policy="evict_first",
            ).to(tl.float32)  # (BLK_POW2, d)
            k_sc = tl.load(
                k_scales_ptr + (blk * sc_head_stride_k + sc_kv_off_k) + k_sc_off,
                mask=row_real[:, None],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)  # (BLK_POW2, n_kg)
            # Dequant: deq[r,dd] = code[r,dd] * scale[r, dd//group]. reshape the d
            # axis to (n_kg, group), broadcast the per-group scale, reshape back.
            k = tl.reshape(
                tl.reshape(k_code, (BLK_POW2, n_kg, k_group)) * k_sc[:, :, None],
                (BLK_POW2, d),
            )  # (BLK_POW2, d) dequant

            if USE_DOT:
                scores = tl.dot(q_rows, tl.trans(k)) * scale  # (GPAD, BLK_POW2)
            else:
                scores = tl.sum(q_rows[:, None, :] * k[None, :, :], axis=2) * scale
            scores = tl.where(tile_mask[None, :], scores, float("-inf"))

            m_tile = tl.max(scores, axis=1)
            m_new = tl.maximum(m, m_tile)
            alpha = tl.exp(m - m_new)
            p = tl.exp(scores - m_new[:, None])
            lse = lse * alpha + tl.sum(p, axis=1)

            # --- V: 2D int8 code load + per-group scale, dequant in-register ---
            v_code = tl.load(
                v_codes_ptr + code_base + code_off,
                mask=row_real[:, None],
                other=0,
                eviction_policy="evict_first",
            ).to(tl.float32)  # (BLK_POW2, d)
            v_sc = tl.load(
                v_scales_ptr + (blk * sc_head_stride_v + sc_kv_off_v) + v_sc_off,
                mask=row_real[:, None],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)  # (BLK_POW2, n_vg)
            v = tl.reshape(
                tl.reshape(v_code, (BLK_POW2, n_vg, v_group)) * v_sc[:, :, None],
                (BLK_POW2, d),
            )  # (BLK_POW2, d) dequant
            # Zero masked rows so they don't contribute to p@v (p is already 0 there
            # via scores=-inf -> exp=0, so v rows are multiplied by 0; safe either way).

            if USE_DOT:
                pv = tl.dot(p, v)  # (GPAD, d)
            else:
                pv = tl.sum(p[:, :, None] * v[None, :, :], axis=1)
            acc = acc * alpha[:, None] + pv
            m = m_new

        head_row = s * h_kv + kv
        acc_off = (head_row * n_q_groups + gp_idx)[:, None] * d + d_idx[None, :]
        tl.store(acc_part_ptr + acc_off, acc, mask=gp_valid[:, None])
        ml_off = head_row * n_q_groups + gp_idx
        tl.store(m_part_ptr + ml_off, m, mask=gp_valid)
        tl.store(lse_part_ptr + ml_off, lse, mask=gp_valid)


def fused_decode_attention_packed(
    q: torch.Tensor,
    k_codes: torch.Tensor,
    v_codes: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    seq_len: int,
    *,
    n_q_groups: int,
    scale: float,
    k_group: int,
    v_group: int,
    num_splits: int | None = None,
    k_tail: torch.Tensor | None = None,
    v_tail: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused split-KV decode over PACKED RTN codes — dequant in-kernel, no dense copy.

    Same contract/output as fused_decode_attention but the resident KV stays packed
    (int8 codes + per-group fp16 scales from build_kv_stacked_packed); the kernel
    dequants each block in-register. int8 = half the bytes of fp16 -> ~2x the
    bandwidth-bound ceiling of the dense path.

    Args:
        q:         (n_q_heads, 1, d) fp16 CUDA.
        k_codes/v_codes:   (max_blocks, h_kv, blk_size, d) int8 CUDA.
        k_scales/v_scales: (max_blocks, h_kv, blk_size, d//group) fp16 CUDA.
        seq_len:   live KV token count (the PACKED committed region only).
        n_q_groups, scale: as fused_decode_attention.
        k_group/v_group: RTN group sizes (scale granularity along d).
        num_splits: None -> pick_num_splits.
        k_tail/v_tail: optional dense fp16 (h_kv, tail_len, d) recent window NOT in
            the packed region (the streaming cache keeps the last W tokens lossless).
            Folded in via the online-softmax merge in PyTorch (tail_len is tiny,
            <= recent_window; not worth a kernel). When None, the GPU merge kernel
            is used directly (no PyTorch round-trip).
    Returns (n_q_heads, 1, d) fp16.
    """
    _require_triton()
    n_q_heads, n_q, d = q.shape
    assert n_q == 1, "decode-only (n_q==1)"
    max_blocks, h_kv, blk_size, _d = k_codes.shape
    assert _d == d, f"k_codes d={_d} != q d={d}"
    assert n_q_heads == h_kv * n_q_groups
    assert seq_len <= max_blocks * blk_size
    assert d % k_group == 0 and d % v_group == 0

    if num_splits is None:
        num_splits = pick_num_splits(seq_len, blk_size, h_kv)
    num_splits = max(1, int(num_splits))

    gpad = 16
    while gpad < n_q_groups:
        gpad *= 2
    blk_pow2 = 1
    while blk_pow2 < blk_size:
        blk_pow2 *= 2
    # tl.dot needs M,N,K>=16: M=GPAD>=16, N=BLK_POW2, K=d. d<16 only in the tiny
    # offline test (-> broadcast cube). BLK_POW2>=16 for real flush blocks.
    use_dot = blk_pow2 >= 16 and d >= 16

    q_kv = q.squeeze(1).view(h_kv, n_q_groups, d).contiguous()
    acc_part = torch.empty(
        num_splits, h_kv, n_q_groups, d, dtype=torch.float32, device=q.device
    )
    m_part = torch.empty(
        num_splits, h_kv, n_q_groups, dtype=torch.float32, device=q.device
    )
    lse_part = torch.empty(
        num_splits, h_kv, n_q_groups, dtype=torch.float32, device=q.device
    )

    _fused_decode_packed_kernel[(h_kv, num_splits)](
        q_kv,
        k_codes,
        v_codes,
        k_scales,
        v_scales,
        acc_part,
        m_part,
        lse_part,
        int(seq_len),
        int(num_splits),
        float(scale),
        h_kv=h_kv,
        blk_size=blk_size,
        d=d,
        n_q_groups=n_q_groups,
        k_group=k_group,
        v_group=v_group,
        BLK_POW2=blk_pow2,
        GPAD=gpad,
        USE_DOT=use_dot,
    )

    if k_tail is None or k_tail.shape[1] == 0:
        # No tail: GPU merge kernel directly (no PyTorch round-trip).
        out = torch.empty(h_kv, n_q_groups, d, dtype=torch.float16, device=q.device)
        _fused_merge_kernel[(h_kv,)](
            acc_part,
            m_part,
            lse_part,
            out,
            int(num_splits),
            h_kv=h_kv,
            d=d,
            n_q_groups=n_q_groups,
        )
        return out.view(n_q_heads, 1, d)

    # Tail present: compute its pre-normalization partial in PyTorch and merge it
    # with the kernel's split partials via the same online-softmax combine
    # (_merge_partials). tail_len <= recent_window is tiny, so this is cheap.
    # Partial layout (num_splits, h_kv, G, ...) -> per-split (n_q_heads,1,d) lists:
    # head index = kv*G + g matches q's (h_kv, G) flatten -> the n_q_heads order.
    assert v_tail is not None, "v_tail required when k_tail is set"
    accs = [acc_part[s].reshape(n_q_heads, 1, d) for s in range(num_splits)]
    ms = [m_part[s].reshape(n_q_heads, 1, 1) for s in range(num_splits)]
    lses = [lse_part[s].reshape(n_q_heads, 1, 1) for s in range(num_splits)]

    # Tail partial: dense fp16 (h_kv, tail_len, d), GQA-expanded to (n_q_heads, .).
    kt = k_tail.to(q.device, torch.float32).repeat_interleave(n_q_groups, dim=0)
    vt = v_tail.to(q.device, torch.float32).repeat_interleave(n_q_groups, dim=0)
    qf = q.float()  # (n_q_heads, 1, d)
    st = torch.einsum("hqd,hkd->hqk", qf, kt) * scale  # (n_q_heads, 1, tail_len)
    mt = st.amax(dim=-1, keepdim=True)  # (n_q_heads, 1, 1)
    pt = torch.exp(st - mt)  # (n_q_heads, 1, tail_len)
    lse_t = pt.sum(dim=-1, keepdim=True)  # (n_q_heads, 1, 1)
    acc_t = torch.einsum("hqk,hkd->hqd", pt, vt)  # (n_q_heads, 1, d) pre-norm
    accs.append(acc_t)
    ms.append(mt)
    lses.append(lse_t)

    return _merge_partials(accs, ms, lses).view(n_q_heads, 1, d)


# ---------------------------------------------------------------------------
# Phase 3a.5: FULL k2b fused decode — the real recipe, all dequant in-kernel.
#
#   K = lowrank_rtn_channel @3b: Us @ Vfac.T + RTN residual, keys PRE-RoPE ->
#       RoPE applied in-kernel on the reconstructed K (ported from the proven
#       per-block _k2b_softmax_block_kernel). All per-head, no cross-head coupling.
#   V = turboquant_mse @2b: codebook gather + Hadamard unrotate (FWHT over the
#       full C=h_kv*d row) + per-row norms. The unrotate COUPLES heads, so it
#       can't run inside a per-head decode program. KEY TRICK (verified exact,
#       scratch_fwht_defer_check.py): the FWHT is linear over C and the softmax
#       weights act on the sequence axis, so they COMMUTE:
#         o = Σ_s p[s]·V_real[s,:] = signs ⊙ H·( Σ_s (p[s]·norms[s])·M_quant[s,:] )
#       The decode kernel accumulates the PER-HEAD raw-codebook value
#         acc_pre[head,:] = Σ_s (p[s]·norms[s])·M_quant[s, head_cols]
#       (M_quant = cb[idx]/√C, pure per-channel gather — fits the per-head kernel);
#       the FWHT+signs run ONCE per query head in the merge epilogue over the
#       assembled full-C row. O(S·C·logC) -> O(C·logC). Survives split-KV merge
#       (linear commutes). norms[s] folds into p̃=p·norms; lse stays Σp (unweighted).
#
# Resident storage stays PACKED throughout (Us/Vfac/res int8 for K; int16 indices
# + norms for V) — no dense KV copy.
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    @triton.jit(do_not_specialize=["seq_len", "num_splits"])
    def _fused_decode_k2b_kernel(
        # Query: (h_kv, n_q_groups, d)
        q_ptr,
        # K factors (lowrank_rtn_channel), stacked per block:
        #   Us:        (max_blocks, blk, rank)            fp16  (shared across heads)
        #   Vfac:      (max_blocks, h_kv*d, rank)         fp16
        #   res_int:   (max_blocks, h_kv*d, blk)          int8  (RTN residual codes)
        #   res_scale: (max_blocks, h_kv*d, blk//k_group) fp16
        us_ptr,
        vfac_ptr,
        res_int_ptr,
        res_scale_ptr,
        # V factors (turboquant_mse PER-HEAD), stacked per block:
        #   v_idx:  (max_blocks, blk, h_kv*d) int16  (codebook indices)
        #   v_norm: (max_blocks, blk, h_kv)   fp16   (per-(row,head) norms)
        v_idx_ptr,
        v_norm_ptr,
        # Codebook (2**vbits,) fp32 — tiny. Per-head d-Hadamard matrix (d,d) fp32
        # + per-channel signs (d,) for the in-kernel unrotate (V = norm·signs·(H·Mq)).
        cb_ptr,
        hmat_ptr,
        vsigns_ptr,
        # RoPE tables for the WHOLE sequence: (max_S, d) fp16 (sliced per block).
        cos_ptr,
        sin_ptr,
        # Partials (pre-normalization):
        #   acc_part: (num_splits, h_kv, G, d) fp32 — normalized-numerator Σ p·V
        #   m_part / lse_part: (num_splits, h_kv, G) fp32
        acc_part_ptr,
        m_part_ptr,
        lse_part_ptr,
        seq_len,
        num_splits,
        scale,
        sqrt_d,  # 1/√d scale folded into the codebook gather (M_quant = cb[idx]/√d)
        h_kv: tl.constexpr,
        blk_size: tl.constexpr,
        d: tl.constexpr,
        n_q_groups: tl.constexpr,
        rank: tl.constexpr,
        k_group: tl.constexpr,
        vbits: tl.constexpr,  # turboquant V bits (codebook size 2**vbits)
        BLK_POW2: tl.constexpr,
        HAS_ROPE: tl.constexpr,
    ):
        """k2b fused decode: in-kernel lowrank-K + RoPE + deferred-FWHT V accumulation.

        Per (kv, split) program: reconstruct K (Us@Vfac.T + RTN residual + RoPE),
        score via GEMV (rank/d may be <16 in tests -> multiply+sum, not tl.dot),
        and accumulate the PER-HEAD raw V value Σ p̃·M_quant (p̃=p·norms). The FWHT
        unrotate is deferred to the merge epilogue (see module comment).
        """
        kv = tl.program_id(0)
        s = tl.program_id(1)
        d_idx = tl.arange(0, d)
        gp_idx = tl.arange(0, n_q_groups)  # query heads in this kv group (no pad: GEMV)
        r_idx = tl.arange(0, BLK_POW2)
        row_real = r_idx < blk_size
        rank_idx = tl.arange(0, rank)
        n_kg: tl.constexpr = blk_size // k_group  # RTN residual groups along blk
        C: tl.constexpr = h_kv * d

        raw = (seq_len + num_splits - 1) // num_splits
        tokens_per_split = ((raw + blk_size - 1) // blk_size) * blk_size
        split_start = s * tokens_per_split
        split_end = tl.minimum(split_start + tokens_per_split, seq_len)
        first_block = split_start // blk_size
        last_block = (split_end + blk_size - 1) // blk_size

        # Per-head query rows (G, d).
        q_off = (kv * n_q_groups + gp_idx)[:, None] * d + d_idx[None, :]
        q_rows = tl.load(q_ptr + q_off).to(tl.float32)  # (G, d)

        m = tl.full((n_q_groups,), float("-inf"), tl.float32)
        lse = tl.zeros((n_q_groups,), tl.float32)
        acc = tl.zeros((n_q_groups, d), tl.float32)  # Σ p̃·M_quant per head

        # rotate_half permutation+sign matrix (D,D), built once (RoPE).
        half: tl.constexpr = d // 2
        j_is_first = d_idx < half
        src_for_j = tl.where(j_is_first, d_idx + half, d_idx - half)
        sign_for_j = tl.where(j_is_first, -1.0, 1.0)
        P = tl.where(d_idx[:, None] == src_for_j[None, :], sign_for_j[None, :], 0.0)

        # Per-head V unrotate operators, loaded once: the (d,d) orthonormal Hadamard
        # matrix and the per-channel signs (V = norm · signs ⊙ (H_d · M_quant)).
        hmat = tl.load(hmat_ptr + d_idx[:, None] * d + d_idx[None, :]).to(
            tl.float32
        )  # (d, d)
        vsigns = tl.load(vsigns_ptr + d_idx).to(tl.float32)  # (d,)

        for blk in range(first_block, last_block):
            row_abs = blk * blk_size + r_idx  # (BLK_POW2,)
            tile_mask = (row_abs < split_end) & row_real

            # --- K lowrank: Us (blk, rank) @ Vfac[head] (d, rank).T -> (blk, d) ---
            us = tl.load(
                us_ptr
                + blk * blk_size * rank
                + r_idx[:, None] * rank
                + rank_idx[None, :],
                mask=row_real[:, None],
                other=0.0,
            ).to(tl.float32)  # (BLK_POW2, rank)
            vfac = tl.load(
                vfac_ptr
                + blk * C * rank
                + (kv * d + d_idx)[:, None] * rank
                + rank_idx[None, :]
            ).to(tl.float32)  # (d, rank)
            k_low = tl.sum(us[:, None, :] * vfac[None, :, :], axis=2)  # (BLK_POW2, d)

            # --- K RTN residual: res_int (d, blk) int8 * per-group scale -> (blk,d) ---
            res = tl.load(
                res_int_ptr
                + blk * C * blk_size
                + (kv * d + d_idx)[:, None] * blk_size
                + r_idx[None, :],
                mask=row_real[None, :],
                other=0,
            ).to(tl.float32)  # (d, BLK_POW2)
            res_sc = tl.load(
                res_scale_ptr
                + blk * C * n_kg
                + (kv * d + d_idx)[:, None] * n_kg
                + (r_idx[None, :] // k_group),
                mask=row_real[None, :],
                other=0.0,
            ).to(tl.float32)  # (d, BLK_POW2)
            k_res = tl.trans(res * res_sc)  # (BLK_POW2, d)
            k = k_low + k_res  # (BLK_POW2, d) pre-RoPE

            if HAS_ROPE:
                cos = tl.load(
                    cos_ptr + row_abs[:, None] * d + d_idx[None, :],
                    mask=row_real[:, None],
                    other=0.0,
                ).to(tl.float32)
                sin = tl.load(
                    sin_ptr + row_abs[:, None] * d + d_idx[None, :],
                    mask=row_real[:, None],
                    other=0.0,
                ).to(tl.float32)
                rot = tl.sum(k[:, :, None] * P[None, :, :], axis=1)  # (BLK_POW2, d)
                k = k * cos + rot * sin

            # scores[g, b] = scale * Σ_dd q[g,dd]*k[b,dd]  (GEMV, rank/d may be <16)
            scores = tl.sum(q_rows[:, None, :] * k[None, :, :], axis=2) * scale
            scores = tl.where(
                tile_mask[None, :], scores, float("-inf")
            )  # (G, BLK_POW2)

            m_new = tl.maximum(m, tl.max(scores, axis=1))
            alpha = tl.exp(m - m_new)
            p = tl.exp(scores - m_new[:, None])  # (G, BLK_POW2)
            lse = lse * alpha + tl.sum(p, axis=1)  # denom (Σ p)

            # --- V: PER-HEAD turboquant dequant, fully in-register over this head's
            # d columns. V[b, dd] = norm[b] · (vsigns[dd] · (H_d · M_quant[b,:])[dd]),
            # M_quant[b,dd] = cb[idx[b,dd]]/√d. The d-point Hadamard is a (d,d) matmul
            # (d>=16 -> tl.dot); per-head means NO cross-head coupling (the cross-head
            # full-C Hadamard could not fuse — QuaRot/SpinQuant use per-head exactly
            # for this). v_norm is per-(row, head).
            v_norm = tl.load(
                v_norm_ptr + blk * blk_size * h_kv + r_idx * h_kv + kv,
                mask=row_real,
                other=0.0,
            ).to(tl.float32)  # (BLK_POW2,) per-row norm for THIS head
            v_idx = tl.load(
                v_idx_ptr
                + blk * blk_size * C
                + r_idx[:, None] * C
                + (kv * d + d_idx)[None, :],
                mask=row_real[:, None],
                other=0,
            ).to(tl.int32)  # (BLK_POW2, d) codebook indices for this head
            m_quant = tl.load(cb_ptr + v_idx).to(tl.float32) * sqrt_d  # (BLK_POW2, d)
            # H_d · M_quant rows (orthonormal d-Hadamard via (d,d) matmul; d>=16 ok),
            # then per-channel signs and the per-row norm -> dequantized V (BLK_POW2,d).
            v = tl.dot(m_quant, hmat) * vsigns[None, :] * v_norm[:, None]
            # p@v via GEMV (multiply+sum) — G=n_q_groups may be <16 so no tl.dot here.
            pv = tl.sum(p[:, :, None] * v[None, :, :], axis=1)  # (G, d)
            acc = acc * alpha[:, None] + pv
            m = m_new

        head_row = s * h_kv + kv
        acc_off = (head_row * n_q_groups + gp_idx)[:, None] * d + d_idx[None, :]
        tl.store(acc_part_ptr + acc_off, acc)
        ml_off = head_row * n_q_groups + gp_idx
        tl.store(m_part_ptr + ml_off, m)
        tl.store(lse_part_ptr + ml_off, lse)


def build_kv_stacked_k2b(
    k_blocks: list,
    v_blocks: list,
    *,
    max_blocks: int,
    h_kv: int,
    blk_size: int,
    d: int,
    device: torch.device | str = "cuda",
):
    """Pre-stack k2b packed factors (lowrank_rtn_channel K + PER-HEAD turboquant V).

    K blocks: standard lowrank_rtn_channel packed dicts (Us, V, res_Q_int, res_scale).
    V blocks: PER-HEAD turboquant dicts — {"indices": (blk, h_kv*d) int16,
              "norms": (blk, h_kv) fp16} from _turboquant_mse_perhead_packed.

    Returns a dict of device tensors the k2b fused kernel consumes:
      us:        (max_blocks, blk, rank)            fp16
      vfac:      (max_blocks, h_kv*d, rank)         fp16
      res_int:   (max_blocks, h_kv*d, blk)          int8
      res_scale: (max_blocks, h_kv*d, blk//k_group) fp16
      v_idx:     (max_blocks, blk, h_kv*d)          int16
      v_norm:    (max_blocks, blk, h_kv)            fp16  (per-(row,head) norms)
    plus rank, k_group (read off block 0).
    """
    C = h_kv * d
    rank = k_blocks[0][0]["Us"].shape[1]
    res_scale0 = k_blocks[0][0]["res_scale"]  # (C, n_groups, 1)
    n_kg = res_scale0.shape[1]
    k_group = blk_size // n_kg

    us = torch.zeros(max_blocks, blk_size, rank, dtype=torch.float16, device=device)
    vfac = torch.zeros(max_blocks, C, rank, dtype=torch.float16, device=device)
    res_int = torch.zeros(max_blocks, C, blk_size, dtype=torch.int8, device=device)
    res_scale = torch.zeros(max_blocks, C, n_kg, dtype=torch.float16, device=device)
    v_idx = torch.zeros(max_blocks, blk_size, C, dtype=torch.int16, device=device)
    v_norm = torch.zeros(max_blocks, blk_size, h_kv, dtype=torch.float16, device=device)

    for i, ((kp, _ks, _ke), (vp, _vs, _ve)) in enumerate(zip(k_blocks, v_blocks)):
        assert i < max_blocks
        us[i] = kp["Us"].to(device).to(torch.float16)
        vfac[i] = kp["V"].to(device).to(torch.float16)
        res_int[i] = kp["res_Q_int"].to(device)
        res_scale[i] = kp["res_scale"].squeeze(-1).to(device).to(torch.float16)
        v_idx[i] = vp["indices"].to(device).to(torch.int16)
        v_norm[i] = vp["norms"].to(device).to(torch.float16)  # (blk, h_kv)

    return {
        "us": us,
        "vfac": vfac,
        "res_int": res_int,
        "res_scale": res_scale,
        "v_idx": v_idx,
        "v_norm": v_norm,
        "rank": rank,
        "k_group": k_group,
    }


def fused_decode_attention_k2b(
    q: torch.Tensor,
    stacks: dict,
    seq_len: int,
    *,
    n_q_groups: int,
    scale: float,
    vbits: int,
    v_seed: int,
    rope_cos: torch.Tensor | None,
    rope_sin: torch.Tensor | None,
    num_splits: int | None = None,
    k_tail: torch.Tensor | None = None,
    v_tail: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full k2b fused decode: in-kernel lowrank+RTN+RoPE K and PER-HEAD turboquant V.

    V uses the per-head Hadamard codec (build_kv_stacked_k2b), so its unrotate is a
    per-head d-point Hadamard done IN-KERNEL (a (d,d) matmul) — no cross-head
    coupling, no o_proj surgery. rope_cos/sin: (max_S, d) tables (None -> keys not
    pre-RoPE). The fp16 recent-window tail is merged in PyTorch.
    """
    _require_triton()
    from bmx.cache.codecs import _hadamard_signs, gaussian_codebook
    from bmx.quant.hadamard import fwht

    n_q_heads, n_q, d = q.shape
    assert n_q == 1, "decode-only"
    h_kv = n_q_heads // n_q_groups
    blk_size = stacks["us"].shape[1]
    max_blocks = stacks["us"].shape[0]
    assert seq_len <= max_blocks * blk_size
    rank = stacks["rank"]
    k_group = stacks["k_group"]
    assert (d & (d - 1)) == 0, f"d={d} must be a power of 2 for the per-head Hadamard"

    if num_splits is None:
        num_splits = pick_num_splits(seq_len, blk_size, h_kv)
    num_splits = max(1, int(num_splits))
    blk_pow2 = 1
    while blk_pow2 < blk_size:
        blk_pow2 *= 2

    cb = gaussian_codebook(vbits).to(q.device, torch.float32)
    sqrt_d = 1.0 / math.sqrt(d)  # M_quant = cb[idx] / √d (per-head rotation)
    # Per-head (d,d) orthonormal Hadamard matrix + per-channel signs for the unrotate.
    # _unrotate(x) = fwht(x) * signs; as a matrix, H_d = fwht(I_d), so the row-wise
    # unrotate is x @ H_d.T * signs. fwht is symmetric, so H_d.T = H_d.
    hmat = fwht(torch.eye(d, dtype=torch.float32, device=q.device))  # (d,d)
    vsigns = _hadamard_signs(d, v_seed).to(q.device, torch.float32)  # (d,)
    has_rope = rope_cos is not None

    q_kv = q.squeeze(1).view(h_kv, n_q_groups, d).contiguous()
    acc_part = torch.empty(
        num_splits, h_kv, n_q_groups, d, dtype=torch.float32, device=q.device
    )
    m_part = torch.empty(
        num_splits, h_kv, n_q_groups, dtype=torch.float32, device=q.device
    )
    lse_part = torch.empty(
        num_splits, h_kv, n_q_groups, dtype=torch.float32, device=q.device
    )

    cos_arg = (
        rope_cos.to(q.device, torch.float16).contiguous() if has_rope else stacks["us"]
    )
    sin_arg = (
        rope_sin.to(q.device, torch.float16).contiguous() if has_rope else stacks["us"]
    )

    _fused_decode_k2b_kernel[(h_kv, num_splits)](
        q_kv,
        stacks["us"],
        stacks["vfac"],
        stacks["res_int"],
        stacks["res_scale"],
        stacks["v_idx"],
        stacks["v_norm"],
        cb,
        hmat,
        vsigns,
        cos_arg,
        sin_arg,
        acc_part,
        m_part,
        lse_part,
        int(seq_len),
        int(num_splits),
        float(scale),
        float(sqrt_d),
        h_kv=h_kv,
        blk_size=blk_size,
        d=d,
        n_q_groups=n_q_groups,
        rank=rank,
        k_group=k_group,
        vbits=vbits,
        BLK_POW2=blk_pow2,
        HAS_ROPE=has_rope,
    )

    # V is fully dequanted in-kernel (per-head) — acc_part/m_part/lse_part are the
    # standard online-softmax partials, so the standard merge applies (no FWHT here).
    if k_tail is None or k_tail.shape[1] == 0:
        out = torch.empty(h_kv, n_q_groups, d, dtype=torch.float16, device=q.device)
        _fused_merge_kernel[(h_kv,)](
            acc_part,
            m_part,
            lse_part,
            out,
            int(num_splits),
            h_kv=h_kv,
            d=d,
            n_q_groups=n_q_groups,
        )
        return out.view(n_q_heads, 1, d)

    # Tail: online-softmax-merge the split partials + the dense fp16 recent window.
    accs = [acc_part[s].reshape(n_q_heads, 1, d) for s in range(num_splits)]
    ms = [m_part[s].reshape(n_q_heads, 1, 1) for s in range(num_splits)]
    lses = [lse_part[s].reshape(n_q_heads, 1, 1) for s in range(num_splits)]
    kt = k_tail.to(q.device, torch.float32).repeat_interleave(n_q_groups, dim=0)
    vt = v_tail.to(q.device, torch.float32).repeat_interleave(n_q_groups, dim=0)
    qf = q.float()
    st = torch.einsum("hqd,hkd->hqk", qf, kt) * scale
    mt = st.amax(-1, keepdim=True)
    pt = torch.exp(st - mt)
    lse_t = pt.sum(-1, keepdim=True)
    acc_t = torch.einsum("hqk,hkd->hqd", pt, vt)
    accs.append(acc_t)
    ms.append(mt)
    lses.append(lse_t)
    return _merge_partials(accs, ms, lses).view(n_q_heads, 1, d)


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


# ---------------------------------------------------------------------------
# Phase 3a.4: PACKED fused decode — dequant int8 RTN codes IN-KERNEL.
#
# The dense fused kernel above throws away the compression (it consumes dense
# fp16 KV). This path keeps the resident storage PACKED (int8 codes + per-group
# fp16 scales) and dequants in-register inside the fused block loop — so the
# memory saving is preserved AND, because int8 is HALF the bytes of fp16, the
# packed kernel's bandwidth-bound ceiling is ~2x the dense one (brain consult:
# decode is bandwidth-bound, dequant FMA rides in idle ALU slack; the per-group
# scale does NOT fold through the q.k dot since group<d, so dequant-then-dot).
#
# Layout (RTN: rtn_quantize_packed on the (S, h_kv*d) matrix; column c -> head
# c//d, channel c%d; per-(row, channel-group) scale, group along d):
#   k_codes/v_codes:   (max_blocks, h_kv, blk_size, d)        int8
#   k_scales/v_scales: (max_blocks, h_kv, blk_size, d//group) fp16
# Dequant: K[r,dd] = code[r,dd] * scale[r, dd//group]  (reshape-broadcast idiom).
# ---------------------------------------------------------------------------


def build_kv_stacked_packed(
    k_blocks: list,
    v_blocks: list,
    *,
    max_blocks: int,
    h_kv: int,
    blk_size: int,
    d: int,
    group: int,
    v_group: int | None = None,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-stack PACKED RTN codes + scales into device tensors (no dense copy).

    RTN arms only (rtn_token / rtn_channel / rotate_rtn_token store Q_int+scale).
    For rtn_channel the matrix is transposed at pack time; this builder assumes the
    rtn_token (S, h_kv*d) layout where Q_int is (S, h_kv*d) and column c maps to
    head c//d. Slots beyond len(k_blocks) are left zero (masked by seq_len).

    Returns (k_codes, v_codes int8 (max_blocks,h_kv,blk,d);
             k_scales, v_scales fp16 (max_blocks,h_kv,blk,d//group)).
    """
    _vg = v_group if v_group is not None else group
    assert d % group == 0, f"d={d} not divisible by k group={group}"
    assert d % _vg == 0, f"d={d} not divisible by v group={_vg}"
    n_kg, n_vg = d // group, d // _vg

    k_codes = torch.zeros(
        max_blocks, h_kv, blk_size, d, dtype=torch.int8, device=device
    )
    v_codes = torch.zeros(
        max_blocks, h_kv, blk_size, d, dtype=torch.int8, device=device
    )
    k_scales = torch.zeros(
        max_blocks, h_kv, blk_size, n_kg, dtype=torch.float16, device=device
    )
    v_scales = torch.zeros(
        max_blocks, h_kv, blk_size, n_vg, dtype=torch.float16, device=device
    )

    def _fill(packed, codes, scales, grp, n_grp):
        # Q_int: (S=blk, C=h_kv*d) int8 ; scale: (S, C//grp, 1) fp16.
        # Reshape per head: column c=head*d+dd -> (blk, h_kv, d). The scale groups
        # tile C in grp-sized runs; head kv owns scale groups [kv*d//grp:(kv+1)*d//grp].
        q_int = packed["Q_int"]  # (blk, h_kv*d) int8
        sc = packed["scale"].squeeze(-1)  # (blk, h_kv*d//grp) fp16
        blk = q_int.shape[0]
        # (blk, h_kv, d) — head is the middle axis after reshape (c = head*d + dd).
        q_hd = q_int.reshape(blk, h_kv, d).to(device)
        sc_hd = sc.reshape(blk, h_kv, n_grp).to(device).to(torch.float16)
        # -> (h_kv, blk, d) / (h_kv, blk, n_grp)
        codes[i] = q_hd.permute(1, 0, 2)
        scales[i] = sc_hd.permute(1, 0, 2)

    for i, ((kpacked, _ks, _ke), (vpacked, _vs, _ve)) in enumerate(
        zip(k_blocks, v_blocks)
    ):
        assert i < max_blocks, (
            f"more blocks ({len(k_blocks)}) than max_blocks ({max_blocks})"
        )
        _fill(kpacked, k_codes, k_scales, group, n_kg)
        _fill(vpacked, v_codes, v_scales, _vg, n_vg)

    return k_codes, v_codes, k_scales, v_scales
