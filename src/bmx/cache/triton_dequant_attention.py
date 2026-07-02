"""Triton fused dequant-attention DECODE kernels.

Two single-launch split-KV decode kernels that dequantize packed codes IN-KERNEL:
  - fused_decode_attention_packed — RTN arms (int8 codes, post-RoPE K).
  - fused_decode_attention_k2b    — the k2b recipe (lowrank_rtn_channel K
    reconstructed + RoPE'd in-kernel; per-head turboquant V dequanted in-kernel).
Non-fused configs fall back to chunked_dequant_attention (PyTorch, fp32-
accumulating). _finalize_decode / _merge_partials handle split-KV combination.

Imports cleanly with TRITON_AVAILABLE=False (AMD/no-CUDA dev box); kernels are
verified on the GH200 VM against the naive oracle + end-to-end logit parity.
Design rationale and staged-build ledger:
  docs/superpowers/specs/2026-06-24-triton-decode-kernel-design.md
"""

from __future__ import annotations

import functools
import math

import torch

from bmx.cache.codecs import _hadamard_signs, gaussian_codebook
from bmx.cache.collect import from_matrix


def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= max(1, n)."""
    p = 1
    while p < n:
        p *= 2
    return p


def _pick_block_n(blk_size: int, cap: int = 64) -> int:
    """KV tile size for the per-block decode loop: the largest power of 2 that is
    <= cap AND divides blk_size, so each tile lies within one stored block
    (contiguous load). Blocks are uniform PAGE=128 tokens under the paged layout,
    so in practice this returns 64; kept general for non-uniform test blocks."""
    bn = 1
    p = 2
    while p <= cap and p <= blk_size:
        if blk_size % p == 0:
            bn = p
        p *= 2
    return bn


@functools.lru_cache(maxsize=16)
def _hadamard_matrix(d: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Orthonormal (d,d) Walsh-Hadamard matrix H_d = fwht(I_d), cached per
    (d, device, dtype). The per-head V unrotate is row-wise `x @ H_d * signs`
    (fwht is symmetric, so H_d.T = H_d). Constant per d — cached so the k2b decode
    launcher doesn't rebuild it (an O(d² log d) FWHT) every token."""
    from bmx.quant.hadamard import fwht

    return fwht(torch.eye(d, dtype=dtype, device=device))


try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TRITON_AVAILABLE = False
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Correctness invariants (see spec for full rationale)
# ---------------------------------------------------------------------------
#
# v_group / v_seed: K and V may use different seed/group; both are accepted as
#   kwargs (default to K's values when omitted, for the RTN-only callers).
#
# Correctness bar: max_abs vs naive_dense_attention < 1e-2 (expect ~2e-4 at fp16).
#   Do NOT loosen — fix the kernel if it drifts.
#
# Split-KV merge invariant (must hold):
#   Each split stores pre-normalization (acc_i, m_i, lse_i), merged as:
#     m = max_i(m_i);  l = sum_i(lse_i * exp(m_i - m));
#     out = sum_i(acc_i * exp(m_i - m)) / l
#   At num_splits=1 this reduces to acc_0 / lse_0 (bit-identical to the serial path).
#
# Base-e consistency: ALL kernels and the merge use natural exp — do NOT mix
#   base-2. A base-2 merge formula is a silent correctness trap.
#
# ---------------------------------------------------------------------------
# Capability guard — fail loud; NO silent fallback.
# ---------------------------------------------------------------------------


def _require_triton() -> None:
    """Raise if Triton + CUDA are not available.

    PackedStreamingLayer.attend checks TRITON_AVAILABLE before calling into this
    module and routes to chunked_dequant_attention otherwise; this guard makes a
    missing capability fail loud rather than fall back silently.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "fused_decode_attention_{packed,k2b} require Triton + CUDA. "
            "TRITON_AVAILABLE=False on this machine (no CUDA or Triton not "
            "installed). PackedStreamingLayer.attend dispatches to "
            "chunked_dequant_attention in that case."
        )


# ---------------------------------------------------------------------------
# Split-KV helpers: merge
# ---------------------------------------------------------------------------


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
        => Bit-identical to the serial path's final division `acc / lse`.

    AT MULTIPLE SPLITS:
        This is exactly online_softmax_update applied across the split axis,
        giving the same result as if all blocks had been processed serially.

    BASE-E NOTE: partial_lse is the raw unnormalized sum-of-softmax-weights
    (lse in online_softmax_update — not the log of that sum).  The correction
    exp(m_i - m) is base-e.  Do NOT use exp2/log2 here (all kernels are base-e).

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
    scales = torch.exp(ms - m_global)  # (S, H, 1, 1) — base-e

    # Merged denominator: sum_i(lse_i * exp(m_i - m))
    l_merged = (lses * scales).sum(dim=0)  # (H, 1, 1)

    # Merged numerator: sum_i(acc_i * exp(m_i - m))  — (S, H, 1, d) * (S, H, 1, 1)
    acc_merged = (accs * scales).sum(dim=0)  # (H, 1, d)

    # Normalize and return in fp16 (`acc / lse.to(q.dtype)`)
    return (acc_merged / l_merged).to(torch.float16)


# ---------------------------------------------------------------------------
# FUSED decode kernels — shared design notes
# (_fused_decode_packed_kernel, _fused_decode_k2b_kernel)
#
# One launch loops over ALL KV blocks INTERNALLY, carrying (m, lse, acc) in fp32
# registers with one output write (vs the per-block launch path, which pays
# n_blocks * h_kv launches per decode step and threads the carry through PyTorch).
# Design:
#   - GQA GROUP FUSION: each program handles ONE kv_head and ALL n_q_groups query
#     heads. The KV tile is loaded ONCE per block and reused across the whole group
#     -> n_q_groups x less KV HBM traffic (the KV load IS the whole cost at M=1
#     decode). (vLLM "3D kernel": process all Q heads of a KV head together.)
#   - REGISTER CARRY: acc[G, D], m[G], lse[G] live in fp32 registers across the
#     whole block loop (acc = 4*128 fp32 = 2KB/program, trivial vs SM reg file).
#     fp16 accumulation over hundreds-thousands of blocks would lose precision.
#   - FIRST-BLOCK -inf: m init -inf, lse/acc init 0. On block 0, alpha =
#     exp(-inf - m_new) = 0 annihilates the garbage init (lse=0*0+sum p,
#     acc=0*0+pv). No special-case needed (the standard flash-attention init).
#   - 128-bit LDG.E.128 loads are AUTOMATIC from contiguous fp16 D=128 inner axis;
#     eviction_policy="evict_first" makes KV a read-once L2 stream so it doesn't
#     evict the reused weight working set.
#   - GEMV (multiply + tl.sum), NOT tl.dot: decode is M=1, bandwidth-bound; tl.dot
#     is useless at M=1 and has a min-dim>=16 constraint.
#
# Split-KV (grid z-dim + merge kernel) parallelizes across SMs at long context
# (no-split underutilizes SMs on a large GPU); num_splits=1 is the serial path.
#
# Correctness bar: max_abs vs naive_dense_attention < 1e-2 (expect ~2-3e-4 at fp16).
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:
    # Import Config directly so Pylance sees the concrete type (not `triton: None`).
    from triton import Config as _TritonConfig

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
    seq_len: int,
    blk_size: int,
    h_kv: int,
    n_sms: int | None = None,
    occupancy_mult: int = 2,
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
    n_sms=None reads the current device's SM count (GH200 = 132, so behavior
    there is unchanged); the 132 fallback keeps CPU-only test boxes deterministic.
    """
    if n_sms is None:
        n_sms = (
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
            if torch.cuda.is_available()
            else 132
        )
    n_blocks = max(1, (seq_len + blk_size - 1) // blk_size)
    target = max(1, occupancy_mult * n_sms // max(1, h_kv))  # oversubscribe SMs
    target = min(target, n_blocks, 64)  # min-work floor + cap
    # Round DOWN to a power of 2 (stable launch grid; avoids odd split sizes).
    p = 1
    while p * 2 <= target:
        p *= 2
    return p


def _finalize_decode(
    acc_part: torch.Tensor,
    m_part: torch.Tensor,
    lse_part: torch.Tensor,
    num_splits: int,
    q: torch.Tensor,
    scale: float,
    n_q_groups: int,
    k_tail: torch.Tensor | None,
    v_tail: torch.Tensor | None,
) -> torch.Tensor:
    """Merge the split partials into the final (n_q_heads, 1, d) output.

    Shared by every fused decode launcher. No tail -> the GPU merge kernel directly
    (no PyTorch round-trip). Tail present -> fold the dense fp16 recent window
    (k_tail/v_tail, <= recent_window tokens) into the split partials via the same
    base-e online-softmax combine (_merge_partials) in PyTorch — tiny, so PyTorch is
    fine. Partial layout (num_splits, h_kv, G, ...) flattens to head index kv*G+g,
    matching q's (h_kv, G) order.
    """
    # acc_part is (num_splits, h_kv, n_q_groups, d) — its group axis equals the
    # n_q_groups parameter by construction (asserted here to keep them in lockstep).
    h_kv, d = acc_part.shape[1], acc_part.shape[3]
    assert acc_part.shape[2] == n_q_groups, (
        f"partial group axis {acc_part.shape[2]} != n_q_groups {n_q_groups}"
    )
    n_q_heads = h_kv * n_q_groups

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

    assert v_tail is not None, "v_tail required when k_tail is set"
    accs = [acc_part[s].reshape(n_q_heads, 1, d) for s in range(num_splits)]
    ms = [m_part[s].reshape(n_q_heads, 1, 1) for s in range(num_splits)]
    lses = [lse_part[s].reshape(n_q_heads, 1, 1) for s in range(num_splits)]

    # Dense tail partial: (h_kv, tail_len, d) GQA-expanded to (n_q_heads, ...).
    kt = k_tail.to(q.device, torch.float32).repeat_interleave(n_q_groups, dim=0)
    vt = v_tail.to(q.device, torch.float32).repeat_interleave(n_q_groups, dim=0)
    qf = q.float()  # (n_q_heads, 1, d)
    st = torch.einsum("hqd,hkd->hqk", qf, kt) * scale  # (n_q_heads, 1, tail_len)
    mt = st.amax(dim=-1, keepdim=True)
    pt = torch.exp(st - mt)
    lse_t = pt.sum(dim=-1, keepdim=True)
    acc_t = torch.einsum("hqk,hkd->hqd", pt, vt)  # pre-norm
    accs.append(acc_t)
    ms.append(mt)
    lses.append(lse_t)
    return _merge_partials(accs, ms, lses).view(n_q_heads, 1, d)


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
        BLOCK_N: tl.constexpr,  # KV tile rows per loop iter (small pow2; divides blk_size)
        GPAD: tl.constexpr,  # G padded up to >=16 so tl.dot's M dim is legal
        USE_DOT: tl.constexpr,  # tl.dot path (dims>=16) vs broadcast cube (tiny test)
    ):
        """Split-KV decode online-softmax, dequanting int8 RTN codes IN-KERNEL.

        Flash-attention tiling: walks its token range in FIXED BLOCK_N-row tiles
        (BLOCK_N small + power of 2), NOT one stored block at a time. The cache
        flushes the whole prefill as one large stored block (thousands of tokens);
        loading that block whole would blow shared memory. BLOCK_N divides blk_size
        (both multiples of the RTN/lowrank group), so each tile lies within ONE
        stored block -> contiguous loads. K/V are int8 codes + per-group fp16 scale,
        dequanted in-register (reshape-broadcast). Packed-resident, no dense copy.
        """
        kv = tl.program_id(0)
        s = tl.program_id(1)
        d_idx = tl.arange(0, d)
        gp_idx = tl.arange(0, GPAD)
        gp_valid = gp_idx < n_q_groups
        n_kg: tl.constexpr = d // k_group  # K scale groups along d
        n_vg: tl.constexpr = d // v_group  # V scale groups along d
        n_idx = tl.arange(0, BLOCK_N)  # (BLOCK_N,) tile-local row index
        kg_idx = tl.arange(0, n_kg)
        vg_idx = tl.arange(0, n_vg)

        # This split's token range, rounded to BLOCK_N so tiles never straddle the
        # split boundary (the last split's tail is masked by split_end).
        raw = (seq_len + num_splits - 1) // num_splits
        tokens_per_split = ((raw + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
        split_start = s * tokens_per_split
        split_end = tl.minimum(split_start + tokens_per_split, seq_len)

        head_stride = h_kv * blk_size * d  # advance one stored block (codes)
        kv_head_off = kv * blk_size * d  # this head within a stored block (codes)
        sc_head_stride_k = h_kv * blk_size * n_kg
        sc_kv_off_k = kv * blk_size * n_kg
        sc_head_stride_v = h_kv * blk_size * n_vg
        sc_kv_off_v = kv * blk_size * n_vg

        q_off = (kv * n_q_groups + gp_idx)[:, None] * d + d_idx[None, :]  # (GPAD, d)
        q_rows = tl.load(q_off + q_ptr, mask=gp_valid[:, None], other=0.0).to(
            tl.float32
        )  # (GPAD, d)

        m = tl.full((GPAD,), float("-inf"), tl.float32)
        lse = tl.zeros((GPAD,), tl.float32)
        acc = tl.zeros((GPAD, d), tl.float32)

        n_tiles = (split_end - split_start + BLOCK_N - 1) // BLOCK_N
        for t in range(n_tiles):
            tok0 = split_start + t * BLOCK_N  # first absolute token of this tile
            tok = tok0 + n_idx  # (BLOCK_N,) absolute token indices
            tile_mask = tok < split_end  # (BLOCK_N,) valid tokens
            # Each tile lies within ONE stored block (BLOCK_N | blk_size): the stored
            # block + row offset for this whole tile come from tok0.
            blk = tok0 // blk_size  # stored block index (scalar)
            row0 = tok0 - blk * blk_size  # tile's first row within that block (scalar)
            r = row0 + n_idx  # (BLOCK_N,) row within the stored block

            code_base = blk * head_stride + kv_head_off
            code_off = r[:, None] * d + d_idx[None, :]  # (BLOCK_N, d)
            k_sc_off = r[:, None] * n_kg + kg_idx[None, :]  # (BLOCK_N, n_kg)
            v_sc_off = r[:, None] * n_vg + vg_idx[None, :]  # (BLOCK_N, n_vg)

            # --- K: int8 codes + per-group scale, dequant in-register ---
            k_code = tl.load(
                k_codes_ptr + code_base + code_off,
                mask=tile_mask[:, None],
                other=0,
                eviction_policy="evict_first",
            ).to(tl.float32)  # (BLOCK_N, d)
            k_sc = tl.load(
                k_scales_ptr + (blk * sc_head_stride_k + sc_kv_off_k) + k_sc_off,
                mask=tile_mask[:, None],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)  # (BLOCK_N, n_kg)
            k = tl.reshape(
                tl.reshape(k_code, (BLOCK_N, n_kg, k_group)) * k_sc[:, :, None],
                (BLOCK_N, d),
            )  # (BLOCK_N, d) dequant

            if USE_DOT:
                scores = tl.dot(q_rows, tl.trans(k)) * scale  # (GPAD, BLOCK_N)
            else:
                scores = tl.sum(q_rows[:, None, :] * k[None, :, :], axis=2) * scale
            scores = tl.where(tile_mask[None, :], scores, float("-inf"))

            m_new = tl.maximum(m, tl.max(scores, axis=1))
            alpha = tl.exp(m - m_new)
            p = tl.exp(scores - m_new[:, None])
            lse = lse * alpha + tl.sum(p, axis=1)

            # --- V: int8 codes + per-group scale, dequant in-register ---
            v_code = tl.load(
                v_codes_ptr + code_base + code_off,
                mask=tile_mask[:, None],
                other=0,
                eviction_policy="evict_first",
            ).to(tl.float32)  # (BLOCK_N, d)
            v_sc = tl.load(
                v_scales_ptr + (blk * sc_head_stride_v + sc_kv_off_v) + v_sc_off,
                mask=tile_mask[:, None],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)  # (BLOCK_N, n_vg)
            v = tl.reshape(
                tl.reshape(v_code, (BLOCK_N, n_vg, v_group)) * v_sc[:, :, None],
                (BLOCK_N, d),
            )  # (BLOCK_N, d) dequant

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

    The resident KV stays packed (int8 codes + per-group fp16 scales from
    build_kv_stacked_packed); the kernel dequants each block in-register. int8 =
    half the bytes of fp16 -> ~2x the bandwidth-bound ceiling of a dense path.

    Args:
        q:         (n_q_heads, 1, d) fp16 CUDA.
        k_codes/v_codes:   (max_blocks, h_kv, blk_size, d) int8 CUDA.
        k_scales/v_scales: (max_blocks, h_kv, blk_size, d//group) fp16 CUDA.
        seq_len:   live KV token count (the PACKED committed region only).
        n_q_groups: GQA query-groups per KV head; scale: 1/sqrt(d) softmax scale.
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

    gpad = _next_pow2(max(16, n_q_groups))
    block_n = _pick_block_n(blk_size)  # KV tile rows (<= 64, divides blk_size)
    # tl.dot needs M,N,K>=16: M=GPAD>=16, N=BLOCK_N, K=d. d<16 / BLOCK_N<16 only on
    # the tiny offline test (-> broadcast cube fallback). Real models: d=128, BLOCK_N=64.
    use_dot = block_n >= 16 and d >= 16

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
        BLOCK_N=block_n,
        GPAD=gpad,
        USE_DOT=use_dot,
    )

    return _finalize_decode(
        acc_part, m_part, lse_part, num_splits, q, scale, n_q_groups, k_tail, v_tail
    )


# ---------------------------------------------------------------------------
# FULL k2b fused decode — the real recipe, all dequant in-kernel.
#
#   K = lowrank_rtn_channel @3b: Us @ Vfac.T + RTN residual, keys PRE-RoPE ->
#       RoPE applied in-kernel on the reconstructed K. All per-head, no
#       cross-head coupling.
#   V = turboquant_mse_perhead @2b: PER-HEAD Hadamard (block-diagonal over heads,
#       had_dim = d_head) — the QuaRot/SpinQuant design. V[b,:] = norm · (signs ⊙
#       (H_d · M_quant[b,:])), M_quant = cb[idx]/√d, over each head's OWN d columns.
#       So V dequant is FULLY per-head and runs IN-KERNEL: codebook gather + a
#       d-point Hadamard (a (d,d) matmul, tl.dot) + signs + norm. No cross-head
#       coupling, so V is a standard online-softmax value and the merge is standard.
#
#   Why per-head, not the full-C turboquant_mse: a single C=h_kv*d Hadamard couples
#   all heads, and under GQA each query head has its own softmax — so that unrotate
#   neither fits a per-head decode program nor folds into o_proj (dimension mismatch
#   + per-head-p commutation failure). Per-head rotation is quality-equivalent (the
#   turboquant distortion bound is dimension-independent in the constant; the
#   Beta→Gaussian concentration is excellent at d=128) and the production-standard
#   choice. (An earlier cross-head "defer the FWHT past the p·v sum" attempt failed
#   for exactly the per-head-p reason; per-head removes the coupling entirely.)
#
# Resident storage stays PACKED throughout (Us/Vfac/res int8 for K; int16 indices
# + per-head norms for V) — no dense KV copy.
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
        BLOCK_N: tl.constexpr,  # KV tile rows per loop iter (small pow2; divides blk_size)
        HAS_ROPE: tl.constexpr,
    ):
        """k2b fused decode: in-kernel lowrank-K + RoPE + per-head turboquant-V.

        Per (kv, split) program: reconstruct K (Us@Vfac.T + RTN residual + RoPE via
        tl.dot), score via GEMV (n_q_groups may be <16 -> multiply+sum, not tl.dot),
        and dequant V FULLY in-kernel per head — codebook gather + a per-head d-point
        Hadamard unrotate (tl.dot(m_quant, hmat)) + per-channel signs + per-row norm —
        then accumulate Σ p·V. Per-head rotation has no cross-head coupling, so V is
        a standard online-softmax value here; the merge is the standard merge (no
        deferred FWHT). acc/m/lse partials are standard.
        """
        kv = tl.program_id(0)
        s = tl.program_id(1)
        d_idx = tl.arange(0, d)
        gp_idx = tl.arange(0, n_q_groups)  # query heads in this kv group (no pad: GEMV)
        n_idx = tl.arange(0, BLOCK_N)  # (BLOCK_N,) tile-local row index
        rank_idx = tl.arange(0, rank)
        n_kg: tl.constexpr = blk_size // k_group  # RTN residual groups along blk
        C: tl.constexpr = h_kv * d

        # Flash-attention tiling: walk the token range in fixed BLOCK_N tiles, NOT one
        # (giant) stored block at a time. BLOCK_N | blk_size so each tile is within one
        # stored block (contiguous) + small enough for SMEM regardless of blk_size.
        raw = (seq_len + num_splits - 1) // num_splits
        tokens_per_split = ((raw + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
        split_start = s * tokens_per_split
        split_end = tl.minimum(split_start + tokens_per_split, seq_len)

        # Per-head query rows (G, d).
        q_off = (kv * n_q_groups + gp_idx)[:, None] * d + d_idx[None, :]
        q_rows = tl.load(q_ptr + q_off).to(tl.float32)  # (G, d)

        m = tl.full((n_q_groups,), float("-inf"), tl.float32)
        lse = tl.zeros((n_q_groups,), tl.float32)
        acc = tl.zeros(
            (n_q_groups, d), tl.float32
        )  # Σ p·V per head (V dequant in-kernel)

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

        n_tiles = (split_end - split_start + BLOCK_N - 1) // BLOCK_N
        for t in range(n_tiles):
            tok0 = split_start + t * BLOCK_N  # first absolute token of this tile
            tok = tok0 + n_idx  # (BLOCK_N,) absolute token indices
            tile_mask = tok < split_end  # (BLOCK_N,) valid tokens
            blk = tok0 // blk_size  # stored block index (scalar; tile within one block)
            r = (
                tok0 - blk * blk_size
            ) + n_idx  # (BLOCK_N,) row within the stored block

            # --- K lowrank: Us (BLOCK_N, rank) @ Vfac[head] (d, rank).T -> (BLOCK_N,d) ---
            us = tl.load(
                us_ptr + blk * blk_size * rank + r[:, None] * rank + rank_idx[None, :],
                mask=tile_mask[:, None],
                other=0.0,
            ).to(tl.float32)  # (BLOCK_N, rank)
            vfac = tl.load(
                vfac_ptr
                + blk * C * rank
                + (kv * d + d_idx)[:, None] * rank
                + rank_idx[None, :]
            ).to(tl.float32)  # (d, rank)
            k_low = tl.dot(us, tl.trans(vfac))  # (BLOCK_N, d)

            # --- K RTN residual: res_int (d, blk) int8 * per-group scale -> (BLOCK_N,d) ---
            res = tl.load(
                res_int_ptr
                + blk * C * blk_size
                + (kv * d + d_idx)[:, None] * blk_size
                + r[None, :],
                mask=tile_mask[None, :],
                other=0,
            ).to(tl.float32)  # (d, BLOCK_N)
            res_sc = tl.load(
                res_scale_ptr
                + blk * C * n_kg
                + (kv * d + d_idx)[:, None] * n_kg
                + (r[None, :] // k_group),
                mask=tile_mask[None, :],
                other=0.0,
            ).to(tl.float32)  # (d, BLOCK_N)
            k_res = tl.trans(res * res_sc)  # (BLOCK_N, d)
            k = k_low + k_res  # (BLOCK_N, d) pre-RoPE

            if HAS_ROPE:
                cos = tl.load(
                    cos_ptr + tok[:, None] * d + d_idx[None, :],
                    mask=tile_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                sin = tl.load(
                    sin_ptr + tok[:, None] * d + d_idx[None, :],
                    mask=tile_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                rot = tl.dot(k, P)  # (BLOCK_N, d) = k @ P (rotate_half); avoids cube
                k = k * cos + rot * sin

            # scores[g, b] = scale * Σ_dd q[g,dd]*k[b,dd]. GEMV (multiply+sum): G=
            # n_q_groups may be <16 so no tl.dot on the G axis (k @ q.T would need M=G).
            scores = tl.sum(q_rows[:, None, :] * k[None, :, :], axis=2) * scale
            scores = tl.where(tile_mask[None, :], scores, float("-inf"))  # (G, BLOCK_N)

            m_new = tl.maximum(m, tl.max(scores, axis=1))
            alpha = tl.exp(m - m_new)
            p = tl.exp(scores - m_new[:, None])  # (G, BLOCK_N)
            lse = lse * alpha + tl.sum(p, axis=1)  # denom (Σ p)

            # --- V: PER-HEAD turboquant dequant, fully in-register over this head's
            # d columns. V[b, dd] = norm[b] · (vsigns[dd] · (H_d · M_quant[b,:])[dd]),
            # M_quant[b,dd] = cb[idx[b,dd]]/√d. The d-point Hadamard is a (d,d) matmul
            # (d>=16 -> tl.dot); per-head means NO cross-head coupling (QuaRot/SpinQuant
            # use per-head exactly for this). v_norm is per-(row, head).
            v_norm = tl.load(
                v_norm_ptr + blk * blk_size * h_kv + r * h_kv + kv,
                mask=tile_mask,
                other=0.0,
            ).to(tl.float32)  # (BLOCK_N,) per-row norm for THIS head
            v_idx = tl.load(
                v_idx_ptr
                + blk * blk_size * C
                + r[:, None] * C
                + (kv * d + d_idx)[None, :],
                mask=tile_mask[:, None],
                other=0,
            ).to(tl.int32)  # (BLOCK_N, d) codebook indices for this head
            m_quant = tl.load(cb_ptr + v_idx).to(tl.float32) * sqrt_d  # (BLOCK_N, d)
            # H_d · M_quant rows (orthonormal d-Hadamard via (d,d) matmul; d>=16 ok),
            # then per-channel signs and the per-row norm -> dequantized V (BLOCK_N,d).
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
    block_n = _pick_block_n(blk_size)  # KV tile rows (<= 64, divides blk_size)

    cb = gaussian_codebook(vbits).to(q.device, torch.float32)
    sqrt_d = 1.0 / math.sqrt(d)  # M_quant = cb[idx] / √d (per-head rotation)
    # Per-head (d,d) orthonormal Hadamard matrix + per-channel signs for the unrotate
    # (row-wise V = (x @ H_d) * signs * norm). hmat is cached per (d, device, dtype).
    hmat = _hadamard_matrix(d, str(q.device), torch.float32)
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
        BLOCK_N=block_n,
        HAS_ROPE=has_rope,
    )

    # V is fully dequanted in-kernel (per-head) — acc_part/m_part/lse_part are the
    # standard online-softmax partials, so the standard merge applies (no FWHT here).
    return _finalize_decode(
        acc_part, m_part, lse_part, num_splits, q, scale, n_q_groups, k_tail, v_tail
    )


# ---------------------------------------------------------------------------
# PACKED fused decode — dequant int8 RTN codes IN-KERNEL.
#
# A dense decode kernel would consume dense fp16 KV, throwing away the
# compression. This path keeps the resident storage PACKED (int8 codes + per-group
# fp16 scales) and dequants in-register inside the fused block loop — so the
# memory saving is preserved AND, because int8 is HALF the bytes of fp16, the
# packed kernel's bandwidth-bound ceiling is ~2x the dense one (decode is
# bandwidth-bound, so the dequant FMA rides in idle ALU slack; the per-group
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
        # The (S, h*x) -> (h, S, x) per-head split is exactly from_matrix (codes use
        # x=d; scales use x=n_grp, same permute since head kv owns scale groups
        # [kv*n_grp:(kv+1)*n_grp]). CLAUDE.md: the head/matrix layout lives ONLY in
        # to_matrix/from_matrix — never hand-roll the permute.
        codes[i] = from_matrix(packed["Q_int"].to(device), h_kv)  # (h_kv, blk, d)
        sc = (
            packed["scale"].squeeze(-1).to(device).to(torch.float16)
        )  # (blk, h_kv*n_grp)
        scales[i] = from_matrix(sc, h_kv)  # (h_kv, blk, n_grp)

    for i, ((kpacked, _ks, _ke), (vpacked, _vs, _ve)) in enumerate(
        zip(k_blocks, v_blocks)
    ):
        assert i < max_blocks, (
            f"more blocks ({len(k_blocks)}) than max_blocks ({max_blocks})"
        )
        _fill(kpacked, k_codes, k_scales, group, n_kg)
        _fill(vpacked, v_codes, v_scales, _vg, n_vg)

    return k_codes, v_codes, k_scales, v_scales
