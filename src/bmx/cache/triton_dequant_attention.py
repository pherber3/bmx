"""Triton fused dequant-attention DECODE kernels.

Two single-launch split-KV decode kernels that dequantize packed codes IN-KERNEL:
  - fused_decode_attention_packed — RTN arms (int8 codes, post-RoPE K).
  - fused_decode_attention_k2b    — the k2b recipe (lowrank_rtn_channel K
    reconstructed + RoPE'd in-kernel; per-head turboquant V dequanted in-kernel).
Non-fused configs fall back to chunked_dequant_attention (PyTorch, fp32-
accumulating). _finalize_decode handles split-KV combination (including the fp16
recent-window tail, folded in as one extra split via _tail_partial + the same GPU
merge kernel — desk review F1b, 2026-07-04).

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


# ---------------------------------------------------------------------------
# Bit-packing (W5-2): pack low-bit-width codes 8/bits-per-byte along the LAST
# axis. Containers today are int16 regardless of bit-width (codecs.py stores
# codebook indices as int16 unconditionally) -- for a 2-bit V code that is 8x
# the bytes actually needed. This packs/unpacks a reference (pure-torch, no
# Triton) representation for the STACKED resident buffers only; the block
# dicts and codecs.py quantize_packed/dequant_packed keep int16 (reference-
# path parity -- see build_kv_stacked_k2b's pack_v flag and the plan doc,
# docs/superpowers/plans/2026-07-05-resident-memory-realization.md, Task 2).
# ---------------------------------------------------------------------------


def pack_codes(idx: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack int codes into uint8 containers, 8/bits codes per byte (last axis).

    Little-endian within the byte: code at channel c lives in byte c//per_byte
    at bit-offset bits*(c % per_byte), per_byte = 8 // bits. Only bit-widths
    that divide 8 and are < 8 are supported (2 and 4 in practice -- vbits for
    the turboquant V codec); the last axis (channel/C) must be divisible by
    per_byte so every output byte is fully populated by real codes.

    idx: (..., C) any integer dtype, values in [0, 2**bits).
    Returns: (..., C // per_byte) uint8.
    """
    assert 8 % bits == 0 and bits < 8, (
        f"pack_codes only supports bit-widths dividing 8 and < 8 (got bits={bits}); "
        "2 and 4 are the supported turboquant/RTN low-bit-width codes."
    )
    per_byte = 8 // bits
    C = idx.shape[-1]
    assert C % per_byte == 0, (
        f"pack_codes: last axis C={C} must be divisible by per_byte={per_byte} "
        f"(8 // bits={bits}) so every packed byte is fully populated."
    )
    idx_u8 = idx.to(torch.uint8)
    grouped = idx_u8.reshape(*idx.shape[:-1], C // per_byte, per_byte)  # (...,C/pb,pb)
    shifts = (torch.arange(per_byte, device=idx.device, dtype=torch.uint8) * bits).to(
        torch.uint8
    )
    # int32 accumulation avoids uint8 overflow when shifting/OR-ing the top code.
    packed = (
        (grouped.to(torch.int32) << shifts.to(torch.int32)).sum(dim=-1).to(torch.uint8)
    )
    return packed


def unpack_codes(packed: torch.Tensor, bits: int, C: int) -> torch.Tensor:
    """Exact inverse of pack_codes. packed: (..., C // per_byte) uint8 -> (..., C) int16."""
    assert 8 % bits == 0 and bits < 8, (
        f"unpack_codes only supports bit-widths dividing 8 and < 8 (got bits={bits})."
    )
    per_byte = 8 // bits
    assert C % per_byte == 0, (
        f"unpack_codes: C={C} must be divisible by per_byte={per_byte} (8 // bits={bits})."
    )
    assert packed.shape[-1] == C // per_byte, (
        f"unpack_codes: packed last axis {packed.shape[-1]} != C // per_byte "
        f"({C} // {per_byte} = {C // per_byte})"
    )
    mask = (1 << bits) - 1
    shifts = torch.arange(per_byte, device=packed.device, dtype=torch.uint8) * bits
    packed_i32 = packed.to(torch.int32).unsqueeze(-1)  # (..., C/pb, 1)
    codes = (packed_i32 >> shifts.to(torch.int32)) & mask  # (..., C/pb, pb)
    return codes.reshape(*packed.shape[:-1], C).to(torch.int16)


def block_v_indices(vp: dict, vbits: int, C: int) -> torch.Tensor:
    """Return a V block dict's int16 codebook indices, transparent to pack_v.

    ``vp`` is a V block dict as stored in PackedStreamingLayer._v_blocks. Under
    the default (unpacked) path it holds ``vp["indices"]`` (int16, (blk, C))
    directly. Under W5-1+W5-2 single-storage pack_v=True re-pointing
    (_repoint_k2b_blocks), "indices" is DELETED and replaced by
    "indices_packed" (a view into the packed uint8 stack buffer) to free the
    int16 block-list copy -- the whole point of packing. This helper is the
    ONE place that knows both representations, so any consumer that needs
    int16 indices from a (possibly re-pointed) stacked block goes through it
    instead of reading "indices" by key directly. The unpack here is
    transient (never stored back onto vp) -- it materializes a fresh int16
    tensor each call, same as the pre-pack_v behavior.
    """
    if "indices" in vp:
        return vp["indices"]
    return unpack_codes(vp["indices_packed"], vbits, C)


def _pick_block_n(blk_size: int, cap: int = 64) -> int:
    """KV tile size for the fused kernels' internal block loop: the largest power of 2 that is
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


@functools.lru_cache(maxsize=16)
def _codebook_dev(bits: int, device: str) -> torch.Tensor:
    """fp32 Gaussian Lloyd-Max codebook already resident on `device`, cached per
    (bits, device). `gaussian_codebook` (codecs.py) is itself lru_cache'd but only
    on CPU — calling `.to(q.device, torch.float32)` on its result every decode
    step re-does the H2D copy + allocation per layer per token (desk review F2:
    docs/2026-07-04-triton-decode-desk-review.md). Cache the device copy too, the
    same pattern as `_hadamard_matrix` above."""
    return gaussian_codebook(bits).to(device, torch.float32)


@functools.lru_cache(maxsize=16)
def _signs_dev(d: int, seed: int, device: str) -> torch.Tensor:
    """fp32 ±1 Hadamard sign vector already resident on `device`, cached per
    (d, seed, device). Same H2D-per-call issue as `_codebook_dev` (desk review
    F2) — `_hadamard_signs` is CPU-cached only."""
    return _hadamard_signs(d, seed).to(device, torch.float32)


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
# Split-KV helpers: tail partial (fp16 recent-window attention, GQA-shaped)
# ---------------------------------------------------------------------------
#
# Online-softmax combine invariants (apply to every merge in this module,
# kernel or PyTorch — moved here from the deleted `_merge_partials`, which this
# tail partial now feeds into the SAME GPU `_fused_merge_kernel` used for the
# no-tail path, desk review F1b):
#
#     m   = max_i(m_i)                            # global running max
#     l   = sum_i(lse_i * exp(m_i - m))           # re-scaled lse sum
#     out = sum_i(acc_i * exp(m_i - m)) / l       # re-scaled acc sum, normalized
#
# CORRECTNESS INVARIANT (num_splits=1, no tail): m=m_0, l=lse_0*exp(0)=lse_0,
# out=acc_0/lse_0 => bit-identical to the serial path's final division.
# BASE-E NOTE: lse is the raw unnormalized sum-of-softmax-weights (not its
# log). The correction exp(m_i - m) is base-e — do NOT mix exp2/log2.
# ---------------------------------------------------------------------------


def _tail_partial(
    q_kv: torch.Tensor,
    k_tail: torch.Tensor,
    v_tail: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the fp16 recent-window tail as ONE extra split partial (acc, m, lse).

    GQA-shaped: works directly on q's (h_kv, G) grouping, NO repeat_interleave to
    n_q_heads (the old per-step fallback expanded k_tail/v_tail to n_q_heads first —
    wasteful; einsum broadcasts the shared kv-head dim across the G query heads for
    free). This lets the tail partial slot into acc_part/m_part/lse_part at row
    index `num_splits` (shape (num_splits+1, h_kv, G, ...)) and be merged by the
    SAME GPU `_fused_merge_kernel` the no-tail path already uses — the old
    dedicated PyTorch merge (96 per-split views + a separate stack-based combine)
    is deleted; there is now exactly one merge path (desk review F1b).

    Args:
        q_kv:   (h_kv, G, d) — already fp16/fp32, upcast internally.
        k_tail: (h_kv, T, d) fp16 dense recent-window keys.
        v_tail: (h_kv, T, d) fp16 dense recent-window values.
        scale:  1/sqrt(d) softmax scale.

    Returns (acc, m, lse), all fp32:
        acc: (h_kv, G, d)  — pre-normalized Σ p·V for this split.
        m:   (h_kv, G)     — running max for this split.
        lse: (h_kv, G)     — Σ p (unnormalized) for this split.
    """
    qf = q_kv.float()  # (h_kv, G, d)
    ktf = k_tail.float()  # (h_kv, T, d)
    vtf = v_tail.float()  # (h_kv, T, d)
    s = torch.einsum("hgd,htd->hgt", qf, ktf) * scale  # (h_kv, G, T)
    m = s.amax(dim=-1)  # (h_kv, G)
    p = torch.exp(s - m.unsqueeze(-1))  # (h_kv, G, T)
    lse = p.sum(dim=-1)  # (h_kv, G)
    acc = torch.einsum("hgt,htd->hgd", p, vtf)  # (h_kv, G, d)
    return acc, m, lse


# ---------------------------------------------------------------------------
# FUSED decode kernels — shared design notes
# (_fused_decode_packed_kernel, _fused_decode_k2b_kernel)
#
# One launch loops over ALL KV blocks INTERNALLY, carrying (m, lse, acc) in fp32
# registers with one output write (vs the retired per-block launch path, which pays
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

    Shared by every fused decode launcher. Both the no-tail and tail-present cases
    now go through the SAME GPU `_fused_merge_kernel` call (desk review F1b — the
    old tail branch ran a dedicated ~96-view PyTorch merge every decode step, since
    the streaming schedule makes the tail non-empty on essentially every real
    step). The caller (each launcher) is responsible for over-allocating
    acc_part/m_part/lse_part with `num_splits + 1` leading slots whenever a
    non-empty tail is expected — slot `num_splits` is left for the tail and is
    never written by the compute kernel (whose grid is `(h_kv, num_splits)`, so its
    internal `program_id(1)` never reaches that index). `_finalize_decode` computes
    the tail partial (`_tail_partial`, GQA-shaped, no repeat_interleave) and writes
    it into slot `num_splits` in place, then calls the merge kernel with the
    *runtime* split count `num_splits + 1` so its internal loop walks through the
    tail row too. Partial layout (splits, h_kv, G, ...) flattens to head index
    kv*G+g, matching q's (h_kv, G) order — identical to the no-tail path.

    Numerics: this is the same online-softmax combine as before (base-e,
    fp32-throughout) — mathematically identical to the old torch-merge result, but
    op ORDER changes (GPU kernel accumulation vs a PyTorch stack+reduce), so
    bitwise logits may differ at fp32-rounding level. The no-tail path is
    byte-for-byte unchanged (same kernel, same math, same call).
    """
    # acc_part is (slots, h_kv, n_q_groups, d) — its group axis equals the
    # n_q_groups parameter by construction (asserted here to keep them in lockstep).
    h_kv, d = acc_part.shape[1], acc_part.shape[3]
    assert acc_part.shape[2] == n_q_groups, (
        f"partial group axis {acc_part.shape[2]} != n_q_groups {n_q_groups}"
    )
    n_q_heads = h_kv * n_q_groups
    has_tail = k_tail is not None and k_tail.shape[1] > 0
    out = torch.empty(h_kv, n_q_groups, d, dtype=torch.float16, device=q.device)

    if not has_tail:
        merge_splits = num_splits
    else:
        assert v_tail is not None, "v_tail required when k_tail is set"
        assert acc_part.shape[0] >= num_splits + 1, (
            "acc_part must be over-allocated with num_splits+1 slots when a "
            "non-empty tail is passed (the launcher's job — see _finalize_decode)"
        )
        q_kv = q.squeeze(1).view(h_kv, n_q_groups, d)  # (h_kv, G, d)
        acc_t, m_t, lse_t = _tail_partial(
            q_kv, k_tail.to(q.device), v_tail.to(q.device), scale
        )
        acc_part[num_splits] = acc_t
        m_part[num_splits] = m_t
        lse_part[num_splits] = lse_t
        merge_splits = num_splits + 1

    _fused_merge_kernel[(h_kv,)](
        acc_part,
        m_part,
        lse_part,
        out,
        int(merge_splits),
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
            Folded into the SAME GPU split-KV merge as one extra "virtual split"
            (desk review F1b) — tail_len is tiny (<= recent_window) so computing
            its partial in PyTorch (_tail_partial) is fine, but the merge itself no
            longer forks into a separate PyTorch code path. When None/empty, the
            merge kernel runs over exactly num_splits (unchanged from before).
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
    has_tail = k_tail is not None and k_tail.shape[1] > 0
    # +1 leading slot for the tail partial (desk review F1b) — the compute kernel
    # below is launched with grid (h_kv, num_splits) so it only ever writes rows
    # [0, num_splits); the extra row is written by _finalize_decode's _tail_partial
    # call and is safe because the merge kernel's row math is driven purely by the
    # runtime split-count arg it's given, not by the buffer's physical extent.
    part_slots = num_splits + 1 if has_tail else num_splits

    gpad = _next_pow2(max(16, n_q_groups))
    block_n = _pick_block_n(blk_size)  # KV tile rows (<= 64, divides blk_size)
    # tl.dot needs M,N,K>=16: M=GPAD>=16, N=BLOCK_N, K=d. d<16 / BLOCK_N<16 only on
    # the tiny offline test (-> broadcast cube fallback). Real models: d=128, BLOCK_N=64.
    use_dot = block_n >= 16 and d >= 16

    q_kv = q.squeeze(1).view(h_kv, n_q_groups, d).contiguous()
    acc_part = torch.empty(
        part_slots, h_kv, n_q_groups, d, dtype=torch.float32, device=q.device
    )
    m_part = torch.empty(
        part_slots, h_kv, n_q_groups, dtype=torch.float32, device=q.device
    )
    lse_part = torch.empty(
        part_slots, h_kv, n_q_groups, dtype=torch.float32, device=q.device
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
        V_PACKED: tl.constexpr,  # W5-2: v_idx_ptr holds packed uint8 (per_byte codes/byte)
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
        # fp16 for the tl.dot operand (entries are exactly ±1/0 — lossless in fp16);
        # all tl.dot ops in this kernel take fp16 operands with fp32 accumulation
        # (tensor-core rate; the kernel is compute-bound — 2026-07-06 speed pass).
        P = tl.where(d_idx[:, None] == src_for_j[None, :], sign_for_j[None, :], 0.0).to(
            tl.float16
        )

        # Per-head V unrotate operators, loaded once: the (d,d) orthonormal Hadamard
        # matrix and the per-channel signs (V = norm · signs ⊙ (H_d · M_quant)).
        hmat = tl.load(hmat_ptr + d_idx[:, None] * d + d_idx[None, :]).to(
            tl.float16
        )  # (d, d) — fp16 dot operand (entries ±1/√d, rel err ~5e-4 « the 1e-2 bar)
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
            )  # (BLOCK_N, rank) fp16 — dot operand, fp32 accumulate
            vfac = tl.load(
                vfac_ptr
                + blk * C * rank
                + (kv * d + d_idx)[:, None] * rank
                + rank_idx[None, :]
            )  # (d, rank) fp16 — dot operand
            k_low = tl.dot(us, tl.trans(vfac))  # (BLOCK_N, d) fp32 acc

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
                # k cast fp16 ONLY for the rotate dot (rel err ~5e-4 on k); the
                # elementwise k*cos below keeps the fp32 k.
                rot = tl.dot(k.to(tl.float16), P)  # (BLOCK_N, d) = k @ P (rotate_half)
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
            c_global = kv * d + d_idx  # (d,) this head's channel range in [0, C)
            if V_PACKED:
                # 8/vbits codes per uint8 byte, little-endian within the byte (see
                # pack_codes): code at channel c lives in byte c//per_byte at bit
                # offset vbits*(c % per_byte). Redundant per-element byte loads
                # across the vbits codes sharing one byte are fine — L2 catches
                # them (per task brief); this does NOT restructure the tile loop.
                per_byte: tl.constexpr = 8 // vbits
                C_packed: tl.constexpr = C // per_byte
                byte_off = c_global // per_byte  # (d,) byte index within the row
                bit_shift = (c_global % per_byte) * vbits  # (d,) shift within the byte
                v_byte = tl.load(
                    v_idx_ptr
                    + blk * blk_size * C_packed
                    + r[:, None] * C_packed
                    + byte_off[None, :],
                    mask=tile_mask[:, None],
                    other=0,
                ).to(tl.int32)  # (BLOCK_N, d)
                v_idx = (v_byte >> bit_shift[None, :]) & ((1 << vbits) - 1)
            else:
                v_idx = tl.load(
                    v_idx_ptr + blk * blk_size * C + r[:, None] * C + c_global[None, :],
                    mask=tile_mask[:, None],
                    other=0,
                ).to(tl.int32)  # (BLOCK_N, d) codebook indices for this head
            m_quant = tl.load(cb_ptr + v_idx).to(tl.float16)  # (BLOCK_N, d)
            # H_d · M_quant rows (orthonormal d-Hadamard via (d,d) matmul; d>=16 ok),
            # then per-channel signs and the per-row norm -> dequantized V (BLOCK_N,d).
            # fp16 dot operands, fp32 result; the scalar sqrt_d commutes through the
            # dot and is applied on the fp32 output (multiplying the fp16 operand by
            # the fp32 scalar arg would silently promote it back to fp32 — Triton
            # dtype-promotion rule, hit on first compile).
            v = tl.dot(m_quant, hmat) * sqrt_d * vsigns[None, :] * v_norm[:, None]
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
    pack_v: bool = False,
):
    """Pre-stack k2b packed factors (lowrank_rtn_channel K + PER-HEAD turboquant V).

    K blocks: standard lowrank_rtn_channel packed dicts (Us, V, res_Q_int, res_scale).
    V blocks: PER-HEAD turboquant dicts — {"indices": (blk, h_kv*d) int16,
              "norms": (blk, h_kv) fp16, "bits": int} from _turboquant_mse_perhead_packed.

    Returns a dict of device tensors the k2b fused kernel consumes:
      us:        (max_blocks, blk, rank)            fp16
      vfac:      (max_blocks, h_kv*d, rank)         fp16
      res_int:   (max_blocks, h_kv*d, blk)          int8
      res_scale: (max_blocks, h_kv*d, blk//k_group) fp16
      v_idx:     (max_blocks, blk, h_kv*d)          int16   -- when pack_v=False
      v_idx_packed: (max_blocks, blk, h_kv*d // per_byte) uint8 -- when pack_v=True
                 (4 codes/byte at vbits=2; per_byte = 8 // vbits)
      v_norm:    (max_blocks, blk, h_kv)            fp16  (per-(row,head) norms)
    plus rank, k_group (read off block 0); when pack_v=True also pack_v=True and
    vbits (read off block 0's "bits" entry) so downstream consumers (the fused
    kernel launcher, _repoint_k2b_blocks) can tell which field/layout is present
    WITHOUT re-deriving vbits from elsewhere. When pack_v=False the output is
    byte-identical to before this flag existed (regression-pinned by
    test_build_kv_stacked_k2b_pack_v_false_matches_today).
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
    v_norm = torch.zeros(max_blocks, blk_size, h_kv, dtype=torch.float16, device=device)

    vbits = int(v_blocks[0][0]["bits"])
    if pack_v:
        per_byte = 8 // vbits
        v_idx_packed = torch.zeros(
            max_blocks, blk_size, C // per_byte, dtype=torch.uint8, device=device
        )
    else:
        v_idx = torch.zeros(max_blocks, blk_size, C, dtype=torch.int16, device=device)

    for i, ((kp, _ks, _ke), (vp, _vs, _ve)) in enumerate(zip(k_blocks, v_blocks)):
        assert i < max_blocks
        us[i] = kp["Us"].to(device).to(torch.float16)
        vfac[i] = kp["V"].to(device).to(torch.float16)
        res_int[i] = kp["res_Q_int"].to(device)
        res_scale[i] = kp["res_scale"].squeeze(-1).to(device).to(torch.float16)
        v_norm[i] = vp["norms"].to(device).to(torch.float16)  # (blk, h_kv)
        idx = vp["indices"].to(device).to(torch.int16)
        if pack_v:
            v_idx_packed[i] = pack_codes(idx, vbits)
        else:
            v_idx[i] = idx

    out = {
        "us": us,
        "vfac": vfac,
        "res_int": res_int,
        "res_scale": res_scale,
        "v_norm": v_norm,
        "rank": rank,
        "k_group": k_group,
    }
    if pack_v:
        out["v_idx_packed"] = v_idx_packed
        out["pack_v"] = True
        out["vbits"] = vbits
    else:
        out["v_idx"] = v_idx
    return out


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
    pre-RoPE). The fp16 recent-window tail's partial is computed in PyTorch
    (_tail_partial) but merged by the same GPU split-KV merge kernel as every other
    split (desk review F1b) — see fused_decode_attention_packed's docstring for the
    full tail-slot design.
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
    has_tail = k_tail is not None and k_tail.shape[1] > 0
    # +1 leading slot for the tail partial (desk review F1b) — see the matching
    # comment in fused_decode_attention_packed for the extra-row safety argument
    # (the compute kernel's grid is (h_kv, num_splits), so it never touches row
    # num_splits; the merge kernel's row indexing is driven by its runtime
    # split-count arg, not the buffer's physical extent).
    part_slots = num_splits + 1 if has_tail else num_splits
    # V_PACKED codegen carries extra (BLOCK_N, d) int32 index/byte temporaries; at
    # production dims (d=128, BLOCK_N=64) that overflowed the GH200's 232KB SMEM
    # (needed 241664 — first real-dims firing, 2026-07-06; the CUDA pack_v oracle
    # tests passed only at tiny dims). Cap the KV tile at 32 rows when packed.
    v_packed = "v_idx_packed" in stacks
    block_n = _pick_block_n(blk_size, cap=32 if v_packed else 64)

    cb = _codebook_dev(vbits, str(q.device))
    sqrt_d = 1.0 / math.sqrt(d)  # M_quant = cb[idx] / √d (per-head rotation)
    # Per-head (d,d) orthonormal Hadamard matrix + per-channel signs for the unrotate
    # (row-wise V = (x @ H_d) * signs * norm). hmat/vsigns cached per (d, device[, seed])
    # — desk review F2: avoid a fresh H2D copy every layer per decode step.
    hmat = _hadamard_matrix(d, str(q.device), torch.float16)
    vsigns = _signs_dev(d, v_seed, str(q.device))  # (d,)
    has_rope = rope_cos is not None

    q_kv = q.squeeze(1).view(h_kv, n_q_groups, d).contiguous()
    acc_part = torch.empty(
        part_slots, h_kv, n_q_groups, d, dtype=torch.float32, device=q.device
    )
    m_part = torch.empty(
        part_slots, h_kv, n_q_groups, dtype=torch.float32, device=q.device
    )
    lse_part = torch.empty(
        part_slots, h_kv, n_q_groups, dtype=torch.float32, device=q.device
    )

    cos_arg = (
        rope_cos.to(q.device, torch.float16).contiguous() if has_rope else stacks["us"]
    )
    sin_arg = (
        rope_sin.to(q.device, torch.float16).contiguous() if has_rope else stacks["us"]
    )

    # W5-2: the stacks dict carries EITHER "v_idx" (int16, default) OR
    # "v_idx_packed" (uint8, pack_v=True build_kv_stacked_k2b) — never both.
    # v_packed detected above (it also gates the SMEM-safe BLOCK_N cap); a plain
    # dict with "v_idx" -> v_packed=False, byte-identical kernel launch to before
    # the flag existed.
    v_idx_arg = stacks["v_idx_packed"] if v_packed else stacks["v_idx"]

    _fused_decode_k2b_kernel[(h_kv, num_splits)](
        q_kv,
        stacks["us"],
        stacks["vfac"],
        stacks["res_int"],
        stacks["res_scale"],
        v_idx_arg,
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
        V_PACKED=v_packed,
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
