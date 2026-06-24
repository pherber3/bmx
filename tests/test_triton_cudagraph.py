"""CUDA-graph-safe decode path — capture/replay test (stage 3d).

THE REAL GATE (from grounding):
  Capture a CUDA graph of triton_decode_attention_graphable at seq_len=S0.
  Update seq_len_dev to S0+k (in-place). Replay the graph.
  Assert replayed output matches a FRESH call to triton_decode_attention_graphable
  at S0+k within max_abs < 2e-2.

  Additionally assert replayed output does NOT match a fresh call at S0 (within
  a loose tolerance), proving the replay used the UPDATED seq_len, not the
  captured S0 value. A Python-int-seqlen implementation would:
    - Bake seq_len=S0 into the kernel specialization.
    - Replay at S0+k would attend only the first S0 tokens → matches fresh-at-S0
      but NOT fresh-at-S0+k → the test catches it.

WHAT THIS TEST COVERS (3d scope):
  - Device-pointer seqlen: seq_len_dev is an int32 CUDA tensor read by the kernel.
  - Fixed launch grid: (max_blocks, h_kv, n_q_groups) — same shape every replay.
  - Block masking: tokens beyond seq_len_dev[0] are masked (partial last block ok).
  - RTN arm only: k2b/lowrank_rtn_channel is deferred (see module docstring).

WHAT IS DEFERRED (honest):
  - k2b path: K's packed factors (Us, V, res_Q_int, res_scale) are heterogeneous
    Python dicts; stacking into a contiguous device tensor requires a paged-block-
    table refactor. Deferred from 3d.
  - Tail window: fp16 residual window is outside the graphable kernel. Callers
    process it outside the captured graph (cheap).
  - RoPE inside graph: apply RoPE to k_stacked BEFORE capture if k_pre_rope=True.
    The graphable path does not re-apply RoPE per replay (pre-RoPE keys already
    applied before stacking).

SKIP: CUDA-gated (skips loud on AMD dev box and any non-CUDA host).
"""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason=(
        "Triton graph-safe decode (stage 3d) — VM/CUDA only. "
        "Skipping on non-CUDA host (AMD dev box or CI without GPU). "
        "Run on the CUDA VM to exercise capture/replay parity."
    ),
)


# ---------------------------------------------------------------------------
# Helper: build RTN packed blocks (same pattern as test_triton_decode_rtn.py)
# ---------------------------------------------------------------------------


def _blocks_cuda(blocks: list) -> list:
    """Move packed dict tensors to CUDA; keep start/end ints."""
    out = []
    for packed, start, end in blocks:
        packed_cuda = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in packed.items()
        }
        out.append((packed_cuda, start, end))
    return out


def _make_rtn_blocks(
    *,
    n_q_heads: int,
    n_q_groups: int,
    d: int,
    blk: int,
    n_blocks: int,
    seed: int = 42,
    group: int = 8,
):
    """Build (q, k_blocks, v_blocks, kwargs) for RTN decode tests."""
    from bmx.cache.codecs import quantize_packed
    from bmx.cache.collect import to_matrix

    h_kv = n_q_heads // n_q_groups
    torch.manual_seed(seed)
    q = torch.randn(n_q_heads, 1, d, dtype=torch.float16)
    k_blocks, v_blocks = [], []
    for i in range(n_blocks):
        start, end = i * blk, (i + 1) * blk
        kM = to_matrix(torch.randn(h_kv, blk, d))
        vM = to_matrix(torch.randn(h_kv, blk, d))
        kp, _ = quantize_packed("rtn_token", kM, bits=4, group=group, seed=seed)
        vp, _ = quantize_packed("rtn_token", vM, bits=4, group=group, seed=seed)
        k_blocks.append((kp, start, end))
        v_blocks.append((vp, start, end))

    kwargs = dict(
        k_arm="rtn_token",
        v_arm="rtn_token",
        group=group,
        seed=seed,
        k_pre_rope=False,
        rope_cos=None,
        rope_sin=None,
        k_tail=None,
        v_tail=None,
        n_q_groups=n_q_groups,
        scale=d**-0.5,
    )
    return q, k_blocks, v_blocks, kwargs


# ---------------------------------------------------------------------------
# Smoke: module imports cleanly on AMD/no-CUDA
# ---------------------------------------------------------------------------


def test_graphable_module_imports():
    """triton_decode_attention_graphable must import cleanly with TRITON_AVAILABLE=False."""
    from bmx.cache.triton_dequant_attention import (  # noqa: F401
        TRITON_AVAILABLE,
        build_kv_stacked,
        triton_decode_attention_graphable,
    )

    assert isinstance(TRITON_AVAILABLE, bool)


def test_build_kv_stacked_raises_on_k2b():
    """build_kv_stacked must raise NotImplementedError for k2b arm (deferred)."""
    from bmx.cache.triton_dequant_attention import build_kv_stacked

    with pytest.raises(NotImplementedError, match="lowrank_rtn_channel"):
        build_kv_stacked(
            k_blocks=[],
            v_blocks=[],
            max_blocks=4,
            h_kv=2,
            blk_size=64,
            d=64,
            k_arm="lowrank_rtn_channel",
            v_arm="turboquant_mse",
            group=32,
            seed=0,
            device="cpu",  # CPU — avoids CUDA requirement for this smoke test
        )


def test_graphable_requires_cuda_seqlen():
    """triton_decode_attention_graphable must raise if seq_len_dev is not a CUDA tensor."""
    from bmx.cache.triton_dequant_attention import triton_decode_attention_graphable
    import bmx.cache.triton_dequant_attention as mod

    # Temporarily pretend Triton is available so we get past _require_triton().
    original = mod.TRITON_AVAILABLE
    try:
        mod.TRITON_AVAILABLE = True
        q = torch.zeros(4, 1, 64, dtype=torch.float16)
        k_stacked = torch.zeros(2, 2, 64, 64, dtype=torch.float16)
        v_stacked = torch.zeros(2, 2, 64, 64, dtype=torch.float16)
        seq_len_dev_cpu = torch.tensor(64, dtype=torch.int32)  # CPU — wrong
        with pytest.raises(AssertionError, match="CUDA tensor"):
            triton_decode_attention_graphable(
                q,
                k_stacked,
                v_stacked,
                seq_len_dev_cpu,
                n_q_groups=2,
                scale=64**-0.5,
            )
    finally:
        mod.TRITON_AVAILABLE = original


def test_graphable_requires_int32_seqlen():
    """triton_decode_attention_graphable must raise if seq_len_dev is not int32."""
    from bmx.cache.triton_dequant_attention import triton_decode_attention_graphable
    import bmx.cache.triton_dequant_attention as mod

    if not torch.cuda.is_available():
        pytest.skip("CUDA needed to create a CUDA tensor for this smoke test")

    original = mod.TRITON_AVAILABLE
    try:
        mod.TRITON_AVAILABLE = True
        q = torch.zeros(4, 1, 64, dtype=torch.float16, device="cuda")
        k_stacked = torch.zeros(2, 2, 64, 64, dtype=torch.float16, device="cuda")
        v_stacked = torch.zeros(2, 2, 64, 64, dtype=torch.float16, device="cuda")
        seq_len_wrong_dtype = torch.tensor(
            64, dtype=torch.int64, device="cuda"
        )  # int64 — wrong
        with pytest.raises(AssertionError, match="int32"):
            triton_decode_attention_graphable(
                q,
                k_stacked,
                v_stacked,
                seq_len_wrong_dtype,
                n_q_groups=2,
                scale=64**-0.5,
            )
    finally:
        mod.TRITON_AVAILABLE = original


# ---------------------------------------------------------------------------
# Oracle comparison: graphable path vs triton_decode_attention (no graph)
# ---------------------------------------------------------------------------


@cuda
def test_graphable_matches_nongraphable_oracle():
    """triton_decode_attention_graphable must match triton_decode_attention.

    This verifies the graphable path (pre-stacked KV, device seqlen) produces
    the same output as the existing 3a/3b path (Python list blocks). No graph
    capture yet — just correctness of the new path.

    Tolerance: max_abs < 2e-2 (fp16 + quantization noise; expect much tighter).
    """
    from bmx.cache.triton_dequant_attention import (
        TRITON_AVAILABLE,
        build_kv_stacked,
        triton_decode_attention,
        triton_decode_attention_graphable,
    )

    assert TRITON_AVAILABLE, (
        "TRITON_AVAILABLE=False on a CUDA box — install Triton: pip install triton"
    )

    torch.manual_seed(7)
    n_q_heads, n_q_groups, d, blk, n_blocks = 8, 4, 64, 64, 4
    q, kb_cpu, vb_cpu, kw = _make_rtn_blocks(
        n_q_heads=n_q_heads,
        n_q_groups=n_q_groups,
        d=d,
        blk=blk,
        n_blocks=n_blocks,
    )
    h_kv = n_q_heads // n_q_groups

    # Non-graphable reference
    q_cuda = q.cuda()
    kb_cuda = _blocks_cuda(kb_cpu)
    vb_cuda = _blocks_cuda(vb_cpu)
    ref = triton_decode_attention(q_cuda, kb_cuda, vb_cuda, **kw)

    # Graphable path
    seq_len = n_blocks * blk
    k_stacked, v_stacked = build_kv_stacked(
        kb_cpu,
        vb_cpu,
        max_blocks=n_blocks,
        h_kv=h_kv,
        blk_size=blk,
        d=d,
        k_arm=kw["k_arm"],
        v_arm=kw["v_arm"],
        group=kw["group"],
        seed=kw["seed"],
        device="cuda",
    )
    seq_len_dev = torch.tensor(seq_len, dtype=torch.int32, device="cuda")
    out = triton_decode_attention_graphable(
        q_cuda,
        k_stacked,
        v_stacked,
        seq_len_dev,
        n_q_groups=n_q_groups,
        scale=kw["scale"],
    )

    diff = (out.float() - ref.float()).abs()
    max_abs = diff.max().item()
    assert max_abs < 2e-2, (
        f"graphable path diverged from non-graphable oracle: max_abs={max_abs:.4e}. "
        "If >> 2e-2 check: (1) block masking logic in _graphable_decode_kernel, "
        "(2) per-block scratch layout in _graphable_reduce, "
        "(3) build_kv_stacked from_matrix layout vs k_stacked (max_blocks, h_kv, blk, d)."
    )


# ---------------------------------------------------------------------------
# THE REAL GATE: capture -> update -> replay -> compare to fresh
# ---------------------------------------------------------------------------


@cuda
def test_cudagraph_capture_replay_parity():
    """THE GATE: captured CUDA graph replays correctly after seq_len update.

    Protocol:
      1. Build k_stacked/v_stacked at max_blocks (enough for S0+k tokens).
      2. Set seq_len_dev = S0.  Capture a CUDA graph of the decode call.
      3. Update seq_len_dev = S0+k in-place (no re-capture).
         Fill k_stacked slots n_blocks_s0..n_blocks_s0+k-1 with new KV data.
      4. Replay the captured graph.
      5. FRESH call (no graph) with seq_len_dev = S0+k.
      6. Assert: replayed ≈ fresh-at-S0+k  (max_abs < 2e-2).
      7. Assert: replayed ≠ fresh-at-S0    (max_abs > 1e-3, proves replay used S0+k).

    Step 7 is the Python-int-seqlen DETECTOR:
      If seq_len were baked in as a Python int at capture time, the replay would
      attend only the first S0 tokens → matches fresh-at-S0 → step 7 FAILS.
      A correct device-pointer implementation passes step 7 because it attends
      S0+k tokens, which differs from S0-token-only output (different attention
      weights from extra blocks).

    WHAT IS COVERED:
      - Device-pointer seqlen: seq_len_dev is int32 CUDA tensor, updated in-place.
      - Fixed grid: (max_blocks, h_kv, n_q_groups) — same at capture and replay.
      - Block masking: blocks 0..n_s0-1 active at S0; 0..n_total-1 at S0+k.
      - RTN arm (pre-stacked K/V in contiguous device tensor).

    WHAT IS DEFERRED:
      - k2b path: see module docstring.
      - Tail window: process outside the captured graph.
    """
    from bmx.cache.triton_dequant_attention import (
        TRITON_AVAILABLE,
        build_kv_stacked,
        triton_decode_attention_graphable,
    )

    assert TRITON_AVAILABLE, (
        "TRITON_AVAILABLE=False on a CUDA box — install Triton: pip install triton"
    )

    # Geometry: S0 = 4 blocks, S0+k = 6 blocks, max_blocks = 8
    torch.manual_seed(17)
    n_q_heads, n_q_groups, d, blk = 8, 4, 64, 64
    h_kv = n_q_heads // n_q_groups
    n_blocks_s0 = 4  # captured at this length
    n_blocks_extra = 2  # added before replay
    n_blocks_total = n_blocks_s0 + n_blocks_extra
    max_blocks = 8  # fixed grid size
    scale = d**-0.5
    group = 8
    seed = 0
    arm = "rtn_token"

    S0 = n_blocks_s0 * blk
    S_total = n_blocks_total * blk

    # Build the S0 blocks (used at capture time)
    q, kb_s0, vb_s0, kw = _make_rtn_blocks(
        n_q_heads=n_q_heads,
        n_q_groups=n_q_groups,
        d=d,
        blk=blk,
        n_blocks=n_blocks_s0,
        seed=42,
        group=group,
    )
    # Build the extra k blocks (added before replay)
    _, kb_extra, vb_extra, _ = _make_rtn_blocks(
        n_q_heads=n_q_heads,
        n_q_groups=n_q_groups,
        d=d,
        blk=blk,
        n_blocks=n_blocks_extra,
        seed=99,
        group=group,
    )
    # Remap start/end of extra blocks to continue from S0
    kb_extra = [(p, S0 + s, S0 + e) for p, s, e in kb_extra]
    vb_extra = [(p, S0 + s, S0 + e) for p, s, e in vb_extra]

    q_cuda = q.cuda()

    # ------------------------------------------------------------------
    # 1. Allocate k_stacked/v_stacked at max_blocks (fixed size).
    #    Fill slots 0..n_blocks_s0-1 with the S0 blocks.
    # ------------------------------------------------------------------
    k_stacked = torch.zeros(
        max_blocks, h_kv, blk, d, dtype=torch.float16, device="cuda"
    )
    v_stacked = torch.zeros(
        max_blocks, h_kv, blk, d, dtype=torch.float16, device="cuda"
    )

    # Fill S0 slots
    k_s0, v_s0 = build_kv_stacked(
        kb_s0,
        vb_s0,
        max_blocks=n_blocks_s0,
        h_kv=h_kv,
        blk_size=blk,
        d=d,
        k_arm=arm,
        v_arm=arm,
        group=group,
        seed=seed,
        device="cuda",
    )
    k_stacked[:n_blocks_s0] = k_s0
    v_stacked[:n_blocks_s0] = v_s0

    # ------------------------------------------------------------------
    # 2. Capture CUDA graph at S0.
    # ------------------------------------------------------------------
    seq_len_dev = torch.tensor(S0, dtype=torch.int32, device="cuda")

    # Warmup outside the graph (required before capture to avoid spurious allocs)
    for _ in range(3):
        _ = triton_decode_attention_graphable(
            q_cuda,
            k_stacked,
            v_stacked,
            seq_len_dev,
            n_q_groups=n_q_groups,
            scale=scale,
        )
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    # output_captured will be written by the graph replay
    output_captured = torch.zeros(n_q_heads, 1, d, dtype=torch.float16, device="cuda")
    with torch.cuda.graph(g):
        output_captured = triton_decode_attention_graphable(
            q_cuda,
            k_stacked,
            v_stacked,
            seq_len_dev,
            n_q_groups=n_q_groups,
            scale=scale,
        )

    # ------------------------------------------------------------------
    # 3. Update seq_len_dev and fill extra K/V slots (no re-capture).
    # ------------------------------------------------------------------
    k_extra, v_extra = build_kv_stacked(
        kb_extra,
        vb_extra,
        max_blocks=n_blocks_extra,
        h_kv=h_kv,
        blk_size=blk,
        d=d,
        k_arm=arm,
        v_arm=arm,
        group=group,
        seed=seed,
        device="cuda",
    )
    # Fill extra slots in-place (graph captures the pointer, sees new data)
    k_stacked[n_blocks_s0:n_blocks_total] = k_extra
    v_stacked[n_blocks_s0:n_blocks_total] = v_extra

    # Update seq_len in-place (THE graph-safety invariant)
    seq_len_dev.fill_(S_total)

    # ------------------------------------------------------------------
    # 4. Replay the captured graph (no re-capture, no new Python args).
    # ------------------------------------------------------------------
    g.replay()
    torch.cuda.synchronize()
    replayed = output_captured.clone()

    # ------------------------------------------------------------------
    # 5. Fresh (non-graphed) call at S_total for ground truth.
    # ------------------------------------------------------------------
    fresh = triton_decode_attention_graphable(
        q_cuda,
        k_stacked,
        v_stacked,
        seq_len_dev,
        n_q_groups=n_q_groups,
        scale=scale,
    )

    # ------------------------------------------------------------------
    # 6. Assert: replayed ≈ fresh-at-S_total (main correctness check).
    # ------------------------------------------------------------------
    diff_fresh = (replayed.float() - fresh.float()).abs()
    max_abs_fresh = diff_fresh.max().item()
    assert max_abs_fresh < 2e-2, (
        f"CUDA graph replay diverged from fresh call at S_total={S_total}: "
        f"max_abs={max_abs_fresh:.4e}. "
        "This means the graph-safe kernel path is incorrect. "
        "Investigate: (1) k_stacked/v_stacked pointer captured correctly, "
        "(2) seq_len_ptr[0] read at runtime (not cached), "
        "(3) block masking logic in _graphable_decode_kernel."
    )

    # ------------------------------------------------------------------
    # 7. Assert: replayed ≠ fresh-at-S0 (Python-int-seqlen DETECTOR).
    #
    # Compute what the output WOULD be if we only attended the first S0 tokens.
    # A Python-int-seqlen implementation bakes seq_len=S0 into the kernel;
    # replay at S_total would produce this S0-only output.
    # If replayed ≈ out_s0_only → the implementation is WRONG (Python int baked in).
    # ------------------------------------------------------------------
    seq_len_dev_s0 = torch.tensor(S0, dtype=torch.int32, device="cuda")
    out_s0_only = triton_decode_attention_graphable(
        q_cuda,
        k_stacked,
        v_stacked,
        seq_len_dev_s0,
        n_q_groups=n_q_groups,
        scale=scale,
    )

    diff_s0 = (replayed.float() - out_s0_only.float()).abs()
    max_abs_s0 = diff_s0.max().item()
    assert max_abs_s0 > 1e-3, (
        f"CUDA graph replay matches S0-only output (max_abs_s0={max_abs_s0:.4e} < 1e-3). "
        f"This strongly suggests seq_len was baked in as a Python int at capture time "
        f"(seq_len={S0}), not read from the device tensor. "
        "The graph-safe invariant is VIOLATED — the kernel must read seq_len_ptr[0] "
        "from the device tensor, not use a Python-int specialization."
    )


# ---------------------------------------------------------------------------
# Parametrized: partial last block (seq_len not a multiple of blk)
# ---------------------------------------------------------------------------


@cuda
@pytest.mark.parametrize(
    "n_blocks_s0,n_blocks_extra,partial_tokens",
    [
        (4, 2, 0),  # both S0 and S_total block-aligned
        (4, 1, 32),  # S_total has a partial last block (32 of 64 tokens used)
        (3, 2, 16),  # S_total partial: 16 tokens in last block
    ],
)
def test_cudagraph_partial_block_masking(n_blocks_s0, n_blocks_extra, partial_tokens):
    """Graph replay with partial last block (seq_len % blk != 0).

    The kernel masks tokens beyond seq_len_dev[0] in the last block.
    This test exercises the `active = min(blk_size, seq_len - block_start)` path.
    """
    from bmx.cache.triton_dequant_attention import (
        TRITON_AVAILABLE,
        build_kv_stacked,
        triton_decode_attention_graphable,
    )

    assert TRITON_AVAILABLE

    torch.manual_seed(31)
    n_q_heads, n_q_groups, d, blk = 8, 4, 64, 64
    h_kv = n_q_heads // n_q_groups
    max_blocks = 12
    group, seed, arm = 8, 0, "rtn_token"
    scale = d**-0.5

    S0 = n_blocks_s0 * blk
    S_total = (n_blocks_s0 + n_blocks_extra) * blk + partial_tokens

    # Need enough blocks to cover S_total
    n_blocks_alloc = n_blocks_s0 + n_blocks_extra + (1 if partial_tokens > 0 else 0)

    q, kb_all, vb_all, _ = _make_rtn_blocks(
        n_q_heads=n_q_heads,
        n_q_groups=n_q_groups,
        d=d,
        blk=blk,
        n_blocks=n_blocks_alloc,
        seed=77,
        group=group,
    )
    q_cuda = q.cuda()

    k_stacked = torch.zeros(
        max_blocks, h_kv, blk, d, dtype=torch.float16, device="cuda"
    )
    v_stacked = torch.zeros(
        max_blocks, h_kv, blk, d, dtype=torch.float16, device="cuda"
    )

    k_all, v_all = build_kv_stacked(
        kb_all,
        vb_all,
        max_blocks=n_blocks_alloc,
        h_kv=h_kv,
        blk_size=blk,
        d=d,
        k_arm=arm,
        v_arm=arm,
        group=group,
        seed=seed,
        device="cuda",
    )
    k_stacked[:n_blocks_alloc] = k_all
    v_stacked[:n_blocks_alloc] = v_all

    # Capture at S0
    seq_len_dev = torch.tensor(S0, dtype=torch.int32, device="cuda")
    for _ in range(2):
        _ = triton_decode_attention_graphable(
            q_cuda,
            k_stacked,
            v_stacked,
            seq_len_dev,
            n_q_groups=n_q_groups,
            scale=scale,
        )
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    output_captured = torch.zeros(n_q_heads, 1, d, dtype=torch.float16, device="cuda")
    with torch.cuda.graph(g):
        output_captured = triton_decode_attention_graphable(
            q_cuda,
            k_stacked,
            v_stacked,
            seq_len_dev,
            n_q_groups=n_q_groups,
            scale=scale,
        )

    # Update to S_total
    seq_len_dev.fill_(S_total)
    g.replay()
    torch.cuda.synchronize()
    replayed = output_captured.clone()

    # Fresh reference at S_total
    fresh = triton_decode_attention_graphable(
        q_cuda,
        k_stacked,
        v_stacked,
        seq_len_dev,
        n_q_groups=n_q_groups,
        scale=scale,
    )

    diff = (replayed.float() - fresh.float()).abs()
    max_abs = diff.max().item()
    assert max_abs < 2e-2, (
        f"partial-block graph replay failed: "
        f"n_blocks_s0={n_blocks_s0}, n_blocks_extra={n_blocks_extra}, "
        f"partial_tokens={partial_tokens}, S_total={S_total}, "
        f"max_abs={max_abs:.4e}. "
        "Check the `active = min(blk_size, seq_len - block_start)` masking path."
    )
