"""CUDA-gated bit-exact test for the Triton RTN decode online-softmax kernel.

Skips entirely on non-CUDA machines (AMD dev box, CI without GPU).
On the VM (CUDA available):
  - Verifies TRITON_AVAILABLE is True (fail loud if Triton not installed).
  - Runs triton_decode_attention vs naive_dense_attention (oracle) on matching
    packed inputs, asserts max_abs < 1e-2 (expect much tighter, ~fp16 rounding).

Skip reason is printed verbosely so the VM operator knows WHY it was skipped
if they accidentally run the suite on a non-CUDA box.
"""

import pytest
import torch

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton decode kernel — VM/CUDA only (skipping on non-CUDA host)",
)


# ---------------------------------------------------------------------------
# Helper: move a packed-block list's dict tensors to CUDA
# ---------------------------------------------------------------------------


def _blocks_cuda(blocks: list) -> list:
    """Move each packed dict's tensors to CUDA; keep start/end ints unchanged."""
    out = []
    for packed, start, end in blocks:
        packed_cuda = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in packed.items()
        }
        out.append((packed_cuda, start, end))
    return out


# ---------------------------------------------------------------------------
# Smoke test: import + TRITON_AVAILABLE gate
# ---------------------------------------------------------------------------


def test_triton_module_imports_with_available_flag():
    """Module must import cleanly on AMD/no-CUDA with TRITON_AVAILABLE=False."""
    from bmx.cache.triton_dequant_attention import TRITON_AVAILABLE  # noqa: F401

    # On non-CUDA hosts TRITON_AVAILABLE is False — that is the correct state.
    # On a CUDA host with Triton installed it should be True.
    # Either way, the import must not raise.
    assert isinstance(TRITON_AVAILABLE, bool)


def test_require_triton_raises_without_cuda():
    """_require_triton() must raise RuntimeError when TRITON_AVAILABLE is False."""
    import bmx.cache.triton_dequant_attention as mod

    original = mod.TRITON_AVAILABLE
    try:
        mod.TRITON_AVAILABLE = False
        with pytest.raises(RuntimeError, match="TRITON_AVAILABLE"):
            mod._require_triton()
    finally:
        mod.TRITON_AVAILABLE = original


# ---------------------------------------------------------------------------
# Bit-exact oracle comparison (CUDA-only)
# ---------------------------------------------------------------------------


@cuda
def test_triton_rtn_decode_matches_oracle():
    """triton_decode_attention must match naive_dense_attention to fp16 tolerance.

    Uses rtn_token arm (the default streaming path), no RoPE, no tail.
    Oracle runs on CPU copies of the same packed data; Triton kernel runs on CUDA.
    Asserts max_abs < 1e-2; expect much tighter (~2e-4 at fp16).
    """
    from bmx.cache.chunked_attention import attention_diff, naive_dense_attention
    from bmx.cache.triton_dequant_attention import (
        TRITON_AVAILABLE,
        triton_decode_attention,
    )

    assert TRITON_AVAILABLE, (
        "TRITON_AVAILABLE=False on a CUDA box — Triton not installed correctly. "
        "Install with: pip install triton"
    )

    from tests.factories import tiny_packed_blocks

    torch.manual_seed(42)
    n_q_heads, n_q_groups, d, blk, n_blocks = 8, 4, 64, 64, 4
    q, kb_cpu, vb_cpu, kw = tiny_packed_blocks(
        n_q_heads=n_q_heads,
        n_q_groups=n_q_groups,
        n_q=1,
        d=d,
        blk=blk,
        n_blocks=n_blocks,
    )

    # Oracle on CPU (matches tiny_packed_blocks defaults: no rope, no tail)
    oracle_kwargs = {k: v for k, v in kw.items() if k != "query_abs_start"}
    ref_cpu = naive_dense_attention(q, kb_cpu, vb_cpu, **oracle_kwargs)

    # Triton kernel on CUDA
    q_cuda = q.cuda()
    kb_cuda = _blocks_cuda(kb_cpu)
    vb_cuda = _blocks_cuda(vb_cpu)
    # rope_cos/sin are None for rtn_token without rope; k_tail/v_tail are None.
    out_cuda = triton_decode_attention(q_cuda, kb_cuda, vb_cuda, **oracle_kwargs)

    # Compare on CPU (move Triton result back)
    out_cpu = out_cuda.cpu()
    diff = attention_diff(out_cpu, ref_cpu)
    assert diff["max_abs"] < 1e-2, (
        f"Triton decode kernel drifted from oracle: {diff}. "
        "If max_abs >= 1e-2 investigate the kernel — do NOT loosen this tolerance."
    )


@cuda
def test_triton_rtn_decode_matches_oracle_prerope():
    """Same as above but with k_pre_rope=True (RoPE applied inside the loop)."""
    from bmx.cache.chunked_attention import attention_diff, naive_dense_attention
    from bmx.cache.triton_dequant_attention import triton_decode_attention

    from tests.factories import tiny_packed_blocks_prerope

    torch.manual_seed(7)
    n_q_heads, n_q_groups, d, blk, n_blocks = 8, 4, 64, 64, 4
    q, kb_cpu, vb_cpu, kw = tiny_packed_blocks_prerope(
        n_q_heads=n_q_heads,
        n_q_groups=n_q_groups,
        d=d,
        blk=blk,
        n_blocks=n_blocks,
    )

    oracle_kwargs = {k: v for k, v in kw.items() if k != "query_abs_start"}
    ref_cpu = naive_dense_attention(q, kb_cpu, vb_cpu, **oracle_kwargs)

    # Move to CUDA: q, kb, vb, plus rope tensors
    q_cuda = q.cuda()
    kb_cuda = _blocks_cuda(kb_cpu)
    vb_cuda = _blocks_cuda(vb_cpu)
    kw_cuda = dict(oracle_kwargs)
    if kw_cuda.get("rope_cos") is not None:
        kw_cuda["rope_cos"] = kw_cuda["rope_cos"].cuda()
    if kw_cuda.get("rope_sin") is not None:
        kw_cuda["rope_sin"] = kw_cuda["rope_sin"].cuda()

    out_cuda = triton_decode_attention(q_cuda, kb_cuda, vb_cuda, **kw_cuda)
    out_cpu = out_cuda.cpu()
    diff = attention_diff(out_cpu, ref_cpu)
    assert diff["max_abs"] < 1e-2, (
        f"Triton decode kernel (pre-RoPE) drifted from oracle: {diff}. "
        "Investigate — do NOT loosen tolerance."
    )


@cuda
def test_triton_decode_asserts_n_q_eq_1():
    """triton_decode_attention must raise if n_q != 1 (prefill guard)."""
    from bmx.cache.triton_dequant_attention import triton_decode_attention

    from tests.factories import tiny_packed_blocks

    _, kb_cpu, vb_cpu, kw = tiny_packed_blocks(
        n_q_heads=4, n_q_groups=2, n_q=1, d=32, blk=32, n_blocks=2
    )
    q_bad = torch.randn(4, 3, 32).cuda()  # n_q=3, not 1
    kwargs = {k: v for k, v in kw.items() if k != "query_abs_start"}
    kb_cuda = _blocks_cuda(kb_cpu)
    vb_cuda = _blocks_cuda(vb_cpu)
    with pytest.raises(AssertionError, match="decode-only"):
        triton_decode_attention(q_bad, kb_cuda, vb_cuda, **kwargs)


# ---------------------------------------------------------------------------
# Stage 3b: split-KV parametrized oracle test
# ---------------------------------------------------------------------------


@cuda
@pytest.mark.parametrize(
    "num_splits,n_blocks",
    [
        (1, 16),  # even: 16 blocks, 1 split  — baseline (3a-identical path)
        (2, 16),  # even: 16 blocks, 2 splits of 8
        (4, 16),  # even: 16 blocks, 4 splits of 4
        (8, 16),  # even: 16 blocks, 8 splits of 2
        (4, 5),  # UNEVEN remainder: 5 blocks, 4 splits → [2,2,1,0] sizes
    ],
)
def test_triton_split_kv_matches_oracle(num_splits, n_blocks):
    """triton_decode_attention(num_splits=N) must match naive_dense_attention.

    Tests even split counts {1,2,4,8} (all divide n_blocks=16 evenly) AND the
    UNEVEN case (n_blocks=5, num_splits=4 → split sizes [2,2,1,0]) that
    exercises the 1-block split and the empty-split skip path.

    A wrong LSE merge produces O(1) error at >1 split — caught here before any
    VM run.

    Tolerance: max_abs < 1e-2.  Expect near fp16 rounding (~2e-4).
    Do NOT loosen this bound — drift means the merge formula is wrong.
    """
    from bmx.cache.chunked_attention import attention_diff, naive_dense_attention
    from bmx.cache.triton_dequant_attention import (
        TRITON_AVAILABLE,
        triton_decode_attention,
    )

    assert TRITON_AVAILABLE, (
        "TRITON_AVAILABLE=False on a CUDA box — Triton not installed correctly. "
        "Install with: pip install triton"
    )

    from tests.factories import tiny_packed_blocks

    torch.manual_seed(0)
    n_q_heads, n_q_groups, d, blk = 8, 4, 64, 64
    q, kb_cpu, vb_cpu, kw = tiny_packed_blocks(
        n_q_heads=n_q_heads,
        n_q_groups=n_q_groups,
        n_q=1,
        d=d,
        blk=blk,
        n_blocks=n_blocks,
    )

    # Oracle: naive_dense_attention on CPU (no num_splits concept; full softmax).
    oracle_kwargs = {k: v for k, v in kw.items() if k != "query_abs_start"}
    ref_cpu = naive_dense_attention(q, kb_cpu, vb_cpu, **oracle_kwargs)

    # Triton split-KV on CUDA.
    q_cuda = q.cuda()
    kb_cuda = _blocks_cuda(kb_cpu)
    vb_cuda = _blocks_cuda(vb_cpu)
    out_cuda = triton_decode_attention(
        q_cuda, kb_cuda, vb_cuda, num_splits=num_splits, **oracle_kwargs
    )
    out_cpu = out_cuda.cpu()

    diff = attention_diff(out_cpu, ref_cpu)
    assert diff["max_abs"] < 1e-2, (
        f"triton_decode_attention(num_splits={num_splits}, n_blocks={n_blocks}) "
        f"drifted from oracle: {diff}.  "
        "If max_abs >= 1e-2 the LSE merge is wrong — do NOT loosen."
    )


@cuda
def test_triton_split_kv_num_splits_1_bit_identical_to_3a():
    """num_splits=1 must be bit-identical to the 3a default path (no num_splits).

    The 3b refactor introduced num_splits as an explicit kwarg; when num_splits=1
    the code takes the same `acc / lse` shortcut as the old 3a serial path.
    This test proves back-compat: calling with num_splits=1 and calling with
    the default (no num_splits arg) must produce EXACTLY the same fp16 tensor
    (torch.equal, not just close), AND both must match naive_dense_attention
    within fp16 tolerance.

    Invariant verified:
      out_default (3a path, no num_splits) == out_split1 (num_splits=1)  [bit-exact]
      out_split1 vs oracle: max_abs < 1e-2

    If torch.equal fails: the 3b refactor changed the num_splits=1 code path.
    If the oracle check fails: the serial path itself is broken.
    """
    from bmx.cache.chunked_attention import attention_diff, naive_dense_attention
    from bmx.cache.triton_dequant_attention import (
        TRITON_AVAILABLE,
        triton_decode_attention,
    )

    assert TRITON_AVAILABLE, "Triton not available — install triton on this host"

    from tests.factories import tiny_packed_blocks

    torch.manual_seed(99)
    n_q_heads, n_q_groups, d, blk, n_blocks = 8, 4, 64, 64, 8
    q, kb_cpu, vb_cpu, kw = tiny_packed_blocks(
        n_q_heads=n_q_heads,
        n_q_groups=n_q_groups,
        n_q=1,
        d=d,
        blk=blk,
        n_blocks=n_blocks,
    )
    oracle_kwargs = {k: v for k, v in kw.items() if k != "query_abs_start"}

    q_cuda = q.cuda()
    kb_cuda = _blocks_cuda(kb_cpu)
    vb_cuda = _blocks_cuda(vb_cpu)

    # 3a default path: no num_splits kwarg (uses the default=1 shortcut).
    out_default = triton_decode_attention(q_cuda, kb_cuda, vb_cuda, **oracle_kwargs)
    # 3b explicit num_splits=1: must follow the identical code branch.
    out_split1 = triton_decode_attention(
        q_cuda, kb_cuda, vb_cuda, num_splits=1, **oracle_kwargs
    )

    # Bit-identical: num_splits=1 IS the 3a path — same tensors, not just close.
    assert torch.equal(out_default, out_split1), (
        "num_splits=1 diverged from the default (3a) path — "
        "the 3b refactor broke back-compat on the serial path."
    )

    # Also confirm both agree with the oracle (catches a broken serial path).
    ref_cpu = naive_dense_attention(q, kb_cpu, vb_cpu, **oracle_kwargs)
    diff = attention_diff(out_split1.cpu(), ref_cpu)
    assert diff["max_abs"] < 1e-2, (
        f"num_splits=1 drifted from oracle: {diff}. "
        "Investigate the serial path — do NOT loosen tolerance."
    )
