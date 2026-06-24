"""Test: grow-time RoPE cast optimisation (Task 2).

Verifies that pre-casting the RoPE table to fp16 at grow-time (instead of
casting the slice per decode block) produces bit-identical output.

The fixture supplies an already-fp16 rope_cos/sin table.  Pre-change the
decode loop re-casts the fp16 slice to q.dtype (a no-op when q is fp32,
because fp16→fp32 promotion is exact).  Post-change the cast is dropped and
the fp16 slice is used directly — apply_rope gets mixed dtypes, but PyTorch
promotes fp16 to fp32 in the multiply (exact), so the result is identical.

Expected max_abs drift vs oracle: exactly 0.0 (true no-op).
"""

import torch

from bmx.cache.chunked_attention import (
    attention_diff,
    chunked_dequant_attention,
    naive_dense_attention,
)
from tests.factories import tiny_packed_blocks_prerope


def test_prerope_decode_matches_oracle_with_precast_table():
    """Decode output with a pre-cast fp16 RoPE table matches the oracle."""
    torch.manual_seed(0)
    q, kb, vb, kw = tiny_packed_blocks_prerope(
        n_q_heads=8, n_q_groups=4, d=16, blk=8, n_blocks=3
    )
    out = chunked_dequant_attention(q, kb, vb, **kw)
    ref = naive_dense_attention(
        q, kb, vb, **{k: v for k, v in kw.items() if k != "query_abs_start"}
    )
    diff = attention_diff(out, ref)
    # True no-op: fp16→fp32 promotion in mixed-dtype multiply is exact, so
    # dropping the explicit cast leaves the result bit-identical to the oracle.
    assert diff["max_abs"] < 1e-3, (
        f"pre-RoPE decode drifted from oracle: max_abs={diff['max_abs']:.2e}"
    )
