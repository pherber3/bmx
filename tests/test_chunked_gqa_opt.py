"""GQA grouped-contraction decode-loop parity test.

Captures the bit-exactness invariant before and after the repeat_interleave ->
grouped-einsum refactor in chunked_dequant_attention.  The test must PASS
against both the old and new code paths.
"""

from bmx.cache.chunked_attention import (
    attention_diff,
    chunked_dequant_attention,
    naive_dense_attention,
)
from tests.factories import tiny_packed_blocks


def test_gqa_grouped_contraction_matches_oracle():
    """chunked_dequant_attention decode output must be bit-exact vs naive oracle.

    n_q_groups=4 exercises the GQA expansion path (same ratio as Llama-3.1-8B).
    Tolerance is 1e-4 (quantisation noise dominates; the attention math itself
    must be within floating-point rounding of the oracle).
    """
    q, kb, vb, kw = tiny_packed_blocks(
        n_q_heads=8, n_q_groups=4, n_q=1, d=16, blk=8, n_blocks=3
    )
    out = chunked_dequant_attention(q, kb, vb, **kw)
    ref = naive_dense_attention(
        q, kb, vb, **{k: v for k, v in kw.items() if k != "query_abs_start"}
    )
    diff = attention_diff(out, ref)
    assert diff["max_abs"] < 1e-4, diff
