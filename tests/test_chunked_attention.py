"""Chunked dequant-attention: oracle, online-softmax exactness, schedule."""

import pytest
import torch

from bmx.cache.chunked_attention import (
    attention_diff,
    chunked_dequant_attention,
    naive_dense_attention,
    online_softmax_update,
)
from bmx.cache.codecs import quantize_packed
from bmx.cache.collect import to_matrix
from bmx.cache.streaming import compute_flush_schedule


def _pack_fp16_block(kv_slice: torch.Tensor) -> torch.Tensor:
    """Dtype-preserving reshape (h, S, d) -> (S, h*d) for fp16-arm test blocks.

    to_matrix() calls .float() which coerces fp64 -> fp32, breaking the
    atol=1e-12 oracle test.  This helper does the same reshape without the
    dtype coercion so the test can verify attention math at fp64 precision.
    """
    h, s, d = kv_slice.shape
    return kv_slice.permute(1, 0, 2).reshape(s, h * d)


def test_oracle_equals_hand_softmax_gqa():
    # The oracle IS the ground truth; pin it against a from-scratch GQA softmax so
    # a future edit to the oracle can't silently corrupt the yardstick.
    torch.manual_seed(0)
    h_kv, n_q_heads, n_q, S, d = 2, 4, 1, 32, 8
    q = torch.randn(n_q_heads, n_q, d, dtype=torch.float64)
    K = torch.randn(h_kv, S, d, dtype=torch.float64)
    V = torch.randn(h_kv, S, d, dtype=torch.float64)
    scale = 1.0 / (d**0.5)
    # fp16-arm packed blocks of 16 (so dequant is identity, isolating attention).
    # NOTE: _pack_fp16_block used (not to_matrix) because to_matrix calls .float()
    # which coerces fp64->fp32, destroying the 1e-12 precision the oracle test needs.
    k_blocks, v_blocks = [], []
    for j in range(0, S, 16):
        k_blocks.append(({"fp16": _pack_fp16_block(K[:, j : j + 16])}, j, j + 16))
        v_blocks.append(({"fp16": _pack_fp16_block(V[:, j : j + 16])}, j, j + 16))
    out = naive_dense_attention(
        q,
        k_blocks,
        v_blocks,
        k_arm="fp16",
        v_arm="fp16",
        group=8,
        seed=0,
        k_pre_rope=False,
        rope_cos=None,
        rope_sin=None,
        k_tail=None,
        v_tail=None,
        n_q_groups=n_q_heads // h_kv,
        scale=scale,
    )
    Kx = K.repeat_interleave(n_q_heads // h_kv, dim=0)
    Vx = V.repeat_interleave(n_q_heads // h_kv, dim=0)
    ref = torch.softmax((q @ Kx.transpose(-1, -2)) * scale, dim=-1) @ Vx
    assert torch.allclose(out, ref, atol=1e-12, rtol=1e-12)


def test_attention_diff_reports_zero_for_identical():
    a = torch.randn(2, 1, 8, dtype=torch.float64)
    d = attention_diff(a, a.clone())
    assert d["max_abs"] == 0.0 and d["max_rel"] == 0.0 and d["mean_abs"] == 0.0


def test_flush_schedule_matches_formula():
    # largest multiple of g leaving >= W recent tokens, else 0.
    assert compute_flush_schedule(S=100, W=32, g=16) == 64
    assert compute_flush_schedule(S=40, W=32, g=16) == 0  # (40-32)//16*16 = 0
    assert compute_flush_schedule(S=20, W=32, g=16) == 0  # S <= W
    assert compute_flush_schedule(S=160, W=32, g=1) == 128


def test_online_softmax_equals_full_softmax():
    torch.manual_seed(0)
    h, n_q, S, d = 2, 1, 48, 8
    q = torch.randn(h, n_q, d, dtype=torch.float64)
    K = torch.randn(h, S, d, dtype=torch.float64)
    V = torch.randn(h, S, d, dtype=torch.float64)
    scale = 1.0 / (d**0.5)

    # Reference: full softmax over all S keys.
    full_scores = (q @ K.transpose(-1, -2)) * scale  # (h, n_q, S)
    ref = torch.softmax(full_scores, dim=-1) @ V  # (h, n_q, d)

    # Streamed in blocks of 16.
    acc = torch.zeros(h, n_q, d, dtype=torch.float64)
    m = torch.full((h, n_q, 1), float("-inf"), dtype=torch.float64)
    lse = torch.zeros(h, n_q, 1, dtype=torch.float64)
    for j in range(0, S, 16):
        Kb, Vb = K[:, j : j + 16], V[:, j : j + 16]
        s = (q @ Kb.transpose(-1, -2)) * scale  # (h, n_q, blk)
        acc, m, lse = online_softmax_update(acc, m, lse, s, Vb)
    out = acc / lse
    assert torch.allclose(out, ref, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "k_arm,v_arm,kw",
    [
        ("fp16", "fp16", {}),
        ("turboquant_mse", "turboquant_mse", dict(bits=2)),
    ],
)
def test_chunked_matches_oracle_no_rope(k_arm, v_arm, kw):
    # chunked dequant-attn must equal the oracle (dequant-all + full softmax) over
    # the SAME packed blocks. Isolates the online-softmax + per-block assembly.
    torch.manual_seed(0)
    h_kv, n_q_heads, n_q, S, d = 2, 4, 1, 48, 8  # GQA: 4 q-heads over 2 kv-heads
    group = 8
    q = torch.randn(n_q_heads, n_q, d, dtype=torch.float64)
    K = torch.randn(h_kv, S, d, dtype=torch.float64)
    V = torch.randn(h_kv, S, d, dtype=torch.float64)
    scale = 1.0 / (d**0.5)

    def pack_side(T, arm):
        blocks = []
        for j in range(0, S, 16):
            M = to_matrix(T[:, j : j + 16])  # (16, h_kv*d)
            packed = (
                {"fp16": M}
                if arm == "fp16"
                else quantize_packed(arm, M, group=group, **kw)[0]
            )
            blocks.append((packed, j, j + 16))
        return blocks

    k_blocks, v_blocks = pack_side(K, k_arm), pack_side(V, v_arm)
    common = dict(
        k_arm=k_arm,
        v_arm=v_arm,
        group=group,
        seed=0,
        k_pre_rope=False,
        rope_cos=None,
        rope_sin=None,
        k_tail=None,
        v_tail=None,
        n_q_groups=n_q_heads // h_kv,
        scale=scale,
    )

    oracle = naive_dense_attention(q, k_blocks, v_blocks, **common)
    fast = chunked_dequant_attention(q, k_blocks, v_blocks, **common)

    drift = attention_diff(fast, oracle)
    assert drift["max_abs"] < 1e-10, drift  # online softmax is exact vs oracle


def test_empty_committed_blocks_degenerates_to_tail_attention():
    """chunked_dequant_attention with no committed blocks equals plain softmax on tail.

    This is the all-fp16-tail case: the first flush has not yet fired, so
    k_blocks/v_blocks are empty and all KV lives in the tail window.
    chunked_dequant_attention must degenerate to exactly naive_dense_attention over
    the tail alone (both should match a from-scratch softmax).
    """
    torch.manual_seed(42)
    h_kv, n_q_heads, n_q, tail_len, d = 2, 4, 1, 16, 8
    q = torch.randn(n_q_heads, n_q, d, dtype=torch.float64)
    k_tail = torch.randn(h_kv, tail_len, d, dtype=torch.float64)
    v_tail = torch.randn(h_kv, tail_len, d, dtype=torch.float64)
    scale = 1.0 / (d**0.5)
    n_q_groups = n_q_heads // h_kv

    common = dict(
        k_arm="fp16",
        v_arm="fp16",
        group=8,
        seed=0,
        k_pre_rope=False,
        rope_cos=None,
        rope_sin=None,
        k_tail=k_tail,
        v_tail=v_tail,
        n_q_groups=n_q_groups,
        scale=scale,
    )

    # Both paths with empty block lists.
    oracle = naive_dense_attention(q, [], [], **common)
    chunked = chunked_dequant_attention(q, [], [], **common)

    # Reference: plain softmax over the expanded tail.
    Kx = k_tail.repeat_interleave(n_q_groups, dim=0)  # (n_q_heads, tail_len, d)
    Vx = v_tail.repeat_interleave(n_q_groups, dim=0)
    ref = torch.softmax((q @ Kx.transpose(-1, -2)) * scale, dim=-1) @ Vx

    assert torch.allclose(oracle, ref, atol=1e-10, rtol=0), (
        "oracle with empty blocks diverged from plain softmax"
    )
    assert torch.allclose(chunked, ref, atol=1e-10, rtol=0), (
        "chunked with empty blocks diverged from plain softmax"
    )
