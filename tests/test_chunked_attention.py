"""Chunked dequant-attention: oracle, online-softmax exactness, schedule."""

import torch

from bmx.cache.chunked_attention import (
    attention_diff,
    naive_dense_attention,
    online_softmax_update,
)
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
