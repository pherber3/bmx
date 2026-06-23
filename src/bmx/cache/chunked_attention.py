"""Chunked dequant-attention + the naive golden reference oracle.

The packed cache stores compressed codes. Two attention paths share one call
shape so they are drop-in comparable:
  - naive_dense_attention  — the ORACLE: dequant everything, ONE full softmax, no
    online trick, no chunking. Slowest, most obviously correct. Every faster path
    (online-softmax, chunked, packed, future Triton) is diffed against THIS, and
    attention_diff() quantifies the drift. The yardstick that keeps us honest.
  - chunked_dequant_attention (Task 5) — dequant ONE block at a time, online
    softmax (exact — Physics of LLM Inference ~line 1931), free the block. Never
    materializes full dense K/V or the full score row.
"""

from __future__ import annotations

import torch

from bmx.cache.codecs import dequant_packed
from bmx.cache.collect import from_matrix
from bmx.cache.rope import apply_rope


def online_softmax_update(acc, m, lse, scores_new, v_new):
    """One online-softmax step.

    acc:(...,n_q,d) m,lse:(...,n_q,1) scores_new:(...,n_q,blk) v_new:(...,blk,d).
    lse is the running log-sum-exp denominator (unnormalized sum of exp weights).
    Returns updated (acc, m, lse). Divide acc by lse after the last block.
    """
    m_new = torch.maximum(m, scores_new.amax(dim=-1, keepdim=True))
    correction = torch.exp(m - m_new)  # (...,n_q,1); <=1, never overflows
    p = torch.exp(scores_new - m_new)  # (...,n_q,blk)
    lse = lse * correction + p.sum(dim=-1, keepdim=True)
    acc = acc * correction + p @ v_new  # (...,n_q,d)
    return acc, m_new, lse


def attention_diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Quantify drift between two attention outputs (oracle vs fast path)."""
    diff = (a.double() - b.double()).abs()
    denom = b.double().abs().clamp_min(1e-12)
    return {
        "max_abs": float(diff.max()),
        "max_rel": float((diff / denom).max()),
        "mean_abs": float(diff.mean()),
    }


def _dequant_block(packed, arm, group, seed, h_kv):
    """packed dict -> (h_kv, blk, d) dense, matching to_matrix layout."""
    M = (
        packed["fp16"]
        if arm == "fp16"
        else dequant_packed(arm, packed, group=group, seed=seed)
    )
    return from_matrix(M, h_kv)


def _dense_kv(blocks, arm, group, seed, h_kv, k_pre_rope, rope_cos, rope_sin):
    """Dequant all blocks to one dense (h_kv, S_committed, d), RoPE-at-read for K."""
    parts = []
    for packed, start, end in blocks:
        B = _dequant_block(packed, arm, group, seed, h_kv)
        if k_pre_rope:
            B = apply_rope(
                B,
                rope_cos[start:end].to(B.dtype),
                rope_sin[start:end].to(B.dtype),
            )
        parts.append(B)
    return torch.cat(parts, dim=1) if parts else None


def naive_dense_attention(
    q,
    k_blocks,
    v_blocks,
    *,
    k_arm,
    v_arm,
    group,
    seed,
    k_pre_rope,
    rope_cos,
    rope_sin,
    k_tail,
    v_tail,
    n_q_groups,
    scale,
):
    """ORACLE: dequant everything, single full softmax, GQA-expand. No chunking.

    Same call shape as chunked_dequant_attention so they are drop-in comparable.
    """
    n_q_heads = q.shape[0]
    h_kv = n_q_heads // n_q_groups
    K = _dense_kv(k_blocks, k_arm, group, seed, h_kv, k_pre_rope, rope_cos, rope_sin)
    V = _dense_kv(v_blocks, v_arm, group, seed, h_kv, False, None, None)
    if k_tail is not None and k_tail.shape[1] > 0:
        K = (
            k_tail.to(q.dtype)
            if K is None
            else torch.cat([K, k_tail.to(q.dtype)], dim=1)
        )
        V = (
            v_tail.to(q.dtype)
            if V is None
            else torch.cat([V, v_tail.to(q.dtype)], dim=1)
        )
    Kx = K.to(q.dtype).repeat_interleave(n_q_groups, dim=0)
    Vx = V.to(q.dtype).repeat_interleave(n_q_groups, dim=0)
    scores = (q @ Kx.transpose(-1, -2)) * scale
    return torch.softmax(scores, dim=-1) @ Vx
