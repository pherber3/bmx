"""Distortion metrics for KV-cache quality assessment.

These metrics compare an approximated cache (Kq, Vq) against the original
(K, V) using probe queries Q, measuring how much attention logits and outputs
change when the cache is compressed.

No causal mask is applied.  These functions probe the stored cache against
arbitrary query positions to characterise how much information is lost — the
probe queries are not causally constrained to any particular decoding step.

fp32 note
---------
All computations are performed in float32 regardless of input dtype.  Inputs
may be fp16 (as returned by collect_cache) and are cast at entry.  This avoids
catastrophic cancellation and overflow in the Frobenius norms, which can occur
in fp16 for large d or S.

GQA expansion
-------------
When the number of query heads h exceeds the number of KV heads h_kv, each KV
head j serves query heads [j*g, (j+1)*g) where g = h // h_kv.  This matches
how grouped-query attention (GQA) expands KV heads in transformers at inference
time.  Expansion is done via repeat_interleave along dim=0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _expand_kv(K: torch.Tensor, h: int) -> torch.Tensor:
    """Expand K from (h_kv, S, d) to (h, S, d) via repeat_interleave if needed."""
    h_kv = K.shape[0]
    if h_kv == h:
        return K
    g = h // h_kv
    return K.repeat_interleave(g, dim=0)


def _frobenius_rel_error(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Relative Frobenius error ||A - B||_F / ||B||_F, per head.

    A, B: (h, T, S) or (h, T, d_v).
    Returns shape (h,).
    """
    diff = (A - B).flatten(1)  # (h, T*S)
    ref = B.flatten(1)  # (h, T*S)
    return diff.norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-12)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def logit_distortion(
    K: torch.Tensor,
    Kq: torch.Tensor,
    Q: torch.Tensor,
) -> float:
    """Mean over heads of ||Q Kq^T - Q K^T||_F / ||Q K^T||_F.

    Parameters
    ----------
    K  : (h_kv, S, d) — original post-RoPE key cache (fp16 or fp32).
    Kq : (h_kv, S, d) — approximated key cache (same dtype/shape as K).
    Q  : (h, T, d)    — probe queries; h may be a multiple of h_kv (GQA).

    Returns
    -------
    Python float — mean relative Frobenius error across heads.

    Notes
    -----
    GQA expansion: K and Kq are repeat_interleaved along dim=0 by g = h // h_kv
    so that KV head j serves query heads [j*g, (j+1)*g), matching the
    standard GQA attention pattern in transformers.

    All computation is in float32; inputs are cast at entry (see module note).
    No causal mask is applied.
    """
    # Cast to fp32 for numerical stability (inputs may be fp16 from collect_cache)
    K = K.float()
    Kq = Kq.float()
    Q = Q.float()

    h = Q.shape[0]
    K_exp = _expand_kv(K, h)  # (h, S, d)
    Kq_exp = _expand_kv(Kq, h)  # (h, S, d)

    # Q @ K^T: (h, T, d) x (h, d, S) -> (h, T, S)
    logits_ref = Q @ K_exp.transpose(-1, -2)  # (h, T, S)
    logits_approx = Q @ Kq_exp.transpose(-1, -2)  # (h, T, S)

    per_head_err = _frobenius_rel_error(logits_approx, logits_ref)  # (h,)
    return per_head_err.mean().item()


def attn_output_distortion(
    K: torch.Tensor,
    V: torch.Tensor,
    Kq: torch.Tensor,
    Vq: torch.Tensor,
    Q: torch.Tensor,
) -> float:
    """Mean over heads of rel Frobenius error of softmax(Q K^T / sqrt(d)) V.

    Parameters
    ----------
    K  : (h_kv, S, d)  — original key cache (fp16 or fp32).
    V  : (h_kv, S, d_v) — original value cache.
    Kq : (h_kv, S, d)  — approximated key cache.
    Vq : (h_kv, S, d_v) — approximated value cache.
    Q  : (h, T, d)      — probe queries; h may be a multiple of h_kv (GQA).

    Returns
    -------
    Python float — mean relative Frobenius error of attention output across heads.

    Notes
    -----
    GQA expansion: K, V, Kq, Vq are all repeat_interleaved by g = h // h_kv.

    No causal mask is applied — metrics probe the full stored cache against
    the probe queries regardless of position.

    All computation is in float32; inputs are cast at entry (see module note).
    """
    # Cast to fp32 for numerical stability
    K = K.float()
    V = V.float()
    Kq = Kq.float()
    Vq = Vq.float()
    Q = Q.float()

    h = Q.shape[0]
    d = Q.shape[-1]

    K_exp = _expand_kv(K, h)  # (h, S, d)
    V_exp = _expand_kv(V, h)  # (h, S, d_v)
    Kq_exp = _expand_kv(Kq, h)  # (h, S, d)
    Vq_exp = _expand_kv(Vq, h)  # (h, S, d_v)

    scale = d**-0.5

    # Reference attention output
    logits_ref = Q @ K_exp.transpose(-1, -2) * scale  # (h, T, S)
    attn_ref = F.softmax(logits_ref, dim=-1)  # (h, T, S)
    out_ref = attn_ref @ V_exp  # (h, T, d_v)

    # Approximated attention output
    logits_approx = Q @ Kq_exp.transpose(-1, -2) * scale  # (h, T, S)
    attn_approx = F.softmax(logits_approx, dim=-1)  # (h, T, S)
    out_approx = attn_approx @ Vq_exp  # (h, T, d_v)

    per_head_err = _frobenius_rel_error(out_approx, out_ref)  # (h,)
    return per_head_err.mean().item()
