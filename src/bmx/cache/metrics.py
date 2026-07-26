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

from bmx.cache.rope import apply_rope


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


def rel_fro(A: torch.Tensor, B: torch.Tensor) -> float:
    """Relative Frobenius error ||A - B||_F / ||B||_F (fp32, clamped denominator).

    A is the approximation, B the reference. Inputs may be fp16; both are cast
    to fp32 at entry (see module note).
    """
    A = A.float()
    B = B.float()
    return ((A - B).norm() / B.norm().clamp_min(1e-12)).item()


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


def logit_distortion_causal(
    K: torch.Tensor,
    Kq: torch.Tensor,
    Q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    q_start: int,
) -> float:
    """The THIRD INSTRUMENT (K4 math review #3(b)/(c)): true causal
    per-position logit error, masked to real attention pairs.

    Delta L_{t,s} = (R_t q_t)^T e_s,  e_s = Kq_s - K_s,  s <= t only.

    R is the FORWARD RoPE rotation (`apply_rope`, positive sin — unlike
    `logit_distortion`, which leaves Q un-rotated and so implicitly measures
    an inverse-rotation, time-reversed, causally-unmasked quadratic form; see
    module docstring of `bmx.cache.spectral.query_position_moment` and math
    review finding #3). By the RoPE-composition identity
    `(R_t q)^T(R_s k) = (R_{t-s} q)^T k` (pinned in
    `tests/test_cache_metrics.py::test_rope_composition_absolute_equals_relative`),
    this is equivalent to forward-rotating q by the relative offset t-s and
    dotting against the UN-rotated key error — this function uses the
    absolute-position form because K/Kq are already stored post-RoPE
    (`apply_rope` at read, the codebase convention) and Q is not.

    Parameters
    ----------
    K, Kq : (h_kv, S, d) — original / approximated POST-RoPE key caches
        (fp16 or fp32), covering the FULL sequence (every causal source
        position s a masked query can attend to).
    Q     : (h, T, d)    — PRE-RoPE probe queries; h may be a multiple of
        h_kv (GQA). Row i sits at TRUE absolute position q_start + i.
    cos, sin : (S, d)    — RoPE tables from `rope_cos_sin`, absolute
        positions [0, S).
    q_start  : absolute position of Q's first row (e.g. S - T for the
        stored last-T-token probe-query window).

    Returns
    -------
    Python float — mean over heads of the relative Frobenius error of the
    causal logit-error matrix: for each head, ||Delta L||_F / ||L_true||_F
    computed over the masked (t, s), s <= t entries only (mirrors
    `logit_distortion`'s per-head relative-Frobenius normalization
    convention — see `_frobenius_rel_error`).

    No GQA head may see a future key: the mask is applied per (t, s) pair
    before the Frobenius reduction, so causality holds exactly regardless of
    head expansion.
    """
    K = K.float()
    Kq = Kq.float()
    Q = Q.float()
    cos = cos.float()
    sin = sin.float()

    h = Q.shape[0]
    T = Q.shape[1]
    S = K.shape[1]
    K_exp = _expand_kv(K, h)  # (h, S, d)
    Kq_exp = _expand_kv(Kq, h)  # (h, S, d)

    # Forward-rotate the probe queries at their TRUE absolute positions.
    q_positions = torch.arange(q_start, q_start + T, device=Q.device)
    assert q_positions.max() < S, (
        f"probe query position {int(q_positions.max())} out of range S={S}"
    )
    Rt_Q = apply_rope(Q, cos[q_positions], sin[q_positions])  # (h, T, d)

    logits_ref = Rt_Q @ K_exp.transpose(-1, -2)  # (h, T, S)
    logits_approx = Rt_Q @ Kq_exp.transpose(-1, -2)  # (h, T, S)

    # Causal mask: query row i (absolute position q_start+i) may only see
    # source columns s <= q_start+i.
    s_positions = torch.arange(S, device=Q.device)
    mask = s_positions.view(1, -1) <= q_positions.view(-1, 1)  # (T, S) bool
    mask = mask.view(1, T, S)  # broadcast over heads

    # Masking both inputs first makes the difference equal the masked
    # difference elementwise, so the module's shared per-head reduction
    # applies unchanged (identical arithmetic, incl. the 1e-12 clamp).
    approx_m = torch.where(mask, logits_approx, torch.zeros_like(logits_approx))
    ref_m = torch.where(mask, logits_ref, torch.zeros_like(logits_ref))
    return _frobenius_rel_error(approx_m, ref_m).mean().item()
