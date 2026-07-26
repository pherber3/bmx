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
import torch.nn.functional as F

from bmx.cache.codecs import dequant_packed
from bmx.cache.collect import from_matrix
from bmx.cache.rope import apply_rope
from bmx.cache.spectral import SpectralPack, spectral_dequant_packed
from bmx.cache.triton_dequant_attention import block_k_res, block_v_indices


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


def _dequant_block(
    packed,
    arm,
    group,
    seed,
    h_kv,
    pack: SpectralPack | None = None,
    k_tier_cols: dict[int, torch.Tensor] | None = None,
    k_quantizer: str = "rtn",
):
    """packed dict -> (h_kv, blk, d) dense, matching to_matrix layout.

    W5-2 pack_v interaction: a k2b V block dict that has been re-pointed under
    pack_v=True (_repoint_k2b_blocks) no longer holds "indices" (int16) --
    "indices" was DELETED and replaced by "indices_packed" (a view into the
    packed uint8 stack buffer), freeing the int16 block-list copy. dequant_packed
    (codecs.py) is arm-generic and always reads "indices" by key, so route
    through block_v_indices here (the one place that knows both
    representations) to transiently materialize int16 indices before handing
    the dict to dequant_packed -- codecs.py itself is untouched. This container
    discipline applies to both "turboquant_mse" and "turboquant_mse_perhead" --
    the spectral K write path (_pack_v_block) sub-byte-packs V's "indices"
    whenever K is spectral, regardless of which turboquant_mse* arm V uses.
    Transient only -- never stored back onto packed (the block_v_indices contract).

    W5-3 pack_k interaction: same pattern, K side. A lowrank_rtn_channel K block
    dict re-pointed under pack_k=True (_repoint_k2b_blocks) no longer holds
    "res_Q_int" (int8) -- it was DELETED and replaced by "res_Q_int_packed" (a
    view into the packed uint8 nibble stack buffer) plus "res_bits" (the K
    residual bit-width, stashed at repoint time since -- unlike V's turboquant
    dict -- the lowrank_rtn_channel dict has no native "bits" key to read).
    Route through block_k_res to transiently materialize int8 codes.

    arm == "spectral": dequant via spectral_dequant_packed (Task 1), the
    corpus-fit-pack inverse of PackedStreamingLayer._pack_k_block's
    spectral_quantize_packed. Returns fp32 (spectral_dequant_packed's native
    output dtype) -- RoPE-at-read (chunked_dequant_attention's decode-loop
    order branch) needs fp32 K for committed-block bitwise parity with
    streaming's fp32-RoPE-then-fp16-cast order; callers that need fp16 cast
    explicitly after RoPE, not here.

    k_tier_cols: precomputed tier_columns(pack.bits) (PackedStreamingLayer._tier_cols),
    threaded through so the cached read path never recomputes it -- tier_columns'
    `.tolist()` is a CUDA->host sync plus up to 7 `.nonzero()` allocations, and this
    fires per committed block, per decode step, per layer. None (the default) keeps
    the standalone-caller behavior: spectral_dequant_packed recomputes it itself.

    k_quantizer: the K spec's `payload_quant` (K4 Lloyd-gate design,
    2026-07-25) -- must match the `quantizer` the block was packed with
    (`PackedStreamingLayer._pack_k_block`'s `spectral_quantize_packed` call),
    since the packed container's signedness convention differs by quantizer
    (see `spectral._pack_tier_codes`). Default "rtn" reproduces every call
    before this parameter existed bit-exactly.
    """
    if arm == "spectral":
        assert pack is not None, (
            "_dequant_block(arm='spectral', ...) requires pack= (the corpus-fit "
            "SpectralPack); it is threaded from PackedStreamingLayer._pack via "
            "chunked_dequant_attention's k_pack kwarg."
        )
        M = spectral_dequant_packed(
            packed, pack, cols_by_tier=k_tier_cols, quantizer=k_quantizer
        )
        return from_matrix(M, h_kv)
    if (
        arm in ("turboquant_mse", "turboquant_mse_perhead")
        and "indices" not in packed
        and "indices_packed" in packed
    ):
        vbits = packed["bits"]
        per_byte = 8 // vbits
        C = packed["indices_packed"].shape[-1] * per_byte
        packed = {
            **packed,
            "indices": block_v_indices(packed, vbits, C),
            "norms": packed["norms"].float(),
        }
    if (
        arm == "lowrank_rtn_channel"
        and "res_Q_int" not in packed
        and "res_Q_int_packed" in packed
    ):
        k_bits = packed["res_bits"]
        blk = packed["res_Q_int_packed"].shape[-1] * 2
        packed = {
            **packed,
            "res_Q_int": block_k_res(packed, k_bits, blk),
        }
    M = (
        packed["fp16"]
        if arm == "fp16"
        else dequant_packed(arm, packed, group=group, seed=seed)
    )
    return from_matrix(M, h_kv)


def _dense_kv(
    blocks,
    arm,
    group,
    seed,
    h_kv,
    k_pre_rope,
    rope_cos,
    rope_sin,
    pack=None,
    k_tier_cols=None,
    k_quantizer="rtn",
):
    """Dequant all blocks to one dense (h_kv, S_committed, d), RoPE-at-read for K.

    arm == "spectral" dequants fp32 (via pack) and RoPE tables are already fp32
    (chunked_dequant_attention's is_prefill path passes fp32 tables for a
    spectral K arm — see _shared_rope's dtype-in-key design), so
    `rope_cos[start:end].to(B.dtype)` is a no-op cast there — fp32-RoPE-then-
    later-fp16-cast (by _assemble_dense_kv's final `.to(dtype)`), matching
    streaming's exact op order without a branch.

    k_tier_cols: forwarded to _dequant_block (see its docstring) — the
    precomputed tier_columns(pack.bits), avoiding a per-block recompute.
    k_quantizer: forwarded to _dequant_block (K4 Lloyd-gate design) — the K
    spec's payload_quant; default "rtn" is inert.
    """
    parts = []
    for packed, start, end in blocks:
        B = _dequant_block(
            packed,
            arm,
            group,
            seed,
            h_kv,
            pack=pack,
            k_tier_cols=k_tier_cols,
            k_quantizer=k_quantizer,
        )
        if k_pre_rope:
            B = apply_rope(
                B,
                rope_cos[start:end].to(B.dtype),
                rope_sin[start:end].to(B.dtype),
            )
        parts.append(B)
    return torch.cat(parts, dim=1) if parts else None


def _assemble_dense_kv(
    k_blocks,
    v_blocks,
    *,
    k_arm,
    v_arm,
    group,
    seed,
    v_group,
    v_seed,
    h_kv,
    k_pre_rope,
    rope_cos,
    rope_sin,
    k_tail,
    v_tail,
    dtype,
    k_pack=None,
    k_tier_cols=None,
    k_quantizer="rtn",
):
    """Dequant all blocks + fold the fp16 tail -> dense (h_kv, S, d) K and V in `dtype`.

    Shared by the oracle and the prefill-SDPA path. Casting is elementwise, so
    cast-then-cat vs cat-then-cast are value-identical; this helper standardizes
    on casting each piece before concatenation.

    k_tier_cols: forwarded to _dense_kv's K side only (V never uses the
    spectral arm) — the precomputed tier_columns(pack.bits).
    k_quantizer: forwarded to _dense_kv's K side only (K4 Lloyd-gate design) —
    the K spec's payload_quant; default "rtn" is inert.
    """
    K = _dense_kv(
        k_blocks,
        k_arm,
        group,
        seed,
        h_kv,
        k_pre_rope,
        rope_cos,
        rope_sin,
        k_pack,
        k_tier_cols,
        k_quantizer,
    )
    V = _dense_kv(v_blocks, v_arm, v_group, v_seed, h_kv, False, None, None)
    if k_tail is not None and k_tail.shape[1] > 0:
        kt = k_tail.to(dtype)
        vt = v_tail.to(dtype)
        K = kt if K is None else torch.cat([K.to(dtype), kt], dim=1)
        V = vt if V is None else torch.cat([V.to(dtype), vt], dim=1)
    else:
        K = K.to(dtype)
        V = V.to(dtype)
    return K, V


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
    v_group: int | None = None,
    v_seed: int | None = None,
    k_pack: "SpectralPack | None" = None,
    k_tier_cols: dict[int, torch.Tensor] | None = None,
    k_quantizer: str = "rtn",
):
    """ORACLE: dequant everything, single full softmax, GQA-expand. No chunking.

    Same call shape as chunked_dequant_attention so they are drop-in comparable.

    v_group / v_seed: allow K and V to use different quantization params
    (k2b oracle tests use K=lowrank_rtn_channel and V=turboquant_mse with
    different seeds). Default to group / seed.
    k_pack: the corpus-fit SpectralPack, required when k_arm == "spectral".
    k_tier_cols: precomputed tier_columns(k_pack.bits); forwarded to
    _dequant_block (see its docstring). None recomputes (the oracle is not the
    hot path — this default keeps standalone/test callers unchanged).
    k_quantizer: the K spec's payload_quant (K4 Lloyd-gate design); default
    "rtn" is inert.
    """
    _v_group = v_group if v_group is not None else group
    _v_seed = v_seed if v_seed is not None else seed
    n_q_heads = q.shape[0]
    h_kv = n_q_heads // n_q_groups
    K, V = _assemble_dense_kv(
        k_blocks,
        v_blocks,
        k_arm=k_arm,
        v_arm=v_arm,
        group=group,
        seed=seed,
        v_group=_v_group,
        v_seed=_v_seed,
        h_kv=h_kv,
        k_pre_rope=k_pre_rope,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        k_tail=k_tail,
        v_tail=v_tail,
        dtype=q.dtype,
        k_pack=k_pack,
        k_tier_cols=k_tier_cols,
        k_quantizer=k_quantizer,
    )
    Kx = K.repeat_interleave(n_q_groups, dim=0)
    Vx = V.repeat_interleave(n_q_groups, dim=0)
    scores = (q @ Kx.transpose(-1, -2)) * scale
    return torch.softmax(scores, dim=-1) @ Vx


def _prefill_dense_attention(
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
    v_group,
    v_seed,
    attn_mask=None,
    k_pack: "SpectralPack | None" = None,
    k_tier_cols: dict[int, torch.Tensor] | None = None,
    k_quantizer: str = "rtn",
):
    """Prefill (n_q > 1) attention: reconstruct dense K/V once, run flash SDPA.

    The per-block online-softmax in chunked_dequant_attention is O(S^2) memory at
    prefill, because each block produces an (heads, n_q=S, blk) score tile and the
    tiles sum to (heads, S, S). At prefill the right tool is flash SDPA, which
    tiles over the query dim internally in O(S) memory and never materializes the
    (S, S) score matrix. We dequant all committed blocks + the fp16 tail into one
    dense K/V (a transient that frees after this one-shot forward), GQA-expand, and
    call F.scaled_dot_product_attention(is_causal=True). This matches what
    transformers' own QuantizedCache does (dequant-to-dense + SDPA). The DECODE
    path (n_q == 1) keeps the chunked online-softmax — that is the resident-memory
    win and is O(S) there (tiny per-block tiles).

    k_tier_cols: precomputed tier_columns(k_pack.bits); forwarded to
    _dequant_block via _assemble_dense_kv (see _dequant_block's docstring).
    k_quantizer: the K spec's payload_quant (K4 Lloyd-gate design); default
    "rtn" is inert.
    """
    K, V = _assemble_dense_kv(
        k_blocks,
        v_blocks,
        k_arm=k_arm,
        v_arm=v_arm,
        group=group,
        seed=seed,
        v_group=v_group,
        v_seed=v_seed,
        h_kv=q.shape[0] // n_q_groups,
        k_pre_rope=k_pre_rope,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        k_tail=k_tail,
        v_tail=v_tail,
        dtype=q.dtype,
        k_pack=k_pack,
        k_tier_cols=k_tier_cols,
        k_quantizer=k_quantizer,
    )
    Kx = K.repeat_interleave(n_q_groups, dim=0)  # (n_q_heads, S, d)
    Vx = V.repeat_interleave(n_q_groups, dim=0)
    # attn_mask (not is_causal) governs masking when provided — see the
    # AttentionMaskInterface registration in packed_streaming.py and
    # docs/2026-06-23-kernel-census-results.md.
    # attn_mask is 4D (b,1,q,kv); add the batch dim to q/K/V so the shapes line up.
    out = F.scaled_dot_product_attention(
        q.unsqueeze(0),
        Kx.unsqueeze(0),
        Vx.unsqueeze(0),
        attn_mask=attn_mask,
        is_causal=(attn_mask is None),
        scale=scale,
    )
    return out.squeeze(0)  # (n_q_heads, n_q, d)


def chunked_dequant_attention(
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
    is_prefill: bool = False,
    v_group: int | None = None,
    v_seed: int | None = None,
    attn_mask=None,
    k_pack: "SpectralPack | None" = None,
    k_tier_cols: dict[int, torch.Tensor] | None = None,
    k_quantizer: str = "rtn",
):
    """Online-softmax attention over per-block dequantized K/V. GQA-aware.

    q: (n_q_heads, n_q, d). k_blocks/v_blocks: list of (packed, start, end).
    k_pre_rope: if True, dequantized K blocks are pre-RoPE and get RoPE applied at
    [start,end) before the contraction. k_tail/v_tail: (h_kv, tail_len, d) fp16
    recent window (post-RoPE for K). Returns (n_q_heads, n_q, d).

    is_prefill: True during prefill (n_q > 1) — delegates to the dense flash-SDPA
    path (the model's attn_mask governs causality). False during decode (n_q == 1)
    — the online-softmax loop runs, no masking needed.
    attn_mask: the model's 4D causal mask, forwarded to the prefill SDPA path.
    v_group / v_seed: dequant params for V blocks; default to group / seed when
    not provided (allows K and V to use different packed formats).
    k_pack: the corpus-fit SpectralPack, required when k_arm == "spectral" (both
    the decode online-softmax loop and the prefill dense path need it).
    k_tier_cols: precomputed tier_columns(k_pack.bits) (PackedStreamingLayer._tier_cols),
    threaded to both the decode loop and the prefill path so spectral_dequant_packed
    never recomputes it (see _dequant_block's docstring — this is the hot-path fix:
    tier_columns' .tolist() is a per-block CUDA->host sync + up to 7 .nonzero() calls).
    None (the default) preserves standalone-caller behavior exactly.
    k_quantizer: the K spec's payload_quant (K4 Lloyd-gate design, 2026-07-25) —
    must match the quantizer the committed blocks were packed with (see
    _dequant_block's docstring). Default "rtn" is inert.
    """
    n_q_heads, n_q, d = q.shape
    h_kv = n_q_heads // n_q_groups
    _v_group = v_group if v_group is not None else group
    _v_seed = v_seed if v_seed is not None else seed

    # Prefill (n_q > 1) delegates to the dense + flash-SDPA path: the per-block
    # online-softmax below is O(S^2) memory at prefill (each block's score tile is
    # (heads, n_q=S, blk)), whereas SDPA tiles internally in O(S).
    if is_prefill:
        return _prefill_dense_attention(
            q,
            k_blocks,
            v_blocks,
            k_arm=k_arm,
            v_arm=v_arm,
            group=group,
            seed=seed,
            k_pre_rope=k_pre_rope,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            k_tail=k_tail,
            v_tail=v_tail,
            n_q_groups=n_q_groups,
            scale=scale,
            v_group=_v_group,
            v_seed=_v_seed,
            attn_mask=attn_mask,
            k_pack=k_pack,
            k_tier_cols=k_tier_cols,
            k_quantizer=k_quantizer,
        )

    # Decode (n_q == 1): the single query at the last position attends ALL cached keys
    # — no causal masking needed. Online-softmax over per-block dequant keeps peak
    # memory at the packed footprint (one tiny block dequantized at a time).
    acc = torch.zeros(n_q_heads, n_q, d, dtype=q.dtype, device=q.device)
    m = torch.full((n_q_heads, n_q, 1), float("-inf"), dtype=q.dtype, device=q.device)
    lse = torch.zeros(n_q_heads, n_q, 1, dtype=q.dtype, device=q.device)

    def attend(K_kv, V_kv):
        nonlocal acc, m, lse
        # Grouped contraction: avoid repeat_interleave materializing an
        # (n_q_heads, blk, d) copy of K/V each block. q viewed as
        # (h_kv, n_q_groups, n_q, d) contracts against (h_kv, blk, d).
        qg = q.view(h_kv, n_q_groups, n_q, d)
        s = torch.einsum("gpnd,gbd->gpnb", qg, K_kv) * scale  # (h_kv, grp, n_q, blk)
        s = s.reshape(n_q_heads, n_q, K_kv.shape[1])
        m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
        correction = torch.exp(m - m_new)
        p = torch.exp(s - m_new)
        lse = lse * correction + p.sum(dim=-1, keepdim=True)
        pg = p.view(h_kv, n_q_groups, n_q, K_kv.shape[1])
        av = torch.einsum("gpnb,gbd->gpnd", pg, V_kv).reshape(n_q_heads, n_q, d)
        acc = acc * correction + av
        m = m_new

    for (kpacked, start, end), (vpacked, _vs, _ve) in zip(k_blocks, v_blocks):
        # Spectral K: fp32-RoPE-then-fp16-cast, matching streaming's exact op
        # order (streaming.py:324-339 — fp32 tables, RoPE in fp32, THEN cast) —
        # required for committed-block BITWISE parity. k2b/other arms keep
        # today's cast-then-rope order verbatim (their parity tests pin it).
        if k_arm == "spectral":
            K_kv = _dequant_block(
                kpacked,
                k_arm,
                group,
                seed,
                h_kv,
                pack=k_pack,
                k_tier_cols=k_tier_cols,
                k_quantizer=k_quantizer,
            )
            K_kv = apply_rope(K_kv, rope_cos[start:end], rope_sin[start:end])
            K_kv = K_kv.to(q.dtype)
        else:
            K_kv = _dequant_block(kpacked, k_arm, group, seed, h_kv).to(q.dtype)
            if k_pre_rope:
                K_kv = apply_rope(K_kv, rope_cos[start:end], rope_sin[start:end])
        V_kv = _dequant_block(vpacked, v_arm, _v_group, _v_seed, h_kv).to(q.dtype)
        attend(K_kv, V_kv)

    if k_tail is not None and k_tail.shape[1] > 0:
        attend(k_tail.to(q.dtype), v_tail.to(q.dtype))

    return acc / lse
