"""Quantized-prefill perplexity evaluation for KV-cache codecs.

Public API
----------
CacheCodecSpec : dataclass
    Codec specification for one side (K or V) of the KV cache.

quantized_prefill_ppl(model, input_ids, n_prefill, k_spec, v_spec) -> dict
    Prefill N tokens, quantize the full prefill cache, write it back, then
    teacher-force the next M tokens and return their NLL perplexity.

Notes
-----
Cache surgery: DynamicCache in transformers 5.x exposes ``cache.layers[i].keys``
and ``.values`` as mutable attributes (shape (1, h_kv, S, d)). Direct assignment
works on transformers 5.11.0 — no update() API fallback required.

Continuation forward label shift: ``model(ids[:, n_prefill:], labels=ids[:, n_prefill:])``
uses transformers' internal shift which yields loss over tokens n_prefill+1..N-1
(n_cont-1 tokens). ``n_eval`` in the returned dict reflects this.

K1 matrix convention: (h_kv, S, d) -> permute(1, 0, 2).reshape(S, h_kv * d)
and inverse: reshape(S, h_kv, d).permute(1, 0, 2).
"""

from __future__ import annotations

import dataclasses

import torch

from bmx.cache.codecs import CACHE_ARMS, quantize_cache
from bmx.cache.collect import _register_hooks
from bmx.cache.rope import apply_rope, rope_cos_sin


@dataclasses.dataclass
class CacheCodecSpec:
    """Codec specification for one side (K or V) of the KV cache.

    Attributes
    ----------
    arm : str
        A member of bmx.cache.codecs.CACHE_ARMS, or ``"fp16"`` for a no-op.
    bits : int
        Quantization bit width.
    rank : int
        Low-rank components for ``lowrank_rtn_channel`` (ignored otherwise).
    group : int
        Group size for rtn_token / rtn_channel / rotate_rtn_token / lowrank arms.
    seed : int
        RNG seed for rotation/sketch arms.
    pre_rope : bool
        If True, capture k_pre via hooks, quantize in pre-RoPE space, then
        apply_rope before writing back into the cache. Ignored for V (V has
        no RoPE in standard transformer families).
    """

    arm: str = "fp16"
    bits: int = 3
    rank: int = 0
    group: int = 64
    seed: int = 0
    pre_rope: bool = False


def _k1_to_matrix(kv: torch.Tensor) -> torch.Tensor:
    """(h, S, d) -> (S, h*d) fp32 matrix for quantize_cache."""
    h, S, d = kv.shape
    return kv.permute(1, 0, 2).reshape(S, h * d).float()


def _matrix_to_k1(M: torch.Tensor, h: int, d: int) -> torch.Tensor:
    """(S, h*d) -> (h, S, d)."""
    S = M.shape[0]
    return M.reshape(S, h, d).permute(1, 0, 2)


def _quantize_kv(
    kv_fp: torch.Tensor,
    spec: CacheCodecSpec,
) -> tuple[torch.Tensor, float]:
    """Quantize (h, S, d) tensor; return (h, S, d) fp32 result and bpe.

    ``kv_fp`` is expected to be fp32.  For ``arm="fp16"``, returns the input
    unchanged and ``bpe=16.0``.
    """
    h, S, d = kv_fp.shape
    if spec.arm == "fp16":
        return kv_fp, 16.0

    assert spec.arm in CACHE_ARMS, (
        f"unknown arm {spec.arm!r}; use one of {CACHE_ARMS} or 'fp16'"
    )
    M = _k1_to_matrix(kv_fp)  # (S, h*d)
    M_hat, bpe = quantize_cache(
        spec.arm,
        M,
        bits=spec.bits,
        seed=spec.seed,
        group=spec.group,
        rank=spec.rank,
    )
    kv_hat = _matrix_to_k1(M_hat, h, d)
    return kv_hat, bpe


def quantized_prefill_ppl(
    model,
    input_ids: torch.Tensor,
    n_prefill: int,
    k_spec: CacheCodecSpec,
    v_spec: CacheCodecSpec,
) -> dict:
    """Prefill N tokens, quantize the KV cache, evaluate M-token continuation ppl.

    Parameters
    ----------
    model :
        HuggingFace CausalLM model (eval mode recommended; not mutated).
    input_ids : torch.Tensor
        Shape (1, N+M).
    n_prefill : int
        Number of prefill tokens N.
    k_spec : CacheCodecSpec
        Codec spec for keys.
    v_spec : CacheCodecSpec
        Codec spec for values.

    Returns
    -------
    dict with keys:
        ``ppl``    — float, perplexity over the M-1 continuation tokens
                     (transformers' internal label shift loses the first token).
        ``bpe_k``  — float, honest bits-per-entry for keys.
        ``bpe_v``  — float, honest bits-per-entry for values.
        ``n_eval`` — int, number of tokens contributing to the loss (M-1).
    """
    assert input_ids.shape[0] == 1, "batch dim must be 1"
    N = input_ids.shape[1]
    assert n_prefill < N, "n_prefill must be < total sequence length"

    # ------------------------------------------------------------------
    # Step 1: prefill, optionally capturing k_pre via hooks
    # ------------------------------------------------------------------
    k_pre_store: dict[str, torch.Tensor] = {}
    handles: list = []

    if k_spec.pre_rope:
        handles, _ = _register_hooks(model, k_pre_store, n_q_keep=1)

    try:
        with torch.no_grad():
            prefill_out = model(input_ids[:, :n_prefill], use_cache=True)
    finally:
        for h in handles:
            h.remove()

    cache = prefill_out.past_key_values
    n_layer = len(cache.layers)

    # ------------------------------------------------------------------
    # Step 2: quantize and write back into cache
    # ------------------------------------------------------------------
    bpe_k_list: list[float] = []
    bpe_v_list: list[float] = []

    for i in range(n_layer):
        layer = cache.layers[i]
        # shapes: (1, h_kv, S, d)
        keys_orig = layer.keys  # (1, h_kv, S, d)
        vals_orig = layer.values  # (1, h_kv, S, d)

        cache_dtype = keys_orig.dtype
        S = keys_orig.shape[2]

        # --- Key quantization ---
        if k_spec.pre_rope:
            # Use captured k_pre (fp16, shape (h_kv, S, d))
            k_pre_fp16 = k_pre_store[f"layer{i}.k_pre"]  # (h_kv, S, d)
            k_pre_fp32 = k_pre_fp16.float()
            k_hat_fp32, bpe_k = _quantize_kv(k_pre_fp32, k_spec)
            # Apply RoPE to get post-RoPE quantized keys
            cos, sin = rope_cos_sin(model.config, S)
            cos = cos.to(k_hat_fp32.dtype)
            sin = sin.to(k_hat_fp32.dtype)
            k_hat_fp32 = apply_rope(k_hat_fp32, cos, sin)
        else:
            k_fp32 = keys_orig.squeeze(0).float()  # (h_kv, S, d)
            k_hat_fp32, bpe_k = _quantize_kv(k_fp32, k_spec)

        # --- Value quantization ---
        v_fp32 = vals_orig.squeeze(0).float()  # (h_kv, S, d)
        v_hat_fp32, bpe_v = _quantize_kv(v_fp32, v_spec)

        # --- Write back (cast to original cache dtype, re-add batch dim) ---
        layer.keys = k_hat_fp32.to(cache_dtype).unsqueeze(0)  # (1, h_kv, S, d)
        layer.values = v_hat_fp32.to(cache_dtype).unsqueeze(0)  # (1, h_kv, S, d)

        bpe_k_list.append(bpe_k)
        bpe_v_list.append(bpe_v)

    # Average bpe across layers (all layers use same spec so they're identical,
    # but average is honest when shapes vary, e.g. lowrank bpe depends on S, C).
    bpe_k_avg = sum(bpe_k_list) / len(bpe_k_list)
    bpe_v_avg = sum(bpe_v_list) / len(bpe_v_list)

    # ------------------------------------------------------------------
    # Step 3: teacher-forced continuation forward
    # ------------------------------------------------------------------
    cont_ids = input_ids[:, n_prefill:]  # (1, M)
    n_eval = cont_ids.shape[1] - 1  # label shift loses first token

    with torch.no_grad():
        out = model(cont_ids, past_key_values=cache, labels=cont_ids)

    ppl = torch.exp(out.loss).item()

    return {
        "ppl": ppl,
        "bpe_k": bpe_k_avg,
        "bpe_v": bpe_v_avg,
        "n_eval": n_eval,
    }
