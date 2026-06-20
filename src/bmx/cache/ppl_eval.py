"""Quantized-prefill perplexity evaluation for KV-cache codecs.

Public API
----------
CacheCodecSpec : dataclass
    Codec specification for one side (K or V) of the KV cache.

PrefillState : dataclass
    Reusable prefill artifacts: the DynamicCache plus the hooked k_pre store.

run_prefill(model, input_ids, n_prefill, capture_pre_rope) -> PrefillState
    The hooked prefill forward, factored out so one prefill can serve many
    codec arms (quantized_prefill_ppl deepcopies the cache before surgery).

quantized_prefill_ppl(model, input_ids, n_prefill, k_spec, v_spec, state=None) -> dict
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

import copy
import dataclasses

import torch

from bmx.cache.codecs import quantize_kv_layout
from bmx.cache.collect import _register_hooks
from bmx.cache.rope import apply_rope, rope_cos_sin
from bmx.cache.specs import CacheCodecSpec  # re-export; was defined here


@dataclasses.dataclass
class PrefillState:
    """Reusable prefill artifacts for quantized_prefill_ppl.

    Attributes
    ----------
    cache :
        The DynamicCache from the prefill forward.  quantized_prefill_ppl
        deepcopies it before surgery, so one state can serve many arms.
    k_pre :
        ``layer{i}.k_pre`` tensors captured via hooks (empty dict when the
        state was built with ``capture_pre_rope=False``).  Read-only.
    """

    cache: object
    k_pre: dict[str, torch.Tensor]


def run_prefill(
    model,
    input_ids: torch.Tensor,
    n_prefill: int,
    capture_pre_rope: bool,
) -> PrefillState:
    """Hooked prefill forward over the first *n_prefill* tokens.

    Set ``capture_pre_rope=True`` if ANY codec arm that will consume this
    state needs pre-RoPE keys (``k_spec.pre_rope=True``).
    """
    k_pre_store: dict[str, torch.Tensor] = {}
    handles: list = []

    if capture_pre_rope:
        handles, _ = _register_hooks(model, k_pre_store, n_q_keep=1)

    try:
        with torch.no_grad():
            prefill_out = model(input_ids[:, :n_prefill], use_cache=True)
    finally:
        for h in handles:
            h.remove()

    return PrefillState(cache=prefill_out.past_key_values, k_pre=k_pre_store)


def _quantize_kv(
    kv_fp: torch.Tensor,
    spec: CacheCodecSpec,
) -> tuple[torch.Tensor, float]:
    """Quantize (h, S, d) tensor; return (h, S, d) fp32 result and bpe.

    ``kv_fp`` is expected to be fp32.  For ``arm="fp16"``, returns the input
    unchanged and ``bpe=16.0``.
    """
    return quantize_kv_layout(kv_fp, spec)


def quantized_prefill_ppl(
    model,
    input_ids: torch.Tensor,
    n_prefill: int,
    k_spec: CacheCodecSpec,
    v_spec: CacheCodecSpec,
    state: PrefillState | None = None,
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
    state : PrefillState | None
        Optional pre-computed prefill (see run_prefill).  When provided, the
        cache is deepcopied before surgery so the state stays reusable across
        arms; the stored k_pre tensors are read-only and shared.  When None,
        run_prefill is called internally (behavior identical to a fresh call).
        A state passed here must have been built from the same
        (model, input_ids[:, :n_prefill]) — and with capture_pre_rope=True if
        ``k_spec.pre_rope``.

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
    assert not v_spec.pre_rope, "pre_rope has no effect on V; set it on k_spec"

    # ------------------------------------------------------------------
    # Step 1: prefill (or reuse), optionally capturing k_pre via hooks
    # ------------------------------------------------------------------
    if state is None:
        state = run_prefill(model, input_ids, n_prefill, k_spec.pre_rope)
        cache = state.cache  # freshly built; safe to mutate in place
    else:
        cache = copy.deepcopy(state.cache)  # surgery mutates; keep state reusable

    k_pre_store = state.k_pre
    n_layer = len(cache.layers)

    # RoPE tables: spec-level, identical for every layer — compute once.
    if k_spec.pre_rope:
        assert k_pre_store, (
            "k_spec.pre_rope=True but state has no k_pre; "
            "build the state with capture_pre_rope=True"
        )
        S = cache.layers[0].keys.shape[2]
        cos, sin = rope_cos_sin(model.config, S)
        cos = cos.float()  # _quantize_kv outputs are fp32
        sin = sin.float()

    # ------------------------------------------------------------------
    # Step 2: quantize and write back into cache
    # ------------------------------------------------------------------
    # bpe is spec-determined and identical across layers (all layers share
    # (S, C)), so a plain per-layer overwrite suffices.
    bpe_k = bpe_v = float("nan")

    for i in range(n_layer):
        layer = cache.layers[i]
        # shapes: (1, h_kv, S, d)
        keys_orig = layer.keys  # (1, h_kv, S, d)
        vals_orig = layer.values  # (1, h_kv, S, d)

        cache_dtype = keys_orig.dtype

        # --- Key quantization ---
        if k_spec.pre_rope:
            # Use captured k_pre (fp16, shape (h_kv, S, d))
            k_pre_fp16 = k_pre_store[f"layer{i}.k_pre"]  # (h_kv, S, d)
            k_pre_fp32 = k_pre_fp16.float()
            k_hat_fp32, bpe_k = _quantize_kv(k_pre_fp32, k_spec)
            # Apply RoPE to get post-RoPE quantized keys
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
        "bpe_k": bpe_k,
        "bpe_v": bpe_v,
        "n_eval": n_eval,
    }
