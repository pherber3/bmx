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
from bmx.cache.collect import register_hooks
from bmx.cache.rope import apply_rope, rope_cos_sin
from bmx.cache.specs import CacheCodecSpec


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
        handles, _ = register_hooks(model, k_pre_store, n_q_keep=1)

    try:
        with torch.no_grad():
            prefill_out = model(input_ids[:, :n_prefill], use_cache=True)
    finally:
        for h in handles:
            h.remove()

    return PrefillState(cache=prefill_out.past_key_values, k_pre=k_pre_store)


def quantized_prefill_ppl(
    model,
    input_ids: torch.Tensor,
    n_prefill: int,
    k_spec: CacheCodecSpec,
    v_spec: CacheCodecSpec,
    state: PrefillState | None = None,
    k_specs: list[CacheCodecSpec] | None = None,
    v_specs: list[CacheCodecSpec] | None = None,
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
        Codec spec for keys. Used for every layer unless ``k_specs`` is given.
    v_spec : CacheCodecSpec
        Codec spec for values. Used for every layer unless ``v_specs`` is given.
    state : PrefillState | None
        Optional pre-computed prefill (see run_prefill).  When provided, the
        cache is deepcopied before surgery so the state stays reusable across
        arms; the stored k_pre tensors are read-only and shared.  When None,
        run_prefill is called internally (behavior identical to a fresh call).
        A state passed here must have been built from the same
        (model, input_ids[:, :n_prefill]) — and with capture_pre_rope=True if
        ``k_spec.pre_rope``.
    k_specs : list[CacheCodecSpec] | None
        Optional per-layer override for keys; ``k_specs[i]`` is used for layer
        i instead of ``k_spec``.  Length must equal the number of cache layers.
        All specs (including ``k_spec``) must share the same ``pre_rope`` value
        — one RoPE regime per run.
    v_specs : list[CacheCodecSpec] | None
        Optional per-layer override for values; ``v_specs[i]`` is used for
        layer i instead of ``v_spec``.  Length must equal the number of cache
        layers.  None of the specs may set ``pre_rope`` (V has no RoPE).

    Returns
    -------
    dict with keys:
        ``ppl``    — float, perplexity over the M-1 continuation tokens
                     (transformers' internal label shift loses the first token).
        ``bpe_k``  — float, honest bits-per-entry for keys.  When ``k_specs``
                     is given, the mean over layers; otherwise the (layer-
                     invariant) single-spec value.
        ``bpe_v``  — float, honest bits-per-entry for values.  Same mean
                     convention as ``bpe_k`` when ``v_specs`` is given.
        ``n_eval`` — int, number of tokens contributing to the loss (M-1).
    """
    assert input_ids.shape[0] == 1, "batch dim must be 1"
    N = input_ids.shape[1]
    assert n_prefill < N, "n_prefill must be < total sequence length"
    assert not v_spec.pre_rope, "pre_rope has no effect on V; set it on k_spec"
    if k_specs is not None:
        assert all(s.pre_rope == k_spec.pre_rope for s in k_specs), (
            "mixed pre_rope across k_specs is rejected; one RoPE regime per run"
        )
    if v_specs is not None:
        assert not any(s.pre_rope for s in v_specs), (
            "pre_rope has no effect on V; set it on k_spec"
        )

    if state is None:
        state = run_prefill(model, input_ids, n_prefill, k_spec.pre_rope)
        cache = state.cache  # freshly built; safe to mutate in place
    else:
        cache = copy.deepcopy(state.cache)  # surgery mutates; keep state reusable

    k_pre_store = state.k_pre
    n_layer = len(cache.layers)

    if k_specs is not None:
        assert len(k_specs) == n_layer, (
            f"k_specs length {len(k_specs)} must equal n_layer {n_layer}"
        )
    if v_specs is not None:
        assert len(v_specs) == n_layer, (
            f"v_specs length {len(v_specs)} must equal n_layer {n_layer}"
        )

    # RoPE tables: spec-level, identical for every layer — compute once.
    if k_spec.pre_rope:
        assert k_pre_store, (
            "k_spec.pre_rope=True but state has no k_pre; "
            "build the state with capture_pre_rope=True"
        )
        S = cache.layers[0].keys.shape[2]
        cos, sin = rope_cos_sin(model.config, S)
        cos = cos.float()  # quantize_kv_layout outputs are fp32
        sin = sin.float()

    # bpe is spec-determined and identical across layers when a single spec is
    # used (all layers share (S, C)), so a plain per-layer overwrite suffices.
    # When per-layer spec lists are given, bpe can vary by layer, so accumulate
    # a running mean instead.
    bpe_k = bpe_v = float("nan")
    bpe_k_sum = bpe_v_sum = 0.0

    for i in range(n_layer):
        layer = cache.layers[i]
        # shapes: (1, h_kv, S, d)
        keys_orig = layer.keys  # (1, h_kv, S, d)
        vals_orig = layer.values  # (1, h_kv, S, d)

        cache_dtype = keys_orig.dtype

        k_spec_i = k_specs[i] if k_specs is not None else k_spec
        v_spec_i = v_specs[i] if v_specs is not None else v_spec

        # --- Key quantization ---
        if k_spec_i.pre_rope:
            # Use captured k_pre (fp16, shape (h_kv, S, d))
            k_pre_fp16 = k_pre_store[f"layer{i}.k_pre"]  # (h_kv, S, d)
            k_pre_fp32 = k_pre_fp16.float()
            k_hat_fp32, bpe_k = quantize_kv_layout(k_pre_fp32, k_spec_i)
            # Apply RoPE to get post-RoPE quantized keys
            k_hat_fp32 = apply_rope(k_hat_fp32, cos, sin)
        else:
            k_fp32 = keys_orig.squeeze(0).float()  # (h_kv, S, d)
            k_hat_fp32, bpe_k = quantize_kv_layout(k_fp32, k_spec_i)

        # --- Value quantization ---
        v_fp32 = vals_orig.squeeze(0).float()  # (h_kv, S, d)
        v_hat_fp32, bpe_v = quantize_kv_layout(v_fp32, v_spec_i)

        bpe_k_sum += bpe_k
        bpe_v_sum += bpe_v

        # --- Write back (cast to original cache dtype, re-add batch dim) ---
        layer.keys = k_hat_fp32.to(cache_dtype).unsqueeze(0)  # (1, h_kv, S, d)
        layer.values = v_hat_fp32.to(cache_dtype).unsqueeze(0)  # (1, h_kv, S, d)

    if k_specs is not None:
        bpe_k = bpe_k_sum / n_layer
    if v_specs is not None:
        bpe_v = bpe_v_sum / n_layer

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
