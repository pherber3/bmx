"""Live-generation perplexity through the StreamingQuantizedCache.

Unlike ppl_eval (which quantizes the whole prefill at once), this prefills N
tokens INTO the streaming cache, then teacher-forces the continuation so each
step attends to the on-append compressed cache. The end-to-end 'in practice'
metric for the K3 verdict.
"""

from __future__ import annotations

import torch

from bmx.cache.specs import CacheCodecSpec
from bmx.cache.streaming import StreamingQuantizedCache


def live_generation_ppl(
    model,
    input_ids: torch.Tensor,
    n_prefill: int,
    k_spec: CacheCodecSpec,
    v_spec: CacheCodecSpec,
    recent_window: int = 32,
) -> dict:
    """Prefill into a streaming cache, teacher-force the continuation, return ppl.

    Parameters
    ----------
    model :
        HuggingFace CausalLM model (eval mode recommended; not mutated).
    input_ids : torch.Tensor
        Shape (1, N+M).
    n_prefill : int
        Number of prefill tokens N. Quantization happens on-append during prefill.
    k_spec : CacheCodecSpec
        Codec spec for keys.
    v_spec : CacheCodecSpec
        Codec spec for values.
    recent_window : int
        Most-recent tokens kept fp16 before flushing (passed to StreamingQuantizedCache).

    Returns
    -------
    dict with keys:
        ``ppl``    — float, perplexity over the M-1 continuation tokens.
        ``bpe_k``  — float, honest bits-per-entry for keys.
        ``bpe_v``  — float, honest bits-per-entry for values.
        ``n_eval`` — int, number of tokens contributing to the loss (M-1).

    Notes
    -----
    # memory_report fields (packed_bytes/fp16_bytes/compression) wired in Task 5.
    """
    assert input_ids.shape[0] == 1, "batch dim must be 1"
    N = input_ids.shape[1]
    assert n_prefill < N, "n_prefill must be < total length"

    cache = StreamingQuantizedCache(
        model.config,
        k_spec=k_spec,
        v_spec=v_spec,
        recent_window=recent_window,
    )
    cache.attach(model)  # pre-RoPE capture; no-op when k_spec.pre_rope is False
    try:
        with torch.no_grad():
            model(input_ids[:, :n_prefill], past_key_values=cache, use_cache=True)

        cont_ids = input_ids[:, n_prefill:]
        n_eval = cont_ids.shape[1] - 1
        with torch.no_grad():
            out = model(cont_ids, past_key_values=cache, labels=cont_ids)
    finally:
        cache.detach()

    bpe_k, bpe_v = cache.bits_per_entry()
    return {
        "ppl": torch.exp(out.loss).item(),
        "bpe_k": bpe_k,
        "bpe_v": bpe_v,
        "n_eval": n_eval,
    }
