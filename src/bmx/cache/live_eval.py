"""Live-generation perplexity through the StreamingQuantizedCache.

Unlike ppl_eval (which quantizes the whole prefill at once), this prefills N
tokens INTO the streaming cache, then teacher-forces the continuation so each
step attends to the on-append compressed cache. The end-to-end 'in practice'
metric for the K3 verdict.

token_by_token mode (Task 11):
    When token_by_token=True, the continuation is scored one token at a time.
    Each decode step appends exactly one token to the cache (triggering
    quantize-on-append), and the logit predicting the NEXT token is scored.
    This is the honest streaming regime: quant errors compound step by step.

    Indexing: after prefill([0:n_prefill]), cache holds [0..n_prefill-1].
    HF causal LM: model(ids[:,a:b], past_key_values=cache) appends tokens
    a..b-1 to the cache; logits[:,-1] predicts token b.
    So to score token i+1, we feed token i (ids[:,i:i+1]); the cache then
    holds [0..i].  Loop: for i in range(n_prefill, N-1) → feed token i,
    score target token i+1.  This yields n_eval = N-1-n_prefill tokens,
    matching the batched path (which uses HF label-shift: cont_ids=[n_prefill:]
    scored over [n_prefill+1..N-1]).

NOTE on tiny-model quality numbers:
    tiny_llama has random weights, so absolute ppl is meaningless (~vocab size).
    The token-by-token gate tests MECHANISM (finite, no explosion, fp16-modes-agree),
    NOT real quality. Real quality numbers come from experiments on real models.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from bmx.cache.specs import CacheCodecSpec
from bmx.cache.streaming import StreamingQuantizedCache, resolve_vocab_size


def compression_for(model, k_spec, v_spec, length: int) -> tuple[float, float, float]:
    """Measured (bpe_k, bpe_v, compression) for an arm at a given sequence length.

    Runs a calibration prefill of `length` tokens first: bits_per_entry is nan until a
    forward pass quantizes a block. Then reads the blended-bpe accounting off the cache.
    """
    cache = StreamingQuantizedCache(model.config, k_spec=k_spec, v_spec=v_spec)
    cache.attach(model)
    g = torch.Generator().manual_seed(0)
    # Generate on CPU (seeded Generator is CPU-only), then move to the model's device.
    vocab = resolve_vocab_size(model.config)
    ids = torch.randint(0, vocab, (1, length), generator=g).to(model.device)
    with cache:
        with torch.no_grad():
            model(ids, past_key_values=cache, use_cache=True)
    bpe_k, bpe_v = cache.bits_per_entry()
    mem = cache.memory_report(seq_len=length)
    return bpe_k, bpe_v, mem["compression"]


def live_generation_ppl(
    model,
    input_ids: torch.Tensor,
    n_prefill: int,
    k_spec: CacheCodecSpec,
    v_spec: CacheCodecSpec,
    recent_window: int = 32,
    token_by_token: bool = False,
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
    token_by_token : bool
        If False (default), teacher-force the full continuation in one batched forward
        (fast; unchanged from before Task 11).
        If True, score the continuation one token at a time so each next-token NLL
        attends to the incrementally-built compressed cache (write-once, errors
        compound).  This is the honest streaming regime for the K3 verdict.

    Returns
    -------
    dict with keys:
        ``ppl``          — float, perplexity over the M-1 continuation tokens.
        ``bpe_k``        — float, honest bits-per-entry for keys.
        ``bpe_v``        — float, honest bits-per-entry for values.
        ``n_eval``       — int, number of tokens contributing to the loss (M-1).
        ``packed_bytes`` — float, honest compressed KV footprint (bpe × entries).
        ``fp16_bytes``   — float, dense fp16 KV baseline footprint.
        ``compression``  — float, fp16_bytes / packed_bytes.
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
    with cache:
        with torch.no_grad():
            # --- Prefill: feed tokens [0..n_prefill-1] ---
            model(input_ids[:, :n_prefill], past_key_values=cache, use_cache=True)

            if not token_by_token:
                # --- Batched continuation (original path) ---
                cont_ids = input_ids[:, n_prefill:]
                n_eval = cont_ids.shape[1] - 1
                out = model(cont_ids, past_key_values=cache, labels=cont_ids)
                ppl = torch.exp(out.loss).item()
            else:
                # --- Token-by-token continuation (honest streaming regime) ---
                #
                # After prefill, cache holds tokens [0..n_prefill-1].
                # To score token i+1: feed token i (ids[:,i:i+1]) → cache appends
                # token i → logits[:,-1] predicts token i+1.
                # Loop: i in [n_prefill, N-1) feeds token i, scores target i+1.
                # This gives n_eval = N-1-n_prefill tokens, matching the batched
                # path's label-shift convention (cont_ids=[n_prefill:N], scored
                # over [n_prefill+1..N-1]).
                total_nll = 0.0
                n_eval = 0
                for i in range(n_prefill, N - 1):
                    step_ids = input_ids[:, i : i + 1]  # token i → appended to cache
                    out = model(step_ids, past_key_values=cache, use_cache=True)
                    logits_next = out.logits[0, -1]  # predicts token i+1
                    target = input_ids[0, i + 1]
                    nll = F.cross_entropy(
                        logits_next.unsqueeze(0), target.unsqueeze(0)
                    ).item()
                    total_nll += nll
                    n_eval += 1
                ppl = math.exp(total_nll / n_eval) if n_eval > 0 else float("nan")

    bpe_k, bpe_v = cache.bits_per_entry()
    mem = cache.memory_report(seq_len=input_ids.shape[1])
    return {
        "ppl": ppl,
        "bpe_k": bpe_k,
        "bpe_v": bpe_v,
        "n_eval": n_eval,
        "packed_bytes": mem["packed_bytes"],
        "fp16_bytes": mem["fp16_bytes"],
        "compression": mem["compression"],
    }
