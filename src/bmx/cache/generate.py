"""Task-agnostic generation + compression accounting through the streaming caches.

generate_through_cache is the ONE generation loop shared by NIAH and LongBench (single
EOS/packed/fp16-routing logic); compression_for reads honest blended bpe off a calibration
prefill.
"""

from __future__ import annotations

import torch

from bmx.cache.hf_compat import resolve_vocab_size
from bmx.cache.packed_streaming import PackedStreamingCache
from bmx.cache.specs import CacheCodecSpec
from bmx.cache.streaming import StreamingQuantizedCache


def generate_through_cache(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    n_prefill: int,
    k_spec: CacheCodecSpec,
    v_spec: CacheCodecSpec,
    max_new_tokens: int,
    strip: bool = True,
    use_packed: bool = False,
) -> str:
    """Prefill prompt_ids into a streaming cache, greedy-decode, return the answer.

    ``use_packed``: when True, run through ``PackedStreamingCache`` (packed codes
    resident + chunked dequant-attention at decode, flash SDPA at prefill) — the
    memory-saving path that keeps the bpe footprint resident instead of a second
    dense KV copy. When False (default), uses ``StreamingQuantizedCache`` (the
    quality/parity reference, dense dequant resident). Both produce identical
    tokens (parity-gated); ``use_packed`` only changes the resident memory.

    The whole prompt is streamed into the cache (prefill block [0, n_prefill) then the rest
    [n_prefill, L) as one forward), and greedy decoding continues from the last prompt logit.
    Both prompt forwards pass ``logits_to_keep=1`` so the (S × vocab) logit tensor is never
    materialized over all positions — at 128k context that all-position logit tensor is ~63 GB
    (vocab 128k × 131k × fp32) and OOMs the GPU even though the compressed KV cache is only a
    few GB. We never read prompt-position logits (only the last one seeds decoding), so keeping
    1 is exact, not an approximation. ``strip`` removes surrounding whitespace (off for
    whitespace-sensitive scorers).
    """
    prompt_ids = prompt_ids.to(model.device)
    L = prompt_ids.shape[1]
    assert n_prefill < L, (
        f"n_prefill={n_prefill} >= prompt length {L}; nothing to generate"
    )
    # Full EOS set, mirroring model.generate(): Llama-3.1-Instruct stops on any of
    # {128001 <|end_of_text|>, 128008 <|eom_id|>, 128009 <|eot_id|>}, whereas
    # tokenizer.eos_token_id is just one (128009). Prefer generation_config (what generate
    # uses), fall back to model.config, then the tokenizer. None => never stop early.
    _eos = (
        getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        or getattr(model.config, "eos_token_id", None)
        or getattr(tokenizer, "eos_token_id", None)
    )
    eos_ids = {_eos} if isinstance(_eos, int) else set(_eos or [])

    # fp16 is the uncompressed baseline — it has no packed representation
    # (PackedStreamingCache's flush would call quantize_packed('fp16'), which raises).
    # Route the all-fp16 spec through the dense cache even under use_packed; its
    # recall is identical either way (no quantization), and fp16-through-packed
    # would save zero memory. Only the compressing arms use the packed path.
    is_fp16 = k_spec.arm == "fp16" and v_spec.arm == "fp16"
    cache_cls = (
        PackedStreamingCache
        if (use_packed and not is_fp16)
        else StreamingQuantizedCache
    )
    cache = cache_cls(model.config, k_spec=k_spec, v_spec=v_spec)
    cache.attach(model)
    new_ids: list[int] = []
    with cache:
        with torch.no_grad():
            # Stream the whole prompt through the cache in two blocks (mirrors the prior
            # prefill/continuation split so cache contents are identical), logits_to_keep=1.
            model(
                prompt_ids[:, :n_prefill],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            out = model(
                prompt_ids[:, n_prefill:],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            nxt = out.logits[0, -1].argmax().view(1, 1)
            new_ids.append(int(nxt))
            for _ in range(max_new_tokens - 1):
                out = model(
                    nxt, past_key_values=cache, use_cache=True, logits_to_keep=1
                )
                nxt = out.logits[0, -1].argmax().view(1, 1)
                tid = int(nxt)
                new_ids.append(tid)
                if tid in eos_ids:
                    break
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text.strip() if strip else text


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
            # logits_to_keep=1: only the cache's bpe accounting is read, never the logits, so
            # don't materialize the (length × vocab) logit tensor (~32 GB at 64k × 128k vocab).
            model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    bpe_k, bpe_v = cache.bits_per_entry()
    mem = cache.memory_report(seq_len=length)
    return bpe_k, bpe_v, mem["compression"]


def avg_bpe(bpe_k: float, bpe_v: float) -> float:
    """Blended KV bits-per-entry (the TurboQuant Table-1 "KV Size" axis).

    One definition shared by every experiment that reports `kv_size_bits`, so the
    K/V blend lives in a single place if it ever changes (e.g. size-weighting K vs V).
    """
    return (bpe_k + bpe_v) / 2
