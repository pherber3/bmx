"""Needle-in-a-haystack retrieval probe through the streaming cache.

The headline 'in practice' benchmark: TurboQuant's own paper test. The
id-level helper (needle_retrieved_from_ids) is tokenizer-free for tests; the
text-level builders are used by the experiment with a real tokenizer.
"""

from __future__ import annotations

import torch

from bmx.cache.specs import CacheCodecSpec
from bmx.cache.streaming import StreamingQuantizedCache


def _argmax_next_at(model, input_ids, query_pos, k_spec, v_spec, n_prefill):
    cache = StreamingQuantizedCache(model.config, k_spec=k_spec, v_spec=v_spec)
    cache.attach(model)
    with cache:
        with torch.no_grad():
            model(
                input_ids[:, :n_prefill],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            # logits_to_keep=1: only the query-position logit is read (argmax), so the
            # (S × vocab) tensor over all positions is never built — would OOM at long context.
            out = model(
                input_ids[:, n_prefill : query_pos + 1],
                past_key_values=cache,
                logits_to_keep=1,
            )
    return out.logits[0, -1].argmax().item()


def needle_retrieved_from_ids(
    model,
    input_ids: torch.Tensor,
    query_pos: int,
    n_prefill: int,
    k_spec: CacheCodecSpec,
    v_spec: CacheCodecSpec,
) -> bool:
    """True if the quantized-cache next-token at query_pos matches the fp16 cache's.

    Tokenizer-free retrieval-fidelity proxy: does compression change the model's
    decision at the query? (Real needle accuracy uses build_needle_ids below.)
    """
    fp16 = CacheCodecSpec(arm="fp16")
    ref = _argmax_next_at(model, input_ids, query_pos, fp16, fp16, n_prefill)
    got = _argmax_next_at(model, input_ids, query_pos, k_spec, v_spec, n_prefill)
    return bool(ref == got)


def build_needle_ids(
    tokenizer,
    n_context: int,
    depth_frac: float,
    needle_text: str = "The secret code is 42.",
    question_text: str = "\nThe secret code is",
):
    """Filler haystack with the needle at depth_frac; returns (ids, answer_id).

    Used by the experiment with a real tokenizer; not exercised in unit tests.
    """
    filler = (" the cat sat on the mat.") * (n_context // 6)
    ids_filler = tokenizer(filler, return_tensors="pt").input_ids
    needle = tokenizer(needle_text, return_tensors="pt").input_ids
    question = tokenizer(question_text, return_tensors="pt").input_ids
    answer = tokenizer(" 42", return_tensors="pt").input_ids[0, -1].item()

    cut = int(ids_filler.shape[1] * depth_frac)
    input_ids = torch.cat(
        [ids_filler[:, :cut], needle, ids_filler[:, cut:], question], dim=1
    )
    return input_ids, answer


def needle_retrieved(
    model, input_ids, answer_token_id, k_spec, v_spec, n_prefill
) -> bool:
    """True if the model's next-token argmax at the end equals answer_token_id."""
    last_pos = input_ids.shape[1] - 1
    predicted = _argmax_next_at(model, input_ids, last_pos, k_spec, v_spec, n_prefill)
    return bool(predicted == answer_token_id)
