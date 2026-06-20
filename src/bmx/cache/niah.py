"""NIAH recall metric: argmax proxy (CI) + ROUGE-1 generate (headline).

Mirrors needle.py's proxy/real split. The argmax proxy is the offline mechanism
gate (tokenizer-free, ≤64 tokens for tiny_llama). The generate path is the headline
recall (ROUGE-1 vs the needle sentence), VM/real-model only — added in Task 3.

All arms route through the same StreamingQuantizedCache path used by the ppl sweep.
"""

from __future__ import annotations

import torch
from rouge_score import rouge_scorer

from bmx.cache.needle import needle_retrieved
from bmx.cache.specs import CacheCodecSpec
from bmx.cache.streaming import StreamingQuantizedCache


def build_niah_ids_synthetic(
    vocab: int,
    n_context: int,
    depth_frac: float,
    *,
    answer_id: int,
    seed: int,
) -> torch.Tensor:
    """Synthetic NIAH id sequence (tokenizer-free, for the offline mechanism gate).

    A repeated-filler id stream with a single distinctive ``answer_id`` planted at
    ``depth_frac`` (the needle). The final positions form a short query so that the
    fp16 model's next-token argmax at the end is a well-defined decision the proxy
    can compare across arms. Returns (1, n_context).
    """
    assert 0 <= depth_frac <= 1, "depth_frac in [0, 1]"
    assert n_context >= 4, "need room for filler + needle + query"
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (1, n_context), generator=g)
    # Plant the needle (answer_id) at depth.
    plant = max(1, min(n_context - 2, int(n_context * depth_frac)))
    ids[0, plant] = answer_id
    # Query tail: make the last token a marker so argmax-at-end is a stable probe.
    ids[0, -1] = answer_id  # last-seen id; mechanism probe only (not a quality claim)
    return ids


def niah_recall_argmax(
    model,
    input_ids: torch.Tensor,
    query_pos: int,
    n_prefill: int,
    k_spec: CacheCodecSpec,
    v_spec: CacheCodecSpec,
    answer_id: int,
) -> bool:
    """True iff the streaming-cache next-token argmax at query_pos equals answer_id.

    Offline mechanism gate: finite, deterministic, indexing-correct. Real recall
    quality is the ROUGE-1 generate path (Task 3), VM only. Delegates to
    needle.needle_retrieved (argmax-equals-answer) so the cache-probe lives in one
    place; query_pos is honored by trimming input_ids to that position.
    """
    return needle_retrieved(
        model, input_ids[:, : query_pos + 1], answer_id, k_spec, v_spec, n_prefill
    )


# Defaults follow the Fu et al. harness (eval/needle/needle_in_haystack.py); the
# Task 0 ledger is the source of truth if the vault refined these.
NEEDLE_TEXT = (
    "\nThe best thing to do in San Francisco is eat a sandwich and sit in "
    "Dolores Park on a sunny day.\n"
)
QUESTION_TEXT = "What is the best thing to do in San Francisco?"
PROMPT_TEMPLATE = (
    "This is a very long story book: <book> {context} </book>.\n"
    "Based on the content of the book, Question: {question}\nAnswer:"
)

_SCORER = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)


def rouge1_recall(needle_text: str, response_text: str) -> float:
    """ROUGE-1 F-measure ×10 (0–10) of the needle vs the response — the headline scorer.

    Matches the Fu et al. metric (needle_in_haystack.py:265). Pure function; the
    streaming-cache generate path feeds the model response in.
    """
    return _SCORER.score(needle_text, response_text)["rouge1"].fmeasure * 10.0


def _insert_needle_at_sentence_boundary(
    tokenizer, context_ids: list[int], needle_ids: list[int], depth_percent: float
) -> list[int]:
    """Insert needle_ids into context_ids at depth_percent, snapped back to a period."""
    if depth_percent >= 100:
        return context_ids + needle_ids
    insertion = int(len(context_ids) * (depth_percent / 100.0))
    period_ids = tokenizer.encode(".", add_special_tokens=False)
    # Walk back to the nearest sentence boundary; check the index directly rather
    # than re-slicing the head each step.
    while insertion > 0 and context_ids[insertion - 1] not in period_ids:
        insertion -= 1
    return context_ids[:insertion] + needle_ids + context_ids[insertion:]


def build_niah_prompt(
    tokenizer,
    context_length: int,
    depth_percent: float,
    *,
    haystack: str,
    needle_text: str = NEEDLE_TEXT,
    question_text: str = QUESTION_TEXT,
    buffer: int = 200,
) -> torch.Tensor:
    """RULER-style NIAH prompt ids (real-tokenizer path; VM only).

    Trims ``haystack`` to ``context_length - buffer`` tokens, inserts ``needle_text``
    at ``depth_percent`` snapped to a sentence boundary, wraps in PROMPT_TEMPLATE +
    question. Returns (1, L).
    """
    needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)
    ctx_ids = tokenizer.encode(haystack, add_special_tokens=False)
    budget = context_length - buffer
    if len(ctx_ids) + len(needle_ids) > budget:
        ctx_ids = ctx_ids[: budget - len(needle_ids)]
    woven = _insert_needle_at_sentence_boundary(
        tokenizer, ctx_ids, needle_ids, depth_percent
    )
    context = tokenizer.decode(woven)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question_text)
    return tokenizer(prompt, return_tensors="pt").input_ids


def generate_through_cache(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    n_prefill: int,
    k_spec: CacheCodecSpec,
    v_spec: CacheCodecSpec,
    max_new_tokens: int,
    strip: bool = True,
) -> str:
    """Prefill prompt_ids into a StreamingQuantizedCache, greedy-generate, return the answer.

    The shared headline generate path for every metric that scores a generation through the
    compressed cache (NIAH ROUGE-1, LongBench code_sim, ...). One fair code path.

    Continuation-only contract (mirrors needle.py:21-22):
      Step 1 fills cache positions [0, n_prefill) via a quantize-on-append forward.
      Step 2 feeds ONLY prompt_ids[:, n_prefill:] to generate(); HF returns the supplied
      continuation followed by the newly-decoded tokens, so the new tokens start at index
      (L - n_prefill) in out[0]. Decoding out[0, L - n_prefill :] yields exactly the answer.
    """
    L = prompt_ids.shape[1]
    cont_len = L - n_prefill
    cache = StreamingQuantizedCache(model.config, k_spec=k_spec, v_spec=v_spec)
    cache.attach(model)
    with cache:
        with torch.no_grad():
            model(prompt_ids[:, :n_prefill], past_key_values=cache, use_cache=True)
            out = model.generate(
                prompt_ids[:, n_prefill:],
                past_key_values=cache,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
    text = tokenizer.decode(out[0, cont_len:], skip_special_tokens=True)
    return text.strip() if strip else text


def niah_recall_generate(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    n_prefill: int,
    k_spec: CacheCodecSpec,
    v_spec: CacheCodecSpec,
    needle_text: str = NEEDLE_TEXT,
    max_new_tokens: int = 50,
) -> float:
    """Prefill the prompt into the streaming cache, greedy-generate, score ROUGE-1.

    Headline recall (VM/real model). Now a thin wrapper over generate_through_cache so the
    generate path lives in one place across metrics.
    """
    response = generate_through_cache(
        model, tokenizer, prompt_ids, n_prefill, k_spec, v_spec, max_new_tokens
    )
    return rouge1_recall(needle_text, response)
