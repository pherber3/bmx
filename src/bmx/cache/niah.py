"""NIAH recall metric: argmax proxy (CI) + ROUGE-1 generate (headline).

The argmax proxy is the offline mechanism gate (tokenizer-free, ≤64 tokens for tiny_llama).
The generate path scores ROUGE-1 against the needle sentence and needs a real model.
"""

from __future__ import annotations

import torch
from rouge_score import rouge_scorer

from bmx.cache.needle import needle_retrieved
from bmx.cache.specs import CacheCodecSpec


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

    Delegates to needle.needle_retrieved by trimming input_ids to query_pos. This is the
    offline mechanism gate (deterministic, indexing-correct), not a recall-quality measure.
    """
    return needle_retrieved(
        model, input_ids[:, : query_pos + 1], answer_id, k_spec, v_spec, n_prefill
    )


# Default needle/question from the Fu et al. NIAH harness.
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
    """ROUGE-1 F-measure ×10 (0–10) of the needle vs the response.

    The Fu et al. metric (needle_in_haystack.py:265). F-measure includes precision, so a
    verbose instruct-model response that contains the needle plus chatter is penalized; see
    rouge1_recall_only for the precision-free retrieval signal.
    """
    return _SCORER.score(needle_text, response_text)["rouge1"].fmeasure * 10.0


def rouge1_recall_only(needle_text: str, response_text: str) -> float:
    """ROUGE-1 recall ×10 (0–10): fraction of needle words present in the response.

    Ignores response verbosity — a response containing all needle words scores ~10 even with
    trailing chatter. The "did the model retrieve the needle" signal, alongside the
    paper-faithful F-measure.
    """
    return _SCORER.score(needle_text, response_text)["rouge1"].recall * 10.0


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
