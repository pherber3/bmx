"""NIAH recall metric: argmax proxy (CI) + ROUGE-1 generate (headline).

The argmax proxy is the offline mechanism gate (tokenizer-free, ≤64 tokens for tiny_llama).
The generate path scores ROUGE-1 against the needle sentence and needs a real model.

Also home to the planted-needle probes (former needle.py) and the PG-essay corpus
loader (former haystack.py).
"""

from __future__ import annotations

import torch
from rouge_score import rouge_scorer

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

    Delegates to needle_retrieved by trimming input_ids to query_pos. This is the
    offline mechanism gate (deterministic, indexing-correct), not a recall-quality measure.
    """
    return needle_retrieved(
        model, input_ids[:, : query_pos + 1], answer_id, k_spec, v_spec, n_prefill
    )


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


PG_ESSAYS_DATASET = "sgoel9/paul_graham_essays"


def load_pg_corpus() -> str:
    """Concatenate the Paul Graham essays from the HF dataset into one string.

    ``datasets`` is imported lazily so importing this module triggers no download.
    """
    from datasets import load_dataset

    ds = load_dataset(PG_ESSAYS_DATASET, split="train")
    texts = [t for t in ds["text"] if t]
    assert texts, f"no essay text in {PG_ESSAYS_DATASET}"
    return "\n".join(texts)
