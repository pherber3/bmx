"""NIAH recall metric: argmax proxy (CI) + ROUGE-1 generate (headline).

Mirrors needle.py's proxy/real split. The argmax proxy is the offline mechanism
gate (tokenizer-free, ≤64 tokens for tiny_llama). The generate path is the headline
recall (ROUGE-1 vs the needle sentence), VM/real-model only — added in Task 3.

All arms route through the same StreamingQuantizedCache path used by the ppl sweep.
"""

from __future__ import annotations

import torch

from bmx.cache.needle import _argmax_next_at
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

    Offline mechanism gate: finite, deterministic, indexing-correct. Real recall
    quality is the ROUGE-1 generate path (Task 3), VM only.
    """
    got = _argmax_next_at(model, input_ids, query_pos, k_spec, v_spec, n_prefill)
    return bool(got == answer_id)
