"""Needle harness mechanics on tiny_llama (no tokenizer download)."""

import torch

from bmx.cache.niah import needle_retrieved_from_ids
from bmx.cache.specs import CacheCodecSpec
from factories import tiny_llama


def test_needle_harness_runs_and_returns_bool():
    # Mechanics only: a synthetic id sequence, the harness returns a bool verdict
    # comparing the argmax next token under fp16 vs quantized at the query position.
    model = tiny_llama()
    g = torch.Generator().manual_seed(31)
    input_ids = torch.randint(0, 97, (1, 40), generator=g)
    got = needle_retrieved_from_ids(
        model,
        input_ids,
        query_pos=30,
        n_prefill=20,
        k_spec=CacheCodecSpec(arm="fp16"),
        v_spec=CacheCodecSpec(arm="fp16"),
    )
    assert isinstance(got, bool)
