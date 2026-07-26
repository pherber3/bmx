"""Tests for optional per-layer codec specs in quantized_prefill_ppl.

Test idiom mirrors tests/test_ppl_eval.py: build from tests/factories.py,
offline, seeded, no downloads.
"""

import torch
from factories import tiny_llama

from bmx.cache.ppl_eval import quantized_prefill_ppl
from bmx.cache.specs import CacheCodecSpec


def _ids(model, n=96):
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, model.config.vocab_size, (1, n), generator=g)


def test_per_layer_specs_identity_invariant():
    """A per-layer list of identical specs must reproduce the single-spec result."""
    model = tiny_llama()
    ids = _ids(model)
    spec = CacheCodecSpec(arm="rtn_channel", bits=3, group=8)
    v = CacheCodecSpec(arm="fp16")
    n_layer = model.config.num_hidden_layers
    a = quantized_prefill_ppl(model, ids, 64, spec, v)
    b = quantized_prefill_ppl(
        model, ids, 64, spec, v, k_specs=[spec] * n_layer, v_specs=[v] * n_layer
    )
    assert abs(a["ppl"] - b["ppl"]) < 1e-6
    assert abs(a["bpe_k"] - b["bpe_k"]) < 1e-9


def test_per_layer_specs_mixed_bits_runs():
    model = tiny_llama()
    ids = _ids(model)
    n_layer = model.config.num_hidden_layers
    k_specs = [
        CacheCodecSpec(arm="rtn_channel", bits=2 + (i % 2), group=8)
        for i in range(n_layer)
    ]
    out = quantized_prefill_ppl(
        model, ids, 64, k_specs[0], CacheCodecSpec(arm="fp16"), k_specs=k_specs
    )
    # rtn_channel bpe includes honest per-group scale/zero-point metadata, so
    # the mean isn't simply mean(bits) — bound around the actual overhead
    # (bits=2 -> bpe=4.0, bits=3 -> bpe=5.0 for this tiny model's group=8),
    # well clear of the fp16 no-op value (16.0).
    assert out["ppl"] > 0 and 3.5 < out["bpe_k"] < 5.5
