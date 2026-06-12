"""Tests for src/bmx/cache/collect.py — offline, tiny random models only.

Test idiom mirrors tests/test_layer_swap.py: build from config, no downloads.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from bmx.cache.collect import collect_cache, load_cache, save_cache


# ---------------------------------------------------------------------------
# Tiny model factories
# ---------------------------------------------------------------------------


def _tiny_gpt2():
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=97, n_positions=64)
    torch.manual_seed(0)
    return GPT2LMHeadModel(cfg)


def _tiny_llama():
    cfg = LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=32,
        intermediate_size=64,
        vocab_size=97,
        max_position_embeddings=64,
    )
    torch.manual_seed(1)
    return LlamaForCausalLM(cfg)


def _ids(vocab=97, seq=12, seed=42):
    return torch.randint(
        0, vocab, (1, seq), generator=torch.Generator().manual_seed(seed)
    )


# ---------------------------------------------------------------------------
# Test 1: Shapes and keys — GPT-2 and Llama
# ---------------------------------------------------------------------------


def test_shapes_keys_gpt2():
    model = _tiny_gpt2()
    ids = _ids(seq=12)
    n_q_keep = 8

    cache = collect_cache(model, ids, n_q_keep=n_q_keep)

    n_layer = model.config.n_layer
    h = model.config.n_head
    h_kv = model.config.n_head  # GPT-2: h_kv == h
    S = ids.shape[1]
    d = model.config.n_embd // model.config.n_head

    for i in range(n_layer):
        assert f"layer{i}.k" in cache, f"missing layer{i}.k"
        assert f"layer{i}.v" in cache, f"missing layer{i}.v"
        assert f"layer{i}.q" in cache, f"missing layer{i}.q"
        assert f"layer{i}.k_pre" in cache, f"missing layer{i}.k_pre"

        k = cache[f"layer{i}.k"]
        v = cache[f"layer{i}.v"]
        q = cache[f"layer{i}.q"]
        k_pre = cache[f"layer{i}.k_pre"]

        assert k.shape == (h_kv, S, d), f"layer{i}.k shape {k.shape}"
        assert v.shape == (h_kv, S, d), f"layer{i}.v shape {v.shape}"
        assert q.shape == (h, min(n_q_keep, S), d), f"layer{i}.q shape {q.shape}"
        assert k_pre.shape == (h_kv, S, d), f"layer{i}.k_pre shape {k_pre.shape}"

        # All tensors stored in fp16
        assert k.dtype == torch.float16, f"layer{i}.k dtype {k.dtype}"
        assert v.dtype == torch.float16
        assert q.dtype == torch.float16
        assert k_pre.dtype == torch.float16


def test_shapes_keys_llama():
    model = _tiny_llama()
    ids = _ids(seq=12)
    n_q_keep = 5

    cache = collect_cache(model, ids, n_q_keep=n_q_keep)

    n_layer = model.config.num_hidden_layers
    h = model.config.num_attention_heads
    h_kv = model.config.num_key_value_heads
    S = ids.shape[1]
    d = model.config.hidden_size // model.config.num_attention_heads

    for i in range(n_layer):
        k = cache[f"layer{i}.k"]
        v = cache[f"layer{i}.v"]
        q = cache[f"layer{i}.q"]
        k_pre = cache[f"layer{i}.k_pre"]

        assert k.shape == (h_kv, S, d), f"layer{i}.k shape {k.shape}"
        assert v.shape == (h_kv, S, d), f"layer{i}.v shape {v.shape}"
        assert q.shape == (h, min(n_q_keep, S), d), f"layer{i}.q shape {q.shape}"
        assert k_pre.shape == (h_kv, S, d), f"layer{i}.k_pre shape {k_pre.shape}"

        assert k.dtype == torch.float16
        assert v.dtype == torch.float16
        assert q.dtype == torch.float16
        assert k_pre.dtype == torch.float16


def test_q_truncation_respects_n_q_keep():
    """n_q_keep larger than S gives q with S positions (no padding)."""
    model = _tiny_gpt2()
    ids = _ids(seq=6)
    S = 6
    n_q_keep = 100  # larger than S

    cache = collect_cache(model, ids, n_q_keep=n_q_keep)
    h = model.config.n_head
    d = model.config.n_embd // model.config.n_head

    q = cache["layer0.q"]
    assert q.shape == (h, S, d), f"expected (h={h}, S={S}, d={d}), got {q.shape}"


# ---------------------------------------------------------------------------
# Test 2: GPT-2 physics invariant — k_pre ≈ k (no RoPE)
# ---------------------------------------------------------------------------


def test_gpt2_kpre_equals_k():
    """GPT-2 has no RoPE; pre-RoPE key must equal post-RoPE key within fp16 noise."""
    model = _tiny_gpt2()
    ids = _ids(seq=16)
    cache = collect_cache(model, ids, n_q_keep=256)

    for i in range(model.config.n_layer):
        k = cache[f"layer{i}.k"].float()
        k_pre = cache[f"layer{i}.k_pre"].float()
        assert torch.allclose(k_pre, k, atol=1e-2), (
            f"layer{i}: k_pre != k; max abs diff = {(k_pre - k).abs().max():.4f}"
        )


# ---------------------------------------------------------------------------
# Test 3: Llama physics invariant — k_pre ≠ k but per-vector norms preserved
# ---------------------------------------------------------------------------


def test_llama_rope_norm_preserving():
    """RoPE is a rotation per head vector, so ||k_pre[h,t,:]|| == ||k[h,t,:]||."""
    model = _tiny_llama()
    ids = _ids(seq=16)
    cache = collect_cache(model, ids, n_q_keep=256)

    for i in range(model.config.num_hidden_layers):
        k = cache[f"layer{i}.k"].float()  # (h_kv, S, d)
        k_pre = cache[f"layer{i}.k_pre"].float()

        # They must differ (RoPE rotates)
        assert not torch.allclose(k_pre, k, atol=1e-3), (
            f"layer{i}: k_pre == k — RoPE was not applied?"
        )

        # Per-vector (per head, per token) norms must be preserved
        norm_k = k.norm(dim=-1)  # (h_kv, S)
        norm_pre = k_pre.norm(dim=-1)  # (h_kv, S)
        rel_diff = ((norm_k - norm_pre).abs() / norm_k.clamp(min=1e-8)).max()
        assert rel_diff < 1e-2, (
            f"layer{i}: RoPE changed vector norms; rel diff = {rel_diff:.4e}"
        )


# ---------------------------------------------------------------------------
# Test 4: save/load round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory,as_str", [(_tiny_gpt2, False), (_tiny_llama, True)])
def test_save_load_roundtrip(factory, as_str):
    cache = collect_cache(factory(), _ids(seq=10), n_q_keep=4)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.safetensors"
        save_cache(cache, str(path) if as_str else path)  # both path types accepted
        loaded = load_cache(str(path) if as_str else path)

    assert set(loaded.keys()) == set(cache.keys())
    for key in cache:
        assert torch.equal(cache[key], loaded[key]), f"round-trip mismatch on {key}"


def test_unsupported_architecture_raises():
    class Dummy:
        config = GPT2Config()

    with pytest.raises(ValueError, match="unsupported architecture"):
        collect_cache(Dummy(), torch.zeros(1, 4, dtype=torch.long))
