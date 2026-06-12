"""Tests for src/bmx/cache/rope.py.

Offline, seeded tiny random models only.
"""

import pytest
import torch
from factories import ids as _ids
from factories import tiny_llama as _tiny_llama
from transformers import GPT2Config

from bmx.cache.collect import collect_cache
from bmx.cache.rope import apply_rope, rope_cos_sin


# ---------------------------------------------------------------------------
# Test 1: cos/sin shapes
# ---------------------------------------------------------------------------


def test_cos_sin_shapes():
    """rope_cos_sin returns (S, d_head) tensors for the llama config."""
    model = _tiny_llama()
    S = 16
    d_head = model.config.hidden_size // model.config.num_attention_heads
    cos, sin = rope_cos_sin(model.config, S)
    assert cos.shape == (S, d_head), f"cos shape {cos.shape} != ({S}, {d_head})"
    assert sin.shape == (S, d_head), f"sin shape {sin.shape} != ({S}, {d_head})"


# ---------------------------------------------------------------------------
# Test 2: GPT-2 raises ValueError (no rotary module)
# ---------------------------------------------------------------------------


def test_gpt2_raises_value_error():
    """GPT2Config has no rope_parameters; rope_cos_sin must raise ValueError."""
    gpt2_cfg = GPT2Config()
    with pytest.raises(ValueError, match="rotary"):
        rope_cos_sin(gpt2_cfg, 16)


# ---------------------------------------------------------------------------
# Test 3: Norm preservation under apply_rope (fp32)
# ---------------------------------------------------------------------------


def test_apply_rope_norm_preservation():
    """apply_rope is a rotation; per-(head, token) vector norms must be preserved."""
    model = _tiny_llama()
    S = 16
    d_head = model.config.hidden_size // model.config.num_attention_heads
    h_kv = model.config.num_key_value_heads

    torch.manual_seed(99)
    x = torch.randn(h_kv, S, d_head)  # fp32
    cos, sin = rope_cos_sin(model.config, S)
    y = apply_rope(x, cos.float(), sin.float())

    norm_x = x.norm(dim=-1)  # (h_kv, S)
    norm_y = y.norm(dim=-1)
    rel_diff = ((norm_x - norm_y).abs() / norm_x.clamp(min=1e-8)).max()
    assert rel_diff < 1e-5, (
        f"Norm not preserved by apply_rope: rel diff = {rel_diff:.4e}"
    )


# ---------------------------------------------------------------------------
# Test 4: Load-bearing self-validation against collect_cache
# ---------------------------------------------------------------------------


def test_apply_rope_matches_collect_cache():
    """apply_rope(k_pre, cos, sin) ≈ k (rel Frobenius < 1e-2, fp16 noise)."""
    model = _tiny_llama()
    ids = _ids(seq=16)
    cache = collect_cache(model, ids, n_q_keep=256)

    S = ids.shape[1]
    cos, sin = rope_cos_sin(model.config, S)

    for i in range(model.config.num_hidden_layers):
        k_pre = cache[f"layer{i}.k_pre"].float()  # (h_kv, S, d)
        k = cache[f"layer{i}.k"].float()

        k_reconstructed = apply_rope(k_pre, cos.float(), sin.float())

        fro_err = (k_reconstructed - k).norm() / k.norm()
        assert fro_err < 1e-2, (
            f"layer{i}: apply_rope(k_pre) does not match k; "
            f"rel Frobenius = {fro_err:.4e}"
        )
