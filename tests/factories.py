"""Shared tiny-model factories for cache tests — offline, seeded, no downloads.

Consolidates the copies previously duplicated across test_cache_collect.py,
test_cache_rope.py, and test_ppl_eval.py.  Both model factories return models
in eval mode.
"""

import torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM


def tiny_gpt2():
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=97, n_positions=64)
    torch.manual_seed(0)
    return GPT2LMHeadModel(cfg).eval()


def tiny_llama():
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
    return LlamaForCausalLM(cfg).eval()


def ids(vocab=97, seq=12, seed=42):
    return torch.randint(
        0, vocab, (1, seq), generator=torch.Generator().manual_seed(seed)
    )
