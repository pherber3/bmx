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
        # 512 so tests can exceed the 128-token PAGE flush threshold (the cache now
        # commits on a fixed 128-token page grid; flushing needs PAGE+recent_window).
        max_position_embeddings=512,
    )
    torch.manual_seed(1)
    return LlamaForCausalLM(cfg).eval()


def tiny_llama_d32():
    """Tiny Llama with head_dim=32 (>=16, power of 2) — for the fused k2b kernel,
    whose tl.dot (lowrank-K, rotate_half, V-Hadamard) needs d>=16 and d a power of 2.
    hidden=64, 2 q heads, 1 kv head -> d_head=32, n_q_groups=2.
    """
    cfg = LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=97,
        max_position_embeddings=512,  # exceed the 128-token PAGE flush threshold
    )
    torch.manual_seed(1)
    return LlamaForCausalLM(cfg).eval()


def ids(vocab=97, seq=12, seed=42):
    return torch.randint(
        0, vocab, (1, seq), generator=torch.Generator().manual_seed(seed)
    )


def tiny_packed_blocks_prerope(*, n_q_heads, n_q_groups, d, blk, n_blocks, seed=0):
    """Build (q, k_blocks, v_blocks, kwargs) for pre-RoPE decode tests.

    rope_cos/sin are pre-cast to fp16 (mirrors the grow-time cast optimisation).
    k_pre_rope=True so chunked_dequant_attention applies RoPE at read.
    """
    from bmx.cache.codecs import quantize_packed
    from bmx.cache.collect import to_matrix

    torch.manual_seed(seed)
    h_kv = n_q_heads // n_q_groups
    S = blk * n_blocks
    q = torch.randn(n_q_heads, 1, d)
    # Already compute-dtype (fp16) — mirrors the grow-time cast (the optimisation).
    cos = torch.randn(S, d).to(torch.float16)
    sin = torch.randn(S, d).to(torch.float16)
    k_blocks, v_blocks = [], []
    for i in range(n_blocks):
        start, end = i * blk, (i + 1) * blk
        kp, _ = quantize_packed(
            "rtn_token",
            to_matrix(torch.randn(h_kv, blk, d)),
            bits=4,
            group=8,
            seed=seed,
        )
        vp, _ = quantize_packed(
            "rtn_token",
            to_matrix(torch.randn(h_kv, blk, d)),
            bits=4,
            group=8,
            seed=seed,
        )
        k_blocks.append((kp, start, end))
        v_blocks.append((vp, start, end))
    kwargs = dict(
        k_arm="rtn_token",
        v_arm="rtn_token",
        group=8,
        seed=seed,
        k_pre_rope=True,
        rope_cos=cos,
        rope_sin=sin,
        k_tail=None,
        v_tail=None,
        n_q_groups=n_q_groups,
        scale=d**-0.5,
    )
    return q, k_blocks, v_blocks, kwargs


def tiny_packed_blocks(
    *, n_q_heads, n_q_groups, n_q, d, blk, n_blocks, arm="rtn_token", group=8, seed=0
):
    """Build (q, k_blocks, v_blocks, kwargs) for chunked/naive attention tests.

    Returns rtn_token packed blocks (decode case: n_q=1).
    """
    from bmx.cache.codecs import quantize_packed
    from bmx.cache.collect import to_matrix

    h_kv = n_q_heads // n_q_groups
    torch.manual_seed(seed)
    q = torch.randn(n_q_heads, n_q, d)
    k_blocks, v_blocks = [], []
    for i in range(n_blocks):
        start, end = i * blk, (i + 1) * blk
        kM = to_matrix(torch.randn(h_kv, blk, d))
        vM = to_matrix(torch.randn(h_kv, blk, d))
        kp, _ = quantize_packed(arm, kM, bits=4, group=group, seed=seed)
        vp, _ = quantize_packed(arm, vM, bits=4, group=group, seed=seed)
        k_blocks.append((kp, start, end))
        v_blocks.append((vp, start, end))
    kwargs = dict(
        k_arm=arm,
        v_arm=arm,
        group=group,
        seed=seed,
        k_pre_rope=False,
        rope_cos=None,
        rope_sin=None,
        k_tail=None,
        v_tail=None,
        n_q_groups=n_q_groups,
        scale=d**-0.5,
    )
    return q, k_blocks, v_blocks, kwargs
