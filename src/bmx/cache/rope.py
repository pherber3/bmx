"""RoPE utilities for the KV-cache codec pipeline.

Provides config-driven cos/sin table construction and the rotate-half
apply_rope that mirrors transformers' apply_rotary_pos_emb exactly.

Supports any HF config that carries `rope_parameters` (llama, mistral,
qwen, etc. — they all use LlamaRotaryEmbedding or an identical copy).
Raises ValueError for configs without a rotary module (e.g. GPT-2).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rope_cos_sin(
    config, S: int, *, start: int = 0, device=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) tables of shape (S, d_head) for *config*.

    Covers absolute positions [start, start+S). Default start=0 gives the
    leading S positions; streaming passes start>0 to extend a table by one
    block without recomputing from 0.

    Uses the model family's own rotary-embedding module (LlamaRotaryEmbedding)
    so rope_scaling variants are inherited from the config, not re-derived.
    Config-only — no weight download.

    Raises ValueError if the config has no rotary embedding (e.g. GPT2Config).
    """
    if not hasattr(config, "rope_parameters"):
        raise ValueError(
            f"Config {type(config).__name__} has no rotary embedding "
            "(rope_parameters missing); rope_cos_sin is not supported."
        )

    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    rotary_emb: nn.Module = LlamaRotaryEmbedding(config=config)
    rotary_emb.eval()
    if device is not None:
        rotary_emb = rotary_emb.to(device)

    d_head = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )

    dummy_x = torch.zeros(1, S, d_head, device=device)  # fp32; forward stays fp32
    position_ids = torch.arange(start, start + S, dtype=torch.long, device=device)

    with torch.no_grad():
        cos, sin = rotary_emb(dummy_x, position_ids.unsqueeze(0))

    return cos.squeeze(0), sin.squeeze(0)  # (1, S, d) -> (S, d)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates the second half of the last dimension into the first half, negated.

    Mirrors transformers' rotate_half exactly:
        x1 = x[..., :d//2]
        x2 = x[..., d//2:]
        return cat(-x2, x1, dim=-1)
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to *x* using the rotate-half convention from transformers.

    Args:
        x:   (h, S, d)  pre-RoPE tensor (any float dtype).
        cos: (S, d)     cosine table from rope_cos_sin.
        sin: (S, d)     sine table from rope_cos_sin.

    Returns:
        Post-RoPE tensor of the same shape and dtype as *x*.

    The formula mirrors transformers' apply_rotary_pos_emb:
        out = (x * cos) + (rotate_half(x) * sin)
    with cos/sin broadcast over the head dimension.
    """
    # Move cos/sin to x's device — rope_cos_sin returns CPU tables (cached),
    # and the streaming _rope_cache also stores CPU; handle both here.
    cos = cos.to(x.device)
    sin = sin.to(x.device)
    # cos/sin are (S, d); broadcast to (1, S, d) so they align with (h, S, d).
    cos = cos.unsqueeze(0)  # (1, S, d)
    sin = sin.unsqueeze(0)  # (1, S, d)

    return (x * cos) + (_rotate_half(x) * sin)
