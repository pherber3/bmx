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


def rope_cos_sin(config, S: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) tables of shape (S, d_head) for *config* and length *S*.

    Uses the model family's own rotary-embedding module (LlamaRotaryEmbedding)
    so rope_scaling variants are inherited from the config rather than
    re-derived.  Config-only — no weight download.

    Supported families: any config with `rope_parameters` (llama, mistral,
    qwen, etc. all share the same rotary math).

    Raises:
        ValueError: if the config has no rotary embedding (e.g. GPT2Config).
    """
    if not hasattr(config, "rope_parameters"):
        raise ValueError(
            f"Config {type(config).__name__} has no rotary embedding "
            "(rope_parameters missing); rope_cos_sin is not supported."
        )

    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    rotary_emb: nn.Module = LlamaRotaryEmbedding(config=config)
    rotary_emb.eval()

    d_head = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )

    # Dummy hidden states: shape (1, S, d_head); float32 so forward stays in fp32.
    dummy_x = torch.zeros(1, S, d_head)
    position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0)  # (1, S)

    with torch.no_grad():
        cos, sin = rotary_emb(dummy_x, position_ids)

    # rotary_emb returns (1, S, d_head); squeeze batch dimension -> (S, d_head)
    cos = cos.squeeze(0)  # (S, d_head)
    sin = sin.squeeze(0)  # (S, d_head)

    return cos, sin


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
    # cos/sin are (S, d); broadcast to (1, S, d) so they align with (h, S, d).
    cos = cos.unsqueeze(0)  # (1, S, d)
    sin = sin.unsqueeze(0)  # (1, S, d)

    return (x * cos) + (_rotate_half(x) * sin)
