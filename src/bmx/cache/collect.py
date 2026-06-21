"""Cache collection via hooked forward passes.

Public API
----------
collect_cache(model, input_ids, n_q_keep=256) -> dict[str, Tensor]
    Single prefill; returns per-layer k/v/q/k_pre tensors in fp16.

save_cache(tensors, path) -> None
load_cache(path) -> dict[str, Tensor]

to_matrix(kv) -> Tensor / from_matrix(M, h) -> Tensor
    The K1 layout convention: (h, S, d) <-> (S, h*d) fp32 matrix.

Supported architectures (dispatched structurally, not by model_type string)
---------------------------------------------------------------------------
- GPT-2 style (``model.transformer.h[i].attn.c_attn`` packed QKV projection):
  no RoPE, so k_pre ≈ k.
- Llama style (``model.model.layers[i].self_attn.q_proj``/``k_proj`` split
  projections — llama/qwen/mistral/...): k_pre is pre-RoPE, k is post-RoPE.

K/V come from ``past_key_values`` (transformers 5.x DynamicCache,
``.layers[i].keys/.values``, shape (1, h_kv, S, d)); Q and pre-RoPE K come
from forward hooks on the projection modules. Hooks truncate Q to the last
n_q_keep positions and cast to fp16 at capture time, so peak transient memory
stays ~one layer of activations, not the full-sequence Q of every layer.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def to_matrix(kv: torch.Tensor) -> torch.Tensor:
    """(h, S, d) cache tensor -> (S, h*d) fp32 matrix (the census/codec layout)."""
    h, S, d = kv.shape
    return kv.permute(1, 0, 2).reshape(S, h * d).float()


def from_matrix(M: torch.Tensor, h: int) -> torch.Tensor:
    """(S, h*d) matrix -> (h, S, d), inverse of to_matrix (dtype preserved)."""
    S, hd = M.shape
    return M.reshape(S, h, hd // h).permute(1, 0, 2)


def _get_kv_layer(past_key_values, i: int):
    """(key, value) of shape (1, h_kv, S, d_head) for layer i (DynamicCache)."""
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[i]
        return layer.keys, layer.values
    raise RuntimeError(
        f"Unknown past_key_values type {type(past_key_values)}; "
        "cannot extract per-layer K/V tensors."
    )


def _reshape_heads(out: torch.Tensor, n_head: int, d: int) -> torch.Tensor:
    """(1, S, n_head*d) projection output -> (n_head, S, d), fp16 contiguous."""
    S = out.shape[1]
    heads = out.reshape(1, S, n_head, d).permute(0, 2, 1, 3).squeeze(0)
    return heads.contiguous().to(torch.float16)


def _register_gpt2_hooks(model, store: dict, n_q_keep: int):
    """Hooks on c_attn (packed Q|K|V columns); returns (handles, n_layer)."""
    cfg = model.config
    h, d = cfg.n_head, cfg.n_embd // cfg.n_head
    n_embd = h * d
    handles = []
    for i, block in enumerate(model.transformer.h):

        def hook(module, inp, out, i=i):
            q = _reshape_heads(out[..., :n_embd], h, d)
            store[f"layer{i}.q"] = q[:, -n_q_keep:, :].contiguous()
            store[f"layer{i}.k_pre"] = _reshape_heads(
                out[..., n_embd : 2 * n_embd], h, d
            )

        handles.append(block.attn.c_attn.register_forward_hook(hook))
    return handles, len(model.transformer.h)


def _register_qkproj_hooks(model, store: dict, n_q_keep: int):
    """Hooks on q_proj/k_proj (Llama-family); returns (handles, n_layer)."""
    from bmx.cache.streaming import resolve_decoder_layers, resolve_text_config

    cfg = resolve_text_config(model.config)  # unwrap multimodal text_config
    h = cfg.num_attention_heads
    h_kv = getattr(cfg, "num_key_value_heads", h)
    # qwen3/gemma-class configs set head_dim != hidden_size // heads explicitly
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // h
    handles = []
    layers = resolve_decoder_layers(model)
    for i, layer in enumerate(layers):

        def q_hook(module, inp, out, i=i):
            q = _reshape_heads(out, h, d)
            store[f"layer{i}.q"] = q[:, -n_q_keep:, :].contiguous()

        def k_hook(module, inp, out, i=i):
            store[f"layer{i}.k_pre"] = _reshape_heads(out, h_kv, d)

        handles.append(layer.self_attn.q_proj.register_forward_hook(q_hook))
        handles.append(layer.self_attn.k_proj.register_forward_hook(k_hook))
    return handles, len(layers)


def _register_hooks(model, store: dict, n_q_keep: int):
    """Structural dispatch: probe for the attention wiring, not model_type."""
    from bmx.cache.streaming import resolve_decoder_layers

    if hasattr(model, "transformer") and hasattr(model.transformer.h[0].attn, "c_attn"):
        return _register_gpt2_hooks(model, store, n_q_keep)
    try:
        layers = resolve_decoder_layers(model)
    except ValueError:
        layers = None
    if layers is not None and hasattr(layers[0].self_attn, "q_proj"):
        return _register_qkproj_hooks(model, store, n_q_keep)
    raise ValueError(
        f"unsupported architecture {model.config.model_type!r}: expected "
        "GPT-2-style packed c_attn or Llama-style q_proj/k_proj attention"
    )


def collect_cache(
    model,
    input_ids: torch.Tensor,
    n_q_keep: int = 256,
) -> dict[str, torch.Tensor]:
    """Single prefill forward; returns per-layer K/V/Q/K_pre tensors in fp16.

    Keys
    ----
    ``layer{i}.k``     — post-RoPE key,   shape (h_kv, S, d)
    ``layer{i}.v``     — post-RoPE value, shape (h_kv, S, d)
    ``layer{i}.q``     — pre-RoPE query (last n_q_keep positions), shape (h, T, d)
                         where T = min(n_q_keep, S)
    ``layer{i}.k_pre`` — pre-RoPE key,    shape (h_kv, S, d)

    All tensors are returned as fp16. Batch dim must be 1.
    """
    assert input_ids.shape[0] == 1, "Batch dim must be 1."

    store: dict[str, torch.Tensor] = {}
    handles, n_layer = _register_hooks(model, store, n_q_keep)
    try:
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()

    result: dict[str, torch.Tensor] = {}
    for i in range(n_layer):
        k_post, v_post = _get_kv_layer(outputs.past_key_values, i)
        result[f"layer{i}.k"] = k_post.squeeze(0).contiguous().to(torch.float16)
        result[f"layer{i}.v"] = v_post.squeeze(0).contiguous().to(torch.float16)
        result[f"layer{i}.q"] = store[f"layer{i}.q"]
        result[f"layer{i}.k_pre"] = store[f"layer{i}.k_pre"]
    return result


def save_cache(tensors: dict[str, torch.Tensor], path: str | Path) -> None:
    """Save cache dict to a safetensors file."""
    save_file(tensors, str(path))


def load_cache(path: str | Path) -> dict[str, torch.Tensor]:
    """Load cache dict from a safetensors file."""
    return load_file(str(path))
