"""HF model/config introspection helpers, model-family agnostic. Layer-0:
imports nothing from bmx.cache — collect.py and rope.py depend on this,
breaking the old upward import into streaming.py."""


def resolve_text_config(model_config):
    """Return the text/LM sub-config, unwrapping multimodal wrappers.

    Qwen3.5 / Gemma4 are ``*ForConditionalGeneration`` models whose head counts
    (num_attention_heads, num_key_value_heads, head_dim, hidden_size) live under
    ``config.text_config``, not at the top level. Llama-family configs have those
    attrs directly. Probe for ``text_config`` and unwrap when present so the cache
    reads the right head geometry on either family.
    """
    tc = getattr(model_config, "text_config", None)
    # A real text config has the head attrs; guard against an unrelated attr.
    if tc is not None and hasattr(tc, "num_attention_heads"):
        return tc
    return model_config


def resolve_vocab_size(model_config) -> int:
    """Vocabulary size, unwrapping multimodal wrappers.

    Gemma4 / Qwen3.5 ``*Config`` put ``vocab_size`` under ``text_config``; Llama-family
    configs have it at the top level. Prefer the text config, fall back to top-level.
    """
    tc = resolve_text_config(model_config)
    return getattr(tc, "vocab_size", None) or model_config.vocab_size


def resolve_decoder_layers(model):
    """Return the list of decoder layers, across Llama / GPT-2 / multimodal nestings.

    Layout probed (most-nested first): ``model.model.language_model.layers``
    (Qwen3.5/Gemma4 multimodal), ``model.model.layers`` (Llama-family),
    ``model.transformer.h`` (GPT-2).
    """
    inner = getattr(model, "model", None)
    if inner is not None:
        lm = getattr(inner, "language_model", None)
        if lm is not None and hasattr(lm, "layers"):
            return lm.layers
        if hasattr(inner, "layers"):
            return inner.layers
    tr = getattr(model, "transformer", None)
    if tr is not None and hasattr(tr, "h"):
        return tr.h
    raise ValueError(
        f"Cannot locate decoder layers for {type(model).__name__}. Expected "
        "model.model.language_model.layers, model.model.layers, or model.transformer.h."
    )


def model_config_n_layers(model) -> int:
    """Number of transformer layers in model (structural probe, not model_type)."""
    return len(resolve_decoder_layers(model))


def resolve_qk_capture_modules(self_attn):
    """(q_module, k_module) whose forward OUTPUT is the pre-RoPE query/key.

    Llama-family attention is {q,k}_proj -> reshape heads -> RoPE, so the
    projection output IS the pre-RoPE tensor. Qwen3/Gemma3-style attention
    inserts a per-head RMSNorm between projection and RoPE
    (k_proj -> view(b, S, h, d) -> k_norm -> RoPE): capturing at k_proj there
    breaks the k == RoPE(k_pre) identity every rope-at-read consumer relies
    on, and q_proj output is not the query attention actually uses (W-moment
    statistics must see the q_norm output). Probe structurally: prefer
    {q,k}_norm when both are present, else the plain projections. The norm
    modules emit the already-headed (b, S, h, d) shape; collect.reshape_heads
    covers both layouts with one numel-equal reshape.
    """
    has_q, has_k = hasattr(self_attn, "q_norm"), hasattr(self_attn, "k_norm")
    # Half-normed attention would make the fallthrough capture un-normed keys
    # silently — fail loud instead (no known architecture does this today).
    assert has_q == has_k, (
        f"attention has {'q_norm' if has_q else 'k_norm'} but not its twin; "
        "capture point ambiguous — extend resolve_qk_capture_modules"
    )
    if has_q:
        return self_attn.q_norm, self_attn.k_norm
    return self_attn.q_proj, self_attn.k_proj
