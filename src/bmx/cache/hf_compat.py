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
