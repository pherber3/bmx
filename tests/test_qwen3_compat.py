"""Second-model-family (Qwen3) compatibility: hf_compat resolution, the qk-norm
pre-RoPE capture point, streaming attach, spectral packs, EOS-list stop.
Everything offline via tiny_qwen3."""

from bmx.cache.hf_compat import (
    model_config_n_layers,
    resolve_decoder_layers,
    resolve_text_config,
    resolve_vocab_size,
)
from tests.factories import tiny_qwen3


def test_hf_compat_resolves_qwen3():
    m = tiny_qwen3()
    assert model_config_n_layers(m) == 2
    layers = resolve_decoder_layers(m)
    sa = layers[0].self_attn
    assert hasattr(sa, "q_proj") and hasattr(sa, "k_proj")
    assert hasattr(sa, "q_norm") and hasattr(sa, "k_norm")  # the qk-norm family marker
    tc = resolve_text_config(m.config)
    assert tc is m.config  # no multimodal wrapper on Qwen3ForCausalLM
    assert tc.num_key_value_heads == 2
    d = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    assert d == 8  # explicit head_dim; Qwen3Config's default (128) must not leak in
    assert resolve_vocab_size(m.config) == 97
