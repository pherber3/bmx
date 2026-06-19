"""Live streaming KV cache that quantizes on append (K2c recipe made live).

Mirrors transformers' QuantizedCache/QuantizedLayer split: a per-layer
DynamicLayer subclass (StreamingQuantizedLayer) that stores only the compressed
representation and RETURNS dequantized K/V from update() for attention, plus a
thin Cache container (StreamingQuantizedCache) that replicates the layer across
the model. Because the layer never persists the dense dequant, resident state is
the compressed footprint — real memory by the official cache contract.

Lands in two stages:
  Stage A (this commit): fp16 passthrough + the layer/container plumbing. With an
    fp16 spec the layer delegates to DynamicLayer — gate is bit-identical logits
    and generation vs a plain default cache.
  Stage B (Task 3): the quantize-on-append path — pre-RoPE K capture + frozen
    subspace, RoPE-at-read, codec-driven _quantize/_dequantize.
"""

from __future__ import annotations

from transformers.cache_utils import Cache, DynamicLayer

from bmx.cache.specs import CacheCodecSpec


class StreamingQuantizedLayer(DynamicLayer):
    """Per-layer streaming-quantized cache. Passthrough until Task 3 adds the codec.

    Parameters
    ----------
    k_spec, v_spec : CacheCodecSpec
        Codec specs for keys and values. ``arm="fp16"`` => passthrough that side.
    model_config :
        HF model config (RoPE tables + head counts, used by the Task 3 codec).
    recent_window : int
        Most-recent tokens kept fp16 before flushing to quantized state (Task 3).
    """

    def __init__(self, k_spec, v_spec, model_config, recent_window: int = 32):
        super().__init__()
        self.k_spec = k_spec
        self.v_spec = v_spec
        self.model_config = model_config
        self.recent_window = recent_window

    def _is_passthrough(self) -> bool:
        return self.k_spec.arm == "fp16" and self.v_spec.arm == "fp16"

    def update(self, key_states, value_states, *args, **kwargs):
        # Stage A: passthrough delegates to DynamicLayer.update (concat + return
        # full keys/values). Task 3 branches here when not passthrough.
        return super().update(key_states, value_states, *args, **kwargs)


class StreamingQuantizedCache(Cache):
    """Cache container replicating StreamingQuantizedLayer across the model.

    Drop-in ``past_key_values=`` for model() / model.generate().
    """

    def __init__(
        self,
        model_config,
        k_spec: CacheCodecSpec,
        v_spec: CacheCodecSpec,
        recent_window: int = 32,
    ):
        # layer_class_to_replicate lazily appends one layer per new layer_idx.
        super().__init__(
            layer_class_to_replicate=lambda: StreamingQuantizedLayer(
                k_spec, v_spec, model_config, recent_window
            )
        )
        self.model_config = model_config
        self.k_spec = k_spec
        self.v_spec = v_spec
        self.recent_window = recent_window
