"""Shared codec specification for one side (K or V) of the KV cache.

Lifted out of ppl_eval so both ppl_eval and the streaming cache can import it
without a cycle. Single source of truth for the spec dataclass.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class CacheCodecSpec:
    """Codec specification for one side (K or V) of the KV cache.

    Attributes
    ----------
    arm : str
        A member of bmx.cache.codecs.CACHE_ARMS, or ``"fp16"`` for a no-op.
    bits : int
        Quantization bit width.
    rank : int
        Low-rank components for ``lowrank_rtn_channel`` (ignored otherwise).
    group : int
        Group size for rtn_token / rtn_channel / rotate_rtn_token / lowrank arms.
    seed : int
        RNG seed for rotation/sketch arms.
    pre_rope : bool
        If True, quantize keys in pre-RoPE space, then apply_rope before use.
        Ignored for V (V has no RoPE in standard transformer families).
    """

    arm: str = "fp16"
    bits: int = 3
    rank: int = 0
    group: int = 64
    seed: int = 0
    pre_rope: bool = False
