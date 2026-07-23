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
    pack_path : str
        File path to a fitted pack (for ``"spectral"`` arm); empty string for
        packless arms (default-inert).
    budget : float
        Quantization budget in bits (for ``"spectral"`` arm); 0.0 for packless
        arms (default-inert).
    dec_quant : str
        Decoder storage precision for the ``"spectral"`` arm (Lever 2).
        ``"fp32"`` (default) is inert -- today's byte-identical compute path.
        ``"int8"`` roundtrips the loaded pack's decoder matrix once at cache
        init (see ``spectral.int8_decoder_roundtrip``) and charges
        ``dec_bits=8`` instead of 16 in the skeptic accounting. Ignored by
        every other arm. A third mode, fp16 (``dec.half().float()``), exists
        only as a measurement arm in ``experiments/k4_dec_quant.py`` (the
        shippability check for what skeptic-v1 charges) and is deliberately
        not a streaming ``dec_quant`` value here.
    """

    arm: str = "fp16"
    bits: int = 3
    rank: int = 0
    group: int = 64
    seed: int = 0
    pre_rope: bool = False
    pack_path: str = ""
    budget: float = 0.0
    dec_quant: str = "fp32"
