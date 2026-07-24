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
        Decoder storage precision for the ``"spectral"`` arm (Lever 2; K4
        local-levers Task 1 collapses this to one tier threshold parsed by
        ``spectral.dec_quant_threshold``). Four forms:
        ``"fp32"`` (default) is inert -- today's byte-identical compute path;
        threshold None, no int8 storage.
        ``"int8"`` roundtrips EVERY used decoder column through int8 once at
        cache init (see ``spectral.int8_decoder_roundtrip``); threshold 8 (the
        top of the standard tier grid, so this is the blanket case) and
        charges the mixed-decoder accounting (``spectral.mixed_dec_charge``)
        at ``c_int8 = c_used``.
        ``"int8_t{T}"`` (e.g. ``"int8_t5"``, 2 <= T <= 8) tier-gates: only
        columns whose allocated bits satisfy ``0 < bits <= T`` are
        int8-roundtripped; used columns above T stay fp32-as-loaded (fp16
        cost in the accounting).
        ``"int8_tl"`` (recipe suffix ``_dec8tl``) derives a PER-LAYER
        threshold at materialization from each layer pack's own certificate
        (``spectral.per_layer_tier_thresholds``, 5% bar); the applied
        threshold is recorded on ``SpectralPack.dec_tier`` and the accounting
        reads that record (never re-derives — see the 2026-07-24 map-drift
        fix). Ignored by every other arm. A fifth mode, fp16
        (``dec.half().float()``), exists only as a measurement arm in
        ``experiments/k4_dec_quant.py`` (the shippability check for what
        skeptic-v1 charges) and is deliberately not a streaming ``dec_quant``
        value here.
    payload_quant : str
        Payload tier codebook for the ``"spectral"`` arm (K4 Lloyd-gate
        design, 2026-07-25; recipe suffix ``_lq``): ``"rtn"`` (default) is
        inert -- today's uniform-step (optionally MSE-refined) RTN codebook,
        threaded to ``spectral.spectral_quantize``/``spectral_quantize_packed``
        as ``quantizer="rtn"``, byte-identical to every call before this field
        existed. ``"lloyd"`` swaps the per-tier codebook for the analytic
        Gaussian Lloyd-Max codebook (``spectral.spectral_quantize``'s
        ``quantizer="lloyd"``) -- same bits per code, same groupwise fp16
        scale accounting (bpe is identical by construction; see
        ``spectral_quantize_packed``'s docstring). Ignored by every non-spectral
        arm and by the V side (turboquant already ships Lloyd via
        ``gaussian_codebook`` — this field only ever reaches the K-side
        spectral quantize calls).
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
    payload_quant: str = "rtn"
