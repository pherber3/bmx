"""Analytic byte-ledger for KV-cache peak memory (Track-B-style honest bytes).

No allocation, no CUDA — arithmetic only. Grounded in the canonical KV formula
2*L*h_kv*S*d*bytes (Physics of LLM Inference; AI Systems Perf Eng).

Three paths modelled, each validated against a MEASURED anchor from
docs/2026-06-21-niah-longbench-frontier-results.md §"128k" (Llama-3.1-8B,
GH200, 94.5 GiB ceiling):

  fp16         = 92.2 GiB  (FITS) — one KV copy in-place
  dense_stream = 99–100 GiB (OOM) — fp16 baseline + codec working-set overhead
  chunked      = ~78 GiB   (fits with margin) — packed codes + fp16 window

Unit convention: the results doc writes "92.2 GB", "16 GB", "94.5 GB" but
these are GiB-rounded values (16 GiB = 17.18 decimal GB, rounded to "16").
Treat the doc's "GB" as GiB throughout.

Term model (one shared act derived once from the fp16 anchor):
  W = weights_bytes (14.9 GiB measured)
  C = one fp16 KV copy = 2*L*h_kv*S*d*2 bytes = 16 GiB at 128k
  A = act_bytes (activations + working-set, shared by all paths)
    = 92.2 - W - C = 92.2 - 14.9 - 16 = 61.3 GiB   (from fp16 anchor)
  dense_overhead = 7.5 GiB (transient codec scratch + partial reassembly;
    the "+7 GB past ceiling" measured in the results doc; NOT a full 2nd C —
    the dense_stream OOM is fp16 + overhead, not fp16 + full extra copy)

  fp16:         W + C + A           = 92.2 GiB
  dense_stream: W + C + A + T       = 92.2 + 7.5 = 99.7 GiB  (in 99-100 band)
  chunked:      W + packed + win + A ≈ 14.9 + 2.5 + 0 + 61.3 = 78.7 GiB
"""

from __future__ import annotations

from dataclasses import dataclass

# Transient codec scratch measured at "+~7 GB past ceiling" for dense_stream
# (results doc §"128k"). Named constant so the honest model is visible.
# This is codec working-set + partial reassembly at OOM moment; NOT a full
# second 16 GiB KV copy (which would overshoot to 108 GiB — not measured).
_DENSE_OVERHEAD_BYTES = int(7.5 * 1024**3)


@dataclass
class KVMemCase:
    seq_len: int
    n_layer: int
    h_kv: int
    d_head: int
    bpe_k: float
    bpe_v: float
    block: int
    recent_window: int
    path: str  # "fp16" | "dense_stream" | "chunked"
    weights_bytes: int
    act_bytes: int
    logits_bytes: int


def _one_fp16_copy_bytes(c: KVMemCase) -> int:
    # K and V, fp16 (2 bytes), all layers, all positions.
    return 2 * c.n_layer * c.h_kv * c.seq_len * c.d_head * 2


def _packed_bytes(c: KVMemCase) -> int:
    entries = c.n_layer * c.h_kv * c.seq_len * c.d_head
    return int(entries * (c.bpe_k + c.bpe_v) / 8)


def predict_peak(case: KVMemCase) -> dict:
    assert case.path in ("fp16", "dense_stream", "chunked"), case.path
    one_copy = _one_fp16_copy_bytes(case)

    # fp16 recent-window resident (K3 streaming recipe: keep last N tokens
    # in full precision). Small but architecturally real.
    window_bytes = 2 * case.n_layer * case.h_kv * case.recent_window * case.d_head * 2

    if case.path == "fp16":
        # Plain fp16 inference: ONE KV copy resident (measured 92.2 GiB anchor).
        resident = one_copy
        transient = 0
        attn = 0

    elif case.path == "dense_stream":
        # Dense streaming codec: fp16 KV copy resident + transient codec working-set.
        # Peak modelled as W + C + A + dense_overhead (NOT W + 2C + A — that would
        # overshoot to 108 GiB; the measured OOM is 99-100 GiB = fp16 + 7-8 GiB).
        # See module docstring for full term derivation.
        resident = one_copy
        transient = _DENSE_OVERHEAD_BYTES  # codec scratch + partial reassembly
        attn = 0

    else:  # chunked
        # Packed quantized codes + fp16 recent window (restored: small but real).
        resident = _packed_bytes(case) + window_bytes
        # One dequantized block of K+V (transient, freed each step).
        transient = 2 * case.n_layer * case.h_kv * case.block * case.d_head * 2
        # Online softmax: (h, block) score tile + (h, d) accumulator.
        attn = case.h_kv * (case.block + case.d_head) * 4

    predicted = (
        resident
        + transient
        + attn
        + case.weights_bytes
        + case.act_bytes
        + case.logits_bytes
    )
    # Compression vs fp16 resident baseline (one full copy).
    compression_at_runtime = one_copy / max(resident, 1)
    return {
        "resident_bytes": resident,
        "transient_bytes": transient,
        "attn_bytes": attn,
        "weights_bytes": case.weights_bytes,
        "act_bytes": case.act_bytes,
        "logits_bytes": case.logits_bytes,
        "predicted_peak_bytes": predicted,
        "compression_at_runtime": compression_at_runtime,
    }
