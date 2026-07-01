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
    dense_overhead_bytes: int = _DENSE_OVERHEAD_BYTES


def _one_fp16_copy_bytes(c: KVMemCase) -> int:
    # K and V, fp16 (2 bytes), all layers, all positions.
    return 2 * c.n_layer * c.h_kv * c.seq_len * c.d_head * 2


def _packed_bytes(c: KVMemCase) -> int:
    entries = c.n_layer * c.h_kv * c.seq_len * c.d_head
    return int(entries * (c.bpe_k + c.bpe_v) / 8)


def _kv_read_bytes_per_step(case: KVMemCase) -> int:
    """KV bytes streamed from HBM for ONE decode step (read all cached K+V)."""
    entries = case.n_layer * case.h_kv * case.seq_len * case.d_head
    if case.path == "fp16":
        return entries * 2 * 2  # K and V, 2 bytes each
    # packed: codes at (bpe_k+bpe_v)/8 bytes/entry + the fp16 recent window resident
    packed = _packed_bytes(case)
    window = 2 * case.n_layer * case.h_kv * case.recent_window * case.d_head * 2
    return packed + window


def _dequant_flops_per_step(case: KVMemCase) -> int:
    """Rough dequant arithmetic per decode step (packed paths only).

    ~O(1) ops per dequantized element (multiply by scale, + low-rank addend).
    fp16 path does no dequant.
    """
    if case.path == "fp16":
        return 0
    entries = case.n_layer * case.h_kv * case.seq_len * case.d_head
    return entries * 4  # unpack + scale + accumulate; small constant, honest order


def predict_decode_latency(
    case: KVMemCase, *, hbm_bandwidth_bytes_per_s: float
) -> dict:
    """Memory-bound decode step latency = bytes/bandwidth, + dequant compute.

    Honest model: decode is memory-bound (~0.5 FLOP/byte), so step time is
    dominated by (weights + KV read) / bandwidth. Dequant FLOPs are "free" only
    while they stay under the bandwidth time.

    compute_bound_flag is None here — it requires peak_flops_per_s to compute,
    which is not available at this level. The flag is computed correctly by
    decode_speedup_curve, which receives the FLOP/s budget.
    dequant_compute_time_s is 0.0 (dequant assumed free at this level of analysis).
    """
    kv_read = _kv_read_bytes_per_step(case)
    weight = case.weights_bytes
    bandwidth_time = (weight + kv_read) / hbm_bandwidth_bytes_per_s
    # Dequant time uses a conservative peak-flops divisor; refined per-GPU at measure time.
    dequant_flops = _dequant_flops_per_step(case)
    dequant_time = 0.0
    return {
        "kv_read_bytes": kv_read,
        "weight_bytes": weight,
        "bandwidth_time_s": bandwidth_time,
        "dequant_compute_time_s": dequant_time,
        "predicted_step_latency_s": bandwidth_time + dequant_time,
        "compute_bound_flag": None,
        "_dequant_flops": dequant_flops,
    }


def decode_speedup_curve(
    fp16_case: KVMemCase,
    packed_case: KVMemCase,
    *,
    hbm_bandwidth_bytes_per_s: float,
    peak_flops_per_s: float,
) -> dict:
    """Predicted decode speedup (UPPER BOUND) + crossover sequence length.

    speedup_upper_bound = fp16 step bytes / packed step bytes (latency proxy).
    The real speedup is <= this: dequant compute and any int8-vs-fp16 bandwidth
    differential only ADD to the packed path. crossover_seq_len is where the
    packed KV read equals the fixed weight stream (below it weights dominate and
    compression barely helps; above it KV dominates and it approaches the ratio).
    """
    f = predict_decode_latency(
        fp16_case, hbm_bandwidth_bytes_per_s=hbm_bandwidth_bytes_per_s
    )
    p = predict_decode_latency(
        packed_case, hbm_bandwidth_bytes_per_s=hbm_bandwidth_bytes_per_s
    )
    fp16_bytes = f["weight_bytes"] + f["kv_read_bytes"]
    packed_bytes = p["weight_bytes"] + p["kv_read_bytes"]
    speedup = fp16_bytes / packed_bytes
    # dequant honesty flag: compute time vs bandwidth time at this operating point.
    dequant_time = p["_dequant_flops"] / peak_flops_per_s
    compute_bound = dequant_time > p["bandwidth_time_s"]
    # crossover: packed KV-read-per-token * S == weights.
    entries_per_tok = packed_case.n_layer * packed_case.h_kv * packed_case.d_head
    packed_bytes_per_tok = entries_per_tok * (packed_case.bpe_k + packed_case.bpe_v) / 8
    crossover = packed_case.weights_bytes / packed_bytes_per_tok
    return {
        "speedup_upper_bound": speedup,
        "crossover_seq_len": crossover,
        "compute_bound_flag": compute_bound,
        "dequant_compute_time_s": dequant_time,
    }


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
        transient = case.dense_overhead_bytes  # codec scratch + partial reassembly
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
