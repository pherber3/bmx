"""Analytic byte-ledger for KV-cache peak memory (Track-B-style honest bytes).

No allocation, no CUDA — arithmetic only. Grounded in the canonical KV formula
2*L*h_kv*S*d*bytes (Physics of LLM Inference; AI Systems Perf Eng). Predicts the
128k peak for the current dense-stream path (should reproduce the ~99-100 GB OOM)
vs the chunked path (packed codes resident). Validated against the VM census
before the chunked prediction is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    path: str  # "dense_stream" | "chunked"
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
    assert case.path in ("dense_stream", "chunked"), case.path
    one_copy = _one_fp16_copy_bytes(case)

    if case.path == "dense_stream":
        # Dequantized frozen prefix + reassembled (prefix+tail) slab ~= 2 copies.
        resident = 2 * one_copy
        # Transient: worst-arm per-flush scratch ~ a couple of full-prefix temps.
        transient = one_copy // case.n_layer  # one layer's block-set scratch, rough
        # Stock SDPA materializes a full (h, S) score row per query (freeable).
        attn = case.h_kv * case.seq_len * 4  # fp32 scores, last query only
    else:  # chunked
        resident = _packed_bytes(case)
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
    dense_resident = 2 * one_copy
    compression_at_runtime = dense_resident / max(resident, 1)
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
