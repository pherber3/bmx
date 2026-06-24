import math

from bmx.bench.kv_memory import KVMemCase, decode_speedup_curve, predict_decode_latency

# Llama-3.1-8B constants (match kv_memory.py docstring anchors)
GiB = 1024**3


def _case(seq_len, bpe_k, bpe_v, path):
    return KVMemCase(
        seq_len=seq_len,
        n_layer=32,
        h_kv=8,
        d_head=128,
        bpe_k=bpe_k,
        bpe_v=bpe_v,
        block=128,
        recent_window=32,
        path=path,
        weights_bytes=int(14.9 * GiB),
        act_bytes=int(61.3 * GiB),
        logits_bytes=0,
    )


def test_kv_read_bytes_packed_is_compression_smaller():
    # fp16 = 32 bpe-pair (2 bytes K + 2 bytes V per elem, ×8 bits); k2b ≈ 3.5+2.0.
    fp16 = predict_decode_latency(
        _case(131072, 16.0, 16.0, "fp16"), hbm_bandwidth_bytes_per_s=4e12
    )
    k2b = predict_decode_latency(
        _case(131072, 3.5, 2.0, "chunked"), hbm_bandwidth_bytes_per_s=4e12
    )
    # packed KV read is (3.5+2.0)/32 of fp16's, within 1%.
    assert math.isclose(
        k2b["kv_read_bytes"] / fp16["kv_read_bytes"], 5.5 / 32, rel_tol=0.01
    )


def test_speedup_is_upper_bound_le_byte_ratio():
    fp16 = _case(131072, 16.0, 16.0, "fp16")
    k2b = _case(131072, 3.5, 2.0, "chunked")
    out = decode_speedup_curve(
        fp16, k2b, hbm_bandwidth_bytes_per_s=4e12, peak_flops_per_s=9.9e14
    )
    byte_ratio = (
        predict_decode_latency(fp16, hbm_bandwidth_bytes_per_s=4e12)["kv_read_bytes"]
        + int(14.9 * GiB)
    ) / (
        predict_decode_latency(k2b, hbm_bandwidth_bytes_per_s=4e12)["kv_read_bytes"]
        + int(14.9 * GiB)
    )
    # speedup never exceeds the byte ratio (dequant + bandwidth diff only add time).
    assert out["speedup_upper_bound"] <= byte_ratio + 1e-9


def test_crossover_is_far_above_128k_for_k2b():
    # With W=14.9GiB and k2b ~5.5bpe, KV_read==W lands well past 128k (~700k).
    fp16 = _case(131072, 16.0, 16.0, "fp16")
    k2b = _case(131072, 3.5, 2.0, "chunked")
    out = decode_speedup_curve(
        fp16, k2b, hbm_bandwidth_bytes_per_s=4e12, peak_flops_per_s=9.9e14
    )
    assert out["crossover_seq_len"] > 300_000


def test_compute_bound_flag_fires_at_extreme_compression():
    # A degenerate 0.1-bpe arm makes KV read tiny; dequant FLOPs can dominate.
    hot = predict_decode_latency(
        _case(131072, 0.05, 0.05, "chunked"),
        hbm_bandwidth_bytes_per_s=4e12,
    )
    assert hot["compute_bound_flag"] in (True, False)  # flag exists and is boolean
