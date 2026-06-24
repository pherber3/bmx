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
    # Property: the byte-ratio is an upper bound on achievable latency speedup because
    # dequant compute ADDS to the packed path's time, making the real speedup strictly
    # smaller than the raw byte ratio.
    BW = 4e12
    PEAK = 9.9e14
    fp16 = _case(131072, 16.0, 16.0, "fp16")
    k2b = _case(131072, 3.5, 2.0, "chunked")
    out = decode_speedup_curve(
        fp16, k2b, hbm_bandwidth_bytes_per_s=BW, peak_flops_per_s=PEAK
    )

    # Recompute the byte ratio (upper bound) directly from the raw read bytes.
    fp16_info = predict_decode_latency(fp16, hbm_bandwidth_bytes_per_s=BW)
    k2b_info = predict_decode_latency(k2b, hbm_bandwidth_bytes_per_s=BW)
    W = int(14.9 * GiB)
    fp16_bytes = W + fp16_info["kv_read_bytes"]
    packed_bytes = W + k2b_info["kv_read_bytes"]
    byte_ratio = fp16_bytes / packed_bytes

    # 1. decode_speedup_curve reports the byte-ratio upper bound (not honest-with-dequant).
    assert math.isclose(out["speedup_upper_bound"], byte_ratio, rel_tol=1e-9)

    # 2. An honest latency speedup that includes non-zero dequant time must be STRICTLY
    #    less than the byte ratio — dequant time only adds to the packed path's cost.
    dequant_time = k2b_info["_dequant_flops"] / PEAK
    assert dequant_time > 0, "dequant_time must be positive for a packed path"
    honest_speedup = (fp16_bytes / BW) / (packed_bytes / BW + dequant_time)
    assert honest_speedup < byte_ratio, (
        f"honest_speedup {honest_speedup:.6f} must be strictly less than "
        f"byte_ratio {byte_ratio:.6f} when dequant_time={dequant_time:.3e} > 0"
    )


def test_crossover_is_far_above_128k_for_k2b():
    # With W=14.9GiB and k2b ~5.5bpe, KV_read==W lands well past 128k (~700k).
    fp16 = _case(131072, 16.0, 16.0, "fp16")
    k2b = _case(131072, 3.5, 2.0, "chunked")
    out = decode_speedup_curve(
        fp16, k2b, hbm_bandwidth_bytes_per_s=4e12, peak_flops_per_s=9.9e14
    )
    assert out["crossover_seq_len"] > 600_000


def test_compute_bound_flag_fires_at_extreme_compression():
    # Property: compute_bound_flag should fire (True) when dequant FLOPs dominate
    # bandwidth time, and stay False at normal k2b operating points.
    #
    # Extreme arm: bpe=0.05+0.05 → tiny packed KV read → bandwidth_time is small,
    # while dequant_flops is proportional to ALL entries (independent of bpe).
    # With a low peak_flops_per_s (1e12 TFLOPs/s), dequant_time swamps bandwidth_time.
    BW = 4e12
    EXTREME_PEAK = 1e12  # deliberately low: forces dequant_time >> bandwidth_time

    # Verify the flag fires at extreme compression / low peak FLOPS.
    fp16 = _case(131072, 16.0, 16.0, "fp16")
    extreme = _case(131072, 0.05, 0.05, "chunked")
    out_extreme = decode_speedup_curve(
        fp16, extreme, hbm_bandwidth_bytes_per_s=BW, peak_flops_per_s=EXTREME_PEAK
    )
    assert out_extreme["compute_bound_flag"] is True, (
        f"Expected compute_bound_flag=True at bpe=0.1 with peak_flops={EXTREME_PEAK:.0e}; "
        f"dequant_time={out_extreme['dequant_compute_time_s']:.3e}"
    )

    # Companion: flag must be False at normal k2b bpe with realistic peak FLOPS.
    k2b = _case(131072, 3.5, 2.0, "chunked")
    out_normal = decode_speedup_curve(
        fp16, k2b, hbm_bandwidth_bytes_per_s=BW, peak_flops_per_s=9.9e14
    )
    assert out_normal["compute_bound_flag"] is False, (
        f"Expected compute_bound_flag=False at k2b bpe with peak_flops=9.9e14; "
        f"dequant_time={out_normal['dequant_compute_time_s']:.3e}"
    )
