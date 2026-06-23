"""Byte-ledger: validate against hand-computed Llama-3.1-8B KV numbers."""

from bmx.bench.kv_memory import KVMemCase, predict_peak

GiB = 1024**3


def _llama31(seq_len, path, bpe_k=16.0, bpe_v=16.0):
    # Llama-3.1-8B: L=32, h_kv=8, d=128. Weights ~14.9 GB, rest working set (weights+kv+act+gradients).
    # fp16 measured baseline ~92 GB at 128k: KV resident (2 copies) ~32 GB, so act_bytes ≈ 47 GB
    # to account for model working memory, attention buffers, other transients during inference.
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
        act_bytes=int(47.0 * GiB),
        logits_bytes=int(0.5 * GiB),
    )


def test_resident_one_fp16_copy_is_16gb_at_128k():
    # 2 * L * h_kv * S * d * 2 bytes = one full K+V fp16 copy.
    # = 2*32*8*131072*128*2 = 17,179,869,184 bytes = 16 GiB.
    case = _llama31(131072, "dense_stream")
    r = predict_peak(case)
    one_copy = 2 * 32 * 8 * 131072 * 128 * 2
    assert one_copy == 16 * GiB
    # dense_stream holds ~2 copies (dequant prefix + reassembled slab).
    assert r["resident_bytes"] == 2 * one_copy


def test_chunked_resident_is_bpe_footprint():
    # 3-bit K, 2-bit V (k2b-ish): resident = L*h_kv*S*d*(bpe_k+bpe_v)/8.
    case = _llama31(131072, "chunked", bpe_k=3.0, bpe_v=2.0)
    r = predict_peak(case)
    expected = int(32 * 8 * 131072 * 128 * (3.0 + 2.0) / 8)
    assert r["resident_bytes"] == expected


def test_dense_stream_128k_reproduces_oom():
    # dense_stream peak should exceed the 94.5 GB GH200 ceiling at 128k.
    case = _llama31(131072, "dense_stream")
    r = predict_peak(case)
    assert r["predicted_peak_bytes"] > 94.5 * GiB


def test_chunked_128k_clears_ceiling():
    case = _llama31(131072, "chunked", bpe_k=3.0, bpe_v=2.0)
    r = predict_peak(case)
    assert r["predicted_peak_bytes"] < 94.5 * GiB
    assert r["compression_at_runtime"] > 1.0
