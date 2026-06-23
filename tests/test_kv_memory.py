"""Byte-ledger: validate against MEASURED anchors from
docs/2026-06-21-niah-longbench-frontier-results.md §"128k"
(Llama-3.1-8B, GH200 94.5 GiB ceiling).

Unit convention: the results doc writes "92.2 GB", "16 GB", "94.5 GB" as
GiB-rounded values (16 GiB = 17.18 decimal GB rounded to "16"). We use GiB
throughout — GiB = 1024**3. GB (decimal 1e9) is used only to convert the
doc's decimal-GB anchor numbers for the derivation comment.
"""

from bmx.bench.kv_memory import KVMemCase, predict_peak

GiB = 1024**3
GB = 1e9  # decimal; used only for the act derivation below


def _llama31(seq_len, path, bpe_k=16.0, bpe_v=16.0):
    # Shared base derived from the MEASURED fp16 anchor (results doc, 128k):
    #   fp16 @128k = 92.2 GiB = weights 14.9 GiB + one fp16 KV copy 16.0 GiB + act 61.3 GiB
    # logits_to_keep=1 => logits ~1 position (negligible), folded into act.
    # One act_bytes for ALL paths — do not tune per-path.
    #   act = 92.2 - 14.9 - 16.0 = 61.3 GiB
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
        logits_bytes=0,  # folded into act post logits_to_keep=1
    )


def test_one_fp16_copy_is_16gib_at_128k():
    # 2 * L * h_kv * S * d * 2 bytes = one full K+V fp16 copy.
    # = 2*32*8*131072*128*2 = 17,179,869,184 bytes = 16 GiB exactly.
    # The doc says "16 GB" meaning 16 GiB (= 17.18 decimal GB, rounded).
    one_copy = 2 * 32 * 8 * 131072 * 128 * 2
    assert one_copy == 16 * GiB
    # Decimal check: confirms the doc's rounding (16 GiB ≈ 17.18 GB decimal).
    assert abs(one_copy / GB - 17.18) < 0.1


def test_fp16_path_reproduces_92gb_anchor():
    # MEASURED: fp16 path = 92.2 GiB at 128k on GH200 (results doc).
    # Model: W + C + A = 14.9 + 16.0 + 61.3 = 92.2 GiB.
    r = predict_peak(_llama31(131072, "fp16"))
    assert abs(r["predicted_peak_bytes"] / GiB - 92.2) < 3.0, r


def test_dense_stream_reproduces_99_100_oom():
    # MEASURED: dense_stream OOM at 99-100 GiB on GH200 (results doc,
    # "+~7 GB past ceiling"). Model: fp16 baseline + dense_overhead (7.5 GiB).
    r = predict_peak(_llama31(131072, "dense_stream"))
    peak_gib = r["predicted_peak_bytes"] / GiB
    assert 98.0 <= peak_gib <= 101.0, peak_gib  # measured 99-100 GiB OOM band


def test_chunked_clears_ceiling_with_margin():
    # chunked path: W + packed + window + A ≈ 78.7 GiB — well under 94.5.
    r = predict_peak(_llama31(131072, "chunked", bpe_k=3.0, bpe_v=2.0))
    assert r["predicted_peak_bytes"] / GiB < 90.0, r  # well under 94.5 GiB
    assert r["compression_at_runtime"] > 1.0


def test_chunked_resident_includes_window():
    # chunked resident = packed bpe footprint + fp16 recent window.
    # (Restored: the K3 streaming recipe keeps a recent fp16 window;
    #  small ~4 MiB at recent_window=32 but architecturally real.)
    r = predict_peak(_llama31(131072, "chunked", bpe_k=3.0, bpe_v=2.0))
    packed = int(32 * 8 * 131072 * 128 * (3.0 + 2.0) / 8)
    window = 2 * 32 * 8 * 32 * 128 * 2  # K+V, recent_window=32 tokens, fp16
    assert r["resident_bytes"] == packed + window
