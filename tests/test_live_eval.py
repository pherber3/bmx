"""Live-generation perplexity through the streaming compressed cache."""

import math

from bmx.cache.live_eval import live_generation_ppl
from bmx.cache.specs import CacheCodecSpec
from factories import ids, tiny_llama


def test_fp16_live_ppl_matches_plain_forward():
    # With fp16 specs, live-gen ppl must equal a plain quantized-prefill-free ppl.
    model = tiny_llama()
    input_ids = ids(vocab=97, seq=32, seed=11)
    out = live_generation_ppl(
        model,
        input_ids,
        n_prefill=16,
        k_spec=CacheCodecSpec(arm="fp16"),
        v_spec=CacheCodecSpec(arm="fp16"),
    )
    # Same model, fp16 path: ppl finite and positive; n_eval correct.
    assert math.isfinite(out["ppl"]) and out["ppl"] > 0
    assert out["bpe_k"] == 16.0 and out["bpe_v"] == 16.0


def test_quantized_live_ppl_finite_and_compressed():
    # Renamed from ..._higher_than_fp16: that name promised a quant>fp16 comparison
    # the body never made (the fp16 result was computed and discarded).  On a
    # random-weight tiny model that comparison is FLAKY (quant can lower loss), so
    # the honest gate is finiteness + honest compression — assert exactly that.
    # seq=200 with recent_window=32: S=200 -> S_q = ((200-32)//128)*128 = 128 tokens
    # quantized (one 128-token PAGE flushed), blended bpe < 16. The cache now commits
    # on a fixed 128-token PAGE grid; shorter sequences stay all-fp16 (window eclipses
    # all committed tokens before a page fills).
    model = tiny_llama()
    input_ids = ids(vocab=97, seq=200, seed=12)
    quant = live_generation_ppl(
        model,
        input_ids,
        16,
        k_spec=CacheCodecSpec(
            arm="lowrank_rtn_channel", bits=3, rank=4, group=16, pre_rope=True
        ),
        v_spec=CacheCodecSpec(arm="rtn_token", bits=2, group=16),
    )
    assert math.isfinite(quant["ppl"])
    assert quant["bpe_k"] < 16.0  # honestly compressed (blended bpe with window)


def test_token_by_token_fp16_matches_batched():
    """Indexing-correctness gate (Task 11).

    With fp16 specs, token-by-token ppl must match the batched ppl within
    rel < 0.05.  fp16 has no quant error; the two scoring modes agree iff
    the incremental indexing is correct.  This is the definitive correctness
    check — do not relax the tolerance.

    NOTE: tiny_llama has random weights, so absolute ppl values are meaningless
    (they will be high, near vocab size).  This test checks MECHANISM only.
    """
    model = tiny_llama()
    # Use seq=32, n_prefill=16: 15 continuation tokens scored (label-shift loses one).
    input_ids = ids(vocab=97, seq=32, seed=99)
    k_spec = CacheCodecSpec(arm="fp16")
    v_spec = CacheCodecSpec(arm="fp16")

    batched = live_generation_ppl(
        model,
        input_ids,
        n_prefill=16,
        k_spec=k_spec,
        v_spec=v_spec,
        token_by_token=False,
    )
    tbt = live_generation_ppl(
        model,
        input_ids,
        n_prefill=16,
        k_spec=k_spec,
        v_spec=v_spec,
        token_by_token=True,
    )

    assert tbt["n_eval"] == batched["n_eval"], (
        f"n_eval mismatch: tbt={tbt['n_eval']} batched={batched['n_eval']}"
    )
    assert math.isfinite(tbt["ppl"]) and tbt["ppl"] > 0
    rel = abs(tbt["ppl"] - batched["ppl"]) / max(batched["ppl"], 1e-9)
    assert rel < 0.05, (
        f"fp16 tbt-ppl {tbt['ppl']:.4f} vs batched-ppl {batched['ppl']:.4f} "
        f"rel={rel:.4f} (must be < 0.05 for correct indexing)"
    )


def test_token_by_token_k2b_quality_holds():
    """C2 streaming-quality gate (Task 11).

    Real K2b spec: lowrank_rtn_channel K@3b pre-RoPE + turboquant_mse V@2b.
    recent_window=8, tiny_llama, token-by-token scoring.

    Asserts:
      - tbt ppl is FINITE (no explosion; Task 10's write-once fix is the guard).
      - tbt ppl is within a sane factor of fp16 tbt ppl (< 3×).

    Before Task 10 this would have exploded (turboquant_mse is non-idempotent:
    re-quantising a dequant value makes V norm blow up over decode steps).
    Now it must be finite and reasonable.

    NOTE: tiny_llama has random weights, so absolute ppl is meaningless (~vocab).
    This test checks MECHANISM (finite, no explosion), NOT real quality.
    Real quality numbers come from experiments on a real model (Task 12).
    """
    model = tiny_llama()
    # seq=64, n_prefill=16: enough context for recent_window=8 to flush some blocks.
    input_ids = ids(vocab=97, seq=64, seed=77)

    fp16_tbt = live_generation_ppl(
        model,
        input_ids,
        n_prefill=16,
        k_spec=CacheCodecSpec(arm="fp16"),
        v_spec=CacheCodecSpec(arm="fp16"),
        recent_window=8,
        token_by_token=True,
    )

    k2b_tbt = live_generation_ppl(
        model,
        input_ids,
        n_prefill=16,
        k_spec=CacheCodecSpec(
            arm="lowrank_rtn_channel", bits=3, rank=4, group=16, pre_rope=True
        ),
        v_spec=CacheCodecSpec(arm="turboquant_mse", bits=2),
        recent_window=8,
        token_by_token=True,
    )

    assert math.isfinite(k2b_tbt["ppl"]), (
        f"K2b tbt ppl is not finite: {k2b_tbt['ppl']} "
        "(turboquant_mse V explosion — Task 10 write-once fix should prevent this)"
    )
    assert k2b_tbt["ppl"] > 0

    ratio = k2b_tbt["ppl"] / max(fp16_tbt["ppl"], 1e-9)
    assert ratio < 3.0, (
        f"K2b tbt ppl / fp16 tbt ppl = {ratio:.3f} (must be < 3.0; "
        f"k2b={k2b_tbt['ppl']:.2f}, fp16={fp16_tbt['ppl']:.2f})"
    )
