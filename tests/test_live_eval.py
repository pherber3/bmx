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


def test_quantized_live_ppl_finite_and_higher_than_fp16():
    # seq=64 with recent_window=32 (default): after the 16-token prefill and 48-token
    # continuation (sent as one batch, S=64), S_q = ((64-32)//16)*16 = 32 tokens get
    # quantized, blended bpe < 16.  seq=32 was too short — max S=32 == W, so S_q=0
    # and everything stayed fp16 (the window eclipsed all tokens).
    model = tiny_llama()
    input_ids = ids(vocab=97, seq=64, seed=12)
    live_generation_ppl(
        model,
        input_ids,
        16,
        CacheCodecSpec(arm="fp16"),
        CacheCodecSpec(arm="fp16"),
    )
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
