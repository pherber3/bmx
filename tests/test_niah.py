from bmx.cache.niah import build_niah_ids_synthetic, niah_recall_argmax
from bmx.cache.specs import CacheCodecSpec
from factories import tiny_llama


def test_build_niah_ids_shape_and_plant():
    ids = build_niah_ids_synthetic(
        vocab=97, n_context=40, depth_frac=0.5, answer_id=7, seed=3
    )
    assert ids.shape == (1, 40)
    # answer_id is planted somewhere in the interior (the needle).
    assert (ids[0] == 7).any()


def test_niah_recall_argmax_returns_bool_fp16():
    model = tiny_llama()
    ids = build_niah_ids_synthetic(
        vocab=97, n_context=40, depth_frac=0.5, answer_id=7, seed=3
    )
    fp16 = CacheCodecSpec(arm="fp16")
    got = niah_recall_argmax(
        model, ids, query_pos=39, n_prefill=20, k_spec=fp16, v_spec=fp16, answer_id=7
    )
    assert isinstance(got, bool)
