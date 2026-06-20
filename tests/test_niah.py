from bmx.cache.niah import (
    build_niah_ids_synthetic,
    generate_through_cache,
    niah_recall_argmax,
    rouge1_recall,
)
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


def test_rouge1_recall_perfect_and_zero():
    needle = "The best thing to do in San Francisco is eat a sandwich in Dolores Park."
    assert rouge1_recall(needle, needle) == 10.0  # identical => fmeasure 1.0 * 10
    assert (
        rouge1_recall(needle, "completely unrelated zzz qqq") < 2.0
    )  # near-zero overlap


def test_rouge1_recall_partial_is_graded():
    needle = "the magic number is one two three four"
    partial = "the magic number is"
    score = rouge1_recall(needle, partial)
    assert 0.0 < score < 10.0  # graded, not binary


def test_generate_through_cache_returns_str(tmp_path):
    import torch
    from bmx.cache.specs import CacheCodecSpec
    from factories import tiny_llama

    class _StubTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return " ".join(map(str, ids.tolist()))

    model = tiny_llama()
    g = torch.Generator().manual_seed(0)
    prompt_ids = torch.randint(0, 97, (1, 24), generator=g)
    fp16 = CacheCodecSpec(arm="fp16")
    out = generate_through_cache(
        model,
        tokenizer=_StubTokenizer(),
        prompt_ids=prompt_ids,
        n_prefill=12,
        k_spec=fp16,
        v_spec=fp16,
        max_new_tokens=4,
    )
    assert isinstance(out, str)
