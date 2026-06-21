import torch

from bmx.cache.niah import (
    build_niah_ids_synthetic,
    generate_through_cache,
    niah_recall_argmax,
    rouge1_recall,
    rouge1_recall_only,
)
from bmx.cache.specs import CacheCodecSpec
from factories import tiny_llama

_INDENTED = "        return x"


class _IndentedStubTokenizer:
    """Always decodes to an indented string regardless of token ids (strip tests)."""

    def decode(self, ids, skip_special_tokens=True):
        return _INDENTED


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


def test_rouge1_recall_only_ignores_verbosity():
    # The needle, retrieved verbatim but buried in chatter: F-measure drops (precision), but
    # recall stays ~10 because every needle word is present.
    needle = "eat a sandwich in Dolores Park"
    verbose = "Eat a sandwich in Dolores Park. Note: this is a humorous answer about the book."
    assert rouge1_recall_only(needle, verbose) == 10.0  # all needle words present
    assert (
        rouge1_recall(needle, verbose) < 8.0
    )  # F-measure penalized by the extra words


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


# --- Regression: I1 strip=False preserves leading indentation for LongBench code_sim ---


def test_generate_through_cache_strip_false_preserves_leading_whitespace():
    """strip=False must NOT remove leading spaces — required for LongBench code_sim fidelity.

    code_sim is whitespace-sensitive on indentation; .strip() removes it before scoring,
    which systematically depresses code_sim on indented completions (the common case for
    lcc/repobench-p). Measured: indent=8 → LongBench 1.000 vs our-stripped 0.820.
    """
    model = tiny_llama()
    g = torch.Generator().manual_seed(1)
    prompt_ids = torch.randint(0, 97, (1, 24), generator=g)
    fp16 = CacheCodecSpec(arm="fp16")

    result = generate_through_cache(
        model,
        tokenizer=_IndentedStubTokenizer(),
        prompt_ids=prompt_ids,
        n_prefill=12,
        k_spec=fp16,
        v_spec=fp16,
        max_new_tokens=4,
        strip=False,
    )
    assert result.startswith("    "), (
        f"strip=False must preserve leading whitespace; got {result!r}"
    )


def test_generate_through_cache_strip_true_removes_leading_whitespace():
    """strip=True (default) must strip leading spaces — NIAH ROUGE-1 path is unchanged."""
    model = tiny_llama()
    g = torch.Generator().manual_seed(2)
    prompt_ids = torch.randint(0, 97, (1, 24), generator=g)
    fp16 = CacheCodecSpec(arm="fp16")

    result = generate_through_cache(
        model,
        tokenizer=_IndentedStubTokenizer(),
        prompt_ids=prompt_ids,
        n_prefill=12,
        k_spec=fp16,
        v_spec=fp16,
        max_new_tokens=4,
        strip=True,
    )
    assert not result.startswith(" "), (
        f"strip=True must remove leading whitespace; got {result!r}"
    )
