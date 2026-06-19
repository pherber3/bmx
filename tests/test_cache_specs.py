"""CacheCodecSpec lives in cache.specs and is re-exported from ppl_eval."""

from bmx.cache.ppl_eval import CacheCodecSpec as SpecFromPpl
from bmx.cache.specs import CacheCodecSpec


def test_spec_defaults():
    s = CacheCodecSpec()
    assert (s.arm, s.bits, s.rank, s.group, s.seed, s.pre_rope) == (
        "fp16",
        3,
        0,
        64,
        0,
        False,
    )


def test_ppl_eval_reexports_same_class():
    assert SpecFromPpl is CacheCodecSpec
