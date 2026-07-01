"""CacheCodecSpec defaults (the codec-spec contract every arm builds on)."""

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
