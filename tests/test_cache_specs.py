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


def test_spec_pack_fields_default_inert():
    s = CacheCodecSpec(arm="rtn_channel", bits=3)
    assert s.pack_path == "" and s.budget == 0.0
