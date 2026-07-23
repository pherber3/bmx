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


def test_dec_quant_default_inert_and_dec8_recipe(tmp_path):
    from bmx.cache.recipes import spec_pair

    assert CacheCodecSpec(arm="rtn_channel", bits=3).dec_quant == "fp32"
    k, v = spec_pair("k4_b2.5_dec8", pack_path="/p/packs.safetensors")
    assert k.arm == "spectral" and k.budget == 2.5 and k.dec_quant == "int8"
    assert v.arm == "turboquant_mse" and v.bits == 2
