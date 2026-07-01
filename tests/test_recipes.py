"""Pin the named-recipe registry: arm string -> (k_spec, v_spec)."""

import pytest

from bmx.cache.recipes import spec_pair
from bmx.cache.specs import CacheCodecSpec


def test_fp16_pair():
    k, v = spec_pair("fp16")
    assert k == CacheCodecSpec(arm="fp16") and v == CacheCodecSpec(arm="fp16")


def test_k2b_canonical():
    k, v = spec_pair("k2b", rank=16, group=64, seed=0)
    assert k == CacheCodecSpec(
        arm="lowrank_rtn_channel", bits=3, rank=16, group=64, seed=0, pre_rope=True
    )
    assert v == CacheCodecSpec(arm="turboquant_mse", bits=2, seed=0)


def test_k2b_parameterized_parsing():
    k, _ = spec_pair("k2b_k2r8", rank=16, group=64, seed=0)
    assert k.bits == 2 and k.rank == 8  # "k2b_k{bits}r{rank}" override


def test_k2b_ph_uses_perhead_v():
    _, v = spec_pair("k2b_ph", seed=0)
    assert v == CacheCodecSpec(arm="turboquant_mse_perhead", bits=2, seed=0)


def test_kivi_pair():
    k, v = spec_pair("kivi", group=64, seed=0)
    assert k.arm == "rtn_channel" and v.arm == "rtn_token"
    assert k.bits == v.bits == 2


def test_turboquant_arms_symmetric():
    for arm in ("turboquant_mse", "turboquant_prod"):
        k, v = spec_pair(arm, seed=0)
        assert k == v and k.arm == arm and k.bits == 2


def test_unknown_arm_raises():
    with pytest.raises(ValueError, match="unknown arm"):
        spec_pair("nope")


def test_census_specs_equivalence():
    # k3_kernel_census previously hand-rolled its own _specs("k2b"); pin that
    # spec_pair with defaults reproduces it exactly so the census swap is a no-op.
    k, v = spec_pair("k2b")
    assert k == CacheCodecSpec(
        arm="lowrank_rtn_channel", bits=3, rank=16, group=64, pre_rope=True
    )
    assert v == CacheCodecSpec(arm="turboquant_mse", bits=2)
