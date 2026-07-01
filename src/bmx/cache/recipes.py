"""Named end-to-end KV-compression recipes: arm string -> (k_spec, v_spec).

The registry behind every K3 experiment's --arms option (NIAH, LongBench,
live-generation, kernel census). One definition; the parquet `arm` column is
these names.
"""

from __future__ import annotations

from bmx.cache.specs import CacheCodecSpec


def spec_pair(
    arm: str, *, rank: int = 16, group: int = 64, seed: int = 0
) -> tuple[CacheCodecSpec, CacheCodecSpec]:
    """(k_spec, v_spec) for a named arm.

    K2b = lowrank K@3b pre-RoPE + rotate/Lloyd V@2b (the quality-first recipe; spends
    bits on keys, so it lands LOWER on compression than turboquant). For an apples-to-
    apples comparison at turboquant's compression, the ``k2b_kNbM`` arms drop the key
    budget to N bits / rank M: ``k2b_k2r8`` lands at ~7.2x (matched to turboquant_mse's
    7.9x and kivi's 7.1x), so quality differences there are at equal bits, not bought
    with extra storage. See the local bpe table in the session notes.
    """
    if arm == "fp16":
        return CacheCodecSpec(arm="fp16"), CacheCodecSpec(arm="fp16")
    # k2b_ph = canonical k2b but with the PER-HEAD Hadamard V codec
    # (turboquant_mse_perhead). Quality-equivalent to k2b (full-C V) and the arm the
    # fused k2b decode kernel runs — use it with --use-packed on CUDA to exercise +
    # regression-check the fused kernel against the recorded k2b results.
    if arm == "k2b_ph":
        return (
            CacheCodecSpec(
                arm="lowrank_rtn_channel",
                bits=3,
                rank=rank,
                group=group,
                seed=seed,
                pre_rope=True,
            ),
            CacheCodecSpec(arm="turboquant_mse_perhead", bits=2, seed=seed),
        )
    if arm == "k2b" or arm.startswith("k2b_k"):
        # Default canonical k2b: keys@3b, rank as passed. Parameterized variants
        # "k2b_k{bits}r{rank}" override the key budget to match compression.
        bits_k, rank_k = 3, rank
        if arm != "k2b":
            # Parse "k2b_k2r8" -> bits_k=2, rank=8.
            body = arm[len("k2b_k") :]
            bits_str, rank_str = body.split("r")
            bits_k, rank_k = int(bits_str), int(rank_str)
        return (
            CacheCodecSpec(
                arm="lowrank_rtn_channel",
                bits=bits_k,
                rank=rank_k,
                group=group,
                seed=seed,
                pre_rope=True,
            ),
            CacheCodecSpec(arm="turboquant_mse", bits=2, seed=seed),
        )
    if arm in ("turboquant_mse", "turboquant_prod"):
        s = CacheCodecSpec(arm=arm, bits=2, seed=seed)
        return s, s
    if arm == "kivi":
        return (
            CacheCodecSpec(arm="rtn_channel", bits=2, group=group, seed=seed),
            CacheCodecSpec(arm="rtn_token", bits=2, group=group, seed=seed),
        )
    raise ValueError(f"unknown arm {arm!r}")
