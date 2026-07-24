"""Named end-to-end KV-compression recipes: arm string -> (k_spec, v_spec).

The registry behind every K3 experiment's --arms option (NIAH, LongBench,
live-generation, kernel census). One definition; the parquet `arm` column is
these names.
"""

from __future__ import annotations

from bmx.cache.specs import CacheCodecSpec


def spec_pair(
    arm: str, *, rank: int = 16, group: int = 64, seed: int = 0, pack_path: str = ""
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
    if arm == "k2t" or arm.startswith("k2t_k"):
        # k2t = the improved-k2b candidate: same structure as k2b (lowrank K
        # pre-RoPE + turboquant V) but the K residual is coded with the
        # turboquant-MSE mechanism instead of per-channel RTN, and the K budget
        # drops to 2b (target ~2.5-2.8 avg measured bits). Motivation: k2b ties
        # turboquant_mse_b3 on LongBench Avg at 3.94 vs 3.21 measured bits —
        # the RTN residual is the suspected bit-waster. Parameterized variants
        # "k2t_k{bits}r{rank}" override the K budget, mirroring k2b_k{bits}r{rank}.
        # (group is inert for lowrank_turboquant; kept for spec symmetry.)
        bits_k, rank_k = 2, rank
        if arm != "k2t":
            body = arm[len("k2t_k") :]
            bits_str, rank_str = body.split("r")
            bits_k, rank_k = int(bits_str), int(rank_str)
        return (
            CacheCodecSpec(
                arm="lowrank_turboquant",
                bits=bits_k,
                rank=rank_k,
                group=group,
                seed=seed,
                pre_rope=True,
            ),
            CacheCodecSpec(arm="turboquant_mse", bits=2, seed=seed),
        )
    if arm.startswith("k4_b"):
        # k4_b{budget}: corpus-fitted spectral K via packs + proven turboquant V@2b.
        # Requires --pack-path (a fitted spectral pack file). Optional "_dec8tl",
        # "_dec8t{T}", or "_dec8" suffix (e.g. "k4_b2.5_dec8tl" / "k4_b2.5_dec8t5" /
        # "k4_b2.5_dec8") selects the int8-decoder Lever-2 variant (same spec,
        # dec_quant="int8_tl" per-layer certificate-derived, dec_quant="int8_t{T}"
        # tier-gated, or dec_quant="int8" blanket). "_dec8tl" is checked FIRST,
        # as an EXACT suffix -- "l" is not a digit, so if "_dec8t" were checked
        # first (as the digit-parse below does) "int8_t" + "l" would try
        # int("l") and crash. Then "_dec8t{T}" (a superstring-prefix of "_dec8"),
        # then the plain "_dec8" float suffix, since the budget itself may
        # contain no further "_" delimiters.
        #
        # CANONICAL SUFFIX ORDER (K4 Lloyd-gate design, 2026-07-25, pinned by
        # test_recipes.py): "_lq" (payload_quant="lloyd") sits BETWEEN the
        # budget and any "_dec8*" suffix -- "k4_b2.5_lq_dec8tl", never
        # "k4_b2.5_dec8tl_lq". The "_dec8*" family is parsed FIRST (unchanged
        # logic above, matched as a SUFFIX of the whole budget_str -- so it
        # strips correctly even with "_lq" still embedded, e.g.
        # "2.5_lq_dec8tl".endswith("_dec8tl") is True), leaving "2.5_lq"
        # behind; "_lq" is then stripped from what remains, so a lone
        # "k4_b2.5_lq" (no dec8 suffix) parses too.
        if not pack_path:
            raise ValueError(
                "k4 arms require --pack-path (a fitted spectral pack file)"
            )
        budget_str = arm[len("k4_b") :]
        dec_quant = "fp32"
        if budget_str.endswith("_dec8tl"):
            dec_quant = "int8_tl"
            budget_str = budget_str[: -len("_dec8tl")]
        elif "_dec8t" in budget_str:
            budget_str, _, t_str = budget_str.partition("_dec8t")
            dec_quant = f"int8_t{t_str}"
        elif budget_str.endswith("_dec8"):
            dec_quant = "int8"
            budget_str = budget_str[: -len("_dec8")]
        payload_quant = "rtn"
        if budget_str.endswith("_lq"):
            payload_quant = "lloyd"
            budget_str = budget_str[: -len("_lq")]
        budget = float(budget_str)
        return (
            CacheCodecSpec(
                arm="spectral",
                pre_rope=True,
                group=group,
                pack_path=pack_path,
                budget=budget,
                dec_quant=dec_quant,
                payload_quant=payload_quant,
            ),
            CacheCodecSpec(arm="turboquant_mse", bits=2, seed=seed),
        )
    if arm in ("turboquant_mse", "turboquant_prod"):
        s = CacheCodecSpec(arm=arm, bits=2, seed=seed)
        return s, s
    # Parametric bit-width variants "turboquant_mse_b{bits}" (e.g. _b3, _b4): the codec's
    # bit-width was never a design constraint — only this registry pinned it at 2. Enables
    # the matched-bits comparison against k2b (~3.94b), the test of the structure claim
    # (TurboQuant's own non-integer 2.5/3.5 rows come from outlier-splitting, a different
    # mechanism we deliberately do NOT replicate; integer bits is the clean comparison).
    if arm.startswith("turboquant_mse_b") or arm.startswith("turboquant_prod_b"):
        base, _, bits_str = arm.rpartition("_b")
        s = CacheCodecSpec(arm=base, bits=int(bits_str), seed=seed)
        return s, s
    # Asymmetric bit-width variants "turboquant_mse_k{kb}v{vb}" (e.g. _k3v2): K and V
    # sensitivity differ (bits belong to K), so turboquant's OWN best frontier point is
    # asymmetric — the honest baseline any new codec must beat, not the symmetric strawman.
    if arm.startswith("turboquant_mse_k") or arm.startswith("turboquant_prod_k"):
        base, _, body = arm.rpartition("_k")
        bits_k_str, _, bits_v_str = body.partition("v")
        return (
            CacheCodecSpec(arm=base, bits=int(bits_k_str), seed=seed),
            CacheCodecSpec(arm=base, bits=int(bits_v_str), seed=seed),
        )
    if arm == "kivi":
        return (
            CacheCodecSpec(arm="rtn_channel", bits=2, group=group, seed=seed),
            CacheCodecSpec(arm="rtn_token", bits=2, group=group, seed=seed),
        )
    raise ValueError(f"unknown arm {arm!r}")
