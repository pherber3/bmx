"""K4 Lever-2 gate: decoder-precision distortion measurement on EXISTING packs.

Skeptic-v1 charges a full C×C fp16 decoder matrix per sequence
(`skeptic_charge`'s default). But the decoder that ships with the codec is
never actually re-derived at fp16 precision from a higher-precision source —
the pack's `dec` tensor IS the fp32 eigenbasis, stored as-is. This harness
asks: if the decoder were ACTUALLY stored/read at fp16 (`dec.half().float()`)
or int8 (`int8_decoder_roundtrip`), how much quality does that cost, relative
to the as-fit fp32 decoder?

Refits nothing: packs are loaded via `load_packs` from an existing pack file
(`k4_fit_packs.py`'s output). Per (cache, layer, budget), the SAME fitted
`SpectralPack.dec` is precision-degraded and re-scored via `spectral_quantize`
on the tail region — mirroring `k4_frontier.py`'s cache-loading/RoPE/scoring
structure and `headline_col` selection.

Win definition (isolates pure decoder-precision damage from the bits question,
which is reported separately and never gated on): per (cache, layer, budget),
win = tq_interp(turboquant_mse k_pre curve, AT THE FP16 ARM's bpe_skeptic_deploy)
/ dist_of_that_mode — the SAME bpe point is used for all three dec modes, so
only the numerator's distortion changes across modes. `int8_win_at_own_bits`
additionally reports the deployment view: win computed at int8's OWN
bpe_skeptic_deploy (dec_bits=8, c_used-accounted) — the actual bits the int8
decoder would cost in deployment, never used for the gate itself.

Acceptance (pre-registered, binding): rel_degradation_int8 =
1 − win_int8/win_fp16 < 5% relative at every budget. Rider: if
rel_degradation_fp16_vs_fp32 = 1 − win_fp16/win_fp32 > 0.5%, flag it — that
would mean skeptic-v1's fp16-decoder charge was optimistic and must be
re-examined (expected ≈ 0).

K4 local-levers Task 2 (tier-threshold sweep): `Config.tier_thresholds`
(default empty = today's behavior, byte-exact) adds `int8_t{T}` dec modes —
`int8_decoder_roundtrip(..., tier_threshold=T)` int8-stores only the used
decoder columns with `0 < bits ≤ T`, shipping the rest fp16. Deploy bpe for
every mode is now priced through `mixed_dec_charge` (endpoint-pinned to the
old `skeptic_charge` formula at fp32/fp16/int8 — zero numeric change there).
THE GATE MOVES when tier modes are present: `gate_pass` binds on
`rel_degradation_int8_t5 < 5%` at every budget (the pre-registered Lever-2
promotion criterion); the blanket `rel_degradation_int8` is still reported
but no longer gates. With `tier_thresholds=()` (no tier modes), `gate_pass`
binds on the blanket exactly as before — existing callers are unaffected.
`cert_vs_measured.parquet` cross-checks the offline certificate
(`int8_decoder_certificate_tiered`) against this harness's own measured
per-layer `1 − win_T(layer)/win_fp16(layer)` — the instrument-validation
table for the "cheap analytic instrument" pattern.
"""

from __future__ import annotations

import dataclasses
import json
import math

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.codecs import quantize_cache
from bmx.cache.spectral import (
    SpectralPack,
    int8_decoder_certificate_tiered,
    int8_decoder_roundtrip,
    load_packs,
    mixed_dec_charge,
    spectral_quantize,
)
from experiments._k4_common import (
    DEPLOY_S,
    _layer_ctx,
    _log_interp,
    _score_tail,
    _tq_layer_curve,
    load_layer_keys,
    setup_rope,
)

DEC_MODES = ("fp32", "fp16", "int8")


@dataclasses.dataclass
class Config:
    pack_path: str
    cache_paths: tuple[str, ...]  # scored caches
    model_label: str = ""
    model_name: str = ""  # HF repo id for RoPE; empty => stored-basis logit only
    budgets: tuple[float, ...] = (2.2, 2.5)
    tq_bits: tuple[int, ...] = (2, 3, 4)
    seed: int = 0
    out_root: str = ""
    # K4 local-levers Task 2: tier-gated int8 sweep. Empty (default) = today's
    # behavior byte-exact (only fp32/fp16/int8 blanket modes run, blanket
    # gates). Non-empty adds int8_t{T} modes for each T and moves the binding
    # gate to int8_t5 (see module docstring).
    tier_thresholds: tuple[int, ...] = ()


def _dec_modes(cfg: Config) -> tuple[str, ...]:
    return DEC_MODES + tuple(f"int8_t{t}" for t in cfg.tier_thresholds)


def _dec_variant(dec: torch.Tensor, bits_pc: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "fp32":
        return dec
    if mode == "fp16":
        return dec.half().float()
    if mode == "int8":
        return int8_decoder_roundtrip(dec, bits_pc)
    if mode.startswith("int8_t"):
        t = int(mode[len("int8_t") :])
        return int8_decoder_roundtrip(dec, bits_pc, tier_threshold=t)
    raise AssertionError(f"unknown dec_mode {mode!r}")


def _mode_c_int8(pack: SpectralPack, mode: str) -> int:
    """Number of used decoder columns int8-stored under `mode` — 0 for
    fp32/fp16, every used column (thr=8) for blanket "int8", and
    `count(0 < bits <= T)` for "int8_t{T}" (via `SpectralPack.c_int8`, the
    shared int8-column count the streaming charge/certificate also use)."""
    if mode in ("fp32", "fp16"):
        return 0
    thr = 8 if mode == "int8" else int(mode[len("int8_t") :])
    return pack.c_int8(thr)


def main(cfg: Config):
    assert cfg.cache_paths, "cache_paths must be non-empty"

    run = (
        create_run("k4_dec_quant", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_dec_quant", cfg)
    )

    model_label = cfg.model_label or "unknown"
    dec_modes = _dec_modes(cfg)
    rows: list[dict] = []  # dec-mode arms -> metrics.parquet
    tq_rows: list[dict] = []  # turboquant_mse baseline curve, verdict-internal only
    headline_col = "logit"  # refined per-cache below (logit_rope when RoPE ready)
    any_rope_ready = False
    # (cache, budget) -> {layer: pack} — retained for the cert_vs_measured
    # table (int8_decoder_certificate_tiered needs the actual SpectralPack,
    # not just the metrics rows).
    packs_by_cache_budget: dict[tuple[str, float], dict[int, SpectralPack]] = {}

    def emit(row: dict, *, dest: list[dict]) -> None:
        dest.append(row)
        budget_str = f"{row['budget']:.2f}" if not math.isnan(row["budget"]) else " n/a"
        print(
            f"  cache={row['cache']:12s} layer={row['layer']:2d} "
            f"arm={row['arm']:14s} dec_mode={row['dec_mode']:5s} budget={budget_str}  "
            f"bpe_model={row['bpe_model']:.3f}  rel_fro={row['rel_fro']:.4f}  "
            f"logit={row['logit']:.4f}  logit_rope={row['logit_rope']:.4f}",
            flush=True,
        )

    for cache_path in cfg.cache_paths:
        cache_label = cache_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        layer_keys = load_layer_keys(cache_path)
        layers = sorted(layer_keys.keys())
        rope_ready, get_cos_sin = setup_rope(cfg.model_name, layer_keys, layers)
        any_rope_ready = any_rope_ready or rope_ready

        # ---- per (cache, layer) turboquant_mse k_pre curve, computed ONCE --
        # (identical across budgets — M_pre/tail don't depend on budget, so
        # hoisting this out of the budget loop below avoids writing
        # exact-duplicate rows into tq_curve.parquet once per budget).
        for layer_i in layers:
            ctx = _layer_ctx(
                layer_keys[layer_i], rope_ready=rope_ready, get_cos_sin=get_cos_sin
            )

            for b in cfg.tq_bits:
                M_hat_tq, bpe_tq = quantize_cache(
                    "turboquant_mse", ctx.M_pre, bits=b, seed=cfg.seed
                )
                rf_tq, lg_tq, lg_rope_tq, _lg_causal_tq = _score_tail(
                    M_hat_tq,
                    ctx.h_kv,
                    ctx.tail,
                    ctx.K_post_true,
                    ctx.Q_fp32,
                    ctx.cos_l,
                    ctx.sin_l,
                    rope_ready,
                    ctx.k_pre_t,
                    ctx.M_pre,
                )
                emit(
                    dict(
                        model=model_label,
                        cache=cache_label,
                        layer=layer_i,
                        kind="k_pre",
                        arm="turboquant_mse",
                        dec_mode="",
                        budget=float("nan"),
                        bpe_model=bpe_tq,
                        bpe_skeptic_deploy=bpe_tq,
                        rel_fro=rf_tq,
                        logit=lg_tq,
                        logit_rope=lg_rope_tq,
                    ),
                    dest=tq_rows,
                )

        for budget in cfg.budgets:
            packs = load_packs(cfg.pack_path, budget)
            packs_by_cache_budget[(cache_label, float(budget))] = packs

            for layer_i in layers:
                if layer_i not in packs:
                    continue
                ctx = _layer_ctx(
                    layer_keys[layer_i], rope_ready=rope_ready, get_cos_sin=get_cos_sin
                )

                pack = packs[layer_i]
                assert pack.enc.shape == (ctx.C, ctx.C), (
                    f"pack C mismatch at layer {layer_i}: {pack.enc.shape} vs C={ctx.C}"
                )

                # ---- decoder-precision arms (fp32/fp16/int8 + int8_t{T}) -----
                for mode in dec_modes:
                    dec_variant = _dec_variant(pack.dec, pack.bits, mode)
                    pack_variant = dataclasses.replace(pack, dec=dec_variant)
                    M_hat, bpe_model = spectral_quantize(ctx.M_pre, pack_variant)
                    rf, lg, lg_rope, _lg_causal = _score_tail(
                        M_hat,
                        ctx.h_kv,
                        ctx.tail,
                        ctx.K_post_true,
                        ctx.Q_fp32,
                        ctx.cos_l,
                        ctx.sin_l,
                        rope_ready,
                        ctx.k_pre_t,
                        ctx.M_pre,
                    )
                    bpe_skeptic_deploy = bpe_model + mixed_dec_charge(
                        ctx.C,
                        DEPLOY_S,
                        pack.tiers,
                        c_used=pack.c_used,
                        c_int8=_mode_c_int8(pack, mode),
                    )
                    emit(
                        dict(
                            model=model_label,
                            cache=cache_label,
                            layer=layer_i,
                            kind="k_pre",
                            arm="spectral",
                            dec_mode=mode,
                            budget=float(budget),
                            bpe_model=bpe_model,
                            bpe_skeptic_deploy=bpe_skeptic_deploy,
                            rel_fro=rf,
                            logit=lg,
                            logit_rope=lg_rope,
                        ),
                        dest=rows,
                    )
    headline_col = "logit_rope" if any_rope_ready else "logit"

    cols = [
        "model",
        "cache",
        "layer",
        "kind",
        "arm",
        "dec_mode",
        "budget",
        "bpe_model",
        "bpe_skeptic_deploy",
        "rel_fro",
        "logit",
        "logit_rope",
    ]
    df = pd.DataFrame(rows)[cols]
    tq_df = pd.DataFrame(tq_rows)[cols]
    write_metrics(run, df)
    write_metrics(run, tq_df, name="tq_curve")

    verdict = _dec_quant_verdict(df, tq_df, headline_col, cfg)
    (run / "dec_quant_verdict.json").write_text(json.dumps(verdict, indent=2))

    # cert-vs-measured instrument-validation table: certificate
    # `implied_rel_degradation` vs this harness's own measured per-layer
    # rel_degradation, for every T in tier_thresholds plus the blanket (8).
    tier_thresholds_incl_blanket = tuple(sorted(set(cfg.tier_thresholds) | {8}))
    tq_curves = {
        cache: _tq_layer_curve(g, headline_col) for cache, g in tq_df.groupby("cache")
    }
    cache_labels = [
        cache_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        for cache_path in cfg.cache_paths
    ]
    cvm_rows: list[dict] = []
    for budget in cfg.budgets:
        packs_by_cache = {
            cache_label: packs_by_cache_budget.get((cache_label, float(budget)), {})
            for cache_label in cache_labels
        }
        cvm_rows.extend(
            _cert_vs_measured_rows(
                df=df,
                tq_curves=tq_curves,
                headline_col=headline_col,
                budget=float(budget),
                packs_by_cache=packs_by_cache,
                tier_thresholds_incl_blanket=tier_thresholds_incl_blanket,
                model_label=model_label,
            )
        )
    cvm_df = pd.DataFrame(cvm_rows)
    write_metrics(run, cvm_df, name="cert_vs_measured")

    print("\n" + "=" * 88)
    print("DEC-QUANT VERDICT — decoder-precision distortion gate (Lever 2)")
    print("=" * 88)
    print(json.dumps(verdict, indent=2))
    print(f"\nTotal rows: {len(df)}")
    print(f"-> {run}")

    return run


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def _per_layer_wins(
    sub: pd.DataFrame,
    tq_curves: dict[str, dict[int, list[tuple[float, float]]]],
    headline_col: str,
    modes: tuple[str, ...],
) -> tuple[dict[str, dict[tuple[str, int], float]], bool]:
    """Per (cache, layer) win for every mode in `modes`, ALL interpolated at
    the fp16 arm's bpe_skeptic_deploy (the shared point that isolates pure
    decoder-precision damage — see module docstring's win definition).
    Returns (wins[mode][(cache, layer)] -> win, extrapolated)."""
    fp16_sub = sub[sub.dec_mode == "fp16"].set_index(["cache", "layer"])
    wins: dict[str, dict[tuple[str, int], float]] = {m: {} for m in modes}
    extrapolated = False

    for (cache, layer), fp16_row in fp16_sub.iterrows():
        pts = tq_curves.get(cache, {}).get(int(layer))
        if not pts:
            continue
        bpe_at = float(fp16_row.bpe_skeptic_deploy)
        tq_dist, ex = _log_interp(pts, bpe_at)
        extrapolated = extrapolated or ex
        for mode in modes:
            row = sub[
                (sub.cache == cache) & (sub.layer == layer) & (sub.dec_mode == mode)
            ]
            if row.empty:
                continue
            dist = max(float(row[headline_col].iloc[0]), 1e-300)
            wins[mode][(cache, layer)] = tq_dist / dist

    return wins, extrapolated


def _ordering_ok(rel_degradation_by_mode: dict[str, float], tol: float = 1e-9) -> bool:
    """True iff rel_degradation is monotone nondecreasing across
    T4 <= T5 <= T6 <= blanket ("int8"), restricted to whichever of those
    modes are present (vacuously True if fewer than two are present)."""
    order = ["int8_t4", "int8_t5", "int8_t6", "int8"]
    present = [
        rel_degradation_by_mode[m] for m in order if m in rel_degradation_by_mode
    ]
    return all(b >= a - tol for a, b in zip(present, present[1:]))


def _cert_vs_measured_rows(
    *,
    df: pd.DataFrame,
    tq_curves: dict[str, dict[int, list[tuple[float, float]]]],
    headline_col: str,
    budget: float,
    packs_by_cache: dict[str, dict[int, SpectralPack]],
    tier_thresholds_incl_blanket: tuple[int, ...],
    model_label: str = "",
) -> list[dict]:
    """One row per (cache, layer, T) in tier_thresholds_incl_blanket: the
    offline certificate's `implied_rel_degradation` (int8_decoder_certificate_tiered)
    vs this harness's own measured per-layer
    `1 - win_T(cache, layer)/win_fp16(cache, layer)` — the instrument-validation
    cross-check (never gated)."""
    sub = df[(df.arm == "spectral") & (df.budget == float(budget))]
    if sub.empty:
        return []

    modes = tuple(
        "int8" if t == 8 else f"int8_t{t}" for t in tier_thresholds_incl_blanket
    )
    wins, _extrapolated = _per_layer_wins(
        sub, tq_curves, headline_col, ("fp16", *modes)
    )

    rows: list[dict] = []
    for cache, layer_packs in packs_by_cache.items():
        for layer, pack in layer_packs.items():
            key = (cache, int(layer))
            win_fp16 = wins["fp16"].get(key)
            if win_fp16 is None:
                continue
            for t, mode in zip(tier_thresholds_incl_blanket, modes):
                win_t = wins[mode].get(key)
                if win_t is None:
                    continue
                cert = int8_decoder_certificate_tiered(pack, t)
                rows.append(
                    dict(
                        model=model_label,
                        cache=cache,
                        budget=float(budget),
                        layer=int(layer),
                        tier_threshold=int(t),
                        implied_rel_degradation=cert["implied_rel_degradation"],
                        measured_rel_deg=1.0 - win_t / win_fp16,
                        frac_int8=cert["frac_int8"],
                        c_used=cert["c_used"],
                        c_int8=cert["c_int8"],
                    )
                )
    return rows


def _dec_quant_verdict(
    df: pd.DataFrame, tq_df: pd.DataFrame, headline_col: str, cfg: Config
) -> dict:
    # Keyed by (cache, layer): a multi-cache invocation must never pool TQ
    # distortions across caches into one curve (that mixes different
    # sequences' distortion scales and can even collapse two caches' points
    # onto the same bpe with different distortions, breaking the log-interp's
    # monotone-x assumption). _tq_layer_curve groups by layer only, so filter
    # to one cache's rows before calling it — keeps _k4_common generic for
    # k4_frontier.py's single-cache caller.
    tq_curves: dict[str, dict[int, list[tuple[float, float]]]] = {
        cache: _tq_layer_curve(g, headline_col) for cache, g in tq_df.groupby("cache")
    }

    modes = _dec_modes(cfg)
    tier_modes = tuple(f"int8_t{t}" for t in cfg.tier_thresholds)
    gate_mode = "int8_t5" if "int8_t5" in tier_modes else "int8"

    per_budget: dict[str, dict] = {}
    gate_pass_all = True
    fp16_flag = False

    for budget in cfg.budgets:
        sub = df[(df.arm == "spectral") & (df.budget == float(budget))]
        if sub.empty:
            continue

        # bpe_skeptic_deploy at the SAME point for all dec modes: the fp16
        # arm's value, per (cache, layer) — the point every mode is
        # interpolated at, isolating pure decoder-precision damage.
        wins_by_layer, extrapolated = _per_layer_wins(
            sub, tq_curves, headline_col, modes
        )
        wins: dict[str, list[float]] = {
            m: list(wins_by_layer[m].values()) for m in modes
        }

        if not wins["fp16"]:
            continue

        win_fp16 = float(pd.Series(wins["fp16"]).mean())
        win_by_mode: dict[str, float] = {
            m: float(pd.Series(wins[m]).mean()) if wins[m] else float("nan")
            for m in modes
        }

        rel_degradation_by_mode: dict[str, float] = {
            m: 1.0 - win_by_mode[m] / win_fp16
            for m in modes
            if m not in ("fp32", "fp16")
        }
        rel_degradation_fp16_vs_fp32 = 1.0 - win_fp16 / win_by_mode["fp32"]

        # Deployment view: win at each int8-family mode's OWN
        # bpe_skeptic_deploy (mixed_dec_charge-accounted) — the actual bits
        # that mode would cost in deployment, never gated on.
        own_bits_wins: dict[str, float] = {}
        for mode in modes:
            if mode in ("fp32", "fp16"):
                continue
            mode_sub = sub[sub.dec_mode == mode].set_index(["cache", "layer"])
            own_wins = []
            for (cache, layer), mode_row in mode_sub.iterrows():
                pts = tq_curves.get(cache, {}).get(int(layer))
                if not pts:
                    continue
                tq_dist_own, ex = _log_interp(pts, float(mode_row.bpe_skeptic_deploy))
                extrapolated = extrapolated or ex
                dist = max(float(mode_row[headline_col]), 1e-300)
                own_wins.append(tq_dist_own / dist)
            own_bits_wins[mode] = (
                float(pd.Series(own_wins).mean()) if own_wins else float("nan")
            )

        budget_pass = bool(rel_degradation_by_mode[gate_mode] < 0.05)
        gate_pass_all = gate_pass_all and budget_pass
        if rel_degradation_fp16_vs_fp32 > 0.005:
            fp16_flag = True

        entry = dict(
            win_fp32=win_by_mode["fp32"],
            win_fp16=win_fp16,
            win_int8=win_by_mode["int8"],
            rel_degradation_int8=rel_degradation_by_mode["int8"],
            rel_degradation_fp16_vs_fp32=rel_degradation_fp16_vs_fp32,
            int8_win_at_own_bits=own_bits_wins["int8"],
            gate_pass=budget_pass,
            gate_mode=gate_mode,
            n_samples=len(wins["fp16"]),
            extrapolated=bool(extrapolated),
        )
        for t in cfg.tier_thresholds:
            m = f"int8_t{t}"
            entry[f"win_{m}"] = win_by_mode[m]
            entry[f"rel_degradation_{m}"] = rel_degradation_by_mode[m]
            entry[f"{m}_win_at_own_bits"] = own_bits_wins[m]
        entry["ordering_ok"] = _ordering_ok(rel_degradation_by_mode)

        per_budget[f"{budget:g}"] = entry

    return dict(
        headline_metric=headline_col,
        gate_mode=gate_mode,
        per_budget=per_budget,
        gate_pass=bool(per_budget) and gate_pass_all,
        fp16_shippability_flag=fp16_flag,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
