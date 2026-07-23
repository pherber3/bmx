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
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Callable, NamedTuple

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.codecs import quantize_cache
from bmx.cache.collect import to_matrix
from bmx.cache.rope import apply_rope
from bmx.cache.spectral import (
    int8_decoder_roundtrip,
    load_packs,
    skeptic_charge,
    spectral_quantize,
)
from experiments._k4_common import (
    DEPLOY_S,
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


def _dec_variant(dec: torch.Tensor, bits_pc: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "fp32":
        return dec
    if mode == "fp16":
        return dec.half().float()
    if mode == "int8":
        return int8_decoder_roundtrip(dec, bits_pc)
    raise AssertionError(f"unknown dec_mode {mode!r}")


class _LayerCtx(NamedTuple):
    """Per-(cache, layer) setup shared by the tq-curve and dec-mode loops."""

    k_pre_t: torch.Tensor  # (h_kv, S, d) fp16, pre-RoPE
    h_kv: int
    S: int
    d: int
    C: int
    Q_fp32: torch.Tensor
    cos_l: torch.Tensor
    sin_l: torch.Tensor
    K_post_true: torch.Tensor | None
    M_pre: torch.Tensor  # (S, C) fp32
    tail: slice


def _layer_ctx(
    kinds_map: dict[str, torch.Tensor],
    *,
    rope_ready: bool,
    get_cos_sin: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
) -> _LayerCtx:
    k_pre_t = kinds_map["k_pre"]  # (h_kv, S, d) fp16, pre-RoPE
    q_t = kinds_map["q"]  # (h, T, d) fp16
    h_kv, S, d = k_pre_t.shape
    Q_fp32 = q_t.float()

    if rope_ready:
        cos_l, sin_l = get_cos_sin(S)
    else:
        cos_l = torch.ones(S, d)
        sin_l = torch.zeros(S, d)
    K_post_true = apply_rope(k_pre_t.float(), cos_l, sin_l) if rope_ready else None

    M_pre = to_matrix(k_pre_t)  # (S, C) fp32
    tail = slice(S // 2, S)

    return _LayerCtx(
        k_pre_t=k_pre_t,
        h_kv=h_kv,
        S=S,
        d=d,
        C=h_kv * d,
        Q_fp32=Q_fp32,
        cos_l=cos_l,
        sin_l=sin_l,
        K_post_true=K_post_true,
        M_pre=M_pre,
        tail=tail,
    )


def main(cfg: Config):
    assert cfg.cache_paths, "cache_paths must be non-empty"

    run = (
        create_run("k4_dec_quant", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_dec_quant", cfg)
    )

    model_label = cfg.model_label or "unknown"
    rows: list[dict] = []  # dec-mode arms (fp32/fp16/int8) -> metrics.parquet
    tq_rows: list[dict] = []  # turboquant_mse baseline curve, verdict-internal only
    headline_col = "logit"  # refined per-cache below (logit_rope when RoPE ready)
    any_rope_ready = False

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
                rf_tq, lg_tq, lg_rope_tq = _score_tail(
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

                # ---- fp32 / fp16 / int8 decoder-precision arms ---------------
                for mode in DEC_MODES:
                    dec_variant = _dec_variant(pack.dec, pack.bits, mode)
                    pack_variant = dataclasses.replace(pack, dec=dec_variant)
                    M_hat, bpe_model = spectral_quantize(ctx.M_pre, pack_variant)
                    rf, lg, lg_rope = _score_tail(
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
                    dec_bits = 8.0 if mode == "int8" else 16.0
                    bpe_skeptic_deploy = bpe_model + skeptic_charge(
                        ctx.C,
                        DEPLOY_S,
                        pack.tiers,
                        c_used=pack.c_used,
                        dec_bits=dec_bits,
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
        fp16_sub = sub[sub.dec_mode == "fp16"].set_index(["cache", "layer"])
        wins: dict[str, list[float]] = {m: [] for m in DEC_MODES}
        extrapolated = False

        for (cache, layer), fp16_row in fp16_sub.iterrows():
            pts = tq_curves.get(cache, {}).get(int(layer))
            if not pts:
                continue
            bpe_at = float(fp16_row.bpe_skeptic_deploy)
            tq_dist, ex = _log_interp(pts, bpe_at)
            extrapolated = extrapolated or ex
            for mode in DEC_MODES:
                row = sub[
                    (sub.cache == cache) & (sub.layer == layer) & (sub.dec_mode == mode)
                ]
                if row.empty:
                    continue
                dist = max(float(row[headline_col].iloc[0]), 1e-300)
                wins[mode].append(tq_dist / dist)

        if not wins["fp16"]:
            continue

        win_fp32 = float(pd.Series(wins["fp32"]).mean())
        win_fp16 = float(pd.Series(wins["fp16"]).mean())
        win_int8 = float(pd.Series(wins["int8"]).mean())

        rel_degradation_int8 = 1.0 - win_int8 / win_fp16
        rel_degradation_fp16_vs_fp32 = 1.0 - win_fp16 / win_fp32

        # Deployment view: win at int8's OWN bpe_skeptic_deploy (dec_bits=8,
        # c_used-accounted) — the actual bits int8 would cost, never gated on.
        int8_sub = sub[sub.dec_mode == "int8"].set_index(["cache", "layer"])
        int8_own_wins = []
        for (cache, layer), int8_row in int8_sub.iterrows():
            pts = tq_curves.get(cache, {}).get(int(layer))
            if not pts:
                continue
            tq_dist_own, ex = _log_interp(pts, float(int8_row.bpe_skeptic_deploy))
            extrapolated = extrapolated or ex
            dist = max(float(int8_row[headline_col]), 1e-300)
            int8_own_wins.append(tq_dist_own / dist)
        int8_win_at_own_bits = (
            float(pd.Series(int8_own_wins).mean()) if int8_own_wins else float("nan")
        )

        budget_pass = bool(rel_degradation_int8 < 0.05)
        gate_pass_all = gate_pass_all and budget_pass
        if rel_degradation_fp16_vs_fp32 > 0.005:
            fp16_flag = True

        per_budget[f"{budget:g}"] = dict(
            win_fp32=win_fp32,
            win_fp16=win_fp16,
            win_int8=win_int8,
            rel_degradation_int8=rel_degradation_int8,
            rel_degradation_fp16_vs_fp32=rel_degradation_fp16_vs_fp32,
            int8_win_at_own_bits=int8_win_at_own_bits,
            gate_pass=budget_pass,
            n_samples=len(wins["fp16"]),
            extrapolated=bool(extrapolated),
        )

    return dict(
        headline_metric=headline_col,
        per_budget=per_budget,
        gate_pass=bool(per_budget) and gate_pass_all,
        fp16_shippability_flag=fp16_flag,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
