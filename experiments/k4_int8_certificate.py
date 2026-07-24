"""K4 int8-decoder certificate table (math review 2026-07-24 finding #9): the
exact offline distortion certificate for every layer of an EXISTING pack file
(refits nothing, loads no caches, needs no GPU).

Per (budget, layer): int8_decoder_certificate(pack) -> added / payload /
noise_to_signal / implied_rel_degradation, plus the mapping onto the
pre-registered VM gate axis (rel_degradation_int8 < 5%). The verdict JSON
carries max_implied_rel_degradation, the margin factor to the 5% line, and
the binding review condition: THE USER REVIEWS THESE NUMBERS BEFORE VM TASK 8
IS RELEASED — the VM task stays queued until explicit release.

Micro-task 6b (tier-gated rescue sweep, pure analysis, no codec change): the
blanket certificate above int8-stores EVERY used decoder column and can fail
the 5% gate — but the failure is driven by high-tier (6/8-bit) columns whose
payload is geometrically tiny (payload_i = lam_i*4^-b_i) while the int8 noise
floor added_i is roughly flat across tiers. "Is int8 dead, or just
misapplied?" For each `tier_thresholds` entry T, this sweep int8-stores ONLY
columns with `bits_i <= T` (columns with `bits_i > T` stay fp16 -> zero added
error on those columns) via `int8_decoder_certificate_tiered` — pure
post-processing of the existing per-layer ddec, zero codec change.

Accounting (skeptic-v2-int8 arithmetic, generalized to a per-column mix, NOT
a new mode — see `mixed_dec_charge`'s docstring in spectral.py for the exact
per-column terms): for each (budget, threshold), summed over every layer's
decoder resident in the cache,

    charge_saving_at_S = sum_layers [ skeptic_charge(dec_bits=16, S)
                                       - mixed_dec_charge(c_int8_layer, S) ]

is the total per-sequence bits/token saved relative to an all-fp16 decoder,
evaluated at S in {4096, 16384, 65536} (the deployment-length grid this repo
reports skeptic charges at). `effective_dec_bits` is the c_used-weighted mix
bit rate of the gated decoder (16.0 = no int8 coverage survives the gate;
8 + 16/C = full blanket int8 coverage), aggregated across layers via total
c_used/c_int8 counts (S-independent, matching effective_dec_bits's contract).
"""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics
from bmx.cache.spectral import (
    effective_dec_bits,
    int8_decoder_certificate,
    int8_decoder_certificate_tiered,
    load_packs,
    mixed_dec_charge,
    per_layer_tier_thresholds,
    skeptic_charge,
)

CHARGE_S_GRID = (4096, 16384, 65536)


@dataclasses.dataclass
class Config:
    pack_path: str
    budgets: tuple[float, ...] = (2.2, 2.5)
    tier_thresholds: tuple[int, ...] = (2, 3, 4, 5, 6)
    model_label: str = ""
    out_root: str = ""


def main(cfg: Config):
    run = (
        create_run("k4_int8_certificate", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_int8_certificate", cfg)
    )
    model_label = cfg.model_label or "unknown"

    # ---- blanket per-(budget, layer) certificate table (unchanged) --------
    rows: list[dict] = []
    packs_by_budget: dict[float, dict[int, object]] = {}
    for budget in cfg.budgets:
        packs = load_packs(cfg.pack_path, budget)
        packs_by_budget[budget] = packs
        for layer_i, pack in sorted(packs.items()):
            cert = int8_decoder_certificate(pack)
            rows.append(
                dict(
                    model=model_label,
                    budget=float(budget),
                    layer=layer_i,
                    c_used=int(pack.c_used),
                    **cert,
                )
            )
            print(
                f"  b={budget:g} layer={layer_i:2d} "
                f"noise_to_signal={cert['noise_to_signal']:.3e} "
                f"implied_rel_degradation={cert['implied_rel_degradation']:.3e}",
                flush=True,
            )
    df = pd.DataFrame(rows)
    write_metrics(run, df)

    max_impl = float(df.implied_rel_degradation.max())
    verdict = dict(
        pack_path=cfg.pack_path,
        budgets=list(cfg.budgets),
        max_noise_to_signal=float(df.noise_to_signal.max()),
        max_implied_rel_degradation=max_impl,
        vm_gate_line=0.05,
        margin_factor=0.05 / max(max_impl, 1e-300),
        certificate_far_inside_gate=bool(max_impl < 0.005),  # 10x margin ask
        user_review_required_before_vm_task8_release=True,
        note=(
            "These numbers are an offline certificate for user review. VM "
            "Task 8's gate measurement stays queued until the user "
            "explicitly releases it; this certificate does not replace it."
        ),
        git_sha=git_sha(),
    )

    # ---- tier-gated rescue sweep (6b) --------------------------------------
    # Per-(budget, threshold, layer) detail table (full transparency). The
    # certs dict is reused by the aggregate loop below (each tiered
    # certificate re-runs the roundtrip + projection — compute it once).
    sweep_layer_rows: list[dict] = []
    tiered_certs: dict[tuple[float, int, int], dict] = {}
    for budget, packs in packs_by_budget.items():
        for layer_i, pack in sorted(packs.items()):
            for T in cfg.tier_thresholds:
                tiered = int8_decoder_certificate_tiered(pack, T)
                tiered_certs[(budget, T, layer_i)] = tiered
                sweep_layer_rows.append(
                    dict(
                        model=model_label,
                        budget=float(budget),
                        layer=layer_i,
                        C=int(pack.enc.shape[0]),
                        **tiered,
                    )
                )
    sweep_layer_df = pd.DataFrame(sweep_layer_rows)
    write_metrics(run, sweep_layer_df, name="tier_sweep_layers")

    # Per-(budget, threshold) aggregate: worst-case implied_rel_degradation
    # across layers (the binding number, same convention as the blanket
    # verdict's max), total int8 coverage, effective_dec_bits, and the
    # charge saving summed over every layer's decoder at each S in the grid.
    sweep_rows: list[dict] = []
    for budget, packs in packs_by_budget.items():
        # effective_dec_bits requires a single C; assert layer uniformity
        # (true for every pack file this repo has fit so far) rather than
        # silently mixing per-layer channel counts into one number.
        Cs = {int(pack.enc.shape[0]) for pack in packs.values()}
        assert len(Cs) == 1, f"non-uniform per-layer C not supported: {Cs}"
        C = next(iter(Cs))
        for T in cfg.tier_thresholds:
            layer_certs = {
                layer_i: tiered_certs[(budget, T, layer_i)] for layer_i in packs
            }
            c_used_total = sum(c["c_used"] for c in layer_certs.values())
            c_int8_total = sum(c["c_int8"] for c in layer_certs.values())
            worst_impl = max(c["implied_rel_degradation"] for c in layer_certs.values())
            worst_n2s = max(c["noise_to_signal"] for c in layer_certs.values())
            eff_bits = effective_dec_bits(C, c_used_total, c_int8_total)

            row = dict(
                model=model_label,
                budget=float(budget),
                tier_threshold=T,
                c_used_total=c_used_total,
                c_int8_total=c_int8_total,
                frac_int8=c_int8_total / c_used_total if c_used_total > 0 else 0.0,
                max_implied_rel_degradation=worst_impl,
                max_noise_to_signal=worst_n2s,
                effective_dec_bits=eff_bits,
            )
            for S in CHARGE_S_GRID:
                saving = sum(
                    skeptic_charge(
                        int(pack.enc.shape[0]),
                        S,
                        pack.tiers,
                        c_used=pack.c_used,
                        dec_bits=16.0,
                    )
                    - mixed_dec_charge(
                        int(pack.enc.shape[0]),
                        S,
                        pack.tiers,
                        c_used=pack.c_used,
                        c_int8=layer_certs[layer_i]["c_int8"],
                    )
                    for layer_i, pack in packs.items()
                )
                row[f"charge_saving_at_S{S}"] = saving
            sweep_rows.append(row)
            print(
                f"  [sweep] b={budget:g} T={T} frac_int8={row['frac_int8']:.3f} "
                f"max_implied_rel_degradation={worst_impl:.4f} "
                f"eff_dec_bits={eff_bits:.3f} "
                f"saving@4096={row['charge_saving_at_S4096']:.4f}",
                flush=True,
            )
    sweep_df = pd.DataFrame(sweep_rows)
    write_metrics(run, sweep_df, name="tier_sweep")

    # Full-int8 charge saving reference (T = max tier, i.e. the blanket
    # int8 decoder) at the same S grid, per budget — "how much of the
    # original int8 charge saving survives" needs this denominator.
    full_int8_saving: dict[str, dict[str, float]] = {}
    for budget, packs in packs_by_budget.items():
        entry = {}
        for S in CHARGE_S_GRID:
            entry[f"S{S}"] = sum(
                skeptic_charge(
                    int(pack.enc.shape[0]),
                    S,
                    pack.tiers,
                    c_used=pack.c_used,
                    dec_bits=16.0,
                )
                - skeptic_charge(
                    int(pack.enc.shape[0]),
                    S,
                    pack.tiers,
                    c_used=pack.c_used,
                    dec_bits=8.0,
                )
                for pack in packs.values()
            )
        full_int8_saving[f"{budget:g}"] = entry

    # Largest threshold clearing the 5% gate, per budget (the interesting
    # readout): scan tier_thresholds ascending, keep the last one whose
    # aggregate max_implied_rel_degradation is still under the line.
    per_budget_rescue: dict[str, dict] = {}
    for budget in cfg.budgets:
        b_rows = [r for r in sweep_rows if r["budget"] == float(budget)]
        b_rows.sort(key=lambda r: r["tier_threshold"])
        passing = [r for r in b_rows if r["max_implied_rel_degradation"] < 0.05]
        best = max(passing, key=lambda r: r["tier_threshold"]) if passing else None
        per_budget_rescue[f"{budget:g}"] = dict(
            largest_passing_threshold=(best["tier_threshold"] if best else None),
            rel_degradation_at_that_threshold=(
                best["max_implied_rel_degradation"] if best else None
            ),
            margin_to_gate=(
                0.05 - best["max_implied_rel_degradation"] if best else None
            ),
            frac_int8_at_that_threshold=(best["frac_int8"] if best else None),
            charge_saving_fraction_at_S4096=(
                best["charge_saving_at_S4096"]
                / max(full_int8_saving[f"{budget:g}"]["S4096"], 1e-300)
                if best
                else 0.0
            ),
            any_threshold_passes=bool(passing),
        )

    verdict["tier_sweep"] = dict(
        tier_thresholds=list(cfg.tier_thresholds),
        charge_s_grid=list(CHARGE_S_GRID),
        full_int8_charge_saving=full_int8_saving,
        per_budget_rescue=per_budget_rescue,
        note=(
            "Rescue sweep: int8-store only columns with bits<=T, fp16 for "
            "the rest (pure post-processing of the existing certificate, no "
            "codec change). largest_passing_threshold is the biggest T whose "
            "aggregate (max-over-layers) implied_rel_degradation stays under "
            "the 5% VM gate line; charge_saving_fraction_at_S4096 is how much "
            "of the FULL blanket-int8 charge saving survives at that T. Same "
            "user-review gate as the blanket certificate above -- this sweep "
            "informs the user's decision, nothing else."
        ),
    )

    # ---- per-layer T_ℓ sweep (K4 estimation-levers Task 3) -----------------
    # per_layer_tier_thresholds(packs, bar=0.05) gives each layer its OWN
    # certified tier ceiling instead of the layer-uniform T=5 (which binds on
    # the single worst layer at every T). The analytic question: how much of
    # uniform T=5's remaining charge-saving gap does per-layer recover, at
    # S=4096 -- the pre-registered MATERIALITY bar (spec Part 2) is
    # saving_delta >= 0.3 bits/token/model; below that, "certified but
    # immaterial" is the expected, honest verdict (recorded here, not shipped
    # into recipes/docs regardless of quality-bar outcome).
    per_layer_rows: list[dict] = []
    per_layer_summary: dict[str, dict] = {}
    for budget, packs in packs_by_budget.items():
        t_map = per_layer_tier_thresholds(packs, bar=0.05)
        saving_delta_by_s: dict[int, float] = {S: 0.0 for S in CHARGE_S_GRID}
        for layer_i, pack in sorted(packs.items()):
            T_l = t_map[layer_i]
            C_l = int(pack.enc.shape[0])
            cert_l = int8_decoder_certificate_tiered(pack, T_l) if T_l > 0 else None
            c_int8_l = pack.c_int8(T_l) if T_l > 0 else 0
            c_int8_uniform5 = pack.c_int8(5)
            per_layer_rows.append(
                dict(
                    model=model_label,
                    budget=float(budget),
                    layer=layer_i,
                    C=C_l,
                    t_layer=T_l,
                    c_used=int(pack.c_used),
                    c_int8_t_layer=c_int8_l,
                    c_int8_uniform_t5=c_int8_uniform5,
                    implied_rel_degradation_at_t_layer=(
                        cert_l["implied_rel_degradation"] if cert_l else 0.0
                    ),
                )
            )
            for S in CHARGE_S_GRID:
                charge_uniform5 = mixed_dec_charge(
                    C_l, S, pack.tiers, c_used=pack.c_used, c_int8=c_int8_uniform5
                )
                charge_t_layer = mixed_dec_charge(
                    C_l, S, pack.tiers, c_used=pack.c_used, c_int8=c_int8_l
                )
                # positive = per-layer cheaper than uniform T=5.
                saving_delta_by_s[S] += charge_uniform5 - charge_t_layer
        per_layer_summary[f"{budget:g}"] = dict(
            t_layer_map={str(k): v for k, v in sorted(t_map.items())},
            **{f"saving_delta_at_S{S}": saving_delta_by_s[S] for S in CHARGE_S_GRID},
            materiality_bar_bits_per_token=0.3,
            materiality_pass_at_S4096=bool(saving_delta_by_s[4096] >= 0.3),
        )
        print(
            f"  [per-layer] b={budget:g} T_l={t_map} "
            f"saving_delta@4096={saving_delta_by_s[4096]:.4f} "
            f"(bar=0.3, pass={per_layer_summary[f'{budget:g}']['materiality_pass_at_S4096']})",
            flush=True,
        )
    per_layer_df = pd.DataFrame(per_layer_rows)
    write_metrics(run, per_layer_df, name="per_layer_tl_sweep")

    verdict["per_layer_tl_sweep"] = dict(
        bar=0.05,
        materiality_bar_bits_per_token_at_S4096=0.3,
        per_budget=per_layer_summary,
        note=(
            "Certificate-derived per-layer T_ℓ (per_layer_tier_thresholds, "
            "same 5% bar as the blanket/tier-sweep certificates above) vs "
            "the uniform T=5 threshold this repo shipped last cycle. "
            "saving_delta_at_S{S} = sum_layers [mixed_dec_charge at uniform "
            "T=5's c_int8 - mixed_dec_charge at T_ℓ's c_int8], bits/token/"
            "model, positive meaning per-layer is CHEAPER than uniform T=5. "
            "THE MATERIALITY BAR (pre-registered, spec Part 2): "
            "saving_delta_at_S4096 >= 0.3 bits/token/model, or per-layer "
            "does NOT ship regardless of the quality bar -- 'certified but "
            "immaterial' is a valid, expected verdict, not a failure to "
            "record honestly."
        ),
    )

    (run / "certificate_verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\n" + "=" * 88)
    print("INT8 DECODER CERTIFICATE — user reviews before VM Task 8 is released")
    print("=" * 88)
    print(json.dumps(verdict, indent=2))
    print(f"-> {run}")
    return run


if __name__ == "__main__":
    main(tyro.cli(Config))
