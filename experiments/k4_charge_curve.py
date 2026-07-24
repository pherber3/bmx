"""Charge-corrected (skeptic-v2) companion curve — analytic re-derivation of
the bits-vs-context table, computed from COMMITTED NIAH parquets + the
fit-pack allocation stats. Never edits old parquets; never re-runs old cells.

`docs/2026-07-15-k4-duel-results.md` §3 measured the bits-vs-context curve
under skeptic-v1 (full-C fp16 decoder charge). This script recomputes those
SAME rows under skeptic-v2 (only-used-decoder-columns) and skeptic-v2-int8
(additionally int8-quantized decoder), per `bmx.cache.spectral.skeptic_charge`.
The correction is purely analytic — it derives from the row's recorded
sequence length S and the fit-pack parquet's per-layer `n_zero_dirs` at the
matching budget (mean over layers -> C_used) — no cache is re-scored.

Per k4 arm row: with C the nominal spectral width, S the row's recorded
sequence length, and C_used the mean-over-layers used-column count at that
arm's budget:

    corrected(db) = measured_kv_bits
        - [skeptic_charge(C, S, tiers) - skeptic_charge(C, S, tiers, c_used=C_used, dec_bits=db)] / 2
        - [scale_bits(group) * (1 - C_used / C)] / 2

The two `/2` factors are the K/V blend (`bmx.cache.generate.avg_bpe` charges
K and V equally); only K carries the spectral pack, so only K's charge
changes. TurboQuant baseline rows (no pack) pass through unchanged into every
column — they have no C_used to correct.
"""

from __future__ import annotations

import dataclasses
import re

import pandas as pd
import tyro

from bmx.cache.codecs import scale_bits
from bmx.cache.spectral import mixed_dec_charge, skeptic_charge

_K4_ARM_RE = re.compile(r"^k4_b(?P<budget>[\d.]+)$")


@dataclasses.dataclass
class Config:
    niah_run_dirs: tuple[str, ...]
    fit_packs_parquet: str = (
        "results/k4_fit_packs/20260713-133628-798d0ef/metrics.parquet"
    )
    budgets: tuple[float, ...] = (2.2, 2.5)
    C: int = 1024
    group: int = 64
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8)
    dec_bits_variants: tuple[float, ...] = (16.0, 8.0)
    # Tier-gated honest replacement for the blanket skeptic-v2-int8 column
    # (rejected by its own certificate, see §3b of the local-levers design
    # doc). Each frac f is applied to C_used as c_int8 = f * c_used through
    # `mixed_dec_charge` — NOT an effective-dec-bits value routed through
    # `skeptic_charge`, which would double-count the fp16-scale term. Empty
    # by default (today's output unchanged).
    int8_frac_variants: tuple[float, ...] = ()
    out_path: str = ""


def _mean_c_used(fit_df: pd.DataFrame, budget: float, C: int) -> float:
    """Mean-over-layers C_used at `budget` from the fit-pack parquet's
    per-layer n_zero_dirs (C_used = C - n_zero_dirs)."""
    rows = fit_df[fit_df["budget"] == budget]
    assert len(rows) > 0, f"no fit-pack rows at budget={budget}"
    mean_n_zero = rows["n_zero_dirs"].mean()
    return C - mean_n_zero


def _load_niah_rows(run_dirs: tuple[str, ...]) -> pd.DataFrame:
    """Explicit run selection (plot-script discipline) — concat only the
    dirs the caller named, never a blind glob of a results root."""
    frames = []
    for d in run_dirs:
        df = pd.read_parquet(f"{d}/metrics.parquet")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _corrected_bits(
    measured: float,
    C: int,
    S: int,
    tiers: tuple[int, ...],
    group: int,
    c_used: float,
    dec_bits: float,
) -> float:
    charge_v1 = skeptic_charge(C, S, tiers)
    charge_v2 = skeptic_charge(C, S, tiers, c_used=c_used, dec_bits=dec_bits)
    decoder_delta = (charge_v1 - charge_v2) / 2.0
    scale_delta = (scale_bits(group) * (1 - c_used / C)) / 2.0
    return measured - decoder_delta - scale_delta


def _corrected_bits_mixed(
    measured: float,
    C: int,
    S: int,
    tiers: tuple[int, ...],
    group: int,
    c_used: float,
    frac: float,
) -> float:
    """Same corrected() blend as `_corrected_bits` (same /2 K-V blend factor,
    same scale_bits term), but the decoder charge routes through
    `mixed_dec_charge(c_int8=frac*c_used)` instead of
    `skeptic_charge(dec_bits=db)` — the tier-gated honest replacement for the
    blanket int8 column (see module docstring / §3b of the local-levers
    design doc). Endpoint-pinned: frac=0.0 == dec_bits=16.0 (skeptic-v2),
    frac=1.0 == dec_bits=8.0 (skeptic-v2-int8) EXACTLY, since
    mixed_dec_charge reduces to skeptic_charge at c_int8 in {0, c_used}."""
    charge_v1 = skeptic_charge(C, S, tiers)
    charge_mixed = mixed_dec_charge(C, S, tiers, c_used=c_used, c_int8=frac * c_used)
    decoder_delta = (charge_v1 - charge_mixed) / 2.0
    scale_delta = (scale_bits(group) * (1 - c_used / C)) / 2.0
    return measured - decoder_delta - scale_delta


def _mode_label(dec_bits: float) -> str:
    return "skeptic-v2" if dec_bits == 16.0 else "skeptic-v2-int8"


def _frac_mode_label(frac: float) -> str:
    return f"skeptic-v2-int8frac{frac:g}"


def _crossover_context(curve: dict[int, float], other: dict[int, float]) -> str:
    """Linear interpolation in 1/S for the length at which `curve` crosses
    below `other`. Both dicts keyed by length (int), values = bits. Returns a
    human-readable bracket string, or 'no crossover in range' if the sign of
    (curve - other) never flips over the shared lengths."""
    lengths = sorted(set(curve) & set(other))
    if len(lengths) < 2:
        return "insufficient shared lengths"
    diffs = [(length, curve[length] - other[length]) for length in lengths]
    for (s0, d0), (s1, d1) in zip(diffs, diffs[1:]):
        if d0 == 0:
            return f"exactly at {s0}"
        if (d0 > 0) != (d1 > 0):
            # Linear interpolation in x = 1/S: d(x) = d0 + (d1-d0)*(x-x0)/(x1-x0)
            x0, x1 = 1.0 / s0, 1.0 / s1
            x_star = x0 + (0 - d0) * (x1 - x0) / (d1 - d0)
            s_star = 1.0 / x_star
            return f"between {s0} and {s1} (~{s_star:,.0f})"
    return "no crossover in range"


def main(cfg: Config) -> None:
    fit_df = pd.read_parquet(cfg.fit_packs_parquet)
    niah_df = _load_niah_rows(cfg.niah_run_dirs)

    c_used_by_budget = {b: _mean_c_used(fit_df, b, cfg.C) for b in cfg.budgets}

    mode_names = (
        ["v1 as-measured"]
        + [_mode_label(db) for db in cfg.dec_bits_variants]
        + [_frac_mode_label(f) for f in cfg.int8_frac_variants]
    )

    # arm -> length -> {mode_name: bits}
    arm_curves: dict[str, dict[int, dict[str, float]]] = {}
    for arm, arm_df in niah_df.groupby("arm"):
        m = _K4_ARM_RE.match(arm)
        for length, length_df in arm_df.groupby("length"):
            measured = length_df["kv_size_bits"].mean()
            row_modes = {"v1 as-measured": measured}
            if m is not None:
                budget = float(m.group("budget"))
                c_used = c_used_by_budget.get(budget)
                if c_used is not None:
                    for db in cfg.dec_bits_variants:
                        row_modes[_mode_label(db)] = _corrected_bits(
                            measured,
                            cfg.C,
                            int(length),
                            cfg.tiers,
                            cfg.group,
                            c_used,
                            db,
                        )
                    for f in cfg.int8_frac_variants:
                        row_modes[_frac_mode_label(f)] = _corrected_bits_mixed(
                            measured,
                            cfg.C,
                            int(length),
                            cfg.tiers,
                            cfg.group,
                            c_used,
                            f,
                        )
            else:
                # TQ baseline (or other non-k4 arm): pass through unchanged.
                for db in cfg.dec_bits_variants:
                    row_modes[_mode_label(db)] = measured
                for f in cfg.int8_frac_variants:
                    row_modes[_frac_mode_label(f)] = measured
            arm_curves.setdefault(arm, {})[int(length)] = row_modes

    lines = []
    lines.append("Charge-corrected companion curve (skeptic-v2)")
    lines.append("")
    lines.append("| arm | length | " + " | ".join(mode_names) + " |")
    lines.append("|---|---|" + "---|" * len(mode_names))
    for arm in sorted(arm_curves):
        for length in sorted(arm_curves[arm]):
            row_modes = arm_curves[arm][length]
            cells = [f"{row_modes.get(name, float('nan')):.2f}" for name in mode_names]
            lines.append(f"| {arm} | {length} | " + " | ".join(cells) + " |")

    # Crossover statements, per mode, k4_b2.5 vs the two TQ baselines.
    lines.append("")
    lines.append("Crossovers (k4_b2.5 vs baselines, per accounting mode):")
    b25_arm = "k4_b2.5"
    for baseline_arm in ("turboquant_mse_b3", "turboquant_mse_k3v2"):
        if b25_arm not in arm_curves or baseline_arm not in arm_curves:
            continue
        for mode_name in mode_names:
            k4_curve = {
                length: vals[mode_name]
                for length, vals in arm_curves[b25_arm].items()
                if mode_name in vals
            }
            base_curve = {
                length: vals[mode_name]
                for length, vals in arm_curves[baseline_arm].items()
                if mode_name in vals
            }
            crossing = _crossover_context(k4_curve, base_curve)
            lines.append(f"- {b25_arm} vs {baseline_arm} [{mode_name}]: {crossing}")

    text = "\n".join(lines) + "\n"
    if cfg.out_path:
        with open(cfg.out_path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main(tyro.cli(Config))
