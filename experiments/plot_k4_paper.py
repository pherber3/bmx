"""K4 paper figures — reads committed parquet/JSON only, never refits.

Four figures, matplotlib (no seaborn), PDF+PNG, deterministic bytes (no
CreationDate timestamp in the PDF):

1. bits-vs-context (the headline, Llama-3.1-8B-Instruct): x=context length
   (log2, 4k-64k), y=blended kv bits. k4_b2.5 / k4_b2.2 under skeptic-v2
   (primary, solid) with skeptic-v1 as-measured as faint companions, plus the
   tq_b3 / tq_k3v2 baselines and the certified tier-gated int8 BAND (shaded
   fill_between the frac=0.893/0.916 curves — the blanket int8 projection
   this line used to draw was REJECTED by its own certificate+measurement,
   see docs/2026-07-25-k4-local-levers-results.md). Reuses
   experiments.k4_charge_curve's correction machinery BY IMPORT — the
   corrected-bits formula lives there once.
2. corpus-transfer matrix + synthesis ladder (gpt2-scale, mechanism-only):
   left panel = D-matrix heatmap (fit corpus x eval corpus) at budget=2.5;
   right panel = the order-ladder (shuf/uni/bi) per eval side as grouped
   bars against the 10% insensitivity line. Every number read verbatim from
   corpus_transfer_verdict.json; nothing recomputed.
3. per-rank subspace overlap (gpt2-scale, mechanism-only): overlap vs rank
   cutoff, one line per corpus pair, mean over layers, from the same run's
   overlap.parquet (kind=='overlap', centered==False).
4. certificate-vs-measured (instrument validation, gpt2-scale): log-log
   scatter of the offline int8-tier certificate's implied_rel_degradation
   against the measured rel_deg, one marker per tier_threshold, y=x
   reference line — the certificate is uniformly conservative (every point
   below the line). Read verbatim from the committed cert_vs_measured.parquet
   (96 points); nothing recomputed.

Source runs (explicit, never a glob of a results root):
  - results/k3_niah/20260715-{080730,080909,081107,081253,110927,111108,
    111257,111443,111948}-21e6d81 (Llama NIAH duel, fig 1)
  - results/k4_fit_packs/20260713-133628-798d0ef/metrics.parquet (fig 1,
    C_used correction inputs)
  - results/k4_corpus_transfer/20260723-220816-9d11538/{corpus_transfer_
    verdict.json,overlap.parquet} (figs 2-3, gpt2-scale)
  - results/k4_dec_quant/20260724-125348-0f49e32/cert_vs_measured.parquet
    (fig 4, gpt2-scale certificate-vs-measured)
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import matplotlib
import pandas as pd
import tyro

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.k4_charge_curve import (
    _corrected_bits,
    _corrected_bits_mixed,
    _crossover_context,
    _load_niah_rows,
    _mean_c_used,
    _mode_label,
)

GPT2_YELLOW_FLAG = (
    "gpt2 scale = mechanism verdict only, not a paper-scale quality claim "
    "(Llama fit-side replication pre-registered before any paper claim; "
    "see corpus_transfer_verdict.json's gpt2_yellow_flag)."
)

# Deterministic PDF bytes: matplotlib stamps a CreationDate into PDF metadata
# by default, which would make repeated runs of this script produce
# byte-different files. None suppresses the field.
_SAVEFIG_KW = dict(metadata={"CreationDate": None})

# ---------------------------------------------------------------------------
# Figure 1: bits-vs-context (Llama, the headline)
# ---------------------------------------------------------------------------

_K4_BUDGETS = (2.5, 2.2)
_K4_ARM_LABELS = {2.5: "k4_b2.5", 2.2: "k4_b2.2"}
_TQ_ARMS = ("turboquant_mse_b3", "turboquant_mse_k3v2")
_TQ_LABELS = {"turboquant_mse_b3": "tq_b3", "turboquant_mse_k3v2": "tq_k3v2"}
_TQ_COLORS = {"turboquant_mse_b3": "#d62728", "turboquant_mse_k3v2": "#9467bd"}
_K4_COLORS = {2.5: "#1f77b4", 2.2: "#2ca02c"}

# Certified tier-gated (T=5) int8 charge-saving fraction band — the two
# `frac_int8` values from the T=5 row of the real gpt2 sweep
# (results/k4_int8_certificate/20260724-113708-ff03bf6/, reproduced in
# docs/2026-07-25-k4-local-levers-results.md §task-6 table: budget 2.2 ->
# 0.916, budget 2.5 -> 0.893). This SUPERSEDES the blanket dec_bits=8.0
# (frac=1.0) column, which was rejected by its own certificate+measurement.
# Both fracs are applied to EACH k4 arm's own c_used to draw a band (not a
# per-budget point) — it is a gpt2-fit ESTIMATE of the Llama band, pending
# exact per-layer tier counts at the refit
# (docs/2026-07-25-k4-local-levers-results.md).
_INT8_TIER_FRAC_BAND = (0.893, 0.916)
_INT8_BAND_LO_MODE = "skeptic-v2-int8-tier-band-lo"
_INT8_BAND_HI_MODE = "skeptic-v2-int8-tier-band-hi"


def _build_bits_vs_context(
    niah_run_dirs: tuple[str, ...],
    fit_packs_parquet: str,
    budgets: tuple[float, ...],
    C: int,
    group: int,
    tiers: tuple[int, ...],
) -> dict:
    """arm -> mode_name -> sorted [(length, bits), ...]; TQ arms pass through
    unchanged into every mode (no spectral pack -> no C_used correction, per
    k4_charge_curve's documented rule)."""
    fit_df = pd.read_parquet(fit_packs_parquet)
    niah_df = _load_niah_rows(niah_run_dirs)

    dec_bits_variants = (16.0,)  # skeptic-v2 only; blanket int8 no longer drawn
    c_used_by_budget = {b: _mean_c_used(fit_df, b, C) for b in budgets}
    mode_names = (
        ["v1 as-measured"]
        + [_mode_label(db) for db in dec_bits_variants]
        + [_INT8_BAND_LO_MODE, _INT8_BAND_HI_MODE]
    )

    curves: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for arm, arm_df in niah_df.groupby("arm"):
        is_k4 = arm in _K4_ARM_LABELS.values()
        budget = next((b for b, lbl in _K4_ARM_LABELS.items() if lbl == arm), None)
        by_mode: dict[str, list[tuple[int, float]]] = {m: [] for m in mode_names}
        for length, length_df in arm_df.groupby("length"):
            measured = float(length_df["kv_size_bits"].mean())
            by_mode["v1 as-measured"].append((int(length), measured))
            c_used = c_used_by_budget.get(budget) if is_k4 else None
            for db in dec_bits_variants:
                mode = _mode_label(db)
                if c_used is not None:
                    val = _corrected_bits(
                        measured, C, int(length), tiers, group, c_used, db
                    )
                else:
                    val = measured  # TQ baseline: pass-through, all modes
                by_mode[mode].append((int(length), val))
            # Certified tier-gated band: both fracs applied to THIS arm's own
            # c_used (frac is a gpt2-band estimate, not a per-arm/budget
            # point) -- lo frac gives the smaller correction (higher bits,
            # "hi" line), hi frac gives the larger correction (lower bits,
            # "lo" line); sort below by band membership, not frac magnitude.
            if c_used is not None:
                band_vals = [
                    _corrected_bits_mixed(
                        measured, C, int(length), tiers, group, c_used, frac
                    )
                    for frac in _INT8_TIER_FRAC_BAND
                ]
                lo_val, hi_val = min(band_vals), max(band_vals)
            else:
                lo_val = hi_val = measured  # TQ baseline: pass-through
            by_mode[_INT8_BAND_LO_MODE].append((int(length), lo_val))
            by_mode[_INT8_BAND_HI_MODE].append((int(length), hi_val))
        for m in mode_names:
            by_mode[m].sort()
        curves[arm] = by_mode

    # Crossover annotations (k4_b2.5 vs the two TQ baselines), reusing
    # k4_charge_curve's own interpolation routine — same numbers as the
    # committed duel doc (docs/2026-07-15-k4-duel-results.md), never
    # recomputed by a second formula.
    crossovers: dict[str, dict[str, str]] = {}
    b25 = _K4_ARM_LABELS[2.5]
    if b25 in curves:
        for tq_arm in _TQ_ARMS:
            if tq_arm not in curves:
                continue
            crossovers[tq_arm] = {}
            for mode in mode_names:
                k4_curve = dict(curves[b25][mode])
                base_curve = dict(curves[tq_arm][mode])
                crossovers[tq_arm][mode] = _crossover_context(k4_curve, base_curve)

    return dict(curves=curves, mode_names=mode_names, crossovers=crossovers)


def make_bits_vs_context_figure(result: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = result["curves"]
    fig, ax = plt.subplots(figsize=(9, 6))

    for budget in _K4_BUDGETS:
        arm = _K4_ARM_LABELS[budget]
        if arm not in curves:
            continue
        color = _K4_COLORS[budget]
        by_mode = curves[arm]

        v1 = by_mode.get("v1 as-measured", [])
        if v1:
            xs, ys = zip(*v1)
            ax.plot(
                xs,
                ys,
                "o-",
                color=color,
                alpha=0.35,
                lw=1.0,
                ms=4,
                label=f"{arm} (v1 as-measured, faint)",
            )

        v2 = by_mode.get("skeptic-v2", [])
        if v2:
            xs, ys = zip(*v2)
            ax.plot(
                xs,
                ys,
                "o-",
                color=color,
                lw=2.2,
                ms=6,
                label=f"{arm} (skeptic-v2, primary)",
            )

        band_lo = by_mode.get(_INT8_BAND_LO_MODE, [])
        band_hi = by_mode.get(_INT8_BAND_HI_MODE, [])
        if band_lo and band_hi:
            xs_lo, ys_lo = zip(*band_lo)
            xs_hi, ys_hi = zip(*band_hi)
            assert xs_lo == xs_hi
            ax.fill_between(
                xs_lo,
                ys_lo,
                ys_hi,
                color=color,
                alpha=0.18,
                lw=0,
                label=f"{arm} (skeptic-v2-int8, tier-gated T=5, certified band)",
            )

    for tq_arm in _TQ_ARMS:
        if tq_arm not in curves:
            continue
        pts = curves[tq_arm].get("v1 as-measured", [])
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(
            xs,
            ys,
            "s--",
            color=_TQ_COLORS[tq_arm],
            lw=1.6,
            ms=5,
            label=_TQ_LABELS[tq_arm],
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length (tokens, log2)")
    ax.set_ylabel("blended KV bits/entry")
    ax.set_title("K4 bits-vs-context (Llama-3.1-8B-Instruct, real-text NIAH duel)")
    ax.legend(fontsize=6.5, loc="upper right", ncol=1)
    ax.grid(alpha=0.2)

    # Crossover annotations (skeptic-v2, the primary mode) — placed below the
    # legend/plot as their own text block, never overlapping the curves.
    crossings = result["crossovers"]
    crossover_lines = []
    for tq_arm, by_mode in crossings.items():
        v1_txt = by_mode.get("v1 as-measured", "")
        v2_txt = by_mode.get("skeptic-v2", "")
        crossover_lines.append(
            f"k4_b2.5 x {_TQ_LABELS[tq_arm]} crossover -- v1: {v1_txt}; v2: {v2_txt}"
        )

    caption = (
        "Model: meta-llama/Llama-3.1-8B-Instruct. Solid = skeptic-v2 (primary "
        "accounting, used-decoder-columns only); faint = skeptic-v1 "
        "as-measured (full-C fp16 decoder charge, every parquet before "
        "2026-07-23). The blanket skeptic-v2-int8 projection this figure "
        "used to draw as a dashed line was REJECTED by its own certificate + "
        "measurement (rel_degradation 13.5%/16.7% at b2.2/b2.5 vs the 5% "
        "line, docs/2026-07-25-k4-local-levers-results.md); the shaded band "
        "replaces it with the CERTIFIED tier-gated (T=5) accounting -- "
        "mixed_dec_charge at int8 charge-saving fraction 0.893/0.916 (the "
        "gpt2 T=5 sweep's per-budget fracs, applied as a band to each arm), "
        "measured rel_degradation 0.50%/0.60% at b2.2/b2.5, well inside the "
        "5% gate. Band is a gpt2-fit ESTIMATE pending exact Llama tier "
        "counts at the refit. TQ baseline rows carry no spectral pack and "
        "are accounting-mode-invariant. "
        + " | ".join(crossover_lines)
        + ". docs/2026-07-15-k4-duel-results.md, "
        "docs/2026-07-25-k4-local-levers-results.md."
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.text(0.01, 0.01, caption, fontsize=6, wrap=True, va="bottom")

    png = out_dir / "k4_bits_vs_context.png"
    pdf = out_dir / "k4_bits_vs_context.pdf"
    fig.savefig(png, dpi=150, **_SAVEFIG_KW)
    fig.savefig(pdf, **_SAVEFIG_KW)
    plt.close(fig)
    return png, pdf


# ---------------------------------------------------------------------------
# Figure 2: corpus-transfer matrix + synthesis ladder (gpt2, mechanism)
# ---------------------------------------------------------------------------

_NATURAL_FIT = ("wiki", "code", "null")
_EVAL_SIDES = ("wiki", "code")
_SYNTH_ORDER = ("shuf", "uni", "bi")  # -> f"{order}{eval_side}"


def make_corpus_transfer_figure(
    verdict: dict, budget: str, out_dir: Path
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pb = verdict["per_budget"][budget]
    D = pb["D"]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left: D-matrix heatmap, fit corpus (rows) x eval corpus (cols). ---
    fit_rows = list(_NATURAL_FIT)
    import numpy as np

    mat = np.full((len(fit_rows), len(_EVAL_SIDES)), np.nan)
    for i, fit_c in enumerate(fit_rows):
        for j, eval_c in enumerate(_EVAL_SIDES):
            if fit_c == eval_c:
                mat[i, j] = 0.0  # matched cell: D undefined (denominator), shown as 0
                continue
            cell = D.get(f"{fit_c}->{eval_c}")
            if cell is not None:
                mat[i, j] = cell["mean"]

    im = ax_l.imshow(mat, cmap="RdYlGn_r", vmin=0.0, vmax=0.6, aspect="auto")
    ax_l.set_xticks(range(len(_EVAL_SIDES)))
    ax_l.set_xticklabels([f"eval={c}" for c in _EVAL_SIDES])
    ax_l.set_yticks(range(len(fit_rows)))
    ax_l.set_yticklabels([f"fit={c}" for c in fit_rows])
    for i in range(len(fit_rows)):
        for j in range(len(_EVAL_SIDES)):
            v = mat[i, j]
            if fit_rows[i] == _EVAL_SIDES[j]:
                txt, color = "matched\n(ref)", "black"
            elif np.isnan(v):
                txt, color = "n/a", "gray"
            else:
                label = D[f"{fit_rows[i]}->{_EVAL_SIDES[j]}"]["label"]
                txt, color = f"{v:.2f}\n{label}", "black" if v < 0.35 else "white"
            ax_l.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax_l, label="D (1 - win(fit!=eval)/win(fit=eval))")
    ax_l.set_title(f"D matrix (budget={budget}, gpt2-scale)")

    # --- Right: synthesis order ladder, grouped bars per eval side. -------
    synth = pb.get("synthesis", {}).get("rules", {})
    x = np.arange(len(_EVAL_SIDES))
    width = 0.22
    order_keys = {"shuf": "D_shuf", "uni": "D_uni", "bi": "D_bi"}
    colors = {"shuf": "#7f7f7f", "uni": "#1f77b4", "bi": "#2ca02c"}
    for k, order in enumerate(_SYNTH_ORDER):
        vals = []
        for eval_c in _EVAL_SIDES:
            rule = synth.get(eval_c, {})
            vals.append(rule.get(order_keys[order]))
        offset = (k - 1) * width
        bar_x = x + offset
        heights = [v if v is not None else 0.0 for v in vals]
        ax_r.bar(bar_x, heights, width=width, label=order, color=colors[order])
        for bx, v in zip(bar_x, vals):
            if v is not None:
                ax_r.text(bx, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)

    ax_r.axhline(0.10, color="black", ls="--", lw=1.2, label="10% rule-a line")
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([f"eval={c}" for c in _EVAL_SIDES])
    ax_r.set_ylabel("D (vs matched fit=eval)")
    ax_r.set_title("Synthesis-order ladder (fit-side-only arms)")
    ax_r.set_ylim(0, ax_r.get_ylim()[1] * 1.18)
    ax_r.legend(fontsize=8, loc="upper center", ncol=4)
    ax_r.grid(alpha=0.2, axis="y")

    caption = (
        f"{GPT2_YELLOW_FLAG} All values read verbatim from "
        f"corpus_transfer_verdict.json (per_budget['{budget}']); nothing "
        "recomputed here. Left: D = 1 - win(fit!=eval)/win(fit=eval), mean "
        "over eval caches; matched fit==eval cells are the D=0 reference, "
        "not a measured D. Right: shuf/uni/bi are the fit-side-only "
        "synthesis-order arms (shufcode/uniwiki+unicode/biwiki+bicode); the "
        "10% line is the pre-registered rule-(a) recipe-confirmed threshold "
        "(D_uni < 0.10). Verdict rule: "
        f"{verdict['verdict_rule']}"
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.text(0.01, 0.01, caption, fontsize=6, wrap=True, va="bottom")

    png = out_dir / "k4_corpus_transfer.png"
    pdf = out_dir / "k4_corpus_transfer.pdf"
    fig.savefig(png, dpi=150, **_SAVEFIG_KW)
    fig.savefig(pdf, **_SAVEFIG_KW)
    plt.close(fig)
    return png, pdf


# ---------------------------------------------------------------------------
# Figure 3: per-rank subspace overlap (gpt2, mechanism)
# ---------------------------------------------------------------------------


def make_overlap_figure(overlap_df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = overlap_df[
        (overlap_df["kind"] == "overlap") & (~overlap_df["centered"].astype(bool))
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {"wiki-code": "#1f77b4", "wiki-null": "#ff7f0e", "code-null": "#2ca02c"}
    for pair, g in sub.groupby("pair"):
        curve = (
            g.groupby("rank")["value"].mean().sort_index()
        )  # mean over layers, per rank
        ax.plot(
            curve.index,
            curve.values,
            "o-",
            color=colors.get(pair, None),
            label=pair,
            lw=2,
        )

    ax.set_xlabel("rank cutoff r")
    ax.set_ylabel("mean squared principal cosine (subspace overlap), mean over layers")
    ax.set_ylim(0, 1.02)
    ax.set_title("Per-rank subspace overlap between corpus-fitted bases (gpt2-scale)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    caption = (
        f"{GPT2_YELLOW_FLAG} overlap.parquet, kind=='overlap' & "
        "centered==False (uncentered E[kk^T] fit, the pack's own whitener); "
        "mean squared principal cosine between span(dec[:, :r]) of two "
        "corpus fits, mean over layers, at each rank cutoff r in "
        "overlap_ranks. 1.0 = identical subspaces; wiki-code/wiki-null/"
        "code-null are the three corpus pairs the Task-6 diagnostics cover."
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.text(0.01, 0.01, caption, fontsize=6, wrap=True, va="bottom")

    png = out_dir / "k4_overlap.png"
    pdf = out_dir / "k4_overlap.pdf"
    fig.savefig(png, dpi=150, **_SAVEFIG_KW)
    fig.savefig(pdf, **_SAVEFIG_KW)
    plt.close(fig)
    return png, pdf


# ---------------------------------------------------------------------------
# Figure 4: certificate-vs-measured (instrument validation, gpt2, mechanism)
# ---------------------------------------------------------------------------

_TIER_THRESHOLD_MARKERS = {4: "o", 5: "s", 6: "^", 8: "D"}
_TIER_THRESHOLD_COLORS = {
    4: "#1f77b4",
    5: "#2ca02c",
    6: "#ff7f0e",
    8: "#d62728",
}


def make_cert_vs_measured_figure(
    cvm_df: pd.DataFrame, out_dir: Path
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    for t, g in cvm_df.groupby("tier_threshold"):
        ax.scatter(
            g["implied_rel_degradation"],
            g["measured_rel_deg"],
            marker=_TIER_THRESHOLD_MARKERS.get(int(t), "x"),
            color=_TIER_THRESHOLD_COLORS.get(int(t), "gray"),
            s=28,
            alpha=0.75,
            label=f"tier_threshold={int(t)}",
        )

    lo = min(cvm_df["implied_rel_degradation"].min(), cvm_df["measured_rel_deg"].min())
    hi = max(cvm_df["implied_rel_degradation"].max(), cvm_df["measured_rel_deg"].max())
    lo = lo * 0.8 if lo > 0 else lo
    hi = hi * 1.2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x (perfect agreement)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("certificate implied_rel_degradation (offline, closed-form)")
    ax.set_ylabel("measured_rel_deg (real logit-distortion measurement)")
    ax.set_title("Certificate vs measured (int8-tier degradation, gpt2-scale)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2, which="both")
    n = len(cvm_df)
    below = int((cvm_df["measured_rel_deg"] <= cvm_df["implied_rel_degradation"]).sum())
    ax.annotate(
        f"certificate is conservative in aggregate (~3-4x over-estimate,\n"
        f"{below}/{n} points at or below y=x); layer 2 is the lone\n"
        "consistent exception (measured 1.2-2.9x over implied, all budgets)",
        xy=(0.42, 0.05),
        xycoords="axes fraction",
        fontsize=7.5,
        ha="left",
    )

    caption = (
        f"{GPT2_YELLOW_FLAG} results/k4_dec_quant/20260724-125348-0f49e32/"
        f"cert_vs_measured.parquet, {n} points ({below}/{n} at or below y=x). "
        "x = the offline closed-form int8-tier certificate's per-layer "
        "implied_rel_degradation (no cache re-scored); y = the same layer's "
        "measured relative logit-distortion degradation. One marker per "
        "tier_threshold in {4,5,6,8}. The instrument-validation evidence for "
        "the cheap-analytic-instruments pattern used throughout K4: the "
        "certificate is a conservative over-estimate in aggregate (~3-4x "
        "throughout, e.g. T=5 layer-mean implied 1.35%/1.39% vs measured "
        "0.36%/0.43% at b2.2/b2.5, docs/2026-07-25-k4-local-levers-results.md) "
        "-- but NOT conservative at every point: layer 2's 8 rows (both "
        "budgets, all 4 tier_thresholds) all measure 1.2-2.9x ABOVE their "
        "implied value, the one layer where the certificate under-states "
        "measured degradation. Ordering (T4<T5<T6<blanket) still holds with "
        "zero sign flips at every layer including layer 2 -- the exception "
        "is a magnitude miscalibration, not a decision-rule failure. Gating "
        "on the aggregate/mean certificate remains conservative; gating on "
        "a single layer's certificate value is not guaranteed to be."
    )
    fig.tight_layout(rect=(0, 0.19, 1, 1))
    fig.text(0.01, 0.01, caption, fontsize=6, wrap=True, va="bottom")

    png = out_dir / "k4_cert_vs_measured.png"
    pdf = out_dir / "k4_cert_vs_measured.pdf"
    fig.savefig(png, dpi=150, **_SAVEFIG_KW)
    fig.savefig(pdf, **_SAVEFIG_KW)
    plt.close(fig)
    return png, pdf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Config:
    niah_run_dirs: tuple[str, ...] = (
        "results/k3_niah/20260715-080730-21e6d81",
        "results/k3_niah/20260715-080909-21e6d81",
        "results/k3_niah/20260715-081107-21e6d81",
        "results/k3_niah/20260715-081253-21e6d81",
        "results/k3_niah/20260715-110927-21e6d81",
        "results/k3_niah/20260715-111108-21e6d81",
        "results/k3_niah/20260715-111257-21e6d81",
        "results/k3_niah/20260715-111443-21e6d81",
        "results/k3_niah/20260715-111948-21e6d81",
    )
    fit_packs_parquet: str = (
        "results/k4_fit_packs/20260713-133628-798d0ef/metrics.parquet"
    )
    corpus_transfer_run_dir: str = "results/k4_corpus_transfer/20260723-220816-9d11538"
    dec_quant_run_dir: str = "results/k4_dec_quant/20260724-125348-0f49e32"
    charge_budgets: tuple[float, ...] = (2.2, 2.5)
    transfer_budget: str = "2.5"
    C: int = 1024
    group: int = 64
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8)
    out_dir: str = "results/figures"


def main(cfg: Config) -> None:
    out_dir = Path(cfg.out_dir)

    bits_result = _build_bits_vs_context(
        cfg.niah_run_dirs,
        cfg.fit_packs_parquet,
        cfg.charge_budgets,
        cfg.C,
        cfg.group,
        cfg.tiers,
    )
    p1_png, p1_pdf = make_bits_vs_context_figure(bits_result, out_dir)
    print(f"-> {p1_png}")
    print(f"-> {p1_pdf}")

    transfer_dir = Path(cfg.corpus_transfer_run_dir)
    verdict = json.loads((transfer_dir / "corpus_transfer_verdict.json").read_text())
    p2_png, p2_pdf = make_corpus_transfer_figure(verdict, cfg.transfer_budget, out_dir)
    print(f"-> {p2_png}")
    print(f"-> {p2_pdf}")

    overlap_df = pd.read_parquet(transfer_dir / "overlap.parquet")
    p3_png, p3_pdf = make_overlap_figure(overlap_df, out_dir)
    print(f"-> {p3_png}")
    print(f"-> {p3_pdf}")

    dec_quant_dir = Path(cfg.dec_quant_run_dir)
    cvm_df = pd.read_parquet(dec_quant_dir / "cert_vs_measured.parquet")
    p4_png, p4_pdf = make_cert_vs_measured_figure(cvm_df, out_dir)
    print(f"-> {p4_png}")
    print(f"-> {p4_pdf}")


if __name__ == "__main__":
    main(tyro.cli(Config))
