"""Publication Pareto figure: quality-DELTA vs KV bits/entry, ours vs TurboQuant's published.

Reads one or more k3_longbench runs' metrics.parquet (explicit --run-dirs, mirrors
plot_longbench_table.py's discipline — never a blind concat of a results root) and never
refits: every score plotted here is exactly a per-item metric k3_longbench already computed,
aggregated the same way plot_longbench_table.py calls "Average (all tasks)" (the task-level
mean, i.e. mean over every task-level score actually present, NOT the mean-of-6-category-
means). That choice is deliberate and is restated in the caption.

Why DELTAS, not absolute scores (docs/2026-07-06-anchor-forensics-results.md): our absolute
LongBench Code numbers run ~15 points above TurboQuant's published Table-1 Code row for
reasons traced to invisible prompt-policy differences (chat-wrap vs raw template), not to
quantization. Absolute cross-harness comparison is therefore not licensed. Delta-from-own-
full-cache-Avg IS licensed: each harness's compression arms are measured against that same
harness's own fp16/Full-Cache row, so harness-specific offsets cancel. This module computes
ours from the parquet's own fp16 rows and theirs from PUBLISHED's own Full Cache row.

The 'kivi' arm is excluded (docs/2026-07-04-kivi-arm-diagnosis.md): our 'kivi' arm is a
symmetric-RTN strawman that collapses at 2 bits for reasons unrelated to real KIVI's behavior,
not a faithful implementation — plotting it against published KIVI rows would mislabel it.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import matplotlib
import pandas as pd
import tyro

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.plot_longbench_table import (
    EXCLUDED_ARMS,
    PUBLISHED,
    KIVI_DIAGNOSIS_DOC,
)

FULL_CACHE_KEY = "Full Cache (published)"
ANCHOR_FORENSICS_DOC = "docs/2026-07-06-anchor-forensics-results.md"

# TurboQuant's 2.5-bit row uses an outlier LongBench split not reproduced on our path — flag
# it wherever the bit-width axis is rendered (task brief's "bit-width caveat").
BITWIDTH_CAVEAT = (
    "TurboQuant's 2.5-bit row uses an outlier LongBench split; not an apples-to-apples "
    "bit-width match to our measured arms."
)

# Panel A y-axis clip: turboquant_prod's delta (~-35) otherwise stretches the axis and
# squashes the interesting [-10, +1] region where every other arm lives. Points beyond the
# clip are drawn AT the clip edge with an open down-arrow marker; the true value is preserved
# in the point's annotation and in the returned `points_below_clip` bookkeeping (never dropped
# from the data, only from the rendered y-range).
PANEL_A_YLIM = (-12.0, 1.5)

# Input-policy columns that must agree across concatenated run dirs (same-inputs discipline —
# mixing truncated/untruncated LongBench rows has burned this program twice; see
# docs/2026-07-06-anchor-forensics-results.md and plot_longbench_table's parity_warning).
POLICY_COLUMNS = ("longbench_version", "max_prompt_tokens")


@dataclasses.dataclass
class ParetoPoint:
    label: str
    x_bits: float
    y_delta: float
    ours: bool


def load_run_dirs(
    run_dirs: tuple[str, ...], allow_arm_override: bool = False
) -> tuple[pd.DataFrame, str]:
    """Load + concat metrics.parquet from multiple explicit run dirs (never a glob of a
    results root — plot-discipline). Returns (df, fp16_run_dir) where fp16_run_dir is the
    first dir listed (the fp16 anchor must come from there; documented at the call site).

    Schemas differ across code vintages (old parquets lack longbench_version/
    max_prompt_tokens/chat_wrap) — concat with NaN fill via plain pd.concat, which already
    NaN-fills missing columns across frames.

    Guards:
      - HARD error if the same arm appears in more than one run dir (ambiguous row
        provenance), unless allow_arm_override=True, in which case the LAST dir listed wins
        and a warning is printed. Exception: 'fp16' is exempt from override — the fp16 anchor
        must come from the first dir listed regardless of allow_arm_override (see below), so a
        duplicate 'fp16' in a later dir still only silences the ambiguity for its OWN row
        selection; if the first dir's fp16 rows are the ones displaced by an override the
        anchor lookup below still fails, by design.
      - HARD error on input-policy mismatch: longbench_version/max_prompt_tokens must agree
        across dirs where the column exists. A dir with the column absent (pre-W3 parquet) is
        treated as compatible with the untruncated v1 policy of the 60h table run — i.e. NaN
        is NOT flagged as a mismatch, only a genuine disagreement between two *present* values
        is. This is a deliberate policy call, not an oversight: the 60h run predates the
        columns entirely, so absence there means "that run's actual policy was the original
        untruncated v1 sweep," not "unknown."
    """
    if not run_dirs:
        raise ValueError("run_dirs must be non-empty")

    frames = []
    arm_to_dir: dict[str, str] = {}
    for d in run_dirs:
        p = Path(d)
        df = pd.read_parquet(p / "metrics.parquet")
        df["_run_dir"] = str(p)
        df["_run_id"] = p.name
        frames.append(df)

        for arm in df["arm"].unique():
            if arm in arm_to_dir and arm_to_dir[arm] != str(p):
                if not allow_arm_override:
                    raise ValueError(
                        f"arm '{arm}' appears in both {arm_to_dir[arm]} and {p} — "
                        f"ambiguous row provenance across --run-dirs. Pass "
                        f"--allow-arm-override to let the last-listed dir win."
                    )
                warnings.warn(
                    f"arm '{arm}' overridden: {arm_to_dir[arm]} -> {p} "
                    f"(--allow-arm-override)",
                    stacklevel=2,
                )
            arm_to_dir[arm] = str(p)

    # Input-policy consistency. Absence (NaN, pre-W3 parquet) MEANS the untruncated-v1
    # policy (per the docstring above) — so NaN is normalized to that policy's canonical
    # value and then compared like any other: a pre-W3 dir + a truncated dir MUST clash.
    # (The first implementation only compared present values, which silently merged the
    # 60h table with a truncated run — the exact violation this guard exists to catch.)
    _NAN_MEANS = {"max_prompt_tokens": -1.0, "longbench_version": "v1"}
    for col in POLICY_COLUMNS:
        seen: dict[object, str] = {}
        for df in frames:
            run_dir = df["_run_dir"].iloc[0]
            if col not in df.columns:
                vals = [_NAN_MEANS[col]] if col in _NAN_MEANS else []
            else:
                vals = [
                    _NAN_MEANS.get(col, v) if pd.isna(v) else v
                    for v in df[col].unique()
                ]
            for v in vals:
                if seen and v not in seen and any(v != sv for sv in seen):
                    conflicting_dir = next(iter(seen.values()))
                    raise ValueError(
                        f"input-policy mismatch on column '{col}': {run_dir} has "
                        f"{v!r}, {conflicting_dir} has {list(seen.keys())!r} — mixing "
                        f"truncated/untruncated (or version-mismatched) rows is the "
                        f"same-inputs violation this program has been burned by twice. "
                        f"Fix the run selection or split into separate figures."
                    )
                seen[v] = run_dir

    # Apply arm overrides: when allow_arm_override, drop earlier-dir copies of any
    # overridden arm before concatenating, so only the last-listed dir's rows for that arm
    # survive.
    if allow_arm_override:
        for i, df in enumerate(frames):
            run_dir = str(Path(run_dirs[i]))
            keep_mask = df["arm"].map(lambda a: arm_to_dir.get(a) == run_dir)
            frames[i] = df[keep_mask]

    combined = pd.concat(frames, ignore_index=True)
    fp16_dir = str(Path(run_dirs[0]))
    fp16_rows = combined[
        (combined["arm"] == "fp16") & (combined["_run_dir"] == fp16_dir)
    ]
    if fp16_rows.empty:
        raise ValueError(
            f"no 'fp16' arm found in the FIRST --run-dirs entry ({run_dirs[0]}) — the fp16 "
            f"anchor must come from the first dir listed, by convention (documented in "
            f"load_run_dirs)."
        )
    return combined, fp16_dir


def _task_level_mean(g: pd.DataFrame) -> float:
    """The 'Average (all tasks)' aggregation from plot_longbench_table.py: equal task weight."""
    return float(g["code_sim"].mean() * 100.0)


def build_pareto(
    df: pd.DataFrame,
    fp16_run_dir: str | None = None,
    y_clip: tuple[float, float] | None = PANEL_A_YLIM,
) -> dict:
    """Turn a k3_longbench metrics frame + the pinned PUBLISHED dict into Pareto-plot points.

    Returns a dict with:
      - "ours": list[ParetoPoint] (measured arms, y = delta from OUR fp16 task-level mean)
      - "ours_abs": list[ParetoPoint] (measured arms, y = absolute task-level mean; for Panel B)
      - "theirs": list[ParetoPoint] (published rows, y = delta from published Full Cache Avg)
      - "ours_frontier" / "theirs_frontier": the non-dominated (upper-envelope) subset of each,
        sorted by x ascending.
      - "fp16_abs": our fp16 arm's absolute task-level mean (Panel B reference line).
      - "y_clip": the (lo, hi) clip applied to Panel A's y-axis (None if not clipping).
      - "points_below_clip": list[ParetoPoint] (from "ours" + "theirs") whose true y_delta
        falls outside y_clip — the true value is never dropped, only the rendered position;
        make_figure draws these at the clip edge with an open down/up-arrow marker.

    Raises if the parquet has no 'fp16' arm — delta is undefined without that anchor. When
    `fp16_run_dir` is given (multi-run path), the fp16 anchor is computed ONLY from that dir's
    fp16 rows (`_run_dir` column), per load_run_dirs's "first dir wins" convention — other
    dirs' fp16 rows (if any; normally there should be none after the duplicate-arm guard) are
    ignored for anchor purposes but still may appear as non-fp16 arms elsewhere.
    """
    if "fp16" not in set(df["arm"].unique()):
        raise ValueError(
            "no 'fp16' arm in this parquet: delta-from-own-full-cache is undefined without "
            "the fp16 anchor row"
        )
    df = df[~df["arm"].isin(EXCLUDED_ARMS)].copy()

    if fp16_run_dir is not None:
        fp16_mask = (df["arm"] == "fp16") & (df["_run_dir"] == fp16_run_dir)
        if not fp16_mask.any():
            raise ValueError(
                f"no 'fp16' arm rows from the designated fp16_run_dir={fp16_run_dir!r} — "
                f"the fp16 anchor must come from the first --run-dirs entry."
            )
        fp16_abs = _task_level_mean(df[fp16_mask])
    else:
        fp16_abs = _task_level_mean(df[df["arm"] == "fp16"])

    ours: list[ParetoPoint] = []
    ours_abs: list[ParetoPoint] = []
    for arm, g in df.groupby("arm", sort=False):
        x_bits = 16.0 if arm == "fp16" else float(g["kv_size_bits"].mean())
        y_abs = _task_level_mean(g)
        ours.append(ParetoPoint(arm, x_bits, y_abs - fp16_abs, ours=True))
        ours_abs.append(ParetoPoint(arm, x_bits, y_abs, ours=True))

    theirs: list[ParetoPoint] = []
    full_cache_avg = PUBLISHED[FULL_CACHE_KEY]["Avg"]
    for method, vals in PUBLISHED.items():
        theirs.append(
            ParetoPoint(
                method, vals["KV Size (bits)"], vals["Avg"] - full_cache_avg, ours=False
            )
        )

    points_below_clip: list[ParetoPoint] = []
    if y_clip is not None:
        lo, hi = y_clip
        for p in [*ours, *theirs]:
            if p.y_delta < lo or p.y_delta > hi:
                points_below_clip.append(p)

    return {
        "ours": ours,
        "ours_abs": ours_abs,
        "theirs": theirs,
        "ours_frontier": _upper_envelope(ours),
        "theirs_frontier": _upper_envelope(theirs),
        "fp16_abs": fp16_abs,
        "y_clip": y_clip,
        "points_below_clip": points_below_clip,
    }


def _upper_envelope(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """Non-dominated subset: a point (x, y) is dominated if another point has x' <= x and
    y' >= y with at least one strict. Sorted by x ascending. Standard 2D Pareto frontier for
    'fewer bits is better, higher delta is better'."""
    pts = sorted(points, key=lambda p: (p.x_bits, -p.y_delta))
    frontier: list[ParetoPoint] = []
    best_y = float("-inf")
    for p in pts:
        if p.y_delta > best_y:
            frontier.append(p)
            best_y = p.y_delta
    return frontier


# Deterministic label-offset table (offset points, by point index modulo table length) — a
# tiny greedy nudge to de-overlap labels without an adjustText dependency. Points that get a
# non-default offset are drawn with a thin leader line back to their marker so the reader can
# still tell which label belongs to which point.
_LABEL_OFFSETS: list[tuple[int, int]] = [
    (4, 4),
    (4, -12),
    (-38, 8),
    (6, 14),
    (-42, -12),
    (4, 20),
    (-38, -20),
    (8, -4),
]


def _annotate_declustered(
    ax, points: list[ParetoPoint], *, color=None, strip_suffix=""
):
    """Annotate each point with a deterministic offset-by-index, drawing a leader line for
    any point whose offset is not the default (4, 4) so de-overlapped labels stay traceable."""
    for i, p in enumerate(points):
        dx, dy = _LABEL_OFFSETS[i % len(_LABEL_OFFSETS)]
        label = p.label.replace(strip_suffix, "") if strip_suffix else p.label
        ax.annotate(
            label,
            (p.x_bits, p.y_delta),
            fontsize=7,
            xytext=(dx, dy),
            textcoords="offset points",
            color=color,
            arrowprops=(
                {
                    "arrowstyle": "-",
                    "lw": 0.5,
                    "color": color or "gray",
                    "shrinkA": 0,
                    "shrinkB": 3,
                }
                if (dx, dy) != (4, 4)
                else None
            ),
        )


def _plot_clipped(
    ax, points: list[ParetoPoint], y_clip: tuple[float, float] | None, **scatter_kwargs
):
    """Scatter `points`, drawing any point outside y_clip AT the clip edge with an open
    down/up-arrow marker instead of stretching the axis. In-range points use the normal
    marker from scatter_kwargs; out-of-range points are drawn separately with marker='v'/'^'.
    Returns (in_range, clipped) point lists actually rendered at in-range vs clip-edge y."""
    if y_clip is None:
        ax.scatter(
            [p.x_bits for p in points], [p.y_delta for p in points], **scatter_kwargs
        )
        return points, []

    lo, hi = y_clip
    in_range = [p for p in points if lo <= p.y_delta <= hi]
    below = [p for p in points if p.y_delta < lo]
    above = [p for p in points if p.y_delta > hi]

    ax.scatter(
        [p.x_bits for p in in_range], [p.y_delta for p in in_range], **scatter_kwargs
    )
    edge_kwargs = dict(scatter_kwargs)
    edge_kwargs.pop("label", None)
    if below:
        ax.scatter(
            [p.x_bits for p in below],
            [lo] * len(below),
            marker="v",
            facecolors="none",
            edgecolors=edge_kwargs.get("color", edge_kwargs.get("edgecolors", "black")),
            s=edge_kwargs.get("s", 60) * 1.5,
            zorder=5,
        )
    if above:
        ax.scatter(
            [p.x_bits for p in above],
            [hi] * len(above),
            marker="^",
            facecolors="none",
            edgecolors=edge_kwargs.get("color", edge_kwargs.get("edgecolors", "black")),
            s=edge_kwargs.get("s", 60) * 1.5,
            zorder=5,
        )
    # Return points repositioned at their rendered (possibly clipped) y for annotation purposes.
    rendered = (
        in_range
        + [ParetoPoint(p.label, p.x_bits, lo, p.ours) for p in below]
        + [ParetoPoint(p.label, p.x_bits, hi, p.ours) for p in above]
    )
    return rendered, below + above


def make_figure(
    result: dict,
    run_id: str | list[str],
    out_dir: Path,
    run_ids: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    """run_id: single-run label kept for backward compatibility; run_ids (optional) lists
    every contributing run's id for the multi-run provenance caption. When run_ids is given,
    it is used for the caption's run_id= field instead of run_id."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel A: delta from own-harness fp16/Full-Cache, ours vs theirs ---
    ours = result["ours"]
    theirs = result["theirs"]
    y_clip = result.get("y_clip")

    ours_rendered, ours_clipped = _plot_clipped(
        ax_a,
        ours,
        y_clip,
        marker="o",
        s=60,
        color="#1f77b4",
        label="ours (measured)",
        zorder=3,
    )
    _annotate_declustered(ax_a, ours_rendered)

    theirs_rendered, theirs_clipped = _plot_clipped(
        ax_a,
        theirs,
        y_clip,
        marker="o",
        s=60,
        facecolors="none",
        edgecolors="#d62728",
        label="TurboQuant (published)",
        zorder=3,
    )
    _annotate_declustered(
        ax_a, theirs_rendered, color="#d62728", strip_suffix=" (published)"
    )

    # Annotate true (unclipped) values for any point drawn at the clip edge, near the arrow
    # marker, so the caption isn't the only place the real number survives.
    for p in ours_clipped + theirs_clipped:
        edge_y = y_clip[0] if p.y_delta < y_clip[0] else y_clip[1]
        ax_a.annotate(
            f"{p.label} ({p.y_delta:+.1f})",
            (p.x_bits, edge_y),
            fontsize=6.5,
            xytext=(4, -16 if p.y_delta < y_clip[0] else 10),
            textcoords="offset points",
            color="dimgray",
            style="italic",
        )

    of = result["ours_frontier"]
    tf = result["theirs_frontier"]
    of_y = [
        max(y_clip[0], min(y_clip[1], p.y_delta)) if y_clip else p.y_delta for p in of
    ]
    tf_y = [
        max(y_clip[0], min(y_clip[1], p.y_delta)) if y_clip else p.y_delta for p in tf
    ]
    ax_a.plot([p.x_bits for p in of], of_y, "-", color="#1f77b4", lw=1.5, zorder=2)
    ax_a.plot([p.x_bits for p in tf], tf_y, "--", color="#d62728", lw=1.5, zorder=2)
    ax_a.scatter(
        [16],
        [0],
        marker="*",
        s=200,
        color="black",
        zorder=4,
        label="fp16 / Full Cache (ref)",
    )
    ax_a.axhline(0, color="gray", lw=0.6)
    if y_clip is not None:
        ax_a.set_ylim(*y_clip)
    ax_a.set_xlabel(
        "KV bits/entry (ours: measured kv_size_bits; theirs: published KV Size)"
    )
    ax_a.set_ylabel("Delta Avg vs own-harness full-cache (points)")
    ax_a.set_title("Panel A: quality-delta vs KV bits (cross-harness via deltas)")
    ax_a.legend(fontsize=7, loc="lower right")

    # --- Panel B: absolute, ours only ---
    ours_abs = result["ours_abs"]
    ax_b.scatter(
        [p.x_bits for p in ours_abs],
        [p.y_delta for p in ours_abs],
        marker="o",
        s=60,
        color="#1f77b4",
        zorder=3,
    )
    _annotate_declustered(ax_b, ours_abs)
    ax_b.axhline(
        result["fp16_abs"], ls="--", color="gray", lw=1, label="our fp16 (reference)"
    )
    ax_b.set_xlabel("KV bits/entry (measured kv_size_bits)")
    ax_b.set_ylabel("Avg (task-level mean, our harness, absolute)")
    ax_b.set_title(
        "Panel B: absolute quality, OUR arms only (no cross-harness absolute claim)"
    )
    ax_b.legend(fontsize=7, loc="lower right")

    provenance = ", ".join(run_ids) if run_ids else run_id
    clip_note = ""
    points_below_clip = result.get("points_below_clip") or []
    if y_clip is not None and points_below_clip:
        clip_vals = "; ".join(
            f"{p.label} ({p.y_delta:+.1f})" for p in points_below_clip
        )
        clip_note = (
            f" Panel A y-axis clipped to [{y_clip[0]:.1f}, {y_clip[1]:.1f}] for readability; "
            f"points outside the clip are drawn at the edge with an open arrow marker and "
            f"their true value is preserved here: {clip_vals}."
        )
    caption = (
        f"run_id={provenance}. Delta-parity rationale: absolute LongBench Code scores differ "
        f"~15pts across harnesses from invisible prompt-policy differences, not quantization "
        f"({ANCHOR_FORENSICS_DOC}); deltas from each harness's own full-cache Avg cancel that "
        f"offset and are the only cross-harness-comparable quantity (Panel A). Panel B avoids "
        f"absolute cross-harness comparison entirely. Bit-width caveat: {BITWIDTH_CAVEAT} "
        f"kivi arm excluded ({KIVI_DIAGNOSIS_DOC}). Y aggregation is the task-level mean "
        f"(plot_longbench_table.py's 'Average (all tasks)'), equal weight per task, not the "
        f"mean of 6 category means.{clip_note}"
    )
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.text(0.01, 0.01, caption, fontsize=6, wrap=True, va="bottom")

    png = out_dir / "pareto.png"
    pdf = out_dir / "pareto.pdf"
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)

    caption_path = out_dir / "pareto_caption.txt"
    caption_path.write_text(caption)
    return png, pdf, caption_path


@dataclasses.dataclass
class Config:
    run_dirs: tuple[
        str, ...
    ]  # explicit path(s) to k3_longbench run dir(s); fp16 anchor = 1st
    allow_arm_override: bool = (
        False  # if an arm appears in >1 dir, last dir wins (warns)
    )
    out_dir: str | None = None  # defaults to the first --run-dirs entry


def main(cfg: Config) -> None:
    combined, fp16_dir = load_run_dirs(
        cfg.run_dirs, allow_arm_override=cfg.allow_arm_override
    )
    result = build_pareto(combined, fp16_run_dir=fp16_dir)
    run_ids = [Path(d).name for d in cfg.run_dirs]
    out_path = Path(cfg.out_dir) if cfg.out_dir else Path(cfg.run_dirs[0])
    png, pdf, caption_path = make_figure(
        result, run_id=run_ids[0], out_dir=out_path, run_ids=run_ids
    )
    print(f"-> {png}")
    print(f"-> {pdf}")
    print(f"-> {caption_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
