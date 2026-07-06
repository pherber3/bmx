"""Publication Pareto figure: quality-DELTA vs KV bits/entry, ours vs TurboQuant's published.

Reads one k3_longbench run's metrics.parquet (explicit --run-dir, mirrors
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


@dataclasses.dataclass
class ParetoPoint:
    label: str
    x_bits: float
    y_delta: float
    ours: bool


def _task_level_mean(g: pd.DataFrame) -> float:
    """The 'Average (all tasks)' aggregation from plot_longbench_table.py: equal task weight."""
    return float(g["code_sim"].mean() * 100.0)


def build_pareto(df: pd.DataFrame) -> dict:
    """Turn a k3_longbench metrics frame + the pinned PUBLISHED dict into Pareto-plot points.

    Returns a dict with:
      - "ours": list[ParetoPoint] (measured arms, y = delta from OUR fp16 task-level mean)
      - "ours_abs": list[ParetoPoint] (measured arms, y = absolute task-level mean; for Panel B)
      - "theirs": list[ParetoPoint] (published rows, y = delta from published Full Cache Avg)
      - "ours_frontier" / "theirs_frontier": the non-dominated (upper-envelope) subset of each,
        sorted by x ascending.
      - "fp16_abs": our fp16 arm's absolute task-level mean (Panel B reference line).

    Raises if the parquet has no 'fp16' arm — delta is undefined without that anchor.
    """
    if "fp16" not in set(df["arm"].unique()):
        raise ValueError(
            "no 'fp16' arm in this parquet: delta-from-own-full-cache is undefined without "
            "the fp16 anchor row"
        )
    df = df[~df["arm"].isin(EXCLUDED_ARMS)].copy()

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

    return {
        "ours": ours,
        "ours_abs": ours_abs,
        "theirs": theirs,
        "ours_frontier": _upper_envelope(ours),
        "theirs_frontier": _upper_envelope(theirs),
        "fp16_abs": fp16_abs,
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


def make_figure(result: dict, run_id: str, out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel A: delta from own-harness fp16/Full-Cache, ours vs theirs ---
    ours = result["ours"]
    theirs = result["theirs"]
    ax_a.scatter(
        [p.x_bits for p in ours],
        [p.y_delta for p in ours],
        marker="o",
        s=60,
        color="#1f77b4",
        label="ours (measured)",
        zorder=3,
    )
    for p in ours:
        ax_a.annotate(
            p.label,
            (p.x_bits, p.y_delta),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax_a.scatter(
        [p.x_bits for p in theirs],
        [p.y_delta for p in theirs],
        marker="o",
        s=60,
        facecolors="none",
        edgecolors="#d62728",
        label="TurboQuant (published)",
        zorder=3,
    )
    for p in theirs:
        ax_a.annotate(
            p.label.replace(" (published)", ""),
            (p.x_bits, p.y_delta),
            fontsize=7,
            xytext=(4, -10),
            textcoords="offset points",
            color="#d62728",
        )
    of = result["ours_frontier"]
    tf = result["theirs_frontier"]
    ax_a.plot(
        [p.x_bits for p in of],
        [p.y_delta for p in of],
        "-",
        color="#1f77b4",
        lw=1.5,
        zorder=2,
    )
    ax_a.plot(
        [p.x_bits for p in tf],
        [p.y_delta for p in tf],
        "--",
        color="#d62728",
        lw=1.5,
        zorder=2,
    )
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
    for p in ours_abs:
        ax_b.annotate(
            p.label,
            (p.x_bits, p.y_delta),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax_b.axhline(
        result["fp16_abs"], ls="--", color="gray", lw=1, label="our fp16 (reference)"
    )
    ax_b.set_xlabel("KV bits/entry (measured kv_size_bits)")
    ax_b.set_ylabel("Avg (task-level mean, our harness, absolute)")
    ax_b.set_title(
        "Panel B: absolute quality, OUR arms only (no cross-harness absolute claim)"
    )
    ax_b.legend(fontsize=7, loc="lower right")

    caption = (
        f"run_id={run_id}. Delta-parity rationale: absolute LongBench Code scores differ "
        f"~15pts across harnesses from invisible prompt-policy differences, not quantization "
        f"({ANCHOR_FORENSICS_DOC}); deltas from each harness's own full-cache Avg cancel that "
        f"offset and are the only cross-harness-comparable quantity (Panel A). Panel B avoids "
        f"absolute cross-harness comparison entirely. Bit-width caveat: {BITWIDTH_CAVEAT} "
        f"kivi arm excluded ({KIVI_DIAGNOSIS_DOC}). Y aggregation is the task-level mean "
        f"(plot_longbench_table.py's 'Average (all tasks)'), equal weight per task, not the "
        f"mean of 6 category means."
    )
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
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
    run_dir: str  # explicit path to a k3_longbench run dir (contains metrics.parquet)


def main(cfg: Config) -> None:
    run_path = Path(cfg.run_dir)
    df = pd.read_parquet(run_path / "metrics.parquet")
    result = build_pareto(df)
    png, pdf, caption_path = make_figure(result, run_id=run_path.name, out_dir=run_path)
    print(f"-> {png}")
    print(f"-> {pdf}")
    print(f"-> {caption_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
