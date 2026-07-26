"""Figures for k3_niah: recall vs length per arm, and length×depth recall heatmaps.

Reads the parquet, never refits. Select runs explicitly upstream (newest_run_with);
this module only renders a passed-in DataFrame.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def make_figures(df, out_dir: str) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # recall_full (precision-free ROUGE-1 recall) is the headline; instruct-model verbosity
    # makes the F-measure `recall` column read low even when retrieval is perfect.
    metric = "recall_full" if "recall_full" in df.columns else "recall"

    # --- Figure 1: recall vs length, one line per arm, annotated with compression. ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for arm, g in df.groupby("arm"):
        gl = g.groupby("length")[metric].mean().sort_index()
        first = g.sort_values("length").iloc[0]
        comp = first["compression"]
        first_len = int(first["length"])
        ax.plot(
            gl.index,
            gl.values,
            marker="o",
            label=f"{arm} ({comp:.1f}× @{first_len})",
        )
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("ROUGE-1 recall ×10 (mean over depth)")
    ax.set_title("NIAH recall vs length under KV compression")
    ax.legend()
    p1 = out / "niah_recall_vs_length.png"
    fig.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p1)

    # --- Figure 2: length×depth recall heatmap per arm (paper's view). ---
    if "depth" in df.columns and df["depth"].nunique() > 1:
        arms = sorted(df["arm"].unique())
        fig, axes = plt.subplots(
            1, len(arms), figsize=(5 * len(arms), 4), squeeze=False
        )
        lengths = sorted(df["length"].unique())
        depths = sorted(df["depth"].unique())
        scores: dict[str, float] = {}
        for ax, arm in zip(axes[0], arms):
            g = df[df["arm"] == arm]
            # Pivot to the (depth × length) grid directly; reindex to the full axes
            # so missing cells stay NaN (masked), repeated cells average.
            grid = (
                g.pivot_table(index="depth", columns="length", values=metric)
                .reindex(index=depths, columns=lengths)
                .to_numpy()
            )
            im = ax.imshow(
                grid, aspect="auto", vmin=0, vmax=10, origin="lower", cmap="viridis"
            )
            # TurboQuant Fig-4 parity: one 0–1 aggregate above each arm's grid.
            # recall is on the 0–10 scale (ROUGE-1 ×10); divide to land in [0, 1].
            # All-NaN grid (an arm missing every cell at some length) → nan, not a crash.
            score = (
                float("nan")
                if np.all(np.isnan(grid))
                else float(np.nanmean(grid)) / 10.0
            )
            scores[str(arm)] = score
            ax.set_xticks(range(len(lengths)))
            ax.set_xticklabels([str(x) for x in lengths])
            ax.set_yticks(range(len(depths)))
            ax.set_yticklabels([f"{d:.0%}" for d in depths])
            ax.set_xlabel("length")
            ax.set_ylabel("depth")
            ax.set_title(f"{arm}\nScore: {score:.3f}")
        fig.colorbar(im, ax=axes[0].tolist(), label="recall (0–10)")
        p2 = out / "niah_recall_heatmap.png"
        fig.savefig(p2, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(p2)

        # Machine-readable sidecar: arm → aggregate score, reusable by the writeup.
        p3 = out / "niah_heatmap_scores.json"
        p3.write_text(json.dumps(scores, indent=2))
        paths.append(p3)

    return paths


if __name__ == "__main__":
    import sys

    import pandas as pd

    df = pd.read_parquet(sys.argv[1])
    print(make_figures(df, sys.argv[2] if len(sys.argv) > 2 else "."))
