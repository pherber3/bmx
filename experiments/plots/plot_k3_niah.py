"""Figures for k3_niah: recall vs length per arm, and length×depth recall heatmaps.

Reads the parquet, never refits. Select runs explicitly upstream (newest_run_with);
this module only renders a passed-in DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def make_figures(df, out_dir: str) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # --- Figure 1: recall vs length, one line per arm, annotated with compression. ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for arm, g in df.groupby("arm"):
        gl = g.groupby("length")["recall"].mean().sort_index()
        comp = g["compression"].iloc[0]
        ax.plot(gl.index, gl.values, marker="o", label=f"{arm} ({comp:.1f}×)")
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("recall (ROUGE-1 ×10, mean over depth)")
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
        for ax, arm in zip(axes[0], arms):
            g = df[df["arm"] == arm]
            grid = np.full((len(depths), len(lengths)), np.nan)
            for _, r in g.iterrows():
                grid[depths.index(r["depth"]), lengths.index(r["length"])] = r["recall"]
            im = ax.imshow(
                grid, aspect="auto", vmin=0, vmax=10, origin="lower", cmap="viridis"
            )
            ax.set_xticks(range(len(lengths)))
            ax.set_xticklabels([str(x) for x in lengths])
            ax.set_yticks(range(len(depths)))
            ax.set_yticklabels([f"{d:.0%}" for d in depths])
            ax.set_xlabel("length")
            ax.set_ylabel("depth")
            ax.set_title(arm)
        fig.colorbar(im, ax=axes[0].tolist(), label="recall (0–10)")
        p2 = out / "niah_recall_heatmap.png"
        fig.savefig(p2, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(p2)

    return paths


if __name__ == "__main__":
    import sys

    import pandas as pd

    df = pd.read_parquet(sys.argv[1])
    print(make_figures(df, sys.argv[2] if len(sys.argv) > 2 else "."))
