"""Figure for k3_longbench: mean code_sim per arm per task (vs fp16 reference).

Reads the parquet, never refits. Select runs explicitly upstream.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def make_figures(df, out_dir: str) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tasks = sorted(df["task"].unique())
    arms = sorted(df["arm"].unique())
    x = range(len(tasks))
    width = 0.8 / max(len(arms), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, arm in enumerate(arms):
        g = df[df["arm"] == arm]
        means = [
            g[g["task"] == t]["code_sim"].mean() if not g[g["task"] == t].empty else 0.0
            for t in tasks
        ]
        comp = g["compression"].iloc[0] if not g.empty else 1.0
        offs = [xi + i * width for xi in x]
        ax.bar(offs, means, width=width, label=f"{arm} ({comp:.1f}×)")

    # fp16 reference line (mean across its tasks), if present.
    if "fp16" in arms:
        fp16_mean = df[df["arm"] == "fp16"]["code_sim"].mean()
        ax.axhline(
            fp16_mean, ls="--", color="gray", lw=1, label="fp16 mean (reference)"
        )

    ax.set_xticks([xi + width * (len(arms) - 1) / 2 for xi in x])
    ax.set_xticklabels(tasks)
    ax.set_ylabel("code_sim (edit-similarity, 0–1)")
    ax.set_title("LongBench Code recall per arm under KV compression")
    ax.legend()
    p = out / "longbench_code_sim.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [p]
