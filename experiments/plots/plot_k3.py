"""K3 figures: quality-vs-bpe and retrieval-vs-arm. Reads parquet, never refits."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def make_figures(df, out_dir: str) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    # Quality vs bits: avg bpe (K,V) on x, ppl on y, one point per arm.
    fig, ax = plt.subplots()
    df = df.copy()
    df["bpe_avg"] = (df["bpe_k"] + df["bpe_v"]) / 2
    for _, r in df.iterrows():
        ax.scatter(r["bpe_avg"], r["ppl"], label=r["arm"])
        ax.annotate(r["arm"], (r["bpe_avg"], r["ppl"]))
    ax.set_xlabel("avg bits/entry (honest)")
    ax.set_ylabel("live-generation perplexity")
    ax.set_title("K3: quality vs bits, live generation")
    p1 = out / "k3_quality_vs_bpe.png"
    fig.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p1)

    return paths


if __name__ == "__main__":
    import sys

    import pandas as pd

    df = pd.read_parquet(sys.argv[1])
    print(make_figures(df, sys.argv[2] if len(sys.argv) > 2 else "."))
