"""Track B figures from a b1_kernel_bench run: factored-vs-dense speedup curves.

Reads metrics.parquet, writes PNGs into the run dir. Never refits.

    uv run python experiments/plots/plot_b1.py --run results/b1_kernel_bench/<run-id>
"""

import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tyro


@dataclasses.dataclass
class Config:
    run: str
    impl: str = "bmm"  # which factored impl to plot against dense


def main(cfg: Config) -> None:
    run = Path(cfg.run)
    df = pd.read_parquet(run / "metrics.parquet")
    dense = df[df.impl == "dense"].set_index(["m", "h", "ell", "batch"]).ms
    fact = df[df.impl == cfg.impl].set_index(["m", "h", "ell", "batch"]).ms
    speedup = (dense / fact).dropna().rename("speedup").reset_index()

    ds = sorted(speedup.m.unique())
    fig, axes = plt.subplots(1, len(ds), figsize=(5 * len(ds), 4), sharey=True)
    axes = [axes] if len(ds) == 1 else list(axes)
    for ax, d in zip(axes, ds):
        sub = speedup[speedup.m == d]
        for (h, ell), grp in sub.groupby(["h", "ell"]):
            grp = grp.sort_values("batch")
            (line,) = ax.plot(
                grp.batch, grp.speedup, marker="o", label=f"h={h} ℓ={ell}"
            )
            ax.axhline(h / ell, color=line.get_color(), ls=":", lw=0.8, alpha=0.5)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("batch")
        ax.set_title(f"d={d}  ({cfg.impl} vs dense; dotted = ideal h/ℓ)")
    axes[0].set_ylabel("speedup (dense ms / factored ms)")
    axes[-1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = run / f"speedup_{cfg.impl}.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
