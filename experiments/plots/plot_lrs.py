"""Rate–distortion view of Stage B: ip_distortion vs bits-per-weight, one
panel per weight, arms as colors. Reads the newest stage_b.parquet."""

import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tyro


@dataclasses.dataclass
class Config:
    run_dir: str = ""  # default: newest results/lrs_residual run with stage_b


def newest_run(root="results/lrs_residual") -> Path:
    runs = sorted(p for p in Path(root).iterdir() if (p / "stage_b.parquet").exists())
    assert runs, f"no stage_b.parquet under {root}"
    return runs[-1]


def main(cfg: Config) -> None:
    run = Path(cfg.run_dir) if cfg.run_dir else newest_run()
    df = pd.read_parquet(run / "stage_b.parquet")
    weights = sorted(df["weight"].unique())
    fig, axes = plt.subplots(
        1, len(weights), figsize=(5 * len(weights), 4), sharey=True
    )
    for ax, w in zip(axes, weights):
        sub = df[df["weight"] == w]
        for arm, marker in zip(sorted(sub["arm"].unique()), "o^sx"):
            a = sub[sub["arm"] == arm].sort_values("bits_per_weight")
            ax.scatter(
                a["bits_per_weight"], a["ip_distortion"], label=arm, marker=marker, s=18
            )
        ax.set_title(w.removeprefix("transformer."), fontsize=9)
        ax.set_xlabel("bits / weight (total, incl. L+S storage)")
        ax.set_yscale("log")
    axes[0].set_ylabel("inner-product distortion")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = run / "lrs_rate_distortion.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
