"""Track A figures: matched-parameter error curves, real stacks vs permutation null.

This is the A4 gate's visual: per object, median (line) and IQR (band) of
rel_error across layers, as a function of param_count, per method. If BMD sits
below the CP/Tucker floor on the real panel and that gap closes on the null
panel, the diag-template structure is real.

    uv run python experiments/plots/plot_a2_a3.py \
        --a2-run results/a2_matched_param/<run-id> \
        --a3-run results/a3_permutation_null/<run-id>
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
    a2_run: str
    a3_run: str | None = None  # omit to plot the real panel only
    out: str | None = None  # default: <a2_run>/error_vs_params.png


def _panel(ax, df: pd.DataFrame, title: str) -> None:
    for method, grp in df.groupby("method"):
        agg = (
            grp.groupby("params").rel_error.agg(["median", "min", "max"]).reset_index()
        )
        agg = agg.sort_values("params")
        (line,) = ax.plot(agg["params"], agg["median"], marker="o", ms=3, label=method)
        ax.fill_between(
            agg["params"], agg["min"], agg["max"], color=line.get_color(), alpha=0.15
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("parameters")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def main(cfg: Config) -> None:
    frames = {"real": pd.read_parquet(Path(cfg.a2_run) / "metrics.parquet")}
    if cfg.a3_run:
        frames["null"] = pd.read_parquet(Path(cfg.a3_run) / "metrics.parquet")

    objects = sorted(frames["real"].object.unique())
    n_rows, n_cols = len(frames), len(objects)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows), squeeze=False
    )
    for r, (kind, df) in enumerate(frames.items()):
        for c, obj in enumerate(objects):
            sub = df[df.object == obj]
            n_layers = sub.layer.nunique()
            _panel(axes[r][c], sub, f"{obj} — {kind} ({n_layers} layers)")
    axes[0][0].set_ylabel("relative error (median, min-max band)")
    if n_rows > 1:
        axes[1][0].set_ylabel("relative error (median, min-max band)")
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    out = Path(cfg.out) if cfg.out else Path(cfg.a2_run) / "error_vs_params.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
