"""F1 figure — measured D_mse / D_prod vs bit-width, with TurboQuant's
closed-form bounds overlaid (Fig-3 parity).

Two log-y panels:
  left  : D_mse vs bitwidth, bounded by [4^-b, sqrt(3)*pi/2 * 4^-b]
  right : D_prod vs bitwidth, bounded by [(1/d) 4^-b, sqrt(3)*pi^2/2 (1/d) 4^-b]

Reads the distortion parquet (measured points per arm); the bound lines are
computed here from the closed forms, never fit. Points that fall between the
two bound lines are the "lands between the bounds" claim; our structure-aware
arm (lowrank_rtn_channel) is expected to sit at/below turboquant on D_prod.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tyro

from bmx.quant.stats import sq_floor

_SQRT3 = math.sqrt(3.0)


def _mse_bounds(b: float) -> tuple[float, float]:
    """(lower, upper) MSE bounds on a unit vector at b bits (TurboQuant §3)."""
    lo = sq_floor(b)  # 4^-b
    hi = _SQRT3 * math.pi / 2.0 * 4.0 ** (-b)
    return lo, hi


def _prod_bounds(b: float, d: int) -> tuple[float, float]:
    """(lower, upper) inner-product distortion bounds at b bits, dimension d."""
    lo = (1.0 / d) * 4.0 ** (-b)
    hi = _SQRT3 * math.pi**2 / 2.0 * (1.0 / d) * 4.0 ** (-b)
    return lo, hi


def make_figures(df: pd.DataFrame, out_dir) -> list[Path]:
    """Emit distortion_vs_bitwidth.png. Returns the list of written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # dimension for the D_prod bound: from the data if present, else 128.
    d = int(df["d"].iloc[0]) if "d" in df.columns and len(df) else 128

    # smooth b-grid for the closed-form bound lines (measured points are at integer b)
    fine = [b + i * 0.05 for b in range(1, 5) for i in range(21)]

    fig, (ax_mse, ax_prod) = plt.subplots(1, 2, figsize=(11, 4.2))
    markers = "o^sxDvP*"

    # --- measured points -----------------------------------------------------
    for arm, marker in zip(sorted(df["arm"].unique()), markers):
        a = df[df["arm"] == arm].sort_values("bitwidth")
        ax_mse.scatter(a["bitwidth"], a["d_mse"], label=arm, marker=marker, s=28)
        ax_prod.scatter(a["bitwidth"], a["d_prod"], label=arm, marker=marker, s=28)

    # --- closed-form bound lines --------------------------------------------
    mse_lo = [_mse_bounds(b)[0] for b in fine]
    mse_hi = [_mse_bounds(b)[1] for b in fine]
    ax_mse.plot(fine, mse_lo, "k--", lw=1, label="lower $4^{-b}$")
    ax_mse.plot(fine, mse_hi, "k:", lw=1, label=r"upper $\frac{\sqrt{3}\pi}{2}4^{-b}$")

    prod_lo = [_prod_bounds(b, d)[0] for b in fine]
    prod_hi = [_prod_bounds(b, d)[1] for b in fine]
    ax_prod.plot(fine, prod_lo, "k--", lw=1, label=r"lower $\frac{1}{d}4^{-b}$")
    ax_prod.plot(
        fine, prod_hi, "k:", lw=1, label=r"upper $\frac{\sqrt{3}\pi^2}{2d}4^{-b}$"
    )

    for ax, title in ((ax_mse, "D_mse (unit-norm)"), (ax_prod, f"D_prod (d={d})")):
        ax.set_yscale("log")
        ax.set_xlabel("bit-width b")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(list(range(1, 5)))
        ax.legend(fontsize=7)
    ax_mse.set_ylabel(r"$\|x-\hat x\|^2$")
    ax_prod.set_ylabel(r"$\mathbb{E}|\langle y,x\rangle-\langle y,\hat x\rangle|^2$")

    fig.suptitle("F1 — distortion vs bit-width (TurboQuant Fig-3 parity)")
    fig.tight_layout()

    out = out_dir / "distortion_vs_bitwidth.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return [out]


# ---------------------------------------------------------------------------
# CLI: read a run's parquet and emit the figure next to it.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Config:
    run_dir: str = ""  # default: newest results/distortion_bounds run


def _newest_run(root="results/distortion_bounds") -> Path:
    runs = sorted(p for p in Path(root).iterdir() if (p / "metrics.parquet").exists())
    assert runs, f"no metrics.parquet under {root}"
    return runs[-1]


def main(cfg: Config) -> None:
    run = Path(cfg.run_dir) if cfg.run_dir else _newest_run()
    df = pd.read_parquet(run / "metrics.parquet")
    for p in make_figures(df, run):
        print(f"-> {p}")


if __name__ == "__main__":
    main(tyro.cli(Config))
