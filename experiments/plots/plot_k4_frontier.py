"""K4 frontier figures: accounting curves (model vs skeptic bpe) and structure tax.

Reads parquet, never refits. Select rows explicitly by arm; unknown arms ignored.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# One row per frontier arm: (arm, extra_mask_fn, label). extra_mask_fn is
# None for arms selected by `df.arm == arm` alone; otherwise it takes the
# already-arm-filtered df and returns an additional boolean mask to AND in.
# Spectral variants are all fit_mode == "oracle": "spectral" is the weighted
# headline, "spectral_unweighted" is the P4 ablation (weighted == False on
# real frames), "spectral_randbasis" is the random-basis control (no
# weighted filter).
ARM_SPECS: list[tuple[str, Callable[[pd.DataFrame], pd.Series] | None, str]] = [
    (
        "spectral",
        lambda df: df.weighted & (df.fit_mode == "oracle"),
        "spectral (weighted, oracle)",
    ),
    (
        "spectral_unweighted",
        lambda df: ~df.weighted & (df.fit_mode == "oracle"),
        "spectral (unweighted, oracle)",
    ),
    (
        "spectral_randbasis",
        lambda df: df.fit_mode == "oracle",
        "random-basis control",
    ),
    ("turboquant_mse", None, "turboquant_mse"),
    ("lowrank_rtn_channel", None, "lowrank_rtn_channel"),
    ("k2t_coeffquant", None, "k2t_coeffquant"),
    ("rtn_channel", None, "rtn_channel"),
]


def _arm_selection(df: pd.DataFrame, arm: str) -> tuple[pd.Series, str]:
    """Row mask + legend label for one frontier arm, per ARM_SPECS."""
    for spec_arm, extra_mask_fn, label in ARM_SPECS:
        if spec_arm == arm:
            mask = df.arm == arm
            if extra_mask_fn is not None:
                mask = mask & extra_mask_fn(df)
            return mask, label
    return df.arm == arm, arm


def _plot_frontier(
    df: pd.DataFrame,
    available_arms: set[str],
    x_col: str,
    xlabel: str,
    title: str,
    out_path: Path,
) -> Path:
    """Frontier figure: one errorbar line per arm, layer-mean distortion vs x_col."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for arm in sorted(available_arms):
        mask, label = _arm_selection(df, arm)
        sub = df[mask]
        if sub.empty:
            continue

        # Group by x_col, compute layer-mean and sem
        grouped = (
            sub.groupby(x_col)
            .agg(
                distortion_mean=("distortion", "mean"),
                distortion_sem=("distortion", lambda x: x.sem() if len(x) > 1 else 0),
            )
            .reset_index()
        )
        grouped = grouped.sort_values(x_col)

        # Plot with error bars
        ax.errorbar(
            grouped[x_col],
            grouped["distortion_mean"],
            yerr=grouped["distortion_sem"],
            marker="o",
            label=label,
            capsize=4,
        )

    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("headline distortion")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_figures(df: pd.DataFrame, out_dir: str) -> list[Path]:
    """Generate K4 frontier figures: model/skeptic accounting + structure tax.

    Args:
        df: DataFrame with columns model, layer, kind, arm, fit_mode, weighted,
            budget, bits, rank, mse_scale, bpe_model, bpe_skeptic,
            bpe_skeptic_deploy, rel_fro, logit, logit_rope.
        out_dir: Output directory for PNG files.

    Returns:
        List of Path objects for generated figures.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    df = df.copy()

    # Compute headline distortion: use logit_rope where non-NaN, else logit.
    df["distortion"] = df["logit_rope"].fillna(df["logit"])

    # Known arms to plot (select explicitly to avoid unknown arms).
    known_arms = {spec_arm for spec_arm, _, _ in ARM_SPECS}
    available_arms = set(df["arm"].unique()) & known_arms

    p1 = _plot_frontier(
        df,
        available_arms,
        "bpe_model",
        "bpe (model accounting)",
        "K4: Distortion vs model accounting (log-scale)",
        out / "k4_frontier_model.png",
    )
    paths.append(p1)

    p2 = _plot_frontier(
        df,
        available_arms,
        "bpe_skeptic_deploy",
        "bpe (skeptic deployment accounting)",
        "K4: Distortion vs skeptic deployment accounting (log-scale)",
        out / "k4_frontier_skeptic.png",
    )
    paths.append(p2)

    # --- Figure 3: Structure tax bar chart at ~3-bit operating point ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # Arms to compare at the 3-bit point. The spectral entry reuses the
    # frontier table's mask for "spectral" (weighted, oracle) with the
    # budget==3.0 point selected on top.
    _spectral_mask, _ = _arm_selection(df, "spectral")
    tax_arms = [
        (
            "turboquant_mse",
            (df.arm == "turboquant_mse") & (df.bits == 3) & (df.kind == "k_pre"),
        ),
        (
            "rtn_channel",
            (df.arm == "rtn_channel") & (df.bits == 3) & ~df.mse_scale,
        ),
        (
            "spectral (oracle)",
            _spectral_mask & (df.budget == 3.0),
        ),
    ]

    bar_labels = []
    bar_values = []

    for label, mask in tax_arms:
        sub = df[mask]
        if sub.empty:
            # Skip bars whose rows are absent
            continue

        # Layer-mean distortion
        distortion_mean = sub["distortion"].mean()
        bar_labels.append(label)
        bar_values.append(distortion_mean)

    if bar_labels:
        x_pos = np.arange(len(bar_labels))
        ax.bar(
            x_pos,
            bar_values,
            color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(bar_labels)],
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bar_labels)
        ax.set_ylabel("headline distortion")
        ax.set_title("K4: Structure tax at ~3-bit operating point")
        ax.set_yscale("log")

    p3 = out / "k4_structure_tax.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p3)

    return paths


if __name__ == "__main__":
    import sys

    df = pd.read_parquet(sys.argv[1])
    print(make_figures(df, sys.argv[2] if len(sys.argv) > 2 else "."))
