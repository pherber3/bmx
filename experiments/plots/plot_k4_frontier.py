"""K4 frontier figures: accounting curves (model vs skeptic bpe) and structure tax.

Reads parquet, never refits. Select rows explicitly by arm; unknown arms ignored.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _arm_selection(df: pd.DataFrame, arm: str) -> tuple[pd.Series, str]:
    """Row mask + legend label for one frontier arm.

    Spectral variants get per-arm filters (all at fit_mode == "oracle"):
    "spectral" is the weighted headline, "spectral_unweighted" is the P4
    ablation (weighted == False on real frames), "spectral_randbasis" is the
    random-basis control (no weighted filter).
    """
    if arm == "spectral":
        mask = (df.arm == arm) & df.weighted & (df.fit_mode == "oracle")
        return mask, "spectral (weighted, oracle)"
    if arm == "spectral_unweighted":
        mask = (df.arm == arm) & ~df.weighted & (df.fit_mode == "oracle")
        return mask, "spectral (unweighted, oracle)"
    if arm == "spectral_randbasis":
        mask = (df.arm == arm) & (df.fit_mode == "oracle")
        return mask, "random-basis control"
    return df.arm == arm, arm


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
    known_arms = {
        "spectral",
        "spectral_unweighted",
        "spectral_randbasis",
        "turboquant_mse",
        "lowrank_rtn_channel",
        "k2t_coeffquant",
        "rtn_channel",
    }
    available_arms = set(df["arm"].unique()) & known_arms

    # --- Figure 1: Frontier vs bpe_model, layer-mean with sem error bars ---
    fig, ax = plt.subplots(figsize=(8, 5))

    for arm in sorted(available_arms):
        mask, label = _arm_selection(df, arm)
        sub = df[mask]
        if sub.empty:
            continue

        # Group by bpe_model, compute layer-mean and sem
        grouped = (
            sub.groupby("bpe_model")
            .agg(
                distortion_mean=("distortion", "mean"),
                distortion_sem=("distortion", lambda x: x.sem() if len(x) > 1 else 0),
            )
            .reset_index()
        )
        grouped = grouped.sort_values("bpe_model")

        # Plot with error bars
        ax.errorbar(
            grouped["bpe_model"],
            grouped["distortion_mean"],
            yerr=grouped["distortion_sem"],
            marker="o",
            label=label,
            capsize=4,
        )

    ax.set_yscale("log")
    ax.set_xlabel("bpe (model accounting)")
    ax.set_ylabel("headline distortion")
    ax.set_title("K4: Distortion vs model accounting (log-scale)")
    ax.legend()
    ax.grid(alpha=0.25)
    p1 = out / "k4_frontier_model.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p1)

    # --- Figure 2: Frontier vs bpe_skeptic_deploy, layer-mean with sem error bars ---
    fig, ax = plt.subplots(figsize=(8, 5))

    for arm in sorted(available_arms):
        mask, label = _arm_selection(df, arm)
        sub = df[mask]
        if sub.empty:
            continue

        # Group by bpe_skeptic_deploy, compute layer-mean and sem
        grouped = (
            sub.groupby("bpe_skeptic_deploy")
            .agg(
                distortion_mean=("distortion", "mean"),
                distortion_sem=("distortion", lambda x: x.sem() if len(x) > 1 else 0),
            )
            .reset_index()
        )
        grouped = grouped.sort_values("bpe_skeptic_deploy")

        # Plot with error bars
        ax.errorbar(
            grouped["bpe_skeptic_deploy"],
            grouped["distortion_mean"],
            yerr=grouped["distortion_sem"],
            marker="o",
            label=label,
            capsize=4,
        )

    ax.set_yscale("log")
    ax.set_xlabel("bpe (skeptic deployment accounting)")
    ax.set_ylabel("headline distortion")
    ax.set_title("K4: Distortion vs skeptic deployment accounting (log-scale)")
    ax.legend()
    ax.grid(alpha=0.25)
    p2 = out / "k4_frontier_skeptic.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p2)

    # --- Figure 3: Structure tax bar chart at ~3-bit operating point ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # Arms to compare at the 3-bit point
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
            (df.arm == "spectral")
            & (df.budget == 3.0)
            & df.weighted
            & (df.fit_mode == "oracle"),
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
