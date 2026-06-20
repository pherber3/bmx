"""Break-even margin vs matrix width across models: the scale-invariance plot.

One point per matrix from every results/frontier_breakeven run: x = harmonic
mean dimension 2mp/(m+p), y = low-rank break-even margin in effective
bits/weight (symlog). Tables (wpe, MoE routers) and layer-0 input-readers are
marked; everything else is a transform weight and should hug the y=0 line."""

import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tyro


@dataclasses.dataclass
class Config:
    root: str = "results/frontier_breakeven"


def classify(row) -> str:
    name = row["weight"]
    if "wpe" in name or name.endswith(".mlp.gate.weight"):
        return "table (wpe / router)"
    reader = any(
        s in name
        for s in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "c_attn")
    )
    if row["layer"] == 0 and reader:
        return "layer-0 input-reader"
    return "transform"


def main(cfg: Config) -> None:
    # newest run only; blind concat double-counts reruns (see CLAUDE.md pitfalls)
    runs = sorted(Path(cfg.root).glob("*/metrics.parquet"))
    assert runs, f"no metrics.parquet under {cfg.root}"
    df = pd.read_parquet(runs[-1])  # timestamps sort lexically
    df["d_h"] = 2 * df.m * df.p / (df.m + df.p)
    df["cls"] = df.apply(classify, axis=1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {
        "gpt2": "o",
        "meta-llama/Llama-3.1-8B": "s",
        "Qwen/Qwen3-30B-A3B-Base": "^",
        "meta-llama/Llama-3.1-70B": "D",
    }
    colors = {
        "transform": "#777777",
        "layer-0 input-reader": "#d62728",
        "table (wpe / router)": "#1f77b4",
    }
    for (model, cls), sub in df.groupby(["model", "cls"]):
        ax.scatter(
            sub.d_h,
            sub.lr_margin_bits,
            s=26,
            marker=markers.get(model, "x"),
            color=colors[cls],
            alpha=0.8,
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlabel("harmonic mean dimension 2mp/(m+p)")
    ax.set_ylabel("low-rank break-even margin (bits/weight)")
    ax.set_title("Side-information margin vs width: transforms hug the Shannon line")
    legend = [
        plt.Line2D([], [], color=c, marker="o", ls="", label=k)
        for k, c in colors.items()
    ]
    legend += [
        plt.Line2D([], [], color="black", marker=m, ls="", label=k.split("/")[-1])
        for k, m in markers.items()
    ]
    ax.legend(handles=legend, fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    out = Path(cfg.root) / "frontier_margin_vs_width.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
