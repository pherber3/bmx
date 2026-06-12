"""K2/K2b headline figures from committed parquets (read, never refit).

Left: Llama-8B keys — attention-logit distortion (post-RoPE basis) vs total
bits/entry per codec; the lowrank-pre-RoPE points dominate the curve.
Right: end-to-end perplexity hit vs average bits/entry (K2b, n_prefill=1792).
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
    k2_root: str = "results/k2_cache_arms"
    k2b_root: str = "results/k2b_cache_ppl"
    out: str = "results/k2_cache_arms/k2_headline.png"


def newest(root: str, require: str = "metrics.parquet") -> pd.DataFrame:
    runs = sorted(Path(root).glob(f"*/{require}"))
    assert runs, f"no {require} under {root}"
    return pd.concat([pd.read_parquet(p) for p in runs], ignore_index=True)


ARM_LABELS = {
    "rtn_token": "per-token RTN",
    "rtn_channel": "per-channel RTN (KIVI-style)",
    "rotate_rtn_token": "rotate + RTN (QuaRot-style)",
    "turboquant_mse": "rotate + Lloyd (TurboQuant core)",
    "turboquant_prod": "TurboQuant two-stage (unbiased)",
    "lowrank_rtn_channel": "low-rank pre-RoPE + per-channel",
}


def main(cfg: Config) -> None:
    k2 = newest(cfg.k2_root)
    k2 = k2[(k2.model == "llama-3.1-8b") & (k2.kind == "k_pre")]
    k2b = newest(cfg.k2b_root)
    k2b = k2b[(k2b.n_prefill == 1792) & (k2b.bits != 16)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for arm, marker in zip(ARM_LABELS, "osv^DP"):
        sub = (
            k2[k2.arm == arm]
            .groupby(["bits", "rank"])
            .agg(bpe=("bpe", "first"), m=("logit_rope", "mean"))
            .reset_index()
            .sort_values("bpe")
        )
        if sub.empty:
            continue
        emph = arm == "lowrank_rtn_channel"
        ax1.plot(
            sub.bpe,
            sub.m,
            marker=marker,
            lw=2.2 if emph else 1.1,
            ms=7 if emph else 5,
            color="#d62728" if emph else None,
            label=ARM_LABELS[arm],
        )
    ax1.set_yscale("log")
    ax1.set_xlabel("total bits per cache entry (all metadata counted)")
    ax1.set_ylabel("attention-score error (real queries)")
    ax1.set_title("Keys: low-rank pre-RoPE dominates at every budget")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.25)

    k2b = k2b.copy()
    k2b["avg_bpe"] = (k2b.bpe_k + k2b.bpe_v) / 2
    k2b["combo"] = [
        ARM_LABELS.get(a, a) if a == v else "recipe: low-rank K + Lloyd V"
        for a, v in zip(k2b.arm_k, k2b.arm_v)
    ]
    for combo, sub in k2b.groupby("combo"):
        sub = sub.sort_values("avg_bpe")
        emph = combo.startswith("recipe")
        ax2.plot(
            sub.avg_bpe,
            sub.dppl_pct.clip(lower=0.05),
            marker="o",
            lw=2.2 if emph else 1.1,
            color="#d62728" if emph else None,
            label=combo,
        )
    ax2.axhline(0.5, color="gray", lw=0.7, ls="--")
    ax2.text(4.1, 0.55, "+0.5% quality", fontsize=7, color="gray")
    ax2.set_yscale("log")
    ax2.set_xlabel("average bits per cache entry (K and V)")
    ax2.set_ylabel("perplexity increase vs fp16 cache (%)")
    ax2.set_title("End-to-end: Llama-3.1-8B, 1792-token context")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out = Path(cfg.out)
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
