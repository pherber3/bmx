"""D1: do trained weight rows Gaussianize under data-oblivious rotation?

Tests entry 3's failure mode 2 directly: weights are trained, correlated
objects — does the random-vector Gaussianization argument survive contact
with them? Reports per-matrix kurtosis and outlier mass, before vs after."""

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.quant.hadamard import random_orthogonal
from bmx.quant.stats import kurtosis, outlier_mass
from bmx.stacks.gpt2 import load_gpt2_state


@dataclasses.dataclass
class Config:
    model_name: str = "gpt2"
    seed: int = 0
    min_dim: int = 64  # skip tiny weights (layernorms, biases)


def main(cfg: Config) -> None:
    sd, _ = load_gpt2_state(cfg.model_name)
    run = create_run("d1_gaussianization", cfg)
    rows = []
    for name, W in sd.items():
        if W.ndim != 2 or min(W.shape) < cfg.min_dim:
            continue
        W = W.to(torch.float64)
        d = W.shape[-1]
        Q = random_orthogonal(d, seed=cfg.seed, dtype=torch.float64)
        Wr = W @ Q.T  # rotate rows
        rows.append(
            {
                "weight": name,
                "shape": str(tuple(W.shape)),
                "kurtosis_before": kurtosis(W, dim=-1).mean().item(),
                "kurtosis_after": kurtosis(Wr, dim=-1).mean().item(),
                "outlier_mass_before": outlier_mass(W).mean().item(),
                "outlier_mass_after": outlier_mass(Wr).mean().item(),
                "outlier_mass_max_before": outlier_mass(W).max().item(),
                "outlier_mass_max_after": outlier_mass(Wr).max().item(),
            }
        )
        print(
            f"{name}: kurtosis {rows[-1]['kurtosis_before']:+.3f} -> "
            f"{rows[-1]['kurtosis_after']:+.3f}"
        )
    write_metrics(run, pd.DataFrame(rows))
    print(f"-> {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
