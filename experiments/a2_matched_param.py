"""A2: matched-parameter comparison on GPT-2 attention stacks.

Fits BMD-RALS vs slice-SVD vs CP vs Tucker vs shared-factor Tucker across
layers and objects; the load-bearing axis downstream is error vs param_count.
"""

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.stacks.gpt2 import load_gpt2_state, stack_by_name
from bmx.stacks.null import permutation_null
from bmx.sweep import decomp_sweep


@dataclasses.dataclass
class Config:
    model_name: str = "gpt2"
    layers: tuple[int, ...] = tuple(range(12))
    objects: tuple[str, ...] = ("wqk", "wov")  # also: raw_q raw_k raw_v raw_o
    dtype: str = "float32"
    device: str = "cpu"  # "cuda" on the VM; BMD + tensorly baselines follow
    null_seed: int | None = None  # set to apply the permutation null (a3)
    bmd_iters: int = 200
    bmd_check_every: int = 5  # sample the dense error check during long fits
    experiment: str = "a2_matched_param"


# rank grids chosen to span comparable param ranges per method on (768, 768, 12);
# BMD ell=8 ~ 4.9M params, so baselines extend to match. CP stays capped: CP-ALS
# beyond rank ~512 is prohibitively slow on CPU at this size (note in metrics).
PLAN = {
    "bmd_rals": [1, 2, 4, 8],
    "slice_svd": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "cp": [8, 32, 128, 512],
    "tucker": [
        (32, 32, 4),
        (64, 64, 8),
        (128, 128, 12),
        (256, 256, 12),
        (384, 384, 12),
        (512, 512, 12),
    ],
    "shared_tucker": [
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (384, 384),
        (512, 512),
    ],
}


def main(cfg: Config) -> None:
    sd, meta = load_gpt2_state(cfg.model_name)
    run = create_run(cfg.experiment, cfg)
    frames = []
    for layer in cfg.layers:
        for obj in cfg.objects:
            stack = stack_by_name(sd, layer, meta["n_head"], obj, model=cfg.model_name)
            stack.tensor = stack.tensor.to(
                dtype=getattr(torch, cfg.dtype), device=cfg.device
            )
            null_seed = None if cfg.null_seed is None else cfg.null_seed + layer
            if null_seed is not None:
                stack.tensor, _ = permutation_null(stack.tensor, seed=null_seed)
            df = decomp_sweep(
                stack,
                PLAN,
                fit_opts={
                    "bmd_rals": {
                        "n_iters": cfg.bmd_iters,
                        "check_every": cfg.bmd_check_every,
                    }
                },
                # always present so a2 and a3 parquet share one schema
                extra_cols={"null_seed": null_seed},
                verbose=True,
            )
            frames.append(df)
            # Rewrite after every block so a mid-sweep crash loses at most
            # one layer/object of fits.
            write_metrics(run, pd.concat(frames, ignore_index=True))
            print(f"layer {layer} {obj}: {len(df)} fits done", flush=True)
    print(f"-> {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
