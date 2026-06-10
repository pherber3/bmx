"""A2: matched-parameter comparison on GPT-2 attention stacks.

Fits BMD-RALS vs slice-SVD vs CP vs Tucker vs shared-factor Tucker across
layers and objects; the load-bearing axis downstream is error vs param_count.
"""

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.stacks.gpt2 import circuit_stack, load_gpt2_state, raw_stack
from bmx.stacks.null import permutation_null
from bmx.sweep import decomp_sweep


@dataclasses.dataclass
class Config:
    model_name: str = "gpt2"
    layers: tuple[int, ...] = tuple(range(12))
    objects: tuple[str, ...] = ("wqk", "wov")  # also: raw_q raw_k raw_v raw_o
    dtype: str = "float32"
    null_seed: int = -1  # >= 0 applies the permutation null (used by a3)
    bmd_iters: int = 200
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


def build_stack(sd, meta, layer: int, obj: str, dtype):
    if obj.startswith("raw_"):
        s = raw_stack(sd, layer, meta["n_head"], which=obj.removeprefix("raw_"))
    else:
        s = circuit_stack(sd, layer, meta["n_head"], kind=obj)
    s.tensor = s.tensor.to(getattr(torch, dtype))
    return s


def main(cfg: Config) -> None:
    sd, meta = load_gpt2_state(cfg.model_name)
    run = create_run(cfg.experiment, cfg)
    frames = []
    for layer in cfg.layers:
        for obj in cfg.objects:
            stack = build_stack(sd, meta, layer, obj, cfg.dtype)
            extra = {}
            if cfg.null_seed >= 0:
                stack.tensor, _ = permutation_null(
                    stack.tensor, seed=cfg.null_seed + layer
                )
                extra = {"null_seed": cfg.null_seed + layer}
            df = decomp_sweep(
                stack,
                PLAN,
                fit_opts={"bmd_rals": {"n_iters": cfg.bmd_iters}},
                extra_cols=extra,
            )
            frames.append(df)
            print(f"layer {layer} {obj}: {len(df)} fits done")
    write_metrics(run, pd.concat(frames, ignore_index=True))
    print(f"-> {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
