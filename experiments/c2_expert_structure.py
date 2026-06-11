"""C2: matched-parameter structure diagnostic on stacked MoE expert matrices.

The same discriminator as a2, on expert stacks (out, in, E): if BMD finds
low-rank diag-template structure below the Tucker/CP floor at matched
parameters, entry 2's shared-template streaming story has its math; if
Tucker dominates, the C1 redundancy is subspace-shaped and TensorLLM-style
methods win the MoE object too.

    uv run python experiments/c2_expert_structure.py --device cuda --layers 0 7 15
"""

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.stacks.moe import expert_stack, moe_layers
from bmx.sweep import decomp_sweep


@dataclasses.dataclass
class Config:
    model_id: str = "allenai/OLMoE-1B-7B-0125"
    checkpoint_dir: str | None = None
    layers: tuple[int, ...] = ()  # empty = all MoE layers
    whichs: tuple[str, ...] = ("gate", "down")
    dtype: str = "float32"
    device: str = "cpu"
    bmd_iters: int = 100
    bmd_check_every: int = 5
    experiment: str = "c2_expert_structure"


# Rank grids for OLMoE-shaped stacks (1024, 2048, 64) ~ 134M dense params.
# BMD ell costs ell*(m*p + (m+p)*E) ~ ell*2.3M; slice_svd r costs E*r*(m+p);
# tucker/shared grids span comparable budgets.
PLAN = {
    "bmd_rals": [1, 2, 4, 8, 16],
    "slice_svd": [1, 2, 4, 8, 16, 32, 64, 128],
    "cp": [16, 64, 256],
    "tucker": [(64, 64, 16), (128, 128, 32), (256, 256, 64), (512, 512, 64)],
    "shared_tucker": [(32, 32), (64, 64), (128, 128), (256, 256)],
}


def main(cfg: Config) -> None:
    if cfg.checkpoint_dir is None:
        from huggingface_hub import snapshot_download

        ckpt = snapshot_download(
            cfg.model_id,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
        )
    else:
        ckpt = cfg.checkpoint_dir

    layers = list(cfg.layers) or moe_layers(ckpt)
    run = create_run(cfg.experiment, cfg)
    model_tag = cfg.model_id.split("/")[-1]
    frames = []
    for layer in layers:
        for which in cfg.whichs:
            stack = expert_stack(ckpt, layer, which, model=model_tag)
            stack.tensor = stack.tensor.to(
                dtype=getattr(torch, cfg.dtype), device=cfg.device
            )
            df = decomp_sweep(
                stack,
                PLAN,
                fit_opts={
                    "bmd_rals": {
                        "n_iters": cfg.bmd_iters,
                        "check_every": cfg.bmd_check_every,
                    }
                },
                extra_cols={"null_seed": None},
                verbose=True,
            )
            frames.append(df)
            write_metrics(run, pd.concat(frames, ignore_index=True))
            print(f"layer {layer} {which}: {len(df)} fits done", flush=True)
            if cfg.device.startswith("cuda"):
                torch.cuda.empty_cache()
    print(f"-> {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
