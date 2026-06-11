"""C1: pairwise expert-redundancy census on fine-grained MoE checkpoints.

The cheap gate for entry 2 (BMD expert streaming): if redundancy is global
(high mean similarity, low participation ratio), a shared-template
decomposition has something to exploit and C2 proceeds; if similarity is near
zero or concentrated in a few mergeable pairs, expect BM-rank near E and stop.

    uv run python experiments/c1_redundancy_census.py \
        --model-id allenai/OLMoE-1B-7B-0125 --device cuda
"""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.census import experts_shared_last, pairwise_similarities, similarity_summary
from bmx.stacks.moe import expert_stack, moe_layers


@dataclasses.dataclass
class Config:
    model_id: str = "allenai/OLMoE-1B-7B-0125"
    checkpoint_dir: str | None = None  # skip download, use local dir
    layers: tuple[int, ...] = ()  # empty = all MoE layers
    whichs: tuple[str, ...] = ("gate", "up", "down")
    top_r: int = 32
    device: str = "cpu"
    dtype: str = "float32"
    experiment: str = "c1_redundancy_census"


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
    sims_dir = Path(run) / "sims"
    sims_dir.mkdir()
    model_tag = cfg.model_id.split("/")[-1]

    rows = []
    for layer in layers:
        for which in cfg.whichs:
            stack = expert_stack(ckpt, layer, which, model=model_tag)
            S = experts_shared_last(stack).to(
                device=cfg.device, dtype=getattr(torch, cfg.dtype)
            )
            sims = pairwise_similarities(S, top_r=cfg.top_r)
            np.savez_compressed(
                sims_dir / f"layer{layer:02d}_{which}.npz",
                **{k: v.cpu().numpy() for k, v in sims.items()},
            )
            for metric, sim in sims.items():
                rows.append(
                    {
                        "model": model_tag,
                        "layer": layer,
                        "which": which,
                        "metric": metric,
                        "d_private": S.shape[1],
                        "d_shared": S.shape[2],
                    }
                    | similarity_summary(sim)
                )
            del S, sims
            if cfg.device.startswith("cuda"):
                torch.cuda.empty_cache()
        # rewrite each layer: crash loses at most one layer
        write_metrics(run, pd.DataFrame(rows))
        last = [r for r in rows if r["layer"] == layer and r["metric"] == "cka"]
        line = "  ".join(f"{r['which']}: cka_mean={r['off_mean']:.3f}" for r in last)
        print(f"layer {layer:2d}  {line}", flush=True)
    print(f"-> {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
