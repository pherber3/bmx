"""Frontier-scale break-even pre-test (Avenue 1 epilogue).

For each sampled weight matrix: would fp16 side-information (low-rank factors
or a sparse spike list) ever pay against the Shannon 4^-b floor? Side info
costing Db bits/weight pays iff the energy fraction eps it removes satisfies
eps > 1 - 4^(-Db) (docs/2026-06-11-lrs-results.md, theoretical postmortem).
Reported headline per matrix: the best margin in effective bits/weight,

    lr_margin = max_r [ log4(1/(1 - eps(r))) - 16 r (m+p)/(m p) ]
    sp_margin = max_k [ log4(1/(1 - eps_S(k))) - k (16 + log2(m p))/(m p) ]

positive margin => that structure pays at this dimension. Only singular
VALUES are needed, so models far larger than RAM stream shard-by-shard
(peak disk ~ one shard; shards deleted after processing unless --keep-shards).
"""

import dataclasses
import gc
import json
import math
import os
import re
import tempfile
from pathlib import Path

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.decomp.lrs import spikiness_ratio
from bmx.quant.breakeven import breakeven_row
from bmx.quant.stats import kurtosis

_LAYER_RE = re.compile(r"(?:\.layers\.|\.h\.)(\d+)\.")
_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")


@dataclasses.dataclass
class Config:
    model: str = "gpt2"  # "gpt2" or an HF repo id, e.g. "meta-llama/Llama-3.1-8B"
    n_layers_sample: int = 4
    experts_per_layer: int = 8  # MoE: only experts with index < this are read
    max_dim: int = 32768  # skip embeddings/lm_head (vocab-sized axes)
    min_dim: int = 64
    max_side_bpw: float = 6.0  # margin search bounded; past this just store fp16
    keep_shards: bool = False
    shard_dir: str = ""  # default: <tmp>/bmx_shards


def sample_layers(n_layers: int, n_sample: int) -> set[int]:
    if n_sample >= n_layers:
        return set(range(n_layers))
    return {round(i * (n_layers - 1) / (n_sample - 1)) for i in range(n_sample)}


def matrix_row(name: str, W: torch.Tensor, cfg: Config) -> dict:
    sigma = W.std()
    noise_max = math.sqrt(2 * math.log(W.numel()))  # expected bulk max, in sigmas
    layer = _LAYER_RE.search(name)
    return {
        "model": cfg.model,
        "weight": name,
        "layer": int(layer.group(1)) if layer else -1,
        "m": W.shape[0],
        "p": W.shape[1],
        **breakeven_row(W, cfg.max_side_bpw),
        "kurtosis": kurtosis(W.double(), dim=-1).mean().item(),
        "spikiness": spikiness_ratio(W),
        "frac_3sigma": (W.abs() > 3.29 * sigma).double().mean().item(),
        "frac_noise_max": (W.abs() > noise_max * sigma).double().mean().item(),
    }


def iter_gpt2(cfg: Config):
    from bmx.stacks.gpt2 import load_gpt2_state

    sd, meta = load_gpt2_state()
    layers = sample_layers(meta["n_layer"], cfg.n_layers_sample)
    for name, W in sd.items():
        lm = _LAYER_RE.search(name)
        in_sampled = lm is not None and int(lm.group(1)) in layers
        if not (in_sampled or name == "transformer.wpe.weight"):
            continue
        if W.ndim == 2 and cfg.min_dim <= min(W.shape) and max(W.shape) <= cfg.max_dim:
            yield name, W.float()


def iter_hf(cfg: Config):
    """Stream shards one at a time; delete each after its matrices are read."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    from safetensors import safe_open

    shard_dir = cfg.shard_dir or str(Path(tempfile.gettempdir()) / "bmx_shards")
    try:
        idx = hf_hub_download(cfg.model, "model.safetensors.index.json")
        weight_map: dict[str, str] = json.loads(Path(idx).read_text())["weight_map"]
    except EntryNotFoundError:
        weight_map = {"": "model.safetensors"}  # unsharded: scan the single file

    layer_ids = [int(g.group(1)) for k in weight_map if (g := _LAYER_RE.search(k))]
    layers = (
        sample_layers(max(layer_ids) + 1, cfg.n_layers_sample) if layer_ids else set()
    )

    def want(name: str) -> bool:
        if not name.endswith(".weight"):
            return False
        lm = _LAYER_RE.search(name)
        if lm is None or int(lm.group(1)) not in layers:
            return False  # also drops embeddings / lm_head / final norm
        em = _EXPERT_RE.search(name)
        return em is None or int(em.group(1)) < cfg.experts_per_layer

    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if name == "" or want(name):
            by_shard.setdefault(shard, []).append(name)

    for shard, names in sorted(by_shard.items()):
        path = hf_hub_download(cfg.model, shard, local_dir=shard_dir)
        with safe_open(path, framework="pt") as f:
            if names == [""]:
                names = [k for k in f.keys() if want(k)]
            for name in sorted(names):
                W = f.get_tensor(name)
                if (
                    W.ndim == 2
                    and cfg.min_dim <= min(W.shape)
                    and max(W.shape) <= cfg.max_dim
                ):
                    yield name, W.float()
        if not cfg.keep_shards:
            os.remove(path)


def main(cfg: Config) -> None:
    run = create_run("frontier_breakeven", cfg)
    rows = []
    source = iter_gpt2(cfg) if cfg.model == "gpt2" else iter_hf(cfg)
    for name, W in source:
        rows.append(matrix_row(name, W, cfg))
        r = rows[-1]
        print(
            f"{name} ({r['m']}x{r['p']}): "
            f"lr_margin={r['lr_margin_bits']:+.3f}b (r={r['lr_best_r']}, "
            f"eps={r['lr_eps']:.3f}, cost={r['lr_db']:.2f}bpw)  "
            f"sp_margin={r['sp_margin_bits']:+.3f}b (k={r['sp_best_k']})  "
            f"stable_rank={r['stable_rank']:.0f}",
            flush=True,
        )
        del W
        gc.collect()
    df = pd.DataFrame(rows)
    write_metrics(run, df)
    pos = df[(df.lr_margin_bits > 0) | (df.sp_margin_bits > 0)]
    print(
        f"\n{len(df)} matrices; {len(pos)} with a positive break-even margin "
        f"(lr max {df.lr_margin_bits.max():+.3f}b, sp max {df.sp_margin_bits.max():+.3f}b)"
    )
    print(f"-> {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
