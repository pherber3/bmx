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


def best_margin(
    eps: torch.Tensor, db: torch.Tensor, max_db: float
) -> tuple[float, int, float, float]:
    """Best (saved-bits - cost-bits) over a budget grid.

    eps: energy fraction captured at each grid point, db: side-info cost in
    bits/weight. Returns (margin, grid index, eps, db) at the argmax.
    """
    eps = eps.double().clamp(max=1 - 1e-9)
    saved = torch.log2(1.0 / (1.0 - eps)) / 2.0  # log4(x) = log2(x)/2
    margin = torch.where(db <= max_db, saved - db, torch.tensor(-torch.inf))
    i = int(margin.argmax())
    return margin[i].item(), i, eps[i].item(), db[i].item()


def matrix_row(name: str, W: torch.Tensor, cfg: Config) -> dict:
    m, p = W.shape
    n = m * p
    w2_total = (W.double() ** 2).sum()

    # low-rank side: cumulative spectrum energy vs fp16 factor cost
    s2 = torch.linalg.svdvals(W).double() ** 2
    eps_r = s2.cumsum(0) / w2_total
    r = torch.arange(1, len(s2) + 1, dtype=torch.float64)
    db_r = 16.0 * r * (m + p) / n
    lr_margin, i, lr_eps, lr_db = best_margin(eps_r, db_r, cfg.max_side_bpw)
    lr_best_r = i + 1

    # sparse side: cumulative top-|entry| energy vs fp16+index cost
    a2 = (W.flatten().double() ** 2).sort(descending=True).values
    idx_bits = (n - 1).bit_length()
    k_grid = torch.unique(
        torch.logspace(0, math.log10(n), steps=512, dtype=torch.float64).long()
    )
    eps_k = a2.cumsum(0)[k_grid - 1] / w2_total
    db_k = k_grid.double() * (16 + idx_bits) / n
    sp_margin, j, sp_eps, sp_db = best_margin(eps_k, db_k, cfg.max_side_bpw)
    sp_best_k = int(k_grid[j])

    sigma = W.std()
    noise_max = math.sqrt(2 * math.log(n))  # expected bulk max, in sigmas
    layer = _LAYER_RE.search(name)
    return {
        "model": cfg.model,
        "weight": name,
        "layer": int(layer.group(1)) if layer else -1,
        "m": m,
        "p": p,
        "lr_margin_bits": lr_margin,
        "lr_best_r": lr_best_r,
        "lr_eps": lr_eps,
        "lr_db": lr_db,
        "eps_r64": eps_r[min(63, len(eps_r) - 1)].item(),
        "sp_margin_bits": sp_margin,
        "sp_best_k": sp_best_k,
        "sp_eps": sp_eps,
        "sp_db": sp_db,
        "stable_rank": (s2.sum() / s2[0]).item(),
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
