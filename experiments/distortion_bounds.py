"""F1 — distortion vs bit-width (TurboQuant Fig-3 parity).

Measures ABSOLUTE distortions on UNIT-NORM vectors in TurboQuant's own
definitions and overlays the closed-form bounds (plot module):

  D_mse(x)  = ||x - x_hat||^2   for ||x|| = 1        (averaged over vectors)
  D_prod    = E |<y, x> - <y, x_hat>|^2  for unit x, unit query y

This is NOT the relative Frobenius error in k2_cache_arms — reusing rel_fro
here and overlaying the bounds would be measuring a different quantity.

Source of vectors:
  --source sphere : i.i.d. unit vectors in R^d (TurboQuant's theoretical
                    setting; the default so tests/CI never touch the gitignored
                    real cache).
  --source cache  : rows of the real KV-cache matrix (opt-in; needs
                    results/cache/<...>.safetensors present).

Each row of the source matrix is one vector; we normalize each to unit norm
before quantizing so the measured distortions match TurboQuant's unit-norm
convention exactly.

Usage
-----
    uv run python experiments/distortion_bounds.py            # sphere control
    uv run python experiments/distortion_bounds.py --source cache \
        --cache-path results/cache/llama-3.1-8b_2048.safetensors
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.codecs import quantize_cache

# turboquant_prod spends one bit on the QJL sign sketch, so it needs bits >= 2.
_PROD_ARMS = frozenset({"turboquant_prod"})
# lowrank_rtn_channel needs an explicit rank (0 is invalid).
_LOWRANK_ARMS = frozenset({"lowrank_rtn_channel"})

_DEFAULT_ARMS = ("turboquant_mse", "turboquant_prod", "lowrank_rtn_channel")


# ---------------------------------------------------------------------------
# Vector sources
# ---------------------------------------------------------------------------


def _sphere_matrix(n_vectors: int, d: int, seed: int) -> torch.Tensor:
    """(n_vectors, d) fp32 matrix whose rows are i.i.d. unit vectors in R^d."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_vectors, d, generator=g, dtype=torch.float32)
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return X


def _cache_matrix(cache_path: str, n_vectors: int, d: int) -> torch.Tensor:
    """Rows of the real KV-cache matrix, trimmed to (<=n_vectors, <=d).

    Loads layer0.k as (h, S, d) and lays it out as the (S, h*d) codec matrix,
    then takes the leading n_vectors rows and leading d columns (d is used as a
    working column budget so the sphere/cache runs share the same axis label).
    """
    from bmx.cache.collect import load_cache, to_matrix

    cache = load_cache(cache_path)
    # Prefer pre-RoPE keys (the quantization target); fall back to k then v.
    for key in ("layer0.k_pre", "layer0.k", "layer0.v"):
        if key in cache:
            M = to_matrix(cache[key])  # (S, C)
            break
    else:
        raise KeyError(f"no layer0.{{k_pre,k,v}} tensor in {cache_path}")
    n = min(n_vectors, M.shape[0])
    c = min(d, M.shape[1])
    return M[:n, :c].contiguous().float()


def _source_matrix(source: str, n_vectors: int, d: int, seed: int, cache_path: str):
    if source == "sphere":
        return _sphere_matrix(n_vectors, d, seed)
    if source == "cache":
        return _cache_matrix(cache_path, n_vectors, d)
    raise ValueError(f"unknown source {source!r}; use 'sphere' or 'cache'")


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure_distortions(
    source: str = "sphere",
    arms: tuple[str, ...] = _DEFAULT_ARMS,
    bit_list: tuple[int, ...] = (1, 2, 3, 4),
    d: int = 128,
    n_vectors: int = 256,
    seed: int = 0,
    *,
    group: int = 64,
    rank: int = 8,
    cache_path: str = "",
) -> pd.DataFrame:
    """Measure per-(arm, bitwidth) D_mse and D_prod on unit-norm vectors.

    Returns a DataFrame with columns
    ``{arm, bitwidth, d_mse, d_prod, bpe, n_vectors, d, source}``.

    D_mse   = mean_i ||x_i - x_hat_i||^2 with each x_i normalized to unit norm.
    D_prod  = mean_i mean_j |<y_j, x_i> - <y_j, x_hat_i>|^2 for random unit y_j.

    turboquant_prod is skipped at bits < 2 (it burns one bit on QJL).
    """
    X = _source_matrix(source, n_vectors, d, seed, cache_path)
    # unit-norm each row so absolute distortions match the paper's convention.
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    n, dd = X.shape

    # Fixed bank of random unit queries for D_prod (shared across arms/bits).
    g = torch.Generator().manual_seed(seed + 1)
    n_query = 64
    Y = torch.randn(n_query, dd, generator=g, dtype=torch.float32)
    Y = Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-12)

    ref_ip = X @ Y.mT  # (n, n_query) true inner products

    rows: list[dict] = []
    for arm in arms:
        for bits in bit_list:
            if arm in _PROD_ARMS and bits < 2:
                # QJL sign bit leaves nothing for the payload at 1 bit.
                continue
            kwargs = dict(bits=bits, seed=seed, group=group)
            if arm in _LOWRANK_ARMS:
                kwargs["rank"] = min(rank, min(n, dd))
            X_hat, bpe = quantize_cache(arm, X, **kwargs)

            err = X_hat - X
            d_mse = (err * err).sum(dim=1).mean().item()

            hat_ip = X_hat @ Y.mT
            d_prod = ((hat_ip - ref_ip) ** 2).mean().item()

            rows.append(
                dict(
                    arm=arm,
                    bitwidth=bits,
                    d_mse=d_mse,
                    d_prod=d_prod,
                    bpe=bpe,
                    n_vectors=n,
                    d=dd,
                    source=source,
                )
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Config:
    source: str = "sphere"  # "sphere" (default; theory control) or "cache"
    cache_path: str = "results/cache/llama-3.1-8b_2048.safetensors"
    arms: tuple[str, ...] = _DEFAULT_ARMS
    bits: tuple[int, ...] = (1, 2, 3, 4)
    d: int = 128
    n_vectors: int = 1024
    group: int = 64
    rank: int = 8
    seed: int = 0


def main(cfg: Config) -> None:
    run = create_run("distortion_bounds", cfg)
    df = measure_distortions(
        source=cfg.source,
        arms=cfg.arms,
        bit_list=cfg.bits,
        d=cfg.d,
        n_vectors=cfg.n_vectors,
        seed=cfg.seed,
        group=cfg.group,
        rank=cfg.rank,
        cache_path=cfg.cache_path,
    )
    write_metrics(run, df)

    print(f"\nsource={cfg.source}  d={cfg.d}  n_vectors={cfg.n_vectors}")
    for _, r in df.iterrows():
        print(
            f"  arm={r['arm']:22s} b={int(r['bitwidth'])} "
            f"D_mse={r['d_mse']:.5g}  D_prod={r['d_prod']:.5g}  bpe={r['bpe']:.3f}"
        )
    print(f"\nTotal rows: {len(df)}")
    print(f"-> {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
