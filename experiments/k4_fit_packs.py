"""K4: corpus pack fitting — fit per-layer SpectralBasis from N corpus caches
and write ONE pack file (+ a small spectra parquet).

Σ_k for each layer is the concat of ALL corpus caches' k_pre matrices (the
deployment-realistic fit: one basis per layer, calibrated on a corpus, not on
the sequence being scored — see k4_spectra.py's "corpus" fit_mode).

W (the query second moment that defines the whitener) has two sources:
  - "corpus": pooled from the corpus caches' OWN stored queries — each cache
    contributes its own query_position_moment (using its own cos/sin, since S
    may differ per cache), averaged across caches. This is the deployment-
    grade choice: at pack-fit time we only ever have corpus-side queries, not
    the queries of the sequence the pack will later score (fixing the
    referee's circularity concern from k4_spectra's oracle/heldout modes).
  - "none": identity_whitener — the unweighted-KLT fallback pack.

Across-layer allocation (the G2 lever): with `alloc_sens_parquet` set (a
k4_alloc run's metrics.parquet, Part-A sensitivity rows), each entry of
`budgets` becomes a TARGET MEAN — per-layer budgets b_l with mean(b_l) ==
target are chosen by k4_alloc's `greedy_layer_allocation` over per-layer
distortion-vs-budget curves built from the fitted spectra, weighted by the
per-layer sensitivities s_l. The pack file keeps the uniform format: layer i's
bits under label `bits_b{target:g}` are simply waterfilled at b_l instead of
at the target, so load_packs/streaming/recipes select allocated packs
unchanged.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics
from bmx.cache.collect import to_matrix
from bmx.cache.spectral import (
    SpectralBasis,
    assemble_whitener,
    fit_spectral_basis,
    identity_whitener,
    pack_from_basis,
    save_pack_file,
)
from experiments._k4_common import corpus_query_moment, load_layer_keys, setup_rope
from experiments.k4_alloc import greedy_layer_allocation

_W_SOURCES = {"corpus", "none"}

# Candidate per-layer budget grid for the across-layer allocator. 1.0 is the
# practical floor (below it the waterfill drops nearly every direction), 4.5
# comfortably brackets every target we run (default budgets top out at 3.2)
# while staying well inside the mean range representable by the default tiers
# (0..8); step 0.25 gives the greedy walk a mean-bits resolution of
# 0.25/n_layer per upgrade.
_ALLOC_GRID = tuple(1.0 + 0.25 * i for i in range(15))  # 1.0 .. 4.5

# Same floor policy as k4_alloc Part A: ppl noise can push a truly-flat
# layer's measured s_i to ~0 or slightly negative; a small positive floor
# keeps every layer eligible for upgrades without changing the ranking of
# genuinely sensitive layers.
_SENS_FLOOR = 1e-6


@dataclasses.dataclass
class Config:
    corpus_cache_paths: tuple[str, ...]
    out_path: str
    model_label: str = ""
    model_name: str = ""  # HF repo id for RoPE; empty => no-RoPE (ones/zeros) tables
    budgets: tuple[float, ...] = (2.0, 2.2, 2.5, 2.7, 3.0, 3.2)
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8)
    group: int = 64
    ridge: float = 1e-3
    position_stride: int = 8
    w_source: str = "corpus"
    # k4_alloc metrics.parquet (Part-A sensitivity rows). Empty = uniform
    # budgets (unchanged behavior); set = budgets are TARGET MEANS allocated
    # across layers via greedy_layer_allocation.
    alloc_sens_parquet: str = ""
    out_root: str = ""


def _load_sensitivities(parquet_path: str, layers: list[int]) -> dict[int, float]:
    """Per-layer s_i from a k4_alloc run's Part-A rows (kind=="sensitivity"),
    floored at _SENS_FLOOR (k4_alloc's own clamp policy)."""
    df = pd.read_parquet(parquet_path)
    sub = df[df.kind == "sensitivity"]
    assert not sub.empty, f"no sensitivity rows in {parquet_path!r} — wrong parquet?"
    s_raw = {int(layer): float(v) for layer, v in zip(sub.layer, sub.s_i)}
    missing = set(layers) - set(s_raw)
    assert not missing, f"sensitivity parquet missing layers {sorted(missing)}"
    return {layer: max(v, _SENS_FLOOR) for layer, v in s_raw.items()}


def _distortion_curves(
    bases: dict[int, SpectralBasis],
    *,
    tiers: tuple[int, ...],
    group: int,
    grid: tuple[float, ...] = _ALLOC_GRID,
) -> dict[int, dict[float, float]]:
    """Per-layer distortion-vs-budget curves from the fitted spectra.

    For each candidate budget b, run the REAL allocator (`pack_from_basis`,
    i.e. allocate_bits_from_variance on `basis.lam64` — the exact fp64 tensor
    it waterfills at save time) and score the Gaussian rate-distortion proxy
    D(b) = Σ_i lam64_i · 4^(−bits_i); dropped (0-bit) directions contribute
    lam_i in full (4^0 = 1).
    """
    curves: dict[int, dict[float, float]] = {}
    for layer_i, basis in bases.items():
        curves[layer_i] = {
            float(b): float(
                (
                    basis.lam64
                    * torch.pow(
                        4.0,
                        -pack_from_basis(
                            basis, b, tiers=tiers, group=group
                        ).bits.double(),
                    )
                ).sum()
            )
            for b in grid
        }
    return curves


def _allocate_layer_budgets(
    cfg: Config, bases: dict[int, SpectralBasis], layers: list[int]
) -> tuple[dict[float, dict[int, float]], dict]:
    """Greedy across-layer allocation for every target in cfg.budgets.

    Returns (layer_budgets, alloc_meta) — the per-target {layer: b_l} maps and
    the sidecar-traceability block (allocation, sensitivities, parquet path).
    """
    s = _load_sensitivities(cfg.alloc_sens_parquet, layers)
    curves = _distortion_curves(bases, tiers=cfg.tiers, group=cfg.group)
    grid = _ALLOC_GRID
    step = grid[1] - grid[0]

    layer_budgets: dict[float, dict[int, float]] = {}
    for target in cfg.budgets:
        assert grid[0] <= target <= grid[-1], (
            f"target mean {target} outside allocator grid [{grid[0]}, {grid[-1]}]"
        )
        alloc = greedy_layer_allocation(curves, s, grid, target)
        realized = sum(alloc.values()) / len(alloc)
        # Same tolerance form as k4_alloc's uniform comparator (grid step /
        # n_layer there is 1.0/n_layer at integer bits; our step is 0.25).
        assert abs(realized - target) <= step / len(alloc) + 1e-9, (
            f"allocated mean {realized:.4f} misses target {target} beyond "
            f"{step}/{len(alloc)} tolerance"
        )
        layer_budgets[target] = alloc
        print(
            f"  [alloc] target={target:g} realized_mean={realized:.4f}: "
            f"{ {layer: b for layer, b in sorted(alloc.items())} }",
            flush=True,
        )

    alloc_meta = {
        "alloc_sens_parquet": cfg.alloc_sens_parquet,
        "alloc": {
            f"{target:g}": {str(layer): b for layer, b in sorted(a.items())}
            for target, a in layer_budgets.items()
        },
        "alloc_sensitivities": {str(layer): s[layer] for layer in sorted(s)},
    }
    return layer_budgets, alloc_meta


def main(cfg: Config):
    assert cfg.w_source in _W_SOURCES, f"w_source={cfg.w_source!r} not in {_W_SOURCES}"
    assert len(cfg.corpus_cache_paths) >= 1, "corpus_cache_paths must be non-empty"

    run = (
        create_run("k4_fit_packs", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_fit_packs", cfg)
    )

    per_cache_layer_keys = [load_layer_keys(p) for p in cfg.corpus_cache_paths]

    layers = sorted(per_cache_layer_keys[0].keys())
    for lk in per_cache_layer_keys[1:]:
        assert sorted(lk.keys()) == layers, "corpus caches disagree on layer set"

    # RoPE setup, per corpus cache (S may differ per cache).
    rope_ready = False
    get_cos_sins = []
    for lk in per_cache_layer_keys:
        ready, get_cos_sin = setup_rope(cfg.model_name, lk, layers)
        rope_ready = rope_ready or ready
        get_cos_sins.append(get_cos_sin)

    bases: dict[int, SpectralBasis] = {}
    rows: list[dict] = []
    model_label = cfg.model_label or "unknown"

    for layer_i in layers:
        # ---- Σ_k: concat of ALL corpus caches' k_pre matrices -------------
        h_kv = d = None
        M_parts = []
        for lk in per_cache_layer_keys:
            k_pre_t = lk[layer_i]["k_pre"]
            this_h_kv, S_c, this_d = k_pre_t.shape
            if h_kv is None:
                h_kv, d = this_h_kv, this_d
            else:
                assert (this_h_kv, this_d) == (h_kv, d), (
                    f"corpus cache layer{layer_i}.k_pre shape "
                    f"{tuple(k_pre_t.shape)} incompatible with (h_kv={h_kv}, d={d})"
                )
            M_parts.append(to_matrix(k_pre_t))
        M_fit = torch.cat(M_parts, dim=0)
        C = h_kv * d

        # ---- W: pooled query second moment, or identity --------------------
        if cfg.w_source == "corpus":
            W_blocks = corpus_query_moment(
                per_cache_layer_keys,
                get_cos_sins,
                rope_ready,
                layer_i,
                h_kv,
                d,
                cfg.position_stride,
            )
            Wh, Wh_inv = assemble_whitener(W_blocks, ridge=cfg.ridge)
        else:  # "none"
            Wh, Wh_inv = identity_whitener(C)

        basis = fit_spectral_basis(M_fit, Wh, Wh_inv)
        bases[layer_i] = basis

        print(
            f"[layer {layer_i}] (h_kv={h_kv}, d={d}, C={C}, "
            f"S_fit={M_fit.shape[0]}) basis fit",
            flush=True,
        )

    # ---- across-layer allocation (needs every layer's basis) --------------
    layer_budgets = None
    alloc_meta: dict = {}
    if cfg.alloc_sens_parquet:
        layer_budgets, alloc_meta = _allocate_layer_budgets(cfg, bases, layers)

    for layer_i in layers:
        basis = bases[layer_i]
        for budget in cfg.budgets:
            b_l = budget if layer_budgets is None else layer_budgets[budget][layer_i]
            pack = pack_from_basis(basis, b_l, tiers=cfg.tiers, group=cfg.group)
            lam = pack.lam
            am_gm = (lam.mean() / lam.clamp_min(1e-12).log().mean().exp()).item()
            top16_energy = (lam[:16].sum() / lam.sum().clamp_min(1e-12)).item()
            n_zero_dirs = int((pack.bits == 0).sum())
            row = dict(
                model=model_label,
                layer=layer_i,
                budget=float(budget),
                am_gm=am_gm,
                top16_energy=top16_energy,
                n_zero_dirs=n_zero_dirs,
            )
            if layer_budgets is not None:
                row["budget_layer"] = float(b_l)
            rows.append(row)

    save_pack_file(
        cfg.out_path,
        bases,
        cfg.budgets,
        tiers=cfg.tiers,
        group=cfg.group,
        meta={
            "model_label": model_label,
            "git_sha": git_sha(),
            "corpus_cache_paths": list(cfg.corpus_cache_paths),
            "w_source": cfg.w_source,
            "ridge": cfg.ridge,
            **alloc_meta,
        },
        layer_budgets=layer_budgets,
    )

    df = pd.DataFrame(rows)
    write_metrics(run, df)

    print(f"\nWrote pack file -> {cfg.out_path}")
    print(f"-> {run}")

    return run


if __name__ == "__main__":
    main(tyro.cli(Config))
