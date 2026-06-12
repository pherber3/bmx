"""Avenue 1: low-rank+sparse quantization residual, two stages in order.

Stage A — structural diagnostic (original basis, NO quantization). One
two-step estimator call per (W, r, k) tests three assumptions at once:
(a) subspace match: does L-hat's column space align with W's own top-r
    left singular subspace (the structure a2 found via Tucker)?
(b) support match: does supp(S-hat) concentrate on the channels d1's
    outlier_mass flags? (Spearman rank corr + top-10 channel overlap.)
(c) spikiness go/no-go: ||L||_max <= alpha/sqrt(d1 d2) — reported as
    spikiness_ratio(L) (alpha-hat); if L-hat is itself spiky the L/S split
    is ill-posed (Wainwright §10.7) and the avenue narrows.

Stage B — compression at matched TOTAL bits: the four bmx.quant.arms pipelines
on a (bits, r, sparse_frac) grid, with L/S storage (values + indices) counted.
"""

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.census import subspace_overlap
from bmx.decomp.lrs import spikiness_ratio, two_step_lrs
from bmx.quant.arms import ARMS, LRS_ARMS, fit_ls, reconstruct_arm, total_bits
from bmx.quant.stats import ip_distortion, kurtosis, outlier_mass
from bmx.stacks.gpt2 import load_gpt2_state


@dataclasses.dataclass
class Config:
    stage: str = "both"  # "a", "b", or "both"
    model_name: str = "gpt2"
    weights: tuple[str, ...] = (
        "transformer.h.5.attn.c_attn.weight",
        "transformer.h.5.attn.c_proj.weight",
        "transformer.h.5.mlp.c_fc.weight",
        "transformer.h.5.mlp.c_proj.weight",
    )
    # d1's worst structured offender; diagnostic-only (not a matmul weight)
    stage_a_extra: tuple[str, ...] = ("transformer.wpe.weight",)
    ranks: tuple[int, ...] = (0, 8, 16, 32, 64)
    sparse_fracs: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2)
    bits: tuple[int, ...] = (2, 3, 4)
    group_size: int = 64
    n_alternations: int = 2
    n_probes: int = 512
    seed: int = 0


def _top_overlap(a: torch.Tensor, b: torch.Tensor, n: int = 10) -> float:
    ia = set(a.topk(n).indices.tolist())
    ib = set(b.topk(n).indices.tolist())
    return len(ia & ib) / n


def stage_a(cfg: Config, sd: dict) -> pd.DataFrame:
    rows = []
    for name in cfg.weights + cfg.stage_a_extra:
        W = sd[name].to(torch.float64)
        m, p = W.shape
        U_W, _, _ = torch.linalg.svd(W, full_matrices=False)
        om = outlier_mass(W)
        spikiness_w = spikiness_ratio(W)
        kurtosis_w = kurtosis(W, dim=-1).mean().item()
        for r in cfg.ranks:
            for frac in cfg.sparse_fracs:
                k = int(frac * m * p)
                if r == 0 or k == 0:
                    continue  # diagnostics need both parts present
                Us, V, S = two_step_lrs(W, r, k, cfg.n_alternations)
                L = Us @ V.mT
                R = W - L - S
                supp_frac = (S != 0).to(torch.float64).mean(dim=0)
                row = {
                    "weight": name,
                    "shape": str((m, p)),
                    "r": r,
                    "sparse_frac": frac,
                    "k": k,
                    "subspace_overlap": subspace_overlap(Us, U_W[:, :r]),
                    "supp_spearman": pd.Series(supp_frac.numpy()).corr(
                        pd.Series(om.numpy()), method="spearman"
                    ),
                    "supp_top10_overlap": _top_overlap(supp_frac, om),
                    "spikiness_W": spikiness_w,
                    "spikiness_L": spikiness_ratio(L),
                    "spikiness_S": spikiness_ratio(S),  # k > 0 in this loop
                    "rel_error_LS": (R.norm() / W.norm()).item(),
                    "kurtosis_W": kurtosis_w,
                    "kurtosis_R": kurtosis(R, dim=-1).mean().item(),
                }
                rows.append(row)
                print(
                    f"[A] {name} r={r} frac={frac:g}: overlap={row['subspace_overlap']:.3f} "
                    f"spearman={row['supp_spearman']:.3f} spike_L={row['spikiness_L']:.1f} "
                    f"kurt {row['kurtosis_W']:+.2f}->{row['kurtosis_R']:+.2f}"
                )
    return pd.DataFrame(rows)


def stage_b(cfg: Config, sd: dict) -> pd.DataFrame:
    g = torch.Generator().manual_seed(cfg.seed)
    rows = []
    for name in cfg.weights:
        W = sd[name].to(torch.float32)
        m, p = W.shape
        # GPT-2 Conv1D computes y = x @ W: probes live on the INPUT dim m,
        # so distortion is measured on W.mT (rows = output features).
        X = torch.randn(cfg.n_probes, m, generator=g, dtype=torch.float32)
        # the L+S fit depends only on (r, k), so fit once per budget point and
        # reuse across bit-widths and both lrs arms (3 SVDs per fit saved 5x)
        ls_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        for bits in cfg.bits:
            for arm in ARMS:
                # plain/rotate arms have no (r, k) grid; lrs arms sweep it
                grid = (
                    [
                        (r, frac)
                        for r in cfg.ranks
                        for frac in cfg.sparse_fracs
                        if r > 0 or frac > 0
                    ]
                    if arm in LRS_ARMS
                    else [(0, 0.0)]
                )
                for r, frac in grid:
                    k = int(frac * m * p)
                    ls = None
                    if arm in LRS_ARMS:
                        if (r, k) not in ls_cache:
                            ls_cache[(r, k)] = fit_ls(W, r, k, cfg.n_alternations)
                        ls = ls_cache[(r, k)]
                    rec, r_st, k_st = reconstruct_arm(
                        arm,
                        W,
                        bits=bits,
                        group_size=cfg.group_size,
                        r=r,
                        k=k,
                        seed=cfg.seed,
                        ls=ls,
                    )
                    tb = total_bits(
                        m, p, bits=bits, group_size=cfg.group_size, r=r_st, k=k_st
                    )
                    row = {
                        "weight": name,
                        "arm": arm,
                        "bits": bits,
                        "r": r_st,
                        "sparse_frac": frac,
                        "k": k_st,
                        "total_bits": tb,
                        "bits_per_weight": tb / (m * p),
                        "rel_error": ((rec - W).norm() / W.norm()).item(),
                        "ip_distortion": ip_distortion(W.mT, rec.mT, X),
                    }
                    rows.append(row)
                    print(
                        f"[B] {name} {arm} b={bits} r={r_st} frac={frac:g}: "
                        f"{row['bits_per_weight']:.2f} bpw  "
                        f"rel={row['rel_error']:.4f} ip={row['ip_distortion']:.4f}"
                    )
    return pd.DataFrame(rows)


def main(cfg: Config) -> None:
    sd, _ = load_gpt2_state(cfg.model_name)
    run = create_run("lrs_residual", cfg)
    if cfg.stage in ("a", "both"):
        write_metrics(run, stage_a(cfg, sd), name="stage_a")
    if cfg.stage in ("b", "both"):
        write_metrics(run, stage_b(cfg, sd), name="stage_b")
    print(f"-> {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
