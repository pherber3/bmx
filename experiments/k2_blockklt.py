"""K2 block-KLT risk gate: how much of the full-CxC eigwaterfill win survives per-head?

The eigenbasis waterfill result (docs/2026-06-21-k2-eigwaterfill-results.md) found a
real 2.24x logit-distortion win for full CxC KLT + reverse waterfill over uniform bits
at matched idealized bpe — killed only on rotation storage (+16*C/S = 8 bpe at S=2048).
The listed amortization fix is a per-head d x d block-diagonal KLT (16*d/S ~ +1 bpe).
This experiment measures the retention fraction of that win under the block restriction.

Arms per (layer, budget), all on k_pre, all lowrank(rank)+residual, scored on logit
distortion vs real stored queries (RoPE-at-read) and rel_fro:
  - uniform            lowrank_rtn_channel (integer budget) / deterministic
                       variance-blind interleaved floor/ceil bits (fractional budget)
  - hadamard_uniform   residual rotated by seeded randomized Hadamard (0 stored bits,
                       the TurboQuant-style baseline), same uniform bits, unrotated
  - eig_full           full CxC KLT + waterfill  (the 2.24x anchor; honest +16*C/S)
  - blockdiag_perhead  per-head d x d KLT + per-head waterfill (honest +16*d/S)
  - blockdiag_shared   per-head d x d KLT + ONE bit vector over eigen-index shared
                       across heads (pooled variance) — isolates allocation vs basis

bpe columns: `bpe` is idealized (rotation-free, all other metadata counted);
`bpe_honest` adds the stored-rotation charge at S=2048; `rot_bpe_deploy` quotes the
same charge amortized at deploy_S (32768) separately.

Usage
-----
    uv run python experiments/k2_blockklt.py \
        --cache-path results/cache/llama-3.1-8b_2048.safetensors \
        --model-label llama-3.1-8b \
        --model-name meta-llama/Llama-3.1-8B
"""

from __future__ import annotations

import dataclasses
import math
import re

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.codecs import (
    _klt_basis,
    _unrotate,
    allocate_channel_bits,
    factor_bits,
    quantize_by_bits,
    quantize_cache,
    scale_bits,
    tier_bits,
)
from bmx.cache.collect import from_matrix, load_cache, to_matrix
from bmx.cache.metrics import logit_distortion, rel_fro
from bmx.cache.rope import apply_rope
from bmx.decomp.lrs import truncated_svd
from bmx.quant.hadamard import randomized_hadamard

_LAYER_RE = re.compile(r"^layer(\d+)\.(k|v|q|k_pre)$")


@dataclasses.dataclass
class Config:
    cache_path: str
    model_label: str = ""
    model_name: str = ""  # HF repo id for RoPE (empty => score in stored basis)
    budgets: tuple[float, ...] = (2.0, 2.5, 3.0)
    group: int = 64
    rank: int = 16
    tiers: tuple[int, ...] = (0, 2, 3, 4)
    seed: int = 0
    layer_stride: int = 1  # evaluate layers 0, stride, 2*stride, ...
    deploy_s: int = 32768  # deployment context for the amortized rotation quote
    out_root: str = ""


def _stored_lowrank(
    M: torch.Tensor, factors: tuple
) -> tuple[torch.Tensor, torch.Tensor]:
    """L from fp16-stored factors (codecs convention) and the residual R = M - L."""
    Us, V = factors
    L = Us.half().float() @ V.half().float().mT
    return L, M - L


def _uniform_bits_vector(C: int, budget: float) -> torch.Tensor:
    """Variance-blind per-channel bits with mean == budget.

    Integer budget -> constant. Fractional -> floor everywhere, ceil on
    round(frac*C) evenly-spread channel indices (deterministic, allocation-free —
    a fair 'uniform bits' baseline at fractional bpe, no data-dependent metadata).
    """
    lo = int(math.floor(budget))
    n_hi = round((budget - lo) * C)
    bits = torch.full((C,), lo, dtype=torch.int64)
    if n_hi > 0:
        idx = (torch.arange(n_hi).double() * (C / n_hi)).long()
        bits[idx] = lo + 1
    assert abs(bits.float().mean().item() - budget) < 1e-6
    return bits


def _uniform_arm(
    M: torch.Tensor, budget: float, group: int, rank: int, factors: tuple
) -> tuple[torch.Tensor, float]:
    """Uniform-bits lowrank arm at possibly-fractional budget (no rotation)."""
    if float(budget).is_integer():
        return quantize_cache(
            "lowrank_rtn_channel",
            M,
            bits=int(budget),
            group=group,
            rank=rank,
            svd_factors=factors,
        )
    S, C = M.shape
    L, R = _stored_lowrank(M, factors)
    R_hat = quantize_by_bits(R, _uniform_bits_vector(C, budget), group)
    bpe = budget + scale_bits(group) + factor_bits(rank, S, C)
    return L + R_hat, bpe


def _hadamard_uniform_arm(
    M: torch.Tensor, budget: float, group: int, rank: int, factors: tuple, seed: int
) -> tuple[torch.Tensor, float]:
    """TurboQuant-style baseline: seeded randomized-Hadamard rotation of the residual
    (0 stored bits), uniform bits in the rotated basis, unrotate before scoring."""
    S, C = M.shape
    L, R = _stored_lowrank(M, factors)
    R_rot = randomized_hadamard(R, seed)
    R_hat_rot = quantize_by_bits(R_rot, _uniform_bits_vector(C, budget), group)
    R_hat = _unrotate(R_hat_rot, seed)
    bpe = budget + scale_bits(group) + factor_bits(rank, S, C)
    return L + R_hat, bpe


def _blockdiag_shared_arm(
    M: torch.Tensor,
    budget: float,
    group: int,
    rank: int,
    tiers: tuple[int, ...],
    factors: tuple,
    h_kv: int,
) -> tuple[torch.Tensor, float]:
    """Per-head d x d KLT with ONE waterfill bit vector over eigen-index, shared
    across heads (allocated on head-pooled variance). Isolates basis-vs-allocation:
    same rotations as blockdiag_perhead, allocation forbidden from adapting per head.
    """
    S, C = M.shape
    assert C % h_kv == 0
    d = C // h_kv
    L, R = _stored_lowrank(M, factors)
    Qs = []
    R_rot = torch.zeros_like(R)
    for hh in range(h_kv):
        sl = slice(hh * d, (hh + 1) * d)
        Qh = _klt_basis(R[:, sl])
        Qs.append(Qh)
        R_rot[:, sl] = R[:, sl] @ Qh
    # Pool heads as extra rows: (h_kv*S, d); variance per eigen-index across all heads.
    pooled = R_rot.reshape(S, h_kv, d).transpose(0, 1).reshape(h_kv * S, d)
    bits_d = allocate_channel_bits(pooled, budget, tiers=tiers, axis=0)  # (d,)
    R_hat = torch.zeros_like(R)
    for hh in range(h_kv):
        sl = slice(hh * d, (hh + 1) * d)
        Rh_hat_rot = quantize_by_bits(R_rot[:, sl], bits_d, group)
        R_hat[:, sl] = Rh_hat_rot @ Qs[hh].mT
    payload = float(bits_d.float().mean().item())
    bpe = payload + scale_bits(group) + factor_bits(rank, S, C) + tier_bits(tiers, S)
    return L + R_hat, bpe


def main(cfg: Config) -> pd.DataFrame:
    run = (
        create_run("k2_blockklt", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k2_blockklt", cfg)
    )

    cache = load_cache(cfg.cache_path)
    layer_keys: dict[int, dict[str, torch.Tensor]] = {}
    for key, tensor in cache.items():
        m = _LAYER_RE.match(key)
        if m is None:
            continue
        layer_keys.setdefault(int(m.group(1)), {})[m.group(2)] = tensor

    rope_ready = False
    hf_config = None
    if cfg.model_name:
        from transformers import AutoConfig

        from bmx.cache.rope import rope_cos_sin

        hf_config = AutoConfig.from_pretrained(cfg.model_name)
        rope_ready = True

    cos_sin: dict[int, tuple] = {}

    def get_cos_sin(S: int):
        if S not in cos_sin:
            cos_sin[S] = rope_cos_sin(hf_config, S)
        return cos_sin[S]

    layers = [i for i in sorted(layer_keys) if i % cfg.layer_stride == 0]
    rows: list[dict] = []
    checked_rotation_charge = False
    for layer_i in layers:
        km = layer_keys[layer_i]
        if "k_pre" not in km or "q" not in km:
            continue
        k_pre = km["k_pre"]
        h_kv, S, d = k_pre.shape
        M = to_matrix(k_pre).float()  # (S, C)
        C = M.shape[1]
        Q = km["q"].float()

        cos = sin = None
        K_post_true = None
        if rope_ready:
            cos, sin = get_cos_sin(S)
            K_post_true = apply_rope(k_pre.float(), cos, sin)

        factors = truncated_svd(M, cfg.rank)

        def score(M_hat: torch.Tensor) -> tuple[float, float]:
            K_hat = from_matrix(M_hat, h_kv)
            rf = rel_fro(M_hat, M)
            if rope_ready:
                K_hat_rope = apply_rope(K_hat.float(), cos, sin)
                lg = logit_distortion(K_post_true, K_hat_rope, Q)
            else:
                lg = logit_distortion(k_pre.float(), K_hat, Q)
            return rf, lg

        rot_full = 16.0 * C / S
        rot_head = 16.0 * d / S
        rot_full_deploy = 16.0 * C / cfg.deploy_s
        rot_head_deploy = 16.0 * d / cfg.deploy_s

        # One-time sanity check: our arithmetic rotation charges must match the
        # codec's own charge_rotation accounting.
        if not checked_rotation_charge:
            _, b_i = quantize_cache(
                "lowrank_eigwaterfill_channel",
                M,
                bits=cfg.budgets[-1],
                group=cfg.group,
                rank=cfg.rank,
                tiers=cfg.tiers,
                svd_factors=factors,
            )
            _, b_h = quantize_cache(
                "lowrank_eigwaterfill_channel",
                M,
                bits=cfg.budgets[-1],
                group=cfg.group,
                rank=cfg.rank,
                tiers=cfg.tiers,
                charge_rotation=True,
                svd_factors=factors,
            )
            assert abs((b_h - b_i) - rot_full) < 1e-9, (b_h, b_i, rot_full)
            _, b_i = quantize_cache(
                "lowrank_blockdiagwaterfill_channel",
                M,
                bits=cfg.budgets[-1],
                group=cfg.group,
                rank=cfg.rank,
                tiers=cfg.tiers,
                h_kv=h_kv,
                svd_factors=factors,
            )
            _, b_h = quantize_cache(
                "lowrank_blockdiagwaterfill_channel",
                M,
                bits=cfg.budgets[-1],
                group=cfg.group,
                rank=cfg.rank,
                tiers=cfg.tiers,
                h_kv=h_kv,
                charge_rotation=True,
                svd_factors=factors,
            )
            assert abs((b_h - b_i) - rot_head) < 1e-9, (b_h, b_i, rot_head)
            checked_rotation_charge = True

        for budget in cfg.budgets:
            uni, bpe_uni = _uniform_arm(M, budget, cfg.group, cfg.rank, factors)
            had, bpe_had = _hadamard_uniform_arm(
                M, budget, cfg.group, cfg.rank, factors, cfg.seed
            )
            eig, bpe_eig = quantize_cache(
                "lowrank_eigwaterfill_channel",
                M,
                bits=budget,
                group=cfg.group,
                rank=cfg.rank,
                tiers=cfg.tiers,
                svd_factors=factors,
            )
            bd, bpe_bd = quantize_cache(
                "lowrank_blockdiagwaterfill_channel",
                M,
                bits=budget,
                group=cfg.group,
                rank=cfg.rank,
                tiers=cfg.tiers,
                h_kv=h_kv,
                svd_factors=factors,
            )
            bds, bpe_bds = _blockdiag_shared_arm(
                M, budget, cfg.group, cfg.rank, cfg.tiers, factors, h_kv
            )

            arm_rows = [
                ("uniform", uni, bpe_uni, 0.0, 0.0),
                ("hadamard_uniform", had, bpe_had, 0.0, 0.0),
                ("eig_full", eig, bpe_eig, rot_full, rot_full_deploy),
                ("blockdiag_perhead", bd, bpe_bd, rot_head, rot_head_deploy),
                ("blockdiag_shared", bds, bpe_bds, rot_head, rot_head_deploy),
            ]
            for arm, M_hat, bpe, rot_bpe, rot_deploy in arm_rows:
                rf, lg = score(M_hat)
                rows.append(
                    dict(
                        model=cfg.model_label or "unknown",
                        layer=layer_i,
                        kind="k_pre",
                        arm=arm,
                        budget=budget,
                        rank=cfg.rank,
                        bpe=bpe,
                        rot_bpe=rot_bpe,
                        bpe_honest=bpe + rot_bpe,
                        rot_bpe_deploy=rot_deploy,
                        rel_fro=rf,
                        logit_rope=lg,
                    )
                )
                print(
                    f"  L{layer_i:2d} b={budget:.1f} {arm:18s} bpe={bpe:.3f} "
                    f"(+rot {rot_bpe:.3f}) logit={lg:.4f} rel_fro={rf:.4f}",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    write_metrics(run, df)

    log4 = math.log(4.0)
    print("\n" + "=" * 78)
    print("SUMMARY (mean over layers; eff_bits = log4(D_uniform/D_arm) at same budget)")
    for budget in cfg.budgets:
        sub = df[df.budget == budget]
        uni_l = sub[sub.arm == "uniform"].set_index("layer")["logit_rope"]
        print(f"\n-- budget {budget:.1f} --")
        for arm in [
            "uniform",
            "hadamard_uniform",
            "eig_full",
            "blockdiag_perhead",
            "blockdiag_shared",
        ]:
            a = sub[sub.arm == arm].set_index("layer")
            ratio = float((uni_l / a["logit_rope"]).mean())
            wins = int((a["logit_rope"] < uni_l.reindex(a.index)).sum())
            eff = math.log(ratio) / log4 if ratio > 0 else float("nan")
            print(
                f"  {arm:18s} bpe={a.bpe.mean():.3f} honest={a.bpe_honest.mean():.3f} "
                f"logit={a.logit_rope.mean():.4f} ratio_vs_uniform={ratio:.3f} "
                f"eff_bits={eff:+.3f} wins={wins}/{len(a)}"
            )
        eig_l = sub[sub.arm == "eig_full"].set_index("layer")["logit_rope"]
        for arm in ["blockdiag_perhead", "blockdiag_shared"]:
            a = sub[sub.arm == arm].set_index("layer")["logit_rope"]
            g_full = math.log(float((uni_l / eig_l).mean())) / log4
            g_arm = math.log(float((uni_l / a).mean())) / log4
            print(
                f"  retention {arm}: {g_arm:.3f} / {g_full:.3f} bits = "
                f"{g_arm / g_full:.1%}"
                if g_full > 0
                else f"  retention {arm}: full-KLT gain <= 0"
            )
    print(f"\n-> {run}")
    return df


if __name__ == "__main__":
    main(tyro.cli(Config))
