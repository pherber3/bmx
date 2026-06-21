"""K2 water-filling kill-or-confirm: per-channel mixed-precision vs uniform on key residuals.

Three arms on k_pre, matched bpe, scored on logit distortion vs real queries:
  - lowrank_rtn_channel @3b   (uniform baseline)
  - lowrank_waterfill_channel (reverse water-filling over per-channel residual variance)
  - outlier_two_tier          (top-k highest-variance residual channels -> fp16, rest low)

Diagnostic column resid_stable_rank distinguishes "low-rank already did the
water-filling" (flat residual spectrum) from "deterministic rounding killed it".

Usage
-----
    uv run python experiments/k2_waterfill.py \
        --cache-path results/cache/llama-3.1-8b_2048.safetensors \
        --model-label llama-3.1-8b \
        --model-name meta-llama/Llama-3.1-8B \
        --budget-bits 3.0 --rank 16
"""

from __future__ import annotations

import dataclasses
import math
import re

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.codecs import quantize_cache
from bmx.cache.collect import from_matrix, load_cache, to_matrix
from bmx.cache.metrics import logit_distortion, rel_fro
from bmx.cache.rope import apply_rope
from bmx.decomp.lrs import truncated_svd

_LAYER_RE = re.compile(r"^layer(\d+)\.(k|v|q|k_pre)$")


@dataclasses.dataclass
class Config:
    cache_path: str
    model_label: str = ""
    model_name: str = ""  # HF repo id for RoPE (empty => score in stored basis)
    budget_bits: float = 3.0
    group: int = 64
    rank: int = 16
    tiers: tuple[int, ...] = (0, 2, 3, 4)
    seed: int = 0
    out_root: str = ""  # override results root (tests pass tmp); empty => default


def _resid_stable_rank(R: torch.Tensor) -> float:
    """(sum of eigenvalues of R^T R) / (max eigenvalue) = (||R||_F^2) / (sigma_max^2)."""
    sv = torch.linalg.svdvals(R.double())
    s2 = sv**2
    return float((s2.sum() / s2[0].clamp_min(1e-30)).item())


def _outlier_two_tier(
    M: torch.Tensor, budget_bits: float, group: int, rank: int, svd_factors
) -> tuple[torch.Tensor, float]:
    """Top-k highest-variance residual channels -> fp16, rest -> low bits, matched bpe."""
    S, C = M.shape
    Us, V = svd_factors
    Us_s = Us.half().float() if M.dtype == torch.float32 else Us
    V_s = V.half().float() if M.dtype == torch.float32 else V
    L = Us_s @ V_s.mT
    R = M - L
    var = R.var(dim=0, unbiased=False)
    # Choose k fp16 channels + b_lo for the rest so the residual payload mean == budget.
    # payload = (k*16 + (C-k)*b_lo)/C = budget. Fix b_lo=2, solve k.
    b_lo = 2
    k = max(0, min(C, round((budget_bits - b_lo) * C / (16 - b_lo))))
    order = torch.argsort(var, descending=True)
    hi_cols = order[:k]
    lo_cols = order[k:]
    R_hat = torch.zeros_like(R)
    R_hat[:, hi_cols] = R[:, hi_cols]  # fp16 == passthrough at experiment precision
    if lo_cols.numel() > 0:
        from bmx.quant.rtn import rtn_quantize

        R_hat[:, lo_cols] = rtn_quantize(R[:, lo_cols].mT, b_lo, group).mT
    M_hat = L + R_hat
    payload = (k * 16.0 + (C - k) * b_lo) / C
    idx_term = math.ceil(math.log2(C)) / S  # store which k channels are fp16
    bpe = payload + 16.0 / group + 16.0 * rank * (S + C) / (S * C) + idx_term
    return M_hat, bpe


def main(cfg: Config) -> pd.DataFrame:
    run = (
        create_run("k2_waterfill", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k2_waterfill", cfg)
    )

    cache = load_cache(cfg.cache_path)
    layer_keys: dict[int, dict[str, torch.Tensor]] = {}
    for key, tensor in cache.items():
        m = _LAYER_RE.match(key)
        if m is None:
            continue
        layer_keys.setdefault(int(m.group(1)), {})[m.group(2)] = tensor

    # RoPE setup (optional).
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

    rows: list[dict] = []
    for layer_i in sorted(layer_keys):
        km = layer_keys[layer_i]
        if "k_pre" not in km or "q" not in km:
            continue
        k_pre = km["k_pre"]
        h_kv, S, d = k_pre.shape
        M = to_matrix(k_pre).float()  # (S, C)
        Q = km["q"].float()

        cos = sin = None
        K_post_true = None
        if rope_ready:
            cos, sin = get_cos_sin(S)
            K_post_true = apply_rope(k_pre.float(), cos, sin)

        factors = truncated_svd(M, cfg.rank)
        sr = _resid_stable_rank(M - (factors[0] @ factors[1].mT))

        def score(M_hat: torch.Tensor) -> tuple[float, float]:
            K_hat = from_matrix(M_hat, h_kv)
            rf = rel_fro(M_hat, M)
            if rope_ready:
                K_hat_rope = apply_rope(K_hat.float(), cos, sin)
                lg = logit_distortion(K_post_true, K_hat_rope, Q)
            else:
                lg = logit_distortion(k_pre.float(), K_hat, Q)
            return rf, lg

        # Uniform baseline @3b (budget rounded to int for the uniform arm).
        uni, bpe_uni = quantize_cache(
            "lowrank_rtn_channel",
            M,
            bits=round(cfg.budget_bits),
            group=cfg.group,
            rank=cfg.rank,
            svd_factors=factors,
        )
        wf, bpe_wf = quantize_cache(
            "lowrank_waterfill_channel",
            M,
            bits=cfg.budget_bits,
            group=cfg.group,
            rank=cfg.rank,
            tiers=cfg.tiers,
            svd_factors=factors,
        )
        ot, bpe_ot = _outlier_two_tier(M, cfg.budget_bits, cfg.group, cfg.rank, factors)

        assert abs(bpe_uni - bpe_wf) < 0.05, (
            f"L{layer_i}: waterfill bpe {bpe_wf:.3f} not matched to uniform {bpe_uni:.3f}"
        )

        for arm, (M_hat, bpe) in {
            "lowrank_rtn_channel": (uni, bpe_uni),
            "lowrank_waterfill_channel": (wf, bpe_wf),
            "outlier_two_tier": (ot, bpe_ot),
        }.items():
            rf, lg = score(M_hat)
            rows.append(
                dict(
                    model=cfg.model_label or "unknown",
                    layer=layer_i,
                    kind="k_pre",
                    arm=arm,
                    rank=cfg.rank,
                    bpe=bpe,
                    rel_fro=rf,
                    logit_rope=lg,
                    resid_stable_rank=sr,
                )
            )
            print(
                f"  L{layer_i:2d} {arm:26s} bpe={bpe:.3f} logit={lg:.4f} "
                f"rel_fro={rf:.4f} sr={sr:.1f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    write_metrics(run, df)

    print("\n" + "=" * 60)
    print("SUMMARY — mean logit_rope per arm (lower is better)")
    for arm in sorted(df.arm.unique()):
        sub = df[df.arm == arm]
        print(
            f"  {arm:26s} logit={sub.logit_rope.mean():.4f} "
            f"bpe={sub.bpe.mean():.3f}  resid_sr={sub.resid_stable_rank.mean():.1f}"
        )
    print(f"\n-> {run}")
    return df


if __name__ == "__main__":
    main(tyro.cli(Config))
