"""K2 water-filling kill-or-confirm: per-channel mixed-precision vs uniform on key residuals.

Five arms on k_pre, matched bpe, scored on logit distortion (headline) and rel_fro (MSE):
  - lowrank_rtn_channel @3b        (uniform baseline)
  - lowrank_waterfill_channel      (reverse water-filling over per-channel residual variance)
  - lowrank_eigwaterfill_channel   (KLT rotation then water-fill — revival candidate)
  - lowrank_randwaterfill_channel  (random rotation control; same bpe, different alignment)
  - outlier_two_tier               (top-k highest-variance residual channels -> fp16, rest low)

Diagnostic columns:
  resid_stable_rank     — distinguishes flat residual spectrum from rogue-channel case
  query_eigen_alignment — fraction of real-query energy in funded KLT eigendirections (KLT+rand)
  bpe_honest            — KLT bpe charged with rotation-matrix cost (KLT only, NaN otherwise)

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
    topk_ks: tuple[int, ...] = (32, 64, 128)
    prefill_fit_len: int = 512


def _resid_stable_rank(R: torch.Tensor) -> float:
    """(sum of eigenvalues of R^T R) / (max eigenvalue) = (||R||_F^2) / (sigma_max^2)."""
    sv = torch.linalg.svdvals(R.double())
    s2 = sv**2
    return float((s2.sum() / s2[0].clamp_min(1e-30)).item())


def _residual_eigengap(R: torch.Tensor) -> float:
    """Max adjacent eigenvalue ratio in the top of the residual channel spectrum.

    A value near 1.0 across the spectrum => no eigengap => frozen rotation likely
    drifts (Davis-Kahan). Returns the largest λ_k/λ_{k+1} among the top-64 to expose
    any elbow; ~1.0 means smooth decay.
    """
    ev = torch.linalg.eigvalsh(R.mT @ R).flip(0).clamp_min(1e-30)
    top = ev[: min(64, len(ev) - 1)]
    ratios = top[:-1] / top[1:]
    return float(ratios.max().item())


def _query_eigen_alignment(
    R: torch.Tensor, q_t: torch.Tensor, h_kv: int, k: int
) -> float:
    """Fraction of real-query energy in the top-k funded KLT eigen-directions of R.

    R: (S, C) residual (C = h_kv*d). q_t: (h_q, T, d) queries (h_q = query heads,
    a GQA multiple of h_kv). k: number of funded eigencols.

    Faithful to the metric's channel layout: the KLT eigenvectors live in the full
    (S, C) channel space, so each query head is GQA-mapped to its kv-head's d-dim
    block of C and projected there. Returns the mean over query rows of the energy
    fraction ||proj_topk||^2 / ||q||^2, averaged over query heads.

    Diagnostic only (directional, not a metric input): if alignment is high yet the
    KLT arm still loses logit, the funded high-variance directions are read by queries
    but quantizing them still costs more than uniform — a clean Outcome-3 signature.
    """
    if k <= 0:
        return 0.0
    C = R.shape[1]
    d = C // h_kv
    _, eigvecs = torch.linalg.eigh(R.mT @ R)
    Qk = eigvecs.flip(dims=(1,))[:, :k]  # (C, k) top-k eigen-directions
    h_q, T, _ = q_t.shape
    group = h_q // h_kv  # GQA: how many query heads share each kv head
    fracs = []
    for j in range(h_q):
        kv = j // group  # this query head's kv-head index
        # embed this head's d-dim query into the full C-dim channel space at its block
        qf = torch.zeros(T, C, dtype=R.dtype)
        qf[:, kv * d : (kv + 1) * d] = q_t[j].to(R.dtype)
        proj = qf @ Qk  # (T, k)
        num = (proj**2).sum(dim=1)
        den = (qf**2).sum(dim=1).clamp_min(1e-30)
        fracs.append((num / den).mean())
    return float(torch.stack(fracs).mean().item())


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

        R_resid = M - (factors[0] @ factors[1].mT)

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
        eig, bpe_eig = quantize_cache(
            "lowrank_eigwaterfill_channel",
            M,
            bits=cfg.budget_bits,
            group=cfg.group,
            rank=cfg.rank,
            tiers=cfg.tiers,
            svd_factors=factors,
        )
        rnd, bpe_rnd = quantize_cache(
            "lowrank_randwaterfill_channel",
            M,
            bits=cfg.budget_bits,
            seed=cfg.seed,
            group=cfg.group,
            rank=cfg.rank,
            tiers=cfg.tiers,
            svd_factors=factors,
        )
        ot, bpe_ot = _outlier_two_tier(M, cfg.budget_bits, cfg.group, cfg.rank, factors)

        for nm, b in (("eig", bpe_eig), ("rand", bpe_rnd)):
            assert abs(bpe_uni - b) < 0.05, (
                f"L{layer_i}: {nm} bpe {b:.3f} vs uniform {bpe_uni:.3f}"
            )
        assert abs(bpe_uni - bpe_wf) < 0.05, (
            f"L{layer_i}: waterfill bpe {bpe_wf:.3f} not matched to uniform {bpe_uni:.3f}"
        )

        # alignment for the KLT arm: k = funded eigencols of the KLT-rotated residual
        from bmx.cache.codecs import allocate_channel_bits

        _, eigvecs_a = torch.linalg.eigh(R_resid.mT @ R_resid)
        Q_klt = eigvecs_a.flip(dims=(1,))
        bits_rot = allocate_channel_bits(
            R_resid @ Q_klt, cfg.budget_bits, tiers=cfg.tiers, axis=0
        )
        k_funded = int((bits_rot > 0).sum().item())
        align = _query_eigen_alignment(R_resid, km["q"].float(), h_kv, k_funded)

        # honest KLT pass: only meaningful if eig beats uniform on logit at this layer
        _, lg_uni = score(uni)
        _, lg_eig = score(eig)
        bpe_eig_honest = float("nan")
        if lg_eig < lg_uni:
            _, bpe_eig_honest = quantize_cache(
                "lowrank_eigwaterfill_channel",
                M,
                bits=cfg.budget_bits,
                group=cfg.group,
                rank=cfg.rank,
                tiers=cfg.tiers,
                charge_rotation=True,
                svd_factors=factors,
            )

        eigengap = _residual_eigengap(R_resid)

        arm_rows = {
            "lowrank_rtn_channel": (uni, bpe_uni, float("nan")),
            "lowrank_waterfill_channel": (wf, bpe_wf, float("nan")),
            "lowrank_eigwaterfill_channel": (eig, bpe_eig, bpe_eig_honest),
            "lowrank_randwaterfill_channel": (rnd, bpe_rnd, float("nan")),
            "outlier_two_tier": (ot, bpe_ot, float("nan")),
        }
        for arm, (M_hat, bpe, bpe_honest) in arm_rows.items():
            rf, lg = score(M_hat)
            align_col = (
                align
                if arm
                in ("lowrank_eigwaterfill_channel", "lowrank_randwaterfill_channel")
                else float("nan")
            )
            rows.append(
                dict(
                    model=cfg.model_label or "unknown",
                    layer=layer_i,
                    kind="k_pre",
                    arm=arm,
                    rank=cfg.rank,
                    bpe=bpe,
                    bpe_honest=bpe_honest,
                    rel_fro=rf,
                    logit_rope=lg,
                    resid_stable_rank=sr,
                    query_eigen_alignment=align_col,
                    resid_eigengap=float("nan"),
                    frozen_oracle_ratio=float("nan"),
                )
            )
            print(
                f"  L{layer_i:2d} {arm:30s} bpe={bpe:.3f} logit={lg:.4f} "
                f"rel_fro={rf:.4f} align={align_col if align_col == align_col else float('nan'):.3f}",
                flush=True,
            )

        # block-diagonal per-head KLT (honest cost folded in directly via charge_rotation)
        bd, bpe_bd = quantize_cache(
            "lowrank_blockdiagwaterfill_channel",
            M,
            bits=cfg.budget_bits,
            group=cfg.group,
            rank=cfg.rank,
            tiers=cfg.tiers,
            h_kv=h_kv,
            charge_rotation=True,
            svd_factors=factors,
        )
        # frozen-prefill full KLT (honest) + oracle control (uncharged)
        fz, bpe_fz = quantize_cache(
            "lowrank_frozenwaterfill_channel",
            M,
            bits=cfg.budget_bits,
            group=cfg.group,
            rank=cfg.rank,
            tiers=cfg.tiers,
            prefill_fit_len=cfg.prefill_fit_len,
            charge_rotation=True,
            svd_factors=factors,
        )
        orc, bpe_orc = quantize_cache(
            "lowrank_oraclewaterfill_channel",
            M,
            bits=cfg.budget_bits,
            group=cfg.group,
            rank=cfg.rank,
            tiers=cfg.tiers,
            svd_factors=factors,
        )
        _, lg_fz = score(fz)
        _, lg_orc = score(orc)
        frozen_oracle_ratio = lg_fz / lg_orc if lg_orc > 0 else float("nan")

        structured = [
            ("lowrank_blockdiagwaterfill_channel", bd, bpe_bd, bpe_bd, float("nan")),
            (
                "lowrank_frozenwaterfill_channel",
                fz,
                bpe_fz,
                bpe_fz,
                frozen_oracle_ratio,
            ),
            (
                "lowrank_oraclewaterfill_channel",
                orc,
                bpe_orc,
                float("nan"),
                float("nan"),
            ),
        ]
        # topk k-sweep, honest cost per k
        C_full = M.shape[1]
        for kk in cfg.topk_ks:
            if kk > C_full:  # guard k <= C
                continue
            tk, bpe_tk = quantize_cache(
                "lowrank_topkwaterfill_channel",
                M,
                bits=cfg.budget_bits,
                group=cfg.group,
                rank=cfg.rank,
                tiers=cfg.tiers,
                topk_k=kk,
                charge_rotation=True,
                svd_factors=factors,
            )
            structured.append(
                (
                    f"lowrank_topkwaterfill_channel_k{kk}",
                    tk,
                    bpe_tk,
                    bpe_tk,
                    float("nan"),
                )
            )

        for arm, M_hat, bpe, bpe_h, fo_ratio in structured:
            rf, lg = score(M_hat)
            rows.append(
                dict(
                    model=cfg.model_label or "unknown",
                    layer=layer_i,
                    kind="k_pre",
                    arm=arm,
                    rank=cfg.rank,
                    bpe=bpe,
                    bpe_honest=bpe_h,
                    rel_fro=rf,
                    logit_rope=lg,
                    resid_stable_rank=sr,
                    query_eigen_alignment=float("nan"),
                    resid_eigengap=eigengap,
                    frozen_oracle_ratio=fo_ratio,
                )
            )
            bpe_h_str = f"{bpe_h:.3f}" if bpe_h == bpe_h else "nan"
            fo_str = f"{fo_ratio:.3f}" if fo_ratio == fo_ratio else "nan"
            print(
                f"  L{layer_i:2d} {arm:34s} bpe={bpe:.3f} bpe_h={bpe_h_str} "
                f"logit={lg:.4f} rel_fro={rf:.4f} gap={eigengap:.3f} fo={fo_str}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    write_metrics(run, df)

    print("\n" + "=" * 70)
    print(
        "SUMMARY — mean logit_rope / rel_fro per arm (lower better); align = query-eigen"
    )
    uni_by_layer = df[df.arm == "lowrank_rtn_channel"].set_index("layer")["logit_rope"]
    # classic arms
    classic_arms = [
        "lowrank_rtn_channel",
        "lowrank_waterfill_channel",
        "lowrank_eigwaterfill_channel",
        "lowrank_randwaterfill_channel",
        "outlier_two_tier",
    ]
    for arm in sorted(df.arm.unique()):
        if arm not in classic_arms:
            continue
        sub = df[df.arm == arm]
        merged = sub.set_index("layer")["logit_rope"]
        wins = int((merged < uni_by_layer.reindex(merged.index)).sum())
        align_mean = sub["query_eigen_alignment"].mean()
        print(
            f"  {arm:34s} logit={sub.logit_rope.mean():.4f} rel_fro={sub.rel_fro.mean():.4f} "
            f"bpe={sub.bpe.mean():.3f} align={align_mean:.3f} beats_uniform={wins}/{sub.layer.nunique()}"
        )
    # structured arms
    structured_arm_names = [a for a in sorted(df.arm.unique()) if a not in classic_arms]
    if structured_arm_names:
        print("\n--- structured rotation arms (HONEST bpe; deployable verdict) ---")
        for arm in structured_arm_names:
            sub = df[df.arm == arm]
            merged = sub.set_index("layer")["logit_rope"]
            wins = int((merged < uni_by_layer.reindex(merged.index)).sum())
            bpe_h_mean = sub["bpe_honest"].mean()
            gap_mean = sub["resid_eigengap"].mean()
            fo_mean = sub["frozen_oracle_ratio"].mean()
            bpe_h_str = f"{bpe_h_mean:.3f}" if bpe_h_mean == bpe_h_mean else "nan"
            gap_str = f"{gap_mean:.3f}" if gap_mean == gap_mean else "nan"
            fo_str = f"{fo_mean:.3f}" if fo_mean == fo_mean else "nan"
            print(
                f"  {arm:34s} logit={sub.logit_rope.mean():.4f} rel_fro={sub.rel_fro.mean():.4f} "
                f"bpe_h={bpe_h_str} gap={gap_str} fo={fo_str} beats_uniform={wins}/{sub.layer.nunique()}"
            )
    print(f"\n-> {run}")
    return df


if __name__ == "__main__":
    main(tyro.cli(Config))
