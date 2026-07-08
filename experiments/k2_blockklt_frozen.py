"""K2 frozen-KLT drift gate: does a full CxC KLT + waterfill allocation frozen at
prefill stay good for later tokens?

Follow-up to experiments/k2_blockklt.py (per-head risk gate): the @32k deployment
story there rests on a full CxC eigenbasis whose storage amortizes over context —
which only works if basis AND allocation can be FIT ON EARLY CONTEXT and reused.
Local proxy on the 2048-token cache:

  - oracle      : KLT basis + waterfill allocation fit on all 2048 tokens
  - frozen_512  : basis + allocation fit on the first 512 tokens only (prefill
                  proxy), applied to all 2048
  - frozen_1024 : same with a 1024-token fit (trend point)
  - uniform     : no-fit baseline (frozen == oracle trivially), for reference

All arms quantize all 2048 tokens; logit distortion is scored REGION-MATCHED on
the streamed tail only (key rows 512: and, separately, rows 1024:), vs the real
stored queries, RoPE-at-read. Drift cost = log4(D_frozen / D_oracle) on the same
region. Allocation drift = # channels whose tier changes when the allocation is
re-derived from tail stats in the SAME frozen basis.

NOTE: a 512-row fit gives XtX rank <= 512 < C=1024 — the bottom half of the
frozen eigenbasis is null-space (arbitrary) and its allocation sees ~zero
variance there. That is not an artifact; it IS the deployment risk under test.

Usage
-----
    uv run python experiments/k2_blockklt_frozen.py \
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
    allocate_channel_bits,
    factor_bits,
    quantize_cache,
    scale_bits,
    tier_bits,
)
from bmx.cache.collect import from_matrix, load_cache, to_matrix
from bmx.cache.metrics import logit_distortion, rel_fro
from bmx.cache.rope import apply_rope
from bmx.decomp.lrs import truncated_svd
from bmx.quant.rtn import rtn_quantize

_LAYER_RE = re.compile(r"^layer(\d+)\.(k|v|q|k_pre)$")


@dataclasses.dataclass
class Config:
    cache_path: str
    model_label: str = ""
    model_name: str = ""  # HF repo id for RoPE (empty => score in stored basis)
    budgets: tuple[float, ...] = (2.0, 2.5, 3.0)
    fit_lens: tuple[int, ...] = (512, 1024)  # frozen-fit prefixes (oracle = full S)
    group: int = 64
    rank: int = 16
    tiers: tuple[int, ...] = (0, 2, 3, 4)
    layer_stride: int = 1
    out_root: str = ""


def _stored_lowrank(
    M: torch.Tensor, factors: tuple
) -> tuple[torch.Tensor, torch.Tensor]:
    """L from fp16-stored factors (codecs convention) and the residual R = M - L."""
    Us, V = factors
    L = Us.half().float() @ V.half().float().mT
    return L, M - L


def _quantize_by_bits(
    R: torch.Tensor, bits_pc: torch.Tensor, group: int
) -> torch.Tensor:
    """Groupwise-RTN each channel at its assigned bit width (0 = drop)."""
    R_hat = torch.zeros_like(R)
    for b in sorted(set(int(x) for x in bits_pc.tolist())):
        if b == 0:
            continue
        cols = (bits_pc == b).nonzero(as_tuple=True)[0]
        R_hat[:, cols] = rtn_quantize(R[:, cols].mT, b, group).mT
    return R_hat


def _eig_arm_fit_on_prefix(
    M: torch.Tensor,
    budget: float,
    group: int,
    rank: int,
    tiers: tuple[int, ...],
    factors: tuple,
    fit_len: int,
) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """Full CxC KLT + waterfill with basis AND allocation fit on M[:fit_len] only,
    applied to all rows. fit_len == S is the oracle. Returns
    (M_hat, bpe_idealized, bits_frozen, Q_frozen)."""
    S, C = M.shape
    L, R = _stored_lowrank(M, factors)
    Q = _klt_basis(R[:fit_len])
    bits_pc = allocate_channel_bits(R[:fit_len] @ Q, budget, tiers=tiers, axis=0)
    R_hat = _quantize_by_bits(R @ Q, bits_pc, group) @ Q.mT
    payload = float(bits_pc.float().mean().item())
    bpe = payload + scale_bits(group) + factor_bits(rank, S, C) + tier_bits(tiers, S)
    return L + R_hat, bpe, bits_pc, Q


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
    regions = sorted(cfg.fit_lens)  # score tails starting at each fit boundary
    rows: list[dict] = []
    for layer_i in layers:
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
        _, R_resid = _stored_lowrank(M, factors)

        def score_tail(M_hat: torch.Tensor, lo: int) -> tuple[float, float]:
            """rel_fro + logit distortion restricted to key rows lo: (streamed tail)."""
            K_hat = from_matrix(M_hat, h_kv)
            rf = rel_fro(M_hat[lo:], M[lo:])
            if rope_ready:
                K_hat_rope = apply_rope(K_hat.float(), cos, sin)
                lg = logit_distortion(K_post_true[:, lo:], K_hat_rope[:, lo:], Q)
            else:
                lg = logit_distortion(k_pre.float()[:, lo:], K_hat[:, lo:], Q)
            return rf, lg

        for budget in cfg.budgets:
            # uniform reference (no fit => no drift by construction)
            uni, bpe_uni = (
                quantize_cache(
                    "lowrank_rtn_channel",
                    M,
                    bits=round(budget) if float(budget).is_integer() else 0,
                    group=cfg.group,
                    rank=cfg.rank,
                    svd_factors=factors,
                )
                if float(budget).is_integer()
                else (None, float("nan"))
            )

            arms: list[tuple[str, torch.Tensor, float, int, torch.Tensor | None]] = []
            if uni is not None:
                arms.append(("uniform", uni, bpe_uni, 0, None))
            orc, bpe_orc, bits_orc, _ = _eig_arm_fit_on_prefix(
                M, budget, cfg.group, cfg.rank, cfg.tiers, factors, S
            )
            arms.append(("eig_oracle", orc, bpe_orc, S, bits_orc))
            frozen_bits: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            for P in cfg.fit_lens:
                fz, bpe_fz, bits_fz, Q_fz = _eig_arm_fit_on_prefix(
                    M, budget, cfg.group, cfg.rank, cfg.tiers, factors, P
                )
                arms.append((f"eig_frozen_{P}", fz, bpe_fz, P, bits_fz))
                frozen_bits[P] = (bits_fz, Q_fz)

            for arm, M_hat, bpe, fit_len, bits_pc in arms:
                row = dict(
                    model=cfg.model_label or "unknown",
                    layer=layer_i,
                    kind="k_pre",
                    arm=arm,
                    budget=budget,
                    fit_len=fit_len,
                    rank=cfg.rank,
                    bpe=bpe,
                    alloc_tiers_changed=float("nan"),
                    alloc_abs_bit_delta=float("nan"),
                )
                for lo in regions:
                    rf, lg = score_tail(M_hat, lo)
                    row[f"rel_fro_tail{lo}"] = rf
                    row[f"logit_tail{lo}"] = lg
                # allocation drift: re-derive allocation from the tail (rows P:)
                # in the SAME frozen basis; count tier changes.
                if arm.startswith("eig_frozen_"):
                    P = fit_len
                    bits_fz, Q_fz = frozen_bits[P]
                    bits_tail = allocate_channel_bits(
                        R_resid[P:] @ Q_fz, budget, tiers=cfg.tiers, axis=0
                    )
                    row["alloc_tiers_changed"] = int((bits_tail != bits_fz).sum())
                    row["alloc_abs_bit_delta"] = float(
                        (bits_tail - bits_fz).abs().float().mean().item()
                    )
                rows.append(row)
                print(
                    f"  L{layer_i:2d} b={budget:.1f} {arm:14s} bpe={bpe:.3f} "
                    + " ".join(
                        f"logit@{lo}:={row[f'logit_tail{lo}']:.4f}" for lo in regions
                    )
                    + (
                        f" tiers_chg={row['alloc_tiers_changed']:.0f}"
                        if row["alloc_tiers_changed"] == row["alloc_tiers_changed"]
                        else ""
                    ),
                    flush=True,
                )

    df = pd.DataFrame(rows)
    write_metrics(run, df)

    log4 = math.log(4.0)
    print("\n" + "=" * 78)
    print("SUMMARY — frozen/oracle logit ratio on the region each fit streams into")
    for budget in cfg.budgets:
        sub = df[df.budget == budget]
        print(f"\n-- budget {budget:.1f} --")
        for P in cfg.fit_lens:
            lo = P  # region-matched: score where this fit actually streams
            orc = sub[sub.arm == "eig_oracle"].set_index("layer")[f"logit_tail{lo}"]
            fz_rows = sub[sub.arm == f"eig_frozen_{P}"].set_index("layer")
            fz = fz_rows[f"logit_tail{lo}"]
            ratio = float((fz / orc).mean())
            worst_layer = int((fz / orc).idxmax())
            worst = float((fz / orc).max())
            drift_bits = math.log(ratio) / log4
            uni_col = sub[sub.arm == "uniform"]
            uni_str = ""
            if len(uni_col):
                u = float(uni_col[f"logit_tail{lo}"].mean())
                uni_str = f" | uniform@tail={u:.4f}"
            print(
                f"  fit@{P:４d} tail@{lo}: oracle={orc.mean():.4f} "
                f"frozen={fz.mean():.4f} frozen/oracle={ratio:.3f} "
                f"(drift {drift_bits:+.3f} eff bits; worst L{worst_layer} {worst:.2f}x)"
                f" tiers_chg={fz_rows.alloc_tiers_changed.mean():.1f}/1024"
                f" |dbits|={fz_rows.alloc_abs_bit_delta.mean():.3f}" + uni_str
            )
        # also the fixed last-1536 region for BOTH fits (coordinator's main table)
        lo = min(cfg.fit_lens)
        orc = sub[sub.arm == "eig_oracle"].set_index("layer")[f"logit_tail{lo}"]
        for P in cfg.fit_lens:
            fz = sub[sub.arm == f"eig_frozen_{P}"].set_index("layer")[f"logit_tail{lo}"]
            ratio = float((fz / orc).mean())
            print(
                f"  [fixed tail@{lo}] fit@{P}: frozen/oracle={ratio:.3f} "
                f"({math.log(ratio) / log4:+.3f} eff bits)"
            )
    print(f"\n-> {run}")
    return df


if __name__ == "__main__":
    main(tyro.cli(Config))
