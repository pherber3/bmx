"""K2d kill-or-confirm gate: lowrank_turboquant on REAL keys at matched bits.

The arm won its planted-low-rank synthetic gate (5.7x logit distortion vs
turboquant_mse@3b at fewer total bits) — planted data is its home turf, so
that proves nothing. This gate re-runs the comparison on real collected
caches (K1 convention) with the layer's real stored queries.

Arms (K side only; every bpe counts ALL metadata):
  - lowrank_turboquant  rank in cfg.ranks @ cfg.lrtq_bits residual, on k_pre
    (lowrank arms are pre-RoPE — the K2 record's convention)
  - turboquant_mse      @ cfg.tq_bits, on BOTH k (post-RoPE) and k_pre
    (K2 ran base arms on both kinds; elementwise codecs were basis-insensitive
    there — we report both to resolve the convention question honestly)
  - lowrank_rtn_channel rank=cfg.incumbent_rank @ cfg.incumbent_bits on k_pre
    (the k2b incumbent, reference)

Headline metric matches docs/2026-06-12-k2-arms-results.md: post-RoPE-basis
logit distortion vs real queries — `logit_rope` (apply_rope at read) for
k_pre arms, plain `logit` for kind=k. Stored-basis `logit` is also recorded.

Usage
-----
    uv run python experiments/k2d_lrtq_gate.py \
        --cache-path results/cache/llama-3.1-8b_2048.safetensors \
        --model-label llama-3.1-8b \
        --model-name meta-llama/Llama-3.1-8B

    uv run python experiments/k2d_lrtq_gate.py \
        --cache-path results/cache/gpt2_1024.safetensors \
        --model-label gpt2
"""

from __future__ import annotations

import dataclasses
import math

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.codecs import quantize_cache
from bmx.cache.collect import from_matrix, to_matrix
from bmx.cache.metrics import logit_distortion, rel_fro
from bmx.cache.rope import apply_rope
from bmx.decomp.lrs import truncated_svd
from experiments._k4_common import load_layer_keys, setup_rope


@dataclasses.dataclass
class Config:
    cache_path: str
    model_label: str = ""
    model_name: str = ""  # HF repo id for RoPE (empty => stored-basis logit only)
    ranks: tuple[int, ...] = (8, 16, 32)
    lrtq_bits: tuple[int, ...] = (2,)
    tq_bits: tuple[int, ...] = (2, 3)
    incumbent_rank: int = 16
    incumbent_bits: int = 3
    group: int = 64
    seed: int = 0
    max_layers: int = 0  # 0 = all layers


def main(cfg: Config) -> None:
    run = create_run("k2d_lrtq_gate", cfg)

    layer_keys = load_layer_keys(cfg.cache_path)

    layers = sorted(layer_keys.keys())
    if cfg.max_layers > 0:
        layers = layers[: cfg.max_layers]

    rope_ready, get_cos_sin = setup_rope(cfg.model_name, layer_keys, layers)

    rows: list[dict] = []
    model_label = cfg.model_label or "unknown"

    def emit(row: dict) -> None:
        rows.append(row)
        rope_str = (
            f"  logit_rope={row['logit_rope']:.4f}"
            if not math.isnan(row["logit_rope"])
            else ""
        )
        rank_str = f"r={row['rank']:2d}" if row["rank"] > 0 else "    "
        print(
            f"  layer={row['layer']:2d} kind={row['kind']:6s} "
            f"arm={row['arm']:20s} b={row['bits']} {rank_str} "
            f"bpe={row['bpe']:.3f}  rel_fro={row['rel_fro']:.4f}  "
            f"logit={row['logit']:.4f}{rope_str}",
            flush=True,
        )

    for layer_i in layers:
        kinds_map = layer_keys[layer_i]
        k_t = kinds_map["k"]  # (h_kv, S, d) fp16, post-RoPE
        k_pre_t = kinds_map["k_pre"]  # (h_kv, S, d) fp16, pre-RoPE
        q_t = kinds_map["q"]  # (h, T, d) fp16
        h_kv, S, d = k_t.shape
        C = h_kv * d
        Q_fp32 = q_t.float()

        cos_l = sin_l = None
        K_post_true = None
        if rope_ready:
            cos_l, sin_l = get_cos_sin(S)
            K_post_true = apply_rope(k_pre_t.float(), cos_l, sin_l)

        M_pre = to_matrix(k_pre_t)  # (S, C) fp32
        M_post = to_matrix(k_t)

        # SVD factors on k_pre, shared across bits per (layer, rank).
        svd_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        def get_svd(rank: int):
            if rank not in svd_cache:
                svd_cache[rank] = truncated_svd(M_pre, rank)
            return svd_cache[rank]

        print(f"\n[layer {layer_i}] (h_kv={h_kv}, S={S}, d={d}, C={C})", flush=True)

        # arm, kind, bits, rank
        jobs: list[tuple[str, str, int, int]] = []
        for rank in cfg.ranks:
            if rank > min(S, C):
                continue
            for b in cfg.lrtq_bits:
                jobs.append(("lowrank_turboquant", "k_pre", b, rank))
        for b in cfg.tq_bits:
            jobs.append(("turboquant_mse", "k", b, 0))
            jobs.append(("turboquant_mse", "k_pre", b, 0))
        jobs.append(
            ("lowrank_rtn_channel", "k_pre", cfg.incumbent_bits, cfg.incumbent_rank)
        )

        for arm, kind, bits, rank in jobs:
            M_orig = M_pre if kind == "k_pre" else M_post
            kwargs: dict = dict(bits=bits, seed=cfg.seed, group=cfg.group)
            if rank > 0:
                kwargs["rank"] = rank
                kwargs["svd_factors"] = get_svd(rank)
            M_hat, bpe = quantize_cache(arm, M_orig, **kwargs)
            K_hat_t = from_matrix(M_hat, h_kv)

            src_t = k_pre_t if kind == "k_pre" else k_t
            logit = logit_distortion(src_t.float(), K_hat_t, Q_fp32)
            logit_rope = float("nan")
            if kind == "k_pre" and K_post_true is not None:
                K_hat_rope = apply_rope(K_hat_t.float(), cos_l, sin_l)
                logit_rope = logit_distortion(K_post_true, K_hat_rope, Q_fp32)

            emit(
                dict(
                    model=model_label,
                    layer=layer_i,
                    kind=kind,
                    arm=arm,
                    bits=bits,
                    rank=rank,
                    bpe=bpe,
                    rel_fro=rel_fro(M_hat, M_orig),
                    logit=logit,
                    logit_rope=logit_rope,
                )
            )

    df = pd.DataFrame(rows)
    write_metrics(run, df)

    # ---- Summary: headline = post-RoPE-basis logit distortion ---------------
    df["headline"] = df.apply(
        lambda r: (
            r.logit_rope
            if (r.kind == "k_pre" and not math.isnan(r.logit_rope))
            else r.logit
        ),
        axis=1,
    )
    print("\n" + "=" * 88)
    print("SUMMARY — mean over layers (headline = post-RoPE-basis logit distortion)")
    print("=" * 88)
    g = df.groupby(["arm", "kind", "bits", "rank"])
    summary = g.agg(
        bpe=("bpe", "mean"),
        headline_mean=("headline", "mean"),
        headline_worst=("headline", "max"),
        logit_stored_mean=("logit", "mean"),
    ).reset_index()
    summary = summary.sort_values("bpe")
    for _, r in summary.iterrows():
        rank_str = f" r={int(r['rank']):2d}" if r["rank"] > 0 else "     "
        print(
            f"  {r['arm']:20s} kind={r['kind']:6s} b={int(r['bits'])}{rank_str}  "
            f"bpe={r['bpe']:.3f}  headline mean={r['headline_mean']:.4f} "
            f"worst-layer={r['headline_worst']:.4f}  "
            f"(stored-basis logit mean={r['logit_stored_mean']:.4f})",
            flush=True,
        )

    print(f"\nTotal rows: {len(df)}")
    print(f"-> {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
