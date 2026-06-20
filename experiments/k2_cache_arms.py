"""K2 Bake-off: matched-bits codec sweep over real KV-cache activations.

For each layer × kind ∈ {k, k_pre, v} × arm × bits the script:
  - builds M (K1 convention: (h,S,d) → permute(1,0,2).reshape(S, h*d), fp32)
  - quantizes with quantize_cache
  - records: model, layer, kind, arm, bits, rank, bpe, rel_fro, logit, logit_rope
  - for kind=k and kind=k_pre → logit_distortion(K_orig, K_hat, Q)
  - for kind=v → attn_output_distortion(K, V, K, V_hat, Q) with true K
  - for kind=k_pre AND model_name nonempty → extra logit_rope column via apply_rope
  - CONTROL rows: random-sphere matrices per (S,C) shape, rel_fro only

Efficiency: SVD factors are cached per (layer, kind, rank) and reused across
the bits loop (SVD depends only on (M, rank), not bits).

Usage
-----
    uv run python experiments/k2_cache_arms.py \
        --cache-path results/cache/gpt2_1024.safetensors \
        --model-label gpt2

    uv run python experiments/k2_cache_arms.py \
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
from bmx.cache.codecs import CACHE_ARMS, quantize_cache
from bmx.cache.collect import from_matrix, load_cache, to_matrix
from bmx.cache.metrics import attn_output_distortion, logit_distortion, rel_fro
from bmx.cache.rope import apply_rope
from bmx.decomp.lrs import truncated_svd

_LAYER_RE = re.compile(r"^layer(\d+)\.(k|v|q|k_pre)$")

# Arms that accept a bits-only sweep (no rank param needed)
_BASE_ARMS = [a for a in CACHE_ARMS if a != "lowrank_rtn_channel"]
_LOWRANK_ARM = "lowrank_rtn_channel"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Config:
    cache_path: str
    model_label: str = ""
    model_name: str = ""  # HF repo id for RoPE (empty => skip rope-eval rows)
    bits: tuple[int, ...] = (2, 3, 4)
    group: int = 64
    ranks: tuple[int, ...] = (8, 16, 32)
    seed: int = 0
    n_random_controls: int = 2


# ---------------------------------------------------------------------------
# Random-sphere control matrices
# ---------------------------------------------------------------------------


def _random_control_matrix(S: int, C: int, ref_fro: float, seed: int) -> torch.Tensor:
    """fp32 matrix of i.i.d. N(0,1), unit-norm rows, then rescaled to match
    the Frobenius norm of the reference matrix."""
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(S, C, generator=g)
    row_norms = M.norm(dim=1, keepdim=True).clamp_min(1e-12)
    M = M / row_norms  # unit-norm rows
    # rescale so ||M||_F == ref_fro
    fro = M.norm()
    if fro > 1e-12:
        M = M * (ref_fro / fro)
    return M


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Config) -> None:
    run = create_run("k2_cache_arms", cfg)

    # -----------------------------------------------------------------------
    # Load cache
    # -----------------------------------------------------------------------
    cache = load_cache(cfg.cache_path)

    # Group by layer index
    layer_keys: dict[int, dict[str, torch.Tensor]] = {}
    for key, tensor in cache.items():
        m = _LAYER_RE.match(key)
        if m is None:
            continue
        layer_i = int(m.group(1))
        kind = m.group(2)
        layer_keys.setdefault(layer_i, {})[kind] = tensor

    # -----------------------------------------------------------------------
    # RoPE setup (optional)
    # -----------------------------------------------------------------------
    rope_ready = False
    cos_sin_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    if cfg.model_name:
        try:
            from transformers import AutoConfig

            from bmx.cache.rope import rope_cos_sin

            hf_config = AutoConfig.from_pretrained(cfg.model_name)
            rope_ready = True
            print(f"RoPE config loaded from {cfg.model_name}", flush=True)
        except ValueError as e:
            print(
                f"WARNING: could not load RoPE config for {cfg.model_name}: {e}",
                flush=True,
            )
            rope_ready = False

    def get_cos_sin(S: int):
        if S not in cos_sin_cache:
            cos_sin_cache[S] = rope_cos_sin(hf_config, S)
        return cos_sin_cache[S]

    # -----------------------------------------------------------------------
    # Once-per-run RoPE validation on layer 0
    # -----------------------------------------------------------------------
    rope_validated = False
    if (
        rope_ready
        and 0 in layer_keys
        and "k_pre" in layer_keys[0]
        and "k" in layer_keys[0]
    ):
        k_pre_t = layer_keys[0]["k_pre"].float()  # (h_kv, S, d)
        k_post_t = layer_keys[0]["k"].float()
        S0 = k_pre_t.shape[1]
        cos0, sin0 = get_cos_sin(S0)
        k_reconstructed = apply_rope(k_pre_t, cos0, sin0)
        rel_err = (k_reconstructed - k_post_t).norm() / k_post_t.norm().clamp_min(1e-12)
        rel_err_val = rel_err.item()
        print(
            f"[rope_validation] layer0 rel_fro(apply_rope(k_pre), k) = {rel_err_val:.4f} "
            f"(threshold 2e-2)",
            flush=True,
        )
        assert rel_err_val < 2e-2, (
            f"RoPE self-validation FAILED: rel={rel_err_val:.4f} >= 2e-2"
        )
        rope_validated = True

    # -----------------------------------------------------------------------
    # Track which (S, C) shapes have already had control rows generated
    # -----------------------------------------------------------------------
    shape_controls_done: set[tuple[int, int]] = set()

    # -----------------------------------------------------------------------
    # SVD factor cache: key = (layer, kind, rank) → (Us, V)
    # -----------------------------------------------------------------------
    svd_cache: dict[tuple[int, str, int], tuple[torch.Tensor, torch.Tensor]] = {}

    # -----------------------------------------------------------------------
    # Row accumulation
    # -----------------------------------------------------------------------
    rows: list[dict] = []
    model_label = cfg.model_label or "unknown"

    def get_svd(layer_i: int, kind: str, rank: int, M: torch.Tensor):
        key = (layer_i, kind, rank)
        if key not in svd_cache:
            svd_cache[key] = truncated_svd(M, rank)
        return svd_cache[key]

    def emit(row: dict) -> None:
        rows.append(row)
        # Per-row print
        logit_str = f"{row['logit']:.4f}" if not math.isnan(row["logit"]) else "  nan "
        rope_str = (
            f"  logit_rope={row['logit_rope']:.4f}"
            if not math.isnan(row["logit_rope"])
            else ""
        )
        rank_str = f"r={row['rank']}" if row["rank"] > 0 else "     "
        print(
            f"  layer={row['layer']:2d} kind={row['kind']:6s} arm={row['arm']:22s} "
            f"b={row['bits']} {rank_str} bpe={row['bpe']:.3f}  "
            f"rel_fro={row['rel_fro']:.4f}  logit={logit_str}{rope_str}",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # Main sweep: layer × kind × arm × bits
    # -----------------------------------------------------------------------
    for layer_i in sorted(layer_keys.keys()):
        kinds_map = layer_keys[layer_i]

        # We need k, v, q, k_pre for this layer
        if "k" not in kinds_map or "v" not in kinds_map or "q" not in kinds_map:
            continue
        if "k_pre" not in kinds_map:
            continue

        k_t = kinds_map["k"]  # (h_kv, S, d) fp16
        v_t = kinds_map["v"]  # (h_kv, S, d) fp16
        q_t = kinds_map["q"]  # (h,    T, d) fp16
        k_pre_t = kinds_map["k_pre"]  # (h_kv, S, d) fp16

        h_kv = k_t.shape[0]
        S = k_t.shape[1]
        d = k_t.shape[2]
        C = h_kv * d  # number of columns in the M matrix

        # For v metric: need true k as (h_kv, S, d)
        K_true = k_t.float()  # fp32 for metric calls
        V_true = v_t.float()
        Q_fp32 = q_t.float()

        # RoPE tables for this layer's sequence length (if available)
        cos_layer: torch.Tensor | None = None
        sin_layer: torch.Tensor | None = None
        if rope_ready:
            cos_layer, sin_layer = get_cos_sin(S)

        # For kind k_pre: true post-RoPE K for logit_rope metric
        K_post_true: torch.Tensor | None = None
        if rope_ready and rope_validated:
            K_post_true = apply_rope(k_pre_t.float(), cos_layer, sin_layer)

        print(f"\n[layer {layer_i}] (h_kv={h_kv}, S={S}, d={d}, C={C})", flush=True)

        for kind in ("k", "k_pre", "v"):
            if kind == "k":
                tensor = k_t
                M_orig = to_matrix(tensor)  # (S, C)
            elif kind == "k_pre":
                tensor = k_pre_t
                M_orig = to_matrix(tensor)
            else:  # v
                tensor = v_t
                M_orig = to_matrix(tensor)

            # ---- base arms -----------------------------------------------
            for arm in _BASE_ARMS:
                for bits in cfg.bits:
                    M_hat, bpe = quantize_cache(
                        arm, M_orig, bits=bits, seed=cfg.seed, group=cfg.group
                    )

                    # Reshape back
                    K_hat_t = from_matrix(M_hat, h_kv)  # (h_kv, S, d)

                    # Metrics
                    rf = rel_fro(M_hat, M_orig)

                    if kind == "k" or kind == "k_pre":
                        # logit distortion in the stored basis
                        logit = logit_distortion(tensor.float(), K_hat_t, Q_fp32)
                        # for k_pre with rope: cross-basis metric
                        logit_rope = float("nan")
                        if kind == "k_pre" and rope_ready and K_post_true is not None:
                            K_hat_rope = apply_rope(
                                K_hat_t.float(), cos_layer, sin_layer
                            )
                            logit_rope = logit_distortion(
                                K_post_true, K_hat_rope, Q_fp32
                            )
                    else:  # v
                        logit = attn_output_distortion(
                            K_true, V_true, K_true, K_hat_t.float(), Q_fp32
                        )
                        logit_rope = float("nan")

                    emit(
                        dict(
                            model=model_label,
                            layer=layer_i,
                            kind=kind,
                            arm=arm,
                            bits=bits,
                            rank=0,
                            bpe=bpe,
                            rel_fro=rf,
                            logit=logit,
                            logit_rope=logit_rope,
                        )
                    )

            # ---- lowrank arm (rank × bits) --------------------------------
            for rank in cfg.ranks:
                if rank > min(S, C):
                    continue
                # Pre-compute SVD once per (layer, kind, rank)
                svd_factors = get_svd(layer_i, kind, rank, M_orig)

                for bits in cfg.bits:
                    M_hat, bpe = quantize_cache(
                        _LOWRANK_ARM,
                        M_orig,
                        bits=bits,
                        seed=cfg.seed,
                        group=cfg.group,
                        rank=rank,
                        svd_factors=svd_factors,
                    )

                    K_hat_t = from_matrix(M_hat, h_kv)
                    rf = rel_fro(M_hat, M_orig)

                    if kind == "k" or kind == "k_pre":
                        logit = logit_distortion(tensor.float(), K_hat_t, Q_fp32)
                        logit_rope = float("nan")
                        if kind == "k_pre" and rope_ready and K_post_true is not None:
                            K_hat_rope = apply_rope(
                                K_hat_t.float(), cos_layer, sin_layer
                            )
                            logit_rope = logit_distortion(
                                K_post_true, K_hat_rope, Q_fp32
                            )
                    else:
                        logit = attn_output_distortion(
                            K_true, V_true, K_true, K_hat_t.float(), Q_fp32
                        )
                        logit_rope = float("nan")

                    emit(
                        dict(
                            model=model_label,
                            layer=layer_i,
                            kind=kind,
                            arm=_LOWRANK_ARM,
                            bits=bits,
                            rank=rank,
                            bpe=bpe,
                            rel_fro=rf,
                            logit=logit,
                            logit_rope=logit_rope,
                        )
                    )

            # ---- random-sphere controls (once per unique (S, C)) ----------
            if (S, C) not in shape_controls_done:
                shape_controls_done.add((S, C))
                ref_fro = M_orig.norm().item()
                for ctrl_idx in range(cfg.n_random_controls):
                    ctrl_seed = cfg.seed * 10000 + ctrl_idx
                    M_ctrl = _random_control_matrix(S, C, ref_fro, ctrl_seed)
                    # quantize controls with each base arm at each bits
                    for arm in _BASE_ARMS:
                        for bits in cfg.bits:
                            M_ctrl_hat, bpe = quantize_cache(
                                arm, M_ctrl, bits=bits, seed=cfg.seed, group=cfg.group
                            )
                            rf = rel_fro(M_ctrl_hat, M_ctrl)
                            emit(
                                dict(
                                    model=model_label,
                                    layer=-1,
                                    kind="random",
                                    arm=arm,
                                    bits=bits,
                                    rank=0,
                                    bpe=bpe,
                                    rel_fro=rf,
                                    logit=float("nan"),
                                    logit_rope=float("nan"),
                                )
                            )

    # -----------------------------------------------------------------------
    # Write parquet
    # -----------------------------------------------------------------------
    df = pd.DataFrame(rows)
    write_metrics(run, df)

    # -----------------------------------------------------------------------
    # End summary: per (kind, bits) — best arm by logit (or rel_fro if logit NaN)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY — best arm per (kind, bits)")
    print("=" * 72)

    real_df = df[df.layer >= 0].copy()

    for kind in sorted(real_df.kind.unique()):
        kind_df = real_df[real_df.kind == kind]
        print(f"\nkind={kind}")
        for bits in sorted(cfg.bits):
            sub = kind_df[kind_df.bits == bits]
            if sub.empty:
                continue
            # Use logit if available, otherwise rel_fro
            use_logit = not sub.logit.isna().all()
            metric_col = "logit" if use_logit else "rel_fro"
            best = sub.groupby(["arm", "rank"])[metric_col].mean()
            best_idx = best.idxmin()
            best_arm, best_rank = best_idx
            best_val = best[best_idx]
            # Get representative bpe for that arm
            arm_sub = sub[(sub.arm == best_arm) & (sub.rank == best_rank)]
            bpe_mean = arm_sub.bpe.mean()
            rank_str = f" rank={best_rank}" if best_rank > 0 else ""
            print(
                f"  bits={bits}: best={best_arm}{rank_str}  "
                f"{metric_col}={best_val:.4f}  bpe={bpe_mean:.3f}"
            )

    # Also print random control summary
    ctrl_df = df[df.layer == -1]
    if not ctrl_df.empty:
        print("\nkind=random (control, rel_fro only)")
        for bits in sorted(cfg.bits):
            sub = ctrl_df[ctrl_df.bits == bits]
            if sub.empty:
                continue
            best = sub.groupby("arm")["rel_fro"].mean()
            best_arm = best.idxmin()
            bpe_mean = sub[sub.arm == best_arm].bpe.mean()
            print(
                f"  bits={bits}: best={best_arm}  rel_fro={best[best_arm]:.4f}  "
                f"bpe={bpe_mean:.3f}"
            )

    print(f"\nTotal rows: {len(df)}")
    print(f"-> {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
