"""K2c: Frozen-subspace generalization — streaming viability test (E5).

Tests the load-bearing assumption of the streaming key codec: a channel
subspace fitted on prefill keys generalises to keys written later during
generation.

Metrics per layer × kind × rank
--------------------------------
* eps_frozen / eps_oracle / eps_ratio   — energy captured by frozen / oracle V
* rel_frozen / rel_oracle / rel_offline — end-to-end codec relative Frobenius
* logit_frozen / logit_oracle / logit_offline — attention logit distortion

Drift curve
-----------
V fitted on block 0 of (n_blocks+1) equal token blocks; eps_frozen recorded
on each of blocks 1..n_blocks.  Answers whether capture decays with distance.

Usage
-----
    uv run python experiments/k2c_subspace_drift.py \\
        --cache-path results/cache/gpt2_1024.safetensors \\
        --model-label gpt2

    uv run python experiments/k2c_subspace_drift.py \\
        --cache-path results/cache/llama-3.1-8b_2048.safetensors \\
        --model-label llama-3.1-8b
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
from bmx.decomp.lrs import truncated_svd

_LAYER_RE = re.compile(r"^layer(\d+)\.(k|v|q|k_pre)$")

NaN = float("nan")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Config:
    cache_path: str
    model_label: str = ""
    ranks: tuple[int, ...] = (8, 16, 32)
    bits: int = 3
    group: int = 64
    fit_frac: float = 0.5
    n_blocks: int = 4
    kinds: tuple[str, ...] = ("k_pre", "k")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def eps_captured(X: torch.Tensor, V: torch.Tensor) -> float:
    """||X @ V @ V.T||_F^2 / ||X||_F^2 via cheaper ((X @ V)**2).sum() / (X**2).sum()."""
    denom = (X**2).sum().item()
    if denom < 1e-30:
        return NaN
    return ((X @ V) ** 2).sum().item() / denom


def codec_arm(
    TEST: torch.Tensor,
    V: torch.Tensor,
    bits: int,
    group: int,
) -> torch.Tensor:
    """Frozen or oracle codec arm: project via V, quantize residual.

    V is (C, r), fp16-roundtripped for honest stored precision.
    Returns M_hat.
    """
    V_stored = V.half().float()  # honest fp16 roundtrip
    L = TEST @ (V_stored @ V_stored.mT)  # (S_test, C)
    R = TEST - L
    R_hat, _ = quantize_cache("rtn_channel", R, bits=bits, group=group)
    M_hat = L + R_hat
    return M_hat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Config) -> None:
    run = create_run("k2c_subspace_drift", cfg)
    model_label = cfg.model_label or "unknown"

    cache = load_cache(cfg.cache_path)

    # Group by layer
    layer_keys: dict[int, dict[str, torch.Tensor]] = {}
    for key, tensor in cache.items():
        m = _LAYER_RE.match(key)
        if m is None:
            continue
        layer_i = int(m.group(1))
        kind = m.group(2)
        layer_keys.setdefault(layer_i, {})[kind] = tensor

    rows: list[dict] = []

    def emit(row: dict) -> None:
        rows.append(row)
        seg = row["segment"]
        eps_r = row["eps_ratio"]
        eps_r_str = f"{eps_r:.4f}" if not math.isnan(eps_r) else "  nan "
        lf = row["logit_frozen"]
        lo = row["logit_offline"]
        lf_str = f"{lf:.4f}" if not math.isnan(lf) else "  nan "
        lo_str = f"{lo:.4f}" if not math.isnan(lo) else "  nan "
        print(
            f"  layer={row['layer']:2d} kind={row['kind']:6s} rank={row['rank']:2d} "
            f"seg={seg:8s} "
            f"eps_frozen={row['eps_frozen']:.4f} "
            f"eps_oracle={row['eps_oracle']:.4f} "
            f"eps_ratio={eps_r_str} "
            f"logit_frozen={lf_str} logit_offline={lo_str}",
            flush=True,
        )

    for layer_i in sorted(layer_keys.keys()):
        kinds_map = layer_keys[layer_i]

        # Need q for logit metrics
        if "q" not in kinds_map:
            continue

        q_t = kinds_map["q"]  # (h, T, d) fp16 — last 256 positions' queries
        Q_fp32 = q_t.float()  # (h, T, d)

        print(f"\n[layer {layer_i}]", flush=True)

        for kind in cfg.kinds:
            if kind not in kinds_map:
                continue

            tensor = kinds_map[kind]  # (h_kv, S, d) fp16
            h_kv = tensor.shape[0]
            S_full = tensor.shape[1]

            M_full = to_matrix(tensor)  # (S_full, C)
            C = M_full.shape[1]

            # Split: FIT / TEST
            S_fit = int(cfg.fit_frac * S_full)
            FIT = M_full[:S_fit]  # (S_fit, C)
            TEST = M_full[S_fit:]  # (S_test, C)
            S_test = TEST.shape[0]

            # Ensure TEST rows are divisible by group (for rtn_channel on residual).
            # Trim to the largest multiple of group that fits in S_test.
            if S_test % cfg.group != 0:
                S_test_use = (S_test // cfg.group) * cfg.group
                TEST = TEST[:S_test_use]
                S_test = S_test_use

            if S_test == 0:
                print(
                    f"  layer{layer_i}.{kind}: S_test={S_test} too small, skipping",
                    flush=True,
                )
                continue

            # SVDs once per segment at the max valid rank, then sliced inside
            # the rank loop — the top-r of a rank-R SVD IS the rank-r SVD.
            ranks_valid = [r for r in cfg.ranks if r <= min(S_fit, C, S_test)]
            if not ranks_valid:
                continue
            r_max = max(ranks_valid)
            _, V_fit_full = truncated_svd(FIT, r_max)  # (C, r_max)
            Us_test_full, V_test_full = truncated_svd(TEST, r_max)

            # Drift-curve subspaces: fit on block 0 of (n_blocks+1) equal
            # blocks of the full seq; per-block oracles for the upper bound.
            n_total_blocks = cfg.n_blocks + 1
            block_size = S_full // n_total_blocks
            drift_ranks = [r for r in ranks_valid if r <= block_size]
            V_drift_full: torch.Tensor | None = None
            block_oracle_V: dict[int, torch.Tensor] = {}
            if drift_ranks:
                r_dmax = max(drift_ranks)
                BLOCK0 = M_full[:block_size]
                _, V_drift_full = truncated_svd(BLOCK0, r_dmax)
                for b in range(1, cfg.n_blocks + 1):
                    BLOCK = M_full[b * block_size : (b + 1) * block_size]
                    if BLOCK.shape[0] == 0:
                        continue
                    block_oracle_V[b] = truncated_svd(BLOCK, r_dmax)[1]

            for rank in ranks_valid:
                # Subspaces (sliced from the shared SVDs)
                V_fit = V_fit_full[:, :rank]  # V_fit: (C, rank)
                V_oracle = V_test_full[:, :rank]  # V_oracle: (C, rank)

                # --- Generalization metrics on TEST --------------------------
                eps_frz = eps_captured(TEST, V_fit)
                eps_orc = eps_captured(TEST, V_oracle)
                eps_ratio = (eps_frz / eps_orc) if eps_orc > 1e-30 else NaN

                # Codec arms: frozen, oracle, offline
                M_hat_frz = codec_arm(TEST, V_fit, cfg.bits, cfg.group)
                M_hat_orc = codec_arm(TEST, V_oracle, cfg.bits, cfg.group)
                M_hat_off, _ = quantize_cache(
                    "lowrank_rtn_channel",
                    TEST,
                    bits=cfg.bits,
                    group=cfg.group,
                    rank=rank,
                    svd_factors=(Us_test_full[:, :rank], V_test_full[:, :rank]),
                )

                rel_frz = rel_fro(M_hat_frz, TEST)
                rel_orc = rel_fro(M_hat_orc, TEST)
                rel_off = rel_fro(M_hat_off, TEST)

                # Logit distortion — need (h_kv, S_test, d) tensors
                K_test_true = from_matrix(TEST, h_kv)
                K_hat_frz_t = from_matrix(M_hat_frz, h_kv)
                K_hat_orc_t = from_matrix(M_hat_orc, h_kv)
                K_hat_off_t = from_matrix(M_hat_off, h_kv)

                logit_frz = logit_distortion(K_test_true, K_hat_frz_t, Q_fp32)
                logit_orc = logit_distortion(K_test_true, K_hat_orc_t, Q_fp32)
                logit_off = logit_distortion(K_test_true, K_hat_off_t, Q_fp32)

                row = dict(
                    model=model_label,
                    layer=layer_i,
                    kind=kind,
                    rank=rank,
                    segment="test",
                    S_fit=S_fit,
                    S_test=S_test,
                    eps_frozen=eps_frz,
                    eps_oracle=eps_orc,
                    eps_ratio=eps_ratio,
                    rel_frozen=rel_frz,
                    rel_oracle=rel_orc,
                    rel_offline=rel_off,
                    logit_frozen=logit_frz,
                    logit_oracle=logit_orc,
                    logit_offline=logit_off,
                    block_idx=NaN,
                    block_token_mid=NaN,
                )
                emit(row)

                # --- Drift curve ---------------------------------------------
                # Fit on block 0 of (n_blocks+1) equal blocks of the full seq
                if block_size < rank:
                    # Degenerate; skip drift
                    continue

                V_drift = V_drift_full[:, :rank]

                for b in range(1, cfg.n_blocks + 1):
                    bstart = b * block_size
                    bend = (b + 1) * block_size
                    BLOCK = M_full[bstart:bend]
                    if BLOCK.shape[0] == 0:
                        continue

                    eps_b = eps_captured(BLOCK, V_drift)
                    # oracle eps for this block (upper bound)
                    eps_b_orc = eps_captured(BLOCK, block_oracle_V[b][:, :rank])
                    eps_b_ratio = (eps_b / eps_b_orc) if eps_b_orc > 1e-30 else NaN
                    token_mid = bstart + (bend - bstart) // 2

                    drift_row = dict(
                        model=model_label,
                        layer=layer_i,
                        kind=kind,
                        rank=rank,
                        segment=f"drift_b{b}",
                        S_fit=block_size,
                        S_test=BLOCK.shape[0],
                        eps_frozen=eps_b,
                        eps_oracle=eps_b_orc,
                        eps_ratio=eps_b_ratio,
                        rel_frozen=NaN,
                        rel_oracle=NaN,
                        rel_offline=NaN,
                        logit_frozen=NaN,
                        logit_oracle=NaN,
                        logit_offline=NaN,
                        block_idx=b,
                        block_token_mid=float(token_mid),
                    )
                    emit(drift_row)

    df = pd.DataFrame(rows)
    write_metrics(run, df)

    # -----------------------------------------------------------------------
    # End summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY — generalization (segment='test' rows)")
    print("=" * 72)

    test_df = df[df.segment == "test"].copy()

    for kind in cfg.kinds:
        kdf = test_df[test_df.kind == kind]
        if kdf.empty:
            continue
        print(f"\nkind={kind}")
        for rank in sorted(cfg.ranks):
            rdf = kdf[kdf["rank"] == rank]
            if rdf.empty:
                continue
            mean_ratio = rdf.eps_ratio.mean()
            min_ratio = rdf.eps_ratio.min()
            mean_lf = rdf.logit_frozen.mean()
            mean_lo = rdf.logit_offline.mean()
            logit_gap_pct = (
                (mean_lf - mean_lo) / mean_lo * 100 if mean_lo > 1e-12 else NaN
            )
            print(
                f"  rank={rank:2d}: eps_ratio mean={mean_ratio:.4f} min={min_ratio:.4f}"
                f"  logit_frozen={mean_lf:.4f} logit_offline={mean_lo:.4f}"
                f"  gap={logit_gap_pct:+.1f}%"
            )

    # Drift curve summary
    drift_df = df[df.segment.str.startswith("drift")].copy()
    if not drift_df.empty:
        print("\n" + "=" * 72)
        print("DRIFT CURVE — eps_ratio by block (fit=block0, rank=32)")
        print("=" * 72)
        r32 = drift_df[drift_df["rank"] == max(cfg.ranks)]
        if not r32.empty:
            for kind in cfg.kinds:
                kdf = r32[r32.kind == kind]
                if kdf.empty:
                    continue
                print(f"\nkind={kind}")
                for b in sorted(kdf.block_idx.dropna().unique()):
                    bdf = kdf[kdf.block_idx == b]
                    mean_eps = bdf.eps_frozen.mean()
                    mean_ratio = bdf.eps_ratio.mean()
                    mid = bdf.block_token_mid.mean()
                    print(
                        f"  block={int(b)} (token_mid~{int(mid):4d}): "
                        f"eps_frozen={mean_eps:.4f}  eps_ratio={mean_ratio:.4f}"
                    )

    print(f"\nTotal rows: {len(df)}")
    print(f"-> {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
