"""K1 Census: break-even margins + distribution diagnostics on cached KV/Q activations.

Usage
-----
    uv run python experiments/k1_cache_census.py \
        --cache-path results/cache/gpt2_1024.safetensors \
        --model-label gpt2

For each layer and each tensor kind (k, k_pre, v, q) the script builds the
(tokens, h*d) fp32 activation matrix and records:
  - break-even margins via bmx.quant.breakeven (same instrument as weights)
  - channel_norm_ratio: max/median of per-channel L2 norms (rogue-channel inventory)
  - kurtosis_token / kurtosis_channel (before rotation)
  - rotation: "hadamard" if h*d is power-of-2 else "random_orthogonal"
  - kurtosis_token_rotated / kurtosis_channel_rotated (after rotation)
  - outlier_mass_mean / outlier_mass_max

Results written as parquet via create_run("k1_cache_census", cfg).
"""

from __future__ import annotations

import dataclasses
import re

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.collect import load_cache, to_matrix
from bmx.quant.breakeven import breakeven_row
from bmx.quant.hadamard import is_power_of_2, random_orthogonal, randomized_hadamard
from bmx.quant.stats import kurtosis, outlier_mass

_LAYER_RE = re.compile(r"^layer(\d+)\.(k|v|q|k_pre)$")


@dataclasses.dataclass
class Config:
    cache_path: str
    model_label: str = ""
    max_side_bpw: float = 6.0


def main(cfg: Config) -> None:
    run = create_run("k1_cache_census", cfg)

    # Load cache
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

    rows = []
    # data-oblivious rotations depend only on the channel count — build once
    rot_cache: dict[int, torch.Tensor] = {}
    for layer_i in sorted(layer_keys.keys()):
        kinds = layer_keys[layer_i]
        for kind in ("k", "k_pre", "v", "q"):
            if kind not in kinds:
                continue

            M = to_matrix(kinds[kind])  # (S, h*d) fp32
            S, channels = M.shape

            col_norms = M.norm(dim=0)
            # dim=-1: over channels => per-token statistic; dim=0: per-channel
            kurt_tok = kurtosis(M, dim=-1).mean().item()
            kurt_ch = kurtosis(M, dim=0).mean().item()

            if is_power_of_2(channels):
                rotation = "hadamard"
                M_rot = randomized_hadamard(M, seed=0)
            else:
                rotation = "random_orthogonal"
                if channels not in rot_cache:
                    rot_cache[channels] = random_orthogonal(
                        channels, seed=0, dtype=M.dtype
                    )
                M_rot = M @ rot_cache[channels].T

            om = outlier_mass(M)  # per-channel fraction (h*d,)
            row = {
                "model": cfg.model_label or "unknown",
                "layer": layer_i,
                "kind": kind,
                "S": S,
                "channels": channels,
                **breakeven_row(M, cfg.max_side_bpw),
                "channel_norm_ratio": (col_norms.max() / col_norms.median()).item(),
                "kurtosis_token": kurt_tok,
                "kurtosis_channel": kurt_ch,
                "rotation": rotation,
                "kurtosis_token_rotated": kurtosis(M_rot, dim=-1).mean().item(),
                "kurtosis_channel_rotated": kurtosis(M_rot, dim=0).mean().item(),
                "outlier_mass_mean": om.mean().item(),
                "outlier_mass_max": om.max().item(),
            }
            rows.append(row)

            print(
                f"layer{layer_i:2d}.{kind:5s} ({S}x{channels}): "
                f"lr_margin={row['lr_margin_bits']:+.3f}b  "
                f"sp_margin={row['sp_margin_bits']:+.3f}b  "
                f"ch_norm_ratio={row['channel_norm_ratio']:.2f}  "
                f"kurt_ch={kurt_ch:.2f}->{row['kurtosis_channel_rotated']:.2f}"
                f"[{rotation[:3]}]  "
                f"outlier_mass_max={row['outlier_mass_max']:.4f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    write_metrics(run, df)

    # Summary
    for kind in ("k", "k_pre", "v", "q"):
        sub = df[df.kind == kind]
        if sub.empty:
            continue
        print(
            f"\n{kind}: lr_margin [{sub.lr_margin_bits.min():+.3f}, "
            f"{sub.lr_margin_bits.max():+.3f}]  "
            f"ch_norm_ratio [{sub.channel_norm_ratio.min():.2f}, "
            f"{sub.channel_norm_ratio.max():.2f}]  "
            f"kurt_ch [{sub.kurtosis_channel.min():.2f}, "
            f"{sub.kurtosis_channel.max():.2f}] -> "
            f"[{sub.kurtosis_channel_rotated.min():.2f}, "
            f"{sub.kurtosis_channel_rotated.max():.2f}] rotated  "
            f"outlier_mass_max max={sub.outlier_mass_max.max():.4f}"
        )

    print(f"\n-> {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
