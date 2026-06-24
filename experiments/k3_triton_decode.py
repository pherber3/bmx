"""Drift-vs-speedup decode ledger: correctness-gated latency per variant.

Thin tyro CLI that drives run_decode_ledger for a set of variants and context
lengths, writes the parquet to results/k3_triton_decode/<run-id>/

Variants measured locally (CPU / CPU-only machine):
  - chunked_dequant: the PyTorch chunked online-softmax path (the reference
    correct path; the same one the gate derives tolerance from).

On a CUDA VM (Task 6) you would add:
  - triton_fused: the Triton kernel (Task 3/4).
This module is structured so that adding a Triton variant is a one-liner here.

Columns written (see triton_bench.LEDGER_COLUMNS for the schema contract):
  variant, seq_len, latency_ms, max_abs_vs_oracle, max_rel_vs_oracle,
  logit_parity_pass, predicted_speedup, measured_speedup
"""

from __future__ import annotations

import dataclasses

import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.chunked_attention import chunked_dequant_attention
from bmx.cache.codecs import quantize_packed
from bmx.cache.collect import to_matrix
from bmx.cache.triton_bench import run_decode_ledger


@dataclasses.dataclass
class Config:
    seq_lens: tuple[int, ...] = (32, 64, 128)
    n_warmup: int = 3
    n_repeat: int = 20
    # attn fixture parameters
    n_q_heads: int = 4
    n_q_groups: int = 2
    d: int = 16
    blk: int = 8
    n_blocks: int = 4
    arm: str = "rtn_token"
    group: int = 8
    seed: int = 42


def _make_packed_blocks(cfg: Config):
    """Build (q, k_blocks, v_blocks, attn_kwargs) from config — no test import."""
    h_kv = cfg.n_q_heads // cfg.n_q_groups
    torch.manual_seed(cfg.seed)
    q = torch.randn(cfg.n_q_heads, 1, cfg.d)
    k_blocks, v_blocks = [], []
    for i in range(cfg.n_blocks):
        start, end = i * cfg.blk, (i + 1) * cfg.blk
        kM = to_matrix(torch.randn(h_kv, cfg.blk, cfg.d))
        vM = to_matrix(torch.randn(h_kv, cfg.blk, cfg.d))
        kp, _ = quantize_packed(cfg.arm, kM, bits=4, group=cfg.group, seed=cfg.seed)
        vp, _ = quantize_packed(cfg.arm, vM, bits=4, group=cfg.group, seed=cfg.seed)
        k_blocks.append((kp, start, end))
        v_blocks.append((vp, start, end))
    attn_kwargs = dict(
        k_arm=cfg.arm,
        v_arm=cfg.arm,
        group=cfg.group,
        seed=cfg.seed,
        k_pre_rope=False,
        rope_cos=None,
        rope_sin=None,
        k_tail=None,
        v_tail=None,
        n_q_groups=cfg.n_q_groups,
        scale=cfg.d**-0.5,
        query_abs_start=None,
    )
    return q, k_blocks, v_blocks, attn_kwargs


def main(cfg: Config) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[k3_triton_decode] device={device}")

    q, k_blocks, v_blocks, attn_kwargs = _make_packed_blocks(cfg)

    if device == "cuda":
        q = q.cuda()
        k_blocks = [
            ({k: v.cuda() if hasattr(v, "cuda") else v for k, v in pb.items()}, s, e)
            for pb, s, e in k_blocks
        ]
        v_blocks = [
            ({k: v.cuda() if hasattr(v, "cuda") else v for k, v in pb.items()}, s, e)
            for pb, s, e in v_blocks
        ]

    # Variants dict: name -> callable(q, k_blocks, v_blocks, **attn_kwargs)
    # Add Triton variant here when available (Task 6).
    variants = {
        "chunked_dequant": lambda q, kb, vb, **kw: chunked_dequant_attention(
            q, kb, vb, **kw
        ),
    }

    df = run_decode_ledger(
        variants=variants,
        q=q,
        k_blocks=k_blocks,
        v_blocks=v_blocks,
        attn_kwargs=attn_kwargs,
        seq_lens=list(cfg.seq_lens),
        device=device,
        n_warmup=cfg.n_warmup,
        n_repeat=cfg.n_repeat,
    )

    run_dir = create_run("k3_triton_decode", cfg)
    write_metrics(run_dir, df, "decode_ledger")

    print()
    print(df.to_string(index=False))
    print(f"\nwrote {run_dir / 'decode_ledger.parquet'}")


if __name__ == "__main__":
    main(tyro.cli(Config))
