"""Drift-vs-speedup decode ledger: correctness-gated latency per variant.

Thin tyro CLI that drives run_decode_ledger across context lengths, writing the
parquet to results/k3_triton_decode/<run-id>/.

Variants measured:
  - chunked_dequant : the PyTorch chunked online-softmax path (the correct
    reference; the gate derives its tolerance from this path's oracle drift).
  - triton_fused    : the Triton decode kernel (triton_decode_attention), added
    only when CUDA is available. This is the variant under test.

Per the run_decode_ledger contract, the fixture is sized to ONE seq_len at a
time (n_blocks = seq_len // blk), and we call the ledger once per seq_len with
timing_seq_len set — so every row's measured columns correspond to a fixture of
that actual length (no silent mis-labelling). The per-block Python launch loop
in the current kernel means latency scales with n_blocks (= seq_len / blk): this
run is the BASELINE for the fused-kernel rewrite.

Columns (see triton_bench.LEDGER_COLUMNS):
  variant, seq_len, latency_ms, max_abs_vs_oracle, max_rel_vs_oracle,
  logit_parity_pass, predicted_speedup, measured_speedup
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.chunked_attention import chunked_dequant_attention
from bmx.cache.codecs import quantize_packed
from bmx.cache.collect import to_matrix
from bmx.cache.triton_bench import run_decode_ledger


@dataclasses.dataclass
class Config:
    # Context lengths to sweep. n_blocks per length = seq_len // blk.
    seq_lens: tuple[int, ...] = (512, 2048, 8192, 32768)
    n_warmup: int = 5
    n_repeat: int = 30
    # Realistic Llama-3.1-8B head geometry (so the numbers transfer).
    n_q_heads: int = 32
    n_q_groups: int = 4  # h_kv = 8
    d: int = 128  # head_dim
    blk: int = 64  # block size (KV codes flushed per block)
    arm: str = "rtn_token"
    group: int = 64
    seed: int = 42


def _make_blocks_for_seqlen(cfg: Config, seq_len: int, device: str):
    """Build (q, k_blocks, v_blocks, attn_kwargs) sized to seq_len.

    n_blocks = seq_len // blk full blocks (the resident packed KV at this context).
    """
    h_kv = cfg.n_q_heads // cfg.n_q_groups
    n_blocks = max(1, seq_len // cfg.blk)
    torch.manual_seed(cfg.seed)
    q = torch.randn(cfg.n_q_heads, 1, cfg.d)
    k_blocks, v_blocks = [], []
    for i in range(n_blocks):
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
    )
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
    return q, k_blocks, v_blocks, attn_kwargs


def main(cfg: Config) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[k3_triton_decode] device={device}  seq_lens={cfg.seq_lens}")

    variants = {
        "chunked_dequant": lambda q, kb, vb, **kw: chunked_dequant_attention(
            q, kb, vb, **kw
        ),
    }
    if device == "cuda":
        from bmx.cache.triton_dequant_attention import triton_decode_attention

        def _triton(q, kb, vb, **kw):
            return triton_decode_attention(q, kb, vb, **kw)

        variants["triton_fused"] = _triton

    # One ledger call per seq_len with a fixture sized to that length, so every
    # row's measured columns are genuine (timing_seq_len == seq_len).
    frames = []
    for sl in cfg.seq_lens:
        q, k_blocks, v_blocks, attn_kwargs = _make_blocks_for_seqlen(cfg, sl, device)
        df_sl = run_decode_ledger(
            variants=variants,
            q=q,
            k_blocks=k_blocks,
            v_blocks=v_blocks,
            attn_kwargs=attn_kwargs,
            seq_lens=[sl],
            timing_seq_len=sl,
            device=device,
            n_warmup=cfg.n_warmup,
            n_repeat=cfg.n_repeat,
            # The Triton kernel stores/carries KV in fp16 (the resident-fp16 point),
            # so its drift vs the fp32 oracle is ~3e-4 — legitimate, not a bug. Gate it
            # at the fp16-appropriate bar (the suite's 1e-2), not the fp32 chunked tol.
            variant_tol={"triton_fused": 1e-2},
        )
        frames.append(df_sl)
        print(f"  seq_len={sl}: {len(v_blocks)} blocks measured")

    df = pd.concat(frames, ignore_index=True)
    run_dir = create_run("k3_triton_decode", cfg)
    write_metrics(run_dir, df, "decode_ledger")

    print()
    print(df.to_string(index=False))
    print(f"\nwrote {run_dir / 'decode_ledger.parquet'}")


if __name__ == "__main__":
    main(tyro.cli(Config))
