"""Collect KV/Q/K_pre cache tensors for a single model + sequence length.

Usage
-----
    uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024
    uv run python experiments/collect_cache.py --model-name meta-llama/Llama-3.1-8B --seq-len 2048

Output is written to results/cache/<model_short>_<seq_len>.safetensors (or --out).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
import tyro

from bmx.cache.collect import collect_cache, save_cache
from bmx.eval.layer_swap import load_eval_tokens


@dataclasses.dataclass
class Config:
    model_name: str = "gpt2"
    seq_len: int = 1024
    n_q_keep: int = 256
    out: str = ""  # override output path; empty => auto


def main(cfg: Config) -> None:
    # Determine output path
    model_short = cfg.model_name.split("/")[-1].lower()
    if cfg.out:
        out_path = Path(cfg.out)
    else:
        out_path = Path("results/cache") / f"{model_short}_{cfg.seq_len}.safetensors"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model (gpt2 is tiny — keep it fp32; everything else bf16)
    print(f"Loading model: {cfg.model_name}", flush=True)
    from transformers import AutoModelForCausalLM

    dtype = None if cfg.model_name == "gpt2" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype)
    model.eval()

    # Load tokens — load_eval_tokens returns a 1-D tensor; collect_cache wants (1, S)
    print(f"Loading {cfg.seq_len} eval tokens for {cfg.model_name}", flush=True)
    tokens = load_eval_tokens(cfg.model_name, n_tokens=cfg.seq_len)
    input_ids = tokens.unsqueeze(0)  # (1, S)

    # Collect
    print("Running collect_cache forward pass...", flush=True)
    cache = collect_cache(model, input_ids, n_q_keep=cfg.n_q_keep)

    # Save
    save_cache(cache, out_path)
    size_mb = out_path.stat().st_size / (1024**2)
    print(f"Saved: {out_path}  ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
