"""Needle-in-a-haystack recall under KV compression.

Sweeps arms × document-lengths × depths through the StreamingQuantizedCache: a single needle
at a given depth, scored by ROUGE-1 recall, recording each arm's measured compression.

When `model` is None: loads the model, tokenizer, and Paul Graham haystack, plants the needle,
generates, and scores ROUGE-1. When `model` is injected (tests): a synthetic argmax proxy at
small lengths (≤64) — schema and mechanism only, no download.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.niah import (
    build_niah_ids_synthetic,
    niah_recall_argmax,
)
from bmx.cache.streaming import StreamingQuantizedCache
from experiments.k3_live_generation import _spec_pair


@dataclasses.dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    arms: tuple[str, ...] = ("fp16", "k2b", "turboquant_mse", "turboquant_prod", "kivi")
    lengths: tuple[int, ...] = (4096, 8192, 16384, 32768)
    depths: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    n_prefill: int = 128
    rank: int = 16
    group: int = 64
    seed: int = 0
    answer_id: int = 7
    """Synthetic needle id for the offline argmax proxy (ignored on the real path)."""
    max_new_tokens: int = 50


def _compression_for(model, k_spec, v_spec, length: int) -> tuple[float, float, float]:
    """Measured (bpe_k, bpe_v, compression) for an arm at a given sequence length.

    Runs a calibration prefill of `length` tokens first: bits_per_entry is nan until a
    forward pass quantizes a block. Then reads the blended-bpe accounting off the cache.
    """
    cache = StreamingQuantizedCache(model.config, k_spec=k_spec, v_spec=v_spec)
    cache.attach(model)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, model.config.vocab_size, (1, length), generator=g)
    with cache:
        with torch.no_grad():
            model(ids, past_key_values=cache, use_cache=True)
    bpe_k, bpe_v = cache.bits_per_entry()
    mem = cache.memory_report(seq_len=length)
    return bpe_k, bpe_v, mem["compression"]


def run(cfg: Config, model=None, root: str = "results"):
    tokenizer = None
    haystack = None
    if model is None:
        # Real run (VM): model + tokenizer + Paul Graham essays.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from bmx.cache.haystack import load_pg_corpus
        from bmx.cache.niah import build_niah_prompt, niah_recall_generate

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch.float16
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        haystack = load_pg_corpus()

    run_dir = create_run("k3_niah", cfg, root=root)
    rows = []
    for arm in cfg.arms:
        k_spec, v_spec = _spec_pair(arm, cfg)
        for length in cfg.lengths:
            bpe_k, bpe_v, compression = _compression_for(model, k_spec, v_spec, length)
            for depth in cfg.depths:
                if tokenizer is None:
                    # Offline: synthetic argmax proxy at this (small) length.
                    ids = build_niah_ids_synthetic(
                        model.config.vocab_size,
                        length,
                        depth,
                        answer_id=cfg.answer_id,
                        seed=cfg.seed,
                    )
                    hit = niah_recall_argmax(
                        model,
                        ids,
                        query_pos=length - 1,
                        n_prefill=cfg.n_prefill,
                        k_spec=k_spec,
                        v_spec=v_spec,
                        answer_id=cfg.answer_id,
                    )
                    recall = 10.0 if hit else 0.0
                    recall_kind = "argmax_proxy"
                else:
                    # Real: ROUGE-1 generate recall.
                    prompt_ids = build_niah_prompt(
                        tokenizer,
                        context_length=length,
                        depth_percent=depth * 100.0,
                        haystack=haystack,
                    )
                    recall = niah_recall_generate(
                        model,
                        tokenizer,
                        prompt_ids,
                        cfg.n_prefill,
                        k_spec,
                        v_spec,
                        max_new_tokens=cfg.max_new_tokens,
                    )
                    recall_kind = "rouge1"
                rows.append(
                    {
                        "arm": arm,
                        "length": length,
                        "depth": depth,
                        "recall": recall,
                        "recall_kind": recall_kind,
                        "bpe_k": bpe_k,
                        "bpe_v": bpe_v,
                        "compression": compression,
                        "n_prefill": cfg.n_prefill,
                    }
                )

    write_metrics(run_dir, pd.DataFrame(rows))
    return run_dir


if __name__ == "__main__":
    run(tyro.cli(Config))
