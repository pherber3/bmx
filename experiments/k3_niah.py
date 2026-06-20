"""K3-NIAH — long-context needle-in-a-haystack recall under KV compression.

Sweeps arms × document-lengths × depths on ONE code path (StreamingQuantizedCache),
following the TurboQuant / Fu et al. setup: single needle, ROUGE-1 recall, length
sweep. Reports each arm's honest measured compression (never a pinned ratio).

Real path (model is None, VM run):
  Loads model + tokenizer + Paul Graham essays, builds RULER-style needle prompts,
  greedy-generates, scores ROUGE-1 (niah_recall_generate).

Offline-test path (model injected):
  Synthetic argmax proxy on small lengths (≤64, tiny_llama). recall is the argmax
  hit ×10 for schema parity. No download, no tokenizer.

Model-agnostic: the SOTA VM run is a --model-name change. Figures: plots/plot_k3_niah.py.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
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


def _compression_for(
    model_config, k_spec, v_spec, length: int
) -> tuple[float, float, float]:
    """Honest (bpe_k, bpe_v, compression) for an arm at a given length.

    Builds a StreamingQuantizedCache only to read its memory_report accounting; this
    matches the deployable blended-bpe number used by the ppl sweep.
    """
    cache = StreamingQuantizedCache(model_config, k_spec=k_spec, v_spec=v_spec)
    # bits_per_entry reads the codec spec deterministically; memory_report is exact at length.
    mem = cache.memory_report(seq_len=length)
    bpe_k, bpe_v = cache.bits_per_entry()
    return bpe_k, bpe_v, mem["compression"]


def run(cfg: Config, model=None, root: str = "results"):
    tokenizer = None
    haystack = None
    if model is None:
        # Real run (VM): model + tokenizer + Paul Graham essays.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from bmx.cache.haystack import pg_essays_dir, read_pg_corpus
        from bmx.cache.niah import build_niah_prompt, niah_recall_generate

        import torch

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch.float16
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        essays = pg_essays_dir()
        assert essays is not None, (
            "Paul Graham essays not found; clone Fu et al. repo at repo root"
        )
        haystack = read_pg_corpus(essays)

    run_dir = create_run("k3_niah", cfg, root=root)
    rows = []
    for arm in cfg.arms:
        k_spec, v_spec = _spec_pair(arm, cfg)
        for length in cfg.lengths:
            bpe_k, bpe_v, compression = _compression_for(
                model.config, k_spec, v_spec, length
            )
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
                rows.append(
                    {
                        "arm": arm,
                        "length": length,
                        "depth": depth,
                        "recall": recall,
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
