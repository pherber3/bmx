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
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.generate import compression_for
from bmx.cache.hf_compat import resolve_vocab_size
from bmx.cache.niah import (
    build_niah_ids_synthetic,
    niah_recall_argmax,
)
from bmx.cache.recipes import spec_pair


@dataclasses.dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    device: str = "cpu"  # "cuda" on the VM
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
    use_packed: bool = False
    """Run generation through PackedStreamingCache (packed codes resident +
    chunked dequant-attention at decode) instead of StreamingQuantizedCache.
    Token-identical output (parity-gated); lower resident memory — the path that
    unblocks the batched 128k sweep. Real path only (ignored offline)."""


def run(cfg: Config, model=None, root: str = "results"):
    tokenizer = None
    haystack = None
    if model is None:
        # Real run (VM): model + tokenizer + Paul Graham essays.
        from experiments._common import load_model_and_tokenizer

        from bmx.cache.generate import generate_through_cache
        from bmx.cache.haystack import load_pg_corpus
        from bmx.cache.niah import (
            NEEDLE_TEXT,
            build_niah_prompt,
            rouge1_recall,
            rouge1_recall_only,
        )

        model, tokenizer = load_model_and_tokenizer(cfg.model_name, cfg.device)
        haystack = load_pg_corpus()

    run_dir = create_run("k3_niah", cfg, root=root)
    rows = []
    for arm in cfg.arms:
        k_spec, v_spec = spec_pair(arm, rank=cfg.rank, group=cfg.group, seed=cfg.seed)
        for length in cfg.lengths:
            bpe_k, bpe_v, compression = compression_for(model, k_spec, v_spec, length)
            for depth in cfg.depths:
                if tokenizer is None:
                    # Offline: synthetic argmax proxy at this (small) length.
                    ids = build_niah_ids_synthetic(
                        resolve_vocab_size(model.config),
                        length,
                        depth,
                        answer_id=cfg.answer_id,
                        seed=cfg.seed,
                    ).to(model.device)
                    hit = niah_recall_argmax(
                        model,
                        ids,
                        query_pos=length - 1,
                        n_prefill=cfg.n_prefill,
                        k_spec=k_spec,
                        v_spec=v_spec,
                        answer_id=cfg.answer_id,
                    )
                    recall = recall_full = 10.0 if hit else 0.0
                    recall_kind = "argmax_proxy"
                else:
                    # Real: generate once, score both F-measure (paper-faithful) and recall
                    # (precision-free; survives instruct-model verbosity).
                    prompt_ids = build_niah_prompt(
                        tokenizer,
                        context_length=length,
                        depth_percent=depth * 100.0,
                        haystack=haystack,
                    ).to(cfg.device)
                    response = generate_through_cache(
                        model,
                        tokenizer,
                        prompt_ids,
                        cfg.n_prefill,
                        k_spec,
                        v_spec,
                        max_new_tokens=cfg.max_new_tokens,
                        use_packed=cfg.use_packed,
                    )
                    recall = rouge1_recall(NEEDLE_TEXT, response)
                    recall_full = rouge1_recall_only(NEEDLE_TEXT, response)
                    recall_kind = "rouge1"
                rows.append(
                    {
                        "arm": arm,
                        "length": length,
                        "depth": depth,
                        "recall": recall,
                        "recall_full": recall_full,
                        "recall_kind": recall_kind,
                        "bpe_k": bpe_k,
                        "bpe_v": bpe_v,
                        "compression": compression,
                        "n_prefill": cfg.n_prefill,
                        "use_packed": cfg.use_packed,
                    }
                )

    write_metrics(run_dir, pd.DataFrame(rows))
    return run_dir


if __name__ == "__main__":
    run(tyro.cli(Config))
