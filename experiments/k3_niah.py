"""Needle-in-a-haystack recall under KV compression.

Sweeps arms × document-lengths × depths through the StreamingQuantizedCache: a single needle
at a given depth, scored by ROUGE-1 recall, recording each arm's measured compression.

When `model` is None: loads the model, tokenizer, and Paul Graham haystack, plants the needle,
generates, and scores ROUGE-1. When `model` is injected (tests): a synthetic argmax proxy at
small lengths (≤64) — schema and mechanism only, no download.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.generate import avg_bpe, compression_for
from bmx.cache.hf_compat import resolve_vocab_size
from bmx.cache.niah import (
    build_niah_ids_synthetic,
    niah_recall_argmax,
)
from bmx.cache.recipes import spec_pair
from experiments._common import (
    assert_resume_identity,
    done_pairs,
    load_shards,
    pair_key,
    print_progress,
    write_samples_shard,
    write_shard,
)


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
    pack_path: str = ""
    """Path to a fitted spectral pack file; only needed for k4_* arms."""
    # Resume a crashed/killed run: path to its run_dir. Skips (arm, length) pairs whose
    # partial/<arm>__<length>.parquet shard already exists; identity-asserted against the
    # stored config.json + git SHA (mismatch is a hard error, not a warning) so a resumed
    # run can never mix rows measured under different code/config. Not itself part of the
    # measured configuration, so it is excluded from that identity check (see
    # experiments._common.RESUME_EXCLUDED_FIELDS).
    resume: str | None = None


def run(
    cfg: Config, model=None, root: str = "results", _stop_after_pairs: int | None = None
):
    tokenizer = None
    haystack = None
    if model is None:
        # Real run (VM): model + tokenizer + Paul Graham essays.
        from experiments._common import load_model_and_tokenizer

        from bmx.cache.generate import generate_through_cache
        from bmx.cache.niah import (
            NEEDLE_TEXT,
            build_niah_prompt,
            load_pg_corpus,
            rouge1_recall,
            rouge1_recall_only,
        )

        model, tokenizer = load_model_and_tokenizer(cfg.model_name, cfg.device)
        haystack = load_pg_corpus()

    if cfg.resume is not None:
        run_dir = Path(cfg.resume)
        assert_resume_identity(run_dir, cfg)
        skip_pairs = done_pairs(run_dir)
        print(
            f"[k3_niah] resuming {run_dir}: {len(skip_pairs)} pair(s) already done",
            flush=True,
        )
    else:
        run_dir = create_run("k3_niah", cfg, root=root)
        skip_pairs = set()

    n_pairs = len(cfg.arms) * len(cfg.lengths)
    pair_i = 0
    n_completed_this_run = 0
    start_time = time.monotonic()
    for arm in cfg.arms:
        k_spec, v_spec = spec_pair(
            arm, rank=cfg.rank, group=cfg.group, seed=cfg.seed, pack_path=cfg.pack_path
        )
        for length in cfg.lengths:
            pair_i += 1
            key = pair_key(arm, length)
            if key in skip_pairs:
                print(
                    f"[k3_niah pair {pair_i}/{n_pairs}] arm={arm} length={length} "
                    "SKIP (resumed)",
                    flush=True,
                )
                continue

            bpe_k, bpe_v, compression = compression_for(model, k_spec, v_spec, length)
            pair_rows = []
            samples_rows = []
            for depth_i, depth in enumerate(cfg.depths, start=1):
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
                pair_rows.append(
                    {
                        "arm": arm,
                        "length": length,
                        "depth": depth,
                        "recall": recall,
                        "recall_full": recall_full,
                        "recall_kind": recall_kind,
                        "bpe_k": bpe_k,
                        "bpe_v": bpe_v,
                        "kv_size_bits": avg_bpe(bpe_k, bpe_v),
                        "compression": compression,
                        "n_prefill": cfg.n_prefill,
                        "use_packed": cfg.use_packed,
                    }
                )
                # One generation per depth is the loop's natural per-sample grain.
                samples_rows.append(
                    {
                        "arm": arm,
                        "length": length,
                        "depth": depth,
                        "sample_idx": depth_i - 1,
                        "recall_full": recall_full,
                    }
                )
                if depth_i % 10 == 0 or depth_i == len(cfg.depths):
                    print_progress(
                        f"arm {arm} length {length} depth",
                        depth_i,
                        len(cfg.depths),
                        start_time,
                        recall=f"{recall:.4f}",
                    )

            # Per-generation recall_full (bootstrap-CI enabler); mirrors the aggregate
            # shard's per-depth rows. Written BEFORE the aggregate shard so the
            # aggregate's existence (the resume key) implies the samples shard exists.
            write_samples_shard(run_dir, samples_rows, arm, str(length))
            write_shard(run_dir, pair_rows, arm, str(length))
            print(
                f"[k3_niah pair {pair_i}/{n_pairs}] arm={arm} length={length} "
                "DONE (checkpointed)",
                flush=True,
            )
            n_completed_this_run += 1
            if (
                _stop_after_pairs is not None
                and n_completed_this_run >= _stop_after_pairs
            ):
                raise RuntimeError(
                    f"injected stop after {n_completed_this_run} pair(s) (test hook)"
                )

    write_metrics(run_dir, load_shards(run_dir))
    return run_dir


if __name__ == "__main__":
    run(tyro.cli(Config))
