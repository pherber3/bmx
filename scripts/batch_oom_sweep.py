"""Batch-concurrency OOM sweep: how many co-resident long-context sequences fit?

The C3 evidence experiment (authoritative-results doc §5b / F4): single-sequence
generation peak shows NO win for k2b (~87 vs ~86 GiB @128k) because weights +
transients dominate one sequence's peak. The KV saving is a RESIDENT-STORAGE
property, so the honest demonstration is concurrency: with N sequences' caches
co-resident (a serving workload), KV becomes the binding constraint and ~4x
KV compression should carry ~4x more sequences before OOM.

Method (residency, not scheduling): for each (mode, ctx) cell, walk N up a grid;
each trial prefills N independent caches (same prompt, batch=1 each) and KEEPS
ALL N RESIDENT, then decodes a few greedy tokens through each cache while the
other N-1 stay resident. All N caches alive at decode time is exactly the
serving-memory question; round-robin interleaving would change nothing about
peak residency. First OOM ends the walk; the cell's result is the last N that
succeeded, plus measured peak and the per-sequence marginal resident GiB.

Modes:
  dense      — DynamicCache, fp16 KV resident (the baseline).
  packed     — PackedStreamingCache with the k2b_ph arm (packed codes resident;
               the deployment path; decode hits fused_decode_attention_k2b).
  streaming  — StreamingQuantizedCache k2b_ph (dequantized-resident reference;
               optional, shows packed's saving vs its own reference).

!!! THIS SCRIPT OOMS THE GPU BY DESIGN. It must NEVER run on a GPU shared with
other jobs. It refuses to start without --confirm-run.

Usage (GH200, idle, user-launched only):
  uv run python scripts/batch_oom_sweep.py --model-name meta-llama/Llama-3.1-8B-Instruct \
      --ctx-lens 32768 65536 --confirm-run
"""

from __future__ import annotations

import dataclasses
import gc
import json
import sys
import time
from pathlib import Path

import torch
import tyro

from bmx.cache.packed_streaming import PackedStreamingCache
from bmx.cache.recipes import spec_pair
from bmx.cache.streaming import StreamingQuantizedCache


@dataclasses.dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    arm: str = "k2b_ph"  # deployment arm — the one the fused kernel serves
    ctx_lens: tuple[int, ...] = (32768, 65536)
    modes: tuple[str, ...] = ("dense", "packed")  # add "streaming" for the reference
    # N grid, walked in order until first OOM. Coarse on purpose: each trial
    # prefills N x ctx tokens, so densifying the grid multiplies hours.
    n_grid: tuple[int, ...] = (1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64)
    decode_tokens: int = 8  # per sequence, with all N caches resident
    rank: int = 16
    group: int = 64
    seed: int = 0
    out: str = "results/batch_oom_sweep.json"
    confirm_run: bool = False  # hard gate: this script OOMs the GPU by design
    # W5-2: bit-packed V indices in the packed cache's stacks (4 codes/byte).
    # Measures the second rung of the memory ladder (~3.6 -> ~1.7 GiB/seq @32k).
    pack_v: bool = False
    # W5-3a: nibble-packed K residual (2 signed codes/byte) — the third memory rung
    # (~2.26 -> ~1.73 GiB/seq @32k predicted).
    pack_k: bool = False


def _prompt_ids(tokenizer, ctx_len: int, device) -> torch.Tensor:
    torch.manual_seed(1234)
    vocab = int(tokenizer.vocab_size)
    return torch.randint(low=10, high=vocab - 10, size=(1, ctx_len), device=device)


def _make_cache(mode: str, model, cfg: Config):
    if mode == "dense":
        from transformers import DynamicCache

        return DynamicCache()
    k_spec, v_spec = spec_pair(cfg.arm, rank=cfg.rank, group=cfg.group, seed=cfg.seed)
    if mode == "streaming":
        return StreamingQuantizedCache(model.config, k_spec=k_spec, v_spec=v_spec)
    return PackedStreamingCache(
        model.config, k_spec=k_spec, v_spec=v_spec, pack_v=cfg.pack_v, pack_k=cfg.pack_k
    )


class _Wired:
    """attach/detach bracket (no-op for cache types without them), so N caches
    can share one model — each holds its own state; only the wiring swaps."""

    def __init__(self, cache, model):
        self.cache, self.model = cache, model

    def __enter__(self):
        attach = getattr(self.cache, "attach", None)
        if attach is not None:
            attach(self.model)
        return self.cache

    def __exit__(self, *exc):
        detach = getattr(self.cache, "detach", None)
        if detach is not None:
            detach()
        return False


def _trial(model, tokenizer, cfg: Config, mode: str, ctx: int, n_seqs: int) -> dict:
    """One residency trial. Raises torch.cuda.OutOfMemoryError on failure."""
    ids_ = _prompt_ids(tokenizer, ctx, model.device)
    torch.cuda.reset_peak_memory_stats()
    caches: list = []
    next_tok: list = []  # each cache's first greedy token, kept from its prefill
    alloc_marks: list[int] = []
    t0 = time.perf_counter()

    # Phase 1: prefill all N caches, keeping every one resident. Each cache also
    # decodes ONE token immediately after its prefill: on the packed path that
    # first decode builds the stacks and (W5-1) re-points/frees the block-list
    # duplicates, so alloc_marks record STEADY-STATE per-sequence residency — the
    # number a serving system actually pays — not the transient prefill layout.
    # (The first sweep, 2026-07-05, prefilled all N before any decode; its OOM
    # boundary was set by the pre-stacking transient and W5-1 could never help.)
    for _ in range(n_seqs):
        cache = _make_cache(mode, model, cfg)
        with _Wired(cache, model), torch.no_grad():
            out = model(ids_, past_key_values=cache, use_cache=True)
            tok = out.logits[:, -1:].argmax(-1)
            out = model(tok, past_key_values=cache, use_cache=True)
            next_tok.append(out.logits[:, -1:].argmax(-1))
        del out, tok
        caches.append(cache)
        # Settle before marking: the W5-1 re-point drops the last references to the
        # block-list duplicates, but if any sit in reference cycles they linger until
        # a GC pass — without this, marks measure garbage-not-yet-collected, not
        # residency (probe read 2.79 GiB/seq while un-gc'd marks read 4.5).
        gc.collect()
        torch.cuda.empty_cache()
        alloc_marks.append(torch.cuda.memory_allocated())

    # Phase 2: decode through each cache while ALL N stay resident.
    tokens0: list[int] | None = None
    for i, cache in enumerate(caches):
        tok = next_tok[i]
        got = [int(tok)]
        with _Wired(cache, model), torch.no_grad():
            for _ in range(cfg.decode_tokens - 1):
                out = model(tok, past_key_values=cache, use_cache=True)
                tok = out.logits[:, -1:].argmax(-1)
                got.append(int(tok))
        if tokens0 is None:
            tokens0 = got
        else:
            # Same prompt + greedy => every sequence must emit identical tokens.
            assert got == tokens0, (
                f"consistency fail: cache {i} tokens {got} != cache 0 {tokens0}"
            )

    peak = torch.cuda.max_memory_allocated() / 2**30
    # Marginal resident cost per extra sequence (allocated delta across prefills).
    marginals = [
        (alloc_marks[i] - alloc_marks[i - 1]) / 2**30
        for i in range(1, len(alloc_marks))
    ]
    return {
        "mode": mode,
        "ctx": ctx,
        "n_seqs": n_seqs,
        "peak_gib": round(peak, 2),
        "marginal_gib_per_seq": round(sum(marginals) / len(marginals), 3)
        if marginals
        else None,
        "trial_s": round(time.perf_counter() - t0, 1),
    }


def main(cfg: Config) -> None:
    if not cfg.confirm_run:
        print(
            "REFUSING TO RUN: this sweep OOMs the GPU by design and must only run "
            "on an idle GPU, launched deliberately. Re-run with --confirm-run."
        )
        sys.exit(2)
    assert torch.cuda.is_available(), "GPU-only experiment"
    from experiments._common import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(cfg.model_name, "cuda")

    results: list[dict] = []
    for ctx in cfg.ctx_lens:
        for mode in cfg.modes:
            best_n = 0
            for n in cfg.n_grid:
                try:
                    row = _trial(model, tokenizer, cfg, mode, ctx, n)
                    best_n = n
                    results.append(row)
                    print(f"[ok ] {row}", flush=True)
                except torch.cuda.OutOfMemoryError:
                    results.append({"mode": mode, "ctx": ctx, "n_seqs": n, "oom": True})
                    print(f"[oom] mode={mode} ctx={ctx} n_seqs={n}", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()
            print(
                f"[cell] mode={mode} ctx={ctx}: max co-resident sequences = {best_n}",
                flush=True,
            )

    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
