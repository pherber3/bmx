"""KV-cache memory census: resident / peak / incremental per arm per length.

Measures both cache paths (StreamingQuantizedCache, PackedStreamingCache) and
compares against the analytic byte-ledger. CUDA-authoritative (VM); falls back to
a tiny CPU smoke run locally. Writes parquet to results/k3_kernel_census/<run-id>.

Columns: `resident_after_prefill` is the LOAD-BEARING number — the total allocated
peak after prefill (weights + KV + activations), what the analytic ledger's
`predicted_peak` corresponds to and what determines whether a path clears the
ceiling. `peak_decode` is the peak during a few decode steps measured after a
SEPARATE `reset_peak_memory_stats()`; it is typically LOWER than the prefill peak
(decode is one token), so `peak_decode_incremental = peak_decode - resident` is
usually negative — it is NOT "how much decode added on top," just the gap between
the two independently-reset peaks. Compare the ledger to `resident_after_prefill`.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.packed_streaming import PackedStreamingCache
from bmx.cache.recipes import spec_pair
from bmx.cache.streaming import StreamingQuantizedCache, resolve_vocab_size


@dataclasses.dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    seq_lens: tuple[int, ...] = (4096, 16384, 32768)
    arms: tuple[str, ...] = ("fp16", "k2b")
    max_new_tokens: int = 4


def _measure(model, input_ids, cache, max_new_tokens: int = 4):
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True)
    if cuda:
        torch.cuda.synchronize()
    resident = torch.cuda.max_memory_allocated() if cuda else 0
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    # Manual decode loop (NOT model.generate). The cache already holds all S prefill
    # tokens from the forward above; generate() would compute new_tokens = given -
    # cached and, given one token already cached, process ZERO tokens -> a 0-length
    # reshape crash. We instead feed genuinely-new single tokens at positions S, S+1,
    # ..., passing cache_position so each appends correctly. This measures a real
    # decode step's peak without generate()'s pre-filled-cache bookkeeping.
    S = input_ids.shape[1]
    next_tok = input_ids[:, -1:]  # arbitrary token id; memory, not text, is measured
    with torch.no_grad():
        for step in range(max_new_tokens):
            pos = torch.tensor([S + step], device=input_ids.device)
            out = model(
                next_tok,
                past_key_values=cache,
                use_cache=True,
                cache_position=pos,
            )
            next_tok = out.logits[:, -1:].argmax(dim=-1)
    if cuda:
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() if cuda else 0
    return resident, peak


def main(cfg: Config):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    ).eval()

    # Create the run dir up front so each cell can be written incrementally — a cell
    # that OOMs (dense_stream @128k is EXPECTED to, >94.5 GB ceiling) must not lose
    # the cells already measured. The parquet is rewritten after every cell.
    run_dir = create_run("k3_kernel_census", cfg)
    rows = []
    for S in cfg.seq_lens:
        input_ids = torch.randint(
            0, resolve_vocab_size(model.config), (1, S), device=model.device
        )
        for arm in cfg.arms:
            k_spec, v_spec = spec_pair(arm)
            for path, Cls in [
                ("dense_stream", StreamingQuantizedCache),
                ("chunked", PackedStreamingCache),
            ]:
                # fp16 is the uncompressed BASELINE — measure it only on
                # dense_stream. The chunked path exists to realize compression, and
                # PackedStreamingCache has no fp16 passthrough (its flush calls
                # quantize_packed, which rejects 'fp16'). fp16-chunked is not a
                # meaningful config; the Phase-3 gate is about the COMPRESSED
                # (k2b) chunked path. Skip the fp16 x chunked cell.
                if arm == "fp16" and path == "chunked":
                    continue
                cache = Cls(model.config, k_spec=k_spec, v_spec=v_spec)
                # Attach PackedStreamingCache so the codec arm routes through
                # chunked_attention_forward. StreamingQuantizedCache only needs
                # attach when pre_rope is True (its existing behavior).
                if isinstance(cache, PackedStreamingCache):
                    cache.attach(model)
                elif k_spec.pre_rope:
                    cache.attach(model)

                # An OOM is a DATA POINT here (the path exceeded the GPU), not a crash:
                # record a sentinel row (oom=True, peak=-1) and continue so the rest of
                # the census still lands. Without this, one expected OOM aborts the
                # whole process before the parquet is written.
                oom = False
                bpe_k = bpe_v = float("nan")
                try:
                    resident, peak = _measure(
                        model, input_ids, cache, cfg.max_new_tokens
                    )
                    # bpe is only meaningful for a cell that actually ran; read it
                    # inside the try so a half-built post-OOM cache can't throw here.
                    if hasattr(cache, "bits_per_entry"):
                        bpe_k, bpe_v = cache.bits_per_entry()
                except torch.cuda.OutOfMemoryError:
                    oom = True
                    resident, peak = -1, -1
                if hasattr(cache, "detach"):
                    cache.detach()
                del cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                rows.append(
                    {
                        "seq_len": S,
                        "arm": arm,
                        "path": path,
                        "resident_after_prefill": resident,
                        "peak_decode": peak,
                        "peak_decode_incremental": (peak - resident if not oom else -1),
                        "oom": oom,
                        "bpe_k": bpe_k,
                        "bpe_v": bpe_v,
                    }
                )
                # Rewrite after every cell so a later OOM/crash keeps prior results.
                write_metrics(run_dir, pd.DataFrame(rows), "census")
                status = "OOM" if oom else f"peak={peak / 1024**3:.1f}GiB"
                print(f"  {arm:5s} {path:12s} S={S:<7d} {status}", flush=True)

    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nwrote {run_dir / 'census.parquet'}")


if __name__ == "__main__":
    main(tyro.cli(Config))
