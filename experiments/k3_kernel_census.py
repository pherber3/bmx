"""KV-cache memory census: resident / peak / incremental per arm per length.

Measures both cache paths (StreamingQuantizedCache, PackedStreamingCache) and
compares against the analytic byte-ledger. CUDA-authoritative (VM); falls back to
a tiny CPU smoke run locally. Writes parquet to results/k3_kernel_census/<run-id>.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.packed_streaming import PackedStreamingCache
from bmx.cache.specs import CacheCodecSpec
from bmx.cache.streaming import StreamingQuantizedCache


@dataclasses.dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    seq_lens: tuple[int, ...] = (4096, 16384, 32768)
    arms: tuple[str, ...] = ("fp16", "k2b")
    max_new_tokens: int = 4


def _specs(arm):
    if arm == "fp16":
        return CacheCodecSpec(arm="fp16"), CacheCodecSpec(arm="fp16")
    if arm == "k2b":
        return (
            CacheCodecSpec(
                arm="lowrank_rtn_channel", bits=3, rank=16, group=64, pre_rope=True
            ),
            CacheCodecSpec(arm="turboquant_mse", bits=2),
        )
    raise ValueError(arm)


def _measure(model, input_ids, cache):
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True)
    resident = torch.cuda.max_memory_allocated() if cuda else 0
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model.generate(
            input_ids,
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
            past_key_values=cache,
        )
    peak = torch.cuda.max_memory_allocated() if cuda else 0
    return resident, peak


def main(cfg: Config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    ).eval()
    rows = []
    for S in cfg.seq_lens:
        input_ids = torch.randint(0, tok.vocab_size, (1, S), device=model.device)
        for arm in cfg.arms:
            k_spec, v_spec = _specs(arm)
            for path, Cls in [
                ("dense_stream", StreamingQuantizedCache),
                ("chunked", PackedStreamingCache),
            ]:
                cache = Cls(model.config, k_spec=k_spec, v_spec=v_spec)
                if k_spec.pre_rope:
                    cache.attach(model)
                resident, peak = _measure(model, input_ids, cache)
                if hasattr(cache, "detach"):
                    cache.detach()
                bpe_k, bpe_v = (
                    cache.bits_per_entry()
                    if hasattr(cache, "bits_per_entry")
                    else (float("nan"), float("nan"))
                )
                rows.append(
                    {
                        "seq_len": S,
                        "arm": arm,
                        "path": path,
                        "resident_after_prefill": resident,
                        "peak_decode": peak,
                        "peak_decode_incremental": peak - resident,
                        "bpe_k": bpe_k,
                        "bpe_v": bpe_v,
                    }
                )
                del cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    run_dir = create_run("k3_kernel_census", cfg)
    write_metrics(run_dir, df, "census")
    print(df.to_string(index=False))
    print(f"\nwrote {run_dir / 'census.parquet'}")


if __name__ == "__main__":
    main(tyro.cli(Config))
