"""K3 — live-generation KV-compression verdict: K2b vs TurboQuant vs KIVI vs fp16.

Sweeps arms on one code path (StreamingQuantizedCache), measuring live-generation
perplexity, honest bpe, and a retrieval-fidelity proxy. Model-agnostic: the SOTA
VM run is a --model-name change. Figures read the parquet (plots/plot_k3.py).
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.live_eval import live_generation_ppl
from bmx.cache.needle import needle_retrieved_from_ids
from bmx.cache.specs import CacheCodecSpec


@dataclasses.dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.2-1B"
    arms: tuple[str, ...] = ("fp16", "k2b", "turboquant_mse", "turboquant_prod", "kivi")
    n_prefill: int = 256
    n_context: int = 512
    rank: int = 16
    group: int = 64
    seed: int = 0
    seq_seed: int = 42
    token_by_token: bool = True
    """Score the continuation one token at a time (honest streaming regime).
    Set to False to use the fast batched path (equivalent to old quantized-prefill
    ppl relabelled live; does not measure compounding quant errors).
    Default True so the experiment measures the honest live regime (K3 verdict)."""


def _spec_pair(arm: str, cfg: Config):
    """(k_spec, v_spec) for an arm. K2b = lowrank K@3b pre-RoPE + rotate/Lloyd V@2b."""
    if arm == "fp16":
        return CacheCodecSpec(arm="fp16"), CacheCodecSpec(arm="fp16")
    if arm == "k2b":
        return (
            CacheCodecSpec(
                arm="lowrank_rtn_channel",
                bits=3,
                rank=cfg.rank,
                group=cfg.group,
                seed=cfg.seed,
                pre_rope=True,
            ),
            CacheCodecSpec(arm="turboquant_mse", bits=2, seed=cfg.seed),
        )
    if arm in ("turboquant_mse", "turboquant_prod"):
        s = CacheCodecSpec(arm=arm, bits=2, seed=cfg.seed)
        return s, s
    if arm == "kivi":
        return (
            CacheCodecSpec(arm="rtn_channel", bits=2, group=cfg.group, seed=cfg.seed),
            CacheCodecSpec(arm="rtn_token", bits=2, group=cfg.group, seed=cfg.seed),
        )
    raise ValueError(f"unknown arm {arm!r}")


def _make_ids(cfg: Config, vocab: int):
    g = torch.Generator().manual_seed(cfg.seq_seed)
    return torch.randint(0, vocab, (1, cfg.n_context), generator=g)


def run(cfg: Config, model=None, root: str = "results"):
    if model is None:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch.float16
        )
        model.eval()

    vocab = model.config.vocab_size
    input_ids = _make_ids(cfg, vocab)
    run_dir = create_run("k3_live_generation", cfg, root=root)

    rows = []
    for arm in cfg.arms:
        k_spec, v_spec = _spec_pair(arm, cfg)
        res = live_generation_ppl(
            model,
            input_ids,
            cfg.n_prefill,
            k_spec,
            v_spec,
            token_by_token=cfg.token_by_token,
        )
        retrieved = needle_retrieved_from_ids(
            model,
            input_ids,
            query_pos=cfg.n_context - 1,
            n_prefill=cfg.n_prefill,
            k_spec=k_spec,
            v_spec=v_spec,
        )
        rows.append(
            {
                "arm": arm,
                "bpe_k": res["bpe_k"],
                "bpe_v": res["bpe_v"],
                "ppl": res["ppl"],
                "n_eval": res["n_eval"],
                "packed_bytes": res["packed_bytes"],
                "fp16_bytes": res["fp16_bytes"],
                "compression": res["compression"],
                "n_prefill": cfg.n_prefill,
                "n_context": cfg.n_context,
                "retrieved": retrieved,
            }
        )

    write_metrics(run_dir, pd.DataFrame(rows))
    return run_dir


if __name__ == "__main__":
    run(tyro.cli(Config))
