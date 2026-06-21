"""K3 — live-generation KV-compression verdict: K2b vs TurboQuant vs KIVI vs fp16.

Sweeps arms on one code path (StreamingQuantizedCache), measuring live-generation
perplexity on real wikitext-2 text (token-by-token honest streaming regime), honest
bpe, a tokenizer-free retrieval-fidelity proxy (``retrieved``), and a real
planted-needle retrieval score (``needle_real``).

Real-text path (model is None, real run on VM):
  Loads wikitext-2 test split via load_eval_tokens → real per-token NLL →
  meaningful live-generation perplexity.  Also builds a planted-needle prompt
  (build_needle_ids) and measures whether each arm still retrieves the fact.

Offline-test path (model injected by the test, input_ids provided):
  Uses synthetic random ids.  No download, no tokenizer.  Only exercises the
  parquet schema and codec mechanics.  needle_real is not computed in this path.

Model-agnostic: the SOTA VM run is a --model-name change.  Figures read the
parquet (plots/plot_k3.py).
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.live_eval import live_generation_ppl
from bmx.cache.needle import (
    build_needle_ids,
    needle_retrieved,
    needle_retrieved_from_ids,
)
from bmx.cache.specs import CacheCodecSpec


@dataclasses.dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.2-1B"
    device: str = "cpu"  # "cuda" on the VM
    arms: tuple[str, ...] = ("fp16", "k2b", "turboquant_mse", "turboquant_prod", "kivi")
    n_prefill: int = 256
    n_context: int = 512
    rank: int = 16
    group: int = 64
    seed: int = 0
    seq_seed: int = 42
    needle_depth: float = 0.5
    """Fractional depth [0, 1] at which the needle is planted in the haystack."""
    token_by_token: bool = True
    """Score the continuation one token at a time (honest streaming regime).
    Set to False to use the fast batched path (equivalent to old quantized-prefill
    ppl relabelled live; does not measure compounding quant errors).
    Default True so the experiment measures the honest live regime (K3 verdict)."""


def _spec_pair(arm: str, cfg: Config):
    """(k_spec, v_spec) for an arm.

    K2b = lowrank K@3b pre-RoPE + rotate/Lloyd V@2b (the quality-first recipe; spends
    bits on keys, so it lands LOWER on compression than turboquant). For an apples-to-
    apples comparison at turboquant's compression, the ``k2b_kNbM`` arms drop the key
    budget to N bits / rank M: ``k2b_k2r8`` lands at ~7.2x (matched to turboquant_mse's
    7.9x and kivi's 7.1x), so quality differences there are at equal bits, not bought
    with extra storage. See the local bpe table in the session notes.
    """
    if arm == "fp16":
        return CacheCodecSpec(arm="fp16"), CacheCodecSpec(arm="fp16")
    if arm == "k2b" or arm.startswith("k2b_k"):
        # Default canonical k2b: keys@3b, rank=cfg.rank. Parameterized variants
        # "k2b_k{bits}r{rank}" override the key budget to match compression.
        bits_k, rank_k = 3, cfg.rank
        if arm != "k2b":
            # Parse "k2b_k2r8" -> bits_k=2, rank=8.
            body = arm[len("k2b_k") :]
            bits_str, rank_str = body.split("r")
            bits_k, rank_k = int(bits_str), int(rank_str)
        return (
            CacheCodecSpec(
                arm="lowrank_rtn_channel",
                bits=bits_k,
                rank=rank_k,
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


def run(cfg: Config, model=None, input_ids=None, root: str = "results"):
    # --- Model and tokenizer loading ---
    # When model is None (real run on VM): load model + tokenizer + real text.
    # When model is injected (offline test): use synthetic ids, skip downloads.
    tokenizer = None
    needle_ids = None
    answer_id = None

    if model is None:
        # Real run: download model, tokenizer, and wikitext-2 real text.
        # This path is exercised only on the VM/real run, NOT in CI.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from bmx.eval.layer_swap import load_eval_tokens

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch.float16
        )
        model = model.to(cfg.device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

        # Real text: wikitext-2 test split → meaningful live-generation perplexity.
        toks = load_eval_tokens(cfg.model_name, n_tokens=cfg.n_context)
        input_ids = toks[: cfg.n_context].unsqueeze(0).to(cfg.device)  # (1, n_context)

        # Real planted needle for retrieval scoring.
        needle_ids, answer_id = build_needle_ids(
            tokenizer, n_context=cfg.n_context, depth_frac=cfg.needle_depth
        )
        needle_ids = needle_ids.to(cfg.device)

    if input_ids is None:
        # Offline test supplied a model but no input_ids: use synthetic ids.
        vocab = model.config.vocab_size
        input_ids = _make_ids(cfg, vocab).to(cfg.device)

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
        # Tokenizer-free retrieval proxy: argmax agreement vs fp16 on the same ids.
        retrieved = needle_retrieved_from_ids(
            model,
            input_ids,
            query_pos=cfg.n_context - 1,
            n_prefill=cfg.n_prefill,
            k_spec=k_spec,
            v_spec=v_spec,
        )
        # Real planted-needle retrieval (real run only; None in offline test).
        if needle_ids is not None and answer_id is not None:
            needle_real = needle_retrieved(
                model, needle_ids, answer_id, k_spec, v_spec, cfg.n_prefill
            )
        else:
            # Offline test: no tokenizer / needle prompt — skip real needle.
            needle_real = None

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
                "needle_real": needle_real,
            }
        )

    write_metrics(run_dir, pd.DataFrame(rows))
    return run_dir


if __name__ == "__main__":
    run(tyro.cli(Config))
