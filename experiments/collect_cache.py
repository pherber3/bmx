"""Collect KV/Q/K_pre cache tensors for a single model + sequence length.

Usage
-----
    uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024
    uv run python experiments/collect_cache.py --model-name meta-llama/Llama-3.1-8B --seq-len 2048
    uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024 --token-offset 1024
    uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024 \
        --token-offset 1024 --dataset-id bigcode/the-stack-smol --data-dir data/python \
        --split train --text-field content --corpus-label code

--token-offset shifts the wikitext slice so distinct offsets act as distinct
documents (calibration corpora for the corpus-vs-heldout transfer test).

Output is written to results/cache/<model_short>_<seq_len>.safetensors, or
..._<seq_len>[_<corpus_label>][_off<token_offset>].safetensors when
--corpus-label and/or --token-offset are set (or --out overrides directly).
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
    token_offset: int = 0  # calibration-corpus slice offset (0 => leading tokens)
    out: str = ""  # override output path; empty => auto
    # ---- corpus passthrough (K4 corpus-transfer gate) ----------------------
    dataset_id: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"  # "" => dataset has no config name
    data_dir: str = ""  # HF data_dir (the-stack-smol style); overrides config
    split: str = "test"
    text_field: str = "text"
    shuffle_seed: int = -1  # >=0 => seeded post-slice token shuffle (null corpus)
    corpus_label: str = ""  # REQUIRED when any corpus knob above is non-default


_WIKI_DEFAULTS = ("Salesforce/wikitext", "wikitext-2-raw-v1", "", "test", "text", -1)


def _corpus_is_default(cfg: Config) -> bool:
    return (
        cfg.dataset_id,
        cfg.dataset_config,
        cfg.data_dir,
        cfg.split,
        cfg.text_field,
        cfg.shuffle_seed,
    ) == _WIKI_DEFAULTS


def _out_path(cfg: Config) -> Path:
    if cfg.out:
        return Path(cfg.out)
    model_short = cfg.model_name.split("/")[-1].lower()
    label = f"_{cfg.corpus_label}" if cfg.corpus_label else ""
    suffix = f"_off{cfg.token_offset}" if cfg.token_offset else ""
    return (
        Path("results/cache")
        / f"{model_short}_{cfg.seq_len}{label}{suffix}.safetensors"
    )


def main(cfg: Config) -> None:
    # Never overwrite wikitext-named caches with a different corpus's content.
    assert _corpus_is_default(cfg) or cfg.corpus_label, (
        "non-default corpus knobs require --corpus-label so the output name "
        "encodes the corpus"
    )
    out_path = _out_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model (gpt2 is tiny — keep it fp32; everything else bf16)
    print(f"Loading model: {cfg.model_name}", flush=True)
    from transformers import AutoModelForCausalLM

    dtype = None if cfg.model_name == "gpt2" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype)
    model.eval()

    # Load tokens — load_eval_tokens returns a 1-D tensor; collect_cache wants (1, S)
    print(f"Loading {cfg.seq_len} eval tokens for {cfg.model_name}", flush=True)
    tokens = load_eval_tokens(
        cfg.model_name,
        cfg.dataset_config,
        n_tokens=cfg.seq_len,
        token_offset=cfg.token_offset,
        dataset_id=cfg.dataset_id,
        data_dir=cfg.data_dir,
        split=cfg.split,
        text_field=cfg.text_field,
        shuffle_seed=cfg.shuffle_seed,
    )
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
