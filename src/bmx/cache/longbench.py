"""LongBench Code eval: code_sim scorer, dataset loader, task registry.

Faithful port of LongBench's Code-category eval (lcc, repobench-p). The scorer matches
LongBench's metrics.py::code_sim_score exactly (verified in .git/sdd/longbench-conventions.md
against the local clone). The dataset loader + scoring path are VM-only (real model); code_sim
is a pure CI-testable function.
"""

from __future__ import annotations

import torch
from fuzzywuzzy import fuzz

from bmx.cache.niah import generate_through_cache
from bmx.cache.specs import CacheCodecSpec

# VERBATIM from LongBench config (verified in .git/sdd/longbench-conventions.md against the
# local clone LongBench/LongBench/config/dataset2prompt.json + dataset2maxlen.json).
# NOTE the exact strings: trailing space after "below. ", and repobench-p has {input}, lcc does
# not. Copy these EXACTLY — do not normalize whitespace.
LONGBENCH_TASKS: dict[str, dict] = {
    "lcc": {
        "prompt_template": "Please complete the code given below. \n{context}Next line of code:\n",
        "max_gen": 64,
    },
    "repobench-p": {
        "prompt_template": "Please complete the code given below. \n{context}{input}Next line of code:\n",
        "max_gen": 64,
    },
}


def build_longbench_prompt(tokenizer, item: dict, task: str) -> torch.Tensor:
    """Apply the task's LongBench prompt template to the item; return (1, L) ids.

    LongBench formats dataset2prompt[task].format(**item); for code tasks the context lives in
    item['context']. (If the ledger records a build_chat wrapper for Llama-Instruct, apply it
    here exactly as LongBench does.)
    """
    template = LONGBENCH_TASKS[task]["prompt_template"]
    prompt = template.format(**item)
    return tokenizer(prompt, return_tensors="pt").input_ids


def load_longbench_task(task: str, n_samples: int | None) -> list[dict]:
    """Lazy-load THUDM/LongBench[task]; return up to n_samples items (all if None). VM-only."""
    from datasets import load_dataset

    ds = load_dataset("THUDM/LongBench", task, split="test")
    items = list(ds if n_samples is None else ds.select(range(min(n_samples, len(ds)))))
    return items


def longbench_code_score(
    model,
    tokenizer,
    item: dict,
    task: str,
    n_prefill: int,
    k_spec: CacheCodecSpec,
    v_spec: CacheCodecSpec,
) -> float:
    """Build the LongBench prompt, generate through the compressed cache, score code_sim.

    Ground truth is item['answers'][0] (LongBench code tasks have a single reference line).
    """
    prompt_ids = build_longbench_prompt(tokenizer, item, task)
    max_gen = LONGBENCH_TASKS[task]["max_gen"]
    response = generate_through_cache(
        model, tokenizer, prompt_ids, n_prefill, k_spec, v_spec, max_gen
    )
    ground_truth = item["answers"][0]
    return code_sim(response, ground_truth)


def code_sim(prediction: str, ground_truth: str) -> float:
    """LongBench code edit-similarity, range 0–1 (verbatim port of code_sim_score).

    Post-process: strip leading blank lines, then keep the FIRST line that contains no
    backtick / '#' / '//' (LongBench's rule), then fuzz.ratio normalized to [0, 1].
    """
    all_lines = prediction.lstrip("\n").split("\n")
    pred = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            pred = line
            break
    return fuzz.ratio(pred, ground_truth) / 100.0
