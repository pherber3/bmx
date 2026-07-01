"""LongBench Code eval (lcc, repobench-p): code_sim scorer, dataset loader, task registry.

code_sim ports LongBench's metrics.py::code_sim_score. The loader and per-item scorer require
a real model and dataset (VM only); code_sim and the registry are pure and CI-testable.
"""

from __future__ import annotations

import torch
from fuzzywuzzy import fuzz

from bmx.cache.generate import generate_through_cache
from bmx.cache.specs import CacheCodecSpec

# LongBench's prompt templates and max_gen for the code tasks. The templates are exact:
# trailing space after "below. ", repobench-p uses {input}, lcc does not. Do not normalize.
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
    item['context']. NO chat/[INST] wrapper: LongBench explicitly skips build_chat for the code
    tasks (lcc, repobench-p are in its exclusion list) — raw template only, even for Instruct
    models.
    """
    template = LONGBENCH_TASKS[task]["prompt_template"]
    prompt = template.format(**item)
    return tokenizer(prompt, return_tensors="pt").input_ids


def load_longbench_task(task: str, n_samples: int | None) -> list[dict]:
    """Load THUDM/LongBench[task]; return up to n_samples items (all if None). VM-only.

    THUDM/LongBench ships as a loader script + data.zip; datasets>=4 no longer runs dataset
    scripts, so read the task's jsonl out of data.zip directly via huggingface_hub.
    """
    import json
    import zipfile

    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    items: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(f"data/{task}.jsonl") as fh:
            for line in fh:
                items.append(json.loads(line))
                if n_samples is not None and len(items) >= n_samples:
                    break
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
        model, tokenizer, prompt_ids, n_prefill, k_spec, v_spec, max_gen, strip=False
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
