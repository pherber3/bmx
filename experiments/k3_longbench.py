"""LongBench Code (lcc, repobench-p) recall under KV compression.

Sweeps arms × tasks through the StreamingQuantizedCache and scores LongBench code_sim,
recording each arm's measured compression.

When `model` is None: loads the model, tokenizer, and THUDM/LongBench, and scores over
n_samples items (all if None) per task. When `model` is injected (tests): scores one synthetic
generation against itself — schema and mechanism only, no download.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.live_eval import compression_for
from bmx.cache.longbench import code_sim
from bmx.cache.niah import generate_through_cache
from bmx.cache.streaming import resolve_vocab_size
from experiments.k3_live_generation import _spec_pair


@dataclasses.dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    device: str = "cpu"  # "cuda" on the VM
    arms: tuple[str, ...] = ("fp16", "k2b", "turboquant_mse", "turboquant_prod", "kivi")
    tasks: tuple[str, ...] = ("lcc", "repobench-p")
    n_samples: int | None = (
        None  # None = full sets (Table-1 comparable); int caps (logged)
    )
    n_prefill: int = 128
    rank: int = 16
    group: int = 64
    seed: int = 0


class _StubTok:
    """Decode stub for the offline path: ids to a deterministic string."""

    def decode(self, ids, skip_special_tokens=True):
        seq = ids.tolist() if hasattr(ids, "tolist") else ids
        return " ".join(str(int(i)) for i in seq)


def run(cfg: Config, model=None, root: str = "results"):
    tokenizer = None
    if model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from bmx.cache.longbench import load_longbench_task, longbench_code_score

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch.float16
        )
        model = model.to(cfg.device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    if cfg.n_samples is not None:
        print(
            f"[k3_longbench] SUBSAMPLED n_samples={cfg.n_samples} — NOT comparable to Table 1"
        )

    run_dir = create_run("k3_longbench", cfg, root=root)

    # A task's dataset is identical across arms; load each once.
    task_items = (
        {task: load_longbench_task(task, cfg.n_samples) for task in cfg.tasks}
        if tokenizer is not None
        else None
    )
    score_kind = "code_sim_offline" if tokenizer is None else "code_sim"
    # Compression-calibration length. Real LongBench code prompts are 4k–16k tokens; calibrate
    # at the MEDIAN tokenized prompt length per task so the compression column is honest (a
    # short fixed proxy understates it badly — the fp16 recent-window is a larger fraction at
    # short length). Offline path has no tokenizer: keep the tiny synthetic length.
    from bmx.cache.longbench import build_longbench_prompt

    def _calib_length(task: str) -> int:
        if tokenizer is None:
            return 32
        lens = sorted(
            build_longbench_prompt(tokenizer, it, task).shape[1]
            for it in task_items[task]
        )
        return lens[len(lens) // 2]  # median; equal across arms, so rankings unaffected

    # Per-task calibration length (depends only on the task's prompts, not the arm).
    calib_length = {task: _calib_length(task) for task in cfg.tasks}

    rows = []
    for arm in cfg.arms:
        k_spec, v_spec = _spec_pair(arm, cfg)
        for task in cfg.tasks:
            bpe_k, bpe_v, compression = compression_for(
                model, k_spec, v_spec, calib_length[task]
            )
            if tokenizer is None:
                # Offline: score one synthetic generation against itself; mechanism only.
                # Generate on CPU (seeded Generator is CPU-only), move to model's device.
                g = torch.Generator().manual_seed(cfg.seed)
                prompt_ids = torch.randint(
                    0,
                    resolve_vocab_size(model.config),
                    (1, calib_length[task]),
                    generator=g,
                ).to(model.device)
                resp = generate_through_cache(
                    model,
                    _StubTok(),
                    prompt_ids,
                    cfg.n_prefill,
                    k_spec,
                    v_spec,
                    max_new_tokens=4,
                )
                score = code_sim(resp, resp)
                n_used = 1
            else:
                items = task_items[task]
                scores = [
                    longbench_code_score(
                        model, tokenizer, it, task, cfg.n_prefill, k_spec, v_spec
                    )
                    for it in items
                ]
                score = sum(scores) / len(scores) if scores else float("nan")
                n_used = len(items)

            rows.append(
                {
                    "arm": arm,
                    "task": task,
                    "code_sim": score,
                    "n_samples": n_used,
                    "bpe_k": bpe_k,
                    "bpe_v": bpe_v,
                    "compression": compression,
                    "n_prefill": cfg.n_prefill,
                    "score_kind": score_kind,
                }
            )

    write_metrics(run_dir, pd.DataFrame(rows))
    return run_dir


if __name__ == "__main__":
    run(tyro.cli(Config))
