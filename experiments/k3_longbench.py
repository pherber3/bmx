"""K3-LongBench — LongBench Code (lcc, repobench-p) recall under KV compression.

Reproduces TurboQuant's Code-column signal: sweeps arms × tasks on ONE code path
(StreamingQuantizedCache via generate_through_cache), scores LongBench code_sim. Reports each
arm's honest measured compression (never a pinned ratio).

Real path (model is None, VM run): loads model + tokenizer + THUDM/LongBench, generates through
the compressed cache, scores code_sim over n_samples (all if None) per task.

Offline-test path (model injected): a tiny synthetic code-ish prompt through tiny_llama;
code_sim against a known target. Mechanism + parquet schema only; no download, no tokenizer.

Model-agnostic: the SOTA VM run is a --model-name change. Figures: plots/plot_k3_longbench.py.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.longbench import code_sim
from bmx.cache.niah import generate_through_cache
from experiments.k3_live_generation import _spec_pair
from experiments.k3_niah import _compression_for


@dataclasses.dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
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
    """Offline decode stub: turns ids into a deterministic 'code-ish' string for code_sim."""

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids.tolist())


def run(cfg: Config, model=None, root: str = "results"):
    tokenizer = None
    if model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from bmx.cache.longbench import load_longbench_task, longbench_code_score

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch.float16
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    if cfg.n_samples is not None:
        print(
            f"[k3_longbench] SUBSAMPLED n_samples={cfg.n_samples} — NOT comparable to Table 1"
        )

    run_dir = create_run("k3_longbench", cfg, root=root)
    rows = []
    for arm in cfg.arms:
        k_spec, v_spec = _spec_pair(arm, cfg)
        for task in cfg.tasks:
            if tokenizer is None:
                # Offline mechanism: one synthetic prompt through the shared generate path.
                g = torch.Generator().manual_seed(cfg.seed)
                prompt_ids = torch.randint(
                    0, model.config.vocab_size, (1, 32), generator=g
                )
                # Mechanism/schema only: a few new tokens suffice (and stay within
                # tiny_llama's max_position_embeddings=64). The real max_gen is used on
                # the VM path; here the response length is irrelevant — we score it
                # against itself.
                resp = generate_through_cache(
                    model,
                    _StubTok(),
                    prompt_ids,
                    cfg.n_prefill,
                    k_spec,
                    v_spec,
                    max_new_tokens=4,
                )
                score = code_sim(resp, resp)  # identical => 1.0; mechanism/schema only
                n_used = 1
                score_kind = "code_sim_offline"
                length = prompt_ids.shape[1]
            else:
                items = load_longbench_task(task, cfg.n_samples)
                scores = [
                    longbench_code_score(
                        model, tokenizer, it, task, cfg.n_prefill, k_spec, v_spec
                    )
                    for it in items
                ]
                score = sum(scores) / len(scores) if scores else float("nan")
                n_used = len(items)
                score_kind = "code_sim"
                # NOTE: representative length for compression accounting only. Real
                # LongBench code prompts are far longer (4k–16k), so this proxy
                # UNDER-states compression (the fixed fp16 recent-window is a larger
                # fraction at short length). It is consistent across arms, so relative
                # rankings hold; the absolute compression column is a lower bound. A
                # future pass could thread the true tokenized prompt length through.
                length = cfg.n_prefill * 2

            bpe_k, bpe_v, compression = _compression_for(model, k_spec, v_spec, length)
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
