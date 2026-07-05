"""LongBench recall under KV compression across the 6 TurboQuant Table-1 categories.

Sweeps arms × tasks through the StreamingQuantizedCache and scores each task with its
LongBench metric (code_sim / qa_f1 / rouge / classification / retrieval / count), recording
each arm's measured compression.

`--categories` expands to the English datasets per category (CATEGORY2DATASETS); `--tasks`
still names individual datasets. When both are empty categories wins; when categories is empty
the explicit `tasks` tuple is used (default: the code pair, for back-compat).

When `model` is None: loads the model, tokenizer, and LongBench, and scores over n_samples
items (all if None) per task. When `model` is injected (tests): scores one synthetic
generation against itself — schema and mechanism only, no download.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.generate import avg_bpe, compression_for, generate_through_cache
from bmx.cache.hf_compat import resolve_vocab_size
from bmx.cache.longbench import CATEGORY2DATASETS, DATASET2METRIC, code_sim
from bmx.cache.recipes import spec_pair
from experiments._common import (
    assert_resume_identity,
    done_pairs,
    load_shards,
    pair_key,
    print_progress,
    write_shard,
)


@dataclasses.dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    device: str = "cpu"  # "cuda" on the VM
    arms: tuple[str, ...] = ("fp16", "k2b", "turboquant_mse", "turboquant_prod", "kivi")
    # Explicit dataset names (default: the code pair). Ignored when `categories` is set.
    tasks: tuple[str, ...] = ("lcc", "repobench-p")
    # TurboQuant Table-1 category names; expand to English datasets via CATEGORY2DATASETS.
    categories: tuple[str, ...] = ()
    # Loader version: 'v1' (THUDM/LongBench full split, parity-run default) or 'v1_e'
    # (LongBench-E, the length-uniform subset TurboQuant Table-1 actually evaluates on —
    # see load_longbench_task's docstring). Not every v1 task has an _e file; requesting
    # 'v1_e' for one raises loudly rather than silently falling back to v1.
    longbench_version: str = "v1"
    n_samples: int | None = (
        None  # None = full sets (Table-1 comparable); int caps (logged)
    )
    # Middle-truncation budget (LongBench pred.py: keep first/last max_prompt_tokens//2 ids).
    # TurboQuant (arXiv 2504.19874) §4 states no prompt-truncation budget; 31500 is the
    # LongBench-convention fallback (LongBench pred.py middle-truncation). None = no-truncation
    # variant (byte-identical to pre-truncation behavior).
    max_prompt_tokens: int | None = 31500
    n_prefill: int = 128
    rank: int = 16
    group: int = 64
    seed: int = 0
    # Resume a crashed/killed run: path to its run_dir. Skips (arm, task) pairs whose
    # partial/<arm>__<task>.parquet shard already exists; identity-asserted against the
    # stored config.json + git SHA (mismatch is a hard error, not a warning) so a resumed
    # run can never mix rows measured under different code/config. Not itself part of the
    # measured configuration, so it is excluded from that identity check (see
    # experiments._common.RESUME_EXCLUDED_FIELDS).
    resume: str | None = None

    def resolved_tasks(self) -> tuple[str, ...]:
        """Datasets to evaluate: categories expanded (dedup, ordered) else explicit tasks."""
        if self.categories:
            # dict.fromkeys dedupes while preserving first-seen order.
            flat = [ds for cat in self.categories for ds in CATEGORY2DATASETS[cat]]
            return tuple(dict.fromkeys(flat))
        return self.tasks


class _StubTok:
    """Decode stub for the offline path: ids to a deterministic string."""

    def decode(self, ids, skip_special_tokens=True):
        seq = ids.tolist() if hasattr(ids, "tolist") else ids
        return " ".join(str(int(i)) for i in seq)


def run(
    cfg: Config, model=None, root: str = "results", _stop_after_pairs: int | None = None
):
    tasks = cfg.resolved_tasks()
    tokenizer = None
    if model is None:
        from experiments._common import load_model_and_tokenizer

        from bmx.cache.longbench import load_longbench_task, longbench_score

        model, tokenizer = load_model_and_tokenizer(cfg.model_name, cfg.device)

    if cfg.n_samples is not None:
        print(
            f"[k3_longbench] SUBSAMPLED n_samples={cfg.n_samples} — NOT comparable to Table 1"
        )

    if cfg.resume is not None:
        run_dir = Path(cfg.resume)
        assert_resume_identity(run_dir, cfg)
        skip_pairs = done_pairs(run_dir)
        print(
            f"[k3_longbench] resuming {run_dir}: {len(skip_pairs)} pair(s) already done",
            flush=True,
        )
    else:
        run_dir = create_run("k3_longbench", cfg, root=root)
        skip_pairs = set()

    # A task's dataset is identical across arms; load each once.
    task_items = (
        {
            task: load_longbench_task(task, cfg.n_samples, cfg.longbench_version)
            for task in tasks
        }
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
            build_longbench_prompt(tokenizer, it, task, cfg.max_prompt_tokens).shape[1]
            for it in task_items[task]
        )
        return lens[len(lens) // 2]  # median; equal across arms, so rankings unaffected

    # Per-task calibration length (depends only on the task's prompts, not the arm).
    calib_length = {task: _calib_length(task) for task in tasks}

    n_pairs = len(cfg.arms) * len(tasks)
    pair_i = 0
    n_completed_this_run = 0
    start_time = time.monotonic()
    for arm in cfg.arms:
        k_spec, v_spec = spec_pair(arm, rank=cfg.rank, group=cfg.group, seed=cfg.seed)
        for task in tasks:
            pair_i += 1
            key = pair_key(arm, task)
            if key in skip_pairs:
                print(
                    f"[k3_longbench pair {pair_i}/{n_pairs}] arm={arm} task={task} "
                    "SKIP (resumed)",
                    flush=True,
                )
                continue

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
                print_progress(
                    "k3_longbench sample",
                    1,
                    1,
                    start_time,
                    arm=arm,
                    task=task,
                    score=f"{score:.4f}",
                )
            else:
                items = task_items[task]
                scores = []
                for sample_i, it in enumerate(items, start=1):
                    s = longbench_score(
                        model,
                        tokenizer,
                        it,
                        task,
                        cfg.n_prefill,
                        k_spec,
                        v_spec,
                        cfg.max_prompt_tokens,
                    )
                    scores.append(s)
                    if sample_i % 10 == 0 or sample_i == len(items):
                        print_progress(
                            f"arm {arm} task {task} sample",
                            sample_i,
                            len(items),
                            start_time,
                            score=f"{s:.4f}",
                        )
                score = sum(scores) / len(scores) if scores else float("nan")
                n_used = len(items)

            pair_rows = [
                {
                    "arm": arm,
                    "task": task,
                    "code_sim": score,  # generic per-item metric value (see `metric` col)
                    "metric": DATASET2METRIC[task].__name__,
                    "n_samples": n_used,
                    "bpe_k": bpe_k,
                    "bpe_v": bpe_v,
                    "kv_size_bits": avg_bpe(bpe_k, bpe_v),
                    "compression": compression,
                    "n_prefill": cfg.n_prefill,
                    "score_kind": score_kind,
                    "max_prompt_tokens": cfg.max_prompt_tokens
                    if cfg.max_prompt_tokens is not None
                    else -1,
                    "longbench_version": cfg.longbench_version,
                }
            ]
            write_shard(run_dir, pair_rows, arm, task)
            print(
                f"[k3_longbench pair {pair_i}/{n_pairs}] arm={arm} task={task} "
                f"score={score:.4f} DONE (checkpointed)",
                flush=True,
            )
            n_completed_this_run += 1
            if (
                _stop_after_pairs is not None
                and n_completed_this_run >= _stop_after_pairs
            ):
                raise RuntimeError(
                    f"injected stop after {n_completed_this_run} pair(s) (test hook)"
                )

    write_metrics(run_dir, load_shards(run_dir))
    return run_dir


if __name__ == "__main__":
    run(tyro.cli(Config))
