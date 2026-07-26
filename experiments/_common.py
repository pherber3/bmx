"""Shared scaffolding for the K3 experiment scripts (real-run path only).

The offline test path injects `model=` and never calls this — the transformers
import stays function-local so importing an experiment module downloads nothing.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import pandas as pd
import torch

from bmx.artifacts import git_sha


def load_model_and_tokenizer(model_name: str, device: str):
    """fp16 CausalLM + tokenizer, moved to device, eval mode. VM/real-run path."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


# --- Checkpoint/resume for long arm x task (x sample) sweeps -----------------
#
# Shards live at <run_dir>/partial/<pair_key>.parquet — UNDER the run-id dir, so
# newest_run_with-style globbing over `<experiment>/*/metrics.parquet` never mistakes a
# partial for a completed run. The canonical metrics.parquet (written once, at the end,
# via bmx.artifacts.write_metrics) stays the only file plot code reads; its rows are the
# concat of every shard. Parquet has no append, so each (arm, task)-style pair gets
# exactly one shard file, written once when that pair completes.
#
# Per-sample shards (bootstrap-CI enabler) ride alongside each aggregate shard at
# <run_dir>/partial/<pair_key>__samples.parquet — written BEFORE the aggregate shard, so
# the aggregate's existence (the resume key) implies its samples shard exists. Both
# load_shards and done_pairs IGNORE samples shards: metrics.parquet stays aggregate-only
# (byte-identical schema to before samples existed) and resume keys stay aggregate-only.


def pair_key(*parts: str) -> str:
    """Filesystem-safe key for a sweep pair/tuple, e.g. pair_key(arm, task)."""
    return "__".join(str(p) for p in parts)


def shard_path(run_dir: Path, *parts: str) -> Path:
    return Path(run_dir) / "partial" / f"{pair_key(*parts)}.parquet"


def write_shard(run_dir: Path, rows: list[dict], *parts: str) -> Path:
    """Write one pair's rows to its shard file. One file per pair; never appended to."""
    path = shard_path(run_dir, *parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


SAMPLES_SUFFIX = "__samples"


def samples_shard_path(run_dir: Path, *parts: str) -> Path:
    # pair_key joins with "__", so appending a "samples" part reproduces
    # SAMPLES_SUFFIX byte-for-byte in the filename.
    return shard_path(run_dir, *parts, "samples")


def write_samples_shard(run_dir: Path, rows: list[dict], *parts: str) -> Path | None:
    """Write one pair's per-sample rows next to its aggregate shard; None if no rows.

    Call BEFORE write_shard for the same pair: the aggregate shard is the resume key,
    so its existence must imply the samples shard exists (a crash between the two
    writes leaves an orphan samples shard, which a resumed run simply overwrites)."""
    if not rows:
        return None
    return write_shard(run_dir, rows, *parts, "samples")


def _is_samples_shard(p: Path) -> bool:
    return p.stem.endswith(SAMPLES_SUFFIX)


def load_shards(run_dir: Path) -> pd.DataFrame:
    """Concat every completed pair's AGGREGATE shard rows (empty frame if none yet).

    Samples shards are excluded: metrics.parquet must keep its pre-samples schema."""
    partial_dir = Path(run_dir) / "partial"
    if not partial_dir.exists():
        return pd.DataFrame()
    shards = sorted(
        p for p in partial_dir.glob("*.parquet") if not _is_samples_shard(p)
    )
    if not shards:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


def load_samples_shards(run_dir: Path) -> pd.DataFrame:
    """Concat every completed pair's SAMPLES shard rows (empty frame if none yet).

    The filter-inverted twin of load_shards: the input for a run-level
    samples.parquet written beside metrics.parquet at run completion."""
    partial_dir = Path(run_dir) / "partial"
    if not partial_dir.exists():
        return pd.DataFrame()
    shards = sorted(p for p in partial_dir.glob("*.parquet") if _is_samples_shard(p))
    if not shards:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


def done_pairs(run_dir: Path) -> set[str]:
    """Pair keys (filenames without .parquet) that already have an AGGREGATE shard.

    Samples shards are excluded — an orphan samples shard (crash between the samples
    write and the aggregate write) must NOT mark its pair as done."""
    partial_dir = Path(run_dir) / "partial"
    if not partial_dir.exists():
        return set()
    return {p.stem for p in partial_dir.glob("*.parquet") if not _is_samples_shard(p)}


# Config fields that legitimately differ between the original run and a --resume
# invocation. `resume` itself always differs (unset originally, set on resume) and
# carries no scientific meaning — it names where to resume, not what was measured.
RESUME_EXCLUDED_FIELDS = frozenset({"resume"})


def assert_resume_identity(run_dir: Path, cfg) -> None:
    """Hard-error if the resuming process's config/code differs from the original run's.

    Mixed-config or mixed-code rows are exactly the double-count/mixed-provenance pitfall
    CLAUDE.md warns about (comparisons must align on a single measured configuration) — a
    resumed run silently continuing under a DIFFERENT config or git SHA would corrupt the
    final parquet with rows that aren't comparable to each other. Any mismatch raises.
    """
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    env_path = run_dir / "env.json"
    if not config_path.exists() or not env_path.exists():
        raise ValueError(
            f"cannot resume {run_dir}: missing config.json/env.json (not a valid run dir)"
        )

    stored_cfg = json.loads(config_path.read_text())
    raw_current_cfg = (
        dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)
    )
    # Round-trip through JSON (same as create_run's config.json write) so tuple-vs-list and
    # other JSON-lossy type differences don't produce false mismatches.
    current_cfg = json.loads(json.dumps(raw_current_cfg, default=str))

    stored_cmp = {
        k: v for k, v in stored_cfg.items() if k not in RESUME_EXCLUDED_FIELDS
    }
    current_cmp = {
        k: v for k, v in current_cfg.items() if k not in RESUME_EXCLUDED_FIELDS
    }
    if stored_cmp != current_cmp:
        diffs = {
            k: (stored_cmp.get(k), current_cmp.get(k))
            for k in stored_cmp.keys() | current_cmp.keys()
            if stored_cmp.get(k) != current_cmp.get(k)
        }
        raise ValueError(
            f"resume config mismatch at {run_dir}: stored vs current differ on {diffs} "
            "— resume requires an identical config (excluding `resume` itself)"
        )

    stored_sha = json.loads(env_path.read_text()).get("git_sha")
    current_sha = git_sha()
    if stored_sha != current_sha:
        raise ValueError(
            f"resume git SHA mismatch at {run_dir}: stored={stored_sha!r} "
            f"current={current_sha!r} — resume requires the same code that started the run"
        )


def print_progress(prefix: str, i: int, n: int, start_time: float, **fields) -> None:
    """Flushed one-line progress: `[prefix i/n] key=val ... elapsed=12.3s`."""
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    elapsed = time.monotonic() - start_time
    print(f"[{prefix} {i}/{n}] {extra} elapsed={elapsed:.1f}s", flush=True)
