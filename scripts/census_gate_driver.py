"""Task-9 96k->128k headroom-gate driver (packed-spectral plan, no human idle-wait).

The packed-spectral VM runbook's 128k headroom gate
(`docs/superpowers/plans/2026-07-23-packed-spectral-path.md`, Task 9) was
originally a human read-and-decide step: watch stage-1 (32k/64k/96k) census
finish, read the parquet, hand-compute a linear projection to 128k per
(path, arm), and only then launch the 128k census invocation for cells that
clear the ceiling -- leaving the GPU idle while a person did the reading.
This script automates that gate end to end:

1. Poll the stage-1 run dir's `census.parquet` until the expected cells
   (32768/65536/98304 for every requested arm x path) are present. A
   staleness guard aborts loudly if the parquet stops growing for
   `--stale-minutes` (default 30) -- the watcher-blind-spot lesson: a silent
   watcher that polls forever without ever flagging a stuck upstream job is
   itself a bug.
2. For each (path, arm) cell, project `resident_128k` via the plan's exact
   rule: `resident_96k + (resident_96k - resident_64k)` (linear growth). The
   arithmetic is printed for every cell -- this is a paper-trail requirement,
   not just a log line.
3. Cells whose projection is <= `--gate-gib` (default 90.0, i.e. >=5.6 GiB
   margin under the 95.6 GiB HBM ceiling) get their 128k census invocation
   launched. Launches are SEQUENTIAL -- one `subprocess.run` (blocking) call
   per cell, never concurrent; this mirrors the `wait` fix in commit
   `d767de3` for the two hand-written 128k launches in the plan itself.
   Cells over the gate are SKIPPED with the projected number printed (not
   silently dropped -- the point of the table is to show the expected-OOM
   class too, just not spend a launch confirming it here).
4. Exits nonzero if EVERY cell was skipped, so a wrapping script can detect
   "the gate closed everything" and stop the batch instead of silently
   proceeding to Task 10 with zero 128k evidence.

`--dry-run` prints exactly what would be launched (arithmetic + argv) without
calling subprocess at all.

Usage (VM, after stage-1 census.parquet is being written):

    uv run python scripts/census_gate_driver.py \\
        --stage1-run-dir results/k3_kernel_census/<stage1-run-id> \\
        --model-name meta-llama/Llama-3.1-8B-Instruct \\
        --pack-path results/cache/k4_packs_llama31_instruct.safetensors
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import tyro

GiB = 1024**3

STAGE1_SEQ_LENS = (32768, 65536, 98304)
GATE_128K_SEQ_LEN = 131072

STALE_MESSAGE = (
    "ABORT: stage-1 census.parquet has stopped growing (no new rows for "
    "--stale-minutes). The watcher will not poll forever -- an upstream job "
    "that died or hung would otherwise leave this driver waiting silently "
    "while the GPU sits idle. Check results/logs/census_stage1.log."
)


@dataclasses.dataclass
class Config:
    stage1_run_dir: str
    """Directory containing the stage-1 (32k/64k/96k) census.parquet."""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    pack_path: str = ""
    """Fitted spectral pack file, forwarded to the 128k census invocation."""
    expected_arms: tuple[str, ...] = ("fp16", "k2b", "k4_b2.5")
    expected_paths: tuple[str, ...] = ("dense_stream", "chunked")
    expected_pairs: tuple[tuple[str, str], ...] = ()
    """Optional explicit (arm, path) pairs to wait for, overriding the
    expected_arms x expected_paths cartesian product (minus the fp16 x
    chunked cell, which census.py never writes). Use this when the arms
    requested don't share the same set of paths."""
    gate_gib: float = 90.0
    """Projected resident_128k must be <= this many GiB to launch (95.6 GiB
    HBM ceiling minus the disclosed prefill-mask-transient margin)."""
    stale_minutes: float = 30.0
    """Abort if the stage-1 parquet's row count hasn't grown for this long."""
    poll_seconds: float = 30.0
    """Seconds between polls of the stage-1 parquet while waiting for cells."""
    log_dir: str = "results/logs"
    dry_run: bool = False


@dataclasses.dataclass
class GateResult:
    exit_code: int
    launched_cells: list[tuple[str, str]]
    skipped_cells: list[tuple[str, str]]


def project_128k(*, resident_64k: float, resident_96k: float) -> float:
    """Linear-growth projection, exactly as the plan words it:
    resident_128k ~= resident_96k + (resident_96k - resident_64k)."""
    return resident_96k + (resident_96k - resident_64k)


def _expected_pairs(cfg: Config) -> list[tuple[str, str]]:
    if cfg.expected_pairs:
        return list(cfg.expected_pairs)
    pairs = []
    for arm in cfg.expected_arms:
        for path in cfg.expected_paths:
            if arm == "fp16" and path == "chunked":
                continue  # census.py skips this cell by design (no fp16 passthrough)
            pairs.append((arm, path))
    return pairs


def _expected_cell_keys(cfg: Config) -> set[tuple[str, str, int]]:
    keys = set()
    for arm, path in _expected_pairs(cfg):
        for s in STAGE1_SEQ_LENS:
            keys.add((arm, path, s))
    return keys


def _present_cell_keys(df: pd.DataFrame) -> set[tuple[str, str, int]]:
    return set(zip(df["arm"], df["path"], df["seq_len"]))


def _wait_for_stage1(cfg: Config) -> pd.DataFrame:
    """Poll census.parquet until every expected cell is present. Aborts (loud
    message + SystemExit) if the row count stalls for stale_minutes."""
    parquet = Path(cfg.stage1_run_dir) / "census.parquet"
    expected = _expected_cell_keys(cfg)

    last_growth_t = time.monotonic()
    last_n_rows = -1
    while True:
        if parquet.exists():
            df = pd.read_parquet(parquet)
            present = _present_cell_keys(df)
            if expected <= present:
                return df
            n_rows = len(df)
        else:
            n_rows = 0

        now = time.monotonic()
        if n_rows > last_n_rows:
            last_n_rows = n_rows
            last_growth_t = now
        elif now - last_growth_t >= cfg.stale_minutes * 60:
            print(STALE_MESSAGE, file=sys.stderr)
            raise SystemExit(1)

        if cfg.poll_seconds > 0:
            time.sleep(cfg.poll_seconds)


def _cells_from_stage1(df: pd.DataFrame, cfg: Config) -> list[dict]:
    """One row per (arm, path) with the 64k/96k resident GiB + projection."""
    cells = []
    for arm, path in _expected_pairs(cfg):
        sub = df[(df["arm"] == arm) & (df["path"] == path)]
        r64_row = sub[sub["seq_len"] == 65536]
        r96_row = sub[sub["seq_len"] == 98304]
        if r64_row.empty or r96_row.empty:
            continue
        oom64 = bool(r64_row["oom"].iloc[0])
        oom96 = bool(r96_row["oom"].iloc[0])
        if oom64 or oom96:
            # Already OOM'd before 128k -- nothing to project; treat as a
            # hard skip (no launch), the OOM itself is the evidence.
            cells.append(
                {
                    "arm": arm,
                    "path": path,
                    "resident_64k_gib": None,
                    "resident_96k_gib": None,
                    "projected_128k_gib": None,
                    "oom_before_128k": True,
                }
            )
            continue
        r64 = float(r64_row["resident_after_prefill"].iloc[0]) / GiB
        r96 = float(r96_row["resident_after_prefill"].iloc[0]) / GiB
        proj = project_128k(resident_64k=r64, resident_96k=r96)
        cells.append(
            {
                "arm": arm,
                "path": path,
                "resident_64k_gib": r64,
                "resident_96k_gib": r96,
                "projected_128k_gib": proj,
                "oom_before_128k": False,
            }
        )
    return cells


def _census_argv(cfg: Config, arm: str) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "-m",
        "experiments.k3_kernel_census",
        "--model-name",
        cfg.model_name,
        "--arms",
        arm,
        "--seq-lens",
        str(GATE_128K_SEQ_LEN),
        "--pack-path",
        cfg.pack_path,
    ]


def _print_cell_arithmetic(cell: dict, gate_gib: float) -> None:
    arm, path = cell["arm"], cell["path"]
    if cell["oom_before_128k"]:
        print(f"[{arm:10s} {path:12s}] OOM already at <=96k -- SKIP (no 128k launch)")
        return
    r64, r96, proj = (
        cell["resident_64k_gib"],
        cell["resident_96k_gib"],
        cell["projected_128k_gib"],
    )
    verdict = "GATE (launch)" if proj <= gate_gib else "SKIP (over gate)"
    print(
        f"[{arm:10s} {path:12s}] projection: {r96:.2f} + ({r96:.2f} - {r64:.2f}) "
        f"= {proj:.2f} GiB vs gate {gate_gib:.1f} GiB -> {verdict}"
    )


def run_gate(cfg: Config) -> GateResult:
    df = _wait_for_stage1(cfg)
    cells = _cells_from_stage1(df, cfg)

    log_dir = Path(cfg.log_dir)
    if not cfg.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)

    launched_cells: list[tuple[str, str]] = []
    skipped_cells: list[tuple[str, str]] = []

    for cell in cells:
        _print_cell_arithmetic(cell, cfg.gate_gib)
        arm, path = cell["arm"], cell["path"]
        clears = (
            not cell["oom_before_128k"] and cell["projected_128k_gib"] <= cfg.gate_gib
        )
        if not clears:
            skipped_cells.append((arm, path))
            continue

        argv = _census_argv(cfg, arm)
        if cfg.dry_run:
            print(f"  DRY-RUN would launch: {' '.join(argv)}")
            continue

        log_path = log_dir / f"census_128k_{arm}_{path}.log"
        print(f"  launching (sequential, blocking): {' '.join(argv)}")
        print(f"  log -> {log_path}")
        with open(log_path, "w") as log_f:
            result = subprocess.run(argv, stdout=log_f, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(
                f"  WARNING: cell ({arm}, {path}) exited nonzero "
                f"({result.returncode}); see {log_path}"
            )
        launched_cells.append((arm, path))

    if cfg.dry_run:
        exit_code = 0
    else:
        exit_code = 0 if launched_cells else 1

    if not launched_cells and not cfg.dry_run:
        print(
            "ALL cells skipped by the headroom gate -- no 128k census launched. "
            "Exiting nonzero so a wrapping script can stop the batch."
        )

    return GateResult(
        exit_code=exit_code, launched_cells=launched_cells, skipped_cells=skipped_cells
    )


def main(cfg: Config) -> None:
    result = run_gate(cfg)
    sys.exit(result.exit_code)


if __name__ == "__main__":
    main(tyro.cli(Config))
