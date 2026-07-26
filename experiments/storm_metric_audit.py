"""Storm Task-1 — per-event metric audit (branch C; GATES THE PAPER TABLES).

Question (plan `docs/superpowers/plans/2026-07-26-storm-gates.md` Task 1): does
ranking codecs by per-event task error agree with ranking by the logit-MSE
instrument (the K4 frontier)? Banked parquets ONLY — no model, no download.

Sides
-----
* TASK side: `results/k3_niah/*/metrics.parquet` (mean needle recall per
  arm×length×depth — NIAH has NO per-sample rows) and
  `results/k3_longbench/*/{metrics,samples}.parquet` (macro + per-category means
  and, where a run banked them, per-sample 0-1 edit-similarity / retrieval scores
  → the per-event win/tie/loss view vs fp16).
* LOGIT side: `results/k4_frontier/*/metrics.parquet` — the query-weighted
  per-layer `logit`/`logit_rope` distortion of the K-tensor codecs (`spectral`
  = the K4 family; `turboquant_mse` = TQ). We reduce it to one scalar per codec
  by the mean over layers (and, for the Qwen case study, tail quantiles).

Arm mapping (task-arm → frontier K-codec, via `bmx.cache.recipes`)
-----------------------------------------------------------------
* `k4_b2.5`            → spectral @ budget 2.5    (K = spectral; V = tq@2b)
* `k4_b2.2`            → spectral @ budget 2.2
* `turboquant_mse_b3`  → turboquant_mse @ bits 3  (K = tq@3b; V = tq@3b)
* `turboquant_mse_k3v2`→ turboquant_mse @ bits 3  (K = tq@3b; V = tq@2b)

Note the structural blind-spot the audit surfaces: `b3` and `k3v2` share the
SAME K-side codec, so the K-logit instrument scores them IDENTICALLY — yet the
task metric ranks them differently (the Qwen collapse). Pairs are ranked only
where BOTH a logit-side and a task-side measurement exist; distinct task arms
that collapse onto one frontier point are recorded as a tie on the logit axis.

Pre-registered gate (verbatim from the plan)
--------------------------------------------
Spearman >= 0.8 on Llama across the duel arms ⇒ the logit instrument is a safe
proxy — keep the shipped metric. Spearman < 0.8 anywhere, or a sign inversion on
any arm pair ⇒ FLAG: the paper's quality tables must carry the per-event column
alongside macro.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.storm_audit import GATE_THRESHOLD, run_audit


@dataclasses.dataclass
class Config:
    # Result roots (banked parquets only; no model, no download).
    niah_root: str = "results/k3_niah"
    longbench_root: str = "results/k3_longbench"
    frontier_root: str = "results/k4_frontier"
    # Run-id substring to EXCLUDE (documented non-duel provenance).
    exclude_run_substr: str = "f9eeafe"
    # Which frontier fit_mode to treat as the shipped/deployment instrument for the
    # spectral arms (heldout = the generalization fit the packs are scored under). The
    # audit records corpus/oracle as robustness columns regardless.
    spectral_fit_mode: str = "heldout"
    # Pre-registered Spearman gate threshold (do not change; plan-locked).
    gate_threshold: float = GATE_THRESHOLD


def run(cfg: Config, root: str = "results") -> Path:
    run_dir = create_run("storm_metric_audit", cfg, root=root)
    verdict, metrics_df = run_audit(
        niah_root=cfg.niah_root,
        longbench_root=cfg.longbench_root,
        frontier_root=cfg.frontier_root,
        exclude_run_substr=cfg.exclude_run_substr,
        spectral_fit_mode=cfg.spectral_fit_mode,
        gate_threshold=cfg.gate_threshold,
    )
    write_metrics(run_dir, metrics_df)
    (run_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
    _print_verdict(verdict)
    return run_dir


def _print_verdict(v: dict) -> None:
    line = "=" * 72
    print(line)
    print("STORM TASK-1 — PER-EVENT METRIC AUDIT — VERDICT")
    print(line)
    for note in v["data_availability"]:
        print(f"  [data] {note}")
    print("-" * 72)
    print("  Provenance:")
    for note in v["provenance"]:
        print(f"    {note}")
    print("-" * 72)
    print("  Spearman (logit-instrument ranking vs task ranking):")
    for row in v["spearman"]:
        flag = " <-- SIGN INVERSION" if row["sign_inversion"] else ""
        below = " <-- BELOW 0.8" if row["below_threshold"] else ""
        rho = "n/a" if row["rho"] is None else f"{row['rho']:+.3f}"
        print(
            f"    {row['model']:<8} {row['task_axis']:<28} "
            f"rho={rho:>7}  n_pairs={row['n_pairs']}{flag}{below}"
        )
    print("-" * 72)
    print("  Per-event (LongBench win-rate vs fp16 — ranking agreement vs mean):")
    for row in v["per_event"]:
        agree = "AGREE" if row["rank_agrees_with_mean"] else "DISAGREE"
        print(
            f"    {row['model']:<8} {row['task']:<22} "
            f"mean-rank vs winrate-rank: {agree} (tau={row['kendall_tau']})"
        )
    print("-" * 72)
    print("  Qwen TQ-collapse case study:")
    for note in v["qwen_case_study"]["notes"]:
        print(f"    {note}")
    print("-" * 72)
    g = v["gate"]
    print(f"  PRE-REGISTERED GATE: {g['outcome']}")
    print(f"    {g['reason']}")
    print(line)


if __name__ == "__main__":
    run(tyro.cli(Config))
