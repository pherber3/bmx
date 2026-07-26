"""Storm Task-1 per-event metric audit — offline unit tests on tiny synthetic parquets.

No downloads, no model. Fixtures write minimal config.json + metrics.parquet (+
samples.parquet where needed) into a tmp results tree, exercising: the model-split,
the f9eeafe exclusion, newest-run dedup, the Spearman computation/alignment, the
per-event win-rate, and the pre-registered gate logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bmx.storm_audit import (
    GATE_THRESHOLD,
    _evaluate_gate,
    _spearman,
    frontier_scalar,
    load_frontier,
    load_longbench,
    load_niah,
    per_event_winrate,
    run_audit,
)


# --- fixture builders --------------------------------------------------------


def _write_run(
    root: Path,
    exp: str,
    run_id: str,
    model_name: str,
    metrics: pd.DataFrame,
    samples=None,
):
    d = root / exp / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_name": model_name}))
    metrics.to_parquet(d / "metrics.parquet", index=False)
    if samples is not None:
        samples.to_parquet(d / "samples.parquet", index=False)
    return d


def _niah_rows(arm_recalls: dict, length=32768, depth=0.5):
    """arm -> recall_full at one (length, depth)."""
    return pd.DataFrame(
        [
            {"arm": a, "length": length, "depth": depth, "recall": r, "recall_full": r}
            for a, r in arm_recalls.items()
        ]
    )


def _frontier_rows(model: str, n_layers=4):
    """Minimal frontier metrics: spectral (budget 2.2/2.5, heldout) + tq (bits 3),
    kind=k_pre, per layer. spectral has lower logit than tq (spectral is 'better')."""
    rows = []
    for layer in range(n_layers):
        for budget, base in ((2.5, 0.02), (2.2, 0.03)):
            rows.append(
                {
                    "model": model,
                    "layer": layer,
                    "kind": "k_pre",
                    "arm": "spectral",
                    "fit_mode": "heldout",
                    "weighted": True,
                    "budget": budget,
                    "bits": -1,
                    "logit": base + 0.001 * layer,
                    "logit_rope": base + 0.001 * layer,
                }
            )
        rows.append(
            {
                "model": model,
                "layer": layer,
                "kind": "k_pre",
                "arm": "turboquant_mse",
                "fit_mode": "baseline",
                "weighted": False,
                "budget": np.nan,
                "bits": 3,
                "logit": 0.15 + 0.001 * layer,
                "logit_rope": 0.15 + 0.001 * layer,
            }
        )
    return pd.DataFrame(rows)


# --- model split + exclusion -------------------------------------------------


def test_model_split_by_config_model_name(tmp_path):
    root = tmp_path
    _write_run(
        root,
        "k3_niah",
        "20260701-000000-aaaaaaa",
        "meta-llama/Llama-3.1-8B-Instruct",
        _niah_rows({"k4_b2.5": 8.0, "turboquant_mse_b3": 7.0}),
    )
    _write_run(
        root,
        "k3_niah",
        "20260701-000001-bbbbbbb",
        "Qwen/Qwen3-8B",
        _niah_rows({"k4_b2.5": 10.0, "turboquant_mse_b3": 2.5}),
    )
    df, _ = load_niah(str(root / "k3_niah"), "f9eeafe")
    assert set(df["model"]) == {"Llama", "Qwen"}
    # arm names are shared; the split keeps them distinct per model.
    llama = df[df["model"] == "Llama"]
    qwen = df[df["model"] == "Qwen"]
    assert llama[llama["arm"] == "turboquant_mse_b3"]["recall_full"].iloc[0] == 7.0
    assert qwen[qwen["arm"] == "turboquant_mse_b3"]["recall_full"].iloc[0] == 2.5


def test_f9eeafe_run_is_excluded(tmp_path):
    root = tmp_path
    _write_run(
        root,
        "k3_niah",
        "20260701-000000-aaaaaaa",
        "Qwen/Qwen3-8B",
        _niah_rows({"k4_b2.5": 10.0}),
    )
    # A newer run whose SHA is f9eeafe MUST be excluded even though it is newest.
    _write_run(
        root,
        "k3_niah",
        "20260715-132205-f9eeafe",
        "Qwen/Qwen3-8B",
        _niah_rows({"k4_b2.5": 3.33}),
    )
    df, notes = load_niah(str(root / "k3_niah"), "f9eeafe")
    assert len(df) == 1
    assert df["recall_full"].iloc[0] == 10.0  # the excluded newer value did NOT win
    assert any("f9eeafe" in n for n in notes)


def test_exclusion_note_records_the_run(tmp_path):
    root = tmp_path
    _write_run(
        root,
        "k3_longbench",
        "20260715-122908-f9eeafe",
        "Qwen/Qwen3-8B",
        pd.DataFrame([{"arm": "fp16", "task": "lcc", "code_sim": 0.6}]),
    )
    _, _, notes = load_longbench(str(root / "k3_longbench"), "f9eeafe")
    assert any("f9eeafe" in n for n in notes)


# --- dedup by newest ---------------------------------------------------------


def test_dedup_keeps_newest_run_per_cell(tmp_path):
    root = tmp_path
    _write_run(
        root,
        "k3_niah",
        "20260701-000000-old",
        "Llama-3.1-8B",
        _niah_rows({"k4_b2.5": 5.0}),
    )
    _write_run(
        root,
        "k3_niah",
        "20260705-000000-new",
        "Llama-3.1-8B",
        _niah_rows({"k4_b2.5": 9.0}),
    )
    df, notes = load_niah(str(root / "k3_niah"), "f9eeafe")
    assert len(df) == 1
    assert df["recall_full"].iloc[0] == 9.0  # newest wins
    assert any("dedup" in n for n in notes)


def test_longbench_samples_dedup_by_newest(tmp_path):
    root = tmp_path
    old = pd.DataFrame([{"arm": "fp16", "task": "lcc", "sample_idx": 0, "score": 0.1}])
    new = pd.DataFrame([{"arm": "fp16", "task": "lcc", "sample_idx": 0, "score": 0.9}])
    _write_run(
        root,
        "k3_longbench",
        "20260701-000000-old",
        "Llama-3.1-8B",
        pd.DataFrame([{"arm": "fp16", "task": "lcc", "code_sim": 0.1}]),
        samples=old,
    )
    _write_run(
        root,
        "k3_longbench",
        "20260705-000000-new",
        "Llama-3.1-8B",
        pd.DataFrame([{"arm": "fp16", "task": "lcc", "code_sim": 0.9}]),
        samples=new,
    )
    _, samples, _ = load_longbench(str(root / "k3_longbench"), "f9eeafe")
    assert len(samples) == 1
    assert samples["score"].iloc[0] == 0.9


# --- Spearman computation ----------------------------------------------------


def test_spearman_perfect_agreement_is_positive_one():
    # lower distortion == higher task metric  -> perfect agreement -> rho = +1.
    logit = {"a": 0.01, "b": 0.05, "c": 0.10}
    task = {"a": 9.0, "b": 5.0, "c": 1.0}  # higher is better
    rho, n, arms = _spearman(logit, task, task_higher_is_better=True)
    assert n == 3
    assert rho == pytest.approx(1.0)


def test_spearman_sign_inversion_is_negative():
    # WORST-distortion codec gets the BEST task score -> inversion -> negative rho.
    logit = {"a": 0.01, "b": 0.05, "c": 0.10}
    task = {"a": 1.0, "b": 5.0, "c": 9.0}
    rho, n, arms = _spearman(logit, task, task_higher_is_better=True)
    assert rho == pytest.approx(-1.0)
    assert rho < 0  # sign inversion flagged downstream


def test_spearman_undefined_when_one_axis_is_flat():
    # b3/k3v2 share one logit point (tie) + everything else equal -> no variance.
    logit = {"a": 0.05, "b": 0.05}
    task = {"a": 3.0, "b": 7.0}
    rho, n, arms = _spearman(logit, task, task_higher_is_better=True)
    assert rho is None  # can't rank on a degenerate axis


def test_spearman_needs_two_pairs():
    rho, n, arms = _spearman({"a": 0.1}, {"a": 5.0}, task_higher_is_better=True)
    assert rho is None
    assert n == 1


# --- frontier scalar ---------------------------------------------------------


def test_frontier_scalar_excludes_layer0_and_averages(tmp_path):
    root = tmp_path
    _write_run(
        root,
        "k4_frontier",
        "20260701-000000-a",
        "llama-3.1-8b-instruct",
        _frontier_rows("llama-3.1-8b-instruct", n_layers=4),
    )
    fr, _ = load_frontier(str(root / "k4_frontier"), "f9eeafe", "heldout")
    # spectral b2.5, layers 1..3 logit = 0.021,0.022,0.023 -> mean 0.022 (layer0=0.020 dropped)
    sc = frontier_scalar(
        fr,
        "llama-3.1-8b-instruct",
        "spectral",
        "budget",
        2.5,
        spectral_fit_mode="heldout",
    )
    assert sc == pytest.approx(0.022, abs=1e-9)
    # tail statistic strictly >= mean.
    p99 = frontier_scalar(
        fr,
        "llama-3.1-8b-instruct",
        "spectral",
        "budget",
        2.5,
        spectral_fit_mode="heldout",
        reducer="p99",
    )
    assert p99 >= sc


def test_frontier_b3_and_k3v2_map_to_identical_logit(tmp_path):
    # The structural blind-spot: b3 & k3v2 differ only in V budget; the K instrument
    # gives them the SAME distortion.
    root = tmp_path
    _write_run(
        root,
        "k4_frontier",
        "20260701-000000-a",
        "qwen3-8b",
        _frontier_rows("qwen3-8b", n_layers=4),
    )
    fr, _ = load_frontier(str(root / "k4_frontier"), "f9eeafe", "heldout")
    s_b3 = frontier_scalar(
        fr, "qwen3-8b", "turboquant_mse", "bits", 3, spectral_fit_mode="heldout"
    )
    # both arms route to the same (turboquant_mse, bits=3) frontier point by ARM_TO_FRONTIER
    assert s_b3 is not None and s_b3 > 0


# --- per-event win/tie/loss --------------------------------------------------


def test_per_event_winrate_vs_fp16():
    # 4 samples: arm beats fp16 twice, ties once, loses once.
    samples = pd.DataFrame(
        [
            {"model": "Qwen", "task": "lcc", "arm": "fp16", "sample_idx": i, "score": s}
            for i, s in enumerate([0.5, 0.5, 0.5, 0.9])
        ]
        + [
            {"model": "Qwen", "task": "lcc", "arm": "tq", "sample_idx": i, "score": s}
            for i, s in enumerate([0.9, 0.9, 0.5, 0.1])
        ]
    )
    pe = per_event_winrate(samples, "Qwen", "lcc")
    row = pe[pe["arm"] == "tq"].iloc[0]
    assert row["win"] == 2 and row["tie"] == 1 and row["loss"] == 1
    assert row["win_rate"] == pytest.approx(0.5)


def test_per_event_total_failure_events():
    # arm scores 0 where fp16 got full credit -> a total-failure event (the collapse
    # signature: event-concentrated, not uniform degradation).
    samples = pd.DataFrame(
        [
            {
                "model": "Qwen",
                "task": "passage_retrieval_en",
                "arm": "fp16",
                "sample_idx": i,
                "score": 1.0,
            }
            for i in range(4)
        ]
        + [
            {
                "model": "Qwen",
                "task": "passage_retrieval_en",
                "arm": "tq",
                "sample_idx": i,
                "score": s,
            }
            for i, s in enumerate([0.0, 0.0, 0.0, 1.0])
        ]
    )
    pe = per_event_winrate(samples, "Qwen", "passage_retrieval_en")
    row = pe.iloc[0]
    assert row["total_fail_rate"] == pytest.approx(0.75)


def test_per_event_none_without_fp16():
    samples = pd.DataFrame(
        [{"model": "Qwen", "task": "lcc", "arm": "tq", "sample_idx": 0, "score": 0.5}]
    )
    assert per_event_winrate(samples, "Qwen", "lcc") is None


# --- gate logic --------------------------------------------------------------


def _srow(model, axis, rho):
    below = rho is not None and rho < GATE_THRESHOLD
    inv = rho is not None and rho < 0
    return {
        "model": model,
        "task_axis": axis,
        "rho": None if rho is None else round(rho, 4),
        "n_pairs": 4 if rho is not None else 0,
        "arms": [],
        "below_threshold": below,
        "sign_inversion": inv,
    }


def test_gate_safe_proxy_when_all_llama_pass():
    rows = [_srow("Llama", "LB macro", 0.9), _srow("Llama", "LB code", 0.85)]
    g = _evaluate_gate(rows, GATE_THRESHOLD)
    assert g["outcome"] == "SAFE_PROXY"


def test_gate_flag_on_below_threshold():
    rows = [_srow("Llama", "LB macro", 0.9), _srow("Llama", "LB code", 0.5)]
    g = _evaluate_gate(rows, GATE_THRESHOLD)
    assert g["outcome"] == "FLAG"
    assert g["any_below_threshold"] is True


def test_gate_flag_on_sign_inversion():
    rows = [_srow("Llama", "LB macro", 0.95), _srow("Llama", "NIAH recall@32768", -0.3)]
    g = _evaluate_gate(rows, GATE_THRESHOLD)
    assert g["outcome"] == "FLAG"
    assert g["any_sign_inversion"] is True


def test_gate_flag_when_no_llama_axis_defined():
    # Every Llama axis degenerate (rho None): cannot certify a proxy -> conservative FLAG.
    rows = [_srow("Llama", "LB macro", None), _srow("Qwen", "LB macro", 0.9)]
    g = _evaluate_gate(rows, GATE_THRESHOLD)
    assert g["outcome"] == "FLAG"
    assert g["n_llama_defined_axes"] == 0


# --- end-to-end smoke on a tiny synthetic tree -------------------------------


def test_run_audit_end_to_end(tmp_path):
    root = tmp_path
    # NIAH: Llama parity-ish, Qwen collapse.
    _write_run(
        root,
        "k3_niah",
        "20260701-000000-a",
        "meta-llama/Llama-3.1-8B-Instruct",
        _niah_rows(
            {
                "k4_b2.5": 8.0,
                "k4_b2.2": 7.8,
                "turboquant_mse_b3": 7.5,
                "turboquant_mse_k3v2": 7.6,
            }
        ),
    )
    _write_run(
        root,
        "k3_niah",
        "20260701-000001-b",
        "Qwen/Qwen3-8B",
        _niah_rows(
            {"k4_b2.5": 10.0, "turboquant_mse_b3": 2.9, "turboquant_mse_k3v2": 5.5}
        ),
    )
    # LongBench metrics + samples (fp16 + duel arms).
    lb_metrics = pd.DataFrame(
        [
            {"arm": a, "task": "lcc", "code_sim": s}
            for a, s in {
                "fp16": 0.7,
                "k4_b2.5": 0.68,
                "turboquant_mse_b3": 0.5,
                "turboquant_mse_k3v2": 0.55,
            }.items()
        ]
    )
    lb_samples = pd.DataFrame(
        [
            {"arm": a, "task": "lcc", "sample_idx": i, "score": v}
            for a, scores in {
                "fp16": [0.7, 0.7, 0.7, 0.7],
                "turboquant_mse_b3": [0.0, 0.0, 0.9, 0.9],
                "turboquant_mse_k3v2": [0.5, 0.5, 0.6, 0.6],
            }.items()
            for i, v in enumerate(scores)
        ]
    )
    _write_run(
        root,
        "k3_longbench",
        "20260701-000002-c",
        "Qwen/Qwen3-8B",
        lb_metrics,
        samples=lb_samples,
    )
    # Frontier for both models.
    _write_run(
        root,
        "k4_frontier",
        "20260701-000003-d",
        "llama-3.1-8b-instruct",
        _frontier_rows("llama-3.1-8b-instruct"),
    )
    _write_run(
        root, "k4_frontier", "20260701-000004-e", "qwen3-8b", _frontier_rows("qwen3-8b")
    )

    verdict, metrics_df = run_audit(
        niah_root=str(root / "k3_niah"),
        longbench_root=str(root / "k3_longbench"),
        frontier_root=str(root / "k4_frontier"),
        exclude_run_substr="f9eeafe",
        spectral_fit_mode="heldout",
        gate_threshold=GATE_THRESHOLD,
    )
    assert verdict["gate"]["outcome"] in ("SAFE_PROXY", "FLAG")
    assert not metrics_df.empty
    # per-event rows exist for the Qwen lcc samples.
    pe_kinds = metrics_df[metrics_df["metric_kind"] == "per_event_win_rate"]
    assert not pe_kinds.empty
    # data-availability note states NIAH lacks per-sample rows.
    assert any(
        "per-sample" in n.lower() or "no per-sample" in n.lower()
        for n in verdict["data_availability"]
    )
    # Qwen case study computed the distortion tail separation.
    assert "distortion_tail" in verdict["qwen_case_study"]


def test_run_audit_gate_matches_preregistered_threshold(tmp_path):
    # Construct data where Llama task ranking perfectly agrees with the logit ranking
    # across all four duel arms (spectral < tq on both axes) -> SAFE_PROXY expected.
    root = tmp_path
    # Task recall MONOTONE with logit goodness: k4_b2.5 > k4_b2.2 > tq (both tq equal-ish).
    _write_run(
        root,
        "k3_niah",
        "20260701-000000-a",
        "meta-llama/Llama-3.1-8B-Instruct",
        _niah_rows(
            {
                "k4_b2.5": 9.0,
                "k4_b2.2": 8.0,
                "turboquant_mse_b3": 5.0,
                "turboquant_mse_k3v2": 5.5,
            }
        ),
    )
    _write_run(
        root,
        "k4_frontier",
        "20260701-000003-d",
        "llama-3.1-8b-instruct",
        _frontier_rows("llama-3.1-8b-instruct"),
    )
    verdict, _ = run_audit(
        niah_root=str(root / "k3_niah"),
        longbench_root=str(root / "k3_longbench"),  # missing -> empty, fine
        frontier_root=str(root / "k4_frontier"),
        exclude_run_substr="f9eeafe",
        spectral_fit_mode="heldout",
        gate_threshold=GATE_THRESHOLD,
    )
    # The single defined Llama NIAH axis should clear the gate (rho high, no inversion).
    llama_niah = [
        r for r in verdict["spearman"] if r["model"] == "Llama" and r["rho"] is not None
    ]
    assert llama_niah, "expected at least one defined Llama axis"
    assert all(r["rho"] >= GATE_THRESHOLD for r in llama_niah)
    assert verdict["gate"]["outcome"] == "SAFE_PROXY"
