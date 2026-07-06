"""Offline tests for the Pareto figure generator (experiments/plot_pareto.py).

Builds small synthetic metrics.parquet-shaped frames (the k3_longbench row schema) and drives
`build_pareto` directly. No matplotlib rendering assertions beyond file-exists (per the
build_pareto/make_figure separation) — the pure aggregation logic is what's under test here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from experiments.plot_longbench_table import PUBLISHED
from experiments.plot_pareto import ParetoPoint, build_pareto, make_figure


def _row(arm, task, score, *, kv_size_bits=3.0):
    return {
        "arm": arm,
        "task": task,
        "code_sim": score,
        "kv_size_bits": kv_size_bits,
    }


def _synthetic_rows():
    """fp16 + two compressed arms over 4 tasks (equal weight, no partial categories needed)."""
    rows = []
    for task, score in [
        ("narrativeqa", 0.40),
        ("qasper", 0.42),
        ("multifieldqa_en", 0.44),
        ("lcc", 0.60),
    ]:
        rows.append(_row("fp16", task, score, kv_size_bits=16.0))
    for task, score in [
        ("narrativeqa", 0.38),
        ("qasper", 0.40),
        ("multifieldqa_en", 0.41),
        ("lcc", 0.50),
    ]:
        rows.append(_row("turboquant_mse", task, score, kv_size_bits=2.5))
    for task, score in [
        ("narrativeqa", 0.39),
        ("qasper", 0.41),
        ("multifieldqa_en", 0.43),
        ("lcc", 0.55),
    ]:
        rows.append(_row("k2b", task, score, kv_size_bits=4.0))
    return rows


def test_missing_fp16_raises():
    df = pd.DataFrame([_row("turboquant_mse", "narrativeqa", 0.4, kv_size_bits=2.5)])
    with pytest.raises(ValueError, match="fp16"):
        build_pareto(df)


def test_delta_computed_correctly_vs_hand_built_frame():
    df = pd.DataFrame(_synthetic_rows())
    result = build_pareto(df)
    fp16_mean = (0.40 + 0.42 + 0.44 + 0.60) / 4 * 100.0
    tq_mean = (0.38 + 0.40 + 0.41 + 0.50) / 4 * 100.0

    fp16_pt = next(p for p in result["ours"] if p.label == "fp16")
    tq_pt = next(p for p in result["ours"] if p.label == "turboquant_mse")
    assert fp16_pt.y_delta == pytest.approx(0.0, abs=1e-9)
    assert fp16_pt.x_bits == pytest.approx(16.0)
    assert tq_pt.y_delta == pytest.approx(tq_mean - fp16_mean)
    assert tq_pt.x_bits == pytest.approx(2.5)


def test_theirs_delta_matches_published_full_cache_anchor():
    df = pd.DataFrame(_synthetic_rows())
    result = build_pareto(df)
    full_cache = next(
        p for p in result["theirs"] if p.label == "Full Cache (published)"
    )
    assert full_cache.y_delta == pytest.approx(0.0)
    kivi3 = next(p for p in result["theirs"] if p.label == "KIVI (KV 3) (published)")
    expected = (
        PUBLISHED["KIVI (KV 3) (published)"]["Avg"]
        - PUBLISHED["Full Cache (published)"]["Avg"]
    )
    assert kivi3.y_delta == pytest.approx(expected)
    assert kivi3.x_bits == pytest.approx(3.0)


def test_kivi_arm_excluded_from_ours():
    rows = _synthetic_rows()
    rows.append(_row("kivi", "narrativeqa", 0.01, kv_size_bits=2.0))
    df = pd.DataFrame(rows)
    result = build_pareto(df)
    labels = {p.label for p in result["ours"]}
    assert "kivi" not in labels


def test_frontier_is_correct_upper_envelope_on_three_point_synthetic():
    # x=8 -> y=2 (fewest bits AND best quality: dominates everything). x=10 -> y=1 is
    # dominated by b (b has both lower bits and higher y). x=14 -> y=2.5 is NOT dominated
    # (worse bits than b, but strictly better y) so it extends the frontier.
    from experiments.plot_pareto import _upper_envelope

    points = [
        ParetoPoint("a", 10.0, 1.0, ours=True),
        ParetoPoint("b", 8.0, 2.0, ours=True),
        ParetoPoint("c", 14.0, 2.5, ours=True),
    ]
    frontier = _upper_envelope(points)
    labels = [p.label for p in frontier]
    assert labels == [
        "b",
        "c",
    ]  # sorted by x ascending; a dominated (higher x, lower y)


def test_frontier_excludes_strictly_dominated_point():
    from experiments.plot_pareto import _upper_envelope

    # d at x=8,y=1 is strictly dominated by b at x=8,y=2 (same x, worse y) — should drop.
    # a at x=10,y=1 is also dominated by b (lower x AND higher y) — only b survives.
    points = [
        ParetoPoint("b", 8.0, 2.0, ours=True),
        ParetoPoint("d", 8.0, 1.0, ours=True),
        ParetoPoint("a", 10.0, 1.0, ours=True),
    ]
    frontier = _upper_envelope(points)
    labels = [p.label for p in frontier]
    assert "d" not in labels
    assert labels == ["b"]


def test_ours_abs_uses_task_level_mean_not_category_mean():
    df = pd.DataFrame(_synthetic_rows())
    result = build_pareto(df)
    fp16_abs_pt = next(p for p in result["ours_abs"] if p.label == "fp16")
    expected = (0.40 + 0.42 + 0.44 + 0.60) / 4 * 100.0
    assert fp16_abs_pt.y_delta == pytest.approx(expected)
    assert result["fp16_abs"] == pytest.approx(expected)


def test_kv_size_bits_is_mean_across_tasks_for_measured_arms():
    rows = [
        _row("fp16", "narrativeqa", 0.4, kv_size_bits=16.0),
        _row("turboquant_mse", "narrativeqa", 0.4, kv_size_bits=2.0),
        _row("turboquant_mse", "qasper", 0.4, kv_size_bits=3.0),
    ]
    df = pd.DataFrame(rows)
    result = build_pareto(df)
    tq_pt = next(p for p in result["ours"] if p.label == "turboquant_mse")
    assert tq_pt.x_bits == pytest.approx(2.5)


def test_make_figure_writes_png_pdf_and_caption(tmp_path):
    df = pd.DataFrame(_synthetic_rows())
    result = build_pareto(df)
    png, pdf, caption_path = make_figure(result, run_id="testrun-123", out_dir=tmp_path)
    assert png.exists()
    assert pdf.exists()
    assert caption_path.exists()
    text = caption_path.read_text()
    assert "testrun-123" in text
    assert "delta" in text.lower()


def test_caption_cites_bitwidth_caveat_and_kivi_diagnosis(tmp_path):
    df = pd.DataFrame(_synthetic_rows())
    result = build_pareto(df)
    _, _, caption_path = make_figure(result, run_id="run1", out_dir=tmp_path)
    text = caption_path.read_text()
    assert "2.5" in text
    assert "kivi-arm-diagnosis" in text
