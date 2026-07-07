"""Offline tests for the LongBench Table-1 generator (experiments/plot_longbench_table.py).

Builds a small synthetic metrics.parquet (the k3_longbench row schema) in tmp_path and
drives `build_table` directly — no model, no downloads, no real run dir required.
"""

from __future__ import annotations

import pandas as pd
import pytest

from experiments.plot_longbench_table import PUBLISHED, TableResult, build_table


def _row(
    arm,
    task,
    score,
    *,
    kv_size_bits=3.0,
    n_samples=10,
    max_prompt_tokens=31500,
    longbench_version="v1_e",
    metric="qa_f1_score",
):
    return {
        "arm": arm,
        "task": task,
        "code_sim": score,
        "metric": metric,
        "n_samples": n_samples,
        "bpe_k": kv_size_bits,
        "bpe_v": kv_size_bits,
        "kv_size_bits": kv_size_bits,
        "compression": 16.0 / kv_size_bits,
        "n_prefill": 128,
        "score_kind": "code_sim",
        "max_prompt_tokens": max_prompt_tokens,
        "longbench_version": longbench_version,
    }


def _synthetic_rows(**overrides):
    """fp16 + turboquant_mse arms, single_qa (3 tasks) full + code (1 of 2 tasks: partial)."""
    rows = []
    # fp16 anchor arm, kv_size_bits=16 (uncompressed).
    for task, score in [
        ("narrativeqa", 0.40),
        ("qasper", 0.42),
        ("multifieldqa_en", 0.44),
        ("lcc", 0.60),
    ]:
        rows.append(_row("fp16", task, score, kv_size_bits=16.0, **overrides))
    # turboquant_mse arm, same tasks, partial code category (only lcc, missing repobench-p).
    for task, score in [
        ("narrativeqa", 0.38),
        ("qasper", 0.40),
        ("multifieldqa_en", 0.41),
        ("lcc", 0.50),
    ]:
        rows.append(_row("turboquant_mse", task, score, kv_size_bits=2.5, **overrides))
    return rows


def test_category_means_correct(tmp_path):
    df = pd.DataFrame(_synthetic_rows())
    result = build_table(df, run_id="20260704-000000-abcdef", anchor_tolerance=2.0)
    fp16_row = result.rows_by_method["fp16 (measured)"]
    assert fp16_row["SingleQA"] == pytest.approx((0.40 + 0.42 + 0.44) / 3 * 100)


def test_missing_category_marked_with_asterisk_and_footnote():
    df = pd.DataFrame(_synthetic_rows())
    result = build_table(df, run_id="run1", anchor_tolerance=2.0)
    tq_row = result.rows_by_method["turboquant_mse (measured)"]
    # code category only has lcc present (repobench-p missing) -> asterisked, not silently averaged.
    assert isinstance(tq_row["Code"], str) and tq_row["Code"].endswith("*")
    assert any("repobench-p" in note for note in result.footnotes)


def test_full_category_not_marked():
    df = pd.DataFrame(_synthetic_rows())
    result = build_table(df, run_id="run1", anchor_tolerance=2.0)
    fp16_row = result.rows_by_method["fp16 (measured)"]
    assert isinstance(fp16_row["SingleQA"], float)


def test_kivi_excluded_with_note():
    rows = _synthetic_rows()
    rows.append(_row("kivi", "narrativeqa", 0.01))
    df = pd.DataFrame(rows)
    result = build_table(df, run_id="run1", anchor_tolerance=2.0)
    assert "kivi (measured)" not in result.rows_by_method
    assert any("kivi" in note.lower() for note in result.footnotes)
    assert any("2026-07-04-kivi-arm-diagnosis" in note for note in result.footnotes)


def test_rtn_style_rows_kept_and_labeled_rtn2():
    rows = _synthetic_rows()
    rows.append(_row("rtn_channel", "narrativeqa", 0.30))
    df = pd.DataFrame(rows)
    result = build_table(df, run_id="run1", anchor_tolerance=2.0)
    assert "rtn_channel (measured, rtn2)" in result.rows_by_method


def test_published_transitive_rows_present():
    df = pd.DataFrame(_synthetic_rows())
    result = build_table(df, run_id="run1", anchor_tolerance=2.0)
    for method in PUBLISHED:
        assert method in result.rows_by_method
        # "Average (all tasks)" carries the paper's own published Avg verbatim (task-level
        # scores aren't published); "Average (categories)" is our recomputed mean-of-6, which
        # deliberately does NOT match (see test_average_reports_both_aggregations...).
        assert result.rows_by_method[method]["Average (all tasks)"] == pytest.approx(
            PUBLISHED[method]["Avg"]
        )


def test_average_reports_both_aggregations_and_they_can_differ():
    df = pd.DataFrame(_synthetic_rows())
    result = build_table(df, run_id="run1", anchor_tolerance=2.0)
    fp16_row = result.rows_by_method["fp16 (measured)"]
    assert "Average (categories)" in fp16_row
    assert "Average (all tasks)" in fp16_row
    # Full Cache published anchor: category-mean avg (48.53) != published avg (50.06).
    full_cache = result.rows_by_method["Full Cache (published)"]
    cat_avg = (
        sum(
            PUBLISHED["Full Cache (published)"][c]
            for c in (
                "SingleQA",
                "MultiQA",
                "Summarization",
                "Few shot",
                "Synthetic",
                "Code",
            )
        )
        / 6
    )
    assert cat_avg != pytest.approx(PUBLISHED["Full Cache (published)"]["Avg"])
    assert full_cache["Average (categories)"] == pytest.approx(cat_avg)


def test_anchor_delta_pass_at_tolerance_boundary():
    # Construct fp16 rows whose SingleQA category mean is exactly published + 2.0 (tolerance).
    published_single_qa = PUBLISHED["Full Cache (published)"]["SingleQA"]
    target = (
        published_single_qa + 2.0
    ) / 100.0  # score fraction, category has 1 task here
    df = pd.DataFrame(
        [
            _row("fp16", "narrativeqa", target, kv_size_bits=16.0),
            _row("turboquant_mse", "narrativeqa", target, kv_size_bits=2.5),
        ]
    )
    result = build_table(df, run_id="run1", anchor_tolerance=2.0)
    verdict = result.anchor_deltas["fp16 (measured)"]["SingleQA"]
    assert verdict["delta"] == pytest.approx(2.0, abs=1e-6)
    assert verdict["verdict"] == "PASS"


def test_anchor_delta_fail_just_past_tolerance_boundary():
    published_single_qa = PUBLISHED["Full Cache (published)"]["SingleQA"]
    target = (published_single_qa + 2.01) / 100.0
    df = pd.DataFrame(
        [
            _row("fp16", "narrativeqa", target, kv_size_bits=16.0),
            _row("turboquant_mse", "narrativeqa", target, kv_size_bits=2.5),
        ]
    )
    result = build_table(df, run_id="run1", anchor_tolerance=2.0)
    verdict = result.anchor_deltas["fp16 (measured)"]["SingleQA"]
    assert verdict["verdict"] == "FAIL"


def test_anchor_delta_uses_turboquant_2_5_row_for_turboquant_mse_arm():
    df = pd.DataFrame(_synthetic_rows())
    result = build_table(df, run_id="run1", anchor_tolerance=2.0)
    assert "turboquant_mse (measured)" in result.anchor_deltas
    # anchor is the published TurboQuant (2.5) row, not TurboQuant (3.5).
    assert (
        result.anchor_source["turboquant_mse (measured)"]
        == "TurboQuant (2.5) (published)"
    )


def test_overall_verdict_fail_when_any_category_fails():
    published_single_qa = PUBLISHED["Full Cache (published)"]["SingleQA"]
    way_off = (published_single_qa + 50.0) / 100.0
    df = pd.DataFrame(
        [
            _row("fp16", "narrativeqa", way_off, kv_size_bits=16.0),
            _row("turboquant_mse", "narrativeqa", 0.4, kv_size_bits=2.5),
        ]
    )
    result = build_table(df, run_id="run1", anchor_tolerance=2.0)
    assert result.overall_verdict == "FAIL"


def test_provenance_warning_fires_for_v1_non_parity_input():
    df = pd.DataFrame(_synthetic_rows(longbench_version="v1"))
    result = build_table(df, run_id="run1", anchor_tolerance=2.0)
    assert result.parity_warning is not None
    assert "v1" in result.parity_warning


def test_provenance_warning_fires_for_truncation_sentinel():
    df = pd.DataFrame(_synthetic_rows(max_prompt_tokens=-1))
    result = build_table(df, run_id="run1", anchor_tolerance=2.0)
    assert result.parity_warning is not None
    assert "-1" in result.parity_warning or "sentinel" in result.parity_warning.lower()


def test_no_provenance_warning_for_parity_input():
    df = pd.DataFrame(
        _synthetic_rows(longbench_version="v1_e", max_prompt_tokens=31500)
    )
    result = build_table(df, run_id="run1", anchor_tolerance=2.0)
    assert result.parity_warning is None


def test_caption_carries_provenance_fields():
    df = pd.DataFrame(_synthetic_rows())
    result = build_table(df, run_id="myrun-123", anchor_tolerance=2.0)
    assert "myrun-123" in result.caption
    assert "v1_e" in result.caption
    assert "31500" in result.caption


def test_render_markdown_and_latex_do_not_crash_and_contain_header():
    df = pd.DataFrame(_synthetic_rows())
    result = build_table(df, run_id="run1", anchor_tolerance=2.0)
    md = result.to_markdown()
    tex = result.to_latex()
    assert "KV Size" in md
    assert "SingleQA" in md
    assert "\\begin{tabular}" in tex or "\\begin{table}" in tex


def test_kv_size_column_is_16_for_fp16_and_matches_mean_bpe_for_measured_arm():
    df = pd.DataFrame(_synthetic_rows())
    result = build_table(df, run_id="run1", anchor_tolerance=2.0)
    assert result.rows_by_method["fp16 (measured)"]["KV Size (bits)"] == pytest.approx(
        16.0
    )
    assert result.rows_by_method["turboquant_mse (measured)"][
        "KV Size (bits)"
    ] == pytest.approx(2.5)


def test_result_is_dataclass_like_container():
    assert hasattr(TableResult, "to_markdown")
    assert hasattr(TableResult, "to_latex")


def test_pre_w3_parquet_missing_provenance_columns_does_not_crash():
    """Old (pre-W3) parquets lack longbench_version/max_prompt_tokens/n_samples entirely —
    build_table must not KeyError on caption/provenance construction (this is exactly the
    crash the w6-1 report observed against results/k3_longbench/20260702-164100-46b9579)."""
    rows = _synthetic_rows()
    df = pd.DataFrame(rows).drop(
        columns=["longbench_version", "max_prompt_tokens", "n_samples"]
    )
    result = build_table(df, run_id="old-run", anchor_tolerance=2.0)
    assert "unknown (pre-W3 parquet)" in result.caption
    assert result.parity_warning is None  # can't assert non-parity without the columns
