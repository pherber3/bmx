"""CPU-testable smoke tests for scripts/census_gate_driver.py.

The driver reads a stage-1 (96k) census parquet, projects resident_128k per
(path, arm) via the plan's linear rule, and launches the 128k census
invocation SEQUENTIALLY for cells that clear the gate. subprocess is never
actually invoked here -- monkeypatched to record calls instead.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from scripts.census_gate_driver import (
    Config,
    STALE_MESSAGE,
    project_128k,
    run_gate,
)

GiB = 1024**3


def _row(seq_len, arm, path, resident_gib, oom=False):
    return {
        "seq_len": seq_len,
        "arm": arm,
        "path": path,
        "resident_after_prefill": (-1 if oom else resident_gib * GiB),
        "oom": oom,
        "bpe_k": 3.0,
        "bpe_v": 2.0,
    }


def _write_parquet(path, rows):
    pd.DataFrame(rows).to_parquet(path, index=False)


def _complete_rows(cells):
    """cells: list of (arm, path, r32, r64, r96) in GiB."""
    rows = []
    for arm, path, r32, r64, r96 in cells:
        rows.append(_row(32768, arm, path, r32))
        rows.append(_row(65536, arm, path, r64))
        rows.append(_row(98304, arm, path, r96))
    return rows


def _cfg(stage1_dir, cells, **overrides):
    """Config whose expected_pairs match exactly the (arm, path) pairs present
    in `cells` -- so the polling loop's completeness check is satisfiable by
    the fixture data (never waits for a cell that was never written; the
    cartesian expected_arms x expected_paths would overgenerate whenever a
    fixture uses different paths for different arms)."""
    pairs = tuple(dict.fromkeys((c[0], c[1]) for c in cells))
    kwargs = dict(
        stage1_run_dir=str(stage1_dir),
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        pack_path="results/cache/fake_packs.safetensors",
        gate_gib=90.0,
        stale_minutes=30,
        poll_seconds=0,
        dry_run=False,
        expected_arms=(),
        expected_paths=(),
        expected_pairs=pairs,
    )
    kwargs.update(overrides)
    return Config(**kwargs)


# ---------------------------------------------------------------------------
# Projection arithmetic
# ---------------------------------------------------------------------------


def test_project_128k_linear_growth():
    # growth 64k->96k is 10 GiB -> projected 128k = 96k + 10 = 106 GiB
    assert project_128k(resident_64k=50.0, resident_96k=60.0) == pytest.approx(70.0)


def test_project_128k_matches_plan_arithmetic_example():
    # resident_96k + (resident_96k - resident_64k), zero growth case
    assert project_128k(resident_64k=40.0, resident_96k=40.0) == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# Gate / skip decisions + dry-run
# ---------------------------------------------------------------------------


def test_cell_under_gate_launches_sequentially(tmp_path, monkeypatch):
    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir()
    parquet = stage1_dir / "census.parquet"
    # 64k->96k growth of 5 GiB -> projected 128k = 55+5=60 GiB, well under 90.
    cells = [("k4_b2.5", "chunked", 40.0, 50.0, 55.0)]
    _write_parquet(parquet, _complete_rows(cells))

    launched: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        launched.append(cmd)

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("scripts.census_gate_driver.subprocess.run", fake_run)

    cfg = _cfg(stage1_dir, cells)
    result = run_gate(cfg)

    assert len(launched) == 1
    assert result.exit_code == 0
    assert result.launched_cells == [("k4_b2.5", "chunked")]
    assert result.skipped_cells == []


def test_cell_over_gate_is_skipped_not_launched(tmp_path, monkeypatch):
    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir()
    parquet = stage1_dir / "census.parquet"
    # 64k->96k growth of 20 GiB -> projected 128k = 83+20=103 GiB > 90 -> SKIP.
    cells = [("fp16", "dense_stream", 63.0, 83.0, 103.0)]
    _write_parquet(parquet, _complete_rows(cells))

    launched: list[list[str]] = []
    monkeypatch.setattr(
        "scripts.census_gate_driver.subprocess.run",
        lambda cmd, **kw: launched.append(cmd),
    )

    cfg = _cfg(stage1_dir, cells)
    result = run_gate(cfg)

    assert launched == []
    assert result.launched_cells == []
    assert result.skipped_cells == [("fp16", "dense_stream")]
    assert result.exit_code != 0  # all cells skipped -> nonzero


def test_mixed_cells_all_skipped_gives_nonzero_exit(tmp_path, monkeypatch):
    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir()
    parquet = stage1_dir / "census.parquet"
    cells = [("fp16", "dense_stream", 63.0, 83.0, 103.0)]  # over gate
    _write_parquet(parquet, _complete_rows(cells))
    monkeypatch.setattr(
        "scripts.census_gate_driver.subprocess.run",
        lambda cmd, **kw: (_ for _ in ()).throw(AssertionError("should not launch")),
    )
    cfg = _cfg(stage1_dir, cells)
    result = run_gate(cfg)
    assert result.exit_code != 0


def test_some_launched_gives_zero_exit(tmp_path, monkeypatch):
    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir()
    parquet = stage1_dir / "census.parquet"
    cells = [
        ("k4_b2.5", "chunked", 40.0, 50.0, 55.0),  # under gate
        ("fp16", "dense_stream", 63.0, 83.0, 103.0),  # over gate
    ]
    _write_parquet(parquet, _complete_rows(cells))
    launched = []
    monkeypatch.setattr(
        "scripts.census_gate_driver.subprocess.run",
        lambda cmd, **kw: launched.append(cmd) or type("R", (), {"returncode": 0})(),
    )
    cfg = _cfg(stage1_dir, cells)
    result = run_gate(cfg)
    assert result.exit_code == 0
    assert len(launched) == 1
    assert result.launched_cells == [("k4_b2.5", "chunked")]
    assert result.skipped_cells == [("fp16", "dense_stream")]


def test_launches_are_sequential_not_concurrent(tmp_path, monkeypatch):
    """Two cells that both clear the gate must run one-at-a-time: subprocess.run
    (blocking) is used, never Popen without a wait -- assert call N+1 doesn't
    start until call N's fake_run has returned."""
    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir()
    parquet = stage1_dir / "census.parquet"
    cells = [
        ("k4_b2.5", "chunked", 40.0, 50.0, 55.0),
        ("k2b", "chunked", 20.0, 25.0, 28.0),
    ]
    _write_parquet(parquet, _complete_rows(cells))

    active = {"n": 0}
    max_concurrent = {"n": 0}

    def fake_run(cmd, **kwargs):
        active["n"] += 1
        max_concurrent["n"] = max(max_concurrent["n"], active["n"])
        time.sleep(0.01)
        active["n"] -= 1

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("scripts.census_gate_driver.subprocess.run", fake_run)

    cfg = _cfg(stage1_dir, cells)
    result = run_gate(cfg)
    assert max_concurrent["n"] == 1
    assert len(result.launched_cells) == 2


def test_dry_run_prints_commands_without_calling_subprocess(
    tmp_path, monkeypatch, capsys
):
    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir()
    parquet = stage1_dir / "census.parquet"
    cells = [("k4_b2.5", "chunked", 40.0, 50.0, 55.0)]
    _write_parquet(parquet, _complete_rows(cells))

    def fail_if_called(cmd, **kwargs):
        raise AssertionError("subprocess.run must not be called in --dry-run")

    monkeypatch.setattr("scripts.census_gate_driver.subprocess.run", fail_if_called)

    cfg = _cfg(stage1_dir, cells, dry_run=True)
    result = run_gate(cfg)
    out = capsys.readouterr().out
    assert "k4_b2.5" in out
    assert "chunked" in out
    assert result.launched_cells == []  # dry-run never actually launches
    assert result.exit_code == 0  # dry-run success is not "all skipped"


# ---------------------------------------------------------------------------
# Staleness guard
# ---------------------------------------------------------------------------


def test_staleness_guard_aborts_when_parquet_stops_growing(tmp_path, monkeypatch):
    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir()
    parquet = stage1_dir / "census.parquet"
    # Incomplete: only 32k/64k rows present, 96k never lands.
    rows = [
        _row(32768, "k4_b2.5", "chunked", 40.0),
        _row(65536, "k4_b2.5", "chunked", 50.0),
    ]
    _write_parquet(parquet, rows)

    cfg = _cfg(
        stage1_dir,
        [("k4_b2.5", "chunked", 0, 0, 0)],  # expected_arms/paths only; values unused
    )

    # Force the poll loop's clock forward past the staleness window without
    # sleeping in the test: monkeypatch time.monotonic to jump ahead each call.
    clock = {"t": 0.0}

    def fake_monotonic():
        clock["t"] += 60 * 20  # +20 min per check -> exceeds 30-min stale window
        return clock["t"]

    monkeypatch.setattr("scripts.census_gate_driver.time.monotonic", fake_monotonic)

    with pytest.raises(SystemExit) as exc_info:
        run_gate(cfg)
    assert exc_info.value.code != 0


def test_staleness_message_is_loud():
    assert "stale" in STALE_MESSAGE.lower()
    assert (
        "abort" in STALE_MESSAGE.lower() or "stopped growing" in STALE_MESSAGE.lower()
    )


# ---------------------------------------------------------------------------
# Expected-cells wait: polls until all (arm, path) x (32768/65536/98304) rows present
# ---------------------------------------------------------------------------


def test_waits_for_incomplete_parquet_then_proceeds(tmp_path, monkeypatch):
    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir()
    parquet = stage1_dir / "census.parquet"
    # Start with only 32k+64k rows.
    _write_parquet(
        parquet,
        [
            _row(32768, "k4_b2.5", "chunked", 40.0),
            _row(65536, "k4_b2.5", "chunked", 50.0),
        ],
    )

    launched = []
    monkeypatch.setattr(
        "scripts.census_gate_driver.subprocess.run",
        lambda cmd, **kw: launched.append(cmd) or type("R", (), {"returncode": 0})(),
    )

    # Simulate the parquet gaining the 96k row after the first poll.
    calls = {"n": 0}
    real_read = pd.read_parquet

    def fake_read_parquet(path, *a, **kw):
        calls["n"] += 1
        if calls["n"] >= 2:
            _write_parquet(
                parquet,
                [
                    _row(32768, "k4_b2.5", "chunked", 40.0),
                    _row(65536, "k4_b2.5", "chunked", 50.0),
                    _row(98304, "k4_b2.5", "chunked", 55.0),
                ],
            )
        return real_read(path, *a, **kw)

    monkeypatch.setattr("scripts.census_gate_driver.pd.read_parquet", fake_read_parquet)

    cfg = _cfg(stage1_dir, [("k4_b2.5", "chunked", 40.0, 50.0, 55.0)])
    result = run_gate(cfg)
    assert result.launched_cells == [("k4_b2.5", "chunked")]
