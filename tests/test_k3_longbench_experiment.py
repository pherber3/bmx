"""k3_longbench emits a parquet with the expected schema (tiny_llama, offline, no download)."""

import dataclasses

import pandas as pd
import pytest

from experiments.k3_longbench import Config, run
from factories import tiny_llama


def test_k3_longbench_run_emits_parquet(tmp_path):
    model = tiny_llama()
    # tiny_llama max_position_embeddings=64 → keep prompt small; group=16 divisibility.
    cfg = Config(
        arms=("fp16", "kivi"),
        tasks=("lcc", "repobench-p"),
        n_prefill=16,
        group=16,
        rank=4,
    )
    run_dir = run(cfg, model=model, root=str(tmp_path))
    df = pd.read_parquet(run_dir / "metrics.parquet")
    for col in (
        "arm",
        "task",
        "code_sim",
        "n_samples",
        "bpe_k",
        "bpe_v",
        "compression",
        "n_prefill",
        "score_kind",
        "max_prompt_tokens",
        "longbench_version",
        "chat_wrap",
    ):
        assert col in df.columns, f"missing column: {col}"
    assert set(df["longbench_version"]) == {"v1"}
    # 2 arms × 2 tasks = 4 rows.
    assert len(df) == 4
    assert set(df["arm"]) <= {
        "fp16",
        "k2b",
        "kivi",
        "turboquant_mse",
        "turboquant_prod",
    }
    assert set(df["score_kind"]) == {"code_sim_offline"}


def test_k3_longbench_chat_wrap_defaults_off_and_is_recorded(tmp_path):
    # Default Config().chat_wrap must be False (comparability with every prior recorded
    # row), and the value must be threaded into the emitted rows verbatim.
    model = tiny_llama()
    cfg = Config(
        arms=("fp16",),
        tasks=("lcc",),
        n_prefill=16,
        group=16,
        rank=4,
    )
    assert cfg.chat_wrap is False
    run_dir = run(cfg, model=model, root=str(tmp_path))
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert (df["chat_wrap"] == False).all()  # noqa: E712 — pandas boolean column comparison


def test_longbench_rows_have_kv_size_bits(tmp_path):
    model = tiny_llama()
    cfg = Config(
        arms=("fp16", "kivi"),
        tasks=("lcc", "repobench-p"),
        n_prefill=16,
        group=16,
        rank=4,
    )
    run_dir = run(cfg, model=model, root=str(tmp_path))
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert "kv_size_bits" in df.columns
    assert (df["kv_size_bits"] > 0).all()
    assert (df["kv_size_bits"] <= 16.0 + 1e-6).all()
    # fp16 K and V are each 16 bpe → average 16.0.
    fp16 = df[df["arm"] == "fp16"]
    assert (fp16["kv_size_bits"] == 16.0).all()


def test_plot_k3_longbench_makes_pngs(tmp_path):
    import pandas as pd
    from experiments.plots.plot_k3_longbench import make_figures

    df = pd.DataFrame(
        [
            {"arm": "fp16", "task": "lcc", "code_sim": 46.0, "compression": 1.0},
            {"arm": "kivi", "task": "lcc", "code_sim": 44.0, "compression": 4.1},
            {
                "arm": "fp16",
                "task": "repobench-p",
                "code_sim": 45.0,
                "compression": 1.0,
            },
            {
                "arm": "kivi",
                "task": "repobench-p",
                "code_sim": 42.0,
                "compression": 4.1,
            },
        ]
    )
    paths = make_figures(df, str(tmp_path))
    assert len(paths) >= 1
    assert all(p.exists() for p in paths)


def test_longbench_checkpoint_resume(tmp_path):
    """A killed run leaves per-(arm,task) shards; --resume finishes the rest, no duplicates."""
    model = tiny_llama()
    cfg = Config(
        arms=("fp16", "kivi"),
        tasks=("lcc", "repobench-p"),
        n_prefill=16,
        group=16,
        rank=4,
    )

    # Simulate a crash after the first (arm, task) pair completes.
    with pytest.raises(RuntimeError, match="injected stop"):
        run(cfg, model=model, root=str(tmp_path), _stop_after_pairs=1)

    runs = list((tmp_path / "k3_longbench").iterdir())
    assert len(runs) == 1
    run_dir = runs[0]

    partial_dir = run_dir / "partial"
    shards = list(partial_dir.glob("*.parquet"))
    assert len(shards) == 1, "exactly one (arm, task) pair should have checkpointed"

    # metrics.parquet must NOT exist yet — the crash happened before the final write.
    assert not (run_dir / "metrics.parquet").exists()

    # Resume: same config, same run_dir -> skips the done pair, completes the rest.
    resume_cfg = dataclasses.replace(cfg, resume=str(run_dir))
    resumed_dir = run(resume_cfg, model=model, root=str(tmp_path))
    assert resumed_dir == run_dir

    df = pd.read_parquet(run_dir / "metrics.parquet")
    # 2 arms x 2 tasks = 4 rows, each (arm, task) pair exactly once.
    assert len(df) == 4
    pairs = list(zip(df["arm"], df["task"]))
    assert len(set(pairs)) == 4

    # All 4 shards should now exist under partial/ (one per completed pair).
    assert len(list(partial_dir.glob("*.parquet"))) == 4


def test_longbench_resume_rejects_config_mismatch(tmp_path):
    """Resuming with a changed config field is a hard error, not a silent continue."""
    model = tiny_llama()
    cfg = Config(
        arms=("fp16", "kivi"),
        tasks=("lcc", "repobench-p"),
        n_prefill=16,
        group=16,
        rank=4,
    )
    with pytest.raises(RuntimeError, match="injected stop"):
        run(cfg, model=model, root=str(tmp_path), _stop_after_pairs=1)

    run_dir = next((tmp_path / "k3_longbench").iterdir())

    mismatched_cfg = dataclasses.replace(cfg, rank=8, resume=str(run_dir))
    with pytest.raises(ValueError, match="(?i)config.*mismatch|mismatch.*config"):
        run(mismatched_cfg, model=model, root=str(tmp_path))
