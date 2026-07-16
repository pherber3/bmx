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
    shards = [
        p for p in partial_dir.glob("*.parquet") if not p.stem.endswith("__samples")
    ]
    assert len(shards) == 1, "exactly one (arm, task) pair should have checkpointed"
    # The samples shard rides along with its aggregate (same checkpoint moment).
    assert shards[0].with_name(f"{shards[0].stem}__samples.parquet").exists()

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

    # All 4 aggregate shards (+ their 4 samples shards) should now exist under partial/.
    all_shards = list(partial_dir.glob("*.parquet"))
    agg = [p for p in all_shards if not p.stem.endswith("__samples")]
    samples = [p for p in all_shards if p.stem.endswith("__samples")]
    assert len(agg) == 4
    assert len(samples) == 4


def test_longbench_samples_parquet_written(tmp_path):
    """Each (arm, task) pair gets a per-sample shard alongside its aggregate shard.

    Schema {arm, task, sample_idx, score}; row count == the aggregate's n_samples; the
    aggregate `code_sim` column is the PLAIN MEAN of the per-sample scores (run() computes
    `sum(scores)/len(scores)`; the max-over-ground-truths happens per sample INSIDE
    longbench_score, before the sample score is emitted). This is the bootstrap-CI enabler.
    """
    model = tiny_llama()
    cfg = Config(
        arms=("fp16", "kivi"),
        tasks=("lcc", "repobench-p"),
        n_prefill=16,
        group=16,
        rank=4,
    )
    run_dir = run(cfg, model=model, root=str(tmp_path))
    partial_dir = run_dir / "partial"
    for arm in cfg.arms:
        for task in cfg.tasks:
            agg_path = partial_dir / f"{arm}__{task}.parquet"
            samples_path = partial_dir / f"{arm}__{task}__samples.parquet"
            assert agg_path.exists(), f"missing aggregate shard for {arm}/{task}"
            assert samples_path.exists(), f"missing samples shard for {arm}/{task}"
            agg = pd.read_parquet(agg_path)
            samples = pd.read_parquet(samples_path)
            assert list(samples.columns) == ["arm", "task", "sample_idx", "score"]
            assert len(samples) == int(agg["n_samples"].iloc[0])
            assert list(samples["sample_idx"]) == list(range(len(samples)))
            assert (samples["arm"] == arm).all()
            assert (samples["task"] == task).all()
            assert samples["score"].mean() == pytest.approx(agg["code_sim"].iloc[0])
    # The final metrics.parquet is unchanged: aggregate rows only, no per-sample columns.
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert len(df) == 4
    assert "sample_idx" not in df.columns


def test_longbench_samples_written_before_aggregate(tmp_path, monkeypatch):
    """A crash between the two writes leaves samples WITHOUT aggregate — never the reverse.

    The aggregate shard's existence is the resume key, so it must imply the samples shard
    exists (samples first, then aggregate)."""
    import experiments.k3_longbench as mod

    model = tiny_llama()
    cfg = Config(arms=("fp16",), tasks=("lcc",), n_prefill=16, group=16, rank=4)

    def boom(*args, **kwargs):
        raise RuntimeError("injected write_shard failure")

    monkeypatch.setattr(mod, "write_shard", boom)
    with pytest.raises(RuntimeError, match="injected write_shard failure"):
        run(cfg, model=model, root=str(tmp_path))

    run_dir = next((tmp_path / "k3_longbench").iterdir())
    partial_dir = run_dir / "partial"
    assert (partial_dir / "fp16__lcc__samples.parquet").exists()
    assert not (partial_dir / "fp16__lcc.parquet").exists()


def test_write_samples_shard_empty_writes_nothing(tmp_path):
    """0 samples -> no samples file (the aggregate-implies-samples rule only holds
    for pairs that actually evaluated at least one sample)."""
    from experiments._common import write_samples_shard

    out = write_samples_shard(tmp_path, [], "fp16", "lcc")
    assert out is None
    assert not (tmp_path / "partial" / "fp16__lcc__samples.parquet").exists()


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
