"""k3_niah emits a parquet with the expected schema (tiny_llama, offline, no download)."""

import dataclasses

import pandas as pd
import pytest

from experiments.k3_niah import Config, run
from factories import tiny_llama


def test_k3_niah_run_emits_parquet(tmp_path):
    model = tiny_llama()
    # tiny_llama max_position_embeddings=64 → keep lengths small; group=16 divisibility.
    cfg = Config(
        arms=("fp16", "kivi"),
        lengths=(32, 48),
        depths=(0.25, 0.5),
        n_prefill=16,
        group=16,
        rank=4,
    )
    run_dir = run(cfg, model=model, root=str(tmp_path))
    df = pd.read_parquet(run_dir / "metrics.parquet")
    for col in (
        "arm",
        "length",
        "depth",
        "recall",
        "recall_full",
        "bpe_k",
        "bpe_v",
        "compression",
        "n_prefill",
        "recall_kind",
    ):
        assert col in df.columns, f"missing column: {col}"
    # 2 arms × 2 lengths × 2 depths = 8 rows.
    assert len(df) == 8
    assert set(df["arm"]) <= {
        "fp16",
        "k2b",
        "kivi",
        "turboquant_mse",
        "turboquant_prod",
    }
    # Offline run uses the argmax proxy mechanism.
    assert set(df["recall_kind"]) == {"argmax_proxy"}


def test_niah_rows_have_kv_size_bits(tmp_path):
    model = tiny_llama()
    cfg = Config(
        arms=("fp16", "kivi"),
        lengths=(32, 48),
        depths=(0.25, 0.5),
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


def test_niah_checkpoint_resume(tmp_path):
    """A killed run leaves per-(arm,length) shards; --resume finishes the rest, no dupes."""
    model = tiny_llama()
    cfg = Config(
        arms=("fp16", "kivi"),
        lengths=(32, 48),
        depths=(0.25, 0.5),
        n_prefill=16,
        group=16,
        rank=4,
    )

    with pytest.raises(RuntimeError, match="injected stop"):
        run(cfg, model=model, root=str(tmp_path), _stop_after_pairs=1)

    runs = list((tmp_path / "k3_niah").iterdir())
    assert len(runs) == 1
    run_dir = runs[0]

    partial_dir = run_dir / "partial"
    shards = [
        p for p in partial_dir.glob("*.parquet") if not p.stem.endswith("__samples")
    ]
    assert len(shards) == 1, "exactly one (arm, length) pair should have checkpointed"
    # The samples shard rides along with its aggregate (same checkpoint moment).
    assert shards[0].with_name(f"{shards[0].stem}__samples.parquet").exists()
    assert not (run_dir / "metrics.parquet").exists()

    resume_cfg = dataclasses.replace(cfg, resume=str(run_dir))
    resumed_dir = run(resume_cfg, model=model, root=str(tmp_path))
    assert resumed_dir == run_dir

    df = pd.read_parquet(run_dir / "metrics.parquet")
    # 2 arms x 2 lengths x 2 depths = 8 rows; 4 distinct (arm, length) pairs.
    assert len(df) == 8
    pairs = list(zip(df["arm"], df["length"]))
    assert len(set(pairs)) == 4
    all_shards = list(partial_dir.glob("*.parquet"))
    agg = [p for p in all_shards if not p.stem.endswith("__samples")]
    samples = [p for p in all_shards if p.stem.endswith("__samples")]
    assert len(agg) == 4
    assert len(samples) == 4


def test_niah_samples_parquet_written(tmp_path):
    """Each (arm, length) pair gets a per-generation shard alongside its aggregate shard.

    The loop's natural grain is one generation per depth, so the samples shard has schema
    {arm, length, depth, sample_idx, recall_full} with one row per depth. The aggregate
    shard already stores per-depth rows (no averaging in the harness), so the samples
    rows must reproduce its recall_full values exactly, row for row (joined on depth).
    """
    model = tiny_llama()
    cfg = Config(
        arms=("fp16", "kivi"),
        lengths=(32, 48),
        depths=(0.25, 0.5),
        n_prefill=16,
        group=16,
        rank=4,
    )
    run_dir = run(cfg, model=model, root=str(tmp_path))
    partial_dir = run_dir / "partial"
    for arm in cfg.arms:
        for length in cfg.lengths:
            agg_path = partial_dir / f"{arm}__{length}.parquet"
            samples_path = partial_dir / f"{arm}__{length}__samples.parquet"
            assert agg_path.exists(), f"missing aggregate shard for {arm}/{length}"
            assert samples_path.exists(), f"missing samples shard for {arm}/{length}"
            agg = pd.read_parquet(agg_path)
            samples = pd.read_parquet(samples_path)
            assert list(samples.columns) == [
                "arm",
                "length",
                "depth",
                "sample_idx",
                "recall_full",
            ]
            assert len(samples) == len(cfg.depths)
            assert list(samples["sample_idx"]) == list(range(len(cfg.depths)))
            assert list(samples["depth"]) == list(cfg.depths)
            assert (samples["arm"] == arm).all()
            assert (samples["length"] == length).all()
            merged = samples.merge(
                agg[["depth", "recall_full"]], on="depth", suffixes=("_s", "_a")
            )
            assert len(merged) == len(cfg.depths)
            assert (merged["recall_full_s"] == merged["recall_full_a"]).all()
    # The final metrics.parquet is unchanged: aggregate rows only, no per-sample columns.
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert len(df) == 8
    assert "sample_idx" not in df.columns


def test_niah_samples_written_before_aggregate(tmp_path, monkeypatch):
    """A crash between the two writes leaves samples WITHOUT aggregate — never the reverse.

    The aggregate shard's existence is the resume key, so it must imply the samples shard
    exists (samples first, then aggregate)."""
    import experiments.k3_niah as mod

    model = tiny_llama()
    cfg = Config(
        arms=("fp16",),
        lengths=(32,),
        depths=(0.25, 0.5),
        n_prefill=16,
        group=16,
        rank=4,
    )

    def boom(*args, **kwargs):
        raise RuntimeError("injected write_shard failure")

    monkeypatch.setattr(mod, "write_shard", boom)
    with pytest.raises(RuntimeError, match="injected write_shard failure"):
        run(cfg, model=model, root=str(tmp_path))

    run_dir = next((tmp_path / "k3_niah").iterdir())
    partial_dir = run_dir / "partial"
    assert (partial_dir / "fp16__32__samples.parquet").exists()
    assert not (partial_dir / "fp16__32.parquet").exists()


def test_niah_resume_rejects_config_mismatch(tmp_path):
    """Resuming with a changed config field is a hard error, not a silent continue."""
    model = tiny_llama()
    cfg = Config(
        arms=("fp16", "kivi"),
        lengths=(32, 48),
        depths=(0.25, 0.5),
        n_prefill=16,
        group=16,
        rank=4,
    )
    with pytest.raises(RuntimeError, match="injected stop"):
        run(cfg, model=model, root=str(tmp_path), _stop_after_pairs=1)

    run_dir = next((tmp_path / "k3_niah").iterdir())

    mismatched_cfg = dataclasses.replace(cfg, rank=8, resume=str(run_dir))
    with pytest.raises(ValueError, match="(?i)config.*mismatch|mismatch.*config"):
        run(mismatched_cfg, model=model, root=str(tmp_path))


def test_plot_k3_niah_makes_pngs(tmp_path):
    import pandas as pd
    from experiments.plots.plot_k3_niah import make_figures

    df = pd.DataFrame(
        [
            {
                "arm": "fp16",
                "length": 4096,
                "depth": 0.5,
                "recall": 10.0,
                "compression": 1.0,
            },
            {
                "arm": "fp16",
                "length": 8192,
                "depth": 0.5,
                "recall": 9.0,
                "compression": 1.0,
            },
            {
                "arm": "kivi",
                "length": 4096,
                "depth": 0.5,
                "recall": 8.0,
                "compression": 4.1,
            },
            {
                "arm": "kivi",
                "length": 8192,
                "depth": 0.5,
                "recall": 6.0,
                "compression": 4.1,
            },
        ]
    )
    paths = make_figures(df, str(tmp_path))
    assert len(paths) >= 1
    assert all(p.exists() for p in paths)


def test_niah_heatmap_has_aggregate_score(tmp_path):
    import json
    import math

    import pandas as pd

    from experiments.plots.plot_k3_niah import make_figures

    # k2b cells average 7.5 → score 0.750; fp16 cells average 9.0 → score 0.900.
    # depth.nunique() > 1 so the heatmap (and its scores) render.
    df = pd.DataFrame(
        [
            {
                "arm": "k2b",
                "length": 4096,
                "depth": 0.25,
                "recall_full": 7.0,
                "compression": 4.1,
            },
            {
                "arm": "k2b",
                "length": 4096,
                "depth": 0.75,
                "recall_full": 8.0,
                "compression": 4.1,
            },
            {
                "arm": "fp16",
                "length": 4096,
                "depth": 0.25,
                "recall_full": 9.0,
                "compression": 1.0,
            },
            {
                "arm": "fp16",
                "length": 4096,
                "depth": 0.75,
                "recall_full": 9.0,
                "compression": 1.0,
            },
        ]
    )
    paths = make_figures(df, str(tmp_path))
    score_paths = [p for p in paths if p.name == "niah_heatmap_scores.json"]
    assert len(score_paths) == 1, "niah_heatmap_scores.json not emitted"
    scores = json.loads(score_paths[0].read_text())
    assert not math.isnan(scores["k2b"]) and abs(scores["k2b"] - 0.75) < 1e-6
    assert not math.isnan(scores["fp16"]) and abs(scores["fp16"] - 0.90) < 1e-6
