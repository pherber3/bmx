"""k3_niah emits a parquet with the expected schema (tiny_llama, offline, no download)."""

import pandas as pd

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
