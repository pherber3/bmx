"""k3_longbench emits a parquet with the expected schema (tiny_llama, offline, no download)."""

import pandas as pd

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
    ):
        assert col in df.columns, f"missing column: {col}"
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
