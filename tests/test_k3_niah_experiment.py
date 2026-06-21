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
