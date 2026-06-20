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
