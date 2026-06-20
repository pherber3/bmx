"""k3 experiment emits a parquet with the expected schema (tiny_llama, offline)."""

import pandas as pd

from experiments.k3_live_generation import Config, run
from factories import tiny_llama


def test_k3_run_emits_parquet(tmp_path):
    model = tiny_llama()
    # Brief specifies n_prefill=12, n_context=28, group=16.
    # However, rtn_channel (KIVI K) and lowrank_rtn_channel (k2b K) assert S % group == 0.
    # In live_generation_ppl the prefill is sent all at once (S=n_prefill) and the
    # continuation is also sent all at once (S=n_context).  So both n_prefill and
    # n_context must be divisible by group=16.  12%16=12 and 28%16=12 both fail.
    # Using n_prefill=16, n_context=32 satisfies the constraint:
    #   16%16=0, 32%16=0, rank=4 <= min(16, C=16).
    cfg = Config(
        arms=("fp16", "k2b", "kivi"), n_prefill=16, n_context=32, rank=4, group=16
    )
    run_dir = run(cfg, model=model, root=str(tmp_path))
    df = pd.read_parquet(run_dir / "metrics.parquet")
    for col in (
        "arm",
        "bpe_k",
        "bpe_v",
        "ppl",
        "n_eval",
        "packed_bytes",
        "fp16_bytes",
        "compression",
        "n_prefill",
        "n_context",
        "retrieved",
    ):
        assert col in df.columns, f"missing column: {col}"
    assert set(df["arm"]) <= {
        "fp16",
        "k2b",
        "kivi",
        "turboquant_mse",
        "turboquant_prod",
    }
