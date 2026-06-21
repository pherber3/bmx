"""k3 experiment emits a parquet with the expected schema (tiny_llama, offline).

The real-text path (load_eval_tokens) and real-needle path (build_needle_ids) are
exercised ONLY on the VM/real run, NOT in CI — they download wikitext-2 and model
weights.  This test exercises the offline mechanics only: it injects both a tiny
model and synthetic input_ids so run() never calls load_eval_tokens or the tokenizer.
"""

import pandas as pd
import torch

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
    # Provide synthetic input_ids so run() never calls load_eval_tokens (no download).
    # Both model and input_ids are injected → fully offline, fast.
    g = torch.Generator().manual_seed(cfg.seq_seed)
    input_ids = torch.randint(
        0, model.config.vocab_size, (1, cfg.n_context), generator=g
    )
    run_dir = run(cfg, model=model, input_ids=input_ids, root=str(tmp_path))
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


def test_k2b_matched_variants_parse_and_lower_key_bits():
    # The k2b_k{bits}r{rank} variants exist to match turboquant's compression by
    # dropping the key budget. Pin the parse + the lowered-bits contract so the
    # matched-compression head-to-head can't silently revert to keys@3b.
    from experiments.k3_live_generation import Config, _spec_pair

    cfg = Config()
    k_can, _ = _spec_pair("k2b", cfg)
    assert (k_can.bits, k_can.rank) == (3, cfg.rank)  # canonical: keys@3b

    k8, v8 = _spec_pair("k2b_k2r8", cfg)
    assert (k8.arm, k8.bits, k8.rank) == ("lowrank_rtn_channel", 2, 8)
    assert v8.arm == "turboquant_mse" and v8.bits == 2  # V side unchanged

    k16, _ = _spec_pair("k2b_k2r16", cfg)
    assert (k16.bits, k16.rank) == (2, 16)


def test_plot_k3_makes_pngs(tmp_path):
    import pandas as pd
    from experiments.plots.plot_k3 import make_figures

    df = pd.DataFrame(
        [
            {
                "arm": "fp16",
                "bpe_k": 16.0,
                "bpe_v": 16.0,
                "ppl": 10.0,
                "n_context": 512,
                "retrieved": True,
                "compression": 1.0,
            },
            {
                "arm": "k2b",
                "bpe_k": 3.0,
                "bpe_v": 2.0,
                "ppl": 10.1,
                "n_context": 512,
                "retrieved": True,
                "compression": 5.3,
            },
        ]
    )
    paths = make_figures(df, str(tmp_path))
    assert len(paths) == 2
    assert all(p.exists() for p in paths)
