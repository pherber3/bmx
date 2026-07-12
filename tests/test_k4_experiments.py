import json

import torch

from bmx.cache.collect import save_cache


def _tiny_cache(path, S=128, C=16, h_kv=2, T=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = C // h_kv
    raw = torch.randn(C, 3, generator=g)
    dirs, _ = torch.linalg.qr(raw)
    z = torch.randn(S, 3, generator=g) * torch.tensor([20.0, 12.0, 8.0])
    M = (z @ dirs.mT + torch.randn(S, C, generator=g)).half()
    K = M.reshape(S, h_kv, d).permute(1, 0, 2)  # from_matrix layout
    tensors = {}
    for i in range(2):  # 2 layers
        tensors[f"layer{i}.k_pre"] = K.contiguous()
        tensors[f"layer{i}.k"] = K.contiguous()
        tensors[f"layer{i}.v"] = torch.randn(h_kv, S, d, generator=g).half()
        tensors[f"layer{i}.q"] = torch.randn(h_kv * 2, T, d, generator=g).half()
    save_cache(tensors, path)


def test_k4_spectra_smoke(tmp_path):
    import pandas as pd

    from experiments.k4_spectra import Config, main

    main_path = tmp_path / "main.safetensors"
    other_path = tmp_path / "other.safetensors"
    _tiny_cache(main_path, seed=0)
    _tiny_cache(other_path, seed=1)
    cfg = Config(
        cache_path=str(main_path),
        corpus_cache_paths=(str(other_path),),
        model_label="tiny",
        budgets=(2.5,),
        group=16,
        out_root=str(tmp_path / "results"),
    )
    run_dir = main(cfg)
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert {"oracle", "heldout", "corpus", "reference"} <= set(df.fit_mode.unique())
    assert {"am_gm", "logit", "bpe_model", "bpe_skeptic"} <= set(df.columns)
    verdict = json.loads((run_dir / "g0_verdict.json").read_text())
    assert "retention_heldout" in verdict and "retention_corpus" in verdict
