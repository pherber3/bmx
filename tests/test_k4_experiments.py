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


def test_k4_frontier_smoke(tmp_path):
    import json

    import pandas as pd

    from experiments.k4_frontier import Config, main

    main_path = tmp_path / "main.safetensors"
    _tiny_cache(main_path, seed=0)
    cfg = Config(
        cache_path=str(main_path),
        model_label="tiny",
        budgets=(2.0, 3.0),
        group=16,
        ranks=(4,),
        coeffquant_rank=4,
        out_root=str(tmp_path / "results"),
    )
    run_dir = main(cfg)
    df = pd.read_parquet(run_dir / "metrics.parquet")
    for arm in (
        "spectral",
        "spectral_unweighted",
        "spectral_randbasis",
        "turboquant_mse",
        "lowrank_rtn_channel",
        "k2t_coeffquant",
        "rtn_channel",
    ):
        assert (df.arm == arm).any(), f"missing arm {arm}"
    v = json.loads((run_dir / "g1_verdict.json").read_text())
    assert "g1_pass" in v and "p3_verdict" in v and "p4_verdict" in v


def test_k4_frontier_figures(tmp_path):
    import pandas as pd

    from experiments.plots.plot_k4_frontier import make_figures

    rows = []
    for arm, base in (
        ("spectral", 0.03),
        ("turboquant_mse", 0.09),
        ("rtn_channel", 0.12),
    ):
        for i, bpe in enumerate((2.0, 3.0, 4.0)):
            rows.append(
                dict(
                    model="tiny",
                    layer=i % 2,
                    kind="k_pre",
                    arm=arm,
                    fit_mode="oracle",
                    weighted=True,
                    budget=bpe,
                    bits=int(bpe),
                    rank=0,
                    mse_scale=False,
                    bpe_model=bpe,
                    bpe_skeptic=bpe + 8.0,
                    bpe_skeptic_deploy=bpe + 0.5,
                    rel_fro=base,
                    logit=base,
                    logit_rope=base * (4.0**-i),
                )
            )
    paths = make_figures(pd.DataFrame(rows), str(tmp_path))
    names = {p.name for p in paths}
    assert {
        "k4_frontier_model.png",
        "k4_frontier_skeptic.png",
        "k4_structure_tax.png",
    } <= names


def test_greedy_layer_allocation_prefers_sensitive_steep_layers():
    from experiments.k4_alloc import greedy_layer_allocation

    grid = (2.0, 3.0, 4.0)
    # Layer 0: sensitive + steep curve; layer 1: insensitive + flat.
    curves = {0: {2.0: 0.4, 3.0: 0.1, 4.0: 0.02}, 1: {2.0: 0.05, 3.0: 0.04, 4.0: 0.039}}
    s = {0: 1.0, 1: 0.05}
    alloc = greedy_layer_allocation(curves, s, grid, target_mean=3.0)
    assert alloc[0] == 4.0 and alloc[1] == 2.0
    assert sum(alloc.values()) / 2 == 3.0


def test_greedy_layer_allocation_uniform_when_symmetric():
    from experiments.k4_alloc import greedy_layer_allocation

    grid = (2.0, 3.0, 4.0)
    curves = {l: {2.0: 0.4, 3.0: 0.1, 4.0: 0.02} for l in range(4)}  # noqa: E741
    s = {l: 1.0 for l in range(4)}  # noqa: E741
    alloc = greedy_layer_allocation(curves, s, grid, target_mean=3.0)
    assert all(v == 3.0 for v in alloc.values())


def test_uniform_bits_by_layer_non_exact_target():
    from experiments.k4_alloc import _uniform_bits_by_layer

    # (32, 3.2): (0.2*32)=6.4 -> round to 6 -> realized 3.1875; must not crash.
    bits = _uniform_bits_by_layer(32, 3.2)
    assert len(bits) == 32
    realized = sum(bits.values()) / 32
    assert abs(realized - 3.2) <= 1.0 / 32 + 1e-9
    # Exact targets stay exact.
    bits = _uniform_bits_by_layer(12, 2.5)
    assert sum(bits.values()) / 12 == 2.5
