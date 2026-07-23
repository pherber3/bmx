import json

import torch

from bmx.cache.collect import save_cache, to_matrix


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


def _fit_tiny_pack_file(
    cache_path, out_path, *, budgets, group, tiers=(0, 2, 3, 4, 5, 6, 8)
):
    """Minimal single-cache mirror of k4_fit_packs.py's fit+save flow: builds
    one SpectralBasis per layer (identity whitener — no RoPE/model needed) and
    writes a pack file via save_pack_file."""
    from bmx.cache.spectral import fit_spectral_basis, identity_whitener, save_pack_file
    from experiments._k4_common import load_layer_keys

    layer_keys = load_layer_keys(str(cache_path))
    bases = {}
    for layer_i, kinds_map in layer_keys.items():
        M_fit = to_matrix(kinds_map["k_pre"])
        C = M_fit.shape[1]
        Ih, Ih_inv = identity_whitener(C)
        bases[layer_i] = fit_spectral_basis(M_fit, Ih, Ih_inv)
    save_pack_file(out_path, bases, budgets, tiers=tiers, group=group)


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


def test_k4_frontier_emits_v2_and_fullc(tmp_path):
    import pandas as pd

    from experiments.k4_frontier import Config, main

    p = tmp_path / "m.safetensors"
    _tiny_cache(p, seed=0)
    cfg = Config(
        cache_path=str(p),
        model_label="tiny",
        budgets=(1.5,),
        group=16,
        max_layers=1,
        out_root=str(tmp_path / "results"),
    )
    run_dir = main(cfg)
    df = pd.read_parquet(run_dir / "metrics.parquet")
    spec = df[(df.arm == "spectral") & (df.fit_mode == "oracle")]
    assert {"bpe_skeptic_fullc", "bpe_skeptic_deploy_fullc"} <= set(df.columns)
    # v2 <= v1 always; strict where the allocation dropped dirs.
    assert (spec.bpe_skeptic <= spec.bpe_skeptic_fullc + 1e-12).all()
    assert (spec.bpe_skeptic < spec.bpe_skeptic_fullc).any()


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


def test_k4_fit_packs_smoke(tmp_path):
    from bmx.cache.spectral import load_packs
    from experiments.k4_fit_packs import Config, main

    p1, p2 = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    _tiny_cache(p1, seed=0)
    _tiny_cache(p2, seed=1)
    out = tmp_path / "packs.safetensors"
    cfg = Config(
        corpus_cache_paths=(str(p1), str(p2)),
        out_path=str(out),
        model_label="tiny",
        budgets=(2.5,),
        group=16,
    )
    main(cfg)
    packs = load_packs(out, 2.5)
    assert 0 in packs and packs[0].enc.shape == (16, 16)
    import json

    side = json.loads(open(str(out) + ".json").read())
    assert side["w_source"] == "corpus"


def test_k4_fit_packs_alloc_mode(tmp_path):
    import pandas as pd

    from bmx.cache.spectral import load_packs
    from experiments.k4_fit_packs import Config, main

    p1, p2 = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    _tiny_cache(p1, seed=0)
    _tiny_cache(p2, seed=1)

    # k4_alloc Part A schema: kind=="sensitivity" rows carrying layer + s_i.
    sens = pd.DataFrame(
        [
            dict(
                model="tiny",
                layer=layer,
                kind="sensitivity",
                s_i=s_i,
                ppl=1.0,
                sens_bits=2,
                n_prefill=768,
            )
            for layer, s_i in ((0, 0.5), (1, 0.05))  # s_0 = 10 * s_1 > 0
        ]
    )
    sens_path = tmp_path / "sens.parquet"
    sens.to_parquet(sens_path)

    out = tmp_path / "packs.safetensors"
    cfg = Config(
        corpus_cache_paths=(str(p1), str(p2)),
        out_path=str(out),
        model_label="tiny",
        budgets=(2.5,),
        group=16,
        alloc_sens_parquet=str(sens_path),
        out_root=str(tmp_path / "results"),
    )
    main(cfg)

    side = json.loads(open(str(out) + ".json").read())
    # (c) sidecar records the allocation map (written by save_pack_file itself,
    # under its reserved "layer_budgets" key) + the sens provenance.
    assert side["alloc_sens_parquet"] == str(sens_path)
    alloc = side["layer_budgets"]["2.5"]
    assert set(alloc) == {"0", "1"}
    assert set(side["alloc_sensitivities"]) == {"0", "1"}
    # (b) across-layer mean of allocated budgets == target within the greedy
    # step tolerance (k4_alloc's grid_step/n_layer form; grid step = 0.25).
    realized = sum(alloc.values()) / len(alloc)
    assert abs(realized - 2.5) <= 0.25 / 2 + 1e-9
    # (a) the sensitive layer gets strictly more budget, hence strictly more
    # mean per-direction bits (both layers share one spectrum by construction,
    # so any difference is attributable to s alone).
    assert alloc["0"] > alloc["1"]
    packs = load_packs(out, 2.5)
    mean_bits = {i: packs[i].bits.double().mean().item() for i in (0, 1)}
    assert mean_bits[0] > mean_bits[1]
    # (d) the file's bits realize the sidecar allocation: waterfill is
    # budget-feasible (mean payload bits <= that layer's allocated budget).
    for i in (0, 1):
        assert mean_bits[i] <= alloc[str(i)] + 1e-9


def test_k4_fit_packs_default_unchanged(tmp_path):
    from safetensors.torch import load_file

    from bmx.cache.collect import to_matrix
    from bmx.cache.spectral import assemble_whitener, fit_spectral_basis, save_pack_file
    from experiments._k4_common import corpus_query_moment, load_layer_keys, setup_rope
    from experiments.k4_fit_packs import Config, main

    p1, p2 = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    _tiny_cache(p1, seed=0)
    _tiny_cache(p2, seed=1)

    out_main = tmp_path / "packs_main.safetensors"
    cfg = Config(
        corpus_cache_paths=(str(p1), str(p2)),
        out_path=str(out_main),
        model_label="tiny",
        budgets=(2.0, 2.5),
        group=16,
        out_root=str(tmp_path / "results"),
    )
    main(cfg)  # alloc_sens_parquet="" (default): must stay uniform

    # Pre-change reference: the same fit driven directly through the library
    # helpers + save_pack_file with uniform budgets.
    per_cache = [load_layer_keys(str(p)) for p in (p1, p2)]
    layers = sorted(per_cache[0].keys())
    get_cos_sins = [setup_rope("", lk, layers)[1] for lk in per_cache]
    bases = {}
    for layer_i in layers:
        M_fit = torch.cat([to_matrix(lk[layer_i]["k_pre"]) for lk in per_cache], dim=0)
        h_kv, _, d = per_cache[0][layer_i]["k_pre"].shape
        W_blocks = corpus_query_moment(
            per_cache, get_cos_sins, False, layer_i, h_kv, d, cfg.position_stride
        )
        Wh, Wh_inv = assemble_whitener(W_blocks, ridge=cfg.ridge)
        bases[layer_i] = fit_spectral_basis(M_fit, Wh, Wh_inv)
    out_ref = tmp_path / "packs_ref.safetensors"
    save_pack_file(out_ref, bases, cfg.budgets, tiers=cfg.tiers, group=cfg.group)

    t_main = load_file(str(out_main))
    t_ref = load_file(str(out_ref))
    assert set(t_main) == set(t_ref)
    for key in t_main:
        assert torch.equal(t_main[key], t_ref[key]), f"tensor {key} differs"


def test_k4_spectra_w_source_corpus(tmp_path):
    import pandas as pd

    from experiments.k4_spectra import Config, main

    main_p, other_p = tmp_path / "m.safetensors", tmp_path / "o.safetensors"
    _tiny_cache(main_p, seed=0)
    _tiny_cache(other_p, seed=1)
    cfg = Config(
        cache_path=str(main_p),
        corpus_cache_paths=(str(other_p),),
        model_label="tiny",
        budgets=(2.5,),
        group=16,
        w_source="corpus",
        out_root=str(tmp_path / "results"),
    )
    run_dir = main(cfg)
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert (df.w_source == "corpus").all()


def test_k4_charge_curve_smoke(tmp_path):
    import pandas as pd

    from experiments.k4_charge_curve import Config, main

    run = tmp_path / "niah" / "r1"
    run.mkdir(parents=True)
    pd.DataFrame(
        {
            "arm": ["k4_b2.5", "k4_b2.5"],
            "length": [8192, 32768],
            "kv_size_bits": [3.60, 2.69],
        }
    ).to_parquet(run / "metrics.parquet")
    fp = tmp_path / "fit.parquet"
    pd.DataFrame(
        {
            "model": ["m"] * 2,
            "layer": [0, 1],
            "budget": [2.5] * 2,
            "n_zero_dirs": [190, 198],
        }
    ).to_parquet(fp)
    out = tmp_path / "table.md"
    main(
        Config(
            niah_run_dirs=(str(run),),
            fit_packs_parquet=str(fp),
            budgets=(2.5,),
            out_path=str(out),
        )
    )
    text = out.read_text()
    assert "skeptic-v2" in text and "as-measured" in text
    # 8k row, mean n_zero = 194 => C_used = 830:
    # v2 = 3.60 - 16*194/(2*8192) - 0.25*(194/1024)/2 = 3.60 - 0.18945 - 0.02368 = 3.3869
    assert "3.39" in text

    # Belt-and-braces: compute the expected value from skeptic_charge/scale_bits
    # directly (the regression pin) and assert the rendered digits match it —
    # never adjust the literal above to make this pass.
    from bmx.cache.codecs import scale_bits
    from bmx.cache.spectral import skeptic_charge

    C, S, tiers, group = 1024, 8192, (0, 2, 3, 4, 5, 6, 8), 64
    c_used = 1024 - (190 + 198) / 2
    expected = (
        3.60
        - (skeptic_charge(C, S, tiers) - skeptic_charge(C, S, tiers, c_used=c_used)) / 2
        - (scale_bits(group) * (1 - c_used / C)) / 2
    )
    assert f"{expected:.2f}" in text


def test_k4_dec_quant_smoke(tmp_path):
    import json

    import pandas as pd

    from experiments.k4_dec_quant import Config, main

    fit, scored = tmp_path / "f.safetensors", tmp_path / "s.safetensors"
    _tiny_cache(fit, seed=0)
    _tiny_cache(scored, seed=1)
    packs_path = tmp_path / "packs.safetensors"
    _fit_tiny_pack_file(
        fit, packs_path, budgets=(2.5,), group=16
    )  # helper mirroring k4_fit_packs
    run_dir = main(
        Config(
            pack_path=str(packs_path),
            cache_paths=(str(scored),),
            model_label="tiny",
            budgets=(2.5,),
            out_root=str(tmp_path / "results"),
        )
    )
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert set(df.dec_mode) == {"fp32", "fp16", "int8"}
    v = json.loads((run_dir / "dec_quant_verdict.json").read_text())
    assert "gate_pass" in v and "rel_degradation_int8" in str(v)
