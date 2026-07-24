import json
import math

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
    assert {"bpe_skeptic_fullc", "bpe_skeptic_deploy_fullc", "c_used"} <= set(
        df.columns
    )
    # v2 <= v1 always; strict where the allocation dropped dirs.
    assert (spec.bpe_skeptic <= spec.bpe_skeptic_fullc + 1e-12).all()
    assert (spec.bpe_skeptic < spec.bpe_skeptic_fullc).any()

    # Exact pin: bpe_skeptic_fullc must carry the TRUE pre-2026-07-23
    # payload-v1 + charge-v1 value, not a payload-v2 + charge-v1 hybrid.
    # For every spectral row, bpe_skeptic_fullc - bpe_skeptic must equal the
    # sum of the charge-side v1-vs-v2 delta and the payload-side v1-vs-v2
    # delta, both computed from that row's own pack c_used.
    from bmx.cache.codecs import scale_bits
    from bmx.cache.spectral import skeptic_charge

    S, C_row, group = (
        128,
        16,
        cfg.group,
    )  # _tiny_cache defaults (S=128, h_kv=2*d=8=>C=16)
    for _, row in spec.iterrows():
        cu = float(row.c_used)
        expected_delta = (
            skeptic_charge(C_row, S, cfg.tiers)
            - skeptic_charge(C_row, S, cfg.tiers, c_used=cu)
        ) + scale_bits(group) * (1 - cu / C_row)
        actual_delta = float(row.bpe_skeptic_fullc) - float(row.bpe_skeptic)
        assert abs(actual_delta - expected_delta) < 1e-9, (
            f"{actual_delta} != {expected_delta} (c_used={cu})"
        )


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


def test_corpus_fit_bases_hoist_pin(tmp_path):
    """K4 local-levers Task 4 hoist: `per_cache_weighted_moments` factors the
    per-cache-matrices + whitener assembly out of `corpus_fit_bases`. Pin two
    things exactly: (1) `corpus_fit_bases`'s bases/M_fits/whiteners are
    UNCHANGED versus a reference built directly on top of the hoisted helper
    (round-trips corpus_fit_bases through the same code path it now calls
    internally -- any accidental reassociation would show up here); (2) the
    hoisted helper's own contract -- the pooled second moment (mean of
    to_matrix(k_pre) concatenated across equal-S caches) equals the exact
    mean of the per-cache second moments (licenses the experiment's Σ̄ = mean_s
    Σ_s convention)."""
    from bmx.cache.spectral import fit_spectral_basis, key_second_moment
    from experiments._k4_common import (
        corpus_fit_bases,
        load_layer_keys,
        per_cache_weighted_moments,
        setup_rope,
    )

    p1, p2, p3 = (tmp_path / f"{n}.safetensors" for n in ("a", "b", "c"))
    for p, seed in ((p1, 0), (p2, 1), (p3, 2)):
        _tiny_cache(p, seed=seed)  # equal S across caches (S=128 default)

    per_cache = [load_layer_keys(str(p)) for p in (p1, p2, p3)]
    layers = sorted(per_cache[0].keys())
    get_cos_sins = [setup_rope("", lk, layers)[1] for lk in per_cache]

    ref = corpus_fit_bases(
        per_cache,
        get_cos_sins,
        rope_ready=False,
        layers=layers,
        w_source="corpus",
        ridge=1e-3,
        position_stride=8,
    )

    for layer_i in layers:
        pcm = per_cache_weighted_moments(
            per_cache,
            get_cos_sins,
            rope_ready=False,
            layer_i=layer_i,
            w_source="corpus",
            ridge=1e-3,
            position_stride=8,
        )
        # (1) corpus_fit_bases's basis == fitting directly on the hoisted
        # helper's M_parts/whitener (exact tensor equality, not tolerance).
        M_fit_direct = torch.cat(pcm.M_parts, dim=0)
        assert torch.equal(M_fit_direct, ref.M_fits[layer_i])
        basis_direct = fit_spectral_basis(M_fit_direct, pcm.Wh, pcm.Wh_inv)
        assert torch.equal(basis_direct.enc, ref.bases[layer_i].enc)
        assert torch.equal(basis_direct.dec, ref.bases[layer_i].dec)
        assert torch.equal(basis_direct.lam, ref.bases[layer_i].lam)
        Wh_ref, Wh_inv_ref = ref.whiteners[layer_i]
        assert torch.equal(pcm.Wh, Wh_ref)
        assert torch.equal(pcm.Wh_inv, Wh_inv_ref)

        # (2) pooled second moment == exact mean of per-cache second moments
        # (equal-S caches here, matching the real gpt2_1024 fleet: every
        # cache_paths entry is a 1024-token window).
        per_cache_sigmas = [key_second_moment(m) for m in pcm.M_parts]
        pooled_direct = key_second_moment(M_fit_direct)
        pooled_mean = sum(per_cache_sigmas) / len(per_cache_sigmas)
        assert torch.allclose(pooled_direct, pooled_mean, atol=1e-10)


def test_k4_fit_packs_tier_counts(tmp_path):
    """Metrics rows gain n_t0,n_t2,n_t3,n_t4,n_t5,n_t6,n_t8 per (layer,
    budget): counts of pack.bits == that tier. Sanity: they sum to C, and
    n_t0 == n_zero_dirs (additive schema; existing columns unchanged)."""
    import pandas as pd

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
        out_root=str(tmp_path / "results"),
    )
    run_dir = main(cfg)
    df = pd.read_parquet(run_dir / "metrics.parquet")

    tier_cols = ["n_t0", "n_t2", "n_t3", "n_t4", "n_t5", "n_t6", "n_t8"]
    assert set(tier_cols) <= set(df.columns)
    # Existing columns must survive untouched.
    assert {"model", "layer", "budget", "am_gm", "top16_energy", "n_zero_dirs"} <= set(
        df.columns
    )

    packs = load_packs(out, 2.5)
    for _, row in df.iterrows():
        layer_i = int(row["layer"])
        C = packs[layer_i].bits.shape[0]
        counts = [int(row[c]) for c in tier_cols]
        assert sum(counts) == C, f"layer {layer_i}: tier counts {counts} != C={C}"
        assert row["n_t0"] == row["n_zero_dirs"]
        # Cross-check against the pack's own bits tensor directly.
        bits = packs[layer_i].bits
        for tier, col in zip((0, 2, 3, 4, 5, 6, 8), tier_cols):
            assert int(row[col]) == int((bits == tier).sum())


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


def _charge_curve_fixture(tmp_path):
    """Shared fixture for the int8_frac_variants tests below — same shape as
    test_k4_charge_curve_smoke's, factored out so both variant tests share it."""
    import pandas as pd

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
    return run, fp


def test_k4_charge_curve_int8_frac_1_matches_blanket_int8(tmp_path):
    """frac=1.0 must reproduce the existing dec_bits=8.0 (skeptic-v2-int8)
    column EXACTLY — mixed_dec_charge(c_int8=c_used) == skeptic_charge
    (dec_bits=8.0) is endpoint-pinned, so this must hold through the full
    corrected() blend, not just at the spectral.py level."""
    from experiments.k4_charge_curve import Config, main

    run, fp = _charge_curve_fixture(tmp_path)

    out_blanket = tmp_path / "blanket.md"
    main(
        Config(
            niah_run_dirs=(str(run),),
            fit_packs_parquet=str(fp),
            budgets=(2.5,),
            dec_bits_variants=(16.0, 8.0),
            out_path=str(out_blanket),
        )
    )
    out_frac = tmp_path / "frac.md"
    main(
        Config(
            niah_run_dirs=(str(run),),
            fit_packs_parquet=str(fp),
            budgets=(2.5,),
            dec_bits_variants=(),
            int8_frac_variants=(1.0,),
            out_path=str(out_frac),
        )
    )

    blanket_text = out_blanket.read_text()
    frac_text = out_frac.read_text()

    # Pull the numeric cells for the k4_b2.5/8192 row out of each table's
    # last column and assert they match exactly (same measured input, same
    # blend, only the accounting path to the decoder charge differs).
    def _last_cell(text: str, needle: str) -> str:
        for line in text.splitlines():
            if line.startswith(f"| {needle} |"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                return cells[-1]
        raise AssertionError(f"row {needle!r} not found in:\n{text}")

    blanket_cell = _last_cell(blanket_text, "k4_b2.5 | 8192")
    frac_cell = _last_cell(frac_text, "k4_b2.5 | 8192")
    assert blanket_cell == frac_cell


def test_k4_charge_curve_int8_frac_between_endpoints(tmp_path):
    """A frac of 0.9 must sit strictly between the dec_bits=16.0 (v2) and
    dec_bits=8.0 (v2-int8) columns for a k4 arm row (the mix is a strict
    convex combination when 0 < frac < 1 and c_used > 0)."""
    from experiments.k4_charge_curve import Config, main

    run, fp = _charge_curve_fixture(tmp_path)

    out = tmp_path / "table.md"
    main(
        Config(
            niah_run_dirs=(str(run),),
            fit_packs_parquet=str(fp),
            budgets=(2.5,),
            dec_bits_variants=(16.0, 8.0),
            int8_frac_variants=(0.9,),
            out_path=str(out),
        )
    )
    text = out.read_text()
    assert "int8frac0.9" in text

    for line in text.splitlines():
        if line.startswith("| k4_b2.5 | 8192 |"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            # columns: arm, length, v1, skeptic-v2 (16.0), skeptic-v2-int8
            # (8.0), skeptic-v2-int8frac0.9
            v2, v2_int8, v2_frac = (float(cells[3]), float(cells[4]), float(cells[5]))
            assert v2_int8 < v2_frac < v2
            break
    else:
        raise AssertionError(f"row not found in:\n{text}")


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


def test_k4_dec_quant_two_caches_keys_tq_curve_per_cache(tmp_path):
    import json

    import pandas as pd

    from experiments.k4_dec_quant import Config, main

    fit = tmp_path / "f.safetensors"
    scored_a = tmp_path / "s_a.safetensors"
    scored_b = tmp_path / "s_b.safetensors"
    _tiny_cache(fit, seed=0)
    _tiny_cache(scored_a, seed=1)
    _tiny_cache(scored_b, seed=2)  # different seed => different distortions
    packs_path = tmp_path / "packs.safetensors"
    _fit_tiny_pack_file(fit, packs_path, budgets=(2.5,), group=16)
    run_dir = main(
        Config(
            pack_path=str(packs_path),
            cache_paths=(str(scored_a), str(scored_b)),
            model_label="tiny",
            budgets=(2.5,),
            out_root=str(tmp_path / "results"),
        )
    )
    v = json.loads((run_dir / "dec_quant_verdict.json").read_text())
    assert "gate_pass" in v
    assert math.isfinite(v["per_budget"]["2.5"]["win_fp16"])

    tq_df = pd.read_parquet(run_dir / "tq_curve.parquet")
    # One curve row per (cache, layer, tq_bit) — no exact-duplicate rows from
    # being recomputed once per budget (budgets=(2.5,) has only one budget
    # here, so re-run with two budgets to actually exercise the hoist).
    assert not tq_df.duplicated(subset=["cache", "layer", "bpe_model"]).any()

    # Per-cache curves must differ (different seeds => different distortions):
    # same (layer, bpe_model) turboquant_mse point scored against each
    # cache's own k_pre should give different logit distortion.
    piv = tq_df.pivot_table(
        index=["layer", "bpe_model"], columns="cache", values="logit"
    )
    assert (piv.iloc[:, 0] != piv.iloc[:, 1]).any()

    # Re-run with two budgets (pack file fit for both) to confirm the TQ pass
    # is hoisted out of the budget loop: exact-duplicate (cache, layer,
    # bpe_model) rows must not appear (previously one identical row was
    # written per budget).
    packs_path2 = tmp_path / "packs2.safetensors"
    _fit_tiny_pack_file(fit, packs_path2, budgets=(2.0, 2.5), group=16)
    run_dir2 = main(
        Config(
            pack_path=str(packs_path2),
            cache_paths=(str(scored_a), str(scored_b)),
            model_label="tiny",
            budgets=(2.0, 2.5),
            out_root=str(tmp_path / "results2"),
        )
    )
    tq_df2 = pd.read_parquet(run_dir2 / "tq_curve.parquet")
    assert not tq_df2.duplicated(subset=["cache", "layer", "bpe_model"]).any()


def test_k4_dec_quant_tier_thresholds_run_and_gate_bind_on_t5(tmp_path):
    """K4 local-levers Task 2: with tier_thresholds set, the run emits
    int8_t{T} dec_mode rows, a cert_vs_measured.parquet, and the verdict
    reports rel_degradation_int8_t5 as a gating field (blanket still
    reported, not gating)."""
    import pandas as pd

    from experiments.k4_dec_quant import Config, main

    fit, scored = tmp_path / "f.safetensors", tmp_path / "s.safetensors"
    _tiny_cache(fit, seed=0)
    _tiny_cache(scored, seed=1)
    packs_path = tmp_path / "packs.safetensors"
    _fit_tiny_pack_file(fit, packs_path, budgets=(2.5,), group=16)
    run_dir = main(
        Config(
            pack_path=str(packs_path),
            cache_paths=(str(scored),),
            model_label="tiny",
            budgets=(2.5,),
            tier_thresholds=(4, 5, 6),
            out_root=str(tmp_path / "results"),
        )
    )
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert set(df.dec_mode) == {"fp32", "fp16", "int8", "int8_t4", "int8_t5", "int8_t6"}

    v = json.loads((run_dir / "dec_quant_verdict.json").read_text())
    pb = v["per_budget"]["2.5"]
    assert "rel_degradation_int8_t5" in pb
    assert "rel_degradation_int8" in pb  # blanket still reported
    assert "ordering_ok" in pb
    assert "gate_pass" in v

    cvm = pd.read_parquet(run_dir / "cert_vs_measured.parquet")
    assert set(cvm.tier_threshold.unique()) == {4, 5, 6, 8}
    for col in (
        "model",
        "cache",
        "budget",
        "layer",
        "tier_threshold",
        "implied_rel_degradation",
        "measured_rel_deg",
        "frac_int8",
        "c_used",
        "c_int8",
    ):
        assert col in cvm.columns


def test_k4_dec_quant_int8_tl_mode(tmp_path):
    """K4 estimation-levers Task 3: dec_tl=True adds the 'int8_tl' dec_mode,
    reported (rel_degradation_int8_tl) but not the binding gate mode -- the
    blanket 'int8' (no tier_thresholds set here) stays gate_mode."""
    import pandas as pd

    from experiments.k4_dec_quant import Config, main

    fit, scored = tmp_path / "f.safetensors", tmp_path / "s.safetensors"
    _tiny_cache(fit, seed=0)
    _tiny_cache(scored, seed=1)
    packs_path = tmp_path / "packs.safetensors"
    _fit_tiny_pack_file(fit, packs_path, budgets=(2.5,), group=16)
    run_dir = main(
        Config(
            pack_path=str(packs_path),
            cache_paths=(str(scored),),
            model_label="tiny",
            budgets=(2.5,),
            dec_tl=True,
            out_root=str(tmp_path / "results"),
        )
    )
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert set(df.dec_mode) == {"fp32", "fp16", "int8", "int8_tl"}
    assert df[df.dec_mode == "int8_tl"].bpe_skeptic_deploy.notna().all()

    v = json.loads((run_dir / "dec_quant_verdict.json").read_text())
    pb = v["per_budget"]["2.5"]
    assert "rel_degradation_int8_tl" in pb
    assert "rel_degradation_int8" in pb
    assert v["gate_mode"] == "int8"  # no tier_thresholds -> blanket still gates
    assert pb["gate_mode"] == "int8"
    assert "gate_pass" in v


def test_k4_dec_quant_int8_tl_and_tier_thresholds_together(tmp_path):
    """dec_tl=True combined with tier_thresholds: int8_tl is reported
    alongside int8_t{T}, and the gate still binds on int8_t5 (dec_tl never
    changes gate selection)."""
    import pandas as pd

    from experiments.k4_dec_quant import Config, main

    fit, scored = tmp_path / "f.safetensors", tmp_path / "s.safetensors"
    _tiny_cache(fit, seed=0)
    _tiny_cache(scored, seed=1)
    packs_path = tmp_path / "packs.safetensors"
    _fit_tiny_pack_file(fit, packs_path, budgets=(2.5,), group=16)
    run_dir = main(
        Config(
            pack_path=str(packs_path),
            cache_paths=(str(scored),),
            model_label="tiny",
            budgets=(2.5,),
            tier_thresholds=(4, 5, 6),
            dec_tl=True,
            out_root=str(tmp_path / "results"),
        )
    )
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert set(df.dec_mode) == {
        "fp32",
        "fp16",
        "int8",
        "int8_t4",
        "int8_t5",
        "int8_t6",
        "int8_tl",
    }
    v = json.loads((run_dir / "dec_quant_verdict.json").read_text())
    assert v["gate_mode"] == "int8_t5"
    pb = v["per_budget"]["2.5"]
    assert "rel_degradation_int8_tl" in pb
    assert "rel_degradation_int8_t5" in pb


def test_dec_quant_verdict_gate_binds_on_int8_t5_when_present():
    """Synthetic frame: blanket int8 fails 5% but int8_t5 passes -> gate_pass
    True (binding moves off the blanket mode). Then flip so t5 fails ->
    gate_pass False, even though nothing else changed."""
    import pandas as pd

    from experiments.k4_dec_quant import Config, _dec_quant_verdict

    def _frame(win_int8, win_int8_t5):
        # tq curve: flat at distortion=1.0 across bpe 0..100 so tq_dist==1.0
        # for every interpolation, making win == 1/dist directly controllable.
        tq_rows = [
            dict(
                model="t",
                cache="c",
                layer=0,
                kind="k_pre",
                arm="turboquant_mse",
                dec_mode="",
                budget=float("nan"),
                bpe_model=bpe,
                bpe_skeptic_deploy=bpe,
                rel_fro=0.0,
                logit=1.0,
                logit_rope=1.0,
            )
            for bpe in (0.0, 100.0)
        ]
        modes = {
            "fp32": 1.0,
            "fp16": 1.0,
            "int8": 1.0 / win_int8,
            "int8_t4": 1.0 / win_int8_t5 * 1.5,  # arbitrary, not gated
            "int8_t5": 1.0 / win_int8_t5,
            "int8_t6": 1.0 / win_int8_t5 * 0.9,  # arbitrary, not gated
        }
        rows = [
            dict(
                model="t",
                cache="c",
                layer=0,
                kind="k_pre",
                arm="spectral",
                dec_mode=mode,
                budget=2.5,
                bpe_model=10.0,
                bpe_skeptic_deploy=10.0,
                rel_fro=0.0,
                logit=dist,
                logit_rope=dist,
            )
            for mode, dist in modes.items()
        ]
        return pd.DataFrame(rows), pd.DataFrame(tq_rows)

    cfg = Config(
        pack_path="unused",
        cache_paths=("unused",),
        budgets=(2.5,),
        tier_thresholds=(4, 5, 6),
    )

    # blanket int8 fails (rel_degradation_int8 = 1 - 0.5/1.0 = 0.5 >> 5%);
    # int8_t5 passes (rel_degradation = 1 - 0.99 = 0.01 < 5%).
    df, tq_df = _frame(win_int8=0.5, win_int8_t5=0.99)
    v = _dec_quant_verdict(df, tq_df, "logit", cfg)
    pb = v["per_budget"]["2.5"]
    assert pb["rel_degradation_int8"] > 0.05  # blanket would have failed
    assert pb["rel_degradation_int8_t5"] < 0.05
    assert v["gate_pass"] is True

    # Now int8_t5 also fails -> gate_pass must flip False even if blanket
    # (no longer binding) were to pass.
    df2, tq_df2 = _frame(win_int8=0.99, win_int8_t5=0.5)
    v2 = _dec_quant_verdict(df2, tq_df2, "logit", cfg)
    pb2 = v2["per_budget"]["2.5"]
    assert pb2["rel_degradation_int8"] < 0.05  # blanket passes here
    assert pb2["rel_degradation_int8_t5"] > 0.05
    assert v2["gate_pass"] is False


def test_dec_quant_verdict_empty_tier_thresholds_preserves_blanket_gate():
    """tier_thresholds=() (default): no int8_t{T} modes are present, and
    gate_pass binds on the blanket rel_degradation_int8 exactly as before
    Task 2 (today's behavior, unchanged)."""
    import pandas as pd

    from experiments.k4_dec_quant import Config, _dec_quant_verdict

    tq_rows = [
        dict(
            model="t",
            cache="c",
            layer=0,
            kind="k_pre",
            arm="turboquant_mse",
            dec_mode="",
            budget=float("nan"),
            bpe_model=bpe,
            bpe_skeptic_deploy=bpe,
            rel_fro=0.0,
            logit=1.0,
            logit_rope=1.0,
        )
        for bpe in (0.0, 100.0)
    ]
    modes = {"fp32": 1.0, "fp16": 1.0, "int8": 1.0 / 0.5}  # win_int8 = 0.5 -> fails
    rows = [
        dict(
            model="t",
            cache="c",
            layer=0,
            kind="k_pre",
            arm="spectral",
            dec_mode=mode,
            budget=2.5,
            bpe_model=10.0,
            bpe_skeptic_deploy=10.0,
            rel_fro=0.0,
            logit=dist,
            logit_rope=dist,
        )
        for mode, dist in modes.items()
    ]
    df = pd.DataFrame(rows)
    tq_df = pd.DataFrame(tq_rows)
    cfg = Config(
        pack_path="unused", cache_paths=("unused",), budgets=(2.5,), tier_thresholds=()
    )
    v = _dec_quant_verdict(df, tq_df, "logit", cfg)
    pb = v["per_budget"]["2.5"]
    assert pb["rel_degradation_int8"] > 0.05
    assert "rel_degradation_int8_t5" not in pb
    assert v["gate_pass"] is False  # blanket still binds when no tier modes present


def test_dec_quant_deploy_bpe_endpoints_match_old_skeptic_charge():
    """mixed_dec_charge-computed bpe_skeptic_deploy for fp32/fp16/int8 modes
    must equal bpe_model + skeptic_charge(..., dec_bits=16 or 8) exactly —
    the old formula, on a small real fitted pack."""
    from bmx.cache.spectral import mixed_dec_charge, skeptic_charge

    C, S = 16, 200
    scales = torch.linspace(0.2, 5.0, C, dtype=torch.float64)
    Wh, Wh_inv = torch.diag(scales), torch.diag(1.0 / scales)
    g = torch.Generator().manual_seed(0)
    M = torch.randn(S, C, generator=g)

    from bmx.cache.spectral import fit_spectral_pack

    pack = fit_spectral_pack(M, Wh, Wh_inv, 2.5, tiers=(0, 2, 3, 4, 5, 6, 8), group=8)
    deploy_s = 32768
    c_used = pack.c_used
    c_int8_blanket = int(((pack.bits > 0) & (pack.bits <= 8)).sum())
    assert c_int8_blanket == c_used  # blanket covers every used column

    old_fp16 = skeptic_charge(C, deploy_s, pack.tiers, c_used=c_used, dec_bits=16.0)
    old_int8 = skeptic_charge(C, deploy_s, pack.tiers, c_used=c_used, dec_bits=8.0)

    new_fp32 = mixed_dec_charge(C, deploy_s, pack.tiers, c_used=c_used, c_int8=0)
    new_fp16 = mixed_dec_charge(C, deploy_s, pack.tiers, c_used=c_used, c_int8=0)
    new_int8 = mixed_dec_charge(
        C, deploy_s, pack.tiers, c_used=c_used, c_int8=c_int8_blanket
    )

    assert abs(new_fp32 - old_fp16) < 1e-12
    assert abs(new_fp16 - old_fp16) < 1e-12
    assert abs(new_int8 - old_int8) < 1e-12


def test_dec_quant_measured_rel_deg_matches_hand_computed():
    """Per-layer measured_rel_deg in cert_vs_measured must equal
    1 - win_T(layer)/win_fp16(layer) hand-computed from the same rows the
    gate uses (single-layer synthetic frame, exact arithmetic check)."""
    import pandas as pd

    from bmx.cache.spectral import fit_spectral_pack, identity_whitener
    from experiments.k4_dec_quant import _cert_vs_measured_rows

    # dist chosen so win_fp16 = 1/0.4 = 2.5, win_t5 = 1/0.5 = 2.0
    # -> measured_rel_deg = 1 - 2.0/2.5 = 0.2 exactly.
    modes = {"fp32": 0.4, "fp16": 0.4, "int8": 0.4, "int8_t5": 0.5}
    rows = [
        dict(
            model="t",
            cache="c",
            layer=0,
            kind="k_pre",
            arm="spectral",
            dec_mode=mode,
            budget=2.5,
            bpe_model=10.0,
            bpe_skeptic_deploy=10.0,
            rel_fro=0.0,
            logit=dist,
            logit_rope=dist,
        )
        for mode, dist in modes.items()
    ]
    df = pd.DataFrame(rows)

    C = 16
    Ih, Ih_inv = identity_whitener(C)
    g = torch.Generator().manual_seed(0)
    M = torch.randn(200, C, generator=g)
    pack = fit_spectral_pack(M, Ih, Ih_inv, 2.5, tiers=(0, 2, 3, 4, 5, 6, 8), group=8)

    tq_curves = {"c": {0: [(0.0, 1.0), (100.0, 1.0)]}}
    cvm_rows = _cert_vs_measured_rows(
        df=df,
        tq_curves=tq_curves,
        headline_col="logit",
        budget=2.5,
        packs_by_cache={"c": {0: pack}},
        tier_thresholds_incl_blanket=(5,),
    )
    row = next(r for r in cvm_rows if r["tier_threshold"] == 5 and r["layer"] == 0)
    assert abs(row["measured_rel_deg"] - 0.2) < 1e-9


def test_dec_quant_ordering_ok():
    """ordering_ok: True when measured rel_degradation is monotone
    nondecreasing T4 <= T5 <= T6 <= blanket; False when violated."""
    from experiments.k4_dec_quant import _ordering_ok

    assert _ordering_ok(
        {"int8_t4": 0.01, "int8_t5": 0.02, "int8_t6": 0.03, "int8": 0.16}
    )
    assert _ordering_ok({"int8_t4": 0.0, "int8_t5": 0.0, "int8_t6": 0.0, "int8": 0.0})
    assert not _ordering_ok(
        {"int8_t4": 0.05, "int8_t5": 0.02, "int8_t6": 0.03, "int8": 0.16}
    )
    assert not _ordering_ok(
        {"int8_t4": 0.01, "int8_t5": 0.02, "int8_t6": 0.20, "int8": 0.16}
    )
    # missing modes (e.g. tier_thresholds empty) -> vacuously True
    assert _ordering_ok({})


def test_corpus_fit_bases_matches_direct_fit(tmp_path):
    from bmx.cache.spectral import fit_spectral_basis, identity_whitener
    from experiments._k4_common import corpus_fit_bases, load_layer_keys, setup_rope

    p1, p2 = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    _tiny_cache(p1, seed=0)
    _tiny_cache(p2, seed=1)
    per_cache = [load_layer_keys(str(p)) for p in (p1, p2)]
    layers = sorted(per_cache[0].keys())
    get_cos_sins = [setup_rope("", lk, layers)[1] for lk in per_cache]

    fit = corpus_fit_bases(
        per_cache,
        get_cos_sins,
        False,
        layers,
        w_source="none",
        ridge=1e-3,
        position_stride=8,
    )
    M_ref = torch.cat([to_matrix(lk[0]["k_pre"]) for lk in per_cache], dim=0)
    Ih, Ih_inv = identity_whitener(M_ref.shape[1])
    ref = fit_spectral_basis(M_ref, Ih, Ih_inv)
    assert torch.equal(fit.bases[0].enc, ref.enc)
    assert torch.equal(fit.bases[0].dec, ref.dec)
    assert torch.equal(fit.M_fits[0], M_ref)
    assert torch.equal(fit.whiteners[0][0], Ih)


def test_layer_ctx_importable_from_k4_common(tmp_path):
    from experiments._k4_common import _layer_ctx, load_layer_keys

    p = tmp_path / "c.safetensors"
    _tiny_cache(p, seed=0)
    lk = load_layer_keys(str(p))
    ctx = _layer_ctx(lk[0], rope_ready=False, get_cos_sin=lambda S: None)
    assert ctx.C == ctx.h_kv * ctx.d == 16
    assert ctx.M_pre.shape == (128, 16)
    assert ctx.tail == slice(64, 128)
    assert ctx.K_post_true is None


def _tiny_corpus_transfer_cfg(tmp_path, budgets=(2.5,)):
    """Matched fit budgets (binding decision 1): 2 fit slices per corpus,
    1 eval cache per side, all S=128/C=16 tiny caches."""
    from experiments.k4_corpus_transfer import Config

    paths, seed = {}, 0
    for name, n in (("wf", 2), ("cf", 2), ("nf", 2), ("we", 1), ("ce", 1)):
        group = []
        for j in range(n):
            p = tmp_path / f"{name}{j}.safetensors"
            _tiny_cache(p, seed=seed)
            seed += 1
            group.append(str(p))
        paths[name] = tuple(group)
    return Config(
        wiki_fit_paths=paths["wf"],
        code_fit_paths=paths["cf"],
        null_fit_paths=paths["nf"],
        wiki_eval_paths=paths["we"],
        code_eval_paths=paths["ce"],
        model_label="tiny",
        budgets=budgets,
        group=16,
        overlap_ranks=(4, 8),
        out_root=str(tmp_path / "results"),
    )


def test_k4_corpus_transfer_smoke(tmp_path):
    import pandas as pd

    from experiments.k4_corpus_transfer import main

    run_dir = main(_tiny_corpus_transfer_cfg(tmp_path))
    df = pd.read_parquet(run_dir / "metrics.parquet")
    spec = df[df.arm == "spectral"]
    assert set(spec.fit_corpus) == {"wiki", "code", "null"}
    assert set(spec.eval_corpus) == {"wiki", "code"}
    # plain cells: fit==w==alloc corpus
    assert (spec.w_corpus == spec.fit_corpus).all()
    assert (spec.alloc_corpus == spec.fit_corpus).all()
    tq = pd.read_parquet(run_dir / "tq_curve.parquet")
    assert (tq.arm == "turboquant_mse").all()

    v = json.loads((run_dir / "corpus_transfer_verdict.json").read_text())
    assert "gpt2_yellow_flag" in v and "verdict_rule" in v
    pb = v["per_budget"]["2.5"]
    assert set(pb["D"]) == {"code->wiki", "null->wiki", "wiki->code", "null->code"}
    for cell in pb["D"].values():
        assert cell["label"] in ("insensitive", "domain-sensitive", "as-measured")
        assert cell["min"] <= cell["mean"] <= cell["max"]
    assert isinstance(pb["model_intrinsic_flag"], bool)
    assert pb["synthesis"] == {}  # no §3b arms provided => empty block


def _tiny_corpus_transfer_cfg_synth(tmp_path, budgets=(2.5,)):
    """Stage-1 tiny cfg + the five §3b synthesis fit arms (2 slices each —
    matched with the natural corpora's 2 slices, binding decision 1)."""
    import dataclasses

    cfg = _tiny_corpus_transfer_cfg(tmp_path, budgets=budgets)
    paths, seed = {}, 100
    for name in ("sc", "uw", "uc", "bw", "bc"):
        group = []
        for j in range(2):
            p = tmp_path / f"{name}{j}.safetensors"
            _tiny_cache(p, seed=seed)
            seed += 1
            group.append(str(p))
        paths[name] = tuple(group)
    return dataclasses.replace(
        cfg,
        shufcode_fit_paths=paths["sc"],
        uniwiki_fit_paths=paths["uw"],
        unicode_fit_paths=paths["uc"],
        biwiki_fit_paths=paths["bw"],
        bicode_fit_paths=paths["bc"],
    )


def test_k4_corpus_transfer_synthesis_smoke(tmp_path):
    import pandas as pd

    from experiments.k4_corpus_transfer import main

    run_dir = main(_tiny_corpus_transfer_cfg_synth(tmp_path))
    df = pd.read_parquet(run_dir / "metrics.parquet")
    spec = df[df.arm == "spectral"]
    synth = ("shufcode", "uniwiki", "unicode", "biwiki", "bicode")
    assert set(spec.fit_corpus) == {"wiki", "code", "null", *synth}
    for c in synth:
        sub = spec[spec.fit_corpus == c]
        # fit-side-only arms, scored on BOTH eval sides in the same run
        assert set(sub.eval_corpus) == {"wiki", "code"}
        assert (sub.w_corpus == c).all() and (sub.alloc_corpus == c).all()

    v = json.loads((run_dir / "corpus_transfer_verdict.json").read_text())
    pb = v["per_budget"]["2.5"]
    assert {
        "uniwiki->wiki",
        "unicode->code",
        "biwiki->wiki",
        "bicode->code",
        "shufcode->code",
        "shufcode->wiki",
    } <= set(pb["D"])
    rules = pb["synthesis"]["rules"]
    assert set(rules) == {"wiki", "code"}
    for eval_c, r in rules.items():
        assert isinstance(r["recipe_confirmed"], bool)
        assert isinstance(r["order2_earns_keep"], bool)
        # pre-registered §3b rules (a)/(b), recomputable from the same JSON
        assert r["recipe_confirmed"] == (r["D_uni"] < 0.10)
        assert r["order2_earns_keep"] == ((r["D_uni"] - r["D_bi"]) >= 0.5 * r["D_uni"])
        assert r["D_shuf"] is not None
    assert pb["synthesis"]["climb_to_order3"] == all(
        r["order2_earns_keep"] for r in rules.values()
    )


def test_k4_corpus_transfer_synthesis_guards(tmp_path):
    import dataclasses

    import pytest

    from experiments.k4_corpus_transfer import main

    cfg = _tiny_corpus_transfer_cfg_synth(tmp_path)
    # partial provision refuses to run (the order ladder needs all five arms)
    with pytest.raises(AssertionError, match="all-or-nothing"):
        main(dataclasses.replace(cfg, bicode_fit_paths=()))
    # the matched-fit-budget guard extends over the synthesis arms
    with pytest.raises(AssertionError, match="matched fit"):
        main(dataclasses.replace(cfg, uniwiki_fit_paths=cfg.uniwiki_fit_paths[:1]))


def test_k4_corpus_transfer_matched_fit_budget_guard(tmp_path):
    import dataclasses

    import pytest

    from experiments.k4_corpus_transfer import main

    cfg = _tiny_corpus_transfer_cfg(tmp_path)
    cfg = dataclasses.replace(cfg, code_fit_paths=cfg.code_fit_paths[:1])
    with pytest.raises(AssertionError, match="matched fit"):
        main(cfg)


def test_k4_corpus_transfer_hybrid_and_wcross(tmp_path):
    import pandas as pd

    from experiments.k4_corpus_transfer import main

    run_dir = main(_tiny_corpus_transfer_cfg(tmp_path))
    df = pd.read_parquet(run_dir / "metrics.parquet")

    hyb = df[df.arm == "spectral_hybrid"]
    assert set(zip(hyb.fit_corpus, hyb.alloc_corpus, hyb.eval_corpus)) == {
        ("wiki", "code", "code"),
        ("code", "wiki", "wiki"),
    }
    assert (hyb.w_corpus == hyb.fit_corpus).all()

    wx = df[df.arm == "spectral_wcross"]
    assert set(zip(wx.fit_corpus, wx.w_corpus)) == {("wiki", "code"), ("code", "wiki")}
    assert set(wx.eval_corpus) == {"wiki", "code"}
    assert (wx.alloc_corpus == wx.fit_corpus).all()

    v = json.loads((run_dir / "corpus_transfer_verdict.json").read_text())
    pb = v["per_budget"]["2.5"]
    assert set(pb["hybrid"]) == {"basis_wiki_alloc_code", "basis_code_alloc_wiki"}
    for h in pb["hybrid"].values():
        assert "recovery" in h and isinstance(h["h3_pass"], bool)
    assert len(pb["wcross"]) == 4  # 2 directions × 2 eval sides


def test_k4_corpus_transfer_diagnostics(tmp_path):
    import pandas as pd

    from experiments.k4_corpus_transfer import main

    run_dir = main(_tiny_corpus_transfer_cfg(tmp_path))
    ov = pd.read_parquet(run_dir / "overlap.parquet")
    assert {"overlap", "tier_agreement", "zero_jaccard", "spectrum", "xretention"} <= (
        set(ov.kind)
    )

    o = ov[ov.kind == "overlap"]
    assert set(o.pair) == {"wiki-code", "wiki-null", "code-null"}
    assert set(o["rank"]) == {4, 8}  # cfg.overlap_ranks, both <= C=16
    assert set(o.centered) == {True, False}
    assert ((o.value >= -1e-9) & (o.value <= 1 + 1e-9)).all()

    t = ov[ov.kind == "tier_agreement"]
    assert ((t.value >= 0) & (t.value <= 1)).all()
    assert (t.tier >= 0).all()

    s = ov[ov.kind == "spectrum"]
    assert set(s.corpus) == {"wiki", "code", "null"}
    # descending spectra per (corpus, layer)
    for _, g in s.groupby(["corpus", "layer"]):
        vals = g.sort_values("rank").value.to_numpy()
        assert (vals[:-1] >= vals[1:] - 1e-12).all()

    x = ov[ov.kind == "xretention"]
    # nothing is evaluated UNDER the null's covariance (fit-side only)
    assert all(not p.endswith("->null") for p in x.pair)
    assert (x.value > 0).all()


def test_rank_overlap_pinned_to_dec_not_enc():
    """Reviewer mutant (Task 6): swapping dec->enc inside _rank_overlap is NOT
    caught by any prior assertion — self-overlap stays 1.0 and values stay in
    [0, 1] under the mutant. Pin the function's output against an expected
    value computed independently, in-test, straight from `dec`, and prove the
    test can discriminate the enc-derived alternative (power check).

    Non-trivial (non-multiple-of-identity) diagonal whitener: distinct
    per-channel scales make enc = Wh@E and dec = Wh_inv@E span genuinely
    different subspaces (Wh != c*Wh_inv for any scalar c), unlike the
    identity_whitener used by most other tiny fixtures in this file where
    enc == dec and the mutant would be invisible.
    """
    from bmx.cache.spectral import fit_spectral_basis
    from bmx.quant.hadamard import orthogonalize
    from experiments.k4_corpus_transfer import _rank_overlap

    C, S = 16, 200
    scales = torch.linspace(0.2, 5.0, C, dtype=torch.float64)
    Wh = torch.diag(scales)
    Wh_inv = torch.diag(1.0 / scales)

    def make_basis(seed):
        g = torch.Generator().manual_seed(seed)
        M = torch.randn(S, C, generator=g, dtype=torch.float32)
        return fit_spectral_basis(M, Wh, Wh_inv)

    basis_a = make_basis(1)
    basis_b = make_basis(2)

    for r in (4, 8):
        # Expected value transcribed independently from _rank_overlap's own
        # reduction (mean of squared singular values of Q_a.T @ Q_b), applied
        # by hand to `dec` — no call into _rank_overlap or subspace_overlap.
        Q_a = orthogonalize(basis_a.dec[:, :r].double())
        Q_b = orthogonalize(basis_b.dec[:, :r].double())
        svals = torch.linalg.svdvals(Q_a.mT @ Q_b)
        expected = float((svals**2).mean())

        got = _rank_overlap(basis_a.dec, basis_b.dec, r)
        assert abs(got - expected) < 1e-9, (
            f"r={r}: _rank_overlap={got} != dec-derived expected={expected}"
        )

        # Power check: the enc-derived alternative must differ meaningfully
        # from the dec-derived expected value, or a dec->enc mutant would
        # slip through this fixture too.
        Q_a_enc = orthogonalize(basis_a.enc[:, :r].double())
        Q_b_enc = orthogonalize(basis_b.enc[:, :r].double())
        svals_enc = torch.linalg.svdvals(Q_a_enc.mT @ Q_b_enc)
        enc_val = float((svals_enc**2).mean())
        assert abs(enc_val - expected) > 0.02, (
            f"r={r}: enc-derived value {enc_val} too close to dec-derived "
            f"expected {expected} -- fixture doesn't discriminate the mutant"
        )

    # Cheap ground-truth pin (self-overlap must be exactly 1 at any rank).
    for r in (4, 8):
        assert abs(_rank_overlap(basis_a.dec, basis_a.dec, r) - 1.0) < 1e-9


def test_k4_corpus_transfer_overlap_row_pinned_to_dec(tmp_path):
    """Closes the reviewer's actual mutant surface: `_rank_overlap` itself is
    attribute-agnostic (it takes raw tensors), so the real dec->enc mutation
    site is the two call sites in `_diagnostics` that read `.dec` off
    `fits[a].bases[layer_i]` / `fits[b].bases[layer_i]`. Neither
    test_k4_corpus_transfer_diagnostics (range/monotonicity only) nor a
    unit-level pin on `_rank_overlap` in isolation exercises those call
    sites. Recompute the expected value independently — via the SAME fit
    path `main()` uses (`_load_side` + `corpus_fit_bases`) — straight from
    `.dec`, and pin it against the actual emitted overlap.parquet row.
    """
    from experiments._k4_common import corpus_fit_bases
    from experiments.k4_corpus_transfer import _load_side, main
    from bmx.quant.hadamard import orthogonalize

    cfg = _tiny_corpus_transfer_cfg(tmp_path)
    run_dir = main(cfg)

    import pandas as pd

    ov = pd.read_parquet(run_dir / "overlap.parquet")
    o = ov[(ov.kind == "overlap") & (~ov.centered)]

    # Independently reproduce fits["wiki"] / fits["code"] the same way main()
    # does (corpus_fit_bases with w_source="corpus", matching cfg.ridge /
    # position_stride) -- a non-identity whitener assembled from the tiny
    # cache's own query moments, so dec != enc here too.
    per_cache_w, gcs_w, rope_w, layers = _load_side(cfg.wiki_fit_paths, cfg.model_name)
    per_cache_c, gcs_c, rope_c, layers_c = _load_side(
        cfg.code_fit_paths, cfg.model_name
    )
    assert layers == layers_c
    fit_wiki = corpus_fit_bases(
        per_cache_w,
        gcs_w,
        rope_w,
        layers,
        w_source="corpus",
        ridge=cfg.ridge,
        position_stride=cfg.position_stride,
    )
    fit_code = corpus_fit_bases(
        per_cache_c,
        gcs_c,
        rope_c,
        layers,
        w_source="corpus",
        ridge=cfg.ridge,
        position_stride=cfg.position_stride,
    )

    layer_i = layers[0]
    r = 4
    Q_a = orthogonalize(fit_wiki.bases[layer_i].dec[:, :r].double())
    Q_b = orthogonalize(fit_code.bases[layer_i].dec[:, :r].double())
    svals = torch.linalg.svdvals(Q_a.mT @ Q_b)
    expected = float((svals**2).mean())

    row = o[(o.pair == "wiki-code") & (o.layer == layer_i) & (o["rank"] == r)]
    assert len(row) == 1, f"expected exactly one matching row, got {len(row)}"
    got = float(row.value.iloc[0])
    assert abs(got - expected) < 1e-6, (
        f"overlap.parquet wiki-code layer={layer_i} rank={r}: {got} != "
        f"dec-derived expected {expected}"
    )

    # Power check: the enc-derived alternative must differ meaningfully from
    # the dec-derived expected value (measured gap ~0.021 on this fixture), or
    # a future fixture change could silently defang this pin against the
    # dec->enc mutant at the actual _diagnostics call sites.
    Q_a_enc = orthogonalize(fit_wiki.bases[layer_i].enc[:, :r].double())
    Q_b_enc = orthogonalize(fit_code.bases[layer_i].enc[:, :r].double())
    svals_enc = torch.linalg.svdvals(Q_a_enc.mT @ Q_b_enc)
    enc_val = float((svals_enc**2).mean())
    assert abs(enc_val - expected) > 0.005, (
        f"enc-derived value {enc_val} too close to dec-derived expected "
        f"{expected} -- fixture doesn't discriminate the dec->enc mutant"
    )


def _tiny_niah_run(tmp_path, run_id, arm, lengths, bits_by_length):
    import pandas as pd

    run = tmp_path / "k3_niah" / run_id
    run.mkdir(parents=True)
    pd.DataFrame(
        {
            "arm": [arm] * len(lengths),
            "length": list(lengths),
            "kv_size_bits": [bits_by_length[length] for length in lengths],
        }
    ).to_parquet(run / "metrics.parquet")
    return run


def test_plot_k4_paper_smoke(tmp_path):
    import pandas as pd

    from experiments.plot_k4_paper import Config, main

    # -- fig 1 inputs: tiny NIAH run dirs (k4_b2.5, k4_b2.2, both TQ arms) +
    # a tiny fit-packs parquet (n_zero_dirs -> C_used correction). ----------
    lengths = (4096, 8192)
    niah_dirs = [
        _tiny_niah_run(tmp_path, "r_b25", "k4_b2.5", lengths, {4096: 4.8, 8192: 3.6}),
        _tiny_niah_run(tmp_path, "r_b22", "k4_b2.2", lengths, {4096: 3.0, 8192: 2.5}),
        _tiny_niah_run(
            tmp_path,
            "r_tqb3",
            "turboquant_mse_b3",
            lengths,
            {4096: 3.42, 8192: 3.22},
        ),
        _tiny_niah_run(
            tmp_path,
            "r_tqk3v2",
            "turboquant_mse_k3v2",
            lengths,
            {4096: 2.94, 8192: 2.73},
        ),
    ]
    fit_packs = tmp_path / "fit.parquet"
    pd.DataFrame(
        {
            "model": ["tiny"] * 4,
            "layer": [0, 1, 0, 1],
            "budget": [2.5, 2.5, 2.2, 2.2],
            "n_zero_dirs": [3, 4, 5, 6],  # tiny C=16 fixture
        }
    ).to_parquet(fit_packs)

    # -- figs 2-3 inputs: a tiny corpus-transfer run dir mirroring the real
    # verdict JSON's shape + a tiny overlap.parquet. -------------------------
    transfer_dir = tmp_path / "k4_corpus_transfer" / "tiny"
    transfer_dir.mkdir(parents=True)
    verdict = {
        "headline_metric": "logit",
        "verdict_rule": "D<0.10 insensitive; D>0.25 domain-sensitive; else as-measured",
        "gpt2_yellow_flag": "tiny-fixture yellow flag",
        "per_budget": {
            "2.5": {
                "D": {
                    "code->wiki": {
                        "mean": 0.46,
                        "min": 0.40,
                        "max": 0.50,
                        "label": "domain-sensitive",
                    },
                    "null->wiki": {
                        "mean": 0.09,
                        "min": 0.07,
                        "max": 0.11,
                        "label": "insensitive",
                    },
                    "wiki->code": {
                        "mean": 0.56,
                        "min": 0.51,
                        "max": 0.61,
                        "label": "domain-sensitive",
                    },
                    "null->code": {
                        "mean": 0.55,
                        "min": 0.50,
                        "max": 0.59,
                        "label": "domain-sensitive",
                    },
                    "shufcode->wiki": {
                        "mean": 0.09,
                        "min": 0.07,
                        "max": 0.11,
                        "label": "insensitive",
                    },
                    "shufcode->code": {
                        "mean": 0.11,
                        "min": 0.09,
                        "max": 0.14,
                        "label": "as-measured",
                    },
                    "uniwiki->wiki": {
                        "mean": 0.12,
                        "min": 0.07,
                        "max": 0.16,
                        "label": "as-measured",
                    },
                    "unicode->code": {
                        "mean": 0.14,
                        "min": 0.13,
                        "max": 0.15,
                        "label": "as-measured",
                    },
                    "biwiki->wiki": {
                        "mean": 0.01,
                        "min": -0.03,
                        "max": 0.06,
                        "label": "insensitive",
                    },
                    "bicode->code": {
                        "mean": 0.03,
                        "min": 0.02,
                        "max": 0.04,
                        "label": "insensitive",
                    },
                },
                "synthesis": {
                    "rules": {
                        "wiki": {
                            "D_uni": 0.12,
                            "D_bi": 0.01,
                            "D_shuf": 0.09,
                            "recipe_confirmed": False,
                            "order2_earns_keep": True,
                        },
                        "code": {
                            "D_uni": 0.14,
                            "D_bi": 0.03,
                            "D_shuf": 0.11,
                            "recipe_confirmed": False,
                            "order2_earns_keep": True,
                        },
                    },
                    "climb_to_order3": True,
                },
            }
        },
    }
    (transfer_dir / "corpus_transfer_verdict.json").write_text(json.dumps(verdict))

    rows = []
    for pair in ("wiki-code", "wiki-null", "code-null"):
        for layer in (0, 1):
            for rank in (8, 16):
                rows.append(
                    dict(
                        kind="overlap",
                        pair=pair,
                        corpus="",
                        layer=layer,
                        rank=rank,
                        budget=float("nan"),
                        tier=-1,
                        centered=False,
                        value=0.5 if rank == 8 else 0.4,
                    )
                )
    pd.DataFrame(rows).to_parquet(transfer_dir / "overlap.parquet")

    # -- fig 4 input: a tiny cert_vs_measured.parquet mirroring the real
    # run's schema (one row below y=x, one above -- exercises both branches
    # of the honest-annotation counting, not just the all-conservative
    # case). ------------------------------------------------------------
    dec_quant_dir = tmp_path / "k4_dec_quant" / "tiny"
    dec_quant_dir.mkdir(parents=True)
    cvm_rows = []
    for budget in (2.2, 2.5):
        for layer in (0, 1):
            for t in (4, 5, 6, 8):
                cvm_rows.append(
                    dict(
                        model="tiny",
                        cache="tiny.safetensors",
                        budget=budget,
                        layer=layer,
                        tier_threshold=t,
                        implied_rel_degradation=0.01 * t,
                        measured_rel_deg=(0.005 * t if layer == 0 else 0.02 * t),
                        frac_int8=0.5,
                        c_used=10,
                        c_int8=5,
                    )
                )
    pd.DataFrame(cvm_rows).to_parquet(dec_quant_dir / "cert_vs_measured.parquet")

    out_dir = tmp_path / "figures"
    cfg = Config(
        niah_run_dirs=tuple(str(d) for d in niah_dirs),
        fit_packs_parquet=str(fit_packs),
        corpus_transfer_run_dir=str(transfer_dir),
        dec_quant_run_dir=str(dec_quant_dir),
        charge_budgets=(2.2, 2.5),
        transfer_budget="2.5",
        C=16,
        group=16,
        out_dir=str(out_dir),
    )
    main(cfg)

    expected = {
        "k4_bits_vs_context.png",
        "k4_bits_vs_context.pdf",
        "k4_corpus_transfer.png",
        "k4_corpus_transfer.pdf",
        "k4_overlap.png",
        "k4_overlap.pdf",
        "k4_cert_vs_measured.png",
        "k4_cert_vs_measured.pdf",
    }
    got = {p.name for p in out_dir.iterdir()}
    assert expected <= got
    for name in expected:
        assert (out_dir / name).stat().st_size > 0

    # Deterministic bytes: re-running must reproduce identical PDF output
    # (no embedded timestamp).
    pdf_before = (out_dir / "k4_bits_vs_context.pdf").read_bytes()
    cvm_pdf_before = (out_dir / "k4_cert_vs_measured.pdf").read_bytes()
    main(cfg)
    pdf_after = (out_dir / "k4_bits_vs_context.pdf").read_bytes()
    cvm_pdf_after = (out_dir / "k4_cert_vs_measured.pdf").read_bytes()
    assert pdf_before == pdf_after
    assert cvm_pdf_before == cvm_pdf_after

    # The blanket-int8 dashed line is GONE; the certified band replaced it.
    from experiments.plot_k4_paper import _build_bits_vs_context

    bits_result = _build_bits_vs_context(
        tuple(str(d) for d in niah_dirs),
        str(fit_packs),
        (2.2, 2.5),
        16,
        16,
        (0, 2, 3, 4, 5, 6, 8),
    )
    assert "skeptic-v2-int8" not in bits_result["mode_names"]
    b25 = bits_result["curves"]["k4_b2.5"]
    lo = dict(b25["skeptic-v2-int8-tier-band-lo"])
    hi = dict(b25["skeptic-v2-int8-tier-band-hi"])
    v2 = dict(b25["skeptic-v2"])
    for length in lo:
        # Both band edges must sit below the fp16-decoder (v2) curve (int8
        # only ever saves charge, never adds it) and the band must have
        # nonzero width (the two fracs 0.893/0.916 are distinct).
        assert lo[length] <= hi[length] <= v2[length]


def test_k4_charge_alloc_smoke(tmp_path):
    import json

    import pandas as pd

    from experiments.k4_charge_alloc import Config, main

    p1, p2 = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    scored = tmp_path / "s.safetensors"
    _tiny_cache(p1, seed=0)
    _tiny_cache(p2, seed=1)
    _tiny_cache(scored, seed=2)
    cfg = Config(
        corpus_cache_paths=(str(p1), str(p2)),
        cache_paths=(str(scored),),
        model_label="tiny",
        plain_budgets=(1.5, 2.0, 2.5, 3.0),
        ca_budgets=(2.0,),
        s_refs=(256, 1024),  # tiny C=16: s = 16/16 + 16*16/256 = 2.0 at 256
        eval_s=(256, 1024, 4096),
        group=16,
        out_root=str(tmp_path / "results"),
    )
    run_dir = main(cfg)
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert set(df.arm) == {"plain", "charge_aware"}
    assert set(df[df.arm == "charge_aware"].s_ref) == {256, 1024}
    assert (df[df.arm == "plain"].s_ref == -1).all()  # sentinel, like bits==-1

    diag = pd.read_parquet(run_dir / "diagnostics.parquet")
    assert {"c_used", "mean_bits"} <= set(diag.columns)
    assert any(c.startswith("n_t") for c in diag.columns)  # tier histogram

    fr = pd.read_parquet(run_dir / "frontier.parquet")
    assert set(fr.s_eval) == {256, 1024, 4096}

    v = json.loads((run_dir / "charge_alloc_verdict.json").read_text())
    assert "a_gate_pass" in v and "honest_negative" in v and "rule" in v
    e = v["per_point"]["b2_s256"]
    for key in (
        "win_ca",
        "win_plain_at_matched_bpe",
        "win_not_worse",
        "bpe_ca",
        "bpe_plain_at_matched_win",
        "bits_saved_k_side",
        "bits_saved_blended",
    ):
        assert key in e, key
    # c_used diagnostic: charge-aware at the harshest s_ref uses no more
    # directions than plain at the same budget (per-layer means).
    cu_ca = diag[(diag.arm == "charge_aware") & (diag.s_ref == 256)].c_used.mean()
    cu_pl = diag[(diag.arm == "plain") & (diag.budget == 2.0)].c_used.mean()
    assert cu_ca <= cu_pl


def test_k4_charge_alloc_verdict_arithmetic(tmp_path):
    """Belt-and-braces regression pin (idiom of test_k4_charge_curve_smoke):
    recompute one verdict entry's bpe_ca from the metrics rows + skeptic_charge
    and assert the JSON carries exactly that number; bits_saved_blended must
    be exactly half of bits_saved_k_side."""
    import json

    import pandas as pd

    from bmx.cache.spectral import skeptic_charge
    from experiments.k4_charge_alloc import Config, main

    p1 = tmp_path / "a.safetensors"
    scored = tmp_path / "s.safetensors"
    _tiny_cache(p1, seed=0)
    _tiny_cache(scored, seed=2)
    cfg = Config(
        corpus_cache_paths=(str(p1),),
        cache_paths=(str(scored),),
        model_label="tiny",
        plain_budgets=(1.5, 2.0, 2.5, 3.0),
        ca_budgets=(2.0,),
        s_refs=(256,),
        eval_s=(256,),
        group=16,
        out_root=str(tmp_path / "results"),
    )
    run_dir = main(cfg)
    df = pd.read_parquet(run_dir / "metrics.parquet")
    v = json.loads((run_dir / "charge_alloc_verdict.json").read_text())
    e = v["per_point"]["b2_s256"]
    sub = df[(df.arm == "charge_aware") & (df.s_ref == 256)]
    expected_bpe = (
        sub.bpe_model
        + sub.apply(
            lambda r: skeptic_charge(
                int(r.C), 256, tuple(cfg.tiers), c_used=float(r.c_used)
            ),
            axis=1,
        )
    ).mean()
    assert abs(e["bpe_ca"] - float(expected_bpe)) < 1e-9
    assert abs(e["bits_saved_blended"] - 0.5 * e["bits_saved_k_side"]) < 1e-12


def test_k4_w_rope_ab_smoke(tmp_path):
    import json

    import pandas as pd

    from experiments.k4_w_rope_ab import Config, main

    p1, p2 = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    scored = tmp_path / "s.safetensors"
    _tiny_cache(p1, seed=0)
    _tiny_cache(p2, seed=1)
    _tiny_cache(scored, seed=2)
    cfg = Config(
        corpus_cache_paths=(str(p1), str(p2)),
        cache_paths=(str(scored),),
        model_label="tiny",
        budgets=(2.0, 2.5),
        group=16,
        overlap_ranks=(4, 8),
        out_root=str(tmp_path / "results"),
    )
    run_dir = main(cfg)
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert set(df.w_rope) == {"frozen", "rotated"}
    ov = pd.read_parquet(run_dir / "overlap.parquet")
    assert set(ov["rank"]) == {4, 8}
    assert ((ov.value >= -1e-9) & (ov.value <= 1 + 1e-9)).all()
    v = json.loads((run_dir / "w_rope_verdict.json").read_text())
    # Circular (frozen-instrument) readout lives ONLY under the demoted
    # record key; the top-level decision derives from the causal instrument.
    rec = v["frozen_instrument_record"]
    assert set(rec["per_budget"]) == {"2", "2.5"}
    for e in rec["per_budget"].values():
        assert {"win_frozen", "win_rotated", "rel_win_delta"} <= set(e)
    assert rec["decision_circular"] in ("scoped_negligible", "rotated_form_required")
    # tiny fixture has no RoPE (model_name="") => the causal instrument is
    # the null control and the top-level decision reflects that, not the
    # circular rule.
    assert v["causal"]["third_instrument_verdict"] == "no_rope_null_control"
    assert v["decision"] == "no_rope_null"
    assert v["llama_refit_required"] is False
    for e in rec["per_budget"].values():
        assert abs(e["rel_win_delta"]) < 1e-6


def test_k4_g_table_smoke(tmp_path):
    import json

    import pandas as pd

    from bmx.cache.codecs import _tier_g
    from experiments.k4_g_table import Config, main

    fit = tmp_path / "f.safetensors"
    _tiny_cache(fit, seed=0)
    packs_path = tmp_path / "packs.safetensors"
    _fit_tiny_pack_file(fit, packs_path, budgets=(2.5,), group=16)
    run_dir = main(
        Config(
            pack_path=str(packs_path),
            corpus_cache_paths=(str(fit),),
            model_label="tiny",
            group=16,
            out_root=str(tmp_path / "results"),
        )
    )
    g = json.loads((run_dir / "g_table.json").read_text())
    tiers = tuple(g["tiers"])
    table = tuple(g["g_table"])
    assert tiers == (0, 2, 3, 4, 5, 6, 8)
    assert table[0] == 1.0  # g(0) exact
    assert all(a > b for a, b in zip(table, table[1:]))  # strictly decreasing
    # The table must be directly consumable by the allocator's validator.
    tiers_t = torch.tensor([float(t) for t in tiers], dtype=torch.float64)
    _tier_g(tiers_t, table)  # raises if not grid-convex
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert {"layer", "tier", "g_hat", "p10", "p90", "n_dirs"} <= set(df.columns)


def test_k4_g_table_deterministic(tmp_path):
    import json

    from experiments.k4_g_table import Config, main

    fit = tmp_path / "f.safetensors"
    _tiny_cache(fit, seed=0)
    packs_path = tmp_path / "packs.safetensors"
    _fit_tiny_pack_file(fit, packs_path, budgets=(2.5,), group=16)
    cfg = dict(
        pack_path=str(packs_path),
        corpus_cache_paths=(str(fit),),
        model_label="tiny",
        group=16,
    )
    r1 = main(Config(**cfg, out_root=str(tmp_path / "r1")))
    r2 = main(Config(**cfg, out_root=str(tmp_path / "r2")))
    g1 = json.loads((r1 / "g_table.json").read_text())["g_table"]
    g2 = json.loads((r2 / "g_table.json").read_text())["g_table"]
    assert g1 == g2


def test_k4_int8_certificate_smoke(tmp_path):
    import json

    import pandas as pd

    from experiments.k4_int8_certificate import Config, main

    fit = tmp_path / "f.safetensors"
    _tiny_cache(fit, seed=0)
    packs_path = tmp_path / "packs.safetensors"
    _fit_tiny_pack_file(fit, packs_path, budgets=(2.0, 2.5), group=16)
    run_dir = main(
        Config(
            pack_path=str(packs_path),
            budgets=(2.0, 2.5),
            model_label="tiny",
            out_root=str(tmp_path / "results"),
        )
    )
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert {
        "budget",
        "layer",
        "added",
        "payload",
        "noise_to_signal",
        "implied_rel_degradation",
    } <= set(df.columns)
    assert (df.implied_rel_degradation >= 0).all()
    v = json.loads((run_dir / "certificate_verdict.json").read_text())
    assert "max_implied_rel_degradation" in v
    assert v["vm_gate_line"] == 0.05
    assert v["user_review_required_before_vm_task8_release"] is True
    # margin_factor consistency pin
    assert (
        abs(v["margin_factor"] - 0.05 / max(v["max_implied_rel_degradation"], 1e-300))
        < 1e-6 * v["margin_factor"]
    )

    # ---- 6b: tier-gated rescue sweep ------------------------------------
    sweep_df = pd.read_parquet(run_dir / "tier_sweep.parquet")
    assert {
        "budget",
        "tier_threshold",
        "c_used_total",
        "c_int8_total",
        "frac_int8",
        "max_implied_rel_degradation",
        "effective_dec_bits",
        "charge_saving_at_S4096",
        "charge_saving_at_S16384",
        "charge_saving_at_S65536",
    } <= set(sweep_df.columns)
    assert set(sweep_df.tier_threshold) == {2, 3, 4, 5, 6}
    assert (sweep_df.frac_int8 >= 0).all() and (sweep_df.frac_int8 <= 1).all()
    # frac_int8 and charge_saving must be non-decreasing in tier_threshold
    # per budget (a higher T can only int8-cover more columns, never fewer).
    for budget in (2.0, 2.5):
        sub = sweep_df[sweep_df.budget == budget].sort_values("tier_threshold")
        assert sub.frac_int8.is_monotonic_increasing
        assert sub.charge_saving_at_S4096.is_monotonic_increasing
        assert sub.effective_dec_bits.is_monotonic_decreasing

    tier_sweep_v = v["tier_sweep"]
    assert tier_sweep_v["tier_thresholds"] == [2, 3, 4, 5, 6]
    for budget_key in ("2", "2.5"):
        rescue = tier_sweep_v["per_budget_rescue"][budget_key]
        assert "largest_passing_threshold" in rescue
        assert "charge_saving_fraction_at_S4096" in rescue

    # ---- K4 estimation-levers Task 3: per-layer T_ℓ sweep -----------------
    from bmx.cache.spectral import (
        load_packs,
        mixed_dec_charge,
        per_layer_tier_thresholds,
    )

    per_layer_df = pd.read_parquet(run_dir / "per_layer_tl_sweep.parquet")
    assert {
        "budget",
        "layer",
        "C",
        "t_layer",
        "c_used",
        "c_int8_t_layer",
        "c_int8_uniform_t5",
        "implied_rel_degradation_at_t_layer",
    } <= set(per_layer_df.columns)
    assert (per_layer_df.implied_rel_degradation_at_t_layer <= 0.05 + 1e-9).all()

    pl_v = v["per_layer_tl_sweep"]
    assert pl_v["materiality_bar_bits_per_token_at_S4096"] == 0.3
    for budget in (2.0, 2.5):
        packs = load_packs(str(packs_path), budget)
        t_map = per_layer_tier_thresholds(packs, bar=0.05)
        entry = pl_v["per_budget"][f"{budget:g}"]
        assert entry["t_layer_map"] == {str(k): v for k, v in sorted(t_map.items())}

        expected_delta = 0.0
        for layer_i, pack in packs.items():
            T_l = t_map[layer_i]
            c_int8_l = pack.c_int8(T_l) if T_l > 0 else 0
            c_int8_5 = pack.c_int8(5)
            C_l = int(pack.enc.shape[0])
            expected_delta += mixed_dec_charge(
                C_l, 4096, pack.tiers, c_used=pack.c_used, c_int8=c_int8_5
            ) - mixed_dec_charge(
                C_l, 4096, pack.tiers, c_used=pack.c_used, c_int8=c_int8_l
            )
        assert abs(entry["saving_delta_at_S4096"] - expected_delta) < 1e-9
        assert entry["materiality_pass_at_S4096"] == (expected_delta >= 0.3)


# ---------------------------------------------------------------------------
# k4_jensen_gap (K4 local-levers Task 4 — determinant-Jensen Gate-A anchor)
# ---------------------------------------------------------------------------


def _diag_cache(path, diag_mags, *, n_layers=1, h_kv=1, T=8, seed=0):
    """A cache whose layer-i k_pre matrix has an EXACTLY diagonal, hand-known
    key second moment: row s has a single nonzero entry `diag_mags[s]` at
    channel s (S == C == len(diag_mags)), so key_second_moment(to_matrix(k_pre))
    == diag(diag_mags**2) / S exactly (no cross terms, no sampling noise).
    h_kv=1 (so C == d == len(diag_mags), identity block layout); q/v are
    unconstrained random fill (unused by the Jensen-gap path, which only
    reads k_pre, but load_layer_keys/setup_rope need the other kinds
    present with consistent shapes)."""
    g = torch.Generator().manual_seed(seed)
    S = len(diag_mags)
    d = S  # C = h_kv * d = d since h_kv=1
    K = torch.diag(torch.tensor(diag_mags, dtype=torch.float32)).unsqueeze(0)  # (1,S,d)
    tensors = {}
    for i in range(n_layers):
        tensors[f"layer{i}.k_pre"] = K.contiguous().half()
        tensors[f"layer{i}.k"] = K.contiguous().half()
        tensors[f"layer{i}.v"] = torch.randn(h_kv, S, d, generator=g).half()
        tensors[f"layer{i}.q"] = torch.randn(h_kv, T, d, generator=g).half()
    save_cache(tensors, path)


def test_k4_jensen_gap_smoke(tmp_path):
    import pandas as pd

    from experiments.k4_jensen_gap import Config, main

    p1, p2, p3 = (tmp_path / f"{n}.safetensors" for n in ("a", "b", "c"))
    for p, seed in ((p1, 0), (p2, 1), (p3, 2)):
        _tiny_cache(p, seed=seed)

    run_dir = main(
        Config(
            cache_paths=(str(p1), str(p2), str(p3)),
            model_label="tiny",
            budgets=(2.2, 2.5),
            n_flat=2,
            out_root=str(tmp_path / "results"),
        )
    )
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert {
        "layer",
        "gm_pool",
        "mean_gm_seq",
        "r_pred",
        "log_gap",
        "n_seq",
        "n_clamped",
        "r_pred_flat",
        "flatness_delta",
        "mixed_r_pred",
        "within_r_pred",
        "r_discrete_b2.2",
        "identity_check_b2.2",
        "abs_gap_b2.2",
        "r_discrete_b2.5",
        "identity_check_b2.5",
        "abs_gap_b2.5",
        "r_pred_debiased",
        "bias_factor_seq",
        "bias_factor_pool",
        "r_pred_flat_debiased",
        "flatness_delta_debiased",
        "mixed_r_pred_debiased",
        "within_r_pred_debiased",
        "abs_gap_debiased_b2.2",
        "abs_gap_debiased_b2.5",
    } <= set(df.columns)
    assert (df.r_pred <= 1.0 + 1e-9).all()
    assert (df.n_seq == 3).all()
    assert df.mixed_r_pred.isna().all()  # no cache_paths_alt given
    assert df.mixed_r_pred_debiased.isna().all()
    assert (df.bias_factor_seq > 0).all() and (df.bias_factor_pool > 0).all()

    verdict = json.loads((run_dir / "jensen_verdict.json").read_text())
    assert "r_pred" in verdict and "per_budget" in verdict and "match" in verdict
    assert set(verdict["per_budget"]) == {"2.2", "2.5"}
    for entry in verdict["per_budget"].values():
        assert {
            "r_discrete",
            "identity_check",
            "abs_gap",
            "match",
            "abs_gap_debiased",
            "match_debiased",
        } <= set(entry)
    assert verdict["flatness"]["n_flat"] == 2
    assert {
        "r_pred_at_n_flat_debiased",
        "r_pred_at_all_debiased",
        "delta_debiased",
    } <= set(verdict["flatness"])
    assert verdict["mixed_domain"] is None

    # Wishart debiasing block: pre-registered raw `match` stays the primary
    # readout; the debiased comparison is explicitly labeled post-hoc.
    wd = verdict["wishart_debiasing"]
    assert wd["post_hoc"] is True
    assert "r_pred_debiased" in wd and "match_debiased" in wd
    assert wd["bias_factor_seq"] > 0 and wd["bias_factor_pool"] > 0
    # Bias factor magnitude sanity: this fixture's n_rows (S=128) comfortably
    # exceeds C (16), so bias factors sit in (0, 1] (small-sample gm
    # underestimation shrinks toward 1 as n/C grows, never overshoots past 1
    # for n >= C -- the Wishart mean is a downward-biased estimator here).
    assert 0.0 < wd["bias_factor_seq"] <= 1.0 + 1e-9
    assert 0.0 < wd["bias_factor_pool"] <= 1.0 + 1e-9


def test_k4_jensen_gap_debiasing_uses_real_row_counts_not_hardcoded(tmp_path):
    """Two caches at S=96 (equal, but != any hardcoded default like 1024):
    the harness must read each cache's row count off the data
    (per_cache_weighted_moments' M_parts) rather than assuming a fixed S --
    verified by checking bias_factor_seq against the closed-form
    Bartlett/digamma correction computed independently in the test from the
    ACTUAL S, and confirming a hardcoded-1024 assumption would give a
    visibly different (wrong) number. Equal lengths are REQUIRED: the
    experiment fail-fasts on unequal per-cache row counts (unweighted
    pooled-mean vs sum-n debias semantics only agree when n_s are equal --
    see the assert in k4_jensen_gap.main)."""
    import math

    import pandas as pd

    from experiments.k4_jensen_gap import Config, main

    p_small, p_large = tmp_path / "small.safetensors", tmp_path / "large.safetensors"
    _tiny_cache(p_small, S=96, C=16, h_kv=2, T=16, seed=0)
    _tiny_cache(p_large, S=96, C=16, h_kv=2, T=16, seed=1)

    run_dir = main(
        Config(
            cache_paths=(str(p_small), str(p_large)),
            model_label="tiny",
            budgets=(2.5,),
            n_flat=1,
            out_root=str(tmp_path / "results"),
        )
    )
    df = pd.read_parquet(run_dir / "metrics.parquet")

    C = 16

    def b(n):
        idx = torch.arange(1, C + 1, dtype=torch.float64)
        psi = torch.special.digamma((n - idx + 1) / 2.0)
        return float((psi + math.log(2.0 / n)).mean())

    # Real per-cache n_rows are 96 (read off the fixture's own S);
    # bias_factor_seq is exp(mean_s(b_s)) over the REAL value.
    b_seq_expected = math.exp(b(96))
    # A hardcoded-1024 bug would instead give exp(b(1024)) for BOTH caches.
    b_seq_wrong_if_hardcoded = math.exp(b(1024))

    for row_val in df.bias_factor_seq.tolist():
        assert abs(row_val - b_seq_expected) < 1e-6, (
            row_val,
            b_seq_expected,
        )
        assert abs(row_val - b_seq_wrong_if_hardcoded) > 1e-3, (
            "bias_factor_seq matches the hardcoded-1024 value -- "
            "n_rows is not being read from the actual cache data"
        )


def test_k4_jensen_gap_mixed_domain_diagnostic(tmp_path):
    """With cache_paths_alt given, mixed_r_pred/within_r_pred populate and the
    mixed-domain verdict block appears (widening the Jensen gap with
    heterogeneity is a reported diagnostic, not gated)."""
    import pandas as pd

    from experiments.k4_jensen_gap import Config, main

    p1, p2 = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    alt1, alt2 = tmp_path / "alt_a.safetensors", tmp_path / "alt_b.safetensors"
    _tiny_cache(p1, seed=0)
    _tiny_cache(p2, seed=1)
    _tiny_cache(alt1, seed=10)
    _tiny_cache(alt2, seed=11)

    run_dir = main(
        Config(
            cache_paths=(str(p1), str(p2)),
            cache_paths_alt=(str(alt1), str(alt2)),
            model_label="tiny",
            budgets=(2.5,),
            n_flat=1,
            out_root=str(tmp_path / "results"),
        )
    )
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert not df.mixed_r_pred.isna().any()
    assert (df.mixed_r_pred <= 1.0 + 1e-9).all()
    assert not df.mixed_r_pred_debiased.isna().any()

    verdict = json.loads((run_dir / "jensen_verdict.json").read_text())
    assert verdict["mixed_domain"] is not None
    assert "mixed_r_pred" in verdict["mixed_domain"]
    assert "within_r_pred" in verdict["mixed_domain"]
    assert "mixed_r_pred_debiased" in verdict["mixed_domain"]
    assert "within_r_pred_debiased" in verdict["mixed_domain"]


def test_k4_jensen_gap_verdict_hand_checked_toy(tmp_path):
    """Hand-checkable 2-layer, C=4 toy: each layer has 2 caches whose k_pre
    key second moment is EXACTLY diagonal (see _diag_cache) with disjoint
    hand-picked magnitudes per layer, identity whitener (no model_name given
    -> no RoPE), w_source='none' isn't wired through Config (jensen_gap
    always uses corpus W by default) -- so instead this test targets the
    no-RoPE path where w_source='corpus' with cos=ones/sin=zeros reduces the
    query moment to a plain pooled outer product; to keep the closed form
    exact we bypass that by calling the experiment's own arithmetic helpers
    directly on the EXACT diagonal T_s matrices (jensen_gap_report,
    allocate_bits_from_variance) and cross-check against independent
    by-hand Python arithmetic -- the "enumerate the waterfill" pin the task
    calls for."""
    from bmx.cache.codecs import allocate_bits_from_variance
    from bmx.cache.spectral import jensen_gap_report
    from experiments.k4_jensen_gap import _TIERS, _discrete_readout

    # layer 0: same two diagonal moments as the closed-form spectral.py test.
    a0 = torch.tensor([1.0, 4.0, 9.0, 16.0], dtype=torch.float64)
    b0 = torch.tensor([2.0, 2.0, 50.0, 0.5], dtype=torch.float64)
    # layer 1: a different pair (checks per-layer independence).
    a1 = torch.tensor([5.0, 5.0, 5.0, 5.0], dtype=torch.float64)
    b1 = torch.tensor([1.0, 100.0, 1.0, 100.0], dtype=torch.float64)

    layers = {0: [torch.diag(a0), torch.diag(b0)], 1: [torch.diag(a1), torch.diag(b1)]}

    for layer_i, T_list in layers.items():
        report = jensen_gap_report(T_list)

        def gm(vals):
            return math.exp(sum(math.log(v) for v in vals) / len(vals))

        vecs = [torch.diagonal(T).tolist() for T in T_list]
        gms = [gm(v) for v in vecs]
        pooled_diag = [sum(vs) / len(vs) for vs in zip(*vecs)]
        expected_r_pred = (sum(gms) / len(gms)) / gm(pooled_diag)
        assert abs(report["r_pred"] - expected_r_pred) < 1e-10, layer_i

        for budget in (2.2, 2.5):
            disc = _discrete_readout(T_list, budget)

            # By-hand oracle side: each diagonal T_s's eigenvalues ARE its
            # diagonal entries (any permutation; allocate_bits_from_variance
            # is permutation-covariant on a 1-D vector, so feed the raw
            # diagonal directly instead of re-deriving eigh's order).
            d_oracle_hand = []
            for T in T_list:
                lam = torch.diagonal(T).clone()
                b = allocate_bits_from_variance(lam, budget, _TIERS)
                d_oracle_hand.append(float((lam * torch.pow(4.0, -b.double())).sum()))
            mean_oracle_hand = sum(d_oracle_hand) / len(d_oracle_hand)
            assert abs(disc["mean_d_oracle"] - mean_oracle_hand) < 1e-9, (
                layer_i,
                budget,
            )

            # By-hand pooled side: pooled diag == mean of the per-cache
            # diagonals (E is a permutation on an already-diagonal matrix
            # -- eigh may reorder/sign-flip columns, but a permutation-plus-
            # sign eigenbasis leaves diag(E^T T E) equal to a permutation of
            # T's own diagonal, and the pooled allocation is computed on the
            # SAME pooled spectrum either way, so the summed charge matches).
            pooled_lam_hand = torch.tensor(pooled_diag, dtype=torch.float64)
            b_bar_hand = allocate_bits_from_variance(pooled_lam_hand, budget, _TIERS)
            charge_bar_hand = torch.pow(4.0, -b_bar_hand.double())
            d_pool_hand = [
                float((torch.diagonal(T) * charge_bar_hand).sum()) for T in T_list
            ]
            mean_pool_hand = sum(d_pool_hand) / len(d_pool_hand)
            assert abs(disc["mean_d_pool"] - mean_pool_hand) < 1e-9, (layer_i, budget)

            expected_r_discrete = mean_oracle_hand / mean_pool_hand
            assert abs(disc["r_discrete"] - expected_r_discrete) < 1e-9

            C = 4
            gm_pooled_hand = gm(pooled_diag)
            expected_identity = mean_pool_hand / (C * gm_pooled_hand * (4.0**-budget))
            assert abs(disc["identity_check"] - expected_identity) < 1e-9


def test_k4_jensen_gap_end_to_end_matches_toy_via_diag_cache(tmp_path):
    """End-to-end integration: run k4_jensen_gap.main on _diag_cache fixtures
    (no model_name => identity RoPE tables, w_source='corpus' default with
    plain-outer-product query moment) and confirm the harness's cache-loading
    path reproduces the fixture's EXACT known raw per-cache second moments
    (the moment-convention check), then that the emitted r_pred/log_gap are
    structurally sane (Minkowski bound, finite) -- the full whitened-moment
    closed form is exercised by the toy test above, which bypasses cache
    loading entirely to control the whitener."""
    import pandas as pd

    from experiments.k4_jensen_gap import Config, main

    a = [1.0, 2.0, 3.0, 4.0]
    b = [4.0, 3.0, 2.0, 1.0]
    p1, p2 = tmp_path / "diag_a.safetensors", tmp_path / "diag_b.safetensors"
    _diag_cache(p1, a, seed=0)
    _diag_cache(p2, b, seed=1)

    run_dir = main(
        Config(
            cache_paths=(str(p1), str(p2)),
            model_label="toy",
            budgets=(2.5,),
            n_flat=1,
            out_root=str(tmp_path / "results"),
        )
    )
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert len(df) == 1  # one layer

    S = len(a)
    Ta = torch.diag(torch.tensor(a, dtype=torch.float64) ** 2) / S
    Tb = torch.diag(torch.tensor(b, dtype=torch.float64) ** 2) / S
    # w_source='corpus' with no RoPE (model_name="") gives an all-ones/zeros
    # cos/sin table, so query_position_moment reduces to the plain pooled
    # query second moment -- NOT identity in general, so the whitener isn't
    # trivially I here. Only check the RAW (unwhitened) per-cache second
    # moments match Ta/Tb exactly (the fixture's own guarantee); the
    # end-to-end r_pred is cross-checked structurally (<=1, finite) rather
    # than against a hand-derived whitened closed form.
    from bmx.cache.spectral import key_second_moment
    from bmx.cache.collect import to_matrix, load_cache

    ka = load_cache(str(p1))["layer0.k_pre"]
    kb = load_cache(str(p2))["layer0.k_pre"]
    assert torch.allclose(key_second_moment(to_matrix(ka)), Ta, atol=1e-6)
    assert torch.allclose(key_second_moment(to_matrix(kb)), Tb, atol=1e-6)

    row = df.iloc[0]
    assert 0.0 < row.r_pred <= 1.0 + 1e-9
    assert math.isfinite(row.log_gap)


# ---------------------------------------------------------------------------
# k4_shrinkage (K4 estimation levers Task 2)
# ---------------------------------------------------------------------------


def test_lw_rows_frame_pin_matches_basis_lam64():
    """MANDATORY frame-pin test (Task 2 brief): the LW rows passed to
    shrink_spectrum must be the fit rows projected through the basis's own
    enc matrix, in fp64 -- rows = M_fit @ enc. Verifies eigvalsh(rows^T rows
    / n) matches basis.lam64 to 1e-6 relative on a tiny fixture, pinning the
    W-weighted frame shrink_spectrum's rho must be computed in (Task 1's
    report: LW rho is NOT invariant to the W-weighting)."""
    from bmx.cache.spectral import assemble_whitener, fit_spectral_basis

    torch.manual_seed(0)
    S, h_kv, d = 200, 2, 6
    C = h_kv * d
    M_fit = torch.randn(S, C, dtype=torch.float32)
    W_blocks = torch.randn(h_kv, d, d, dtype=torch.float64)
    W_blocks = torch.einsum("hij,hkj->hik", W_blocks, W_blocks) + 0.1 * torch.eye(
        d, dtype=torch.float64
    )
    Wh, Wh_inv = assemble_whitener(W_blocks, ridge=1e-3)
    basis = fit_spectral_basis(M_fit, Wh, Wh_inv)

    rows = M_fit.double() @ basis.enc.double()  # the frame under test
    n = rows.shape[0]
    Sigma_rows = rows.mT @ rows / n
    lam_rows = torch.linalg.eigvalsh(Sigma_rows).flip(0)  # descending

    rel_err = ((lam_rows - basis.lam64).abs() / basis.lam64.clamp_min(1e-12)).max()
    assert float(rel_err) < 1e-6, f"frame mismatch: rel_err={float(rel_err):.3g}"


def test_shrinkage_verdict_gate_both_budgets_and_1_02_factor():
    """Gate logic on a synthetic frame: PROMOTE iff at BOTH budgets
    win(lw) >= 1.02*win(plain) AND no matched-budget bpe_v2 regression > 0.02.
    One budget failing the ratio -> gate_pass False."""
    from experiments.k4_shrinkage import _shrinkage_verdict

    rows = []
    for budget in (2.2, 2.5):
        for arm, win, bpe in (
            ("plain", 10.0, budget),
            ("lw", 10.3, budget + 0.01),  # ratio 1.03 >= 1.02, bpe delta ok
            ("oas", 10.1, budget),
        ):
            rows.append(
                dict(
                    model="tiny",
                    cache="c",
                    layer=0,
                    arm=arm,
                    method=arm if arm != "plain" else "",
                    budget=budget,
                    n_fit=0,
                    win=win,
                    bpe_v2=bpe,
                    c_used=8.0,
                    rho=0.1,
                )
            )
    import pandas as pd

    df = pd.DataFrame(rows)
    v = _shrinkage_verdict(df, budgets=(2.2, 2.5), win_factor=1.02, bpe_guard=0.02)
    assert v["gate"]["2.2"]["win_ratio"] > 1.02
    assert v["gate"]["2.5"]["win_ratio"] > 1.02
    assert v["gate"]["2.2"]["bpe_regression_ok"] is True
    assert v["gate"]["2.5"]["bpe_regression_ok"] is True
    assert v["gate_pass"] is True

    # Flip one budget's lw win below the 1.02 factor -> overall gate fails.
    df2 = df.copy()
    df2.loc[(df2.arm == "lw") & (df2.budget == 2.5), "win"] = 10.0  # ratio 1.0 < 1.02
    v2 = _shrinkage_verdict(df2, budgets=(2.2, 2.5), win_factor=1.02, bpe_guard=0.02)
    assert v2["gate"]["2.2"]["win_ratio"] > 1.02
    assert v2["gate"]["2.5"]["win_ratio"] <= 1.02
    assert v2["gate_pass"] is False


def test_shrinkage_verdict_bpe_regression_guard():
    """A budget can pass the win-ratio factor but fail on an oversized
    matched-budget bpe_v2 regression (lw bpe_v2 > plain bpe_v2 + 0.02) ->
    gate_pass False for that budget, and overall."""
    from experiments.k4_shrinkage import _shrinkage_verdict
    import pandas as pd

    rows = []
    for budget in (2.2, 2.5):
        lw_bpe = budget + (0.05 if budget == 2.2 else 0.0)  # regress at 2.2
        for arm, win, bpe in (
            ("plain", 10.0, budget),
            ("lw", 10.5, lw_bpe),  # ratio 1.05, comfortably over factor
        ):
            rows.append(
                dict(
                    model="tiny",
                    cache="c",
                    layer=0,
                    arm=arm,
                    method=arm if arm != "plain" else "",
                    budget=budget,
                    n_fit=0,
                    win=win,
                    bpe_v2=bpe,
                    c_used=8.0,
                    rho=0.1,
                )
            )
    df = pd.DataFrame(rows)
    v = _shrinkage_verdict(df, budgets=(2.2, 2.5), win_factor=1.02, bpe_guard=0.02)
    assert v["gate"]["2.2"]["bpe_regression_ok"] is False
    assert v["gate"]["2.2"]["gate_pass"] is False
    assert v["gate"]["2.5"]["bpe_regression_ok"] is True
    assert v["gate"]["2.5"]["gate_pass"] is True
    assert v["gate_pass"] is False  # both-budgets AND


def test_shrinkage_verdict_n_scaling_table_shape_and_rho_carried():
    """n-scaling table: mean win by (arm, n_fit, budget); rho column present
    per (layer, method, n_fit) in the diagnostics."""
    from experiments.k4_shrinkage import _shrinkage_verdict
    import pandas as pd

    rows = []
    for n_fit in (768, 0):
        for budget in (2.2, 2.5):
            for arm, method, rho in (
                ("plain", "", 0.0),
                ("lw", "lw", 0.3 if n_fit == 768 else 0.05),
                ("oas", "oas", 0.25 if n_fit == 768 else 0.04),
            ):
                rows.append(
                    dict(
                        model="tiny",
                        cache="c",
                        layer=0,
                        arm=arm,
                        method=method,
                        budget=budget,
                        n_fit=n_fit,
                        win=10.0 + rho,
                        bpe_v2=budget,
                        c_used=8.0,
                        rho=rho,
                    )
                )
    df = pd.DataFrame(rows)
    v = _shrinkage_verdict(df, budgets=(2.2, 2.5), win_factor=1.02, bpe_guard=0.02)
    assert "n_scaling" in v
    # keyed by arm -> n_fit -> budget -> mean win
    assert set(v["n_scaling"].keys()) >= {"plain", "lw", "oas"}
    assert set(v["n_scaling"]["lw"].keys()) == {"768", "0"}
    assert "rho_summary" in v
    assert "lw" in v["rho_summary"] and "oas" in v["rho_summary"]
    # rho at n_fit=768 should be reported distinctly from n_fit=0 (full)
    assert set(v["rho_summary"]["lw"].keys()) == {"768", "0"}


def test_k4_shrinkage_smoke(tmp_path):
    """End-to-end smoke on tiny fixtures: metrics.parquet has the expected
    arm/method/n_fit/budget columns and shrinkage_verdict.json has the gate
    + diagnostics keys."""
    import json

    import pandas as pd

    from experiments.k4_shrinkage import Config, main

    fit_paths = []
    for i in range(2):
        p = tmp_path / f"fit{i}.safetensors"
        _tiny_cache(p, seed=i)
        fit_paths.append(str(p))
    heldout_paths = []
    for i in range(2):
        p = tmp_path / f"held{i}.safetensors"
        _tiny_cache(p, seed=100 + i)
        heldout_paths.append(str(p))

    run_dir = main(
        Config(
            fit_cache_paths=tuple(fit_paths),
            heldout_cache_paths=tuple(heldout_paths),
            model_label="tiny",
            budgets=(2.2, 2.5),
            n_fits=(64, 0),
            methods=("lw", "oas"),
            group=16,
            out_root=str(tmp_path / "results"),
        )
    )
    df = pd.read_parquet(run_dir / "metrics.parquet")
    for col in (
        "model",
        "cache",
        "layer",
        "arm",
        "method",
        "budget",
        "n_fit",
        "win",
        "bpe_v2",
        "c_used",
        "rho",
    ):
        assert col in df.columns, col
    assert set(df.arm) == {"plain", "lw", "oas"}
    assert set(df.n_fit) == {64, 0}
    assert set(df.budget) == {2.2, 2.5}
    # plain arm carries no shrinkage intensity
    assert (df[df.arm == "plain"].rho == 0.0).all()

    v = json.loads((run_dir / "shrinkage_verdict.json").read_text())
    for key in ("gate", "gate_pass", "n_scaling", "rho_summary", "c_used_stability"):
        assert key in v, key


def test_k4_shrinkage_n_fit_zero_matches_standard_fit_path(tmp_path):
    """n_fit=0 (full) must fit BIT-EXACT the same basis as the standard
    corpus_fit_bases path (no subsampling applied) -- pinned by comparing
    the plain-arm pack's bits directly against a corpus_fit_bases +
    pack_from_basis reference computed independently in this test."""
    import torch as _torch

    from bmx.cache.spectral import pack_from_basis
    from experiments._k4_common import corpus_fit_bases, load_layer_keys, setup_rope
    from experiments.k4_shrinkage import _fit_bases_at_n

    fit_paths = []
    for i in range(2):
        p = tmp_path / f"fit{i}.safetensors"
        _tiny_cache(p, seed=i)
        fit_paths.append(str(p))

    per_cache = [load_layer_keys(p) for p in fit_paths]
    layers = sorted(per_cache[0].keys())
    rope_ready = False
    get_cos_sins = []
    for lk in per_cache:
        ready, gcs = setup_rope("", lk, layers)
        rope_ready = rope_ready or ready
        get_cos_sins.append(gcs)

    ref = corpus_fit_bases(
        per_cache,
        get_cos_sins,
        rope_ready,
        layers,
        w_source="corpus",
        ridge=1e-3,
        position_stride=8,
    )
    ref_pack = pack_from_basis(ref.bases[layers[0]], 2.5, group=16)

    fit_n0 = _fit_bases_at_n(
        per_cache,
        get_cos_sins,
        rope_ready,
        layers,
        ridge=1e-3,
        position_stride=8,
        n_fit=0,
        seed=0,
    )
    test_pack = pack_from_basis(fit_n0[layers[0]][0], 2.5, group=16)

    assert _torch.equal(ref_pack.bits, test_pack.bits)
    assert _torch.allclose(ref.bases[layers[0]].lam64, fit_n0[layers[0]][0].lam64)
