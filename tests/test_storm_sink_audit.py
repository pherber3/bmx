"""Storm Task-3 — sink carve-out audit: position-decomposition math pins.

The load-bearing claim the gate rests on is that the per-position query-weighted
distortion Σ_s e_sᵀ W e_s decomposes EXACTLY by key position, and that a
synthetic cache whose reconstruction error is concentrated entirely on
position 0 attributes ~100% of the weighted distortion to the sinks. These are
tiny-factory, offline (no model, no download) tests.
"""

import json

import torch

from bmx.cache.collect import save_cache, to_matrix
from bmx.cache.spectral import (
    fit_spectral_basis,
    identity_whitener,
    pack_from_basis,
    save_pack_file,
)
from experiments.storm_sink_audit import (
    N_SINKS,
    attention_mass_by_position,
    position_weighted_distortion,
    sink_share,
)


def _tiny_pack(S=128, C=16, h_kv=2, seed=0, budget=2.5, group=16):
    """A real fitted SpectralPack (identity whitener) on a low-rank-plus-noise
    (S, C) key matrix — enough tiers get allocated for a nontrivial residual."""
    g = torch.Generator().manual_seed(seed)
    d = C // h_kv
    raw = torch.randn(C, 3, generator=g)
    dirs, _ = torch.linalg.qr(raw)
    z = torch.randn(S, 3, generator=g) * torch.tensor([20.0, 12.0, 8.0])
    M = z @ dirs.mT + torch.randn(S, C, generator=g)
    Ih, Ih_inv = identity_whitener(C)
    basis = fit_spectral_basis(M, Ih, Ih_inv)
    pack = pack_from_basis(basis, budget, group=group)
    return M.float(), pack, d


def test_w_equals_enc_enc_t_identity():
    """W = enc @ encᵀ to fp64 round-off — the identity that lets the
    per-position decomposition use ONLY the pack's enc (no W refit). Built on a
    weighted (non-identity) whitener so the identity is nontrivial."""
    from bmx.cache.spectral import assemble_whitener

    torch.manual_seed(0)
    h_kv, d = 2, 8
    C = h_kv * d
    # A per-block query second moment (h_kv, d, d), PSD.
    A = torch.randn(h_kv, d, d, dtype=torch.float64)
    W_blocks = A @ A.mT + torch.eye(d, dtype=torch.float64)
    Wh, Wh_inv = assemble_whitener(W_blocks, ridge=1e-3)
    M = torch.randn(64, C, dtype=torch.float64)
    basis = fit_spectral_basis(M.float(), Wh, Wh_inv)
    enc = basis.enc.double()
    W = Wh @ Wh  # W^{1/2} @ W^{1/2}
    rel = ((enc @ enc.mT - W).norm() / W.norm()).item()
    assert rel < 1e-6, f"enc encᵀ != W: rel {rel}"


def test_position_decomposition_matches_direct_quadratic_form():
    """The per-position sum d_s = ||encᵀ e_s||² must equal the direct
    Σ_s e_sᵀ W e_s (W = enc encᵀ) to fp round-off — the decomposition is a
    reorganization of the same scalar, not an approximation."""
    M, pack, _d = _tiny_pack()
    d_pos = position_weighted_distortion(M, pack)
    # direct quadratic form
    from bmx.cache.spectral import spectral_quantize

    M_hat, _ = spectral_quantize(M, pack)
    err = (M - M_hat).double()
    W = pack.enc.double() @ pack.enc.double().mT
    direct = torch.einsum("sc,cd,sd->", err, W, err).item()
    assert abs(float(d_pos.sum()) - direct) / direct < 1e-4
    assert (d_pos >= 0).all()


def test_all_error_at_position_zero_attributes_to_sinks():
    """SYNTHETIC GATE CHECK (plan Task 3): a reconstruction whose error is
    planted ENTIRELY at position 0 must attribute ~100% of the weighted
    distortion to the sinks. We synthesize this directly on the decomposition
    quantity: build an error matrix that is nonzero only on row 0, feed it
    through the same enc projection, and confirm sink_share == 1.0."""
    _M, pack, _d = _tiny_pack()
    C = pack.enc.shape[0]
    S = 128
    err = torch.zeros(S, C)
    err[0] = torch.randn(C)  # ALL error on the first sink position
    proj = err @ pack.enc.float()
    d_pos = (proj * proj).sum(dim=1)
    assert sink_share(d_pos, n_sinks=N_SINKS) == 1.0
    # And a mid-window-only error attributes 0% to sinks.
    err2 = torch.zeros(S, C)
    err2[S // 2] = torch.randn(C)
    d_pos2 = (err2 @ pack.enc.float()).pow(2).sum(dim=1)
    assert sink_share(d_pos2, n_sinks=N_SINKS) == 0.0


def test_position_weighted_distortion_via_real_reconstruction_lifts_sink_share():
    """End-to-end on a real pack: corrupting the SINK rows with a large offset
    along the codec's WEAKEST (least-allocated) direction — the residual axis
    the waterfill can't represent — lifts the measured sink distortion share
    far above the uniform-position baseline (n_sinks/S). This exercises the
    full `spectral_quantize` reconstruction path, not a hand-built error."""
    M, pack, _d = _tiny_pack()
    S, C = M.shape
    base_share = sink_share(position_weighted_distortion(M, pack), n_sinks=N_SINKS)
    # dec's last column is the lowest-eigenvalue (least-allocated) direction.
    weak_dir = pack.dec[:, -1].float()
    weak_dir = weak_dir / weak_dir.norm()
    M2 = M.clone()
    M2[:N_SINKS] = M2[:N_SINKS] + 30.0 * weak_dir  # rogue offset on the sink rows
    lifted = sink_share(position_weighted_distortion(M2, pack), n_sinks=N_SINKS)
    assert lifted > base_share
    assert lifted > (N_SINKS / S)  # well above the uniform-position baseline


def test_attention_mass_sums_to_one_and_is_causal():
    """Mass over positions is a distribution (sums to 1); with the queries at
    the last T positions, sink mass is a valid fraction in [0, 1]."""
    torch.manual_seed(0)
    h_kv, h, S, T, d = 2, 4, 64, 16, 8
    q = torch.randn(h, T, d)
    k_post = torch.randn(h_kv, S, d)
    mass, sink_m = attention_mass_by_position(q, k_post, n_sinks=N_SINKS)
    assert abs(float(mass.sum()) - 1.0) < 1e-5
    assert 0.0 <= sink_m <= 1.0
    # A query row can attend to a sink (s=0 <= any query position), so with
    # random keys the sinks carry SOME mass.
    assert sink_m > 0.0


def test_sink_share_handles_zero_total():
    assert sink_share(torch.zeros(64), n_sinks=N_SINKS) == 0.0


def test_storm_sink_audit_smoke(tmp_path):
    """Full experiment on a tiny factory cache + on-the-fly-fit pack file:
    parquet + verdict written, gate evaluated, honest-null branch reachable."""
    import pandas as pd

    from experiments.storm_sink_audit import Config, main

    # A tiny 2-layer cache (mirrors the k4 test factory) as BOTH corpus and
    # score cache — smoke only; the real run uses distinct offset caches.
    g = torch.Generator().manual_seed(3)
    S, C, h_kv, T = 128, 16, 2, 16
    d = C // h_kv
    raw = torch.randn(C, 3, generator=g)
    dirs, _ = torch.linalg.qr(raw)
    z = torch.randn(S, 3, generator=g) * torch.tensor([20.0, 12.0, 8.0])
    M = (z @ dirs.mT + torch.randn(S, C, generator=g)).half()
    K = M.reshape(S, h_kv, d).permute(1, 0, 2).contiguous()
    tensors = {}
    for i in range(2):
        tensors[f"layer{i}.k_pre"] = K.clone()
        tensors[f"layer{i}.k"] = K.clone()
        tensors[f"layer{i}.q"] = torch.randn(h_kv * 2, T, d, generator=g).half()
        tensors[f"layer{i}.v"] = torch.randn(h_kv, S, d, generator=g).half()
    cache_path = tmp_path / "score.safetensors"
    save_cache(tensors, str(cache_path))

    # Fit a real pack file (identity whitener) with both budgets present.
    layer_keys = {
        0: {"k_pre": tensors["layer0.k_pre"]},
        1: {"k_pre": tensors["layer1.k_pre"]},
    }
    bases = {}
    for li, km in layer_keys.items():
        Mfit = to_matrix(km["k_pre"])
        Ih, Ih_inv = identity_whitener(Mfit.shape[1])
        bases[li] = fit_spectral_basis(Mfit, Ih, Ih_inv)
    pack_path = tmp_path / "packs.safetensors"
    save_pack_file(str(pack_path), bases, (2.2, 2.5), group=16)

    cfg = Config(
        score_cache_path=str(cache_path),
        pack_path=str(pack_path),
        model_label="tiny",
        model_name="",
        budgets=(2.2, 2.5),
        n_sinks=N_SINKS,
        out_root=str(tmp_path / "results"),
    )
    run_dir = main(cfg)
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert {
        "model",
        "budget",
        "layer",
        "sink_distortion",
        "total_distortion",
        "sink_dist_share",
        "sink_attn_mass",
    } <= set(df.columns)
    assert set(df.budget.unique()) == {2.2, 2.5}
    verdict = json.loads((run_dir / "verdict.json").read_text())
    assert "gate_outcome" in verdict and "carve_out_confirmed" in verdict
    assert set(verdict["per_budget"].keys()) == {"2.2", "2.5"}
    for v in verdict["per_budget"].values():
        assert 0.0 <= v["sink_dist_share_pct"] <= 100.0
        assert 0.0 <= v["sink_attn_mass_pct"] <= 100.0
