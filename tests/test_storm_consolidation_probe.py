"""Storm Task-9 — consolidation probe: offline tiny-factory pins.

The gate rests on three load-bearing pieces, each pinned here without any
model download: (1) the whitened k-means recovers planted clusters AND
respects the query metric (clusters split along the direction the metric
upweights, not the high-variance nuisance axis); (2) the pre-registered bit
accounting (pure counts/positions vs per-token assignment table, floor
matching maximal-but-never-over); (3) the degenerate case m = S reproduces
the exact cache bit-for-bit (zero distortion through the shipped
instrument), and the expanded evaluation embedding equals the merged
count-weighted softmax exactly.
"""

import json

import torch

from bmx.cache.collect import from_matrix, save_cache, to_matrix
from bmx.cache.metrics import logit_distortion
from bmx.cache.spectral import fit_spectral_basis, identity_whitener, save_pack_file
from experiments.storm_consolidation_probe import (
    assign_bits,
    ceil_log2,
    cluster_mean_rows,
    consolidate,
    expanded_key_matrix,
    kmeans_whitened,
    matched_m_assign,
    matched_m_pure,
    pure_bits,
)


def test_whitened_kmeans_recovers_planted_clusters():
    """Well-separated planted clusters are recovered exactly (assignment
    purity 1, one recovered label per planted group) and the centroids land
    on the planted centers within the noise scale."""
    g = torch.Generator().manual_seed(0)
    m0, per, D = 5, 40, 8
    centers = torch.randn(m0, D, generator=g) * 10.0
    X = centers.repeat_interleave(per, dim=0) + 0.1 * torch.randn(
        m0 * per, D, generator=g
    )
    assign, cent = kmeans_whitened(X, m0, seed=0)
    lab = assign.view(m0, per)
    assert (lab == lab[:, :1]).all(), "a planted cluster was split"
    assert lab[:, 0].unique().numel() == m0, "two planted clusters merged"
    for gi in range(m0):
        j = int(lab[gi, 0])
        assert float((cent[j] - centers[gi]).norm()) < 0.5


def test_kmeans_respects_query_metric():
    """Two clusters separated ONLY along coordinate 0 (small gap), with a
    large-variance nuisance along coordinate 1. Under the metric transform
    T = diag(10, 0.01) (so W = T Tᵀ upweights coord 0 by 1e2 and kills the
    nuisance by 1e-4), k-means on M @ T must split exactly by the sign of
    coord 0 — while plain Euclidean k-means is dominated by the nuisance
    axis and does not recover that split. This pins 'clustering respects the
    query metric'."""
    g = torch.Generator().manual_seed(1)
    n = 100
    sign = torch.cat([torch.ones(n), -torch.ones(n)])
    M = torch.zeros(2 * n, 2)
    M[:, 0] = sign + 0.05 * torch.randn(2 * n, generator=g)
    M[:, 1] = 10.0 * torch.randn(2 * n, generator=g)
    T = torch.diag(torch.tensor([10.0, 0.01]))
    assign_w, _ = kmeans_whitened(M @ T, 2, seed=0)
    a, b = assign_w[:n], assign_w[n:]
    assert (a == a[0]).all() and (b == b[0]).all() and a[0] != b[0]
    # Contrast: identity-metric k-means splits the nuisance axis instead.
    assign_e, _ = kmeans_whitened(M, 2, seed=0)
    agree = max(
        float((assign_e == assign_w).float().mean()),
        float((assign_e != assign_w).float().mean()),
    )
    assert agree < 0.9, "Euclidean clustering unexpectedly matched the metric split"


def test_consolidation_bit_accounting():
    """Hand-computed pins of the pre-registered storage charges + the floor
    matching invariant (maximal m that never exceeds the target)."""
    assert ceil_log2(1) == 0 and ceil_log2(2) == 1
    assert ceil_log2(1024) == 10 and ceil_log2(1025) == 11
    S, C = 1024, 768
    # pure: m fp16 rows + count (+ representative position iff RoPE)
    assert pure_bits(10, S, C, rope=False) == 10 * (16 * 768 + 10) == 122980
    assert pure_bits(10, S, C, rope=True) == 10 * (16 * 768 + 20) == 123080
    # assign: m fp16 rows + S-entry ceil_log2(m)-bit table
    assert assign_bits(16, S, C) == 16 * 16 * 768 + 1024 * 4 == 200704
    for target_bpe in (2.2, 2.5, 3.0):
        target = target_bpe * S * C
        for rope in (False, True):
            m = matched_m_pure(target, S, C, rope=rope)
            assert pure_bits(m, S, C, rope=rope) <= target
            assert pure_bits(m + 1, S, C, rope=rope) > target
        ma = matched_m_assign(target, S, C)
        assert assign_bits(ma, S, C) <= target
        assert assign_bits(ma + 1, S, C) > target


def test_m_equals_S_reproduces_exact_cache():
    """Degenerate case (pre-registered): m = S must reproduce the exact cache
    — every token its own centroid (counts all 1, representative positions
    the identity), the expanded matrix bit-equal to the source, and zero
    distortion through the shipped logit instrument. M is built from fp16
    values so the storage roundtrip is exact by construction."""
    g = torch.Generator().manual_seed(2)
    S, C, h_kv = 32, 8, 2
    M = torch.randn(S, C, generator=g).half().float()
    cons = consolidate(M, M.clone(), S, iters=10, seed=0, n_init=1)
    assert torch.equal(cons.counts, torch.ones(S, dtype=torch.int64))
    assert torch.equal(cons.rep_pos, torch.arange(S))
    assert torch.equal(expanded_key_matrix(cons), M)
    K = from_matrix(M, h_kv)
    Q = torch.randn(2 * h_kv, 8, C // h_kv, generator=g)
    assert logit_distortion(K, from_matrix(expanded_key_matrix(cons), h_kv), Q) == 0.0
    Vm = torch.randn(S, C, generator=g).half().float()
    assert torch.equal(cluster_mean_rows(Vm, cons), Vm)


def test_expanded_softmax_equals_merged_counts_softmax():
    """The evaluation embedding identity: the expanded S-row softmax
    denominator Σ_s exp(q·c_{a(s)}) equals the merged m-term count-weighted
    denominator Σ_j n_j exp(q·c_j) — the pure storage model needs no
    per-token assignment at read."""
    g = torch.Generator().manual_seed(3)
    S, C, m = 40, 6, 5
    M = torch.randn(S, C, generator=g)
    cons = consolidate(M, M.clone(), m, iters=50, seed=0, n_init=2)
    q = torch.randn(C, generator=g)
    expanded = (expanded_key_matrix(cons) @ q).exp().sum()
    merged = (cons.counts.float() * (cons.centroids @ q).exp()).sum()
    assert torch.allclose(expanded, merged, rtol=1e-5)


def test_storm_consolidation_probe_smoke(tmp_path):
    """Full experiment on a tiny factory cache + fitted pack file (identity
    whitener): parquet + verdict written, both budgets scored, every
    consolidation row at or under its pack bit target, gate evaluated."""
    import pandas as pd

    from experiments.storm_consolidation_probe import Config, main

    g = torch.Generator().manual_seed(4)
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

    bases = {}
    for li in range(2):
        Mfit = to_matrix(tensors[f"layer{li}.k_pre"])
        Ih, Ih_inv = identity_whitener(C)
        bases[li] = fit_spectral_basis(Mfit, Ih, Ih_inv)
    pack_path = tmp_path / "packs.safetensors"
    save_pack_file(str(pack_path), bases, (2.2, 2.5), group=16)

    cfg = Config(
        cache_path=str(cache_path),
        pack_path=str(pack_path),
        model_label="tiny",
        model_name="",
        budgets=(2.2, 2.5),
        kmeans_iters=25,
        kmeans_restarts=2,
        out_root=str(tmp_path / "results"),
    )
    run_dir = main(cfg)
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert {
        "model",
        "layer_class",
        "budget",
        "arm",
        "m",
        "m_over_S",
        "bpe",
        "bpe_target",
        "dist",
        "worst_ratio",
        "out_err",
        "out_err_causal",
    } <= set(df.columns)
    assert set(df.budget.unique()) == {2.2, 2.5}
    assert set(df.arm.unique()) == {"pack", "cons_pure", "cons_assign"}  # no RoPE arm
    cons_rows = df[df.arm != "pack"]
    assert (cons_rows.bpe <= cons_rows.bpe_target + 1e-9).all()
    assert (cons_rows.m >= 1).all() and (cons_rows.m < S).all()

    verdict = json.loads((run_dir / "verdict.json").read_text())
    assert "gate_outcome" in verdict and "confirmed" in verdict
    assert set(verdict["per_layer_class"]) == {"friendly", "steep"}
    for blk in verdict["per_layer_class"].values():
        assert set(blk["per_budget"]) == {"2.2", "2.5"}
        for b in blk["per_budget"].values():
            assert "cons_pure" in b["arms"] and "cons_assign" in b["arms"]
            assert isinstance(b["arms"]["cons_pure"]["holds"], bool)
