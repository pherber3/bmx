"""Storm Task-2 — leverage-score token coreset gate: offline pins.

Tiny-factory (no model, no download) tests for the load-bearing pieces the
pre-registered gate rests on: (1) the leverage computation must select a
planted high-leverage row (and ALWAYS the sinks), (2) the plan-locked bit
accounting formula 16*k*C/S + ceil(log2 S)*k/S bits/token and its downward
(floor) matching, (3) the drop-mass account against a hand-computed softmax,
and (4) the zero-row-embedding identity the shipped-instrument scoring path
reduces to for a coreset.
"""

import json
import math

import torch

from bmx.cache.collect import from_matrix, save_cache, to_matrix
from bmx.cache.metrics import logit_distortion
from experiments.storm_coreset_gate import (
    N_SINKS,
    coreset_bpe,
    coreset_reconstruct,
    coreset_total_bits,
    drop_diagnostics,
    leverage_scores,
    matched_coreset_k,
    mixed_arm,
    select_coreset,
    stable_rank,
    token_index_bits,
    true_causal_logits,
)


def _planted_matrix(S=64, C=16, dim=3, needle_row=37, seed=0):
    """Rows live in a `dim`-dimensional subspace, EXCEPT `needle_row`, which is
    a moderate-norm row along a direction orthogonal to that subspace — the
    canonical high-leverage row (it owns its own singular direction). The sink
    rows 0..3 are low-leverage bulk rows (subspace combinations)."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(C, dim + 1, generator=g, dtype=torch.float64)
    basis, _ = torch.linalg.qr(raw)
    dirs, ortho = basis[:, :dim], basis[:, dim]
    z = torch.randn(S, dim, generator=g, dtype=torch.float64) * 10.0
    M = z @ dirs.mT
    M[needle_row] = 2.0 * ortho  # small norm, but the ONLY mass off-subspace
    return M.float()


def test_leverage_selects_planted_high_leverage_row():
    """A moderate-norm row orthogonal to the bulk subspace must carry ~unit
    leverage in the top-r subspace (r > dim) and be the argmax — leverage is
    about direction ownership, not row norm."""
    S, needle = 64, 37
    M = _planted_matrix(S=S, needle_row=needle)
    lev = leverage_scores(M, r=8)
    assert lev.shape == (S,)
    assert int(lev.argmax()) == needle
    assert float(lev[needle]) > 0.9  # the row owns its singular direction
    # Leverage scores of a top-r subspace sum to r and lie in [0, 1].
    assert abs(float(lev.sum()) - 8.0) < 1e-4
    assert float(lev.max()) <= 1.0 + 1e-5 and float(lev.min()) >= -1e-6


def test_select_coreset_includes_needle_and_always_the_sinks():
    """The keep set must contain the planted needle row AND positions 0..3
    (sinks are unconditional — plan Task 2), with exactly k sorted uniques."""
    M = _planted_matrix(needle_row=37)
    lev = leverage_scores(M, r=8)
    # The sink rows are bulk rows: their leverage is far below the needle's.
    assert float(lev[:N_SINKS].max()) < float(lev[37])
    keep = select_coreset(lev, k=8, n_sinks=N_SINKS)
    keep_set = set(keep.tolist())
    assert keep.numel() == 8 and len(keep_set) == 8
    assert {0, 1, 2, 3} <= keep_set
    assert 37 in keep_set
    assert torch.equal(keep, keep.sort().values)


def test_coreset_bit_accounting_formula():
    """The plan-locked formula: total = k*(16*C + ceil(log2 S)); per token
    16*k*C/S + ceil(log2 S)*k/S; matched_coreset_k is the floor inverse
    (never exceeds the target bpe, deficit < one token's cost)."""
    S, C = 1024, 768
    assert token_index_bits(S) == 10
    for k in (4, 129, 500):
        total = coreset_total_bits(k, S, C)
        assert total == k * (16 * C + 10)
        # bits/token identity, exactly as the plan states it
        assert abs(total / S - (16 * k * C / S + 10 * k / S)) < 1e-9
        assert abs(coreset_bpe(k, S, C) - total / (S * C)) < 1e-15
    for target in (2.0208, 2.5, 3.25, 4.25):
        k = matched_coreset_k(target, S, C)
        assert coreset_bpe(k, S, C) <= target + 1e-12  # never over budget
        assert coreset_bpe(k + 1, S, C) > target  # largest such k
        # deficit below one token's amortized cost
        assert target - coreset_bpe(k, S, C) < (16 * C + 10) / (S * C)
    # clamp: never below the sink count
    assert matched_coreset_k(1e-6, S, C) == N_SINKS


def test_mixed_arm_accounting_and_structure():
    """Mixed arm: measured bpe <= target; window rows quantized (not exact),
    kept tail rows exact, dropped tail rows zero, sinks retained exactly."""
    g = torch.Generator().manual_seed(1)
    S, C = 128, 16
    M = torch.randn(S, C, generator=g)
    lev = leverage_scores(M, r=4)
    target = 3.0
    M_hat, bpe, k_tail, W = mixed_arm(M, target, lev, level_bits=3.0)
    assert bpe <= target + 1e-9
    assert W == S // 4 and N_SINKS <= k_tail <= S - W
    # accounting identity: window (turboquant bpe) + tail rows + boundary int
    idxb = token_index_bits(S)
    window_bits = bpe * S * C - k_tail * (16 * C + idxb) - idxb
    assert window_bits > 0
    # sinks exact
    assert torch.equal(M_hat[:N_SINKS], M[:N_SINKS])
    # window rows are quantized: none exactly equal, all nonzero
    assert not torch.equal(M_hat[S - W :], M[S - W :])
    assert (M_hat[S - W :].abs().sum(dim=1) > 0).all()
    # tail rows are exact-or-zero
    tail = M_hat[: S - W]
    exact = (tail == M[: S - W]).all(dim=1)
    zero = (tail == 0).all(dim=1)
    assert (exact | zero).all()
    assert int(exact.sum()) == k_tail


def test_drop_mass_account_matches_hand_softmax():
    """h=1, T=1 query at the last position over S=4 keys: the drop mass must
    equal the hand-computed softmax probability of the dropped position, and
    the worst-case number must equal that position's |scaled logit|."""
    d = 2
    k_post = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.5, 0.5]]])
    q = torch.tensor([[[1.0, 1.0]]])  # (h=1, T=1, d)
    logits, causal = true_causal_logits(q, k_post)
    assert causal.all()  # last-position query sees all 4 keys
    hand = (q[0, 0] @ k_post[0].mT) / math.sqrt(d)  # (4,)
    assert torch.allclose(logits[0, 0], hand, atol=1e-6)
    probs = torch.softmax(hand, dim=0)
    retained = torch.tensor([0, 1, 3])  # drop position 2 (the largest logit)
    dm, worst, max_true = drop_diagnostics(logits, causal, retained)
    assert abs(dm - float(probs[2])) < 1e-6
    assert abs(worst - float(hand[2].abs())) < 1e-6
    assert abs(max_true - float(hand.abs().max())) < 1e-6
    # boundary cases: retain everything -> 0; retain nothing -> 1
    dm_all, worst_all, _ = drop_diagnostics(logits, causal, torch.arange(4))
    assert dm_all == 0.0 and worst_all == 0.0
    dm_none, _, _ = drop_diagnostics(logits, causal, torch.arange(0))
    assert abs(dm_none - 1.0) < 1e-6


def test_drop_mass_respects_causal_mask():
    """A dropped position AFTER every query position must carry zero mass and
    zero worst-case logit (no query can see it)."""
    g = torch.Generator().manual_seed(2)
    h, T, S, d = 2, 4, 16, 8
    q = torch.randn(h, T, d, generator=g)
    k_post = torch.randn(1, S, d, generator=g)
    logits, causal = true_causal_logits(q, k_post)
    # queries sit at positions S-T..S-1; position S-1 is visible only to the
    # last query; drop it and mass must be strictly between 0 and 1.
    retained = torch.arange(S - 1)
    dm, worst, _ = drop_diagnostics(logits, causal, retained)
    assert 0.0 < dm < 1.0 and worst > 0.0
    # per-position mass is a distribution: complementary sets sum to 1
    dm_c, _, _ = drop_diagnostics(logits, causal, torch.tensor([S - 1]))
    assert abs(dm + dm_c - 1.0) < 1e-5


def test_coreset_scoring_reduces_to_dropped_column_energy():
    """Zero-row embedding identity: with kept rows exact, the shipped
    logit_distortion of the coreset reconstruction equals, per head, the
    Frobenius ratio of the DROPPED logit columns to all logit columns."""
    g = torch.Generator().manual_seed(3)
    h_kv, h, S, d = 2, 4, 32, 4
    K = torch.randn(h_kv, S, d, generator=g, dtype=torch.float64).float()
    Q = torch.randn(h, 8, d, generator=g, dtype=torch.float64).float()
    M = to_matrix(K)  # (S, h_kv*d) — the one sanctioned layout helper
    keep = select_coreset(torch.rand(S, generator=g), k=10, n_sinks=N_SINKS)
    K_hat = from_matrix(coreset_reconstruct(M, keep), h_kv).float()
    got = logit_distortion(K, K_hat, Q)
    # direct: per (GQA-expanded) head, ||L[:, dropped]||_F / ||L||_F
    dropped = torch.ones(S, dtype=torch.bool)
    dropped[keep] = False
    K_exp = K.repeat_interleave(h // h_kv, dim=0).double()
    L = Q.double() @ K_exp.transpose(-1, -2)  # (h, T, S)
    per_head = L[:, :, dropped].flatten(1).norm(dim=1) / L.flatten(1).norm(dim=1)
    assert abs(got - float(per_head.mean())) < 1e-5


def test_storm_coreset_gate_smoke(tmp_path):
    """Full experiment on a tiny factory cache (no packs, no RoPE): parquet +
    verdict written, gate evaluated verbatim, all rows carry the accounting."""
    import pandas as pd

    from experiments.storm_coreset_gate import Config, main

    g = torch.Generator().manual_seed(4)
    S, C, h_kv, T = 128, 16, 2, 16
    d = C // h_kv
    raw = torch.randn(C, 3, generator=g)
    dirs, _ = torch.linalg.qr(raw)
    z = torch.randn(S, 3, generator=g) * torch.tensor([20.0, 12.0, 8.0])
    M = (z @ dirs.mT + torch.randn(S, C, generator=g)).half()
    K = from_matrix(M, h_kv).contiguous()
    tensors = {}
    for i in range(2):
        tensors[f"layer{i}.k_pre"] = K.clone()
        tensors[f"layer{i}.k"] = K.clone()
        tensors[f"layer{i}.q"] = torch.randn(h_kv * 2, T, d, generator=g).half()
        tensors[f"layer{i}.v"] = torch.randn(h_kv, S, d, generator=g).half()
    cache_path = tmp_path / "cache.safetensors"
    save_cache(tensors, str(cache_path))

    cfg = Config(
        cache_path=str(cache_path),
        pack_path="",
        model_label="tiny",
        model_name="",
        rtn_bits=(2,),
        tq_bits=(2,),
        spectral_budgets=(),
        r_grid=(4,),
        use_stable_rank_r=True,
        mixed_r=4,
        group=16,
        out_root=str(tmp_path / "results"),
    )
    run_dir = main(cfg)
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert set(df.kind.unique()) == {"uniform", "coreset", "mixed"}
    # every selection row is at-or-under its matched target
    sel = df[df.kind.isin(["coreset", "mixed"])]
    assert (sel.bpe <= sel.bpe_target + 1e-9).all()
    assert (sel.drop_mass >= 0).all() and (sel.drop_mass <= 1 + 1e-6).all()
    verdict = json.loads((run_dir / "verdict.json").read_text())
    assert verdict["gate_win_frac"] == 0.9
    assert "gate_outcome" in verdict and "confirmed" in verdict
    for block in verdict["per_level"].values():
        assert 0.0 <= block["best_variant_win_frac"] <= 1.0
        for v in block["per_variant"].values():
            assert 0.0 <= v["layer_win_frac"] <= 1.0


def test_stable_rank_of_isotropic_matrix_is_high():
    g = torch.Generator().manual_seed(5)
    M = torch.randn(256, 32, generator=g)
    _, svals, _ = torch.linalg.svd(M, full_matrices=False)
    sr = stable_rank(svals)
    assert 10.0 < sr <= 32.0  # near-isotropic: stable rank ~ C
    # rank-1 matrix: stable rank ~ 1
    M1 = torch.outer(torch.randn(256, generator=g), torch.randn(32, generator=g))
    _, sv1, _ = torch.linalg.svd(M1, full_matrices=False)
    assert stable_rank(sv1) < 1.5
