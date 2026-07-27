"""Storm Task-8 — attention predictability: offline pins.

Plan-required pins (docs/superpowers/plans/2026-07-26-storm-gates.md Task 8):
  (1) recall@k arithmetic on PLANTED attention maps — a static map must give
      recall exactly 1.0; a permuted (disjoint-support) map must give the
      known lower value (0.0 for identity; exactly 0.5 when half the realized
      mass sits in the recency window and the recency-augmented predictor
      catches it);
  (2) the mass-coverage computation on planted numbers.

Plus the risky mechanics: predictor-set budgets/membership (recency union
excludes double-counted positions), the sink/recent/content partition,
EMA-update arithmetic over the growing support, the micro-aggregation and
gate arithmetic (boundary inclusive), and an end-to-end tiny-factory smoke of
run_for_model + assemble_verdict on both model families (offline, no
downloads — gpt2-style and qwen3-style tiny factories).
"""

import math

import pandas as pd
import pytest
import torch

from experiments.storm_attn_predict import (
    CLASSES,
    PREDICTORS,
    Config,
    agg_micro,
    assemble_verdict,
    class_masks,
    ema_update,
    gate_eval,
    predictor_sets,
    score_sets,
    set_mask,
)
from factories import ids, tiny_gpt2, tiny_qwen3

# ---------------------------------------------------------------------------
# Planted-map recall@k pins (plan-required)
# ---------------------------------------------------------------------------


def _planted(peaks: dict[int, float], S: int) -> torch.Tensor:
    a = torch.zeros(1, S)
    for pos, mass in peaks.items():
        a[0, pos] = mass
    return a


def test_static_map_recall_exactly_one():
    """Same attention every step (mass 0.4/0.3/0.2 on fixed positions): the
    identity predictor's recall must be exactly 1.0 and its mass coverage
    exactly the planted 0.9 (extra top-k picks land on zero-mass positions)."""
    a_prev = _planted({2: 0.4, 7: 0.3, 11: 0.2}, 31)
    a_next = _planted({2: 0.4, 7: 0.3, 11: 0.2}, 32)  # one new (zero-mass) pos
    sets = predictor_sets(a_prev, ema_prev=a_prev, k=5, n_recent=1)
    masks = class_masks(32, n_sink=1, n_recent=1)
    for pred in PREDICTORS:
        s = score_sets(set_mask(sets[pred], 32), a_next, 0.01, masks)
        assert int(s["n_realized"]) == 3
        assert int(s["n_hit"]) == 3, f"{pred} must catch a static map"
        assert float(s["mass_cov"]) == pytest.approx(0.9, abs=1e-7)


def test_permuted_map_identity_recall_zero():
    """t+1's realized set disjoint from t's top-k: identity recall exactly 0
    and mass coverage exactly 0."""
    a_prev = _planted({2: 0.4, 7: 0.3, 11: 0.2}, 31)
    a_next = _planted({5: 0.45, 20: 0.45}, 32)
    sets = predictor_sets(a_prev, ema_prev=a_prev, k=3, n_recent=1)
    masks = class_masks(32, n_sink=1, n_recent=1)
    s = score_sets(set_mask(sets["identity"], 32), a_next, 0.01, masks)
    assert int(s["n_realized"]) == 2
    assert int(s["n_hit"]) == 0
    assert float(s["mass_cov"]) == 0.0


def test_recency_augmentation_catches_the_newest_position():
    """Half the realized mass moves to the brand-new position (index 31 —
    invisible to the identity predictor by construction): identity recall is
    exactly 0.5's complement structure — identity 0.0 on it, identity_recency
    exactly 0.5 with mass coverage exactly 0.45."""
    a_prev = _planted({2: 0.4, 7: 0.3, 11: 0.2}, 31)
    a_next = _planted({5: 0.45, 31: 0.45}, 32)
    sets = predictor_sets(a_prev, ema_prev=a_prev, k=3, n_recent=1)
    masks = class_masks(32, n_sink=1, n_recent=1)
    s_id = score_sets(set_mask(sets["identity"], 32), a_next, 0.01, masks)
    s_rec = score_sets(set_mask(sets["identity_recency"], 32), a_next, 0.01, masks)
    assert int(s_id["n_hit"]) == 0
    assert int(s_rec["n_realized"]) == 2
    assert int(s_rec["n_hit"]) == 1  # recall exactly 0.5
    assert float(s_rec["mass_cov"]) == pytest.approx(0.45, abs=1e-7)


def test_mass_coverage_is_threshold_free():
    """Mass coverage counts ALL covered mass, including sub-threshold tokens:
    predicted set covering a 0.5%-mass token still earns that mass."""
    a_prev = _planted({2: 0.9, 7: 0.005}, 31)  # 7 is sub-threshold at X=1%
    a_next = _planted({2: 0.9, 7: 0.005}, 32)
    sets = predictor_sets(a_prev, ema_prev=a_prev, k=2, n_recent=1)
    masks = class_masks(32, n_sink=1, n_recent=1)
    s = score_sets(set_mask(sets["identity"], 32), a_next, 0.01, masks)
    assert int(s["n_realized"]) == 1  # only the 0.9 token clears 1%
    assert float(s["mass_cov"]) == pytest.approx(0.905, abs=1e-7)


# ---------------------------------------------------------------------------
# Predictor-set mechanics
# ---------------------------------------------------------------------------


def test_predictor_sets_budgets_and_membership():
    torch.manual_seed(0)
    a_prev = torch.rand(3, 40)
    ema_prev = torch.rand(3, 40)
    k, n_recent = 6, 2
    sets = predictor_sets(a_prev, ema_prev, k=k, n_recent=n_recent)
    assert set(sets) == set(PREDICTORS)
    for name, idx in sets.items():
        assert idx.shape == (3, k)  # every predictor spends exactly budget k
        assert int(idx.max()) < 41 and int(idx.min()) >= 0
        # No duplicate indices inside a row (recency union must not double-count)
        for r in range(3):
            assert len(set(idx[r].tolist())) == k, name
    # Recency variants contain ALL n_recent newest positions (39, 40)
    for name in ("identity_recency", "ema_recency"):
        for r in range(3):
            assert {39, 40} <= set(sets[name][r].tolist())
    # identity is exactly the top-k of a_prev; ema of ema_prev
    assert torch.equal(
        sets["identity"].sort(-1).values, a_prev.topk(k, -1).indices.sort(-1).values
    )
    assert torch.equal(
        sets["ema"].sort(-1).values, ema_prev.topk(k, -1).indices.sort(-1).values
    )


def test_predictor_sets_rejects_bad_budget():
    a = torch.rand(1, 10)
    with pytest.raises(AssertionError):
        predictor_sets(a, a, k=2, n_recent=2)  # n_recent must be < k
    with pytest.raises(AssertionError):
        predictor_sets(a, a, k=11, n_recent=1)  # k must be <= S_prev


# ---------------------------------------------------------------------------
# Partition + decomposition + EMA arithmetic
# ---------------------------------------------------------------------------


def test_class_masks_partition():
    masks = class_masks(10, n_sink=2, n_recent=2)
    total = masks["sink"] | masks["recent"] | masks["content"]
    assert bool(total.all())  # partition covers every position
    assert int((masks["sink"] & masks["recent"]).sum()) == 0
    assert masks["sink"].nonzero().flatten().tolist() == [0, 1]
    assert masks["recent"].nonzero().flatten().tolist() == [8, 9]
    with pytest.raises(AssertionError):
        class_masks(3, n_sink=2, n_recent=2)  # overlap must fail loudly


def test_score_sets_class_decomposition_exact():
    """Planted events in each class: per-class counts and masses pin exactly."""
    masks = class_masks(10, n_sink=2, n_recent=2)
    a_next = _planted({0: 0.3, 5: 0.3, 9: 0.3, 3: 0.1}, 10)
    pred = torch.tensor([[0, 5]])
    s = score_sets(set_mask(pred, 10), a_next, 0.01, masks)
    assert int(s["n_realized"]) == 4 and int(s["n_hit"]) == 2
    assert int(s["n_realized_sink"]) == 1 and int(s["n_hit_sink"]) == 1
    assert int(s["n_realized_recent"]) == 1 and int(s["n_hit_recent"]) == 0
    assert int(s["n_realized_content"]) == 2 and int(s["n_hit_content"]) == 1
    assert float(s["mass_sink"]) == pytest.approx(0.3)
    assert float(s["mass_recent"]) == pytest.approx(0.3)
    assert float(s["mass_content"]) == pytest.approx(0.4)
    assert float(s["mass_cov_sink"]) == pytest.approx(0.3)
    assert float(s["mass_cov_recent"]) == 0.0
    assert float(s["mass_cov_content"]) == pytest.approx(0.3)


def test_score_sets_batch_matches_reference():
    """The batched scorer the loop actually runs must agree EXACTLY with the
    reference score_sets on random normalized rows (oracle-gated fast path)."""
    from experiments.storm_attn_predict import score_sets_batch, set_mask_batch

    torch.manual_seed(1)
    R, S_prev = 5, 60
    a_prev = torch.softmax(torch.randn(R, S_prev) * 3, dim=-1)
    ema_prev = torch.softmax(torch.randn(R, S_prev) * 3, dim=-1)
    a_next = torch.softmax(torch.randn(R, S_prev + 1) * 3, dim=-1)
    S_next = S_prev + 1
    masks = class_masks(S_next, n_sink=4, n_recent=8)
    sets = predictor_sets(a_prev, ema_prev, k=12, n_recent=8)
    stacked = set_mask_batch(torch.stack([sets[p] for p in PREDICTORS]), S_next)
    x_fracs = [0.005, 0.01, 0.02]
    batched = score_sets_batch(stacked, a_next, x_fracs, masks)
    for pi, pred in enumerate(PREDICTORS):
        pmask = set_mask(sets[pred], S_next)
        assert torch.equal(stacked[pi], pmask)
        for xf in x_fracs:
            ref = score_sets(pmask, a_next, xf, masks)
            for f, v in ref.items():
                got = batched[xf][f][pi]
                assert torch.allclose(got.double(), v.double(), atol=1e-6), (
                    f"{pred}/{xf}/{f}"
                )


def test_ema_update_arithmetic():
    ema_prev = torch.tensor([[0.4, 0.6, 0.0, 0.0]])
    a_next = torch.tensor([[0.0, 0.2, 0.2, 0.2, 0.4]])
    out = ema_update(ema_prev, a_next, lam=0.5)
    # (1-lam)*pad + lam*a: new position enters with zero history
    assert torch.allclose(out, torch.tensor([[0.2, 0.4, 0.1, 0.1, 0.2]]))
    with pytest.raises(AssertionError):
        ema_update(ema_prev, torch.zeros(1, 6), lam=0.5)  # must grow by one


# ---------------------------------------------------------------------------
# Aggregation + gate arithmetic
# ---------------------------------------------------------------------------


def test_agg_micro_planted():
    df = pd.DataFrame(
        [
            dict(n_realized=4.0, n_hit=3.0, n_steps=10, mass_cov_sum=8.0),
            dict(n_realized=1.0, n_hit=1.0, n_steps=10, mass_cov_sum=6.0),
        ]
    )
    out = agg_micro(df)
    assert out["recall"] == pytest.approx(4 / 5)  # micro: (3+1)/(4+1)
    assert out["mass_coverage"] == pytest.approx(14 / 20)
    assert out["n_realized"] == 5.0


def test_gate_eval_boundary_inclusive_and_kill_branch():
    g = gate_eval({"gpt2": 0.95, "qwen3-0.6b": 0.9})
    assert g["gate_pass"] is True  # 0.9 exactly clears (>= is inclusive)
    assert "graduates to a spec" in g["gate_outcome"]
    g = gate_eval({"gpt2": 0.95, "qwen3-0.6b": 0.89})
    assert g["gate_pass"] is False  # ANY model below kills
    assert g["min_model"] == "qwen3-0.6b"
    assert g["min_recall"] == pytest.approx(0.89)
    assert "killed by prefetch pollution" in g["gate_outcome"]


# ---------------------------------------------------------------------------
# End-to-end tiny-factory smoke (offline, both model families)
# ---------------------------------------------------------------------------


def _smoke_cfg() -> Config:
    return Config(
        prompt_len=24,
        decode_steps=8,
        x_pct_grid=(1.0, 5.0),
        k_frac_grid=(0.1, 0.2),
        gate_x_pct=1.0,
        gate_k_frac=0.1,
        n_recent=1,
        n_sink=1,
    )


@pytest.mark.parametrize(
    "factory,n_layers,n_heads",
    [(tiny_gpt2, 2, 2), (tiny_qwen3, 2, 4)],
    ids=["gpt2-family", "qwen3-family"],
)
def test_run_for_model_smoke_invariants(factory, n_layers, n_heads):
    cfg = _smoke_cfg()
    rows, steps, meta = run_smoke(cfg, factory)
    assert meta["n_layers"] == n_layers and meta["n_heads"] == n_heads
    df = pd.DataFrame(rows)
    # Row census: layers x predictors x grid x (heads + pooled)
    assert len(df) == n_layers * 4 * 2 * 2 * (n_heads + 1)
    pooled = df[df.scope == "layer_pooled"]
    assert set(pooled["head"].unique()) == {-1}
    assert set(df[df.scope == "head"]["head"].unique()) == set(range(n_heads))
    # recall in [0,1] (or nan when a head-cell saw no realized events)
    live = df[df.n_realized > 0]
    assert ((live.recall >= 0) & (live.recall <= 1)).all()
    assert ((df.mass_cov_mean >= 0) & (df.mass_cov_mean <= 1 + 1e-5)).all()
    # sink/recent/content partition the events and the hits exactly
    for f in ("n_realized", "n_hit"):
        parts = sum(df[f"{f}_{c}"] for c in CLASSES)
        assert (parts == df[f]).all()
    # class masses partition the (normalized) read mass: sum ~= n_steps
    mass_parts = sum(df[f"mass_{c}_sum"] for c in CLASSES)
    assert mass_parts.to_numpy() == pytest.approx(df.n_steps.to_numpy(), rel=1e-3)
    # per-step rows: (decode_steps - 1) x predictors x scopes
    sdf = pd.DataFrame(steps)
    assert len(sdf) == (cfg.decode_steps - 1) * 4 * 2
    assert set(sdf.predictor.unique()) == set(PREDICTORS)


def run_smoke(cfg: Config, factory):
    from experiments.storm_attn_predict import run_for_model

    model = factory()
    tokens = ids(vocab=97, seq=cfg.prompt_len + cfg.decode_steps, seed=7)
    return run_for_model(cfg, "tiny", "tiny", model=model, tokens=tokens)


def test_assemble_verdict_smoke_gate_consistency():
    cfg = _smoke_cfg()
    rows, steps, meta = run_smoke(cfg, tiny_gpt2)
    df, sdf = pd.DataFrame(rows), pd.DataFrame(steps)
    verdict = assemble_verdict(df, sdf, cfg, [meta])
    pm = verdict["per_model"]["tiny"]
    assert pm["best_predictor"] in PREDICTORS
    assert verdict["gate"]["gate_pass"] == (
        pm["best"]["recall"] >= verdict["gate"]["threshold"]
    )
    assert verdict["pre_registered"]["gate_x_pct"] == cfg.gate_x_pct
    # grid carries all 4 cells and the gate cell agrees with `best`
    assert len(pm["grid"]) == len(cfg.x_pct_grid) * len(cfg.k_frac_grid)
    gate_cell = pm["grid"][f"X={cfg.gate_x_pct:g}%|k={cfg.gate_k_frac:g}S"]
    assert gate_cell["recall"] == pytest.approx(pm["best"]["recall"])
    # decomposition shares sum to 1 across the partition
    dec = pm["decomposition"]
    assert sum(dec[c]["realized_event_share"] for c in CLASSES) == pytest.approx(1.0)
    assert sum(dec[c]["realized_mass_share"] for c in CLASSES) == pytest.approx(
        1.0, rel=1e-3
    )
    assert dec["static_realized_mass_share"] <= 1.0 + 1e-6
    # stability is over the per-step series at the best predictor
    assert pm["stability"]["n_steps"] == cfg.decode_steps - 1


def test_k_schedule_matches_preregistration():
    """The gate k is ceil(0.05 * S_{t+1}) — pin the arithmetic the loop uses."""
    assert max(1, math.ceil(0.05 * 1025)) == 52
    assert max(1, math.ceil(0.025 * 769)) == 20  # smallest grid cell, gpt2 S_min
    cfg = Config()
    assert cfg.gate_x_pct == 1.0 and cfg.gate_k_frac == 0.05  # pre-registered
    assert cfg.n_recent == 16 and cfg.n_sink == 4
    assert cfg.gate_x_pct in cfg.x_pct_grid and cfg.gate_k_frac in cfg.k_frac_grid
