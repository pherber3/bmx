"""Storm Task-5 — prompt-policy pack robustness: offline pins.

Plan-required pins (docs/superpowers/plans/2026-07-26-storm-gates.md Task 5):
  (1) chat-wrap inertness — collect_cache's --chat-wrap flag OFF is BYTE-EXACT
      inert (the identical token object passes through untouched, and the
      corpus-label guard fires before any model/tokenizer load when it's on
      without a label);
  (2) retention arithmetic on planted numbers (retention_ratios + gate_eval,
      both the ROBUST and FLAG branches, threshold boundary inclusive).

Plus the risky mechanics: chat_wrap_token_ids' exact-target-length trimming on
a reversible stub tokenizer (including the fail-fast path when no content
length can hit the target), the policy_shift_gap identity/symmetry properties,
and an end-to-end tiny-factory smoke of the experiment main (no model, no
download, no RoPE).
"""

import json

import pytest
import torch

from bmx.cache.collect import save_cache
from bmx.eval.layer_swap import chat_wrap_token_ids
from experiments.collect_cache import Config as CollectConfig
from experiments.collect_cache import _maybe_chat_wrap
from experiments.collect_cache import main as collect_main
from experiments.storm_policy_robustness import (
    GATE_RETENTION,
    gate_eval,
    policy_shift_gap,
    retention_ratios,
)

# ---------------------------------------------------------------------------
# (1) chat-wrap inertness (flag off = byte-exact) + guard
# ---------------------------------------------------------------------------


def test_chat_wrap_flag_off_is_byte_exact_inert():
    """Default Config (chat_wrap=False) passes the token stream through as the
    IDENTICAL object — the shipped collect path is untouched by the new flag."""
    tokens = torch.arange(100)
    cfg = CollectConfig()
    assert cfg.chat_wrap is False  # default off
    out = _maybe_chat_wrap(cfg, tokens)
    assert out is tokens  # same object, not merely equal — byte-exact inert


def test_chat_wrap_requires_corpus_label():
    """--chat-wrap without --corpus-label must fail loudly BEFORE any model or
    tokenizer load (the wikitext-cache-name guard, extended to prompt policy)."""
    with pytest.raises(AssertionError, match="corpus-label"):
        collect_main(CollectConfig(chat_wrap=True))


# ---------------------------------------------------------------------------
# chat_wrap_token_ids mechanics on a reversible stub tokenizer
# ---------------------------------------------------------------------------


class _StubTok:
    """Reversible word-level stub: token id i <-> word f"w{i}". The chat
    template adds 2 header markers + 1 trailer marker (ids 9001/9002/9003), so
    wrapped length = n_content + 3 exactly."""

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{int(t)}" for t in ids.tolist())

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        return_tensors=None,
    ):
        assert messages[0]["role"] == "user"
        assert add_generation_prompt and tokenize and return_dict
        content = [int(w[1:]) for w in messages[0]["content"].split()]
        return {"input_ids": torch.tensor([[9001, 9002] + content + [9003]])}


class _UnreachableTok(_StubTok):
    """Wrapped length = n_content + 3 + (n_content % 2): every odd n adds one
    extra token, making EVEN wrapped totals with odd overhead unreachable for
    some targets — exercises the fail-fast path."""

    def apply_chat_template(self, messages, **kw):
        out = super().apply_chat_template(messages, **kw)
        ids = out["input_ids"]
        n_content = ids.shape[1] - 3
        if n_content % 2:
            ids = torch.cat([ids, torch.tensor([[9004]])], dim=1)
        return {"input_ids": ids}


def test_chat_wrap_token_ids_exact_length_and_structure():
    tokens = torch.arange(64)
    wrapped = chat_wrap_token_ids(_StubTok(), tokens, target_len=32)
    assert wrapped.numel() == 32  # EXACT target length
    assert wrapped[:2].tolist() == [9001, 9002]  # role header present
    assert int(wrapped[-1]) == 9003  # generation-prompt trailer present
    # Content is the PREFIX of the raw stream (same underlying text).
    assert wrapped[2:-1].tolist() == list(range(29))


def test_chat_wrap_token_ids_target_equals_stream_length():
    """target_len == stream length: content must self-trim by the overhead."""
    tokens = torch.arange(50)
    wrapped = chat_wrap_token_ids(_StubTok(), tokens, target_len=50)
    assert wrapped.numel() == 50
    assert wrapped[2:-1].tolist() == list(range(47))


def test_chat_wrap_token_ids_fails_fast_when_unreachable():
    """_UnreachableTok: length(n) = n + 3 + (n%2) takes only ODD values >= 4
    at even n (n+3) and odd-n values are also odd (n+4) — so an EVEN target is
    unreachable and must raise, never silently return a near-miss."""
    tokens = torch.arange(64)
    lengths = {n + 3 + (n % 2) for n in range(1, 60)}
    assert 32 not in lengths  # sanity: the target really is unreachable
    with pytest.raises(AssertionError, match="wraps to exactly"):
        chat_wrap_token_ids(_UnreachableTok(), tokens, target_len=32)


# ---------------------------------------------------------------------------
# (2) retention arithmetic + gate on planted numbers
# ---------------------------------------------------------------------------


def test_retention_ratios_planted():
    wins = {
        ("raw", "raw"): 2.0,  # same-policy baseline for the raw heldout
        ("chat", "chat"): 1.6,  # same-policy baseline for the chat heldout
        ("raw", "chat"): 1.2,  # raw pack scored on chat heldout
        ("chat", "raw"): 1.9,  # chat pack scored on raw heldout
    }
    r = retention_ratios(wins)
    assert r["raw_to_chat"] == pytest.approx(1.2 / 1.6)  # 0.75
    assert r["chat_to_raw"] == pytest.approx(1.9 / 2.0)  # 0.95


def test_gate_eval_flag_branch_and_min_localization():
    gate = gate_eval(
        {
            "2.2": {"raw_to_chat": 0.75, "chat_to_raw": 0.95},
            "2.5": {"raw_to_chat": 0.92, "chat_to_raw": 1.01},
        }
    )
    assert gate["gate_pass"] is False
    assert gate["min_retention"] == pytest.approx(0.75)
    assert gate["min_at"] == "budget 2.2 raw_to_chat"
    assert "FLAG" in gate["gate_outcome"]
    assert "refit" in gate["gate_outcome"]  # the flag branch specs the note


def test_gate_eval_robust_branch_and_boundary_inclusive():
    gate = gate_eval(
        {
            "2.2": {"raw_to_chat": GATE_RETENTION, "chat_to_raw": 1.10},
            "2.5": {"raw_to_chat": 0.95, "chat_to_raw": 1.02},
        }
    )
    assert gate["gate_pass"] is True  # exactly 0.9 passes (gate is >= 0.9)
    assert "ROBUST" in gate["gate_outcome"]
    assert gate["threshold"] == GATE_RETENTION == 0.9


# ---------------------------------------------------------------------------
# policy_shift_gap properties
# ---------------------------------------------------------------------------


def test_policy_shift_gap_zero_iff_equal_positive_and_symmetric():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(64, 8, generator=g, dtype=torch.float64)
    Y = torch.randn(64, 8, generator=g, dtype=torch.float64) * 2.0
    A = X.mT @ X / 64 + 0.1 * torch.eye(8, dtype=torch.float64)
    B = Y.mT @ Y / 64 + 0.1 * torch.eye(8, dtype=torch.float64)
    assert policy_shift_gap(A, A.clone()) == pytest.approx(0.0, abs=1e-9)
    gap_ab = policy_shift_gap(A, B)
    assert gap_ab > 0.0  # Minkowski/Jensen: strict for distinct moments
    assert policy_shift_gap(B, A) == pytest.approx(gap_ab, rel=1e-9)


# ---------------------------------------------------------------------------
# End-to-end tiny-factory smoke (no model, no download, no RoPE)
# ---------------------------------------------------------------------------


def _tiny_cache(seed: int, S=128, C=16, h_kv=2, T=16, n_layers=2) -> dict:
    """Low-rank-plus-noise factory cache matching the collect layout (k_pre/
    k/q/v per layer, fp16; q has h = 2*h_kv heads — a GQA group of 2)."""
    g = torch.Generator().manual_seed(seed)
    d = C // h_kv
    tensors = {}
    for i in range(n_layers):
        raw = torch.randn(C, 3, generator=g)
        dirs, _ = torch.linalg.qr(raw)
        z = torch.randn(S, 3, generator=g) * torch.tensor([20.0, 12.0, 8.0])
        M = (z @ dirs.mT + torch.randn(S, C, generator=g)).half()
        K = M.reshape(S, h_kv, d).permute(1, 0, 2).contiguous()
        tensors[f"layer{i}.k_pre"] = K.clone()
        tensors[f"layer{i}.k"] = K.clone()  # no-RoPE: k == k_pre
        tensors[f"layer{i}.q"] = torch.randn(h_kv * 2, T, d, generator=g).half()
        tensors[f"layer{i}.v"] = torch.randn(h_kv, S, d, generator=g).half()
    return tensors


def test_storm_policy_robustness_smoke(tmp_path):
    """Full experiment main on tiny factory caches for BOTH policies (the chat
    stand-ins differ by seed — the pipeline doesn't care how the caches were
    produced): parquets + verdict written, retention finite and positive, gate
    evaluated, shift diagnostic rows present."""
    import pandas as pd

    from experiments.storm_policy_robustness import Config, main

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    seq_len = 128
    # Policy naming mirrors collect_cache: raw = no label, chat = "_chat".
    specs = [
        ("raw", 0, ""),
        ("raw", 2048, "_off2048"),
        ("raw", 4096, "_off4096"),
        ("chat", 0, "_chat"),
        ("chat", 2048, "_chat_off2048"),
        ("chat", 4096, "_chat_off4096"),
    ]
    for policy, off, suffix in specs:
        seed = {"raw": 10, "chat": 20}[policy] + off // 1024
        save_cache(
            _tiny_cache(seed),
            str(cache_root / f"tiny_{seq_len}{suffix}.safetensors"),
        )

    cfg = Config(
        model_name="",  # no-RoPE tiny substrate
        model_label="tiny",
        seq_len=seq_len,
        fit_offsets=(2048, 4096),
        heldout_offset=0,
        budgets=(2.2, 2.5),
        group=16,  # S=128 divisible; tiny C=16 needs the smaller group
        collect_missing=False,
        cache_root=str(cache_root),
        out_root=str(tmp_path / "results"),
    )
    run_dir = main(cfg)

    df = pd.read_parquet(run_dir / "metrics.parquet")
    spec = df[df.arm == "spectral"]
    assert set(spec.pack_policy.unique()) == {"raw", "chat"}
    assert set(spec.score_policy.unique()) == {"raw", "chat"}
    assert set(spec.budget.unique()) == {2.2, 2.5}
    assert (spec.win_model > 0).all()
    tq = df[df.arm == "turboquant_mse"]
    assert set(tq.bits.unique()) == {2, 3, 4}

    df_shift = pd.read_parquet(run_dir / "shift.parquet")
    assert {"gap_sigma_cross", "gap_w_cross", "gap_sigma_within_raw"} <= set(
        df_shift.columns
    )
    assert (df_shift.gap_sigma_cross >= 0).all()

    verdict = json.loads((run_dir / "verdict.json").read_text())
    assert verdict["gate"]["threshold"] == GATE_RETENTION
    for key, r in verdict["gate"]["retention"].items():
        assert r > 0 and torch.isfinite(torch.tensor(r)), (key, r)
    assert verdict["gate"]["gate_outcome"]
    assert "shift_diagnostic" in verdict and "per_cell" in verdict
    assert len(verdict["per_cell"]) == 8  # 2 packs x 2 heldouts x 2 budgets
