"""Ledger gate test: correctness-gated latency, schema enforcement.

Two variant contracts exercised:
  - correct_chunked: wraps chunked_dequant_attention — should match the oracle
    within the measured drift tol, so latency_ms is FINITE and
    logit_parity_pass is True.
  - broken_scale:  returns oracle * 1.5 — catastrophically wrong, so latency_ms
    is NaN and logit_parity_pass is False and measured_speedup is NaN.

The test would FAIL if someone removed the correctness gate (latency recorded
unconditionally), because the broken variant has no finite latency to assert on.
"""

from __future__ import annotations

import math

import pandas as pd

from tests.factories import tiny_packed_blocks


# ── required schema ────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "variant",
    "seq_len",
    "latency_ms",
    "max_abs_vs_oracle",
    "max_rel_vs_oracle",
    "logit_parity_pass",
    "predicted_speedup",
    "measured_speedup",
]


# ── tiny fixture ───────────────────────────────────────────────────────────────
def _make_attn_inputs():
    """Return (q, k_blocks, v_blocks, kwargs) for a tiny decode step."""
    return tiny_packed_blocks(
        n_q_heads=4,
        n_q_groups=2,
        n_q=1,
        d=16,
        blk=8,
        n_blocks=4,
        arm="rtn_token",
        group=8,
        seed=7,
    )


# ── variant callables ──────────────────────────────────────────────────────────
def _correct_variant(q, k_blocks, v_blocks, **kwargs):
    """Thin wrapper around chunked_dequant_attention — the locally-correct path."""
    from bmx.cache.chunked_attention import chunked_dequant_attention

    return chunked_dequant_attention(q, k_blocks, v_blocks, **kwargs)


def _broken_variant(q, k_blocks, v_blocks, **kwargs):
    """Returns oracle * 1.5 — catastrophically wrong, guaranteed to fail the gate.

    We call naive_dense_attention (the oracle) then corrupt the output.
    naive_dense_attention does not accept query_abs_start / attn_mask etc.,
    so strip extras before forwarding (same strip the bench module applies).
    """
    from bmx.cache.chunked_attention import naive_dense_attention

    _NAIVE_KEYS = {
        "k_arm",
        "v_arm",
        "group",
        "seed",
        "k_pre_rope",
        "rope_cos",
        "rope_sin",
        "k_tail",
        "v_tail",
        "n_q_groups",
        "scale",
    }
    oracle_out = naive_dense_attention(
        q, k_blocks, v_blocks, **{k: v for k, v in kwargs.items() if k in _NAIVE_KEYS}
    )
    return oracle_out * 1.5


# ── tests ──────────────────────────────────────────────────────────────────────


def test_schema_exact():
    """Column set must EXACTLY match REQUIRED_COLUMNS (no missing, no extra)."""
    from bmx.cache.triton_bench import run_decode_ledger

    q, k_blocks, v_blocks, kwargs = _make_attn_inputs()
    variants = {"correct_chunked": _correct_variant}
    df = run_decode_ledger(
        variants=variants,
        q=q,
        k_blocks=k_blocks,
        v_blocks=v_blocks,
        attn_kwargs=kwargs,
        seq_lens=[32],
        device="cpu",
    )
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == REQUIRED_COLUMNS, (
        f"Column mismatch.\n  got:      {list(df.columns)}\n  expected: {REQUIRED_COLUMNS}"
    )


def test_correct_variant_gets_finite_latency():
    """A correct variant must produce a finite latency_ms and logit_parity_pass True."""
    from bmx.cache.triton_bench import run_decode_ledger

    q, k_blocks, v_blocks, kwargs = _make_attn_inputs()
    variants = {"correct_chunked": _correct_variant}
    df = run_decode_ledger(
        variants=variants,
        q=q,
        k_blocks=k_blocks,
        v_blocks=v_blocks,
        attn_kwargs=kwargs,
        seq_lens=[32],
        device="cpu",
    )
    row = df[df["variant"] == "correct_chunked"].iloc[0]
    assert row["logit_parity_pass"] is True or row["logit_parity_pass"] == True  # noqa: E712
    assert math.isfinite(float(row["latency_ms"])), (
        f"Expected finite latency for correct variant, got {row['latency_ms']}"
    )
    assert math.isfinite(float(row["measured_speedup"])), (
        f"Expected finite measured_speedup for correct variant, got {row['measured_speedup']}"
    )


def test_broken_variant_gets_nan_latency():
    """A wrong variant must produce NaN latency_ms and logit_parity_pass False.

    This test FAILS if the gate is removed (latency recorded unconditionally):
    removing the gate would give a finite latency here, violating the assert.
    """
    from bmx.cache.triton_bench import run_decode_ledger

    q, k_blocks, v_blocks, kwargs = _make_attn_inputs()
    variants = {"broken_scale": _broken_variant}
    df = run_decode_ledger(
        variants=variants,
        q=q,
        k_blocks=k_blocks,
        v_blocks=v_blocks,
        attn_kwargs=kwargs,
        seq_lens=[32],
        device="cpu",
    )
    row = df[df["variant"] == "broken_scale"].iloc[0]
    assert row["logit_parity_pass"] is False or row["logit_parity_pass"] == False  # noqa: E712
    assert math.isnan(float(row["latency_ms"])), (
        f"Expected NaN latency for broken variant, got {row['latency_ms']} — "
        "the correctness gate may have been removed"
    )
    assert math.isnan(float(row["measured_speedup"])), (
        f"Expected NaN measured_speedup for broken variant, got {row['measured_speedup']}"
    )


def test_both_variants_same_run():
    """Both variants in one call: correct row finite, broken row NaN."""
    from bmx.cache.triton_bench import run_decode_ledger

    q, k_blocks, v_blocks, kwargs = _make_attn_inputs()
    variants = {
        "correct_chunked": _correct_variant,
        "broken_scale": _broken_variant,
    }
    df = run_decode_ledger(
        variants=variants,
        q=q,
        k_blocks=k_blocks,
        v_blocks=v_blocks,
        attn_kwargs=kwargs,
        seq_lens=[32],
        device="cpu",
    )
    assert len(df) == 2

    correct_row = df[df["variant"] == "correct_chunked"].iloc[0]
    broken_row = df[df["variant"] == "broken_scale"].iloc[0]

    # Correct: gate open
    assert correct_row["logit_parity_pass"] == True  # noqa: E712
    assert math.isfinite(float(correct_row["latency_ms"]))

    # Broken: gate closed
    assert broken_row["logit_parity_pass"] == False  # noqa: E712
    assert math.isnan(float(broken_row["latency_ms"]))
    assert math.isnan(float(broken_row["measured_speedup"]))


def test_predicted_speedup_populated_for_all_rows():
    """predicted_speedup comes from decode_speedup_curve; present even for broken."""
    from bmx.cache.triton_bench import run_decode_ledger

    q, k_blocks, v_blocks, kwargs = _make_attn_inputs()
    variants = {"broken_scale": _broken_variant}
    df = run_decode_ledger(
        variants=variants,
        q=q,
        k_blocks=k_blocks,
        v_blocks=v_blocks,
        attn_kwargs=kwargs,
        seq_lens=[32],
        device="cpu",
    )
    # predicted_speedup is an analytic prediction, not gated on correctness
    row = df.iloc[0]
    assert math.isfinite(float(row["predicted_speedup"])), (
        f"predicted_speedup should be finite even for broken variant, got {row['predicted_speedup']}"
    )
