"""Ledger gate test: correctness-gated latency, schema enforcement.

Two variant contracts exercised:
  - correct_chunked: wraps chunked_dequant_attention — should match the oracle
    within the measured drift tol, so latency_ms is FINITE and
    logit_parity_pass is True.
  - broken_scale:  returns oracle * 1.5 — catastrophically wrong, so latency_ms
    is NaN and logit_parity_pass is False and measured_speedup is NaN.

The test would FAIL if someone removed the correctness gate (latency recorded
unconditionally), because the broken variant has no finite latency to assert on.

MEASURED vs ANALYTIC ROWS
--------------------------
When multiple seq_lens are passed to run_decode_ledger, only the row matching
``timing_seq_len`` gets real measured columns. All other rows have NaN / None
for the measured columns but a finite predicted_speedup. This is tested
explicitly in test_unmeasured_seqlen_rows_have_nan_measured_columns.

DISTINGUISHING "NOT MEASURED" from "MEASURED AND FAILED"
---------------------------------------------------------
- logit_parity_pass is None  -> row was NOT measured (wrong seq_len for fixture)
- logit_parity_pass is False -> row WAS measured but variant failed the gate
Both states have NaN latency_ms; the parity field distinguishes them.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

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

    DISTINCT from "not measured" (other seq_len): that state has logit_parity_pass=None.
    Here it is False — the variant WAS measured at timing_seq_len, and it failed.
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
    # Measured at timing_seq_len and failed -> False (not None)
    assert row["logit_parity_pass"] is False or row["logit_parity_pass"] == False  # noqa: E712
    assert row["logit_parity_pass"] is not None, (
        "logit_parity_pass must be False (measured-and-failed), not None (not-measured)"
    )
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

    # Broken: gate closed — False (measured-and-failed), not None (not-measured)
    assert broken_row["logit_parity_pass"] == False  # noqa: E712
    assert broken_row["logit_parity_pass"] is not None
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


def test_unmeasured_seqlen_rows_have_nan_measured_columns():
    """Rows for seq_lens != timing_seq_len have NaN measured columns but finite predicted_speedup.

    The fixture is sized for seq_len=32 (timing_seq_len=32). The row for
    seq_len=1024 is analytic-only: measured columns are NaN/None, predicted_speedup
    is finite. This makes the "fixture doesn't represent this length" gap
    structurally visible instead of silently wrong.
    """
    from bmx.cache.triton_bench import run_decode_ledger

    q, k_blocks, v_blocks, kwargs = _make_attn_inputs()
    variants = {"correct_chunked": _correct_variant}
    df = run_decode_ledger(
        variants=variants,
        q=q,
        k_blocks=k_blocks,
        v_blocks=v_blocks,
        attn_kwargs=kwargs,
        seq_lens=[32, 1024],
        timing_seq_len=32,
        device="cpu",
    )
    assert len(df) == 2

    # Timed row (seq_len=32): measured columns populated
    timed_row = df[df["seq_len"] == 32].iloc[0]
    assert timed_row["logit_parity_pass"] is True
    assert math.isfinite(float(timed_row["latency_ms"])), (
        f"timing_seq_len row should have finite latency, got {timed_row['latency_ms']}"
    )
    assert math.isfinite(float(timed_row["predicted_speedup"]))

    # Analytic-only row (seq_len=1024): measured columns are NaN/None
    analytic_row = df[df["seq_len"] == 1024].iloc[0]
    assert analytic_row["logit_parity_pass"] is None, (
        f"Unmeasured row should have logit_parity_pass=None, got {analytic_row['logit_parity_pass']!r}"
    )
    assert math.isnan(float(analytic_row["latency_ms"])), (
        f"Unmeasured row should have NaN latency_ms, got {analytic_row['latency_ms']}"
    )
    assert math.isnan(float(analytic_row["max_abs_vs_oracle"])), (
        f"Unmeasured row should have NaN max_abs_vs_oracle, got {analytic_row['max_abs_vs_oracle']}"
    )
    assert math.isnan(float(analytic_row["measured_speedup"])), (
        f"Unmeasured row should have NaN measured_speedup, got {analytic_row['measured_speedup']}"
    )
    # predicted_speedup is analytic — always finite
    assert math.isfinite(float(analytic_row["predicted_speedup"])), (
        f"Unmeasured row should still have finite predicted_speedup, got {analytic_row['predicted_speedup']}"
    )

    # Confirm the two states are DISTINGUISHABLE:
    # None = not measured; False = measured and failed
    assert analytic_row["logit_parity_pass"] is None  # not-measured
    # (broken variant at timing_seq_len would give False — tested separately)


def test_timing_seq_len_required_when_multiple_seqlens():
    """ValueError must fire when seq_lens has >1 entry and timing_seq_len is None.

    Silently timing the same fixture and labelling rows with different seq_lens
    would produce misleading numbers — fail fast instead.
    """
    from bmx.cache.triton_bench import run_decode_ledger

    q, k_blocks, v_blocks, kwargs = _make_attn_inputs()
    variants = {"correct_chunked": _correct_variant}

    with pytest.raises(ValueError, match="timing_seq_len"):
        run_decode_ledger(
            variants=variants,
            q=q,
            k_blocks=k_blocks,
            v_blocks=v_blocks,
            attn_kwargs=kwargs,
            seq_lens=[32, 1024],
            timing_seq_len=None,  # explicit None with multiple seq_lens -> ValueError
            device="cpu",
        )


def test_broken_variant_vs_unmeasured_row_are_distinct():
    """Confirm "measured-and-failed" (False) is distinguishable from "not-measured" (None).

    Same ledger run: broken_scale at timing_seq_len=32 gets logit_parity_pass=False;
    the same broken_scale at seq_len=1024 (not timed) gets logit_parity_pass=None.
    Both have NaN latency_ms but the parity field distinguishes them.
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
        seq_lens=[32, 1024],
        timing_seq_len=32,
        device="cpu",
    )
    assert len(df) == 2

    # measured-and-failed row
    failed_row = df[df["seq_len"] == 32].iloc[0]
    assert failed_row["logit_parity_pass"] is False, (
        f"Broken variant at timing_seq_len should be False (measured-and-failed), "
        f"got {failed_row['logit_parity_pass']!r}"
    )
    assert math.isnan(float(failed_row["latency_ms"]))

    # not-measured row
    unmeasured_row = df[df["seq_len"] == 1024].iloc[0]
    assert unmeasured_row["logit_parity_pass"] is None, (
        f"Non-timing-seq_len row should be None (not-measured), "
        f"got {unmeasured_row['logit_parity_pass']!r}"
    )
    assert math.isnan(float(unmeasured_row["latency_ms"]))

    # The two states are distinct
    assert failed_row["logit_parity_pass"] is not unmeasured_row["logit_parity_pass"]
