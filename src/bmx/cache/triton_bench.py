"""Correctness-gated decode latency ledger.

run_decode_ledger measures per-(variant, seq_len) attention latency on CPU or CUDA,
but ONLY records latency_ms when the variant passes BOTH:
  (a) attention_diff vs naive_dense_attention (oracle) < tol
  (b) logit_parity_pass — the same numerical check, confirming the variant's
      output is within tolerance of the oracle on the same inputs.

"Logit parity" in this module means attention-output parity vs the oracle —
we compare attention outputs (not model logits, which would need a full
forward pass and model weights). This is the right gate here: the oracle IS
the ground-truth attention output, and any divergence there propagates into
final logits. Documented choice: full-model logit comparison is not available
without model weights locally; attention-output comparison is the tightest
device-independent correctness check available for the KV-attention variants
this module measures.

Tolerance: derived from the measured chunked-vs-oracle drift at startup
(compute once, log it). NOT hardcoded.

MEASURED vs ANALYTIC COLUMNS
-----------------------------
The ``timing_seq_len`` parameter controls which row gets real timing:
  - ``predicted_speedup`` is ANALYTIC (decode_speedup_curve formula) — populated
    for every (variant, seq_len) row regardless of the fixture size.
  - ``latency_ms``, ``max_abs_vs_oracle``, ``max_rel_vs_oracle``,
    ``logit_parity_pass``, and ``measured_speedup`` are MEASURED — they require
    running the variant on the actual fixture tensors. These are populated ONLY
    for the row where ``row.seq_len == timing_seq_len`` (the context length the
    supplied fixture actually represents). All other rows carry NaN / None for
    these columns.

This makes the gap between "what we timed" and "what we're labelling"
structurally visible in the DataFrame rather than silently wrong.

Schema (exact): ["variant", "seq_len", "latency_ms", "max_abs_vs_oracle",
                  "max_rel_vs_oracle", "logit_parity_pass",
                  "predicted_speedup", "measured_speedup"]
"""

from __future__ import annotations

import time
from typing import Callable

import pandas as pd
import torch

from bmx.bench.kv_memory import KVMemCase, decode_speedup_curve
from bmx.cache.chunked_attention import attention_diff, naive_dense_attention

# ── hardware constants (GH200 defaults; override when measured on a real GPU) ──
_HBM_BANDWIDTH = 4.0e12  # bytes/s  (GH200 ~4 TB/s)
_PEAK_FLOPS = 1.0e15  # FLOP/s   (GH200 fp16 ~1000 TFLOPS; conservative)

# ── reference arch (Llama-3.1-8B) — used for predicted_speedup column ─────────
_LLAMA8B_LAYERS = 32
_LLAMA8B_H_KV = 8
_LLAMA8B_D_HEAD = 128
_LLAMA8B_WEIGHTS_BYTES = int(14.9 * 1024**3)
_LLAMA8B_ACT_BYTES = int(61.3 * 1024**3)
_LLAMA8B_BLOCK = 128
_LLAMA8B_RECENT_WINDOW = 32


_NAIVE_DENSE_PARAMS = {
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


def _oracle_attn_kwargs(kwargs: dict) -> dict:
    """Strip keys that naive_dense_attention does not accept.

    naive_dense_attention's signature is a strict subset of
    chunked_dequant_attention's — it has no query_abs_start, attn_mask,
    v_group, or v_seed. Callers can pass the full chunked kwargs dict here;
    this strips the extras so the oracle call doesn't TypeError.
    """
    return {k: v for k, v in kwargs.items() if k in _NAIVE_DENSE_PARAMS}


def _predicted_speedup(seq_len: int) -> float:
    """Analytic upper-bound speedup for k2b chunked vs fp16 at seq_len."""
    common = dict(
        n_layer=_LLAMA8B_LAYERS,
        h_kv=_LLAMA8B_H_KV,
        d_head=_LLAMA8B_D_HEAD,
        block=_LLAMA8B_BLOCK,
        recent_window=_LLAMA8B_RECENT_WINDOW,
        weights_bytes=_LLAMA8B_WEIGHTS_BYTES,
        act_bytes=_LLAMA8B_ACT_BYTES,
        logits_bytes=0,
    )
    fp16_case = KVMemCase(
        seq_len=seq_len,
        bpe_k=16.0,
        bpe_v=16.0,
        path="fp16",
        **common,
    )
    packed_case = KVMemCase(
        seq_len=seq_len,
        bpe_k=3.0,  # k2b recipe: keys @3b pre-RoPE lowrank+channel
        bpe_v=2.0,  # values @2b rotate+Lloyd
        path="chunked",
        **common,
    )
    result = decode_speedup_curve(
        fp16_case,
        packed_case,
        hbm_bandwidth_bytes_per_s=_HBM_BANDWIDTH,
        peak_flops_per_s=_PEAK_FLOPS,
    )
    return float(result["speedup_upper_bound"])


def _time_variant(
    variant_fn: Callable,
    q: torch.Tensor,
    k_blocks: list,
    v_blocks: list,
    kwargs: dict,
    *,
    n_warmup: int = 3,
    n_repeat: int = 10,
    device: str = "cpu",
) -> float:
    """Return mean wall-clock ms for variant_fn(q, k_blocks, v_blocks, **kwargs).

    Uses torch.cuda.synchronize() on CUDA, plain time.perf_counter on CPU.
    """
    cuda = device != "cpu" and torch.cuda.is_available()

    def _run():
        return variant_fn(q, k_blocks, v_blocks, **kwargs)

    for _ in range(n_warmup):
        _run()
    if cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        _run()
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return sum(times) / len(times)


def run_decode_ledger(
    variants: dict[str, Callable],
    q: torch.Tensor,
    k_blocks: list,
    v_blocks: list,
    attn_kwargs: dict,
    seq_lens: list[int],
    *,
    timing_seq_len: int | None = None,
    device: str = "cpu",
    n_warmup: int = 3,
    n_repeat: int = 10,
    tol_scale: float = 10.0,
) -> pd.DataFrame:
    """Measure per-(variant, seq_len) decode latency, gated on correctness.

    MEASURED vs ANALYTIC COLUMNS
    ----------------------------
    ``timing_seq_len`` is the context length that the supplied fixture tensors
    (q, k_blocks, v_blocks) actually represent.  For the row where
    ``seq_len == timing_seq_len`` the following columns are populated from real
    measurements (subject to the correctness gate):
        latency_ms, max_abs_vs_oracle, max_rel_vs_oracle,
        logit_parity_pass, measured_speedup
    For all OTHER seq_len rows these columns are NaN / None — the fixture does
    not correspond to those lengths, so timing them would be silently wrong.
    ``predicted_speedup`` is always populated (it is purely analytic).

    If ``timing_seq_len`` is None and ``len(seq_lens) == 1``, it defaults to
    that single entry.  If ``len(seq_lens) > 1`` and ``timing_seq_len`` is None
    a ``ValueError`` is raised — callers must be explicit about which seq_len
    the fixture represents to prevent silent mis-labelling.

    On the VM, call run_decode_ledger once per seq_len with a correctly-sized
    fixture and timing_seq_len set to match:
        for sl in seq_lens:
            df_sl = run_decode_ledger(..., seq_lens=[sl], timing_seq_len=sl)

    Steps for each (variant, seq_len) where seq_len == timing_seq_len:
      1. Run the oracle (naive_dense_attention) on (q, k_blocks, v_blocks).
      2. Run the variant.
      3. attention_diff(variant_out, oracle_out) -> max_abs, max_rel.
      4. tol is derived once from the chunked reference drift measured at call
         start (see below). logit_parity_pass = (max_abs < tol).
      5. latency_ms and measured_speedup are ONLY recorded when BOTH:
           max_abs_vs_oracle < tol  AND  logit_parity_pass is True.
         Else latency_ms = NaN and measured_speedup = NaN.

    Tolerance derivation:
      We call chunked_dequant_attention on the supplied inputs as the "reference
      correct" path, measure its drift from the oracle, then set
        tol = max(chunked_drift * tol_scale, 1e-6)
      Floor is 1e-6 — one order of magnitude above the measured fp16 roundoff
      drift (~2.4e-7 chunked vs oracle). This is tight enough to catch subtle
      masking or quantisation bugs whose drift falls in the [1e-7, 1e-5] range.
      The old 1e-4 floor was ~420× the real drift and would silently pass a
      variant with up to 1e-4 error, defeating the gate's purpose.

    Args:
        variants:        {name: callable(q, k_blocks, v_blocks, **attn_kwargs)}
        q, k_blocks, v_blocks, attn_kwargs: attention inputs (sized for timing_seq_len).
        seq_lens:        context lengths to include in the ledger.
        timing_seq_len:  the seq_len the fixture actually represents; measured
                         columns are populated only for this row.  Defaults to
                         seq_lens[0] when len(seq_lens)==1; must be explicit
                         when len(seq_lens)>1.
        device:          "cpu" or "cuda".
        n_warmup / n_repeat: timing parameters.
        tol_scale:       multiplier on reference drift for the pass threshold.

    Returns:
        pd.DataFrame with EXACTLY columns:
        ["variant", "seq_len", "latency_ms", "max_abs_vs_oracle",
         "max_rel_vs_oracle", "logit_parity_pass", "predicted_speedup",
         "measured_speedup"]
        Rows where seq_len != timing_seq_len have NaN for all measured columns
        and None for logit_parity_pass; predicted_speedup is finite for all rows.
    """
    from bmx.cache.chunked_attention import chunked_dequant_attention

    # ── Resolve timing_seq_len ────────────────────────────────────────────────
    if timing_seq_len is None:
        if len(seq_lens) == 1:
            timing_seq_len = seq_lens[0]
        else:
            raise ValueError(
                "timing_seq_len must be provided when len(seq_lens) > 1. "
                "The fixture tensors correspond to one specific context length; "
                "timing the same fixture and labelling it as a different seq_len "
                "is silently wrong. Pass timing_seq_len=<the fixture's real context "
                "length>, or call run_decode_ledger once per seq_len."
            )

    # naive_dense_attention does not accept query_abs_start / attn_mask / v_group /
    # v_seed — strip them so callers can pass full chunked kwargs unchanged.
    _oracle_kwargs = _oracle_attn_kwargs(attn_kwargs)

    # ── Step 0: derive tolerance from reference chunked path drift ────────────
    with torch.no_grad():
        oracle_ref = naive_dense_attention(q, k_blocks, v_blocks, **_oracle_kwargs)
        chunked_ref = chunked_dequant_attention(q, k_blocks, v_blocks, **attn_kwargs)
    ref_diff = attention_diff(chunked_ref, oracle_ref)
    measured_drift = ref_diff["max_abs"]
    # Floor at 1e-6: one order above the measured fp16 roundoff drift (~2.4e-7).
    # Tight enough to catch subtle masking/quant bugs whose drift lands in
    # [1e-7, 1e-5]. The old 1e-4 floor was ~420x the real drift and would
    # silently pass a buggy variant.
    tol = max(measured_drift * tol_scale, 1e-6)
    print(
        f"[triton_bench] reference chunked drift: max_abs={measured_drift:.3e}  "
        f"tol={tol:.3e}  (floor=1e-6, tol_scale={tol_scale})"
    )

    # ── Step 1: measure fp16 baseline latency (for measured_speedup ratio) ────
    # Only needed for the timing_seq_len row; compute once unconditionally since
    # it is cheap and we need it as the denominator regardless.
    # oracle baseline uses filtered kwargs (no query_abs_start etc.)
    fp16_ms = _time_variant(
        naive_dense_attention,
        q,
        k_blocks,
        v_blocks,
        _oracle_kwargs,
        n_warmup=n_warmup,
        n_repeat=n_repeat,
        device=device,
    )

    # ── Step 2: build rows — measure only at timing_seq_len ──────────────────
    rows = []
    for seq_len in seq_lens:
        pred_speedup = _predicted_speedup(seq_len)
        is_timed_row = seq_len == timing_seq_len

        for name, fn in variants.items():
            if is_timed_row:
                # Real measurement: correctness check + conditional timing
                with torch.no_grad():
                    oracle_out = naive_dense_attention(
                        q, k_blocks, v_blocks, **_oracle_kwargs
                    )
                    variant_out = fn(q, k_blocks, v_blocks, **attn_kwargs)

                diff = attention_diff(variant_out, oracle_out)
                max_abs = diff["max_abs"]
                max_rel = diff["max_rel"]
                parity_pass = bool(max_abs < tol)

                if parity_pass:
                    latency_ms = _time_variant(
                        fn,
                        q,
                        k_blocks,
                        v_blocks,
                        attn_kwargs,
                        n_warmup=n_warmup,
                        n_repeat=n_repeat,
                        device=device,
                    )
                    measured_speedup = fp16_ms / latency_ms
                else:
                    latency_ms = float("nan")
                    measured_speedup = float("nan")
            else:
                # Analytic-only row: fixture does not represent this seq_len.
                # All measured columns are NaN / None to make the gap visible.
                max_abs = float("nan")
                max_rel = float("nan")
                parity_pass = None  # None = "not measured", distinct from False = "measured and failed"
                latency_ms = float("nan")
                measured_speedup = float("nan")

            rows.append(
                {
                    "variant": name,
                    "seq_len": seq_len,
                    "latency_ms": latency_ms,
                    "max_abs_vs_oracle": max_abs,
                    "max_rel_vs_oracle": max_rel,
                    "logit_parity_pass": parity_pass,
                    "predicted_speedup": pred_speedup,
                    "measured_speedup": measured_speedup,
                }
            )

    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


# canonical schema exposed for external validation
LEDGER_COLUMNS = [
    "variant",
    "seq_len",
    "latency_ms",
    "max_abs_vs_oracle",
    "max_rel_vs_oracle",
    "logit_parity_pass",
    "predicted_speedup",
    "measured_speedup",
]
