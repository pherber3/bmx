# Kernel Phases 1+2 — state handoff (2026-06-23)

Full-suite green gate passed. All four components are built, tested, and committed.
The VM census run is the next step.

---

## Components built

1. **Byte-ledger** (`src/bmx/bench/kv_memory.py`) — pure-Python analytic memory
   model: `KVMemCase` dataclass + `predict_peak()` → resident / transient /
   attention-working-set / peak per path (`fp16`, `dense_stream`, `chunked`), plus
   `compression_at_runtime` ratio. Tested in `tests/test_kv_memory.py`.

2. **Chunked dequant-attention + golden oracle** (`src/bmx/cache/chunked_attention.py`)
   — `chunked_dequant_attention()`: packed codes resident, online-softmax over
   per-block dequant (block freed after each step), RoPE applied per block at
   absolute positions, GQA-aware. `naive_dense_attention()` is the single named
   ground truth oracle (dequant everything, one full softmax, no tricks).
   `attention_diff(a, b)` quantifies drift: `{max_abs, max_rel, mean_abs}`.
   Codec split (`quantize_packed`/`dequant_packed`) lives in
   `src/bmx/cache/codecs.py`. Tested in `tests/test_chunked_attention.py` and
   `tests/test_codec_split.py`.

3. **PackedStreamingCache** (`src/bmx/cache/packed_streaming.py`) —
   `PackedStreamingCache` / `PackedStreamingLayer`: committed prefix stored as
   packed codes + frozen subspace (bpe footprint, resident), fp16 recent window
   only. Never builds `_q_prefix_k/v` or `k_hat`/`v_hat` slabs. Routes attention
   through `chunked_dequant_attention` via the transformers 5.x
   `ALL_ATTENTION_FUNCTIONS` registry (`"chunked_dequant"`).

4. **Census instrument** (`experiments/k3_kernel_census.py`) — tyro CLI; wraps
   prefill + decode with `torch.cuda.reset_peak_memory_stats()` /
   `max_memory_allocated()`. Reports per arm × context length:
   `resident_after_prefill`, `peak_decode`,
   `peak_decode_incremental = peak_decode − resident_after_prefill` — the
   incremental split lets the ledger's `resident + transient` be checked
   term-by-term rather than against a single cumulative number. Runs both
   `StreamingQuantizedCache` and `PackedStreamingCache`. Writes parquet to
   `results/k3_kernel_census/<run-id>/` via `artifacts.py`.

---

## Ledger predictions at 128k (Llama-3.1-8B, k2b: bpe_k=3.0, bpe_v=2.0)

Actual `predict_peak` output (run 2026-06-23, see Step 2 of task-8):

```
fp16          92.2 GiB  resident  16.00 GiB  compr 1.00
dense_stream  99.7 GiB  resident  16.00 GiB  compr 1.00
chunked       78.7 GiB  resident   2.50 GiB  compr 6.39
```

Interpretation:

- **fp16 → 92.2 GiB**: baseline (weights + acts + KV fp16, no compression).
- **dense_stream → 99.7 GiB**: reproduces the observed ~99–100 GiB OOM on the
  current `StreamingQuantizedCache` path, validating the ledger. The double dense
  copy (`_q_prefix_k/v` + `k_hat`/`v_hat`) dominates; bpe compression is an
  accounting fiction at runtime on this path.
- **chunked → 78.7 GiB**: well under the 94.5 GiB batched-sweep ceiling.
  Compression is real at runtime (6.39× resident KV reduction vs fp16).

**Prediction:** the chunked-PyTorch path clears the 128k batched-sweep ceiling
analytically. The VM census must confirm this measurement is authoritative
(allocator fragmentation is an unmodeled term; that fudge factor is measured, not
guessed). A Triton kernel is not required to unblock the sweep if the census
agrees.

---

## VM census command

Run SOLO at long context (not batched with other jobs — `max_memory_allocated` is
process-wide; a concurrent process inflates the reading):

```bash
uv run python experiments/k3_kernel_census.py \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --seq-lens 4096 16384 32768 65536 131072 \
  --arms fp16 k2b \
  --max-new-tokens 4
```

After the run, confirm that `torch.cuda.synchronize()` was called before each
`max_memory_allocated()` read (it is, per the implementation) — this ensures the
reported numbers are synchronize'd and authoritative, not asynchronous estimates.

---

## Phase-3 (Triton) decision rule — VERBATIM from spec

> Build Triton **only if** (a) the Phase-1 ledger predicts *or* the Phase-2
> census measures that chunked-PyTorch's `peak_decode` at 128k still exceeds
> 94.5 GB; **or** (b) a deployment claim needs the literal process-RSS / speed
> win (the "5×") that PyTorch's allocator + lack of true fusion can't deliver.

If the Triton kernel is built, it MUST be validated against **`naive_dense_attention`
(the oracle)** via `attention_diff` — NOT against the chunked path. The chunked
path is itself an optimization that shares code with the kernel; the oracle shares
the least code with the thing under test and is the trustworthy quality yardstick.
No faster path is accepted on speed alone without a quantified diff from the oracle.
The algorithm is already proven by Phase 2; Triton only changes where the
computation runs. VM-only; a separate spec when triggered.

---

## Pending: `simplify` pass

The post-leg `simplify` pass is still pending. Dead code flagged: the original
`_turboquant_mse` helper (and `_turboquant_prod`) in `src/bmx/cache/codecs.py`
are now unused — `quantize_packed` / `dequant_packed` for the `turboquant_mse`
and `turboquant_prod` arms use `_turboquant_mse_packed` / `_turboquant_mse_dequant`
directly, bypassing the original monolithic helpers.
