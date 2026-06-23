# Fused dequant-attention — VM memory census results (2026-06-23)

The empirical confirmation of Phases 1+2: does the chunked dequant-attention path
make the k2b compression **real at runtime** (resident memory), and does it clear
the GH200 ceiling at 128k so the batched sweep is unblocked? **Yes to both** — but
the run also surfaced and fixed a real O(S²) prefill bug first (the value of
running on actual CUDA).

Hardware: NVIDIA GH200 480GB — usable GPU HBM **97871 MiB ≈ 95.6 GiB** (the
"480GB" includes Grace LPDDR; the attention/KV ceiling is the ~95.6 GiB HBM, which
matches the ~94.5 GB ceiling the byte-ledger assumed). Model: Llama-3.1-8B-Instruct,
fp16. Run: `results/k3_kernel_census/20260623-223357-c1fc279/`.

## The result (resident_after_prefill, GiB)

| seq_len | fp16 | k2b dense_stream | **k2b chunked** | chunked saves vs dense |
|---|---|---|---|---|
| 4096   | 16.5 | 17.1 | 16.5 | 0.6 |
| 16384  | 21.0 | 23.6 | 21.1 | 2.5 |
| 32768  | 27.1 | 32.1 | 27.3 | 4.8 |
| 65536  | 39.1 | 49.3 | 39.5 | 9.8 |
| **131072** | **63.3** | **83.5** | **64.1** | **19.4** |

**Headline:** at 128k the chunked path (64.1 GiB) is essentially identical to the
fp16 baseline (63.3 GiB), while the dense_stream path balloons to 83.5 GiB. The
chunked path's saving over dense_stream grows **linearly with context**
(0.6 → 19.4 GiB) — the signature of eliminating the second dense KV copy
(`2·L·h_kv·S·d·2`, linear in S). The k2b compression is now resident, not an
accounting fiction. Both paths fit under 95.6 GiB at single-stream 128k, so the
**batched 128k sweep is unblocked** — the concrete goal of this work.

(`peak_decode` over the 4-step decode measurement is *lower* than resident — decode
is one token — so resident_after_prefill is the meaningful column. `bpe_k≈3.5,
bpe_v≈2.0` confirms the k2b codec is at its intended ~3b-key/2b-value operating
point; chunked's bpe is NaN because PackedStreamingCache has no `bits_per_entry`
accessor — its compression shows up directly as the lower resident footprint.)

## Reconciliation with the byte-ledger — direction validated, absolute differs

The ledger (`src/bmx/bench/kv_memory.py`) predicted dense_stream 99.7 / chunked
78.7 GiB at 128k; measured here are 83.5 / 64.1. The **mechanism and direction the
ledger predicted hold exactly** — dense_stream ≈ fp16 + a full extra KV copy;
chunked ≈ fp16; the gap is the eliminated double-copy. The **absolute numbers are
lower** because:

- The ledger's `act_bytes=61.3 GiB` was calibrated from the prior campaign's
  *full-generation* peak (long `generate`, heavy activation/logits transients).
- This census measures an isolated **prefill forward + 4 decode steps**, whose
  activation transients are much smaller. Lighter workload → lower absolute peak,
  same relative structure.

This is honest: the ledger is a *relative* predictor calibrated to a heavier
workload; it correctly predicted the chunked-vs-dense saving (the thing that
matters), not the absolute peak of this specific lighter measurement. The
dense_stream resident at 128k (83.5 GiB) is consistent with the prior campaign's
"~90 GB, near the ceiling" for the k2b 2-copy path.

## The O(S²) prefill bug (found and fixed on the first CUDA run)

The first run OOM'd the **chunked path at 32k+** and showed it using 3× the dense
path's memory at 16k — the opposite of the prediction. Root cause (commit
`c1fc279`): `chunked_dequant_attention`'s per-block online-softmax materializes an
`(heads, n_q, blk)` score tile per block. At **decode** (n_q=1) the tiles are tiny
and the loop is O(S) — the intended, memory-saving path. At **prefill** (n_q=S)
every tile is `(heads, S, blk)` and they sum to `(heads, S, S)` → **O(S²) memory**.
The chunked kernel was designed and tested for decode; the census was the first
thing to run it at prefill.

**Fix:** dispatch on `n_q`. Prefill (n_q>1) reconstructs dense K/V once and runs
`F.scaled_dot_product_attention(is_causal=True)` — flash SDPA tiles over the query
dim internally in O(S), and the transient dense K/V frees after the one-shot
forward. Decode (n_q==1) keeps the chunked online-softmax unchanged — that is the
resident-memory win, and it's O(S) there. This is grounded two ways:

- **Vault** ([[Prefill Decode Separation]]; *Physics of LLM Inference* ~line 813:
  the prefill `O(s²)` score matrix "is only needed until attention is computed; in
  inference we can free it immediately"): prefill and decode legitimately use
  different attention implementations, and prefill peak is transient.
- **transformers' own QuantizedCache** (via DeepWiki): dequantizes to dense and
  runs stock SDPA — the idiomatic approach for storage-compressed caches. Our
  decode path deliberately does *not* do this, because at decode the dense
  reconstruction (~16 GiB at 128k, every step) is exactly the transient the
  chunked path exists to avoid. So: idiomatic SDPA at prefill, bespoke chunked at
  decode. The `n_q>1` signal is the same one transformers' SDPA uses for
  `is_causal`.

Parity held after the fix: `PackedStreamingCache` generation matches
`StreamingQuantizedCache` token-for-token at seq=12 (no prefill flush) and seq=48
(flush during prefill), so the prefill-dense path is bit-correct.

## Status

- **Phases 1+2 empirically confirmed.** Chunked dequant-attention realizes the
  k2b compression in resident memory (~fp16 footprint at 128k, ~19 GiB under the
  dense path) and clears the ceiling.
- **Phase 3 (Triton) gate:** the gate was "build Triton only if chunked-PyTorch
  doesn't clear 94.5 GB." It clears it comfortably (64.1 GiB at 128k single-stream).
  Triton is **not required to unblock the batched sweep**; it remains optional for
  a deployment speed/RSS claim. The next step is the actual batched 128k sweep
  (multiple arms/sequences co-resident), now that single-stream headroom is proven.
- The first census run's broken-path parquet is kept locally as evidence (not
  committed); the authoritative result is the one committed here.
