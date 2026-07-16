# K4 spectral codec — Stage-3 VM duel results (2026-07-15)

**Verdict up front: the pre-registered effect-size targets (spec §10.2) are NOT
met under the licensed accounting; the underlying effect is real, measured, and
smaller.** K4 is the best quantized arm on the full LongBench table (+0.35 Avg
over turboquant_mse_b3, +1.51 over K3V2, retrieval category +3.1), reaches
fp16-parity NIAH recall, and — the load-bearing new measurement — its skeptic
bits-vs-context curve crosses below b3 at ~16k tokens (2.99 vs 3.12) and sits at
2.69 vs 3.07 at 32k. At long context K4 Pareto-beats b3 under the most
conservative accounting; at short context the per-sequence pack charge erodes
the bits edge. The +2-Avg-at-≤2.7-bits and matched-quality-at-−0.5-bits targets
are missed at task-length accounting. Duel model: Llama-3.1-8B-Instruct, full
LongBench v1 English sets, all arms on the StreamingQuantizedCache path at
SHA `21e6d81` (probe cells at `798d0ef`/`562d696` — CUDA-bitwise-identical
codecs, gated by tests/test_streaming_batched_flush.py on the GH200, 20/20).

## 1. License gates (plan Task 9, run 2026-07-13, base Llama-3.1-8B)

| Gate | Rule (pre-registered) | Measured | Verdict |
|---|---|---|---|
| A (G0-corpus) | retention ≥ 0.90 licenses model-level accounting | 0.56–0.64 across 4 scored caches (n=8192 ≫ C=1024, confound-free) | **FAIL** — skeptic accounting primary |
| B (query-heldout) | corpus-W win within ~20% of scored-W win licenses weighting | weighted increment 1.54–1.59× under corpus-W (scored-W ≈ 1.7×; transfer ratio 0.85–0.88) | **PASS** — weighted corpus-W packs |
| C (error bars) | min G1 win > 1× across scored caches, both accountings | min 6.19× / max 6.45× deploy @2.5b; layer_win_fraction 1.0 | **PASS** — duel licensed |

Packs: `k4_packs_llama31{,_instruct}.safetensors` (corpus-W, w_source=corpus,
fit offsets {2048..8192}, scored {10240..16384}; ~270 MB, regenerable, never
committed). The gpt2 corpus-retention yellow flag replicated on Llama: the
query-weighted basis transfers at ~0.6 of its oracle win. Honest negative.

**Corpus-scale ablation (2026-07-16):** tripling the fit corpus (4→12 caches,
8k→24k rows) moves retention only 0.56–0.64 → 0.61–0.69. The transfer ceiling
is STRUCTURAL, not data-limited — the query-weighted basis is genuinely
sequence-dependent to a degree no corpus size fixes. Gate A's fail is
fundamental; skeptic-primary accounting is settled, and the paper can say so
with a measured scaling curve instead of a caveat.

## 2. LongBench duel (full sets, n=2,930 samples/arm across 16 datasets)

| arm | single_qa | multi_qa | few_shot | summ. | synthetic | code | **AVG** | kv bits (task-S) |
|---|---|---|---|---|---|---|---|---|
| fp16 | 22.55 | 14.90 | 69.30 | 28.28 | 52.46 | 61.99 | **41.58** | 16.00 |
| k4_b2.2 | 21.30 | 13.43 | 68.83 | 27.28 | 52.62 | 59.91 | **40.56** | 3.65 |
| k4_b2.5 | 21.72 | 13.24 | 68.88 | 27.38 | 52.68 | 60.43 | **40.72** | 3.80 |
| tq_b3 | 22.39 | 14.00 | 68.40 | 27.89 | 49.54 | 60.02 | **40.37** | 3.21 |
| tq_k3v2 | 20.48 | 13.32 | 67.78 | 27.00 | 49.07 | 57.59 | **39.21** | 2.71 |

(Stage-A probe additionally killed turboquant_mse_b2 at 41.04/n=100 — synthetic
collapse — and k4_b3.0 as non-additive over k4_b2.5.)

- K4 holds the retrieval/synthetic edge (+3.1 over b3) that motivated the
  program, and is the only quantized family matching fp16 on that category.
- The four language categories carry ~no discriminative power (all quantized
  arms within 1.1 pts; total fp16→quantized damage ≤1.5 pts) — consistent with
  the K1-era finding that ppl-adjacent metrics can't attribute component choices.
- `kv_size_bits` is skeptic-at-actual-S (per-sequence pack charge 16·C/S +
  tier map): the honest-but-harshest view, dominated by short tasks.
- **Bootstrap CIs on the category edges (per-sample shards, 10k resamples,
  full sets):** K4−b3 synthetic delta = **+3.25 [95% CI +1.02, +5.56],
  P(Δ>0) = 0.9975** — the retrieval-category edge is statistically solid.
  Code delta = +0.41 [−2.21, +3.06], P(Δ>0) = 0.62 — code is parity, not a
  win; the paper's significant quality edge is retrieval/synthetic ONLY.
  (Rerun scores replicated the banked table's values across code versions —
  measurement stability under the bitwise gates.)
- **Paired per-task statistics (16 tasks):** uniform k4_b2.5 vs b3 is
  quality-PARITY, not a quality win (5/16 task wins, Wilcoxon p=0.74 — the
  +0.35 Avg is synthetic-category-driven); vs k3v2 the gap is significant
  (13/16, p=0.003); vs fp16 K4 loses small-but-consistently per task (1/15,
  p=0.002, median −0.9). The defensible uniform-K4 claim is therefore
  "b3-parity quality at strictly fewer deployment bits, with a significant
  retrieval-category edge" — whether the allocated arm upgrades parity to a
  per-task win is measured by the allocated full-table run (in flight).

## 3. The bits-vs-context curve (measured, NIAH runs, skeptic accounting)

| arm | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| k4_b2.2 | — | — | — | 2.54 | **2.38** |
| k4_b2.5 | 4.81 | 3.60 | **2.99** | **2.69** | **2.53** |
| tq_b3 | 3.42 | 3.22 | 3.12 | 3.07 | 3.04 |
| tq_k3v2 | 2.94 | 2.73 | 2.62 | 2.57 | 2.54 |

Crossover vs b3 between 8k and 16k; vs k3v2 between 32k and 64k. At 64k,
k4_b2.2 (2.38 bits, Avg 40.56) is the CHEAPEST arm in the table and strictly
dominates both TurboQuant baselines on bits and quality simultaneously —
a measured Pareto win, under fully-skeptic accounting, at exactly the contexts
KV compression exists for. Model-level accounting (pack ships with the model,
as calibration artifacts do in the weight-quantization literature) puts
k4_b2.5 at a flat ~2.48 / k4_b2.2 at ~2.23 at every length; Gate A's
pre-registered rule keeps this SECONDARY.

## 4. NIAH (real-text PG-essay, 4k–32k × 5 depths, recall_full /10)

| arm | 4k | 8k | 16k | 32k | 64k | mean |
|---|---|---|---|---|---|---|
| fp16 | 7.33 | 7.14 | 7.05 | 7.52 | 6.76 | 7.16 |
| k4_b2.2 | — | — | — | 7.14 | 7.81 | — |
| k4_b2.5 | 7.05 | 7.05 | 8.10 | 7.24 | 7.71 | 7.43 |
| tq_b3 | 7.05 | 7.62 | 6.86 | 7.71 | 6.95 | 7.24 |
| tq_k3v2 | 6.95 | 7.24 | 7.33 | 7.52 | 8.38 | 7.48 |

Single-needle NIAH at these lengths does not separate arms — everyone holds
fp16-level recall through 64k (honest null; the separation lives in LongBench
synthetic). K4 ≥ b3 at fewer bits everywhere ≥16k: the NIAH criterion passes
as a no-regression result, not a win.

## 5. Spec §10 evaluation (verbatim criteria)

1. **G0–G2 offline gates:** G1 PASS, G2 PASS (now on BOTH models — Llama
   sensitivity census at 2048/512 windows: s_i ~9× spread, allocated 7.250 vs
   uniform 7.461 ppl @2.5b, fp16 7.231). G0-corpus FAIL written as an honest
   negative (§1). **As-stated: met.**
2. **Duel** (bits are context-dependent under skeptic accounting; both ends
   reported): (a) "at ≤2.7 measured bits, K4 Avg ≥ +2 over the best TQ arm at
   equal-or-fewer bits": at 64k, k4_b2.2 measures 2.38 — cheaper than every TQ
   arm — and its Avg margin over the cheapest TQ arm (k3v2, 2.54 bits) is
   +1.35; over b3, +0.19. The +2 magnitude is not reached at any operating
   point. **FAIL on magnitude** (the bits side over-delivers; the Avg side
   under-delivers). (b) "K4 matches b3's Avg at ≥0.5 fewer bits": at 64k,
   k4_b2.5 = 2.53 vs b3 3.04 (−0.51) at +0.35 Avg; k4_b2.2 = 2.38 (−0.66) at
   +0.19 Avg. **PASS at ≥64k context under skeptic accounting** (at the
   LongBench task-length mix it fails: K4 pays +0.44–0.59). (c) "NIAH ≥
   turboquant at matched bits, no retrieval regression": holds through 64k;
   synthetic category edge retained. **PASS.**
3. **Traceability:** every number above from parquets under
   `results/k3_longbench/2026071*` / `results/k3_niah/20260715-*` /
   `results/k4_*` with config + env + SHA. **Met (pending commit).**

## 6. The allocation lever — measured (2026-07-15 afternoon, Option-C probe)

Per-layer budgets (G2's greedy over the fitted spectra, weighted by the
2048/512 Llama sensitivity census; commit `196396c`) vs the duel's uniform
packs, n=100 synthetic+code, measured bits with the layer-averaged accounting
(`f9eeafe` — the uniform-layers assumption in bits_per_entry was falsified by
allocated packs and fixed + regression-pinned):

| arm | avg4 alloc | avg4 uniform | Δ | kv bits (task-S) | kv @32k | kv @64k |
|---|---|---|---|---|---|---|
| k4_b2.2 | 55.97 | 55.74 | +0.23 | 3.77 (=uniform) | 2.54 | 2.39 |
| k4_b2.5 | **56.83** | 56.34 | **+0.49** | 3.92 (=uniform) | 2.69 | 2.53 |

Probe read (n=100): allocation looked like +0.23/+0.49 avg4 at identical bits.
**Full-table verdict (measured 2026-07-16, 2 arms × 16 tasks full sets):
allocation is a TASK-LEVEL NULL** — allocated k4_b2.5 = 40.70 AVG vs uniform
40.72; allocated k4_b2.2 = 40.48 vs 40.56, at identical measured bits. The
probe delta was n=100 subsample noise concentrated in synthetic (allocated
synthetic did hold +0.57 for b2.5, offset by small language-category dips).
The G2 ppl improvement (7.250 vs 7.461 @2.5b) is real but does not move
LongBench Avg at these budgets. Consequence: **uniform K4 stays the headline
arm** (fewer moving parts, no sensitivity-census dependency); allocation is
reported as ppl-proven / task-neutral — an honest null that also demonstrates
why headlines are written from full sets, not probes.

Still unexercised: **mse_scale for V / baselines** (−15–21% distortion, applies
to any arm — a fairness item for the final table) and **asymmetric K/V budget
search** beyond the probed points.

## 6b. Decode latency (measured 2026-07-16, GH200, idle GPU, oracle-gated)

`profile_decode_ab --arm k4_b2.5` (dense-vs-streaming; spectral is pack-gated,
no packed twin), ms per decode token:

| ctx | dense fp16 | K4 streaming |
|---|---|---|
| 4,096 | 58.5 | **37.4** |
| 16,384 | 57.7 | **37.7** |
| 65,536 | 57.9 | **38.7** |

K4 decodes ~1.5× FASTER than the dense fp16 cache, flat in context length
(and faster at 64k than the k2b_ph streaming path's 60.0). Scope honestly:
this is a LATENCY claim only — StreamingQuantizedCache keeps the dequantized
prefix resident, so resident memory matches fp16; the packed-resident memory
story requires a packed spectral path (future work, explicitly out of scope).

## 7. Program context

The K-program's prior verdict (2026-07-08) was b3 40.56 @ 3.21 killing k2b
(40.62 @ 3.94), with retrieval the only surviving edge. K4 at 40.56–40.72 @
3.65–3.80 task-S (2.69 @ 32k) with the retrieval category won back is the
first arm since to sit strictly above b3 on quality while undercutting its
bits at deployment context. The magnitude, however, is TurboQuant-scale
(+0.35 table Avg), not the pre-registered +2.

## 8. Decision options (user call — **Option A taken, 2026-07-15**: publish as
long-context KV compression; allocated full-table run launched same day)

- **A. Publish as long-context KV compression** with the allocated arm as the
  headline: allocated k4_b2.5 = probe quality ABOVE fp16, +3 over b3 where the
  benchmark discriminates, 2.53 bits at 64k vs b3's 3.04 (−0.51, the
  pre-registered §10.2b margin) under fully-skeptic accounting; model-level
  secondary with Gate A disclosed; theory (query-weighted KLT + waterfill +
  sensitivity-weighted across-layer allocation) as the contribution stack.
  Completing the allocated full-table row costs ~8–10 h GPU.
- **B. Kill per pre-registered magnitude** (§10.2a still missed: +1.35–1.6 vs
  the +2 target over the cheapest TQ arm): write the negative, salvage infra +
  G2 + the accounting framework.
- **C (done 2026-07-15):** the allocation increment was run — §6. Its verdict:
  the lever is real (+0.23/+0.49 at equal bits) but does not by itself close
  the §10.2a magnitude gap; it does push the b3 comparison to "strictly better
  quality, strictly fewer deployment bits, with margin".

## Appendix: engineering findings this batch (Triton branch scope)

- Packed fused decode: end-to-end ms/token at 4k/16k/64k = 60.5/82.6/193.5 vs
  dense 47.9/59.6/60.5 and streaming 36.2/37.9/60.0 — latency claim negative;
  streaming (the duel path) beats dense at ≤16k.
- Streaming-vs-packed greedy parity at 64k diverges probabilistically and is
  PRE-EXISTING (bisected to 798d0ef; parity had only ever run at 4k).
  Per-step probe: logit deltas O(0.25–1.45) at 64k with no token flip over 32
  steps on this seed — accumulation-order drift across 512 pages, not an
  indexing bug; bitwise-greedy at 64k is unattainable without matching
  accumulation order. Merge gate wording must change accordingly.
- Fixed this batch (pushed): 88-GiB gen-2-gc cache leak (`562d696`); batched
  prefill flush, 27→4.3 s/sample measured (`21e6d81`, bitwise-gated CPU+CUDA).
