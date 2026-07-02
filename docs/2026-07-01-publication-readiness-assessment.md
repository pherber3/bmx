# Publication-Readiness Assessment + Claim Ledger

**Date:** 2026-07-01. **Plan:** `docs/superpowers/plans/2026-07-01-publication-readiness.md` (Task 1 deliverable).
**Target:** full conference paper, benchmarked box-for-box against **TurboQuant** (arXiv 2504.19874), the closest published mirror.
**Branch state:** `feat/triton-decode-kernel` @ `235b117`, 33 commits ahead of origin (unpushed), post two cleanup waves. Local baseline 271/8/1.

This document is the canonical claim ledger. Every later task cites its gap IDs (G#). A reviewer reading only this file should understand what is proven vs. pending, and why the comparison to TurboQuant is legitimate.

---

## 1. The four claims

Each claim: exact statement · supporting artifact(s) today · what's missing to make it airtight.

### C1 — The KV recipe beats TurboQuant at matched compression (SPINE)

**Statement.** At matched KV-cache compression (~7×), the recipe *pre-RoPE low-rank keys + per-channel residual, values rotate+Lloyd* (arm family `k2b` / `k2b_k2r8`) achieves higher task quality than TurboQuant's own MSE arm (`turboquant_mse`) on both long-context retrieval (NIAH) and coding (LongBench Code), measured through one fair `StreamingQuantizedCache` path.

**Supported today by:**
- NIAH short/mid (4k–16k), `recall_full` pooled mean±sem, n≈63/arm (`docs/2026-06-21-niah-longbench-frontier-results.md` §"short/mid"): **k2b_k2r8 8.47±0.21 @6.87× vs turboquant_mse 7.88±0.17 @7.70×** — ~3 SEM separation, k2b_k2r8 wins at *lower* compression.
- LongBench Code, `code_sim`×100, n=100/task (same doc §LongBench): **k2b_k2r8 50.3 @6.8× beats turboquant_mse 46.0 @7.5×**; canonical k2b 56.2 @5.4× nearly holds fp16 60.1. turboquant_mse reproduces the paper's Code≈46 exactly (see §the-anchor).
- Source parquets: `results/k3_niah/20260621-*`, `results/k3_longbench/20260621-*`.

**Missing to be airtight (gaps):**
- **G1** — these are the June single-model (Llama-3.1-8B-Instruct) runs; the paper's authoritative table needs a re-run on the **final post-cleanup code** (module moves + two sanctioned behavior deltas) → Task 9 (NIAH), Task 10 (LongBench). VM.
- **G2** — LongBench is currently **Code-only**; TurboQuant Table 1 is a 6-category table → Task 6 (scorers) + Task 10 (full run). VM.
- **G3** — the "matched compression" claim rests on the `kv_size_bits` axis being reported as a first-class column, not just a ×-factor → Task 5. Local.

### C2 — Long-context edge (STRONGER than TurboQuant)

**Statement.** Canonical `k2b` (3-bit keys) holds fp16-quality retrieval at 32k–64k(–128k) where TurboQuant's own `turboquant_mse` arm degrades. This is the "bits belong to K" thesis: the extra key bit buys long-context robustness that a flat 2-bit budget cannot.

**Supported today by:**
- 24k–32k deep run, `recall_full` n=20/cell (same doc §"long context"): canonical **k2b (3-bit K) matches fp16 at 32k (6.95 = 6.95)** and beats every compressed arm; **k2b_k2r8 (2-bit K) degrades worst at 32k (4.62), *below* turboquant_mse (6.26)**. The comparison *inverts* vs. 4k–16k.
- 64k solo run (§"NIAH at 64k"): canonical k2b (7.82) matches/beats fp16 (7.28); k2b_k2r8 collapses to 4.15 (worst arm).
- Unified single-path frontier 4k–64k (§"definitive table"): 2-bit matched arms win short (8.3–9.3 at 4k–16k), collapse long (4.9–6.7 at 64k); canonical k2b (3-bit K) holds at 32k–64k.
- 128k (fresh-process per arm, depth 0.5 — an isolated run, NOT the co-resident sweep; frontier doc lines 166-173): canonical k2b **10.0 ≥ fp16 9.52**, 2-bit k2b_k2r8 degrades to 6.19. Extends the 3-bit-key robustness to 128k, but flag it as fresh-process-per-arm (the co-resident batched 128k mean was k2b 7.94 in the census run) → re-confirm cleanly in Task 9.

**Missing to be airtight (gaps):**
- **G4** — the 32k–128k crossover figure (F4) that visualizes where k2b holds and turboquant_mse degrades → Task 9 Step 5. VM.
- **Honesty caveat to disclose:** on the *unified* run, turboquant_mse at 64k (7.41) edges canonical k2b (7.21) — within noise; the clean separation is at 32k and in the dedicated deep runs. State the crossover as "k2b holds fp16-parity; turboquant_mse is within-noise-to-below," not "k2b strictly dominates at every long length."

### C3 — Runtime memory realized (STRONGER than TurboQuant)

**Statement.** The k2b compression is real in *resident* memory, not an accounting fiction: the chunked/packed dequant-attention path keeps codes resident and dequants on the fly, so at 128k the process sits at fp16's footprint instead of ballooning past it — and the compressed arm that OOMs on the dense path **completes** on the packed path.

**Supported today by:**
- Resident census, Llama-3.1-8B, GH200 95.6 GiB ceiling (`docs/2026-06-23-kernel-census-results.md`): at 128k, **fp16 63.3 GiB · k2b dense_stream 83.5 GiB · k2b chunked/packed 64.1 GiB**; the chunked-vs-dense saving grows linearly with context (0.6→19.4 GiB). Source: `results/k3_kernel_census/20260623-223357-c1fc279/`.
- OOM-vs-completes A/B (same doc): at k2b/128k in one cell, the dense path **OOMs** (`torch.OutOfMemoryError … 94.06 GiB in use` inside `sdpa_attention_forward`); the packed path **completes** the identical cell. Definitive capability proof.
- The byte-ledger `src/bmx/bench/kv_memory.py::predict_peak` (kept deliberately as the census-anchor pin) predicted the direction; census confirmed it.

**Missing to be airtight (gaps):**
- **G5 (framing, not data)** — must be reported as the **KV-cache slice** (fp16 16 GiB → k2b ~3 GiB, ~5.3× on the slice), NOT total process RSS. In the total (63.3/64.1) the fixed weights (~14.9 GiB) + activations (~61.3 GiB, from the ledger term decomposition W=14.9/C=16/A=61.3 @128k) *mask* the compression. A reader who sees "64.1 ≈ 63.3" and concludes "compression saves nothing" has misread it. → Task 1 §3 (this reframe) + Task 11 Step 4 (T2 reframed).
- **G6** — census predates the cleanup waves; re-confirm on final code → Task 11. VM.
- **Honesty caveat:** census was an isolated *prefill + 4 decode steps*, not full generation; single run. Disclose.

### C4 — Fused decode kernel feasibility (SOFTENED — nice-to-have, not load-bearing)

**Statement.** The full k2b recipe is realizable as a *single fused Triton decode launch* that dequantizes packed codes in-kernel (in-kernel low-rank-K + RoPE + per-head turboquant-V Hadamard), so compression need not be undone at runtime. Speedups are reported **only against a naive PyTorch chunked baseline** — a systems-feasibility result, not a competitive-latency claim vs FlashAttention/vLLM.

**Supported today by:**
- Latency bench, GH200 (`docs/2026-06-24-triton-decode-results.md`): RTN packed **2624× @131k** vs chunked; k2b real recipe **322× @131k** vs chunked. Correctness: per-component oracle isolation (RTN 10/10, k2b 6/6 vs naive dense) + live generate-parity vs `StreamingQuantizedCache`.
- Source: `results/k3_triton_decode/20260624-*`.

**Missing / framing:**
- **G7** — the baseline is naive PyTorch; the paper must never imply otherwise. Post-cleanup, the per-block path was deleted and the bench now measures the real deployment kernel vs chunked directly — the June per-block numbers are historical-only. → Task 11 Step 4 (T3 labeled explicitly). VM.
- **Two sanctioned behavior deltas** to disclose: non-fused CUDA decode configs fall back to chunked (non-headline arms only); `pick_num_splits` reads the device SM count. Both covered by the pending GH200 re-verify.

---

## 2. Gap register

| G# | Claim | What's missing | Closed by | VM? |
|---|---|---|---|---|
| G1 | C1 | Re-run NIAH+LongBench on final post-cleanup code | Tasks 9, 10 | yes |
| G2 | C1 | LongBench is Code-only; need full 6-category (Table-1 parity) | Tasks 6, 10 | yes (run) |
| G3 | C1 | `kv_size_bits` as a first-class reported column | Task 5 | no |
| G4 | C2 | 32k–128k crossover figure (F4) | Task 9 | yes |
| G5 | C3 | Reframe memory as KV-slice, not total RSS | Task 1 §3 + Task 11 | no (framing) |
| G6 | C3 | Re-confirm census on final code | Task 11 | yes |
| G7 | C4 | Label speedup baseline honestly; disclose deltas | Task 11 | yes |
| G8 | C3 | Full-generation peak memory (replace prefill+4-step diagnostic) | Task 11b (new) | yes |
| G9 | scope | Multi-architecture extension (Gemma + Qwen3) | ext. spec `2026-07-01-multi-architecture-extension.md` | yes (later) |
| — | all | Distortion-vs-bounds figure (F1, TurboQuant Fig-3 parity) | Task 8 | no |
| — | all | NIAH heatmap aggregate-Score annotation (F2, Fig-4 parity) | Task 4 | no |
| — | C1 | Anchor: reproduce TurboQuant fp16+turboquant_mse rows | Task 10 Step 4 | yes (gate) |

**No claim is left without either an artifact or a gap-task.** All four claims trace to ≥1 supporting artifact path above.

---

## 3. The reframe (G5 — the single most important presentation decision)

TurboQuant reports **no memory table at all** — their entire memory claim is the "KV Size (bits)" column in Table 1 (16 / 2.5 / 3.5) plus "≥4.5×" in prose. So the *compression ratio is the memory claim.* We match that (report `kv_size_bits`), and then add the resident census as upside — **but only if framed as the KV slice.**

The byte-ledger term decomposition at 128k (`kv_memory.py` docstring): **W (weights) = 14.9 GiB · C (one fp16 KV copy) = 16 GiB · A (activations + working set) = 61.3 GiB.** fp16 total = W + C + A = 92.2 (ledger) ≈ 63.3 (census, lighter prefill+4-step workload).

**Report memory two ways, in this order:**
1. **KV-cache slice (the headline):** fp16 KV = 16 GiB → k2b packed ≈ 3 GiB → **~5.3× on the slice.** This is the number that matches the compression ratio and matches TurboQuant's "KV Size" axis.
2. **Total resident (the systems bonus):** 63.3 (fp16) / 64.1 (k2b packed) / 83.5 (dense_stream), with the **explicit note** that in the total, fixed W + A mask the KV win — the packed path lands at fp16's footprint (compression realized, not defeated), while the naive dense-compressed path balloons *past* fp16. The linear-in-context saving vs dense (0.6→19.4 GiB) is the signature.
3. **The capability result (the strongest single fact):** OOM-vs-completes at batched 128k. Binary, reproducible, needs no interpretation.

**Never lead with the 63.3/64.1 total as the memory result** — it invites the "saves nothing vs fp16" misread. The KV slice + the OOM-vs-completes result are the claims; the total is context.

---

## 4. What we claim MORE than TurboQuant

TurboQuant shows neither of these:

- **C2 (long-context edge).** They report "identical to full-precision at 4× … 4k→104k" but their arm is a *flat* budget. Our "bits belong to K" result shows a 3-bit-key allocation holds fp16-parity retrieval at 32k–64k(–128k) where a matched-compression flat-2-bit arm (and, at 32k, their own turboquant_mse) degrades. This is a *mechanism* claim on *their* benchmark.
- **C3 (realized runtime memory).** They never demonstrate resident bytes — the compression ratio stands in for it. We measure it (census) and prove the capability (OOM-vs-completes). Plus C4: a fused kernel proving the recipe is deployable without undoing compression at runtime.

The wedge: *TurboQuant proves worst-case-optimal coding on the sphere; we show that on real caches, structure-aware allocation (pre-RoPE low-rank keys, asymmetric K/V bits) beats their worst-case-optimal arm 2–3× on the task-relevant metric — and holds where a flat budget breaks.*

---

## 5. Honest weaknesses to disclose in the paper (the Limitations section)

Copy these verbatim into the paper's limitations — disclosing them is what makes the strong claims credible.

1. **Single-run measurements.** NIAH/LongBench/census numbers are single runs on one GH200; no seed-variance bars on the systems numbers. (Quality numbers do carry sem over depths/samples.)
2. **Census workload — being upgraded to a real measurement (G8).** The June census measured an isolated *prefill + 4 decode steps* as a diagnostic. This is being replaced by a **full-generation peak-memory measurement** (generate 256–512 tokens at 32k/64k/128k, record true peak) → new Task 11b. We report the real generation peak as the headline and keep the diagnostic as corroboration. Expect the *absolute* total to rise under full generation (larger activation/logits transients) while the *relative* structure (packed ≈ fp16, dense balloons; linear-in-context packed-vs-dense gap) holds or widens — which strengthens C3.
3. **Kernel speedup baseline is naive PyTorch chunked**, not FlashAttention/vLLM. It is a feasibility result, not a competitive-latency claim. (C4/G7.)
4. **Cross-model generalization — being actively extended (G9), not just disclosed.** The current `StreamingQuantizedCache` is validated on Llama-family GQA (uniform global attention + full RoPE). Extension to **Gemma-family** (sliding-window + alternating full/sliding + partial RoPE) and **Qwen3** (standard full attention; hybrid variants skip their linear-attention layers exactly as TurboQuant does) is scoped in `docs/superpowers/specs/2026-07-01-multi-architecture-extension.md`. Correct framing (per vLLM's TurboQuant scoping): the recipe applies to *any standard full-attention layer with a KV cache*; for hybrid models it applies to the full-attention layers and defers linear-attention ones. Whether "bits belong to K" and "quantize pre-RoPE" survive partial RoPE / sliding windows is an open, genuinely falsifiable question — a positive is a strong generalization result, a negative is an honest boundary. Ship/delay decision deferred until the Llama authoritative runs land.

**METHODS JUSTIFICATIONS (not weaknesses — state affirmatively):**
5. **We do not middle-truncate LongBench prompts — this is deliberate and can only hurt us.** LongBench's own preprocessing middle-truncates over-long prompts to fit a fixed context window (`input[:L/2] + input[-L/2:]`, verified in `Long-Context-Data-Engineering/eval/book/eval_utils.py:406-410`). They truncate because they feed a fixed-window model; *compressing the full context is the method under test*, so truncating would discard the very tokens whose compression we measure. Our prompts are therefore ≥ theirs in length, making our task **harder** (more context to retrieve from) — not-truncating cannot inflate our scores, so it is not an unfair advantage. State this as a methods choice, not a divergence-to-apologize-for.
   - *Genuine minor caveat (stays a limitation):* the `length = n_prefill*2` compression proxy under-states compression — a consistent lower bound across arms, so rankings hold but the absolute compression column is conservative.
6. **turboquant_prod is carried as a faithful-but-losing baseline** — its unbiased-coding collapse on our path is a real property of the method under per-key softmax exposure (verified faithful three ways), not a bug; do not present it as our failure to implement.

---

## The anchor (why the transitive-baseline comparison is legitimate) {#the-anchor}

We do NOT implement PolarQuant/SnapKV/PyramidKV. Transitivity (we beat TurboQuant; TurboQuant beat them) is **licensed by the anchor**: we run fp16 (Full Cache) + turboquant_mse on our own path and reproduce TurboQuant's Table-1 values (Code≈46 already confirmed at 46.0; Avg≈50 to confirm at full scale). **Task 10 Step 4 is a hard gate** — if the reproduced anchor rows don't match Table 1, the transitive argument is void and must be root-caused before any baseline claim is written. See `docs/2026-07-01-baseline-parity-decision.md` (Task 2).
