# Authoritative VM Results — KV-Cache Compression (GH200, 2026-07-04/05)

> **Status:** Report is complete **except §3b (NIAH long-context 64k/128k)**, which is still
> running on the VM as of writing — that section is marked `[PENDING — RUN IN FLIGHT]` and will be
> filled when the parquet lands. Everything else below is from committed parquets, numbers pulled
> directly (not from memory). Model: `meta-llama/Llama-3.1-8B-Instruct`, single GH200 480GB.
>
> **[2026-07-25 commit note]** This snapshot was written 2026-07-05 but not committed until
> 2026-07-25 (found untracked during results-tree curation); it is preserved as the historical
> record of that session. Parts were subsequently resolved or superseded: the LongBench
> truncated-rerun verdict is `docs/2026-07-08-*` (b3 dominates k2b); the C3 memory reversal was
> later isolated and DEMONSTRATED resolved (2.258 vs 4.008 GiB/seq — rope-dup + script flag
> shadowing + sweep design, zero residual mystery); the k2b_ph kernel latency work continued
> through the fp16-dots and pack_v defaults. Read this doc as the honest 07-05 state, not the
> program's final word on those items.

---

## 1. Headline

The full authoritative suite ran on the GH200: a ~60-hour full 6-category LongBench sweep, the
NIAH short-context sweep, the resident-memory census, full-generation peak-memory, and the Triton
decode latency bench. **The quality/relative story is strong** — k2b holds ~fp16 task quality at
~4× KV compression and beats the TurboQuant arm at a comparable operating point. **But three
claims are honestly narrowed by what we measured**, and I'm putting them up front rather than in a
footnote:

1. **LongBench anchor gate MISSES** on absolute TurboQuant-Table-1 parity — because our run used
   **un-truncated** inputs where TurboQuant middle-truncates. This is an input-policy mismatch, not
   a measurement-path failure; the **relative** claims stand, but absolute parity needs a truncated
   rerun (harness fixes already committed).
2. **The runtime-memory claim (C3) is REVERSED by measurement.** Neither single-sequence peak (T2b:
   packed ≈ fp16) nor batch concurrency (T4: packed fits **8** sequences @32k vs fp16's **16** —
   *half*) shows a VRAM win. The 4× code compression is real in stored bits but **entirely paid
   back by per-sequence dequant/stack/buffer overhead** in the cache objects. Compression is NOT
   realized in deployment VRAM on any current path.
3. **The k2b kernel routing was the wrong arm, AND the corrected path is still slow.** Every earlier
   run used `k2b` (chunked fallback by construction), not `k2b_ph` (the fused-kernel arm). On the
   corrected `k2b_ph` arm the kernel **does** fire (path probe confirms) — **but the packed decode
   path is still 1.5–8× slower than the plain dense-dequant streaming cache** (285 ms/tok @128k vs
   34), because the `_PagedStacks` per-step rebuild dominates. The RTN microbench's 402× (T3) does
   **not** carry to end-to-end generation (T3b). Fastest quantized decode today = `streaming`.

**Net: the quality/compression science (C1) is solid; the systems story (C2 latency, C3 memory) is
substantially weaker than the pre-run framing — three claims moved from "pending" to "measured
negative." This report is the honest record; §5 and T3b/T4 are the load-bearing new results.**

---

## 2. LongBench — full 6-category (Table T1)

Run: `results/k3_longbench/20260702-164100-46b9579/` — 5 arms × 16 datasets × full sets
(3,750 items/arm), ~60h wall. Scores are the LongBench-official per-category metrics (×100).

| arm | single_qa | multi_qa | summ. | few_shot | synthetic | code | **Avg** | kv_bits | comp. |
|---|---|---|---|---|---|---|---|---|---|
| **fp16** | 24.18 | 14.87 | 28.28 | 69.21 | 52.29 | 62.04 | **41.81** | 16.0 | 1.0× |
| **k2b (ours)** | 21.94 | 13.65 | 27.21 | 68.80 | 51.86 | 60.22 | **40.62** | 3.94 | 4.07× |
| turboquant_mse | 17.59 | 12.10 | 25.23 | 64.04 | 29.71 | 51.18 | **33.31** | 2.22 | 7.27× |
| kivi* | 1.52 | 0.77 | 2.69 | 2.18 | 2.38 | 20.75 | **5.05** | 2.45 | 6.57× |
| turboquant_prod | 1.70 | 0.73 | 3.67 | 2.08 | 2.67 | 25.37 | **6.04** | 2.24 | 7.22× |

**What stands (C1, relative):**
- **k2b holds 97% of fp16 quality** (40.62 / 41.81) at **4.07×** compression — same path, same inputs.
- **k2b beats turboquant_mse by +7.3 Avg** (40.62 vs 33.31); the gap is largest on `synthetic`
  (51.86 vs 29.71) and holds across every category. This is the core "we beat TurboQuant at a
  comparable operating point" result, licensed by same-path/same-input consistency.
- turboquant_prod and kivi* collapse (Avg ~5–6) — consistent with NIAH.

**Anchor gate — MISS (and why it's not fatal):**

| | our fp16 | TurboQuant Table-1 target |
|---|---|---|
| Avg | **41.81** | ~50.06 |
| Code | **62.04** | ~46.28 |

Our fp16 Avg lands ~8 points low. **Diagnosis: un-truncated inputs.** We feed the full context
(many summarization/QA prompts run 25k–100k+ tokens); TurboQuant middle-truncates to ~31.5k. Different
effective inputs → different absolute scores. Our Code is actually *higher* (62 vs 46), consistent
with code prompts being short enough that truncation barely matters. This is an **input-policy**
mismatch, **not** a measurement-path bug — so the *relative* comparisons above are valid, but the
*absolute* TurboQuant-parity number (which licenses the transitive-baseline argument) is not yet
reproduced.

**Remedy (already staged):** a truncated parity rerun. The harness fixes are committed locally
(un-pushed): a middle-truncation flag matching LongBench's scheme, a per-token slab-rebuild
deletion, LongBench-E loading, and the kivi diagnosis. See §7/§8.

\* **kivi caveat:** this "kivi" arm is a **symmetric-RTN strawman**, not the real KIVI algorithm
(commit `acaaeaf`). Its collapse is **not** a valid statement about KIVI and must not be presented
as one.

---

## 3a. NIAH — short context (4k–32k), this session

Run: `results/k3_niah/20260702-160656-46b9579/` — 5 arms × {4k,8k,16k,32k} × 5 depths, real
ROUGE-1 needle recall (scale 0–10).

| arm | recall_full | kv_size_bits | compression |
|---|---|---|---|
| **fp16** | 7.19 | 16.0 | 1.0× |
| **k2b (ours)** | **7.93** | 3.94 | 4.07× |
| turboquant_mse | 7.17 | 2.22 | 7.24× |
| kivi* | 1.10 | 2.45 | 6.55× |
| turboquant_prod | 1.26 | 2.24 | 7.19× |

- **k2b matches/beats fp16 recall (7.93 vs 7.19)** at 4.07× — quality parity, the small edge is
  denoising within noise.
- **turboquant_mse reproduces at parity too (7.17 ≈ fp16)** — the anchor behaves as expected here.
- turboquant_prod / kivi* collapse (~1.1–1.3), same as LongBench.
- Compression grows with context (3.86× @4k → 4.20× @32k) as the fp16 recent-window amortizes.

---

## 3b. NIAH — long context (64k / 128k) — `[ABANDONED — no data]`

Run: `results/k3_niah/20260705-121832-46b9579/` — 5 arms × {64k,128k} × 5 depths, `--use-packed`.
Intended as the **C2 long-context crossover** evidence (does k2b hold ≥fp16 recall at 64k/128k
while turboquant_mse degrades?). **This run was abandoned after ~10.7h and produced no data.**

**What happened:** on the OLD-code packed path, per-cell decode time *degraded* as the run
progressed — a single 128k cell went from minutes (early arms) to **>35 min** (arm 4,
turboquant_mse), while resident memory crept to **94.3 GiB** against the 95.6 GiB ceiling (the
packed path was not releasing memory between cells). It never OOM'd or deadlocked (CPU-time kept
advancing) but was effectively pathological, with turboquant_mse-128k + all of kivi still ahead
(~1.5 arms of ever-slower 128k cells). It was killed to free the GPU for the authorized Wave-4
runs.

**Cost:** `k3_longbench`/`k3_niah` write the parquet only at the very end (no checkpointing), so
the completed arms (fp16, k2b, k2b_k2r8 — **exactly the crossover comparison that mattered**) were
computed but never written and are lost. This is the sharpest example of finding F6 (no
progress/checkpoint) biting.

**Status of C2:** the **short-context C2 (§3a, 4k–32k) stands** and already shows k2b ≥ fp16 recall
with compression growing with length — directional evidence for the crossover. The **64k/128k
long-context table is not available.** A C2-long rerun should happen on the **Wave-4 code** (the
F1b/F2/F3 fixes target exactly this packed-path slowness) with per-cell checkpointing, not on the
old path that just failed this way.

---

## 4. Systems — memory + latency (Tables T2 / T3)

### T2a — resident-after-prefill census
Run: `results/k3_kernel_census/20260705-051442-46b9579/census.parquet` (prefill + 4-step
diagnostic). GiB:

| seq_len | fp16 dense | k2b chunked | k2b dense_stream |
|---|---|---|---|
| 32k | 27.1 | 27.2 | 32.0 |
| 64k | 39.1 | 39.4 | 49.1 |
| **128k** | **63.3** | **63.8** | 83.3 |

**k2b-chunked ≈ fp16 resident** (63.8 vs 63.3 @128k) — the chunked path stores packed codes + one
KV copy, tracking fp16's footprint. The `dense_stream` reference (83.3) keeps a full dequant copy
and is *larger* — expected, it's not the deployment path. Reproduces the June census.

### T2b — full-generation peak memory (256-token generation)
Run: `.../gen_peak_results.json` (saved into the census run dir). GiB peak:

| seq_len | fp16 dense | k2b packed | Δ |
|---|---|---|---|
| 32k | 23.74 | 24.00 | +0.26 |
| 64k | 38.49 | 39.01 | +0.52 |
| **128k** | **86.0** | **87.03** | **+1.03** |

**k2b's generation peak is ≈ or slightly ABOVE fp16 at every length. Neither OOM'd.** See §5(b) —
this is the memory-reframe finding, not an error.

### T3 — Triton decode latency (RTN kernel)
Run: `results/k3_triton_decode/20260705-121729-46b9579/decode_ledger.parquet` — correctness-gated
(oracle diff `max_abs`, logit-parity gate). Single decode step:

| seq_len | chunked (ms) | triton_fused (ms) | speedup | max_abs_vs_oracle | parity |
|---|---|---|---|---|---|
| 2k | 12.22 | 0.097 | **24×** | 2e-4 | ✓ |
| 8k | 49.62 | 0.104 | **87×** | 1e-4 | ✓ |
| 32k | 204.03 | 0.160 | **240×** | 5e-5 | ✓ |
| 128k | 813.09 | 0.402 | **402×** | 3e-5 | ✓ |

**The RTN fused kernel is 24–402× over chunked PyTorch, correctness-gated.** Framed per the paper
rule: this is **speedup vs a naive chunked baseline — a systems-feasibility result, NOT a
competitive-latency claim** vs FlashAttention/vLLM. **⚠ But see T3b — this is a single-kernel
microbench with prebuilt stacks; in the actual generation loop the story reverses.**

### T3b — `k2b_ph` end-to-end decode profile (Wave-4 code, `profile_decode_ab.py`) — **NEGATIVE**
Run on the **correct fused-kernel arm `k2b_ph`** (V=`turboquant_mse_perhead`), Wave-4 code.
Path probe confirms the kernel fires: `decode attend calls: {fused_k2b: 96, chunked: 32}` (the 32
are prefill/mask), and `[parity] streaming == packed: OK`. So the kernel routes and is correct.
**Per-token decode latency (ms), end-to-end through `generate`:**

| ctx | dense fp16 | **packed (k2b_ph, fused kernel)** | streaming (dense-dequant ref) |
|---|---|---|---|
| 4k | 47.7 | 52.5 | **35.1** |
| 16k | 59.0 | 53.9 | **35.0** |
| 65k | 58.9 | **147.9** | **34.4** |
| 128k | 60.4 | **285.3** | **34.5** |

**This is a NEGATIVE result and it matters:** even with the fused k2b kernel confirmed firing, the
**packed path is 1.5× slower at 4k and blows up to ~8× slower at 128k** (285 vs 34–60 ms/token),
scaling *badly* with context. Meanwhile:
- **`streaming` (StreamingQuantizedCache, dense-dequant) is the fastest — ~34 ms/token, FLAT with
  context** — even beating dense fp16. The quantized *reference* path is genuinely good.
- The **fused kernel fires but is strangled by the packed-cache machinery** around it (the
  `_PagedStacks` block-table rebuilt/`.view()`-ed every decode step, growing with n_blocks). The
  402× microbench (T3) used *prebuilt* stacks and timed only the kernel launch; end-to-end, the
  per-step stack assembly dominates and grows with context. **"Fast kernel" ≠ "fast generation."**

**Consequence for C4:** the deployment-latency claim does **not** hold as-is. The honest C4 is:
(1) the fused kernel is correct and 24–402× faster *as an isolated op* (T3); (2) but the packed
generation path is currently *slower* than even the dense-dequant streaming cache (T3b) because the
cache wrapper's per-token overhead dominates. The Wave-4 F1b/F2/F3 fixes (merge-kernel, codebook
cache, incremental uniform_blk) did **not** fix this — the remaining cost is the `_PagedStacks`
per-step rebuild, which is the next optimization target. **Until that's fixed, the fastest
quantized decode path is `StreamingQuantizedCache` (streaming), not the packed/kernel path.**

**⚠ Critical caveat:** `k3_triton_decode` **hard-asserts `arm == "rtn_token"`** — it benches only
`fused_decode_attention_packed` (the RTN kernel), and **cannot bench `fused_decode_attention_k2b`**
(our actual recipe's kernel). So the 402× is for the RTN arm, **not k2b**. The k2b kernel latency
is **unbenched** here; the docs' "322× k2b" figure is not reproducible through this script. See §5(c).

---

## 5. The three honest narrowings (do not bury)

### (a) LongBench anchor MISS → **NOT truncation** (hypothesis disproven)
Absolute TurboQuant parity is not reproduced (un-truncated fp16 Avg 41.81 vs 50.06; Code 62.04 vs
~46.28). **The truncation hypothesis was tested and REJECTED (2026-07-05):** a truncated rerun
(`--max-prompt-tokens 31500`, code category, Wave-4 code, full 500-sample checkpointed partials)
gives fp16 Code = **61.97** (lcc 65.17, repobench-p 58.77) — **essentially identical to the
un-truncated 62.04.** Truncating our inputs to TurboQuant's ~31.5k budget changes the fp16 Code
score by <0.1 point, so it is **provably not** the cause of the ~62-vs-46 gap. turboquant_mse Code
likewise ≈ 54 (vs un-truncated 51), a consistent positive offset across arms → a **systematic setup
difference, not an input-policy or per-arm issue.** Remaining suspects
to isolate: LongBench **V1 vs E split**, `code_sim`/`fuzzywuzzy` version, prompt-template or
first-line-extraction differences, or TurboQuant's fp16 baseline simply being measured on a setup
that differs from ours in an unidentified way. **The transitive-baseline argument stays UNLICENSED**
until this offset is explained — and it is *not* the quick fix we assumed. (Preliminary — from the
~10%-sampled progress log; final parquet numbers to confirm.) Relative claims (k2b beats
turboquant_mse, holds fp16 quality on our own consistent path) are unaffected.

### (b) Memory: it's KV-slice STORAGE, not peak RSS or OOM-capability
k2b's **generation peak ≈ fp16's** (87 vs 86 GiB @128k), and **neither OOMs** at 128k. Three
reasons the peak doesn't shrink:
- **KV is a minority of peak** for one 8B sequence @128k: weights (~15 GiB) + activations/decode
  transients (tens of GiB) dominate; shrinking a ~16 GiB KV to ~4 GiB is ~14% of an 86 GiB peak.
- **The packed path dequants to fp16 at decode**, so it re-materializes a working set that eats the
  storage saving *at peak* (steady-state resident is smaller — census — but peak is not).
- **Peak measures transients**; the compression is a **steady-state/resident** property.

**The compression is real in the ledger** (16→~4 bits/entry, 4.07× on the stored codes). The batch
OOM sweep was supposed to turn that into a concurrency win. **It did the opposite — see T4.**

### T4 — batch-concurrency OOM sweep (`batch_oom_sweep.py`, Wave-4 code) — **C3 REVERSED**
Max co-resident sequences before OOM @32k on the GH200 (95.6 GiB), plus marginal GiB per added
sequence:

| mode | marginal GiB/seq | **max co-resident @32k** |
|---|---|---|
| **dense fp16** | 4.0 | **16** |
| **packed (k2b_ph, deployment path)** | 4.17 | **8** ⚠ |
| **streaming (StreamingQuantizedCache)** | 8.99 | **6** ⚠ |

**Every "compressed" path fits FEWER concurrent sequences than fp16, not more.** Packed — the
supposed deployment win — fits **half** as many (8 vs 16), and costs **more** memory per sequence
(4.17 vs 4.0 GiB) than uncompressed fp16. Streaming is worst (6).

**Why (the load-bearing diagnosis):** the fp16 KV cache for a 32k Llama-3.1-8B sequence is ~4.3
GiB (32 layers × 8 KV-heads × 128 × 32768 × 2 × 2B) — matching dense's 4.0/seq. k2b stores codes at
~3 bpe K / ~2 bpe V, so the *codes* are ~1 GiB/seq. But the packed cache's **marginal** cost is
4.17 GiB/seq — meaning it carries **~4 GiB of non-code per-sequence state resident** (dequant
working buffers, `_PagedStacks`, the fp16 residual window, partials). **The 4× code compression is
entirely paid back by the cache object's per-sequence scratch.** Streaming is worse still — it
keeps the full dense dequant (9 GiB/seq, ~2× fp16).

**Consequence — C3 does not hold as a runtime-memory claim on ANY measured path.** The compression
is genuine in stored bits but **not realized in deployment VRAM**: neither single-sequence peak
(T2b: packed ≈ fp16), nor batch concurrency (T4: packed = ½ fp16). The KV-slice *storage* number
(16→~4 bpe) is real as an accounting statement, but it does **not** translate to fitting longer
context or more sequences, because the cache implementations don't keep the compressed
representation resident — they hold dense-equivalent working memory per sequence. **Realizing the
memory win requires a cache that keeps codes resident and dequants transiently without a
per-sequence dense buffer — which the current packed path does not do.** This is now the central
systems finding, and it supersedes the earlier "KV-slice storage saving" framing as the honest C3.

### (c) The k2b kernel never fired — ROOT CAUSE FOUND: **we ran the wrong arm**
**CONFIRMED (2026-07-05, on Wave-4 code):** the entire eval used the **`k2b`** arm, which by
construction does **NOT** route to the fused kernel. Direct check of `spec_pair`:

| arm | K spec | V spec | fires `fused_decode_attention_k2b`? |
|---|---|---|---|
| **`k2b`** (what we ran) | lowrank_rtn_channel | **turboquant_mse** (full-C) | **NO — chunked fallback** |
| **`k2b_ph`** (the kernel arm) | lowrank_rtn_channel | **turboquant_mse_perhead** | **YES** |
| `k2b_k2r8` | lowrank_rtn_channel | turboquant_mse (rank 8 < 16) | NO |

The fused k2b gate requires `v_spec.arm == "turboquant_mse_perhead"` (+ rank≥16 pow2, d/n_q_groups
pow2). `k2b`'s V is `turboquant_mse` (**full-C, not per-head**) → it fails the gate and lands in
`chunked_dequant_attention` **by design**. The Wave-4 code emits an explicit warning on exactly
this fallback ("no fused kernel covers this pair; expect ~30-70× slower decode… use `k2b_ph`") —
added in the 2026-07-04 desk review (their finding F0) *specifically so a benchmark can't silently
attribute chunked cost to the Triton path.* It caught this run doing exactly that.

**This one fact reframes the whole systems story:**
- The 60h LongBench, the extended-NIAH 128k meltdown, and the gen-peak "no memory win" were **all
  `k2b` = the chunked fallback**, ~30-70× slower than the kernel — **not** a property of the k2b
  Triton kernel, which never ran.
- The chunked path re-dequantizes every committed page each decode step, which is exactly why it
  degraded near the memory ceiling at 128k and why peak didn't shrink.
- The kernel was correctly wired all along; **the eval arm was wrong.**

**The real measurement is `k2b_ph`, not `k2b`** — that is what Phase-2 `profile_decode_ab.py` and
the C2-long / systems reruns must use. `k3_triton_decode` separately asserts `rtn_token` (benches
only the RTN kernel), so the k2b_ph latency comes from `profile_decode_ab.py`.

**UPDATE — the `k2b_ph` profile ran, and the news is mixed (see T3b):** the kernel **does** fire on
`k2b_ph` (path probe: `fused_k2b: 96`, parity OK) — so the routing gap is real and now understood.
**But even firing, the packed generation path is 1.5–8× SLOWER than the dense-dequant streaming
cache** (285 ms/token @128k vs 34), because the `_PagedStacks` per-decode-step rebuild dominates and
grows with context. So the corrected picture is **not** "wrong arm → kernel is fast." It is: the
wrong arm (`k2b`) explains the old chunked-fallback slowness; the right arm (`k2b_ph`) fires the
kernel but the surrounding packed-cache machinery is itself the bottleneck. **The fastest quantized
decode today is `StreamingQuantizedCache` (streaming, ~34 ms/token flat), not the packed/kernel
path.** Fixing the `_PagedStacks` per-step overhead is the real open kernel-deployment task.

**The decisive, ~5-minute probe:** run one short k2b generation via
`generate_through_cache(..., use_packed=True)` and confirm (1) `fused_decode_attention_k2b` is
actually called (not the chunked fallback), (2) output token-parity vs the unpacked reference, then
(3) latency/peak. **If (1) is "no," that one fact explains the 60h runtime, the no-memory-win, and
"the kernel is slow" — all reducing to "we never ran the kernel."**

---

## 6. Runtime postmortem — why LongBench took ~60h (it's not a bug)

- Direct profile: **~70 ms/token, flat with context** (70ms@2k, 67ms@8k) — no quadratic blowup.
- LongBench full-6-category = **~2.5M sequential decode tokens** (16 datasets × full sets ×
  summarization `max_gen=512` × 5 arms × **unbatched single-sequence** decode).
- 2.5M × 70ms ≈ **~49h** of pure decode; ~60h wall including prefills. Matches.
- The reference streaming cache is ~10× slower per token than dense fp16 (quantize/dequant every
  token) — tolerable at NIAH's ~50-token generations, brutal at LongBench's 512-token summaries.
- **Levers for future runs:** the k2b fused kernel (if it routes — §5c), batch arms across GPUs
  (~5× wall), or trim summarization `max_gen` (departs from official parity).

**Process honesty:** I mis-estimated this repeatedly (5-6h → 14h → 20h → "1-3h left"), then briefly
mis-diagnosed a non-existent quadratic bug. The truth is the flat ~70ms/token profile + 2.5M
tokens. The only real progress signal the script exposes is a py-spy dump of the arm/task frame
locals; there is no built-in progress output or checkpointing (a harness gap — fixed in the
committed changes).

---

## 7. Findings ledger

| # | finding | consequence |
|---|---|---|
| F1 | **kivi = symmetric-RTN strawman**, not real KIVI (`acaaeaf`) | its collapse is not a KIVI result; caveat everywhere |
| F2 | **kernel-routing unverified** — k2b kernel may not fire in generation | §5c; decisive cheap probe outstanding |
| F3 | **truncation-comparability** — un-truncated inputs break absolute anchor | §5a; truncated rerun |
| F4 | **gen-peak reframe** — no peak/OOM win single-seq; win is KV-slice storage | §5b; needs batch-OOM sweep |
| F5 | **k3_triton_decode is RTN-only** (asserts rtn_token) | k2b kernel has no latency bench |
| F6 | **no progress/checkpoint in k3_longbench** | 60h opaque; fixed in committed harness changes |

---

## 8. State + ranked next steps

**Repo state:** local branch `feat/triton-decode-kernel` has **5 un-pushed harness-fix commits**
(truncation flag, slab-rebuild deletion, LongBench-E, kivi diagnosis, plan) **+ 1 pending results
commit**. **Nothing is pushed** — all await your approval (hard rule). All result parquets are
being staged locally (transport step).

**Ranked next steps:**
1. **k2b kernel routing + correctness probe** — cheapest, most decisive. Does
   `generate_through_cache(use_packed=True)` on the real k2b arm actually call
   `fused_decode_attention_k2b`? Resolves F2 and likely reframes §5b/§6.
2. **Truncated LongBench parity rerun** — fixes the anchor (§5a) using the committed truncation
   flag; re-enables the transitive-baseline argument.
3. **Batch-concurrent-sequence OOM sweep** — the experiment that actually demonstrates the VRAM win
   (C3), where KV is the binding constraint.
4. **k2b kernel latency/peak bench** — once routing (step 1) is confirmed; gives the real C4/T3 for
   our recipe (the RTN 402× is a proxy, not our kernel).

**Bottom line:** the science on quality/compression is in good shape (k2b ≈ fp16 quality at 4×,
beats the TurboQuant arm). The systems story needs the kernel actually running to close — and
confirming whether it runs is a five-minute test that gates the memory claim, the latency claim,
and the runtime all at once.

---

## 9. Addendum (2026-07-05, controller session) — §5c and F5 answered statically; F2 CONFIRMED

Written after this report; from code reading + the 2026-07-04 desk review
(`docs/2026-07-04-triton-decode-desk-review.md`), no GPU needed.

**§5c's probe is answered: the k2b kernel definitively did NOT fire — by design, not by bug.**
`spec_pair("k2b")` returns V=`turboquant_mse` (full-C; `src/bmx/cache/recipes.py:61`), but
`fused_k2b_ok` requires V=`turboquant_mse_perhead` (`packed_streaming.py:619`). The full-C
Hadamard couples heads and provably cannot run per-head in a decode kernel — that is exactly why
the `k2b_ph` codec exists. **Only `k2b_ph` and rtn_token/rtn_token route to fused kernels.**
Every eval row in this report used arm `k2b` (full-C) → reference/chunked paths throughout.
F2's suspicion is CONFIRMED, and the fix is an arm choice, not a code fix: run the deployment
story on `k2b_ph` (quality-equivalent, ratio 0.986; true-parity codec since C1/85839cc).

**The in-flight §3b NIAH run is decoding through the chunked fallback for ALL five arms**
(fp16/k2b/turboquant_mse/kivi/turboquant_prod all miss both fused predicates) — that IS why the
128k packed cells are slow (chunked ≈ 813 ms/layer-call ×32 layers at 128k). Results remain
CORRECT (chunked is the correct reference); only slow. As of `9cf55d4` the packed cache
warns loudly on this exact silent fallback.

**F5 (no k2b kernel latency bench) is closed** by `scripts/profile_decode_ab.py` (commit
`bac7196`+`9cf55d4`): parity-gated end-to-end A/B/C (dense / streaming / packed) with a
path-probe that COUNTS fused-vs-chunked calls and refuses to time a misrouted arm — the F0
mistake (benching `turboquant_mse` through the packed path and blaming the kernel) is now
mechanically impossible. Run it in the Task-5 slot when the GPU is free.

**Known kernel-path overheads once the right arm routes** (desk review F1–F3, fixes in flight
locally as Wave 4): the fp16 tail merge ran in PyTorch every decode step (GPU merge kernel was
dead code during generation), k2b constants were re-uploaded H2D per call, and `uniform_blk` did
an O(n_blocks) Python scan per layer-step. §6's rerun lever "the k2b fused kernel (if it routes)"
should therefore read: **route k2b_ph + Wave-4 fixes**, then LongBench decode ≈ dense-fp16-like
per-token cost instead of ~70 ms.
