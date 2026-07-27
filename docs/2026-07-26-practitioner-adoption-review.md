# bmx / K4 KV-cache codec — staff inference engineer adoption review

**Reviewer stance:** skeptical staff engineer at a serving company. I do not care
about the break-even inequality or the Minkowski transfer-ceiling theorem. I care
about: what do I integrate, how much memory do I actually get back per GPU, what does
it cost me in tokens/sec, and what breaks in production. Every number below is cited to
the repo's own record. Where the record is silent I write **ADOPTER GAP** rather than
guess.

Model scope of ALL evidence: `meta-llama/Llama-3.1-8B-Instruct` and `Qwen3-8B`, both
full-rotary, on a single GH200 480GB. n=2 models, n=1 GPU type.

---

## 1. What is the artifact, and how far is it from my stack?

**What I would integrate.** This is transformers-native research code. The deliverable
is a set of `Cache` subclasses plus a per-model calibration pack:

- `StreamingQuantizedCache` / `StreamingQuantizedLayer` — subclass transformers 5.11
  `DynamicCache`/`DynamicLayer`, quantize-on-append, fp16 recent window. This is the
  **latency path** (see §3). (`src/bmx/cache/streaming.py`; README §3.)
- `PackedStreamingCache` — keeps packed codes resident, chunked dequant-attention at
  decode, flash SDPA at prefill. This is the **memory path**
  (`src/bmx/cache/packed_streaming.py`).
- The two Triton fused decode kernels: `fused_decode_attention_packed` (RTN arm) and
  `fused_decode_attention_k2b` (the k2b_ph arm). **No fused spectral/k4 kernel exists**
  (`src/bmx/cache/triton_dequant_attention.py:731,1232`; verified — only two fused
  entry points in the whole tree).
- A per-model **spectral pack file** (`k4_packs_<model>.safetensors` + a `.json`
  sidecar), loaded once at attach (`spectral.py:1137 load_packs`,
  `packed_streaming.py:1091`). For the shipped k4 arm you MUST pass `--pack-path` or the
  recipe raises (`recipes.py:110-113`).

**Distance to vLLM / sglang: this is the single biggest integration cost and it is
unmeasured.** Everything rides on transformers' `Cache` / `AttentionInterface` /
`AttentionMaskInterface` registration machinery. The program itself hit two bugs
proving how tightly coupled this is to transformers internals: a custom
`AttentionInterface` that ran **maskless** because no `AttentionMaskInterface` was
registered (garbage logits until fixed), and a prefill/decode dispatch bug
(chronicle Part IV, "the kernel program"). vLLM and sglang do NOT use transformers'
`Cache` abstraction — they have their own paged-KV managers and their own attention
kernels. **ADOPTER GAP:** there is zero evidence in the record of a vLLM or sglang
port, a PagedAttention integration, or even a design sketch for one. The paged layout
here (`_PagedStacks`, uniform PAGE=128) is bespoke to this codebase. Porting the
in-kernel dequant (lowrank-K + RoPE + per-head Hadamard-V) into a vLLM attention
backend is a from-scratch kernel-engineering project, not a config change.

**Calibration workflow + cost.** Fit a spectral pack from a handful of general-text
caches:

- Collect ~4 caches of 2048 tokens (the shipped Llama pack used
  `4×2048-token Instruct caches`, offsets 2048–8192; rental §2).
- Run the fit (`experiments/k4_fit_packs.py`), which computes per-layer weighted-KLT
  bases + reverse-waterfill bit allocation and writes the pack.
- **Cheap:** G1 passes from a **single 2048-token cache** on both models (rental §5,
  calibration ladder), so the input-data requirement is trivial. Even better for
  privacy/ops: the pack can be synthesized from **trigram count tables alone**
  (D_tri 0.036–0.074 < 0.10 bar, both models; rental §5) — you never need to ship raw
  calibration text.
- The fit itself is minutes-scale (rental stage 3 was 06:50→07:40 UTC and included the
  duel + calibration ladder, not just one fit). ADOPTER GAP: no isolated wall-clock for
  a single production pack-fit is cited, but the evidence points to "cheap, one-time,
  per-model, offline."

**Bottom line for Q1:** the science is real and the cache classes work under
`generate()`, but what you integrate is a transformers-5.11-coupled research cache, and
the distance to a production serving engine (vLLM/sglang) is an unmeasured, non-trivial
kernel + KV-manager port. That port is the gating adoption cost, and the record is
silent on it.

---

## 2. Memory — what do I get, and when?

The memory win is **real and it is the strongest part of the deployment story**, but
only on the *chunked* path and only as a resident/steady-state property, not a peak
property.

**Per-sequence resident, measured (GH200, chunked = the deployment memory path)**
(rental appendix-systems §3, run `20260725-200112` + 128k cells):

| context | fp16 dense | k2b chunked | **k4_b2.5 chunked** | k4 vs fp16 |
|---|---|---|---|---|
| 32k | 27.07 GiB | 26.72 | **24.08** | −11% |
| 64k | 39.19 | 38.48 | **32.90** | −16% |
| 96k | 51.32 | 50.23 | **41.72** | −19% |
| 128k | 63.30 | 61.89 | **50.48** | **−20%** |

The saving grows with context (the fp16 recent-window amortizes). At 128k k4 is 20%
under fp16 dense; k2b already undercuts fp16 too (61.89 vs 63.30). bpe_k for k4 at 128k
is 2.82 (rental appendix-systems §3).

**Co-residency — the metric I actually buy (sequences per GPU).** This is where the
history is ugly and you must read it carefully:

- The `batch_oom_sweep` (2026-07-05, T4) measured packed fitting **8** co-resident 32k
  sequences vs fp16's **16** — *half* — at a marginal 4.17 GiB/seq vs fp16's 4.0
  (2026-07-05 §T4). That was a genuine, published NEGATIVE: "every compressed path fits
  FEWER concurrent sequences than fp16."
- It was then partially decomposed and reversed (2026-07-06 overnight): the negative
  was three fixable artifacts — an all-prefill sweep design, per-layer RoPE table
  duplication (~0.5 GiB/cache @32k), and a `pack_v=False` script-flag shadow. After the
  fixes, **packed marginal 2.258 GiB/seq @32k vs dense fp16's 4.008** — i.e.
  ~1.8× more sequences per GiB, with "zero residual mystery" (2026-07-06 final
  addendum). The fixed steady-state sweep measured packed **12** co-resident 32k seqs
  vs dense 16 in one run, with the ledger predicting **28–34** at the true 2.3–2.8
  GiB/seq footprint (2026-07-06 addendum).
- **ADOPTER GAP (important):** the clean 2.258 GiB/seq marginal was measured for the
  **k2b_ph** packed cache (pack_v path). I find **no co-residency / GiB-per-seq sweep
  for the shipped k4 spectral arm** on the packed path. The k4 numbers I have are
  single-sequence resident census only (table above). The chronicle explicitly lists
  "a wider batched-128k co-residency sweep" as still-open work (Part X; README line 71).
  So the headline co-residency number (2.258 GiB/seq) is a k2b number, and the k4
  equivalent is a projection at best.

**Where the saving does NOT materialize (be honest with yourself):**

1. **Peak RSS, single sequence.** k2b generation peak ≈ fp16 at every length (87.03 vs
   86.0 GiB @128k, 2026-07-05 §T2b). KV is a minority of peak for one 8B sequence
   (weights ~15 GiB + activation transients dominate; §5b). The compression is a
   steady-state/resident property, not a peak-memory or single-seq-OOM property.
2. **Short context.** The pack charge amortizes over sequence length; the spectral
   decoder cost is `dec_bits·c_used/S` bpe (`spectral.py:898-910`), so at small S the
   pack tax dominates — a pack codec's bpe goes ~8 bpe @2k → ~0.5 @32k (duel doc; k4
   crosses below tq_b3 only at **~5.6–5.7k context**, chronicle Part VI). Below ~5–6k
   tokens, k4 is spending MORE bits than the strongest scalar baseline. LongBench
   corroborates: lcc's quality win "costs +0.91 bits at short sequences" (rental §3).
3. **Dense-stream path.** The naive "compress then keep a dense dequant copy" path
   (`dense_stream`) is *larger* than fp16 (83.56 GiB @128k for k4; appendix-systems §3).
   You get the win ONLY on chunked. Pick the wrong path and compression is negative.

**Scratch overheads and their fixes (already landed).** The 4 GiB/seq of non-code
scratch that caused the original T4 negative decomposed into: sweep-design transient,
per-layer RoPE duplication (fixed, shared per cache — `e4720b2`), int16 V-index bloat
(fixed, pack_v default ON — `90c32dc`), and W5-1/W5-2 single-storage page dedup. The
remaining ladder to the honest-bpe ~1.1 GiB/seq (W5-3 K-residual 3-bit packing −0.67,
fp16-ing fp32 metadata −0.3) is "optional post-paper" (2026-07-06 final addendum) —
i.e. NOT done.

**Bottom line for Q2:** ~20% single-seq resident at 128k on the chunked path is real
and reproduced. The co-residency win (the thing I buy) is real for k2b (~2.26 GiB/seq)
but **unmeasured for the shipped k4 arm**, and the whole win evaporates on short
contexts (<~6k) and on the wrong (dense-stream / peak) path.

---

## 3. Latency — what do I pay? (The critical question.)

Here is the menu I would actually run in production, per deployable arm. **The flagship
k4 spectral arm has no fused kernel and routes to the chunked decode path** — I verified
this directly in the source, not just the docs.

### The routing, from source (`packed_streaming.py` attend, lines 904–1041):

- `fused_packed_ok` requires `k_spec.arm == "rtn_token"` AND `v_spec.arm == "rtn_token"`
  (lines 908–909).
- `fused_k2b_ok` requires `k_spec.arm == "lowrank_rtn_channel"` AND
  `v_spec.arm == "turboquant_mse_perhead"` (lines 971–972).
- **`spectral` (k4) matches NEITHER predicate.** It falls through to
  `chunked_dequant_attention` (line 1042), and the code emits a dedicated warning
  (lines 1027–1032): *"PackedStreamingCache spectral decode runs the CHUNKED path BY
  DESIGN … no fused spectral kernel; expect chunked-class decode latency."*
- `recipes.py:139-149` confirms the k4 arm returns K=`spectral`, V=`turboquant_mse`
  (full-C, not per-head) — so it cannot even borrow the k2b kernel.

The stage-2 path probe on the GH200 confirms it in production: `[path probe] chunked
path taken by design, 128 calls (no fused spectral kernel exists yet)`
(appendix-systems §4).

### The measured latency table (2026-07-05 §T3b, k2b_ph, end-to-end through `generate`, ms/token):

| context | dense fp16 | **packed k2b_ph (fused kernel)** | streaming (dense-dequant) |
|---|---|---|---|
| 4k | 47.7 | 52.5 | **35.1** |
| 16k | 59.0 | 53.9 | **35.0** |
| 65k | 58.9 | **147.9** | **34.4** |
| 128k | 60.4 | **285.3** | **34.5** |

The overnight fixes (fp16 tensor-core dots, pack_v) improved the packed path to
**51.8 / 63.2 / 132.6 / 256.7 ms** at 4k/16k/65k/128k (2026-07-06 dawn addendum) — still
2–8× worse than streaming. The isolated kernel microbench is fast (1.85× vs chunked;
2026-07-06), but "fast kernel ≠ fast generation" — the `_PagedStacks`/tail-merge glue
dominates.

### The production menu, per deployable arm:

| arm | production path | measured decode cost | quality (LongBench macro) |
|---|---|---|---|
| **k4_b2.5_dec8tl** (flagship) | **chunked** (no fused kernel) | **ADOPTER GAP** — no end-to-end k4 chunked ms/token at scale is in the record | 40.85 @ 3.081 bits (rental §3) |
| **k2b_ph** | packed fused kernel, OR streaming | packed 51.8→256.7 ms/tok (4k→128k); streaming ~34 ms/tok flat | 58.25 Code avg, −3.72 vs fp16 (2026-07-06) |
| **rtn** (rtn_token/rtn_token) | packed fused kernel | ADOPTER GAP — no end-to-end rtn ms/token in the record (only the microbench 24–402× vs chunked) | not a shipped-quality arm |

**The load-bearing gap.** The chronicle's own honest ceiling is that decode wall-clock
is **KV-fraction-bounded, not compression-ratio-bounded** (memory
`triton-decode-win-prediction`): at LongBench contexts attention is a small slice of the
~70 ms/token step (MLP/linears dominate), so even a zero-overhead packed path *roughly
ties* the dense cache on speed — the honest claim is "memory at speed parity, not a
speedup headline" (2026-07-04 desk review, "What a fix buys"). But that is the
argument for the *streaming/k2b* path.

For the shipped **k4 chunked** path I have NO end-to-end ms/token at any scale. The only
chunked-latency data point anywhere is a *plan* line asserting "the chunked path decoded
**k2b** at ~60 ms/tok at 64k (acceptable)" (`plans/2026-07-23-packed-spectral-path.md:7`)
— that is (a) a plan, not a measured result, and (b) k2b, not k4. The chunked
`chunked_dequant_attention` path re-dequantizes **every committed page, every layer,
every token** (desk review F0; §5c) — this is the O(S)/step path the fused kernels exist
to replace, and it is what made the abandoned 128k NIAH run degrade to >35 min/cell and
creep to 94.3 GiB (2026-07-05 §3b). The spectral pack adds a per-layer **C×C decoder
matmul** at every decode step on top of that.

**Estimate I would carry into a capacity plan (clearly marked as my inference, not the
repo's):** the k4 chunked decode does everything the k2b chunked path does PLUS a C×C
(1024×1024) reconstruction per layer per step. The k2b chunked path is the "~30–70×
slower than a fused/dense step at 8k, ×n_layers" path the code warns about
(`packed_streaming.py:1017-1021`). At the observed ~813 ms/layer-call chunked cost at
128k (2026-07-05 §T3), k4 chunked at 128k is plausibly **seconds/token**, i.e.
serving-unviable at long context. **This is an inference; the repo does not measure it,
so it is an ADOPTER GAP, and it is the most adoption-blocking gap I found.**

**What actually ships fast today:** `StreamingQuantizedCache` at ~34 ms/token, FLAT with
context, beating even dense fp16 (2026-07-05 §T3b; GQA-aware attend). But streaming is
the k2b-recipe path, and it fits FEWER co-resident sequences (6 @32k, worst of all —
2026-07-05 §T4) because it keeps a full dense dequant. So streaming buys you latency at
the cost of the memory win. **You cannot have the k4 memory win AND low latency on any
path in the current code** — the k4 memory win lives on chunked, and chunked is slow.

**Bottom line for Q3:** the flagship k4 arm is a **memory play with an unmeasured and
likely serving-hostile decode latency at long context**. k2b_ph gives you a fused
kernel but no speedup over fp16 (KV-fraction-bound), and its memory-win variant
(streaming) is the slowest on co-residency. There is no arm that is simultaneously
best-memory and best-latency.

---

## 4. Quality — the risk surface

**Two-model evidence, and it is decent within scope:**

- LongBench macro **40.85 @ 3.081 mean bits** (Llama, full 3750-sample splits), +0.48
  over the strongest TurboQuant baseline (tq_b3, 40.37) at −0.125 bits, +0.13 over its
  own fp32 predecessor at −0.72 bits (rental §3). The edge is entirely
  synthetic/retrieval (+3.36) and code (+1.40); the four **language categories slightly
  favor tq_b3** (rental §3, §11). So on QA/summarization you are at rough parity, not
  ahead — monitor those categories if your traffic is language-heavy.
- NIAH is an **honest null on Llama** — all arms (k4, tq_b3, tq_k3v2, fp16) within
  codec-RNG noise at 32k/64k over 5 seeds (rental §4). "Parity at fewer bits" is the
  claim; there is no measured retrieval *advantage* on Llama.
- Qwen LongBench full-set was **NOT run** — only an n=100 directional probe (rental §7).
  So the two-model quality claim is Llama-full + Qwen-probe, not two full evals.

**Per-model scope limits (hard boundaries):**

- **Full-rotary models only.** The W-instrument covers Llama/Qwen3; "partial-rotary
  needs the non-rotated sub-block added" (paper-shelf, W-instrument scoping). **Hybrid /
  linear-attention / unified-KV architectures are explicitly excluded** — "linear layers
  have no growing KV; unified-KV dissolves the K/V split … bounds where this codec
  applies" (chronicle Part X). If your fleet is moving to hybrid attention, this codec
  does not apply to the linear layers.
- **rotated-W is Llama-licensed only.** The Qwen pack is frozen-W by scope (rental §2
  scope note); the rotated-vs-frozen choice was validated on Llama's causal instrument,
  which never ran on Qwen. So the exact shipped config is one-model-validated.

**Calibration sensitivity axes (the production drift risk):**

- **Domain matters.** Basis does NOT transfer across domains — hybrid recovery
  0.63–0.75 < 0.9 bar on both models (rental §5, H3). Cross-domain transfer penalty
  (D 0.229–0.440) is real. The exploitation lever is **whole-pack per-domain fitting** —
  i.e. if you serve code and prose from one pack, you leave quality on the table.
- **Word order matters at scale.** The gpt2 "token-marginal" calibration shortcut
  **reverses at 8B** — shuffled-order calibration is *worse* than cross-domain (rental
  §5). So you cannot calibrate from a bag-of-tokens; order-preserving text (or ≥trigram
  stats) is required.
- **Prompt policy is a MASSIVE unquantified axis.** Anchor forensics found raw-template
  vs chat-wrapped prompts move LongBench code_sim by **43.7 points** — an order of
  magnitude larger than any quantization effect in the entire program (chronicle Part
  IV). This is why the program retreated to "delta-parity" comparisons. **Task-5
  (prompt-policy sensitivity) is pending** — ADOPTER GAP. If your serving prompt
  template differs from the calibration/eval template, absolute quality is unpredictable.

**What I would monitor in production:** (1) language-category task quality (KV codec is
weakest there, slightly behind tq_b3); (2) domain drift between calibration corpus and
live traffic (basis non-transfer); (3) prompt-template consistency (43.7-pt sensitivity);
(4) the Qwen-style TQ-collapse is a *baseline* fragility not a k4 one, but it signals
that model-family interactions are real and unmodeled (n=1 model-pair, mechanism open —
rental §4).

**Bottom line for Q4:** quality is parity-with-retrieval/code-edge on two full-rotary 8B
models, with real, measured calibration-drift axes and one enormous *unquantified* axis
(prompt policy, Task-5 pending). The risk surface is manageable for a fixed
domain+template, dangerous if either drifts.

---

## 5. Operational costs & failure modes

**Pack files — per model.** One `k4_packs_<model>.safetensors` + a small `.json`
sidecar, stored per-layer `enc`/`dec` C×C fp32 matrices + `lam` + per-budget bit
allocations (`spectral.py:1108-1119, 185-186`). Measured on disk:
- gpt2 (C=768, 12 layers): **57 MB** (`k4_packs_gpt2.safetensors`).
- Qwen3-0.6B (C=1024, 28 layers): **236 MB** (`k4_packs_qwen3_06b.safetensors`).
- **8B model (C=1024, 32 layers):** ADOPTER GAP on exact size — the shipped
  `k4_packs_llama31_instruct_rotw.safetensors` is **VM-side and gitignored**
  (rental §2), never committed. Scaling the 0.6B/28-layer pack to 32 layers gives
  ≈**270 MB fp32** per 8B model (my arithmetic; enc+dec = 2·C²·4·n_layers). The int8_tl
  ships an int8 decoder (`spectral.py:900-903`), which is an *accounting/deploy* charge
  reduction (dec stored int8 at 8 bits/entry) — whether the on-disk pack is also halved
  is an ADOPTER GAP. Loading is a one-time `load_file` at cache attach
  (`spectral.py:1149`, `packed_streaming.py:1091`) — cheap, but it is a per-model
  artifact your deploy system must version, distribute, and pin to the model SHA.

**Per-layer thresholds.** The shipped `int8_tl` decoder picks, per layer, the largest
tier T whose offline certificate ≤ the 5% bar (appendix-systems §1). These T-maps are
baked into the pack. The certificate is a conservative screen (5.3–8.6× conservative;
rental §6), which is good — but the binding-layer nominal margins sit right on the bar
(**1.00–1.09×**, e.g. Llama b2.2 layer 27 = 1.001×; rental §2, §6). The *measured*
damage is 5–9× under the bar, so there is real headroom, but the nominal margin is a
razor edge — you are trusting the certificate's conservatism, not a comfortable margin.
For Qwen b2.2 the binding number (1.09×) is **log-cited only, not reproducible from the
committed parquet** (appendix-systems §1) — a provenance hole in one deployment cell.

**Model updates / finetunes.** The pack is calibrated to a specific model's key/query
second-moment structure. Any finetune that shifts activation statistics invalidates the
basis (domain non-transfer, rental §5, is the direct evidence). **ADOPTER GAP:** there
is no measurement of how much a finetune degrades a pack, nor a re-calibration trigger/
policy. Operationally you must treat "refit the pack" as part of every model-update
pipeline. The good news: refitting is cheap (nc=1, 2048 tokens, or trigram tables), so
this is a pipeline-plumbing cost, not a data cost.

**The arm-misconfiguration trap (the F0 lesson) — and whether the warning is enough.**
This is the scariest operational footgun and it is well-documented. Running the wrong
arm through the packed path silently falls to the chunked fallback, **~30–70× slower
per decode step** (`packed_streaming.py:1017-1021`), with correct outputs. The program
itself burned a **60-hour LongBench run + a 10.7-hour abandoned NIAH run** on exactly
this mistake (ran `k2b` — full-C V — which routes to chunked by construction, instead
of `k2b_ph`; 2026-07-05 §5c, §3b). The mitigation is a one-time `warnings.warn` on the
CUDA chunked fallback (`packed_streaming.py:1033-1041`), plus a path-probe in
`profile_decode_ab.py` that counts fused-vs-chunked calls and refuses to time a
misrouted arm.

**Is the shipped warning sufficient? No, for a serving deployment.** A `warnings.warn`
(a) fires once per process and is trivially swallowed by log filters, (b) does not fail
the request, and (c) does NOT fire for the *flagship k4 arm* — for spectral the "warning"
literally says the chunked path is *by design* (`packed_streaming.py:1027-1032`). So the
guardrail that catches the k2b-vs-k2b_ph footgun explicitly greenlights the k4 slow
path as intended behavior. A serving system needs a hard startup assertion ("this arm
has no fused kernel; refuse or require explicit opt-in") or a metrics-plane alarm on
chunked-call-count, neither of which exists. **ADOPTER GAP:** no hard guard, only a
soft warn, and it is silent-by-design for the flagship arm.

**Other ops honesty from the record (credit where due):** the program documents its own
orchestration bugs (bash `set -e` suppression, `sys.path[0]` trap, test-debris commits
polluting `results/`; rental §9). This is a research repo's honesty, not a serving
system's hardening — expect to do the hardening yourself.

---

## 6. The verdict — three adopter personas

### (a) High-batch API serving, 8–32k contexts

**PASS today. Adopt-after-X.**

Rationale: at 8–32k, decode wall-clock is KV-fraction-bounded, so there is no latency
win to be had (best case is parity — desk review). The k4 arm's memory win is thin here
(−11% resident @32k; and BELOW ~5.6k context k4 spends MORE bits than tq_b3 — chronicle
Part VI). The one metric that matters — sequences per GPU — has **no measured k4 number**
(the 2.26 GiB/seq co-residency win is k2b, not k4). And the flagship arm routes to a
chunked decode path whose end-to-end latency at high batch × 32k is unmeasured and, by
my inference, likely a throughput regression. For a throughput-serving shop the honest
value proposition (memory at speed parity) is exactly the regime where the memory win is
smallest.

**X = (1) a fused spectral decode kernel** OR a demonstration that k2b_ph-streaming (not
k4) meets your quality bar at ~34 ms/token, **AND (2) a measured co-residency sweep for
the actual shipped arm**, **AND (3) a vLLM/sglang plugin** — you will not run a
transformers-`generate` loop in high-batch production.

### (b) Long-context single-user, 128k+

**Adopt-after-X — this is the persona the memory story is built for, but latency blocks it.**

Rationale: at 128k the resident win is real and biggest (−20%, 50.48 vs 63.30 GiB;
appendix-systems §3), NIAH holds fp16 parity through 128k on the packed path (rental §4),
and quality is neutral-to-better. If you are memory-bound fitting one long sequence, k4
chunked genuinely lets you hold 128k in ~50 GiB instead of ~63. **But** the decode
latency at 128k on the chunked spectral path is the single unmeasured, likely-fatal
number (my inference: seconds/token). Single-user 128k means every token pays the full
per-page re-dequant + C×C reconstruction. If that lands at seconds/token, the memory win
is academic.

**X = the fused spectral decode kernel** (Phase B, explicitly gated and NOT built —
`plans/2026-07-23-packed-spectral-path.md:7`, README line 74), OR a measured k4 chunked
ms/token at 128k that proves it is tolerable for interactive single-user use. Without one
of those two, you cannot size the SLA.

### (c) On-device / edge

**PASS.**

Rationale: nothing in the record targets edge. All evidence is GH200 (480 GB, an
NVIDIA datacenter part); the kernels are Triton+CUDA and hard-fail without CUDA
(`triton_dequant_attention.py:297`). The repo was developed against AMD (no CUDA) and
explicitly runs the CUDA-authoritative path only on rented NVIDIA (CLAUDE.md). No mobile
NPU, no ARM, no CoreML/TFLite path, no int-only decode without Triton. The chunked path
runs in PyTorch (CPU-capable) but at chunked latency, which is a non-starter on-device.
There is no memory or latency measurement below datacenter GPU scale.

**X = a from-scratch edge port** (int-only or Metal/NNAPI dequant-attention kernel +
CPU/NPU memory measurements). This is effectively a different project.

---

## ADOPTER GAPS — ranked by adoption-blocking severity

1. **[BLOCKER] No end-to-end decode latency (ms/token) for the shipped k4 spectral arm
   at any scale.** The flagship arm routes to the chunked path
   (`packed_streaming.py:1027-1042`, confirmed in source + GH200 path probe), and the
   chunked path re-dequantizes every page every layer every token PLUS a C×C spectral
   reconstruction. No measured number exists; my inference is seconds/token at 128k.
   This single gap blocks personas (a) and (b). — *The most adoption-blocking fact.*

2. **[BLOCKER] No fused spectral decode kernel exists.** Only
   `fused_decode_attention_packed` (RTN) and `_k2b` are implemented
   (`triton_dequant_attention.py:731,1232`). It is explicitly "Phase B, gated, not
   planned" (`plans/2026-07-23-packed-spectral-path.md:7`; README line 74). Without it
   the k4 memory win is inseparable from chunked latency.

3. **[BLOCKER] No vLLM / sglang / PagedAttention integration or design.** Everything is
   transformers-5.11 `Cache`-coupled. No production-serving-engine port is in the record.
   High-batch serving (persona a) cannot run a `generate()` loop.

4. **[HIGH] No co-residency (sequences-per-GPU) measurement for the k4 arm.** The clean
   2.258 GiB/seq marginal is k2b_ph, not k4 (2026-07-06). The k4 co-residency sweep is
   listed as open work (chronicle Part X). This is the exact metric a serving engineer
   buys, and it is missing for the shipped arm.

5. **[HIGH] Prompt-policy quality sensitivity (Task-5) is pending, and the axis is
   enormous** — 43.7 LongBench points from template alone (chronicle Part IV), vs <1.5
   points for the whole quantization effect. Absolute production quality is unpredictable
   if your template ≠ the calibration template.

Additional (medium/low): finetune/model-update pack-degradation is unmeasured (§5); the
8B pack file is gitignored/VM-side so its exact size and int8-on-disk footprint are not
in the tree (§5); the chunked-fallback guard is a soft `warnings.warn`, silent-by-design
for k4, not a hard serving guard (§5); one deployment cell (Qwen b2.2 binding margin
1.09×) is log-cited only, not parquet-reproducible (appendix-systems §1); the certificate
runs anti-conservative on layer 2 up to ~3× (absorbed by margin, but stated — chronicle
Part VII).

---

## One-paragraph adopter summary

This is rigorous, honest research with a real, reproduced **memory** result: ~20%
single-sequence resident savings at 128k on the chunked path, quality-neutral-to-better
on two full-rotary 8B models, calibratable cheaply from one 2048-token cache or trigram
tables. But as a serving artifact it is not ready: the flagship k4 arm has **no fused
kernel** and decodes through a chunked path whose **latency is unmeasured and likely
serving-hostile at long context**; the memory-vs-latency tradeoff is unresolved (no arm
is both best-memory and low-latency); the **co-residency win is measured for k2b, not
k4**; and there is **no path to vLLM/sglang**. I would not adopt today for any of the
three personas. The nearest adoptable slice is the long-context single-user memory play
(persona b) — and it unlocks the moment someone measures (or builds a kernel for) k4
chunked decode latency at 128k.
