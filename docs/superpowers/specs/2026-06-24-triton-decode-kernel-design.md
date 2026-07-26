# Triton fused dequant-attention decode kernel — design (2026-06-24)

Phase 3 of the fused dequant-attention program. Phases 1+2 (`PackedStreamingCache`,
chunked dequant-attention at decode, flash SDPA at prefill) are merged to main
(`fb7209a`) and GH200-confirmed: resident ~64 GiB at 128k (≈ fp16, ~19 GiB under
the dense path), the batched 128k NIAH sweep unblocked, quality parity confirmed
(`docs/2026-06-23-kernel-census-results.md`). Suite: 243 passed.

This spec covers the **deployment-grade Triton decode kernel** — the literal
process speed/RSS artifact to ship alongside the paper, plus the practice of
writing it well. It is engineering, not science: the science (key-bits↔context
tradeoff, k2b beats turboquant at matched compression) is settled and committed.

## The honesty constraint this design rests on (the crux)

The Phase-3 gate (from `2026-06-23-fused-dequant-attention-design.md`) is: **build
Triton ONLY if a deployment speed/RSS claim needs it** — it is NOT required to
clear the memory ceiling (chunked-PyTorch already does, 64.1 GiB at 128k). So the
first deliverable is not a kernel; it is an honest *prediction of what the kernel
buys*, computed before building.

**The predicted decode speedup is KV-fraction-bounded, NOT the compression ratio.**
A batch=1 decode step is memory-bound (~0.5 FLOP/byte, <5% MFU — vault
[[Prefill Decode Separation]], [[Roofline Model]]). Step latency ≈ bytes streamed
from HBM ÷ achieved HBM bandwidth. Bytes per step:

```
fp16 path:    W + KV_read_fp16(S)     KV_read_fp16(S)   = 2·L·h_kv·S·d·2
packed path:  W + KV_read_packed(S)   KV_read_packed(S) = L·h_kv·S·d·(bpe_k+bpe_v)/8
                                                          + fp16 recent-window bytes
predicted_speedup(S) = (W + KV_read_fp16(S)) / (W + KV_read_packed(S))
```

`W` is the **fixed** per-step weight stream (~14.9 GiB for Llama-3.1-8B, already a
measured constant in `src/bmx/bench/kv_memory.py`). Our compression shrinks ONLY
the KV-read term — it cannot touch `W`. So the win approaches the compression
ratio ONLY when KV dominates weights, i.e. at long context. At short context the
honest answer is ~1×.

**This byte-ratio is an UPPER BOUND on the latency speedup, not the speedup
itself.** Actual latency = bytes / achieved-bandwidth + dequant-compute-time. Two
effects make the real speedup ≤ the byte-ratio, never more: (a) packed codes are
int8, so achieved HBM bandwidth may differ from fp16 sequential reads; (b) dequant
FLOPs consume SM cycles — "free" only while they stay under the bandwidth time.
Both only ADD to the packed path's latency. In practice `BW_fp16 ≈ BW_packed`
(both sequential HBM reads) and dequant time `<<` bandwidth time at k2b's 3+2-bit
compression, so the byte-ratio is a good first-order approximation — but the model
reports it AS an upper bound, and the stage-0 unit test asserts the
compute-approaching-bandwidth flag fires at the compression ratio where dequant
FLOPs stop being free.

**The crossover context is computed in code (stage 0), not hardcoded here** — it
is sensitive to exact bpe and to the bits→bytes factor, and a hand-computed number
is error-prone. As a calibration check against the ledger's measured anchors (one
fp16 KV copy = 16 GiB at 128k; `W` = 14.9 GiB; k2b ≈ 5.5 bpe vs 32 bpe-pair fp16),
the crossover where `KV_read_packed = W` lands around **~700k tokens** for
Llama-3.1-8B/k2b — far above 128k. The blunt implication the gate must surface:
**at 128k the predicted decode *latency* win over fp16 is modest** (≈ (W + 16 +
A)/(W + 2.75 + A) ≈ 1.17× with A ≈ 61 GiB), because weights+activations still
dominate the per-step byte stream. So the honest deployment claim leans primarily
on the **memory/RSS win** (real, ~C× resident) and only secondarily on a latency
win that materializes at much longer context. Stage 0 produces the exact curve and
crossover; if the predicted latency win at the target context is thin, that IS the
finding (kill-or-confirm) — the kernel's value is then the RSS artifact + the
craft, not a decode-latency headline.

### Grounding (Personal Brain vault — foundational + wiki)

- **Megakernels target the weight stream; we target the KV cache — a different,
  smaller lever.** `foundational/ai-performance-engineering/Grokking Megakernels.md`:
  decode is a streaming problem and "the weights are the stream" (~line 371); INT4
  weight quant gives ~4× because weights ARE ~the whole stream (~line 2057). The
  KV-cache read "grows with sequence length" and is the term that pulls the
  megakernel off its bandwidth ceiling as position grows (~line 290, ~line 1181).
  Our kernel shrinks that KV term only — hence KV-fraction-bounded.
- **Dequant FLOPs are free in the memory-bound regime** — "the extra dequantization
  fits cleanly into the existing compute slack" (`Grokking Megakernels.md`
  ~line 2057, ~line 3588). The model treats unpack FLOPs as free BUT flags when
  predicted compute time approaches predicted bandwidth time, so we never silently
  assume slack that isn't there.
- **Decode parallelism comes from splitting the KV traversal, not Q-blocks**
  ([[vLLM Triton Attention Backend Deep Dive]]): "Q blocks help prefill but not
  decode (single query token). For decode, parallelism is added by splitting the
  KV cache traversal across multiple kernel instances ... reduced in a second
  kernel" (Triton has no global barrier). The same source: vLLM's Triton paged
  attention hit **100.7% of FA3 on H100 long decode** with ~800 lines vs FA3's
  ~70,000 — FA-parity for the decode kernel shape is achievable in Triton.
- **Autotune recompile trap:** keying/specializing on context-length triggers a
  fresh compile per shape — AWS saw a 10× TTFT regression
  ([[GPU Kernel Auto-tuning]]). Use `do_not_specialize` on the context-length arg.
- Online-softmax is mathematically exact ([[How does the online softmax trick
  enable tiled attention computation]]); the existing `online_softmax_update` is
  the reference primitive.

## Scope, sequencing, and "done"

**Bar:** deployment-grade — parity with a strong baseline, split-KV decode
parallelism, autotune, CUDA-graph friendliness.

**Baseline:** fp16 flash-SDPA decode path (the deployment default, no compression)
+ our own tuned PyTorch chunked path. No new serving-engine dependency. Claim
shape: *"matches fp16 flash-SDPA decode latency within X% at 128k while resident
memory is ~C× smaller; the latency win over fp16 is KV-fraction-bounded and
materializes only past ~Nk context, as predicted."*

**Kernel scope:** k2b recipe only (the shipped recipe). RTN-style unpack first to
get the skeleton + parallelism + autotune working bit-exact, THEN k2b's real
lowrank-key / Lloyd-value unpack. RoPE applied **in-kernel** from a resident
cos/sin table (keys stored pre-RoPE — post-RoPE storage would smear the low-rank
structure k2b exploits; CLAUDE.md pitfall).

**Sequencing (Approach A — prediction-gated, staged):**

0. **Predict** (gate). Extend `kv_memory.py` with the decode-latency model →
   predicted speedup-vs-context curve per arm + crossover context. Sets go/no-go,
   the target number the kernel chases, and the split-KV decision (KV-read bytes
   per step). Cheap, no GPU.
1. **PyTorch decode-loop opts.** GQA grouped contraction (no per-block
   `repeat_interleave`) + grow-time RoPE cast, in `chunked_attention.py`. Gives a
   tuned baseline AND a faster-but-bit-exact oracle. Bit-exact vs the naive oracle
   before/after each opt.
2. **Triton kernel, staged (3a–3d below).**
3. **Measure & write the honest claim.**

**Triton sub-stages** (`src/bmx/cache/triton_dequant_attention.py`, new):
- **3a — RTN single-block, decode-only.** Unpack int codes + per-group scales
  in-register, RoPE in-kernel from resident cos/sin, online-softmax, serial over
  blocks. Correctness-first skeleton.
- **3b — split-KV decode parallelism + autotune.** Split the KV traversal across
  programs → partial (acc, m, lse) → second reduction kernel merges partial LSEs.
  `@triton.autotune` over block/num_warps/num_stages/split, `do_not_specialize` on
  context-length. **DeepWiki checkpoint:** ask FlashInfer/SGLang for the split-KV
  grid layout + partial-LSE merge + split-count heuristic.
- **3c — k2b real unpack.** Replace RTN with lowrank+per-channel keys (frozen
  `Us @ V_frozenᵀ` + per-channel) and Lloyd-codebook values; K/V may differ.
- **3d — CUDA-graph capture.** Persistent-kernel pattern (fixed grid, work read
  from memory) so a captured graph replays as S grows. **DeepWiki checkpoint:**
  how vLLM/SGLang keep variable-length decode CUDA-graph-compatible.

Prefill is unchanged — stays flash-SDPA (Phases 1+2 settled this). The kernel is
decode-only.

**Done =** predicted-vs-measured speedup curve documented; the honest, scoped
claim written (parity-or-not reported either way — kill-or-confirm); Phase-3
status in CLAUDE.md/memory moved from "open/gated" to its measured outcome.

## The correctness spine (gated at every stage)

```
naive_dense_attention  ──diff──>  [variant]  ──logit parity──>  fp16 SDPA generation
  (oracle: dequant-all,            (PyTorch-tuned,               (end-to-end truth)
   one softmax, no chunk)           then Triton 3a–3d)
```

Two anchors, **both required per stage, both before any latency is recorded**:

1. **Kernel-level:** `attention_diff(variant, naive_dense_attention(...))` →
   `max_abs` under an fp16 tolerance derived from the *current measured*
   chunked-vs-oracle drift (not an arbitrary number; derivation goes in the plan).
2. **End-to-end:** logit parity vs fp16 SDPA on the two-block cached-prefill path
   (extend the `c4098d4` regression test) + first-token equality vs dense
   `StreamingQuantizedCache`.

Why both: the Phases-1+2 prefill-mask bug had a kernel diff of ~8e-5 (looked
perfect) yet produced garbage generation — the defect was in masking/integration,
not the contraction. Kernel-diff is necessary but NOT sufficient.

**Drift-vs-speedup ledger** (structural enforcement): the experiment writes a
parquet with one row per (variant, context):
`latency_ms, max_abs_vs_oracle, max_rel_vs_oracle, logit_parity_pass,
predicted_speedup, measured_speedup`. The writeup reads this parquet, so a
measured speedup is physically adjacent to its correctness columns and its
prediction. Figures never refit (repo convention).

## Fail-loud / fallback-ok taxonomy

Fallback is allowed ONLY for capability absence — NEVER for correctness or
unexpected errors. Dispatch checks capability explicitly (is CUDA available, is
the kernel compiled); no `try/except` catch-all that swallows a real failure.

| Situation | Behavior |
|---|---|
| Triton/CUDA absent (local AMD box) | **Transparent fallback** to chunked PyTorch, logged once. The *ok* fallback. |
| Kernel output drifts past oracle tolerance | **Fail loud** (raise/assert). Never silently fall back — a correctness regression must not hide as "fallback worked." |
| Kernel errors / compiles wrong when present | **Fail loud** in tests and harness. A kernel that should run and doesn't is a defect. |
| Unsupported shape/group config | **Fail loud** with a boundary shape assert (repo "fail fast, no silent coercion"). |

## Testing & verification

**Local (AMD, no CUDA) — always runs, never goes dark:**
- Chunked PyTorch vs `naive_dense_attention` oracle (existing bit-exact floor).
- The `c4098d4` two-block cached-prefill logit-parity regression.
- PyTorch decode-loop opts (stage 1): diff vs oracle before AND after — proves the
  speedup didn't change the output. Pure PyTorch, AMD-runnable.
- Prediction-model unit tests: known-input → known speedup; the
  compute-approaching-bandwidth honesty flag fires when it should.

**VM (CUDA) — the Triton tests:**
- `@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel —
  VM/CUDA only")` — skip LOUD locally (skip, not xfail: not-runnable-here, not
  expected-broken). A green local run must not be misread as "Triton verified."
- Per sub-stage (3a–3d): bit-exact vs oracle AND logit parity vs fp16 SDPA. **Fail
  loud** on drift.
- **"Did it actually run" guard:** on a CUDA box, assert the GPU-required tests
  were collected-and-run, not all-skipped — an all-skipped green run on the VM that
  is supposed to test the kernel is itself a failure.

`xfail` is reserved for genuinely-expected failures (like the existing
`test_cold_start_recovery`), never to quiet a GPU-absent test.

## Components

1. `src/bmx/bench/kv_memory.py` — extend with the decode-latency model (reuses the
   measured `weights_bytes`/`act_bytes` constants; HBM bandwidth sourced per-GPU at
   measure-time, not hardcoded). The prediction gate.
2. `src/bmx/cache/chunked_attention.py` — the two PyTorch decode-loop opts, kept
   bit-exact vs oracle.
3. `src/bmx/cache/triton_dequant_attention.py` (new) — the staged Triton kernel.
   `PackedStreamingLayer.attend` gains a decode dispatch: Triton when available +
   CUDA, else the chunked PyTorch path (the bit-exact reference).
4. An experiment script (thin tyro CLI) that runs the drift-vs-speedup ledger on
   the VM and writes the parquet to `results/<exp>/<run-id>/`.
5. Results doc (`docs/2026-06-2x-triton-decode-results.md`) reading the parquet.

## Deliverables

- Code (1–4 above).
- Results doc: predicted-vs-measured speedup curve, drift-vs-speedup ledger,
  crossover context, the honestly-scoped claim (parity-or-not, kill-or-confirm).
- CLAUDE.md / memory: Phase-3 status → measured outcome.

## Constraints

- Triton authoring is VM-only (local AMD has no CUDA; local stays green via
  skipif). `triton` 3.7.0 is **already** in `uv.lock` as a transitive dep of
  `torch` with `marker = "sys_platform == 'linux'"` (no Windows wheels) — so it is
  installed on the Linux VM, absent on the Windows AMD box, and **no `pyproject`
  change is needed**. The module uses an explicit `try: import triton` capability
  guard; the plan verifies `uv run python -c "import triton"` succeeds on the VM
  and fails cleanly on Windows.
- Exact tolerances derived from current measured chunked-vs-oracle drift, not
  picked arbitrarily (derivation in the plan).
- No `git commit` without explicit approval. Before any commit: ruff format → ruff
  check → pytest -q clean.
- GPU-authoritative work runs on the rented NVIDIA VM via git transport
  (push → pull → run → commit parquet back).

## Plan-level notes (carried from spec review, resolved at plan time)

- **Exact `max_abs` / `max_rel` tolerances** — derive from measured
  chunked-vs-oracle drift at the start of stage 1, not picked arbitrarily.
- **Stage 1 expected gain** — the GQA grouped contraction avoids
  `repeat_interleave` materializing an `(n_q_heads, blk, d)` copy (n_q_groups× the
  block memory; n_q_groups=4 for Llama-3.1-8B). Small per-block (~2 MB at
  block=128) but × (S/block) blocks × L layers; quantify in the plan.
- **Split-KV split-count heuristic** — pre-load the vLLM-style
  `num_splits = max(1, ceil(S / (block · num_SMs)))` so stage 3b doesn't start from
  scratch; refine via the DeepWiki/FlashInfer checkpoint.
- **CUDA-graph seqlen via device pointer (hard requirement, stage 3d)** — the
  growing cache means KV-read length changes per step; pass sequence length as a
  **device tensor pointer**, NOT a Python int (a Python int kernel arg triggers
  per-step recompile). The graph captures the pointer-read and replays correctly as
  S grows. This is what vLLM does.
- **`do_not_specialize` on context-length (hard rule, stage 3b)** — NOT a note:
  `@triton.autotune` with context-length as a specialized arg recompiles per S,
  which at growing-decode is a recompile per step (the AWS 10× TTFT regression).
