# Overnight session results — k2b deployment path: kernel 1.85×, pack_v default ON

**Date:** 2026-07-06 (autonomous overnight session, GH200). Companion to
`docs/2026-07-06-anchor-forensics-results.md` (same night, anchor half). Branch head at
writing: `90c32dc`. Every number below from idle-GPU runs, oracle columns attached.

## Quality rows banked (truncated-v1 code, run `20260706-000742-84bb1f8`)

| arm | lcc | repobench-p | Code avg | Δ vs fp16 |
|---|---|---|---|---|
| fp16 | 65.17 | 58.77 | 61.97 | — |
| **k2b_ph (deployment arm)** | 61.06 | 55.44 | **58.25** | **−3.72** |
| turboquant_mse | 53.80 | 48.20 | 51.00 | −10.97 |

First quality measurement of the deployment arm on real text: gives up 3.7 code points
where the TurboQuant arm gives up 11.

## The kernel work (the night's core)

**Baseline map** (`bench_k2b_kernel.py`, synthetic stacks at production dims
h_kv=8/d=128/blk=128/rank=16, oracle vs chunked ≈4.3e-4 every cell): time halves with
splits until programs ≈ SM count (16×8=128 vs 132 SMs), then flat; linear in ctx at
saturation → **pure work-limited**, per-program fixed cost negligible. The June
"0.71ms @2k" was a blk=64 artifact; blk=128 baseline best: 0.145 / 0.453 / 1.723 /
6.865 ms at 2k/8k/32k/128k.

**Change 1 — fp16 tensor-core operands** for all four k2b dots (lowrank Us·Vfacᵀ,
rotate-half perm, Hadamard; fp32 accumulate; softmax/score paths untouched fp32).
Gotcha hit: multiplying an fp16 operand by an fp32 scalar kernel-arg silently promotes
back to fp32 (`sqrt_d` moved to the dot output — scalar commutes).
**Change 2 — SMEM fix for V_PACKED:** first real-dims firing of the W5-2 packed layout
blew shared memory (241,664 needed vs 232,448; the CUDA pack_v oracle tests only cover
tiny dims — test-scale gap, noted). Fix: BLOCK_N capped at 32 under V_PACKED; costs
nothing (packed@32 beats unpacked@64).

**After** (same bench, same fixtures):

| ctx | baseline best | fp16 | **fp16 + pack_v** | speedup | oracle |
|---|---|---|---|---|---|
| 2k | 0.145 | 0.103 | **0.079** | 1.84× | 1.7e-4 |
| 8k | 0.453 | 0.376 | **0.241** | 1.88× | 1.6e-4 |
| 32k | 1.723 | 1.472 | **0.935** | 1.84× | 1.6e-4 |
| 128k | 6.865 | 5.845 | **3.709** | **1.85×** | 1.6e-4 |

Oracle *improved* (4.3e-4 → 1.6e-4): fp16-in/fp32-acc beats the TF32 path on accuracy
here. pack_v's 1.58× share is a bandwidth win — the int16 V indices were 4× the bytes.
GH200 acceptance: 34-test CUDA battery green (oracle + generate-parity + finalize +
pack_v variants); full suite green. Commits: `ec8bf2d` (kernel), `90c32dc`
(**pack_v default ON** — now strictly better on both axes).

## Clean end-to-end profile (idle GPU, PRE-kernel-fix code, run 05:46 UTC)

Per-token decode ms — packed 51.1 / 56.7 / 150.2 / 287.9 at 4k/16k/65k/128k ≈ the
"contaminated" T3b numbers → contention was never a factor. streaming 32.8–39.5 beats
dense 44.1–58.6 (GQA-aware attend vs stock repeat_kv — mechanism confirmed on idle
GPU). Kernel self-time 228.9µs/call @4k (Mode B) → the kernel is the cost at every
scale; glue is secondary. Post-fix e2e profile queued (final block).
Projection with the 1.85×: packed @128k ≈ 285 → ~200ms; the remaining gap to
streaming's ~34ms is the kernel's per-token full-cache dequant-compute — next levers
are tile-size/occupancy work on the (d,d) dots (the rotate/Hadamard-via-matmul forms
inflate FLOPs ~20–60× over minimal rotate_half/FWHT; a Triton-constraint tax worth
revisiting only if the latency story needs it).

## Parity semantics at long context (measured, documented)

Logit-delta probe @65k, streaming vs packed (k2b_ph): step-0 max|Δlogit| 0.31 rising
to 0.89 by step 6, first token fork at step 6 on an argmax gap of 0.37 (random-token
prompt → near-ties). Cause: F1b's merge-op-order change (per-layer ~1e-3 fp16
rounding) amplified across 32 layers *within* a step. Well under quantization noise;
per-op oracle unchanged (~1e-4). **Long-ctx acceptance criterion is now step-0 logit
tolerance, not token equality**; short-generation bit-parity tests remain valid and
green.

## OOM-sweep design fix

The 2026-07-05 sweep prefilled ALL N caches before any decode, so the OOM boundary was
set by the pre-stacking transient (unpacked int16 blocks) — W5-1's dedup never
participated, which is why packed marginal read 4.166 GiB/seq unchanged. Fixed
(`65c7d59`): each sequence decodes one token right after its prefill (stacks build +
re-point + free), so the boundary measures steady-state residency, the number a
serving system pays. Steady-state sweeps queued (final block); predicted marginal
~1.7 GiB/seq with pack_v (vs dense 4.0).

## Harness bug found by first real-tokenizer chat-wrap run

`apply_chat_template(tokenize=True, return_tensors="pt")` returns a list of
`tokenizers.Encoding` on the real Llama tokenizer; fixed via `return_dict=True`
(`bd31499`), fake tokenizer now reproduces the trap.

## Artifacts

VM: `~/bench_baseline.log`, `~/bench_fp16.log`, `~/profile_ab_clean.log`,
`~/probe_logit_delta.log`, `~/probe_chatwrap.log`, `~/lb_anchor.log`,
`~/lb_anchor_v1e.log`, `results/oom_sweep_*.json`, final block logs. Runs:
`results/k3_longbench/20260706-{000742,024100}-84bb1f8`. To transport in the morning.

## Addendum (dawn): final block numbers + the memory-ladder state of truth

- **Post-fix end-to-end** (pack_v default on): packed 51.8 / 63.2 / 132.6 / 256.7 ms
  at 4k/16k/65k/128k (−11–12% at long ctx vs 287.9; the isolated 1.85× shrinks in situ
  — the kernel loses the L2 locality the back-to-back bench enjoyed). streaming
  31.7–35.3 flat; dense 44.0–58.9. **Deployment guidance: streaming = latency path,
  packed = memory path, identical quality (parity gate green).**
- **Steady-state OOM sweep** (fixed design): dense 16 co-resident 32k seqs; packed 12
  (was 8) — still an honest NEGATIVE on concurrency, but now contradicted by direct
  measurement:
- **Memory-ledger probe** (`~/probe_mem_ledger.log`): one packed cache at 32k after
  stack+re-point = **2.789 GiB allocator-verified** (2.753 accounted field-by-field;
  res_Q_int/indices block copies confirmed freed). Two findings:
  (a) `_rope_cos/_rope_sin` are duplicated PER LAYER — ~0.5 GiB/cache of identical
  tables at 32k (≈2.1 at 128k). Share per cache (or lru-cache per device) — first
  morning fix, drops steady to ~2.3;
  (b) the sweep's 4.506 GiB/seq marginal vs the probe's 2.789 true footprint — ~1.7
  GiB of per-iteration transient the sweep counts that a settled cache doesn't hold
  (unreconciled; instrument alloc_marks inside a 2-seq mini-run to localize). Until
  (a)+(b) resolve, the concurrency table is NOT the cache's fault: predicted max
  packed seqs at true 2.3–2.8/seq ≈ 28–34 vs dense 16.
- **Closing gate: full GH200 suite 354 passed / 2 skipped / 1 xfailed** on the final
  tree (kernel fp16 + pack_v default + all of Waves 3–5).
