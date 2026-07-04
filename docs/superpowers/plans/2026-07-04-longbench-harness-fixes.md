# LongBench Harness Fixes — for the truncated parity rerun

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Per-task auto-commit pre-authorized (user 2026-07-04, same working agreement as the two
> cleanup waves): commit with each task's exact proposed message after the gate; explicit
> paths only, never `git add -A`; NO AI attribution. Run ledger: `.superpowers/sdd/progress.md`.

> **Trigger:** the 54h+ un-truncated LongBench run (`results/k3_longbench/20260702-164100-46b9579`,
> pid 10858 on the GH200) is ~82% done (py-spy: arm `kivi`, task `qasper` — 4.1/5 arms).
> Let it finish; its output is the **no-truncation variant**. This plan fixes the harness so a
> **truncated parity rerun** (the version actually comparable to TurboQuant Table 1) is
> observable, resumable, and scientifically valid. **Tasks 1–4 are local code — dispatch them
> NOW, in parallel with the VM run.** Only the rerun itself (and Task 5's isolated profiling)
> waits for the current run to finish. Do NOT touch the VM while pid 10858 lives — every
> completed row exists only in that process's memory until the final parquet write.

**Why this exists (the three findings):**

1. **Truncation is a *comparability* issue, not just speed (the important one).** TurboQuant's
   Table 1 was produced on **middle-truncated** inputs (~31.5k tokens for Llama-3.1-8B). Our run
   does **not** truncate. So our fp16 anchor row runs on *different effective inputs* than theirs
   — the Task 10 Step 4 anchor gate (fp16 Avg≈50.06, turboquant_mse Code≈46.28) can fail for
   **input-policy** reasons unrelated to our measurement path, spuriously unlicensing the
   transitive-baseline argument. For the parity table, matching their truncation is **required**,
   not optional. Keep no-truncation as an explicit named variant (it's a legitimate "we compress
   the full context" result — just not the anchor-comparable one).

2. **Per-token full-slab rebuild in the dense streaming cache — and it is provably a NO-OP.**
   `streaming.py::update()`'s no-flush branch does `torch.cat([self._q_prefix_k, k_tail])` for K
   and V, every layer, every decoded token. Trace the invariant: at step t we set
   `self.keys = cat(prefix, tail)`; at step t+1 `super().update()` concatenates the new token onto
   that SAME tensor, so the `keys` the parent returns is already `[prefix, tail, new_token]` — our
   rebuild slices the tail back out and re-cats it onto the prefix, reconstructing the identical
   tensor. The fix is therefore *deletion*, not incremental-append machinery (see Task 2).
   Measured cost context: **~70 ms/token, roughly flat with context** (70ms@2k, 67ms@8k) —
   consistent with constant Python/allocation overhead dominating (at ≤31.5k the raw slab copy is
   sub-millisecond at HBM speeds), NOT quadratic blowup. **CAVEAT: these numbers were measured
   while pid 10858 occupied ~1/3 of the GPU — treat every figure as provisional until Task 5's
   isolated re-profile.** Rough budget: 2.5M decoded tokens × 70ms ≈ 49h, which matches the
   observed wall time; the rebuild is plausibly a large share of the 70ms (2 tensors × 32 layers
   of O(S) cat + alloc per token), NOT the codec (quantization fires once per 128-token page
   flush, and the dense path stores the prefix already-dequantized — there is no per-token codec
   work on this path).

3. **Zero progress output + zero checkpointing.** `write_metrics` fires **once** at the very end;
   nothing prints between the startup line and the final DataFrame. A 50h run yields **zero rows**
   if anything hiccups — no resume, no partial salvage. From outside, "50h of correct slow
   progress" and "hung at hour 2" are indistinguishable (only py-spy on frame locals told us the
   arm/task). This is the defect that made the whole ordeal opaque.

**Global constraints** (from CLAUDE.md — every task): NEVER commit without explicit approval;
before any commit `uv run ruff format . && uv run ruff check . && uv run pytest -q` clean; deps via
`uv add` only; Bash (git bash) not PowerShell; comparisons align on total bits; rank codecs on
IP/logit distortion. VM transport is git. Relaunch pulls the current branch head first (the 33
cleanup commits are parity-preserving; generation speed unchanged, but paper artifacts should come
from final code).

---

## Task 1: [FIGURE/CODE] Middle-truncation flag matching LongBench's official scheme

**Files:** `src/bmx/cache/longbench.py` (add truncation to `build_longbench_prompt`), 
`experiments/k3_longbench.py` (add `--max-prompt-tokens` / `--truncate` to Config), 
`tests/test_longbench.py` / `tests/test_longbench_scorers.py`.

**What LongBench does** (from its `pred.py`): if `len(tokenized_prompt) > max_length`, keep the
**first and last** `max_length//2` tokens (middle-truncation), then decode back to text and only
THEN apply the chat wrapper — truncation operates on the task prompt, not the chat-wrapped ids.

- [ ] **Step 0: Pin the constant and the ordering against TURBOQUANT, not generic LongBench.**
  The anchor argument requires matching *TurboQuant's* input policy. LongBench's own
  `model2maxlen` predates Llama-3.1 (31500 is NOT verified for this model); TurboQuant ran a
  single A100-80GB, which implies truncation, but the exact budget must come from their paper
  (§4 / appendix) or released code. Deliverable: the verified number + a one-line citation in the
  flag's docstring. If it genuinely cannot be pinned, default to 31500 with the docstring saying
  "LongBench-convention fallback; TurboQuant's exact budget unverified" — honest label, user
  decides at rerun time. ALSO verify our `build_longbench_prompt` can truncate at the right
  layer: LongBench truncates the task prompt then chat-wraps; if our builder fuses template +
  chat-wrap + tokenize in one step, the truncation must be inserted between them, not applied to
  the final ids.

- [ ] **Step 1: Failing test.** `test_middle_truncation_keeps_head_and_tail`: build a prompt whose
  tokenization exceeds `max_prompt_tokens`, assert the result is exactly
  `head[:n//2] + tail[-n//2:]` (ids), length `== max_prompt_tokens`, and that a short prompt is
  returned unchanged.
- [ ] **Step 2:** Run, verify fail.
- [ ] **Step 3: Implement.** Add `max_prompt_tokens: int | None` param to `build_longbench_prompt`;
  when set and exceeded, middle-truncate the input ids. Thread `--max-prompt-tokens` through
  `k3_longbench.Config` (default **31500** for the parity run; `None` = no-truncation variant).
  Record the value in the emitted rows (a `max_prompt_tokens` column) so the parquet is
  self-describing.
- [ ] **Step 4:** Run, verify pass. Ruff + full suite.
- [ ] **Step 5: Commit.** `feat(longbench): middle-truncation flag (LongBench-official; parity-comparable inputs)`. STOP for approval.

**Acceptance:** the parity run's inputs match LongBench's truncation policy exactly; the
no-truncation variant is still reachable via `--max-prompt-tokens None` and clearly labeled in the
parquet.

---

## Task 2: [FIGURE/CODE] Incremental slab maintenance in the dense streaming cache

**Files:** `src/bmx/cache/streaming.py` (the no-flush branch of `update()`), 
`tests/test_streaming_cache.py`.

**Change — SIMPLER THAN ORIGINALLY SKETCHED (identity proof, not incremental machinery):** in the
`new_S_q <= self._committed_S_q` branch (no new page to flush), the rebuild is provably the
identity. Invariant: after every `update()`, `self.keys` holds `[quantized_prefix | fp16_tail]`
assembled. On the next decode step, `super().update()` (DynamicLayer) concatenates the new token
onto that stored tensor and returns it — so `keys` ALREADY equals `[prefix | tail | new_token]`.
The branch's `k_tail = keys[..., committed:, :]` + `cat(prefix, k_tail)` reconstructs the very
tensor it sliced. **Fix: replace the rebuild with `self.keys, self.values = keys, values`** (keep
the blended-bpe recompute exactly as is). Bit-identical by construction: fp16→fp16 `.to()` is a
no-op and the cat operands are the same storage. Flush boundaries still take the flush branch
(prefix grows there — reassembly stays, once per 128 tokens). NO persistent-buffer or in-place
append machinery — that would fight the parent's own concat and is not needed.

- [ ] **Step 1: Pin numerics.** Confirm the existing streaming parity test
  (`test_streaming_cache.py`) asserts token-identical output vs the reference; if it only checks a
  single forward, add a **multi-token greedy-decode** parity assertion first (old path vs itself
  as baseline, ≥64 decoded tokens crossing at least one flush boundary so both branches execute).
  This is the guard the change rides on.
- [ ] **Step 2:** Run, verify current code passes it (baseline).
- [ ] **Step 3: Implement** the deletion-fix above. VERIFY THE INVARIANT FIRST from the code: check
  nothing mutates `self.keys` between updates in a way that breaks prefix-prefix identity (the
  attend paths read it; the flush branch reassigns it — both fine). If you find a real mutation
  path, STOP and report BLOCKED with the evidence — do not fall back to the incremental-append
  design without controller sign-off.
- [ ] **Step 4:** Run parity test → identical. Add a micro-benchmark note (ms/token before/after at
  8k, CPU is fine for the ratio) to the commit body.
- [ ] **Step 5:** Ruff + full suite (current local baseline 272/8/1).
- [ ] **Step 6: Commit.** `perf(streaming): drop per-token slab rebuild in the no-flush branch (provably identity; numerics pinned)`.

**Acceptance:** multi-token decode is bit-identical; per-token cost drops; no recorded metric
changes.

**Caveat (corrected):** the true per-token cost split is UNKNOWN until Task 5's isolated profile —
the rebuild is plausibly a large share (2 tensors × 32 layers of O(S) cat/alloc per token), and
there is NO per-token codec cost on this path (quantization is per-page-flush; the prefix is
stored dequantized). Do not claim a rerun ETA from contaminated numbers; measure after this lands.

---

## Task 3: [FIGURE/CODE] Per-sample progress + parquet checkpointing with resume

**Files:** `experiments/k3_longbench.py` (the arm×task×sample loop), `src/bmx/artifacts.py`
(if a checkpoint-append helper belongs there), `tests/test_k3_longbench_experiment.py`.

**Change:** three things so a long run is never opaque or lost:
- **Progress lines:** print `[arm i/N task j/M sample k/K] score=... elapsed=...` per sample (or
  per 10), flushed. Makes py-spy unnecessary for position.
- **Checkpoint:** after each (arm, task) completes, append its rows to
  `<run_dir>/metrics_partial.parquet` (or a per-task shard). The final `write_metrics` still writes
  the canonical `metrics.parquet`, but partials survive a crash.
- **Resume:** on startup, if `metrics_partial.parquet` exists in a `--resume <run_dir>`, skip
  (arm, task) pairs already present and continue. Idempotent on (arm, task).

Implementation constraints (from the repo's plot-discipline pitfalls):
- Shards live UNDER the same run-id dir (`<run_dir>/partial/<arm>__<task>.parquet`), so
  `newest_run_with`-style globbing can never double-count a partial as a separate run; the
  canonical `metrics.parquet` at the end is the only file plot code reads.
- **Resume must assert identity:** on `--resume <run_dir>`, compare the stored `config.json` and
  git SHA (from the run dir's env record) against the resuming process's; mismatch = hard error,
  not a warning. Mixed-code/mixed-config rows are exactly the double-count pitfall CLAUDE.md
  warns about.
- Parquet has no append — write one shard file per completed (arm, task), concat at the end.

- [ ] **Step 1: Failing test.** `test_longbench_checkpoint_resume`: run the tiny offline path for 2
  arms × 2 tasks, kill after 1 task (simulate by calling the loop with an injected stop), assert
  the completed (arm, task) shard exists; re-run with `--resume`, assert it skips the done pair,
  completes the rest, and the final parquet has all 4 (arm, task) groups exactly once. Add
  `test_longbench_resume_rejects_config_mismatch` (resume with a changed config field → hard
  error).
- [ ] **Step 2:** Run, verify fail.
- [ ] **Step 3: Implement** progress prints + per-(arm,task) shard write + `--resume` skip logic +
  the identity assert.
- [ ] **Step 4:** Run, verify pass. Ruff + full suite.
- [ ] **Step 5: Commit.** `feat(longbench): per-sample progress + parquet checkpoint/resume (no more opaque 50h runs)`.

**Acceptance:** a killed run loses at most one (arm, task)'s worth of work; a re-run with
`--resume` completes the rest; live position is readable from stdout without py-spy; resume
refuses mixed code/config.

**Generalize cheaply:** `k3_niah.py` (publication Task 9's sweep) has the identical opacity
failure mode. If the progress/checkpoint helper lands as a small shared utility (e.g. in
`experiments/_common.py`), wiring NIAH's (arm, length) loop into it is ~20 minutes. Do it in this
task if the helper factored cleanly; otherwise note it as an explicit follow-up in the commit
body — do not silently skip it.

---

## Task 4: [KILL-OR-CONFIRM] The kivi arm — diagnose the near-zero scores BEFORE the rerun

**Files:** read `src/bmx/quant/rtn.py`, `src/bmx/cache/recipes.py` (the kivi spec pair);
deliverable is a short diagnosis doc `docs/2026-07-04-kivi-arm-diagnosis.md` + a launch decision.

**The evidence:** the live run shows kivi scoring ~0 across tasks (narrativeqa 0.007; qasper
scores almost all literal zeros) while TurboQuant's published KIVI averages ~47. **Candidate
diagnosis:** our "kivi" arm is *symmetric* groupwise RTN (`rtn_channel`/`rtn_token`), while real
KIVI uses *asymmetric* quantization with per-group zero-points — at 2 bits, symmetric has ~3
usable levels and is known to collapse. If that holds, the row as-labeled would strawman the
baseline and cannot be printed.

- [ ] **Step 1:** Confirm or refute from the code: is `rtn_quantize` symmetric (scale-only, no
  zero-point)? Compare against KIVI's actual scheme (paper §3 / their repo). Cite lines.
- [ ] **Step 2:** Cheap empirical corroboration (CPU, tiny model from `tests/factories.py`): 2-bit
  symmetric RTN reconstruction error vs 2-bit asymmetric (a 10-line throwaway using min/max
  zero-point) on a real-ish activation distribution — show the gap. This is corroboration, not
  proof; the code reading is the verdict.
- [ ] **Step 3:** Write the diagnosis doc + recommendation. DEFAULT DECISION (controller-approved,
  user can override at launch): **drop kivi from the truncated parity rerun** and cite
  TurboQuant's published KIVI row in T1 (the locked transitive-baseline strategy explicitly
  licenses this); relabel any use of our arm as `rtn2` where it appears. Implementing faithful
  asymmetric KIVI (new zero-point bpe accounting: +16/group more metadata bits per the honest
  rule) is post-paper work unless the user asks for it now.
- [ ] **Step 4: Commit** the doc: `docs: kivi arm diagnosis — symmetric-RTN strawman vs real KIVI (kill-or-confirm + rerun decision)`.

**Acceptance:** the rerun launch command's arm list is justified line-by-line; nobody discovers
the kivi question AFTER burning GPU-hours on it.

---

## Task 5: [VM, after pid 10858 completes] Isolated profile — replace every contaminated number

**Do not start while the current run lives.** One sample, GPU otherwise idle: prefill and decode
timed separately, per-token decode breakdown (parent cat / our branch / attention / HF overhead —
`torch.profiler` or coarse timers), before AND after Task 2's fix is pulled. Deliverable: a
ms/token table in `docs/2026-07-04-longbench-perf-profile.md` and a defensible rerun ETA. Every
number from the 07-02..07-04 window (70ms flat, the kernel A/B) is provisional until this exists.

---

## Re-sequencing the rerun itself (not a code task — a run-order rule)

Once the harness fixes land, **do not launch the full matrix blind.** Order:

1. **Anchor rows first, truncated:** `--arms fp16 turboquant_mse --categories code
   --max-prompt-tokens <Task-1-Step-0 value>`. Cheap. Read the Task 10 Step 4 gate: fp16 Code and
   turboquant_mse Code must reproduce TurboQuant's Table-1 Code values. **If this passes**,
   transitivity is licensed and the full sweep is worth it. **If it fails on truncated inputs**,
   the problem is the measurement path (not input policy) and must be root-caused before burning
   the full matrix.
2. **Then the full truncated sweep with the CORRECT arm list:**
   `fp16 k2b k2b_k2r8 turboquant_mse turboquant_prod` **+ kivi only per Task 4's decision**.
   NOTE: the current 54h run omitted `k2b_k2r8` — the matched-compression arm that makes C1's
   comparison at-equal-bits. The rerun without it repeats the disqualifying gap. Six categories,
   progress + checkpointing on.
3. **Arm-parallelism across GPUs is the DEFAULT launch mode, not an option** (N arms → ~1/N wall
   time; each arm is an independent process with its own `--arms <one>` + shared run-id shards, so
   Task 3's per-(arm,task) checkpointing composes with it for free). Truncation alone does NOT
   shrink the rerun much: decode tokens don't truncate — if decode really is ~70ms/token, a serial
   rerun is still ~2 days. Parallel arms + Task 2's fix are the ETA levers; Task 5's profile turns
   that into a real number before launch.

## Open, separate (NOT in this plan)

- **Triton kernel through the generation path.** The fused decode kernel benched 2624×/322× vs
  chunked in `k3_triton_decode`'s dedicated harness (pre-built stacks), but measured *slower*
  through `generate_through_cache → PackedStreamingCache` in one contaminated A/B. Whether the
  kernel is fast-but-strangled-by-the-wrapper or unsuited to the generation loop is the **C4
  writeup's** real open question. Investigate with the GPU free, timing only the decode loop
  (not prefill), run in isolation. This is the actual path to making compressed generation fast —
  Tasks 1–3 make the *reference* path tolerable, they don't accelerate decode itself.
