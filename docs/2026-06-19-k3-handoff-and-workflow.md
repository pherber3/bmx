# K3 handoff + workflow notes (2026-06-19)

For the next session. The K3 *findings* live in
`docs/2026-06-19-k3-streaming-cache-results.md`; this doc is **state + how we worked**,
so a fresh chat starts warm.

## Branch state (as of pause)

- Branch `k3-live-streaming-cache`, 22 commits off `main`, **not merged, not pushed**.
  Tree clean, `155 passed, 1 xfailed` (the intentional `test_cold_start_recovery`).
- Solo repo → merge straight when ready: `git checkout main && git merge
  k3-live-streaming-cache`. No PR needed.
- Final whole-branch re-review verdict: **MERGE-READY** with three deferred caveats
  (kernel for process-RSS; SOTA VM run for headline numbers; blended-bpe is
  conservative at short context).
- Progress ledger with per-task commit ranges + every Minor finding:
  `.git/sdd/progress.md`. Task briefs/reports: `.git/sdd/task-*.{brief,report}.md`.

## What K3 delivered (one paragraph)

`StreamingQuantized{Layer,Cache}` (subclass transformers 5.11 `DynamicLayer`/`Cache`,
mirrors HF `QuantizedCache`) — quantize-on-append KV cache that streams token-by-token
under real `generate()`. **Write-once quantized storage** (each token quantized once
from its pristine source) was the load-bearing fix; it killed a real bug where the K2b
value codec (`turboquant_mse`) compounded under naive per-step re-quantization (V norm
98×). Frozen pre-RoPE subspace, fp16 residual window, honest packed bpe, all arms on one
fair code path. Quality holds (1.001× fp16 tbt-ppl).

## Open work, in priority order (engineering, not science)

1. **Authoritative SOTA-model VM run.** The experiment (`experiments/k3_live_generation.py`)
   is model-agnostic — `--model-name` is the only knob. Run on Llama-3.2-1B (or a
   Qwen3/Gemma-class SOTA model) on the NVIDIA VM with real wikitext + planted needle
   (already wired). Produces the headline K2b-vs-TurboQuant-vs-KIVI numbers. This is the
   first thing to do — the branch makes it *trustworthy*; it doesn't *execute* it (no
   CUDA on the 7900 XTX).
2. **Fused dequant-attention kernel** for the literal process-RSS 5×. `reconstruct_layer`
   currently materializes the dequant slab for the model to read (Stage-B contract).
   **Predict the win with the Track B byte model in `src/bmx/bench/` BEFORE writing CUDA.**
3. **32k-context drift re-check** (drift measured to 2k in K2c).
4. Minor nits recorded for the writeup, not blockers: `memory_report` compression is
   blended-bpe (conservative short-context); proxy `retrieved` vs real `needle_real` are
   two different signals; true uint8 packed storage is part of the kernel work.

## The workflow that worked (reuse this)

This was brainstorm → spec → plan → subagent-driven execution with reviews → simplify →
whole-branch review. The process lessons, in rough order of value:

1. **Consult the personal-brain (`personal-brain` skill / `mcp__wiki__*`) and DeepWiki
   BEFORE architecting, not after.** The brain changed the design twice this round:
   (a) it pointed at HF `QuantizedCache` which dissolved a memory-mechanism fork I was
   stuck on; (b) it reframed three separate Critical patches into ONE root fix
   (write-once = canonical KV semantics). DeepWiki on vLLM/SGLang confirmed the
   dequant-on-read contract was right. **When you hit an architecture decision, search
   the vault first — it knows more than the model's priors.**
2. **The whole-branch review is non-negotiable and must RUN the code.** Per-task reviews
   passed everything green; the cross-cutting final review (on the most capable model,
   `ml-research-reviewer` agent) found 3 Criticals the per-task reviews structurally
   could not see — including a V-explosion bug hiding under green tests (the per-task
   streaming test used an *idempotent* V codec and never checked V quality). The reviewer
   that **injected a bug to prove a test gap** (RoPE positions) was the gold standard —
   ask reviewers to verify by execution/probe, not just by reading the diff.
3. **Kill-or-confirm earns its keep.** A branch that looked "done" (10 tasks, all
   reviewed, 155 green) had a real bug. The honest-negative ethos is what surfaced it.
   Independently re-derive headline numbers from the artifact, never take them on a
   subagent's report (the controller re-ran the V-stability check by hand each time).
4. **Subagent-driven specifics that paid off:** one fresh subagent per task; brief +
   report as FILES (not pasted context — keeps the controller's context clean);
   model tiered to task (haiku for mechanical/transcription, sonnet for integration,
   `ml-research-reviewer`/most-capable for the whole-branch gate); a durable ledger
   (`.git/sdd/progress.md`) so progress survives compaction.
5. **Insert tasks mid-stream when a finding warrants it.** Task 3.5 (residual window)
   and Tasks 10–12 (the Critical fixes) were not in the original plan — they came from
   findings. Don't force a finding into the existing plan; add a task with its own gate.

## Commit convention (this repo)

Conventional prefixes (`feat:`/`fix:`/`test:`/`refactor:`/`docs:`), imperative, scoped.
**NEVER any `Co-Authored-By` or AI attribution** — checked across all 22 commits, clean.
Pre-commit gate every time: `uv run ruff format .` → `uv run ruff check .` →
`uv run pytest -q` (expect `155 passed, 1 xfailed` on this branch; `129 passed, 1 xfailed`
on main before K3). User is fine with per-task auto-commits *as long as* they follow
precedent and carry no attribution — confirmed this round.

## Fastest way to resume in a new chat

Point the new session at: this doc, `docs/2026-06-19-k3-streaming-cache-results.md`
(findings), and `.git/sdd/progress.md` (ledger). Then the obvious next task is the
SOTA VM run (#1 above) — it needs no new code, just the VM and `--model-name`.
