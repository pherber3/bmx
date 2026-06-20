# LongBench Code eval under KV compression — design (2026-06-20)

Reproduce TurboQuant's **coding signal** — the `Code` column of their LongBench Table 1 —
by running `lcc` + `repobench-p` end-to-end through the **same `StreamingQuantizedCache`
path** as the NIAH metric, scored by LongBench's official `code_sim` (edit-similarity).
It answers what NIAH and perplexity cannot: *does the model still write correct code
completions when its long context is 2-bit-compressed?*

This is the **coding half** of next-direction #0 (the other half, NIAH retrieval, is built —
`docs/superpowers/specs/2026-06-20-niah-retrieval-metric-design.md`). Both run alongside on
the VM. We are treating the TurboQuant paper as the **reference protocol** and reproducing
its eval families on our compression arms through our honest cache path.

## Why this, and why LongBench Code specifically

Perplexity says the distribution held; NIAH says a planted fact is retrievable; neither says
the model still *writes correct code* from a long compressed context. That is exactly what
local-model coding users care about. Critically, **the paper got its coding signal from the
`Code` subset of LongBench-V1, not from HumanEval pass@1** — LongBench Code is graded
(edit-similarity, built-in headroom), execution-free (no sandbox, no pass@1 variance), and
runs through the identical prefill-then-generate cache path we already built for NIAH. So we
match the paper: LongBench Code, not HumanEval.

## Anchor: TurboQuant Table 1 (LongBench-V1, Llama-3.1-8B-Instruct)

Per-category scores at a KV-size budget; the `Code` column is our target
(full-precision ≈ 46.28, KIVI@2 ≈ 46.61, TurboQuant ≈ 45.7–46.9). Tasks in the Code
category: **`lcc`** (long code completion) and **`repobench-p`** (repo-level completion),
each scored by LongBench's **`code_sim`** = edit-similarity (0–100). All methods compared at
a fixed memory ratio (paper uses ~0.25 / 4×).

## Decisions (locked in brainstorm)

1. **Scope = Code subset** (`lcc`, `repobench-p`), built on a **generic per-task runner** so
   other LongBench categories *could* be added later as config, not a rewrite. The full
   Table 1 is explicitly out of scope (mostly non-coding work).
2. **Compression regime = whole-context prefill then generate** through
   `StreamingQuantizedCache` — the entire long context is quantized on-append (minus the
   fp16 recent window, negligible at LongBench code lengths), and the generated code attends
   to the compressed cache. The SAME path `niah_recall_generate` uses → one fair code path
   across both metrics. Per-arm honest measured compression (`memory_report`), never a pinned
   ratio; the paper's 4× / fp16 score is a **reference line**, not a target.
3. **Scorer = LongBench's official `code_sim`, ported faithfully, using `fuzzywuzzy`** (the
   exact library LongBench uses — wraps `python-Levenshtein`) so the number is provably
   identical to the published scorer. `rapidfuzz` is a documented FALLBACK only if the
   `python-Levenshtein` wheel will not install via `uv` on this box (Task 1 verifies; if it
   falls back, the substitution is logged and the edge-case-tie caveat noted).
4. **Prompt templates + `max_gen` = LongBench's own, verbatim.** Sourced from LongBench's
   `dataset2prompt` / `dataset2maxlen` config, with the model-specific `build_chat` wrapper
   LongBench applies for Llama-Instruct. Task 0 locks these against the authoritative source.
5. **Full sets by default** (`n_samples=None` ⇒ all 500 `lcc` + 500 `repobench-p`) for direct
   comparability to Table 1. `--n-samples` caps for a fast look; when capped, the run is
   logged as **"subsampled, not comparable to Table 1"** (honest, no silent truncation).
6. **Run split (the NIAH/K3 pattern):** offline CI gates *mechanism* on `tiny_llama`
   (random weights → code_sim meaningless, like NIAH recall); the headline quality numbers
   are a VM `--model-name` run (no CUDA locally). Headroom guard applied at analysis time,
   identical to NIAH.

## Architecture — shared generate path, LongBench-specific scorer + loader

Maximize reuse of the NIAH machinery; the only LongBench-specific code is the scorer, the
dataset loader, and the task registry.

- **Shared refactor (NIAH + LongBench), `src/bmx/cache/niah.py`:** extract
  `generate_through_cache(model, tokenizer, prompt_ids, n_prefill, k_spec, v_spec, max_new_tokens) -> str`
  — prefill into `StreamingQuantizedCache`, `generate()` the continuation (greedy), decode
  the new tokens. `niah_recall_generate` becomes `generate_through_cache(...)` +
  `rouge1_recall(...)`. **One fair path, literally one function**; the double-prefill fix
  (continuation-only, decode `out[0, L-n_prefill:]`) lives in one place for both metrics.
  This is a contained, justified refactor: LongBench is the second consumer of that code.
- **`src/bmx/cache/longbench.py`** — LongBench-specific:
  - `code_sim(prediction: str, ground_truth: str) -> float` — faithful port of LongBench's
    official scorer (post-process: strip comment/blank lines per LongBench's rule, then
    `fuzz.ratio`, 0–100). Pure function, CI-testable, no model.
  - `load_longbench_task(task: str, n_samples: int | None) -> list[dict]` — lazy
    `load_dataset("THUDM/LongBench", task)`; returns items carrying the prompt-template-applied
    input + ground truth + `max_gen`. VM-only (CI never downloads).
  - `longbench_code_score(model, tokenizer, item, n_prefill, k_spec, v_spec) -> float` —
    build the prompt, `generate_through_cache(...)`, `code_sim` vs ground truth.
  - `LONGBENCH_TASKS = {task: {prompt_template, max_gen, scorer}}` for `lcc`/`repobench-p`,
    sourced verbatim from LongBench config (Task 0). The generic seam for future categories.

## Experiment & data flow

- **`experiments/k3_longbench.py`** — thin tyro CLI, same shape as `experiments/k3_niah.py`.
  Config: `model_name` (default `meta-llama/Llama-3.1-8B-Instruct`), `arms`
  (the same `_spec_pair` arms: fp16, k2b, turboquant_mse, turboquant_prod, kivi),
  `tasks: tuple[str,...]` (default `("lcc", "repobench-p")`), `n_samples: int | None` (None =
  full sets), `n_prefill`, `rank`, `group`, `seed`. Sweeps **arms × tasks**; per (arm, task)
  records `code_sim` (mean over samples), `n_samples`, honest `bpe_k/bpe_v/compression` (via
  the same `_compression_for` calibration as NIAH — reuse it), and `score_kind` (`"code_sim"`)
  so the parquet is self-describing (the LongBench analog of NIAH's `recall_kind`). Writes
  parquet via `artifacts.create_run` / `write_metrics`.
- **Offline-test path** (model injected, no `model_name`): a tiny synthetic code-ish prompt
  through `tiny_llama` (≤64 tokens), exercising streaming + parquet schema only. No download,
  no tokenizer — airtight lazy-import guard, identical to `k3_niah`.
- **`experiments/plots/plot_k3_longbench.py`** — `make_figures(df, out_dir) -> list[Path]`:
  bar chart of mean code-sim per arm per task, each arm annotated with its measured
  compression; the fp16 (full-precision) score drawn as the reference line (the analog of
  NIAH's 4× line — here "do we hold the fp16 code-sim?"). Reads parquet, never refits;
  selects runs explicitly upstream.

## Testing & the headroom guard

Offline CI gates *mechanism*; VM produces *quality* (same split as NIAH):

- **`code_sim` pure-function tests (the CI-testable surface):** `code_sim(x, x) == 100`,
  disjoint strings → near 0, partial overlap → graded strictly between. No model.
- **`generate_through_cache` mechanism test:** streams a tiny synthetic prompt through
  `tiny_llama` without crashing (finite, indexing-correct); and **the existing NIAH tests
  still pass after the refactor** (proves the `generate_through_cache` extraction is
  behavior-preserving — re-run `tests/test_niah.py`).
- **Schema test:** `k3_longbench.run` offline emits columns
  `arm, task, code_sim, n_samples, bpe_k, bpe_v, compression, score_kind` with NO download
  (airtight lazy-import guard test).
- **Headroom guard (analysis-time, identical to NIAH):** the VM run reports **fp16 code-sim
  per task**; if fp16 is at floor (model can't complete the code) or every arm ties at
  ceiling, flag that task as **non-discriminating** rather than reporting a vacuous tie. The
  discriminating signal is where fp16 ≈ 46 and the 2-bit arms hold or drop.

## Execution model (for the implementation plan)

- **Subagent-driven development**, controller as orchestrator/reviewer (the NIAH/K3 workflow):
  one fresh subagent per task; brief + report as FILES; model tiered to task; durable ledger;
  per-task auto-commit pre-authorized (conventional prefix, NO AI attribution). A `/simplify`
  quality pass after the tasks, then the **whole-branch `ml-research-reviewer` review that
  RUNS the code** — non-negotiable (it caught two real bugs in NIAH that green per-task tests
  missed).
- **Task 0 = lock conventions against the authoritative LongBench source BEFORE code.** Read
  LongBench's `metrics.py` (`code_sim` post-processing + `fuzz.ratio`), `config/dataset2prompt`
  + `dataset2maxlen` (the `lcc`/`repobench-p` templates and gen lengths), and its `build_chat`
  model-prompt wrapper for Llama-Instruct. Record verbatim constants the later tasks cite.
  (Personal-brain vault was offline last session — re-check if connected; otherwise the
  LongBench repo is the authoritative source.)
- Independently re-derive headline numbers from the artifact; never take them on a subagent's
  report.

## Risks / explicitly deferred

- **`fuzzywuzzy` / `python-Levenshtein` install** — Task 1 verifies the wheel installs via
  `uv add` on this box; `rapidfuzz` fallback ONLY if it won't, with the substitution logged.
- **Prompt-template fidelity** — Llama-3.1-8B-Instruct needs its chat template; LongBench
  applies a model-specific `build_chat` wrapper. Task 0 records exactly what LongBench does so
  we match their prompt construction, not a guess.
- **Full-set runtime** — `n_samples=None` is 1000 long-context prefill+generate passes per arm
  × 5 arms; the VM handoff note states the expected wall-clock and that `--n-samples` exists
  for a fast first look.
- **HumanEval pass@1** — explicitly NOT this; the paper used LongBench Code, so we do too. An
  execution-based pass@1 would be a separate future spec.
- **Other LongBench categories** (QA/summ/synthetic) — out of scope; the generic task registry
  leaves the door open without committing to them now.

## Commit convention (this repo)

Conventional prefixes (`feat:`/`fix:`/`test:`/`refactor:`/`docs:`), imperative, scoped.
**NEVER any `Co-Authored-By` or AI attribution.** Pre-commit gate every time:
`uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q`. Dependencies via
`uv add` only. Per-task auto-commit pre-authorized for this plan (NIAH precedent).
