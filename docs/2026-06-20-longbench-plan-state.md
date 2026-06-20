# LongBench Code eval — state + VM handoff (2026-06-20)

The coding half of next-direction #0 (the retrieval half is NIAH —
`docs/2026-06-20-niah-plan-state.md`). Harness **built, reviewed, merged-ready**; what
remains is the authoritative **VM run** on a real model. Both metrics run in one VM session
(they share `generate_through_cache`).

Spec: `docs/superpowers/specs/2026-06-20-longbench-code-eval-design.md`
Plan + ledger: `docs/superpowers/plans/2026-06-20-longbench-code-eval.md`,
`.superpowers/sdd/longbench-progress.md`. Conventions: `.git/sdd/longbench-conventions.md`.

## What was built (one paragraph)

Reproduces TurboQuant's Table-1 **Code** signal: `lcc` + `repobench-p` run end-to-end through
the **same `StreamingQuantizedCache` path** as NIAH (the shared `generate_through_cache`
helper, extracted from `niah_recall_generate`), scored by LongBench's **exact** `code_sim`
edit-similarity (`fuzzywuzzy`, range **0–1**, a verbatim port of `metrics.py::code_sim_score`).
New code: `src/bmx/cache/longbench.py` (`code_sim`, task registry, dataset loader, per-item
scorer), `experiments/k3_longbench.py` (sweep arms × tasks → parquet, honest measured
compression per arm, `score_kind` column), `experiments/plots/plot_k3_longbench.py`
(code_sim per arm/task vs the fp16 reference line). All arms route through `_spec_pair` —
structurally fair, same as NIAH.

## The VM run (the open work — engineering, not science)

```bash
cd /d/Projects/bmx   # on the NVIDIA VM (no CUDA on the 7900 XTX dev box)
uv run python experiments/k3_longbench.py \
  --model-name meta-llama/Llama-3.1-8B-Instruct
# n_samples defaults to None = FULL sets (500 lcc + 500 repobench-p) → Table-1 comparable.
# --n-samples N caps for a fast first look (prints a "NOT comparable to Table 1" warning).
```

Needs the LongBench dataset (`THUDM/LongBench`, auto-downloaded via `datasets` on first run)
and the local clone at `LongBench/` for reference (gitignored; re-clone on the VM if needed).
Figures: `experiments/plots/plot_k3_longbench.py::make_figures(df, out_dir)` over the parquet.

**Runtime warning:** full sets = 1000 long-context prefill+generate passes per arm × 5 arms.
This is slow (hours). Use `--n-samples` for a smoke test first, then the full run.

## Headroom guard — apply at analysis time (do NOT skip)

Identical to NIAH. The metric is meaningful only where **fp16 code_sim is high but the 2-bit
arms can diverge.** When reading the VM parquet:
- If **fp16 code_sim is at floor** for a task (the model can't complete the code at all),
  that task is **non-discriminating** — flag it, don't report a vacuous tie.
- If **every arm ties at ceiling**, likewise non-discriminating.
- The discriminating signal is where fp16 ≈ 0.46 (Table 1's Code ≈ 46/100) and KIVI/TurboQuant
  hold or drop. Read against TurboQuant Table 1's Code column.

## Fidelity notes from the LongBench source (Task 0)

- **`code_sim` returns 0–1** (LongBench divides `fuzz.ratio` by 100). The plot y-axis and all
  thresholds are 0–1, NOT 0–100. Table 1 reports ×100 (≈46) — multiply our number by 100 to
  compare.
- **Code tasks are NOT chat-wrapped.** LongBench explicitly skips `build_chat` for `lcc` /
  `repobench-p` ("chat models are better off without build prompts on these tasks"). Our
  `build_longbench_prompt` uses the raw template — correct, even for Llama-Instruct.
- **Templates are exact** (trailing space after "below. "; `repobench-p` has `{input}`, `lcc`
  does not); `max_gen` = 64 for both. Verified verbatim against the clone.
- **KNOWN DIVERGENCE (for the writeup):** LongBench middle-truncates over-long prompts
  (`prompt[:half] + prompt[-half:]`). We do NOT truncate — we compress the FULL context (that
  is the point: the whole context goes through the compressed cache). So our prompts may be
  longer than LongBench's truncated ones; this is intentional, flag it when reporting.
- **Compression-length proxy:** `k3_longbench` uses `length = n_prefill*2` as a representative
  length for the per-arm compression accounting (real prompts are far longer). This
  UNDER-states compression (the fixed fp16 recent-window is a larger fraction at short
  length); it is consistent across arms so rankings hold, but the absolute compression column
  is a lower bound. A future pass could thread the true tokenized prompt length through.

## Deferred (not blockers)

- **HumanEval pass@1** — explicitly NOT pursued. The paper used LongBench Code; that is the
  coding-task answer. An execution-based pass@1 would be a separate future spec.
- **Other LongBench categories** (QA/summ/synthetic) — out of scope; the generic task registry
  leaves the door open.
- Minor nits in `.superpowers/sdd/longbench-progress.md` (style observations) — for the
  writeup, not blockers.
