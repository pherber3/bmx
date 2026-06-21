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
cd ~/bmx   # on the NVIDIA VM (no CUDA on the 7900 XTX dev box)
# Run as a module (-m): `python experiments/k3_longbench.py` fails on the experiments-package import.
uv run python -m experiments.k3_longbench \
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

## 2026-06-21 — smoke results, turboquant_prod diagnosis, matched-compression arms

**Smoke (n=5, lcc, Llama-3.1-8B-Instruct, NOT Table-1-comparable):**

| arm            | code_sim | compression |
|----------------|----------|-------------|
| fp16           | 0.768    | 1.00x       |
| k2b            | 0.626    | 2.44x       |
| turboquant_mse | 0.584    | 4.25x       |
| turboquant_prod| 0.178    | 4.24x       |
| kivi           | 0.120    | 2.81x       |

(NIAH smoke, length 4096, 1 depth: fp16 7.14, k2b 6.67 @5.3x, turboquant_mse 7.14 @7.5x,
turboquant_prod 1.43 @7.5x, kivi 2.38 @6.5x — recall_full column.)

**Two findings:**

1. **The head-to-head was NOT compression-matched.** k2b (keys@3b, "bits belong to K") sits at
   2.4–5.3x; turboquant_mse at 4.3–7.9x. k2b scoring higher than turboquant is bought with extra
   storage, not a fair win. Fix: `k2b_k{bits}r{rank}` arms drop the key budget. On Llama-3.1 head
   dims (h_kv=8, d=128, group=64, S≈4096): **`k2b_k2r8` → 7.24x** (matched to turboquant_mse 7.94x
   / kivi 7.11x), `k2b_k2r16` → 6.99x. Canonical `k2b` stays at keys@3b ≈ 5.74x. Pinned by
   `tests/test_k3_experiment.py::test_k2b_matched_variants_parse_and_lower_key_bits`.

2. **turboquant_prod collapse is NOT our bug — it is faithful.** Verified three ways: (a) the
   `qjl_reconstruct` formula matches TurboQuant Alg. 2 / Theorem 2 verbatim (vault:
   `[[Two-Stage Quantization for Unbiased Inner Products]]`, `[[Quantized Johnson-Lindenstrauss
   Transform]]`); (b) the single-seed QJL inner-product estimate is **unbiased** (mean 0.3015 vs
   true 0.3007); (c) its variance **matches the paper bound** π/(2d)·‖r‖²·‖y‖² to 0.97x. The
   collapse is a real property of the method on our path: the per-key IP noise std (~0.31 at
   d=128) is as large as the IP itself, and `StreamingQuantizedCache` exposes every per-key score
   to softmax (no aggregation to average the unbiased noise down). The paper's own _mse_ arm beats
   its _prod_ arm here. Carry prod as a faithful baseline that loses; do not "fix" it.

**The matched-compression VM sweep (run after the canonical smoke):**

```bash
cd ~/bmx
uv run python -m experiments.k3_longbench \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --arms fp16 k2b k2b_k2r8 k2b_k2r16 turboquant_mse kivi \
  --n-samples 50   # then drop --n-samples for the full Table-1 run
# Read k2b_k2r8 (≈7.2x) vs turboquant_mse (≈7.9x) / kivi (≈7.1x): the fair head-to-head.
# turboquant_prod dropped from the matched sweep (faithful-but-loses; keep in a separate
# baseline run if a reviewer asks).
```

The same `--arms` list applies to `k3_niah`. Confirm the compression column lands near the
table above before trusting the quality numbers — if k2b_k2r8 is not within ~10% of
turboquant_mse's compression, the head-to-head is still mismatched.

## Deferred (not blockers)

- **HumanEval pass@1** — explicitly NOT pursued. The paper used LongBench Code; that is the
  coding-task answer. An execution-based pass@1 would be a separate future spec.
- **Other LongBench categories** (QA/summ/synthetic) — out of scope; the generic task registry
  leaves the door open.
- Minor nits in `.superpowers/sdd/longbench-progress.md` (style observations) — for the
  writeup, not blockers.
