# KV code cleanup — results (2026-07-01)

Branch `chore/kv-cleanup` off `feat/triton-decode-kernel` @ `497223d`. Plan:
`docs/superpowers/plans/2026-07-01-kv-code-cleanup.md`. Progress ledger:
`.superpowers/sdd/progress.md`.

## Verdict

A five-agent design review of the KV-program code (`src/bmx/cache/`,
`src/bmx/cache/triton_dequant_attention.py`) concluded the codebase was **not
slop — accretion debt**: dead first-generation arm implementations superseded
by the packed split, a registry re-implemented across two experiment files,
an upward import cycle dodged with function-local imports, and commit-message
narration fossilized as source comments. All of it traces to real, sequential
design evolution (per-block → packed → paged → Triton), not carelessness.
This plan paid that debt down with a strict parity rule: only deletions of
verified-dead code, verbatim moves, renames, and provably-equivalent dispatch
plumbing. Any numeric-expression change was an explicit STOP condition.
None were hit.

## Final gate (this task)

```
uv run ruff format .   → 130 files left unchanged
uv run ruff check .    → All checks passed!
uv run pytest -q       → 271 passed, 17 skipped, 1 xfailed  (≈50s)
```

Matches the ledger's tracked arithmetic: 264 (plan start) → 272 after Task 4
(+8, the new `tests/test_recipes.py`) → 271 after Task 11 (−1,
`test_synthetic_filler_is_deterministic_and_scales` deleted with the dead
`synthetic_filler` helper it pinned). `CLAUDE.md`'s Commands section and the
publication plan's Global Constraints section are both updated from 264 to
271; the GH200 Triton figure (342) is left alone — it is not locally
verifiable and should get a small (~+8, tracking the same net test delta)
adjustment on the next VM run.

## Per-task ledger

15 commits, base `497223d` → HEAD (14 tasks + 1 review-driven fix commit on
Task 11). Net across the whole branch: **36 files changed, 587 insertions(+),
1015 deletions(-)** (`git diff --shortstat 497223d..HEAD`).

| # | Commit | Subject | Net lines |
|---|---|---|---|
| 1 | `8b33e92` | refactor(triton): delete dead `_fused_decode_kernel` missed by `7b07552` debloat | +8 / −166 (2 files) |
| 2 | `bcecf18` | refactor(triton): drop dead `n_blocks_in_split` param; `pick_num_splits` reads device SM count | +17 / −14 (1 file) |
| 3 | `e1e8093` | docs(triton): fix stale module/helper docstrings; dedupe retired-kernel eulogies | +17 / −19 (2 files) |
| 4 | `1adf612` | refactor(cache): move recipe registry to `bmx.cache.recipes.spec_pair` | +140 / −84 (7 files; new `recipes.py` + `test_recipes.py`) |
| 5 | `aefffa4` | refactor(exp): shared `load_model_and_tokenizer` in `experiments/_common.py` | +26 / −23 (4 files; new `_common.py`) |
| 6 | `09d40f2` | refactor(cache): extract `hf_compat.py`; kill collect/rope upward imports; de-privatize `reshape_heads`/`register_hooks` | +93 / −85 (12 files; new `hf_compat.py`) |
| 7 | `593725f` | refactor(cache): delete dead streaming code — `_group_size`, subsumed `new_S_q<=0` branch, `_quantize_matrix` wrapper | +3 / −35 (1 file) |
| 8 | `bdca830` | refactor(codecs): delete dead first-generation arm impls | +0 / −87 (1 file) |
| 9 | `8bbc43e` | refactor(codecs): waterfill base arm = `rotation="identity"`; dispatch ladder → mode table | +33 / −139 (1 file) |
| 10 | `5ea8a1c` | refactor(codecs): derive the three arm registries from one `_ARM_TABLE`; fix stale docstring (six arms → 14) | +42 / −53 (1 file) |
| 11 | `a2633cb` + `deee952` | refactor(cache): delete dead eval code — `niah_recall_generate`, `synthetic_filler`, `_quantize_kv`, `ppl_eval` re-export (+ 1 review fix: `test_ppl_eval` import site) | +10 / −58 (5 files) |
| 12 | `e88841a` | refactor(cache): move `generate_through_cache` + `compression_for` to `bmx.cache.generate` | +135 / −124 (8 files; new `generate.py`) |
| 13 | `b625608` | refactor(cache): `query_abs_start` int-as-flag → `is_prefill` bool | +31 / −59 (10 files) |
| 14 | `d581a8b` | docs(cache): comment debloat — ticket-ID narration removed, war stories deduped | +33 / −70 (6 files) |

Three new modules landed: `src/bmx/cache/recipes.py` (named arm→spec pairs,
Task 4), `src/bmx/cache/hf_compat.py` (HF model/config introspection,
Task 6), `src/bmx/cache/generate.py` (shared generation loop + compression
accounting, Task 12) — all three now listed in `CLAUDE.md`'s Architecture
section.

## Parity statement

Every change in this plan falls into one of: DELETE verified-dead code
(grep-confirmed zero callers before deletion), MOVE code verbatim between
modules (byte-identity checked at review for Tasks 6 and 12), RENAME symbols
(`reshape_heads`/`register_hooks`, `is_prefill`), or REPLACE dispatch
plumbing with provably-equivalent plumbing (the waterfill `rotation="identity"`
merge in Task 9, the `_ARM_TABLE` registry derivation in Task 10, both with
explicit equivalence proofs re-derived by the implementer and independently
re-verified by the reviewer). Comment/docstring edits (Tasks 3, 14) touched
no code lines — confirmed via `git diff --stat` showing only comment hunks.

No numeric expression was changed anywhere in the branch. Every one of the
15 commits was gated by the full suite
(`uv run ruff format . && uv run ruff check . && uv run pytest -q`) before
being proposed, and every task was independently review-cleared per the
progress ledger (`.superpowers/sdd/progress.md`) — risk classification for
every change was **NONE or LOW**, with LOW-risk steps (Tasks 7, 9, 13)
carrying an explicit written equivalence proof checked twice (once by the
implementer, once by the reviewer) before landing. Task 11 required one
review-driven fix loop (an incidental `CacheCodecSpec` re-export import site
in `test_ppl_eval.py`) — caught by review, fixed same-task, re-gated clean.

One sanctioned exception to universal bit-identity: `pick_num_splits` now reads the device SM count (Task 2) — bit-identical on the GH200 (132 SMs) and CPU boxes, but on other CUDA devices `num_splits` (and thus split-partial merge order) may differ; the GH200 re-verify covers the authoritative hardware.

## Deferred (explicitly not in this plan)

| Item | Why deferred |
|---|---|
| Unify `StreamingQuantizedLayer`/`PackedStreamingLayer` into one flush engine + storage backends | HIGH parity risk near the GH200 merge gate; the RoPE-table dtype divergence (streaming slices fp32, packed casts fp16 at grow) is an intentional per-backend difference a merge must parameterize. Revisit only if the program reopens post-paper. |
| Delete the ~574-line per-block Triton path | Entangled with `k3_triton_decode`'s variant ledger and publication Task 11's latency re-run; do post-paper. |
| Merge the two fused kernels under an `IS_K2B` constexpr | Reviewed and rejected: disjoint dequant bodies + pointer lists; readability regression, zero codegen benefit. |
| Named bpe-term helpers (`scale_bits()` etc.); full-C turboquant = perhead h=1 | LOW-not-NONE (float reassociation risk / touches a headline baseline arm) with authoritative VM runs imminent. Post-paper polish. |
| `predict_peak` retirement; needle/haystack file folds; `_PagedStacks` dict-only normalization; `_assemble_dense_kv` extraction; experiment `arm`→`recipe` rename | Cosmetic or breaks parquet/plot/doc continuity; value below churn cost right now. |

Two additional minor items surfaced during the run's per-task roll-up
(neither rose to plan-worthy; noted here for the next pass over this code):

- **`quantize_cache`'s docstring still under-documents `svd_factors`/`tiers`.**
  Those two parameter entries name only 3 of the waterfill arms, a
  pre-existing staleness from before Task 10's registry unification — all 7
  waterfill arms now accept both parameters uniformly, but the docstring
  text was never widened to match. Candidate one-line fix next time codecs.py
  is touched.
- **`test_ppl_eval_reexports_same_class` (`tests/test_cache_specs.py`) may be
  vestigial.** Before Task 11 it pinned a deliberate backcompat re-export
  (`CacheCodecSpec` importable from `bmx.cache.ppl_eval`); Task 11 removed the
  `# re-export` framing and the import is now incidental (kept only because
  `ppl_eval.py`'s own signatures reference the name). Worth a final judgment
  call on whether the test still earns its keep, or should be deleted /
  repointed at `bmx.cache.specs` directly.

## Wave 2 (same day)

**Directive:** the user decided nothing worth doing gets deferred past the
paper — every remaining item from the review above that had real merit (not
the merits-rejected ones) got its own task NOW, before the single
authoritative GH200 re-verify, so that VM run generates artifacts from the
final, inspection-ready code. Plan:
`docs/superpowers/plans/2026-07-01-kv-code-cleanup-wave2.md` (9 tasks, branch
`chore/kv-cleanup`, same commit policy and parity discipline as Wave 1).
Wave-2 start baseline: 271 passed, 17 skipped, 1 xfailed.

### Per-task ledger

Through `56272c2` (all code + sweep commits, pre-this-close-out doc): 11
commits, 19 files changed, 426 insertions(+), 1072 deletions(−)
(`git diff --shortstat 0a9dd2d..56272c2`); the close-out and review-fix
commits land after that snapshot.

| Task | Commit(s) | Subject | Verdict / count delta |
|---|---|---|---|
| W2-1 (trivial trio) | `46bd730`, `9fbf68c`, `453f0ef` | Drop vestigial `ppl_eval` re-export test/import; recovery-doc reconciliation note; over-long comment/docstring line wraps (2 rounds — a review-caught miss, then a controller sweep) | complete, review clean; 270/17/1 (−1 passed, the deleted re-export test) |
| W2-2 (bpe terms) | `096dfd7` | Name the honest-bpe metadata terms (`scale_bits`/`norm_bits`/`factor_bits`/`tier_bits`) as an expression-identical audit surface; add a pinning test | complete, review clean, 7-hit grep re-verified all call sites argument-identical; 271/17/1 (+1 passed, the new pin) |
| W2-3 (turboquant merge, kill-or-confirm) | `b4747a7` | Collapse full-C turboquant into the perhead path as `h=1` | **VERDICT: CONFIRM** — bit-path identity verified by the implementer and independently re-verified by the reviewer with a wider sweep (`C ∈ {128, 96, 48, 1, 2, 3}` at internal and public call levels). The controller's own pre-task prediction (that non-power-of-2 `C` would diverge between the full-C and perhead code paths) was **wrong** — the merge is bit-identical across the swept range. Recorded honestly because the gate is what worked, not the prediction. |
| W2-4 (per-block deletion) | `74c73cd` | Delete the legacy ~570-line per-block Triton decode path (`_online_softmax_block_kernel`, `_online_block_kernel_launch`, `_partition_blocks`, `triton_decode_attention`); fused kernels + chunked fallback cover all configs; fail-loud tests retargeted at the fused entry points | complete, review clean, reviewer reproduced the gate bit-for-bit; **NEW LOCAL BASELINE 271/8/1** (−9 skipped — those tests were skip-gated on the now-deleted per-block path) |
| W2-5 (bench retarget) | `e011b39` | `k3_triton_decode`'s `triton_fused` variant now measures `fused_decode_attention_packed` directly (stacks prebuilt, honest timing) — the per-block baseline it used to measure is gone | complete, review clean; honest timing + field-for-field mapping + same-fixture oracle all independently verified |
| W2-6 (module folds) | `597c88d` | Fold `needle.py` + `haystack.py` into `niah.py` — one module per concern (NIAH) | complete, review clean; byte-identity re-verified independently |
| W2-7 (dedup helper) | `b5d662d` | Extract shared `_assemble_dense_kv` for the oracle and prefill paths (cast-commutes-with-cat, value-identical) | complete, review clean; oracle untouched in value, trace table verified, no test motion |
| W2-8 (kv_memory hygiene) | `b0ca88f` | `decode_speedup_curve` computes dequant FLOPs directly (drop the `_dequant_flops` private-dict-key smuggle); `predict_peak` gets a one-line role docstring | complete, review clean, `packed_case` arg verified. **STOP-escalation:** implementing the fix required `tests/test_kv_memory_latency.py` to stop reading `k2b_info["_dequant_flops"]` (a private dict key on the object being retired) — the implementer stopped and escalated rather than quietly threading the private key through; the controller approved repointing the test onto `_dequant_flops_per_step(k2b)` directly (same function, same case, identical value, no behavior change). |
| W2-9 (this close-out) | `56272c2` | Minor sweep: drop the vestigial `C` param from `_turboquant_mse_dequant` (surfaced by W2-3); `_pick_block_n` docstring wording (surfaced by W2-4); recovery-doc explicit SHA (surfaced by W2-4); `niah.py` docstring self-reference (surfaced by W2-6) | complete; full gate green at 271/8/1 both before and after |

### `predict_peak` keep-decision reversal

Wave 1's deferred list (above) originally leaned toward retiring
`predict_peak` as cosmetic churn. Wave 2 reversed that lean: `predict_peak`'s
tests pin the *measured* census anchors backing the paper's systems claim
(92.2 GiB fp16 resident, 99–100 GiB dense-stream OOM, chunked clears the
ceiling — `docs/2026-06-23-kernel-census-results.md`). Deleting it would
delete the only place those anchor numbers are pinned in code. The actual
wart was never `predict_peak` itself — it was `predict_decode_latency`
smuggling a private `"_dequant_flops"` key through a returned dict for
`decode_speedup_curve` to finish computing outside the function that owns
the math. W2-8 fixed the real wart (`decode_speedup_curve` now computes
dequant FLOPs directly) and gave `predict_peak` a one-line docstring
declaring its role as a census anchor, kept deliberately.

### Final gate (Wave 2)

```
uv run ruff format .   → 128 files left unchanged
uv run ruff check .    → All checks passed!
uv run pytest -q       → 271 passed, 8 skipped, 1 xfailed  (≈53s)
```

`CLAUDE.md`'s Commands section and the publication plan's Global Constraints
baseline line are both updated from 271/17/1 to **271/8/1**. The GH200
Triton figure (342, carried from the Wave-1 write-up) is left alone — still
not locally verifiable, and now needs a downward adjustment tracking the
same net per-block-path-deletion delta on the next VM run, rather than the
small `+8` bump anticipated after Wave 1.

### Deferred — now empty except merits-rejected items

Every Wave-1 deferred item with real merit got a Wave-2 task (bpe terms →
W2-2, turboquant merge → W2-3 kill-or-confirm, per-block path deletion →
W2-4, `predict_peak` → resolved by keeping it with the real wart fixed
elsewhere in W2-8, needle/haystack folds → W2-6, `_assemble_dense_kv` →
W2-7, the trivial trio → W2-1). What remains is **only** the items the
review process rejected on their merits, not deferred for scheduling
reasons:

| Item | Why rejected |
|---|---|
| Unify `StreamingQuantizedLayer`/`PackedStreamingLayer` into one flush engine + storage backends | The per-backend divergence (RoPE-table dtype: streaming slices fp32, packed casts fp16 at grow) is intentional, not accidental duplication — the "redundancy" IS the verification structure (two independently-checkable implementations of related but distinct contracts). Merging would remove a cross-check, not just lines. |
| Merge the two fused Triton kernels under an `IS_K2B` constexpr | Reviewed and rejected in Wave 1: disjoint dequant bodies + pointer lists; readability regression, zero codegen benefit. Re-affirmed in Wave 2 — no new information changes the verdict. |
| Experiment `arm` → `recipe` rename | Cosmetic; `arm` is also a load-bearing parquet column name (`LEDGER_COLUMNS`, forbidden-touch in both waves) — renaming the Python-side variable while the column name stays `arm` buys inconsistency, not clarity. |
| Historical-doc rewrites | `docs/` is a decision record — Wave 1's Global Constraints explicitly forbid rewriting old plans/results to match new identifiers, and that rule carries into Wave 2 unchanged. |
