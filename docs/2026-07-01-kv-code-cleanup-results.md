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
