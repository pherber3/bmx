# KV Code Cleanup Wave 2 — Pre-Paper Finish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Every task is behavior-preserving under the same parity discipline as Wave 1 — if you find yourself changing a numeric result, STOP and escalate.

**Goal:** Land every remaining worthwhile cleanup from the 2026-07-01 review NOW — before the single GH200 re-verify — so the paper's authoritative VM artifacts are generated from the final, inspection-ready code (user decision 2026-07-01: nothing worth doing gets deferred past the paper).

**Architecture:** Nine tasks on branch `chore/kv-cleanup` (currently at `e448cb7`, equal to `feat/triton-decode-kernel`). The headline is Task 4 (delete the ~570-line legacy per-block Triton path and retarget the bench at the real fused kernels); the rest are small: named bpe-term helpers, a kill-or-confirm turboquant merge, module folds, one dedup helper, kv_memory hygiene, and the trivial trio from the Wave-1 final review. One fable whole-branch review at the end, then fast-forward back into `feat/triton-decode-kernel`.

**Tech Stack:** Python 3.12, PyTorch 2.12, pytest, ruff, uv. No CUDA locally — Triton code must import cleanly with `TRITON_AVAILABLE=False`; CUDA-gated tests skip locally and run on the GH200 re-verify.

## Global Constraints

Identical to Wave 1 (`docs/superpowers/plans/2026-07-01-kv-code-cleanup.md` Global Constraints) — copied essentials:

- **Commit policy:** per-task auto-commit pre-authorized (user, 2026-07-01) with the task's exact proposed message; stage EXPLICIT paths only, never `git add -A`; NO AI attribution ever.
- Gate per task: `uv run ruff format . && uv run ruff check . && uv run pytest -q` — all clean. **Local baseline at Wave-2 start: 271 passed, 17 skipped, 1 xfailed.** Tasks 1, 2, and 4 change counts deliberately (each states its delta); Task 9 records the final counts everywhere.
- Allowed change classes: DELETE verified-dead/legacy code; MOVE verbatim; RENAME; provably-equivalent plumbing; text edits. Forbidden-touch list unchanged: bpe arithmetic (moving a formula into a named helper with the IDENTICAL expression is allowed — reassociating it is not), PAGE formula, RoPE-table dtypes, LongBench templates, EOS ordering, `logits_to_keep=1`, `LEDGER_COLUMNS` names, `arm` parquet column.
- Kill-or-confirm ethos: Task 3 explicitly permits the conclusion "not merged, here's the measured reason" — an honest negative is a valid deliverable.
- Historical docs under `docs/` are a decision record: never rewrite old plans/results to match new identifiers.

---

### Task 1: The trivial trio — vestigial re-export test, recovery-doc note, pointer-line polish

**Files:**
- Modify: `tests/test_cache_specs.py`
- Modify: `docs/2026-06-24-decode-path-debloat-removal.md`
- Modify: `src/bmx/cache/packed_streaming.py` (one comment line), `src/bmx/cache/triton_dequant_attention.py` (one comment line, if over-long)

- [ ] **Step 1: Delete the vestigial re-export test.** In `tests/test_cache_specs.py`: delete `test_ppl_eval_reexports_same_class` and the `from bmx.cache.ppl_eval import CacheCodecSpec as SpecFromPpl` import; replace the module docstring with `"""CacheCodecSpec defaults (the codec-spec contract every arm builds on)."""`. `test_spec_defaults` stays untouched. (Rationale: ppl_eval's import of the name survives only because its own signatures use it — there is no deliberate re-export contract left to pin; the old test would fail on a harmless import-style change.)

- [ ] **Step 2: Recovery-doc reconciliation note.** Append one sentence to the Addendum section of `docs/2026-06-24-decode-path-debloat-removal.md`:

```markdown
Note for anyone recovering: code taken from `93751eb` predates the 2026-07-01 cleanup
(`query_abs_start` → `is_prefill`, module extractions, and the per-block decode path
removal) — reconcile signatures against current `chunked_attention.py` /
`packed_streaming.py` before wiring it back in.
```

- [ ] **Step 3: Pointer-line polish.** In `packed_streaming.py`, find the Task-14 one-liner that reads `# ... see the AttentionMaskInterface registration in packed_streaming.py and docs/...` — it self-references its own file; reword to `# ... see the AttentionMaskInterface registration at the top of this module and docs/2026-06-23-kernel-census-results.md.` Then check the two retired-kernel pointer lines (grep `_k2b_softmax_block_kernel`); if any exceeds ~100 chars, wrap it onto two comment lines with the same text.

- [ ] **Step 4: Gate.** Full ruff + pytest → **270 passed** (−1: the deleted test), 17 skipped, 1 xfailed.

- [ ] **Step 5: Commit.** Stage the three files; propose-and-commit: `refactor(tests): drop vestigial ppl_eval re-export pin; recovery-doc reconciliation note; pointer-line polish`.

---

### Task 2: Named bpe-term helpers in `codecs.py` (the honest-accounting audit surface)

**Files:**
- Modify: `src/bmx/cache/codecs.py`
- Test: `tests/test_cache_codecs.py` (one new test)

**Interfaces:**
- Produces: module-level `scale_bits(group)`, `norm_bits(h, C)`, `factor_bits(rank, S, C)`, `tier_bits(tiers, S)` in `bmx.cache.codecs`. Bodies are the EXACT existing expressions — this task names them, it does not re-derive them.

- [ ] **Step 1: Write the failing test** (add to `tests/test_cache_codecs.py`):

```python
def test_bpe_term_helpers_are_the_audit_surface():
    # The named metadata terms: one place to audit "ALL metadata counted".
    from bmx.cache.codecs import factor_bits, norm_bits, scale_bits, tier_bits

    assert scale_bits(64) == 16.0 / 64
    assert norm_bits(1, 128) == 16.0 / 128
    assert norm_bits(8, 1024) == 16.0 * 8 / 1024
    assert factor_bits(16, 256, 1024) == 16.0 * 16 * (256 + 1024) / (256 * 1024)
    assert tier_bits((0, 2, 3, 4), 256) == 2 / 256  # ceil(log2(4)) = 2
```

- [ ] **Step 2: Run, verify fail.** `uv run pytest tests/test_cache_codecs.py::test_bpe_term_helpers_are_the_audit_surface -v` → FAIL (ImportError).

- [ ] **Step 3: Add the helpers** near the top of `codecs.py` (after the registry block), with this exact code:

```python
# ---------------------------------------------------------------------------
# Honest-bpe metadata terms — the audit surface for "ALL metadata counted".
# Every arm's bpe is payload bits + a sum of these named terms; the expressions
# are the scientific record and must not be re-derived or reassociated.
# ---------------------------------------------------------------------------


def scale_bits(group: int) -> float:
    """fp16 groupwise-RTN scale: one fp16 per `group` entries."""
    return 16.0 / group


def norm_bits(h: int, C: int) -> float:
    """fp16 per-row norms, `h` per row of C channels (h=1: one full-row norm)."""
    return 16.0 * h / C


def factor_bits(rank: int, S: int, C: int) -> float:
    """fp16 low-rank factors Us (S×r) + V (C×r), amortized per entry."""
    return 16.0 * rank * (S + C) / (S * C)


def tier_bits(tiers: tuple[int, ...], S: int) -> float:
    """Per-channel tier map: ceil(log2(n_tiers)) bits per channel, amortized."""
    return math.ceil(math.log2(len(tiers))) / S
```

- [ ] **Step 4: Replace each formula site with the helper call — expression-identical, sites enumerated:**
  - `quantize_packed`: the three RTN returns `bits + 16.0 / group` → `bits + scale_bits(group)` (rtn_token, rtn_channel, rotate_rtn_token); `turboquant_mse` return `bits + 16.0 / C` → `bits + norm_bits(1, C)`; `turboquant_mse_perhead` return `bits + 16.0 * h / C` → `bits + norm_bits(h, C)`; the lowrank tail `bpe = bits + 16.0 / group + 16.0 * rank * (S + C) / (S * C)` → `bpe = bits + scale_bits(group) + factor_bits(rank, S, C)`.
  - `turboquant_prod`'s `(bits - 1) + 1 + 32.0 / C` stays VERBATIM — add the comment `# payload + 1 sign bit + two fp16 norm vectors (mse + qjl) = 2 * norm_bits(1, C)` above it. (Rewriting 32.0/C risks nothing mathematically, but verbatim + comment keeps the record untouched.)
  - `_lowrank_rotwaterfill_channel`'s tail: `scale_term = 16.0 / group` → `scale_term = scale_bits(group)`; `factor_term = 16.0 * rank * (S + C) / (S * C)` → `factor_term = factor_bits(rank, S, C)`; `tier_term = math.ceil(math.log2(len(tiers))) / S` → `tier_term = tier_bits(tiers, S)`. The `bpe = mean_payload + scale_term + factor_term + tier_term + rot_bits` sum is untouched.
  - Verify no formula site was missed: `grep -n "16.0 /\|16.0 \*" src/bmx/cache/codecs.py` — every remaining hit must be either inside a helper body, the turboquant_prod verbatim line, a rot_bits charge expression inside the waterfill modes (`16.0 * C / S`, `16.0 * kk / S`, `16.0 * d / S` — these are per-mode rotation charges, LEAVE them), or a comment.

- [ ] **Step 5: Gate.** Full ruff + pytest → **271 passed** (+1), 17 skipped, 1 xfailed. The bpe pin tests in `test_cache_codecs.py` are the parity authority — any change there means an expression drifted: revert and report.

- [ ] **Step 6: Commit.** `refactor(codecs): name the honest-bpe metadata terms (scale/norm/factor/tier_bits) — expression-identical audit surface`.

---

### Task 3: [KILL-OR-CONFIRM] Merge full-C turboquant into perhead(h=1)

**Files:**
- Modify: `src/bmx/cache/codecs.py` (`_turboquant_mse_packed`/`_dequant` vs `_turboquant_mse_perhead_packed`/`_dequant`)

**The gate (run BEFORE any edit):** the merge is legal ONLY IF `_turboquant_mse_perhead_packed(M, bits, seed, h=1)` is exactly `_turboquant_mse_packed(M, bits, seed)` for every C the full-C arm supports. Read both bodies and answer specifically:
1. Rotation path: what rotation does each apply? If full-C uses `_rotate` (Hadamard when C is a power of 2, `random_orthogonal` otherwise) while perhead applies a Hadamard over d unconditionally (or asserts power-of-2), then for non-power-of-2 C the h=1 case DIVERGES and the merge is dead — SKIP.
2. Norm/index dtypes and codebook calls identical?
3. Packed-dict schema: full-C stores `{indices, norms, bits}`, perhead `{indices, norms, bits, h}` — consumers (`dequant_packed`, the fused kernel's stack builder) key on schema; a merge must keep both dict shapes emitted per arm name.

- [ ] **Step 1:** Run the gate: read both pack/dequant pairs; write the verdict in your report with line evidence.
- [ ] **Step 2 (CONFIRM branch):** implement `_turboquant_mse_packed(M, bits, seed)` as `return _turboquant_mse_perhead_packed(M, bits, seed, h=1)` reshaped to the full-C dict schema (and the dequant analogously) ONLY if Step 1 proved bit-path identity for all supported C. Run `uv run pytest tests/test_codec_split.py tests/test_cache_codecs.py -q` — any diff, revert.
- [ ] **Step 2 (KILL branch):** if Step 1 found divergence (expected: the non-power-of-2-C rotation path), make NO code change; instead add one comment above `_turboquant_mse_packed`: `# NOT the h=1 case of the perhead codec: full-C falls back to random_orthogonal for non-pow2 C, perhead requires pow2 d. Kept separate deliberately (2026-07-01 kill-or-confirm).` This is the honest-negative deliverable.
- [ ] **Step 3: Gate.** Full ruff + pytest → same counts as after Task 2.
- [ ] **Step 4: Commit.** CONFIRM: `refactor(codecs): full-C turboquant = perhead h=1 (bit-path identity verified)`. KILL: `docs(codecs): record why full-C turboquant is NOT the perhead h=1 case (kill-or-confirm)`.

---

### Task 4: Delete the legacy per-block Triton decode path

The ~570-line stage-1 scaffold (`_online_softmax_block_kernel` + `_AUTOTUNE_CONFIGS` + `_online_block_kernel_launch` + `_partition_blocks` + `triton_decode_attention`) predates the fused kernels; it serves only CUDA-decode configs that miss both fused predicates — non-headline eval arms the chunked fallback already covers within tolerance.

**Files:**
- Modify: `src/bmx/cache/triton_dequant_attention.py`
- Modify: `src/bmx/cache/packed_streaming.py` (import + the fallback route)
- Modify: `tests/test_triton_decode_rtn.py` (keep 2 smoke tests, delete per-block tests)
- Modify: `tests/test_packed_dispatch.py` (retarget the fail-loud tests at the fused entry points)
- Modify: `docs/2026-06-24-decode-path-debloat-removal.md` (addendum sentence)

**MUST SURVIVE (grep-verified consumers):** `_merge_partials` (used by `_finalize_decode`'s tail path), `_pick_block_n` (used by both fused launchers), `_require_triton`, `_next_pow2`, `_hadamard_matrix`, `_FUSED_AUTOTUNE_CONFIGS`, `_fused_merge_kernel`, `_finalize_decode`, `pick_num_splits`, both fused kernels + launchers + both stack builders.

- [ ] **Step 1: Verify the deletion set's callers.** `grep -rn "_online_softmax_block_kernel\|_online_block_kernel_launch\|_partition_blocks\|triton_decode_attention\b\|_AUTOTUNE_CONFIGS\b" src/ tests/ experiments/ --include="*.py"` — expected consumers: the triton module itself, `packed_streaming.py` (import + one call), `tests/test_triton_decode_rtn.py`, `tests/test_packed_dispatch.py` (monkeypatch targets), `experiments/k3_triton_decode.py` (Task 5 handles that file — do NOT touch it here; its import is function-local under `device == "cuda"`, so the suite stays green between Tasks 4 and 5 on this CUDA-less box, and Task 5 lands before any CUDA machine runs it). Anything else → STOP.

- [ ] **Step 2: Delete in `triton_dequant_attention.py`:** the AUTOTUNE NOTE block that documents the per-block kernel (~:147–153); the `if TRITON_AVAILABLE:` block containing `_AUTOTUNE_CONFIGS` and `_online_softmax_block_kernel` (~:155–283) — KEEP the `from triton import Config as _TritonConfig` import if `_FUSED_AUTOTUNE_CONFIGS` needs it (it does — move that import line next to `_FUSED_AUTOTUNE_CONFIGS` if it lived in the deleted block); `_online_block_kernel_launch` (~:285–366); `_partition_blocks` (~:367–389); `triton_decode_attention` and its nested helpers (~:452–~704). KEEP `_merge_partials` (~:390–451) — it sits between deletion targets; cut around it. Update the module docstring's entry-point list (remove the `triton_decode_attention` fallback mention; the fallback is now chunked for all non-fused configs).

- [ ] **Step 3: Reroute dispatch in `packed_streaming.py`:** remove `triton_decode_attention` from the import at ~:26; delete the entire `if (TRITON_AVAILABLE and is_decode and q.is_cuda and self.k_spec.arm != "lowrank_rtn_channel"): return triton_decode_attention(...)` block (~:645–668) plus its lead comment; the flow falls through to the existing `chunked_dequant_attention` call. Update the dispatch comment above `attend`'s routing to say: fused packed / fused k2b are the CUDA fast paths; everything else — including CUDA decode on non-fused arms — runs chunked (fp32-accumulating reference path). The FAIL-LOUD comment stays: it now governs the two fused routes only.

- [ ] **Step 4: Tests.**
  - `tests/test_triton_decode_rtn.py`: KEEP `test_triton_module_imports_with_available_flag` and `test_require_triton_raises_without_cuda` (locally-running smoke tests). DELETE the per-block tests (`test_triton_rtn_decode_matches_oracle`, `test_triton_rtn_decode_matches_oracle_prerope`, `test_triton_decode_asserts_n_q_eq_1`, `test_triton_split_kv_matches_oracle`, `test_triton_split_kv_num_splits_1_bit_identical_to_3a`) and their now-unused imports/fixture helpers within this file. Rename nothing; the file keeps its name (it still smoke-tests the triton module).
  - `tests/test_packed_dispatch.py`: the fail-loud contract must survive, retargeted. Rewrite `test_no_silent_swallow` to monkeypatch `ps_mod.fused_decode_attention_packed` with the raise-sentinel (config: rtn_token/rtn_token post-RoPE uniform blocks — the fused_packed predicate) and assert attend propagates the sentinel; rewrite `test_k2b_pre_rope_falls_back_to_chunked` to monkeypatch BOTH `ps_mod.fused_decode_attention_packed` and `ps_mod.fused_decode_attention_k2b` with raise-sentinels and assert a k2b-with-non-pow2-rank config completes via chunked without calling either. Preserve each test's existing skipif markers and fixture style (they already build configs via `tests/factories.py`).

- [ ] **Step 5: Recovery-doc addendum.** Append: `2026-07-01 Wave 2: the per-block launch path (_online_softmax_block_kernel, _online_block_kernel_launch, _partition_blocks, triton_decode_attention) was also removed — the fused kernels + chunked fallback cover all configs. Recover from the parent of the Wave-2 removal commit if ever needed.`

- [ ] **Step 6: Gate.** Full ruff + pytest → passed count unchanged from Task 3; **skipped count DROPS** (the deleted CUDA-gated tests were skips locally) — record exact new counts for Task 9.

- [ ] **Step 7: Commit.** `refactor(triton): delete legacy per-block decode path — fused kernels + chunked fallback cover all configs; fail-loud tests retargeted at fused entry points`.

---

### Task 5: Retarget `k3_triton_decode`'s `triton_fused` variant at the real fused kernel

The bench's `triton_fused` variant currently calls the just-deleted per-block path — mislabeled since the fused kernels landed. Retarget it at `fused_decode_attention_packed` (the deployment RTN kernel), timing stacks honestly.

**Files:**
- Modify: `experiments/k3_triton_decode.py`

**Interfaces:**
- Consumes: `fused_decode_attention_packed` and `build_kv_stacked_packed` from `bmx.cache.triton_dequant_attention` — READ BOTH SIGNATURES FIRST (the launcher takes `(q, k_codes, v_codes, k_scales, v_scales, seq_len, *, n_q_groups, scale, k_group, v_group, k_tail, v_tail, ...)`; match it exactly, including any num_splits parameter and defaults).

- [ ] **Step 1:** Read both signatures + `PackedStreamingLayer.attend`'s `fused_packed_ok` call site (packed_streaming.py ~:557–588) — it is the production usage to mirror.
- [ ] **Step 2:** In `_make_blocks_for_seqlen`, when `device == "cuda"`, ALSO prebuild the stacked views once per fixture (mirroring `_PagedStacks`' amortized production behavior — stack building must NOT be inside the timed closure): call `build_kv_stacked_packed` over `k_blocks`/`v_blocks` with the fixture's `h_kv`, `blk_size=cfg.blk`, `d=cfg.d`, `group=cfg.group`, `v_group=cfg.group`, and return the stack tensors alongside the existing outputs.
- [ ] **Step 3:** Replace the `_triton` variant closure: it ignores `kb`/`vb` and calls `fused_decode_attention_packed(q, <stacks...>, seq_len, n_q_groups=cfg.n_q_groups, scale=cfg.d**-0.5, k_group=cfg.group, v_group=cfg.group, k_tail=None, v_tail=None)`. Keep `variant_tol={"triton_fused": 1e-2}` (same fp16-resident rationale). Keep the `chunked_dequant` reference variant untouched.
- [ ] **Step 4:** Rewrite the module docstring: variants are now `chunked_dequant` (PyTorch reference) and `triton_fused` (= `fused_decode_attention_packed`, the single-launch split-KV deployment kernel); DELETE the stale "per-block Python launch loop … this run is the BASELINE for the fused-kernel rewrite" narrative. Note the fixture constraint: `cfg.arm` must be `rtn_token` post-RoPE (the fused packed layout) — add `assert cfg.arm == "rtn_token", "fused packed kernel benches rtn_token"` at the top of `main`.
- [ ] **Step 5: Gate.** Full ruff + pytest → counts unchanged from Task 4 (this experiment has no local test coverage of the CUDA branch; the offline suite only imports it). CUDA verification is the GH200 re-verify (the ledger's oracle columns gate correctness there — `run_decode_ledger` diffs every variant against `naive_dense_attention` per its existing contract).
- [ ] **Step 6: Commit.** `feat(bench): k3_triton_decode triton_fused variant now measures fused_decode_attention_packed (stacks prebuilt, honest timing) — per-block baseline retired`.

---

### Task 6: Fold `needle.py` + `haystack.py` into `niah.py` (one NIAH module)

**Files:**
- Modify: `src/bmx/cache/niah.py` (absorb both), Delete: `src/bmx/cache/needle.py`, `src/bmx/cache/haystack.py`
- Modify: `tests/test_needle.py` (:5 import), `tests/test_haystack.py` (:1 import), `experiments/k3_live_generation.py` (:32 import block), `experiments/k3_niah.py` (:57 function-local import)

- [ ] **Step 1:** Move VERBATIM into `niah.py`: from `needle.py` — `_argmax_next_at`, `needle_retrieved_from_ids`, `needle_retrieved`, `build_needle_ids` (all four; carry the `StreamingQuantizedCache` import; `CacheCodecSpec`/`torch` already imported); from `haystack.py` — `PG_ESSAYS_DATASET` and `load_pg_corpus` (keep the lazy `datasets` import inside the function). Place the needle block after the existing synthetic-ids section, the corpus loader at the end. Remove niah.py's now-internal `from bmx.cache.needle import needle_retrieved`. Extend niah.py's module docstring by one line: `Also home to the planted-needle probes (former needle.py) and the PG-essay corpus loader (former haystack.py).`
- [ ] **Step 2:** Delete both source modules. Update the four import sites to `from bmx.cache.niah import ...` (same names). Grep: `grep -rn "bmx.cache.needle\|bmx.cache.haystack" src/ tests/ experiments/ --include="*.py"` → zero.
- [ ] **Step 3: Gate.** Full ruff + pytest → counts unchanged (test files keep their names; only imports moved).
- [ ] **Step 4: Commit.** `refactor(cache): fold needle.py + haystack.py into niah.py — one module per concern (NIAH)`.

---

### Task 7: Extract `_assemble_dense_kv` in `chunked_attention.py`

`naive_dense_attention` and `_prefill_dense_attention` duplicate the dense K/V assembly (dequant-all-blocks + tail concat). One helper, both call it.

**Files:**
- Modify: `src/bmx/cache/chunked_attention.py`

- [ ] **Step 1:** Add above `naive_dense_attention`:

```python
def _assemble_dense_kv(
    k_blocks,
    v_blocks,
    *,
    k_arm,
    v_arm,
    group,
    seed,
    v_group,
    v_seed,
    h_kv,
    k_pre_rope,
    rope_cos,
    rope_sin,
    k_tail,
    v_tail,
    dtype,
):
    """Dequant all blocks + fold the fp16 tail -> dense (h_kv, S, d) K and V in `dtype`.

    Shared by the oracle and the prefill-SDPA path. Casting is elementwise, so
    cast-then-cat vs cat-then-cast are value-identical; this helper standardizes
    on casting each piece before concatenation.
    """
    K = _dense_kv(k_blocks, k_arm, group, seed, h_kv, k_pre_rope, rope_cos, rope_sin)
    V = _dense_kv(v_blocks, v_arm, v_group, v_seed, h_kv, False, None, None)
    if k_tail is not None and k_tail.shape[1] > 0:
        kt = k_tail.to(dtype)
        vt = v_tail.to(dtype)
        K = kt if K is None else torch.cat([K.to(dtype), kt], dim=1)
        V = vt if V is None else torch.cat([V.to(dtype), vt], dim=1)
    else:
        K = K.to(dtype)
        V = V.to(dtype)
    return K, V
```

- [ ] **Step 2:** Rewire both callers to `K, V = _assemble_dense_kv(..., dtype=q.dtype)` followed by their existing GQA expansion (`repeat_interleave`) and attention math. Delete the duplicated assembly lines from each. NOTE the one intentional normalization: `naive_dense_attention` previously cast K/V to `q.dtype` at the `repeat_interleave` line (cat-then-cast) while `_prefill_dense_attention` cast pieces first — elementwise casts commute with `torch.cat`, so values are identical; state this in the commit body if the oracle-diff tests are within existing tolerances (they pin this: `tests/test_chunked_gqa_opt.py`, oracle tests).
- [ ] **Step 3: Gate.** Full ruff + pytest → counts unchanged. If ANY oracle/parity test moves, revert and report — the cast-commutes argument failed and the change is not NONE-risk.
- [ ] **Step 4: Commit.** `refactor(cache): shared _assemble_dense_kv for oracle + prefill paths (cast-commutes-with-cat, value-identical)`.

---

### Task 8: `kv_memory.py` hygiene — keep `predict_peak` (labeled), remove the `_dequant_flops` smuggle

**DECISION REVERSAL (user's gorgeous-repo directive, scoped honestly):** `predict_peak` is NOT retired — its tests pin the measured census anchors (92.2 GiB fp16, 99–100 GiB dense-stream OOM, chunked-clears-ceiling) that back the paper's C3 memory claim. The actual wart is `predict_decode_latency` returning a private `"_dequant_flops"` key for `decode_speedup_curve` to finish computing.

**Files:**
- Modify: `src/bmx/bench/kv_memory.py`
- Test: `tests/test_kv_memory.py` (existing tests are the pin; check none read `"_dequant_flops"`)

- [ ] **Step 1:** `grep -n "_dequant_flops" src/ tests/ experiments/ --include="*.py"` — expected: the def, the dict key in `predict_decode_latency`, the read in `decode_speedup_curve`, and possibly `triton_bench.py`. If `triton_bench.py` (or any test) reads the KEY `"_dequant_flops"` from the returned dict, STOP and report — the smuggle is load-bearing beyond the two functions.
- [ ] **Step 2:** In `predict_decode_latency`: delete the `dequant_flops = _dequant_flops_per_step(case)` line and the `"_dequant_flops": dequant_flops,` dict entry (the `"dequant_compute_time_s": 0.0` stays, as does its docstring rationale). In `decode_speedup_curve`: replace `dequant_time = p["_dequant_flops"] / peak_flops_per_s` with `dequant_time = _dequant_flops_per_step(packed_case) / peak_flops_per_s` — same function, same case, same value.
- [ ] **Step 3:** Add one role line to `predict_peak`'s docstring (it currently has none — it starts with a bare assert): `"""Analytic peak-memory model for the census paths; its tests pin the MEASURED anchors (92.2 GiB fp16, 99–100 GiB dense-stream OOM) — see docs/2026-06-23-kernel-census-results.md."""`
- [ ] **Step 4: Gate.** Full ruff + pytest → counts unchanged (`test_kv_memory.py` all green — they pin the anchors).
- [ ] **Step 5: Commit.** `refactor(bench): decode_speedup_curve computes dequant FLOPs directly (drop _dequant_flops dict smuggle); predict_peak role docstring (census-anchor pin, kept deliberately)`.

---

### Task 9: Wave-2 close-out — baselines, results doc, publication-plan count

**Files:**
- Modify: `CLAUDE.md`, `docs/2026-07-01-kv-code-cleanup-results.md`, `docs/superpowers/plans/2026-07-01-publication-readiness.md`

- [ ] **Step 1:** Final gate run; record exact counts (expect ~271 passed / ~12–15 skipped / 1 xfailed — Task 1 −1, Task 2 +1, Task 4 −skips; use ACTUALS).
- [ ] **Step 2:** Update `CLAUDE.md`'s pytest-count line and the publication plan's baseline line to the actuals (both currently say 271/17/1). In CLAUDE.md's research-state Triton bullet, update the parenthetical about the per-block path if it mentions one (grep `per-block` in CLAUDE.md).
- [ ] **Step 3:** Append a `## Wave 2 (same day)` section to `docs/2026-07-01-kv-code-cleanup-results.md`: the user's do-it-now directive, the per-task ledger (from `git log --stat` over the Wave-2 commits), the Task-3 kill-or-confirm verdict, the predict_peak keep-decision reversal + rationale, and the now-empty deferred list (only the merits-rejected items remain: cache-fork unification — redundancy is the verification structure; `IS_K2B` merge; `arm`→`recipe` rename; historical-doc rewrites).
- [ ] **Step 4: Gate** (ruff only for doc edits + one final full pytest for the record). **Commit.** `docs: Wave-2 cleanup ledger — per-block path retired, bpe terms named, folds landed; baselines updated`.

---

## Self-Review

- **Directive coverage:** every non-merits-rejected deferred item has a task — trivial trio (T1), bpe terms (T2), turboquant merge as kill-or-confirm (T3), per-block deletion + bench retarget (T4+T5), folds (T6), `_assemble_dense_kv` (T7), predict_peak resolved-by-keeping + real wart fixed (T8), baselines (T9). Merits-rejected items are documented in T9's ledger, not silently dropped. ✓
- **Placeholder scan:** T5 requires reading two signatures before wiring — that is a verify-first step against live code, with the call shape and production mirror named, not a TBD. All other code steps carry exact code or exact anchors + keep-lists. ✓
- **Sequencing hazards:** T4 before T5 (T5 rewires what T4 orphans; the orphaned import is function-local CUDA-gated so the suite stays green between them — stated in T4 Step 1). T2 before T3 (T3 reads the turboquant bodies T2 just touched — T2 changes only the bpe return lines, not the pack math). Counts thread T1(−1) → T2(+1) → T4(−skips) → T9 records actuals. ✓
- **Parity:** the only tasks touching numeric-adjacent code are T2 (expression-identical naming, pinned by bpe tests), T3 (gated, skip-permitted), T7 (cast-commutes argument with revert-on-any-test-motion), T4/T5 (deletion + bench retarget; headline paths untouched, chunked fallback for stragglers is the plan-sanctioned LOW change, GH200 re-verify covers CUDA). ✓
