# Transitive-Baseline Justification Memo

**Date:** 2026-07-01. **Plan:** `docs/superpowers/plans/2026-07-01-publication-readiness.md` (Task 2 deliverable).
**Decision (locked by user 2026-07-01): baselines are TRANSITIVE.** We do NOT implement PolarQuant / SnapKV / PyramidKV.

This memo makes the transitive-baseline argument airtight by pinning it to a reproduced anchor. It is referenced by publication-plan Tasks 9, 10 (VM runs) and 12 (outline).

## 1. The transitive argument

TurboQuant (arXiv 2504.19874) already benchmarks against KIVI, PolarQuant, SnapKV, PyramidKV, and Full-Precision on the *same* tasks we use (NIAH Fig 4, LongBench Table 1, Llama-3.1-8B-Instruct). We benchmark against TurboQuant. Therefore:

> We beat `turboquant_mse` at matched compression (C1) → TurboQuant beats KIVI/PolarQuant/SnapKV/PyramidKV in its own Table 1 / Fig 4 → we dominate them transitively.

This is standard and accepted in the KV-compression literature *provided the measurement paths are comparable* (§2). It deletes the need to implement three additional baselines from scratch — the former publication-plan Task 7 (PolarQuant arm) is removed.

## 2. The seam, and how we close it — the ANCHOR

Transitivity holds **only if our measurement path is numerically comparable to TurboQuant's.** A reviewer will not ask "did you re-run SnapKV?"; they will ask "is your Full-Cache row, and your reproduced-TurboQuant row, the same quantity as TurboQuant's Table 1?" We answer yes, with evidence:

**The anchor = we run `fp16` (Full Cache) + `turboquant_mse` on our own `StreamingQuantizedCache` path and show they reproduce TurboQuant's published Table-1 values.**

- LongBench Code already reproduces: our `turboquant_mse` = **46.0 ×100** vs TurboQuant Table-1 Code ≈ 46.28. (`docs/2026-06-21-niah-longbench-frontier-results.md`.)
- Full-scale confirmation of Full-Cache Avg ≈ 50.06 and turboquant_mse Avg is the job of **publication-plan Task 10 Step 4** — a HARD GATE: if the reproduced anchor rows do not match Table 1, the transitive argument is void and the mismatch must be root-caused before any baseline claim is written.

If the anchor matches, every transitive claim about the un-run baselines is licensed: our path measures the same thing theirs does, so their numbers for the un-run methods slot directly into our table as reference rows.

## 3. The three baselines we RUN, and why each is the anchor (not a courtesy)

These are NOT optional comparison points we could drop — they are the machinery that licenses the whole comparison. All three are already implemented; `spec_pair(arm, *, rank, group, seed)` in `src/bmx/cache/recipes.py` accepts them (confirmed 2026-07-01):

| Arm (via `spec_pair`) | Role | Why it's the anchor, not a courtesy |
|---|---|---|
| `fp16` | Full Cache / quality ceiling | Reproduces TurboQuant's "Full Cache" row (KV Size 16). The comparability datum: if our fp16 Avg ≈ 50, our path == their path. |
| `turboquant_mse` (+`turboquant_prod`) | The mirror method | Reproducing its Table-1 row (Code ≈ 46) on our path is THE proof of comparability. `turboquant_prod` carried as the faithful-but-losing unbiased variant (documented, not a bug). |
| `kivi` (`rtn_channel` K+V @2b + fp16 window) | One real scalar-quant baseline | The one non-transitive baseline we run directly — cheap, already implemented, and the weakest-strong baseline so a direct win is on record, not only transitive. |

Our own arms in the same table: `k2b` (canonical 3-bit K) and `k2b_k2r8` (compression-matched 2-bit K) — the C1/C2 contributions.

## 4. The un-run baselines — reported as TurboQuant's attributed reference rows

In Table T1 (LongBench) and the NIAH figure, PolarQuant / SnapKV / PyramidKV appear as **reference rows with their TurboQuant-published values, clearly attributed** ("as reported in Zandieh et al. 2025, Table 1"), never presented as our own measurements. Justification per method:

- **PolarQuant** — a quantization peer (same author group, arXiv 2502.02617). Transitive via the anchor; not re-implemented. If a reviewer specifically demands a direct PolarQuant head-to-head, the extension is a known, scoped follow-on (a `polarquant` arm in `CACHE_ARMS`), not a blocker for this paper.
- **SnapKV / PyramidKV** — **token-eviction** methods, a *different mechanism* from quantization. Their absence as direct arms is **scope**, not omission: this paper is about KV *quantization*, and eviction is orthogonal (composable, not competing). State this explicitly so the absence reads as a scoping choice.

## 5. What Task 10 must verify (the gate this memo rests on)

Publication-plan **Task 10 Step 4 is the enforcement point.** Before any transitive or baseline claim is written into the paper:
1. Re-derive our `fp16` (Full Cache) per-category scores from the parquet; confirm they land on TurboQuant Table-1 Full-Cache values (Avg ≈ 50.06, Code ≈ 46.28) within reasonable tolerance.
2. Re-derive our `turboquant_mse` row; confirm it reproduces its Table-1 row.
3. If either fails → STOP; transitivity is unlicensed until root-caused. Do not paper over a mismatch.

Only after the anchor is confirmed do the un-run baselines' published numbers enter the table as licensed reference rows.
