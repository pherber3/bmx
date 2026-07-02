# Paper Results-Section Skeleton (TurboQuant §4 structure)

**Date:** 2026-07-01. **Plan:** Task 3 deliverable. **Consumes:** Task 1 (claim ledger, `docs/2026-07-01-publication-readiness-assessment.md`), Task 2 (baseline memo, `docs/2026-07-01-baseline-parity-decision.md`).

Mirrors TurboQuant's §4 layout so the paper is a box-for-box comparison, then adds the two contributions TurboQuant does not make (systems + long-context edge). Each element is tagged `[HAVE]` / `[NEEDS-FIGURE]` / `[NEEDS-VM]` and carries a **stable ID** (F1–F4, T1–T3) reused verbatim by every downstream task. Producing task in brackets.

## Element IDs (the canonical registry — do not renumber)

| ID | Element | Mirrors TurboQuant | Status | Producing task |
|---|---|---|---|---|
| **F1** | Distortion (D_mse, D_prod) vs bit-width, overlaid with theoretical upper/lower bounds | Fig 3 | `[HAVE data, NEEDS-FIGURE]` | Task 8 |
| **F2** | NIAH length×depth recall heatmap per arm, + single aggregate Score | Fig 4 | `[NEEDS-VM]` (heatmap code exists; needs Score annotation + full sweep) | Tasks 4, 9 |
| **F3** | NIAH recall-vs-length line, one per arm, annotated with compression | (their prose) | `[NEEDS-VM]` | Task 9 |
| **F4** | Long-context crossover: k2b (3-bit K) holds fp16-parity at 32k–128k where turboquant_mse degrades | — (OUR addition, C2) | `[NEEDS-VM]` | Task 9 |
| **T1** | LongBench per-category table (6 cats + Avg), KV-Size(bits) column, our arms + anchor + attributed reference rows | Table 1 | `[NEEDS-VM]` (Code-only today; full 6-cat needs scorers) | Tasks 6, 10 |
| **T2** | Resident-memory: KV-slice (16→~3 GiB) headline + total-resident context + OOM-vs-completes | — (OUR addition, C3) | `[HAVE]` (re-confirm on final code) | Tasks 11, 11b |
| **T3** | Fused-kernel decode latency vs chunked PyTorch (feasibility framing) | — (OUR addition, C4) | `[HAVE]` (re-confirm) | Task 11 |

## Section-by-section skeleton

### §1 Empirical validation (mirrors TurboQuant §4.1)
- **F1** — measured D_mse / D_prod for `turboquant_mse` / `turboquant_prod` on **real caches** vs the paper's closed-form bounds (`√3·π/2·4^-b` upper, `4^-b` lower). `[HAVE data, NEEDS-FIGURE — Task 8]`
- **The wedge (our sharpening):** their worst-case bounds hold on real caches, but our structure-aware arms (`lowrank_rtn_channel` on pre-RoPE keys) sit *below* the curve — 2–3× better on the IP-relevant metric. This is the transition from "we replicate their theory" to "structure beats worst-case-optimal on real data." (C1 mechanism.)

### §2 Needle-in-a-Haystack (mirrors TurboQuant §4.2 / Fig 4)
- Setup: Llama-3.1-8B-Instruct, Fu et al. protocol, 4k→128k, ROUGE-1 `recall_full`, length×depth. Arms: `fp16`, `k2b`, `k2b_k2r8`, `turboquant_mse`, `kivi` (anchor + ours; §baseline memo). `[NEEDS-VM — Task 9]`
- **F2** — per-arm heatmap + aggregate Score (their signature figure). `[Tasks 4, 9]`
- **F3** — recall-vs-length lines. `[Task 9]`
- Headline (C1): at matched ~7×, k2b_k2r8 (8.47) beats turboquant_mse (7.88), short/mid context.

### §3 Long-context edge — OUR addition (C2)
- **F4** — the crossover. Short/mid (4k–16k): 2-bit matched arms win. Long (32k–128k): they collapse; canonical **k2b (3-bit K) holds fp16-parity** (32k: 6.95=6.95; 64k: 7.82≥7.28; 128k fresh: 10.0≥9.52). `[NEEDS-VM — Task 9]`
- The "bits belong to K" thesis, on TurboQuant's own benchmark. Honesty caveat (Task 1 C2): on the unified run turboquant_mse edges k2b at 64k within noise — frame as "holds fp16-parity," not "strictly dominates."

### §4 LongBench (mirrors TurboQuant §4.3 / Table 1)
- **T1** — full 6-category table (SingleQA / MultiQA / Summ / Fewshot / Synthetic / Code / Avg), **KV Size (bits)** column, LongBench-V1. Our arms + reproduced anchor (fp16, turboquant_mse) + attributed reference rows (PolarQuant/SnapKV/PyramidKV, per §baseline memo). `[NEEDS-VM — Tasks 6 (scorers), 10 (run)]`
- **Anchor gate (Task 10 Step 4):** fp16 Avg ≈ 50.06 + turboquant_mse Code ≈ 46 must reproduce, or the transitive baselines are unlicensed.
- Methods note (Task 1 §5.5): we do not middle-truncate — stated affirmatively (harder task, cannot inflate scores), not as a caveat.

### §5 Systems — OUR addition (C3, C4)
- **T2** (C3) — memory reported KV-slice-first (fp16 16 GiB → k2b ~3 GiB, ~5.3×), then total-resident as context (63.3/64.1/83.5 @128k, with the "fixed weights+activations mask the total" note), then the **OOM-vs-completes** capability result (dense OOMs at 94.06 GiB; packed completes). Full-generation peak (Task 11b/G8) is the headline number; prefill+4-step diagnostic corroborates. `[HAVE — Tasks 11, 11b]`
- **T3** (C4) — fused decode latency vs chunked PyTorch, **explicitly labeled "vs naive PyTorch baseline — feasibility, not a competitive-latency claim."** RTN 2624×, k2b 322× @131k. `[HAVE — Task 11]`

### §6 Limitations
- All six from Task 1 §5, verbatim: single-run; census workload (→ upgraded by Task 11b); kernel baseline is naive PyTorch; cross-model scope (→ actively extended, spec `2026-07-01-multi-architecture-extension.md`); LongBench proxy caveat; turboquant_prod faithful-but-losing.

### §7 Conclusion
- The wedge restated: on real caches, structure-aware allocation (pre-RoPE low-rank keys, asymmetric K/V bits) beats worst-case-optimal coding on the task metric AND holds long-context where a flat budget breaks — realized in resident memory and a fused kernel. Generalization to Gemma/Qwen3 in progress.

## Claim → section coverage check (Task 3 acceptance criterion)

Every claim is visible in ≥1 section, and every checklist-table element (Task-1 top-of-plan) has an ID + producing task:
- **C1** (recipe beats TurboQuant) → §1 (F1 wedge), §2 (F2/F3), §4 (T1). ✓
- **C2** (long-context edge) → §3 (F4). ✓
- **C3** (runtime memory realized) → §5 (T2). ✓
- **C4** (fused kernel feasibility) → §5 (T3). ✓
- No orphan figures/tables; F1–F4 and T1–T3 each resolve to a producing task. ✓
