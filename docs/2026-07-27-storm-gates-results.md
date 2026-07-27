# Storm gates — battery results (2026-07-27)

Plan: `docs/superpowers/plans/2026-07-26-storm-gates.md` (all gates
pre-registered before running). Source brainstorm:
`docs/2026-07-26-storm-kv-mechanisms-briefing.md`. Strategic frame:
`docs/2026-07-26-program-map.md` (branch B timeboxed + branch C's audit).
All nine tasks ran locally at mechanism scale (gpt2 / qwen3-0.6b; banked
parquets for Task 1); commits on `feat/storm-metric-audit`
(`92dd83b` through `74303ab`); suite grew 651 → 741 passed (90 new offline
tests; no pre-existing test touched). Provenance convention note: each run
dir's recorded git SHA is the PARENT of the commit that introduced its
generating code (run-then-commit on a dirty tree); a dirty-flag in
bmx.artifacts is a hygiene candidate. Plan-level bars were pre-registered
in the plan doc; three delegated pre-registrations (T4 primary regression,
T8 X/k, T9 operationalization) live in experiment docstrings that landed
in the same commit as results — a weaker audit trail, noted.
Two host crashes interrupted the battery; recovery was clean both times
(per-task commits + on-disk partials; no data loss).

## Verdict table

| # | Gate | Pre-registered bar | Outcome | The number |
|---|---|---|---|---|
| 1 | Per-event metric audit | Spearman ≥0.8 vs logit ranking | **FLAG** | inversions to −0.74 (Llama NIAH; n=4 arms with near-tie recalls — adjacent-length sign flips are tie-noise-shaped, and FLAG is the conservative pre-registered outcome either way); tq V-axis invisible to the K instrument; Qwen collapse event-concentrated (total-failure 0.77); **p99/max tail separates Qwen fragility 2.9× where the mean gives 1.69×** |
| 2 | Leverage-score coreset | beat uniform at matched bits, ≥90% layers, ≥1 budget | **NEGATIVE** (mechanism measured) | 28–68× distortion loss at the spectral budgets (2.2–3); 1.6–17× at bits 2/4 where the uniform baseline is itself weak; 37–48% of true attention mass on dropped tokens; the max-logit token itself dropped in 53–77% of challenger (variant, layer, budget) cells (tie tolerance 1e-3) — **leverage ≠ attention** |
| 3 | Sink carve-out | ≥5% of distortion budget | **NULL (validating)** | sinks: 31–47% of attention mass, **<0.5%** of weighted-distortion budget — W prices sinks automatically |
| 4 | Head-role regression | R² ≥ 0.5 | **SPLIT** (lead: null) | qwen3 R²=0.082 (spectrum subsumes function on RoPE+QK-norm); gpt2 0.514 marginal/fragile; the QK-norm model shows **~190× smaller head-sensitivity spread (14,943× → 78×)** — attribution to QK-norm is the standing hypothesis (confounded by RoPE/GQA/d_head/scale), pending the on/off probe |
| 5 | Prompt-policy robustness | cross-policy retention ≥0.9 | **PASS (robust)** | retention 0.987–1.000, 28/28 layers; chat wrap moves the consumed second moments **~30× less** than document heterogeneity (matched pairing: Σ-gap 0.0061 vs 0.197; the 4v4 pairing gives ~54×: 0.0036 vs 0.197) |
| 6a | Oracle-vs-average W | gap ≈ 0 (<0.10) | **NULL** | 0.012 / 0.021 mean rel gap in-distribution — average W is a sufficient statistic on the fit distribution (the held-out-slice OOD gap 0.26–0.50 is distribution shift, T5's axis, per the verdict JSON) |
| 6b | Post-KLT residual predictor | ≥0.5 bit on ≥90% layers | **NULL** | +0.034–0.038 bit-equiv, 0% of layers |
| 6c | h-cache / MLA arithmetic | settle | **SETTLED** | h-cache = **2.0×** fp16 GQA-KV bytes (both 8B geometries); k4 pack 0.193×; MLA reference-only caveat produced |
| 6d | vMF radial codec | radial ≥ top-1 KLT | **NULL** | the per-head radial coordinate carries 92.5% *as much* weighted metric as the top KLT direction (0.802 vs 0.862; near-alignment implied, not directly measured) — spherical structure already inside the codec |
| 6e | W_OV read-subspace V | survive GQA union | **GQA-NULL** | union read-fraction 1.000 (per-head 0.890) |
| 6f | Mean-centering lever | ≥0.5 bit at matched bpe | **NULL** | +0.038/+0.034 bit at bpe gap 0.0000 (note: 6b's global-center row and 6f are the SAME computation under two ledger framings — one lever, counted once) |
| 7 | Break-even blind-spot audit | any untested free-currency class | **PARTIALLY CLOSED** | two surviving classes found: mean-centering measured null (6f); **per-position scale schedules remain spec'd-untested** (predicted null, not measured) — the taxonomy-closure claim is one cheap test short of its positive argument |
| 8 | Attention predictability | recall@k ≥0.9 (X=1%, k=5%S pre-registered) | **KILL** (at the registered cell) | 0.826/0.864 pooled; **content reads 0.62–0.72 recall, 0.36–0.46 mass coverage** — predictable only where static (sinks+recency = 64–68% of mass). Honesty note: at the grid's k=10%S corner (the top of the plausible budget) recall crosses 0.9 (0.902/0.922) — the kill is at the conservative registered cell, and the content-read decomposition is the load-bearing argument |
| 9 | Consolidation probe | hold within codec margin at matched bits | **FAIL (decisive)** | 7.7–30× loss; position-merging costs ~4.8× on the consolidation-friendly layer (1.6× on the steep layer) and its error exceeds the clustering error additively (0.404 vs 0.086); oracle-position VQ bridge still 5.4× — **token identity directly defended** |

Run-ids and per-gate detail: each task's run dir under `results/storm_*`
(config/env/SHA per run); commits `92dd83b` (T1), `3d7586b` (T3),
`7b53228` (T6), `feb10e4` (T2), `bf805be` (T4), `1cd2f90` (T5),
`131eb1f` (T8), `74303ab` (T9); T7 is the staged doc
`2026-07-26-breakeven-blindspot-audit.md` + 6f's number. (Commit 74303ab's
message says "7.7-26x"; the parquet maximum is 30.47× — this doc is
authoritative on the range.)

## What the battery means (the synthesis)

1. **The codec's design choices were re-validated five independent ways.**
   The W-metric automatically prices sinks (T3); the average query moment
   is sufficient (6a); the spectrum subsumes head function on modern
   geometry (T4); the spherical/vMF structure is already captured (6d);
   the KLT leaves no predictable residual (6b). The storm's twin
   unification ("climbing structure without naming it") now has five
   measured confirmations to cite.
2. **Every alternative storage axis TESTED measured dead at mechanism
   scale.** Geometry-scored selection (T2 — with the leverage≠attention
   mechanism), two-tier prefetch (T8 — static-only predictability, at the
   registered cell), consolidation (T9 — position merging is structural),
   recompute-from-hidden (6c — 2× arithmetic loss). The program-map's
   "per-entry coding is at its floor" verdict is now measured, not argued.
   Scope limits: attention-scored eviction (SnapKV/H2O family) was NOT
   tested — transitively covered by TurboQuant's tables; T2 kills the
   geometry-scored variant and explains why attention-blind selection
   fails. The plan's two deferred swings stay deferred with their gates
   now resolved AGAINST them: computed-shortcut head replacement was
   gated on T4 (role signal — null on the deployment geometry) and
   sequence-axis stacking touches T9's measured position-merging wall;
   neither is licensed at mechanism scale.
3. **The two live outputs are paper-affecting, not codec-changing.**
   (i) T1's FLAG: the quality tables carry a per-event column; the
   methods section states the K-instrument's V-axis scope limit; and the
   **tail-quantile (p99/max) fragility screen** is a new cheap instrument
   candidate — it separates the Qwen TQ-collapse where means do not, and
   feeds the collapse-mechanism note. (ii) T5's PASS: the deployment
   section gains a measured robustness claim (template wrapping barely
   moves the consumed moments) with its honest scope note (wrap ≠ content
   shift; content shift is the corpus-transfer gate's territory).
4. **QK-norm emerges as a cross-cutting thread**: it compresses the
   head-sensitivity spread ~200× (T4) and is the standing suspect for the
   TQ-collapse (rental doc §4). A small mechanism probe (QK-norm on/off
   at mechanism scale against tail-quantile fragility) is the natural
   next-rental rider.
5. **Nothing in the battery warrants a Llama-scale rental leg on its
   own.** T1's per-event analysis is already at full scale (banked
   parquets); T5's Llama confirmation can ride any future rental as a
   cheap rider; everything else closed negative.

## Caveats the record must carry

- Mechanism scale: n=2 small models (gpt2, qwen3-0.6b); Llama-scale
  behavior asserted nowhere. T5 additionally qwen-only (gpt2 has no chat
  template).
- T1's per-event granularity is LongBench-only (NIAH parquets carry no
  per-sample rows — a harness limitation already on the open-items list).
- T2/T9 charged the pack at model-level vs per-sequence for the
  challengers (the shipped convention; handicaps challengers only
  marginally and both lost by 1–2 orders).
- T8 is teacher-forced; free-running drift unmeasured (moot — the gate
  killed the mechanism).
- The instrument scoped by T1's FLAG is the same instrument underlying
  the T2/T3/T5/T6/T9 verdicts. The negative margins (28–68× at spectral
  budgets, 7.7–30×) dwarf the metric's demonstrated ranking noise — but
  the smallest T2 margin is 1.6× (qwen bits-2, where the uniform baseline
  is itself weak), and T5's PASS (the battery's one positive deployment
  claim) has no task-side/per-event confirmation; the flag stands for
  close calls and for T5's eventual Llama-scale rider.
