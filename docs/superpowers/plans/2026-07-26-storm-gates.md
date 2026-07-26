# Storm gates — the pre-registered battery (2026-07-26)

> **For agentic workers:** execute task-by-task with fresh subagents
> (superpowers:subagent-driven-development). Every task pre-registers its
> gate BEFORE running. Source: `docs/2026-07-26-storm-kv-mechanisms-briefing.md`
> (the six-lens STORM brainstorm); strategic frame:
> `docs/2026-07-26-program-map.md` (this cycle = branch B timeboxed to the
> gate battery + branch C's audit, which gates the paper's tables).

**Goal:** run every offline, GPU-free storm gate at mechanism scale (gpt2 /
qwen3-0.6b local; Llama-scale confirmation rides a future rental only for
survivors). Honest negatives are results; each task commits its parquet +
verdict.

**Global constraints:** no web; no model downloads in tests (tiny factories
only; experiments may pull gpt2/qwen3-0.6b); all comparisons matched on
total bits, all metadata counted; the logit metric convention per CLAUDE.md;
gauntlet before every commit; commits per established per-task arrangement.

---

## Task 1 — Per-event metric audit (branch C; GATES THE PAPER TABLES)

Question: does ranking codecs by per-event task error (0/1 retrieval /
argmax events) agree with ranking by the logit-MSE instrument?
Data: banked parquets ONLY (results/k3_niah/*, results/k3_longbench/*
incl. samples.parquet per-sample scores, results/k4_frontier/*), split by
model, f9eeafe excluded per the rental doc's hygiene note.
Method: (a) per (model, budget-matched arm set): Spearman between the
frontier logit-distortion ranking and task-metric rankings (NIAH recall,
LongBench per-category and per-sample-event win-rate vs fp16); (b) the Qwen
case study: where in the per-sample distribution does the TQ collapse live
(uniform vs event-concentrated) and would any logit-side statistic
(tail quantiles of distortion rather than mean) have predicted it.
**Pre-registered gate:** Spearman ≥ 0.8 on Llama across the duel arms ⇒ the
logit instrument is a safe proxy — record and keep the shipped metric.
Spearman < 0.8 anywhere, or a sign inversion on any arm pair ⇒ FLAG: the
paper's quality tables must carry the per-event column alongside macro.
Either way: the Qwen analysis feeds the TQ-collapse mechanism note.

## Task 2 — Leverage-score token coreset vs uniform quantization

Question: at matched total bits, does keeping top-k tokens by leverage
score (of pre-RoPE K's top-r subspace; sinks always included) exactly, and
dropping the rest, beat quantizing all tokens uniformly — on the logit
instrument with real stored queries and RoPE-at-read?
Substrate: existing/regenerable local caches (gpt2 S=1024; qwen3-0.6b).
Accounting: coreset charged 16·k·C/S + ceil(log2(S))·k/S bits/token
(retained fp16 + indices) vs uniform arms at the same bpe; also a mixed arm
(coreset tail + quantized recent window).
**Pre-registered gate:** coreset (or mixed) beats the best uniform arm at
matched bpe on ≥90% of layers at ≥1 budget ⇒ CONFIRM (opens the selection
axis; Llama rides next rental). Loses everywhere ⇒ honest negative with the
mechanism (expected failure mode: softmax-denominator bias / worst-case
needle loss — report which).

## Task 3 — Sink carve-out audit

Question: what fraction of the query-weighted distortion budget does the
K4 codec spend on positions 0–3 (the sink tokens), and does exempting them
(stored fp16, excluded from the codec objective) reclaim measurable bits?
**Pre-registered gate:** sink share ≥5% of the weighted-distortion budget
at either budget ⇒ CONFIRM carve-out (spec the recipe change). <5% ⇒
honest null: the W-weighting already prices sinks correctly — record as a
validation of the metric.

## Task 4 — Head-role regression (does function explain the 35× spread?)

Question: do weight-readable per-head statistics (QK-circuit effective
rank; pre-RoPE Q/K concentration; local-vs-long-range preference derived
from weights — re-derived from weights, NOT trusted from the wiki note)
explain the measured per-layer/per-head sensitivity spread in the K4 packs?
Substrate: qwen3-0.6b (RoPE + QK-norm) + gpt2; existing packs/frontier
parquets for the sensitivity side.
**Pre-registered gate:** R² ≥ 0.5 ⇒ role is real allocation signal (spec a
role-prior lever). R² < 0.5 ⇒ honest null: the query-weighted spectrum
already subsumes function — a paper-strengthening statement either way.

## Task 5 — Prompt-policy pack robustness

Question: does a pack fitted on raw-template caches hold its G1 win when
scored on chat-wrapped caches (and vice versa) — the 43.7-pt axis?
Substrate: qwen3-0.6b local: collect raw vs chat-template caches (same
tokens budget), fit packs each way, cross-score with the frontier
instrument.
**Pre-registered gate:** cross-policy win retention ≥0.9 of same-policy ⇒
the shipped-pack story is robust to the largest measured shift axis.
<0.9 ⇒ FLAG for the paper's deployment section + spec a refit/shift-detector
note (refit is cheap: nc=1 suffices).

## Task 6 — Tier-2 closure batch (pack algebra + arithmetic; one task)

(a) Oracle-vs-average query gap: measured `Σ e_sᵀW e_s` with per-read
oracle W vs shipped average W — the second-moment-sufficiency prediction is
gap ≈ 0; record the number. (b) Post-KLT residual predictor: does a global
center/local-window predictor cut weighted residual energy ≥0.5 bit on
≥90% layers (random-access-safe variants only)? (c) h-cache vs K/V vs
MLA-latent bytes/token table for Llama-3.1-8B + Qwen3-8B geometry (pure
arithmetic; settles skeptic challenge 3 and produces the MLA caveat
sentence). (d) vMF radial check: fraction of weighted metric carried by the
per-head radial coordinate t = μᵀk vs the top KLT directions.
(e) W_OV read-subspace effective rank per head + V-energy routed through it
(GQA-null prediction). **Gates:** each sub-item pre-registers its threshold
in the task brief exactly as phrased in the briefing; expected outcome is
honest nulls that close the ledger — nulls are the deliverable.

## Task 7 — Break-even blind-spot audit (theory page)

Enumerate the correction classes the instrument prices vs the classes the
program shipped (additive low-rank; multiplicative scales; basis rotation;
deterministic decoder perturbation). For each shipped class outside
ε > 1−4^(−Δb), the one-line reason; then enumerate UNSHIPPED classes
sharing those reasons (per-token magnitude codes; learned bias vectors;
other affine/deterministic corrections) and mark each tested/untested.
**Gate:** any untested free-currency class found ⇒ spec its cheap test;
none ⇒ the "taxonomy closed" claim gets its first positive argument.
Reviewed by ml-research-reviewer like any results doc.

## Task 8 — Attention-predictability measurement (Tier-3 gatekeeper)

Question: is the top-k attention read set one-step predictable (the
prefetch/two-tier mechanism's life-or-death number)?
Substrate: local tiny/small models with attention capture during
generation on real text (gpt2 / qwen3-0.6b; no VM).
**Pre-registered gate:** one-step-ahead predictor recall@k ≥0.9 for tokens
carrying >X% attention mass ⇒ the two-tier mechanism graduates to a spec.
<0.9 ⇒ killed by prefetch-pollution before any engineering.

## Task 9 — Consolidation probe (superposed storage, one layer)

Question: on the layer with the flattest query-eigenspectrum, do m ≪ S
consolidated representatives at matched total bits hold needle
recall/logit distortion vs the per-token pack?
**Pre-registered gate:** holds within the codec's own quality margin ⇒ a
genuinely new axis opens (order-cost accounting next). Fails ⇒ the
token-identity doctrine gets its first direct positive defense.

## Deferred (spec-level only, post-battery): computed-shortcut head
replacement (circuits M3) and sequence-axis stacking — both gated on
Task 4/8 outcomes and scoped in the briefing.

## Order and cadence

Task 1 FIRST (paper-gating). Then 3, 6, 7 (pure algebra/analysis) in
parallel-ish; then 2, 4, 5 (need cache generation); then 8, 9. Each task:
fresh implementer subagent, pre-registered gate in the brief, results
parquet + short verdict section appended to a single results doc
(docs/2026-07-XX-storm-gates-results.md), reviewer pass at the end of the
battery. Local only; anything needing Llama-scale goes on the
next-rental list, not run here.
