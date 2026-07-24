# K4 estimation levers — results (2026-07-25)

Provenance: spec `docs/superpowers/specs/2026-07-25-k4-estimation-levers-design.md`,
plan `docs/superpowers/plans/2026-07-25-k4-estimation-levers.md`. All local
(gpt2 mechanism scale + analytic). Run ids:
- shrinkage gate: `results/k4_shrinkage/20260724-145553-9661897/`
- per-layer certificate sweep: `results/k4_int8_certificate/20260724-151346-6b9764c/`
- per-layer measured confirm: `results/k4_dec_quant/20260724-151413-6b9764c/`

**Verdict summary.** (1) Eigenvalue shrinkage of the allocation input is an
**honest negative, banked with its mechanism**: Ledoit–Wolf (the gated arm)
loses 34–37% of heldout win AND adds 0.14–0.16 bpe at both budgets, with no
small-n rescue anywhere on the n-scaling ladder — linear shrinkage toward the
spectrum mean is structurally wrong for a 13-orders heavy-tailed spectrum
feeding a log-scale waterfill. **The Llama rotated-W refit ships WITHOUT
shrinkage** (`lam_alloc=None` stands) — the refit-blocking question this
cycle existed to answer. (2) Certificate-derived **per-layer int8 tier
thresholds (int8_tl) SHIP** under both pre-registered bars: charge saving
over uniform T=5 is 0.594/0.836 bits/token/model at S=4096 (2–2.8× the 0.3
materiality bar) at measured rel_degradation 0.75%/0.93% (vs the 5% line) —
and the per-layer map dissolves last cycle's T=6 tension at depth: the
uniform-T=6 certificate failure was ONLY layer 1, which int8_tl holds at
T=5 while certifying T=6 everywhere else. The deployment-arm recommendation
updates to `int8_tl` (uniform `int8_t5` remains the simple fallback).
(3) The paper's bits-vs-context figure no longer shows the rejected blanket
projection; the certified tier-gated band replaces it, and the
certificate-vs-measured instrument-validation scatter becomes a figure (§3).

## §1 Eigenvalue shrinkage (finding #7) — HONEST NEGATIVE, refit unaffected

Pre-registered gate (binding): at BOTH budgets, heldout win(lw) ≥ 1.02 ×
win(plain) at matched skeptic-v2 bpe AND matched-budget bpe regression
≤ 0.02. Substrate: the corpus-transfer wiki split (4 fit caches, 2 heldout),
γ = C/n ≈ 0.19 at full fit rows — the deployment-like regime (Llama duel fit
γ ≈ 0.125). Arms: plain / LW (gated) / OAS (reported).

| budget | win plain | win lw | ratio (need ≥1.02) | bpe plain | bpe lw | regression (need ≤0.02) | gate |
|---|---|---|---|---|---|---|---|
| 2.2 | 9.736 | 6.179 | 0.635 | 2.614 | 2.750 | +0.136 | **FAIL** |
| 2.5 | 9.207 | 6.098 | 0.662 | 2.946 | 3.102 | +0.155 | **FAIL** |

OAS is uniformly slightly worse than LW. `gate_pass = false` at both
budgets; the allocator input stays raw; **no shrinkage rides the rental**.

**Mechanism (diagnosed, not just recorded).** The fitted spectra span ~13
orders of magnitude with ~90% of the shrinkage-intensity denominator d² in
the single top eigenvalue, so the estimated intensities are tiny (ρ_lw
≈ 4e-4 at full n) — the textbook LW failure mode on heavy tails. But even
tiny ρ is destructive here: shrinking toward μ̂ = mean(λ̂) lifts near-zero
TAIL eigenvalues by orders of magnitude relative to themselves
(λ' ≈ ρ·μ̂ ≫ λ̂ in the tail), and the waterfill — which reads the spectrum
on a log scale — funds those directions: c_used explodes from ~510/549
(plain, b2.2/b2.5 — reproducing the A-gate cycle's recorded values exactly)
to ~678/739 (lw), damaging win and charge simultaneously. The additive
shrink direction is exactly wrong for a multiplicative consumer.

**No small-n rescue.** ρ is monotone-decreasing in n as theory predicts
(0.0020 at n=768 → 0.0004 at full), but the win ratio stays in a flat
0.57–0.77 band across the entire n ladder {768, 1536, 3072, full} — the
anticipated "shrinkage only matters below deployment n" contingency did NOT
materialize; the negative is unconditional in the tested family.

**Scope (what this kills and what it does not).** Killed: LINEAR shrinkage
toward the mean (LW and OAS — the math review's own named floor), at
mechanism scale, in the tested n range. Not licensed either way: log-domain
or nonlinear (order-preserving-in-log) shrinkage, which does not share the
additive-lift failure mode — recorded as future work REQUIRING its own
validation design, not as a promising lead (the identity-check motivation
from the Jensen anchor stands, but two estimator families have now failed
to beat raw: the burden of proof is on the next design).

## §2 Per-layer int8 tier thresholds — SHIP (both bars pass)

Pre-registered rule (binding, both bars): measured rel_degradation(int8_tl)
< 5% at both budgets AND analytic charge saving over uniform T=5 ≥ 0.3
bits/token/model at S=4096; "certified but immaterial" was the expected
outcome and would NOT have shipped.

**T_ℓ map** (certificate bar 5%, identical at both budgets): layer 1 → T=5
(the known binder), all 11 other layers → T=6. The map is derived
deterministically from the pack at materialization
(`per_layer_tier_thresholds`, no new metadata; `dec_quant="int8_tl"`, recipe
suffix `_dec8tl`).

| bar | b2.2 | b2.5 | line | result |
|---|---|---|---|---|
| materiality: saving over uniform T=5 @S4096 | 0.594 | 0.836 | ≥ 0.3 | **PASS** (2.0–2.8×) |
| quality: measured rel_degradation | 0.755% | 0.935% | < 5% | **PASS** |

(Saving deltas at longer context scale as 1/S: 0.149/0.209 at 16k,
0.037/0.052 at 64k — the lever matters most exactly where the duel is
tightest, the 4–8k region.)

**This dissolves the T=5-vs-T=6 question at depth.** Last cycle recorded
"measured T=6 passes but its certificate fails — T=5 ships" as an honest
tension. The per-layer sweep shows the uniform-T=6 certificate failure was
entirely LAYER 1: held at T=5 per-layer, every other layer certifies T=6
individually. int8_tl is not a compromise between T=5 and T=6 — it is the
resolution that explains why both uniform readings were what they were.

**Deployment.** Own-bits win int8_tl = 9.863/9.342 — the highest of every
mode measured (above int8_t5's 9.845/9.316 and fp16's 9.159/8.646). The
deployment-arm recommendation updates to `int8_tl`; `int8_t5` stays as the
uniform fallback (one-integer spec, no certificate dependency at
materialization). Margin note (recomputed at final review): per-layer
certificate-to-bar margins at T_ℓ run min 1.1–1.3× / median 1.4–1.8× —
layer 1 at b2.5 sits essentially ON the 5% boundary (~1.0×) — vs uniform
T=5's min 2.4× / median ~3×. By construction the per-layer rule selects
each layer's LAST passing tier, so thin certificate margins are inherent to
it; the safety is carried by the certificate's aggregate conservatism (§3
refinement: 3–4× layer-mean, though layer 2 runs anti-conservative ≤3×)
plus the measured confirmation. The Llama refit re-runs
`per_layer_tier_thresholds` and re-checks both margins at scale before
trusting the map there (one CPU call).

## §3 Paper-figure integrity refresh

The committed `k4_bits_vs_context` figure drew the REJECTED blanket int8
accounting as its dashed "GATED PROJECTION" line. Refreshed: the certified
tier-gated band (mixed_dec_charge, gpt2 frac band [0.893, 0.916]) replaces
it, caption states the resolution; new figure `k4_cert_vs_measured` plots
the 96-point certificate-vs-measured scatter (log-log, y=x line) — the
instrument-validation evidence for the cheap-analytic-instruments pattern.

**Refinement found while making the figure (a real per-layer effect, not
noise):** the certificate is conservative for 87/96 cells, but LAYER 2
consistently measures 1.2–2.9× ABOVE its implied value (all 8 of its rows,
both budgets, every tier) — "uniformly conservative" (the phrasing in the
2026-07-25 local-levers doc §1, corrected there) is too strong; the honest
statement is: conservative in aggregate (layer-mean 3–4×) and for 10 of 12
layers, anti-conservative by ≤3× on layer 2 (all 8 rows) with a single
additional marginal excursion on layer 4 (b2.5, T=4, 1.007× — a tie at the
~0.24% noise floor). T-ordering is exact everywhere including layer 2. No gate outcome changes (per-layer margins at
the shipped thresholds are 5–10×, absorbing a 3× layer excursion), but this
is exactly why §2's margin note exists: the Llama refit re-runs
`per_layer_tier_thresholds` and re-checks the certificate margin at scale
before trusting the map there.

## §4 VM addendum deltas (rides the queued rotated-W refit; no new tasks)

- Refit fits with `lam_alloc=None` (shrinkage rejected, §1).
- Refit deployment arm: `dec_quant="int8_tl"` (pending user ratification of
  §2's promotion; `int8_t5` fallback). One extra CPU line at refit:
  `per_layer_tier_thresholds` on the Llama packs (records the Llama T_ℓ map
  and re-checks the certificate margin at scale).
- Unchanged riders: exact per-tier counts (`n_t0…n_t8` schema), per-cache
  logdet scalars for the Llama Jensen point.

## §5 Future work (recorded, not started)

- Log-domain / nonlinear spectral shrinkage — only with its own validation
  design; two linear families are now measured-dead (§1 scope note).
- Tier-grid densification near the 0↔2 boundary (Jensen identity check
  quantifies the coarse-grid cost).
- Per-layer certificate bar (the current 5% per-layer bar is uniform; a
  budget-aware per-layer bar is the next refinement if the Llama T_ℓ map
  turns out less degenerate than gpt2's 11-of-12).
