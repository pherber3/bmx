# K4 local levers — results (2026-07-24/25)

Provenance: spec `docs/superpowers/specs/2026-07-24-k4-local-levers-design.md`,
plan `docs/superpowers/plans/2026-07-24-k4-local-levers.md`. Trigger: USER
DECISIONS 2026-07-24 — VM Task 8 released on the int8 certificate (blanket
rejected offline), tier-gated T=5 promoted as the Lever-2 redesign, VM rental
deferred. All local (gpt2 mechanism scale + analytic). Run ids:
- measured dec gate: `results/k4_dec_quant/20260724-125348-0f49e32/`
- §3b recompute: `experiments/k4_charge_curve.py` on the §3 recorded inputs
  (nine `results/k3_niah/20260715-*` dirs +
  `results/k4_fit_packs/20260713-133628-798d0ef/metrics.parquet`)
- Jensen anchor: `results/k4_jensen_gap/20260724-132108-e03afb9/` (raw) +
  `results/k4_jensen_gap/20260724-133843-1ee62ad/` (Wishart-debiased columns)

**Verdict summary.** (1) The tier-gated int8 decoder **passes its
pre-registered measured gate decisively** — rel_degradation 0.50%/0.60% at
b2.2/b2.5 vs the 5% line, where blanket int8 fails at 13.5%/16.7% — and is
deployment-DOMINANT: at its own (cheaper) bits it wins MORE than the fp16
decoder. `int8_t5` ships as the Lever-2 arm. (2) The certificate that
licensed skipping the VM distortion gate is **validated as an instrument**:
conservative ~3× against measurement, ordering exact across 96
(budget, layer, T) cells, zero violations. (3) The §3b bits-vs-context
column built on blanket int8 is honestly superseded: under the certified
tier-gated accounting the k4_b2.5-vs-tq_b3 crossover is **~5.6–5.7k
(< the 8k target)**, vs ~5.2k blanket (rejected) and ~10.2k v2-fp16.
(4) The determinant-Jensen Gate-A anchor: mechanism fully confirmed
(Minkowski bound, exact W-cancellation, heterogeneity widens the gap at
every layer, corpus-size flatness), the pre-registered raw
|r_pred − r_discrete| ≤ 0.10 readout is **false as registered** — and the
diagnosis is quantitative: the raw population functional at this fleet's
aspect ratio carries a known, closed-form Wishart log-det bias (asymmetric
factor 0.623 between per-cache and pooled moments); removing it moves r_pred
from 0.365 to ≈0.586, **inside the measured Gate-A retention band
(0.56–0.69)** — a labeled post-hoc analysis, reported beside the registered
readout, connecting math-review finding #7 (spectrum estimation) to the
anchor.

## §1 Tier-gated int8 decoder — measured gate (PASS, promoted)

Pre-registered rule (binding, from the spec): measured
`rel_degradation_int8_t5 = 1 − win_int8_t5/win_fp16 < 5%` at budgets
{2.2, 2.5}, same harness/axis as the blanket measurement. Run
`20260724-125348-0f49e32` (gpt2 packs, cache `gpt2_1024`, modes
fp32/fp16/int8 + int8_t{4,5,6}, `logit` headline — gpt2 has no RoPE):

| budget | mode | win | rel_degradation vs fp16 | win at OWN bits |
|---|---|---|---|---|
| 2.2 | fp16 | 9.159 | — | — |
| 2.2 | int8_t4 | 9.135 | 0.26% | 9.774 |
| 2.2 | **int8_t5** | **9.113** | **0.50%** | **9.845** |
| 2.2 | int8_t6 | 9.077 | 0.89% | 9.861 |
| 2.2 | int8 (blanket) | 7.925 | **13.47% — FAIL** | 8.632 |
| 2.5 | fp16 | 8.646 | — | — |
| 2.5 | int8_t4 | 8.624 | 0.25% | 9.234 |
| 2.5 | **int8_t5** | **8.594** | **0.60%** | **9.316** |
| 2.5 | int8_t6 | 8.548 | 1.13% | 9.339 |
| 2.5 | int8 (blanket) | 7.204 | **16.68% — FAIL** | 7.898 |

`gate_pass = true` (int8_t5 at both budgets, ~10% of the 5% line);
`fp16_shippability_flag = false`; no extrapolated interpolation points. The
blanket b2.5 number reproduces the prior measurement (16.68%,
`results/k4_dec_quant/20260723-130005-b32de01`) exactly — axis
comparability confirmed.

**Deployment dominance (stronger than the gate).** At its OWN skeptic bits
(`mixed_dec_charge`, c_int8 = count(0 < bits ≤ 5)), int8_t5 wins 9.845/9.316
— MORE than the fp16 decoder's 9.159/8.646 at its bits: the ~7 bits/column
saved on ~90% of used columns buys more headroom than the 0.5% distortion
costs. Tier-gated int8 is not a tradeoff at these operating points; it is a
free improvement over the fp16-decoder accounting.

**Why T=5 and not T=6.** Measured T=6 also clears the 5% line (0.89%/1.13%)
but its CERTIFICATE fails (0.0564 at b2.2 — over the line; b2.5 lands
exactly on it). The promotion rule pre-registered both instruments: the
offline certificate is what licenses skipping VM distortion gates, so the
shipped variant must clear BOTH. T=6 clears one. It also buys almost
nothing: +0.016/+0.023 own-bits win over T=5. T=5 ships; T=6 is recorded as
a measured-passing/certificate-failing observation (the certificate's
conservatism, not a lost opportunity).

**Certificate validated as an instrument (the "cheap analytic instruments"
pattern, now with evidence).** Per-layer certificate
`implied_rel_degradation` vs measured `1 − win_T(layer)/win_fp16(layer)`
(`cert_vs_measured.parquet`, 96 rows):

- Ordering: measured T4 < T5 < T6 < blanket at every budget; `ordering_ok =
  true`; zero per-layer sign flips.
- Magnitude: certificate is a conservative over-estimate ~3–4× throughout
  (T=5 layer-mean implied 1.35%/1.39% vs measured 0.36%/0.43%; blanket
  implied 26.4%/30.7% vs measured 8.3%/10.7% — same factor) — consistent
  with its documented un-modeled second-order terms all sitting on the
  "over-count" side at these budgets. (The §1 gate numbers 0.50%/0.60% are
  the aggregate-win form 1 − win_t5/win_fp16; the per-layer means here are
  mean-of-ratios — both reported, same verdict.)

This is the empirical license for the pattern the MA results §E proposed:
an offline closed-form certificate whose errors are one-sided-conservative
can gate GPU spend. It predicted blanket's failure (confirmed 13.5%/16.7%
measured) and tier-gated's pass (confirmed 0.5%/0.6%) with zero VM hours.

The `int8_t{T}` variant is live end-to-end: `dec_quant="int8_t5"` parses
through recipes (`k4_b2.5_dec8t5`), materializes via the gated
`int8_decoder_roundtrip(tier_threshold=)` at pack load, and is charged by
`mixed_dec_charge` in streaming/packed accounting (endpoint-pinned
zero-numeric refactor: fp32 ≡ c_int8=0, blanket ≡ c_int8=c_used ≡ T=8 on
the {0,2,3,4,5,6,8} grid).

## §2 §3b supersession — the honest tier-gated bits-vs-context column

The duel doc's `skeptic-v2-int8` column assumed BLANKET int8 (dec_bits=8 on
every used column) — now rejected by certificate AND measurement. The
replacement column charges the certified mix through `mixed_dec_charge`
(int8 for the ≤T=5 columns + fp16 scale, fp16 for the rest — NOT an
effective-dec-bits value through `skeptic_charge`, which double-counts the
fp16-scale term by 16·c_used/(S·C)). Exact per-tier counts for the Llama
duel packs exist in no committed artifact, so the gpt2-measured frac_int8
band **[0.893, 0.916]** is applied to the Llama C_used as a labeled
estimate; `k4_fit_packs.py` now emits `n_t0…n_t8` per-tier counts so the
REQUIRED rotated-W refit records exact Llama numbers and the band collapses
to a point on the next rental.

Recomputed crossovers (k4_b2.5, linear interpolation in 1/S, same recorded
inputs as §3):

| comparison | v1 | v2 (fp16 dec) | v2-int8 blanket (rejected) | **v2-int8t5 band (certified)** |
|---|---|---|---|---|
| vs tq_b3 | ~13.1k | ~10.2k | ~5.19k | **~5.61k – 5.73k** |
| vs tq_k3v2 | ~61.8k | ~42.5k | ~21.4k | **~23.1k – 23.6k** |

The certified accounting gives back ~0.4–0.5k of the blanket column's
crossover (the fp16 columns it honestly re-prices) and still clears the
< 8k target against tq_b3 with ~2.3k of margin. The blanket column stays in
the historical table as a record, annotated rejected (edit in
`docs/2026-07-15-k4-duel-results.md` §3).

## §3 Determinant-Jensen Gate-A anchor (mechanism confirmed; registered
readout false; bias-diagnosed and post-hoc reconciled)

Theorem (math review #6, now in-code): with fixed shared W, pooled fit
Σ̄ = E_s[Σ_s], continuous high-rate allocation —
identity `E_s[D_pool] = C·GM(λ̄)·4^{−B̄}` (pooled-basis misalignment costs
zero on average; the transfer shortfall is entirely on the oracle side), and
Minkowski bound `R = E_s[D_oracle]/E_s[D_pool] =
E_s[det^{1/C}(Σ_s)]/det^{1/C}(Σ̄) ≤ 1` — a population functional,
corpus-size-independent. `jensen_gap_report` (fp64, eigvalsh, PSD-guarded)
+ `experiments/k4_jensen_gap.py` (per-cache moments via the SAME
`corpus_fit_bases` conventions, hoisted `per_cache_weighted_moments`,
zero-numeric pin). Substrate: 6 wiki + 6 code gpt2 caches, C=768, 12 layers.

**Confirmed mechanism (all four, gpt2):**
- W-cancellation: whitened r_pred bit-identical to raw — the theorem's
  frame-invariance, verified exactly, not assumed.
- Minkowski: r_pred ≤ 1 at every layer (and in every synthetic test).
- Heterogeneity widens the gap: mixed wiki+code r_pred 0.234 < within-wiki
  0.365, at EVERY layer — the corpus-transfer domain-sensitivity verdict and
  Gate-A retention are the same functional read at two mixture widths.
- Corpus-size flatness: r_pred@3-caches 0.405 vs @6 0.365 (Δ 0.040) — the
  population-functional signature that motivated the theorem.

**Registered readout — false, with quantitative attribution.** Layer-mean
r_pred (raw) 0.365 vs r_discrete 0.732 (b2.2) / 0.769 (b2.5): abs_gap
0.37/0.40 > 0.10, `match = false`. Two pre-stated effects account for it:

1. **Discrete-regime distance** (identity check): `mean_s D_pool_disc /
   (C·GM(λ̄)·4^{−B̄})` = 7.6× (b2.2) / 10.5× (b2.5), layer range 1.6–62×.
   Isolating contributions: the zero-bit directions' own share is a
   minority (≈4–18% depending on the attribution convention — the two
   independent decompositions run during review disagree on the split but
   not the conclusion); the dominant term either way is the tier-rounding
   penalty on FUNDED directions (4^{−b} snapped to the {0,2,3,4,5,6,8}
   grid vs continuous b_i), worst exactly at the eigenvalue-peaked early
   layers. The continuous-rate
   theorem's regime is genuinely far from the deployed tiered allocator at
   these budgets — worth knowing on its own (it bounds how literally the
   waterfill closed forms can be quoted for deployed packs).
2. **Wishart log-det bias** (the r_pred depression): per-cache moments are
   estimated at n≈1024 rows vs C=768 (γ≈0.75). The Gaussian closed form
   `E[log det Σ̂] − log det Σ = Σ_{i≤C}[ψ((n−i+1)/2) + log(2/n)]` gives
   det^{1/C} bias factors 0.583 (per-cache) vs 0.937 (pooled, n=6144) —
   an ASYMMETRIC factor 0.623 baked into raw r_pred by construction.

**Post-hoc debiased reading (labeled as such; the registered readout above
stands as registered).** Applying the closed-form correction (torch
digamma; exact under Gaussian iid rows, first-order under token
autocorrelation — it under-corrects):

Rerun `results/k4_jensen_gap/20260724-133843-1ee62ad/` (`jensen_gap_report`
gains `n_rows`; real per-cache row counts, never hardcoded; `post_hoc: true`
in the verdict JSON):

| quantity | raw | debiased |
|---|---|---|
| r_pred (layer-mean) | 0.365 | **0.586** |
| vs measured Gate-A retention band 0.56–0.69 | below | **inside** |
| abs gap to r_discrete (b2.2 / b2.5) | 0.366 / 0.404 | 0.145 / 0.183 |
| match (≤ 0.10) | false | false (post-hoc) |
| flatness delta (3 vs 6 caches) | 0.040 | 0.019 |
| mixed-domain r_pred (vs within) | 0.234 (vs 0.365) | 0.389 (vs 0.586) |

Bias factors: 0.5834 per-cache, 0.9367 pooled (bit-exact to the analytic
digamma hand-derivation). The debiasing closes ~60% of the r_pred-vs-
r_discrete gap on the layer-mean axis (gap-of-means; the per-layer
mean-of-absolute-gaps closes ~40%, 0.219/0.228 — both aggregations reported,
same verdict); the residual 0.15–0.18 is the SEPARATE discrete-allocator
mechanism (the identity check's tier-rounding term, item 1 above) — the two
effects are deliberately not conflated. Every mechanism signature survives
debiasing (flatness tightens; mixed < within at every layer). Token
autocorrelation means effective n < 1024, so the correction UNDER-corrects
— the true population r_pred sits somewhat above 0.586.

Analytically, 0.365/0.623 ≈ 0.586 — inside the measured Gate-A retention
band (0.56–0.69). Scope: that band was measured on Llama and on a
win-ratio axis; a gpt2-scale determinant functional landing in it is
CONSISTENCY, not cross-model confirmation — the Llama r_pred one-liner
rides the refit (§4) to close that loop. The paper's claim takes the
scoped form: *the transfer ceiling is between-sequence spectral
heterogeneity (a population information quantity, corpus-size-independent;
Minkowski-bounded, W-invariant), measured at gpt2 scale to sit in the
recorded retention band once the known finite-sample log-det bias is
removed; no calibration corpus fixes it.* The un-debiased match rule failed
as registered and is reported so; the debias correction is closed-form and
parameter-free, not a fitted rescue.

## §4 VM addendum deltas (rides the already-queued rotated-W refit)

- Llama refit records `n_t0…n_t8` per-tier counts (schema live) → §2's band
  collapses to exact Llama numbers; refit packs ship `dec_quant="int8_t5"`
  as the deployment arm (task-level confirmation row stays on the VM
  checklist per repo discipline — distortion is certified+measured, task
  quality is not).
- OPTIONAL one-liner at refit time: per-cache `(1/C)·logdet` scalars of the
  Llama fit moments → Llama r_pred at n/C≈8 (bias factor ~0.94, nearly
  clean) — upgrades §3's gpt2-scale claim to the paper model.

## §5 Future work (recorded, not started)

- MP/Ledoit-Wolf eigenvalue shrinkage before the waterfill (finding #7) —
  now doubly motivated: the allocation-input jitter AND the §3 identity
  check both sit at the same estimation boundary.
- Per-layer tier threshold T_ℓ for int8 (layer 1 binds the uniform T).
- Tier-grid densification near the 0↔2 boundary (the §3 identity check
  quantifies what the coarse grid costs in the continuous-comparison sense;
  a {0,1,2,…} grid would close part of it — needs its own distortion gate).
