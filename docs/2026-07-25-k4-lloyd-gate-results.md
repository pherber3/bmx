# K4 Lloyd payload-quantizer gate — results (2026-07-25)

Provenance: spec `docs/superpowers/specs/2026-07-25-k4-lloyd-gate-design.md`,
plan `docs/superpowers/plans/2026-07-25-k4-lloyd-gate.md`. Trigger: the
pre-rental foundational-textbook pass measured 44%/45% excess distortion at
tiers 2/3 for the K-side uniform-RTN payload quantizer vs the Gaussian
Lloyd-Max codebook the V-side already ships. Run ids:
- ĝ tables: `results/k4_g_table/20260724-185928-eef8205/` (rtn — reproduces
  the recorded 20260724-111214 table byte-identically),
  `results/k4_g_table/20260724-185949-eef8205/` (lloyd)
- gate: `results/k4_lloyd_gate/20260724-190035-eef8205/`

**Verdict summary.** The blanket Lloyd swap **FAILS its pre-registered gate
decisively — honest negative, banked with a mechanism that improves the
paper**: Lloyd wins exactly where the theory said (tiers 2–4, measured
20–24% per-tier) and loses 2–26× at tiers 5–8. Two independent legs explain
it (both re-measured for this doc, §2): (i) at ≥5 bits RTN's per-group
MSE-scale beats a fixed Gaussian-Lloyd codebook even on a pure N(0,1) source
(bit-depth, shape-independent); and (ii) the top directions — which carry
~99.8% of the λ-energy and thus dominate the λ-weighted logit metric — are
**sub-Gaussian / platykurtic** (λ-weighted Pearson kurtosis 2.58, i.e. the
"2.0–2.6" figure the pre-gate pass reported), so a Gaussian codebook is
structurally mismatched there and uniform RTN is near-optimal. (The low
tiers are heavy-tailed, not near-Gaussian — an earlier draft had that
backwards; corrected in §2. Lloyd's low-tier win is a coarse-quantization
effect.) The offline certificate agreed in sign and magnitude before the
measurement (predicted −3.8/−4.3×, measured win ratios 0.25/0.22). The
natural per-tier mix (Lloyd ≤4, RTN ≥5) was then certified analytically from
the same tables: **+4.85%/+3.39% payload-distortion reduction, confined to
tiers carrying ≈0.02% of the λ-weighted metric** ⇒
**certified-but-immaterial, declined without a measured gate** (§3 states the
materiality on the correct axis — the bare "0.03 bits/entry vs 0.3
bits/token/model" comparison is cross-unit and does not itself carry the
call). RTN stands as the shipped payload quantizer; the Llama refit is
unchanged; and the negative's mechanism upgrades the paper's "why uniform
RTN" from a default into a measured, shape-matched design choice.

## §1 The pre-registered gate (FAIL)

Rule (binding, from the spec): heldout win(lloyd)/win(rtn) ≥ 1.02 at BOTH
budgets on the existing gpt2 packs (same packs, same allocation — bpe
identical by construction, asserted per cell) AND measured ĝ_lloyd
grid-convex.

| budget | win_rtn | win_lloyd | ratio (need ≥ 1.02) | gate |
|---|---|---|---|---|
| 2.2 | 9.736 | 2.452 | 0.252 | **FAIL** |
| 2.5 | 9.207 | 1.988 | 0.216 | **FAIL** |

ĝ_lloyd grid-convexity: PASS (the convexity leg was never the problem).
`gate_pass = false`; `quantizer="rtn"` / `payload_quant="rtn"` stand.

## §2 The mechanism (measured three independent ways)

Measured per-tier relative distortion (calibration codes, same pipeline for
both arms; analytic = Gaussian-Lloyd reference on N(0,1)):

| tier | ĝ_rtn | ĝ_lloyd | analytic Lloyd-Gaussian | lloyd vs rtn |
|---|---|---|---|---|
| 2 | 0.16997 | 0.13554 | 0.11765 | **−20%** (win) |
| 3 | 0.05033 | 0.03836 | 0.03467 | **−24%** (win) |
| 4 | 0.01320 | 0.01051 | 0.00996 | **−20%** (win) |
| 5 | 0.00308 | 0.00377 | 0.00360 | +23% (loss) |
| 6 | 0.000730 | 0.001601 | 0.001439 | +119% (loss) |
| 8 | 4.36e-5 | 3.50e-4 | 2.54e-4 | +702% (loss) |

Three independent confirmations agree (the measured table; a zero-real-data
synthetic N(0,1) check; the end-to-end gpt2 run, where the live tier-8
regression reached 26×). The λ-weighted metric concentrates ~99.8% of
eigenvalue energy in the top-tier directions, so the high-tier loss
dominates and the aggregate collapses to 0.22–0.25×.

**Why RTN wins the top tiers (the paper-grade finding).** The rtn ĝ at
tiers 5/6/8 sits BELOW the analytic Gaussian-Lloyd optimum — impossible for
a Gaussian source, and previously flagged as sampling noise. The correct
reading has two independent legs, both re-measured directly for this doc
(gpt2 k_pre projected onto the shipped `enc` eigenbasis, per-direction
kurtosis over the 4 calibration caches, all 12 layers):

1. **Bit-depth (the dominant, shape-independent leg).** The crossover
   reproduces on a *pure* i.i.d. N(0,1) source with no real data (task-2
   report §"2"): at ≥5 bits, RTN's per-group (group=64) MSE-refined scale
   resolves local variance far more finely than a single fixed
   Gaussian-Lloyd codebook with one alternating-minimization scale. At 256
   levels (8 bits) locally-adaptive uniform quantization is already
   near-lossless and nonuniform spacing's marginal benefit is swamped.
   This alone predicts the tier-5/6/8 losses.

2. **Source shape (the corroborating leg — measured, with the sign
   corrected from the pre-gate pass).** The metric is λ-weighted and ~99.8%
   of the λ-energy sits in the tier-8 directions, so what the metric
   effectively "sees" is the shape of the top directions: their
   **λ-weighted mean Pearson kurtosis is 2.58** (Gaussian = 3.0), and the
   top-5-by-λ directions per layer run Pearson kurtosis mean 2.57 / median
   2.18 — i.e. **sub-Gaussian / platykurtic** in exactly the range the
   pre-gate pass reported as "2.0–2.6" (that figure is Pearson, not excess,
   kurtosis; excess ≈ −0.4 λ-weighted). A group-adaptive uniform quantizer
   legitimately beats a GAUSSIAN-Lloyd reference on such a flat-topped
   source, so leg 2 *adds* to leg 1 at the top tiers rather than opposing
   it. Caveat measured here: the tier-8 *tranche* as a whole is not
   uniformly platykurtic — its single highest-λ direction per layer is
   near-Gaussian (Pearson ≈ 2.9) and ~7 of 204 tier-8 directions are
   heavy-tailed outliers (excess up to +20); the sub-Gaussian statement is
   a λ-weighted / top-few-directions statement, which is the relevant one
   because those directions carry the metric.

   The low tiers are **not** "closer to Gaussian" — the earlier draft had
   this backwards. Measured Pearson kurtosis climbs monotonically as the
   tier drops: tier 2/3/4 run 4.19/4.21/4.15 (excess ≈ +1.1–1.2,
   leptokurtic / heavy-tailed), tier 5 → 3.9, tier 6 → 3.5, tier 8 → 3.2
   (mean) / 2.58 (λ-weighted). Lloyd's 20–24% win at tiers 2–4 is therefore
   a **coarse-quantization** effect (nonuniform levels help at 2–4 bits
   regardless of shape — the synthetic N(0,1) check wins there too), not a
   "these directions are Gaussian" effect. The net: RTN is well-matched at
   the top (few-bit-per-code, sub-Gaussian, metric-dominant) tiers where it
   matters, and the deployed all-RTN choice is near-optimal on the metric.

The `sampling_limited` flag's docstring is corrected to "sub-Gaussian source
OR sampling-limited" (it cannot distinguish without per-direction shape
statistics); the earlier recommendation to pin high-tier ĝ to analytic
Lloyd values is WITHDRAWN — the measured values are real. (Note: the
docstring's own inline "top eigendirections are platykurtic (kurtosis
2.0–2.6)" phrasing carries the same top-few/λ-weighted qualification as
above; this §2 is the authoritative statement.)

## §3 The per-tier mix — certified-but-immaterial, declined by rule

The measured tables license an exact offline certificate for the natural
follow-up (Lloyd at tiers ≤4, RTN at ≥5; zero metadata, static rule):
predicted payload-distortion reduction **+4.85% (b2.2) / +3.39% (b2.5)**
(Σλ·Δĝ over each pack's own bits; both figures recomputed from the two
measured g-tables and the packs' own bits — 4.853% / 3.385%, exact).

**Why it is immaterial (stated on the correct axis).** The naive
"log₄(1/(1−0.05)) ≈ 0.03 bits/entry vs the 0.3 bits/token/model bar" reads
as an order of magnitude, but that comparison is **apples-to-oranges** and
does not itself carry the declination — a hostile reviewer catches it
immediately:

- The `int8_tl` bar (0.3, shipped at 0.59–0.84) is a **charge saving,
  summed over all 12 layers, per token** — real bits removed at fixed
  quality.
- The mix saves **zero charge** (bpe identical by construction); its
  0.03-bits/entry is a *per-code, quality-equivalent* figure, not a
  per-token/model charge. Put on the int8_tl axis, the mix delivers no
  charge saving at all, and *crediting* its distortion headroom back as bits
  on the metric-dominant (tier-8) directions would come to ≈5 bits/token/
  model — which is **above**, not below, the 0.3 bar. So the raw
  0.03-vs-0.3 comparison is not just cross-unit, it points the wrong way
  once the units are fixed.

The declination survives on the axis that actually matters — **the
λ-weighted metric the gate scores.** The mix's entire +4.85%/+3.39%
reduction lands in tiers 2/3/4, which carry ~15–22% of the *certificate*
distortion (Σλĝ) but only ≈0.02% of the **λ-energy** — and the top-tier
directions (99.8% of λ, and thus ~99.8% of the logit metric's weight) are
left bit-identical by the mix. So the mix's predicted effect on the scored
logit metric is ≈0.02% relative, indistinguishable from noise — the same
metric on which the blanket swap already measured a 4–5× *loss*. That is the
honest materiality statement: not "0.03 < 0.3 bits", but "the improvement is
confined to directions carrying ~0.02% of the metric, so it cannot move the
gate." Per the program's materiality precedent (a lever whose ceiling is a
rounding-error fraction of the scored metric does not ship its codec
complexity), the mix is **declined without a measured gate** — the per-tier
codebook dispatch through quantize, containers, and both read paths is not
bought by a 0.02%-of-metric gain. Recorded as the honest declination beside
the sign-tier and entropy-coding declinations.

## §4 Instrument validation (the certificate, again)

The blanket-Lloyd offline certificate predicted −3.79×/−4.34× mean payload
degradation; the measured win ratios (0.252/0.216 ⇒ ×4.0/×4.6 distortion)
match nearly exactly — the certificate's third validated deployment (int8
blanket: conservative 3–4×; int8 tier sweep: exact ordering; Lloyd blanket:
near-exact magnitude). The cheap-analytic-instruments pattern now has three
independent confirmations across two different physical mechanisms.

## §5 Consequences

- **Refit unchanged**: RTN payload quantizer, `lam_alloc=None`,
  `dec_quant="int8_tl"` — the rental plan stands exactly as registered.
- The `quantizer=`/`payload_quant=` machinery stays (default-inert, fully
  pinned incl. packed containers at every tier) — the measured-ĝ
  infrastructure now supports per-quantizer tables if a future source-shape
  finding ever revives the question.
- Paper: the §2 shape-matched-codebook finding becomes the "why uniform
  RTN" paragraph; §3's materiality declination and §1's honest negative
  join the declination ledger; the aggregation-regime reconciliation
  (unbiased coding) and this result together give the paper a complete,
  measured account of its quantizer choices.
