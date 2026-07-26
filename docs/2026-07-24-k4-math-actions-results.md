# K4 math-review actions — results (2026-07-24)

Provenance: spec `docs/superpowers/specs/2026-07-24-k4-math-actions-design.md`
(math `docs/2026-07-24-k4-math-review.md` #1/#2/#3/#4/#9), plan
`docs/superpowers/plans/2026-07-24-k4-math-actions.md`. All local
(gpt2 mechanism scale + qwen3-0.6b for the RoPE-dependent A/B + analytic);
Llama confirmation of survivors rides the next rental (§D). Run ids:
- A-gate: `results/k4_charge_alloc/20260724-101819-570b407/` (model-g),
  `results/k4_charge_alloc/20260724-111324-a7370f4/` (measured-g rerun)
- g-table: `results/k4_g_table/20260724-111214-a7370f4/`
- w_rope A/B: `results/k4_w_rope_ab/20260724-103946-5b8340f/` (circular
  logit_rope readout), `results/k4_w_rope_ab/20260724-110013-fb93851/`
  (third instrument: logit_causal)
- certificate: `results/k4_int8_certificate/20260724-113708-ff03bf6/`

**Verdict summary.** Three math-review levers, three local kill-or-confirm
verdicts: the charge-aware allocator is an **honest negative** (dead at gpt2
mechanism scale under the pre-registered bar, unchanged under a measured ĝ
table); the W-instrument RoPE correction is **confirmed by theory** (the
frozen A/B was circular — its instrument was the frozen arm's own objective —
and the non-circular causal instrument shows rotated wins 1.20×, so the Llama
rotated-W refit is REQUIRED); the blanket int8 decoder **fails its own
certificate** (implied rel_deg up to 54% vs the 5% line) but is rescued by a
tier gate (T=5 passes at ~0.021 both budgets with ~90% of the charge saving
surviving) — int8 misapplied, not dead.

## §A Charge-aware allocation (A-gate, finding #2)

Pre-registered rule (binding, from the spec): per (budget, S_ref) at MATCHED
skeptic-v2 bpe@S_ref, charge-aware heldout G1 win ≥ plain's, AND skeptic-v2
bpe@S_ref at matched win undercuts plain by ≥ 0.4 blended bits at
S_ref=4096 (blended = K-side/2; half the ~0.8-blended projection). Both
budgets fail ⇒ honest negative, allocator stays as-is. S_ref=16384
reported, not gated. Headline metric `logit` (gpt2 has no RoPE).

**VERDICT — HONEST NEGATIVE.** At S_ref=4096: b2.2 win_ca 5.562 vs
win_plain_at_matched_bpe 5.384 (win_not_worse=true), bits_saved_blended
−0.052 < 0.4; b2.5 win_ca 5.100 vs 4.914 (win_not_worse=true),
bits_saved_blended −0.063 < 0.4. a_gate_pass=false; the allocator stays
as-is. The charge lever is dead at gpt2 mechanism scale under the
pre-registered bar; recorded, not retried with moved goalposts.

**Why it fails (the mechanism, for the paper).** The projection that
motivated this lever compared same-budget packs. But the plain budget knob
*already walks the same (c_used, bpe) locus*: dropping directions to lower
c_used is exactly what a lower plain budget does. The charge-aware pack at
b2.2/S_ref=4096 lands at bpe 2.197 with c_used 268 — indistinguishable from
plain **b1.0** (bpe 2.195, `frontier.parquet` s_eval=4096), within 0.003 bpe.
The allocator IS doing the intended thing (see diagnostics), but the plain
frontier already reaches those (c_used, bpe) points for free, so the matched
comparison shows no bits saved. `win_not_worse` holds throughout — quality is
not sacrificed by the reallocation — there is simply no net bit win to bank.
Scope: gpt2-only; the ml-research adjudication found the negative robust to
the interpolation basis and the TQ grid (the gated points sit interior,
`extrapolated=false` at b2.5/s4096; frac_extrap=0 on the deciding comparison).

Diagnostics (reported, never gated):
- c_used vs S_ref (mean over layers, model-g): b2.2 plain 509.6 → s16384
  412.9 → s4096 268.1 (monotone_decreasing=true); b2.5 plain 548.7 → s16384
  454.9 → s4096 301.6 (monotone_decreasing=true) — the math doc's prediction.
- Tier-map shift (0↔2 boundary movement, mean n_t0/n_t2 per layer, model-g,
  from `diagnostics.parquet`): b2.5/s4096 vs plain b2.5 moves n_t0 219.3 →
  466.4 and n_t2 179.8 → 122.8 (c_used 548.7 → 301.6); b2.2/s4096 vs plain
  b2.2 moves n_t0 258.4 → 499.9, n_t2 192.2 → 116.7 (c_used 509.6 → 268.1).
  The allocator prices the decoder charge and drops directions across the
  0↔2 boundary exactly as predicted.
- bpe-vs-S frontier (`frontier.parquet`, the "does optimizing for 4k hurt
  64k?" curve) — the S_ref=4096 charge-aware pack evaluated at other S:

  | arm | budget | s_ref | s_eval | bpe | win |
  |---|---|---|---|---|---|
  | charge_aware | 2.2 | 4096 | 4096 | 2.197 | 5.562 |
  | charge_aware | 2.2 | 4096 | 8192 | 1.673 | 8.035 |
  | charge_aware | 2.2 | 4096 | 16384 | 1.411 | 9.659 |
  | charge_aware | 2.2 | 4096 | 32768 | 1.280 | 10.590 |
  | charge_aware | 2.2 | 4096 | 65536 | 1.215 | 11.089 |
  | charge_aware | 2.5 | 4096 | 4096 | 2.499 | 5.100 |
  | charge_aware | 2.5 | 4096 | 8192 | 1.910 | 7.709 |
  | charge_aware | 2.5 | 4096 | 16384 | 1.615 | 9.478 |
  | charge_aware | 2.5 | 4096 | 32768 | 1.468 | 10.510 |
  | charge_aware | 2.5 | 4096 | 65536 | 1.394 | 11.067 |

  No 4k-vs-64k tradeoff: bpe drops and win rises monotonically as s_eval
  grows past the optimized S_ref=4096 (skeptic charge amortizes the fixed
  decoder cost over more sequence; c_used is fixed once fit). The
  S_ref=16384 packs show the same shape shifted right (full curve in
  `frontier.parquet`).

### §A.1 Measured-ĝ rerun (finding #4; reported beside, not gated)

g_table = [1.0, 0.16997, 0.05033, 0.013198, 0.0030774, 0.00073008,
4.3638e-05] (tiers 0/2/3/4/5/6/8; n=4096 calibration rows; `g_table.json`).
Per-tier p10/p90 spread (the shared-shape audit): tier 2 [0.1483, 0.2055],
tier 3 [0.0431, 0.0630], tier 4 [0.0109, 0.0171], tier 5 [0.00253, 0.00406],
tier 6 [0.000598, 0.000963], tier 8 [3.58e-05, 5.75e-05] — tight per tier.

A-gate quantities under measured-g vs model-g at the gate points:

| | model-g 4^(−b) (570b407) | measured ĝ (a7370f4) |
|---|---|---|
| a_gate_pass | false | false |
| honest_negative | true | true |
| b2.2_s4096 win_not_worse | true | false |
| b2.2_s4096 bits_saved_blended | −0.0520 | +0.0100 |
| b2.5_s4096 win_not_worse | true | false |
| b2.5_s4096 bits_saved_blended | −0.0625 | +0.0185 |

**Negative UNCHANGED under measured ĝ** — the gate stays closed at both
points under both g-models (strengthens the bank). The constituent failure
mode flips (under 4^(−b): win_not_worse holds but bits saved is slightly
negative; under measured ĝ: bits_saved_blended turns marginally positive,
+0.01/+0.02, still ≫ short of the 0.4 bar, but win_not_worse now fails) — the
tier misplacement changes which arm's distortion is lower at matched bpe, not
by enough in either direction to flip a_gate_pass. The **fragile leg** of
finding #4 shows here: the 0↔2 boundary is where the measured table deviates
most from 4^(−b) (ĝ(2) ≈ 0.170 vs model 0.0625, a 2.7× ratio, off-model), and
that boundary is exactly the c_used "money term" the charge-aware allocator
acts on — so this is the more honest input even though it did not flip the
verdict. Note: the exact Lagrangian enumeration SUBSUMES the +0.443/+0.782
threshold-offset fix entirely (spec §A(i)) — there is no separate rounding
change to report.

### §A.2 Not included

Eigenvalue shrinkage (finding #7) is NOT included — it needs its own
validation design (future work, §E).

## §B W-instrument RoPE A/B (finding #3)

Substrate: qwen3-0.6b (controller-approved deviation from the spec's "gpt2" —
gpt2 has no RoPE, the two variants are provably identical there; pinned by
`test_query_moment_rotated_null_on_no_rope`). The pre-registered decision rule
was `|rel_win_delta| < 2%` on the frozen A/B → scoped claim, else the Llama
refit is REQUIRED.

**The pre-registered branch is SUSPENDED, explicitly.** The rule presupposed a
neutral referee. The instrument it read — `logit_rope` — leaves the probe
query un-rotated and scores it as a frozen, causally-unmasked quadratic form,
which is *exactly the objective the frozen arm's W is fit to match*. So the
A/B compared the frozen arm against its own objective and could not
distinguish frozen from rotated on independent grounds. Its `rel_win_delta`
(−20.5% at b2.2, −21.7% at b2.5, 56/56 layers agreeing, sign reversed from a
plain-logit spot check) is a **circularity signature**, not a measurement of
the correction. We state the identity rather than silently relabel it: the
old `decision`/`llama_refit_required` fields in the first run's verdict JSON
(`5b8340f`) were computed from this circular rule and are **superseded by the
causal readout below** — read them as the circularity record, not the finding.

**VERDICT — ROTATED FORM REQUIRED (via the third instrument).** The deciding
readout is `logit_distortion_causal` (`fb93851`): the true masked
per-position causal logit error `(R_{t−s} q_t)ᵀ e_s`, `s ≤ t`, with the
RoPE-composition identity `(R_t q)ᵀ(R_s k) = (R_{t−s} q)ᵀ k` pinned by
`test_rope_composition_absolute_equals_relative`. Direct ratio
`dist_causal_frozen / dist_causal_rotated` per (cache, layer), no TQ-curve
interpolation, no metric shared with either W's fitting objective:

| budget | ratio mean | ratio min | ratio max | dist_frozen | dist_rotated | n_layer_cache |
|---|---|---|---|---|---|---|
| 2.2 | 1.2004 | 1.1302 | 1.2703 | 0.036909 | 0.030787 | 28 |
| 2.5 | 1.2095 | 1.1480 | 1.2923 | 0.030339 | 0.025107 | 28 |

Rotated wins by ~20% (min 1.13) at both budgets, consistently in sign,
`third_instrument_verdict = "rotated_preferred_causal"`. This is the exact
mirror of the circular readout (which showed rotated looking *worse* under
the frozen metric): under the true masked causal metric — the one math review
#3(c) proves the rotated W is provably closer to in expectation — rotated is
the arm to ship. **The Llama rotated-W refit rides the rental as a REQUIRED
item (§D), justified non-circularly by this causal readout.** Per-rank
frozen-vs-rotated basis overlap (mean over 20 layers): rank 8 → 0.922, rank
16 → 0.921, rank 32 → 0.929, rank 64 → 0.932 — a moderate, not wholesale,
change of basis.

**PROGRAM-WIDE scoping.** All G1 `logit_rope` numbers across the K4 program
are **frozen-instrument** numbers — a fair codec-vs-codec comparison, NOT "the
deployment metric." The methods section must state this: `logit_rope` is the
frozen quadratic form W is fit to, so it is the right axis to compare two
codecs on equal footing, but it is not the true causal-attention logit error;
the causal instrument above is.

Methods-section footnote (enters the paper either way): the instrumented W
freezes the query's own rotation — relative to true causal logits the odd
sin·cos plane components enter sign-flipped and offsets are uniform-strided
rather than triangular; W is exact for the instrument, an approximation to
attention (math review #3(a)/(b)).

## §C int8 decoder certificate (finding #9)

Formula (exact, offline, per pack): `added = Σ_i lam_i·‖encᵀ Δdec[:,i]‖²`
with `Δdec = dec_int8 − dec`; `payload = Σ_i lam_i·4^{−b_i}`;
`implied_rel_degradation = added/(payload+added)` — the same axis as the
pre-registered VM gate `rel_degradation_int8 < 5%`. Analytic-identity test
pins the closed form to a brute-force weighted row norm to 1e-9.

**Blanket int8 FAILS its own certificate.** gpt2 packs (`k4_packs_gpt2`,
budgets 2.2/2.5): max noise_to_signal 1.175, max implied_rel_degradation
**0.540** (layer 1, budget 2.5), margin_factor 0.093 (< 1 ⇒ inside-out of the
gate), `certificate_far_inside_gate=false`. Per-layer table
(`metrics.parquet`, worst layers bold):

| budget | layer | c_used | added | payload | noise_to_signal | implied_rel_deg |
|---|---|---|---|---|---|---|
| 2.2 | 0 | 488 | 0.453 | 2.536 | 0.179 | 0.151 |
| 2.2 | **1** | 434 | 0.502 | 0.476 | **1.054** | **0.513** |
| 2.2 | 2 | 481 | 2.237 | 8.664 | 0.258 | 0.205 |
| 2.2 | 3 | 513 | 3.278 | 10.974 | 0.299 | 0.230 |
| 2.2 | 4 | 519 | 6.577 | 14.537 | 0.452 | 0.312 |
| 2.2 | 5 | 518 | 2.078 | 4.927 | 0.422 | 0.297 |
| 2.2 | 6 | 526 | 1.496 | 3.897 | 0.384 | 0.277 |
| 2.2 | 7 | 521 | 1.989 | 4.480 | 0.444 | 0.308 |
| 2.2 | 8 | 531 | 1.528 | 3.422 | 0.447 | 0.309 |
| 2.2 | 9 | 532 | 1.045 | 3.325 | 0.314 | 0.239 |
| 2.2 | 10 | 526 | 0.637 | 2.509 | 0.254 | 0.203 |
| 2.2 | 11 | 526 | 0.255 | 1.700 | 0.150 | 0.131 |
| 2.5 | 0 | 524 | 0.453 | 2.217 | 0.204 | 0.170 |
| 2.5 | **1** | 467 | 0.502 | 0.427 | **1.175** | **0.540** |
| 2.5 | 2 | 517 | 2.237 | 8.528 | 0.262 | 0.208 |
| 2.5 | 3 | 551 | 3.278 | 10.440 | 0.314 | 0.239 |
| 2.5 | 4 | 560 | 6.577 | 13.684 | 0.481 | 0.325 |
| 2.5 | 5 | 560 | 2.079 | 3.821 | 0.544 | 0.352 |
| 2.5 | 6 | 567 | 1.496 | 3.025 | 0.495 | 0.331 |
| 2.5 | 7 | 562 | 1.990 | 3.366 | 0.591 | 0.372 |
| 2.5 | 8 | 570 | 1.528 | 2.514 | 0.608 | 0.378 |
| 2.5 | 9 | 573 | 1.045 | 2.323 | 0.450 | 0.310 |
| 2.5 | 10 | 566 | 0.637 | 1.729 | 0.368 | 0.269 |
| 2.5 | 11 | 567 | 0.256 | 1.089 | 0.235 | 0.190 |

**Why this inverts the math doc's 7e-5 back-of-envelope.** The review's
estimate was **budget-averaged** (per-column int8 relative noise
`crest²/(12·127²)` vs the mean payload distortion). But `added` and `payload`
are both `lam_i`-weighted and dominated by the SAME top eigendirections, and
those directions sit at the top bit tiers (5/6/8) where `payload_i =
lam_i·4^{−b_i}` has already shrunk by 4^5–4^8 while the int8 decoder-storage
noise `added_i` has not shrunk at all. So the certificate is **tier-dependent
payload**, not budget-averaged: at low tiers int8 storage is negligible
(added/payload 3.9–19%), at tiers 6/8 it exceeds the payload it is meant to
serve (added/payload > 1). This is consistent with the measured **16.68%**
gpt2 dec_quant preflight (`results/k4_dec_quant/20260723-130005-b32de01`) —
both say blanket int8 is well over the line at these budgets.

**Tier-gated rescue (int8 misapplied, not dead).** Pure post-processing of
the same certificate: int8-store only columns with `bits ≤ T`, fp16 for the
rest (`tier_sweep.parquet`, `max_implied_rel_degradation` = worst layer):

| budget | T | frac_int8 | max_implied_rel_deg | eff_dec_bits | saving@S=4096 |
|---|---|---|---|---|---|
| 2.2 | 2 | 0.377 | 0.0013 | 12.99 | 4.49 |
| 2.2 | 3 | 0.629 | 0.0043 | 10.98 | 7.49 |
| 2.2 | 4 | 0.815 | 0.0095 | 9.50 | 9.70 |
| 2.2 | **5** | **0.916** | **0.0211** | **8.69** | **10.91** |
| 2.2 | 6 (=blanket) | 0.975 | 0.0564 | 8.22 | 11.61 |
| 2.5 | 2 | 0.328 | 0.0012 | 13.38 | 4.20 |
| 2.5 | 3 | 0.571 | 0.0045 | 11.44 | 7.33 |
| 2.5 | 4 | 0.771 | 0.0111 | 9.85 | 9.89 |
| 2.5 | **5** | **0.893** | **0.0210** | **8.88** | **11.45** |
| 2.5 | 6 (=blanket) | 0.969 | 0.0500 | 8.27 | 12.43 |

Largest passing T per budget = **5 at both** (0.0211 at b2.2, 0.0210 at b2.5,
each ~42% of the 5% line, margin ~0.029). At T=5, ~90% of used columns stay
int8 (frac_int8 0.916/0.893) and the surviving charge-saving fraction at
S=4096 is **91.6%** (10.91 of 11.91 bits/token/model) / **89.3%** (11.45 of
12.83). Excluding only the top ~1–2 tiers rescues the certificate to well
inside the gate while keeping ~90% of the saving — the **Lever-2 redesign
candidate** (int8-store tiers ≤5, fp16 for 6/8-bit columns). (T=6/blanket
fails at 2.2: 0.0564; at 2.5 lands exactly on the line, 0.0500 — inside by
numerical noise only, not a clean pass. The layer-uniform threshold binds on
layer 1 at every T; a per-layer threshold is a future note, §E.)

Honest limits (what the certificate does NOT capture): the diag(lam) model
of the code moment (payload-shift + payload × decoder cross-term, both O(g(b))
relative on the per-column noise base); query-distribution interaction beyond
the modeled second moment; task-level effects — the VM half of the ledger is
NOT certified.

**THE CERTIFICATE INFORMS THE USER'S DECISION; VM TASK 8 STAYS QUEUED UNTIL
THE USER EXPLICITLY RELEASES IT.** If released, the §3b "accounting projection
only" caveat upgrades to "distortion-certified" and Task 8's gate measurement
is replaced by this certificate (blanket int8 rejected; tier-gated T=5 the
candidate to confirm); task-level confirmation stays on the VM checklist per
repo discipline.

## §D VM addendum (one line per surviving item)

- w_rope Llama rotated-W refit — REQUIRED (not a spot-check), justified by the
  causal readout (§B): refit the K4 packs with `w_rope="rotated"` at Llama
  scale, rerun the G1/NIAH point, adopt the rotated numbers in the paper.
- int8 decoder: IF the user releases Task 8 on the certificate, drop the VM
  distortion gate and keep only the task-level confirmation row — confirming
  the tier-gated T=5 variant (blanket int8 already rejected offline).
- (The charge-aware lever does NOT ride — dead at §A, allocator stays as-is;
  no Llama refit for it.)

## §E Future work (each needs its own validation design; none started)

- Per-layer tier threshold for int8 (the layer-uniform T binds on layer 1;
  gating layer 1 tighter than the rest should improve both gate margin and
  surviving charge fraction — §C).
- Eigenvalue shrinkage before the waterfill (finding #7, MP-edge clipping /
  Ledoit-Wolf).
- The 'cheap analytic instruments' pattern that closed §C offline before any
  VM spend — the int8-certificate closed form and the determinant–Jensen
  Gate-A theorem (math review #6) are the same move (settle a distortion
  question on pack-side algebra, gate the rental on the answer); worth
  formalizing as the default first pass for any future lever.
