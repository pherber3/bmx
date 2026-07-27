# K4 paper shelf — banked paper-ready material (2026-07-25)

Consolidated at the pre-rental pause so nothing lives only in session scratch.
Each item names its source doc (the number-bearing record). Drafting is HELD
until all results are in (user's call); this is the shelf, not the draft.

## Method / theory section

- **Everett-tightness of the tier allocator** (foundational pass, ledger):
  the Lagrangian allocation is exactly primal-optimal at its achieved charge;
  vs the target budget the duality gap is ≤ κ*·Δ_max/C ≈ 0.002 bpe at C=1024
  (staircase perturbation function; requires grid-convexity of g, which
  `_tier_g` asserts; allocator can land slightly under budget at a plateau,
  never over). Frame the allocator as a Lagrangian slice of the
  rate-distortion frontier (κ_L ≡ the RD multiplier).
- **The Jensen residual is an exact AM/GM gap**
  (`docs/2026-07-25-k4-local-levers-results.md` §3 + estimation-axis pass):
  continuous waterfill equalizes per-direction distortion (AM=GM, ratio 1);
  the tier grid makes it non-flat; the 7.6–10.5× identity-check residual IS
  the discretization AM/GM gap — closed-form decomposition zero-bit (12–17%)
  vs funded-tier-rounding (83–88%). Optional 4-line emit in
  `k4_jensen_gap._discrete_readout` makes future runs self-documenting.
- **Transfer ceiling = mutual information** (Matrix Analysis Thm 7.8.21
  Minkowski + Thm 7.6.6 log-det concavity; HDS §15.3 eq. 15.50 uses the same
  inequality to bound I(Z;J)): the corpus-transfer ceiling
  (1/C)(log det Σ̄ − E_s log det Σ_s) is (2/C)× a sequence-identity mutual
  information — intrinsic, corpus-size-independent. Measured: r_pred
  debiased 0.586 in the Gate-A band; honest bracket **[0.586, 0.72]**
  (effective-n n_eff≈848 correction overshoots the band → raw-n anchor
  retained as conservative; Wishart debias exact in the log-det first
  moment, exp step exact to O(1/C²)).
- **W-instrument scoping** (`docs/2026-07-24-k4-math-actions-results.md` §B):
  all G1 `logit_rope` numbers are frozen-instrument (fair codec-vs-codec,
  not the deployment metric); the causal instrument decides; sign-flip
  footnote; triangular weighting = the matched offset density of a causal
  window (measured AT the peak — every recency-discounted variant worse);
  mechanism for first-order size: 29/64 Llama rotary planes have wavelength
  > 64k so the odd sin·cos term never phase-averages. Scope note: the
  instrument covers full-rotary models (Llama/Qwen3); partial-rotary needs
  the non-rotated sub-block added.
- **Why scalar quantization on eigencoordinates** (MacKay Ch. 20 unequal-
  variance VQ failure modes + the 13-order spectrum): scalar-per-direction
  after KLT is the right structure; block-VQ would need variance-aware
  distance to not degenerate.

## Quantizer-choice account (the declination ledger, each with numbers)

- **Why uniform RTN (the shape-matched finding)**
  (`docs/2026-07-25-k4-lloyd-gate-results.md`): blanket Gaussian-Lloyd fails
  0.25/0.22× (two legs: ≥5-bit per-group MSE-scale beats a fixed codebook
  even on N(0,1); metric-dominant top directions are sub-Gaussian,
  λ-weighted Pearson kurtosis 2.58 — low tiers are heavy-tailed 4.2, Lloyd's
  low-tier 20–24% win is coarse-quantization). Per-tier mix certified
  +3.4–4.9% but confined to ~0.02% of the λ-metric — immaterial.
- **Unbiased coding reconciliation** (RD-axis pass): unbiasedness pays only
  under long aggregation (TurboQuant's NIAH regime); our per-position causal
  logit metric is low-aggregation → biased RTN's lower per-estimate
  distortion dominates (measured 0.197 vs 0.090 at b3). Predicts where the
  verdict would flip — falsifiable.
- **Entropy coding declined** (foundational pass): deployed-code entropy gap
  is real (1.2–1.8 bits — it IS the uniform-codebook inefficiency) but
  random access for paged decode forecloses stream coding; the fix belongs
  at the codebook, which was then measured and declined (above). One
  limitations paragraph.
- **Sign tier / ternary tier**: ternary non-convex vs measured g (dominated);
  1-bit sign raises c_used at the deployed path (+15 dirs ⇒ +0.059 bpe ≫
  the payload gain) and is never selected at the charge-aware point.
- **Linear shrinkage (LW/OAS)** (`docs/2026-07-25-k4-estimation-levers-results.md`):
  additive-lift mechanism, no small-n rescue; row truncation right-signed
  but no fuel (row-norm spread 1.24–1.49) at cost +0.4–3.2% distortion;
  log-domain shrinkage not licensed (two families dead, burden on the next
  design). Charge-aware allocation dead (budget knob walks the same locus).
- **V-side Gaussian codebook exactness**: √d-scaled post-Hadamard marginal
  at d=128: variance 0.999, kurtosis 2.957 — the Gaussian Lloyd codebook is
  exact-optimal for V with no asymptotics appeal.

## Accounting / deployment section

- **AutoQuantize framing**: `mixed_dec_charge` + `int8_tl` = operator-level
  mixed-precision PTQ with a CLOSED-FORM sensitivity (the certificate's
  per-column `added_i`), cleaner than diagonal-Fisher; footnote where the
  two sensitivities coincide.
- **The cheap-analytic-instruments pattern, 3 validations**: int8 blanket
  (conservative 3–4×, layer 2 anti-conservative ≤3× — stated), int8 tier
  ordering (exact, 0/96), Lloyd blanket (near-exact magnitude). This is a
  methods contribution: offline certificates gating GPU spend.
- **Heavy-tail localization one-liner** (probability pass): whitened-tail
  kurtosis is CREATED by W^{-1/2} amplification in near-null query
  directions (layer-0-localized, funded mid-tier) — the principled account
  of the 0↔2 boundary fragility; ridge is RELATIVE (model-portable);
  whitener conditioning bounded by ridge cap (κ(W̃) ≤ ridge^{-1/2} ≈ 31.6).
- **Measured-ĝ hygiene**: `sampling_limited` flag reads "sub-Gaussian source
  OR sampling-limited" (cannot distinguish without shape stats); the
  measured high-tier values are real (platykurtic), do not pin analytic.

## PRACTITIONER CAVEAT (2026-07-26, from
## `2026-07-26-practitioner-adoption-review.md` — binding on the paper's
## deployment section)

- "Resident memory at speed parity" is **k2b_ph's** measured claim, NOT
  k4's: the spectral arm has no fused decode kernel (routes chunked BY
  DESIGN, `packed_streaming.py` — verified by the GH200 path probe) and its
  end-to-end ms/token is UNMEASURED at any scale. The paper must state the
  k4 systems claim as: resident memory (−20% vs fp16 @128k) with decode
  latency an open engineering item (fused spectral kernel = the named
  adoption blocker for serving personas). Do not let k2b's latency numbers
  stand in for k4's.
- k4 co-residency (seqs/GPU) is unmeasured (the 2.258 GiB/seq figure is
  k2b) — next-rental item alongside k4 chunked ms/token.
- Below ~5.6k context the pack charge makes k4's bits WORSE than tq_b3 —
  the crossover is the claim, never a blanket win.

## RENTAL ADDITIONS (2026-07-26 — every number now measured; the pre-refit
## section below is superseded by `2026-07-26-gh200-rental-results.md`)

- **Headline table row (the shipped recipe, measured)**: k4_b2.5_dec8tl
  LongBench macro 40.85 @ 3.081 mean bits — +0.48 over tq_b3 (40.37) at
  −0.125 bits; +0.13 over own fp32/frozen at −0.72 bits (the int8_tl
  accounting win as a single number). Macro convention mandatory. Category
  composition: synthetic +3.36 / code +1.40; language cats slightly favor b3.
- **NIAH figure**: honest null on Llama (all arms within noise at 32k/64k,
  5 seeds) = parity-at-fewer-bits; seed bars are codec-RNG bars (one fixed
  needle/length — stated limitation). Qwen: TQ-family collapse at 32k
  (tq_b3 → 2.60 vs fp16 10.00 flat) while k4 holds parity — cross-model
  robustness as a differentiator (n=1 model-pair, mechanism open).
- **Deployment section**: measured int8_tl 0.80/0.90% (Llama) and
  0.55/0.57% (Qwen) vs the 5% bar; certificate conservatism 5.3–8.6×
  (validations 3+4 of the cheap-instruments pattern); 128k census measured
  k4 chunked 50.48 GiB vs fp16 63.30 (20% below) with packed NIAH healthy
  at 128k; k2b 128k dense FITS post-mask-fix (83.31 — the June OOM was
  pre-fix).
- **Calibration story (the "general corpus" section)**: cross-domain
  penalty halves at 8B; token-marginal REVERSES at 8B on both models (order
  matters — scale-scoped narrative required); trigram count-table recipe
  CONFIRMS both models (D_tri 0.036–0.074 < 0.10) and the ladder
  self-terminates at order 3 by the pre-registered both-sides rule; G1
  passes from nc=1 (a single 2048-token cache) on both models, win curve
  monotone to nc=8. Privacy framing: ship general-text packs; the n-gram
  recipe needs order-3 statistics, not raw text.
- **Theory anchors extended**: Jensen debiased r_pred 0.684 at Llama inside
  the gpt2 band with the identity matching (0.008–0.040) — two-scale
  anchor; H3 basis non-transfer replicates at 8B (0.63–0.75 < 0.9).
- **Methods honesty items**: Gate-C amendment (drift-inexplicable flips;
  pre-registered by the duel doc itself); k3v2 = our asymmetric steelman
  (`5720588`); rotated-W is Llama-licensed (Qwen frozen-W by scope); the
  full incident record (rental doc §9) as ops-transparency material.

## THE NEGATIVE-RESULTS APPENDIX (2026-07-27 — "what we tried and what
## killed it"; the user plans this as an appendix or wherever it fits)

Source docs: `2026-07-27-storm-gates-results.md` (the battery, reviewed),
`2026-07-26-storm-kv-mechanisms-briefing.md` (provenance of the ideas),
`2026-07-26-breakeven-blindspot-audit.md`, the chronicle Parts III/VII
(the pre-storm negatives), and the quantizer declination ledger above.
Every entry is a pre-registered gate with a measured kill and a mechanism —
the appendix's thesis: **the codec's design is what remains after
everything else measurably failed.** Suggested grouping:

1. **Allocation/basis attempts (June era):** raw-basis waterfill (uniform
   wins 32/32 — funded channels aren't query-read); eigenbasis KLT (wins
   2.24× but +8 bpe per-sequence basis charge — the KV-side break-even
   law; the kill that DESIGNED K4's amortization); structured/streamable
   rotations (blockdiag 2× worse; frozen rotation drifts, eigengap ≈ 1).
2. **Codec-refinement attempts (July era):** charge-aware allocation (the
   plain budget knob walks the same locus); LW/OAS shrinkage (log-scale
   waterfill amplifies relative tail lifts; c_used explodes); Gaussian-
   Lloyd K codebooks (shape-matched RTN wins where the metric lives —
   the tangent-normal/sphere-Gaussianization account); blanket int8
   decoders (certified 54% implied degradation → rescued only tier-gated);
   mean-centering (+0.038 bit at matched bpe); unbiased/aggregation
   coding (2× variance never repaid; QJL-class collapses at softmax);
   entropy coding (foreclosed by paged random access).
3. **Alternative storage axes (the storm battery):** geometry-scored
   token selection (leverage ≠ attention: 37–48% of true attention mass
   dropped; the max-logit token itself dropped in 53–77% of cells; 28–68×
   distortion loss at spectral budgets); two-tier speculative prefetch
   (attention predictable only where static — content reads 0.36–0.46
   mass coverage); sequence consolidation (7.7–30× at matched bits;
   position-merging alone ~4.8× on the friendliest layer — token identity
   defended); recompute-from-hidden (2.0× fp16-KV bytes on GQA geometry,
   arithmetically); head-role priors (R²=0.08 on RoPE+QK-norm — the
   spectrum subsumes function); per-read query conditioning (average W is
   a sufficient statistic, gap 0.01–0.02 in-distribution); vMF/radial
   codebooks (the radial coordinate is already ~92%-captured by the top
   KLT direction); W_OV V-routing (GQA union fills the space).
4. **Validations-by-failure worth stating positively:** sinks are priced
   automatically by the W-metric (<0.5% of budget at 31–47% of attention
   mass); prompt-policy robustness PASSES (retention ≥0.987; chat wrap
   moves consumed moments ~30× less than document variation).
5. **The two flags the appendix must not bury:** the per-event metric
   FLAG (logit ranking is not a safe proxy in parity regimes; the
   K-instrument cannot see the V-budget axis; the p99-tail screen
   separates architecture fragility where means do not) and the
   still-untested per-position scale schedule (spec'd, predicted null).

Scope sentence for the appendix header: all battery results are mechanism
scale (gpt2 / qwen3-0.6b, n=2 small models); the pre-storm negatives are
gpt2-mechanism with Llama confirmations where noted in their docs.

## Numbers the headline tables cite (SUPERSEDED — pre-refit; see the rental
## additions above)

- Duel crossovers under certified accounting: k4_b2.5 vs tq_b3 ~5.6–5.7k
  (band → exact at refit); vs tq_k3v2 ~23.1–23.6k
  (`docs/2026-07-15-k4-duel-results.md` §3 supersession block).
- int8_tl: measured 0.75%/0.93%, saving 0.594/0.836 bits/token over uniform
  T=5; T_ℓ map layer1→5 others→6
  (`docs/2026-07-25-k4-estimation-levers-results.md` §2).
- Figures: `results/figures/` (bits-vs-context with certified band;
  cert-vs-measured scatter; corpus matrix + synthesis ladder; overlap).
