# K4 spectral codec — mathematical review (2026-07-24)

Scope: `src/bmx/cache/spectral.py` (whitening, eig, pack/charge arithmetic),
`src/bmx/cache/codecs.py` (`allocate_bits_from_variance`, `quantize_by_bits`,
`tier_bits`/`scale_bits`), `experiments/_k4_common.py`
(`greedy_layer_allocation`), read against the measured record
(`docs/2026-07-15-k4-duel-results.md` §1–3, §3b;
`docs/2026-07-23-k4-corpus-transfer-results.md` §5/§8). This is a theory
review, not a bug hunt: each finding states (a) the mathematical claim or gap,
(b) whether it is paper-ready / code-relevant / a dead end, (c) effort. No
code was edited. Notation: `C` channels, tiers `T = {0,2,3,4,5,6,8}`, spectrum
`λ_i` = eigenvalues of `W^{1/2} Σ W^{1/2}` (descending), model distortion
`g(b) = 4^{−b}`, per-direction distortion `D_i(b) = λ_i·g(b)`.

Summary verdicts up front:

| # | finding | verdict | class |
|---|---|---|---|
| 1 | discrete allocator: threshold-family-optimal, but round-to-nearest ≠ Lagrangian tier choice; optimality lemma available | CONFIRM with a fix | paper-ready lemma + 5-line change |
| 2 | allocator's budget ≠ the reported bpe: per-used-direction fixed charges (scale + decoder columns) are unpriced; at S=4k the unpriced charge is 2.1× the priced payload | GAP — the strongest lever found | code-relevant, big expected effect at short S |
| 3 | W is EXACT for the instrumented metric; the instrument is time-reversed and mis-weighted relative to real causal attention logits | CONFIRM (internal) + GAP (external) | paper-ready condition + 1-line fix + A/B |
| 4 | allocation optimality needs only: separability, grid-convexity, shared shape g; the 6 dB/bit constant never enters — only per-step ratios, and the true RTN ratios shift tier bands by up to ~0.9 log₂λ | CONFIRM structure, sharpen constants | paper-ready + cheap allocator upgrade |
| 5 | corpus result: exact conditional-moment decomposition + max-ent ladder formalizes "order-r suffices"; Markov concentration bounds the sampling half; an n^{−1/2} falsifiable prediction splits sampling from truncation | CONFIRM with a proposition | paper-ready |
| 6 | Gate-A's structural transfer ceiling is a determinant Jensen gap — an exact identity plus Minkowski concavity; predictable from cached moments | NEW THEOREM-SHAPED ANCHOR | paper-ready, testable |
| 7 | spectrum estimated at n/C ≈ 8–24: Marchenko–Pastur eigenvalue spreading biases the allocation tails and c_used | GAP | code-relevant (shrinkage), small-moderate |
| 8 | ridge/fp64/bisection numerics: sound, with one unswept knob (ridge) and one exact invariance worth stating | CONFIRM | mostly paper hygiene |
| 9 | int8 decoder: the distortion gate is exactly computable offline — a certificate, not a bound | CONFIRM cheaply | code-relevant, removes a VM dependency |
| 10 | small items: mean-centering lever, 1-bit tier, dec-not-enc accounting stance, layer-greedy convexity condition | mixed | minor |

---

## 1. The discrete allocator: what is provable, what is heuristic (Q1)

**What the code does.** `allocate_bits_from_variance` computes the continuous
reverse-waterfill `b_cont(λ) = max(0, ½log₂(λ/κ))`, rounds each `b_cont` to
the **nearest tier in bit-space**, and bisects `κ` for the smallest value whose
rounded mean is feasible. Two separate correctness questions:

**(i) The bisection is exact within its own family.** `b_cont` is monotone in
`κ`, and round-to-nearest of a monotone function is monotone, so the rounded
mean is non-increasing in `κ`; the feasible set of `κ` is an interval and
bisection converges to its left endpoint. So the allocator provably returns
the maximal-bits member of the *midpoint-threshold family*. Confirmed sound.

**(ii) The midpoint-threshold family does not contain the optimum.** The
discrete problem is: minimize `Σ_i λ_i g(b_i)` s.t. `Σ b_i ≤ B`, `b_i ∈ T`.
An exchange argument (swap `b_i < b_j` when `λ_i > λ_j`; distortion change
`(λ_i − λ_j)(g(b_j) − g(b_i)) ≤ 0` at unchanged budget) shows every optimum is
monotone in λ — a threshold solution. But the *optimal* thresholds are the
Lagrangian ones: switch tier `t_{j−1} → t_j` (gap `Δ_j`) at
`λ*_j = κ_L·Δ_j/(g(t_{j−1}) − g(t_j))`. Mapping to `b_cont` coordinates via
`κ_L = κ·ln4` gives the optimal switch point

    b_cont(λ*_j) = t_{j−1} + ½·log₂( Δ_j·ln4 / (1 − 4^{−Δ_j}) )

= `t_{j−1} + 0.443` for unit gaps and `t_{j−1} + 0.782` for the 2-bit gaps
(0↔2 and 6↔8) — versus the implemented midpoints `+0.5` and `+1.0`. The
implemented rule therefore under-allocates directions whose continuous
allocation lands in the windows `(0.443, 0.5)` (width 0.057 bits) and
`(0.782, 1.0)` (width 0.218 bits) above each boundary; the bisection then
refunds the saved bits globally, producing a provably dominated allocation at
matched achieved budget. The 0↔2 boundary matters most: at budgets 2.2–2.5 it
carries 194–263 of 1024 directions (duel §3b), so its window is populated.

**Optimality lemma (paper-ready, 3–6 lines).** *Let `g` be strictly
decreasing and grid-convex on `T` (per-bit marginal gains
`(g(t_{j−1})−g(t_j))/Δ_j` strictly decreasing in `j` — true for `4^{−b}` on
`{0,2,3,4,5,6,8}`, and for the measured RTN curve, see Finding 4). Then (a)
the Lagrangian selection `b_i(κ_L) = argmin_{b∈T} λ_i g(b) + κ_L·b` is
optimal for its achieved budget (Everett's theorem: any feasible allocation
with the same or smaller budget has Lagrangian objective ≥, hence distortion
≥); (b) grid-convexity lets each multi-bit tier step be split into equal-density
unit steps, so greedy-by-marginal-density traces the same solution family, and
for budgets between the family's achievable points the true optimum differs
from the Lagrangian solution by at most one direction moved one tier —
an `O(1/C)` bpe correction (`C = 1024`: negligible).* The knapsack caveat
(greedy-by-density is not optimal with variable step sizes in general) is
discharged exactly by grid-convexity; without it the lemma is false.

**Closed form vs the continuous relaxation (Q1's second half).** Under a
uniform-residual model (`b_cont − t` uniform over each gap, budget balanced in
expectation), midpoint rounding inflates distortion by
`E[4^{−u}] = (4^{δ/2} − 4^{−δ/2})/(δ·ln4)` per direction with gap δ:
**1.082 (≈ 0.057 bits) for unit gaps, 1.353 (≈ 0.218 bits) for the 2-bit
gaps.** This is the price of the tier set itself, paid by any rounding rule;
it quantifies what a finer tier ladder buys (Finding 10c prices the 1-bit
tier).

**Verdict / effort.** The allocator is a good heuristic, exactly optimal
within a slightly-wrong threshold family. Replace `_round_to_tiers(b_cont)`
with `argmin_{b∈T} λ·4^{−b} + κ_L·b` inside the same bisection (~5 lines,
bit-exact reproducibility of old packs preserved by keeping the old path
callable) → the lemma above then covers the shipped allocator verbatim.
Expected numeric effect: small (only the offset windows move, ≲ 1% of total
distortion for smooth spectra) — the value is the provable-optimality sentence
in the paper, plus honesty: today's text cannot claim optimality.

## 2. The allocator optimizes a different budget than the paper reports (new; strongest lever found)

Payload-v2 accounting (module docstring, `spectral_payload_bpe`) is

    bpe = (1/C)·Σ_i [ b_i + (16/group)·1[b_i>0] ]

and skeptic-v2 adds `dec_bits·c_used/S`, i.e. per direction
`+ dec_bits·(C/S)·1[b_i>0]` in the same per-direction-bit units. The
allocator prices a direction upgrade `0 → 2` at **2** bits. Its true price
under the accounting the paper reports:

| S | scale term | decoder term (fp16) | true price of 0→2 | allocator's price |
|---|---|---|---|---|
| model-level | 0.25 | 0 | 2.25 | 2 |
| 32768 | 0.25 | 0.5 | 2.75 | 2 |
| 4096 | 0.25 | 4.0 | **6.25** | 2 |

At S = 4k the unpriced fixed charge (4.25) is **2.1× the priced payload** —
the allocator systematically over-opens directions exactly where the §3 curve
loses to b3 (4k–8k). The duel's measured `c_used ≈ 830` at b2.5 feeds
`16·830/4096 ≈ 3.24` bpe of decoder charge on K (≈ 1.62 blended), the
dominant term in the 4k skeptic number (4.41 under v2).

**The fix is the same Lagrangian machinery with a fixed charge.** Per
direction, select `b_i = argmin_{b∈T} λ_i·g(b) + κ_L·(b + s·1[b>0])` with
`s = 16/group + dec_bits·C/S_deploy`. The fixed charge makes the
per-direction cost non-convex at 0, but per-direction argmin over 7 tiers is
exact enumeration, and Everett's theorem still gives optimality at the
achieved *total* charge (the lemma in Finding 1(a) never used convexity —
only (b) did, and greedy is not needed here). Directions whose entire
distortion `λ_i` is worth less than `κ_L·(2+s)` close; freed bits fund
survivors. Expected effect at S=4k: back-of-envelope, dropping the ~400
marginal directions whose λ sits within a factor ~4^2 of the drop boundary
cuts the decoder charge by ~0.8 blended bpe against a distortion increase
bounded by the dropped directions' λ-mass (measurable offline in the G1
instrument before any VM time). This attacks the ONLY region where K4 loses
the bits-vs-context duel; the crossover vs b3 (~10.2k under v2) should move
left materially, without invoking the int8 lever.

**Discipline note.** This is an *allocation* change, not an accounting
change: the bpe expressions stay frozen; `c_used` simply becomes smaller.
Packs become deploy-S-banded (`bits_b{budget}_S{band}`), which
`save_pack_file`'s labeling already accommodates structurally
(`layer_budgets` precedent). Effort: ~20 lines + a G1 offline sweep;
paper-ready as "deployment-context-aware allocation" with the same optimality
lemma. **This is the one finding I would act on first.**

## 3. The query weighting: exact for the instrument; the instrument is time-reversed relative to attention (Q2)

**(a) Internal exactness — CONFIRM, and the paper can claim it.** The repo's
measured metric (`logit_distortion`) is
`Σ_{t,s} (q_tᵀ R_s e_s)²` — stored **pre-RoPE** probe queries against
**post-RoPE** keys (`apply_rope` at read; Q never rotated: verified in
`metrics.logit_distortion` and every k2/k4 experiment call site). This is a
quadratic form in the errors: it equals `Σ_s e_sᵀ W e_s` with
`W = E_s[R_sᵀ q qᵀ R_s]` **exactly** — no cross-terms, because queries enter
only through their second moment. That is precisely what
`query_position_moment` computes (inverse rotation at strided absolute
positions). Three further exactness claims the code already licenses:

- **GQA pooling is exact**: summing squared logit errors over the query heads
  in a kv-group gives the pooled `W_j` — not an approximation.
- **Block-diagonality is exact**: logits never mix kv-heads, so `W` is
  block-diagonal per head even though `Σ` and the eigenbasis are full-C.
- **The whitening reduction is exact**: with `enc = W̃^{1/2}E`,
  `dec = W̃^{−1/2}E` (W̃ = ridge-floored W), `‖W̃^{1/2}(k − k̂)‖ = ‖y − ŷ‖`
  identically (E orthogonal), so per-direction MSE in code space IS the
  weighted metric, and the eigenbasis of `W̃^{1/2} Σ W̃^{1/2}` with
  reverse-waterfill is the textbook-optimal transform+allocation for it —
  under the standard conditions (orthogonal transform class, shared-shape
  scalar quantizers; Gaussian marginals for strict KLT optimality).

**(b) External gap — the instrument is not the attention logit.** The true
logit error is `(R_t q_t)ᵀ R_s e_s = (R_{t−s} q_t)ᵀ e_s` with causal offset
`m = t − s ≥ 0` — **forward** RoPE applied to the query at *relative* offsets
with a **triangular** aggregate weight (`#pairs at offset m ∝ S − m`). The
instrument (and hence W) uses `(R_{−s} q)ᵀ e_s` — the query's own rotation
frozen at zero, equivalent to offsets of the **opposite sign** with
uniform-strided weights. Per rotary plane with query moment
`M = [[a,c],[c,d]]` and angle φ = θ_j·offset:

    R_φ M R_φᵀ = cos²φ·M + sinφcosφ·(JM − MJ) − sin²φ·JMJ

The even terms survive either sign convention (differing only via the offset
*distribution*, uniform vs triangular); the odd term `sinφcosφ·(JM − MJ)`
**flips sign** under time reversal. So `W_implemented − W_true` is, per
plane, `2·E[sinφcosφ]·[[−2c, a−d],[a−d, 2c]]` plus the even-term
reweighting. For high-frequency planes `E[sin 2φ] ≈ 0` (phase averaging);
for low-frequency planes (rotary wavelength ≳ S — the slowest Llama dims at
S ≤ 64k) the error is first-order in the plane anisotropy `(a−d, c)`.
Precise statement of the condition under which the current W is correct
anyway: *per-plane query second moments isotropic within each rotary plane,
or all plane phases equidistribute over the sampled offsets.* Neither is
guaranteed; neither has been measured.

**(c) Status.** Not a bug — code and its own docstring/metric are mutually
consistent, and Gate B validated the current W empirically (weighted
increment 1.54–1.7×). But the paper's story "W is exactly the right weight
under the measured metric" must be scoped: exact for the instrument, an
approximation to attention. The fix to test: forward-RoPE the probe queries
(sign of `sin`), sample relative offsets with triangular weights
(one line each in `query_position_moment`), refit, and A/B on the
G1/Gate-B instruments. Since the corrected W matches the true metric's
quadratic form, it can only improve the *true*-metric win in expectation;
whether the instrumented metric or task scores move is an empirical
question. Effort: ~1 hour offline. Also record: attention-output/softmax
weighting (Fisher-style) is the yet-sharper objective but is per-sequence —
it breaks corpus-level packs, and is a **dead end** for this codec (note it
as considered).

## 4. What the allocation needs from the distortion model — stated precisely (Q3)

The allocation is optimal iff three conditions hold; `6 dB/bit` per se is
never one of them:

1. **Separability** — distortion sums over directions. Holds: groupwise RTN
   acts per column; scales are per (direction, 64-token group).
2. **Scale-equivariance + shared shape** — `D_i(b) = λ_i·g(b)` with a common
   `g`. Scale-equivariance is *exact* for RTN (the quantizer commutes with
   positive scaling, so distortion ∝ second moment). The shared-shape half is
   the fragile leg: it requires the normalized (unit-variance) distribution of
   each eigencoordinate to be the same. Heavy-tail heterogeneity across
   directions (top eigendirections are plausibly spikier) breaks λ-only
   ranking — this is the condition to state in the paper, and it is cheaply
   auditable: one calibration pass measuring per-direction per-tier empirical
   ratios `ĝ_i(t_j)/ĝ_i(t_{j−1})`.
3. **Grid-convexity of the true curve** — needed for Finding 1's lemma. It
   holds for the actual RTN-on-Gaussian curve: measured/classical
   optimal-uniform values `g = 1, 0.119, 0.0374, 0.0115, 0.0035, 0.0010,
   ~9·10⁻⁵` at `b = 0,2,3,4,5,6,8` have strictly decreasing per-bit marginals
   (0.441, 0.081, 0.026, 0.008, 0.0024, 0.0005). Note `g(0) = 1` is **exact**
   (a dropped direction's error is the coordinate itself), so the reverse
   waterfill's `min(λ, θ)` structure at the drop boundary needs no quantizer
   model at all — worth one sentence in the paper.

**Does the ranking survive the 4^{−b} deviation?** Yes — any common
monotone `g` yields the same monotone-in-λ threshold structure; the model
only enters through *threshold placement*. But the placement error is not
negligible: the true low-rate per-step ratios are ≈ 0.30–0.32 (asymptotically
`D ∝ b·4^{−b}` for optimal-step uniform quantization, ratio
`0.25·(1+1/b)`), not 0.25. The log₂λ band assigned to tier 2 under the model
is `log₂(0.469/0.047) = 3.32` wide; under the true curve
`log₂(0.441/0.081) = 2.44` — a **0.9 log₂λ (≈ 0.44 bit) misplacement of the
2↔3 boundary**, larger than Finding 1's midpoint-vs-Lagrangian offsets. Fix:
tabulate `ĝ` once (measured on calibration codes, mse_scale=True, group=64 —
NOT taken from Gaussian tables) and use it in the Finding-1 Lagrangian
selection. Effort: ~10 lines + one calibration pass; expected effect small
in total distortion (boundary bands only) but it upgrades the paper claim
from "6 dB/bit heuristic" to "optimal against the measured tier curve", and
it composes with Findings 1–2 (same allocator change).

## 5. The corpus result: a decomposition that formalizes the synthesis ladder (Q4)

**(a) Exact decomposition (no cross-terms).** Conditioning on the current
token `t` and using the law of total expectation:

    Σ = E_{t~p}[ μ_t μ_tᵀ ] + E_{t~p}[ C_t ],   μ_t = E[k|t], C_t = Cov(k|t)

— exact for any process; the cross-terms vanish identically. The caveat that
gives §5/§8 their content: `μ_t` and `C_t` are conditional moments **under the
process**, not model constants — context shifts move them. So "token-marginal
driven" is not a theorem about Σ; it is the *measured* statement that on these
corpora the `p`-weighted first term (plus short-range context) carries ~91%
(shuffle) / ~99% (bigram) of the codec-relevant mass.

**(b) The ladder is the max-ent hierarchy — proposition sketch (3–6 lines,
paper-ready).** *Let `P` be the true stationary token process and `P_r` its
order-r maximum-entropy surrogate (the (r−1)-order Markov chain matching P's
r-gram marginals; r=1 unigram sampling, r=2 the bigram chain). (i) If the key
map factors through the last r tokens, `Σ(P_r) = Σ(P)` exactly — Σ is a
linear functional of the r-gram law. (ii) In general the fit error splits as
`D_r = T(r) + ε_n`: a truncation term `T(r)` (mass of Σ carried by context
beyond the r-window, a property of the model, not estimable without running
it) and a sampling term `ε_n`. (iii) For the Markov surrogate, the empirical
second moment of an n-token sample concentrates:
`‖Σ̂ − Σ(P_r)‖_op = O(σ²·√(t_mix·(C + log(1/δ))/n))` (matrix Bernstein for
uniformly ergodic chains, Paulin 2015).* The measured ladder then has exact
interpretations: `D_shuf` = context destruction at zero sampling noise
(without-replacement, exact multiset); `D_uni − D_shuf ≈ 2 pts` = the
with-replacement multinomial noise at n = 4096; `D_bi ≈ 1–3%` = `T(2) + ε_n`.

**(c) A falsifiable prediction the harness can run this week.** If the
decomposition is right, `D_uni − D_shuf` scales as `n^{−1/2}` in calibration
length while `D_shuf` (pure truncation at order ∞ sampling) is flat. Doubling
the fit slices from 4×1024 to 8×1024 should cut the uni-vs-shuf gap by ~1.4×
and leave `D_shuf` unchanged. If measured otherwise, (b)'s decomposition is
wrong as an account of the ladder — kill it, don't keep the proposition.
Effort: one config change, CPU-scale rerun.

**(d) One remark the §4 tier-inversion licenses.** "Which channels get cut"
transferring better than "which channels get top bits" has a clean spectral
reading: tier boundaries live at fixed positions in `log₂λ`; the zero
boundary sits where the *cumulative* tail mass crosses the water level (a
coarse functional, stable under corpus perturbation), while top-tier
membership depends on individual leading eigenvalues whose corpus shift
`Δλ/λ` is largest (loading re-weighting concentrates in dominant
directions). A histogram of `log₂λ` with tier thresholds overlaid, per
corpus, would make this a one-figure explanation. Diagnostic, not a theorem;
cheap.

## 6. Gate-A's transfer ceiling is a determinant Jensen gap (new; the sharpest paper-ready item)

Gate A measured corpus-basis retention 0.56–0.69 of the per-sequence oracle
win, *flat under corpus tripling* — reported as "structural, not
data-limited". There is a two-line theory anchor for exactly this, with an
exact identity as a bonus.

Assume the high-rate continuous allocation (all directions active), fixed
shared `W̃`, shared shape `g`, pooled fit `Σ̄ = E_s[Σ_s]` over the sequence
population, pooled spectrum `λ̄` with water level `κ̄` (`g(b̄_i) = κ̄/λ̄_i`).

**Identity (unbiasedness of the pooled water level).** For a sequence with
whitened moment `Σ_s`, the pooled-basis codec's distortion is
`D_pool(s) = κ̄·Σ_i (EᵀΣ_sE)_{ii}/λ̄_i`. Taking `E_s` and using
`E_s[EᵀΣ_sE]_{ii} = λ̄_i`:

    E_s[D_pool(s)] = C·κ̄ = C·GM(λ̄)·4^{−B̄}    (exact)

The pooled codec loses **nothing on average** relative to its design point —
basis misalignment per se costs zero in expectation. The entire transfer
shortfall is on the oracle side:

**Bound (Minkowski).** `E_s[D_oracle(s)] = C·4^{−B̄}·E_s[GM(λ(Σ_s))]`, and
`det^{1/C}` is concave on PSD matrices (Minkowski's determinant inequality +
degree-1 homogeneity), so

    retention R = E_s[D_oracle]/E_s[D_pool] = E_s[det(Σ_s)^{1/C}] / det(Σ̄)^{1/C} ≤ 1

with the gap the **spectral Jensen gap** of the sequence-moment mixture — a
population functional, hence *corpus-size-independent*: exactly the measured
signature (retention 0.56–0.64 → only 0.61–0.69 under 3× corpus). Up to
second order it is the Gaussian sequence-identity information
`(2/C)·I(s; k)`-shaped quantity `(1/C)(log det Σ̄ − E_s log det Σ_s)`.

**Why this earns its place:** it converts the program's honest negative
("query-weighted bases don't transfer, structurally") into a quantified,
*predictive* statement — compute `R_pred = E_s[det^{1/C}(Σ_s)]/det^{1/C}(Σ̄)`
from the already-cached per-sequence moments and compare to the measured
0.56–0.69. Conditions to state: continuous high-rate allocation (at b2.2–2.5
~20–25% of directions are zero-bit — expect `R_pred` to be an optimistic
bound, not an equality), fixed W, and Gate A's per-cache win-ratio
aggregation vs this expectation-ratio (report both). If `R_pred` lands near
the measured band, the paper gets "the ceiling is the between-sequence
spectral heterogeneity, an intrinsic information quantity — no calibration
corpus fixes it"; if it does not, the gap is in the tier/zero-bit effects and
that too is worth knowing. Effort: ~30 lines against cached moments, no GPU.

## 7. Spectrum estimation at n/C ≈ 8–24: the allocation input is MP-biased (new)

Fit sizes on record: 4 caches × 2048 rows = 8192 (duel) to 24k (ablation)
against `C = 1024`, so `γ = C/n ≈ 0.04–0.125` — and token rows are
autocorrelated, so effective n is smaller. Sample-covariance eigenvalues at
these aspect ratios are spread relative to truth (Marchenko–Pastur):
top eigenvalues biased up, bulk/tail biased down, with multiplicative
fluctuation `O(√γ) ≈ 20–35%` ≈ **0.13–0.22 bits of jitter through
`½log₂λ`** — concentrated exactly at the 0↔2 drop boundary that sets
`c_used` (which Finding 2 makes the money term). Nonlinear shrinkage
(Ledoit–Wolf as a floor; even simple eigenvalue clipping at the MP edge)
before the waterfill is a 3-line, provably-variance-reducing correction to
the allocation input, and doubles as regularization for the corpus-transfer
setting (the `D_uni` sampling term of Finding 5 is the same mechanism seen
from the other side). Expected effect: fewer spuriously-opened tail
directions → smaller `c_used` → smaller decoder charge; direction of every
effect favorable, magnitude to be measured in G1. Code-relevant; half a day
including the audit.

## 8. Numerics and conditioning — mostly CONFIRM (Q5)

- **The ridge (`1e-3·λ_max` floor per W-block).** Two exact statements worth
  recording: (i) the codec optimizes the *floored* metric exactly — the
  whitening identity in Finding 3(a) holds for W̃, so no "amplification by
  `W^{−1/2}`" ever hits the weighted objective; (ii) what the ridge actually
  buys is robustness to W-*estimation* error and a bound on *unweighted*
  error amplification: `‖k − k̂‖ ≤ ridge^{−1/2}·λ_max(W)^{−1/2}·‖y − ŷ‖`,
  i.e. ≤ ~31.6× at 1e-3. The cost: floored (query-null) directions can still
  be funded if Σ is large there — bits spent where the true metric says
  worthless. The knob was never swept; Gate-B's heldout-query instrument
  measures exactly the right tradeoff. Effort: an afternoon; expected effect
  unknown (that is the point of the sweep).
- **fp64 moment discipline**: correct and necessary (`κ(T) ≤ κ(W̃)·κ(Σ) ≤
  10³·κ(Σ)`; fp64 eigh fine). `lam.clamp_min(0)` before `log₂` with the
  `1e-30` floor in the allocator: no issue.
- **Bisection**: 40 iterations over a ~28-log-unit bracket → final interval
  ~1e-8 log units; step-function plateaus are handled correctly by keeping
  the best feasible candidate. Confirmed.
- **`_mse_refine_scale`**: alternating minimization is monotone in MSE but
  converges to a local optimum in scale; with absmax init this is the
  standard, adequate choice. Not worth sharpening (known ≲1% from optimal
  step at these bit-widths). Dead end.

## 9. The int8 decoder gate is exactly computable offline (Q5)

`int8_decoder_roundtrip` produces a *deterministic* perturbation
`Δ = dec_int8 − dec`. The added reconstruction error on a row is `Δŷ` — not a
random variable. Therefore the added weighted distortion
`mean_rows ‖W̃^{1/2}Δŷ‖²` (and its cross-term with the payload error) is an
exact computable number per pack on calibration data — a **certificate**, not
a bound and not a VM measurement. Scaling argument for what to expect:
per-column int8 noise-to-signal `≈ crest²/(12·127²) ≈ 2ln C/193548 ≈ 7·10⁻⁵`,
versus payload distortion `g(2.5) ≈ 0.05–0.1` — three orders of margin; fp16
scale rounding (2⁻¹¹ relative) is noise on noise. Recommendation: compute the
certificate at pack-save time and assert `added/payload < 1%`; this closes
the distortion half of Lever 2's "pending quality gate" offline (task-level
confirmation can stay on the VM checklist per repo discipline, but the §3b
"accounting projection only" caveat can be upgraded to "distortion-certified"
for free). Effort: ~15 lines in `save_pack_file` or the G1 instrument.

## 10. Small items

**(a) Mean-centering (uncentered moments feed the waterfill).** Symmetric RTN
around zero on a direction with `|μ| ≫ σ` wastes exactly 1 bit
asymptotically (the sign level never varies), and the waterfill funds it by
`μ² + σ²` rather than `σ²`. A per-direction mean stored in the pack is
model-level (16·C bits total — zero in every accounting mode) and reduces the
allocation input to `σ²`. Prior evidence says the effect is small (corpus doc
§4: centering moves subspace overlap ~2%; in the eigenbasis the mean
concentrates in top directions that get 8 bits anyway) — but it is free and
one line to test in G1. Low expected effect; do it only as a sweep row.

**(b) The dec-but-not-enc skeptic charge has a principled name — use it.**
The skeptic charge counts the decoder matrix but not the encoder, which is
also required at decode time (every appended token is encoded). The clean
defense is the **source-coding stance**: the cache is a code stream; the
charge is what a *reader* of the stored cache needs (codes + scales + tier
map + dec). The encoder stays on the writer's side like the model weights
(which no baseline charges either). One sentence in the paper preempts the
referee; without it, a hostile reading doubles the pack charge (`32·C/S`) and
moves the §3 crossovers right. Currently the stance is implicit — make it
explicit. Effort: prose only.

**(c) A 1-bit tier is excluded by implementation, not math.** The assert
`1 not in tiers` is correct for symmetric RTN (`qmax = 0`), but a two-level
sign quantizer with MSE-optimal scale (`±E|x|`, `g(1) = 1 − 2/π ≈ 0.363` for
Gaussian) is grid-convex-compatible (marginals 0.637, 0.244, 0.081, …
decreasing) and has the *largest* per-bit gain in the ladder — it would be
picked precisely at the populated 0↔2 boundary, where Finding 1's closed form
says the current ladder pays its worst (~35%) discretization penalty.
Honest expectation: modest (the true `g(1)=0.363` is well above the model's
0.25, so the gain over jumping to 2 bits is limited); needs a 1-bit container
(`pack_codes` width-1 offset-binary — machinery exists). Moderate effort;
try after Findings 2/4 land, in the same G1 sweep.

**(d) `greedy_layer_allocation` optimality condition.** The layer greedy is
marginal analysis on *measured* distortion curves; it is optimal iff each
`s_l·D_l(·)` is convex on the budget grid (decreasing marginal gains). The
code neither checks nor enforces this; measurement noise can produce a
concave kink that stalls greedy on a dominated rung. Fix: lower-convex-hull
each layer's curve before the loop (3 lines) — restores the guarantee at zero
cost. Low stakes (allocation measured task-null; uniform is the headline
arm), but the fix is cheaper than the caveat.

## Dead ends considered (so nobody re-treads)

- **Softmax/Fisher-weighted objective** (attention-probability-weighted W):
  sharper in principle, per-sequence in practice — incompatible with
  corpus-level packs, which are the program's accounting edge. Dead end for
  K4; one sentence in the paper's limitations.
- **Lloyd/lattice codebooks on eigencoordinates**: already scoped out (spec
  §6); the ~1%-over-uniform constant plus Finding 4's measured-ĝ upgrade
  leaves nothing on this table.
- **Non-orthogonal transforms in whitened space**: could beat KLT only via
  non-Gaussian shaping; the shared-shape audit in Finding 4 subsumes the
  question — if shapes are near-Gaussian (audit says), KLT is optimal in its
  class and the point is moot.
- **Exact knapsack (DP) over tiers**: unnecessary — Finding 1's lemma gets
  within `O(1/C)` bpe of optimal with the Lagrangian rule; a DP would buy
  one direction's tier at C=1024. Dead end.

## Recommended order of operations

1. Finding 2 (charge-aware allocation) + Finding 1 (Lagrangian selection) —
   one allocator change, offline G1 verdict, directly targets the 4k–8k
   region where the duel is lost.
2. Finding 6 (Jensen-gap retention prediction) — pure analysis on cached
   moments; if it matches, the paper's honest negative becomes a theorem-
   anchored claim.
3. Finding 3(c) (causal-W A/B) + Finding 4 (measured-ĝ) + Finding 7
   (shrinkage) — one combined G1 sweep.
4. Finding 9 (int8 certificate) + 10(b) (accounting stance prose) — free.
5. Finding 5(c) (n^{−1/2} prediction) — rides the next corpus-transfer rerun.
