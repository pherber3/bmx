# K4 corpus-transfer gate — results (gpt2 mechanism scale)

> **SCALE-SCOPING (2026-07-26,** `docs/2026-07-26-gh200-rental-results.md` §5**):**
> the token-marginal verdict below (shuffle-null ≈ in-domain,
> `model_intrinsic_flag=True`) is a gpt2-scale result and REVERSES at 8B on
> BOTH Llama-3.1-8B-Instruct and Qwen3-8B: the shuffled-order null is WORSE
> than cross-domain transfer on every side (flag False both models) — word
> order in calibration text matters at deployment scale. Meanwhile the
> cross-domain penalty roughly halves vs gpt2, and the n-gram synthesis
> ladder CONFIRMS at order 3 (D_tri < 0.10 both models; ladder
> self-terminates). H3 and the top-tier findings below replicate at 8B.

Spec: `docs/superpowers/specs/2026-07-23-k4-corpus-transfer-design.md`.
Run: `results/k4_corpus_transfer/20260723-190823-8dced47` (git SHA `8dced47`).
Harness: `experiments/k4_corpus_transfer.py`. Fit budgets MATCHED by
construction: 4 slices × 1024 tokens per corpus (offsets 1024/2048/3072/4096);
eval = 2 held-out slices × 1024 per natural corpus (offsets 0, 5120). Shuffle-
null seed 20260723 (post-slice permutation, per-slice generator seed
`20260723 + offset`). Code corpus: `bigcode/the-stack-smol` is **gated** on
the Hub (no access from this environment) — fell back to
`codeparrot/codeparrot-clean-valid`, `split="train[:200]"` (Task 2's
row-count workaround for a real `load_eval_tokens` full-split
materialize-then-truncate memory bug, not a corpus-identity change — the
first 200 rows are a strict prefix of the full split, and only the first
~6144 tokens were ever consumed at these offsets, so the collected token
content is identical to what `split="train"` would have produced; see
`.superpowers/sdd/task-2-report.md`). Budgets 2.2 / 2.5; headline metric
`logit` (gpt2, no RoPE); win = per-pack bits-normalized TQ-curve ratio at
each pack's OWN `bpe_skeptic_deploy`. Independent recomputation of the
`code->wiki` @ b2.5 cell from `metrics.parquet` + `tq_curve.parquet` matched
the stored verdict to `<1e-9`; `extrapolated: false` on every cell, every
hybrid arm, and every W-cross arm at both budgets (576 total metric rows).

**YELLOW FLAG (every table below):** gpt2 scale = mechanism verdict only
(corpus-W retention ~0.47–0.52, `docs/2026-07-15-k4-duel-results.md`);
Llama fit-side replication is pre-registered (plan addendum) before any
paper claim.

## 1. Win matrix (win_mean; gpt2 mechanism scale — see yellow flag)

| fit \ eval | wiki-held | code-held |
|---|---|---|
| wiki | 9.736 / 9.207 (b2.2/b2.5) | 7.165 / 6.836 |
| code | 5.274 / 4.991 | 17.488 / 15.876 |
| null (shuffled) | 8.813 / 8.335 | 7.375 / 7.105 |

## 2. Cross-fit degradation D = 1 − win(cross)/win(matched) (§4 rules; gpt2 — see yellow flag)

| cell | b2.2 mean [min,max] | b2.5 mean [min,max] | label |
|---|---|---|---|
| code→wiki | 0.456 [0.409, 0.502] | 0.455 [0.408, 0.502] | domain-sensitive |
| wiki→code | 0.585 [0.540, 0.630] | 0.564 [0.519, 0.609] | domain-sensitive |
| null→wiki | 0.094 [0.076, 0.111] | 0.094 [0.074, 0.113] | insensitive |
| null→code | 0.573 [0.529, 0.617] | 0.547 [0.501, 0.594] | domain-sensitive |

Rule: D < 10% → corpus-insensitive; D > 25% → domain-sensitive; between →
as measured. model_intrinsic_flag (null≈wiki on wiki eval): **true**
(both budgets).

## 3. Hybrid (H3) + W-cross (gpt2 — see yellow flag)

| arm | cell | win_mean | recovery vs matched | h3_pass (≥0.9) |
|---|---|---|---|---|
| hybrid (b2.2) | basis wiki + alloc code → code | 7.433 | 0.425 | false |
| hybrid (b2.2) | basis code + alloc wiki → wiki | 5.514 | 0.566 | false |
| hybrid (b2.5) | basis wiki + alloc code → code | 7.110 | 0.448 | false |
| hybrid (b2.5) | basis code + alloc wiki → wiki | 5.268 | 0.572 | false |
| W-cross (b2.2) | Σ wiki × W code → wiki / → code | 7.740 / 10.864 | — | — |
| W-cross (b2.2) | Σ code × W wiki → wiki / → code | 6.915 / 12.332 | — | — |
| W-cross (b2.5) | Σ wiki × W code → wiki / → code | 7.331 / 10.203 | — | — |
| W-cross (b2.5) | Σ code × W wiki → wiki / → code | 6.541 / 11.369 | — | — |

Both hybrid arms fail H3 at both budgets (0.42–0.57, well under the 0.9
bar). Recovery is defined as `win_mean(hybrid) / win_mean(matched same-
corpus fit)` — independently confirmed against the verdict JSON's stored
`recovery` field to machine precision.

## 4. Mechanism diagnostics (overlap.parquet; gpt2 — see yellow flag)

- Per-rank subspace overlap (pair mean over layers, uncentered): wiki–code
  r=8 0.646, r=16 0.526, r=32 0.492, r=64 0.519; wiki–null r=8 0.814,
  r=16 0.764, r=32 0.770, r=64 0.790; code–null r=8 0.623, r=16 0.523,
  r=32 0.503, r=64 0.537.
  → H2 check: does divergence grow with rank? **No, not monotonically** —
  every pair's overlap DROPS from r=8 to r=16/32 (biggest single-rank
  component dominates and is shared) then partially RECOVERS by r=64
  (broader subspaces converge back toward each other, plausibly because a
  64-dim subspace of a 64-dim head captures most of the space regardless of
  fit corpus). The wiki–null gap stays far above wiki–code / code–null at
  every rank (≥0.76 vs ≤0.65) — the separation is by CORPUS IDENTITY
  (wiki shares more subspace with its own shuffled-token null (0.76–0.81)
  than with code (0.49–0.65)), not a rank-monotone trend.
- Centered vs uncentered at r=16 (wiki–code): uncentered 0.526 vs centered
  0.515 → H1 check: agreement drop when the mean/rogue component is
  removed? **Small (−0.011, ~2% relative)** — centering barely moves
  overlap, so the wiki/code subspace disagreement is NOT concentrated in a
  removable mean/DC term; it's distributed through the covariance
  structure itself.
- Tier agreement: top tiers (≥4 bits) 0.770 (wiki-code) / 0.679 (code-null)
  / 0.974 (wiki-null) vs zero-set Jaccard 0.871 (wiki-code) / 0.872
  (code-null) / 0.992 (wiki-null) (b2.5, layer-mean) → H2 as pre-registered
  predicted disagreement concentrates in the LOW tiers (the 0/2-bit pruning
  boundary). **Refuted — the measurement shows the opposite direction.** For
  wiki-code, the zero-bit (pruned) boundary agrees MORE than the top-tier
  boundary (0.871 vs 0.770), and wiki-null is likewise higher at the zero-set
  than the top tier (0.992 vs 0.974); code-null shows the same ordering
  (0.872 vs 0.679, the largest gap of the three). Disagreement concentrates
  in the TOP-tier allocation, not the low-tier/pruning boundary — the
  opposite of H2's low-tier-boundary prediction, and (combined with the
  already-killed divergence-grows-with-rank sub-prediction above) both halves
  of H2 as pre-registered are refuted. The supported finding is the
  inversion itself: "which channels get cut" transfers MORE across corpora
  than "which channels get the richest allocation" — coarse pruning
  decisions are closer to model-intrinsic, fine-grained allocation decisions
  are the corpus-sensitive part of the pack. wiki-null dominates both tiers
  for every pair, again pointing at corpus identity (natural vs shuffled) as
  the primary axis.
- Analytic cross-retention (D_own/D_cross, b2.5 layer-mean): wiki→code
  0.430, code→wiki 0.496, null→wiki 0.792, null→code 0.468 (diagnostic
  only — the cross term is measured in the src basis's whitened
  coordinates). This independently reproduces the win-matrix pattern:
  wiki↔code cross-retention sits at 0.43–0.50 (the basis+allocation fit on
  one natural corpus captures under half the achievable distortion
  reduction on the other), while null→wiki retention is 0.79 — closer to
  its own matched fit than wiki↔code are to each other. null→code (0.468)
  is as poor as wiki↔code, meaning code, not null, is the outlier corpus
  here — shuffled-token statistics transfer better to wiki than code's own
  natural statistics transfer either direction.

## 5. WHY (mechanism)

The four diagnostics converge on one story: **the top-rank basis is
model/token-intrinsic on wikitext, but code prose has a genuinely different
second-moment geometry that no rank or centering choice bridges.** The
win-matrix D's cleanly split into two regimes — null→wiki at D≈9% (inside
the <10% insensitive band, `model_intrinsic_flag: true`) versus code→wiki,
wiki→code, and null→code all at D≈45–58% (all comfortably past the >25%
domain-sensitive line, and all three straddle a similarly narrow band,
45–58%, rather than one being a clear outlier) — and every mechanism number
in §4 explains that split without contradiction. First, the per-rank
overlap ranks corpus pairs the same way at every rank tested (8/16/32/64):
wiki–null is always highest (0.76–0.81) and wiki–code / code–null are
always lower and close to each other (0.49–0.65) — this is exactly
`model_intrinsic_flag`'s null≈wiki finding, restated in subspace-geometry
terms rather than distortion terms. Second, that ranking does NOT sharpen
with rank (H2's naive form — "divergence should grow as you fit more of
the tail" — is killed): the gap is already present at r=8 and doesn't
widen; if anything, the biggest single component (r=8) is where wiki-code
and code-null show their highest overlap, with a dip through the middle
ranks and partial recovery at r=64. That is more consistent with a few
shared dominant directions (token-frequency / positional statistics common
to any English-derived tokenizer output) sitting on top of corpus-specific
mid-rank structure than with the "more rank = more divergence" story H2
predicted. Third, the centered-vs-uncentered probe at r=16 kills H1 as
stated: removing the per-corpus mean only moves wiki-code overlap by
~2% relative (0.526→0.515) — so the disagreement is not a removable
rogue/DC-offset artifact riding on top of an otherwise-shared covariance;
it is baked into the covariance shape itself. Fourth, tier agreement
REFUTES H2 as pre-registered rather than confirming it — H2 predicted
disagreement would concentrate in the LOW tiers (the 0/2-bit pruning
boundary), but the zero-bit/pruned boundary (which channels get cut
entirely) transfers noticeably *better* than the top-tier boundary (which
channels get the richest allocation) for the wiki-code pair (0.871 vs 0.770)
and is near-ceiling for wiki-null (0.992 vs 0.974) — the disagreement is in
the TOP tier, the opposite of what H2 predicted. Combined with the
divergence-grows-with-rank sub-prediction killed above, both halves of H2
as pre-registered are refuted; the supported finding is the inversion:
coarse pruning decisions are closer to model-intrinsic, fine-grained
allocation decisions carry the corpus-sensitive signal — which is exactly
what predicts the H3 hybrid failure below (swapping only the allocation
map, the corpus-sensitive half, onto a foreign basis cannot recover
matched-fit quality). Finally, the analytic
xretention ratios (D_own/D_cross ≈ 0.43–0.50 for every wiki↔code
direction, ≈0.79 for null→wiki) reproduce the empirical win-matrix D's
using a completely independent code path (whitened-space proxy distortion
rather than the TQ-curve win metric), which rules out a win-metric
artifact as the explanation — the domain-sensitivity is a property of the
fitted second-moment/basis geometry itself, not of how the win score is
computed downstream. Net: H3 (hybrid: shared basis + per-domain
allocation) also fails cleanly (recovery 0.42–0.57, both directions, both
budgets) — consistent with the tier-agreement finding that allocation is
the MORE corpus-sensitive half of the pack, so swapping only the
allocation map onto a foreign basis leaves most of the cross-domain loss
on the table. The W-cross split (§3) points the same direction: at b2.5,
`Σ_code × W_wiki → code` (11.369) sits well below the matched `code→code`
baseline (15.876) despite USING the code corpus's own covariance basis —
the wiki-derived whitener alone accounts for a large share of the loss,
confirming allocation/whitening geometry (not just the raw top-subspace
direction) carries the corpus-specific signal.

> This is the pattern the theory predicts. Activation outlier channels are
> persistent across inputs — always the same channels, with only their
> magnitudes moving between domains like chemistry prose and Python code
> ([[How does SmoothQuant address the activation outlier problem]]) — and
> pre-RoPE Q/K directions cluster around model-intrinsic centers that are
> nearly calibration-corpus-invariant ([[Q-K Concentration in Pre-RoPE
> Space]]). Mechanistically, the residual stream is additive, with token
> embeddings occupying a small fixed subspace ([[A Mathematical Framework
> for Transformer Circuits]]), so the key second moment inherits a
> token-marginal-weighted embedding component: shuffling the calibration
> text preserves that marginal exactly, while a domain shift re-weights
> which tokens — and hence which fixed channels and directions — carry
> mass, through Σ and the query weighting W alike. The same sensitivity is
> prior art in weight PTQ, where salient-channel identification is only as
> good as the calibration distribution ([[Compare GPTQ and AWQ]]). A fitted
> basis is, in the VQ taxonomy, an offline data-dependent codebook — corpus
> dependence is the price of beating data-oblivious rotation, whose
> coordinates are universal by construction ([[Vector Quantization
> Distortion Objectives]], [[Random Rotation Induces Beta-Distributed
> Coordinates]]).

Scope note: the vault prior above establishes channel-identity persistence
and center invariance, but it does not by itself quantify the
token-identity-vs-contextualization split of the key second moment — that
fraction rests on THIS measurement (the shuffle null), not on the vault.
The vault's domain-invariance evidence also concerns directional CENTERS
(a first-moment claim); the vault-consistent reading of the measured ~50%
domain effect here is that domain shift moves loadings and the residual
spectrum around model-fixed dominant directions, rather than moving the
directions themselves — consistent with the measured per-rank overlap
(0.49–0.65, wiki–code) sitting well above zero but well below the wiki–null
shuffled-token twin (0.76–0.81).

## 5b. Reconciling with the Llama task-level code parity

This result needs to sit next to a fact already on record:
`docs/2026-07-15-k4-duel-results.md` §2 measured wikitext-fit K4 packs at
LongBench **Code task-level PARITY** with TurboQuant b3 on real
Llama-3.1-8B (60.43 vs 60.02, delta +0.41, bootstrap CI [−2.21, +3.06]) — a
pack that never saw a line of code, on the one task category that is most
plausibly domain-mismatched, with code not collapsing relative to the other
categories. On its face that looks like it could be in tension with a
result showing ~45–58% relative-win loss crossing domains at gpt2 scale.
It is not — three candidate reconciliations, in order of defensibility:

(a) **The D-metric here is a RELATIVE win ratio, not an absolute-quality
comparison.** D measures how much of the *wiki-fit pack's own oracle
distortion-reduction win* survives when evaluated on code, relative to a
code-fit pack's win on code. A halved relative win can still leave the
cross-fit pack's absolute quality at or above the TurboQuant baseline —
which is exactly what the task metric (LongBench code_sim) compares. Losing
half of a large win over baseline can still mean "at or above baseline";
the duel's task-level number is an absolute comparison against b3, not
against an oracle code-fit K4 pack (which was never run on Llama code
tasks), so the two numbers are not measuring the same ratio and are not
contradictory even at face value.

(b) **Edit-similarity task metrics are far coarser than tail-logit
distortion.** This is the K1-era finding restated: ppl-adjacent /
downstream-task metrics can't attribute component choices the way a
distortion metric can. A pack can lose a large fraction of its *achievable*
distortion-reduction win on an out-of-domain corpus while the residual win
is still enough to clear a coarse, thresholded, edit-similarity task score
— especially against a baseline (b3) that has no corpus-fit component to
lose in the first place.

(c) **Scale.** gpt2 is the mechanism testbed here (yellow flag on every
table above) — a smaller model, no RoPE, no instruction-tuning. The
pre-registered VM addendum's Llama fit-side replication (§7 of this doc)
will measure the distortion-level cross-corpus effect directly at the scale
the duel's task numbers were produced at, closing this gap with a same-model
measurement rather than an analogy.

State plainly: **the duel's task-level parity stands as the
deployment-relevant fact** — a wikitext-fit K4 pack does not break Code on
Llama. This result does not overturn that; it refines WHERE the corpus
signal enters (the relative win magnitude, via token-marginal second-moment
statistics — §5's WHY) rather than showing wikitext packs are broken on
code. If anything, it predicts the opposite of a threat to the duel
headline: a domain-matched (code-fit) pack should RAISE the Llama code-side
win further above b3, not that the wikitext-fit pack's current parity is
fragile.

## 6. Verdict

### Template B — domain-sensitive

Fit corpus matters (figures below are the per-budget D *means*, i.e. the
range across the two budgets {b2.2, b2.5}; the wider per-cache min/max band
noted once at the end): `wiki→code` degrades 56–58%, `code→wiki` degrades
45–46%, and `null→code` degrades 55–57% — all three comfortably past the
25% domain-sensitive line, at both budgets, with `extrapolated: false`
everywhere. (Pooling both budgets' individual eval-cache values instead of
budget means widens each band to its full min/max: `code→wiki` 41–50%,
`wiki→code` 52–63%, `null→code` 50–62% — same conclusion, domain-sensitive
with margin.) Only `null→wiki` (7–11%) falls in the corpus-insensitive band,
confirming `model_intrinsic_flag: true` for the wiki/null pair specifically
— not a general "corpus doesn't matter" result. The exploitation lever is
decided by H3: hybrid recovery 0.42–0.57 < 0.9 at both budgets and both
transfer directions — allocation transfer is insufficient; the lever is
whole-pack per-domain fitting, not a shared basis with a swapped allocation
map. W-cross localizes the sensitivity to **both Σ (basis) and W
(whitener/allocation), with W carrying a substantial independent share**:
`Σ_code × W_wiki → code` (11.37 @ b2.5) sits ~28% below the matched
`code→code` win (15.88) even though the covariance basis is code's own —
the wiki-fit whitener alone destroys much of the code-domain win. Llama
replication is pre-registered before the paper states this.

## 7. VM addendum (pre-registered — rides the next rental)

See the plan's "VM addendum" section
(`docs/superpowers/plans/2026-07-23-k4-corpus-transfer.md`): Llama-Instruct
fit-side replication (same matrix, matched budgets at S=2048), plus the
OPTIONAL LongBench-code probe cell (n=100, paired vs the wikitext-fit arm)
ONLY if H3 confirms here and there. **H3 did NOT confirm here** (hybrid
recovery 0.42–0.57 < 0.9 at both budgets) — per the plan's stated
condition, the LongBench-code probe cell is gated off pending the Llama
replication's own H3 result; do not run it speculatively.

## 8. Stage 2 — synthesis order (§3b addendum; run `20260723-220816-9d11538`, git SHA `9d11538`)

Motivation: Stage 1 measured D(shuf_wiki→wiki) ≈ 9.4% — the unigram token
histogram carries ~91% of the matched-fit win for English. Stage 2 asks
whether the literal deployment recipe (SAMPLE a calibration stream from a
traffic token histogram) works, on both domains, and whether order 2 buys
anything (spec §3b). Five fit-side arms at matched budgets (4 × 1024 tokens,
offsets 1024/2048/3072/4096; per-slice statistics; seed 20260723 for both
shuffle and synthesis, generator seeded `20260723 + offset`); code source =
the Stage-1 codeparrot fallback (identity with Stage-1 code windows). FULL
matrix rerun — all Stage-1 and Stage-2 cells share this run-id; Stage-1 D
cells reproduced against run `20260723-190823-8dced47` to < 1e-9 (Step-2
check), so §§1–7 above stand unchanged.

**YELLOW FLAG:** gpt2 scale = mechanism verdict only (corpus-W retention
~0.47–0.52, `docs/2026-07-15-k4-duel-results.md`); Llama fit-side
replication pre-registered.

### 8.1 Synthesis-arm win matrix (win_mean, b2.2 / b2.5; gpt2 — see yellow flag)

| fit arm \ eval | wiki-held | code-held |
|---|---|---|
| shuf_code | 5.270 / 4.985 | 15.411 / 14.033 |
| uni_wiki | 8.623 / 8.113 | 7.636 / 7.287 |
| uni_code | 5.085 / 4.810 | 15.019 / 13.635 |
| bi_wiki | 9.619 / 9.075 | 7.132 / 6.822 |
| bi_code | 5.260 / 4.958 | 16.934 / 15.351 |

(Reference matched cells, §1: wiki→wiki 9.736/9.207, code→code 17.488/15.876.)

### 8.2 Matched-side D + order-ladder rules (§3b; gpt2 — see yellow flag)

| eval side | D_shuf (control) | D_uni (recipe) | D_bi | rule (a) D_uni<10% | rule (b) gap-closed ≥ ½ |
|---|---|---|---|---|---|
| wiki | 0.094 (Stage-1 null→wiki) | 0.112 / 0.116 (b2.2/b2.5) | 0.009 / 0.012 | fail | pass |
| code | 0.116 / 0.114 (shufcode→code) | 0.140 / 0.140 (b2.2/b2.5) | 0.030 / 0.032 | fail | pass |

climb_to_order3: true (rule b on BOTH sides — the ladder is climbed
one measured rung at a time; no higher orders otherwise).

The shuf-vs-uni gap isolates with-vs-without replacement at matched order-1
statistics: on wiki, D_uni (0.112–0.116) sits 1.8–2.2 points above D_shuf
(0.094) — sampling WITH replacement from the fit-slice histogram loses a
small but consistent slice of win relative to a straight without-replacement
permutation of the same tokens, at matched order-1 marginal statistics. On
code the gap is similar in size (2.4–2.6 points, D_uni 0.140 vs D_shuf
0.114–0.116) despite code's D_shuf already sitting above wiki's — the
with-replacement sampling-noise cost is roughly constant in absolute D
across both domains, not a code-specific problem.

### 8.3 Recipe verdict (delete the branch the numbers kill, per rule per side)

**(a) killed branch:** the sampled-unigram recipe fails for both wiki and
code — D(uni→E) is 0.112–0.116 (wiki) and 0.140 (code), both ≥ 10% at both
budgets, vs the shuf control at 0.094 (wiki) and 0.114–0.116 (code): shuf
passes (< 10%, "insensitive"/"as-measured" band) where uni fails on wiki,
and both sit in the same "as-measured" (10–25%) band on code but uni is
still further from the 10% line than shuf — so the loss is the
with-replacement sampling noise / multiset drift on top of the order
statistics, not the order-1 marginal itself. The histogram recipe does not
transfer at the literal-sampling level (order 1) on either side at gpt2
mechanism scale; per-domain NATURAL calibration text remains the fallback
UNLESS the bigram-order recipe below closes enough of the gap to serve as
the floor instead (it does — see the earns-keep branch next).

**(b) earns-keep branch:** order 2 closes 90.0–91.6% of the unigram gap on
wiki and 77.3–78.3% on code (≥ ½ bar on both sides) — the bigram chain is
the recipe floor there; order 3 is licensed for the ladder
(`climb_to_order3: true`, both budgets) since rule (b) passes on both eval
sides, not just one.

### 8.4 VM rider

If rule (a) confirms here, the pre-registered Llama A1/A2 replication
carries the five synthesis arms (exact flags in
`docs/superpowers/plans/2026-07-23-k4-synthesis-order.md`, VM NOTE). Rule
(a) did NOT confirm here on either side (§8.3) — the literal
sampled-unigram recipe is killed at gpt2 mechanism scale, so the histogram-
only privacy claim (ship histograms, not texts) is not licensed by this
result. Order 2 (bigram) earns its keep on both sides instead and licenses
climbing to order 3 (`climb_to_order3: true`); the pre-registered VM rider's
CONDITION was rule (a), not rule (b), so the synthesis arms do not
automatically ride the Llama A1/A2 replication under this outcome — a
bigram-recipe-specific VM extension would need its own pre-registration
before running on Llama.
