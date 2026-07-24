# K4 math-review actions — design (charge-aware allocation, W-instrument A/B, int8 certificate)

Provenance: `docs/2026-07-24-k4-math-review.md` (committed 567fc40), findings
#2, #3, #9. User approved all three 2026-07-24 with the condition: the int8
certificate's numbers are REVIEWED BY THE USER before it may replace VM
Task 8's gate measurement (the VM task stays queued until then).

All work is local (gpt2 mechanism scale + analytic); Llama confirmation of
anything that survives rides the next rental. Nothing here touches shipped
packs, duel parquets, or the frozen streaming numerics.

## A. Charge-aware allocation (finding #2 — the ~0.8-bpe-at-4k lever)

**The gap:** the reverse-waterfill prices a direction's tier upgrade at its
payload bits only. Under skeptic accounting every USED direction also costs a
16-bit fp16 decoder column of length C (16·C bits per layer per sequence) plus
its group-scale share — at S=4096, C=1024 that is an extra ~4 bits/entry-
equivalent the allocator never sees, so it keeps directions whose marginal
distortion gain cannot pay their true storage cost at short context.

**Mechanism:** allocation cost of tier b for direction i becomes
`b·S_ref + (16·C + scale_bits(group)·S_ref/group_count_norm)·[b>0]` (exact
per-direction enumeration over the tier grid with an Everett/Lagrangian λ
search — the math doc's formulation is authoritative; ~20 lines beside
`allocate_bits_from_variance`, default-inert). Packs become (budget, S_ref)
parameterized: `s_ref: int | None = None` on the fitting path, `None` =
today's behavior bit-exact (pinned). Recipe alias `k4_b{budget}_s{S_ref}`
NOT added to CACHE_ARMS/recipes yet — fitting-path + experiment surface only
(the arm ships only if the gate passes and the user promotes it).

**Pre-registered gate (kill-or-confirm, gpt2 mechanism scale):** fit
charge-aware packs at S_ref ∈ {4096, 16384} beside plain packs, same corpus
caches (the existing gpt2 fit slices), budgets {2.2, 2.5}. Verdict per
(budget, S_ref): at MATCHED skeptic-v2 bpe evaluated at S_ref, the
charge-aware pack's heldout G1 win must be ≥ the plain pack's (quality not
sacrificed) AND its skeptic-v2 bpe@S_ref at matched win must undercut the
plain pack by ≥ 0.4 blended bits at S_ref=4096 (half the projection — the
projection itself is the optimistic bound). Both fail → honest negative,
recorded, allocator stays as-is. Diagnostics: c_used vs S_ref (expect
monotone decrease), the tier-map shift (expect 0↔2 boundary movement — the
math doc's prediction), and the bpe-vs-S curve of the S_ref=4096 pack
evaluated at OTHER S (does optimizing for 4k hurt 64k? the frontier
question — report the full curve, no gate).

**Also folded in (same allocator visit, from findings #1/#4/#7 — cheap,
severable):** (i) the optimality-lemma threshold offsets (+0.443/+0.782
in place of midpoint rounding) IF the exact enumeration doesn't subsume
rounding entirely (it does — note in the doc); (ii) a measured-ĝ table
option `g_table: tuple | None = None` replacing 4^(−b) ratios with measured
per-tier distortion ratios (default None = 4^(−b) bit-exact; ONE gpt2
measurement produces the table, reported beside the charge-aware gate);
(iii) eigenvalue shrinkage is NOT included (needs its own validation design
— future work, one line in the results doc).

## B. W-instrument RoPE A/B (finding #3 — scope the exactness claim)

**The gap:** the instrumented query moment freezes the query's own RoPE
rotation; relative to true causal-attention logits the odd sin·cos plane
components of W enter sign-flipped, and position offsets are uniform-strided
rather than triangular. The measured record is valid AS MEASURED; the paper's
"W is exactly the right weighting" claim must be scoped.

**Mechanism:** implement the corrected variant behind a flag on the W-moment
construction (`w_rope: str = "frozen"` default — today's path bit-exact —
vs `"rotated"` applying the query-side rotation per the math doc's
formulation). Fit gpt2 bases both ways on the same caches.

**Pre-registered readout (measurement, not a pass/fail gate):** heldout G1
win ratio (rotated vs frozen) and per-rank subspace overlap between the two
bases, per layer. Decision rule for the PAPER text: |relative win delta| <
2% at both budgets → claim scoped as "the frozen-rotation approximation is
measured-negligible at gpt2 scale" (Llama spot-check queued for the rental);
≥ 2% → the paper uses the rotated form's numbers going forward and the Llama
refit rides the rental as a REQUIRED item. Either way the sign-flip footnote
enters the methods section.

## C. int8 decoder certificate (finding #9 — user reviews before it replaces VM Task 8)

**Mechanism:** the int8 roundtrip's perturbation Δdec = dec_int8 − dec is
deterministic per pack, so the added reconstruction distortion is exactly
computable pack-side (no caches, no GPU): per layer, the weighted added-error
second moment from Δdec and the pack's own lam/moments per the math doc's
closed form, reported as noise-to-signal against the payload distortion
(expected ~7e-5 per the review) and mapped onto the SAME axis as the
pre-registered VM gate (rel_degradation_int8 < 5%): the certificate's implied
bound must sit far inside the gate for the replacement argument to be made.

**Deliverable:** `int8_decoder_certificate(pack)` in spectral.py + a tiny
experiment emitting the per-layer certificate table for the existing gpt2
packs (and the fit-cache-free analytic identity test), plus a one-page doc
section presenting: the formula, the numbers, the mapping to the 5% gate,
and the honest limits of the certificate (what it does NOT capture — e.g.
interaction with the query distribution beyond the modeled moments).
**Acceptance is the USER'S call after reading; VM Task 8 stays in the queue
until explicitly released.**

## Non-goals

- No recipe/CACHE_ARMS registration of charge-aware arms before the gate
  verdict + user promotion; no touching shipped packs or duel numbers.
- No eigenvalue shrinkage, no 1-bit tier, no mean-centering lever (future
  work list in the results doc — each needs its own validation design).
- No Llama fitting locally; every surviving item gets one line in the VM
  addendum list.

## Constraints

Repo hard rules (battery before commit, tyro, fp64 moments/fp32 experiments,
deterministic seeds, explicit run selection, module-form invocation, tiny
offline models in tests, no downloads in tests). Default-inert everywhere:
`s_ref=None`, `g_table=None`, `w_rope="frozen"` reproduce today's outputs
bit-exactly, each pinned by a test. The math doc is the authoritative source
for formulas; where this spec and that doc disagree, STOP and reconcile
before implementing.
