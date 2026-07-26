# K4 estimation levers — design (eigenvalue shrinkage gate, per-layer int8 threshold, figure refresh)

Provenance: user 2026-07-25 "continue with the local work candidates" after
the local-levers results commit (`fe29d99`). Sources: math review
`docs/2026-07-24-k4-math-review.md` finding #7 (MP-biased spectrum at the
fit aspect ratio — now DOUBLY motivated: the Jensen identity check and the
allocation jitter sit at the same estimation boundary,
`docs/2026-07-25-k4-local-levers-results.md` §3/§5); local-levers §5
(per-layer tier threshold); paper-integrity: the committed
`k4_bits_vs_context` figure still shows the BLANKET skeptic-v2-int8 dashed
line labeled "pending its own quality gate" — that gate has resolved
(blanket rejected, tier-gated certified), the figure must say so.

**Ordering rationale (why this cycle precedes the rental):** the REQUIRED
rotated-W Llama refit should ship with the winning allocation input. If
shrinkage passes its gate, the refit fits with it; deciding after the rental
means refitting twice. All work local (gpt2 mechanism scale + analytic).

## Part 1 — Eigenvalue shrinkage before the waterfill (finding #7, kill-or-confirm)

**The gap:** allocation inputs λ̂ come from sample covariances at
γ = C/n ≈ 0.09–0.13 (deployment corpus fits) with autocorrelated rows —
eigenvalues spread vs truth (top biased up, bulk/tail down), ~0.13–0.22 bits
of jitter through ½log₂λ concentrated at the 0↔2 boundary that sets c_used.
Shrinkage is a provably-variance-reducing correction to the allocation
input; it changes NO accounting expression and NO basis — only the waterfill
input, via the existing `pack_from_basis(lam_alloc=…)` hook.

**Mechanism:** `shrink_spectrum` in spectral.py with two estimators:
- `lw` (the gated arm; Ledoit–Wolf linear shrinkage toward μI): intensity ρ
  estimated from the fit ROWS (standard LW formula on the same W-weighted
  row matrix the moment is built from); shrunk allocation input
  λ_lw = (1−ρ)·λ̂ + ρ·μ̂ with μ̂ = mean(λ̂). Same eigenvectors, same basis —
  monotone map of the spectrum.
- `oas` (reported beside, never gated): Oracle-Approximating Shrinkage —
  closed form from (λ̂, n) alone, no rows needed (attractive because it
  could run at any call site); same shrink target.
Both fp64. `lam_alloc=None` paths stay bit-exact (already pinned).

**Pre-registered gate (binding, gpt2 mechanism scale, deployment-like n):**
corpus-fit packs (the standard wiki fit fleet, full fit rows — γ ≈ 0.13,
matching the Llama duel fit's 0.125) with `lw`-shrunk allocation vs plain
allocation, budgets {2.2, 2.5}, heldout caches, win measured on the
A-gate/dec-quant axis (TQ-curve interpolation at matched skeptic-v2 bpe).
PROMOTE (Llama refit fits with shrinkage) iff at BOTH budgets:
heldout win(lw) ≥ 1.02 × win(plain) (a ≥ +2% relative improvement — a real
effect, not noise) AND no matched-budget bpe regression > 0.02. Otherwise
HONEST NEGATIVE — recorded, allocator input stays raw, refit unchanged.

Diagnostics (reported, never gated):
- n-scaling: refit at subsampled fit rows n ∈ {768, 1536, 3072, full} ×
  {plain, lw, oas} — the mechanism signature is improvement GROWING as n
  shrinks. If full-n is null but small-n is real, the verdict is "shrinkage
  matters below deployment n; refit unaffected" — decision-relevant either
  way.
- c_used stability: shrinkage should prune spuriously-funded tail
  directions (c_used variance across fit subsamples shrinks).
- ρ per layer (the estimated intensity — near 0 means the data says
  "don't shrink", itself informative).
- 0↔2 boundary movement (tier-map deltas), the pre-stated jitter site.

## Part 2 — Per-layer int8 tier threshold (analytic-first, materiality-barred)

**The gap:** the layer-uniform T=5 binds on layer 1 at every T; per-layer
T_ℓ should recover part of the ≤8% of blanket's charge saving that
uniform-T leaves.

**Mechanism:** T_ℓ(pack) = max{T ∈ {2,…,6,8} : certificate
`implied_rel_degradation(pack_ℓ, T)` ≤ 5%} per layer — derived
deterministically from the pack at materialization (no new metadata).
`dec_quant="int8_tl"` parses through the existing threshold plumbing;
`load_packs_for_spec` applies the per-layer gated roundtrip; streaming
accounting already computes per-layer c_int8 (extend the single-thr
computation to per-layer thr — small).

**Pre-registered rule (two bars, both binding):**
1. Quality: measured (k4_dec_quant, new `int8_tl` mode) rel_degradation < 5%
   at both budgets — same axis as T=5's gate.
2. MATERIALITY (YAGNI kill): per-layer must beat uniform T=5's charge saving
   by ≥ 0.3 bits/token/model at S=4096 (uniform T=5 achieved 10.91/11.45 of
   blanket's 11.61/12.43). Below that, the codec complexity does NOT ship —
   recorded as "certified but immaterial", the honest negative for a lever
   whose ceiling is ~8% of an already-won number.

## Part 3 — Paper-figure integrity refresh

`experiments/plot_k4_paper.py` bits-vs-context panel: replace the blanket
skeptic-v2-int8 dashed line + "ACCOUNTING PROJECTION ONLY pending its own
quality gate" caption with the CERTIFIED tier-gated band (mixed_dec_charge,
frac band [0.893, 0.916], band shading between the two frac curves) and a
caption stating the resolution (blanket rejected by certificate+measurement;
tier-gated T=5 certified+measured, gpt2-band estimate pending exact Llama
tier counts at the refit). Add one new figure: certificate-vs-measured
agreement (the instrument-validation scatter, 96 points, log-log, y=x line —
the §E pattern's evidence plot). Jensen gets a figure ONLY if it earns paper
space later — not in this cycle (avoid figure sprawl). Regenerate committed
PNGs/PDFs from committed parquets only (explicit run selection as the script
already does).

## Non-goals

- No nonlinear/analytical spectral shrinkage (LW linear + OAS only — the
  review's named floor); no shrinkage of the STORED lam (accounting/
  certificate weights stay as-fit; only the allocation input shrinks).
- No tier-grid densification (recorded future work; needs its own gate).
- No Llama fitting locally; the refit consumes the verdict, not new code
  beyond `lam_alloc` wiring already present.
- Part 2 ships nothing if the materiality bar fails, regardless of quality
  bar.

## Constraints

Repo hard rules (battery before commit; tyro; fp64 moments; deterministic
seeds; explicit run selection; tiny offline test models; results-doc commits
stop for the user). Default-inert: no default-path numeric change anywhere;
every new knob None/off by default, pinned. Where this spec and the math
review disagree, STOP and reconcile.
