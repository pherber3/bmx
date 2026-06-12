# Next research avenues — from the BM negative results to structured-residual compression

Date: 2026-06-10. Written at the close of the H100 session, as the forward-looking
companion to `2026-06-10-h100-session-results.md`. The BM program produced clean
negatives; those negatives *characterized* the actual structure of trained weights, and
that structure points at specific, foundational math. This note records the three avenues,
the foundational grounding for each (read, not skimmed), and a concrete first experiment.

The throughline: **stop modeling weight stacks as template-shaped (falsified); model the
quantization residual as structured (low-rank + sparse), which is what we actually
measured.**

---

## What the negatives measured (the empirical priors for what follows)

1. **Weight stacks are subspace-shaped, not template-shaped.** Permutation null: Tucker
   keeps a real cross-slice advantage (0.06–0.10 RE) that vanishes on the null; BMD's
   real-vs-null gap is ≈0. → the right structural model is *shared low-rank subspace +
   private per-slice coefficients*.
2. **MoE experts are orthogonal-as-vectors but share ~10 global second-moment modes**
   (C1 census, 3 checkpoints; participation ratio ~0.15·E). → a *shared-subspace +
   private-coefficient* (CCA-shaped) object, measured directly.
3. **Rotation Gaussianizes the bulk but leaves stubborn channel outliers** (D1:
   worst-channel outlier mass 0.69 → 0.23, not → 0). → *Gaussian bulk + sparse residual*.
4. **Decode is byte-bound and the matvec is linear** (Track B + ncu: dense reads h·m·p
   bytes exactly; factored reads ℓ·m·p; speedup tracks the byte ratio). → a linear map
   under a byte budget — the home of sketching / randomized linear algebra.

---

## Avenue 1 (recommended) — low-rank-plus-sparse quantization residual

**Thesis.** Decompose each weight matrix `W = L + S + Q(R)`: a low-rank `L` (the shared
subspace facts 1–2 say exists), an elementwise-sparse `S` (the stubborn outliers fact 3
measured), and a cleanly-quantizable dense residual `R = W − L − S`. The bulk quantizes
*well* precisely because you have subtracted off the two structures (subspace + outliers)
that wreck its histogram. **AWQ's salient-channel trick `Q(W·diag(s)⁻¹)·diag(s)` is
already a crude *column*-sparse correction** — this is its principled, two-structure
generalization with recovery guarantees.

**Foundational grounding (read, Wainwright *High-Dimensional Statistics* §10.7):**
- The object: additive matrix decomposition `Y = Λ* + Γ* + W`, low-rank `Λ*` plus
  elementwise-sparse `Γ*` (Example 10.19 *factor analysis with sparse noise* is the exact
  shape; `Σ = LLᵀ + Γ`).
- The estimator (Eq. 10.53): convex,
  `min_{Λ,Γ} ½‖Y − (Γ+Λ)‖²_F + λₙ‖Γ‖₁ + ωₙ‖Λ‖_nuc`, the familiar
  ℓ₁-plus-nuclear-norm program.
- **The load-bearing identifiability condition: "spikiness," not incoherence.** The
  low-rank/sparse split is ill-posed without excluding matrices that are *both* (Θ_bad,
  Example 10.16). Wainwright's key move (more robust than singular-vector incoherence
  under noise): bound the low-rank part's max entry, `‖Λ*‖_max ≤ α/√(d₁d₂)`. This is the
  condition we must check holds for weight matrices — and it is checkable empirically.
- The guarantee (Corollary 10.22): with `λₙ ≥ 2(‖W‖_max + 4α/√(d₁d₂))` and
  `ωₙ ≥ 2‖W‖₂/λₙ`, the squared Frobenius error is oracle-bounded
  `≲ ω²ₙλ²ₙ·rank(Λ*) + λ²ₙ·|supp(Γ*)|` — i.e. error scales with the *true* rank and
  sparsity, which is exactly the rate-distortion lever a systems person wants.

**Why it's the pick.** Unique convergence of three measured facts (subspace + outliers +
byte budget); the foundational layer has the *complete* theory (estimator + identifiability
+ recovery rate) where the field (AWQ/LLM.int8) is heuristic; and it **composes with the
Track-B fused-dequant kernel we already validated** — store `[INT4 bulk] + [L: two skinny
fp16 factors] + [S: a few fp16 outliers]`, reconstruct `W = dequant(bulk) + L + S` inside
the matmul prologue so `W` never lands in HBM. It is also the natural continuation of the
one D-track thread that survived D0 (structured residuals / unbiased estimators), so it
extends the program rather than restarting it.

**The cheap estimator to implement first** (verified against the text, Wainwright
§11.4.2 Eq. 11.58 / Proposition 11.19): a two-step **hard**-threshold-then-residual
procedure — `T_ν(v) = v·𝟙[|v| > ν]` (NOT soft-thresholding; the operator defined at
Eq. 11.58 keeps the value, it does not shrink it). Adapted to our object: `Ŝ = T_ν(W)`
(the large entries), `L̂ = truncated-SVD(W − Ŝ)`, optionally alternated 2–3 times.
Attribution that matters for follow-up reading: the direct thresholding+truncated-SVD
approach and the *noisy-setting* spikiness analysis are **Agarwal et al. 2012**
(Wainwright's bibliographic note, §11.5); exact-recovery originals are Chandrasekaran
et al. 2011 and Candès et al. 2011 (robust PCA). Implement the cheap estimator first;
the convex program (10.53) is the referee, not the workhorse.

**The basis decision (load-bearing — settled by grounding, not taste).** Rotation and
sparse-extraction are *competing treatments of the same outlier structure*. The
Beta-coordinate theorem (the same fact that makes rotation good for RTN) says a rotated
vector's coordinates go to ~N(0, 1/d) regardless of input — i.e. rotation provably
**spreads a sparse vector's mass across all d coordinates** at scale ‖x‖/√d. So fitting
a sparse `S` *after* rotation is structurally self-defeating: the rotation already
destroyed the concentration that `S` needs. The same principle is triangulated in
practice across three quantization domains (FA3 FP8 attention §3.3, NVFP4 wgrad RHT,
QuIP/QuIP# weight PTQ — see the vault's "Incoherence Processing Across Quantization
Domains" note, whose failure-mode list states it directly: "RHT helps when outliers are
sparse-and-large; it does nothing when the distribution is already approximately
Gaussian"). **Therefore: fit L+S in the ORIGINAL basis** (where D1 measured worst-channel
outlier fraction 0.69 — concentrated), and treat rotation as the *alternative* arm, plus
a composed arm (L+S first, then rotate the residual) which should out-Gaussianize either
alone. Null result noted: the AI-perf foundational corpus covers AWQ/GPTQ/SmoothQuant
but has no QuIP/rotation-incoherence material — the practice grounding is the primary
papers via the vault note.

**Metric calibration (correcting an easy misreading).** bmx's `outlier_mass` is the
*fraction of entries* in a channel beyond 3σ (a count, not Frobenius mass), and the
headline 0.23 was the **worst channel of the worst matrix, post-rotation** (pre-rotation
0.69; medians far lower — likely one structured offender such as `wpe`). Baselines for
the sparsity-match check come from the per-matrix distribution in the committed d1
parquet, not from the single worst-case number.

**First experiment — two stages, in this order** (the decompose-clean-then-code-residual
ordering has a direct precedent in TurboQuant's two-stage construction, vault note
"Two-Stage Quantization for Unbiased Inner Products": stage 1 captures the bulk with the
primitive suited to it, stage 2 codes the residual stage 1 missed; composition works
because stage 2 operates on the clean object's leftover, never on the coded output):

- **Stage A — diagnostic (one matrix, one estimator call).** On a GPT-2 attention
  projection `W` in the original basis, run the two-step estimator. Validate:
  (a) does `rank(L̂)` match the subspace Tucker found independently in a2?
  (b) does `supp(Ŝ)` concentrate on the channels D1 flagged?
  (c) does `‖L̂‖_max ≤ α/√(d₁d₂)` hold (the spikiness go/no-go — if the recovered
  low-rank part is itself spiky, the decomposition is ill-posed and the avenue narrows)?
  This treats quantization later; here the "noise" slot of Wainwright's model is just
  the unmodeled bulk. Three structural assumptions tested in one shot.
- **Stage B — compression (the payoff number).** Fit `(L̂, Ŝ)` on **clean** `W`, form
  `R = W − L̂ − Ŝ`, quantize with `bmx.quant.rtn`, reconstruct
  `Ŵ = dq(Q(R)) + L̂ + Ŝ`. Compare at **matched total bits (counting L̂ and Ŝ storage)**
  against: plain `dq(Q(W))`, rotate-then-RTN, and the composed arm. Metrics:
  `ip_distortion`, per-row kurtosis of `R` vs `W`, then layer-swap perplexity
  (`bmx.eval.layer_swap`, to be implemented) if Stage A cleared. Natural stage-3
  extension once this works: an unbiased QJL pass on the *quantization* residual —
  TurboQuant's exact composition — which is how this avenue plugs into the (c)-cluster
  (unbiased weight matvecs) that survived D0.

**The composite model (synthesis worth testing once A/B clear).** The CCA-shaped C1
finding and the L+S decomposition are the same additive split at different scales
(cross-matrix vs within-matrix), suggesting `W_e = L_shared + S_e + B_e` for expert
stacks: cross-expert shared low-rank (the ~10 CKA modes), per-expert sparse outliers,
per-expert private low-rank. Caveats recorded now: 3-way identifiability has no
closed-form theory (the 2-way spikiness condition doesn't trivially extend), and the
per-component compression factors apply to *different slices of the mass* — they do not
multiply into a combined ratio.

Most of the infrastructure exists: `quant/` (rotation, RTN, stats, IP-distortion), the
matched-parameter `sweep` engine, `eval/layer_swap` stub, the artifacts/plotting harness.
What's new: an `lrs` (low-rank+sparse) module behind the existing `Decomposition`
protocol (the hard-threshold + truncated-SVD two-step, unit-tested on planted L+S), and
the two-stage experiment. Systems composition support from the foundational layer:
Grokking Megakernels' INT4 analysis ("dequantization fits cleanly into the existing
execution timeline" because decode is bandwidth- not compute-bound) applies verbatim to
adding the `+ L̂ + Ŝ` terms in the dequant epilogue.

---

## Avenue 2 — sketched decode matvec with a *uniform* error bound

**Thesis.** Decode applies the same `W` against thousands of tokens. The QuIP#/QJL line
sketches weights and proves *per-vector* inner-product bounds; what actually governs decode
quality is a bound *uniform over the whole token stream*. Vershynin *High-Dimensional
Probability* Ch. 9 has the sharper tools the quantization papers don't cite:
- **Matrix deviation inequality (Thm 9.1.1):** for a sub-Gaussian sketch `A`,
  `sup_{x∈T} | ‖Ax‖₂ − √m‖x‖₂ | ≲ γ(T)` — a *uniform-over-a-set* guarantee, i.e. one
  bound covering all tokens at once, not a union bound per token.
- **Dvoretzky–Milman (Thm 9.7.2):** a random projection of any bounded set is
  *provably near-round* — the rigorous version of "rotation kills outliers" that D1 showed
  empirically and QuaRot/SpinQuant assert heuristically.

**Lead.** A sketched/projected decode matvec carrying a uniform error bound across the
token stream, paired with the *unbiased-estimator* gap D0 left open (QJL is unbiased for
KV/activations; the weight-side unbiased matvec, and why QuIP found unbiased stochastic
rounding empirically *worse*, are open). Cleaner math than Avenue 1, slightly less direct
systems payoff. Good second priority.

---

## Avenue 3 — CCA-structured expert streaming (the honest reframe of entry 2)

**Thesis.** Fact 2 *is* a canonical-correlation-analysis finding. Murphy *PML: Advanced
Topics* §28.3.4.3: CCA models each view as a *shared* latent subspace `zˢ` plus a *private*
subspace (`zˣ`, `zʸ`), `Σ = W_s W_sᵀ + (private)`. The entry-2 streaming dream — templates
resident, gains streamed — with the **correct** decomposition substituted: experts =
shared subspace `L` (the ~10 global modes, HBM-resident) + private per-expert coefficients
`Cₑ` (small, streamed on routing). BMD assumed *diagonal-gain* sharing (too rigid,
falsified); CCA assumes *linear-subspace* sharing (which C1 measured to exist).

**Why it's avenue 3, not 1.** C2 already showed the shared part is ~10 modes against 2048 —
likely too thin to compress the bulk. So this is "the streaming idea was right, the
redundancy is just smaller than hoped." Worth a *clean* shared-subspace + private-coeff fit
(partial_tucker / CCA — both already wired in `bmx.decomp.baselines` and `bmx.census`) to
put a real number on exactly how thin, rather than inferring it from the BMD failure. Low
cost (reuses C1/C2 infrastructure), modest expected payoff, honest closure on entry 2.

---

## Recommendation

Pursue **Avenue 1** first: it's the highest-information, best-grounded, and systems-composable
lead, and it picks up the surviving D-track thread. Avenue 3 is a cheap honest-closure run
on entry 2 that can ride alongside. Avenue 2 is the deeper-math follow-up once Avenue 1's
residual structure is characterized (the sketch quality depends on what L+S leaves behind).
