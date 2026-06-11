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

**First experiment (cheap, local, mostly built):**
1. On GPT-2 / Llama-1B weight matrices, fit `L + S` via the convex program (or the simpler
   two-step thresholding estimator, Wainwright §11.4.2 / Prop 11.19: soft-threshold for S,
   residual SVD for L) at several `(rank(L), nnz(S))` budgets.
2. **Empirically verify the spikiness condition** `‖L‖_max ≤ α/√(d₁d₂)` on real matrices —
   this is the go/no-go: if the recovered low-rank part is itself spiky, the decomposition
   is ill-posed and the avenue narrows.
3. Quantize the residual `R = W − L − S` with the existing `bmx.quant.rtn` and measure
   **inner-product distortion** (`bmx.quant.stats.ip_distortion`) and per-row kurtosis vs
   quantizing `W` directly. The claim under test: at matched total bits (counting L and S),
   structured-residual quantization beats round-to-nearest and matches/beats AWQ.
4. If it clears: functional eval (layer-swap perplexity, the `bmx.eval.layer_swap` stub),
   then the fused-kernel byte accounting on the VM.

Most of the infrastructure exists: `quant/` (rotation, RTN, stats, IP-distortion), the
matched-parameter `sweep` engine, `eval/layer_swap` stub, the artifacts/plotting harness.
What's new: an `lrs` (low-rank+sparse) decomposition module behind the existing
`Decomposition` protocol, and a quant-residual experiment.

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
