# D0 Literature Notes: Rotation-Based Weight Quantization Theory

Track D0 of the research plan. Question: how much of entry 3's planned theory contribution —
(a) Haar/random-rotation Gaussianization of coordinates, (b) MSE vs inner-product distortion
with 4^{-b} lower-bound floors, (c) unbiased inner-product estimation via sign sketches — is
already published as a foundation *for weight quantization*?

Date of pass: 2026-06-10.

---

## QuIP (Chee, Cai, Kuleshov, De Sa — arXiv 2307.13304, NeurIPS 2023)

Link: https://arxiv.org/abs/2307.13304

The only one of the four "practical rotation" papers with substantial theory. Key results
(numbering from the arXiv PDF):

- **Definition 1 (mu-incoherence).** Hessian H is mu-incoherent if its eigenvector matrix Q
  satisfies |Q_ij| <= mu/sqrt(n); weight W is mu-incoherent if |W_ij| <= mu * ||W||_F / sqrt(mn).
  This is a **max-entry (L-infinity) bound, not a distributional statement**.
- **Theorem 1 (LDLQ optimality).** LDLQ (adaptive rounding with linear feedback) is worst- and
  average-case optimal *within the class* of rounding methods with linear feedback U a function
  of H only: L_worst = (m/4) tr(D), L_avg = (m/c) tr(D), c = 12 (nearest) / 6 (stochastic),
  D from the LDL decomposition of H. This is a within-class optimality result, not an
  information-theoretic lower bound.
- **Lemma 2.** If H is mu-incoherent, tr(D) <= (mu^2/n) tr(H^{1/2})^2 — the bridge from
  incoherence to the proxy-loss bound.
- **Lemma 3 / Theorem 4.** Proxy losses of plain nearest/stochastic rounding ((m/4) tr H worst
  case); without incoherence no spectral bound can separate LDLQ from these baselines.
- **Lemma 5 (incoherence processing).** For U, V Kronecker products of k independent Haar
  orthogonal factors, V H V^T is mu_H-incoherent and U W V^T is mu_W-incoherent w.p. >= 1-delta
  with mu_H = A^{k/2} log(C^k n^2 / delta)^{k/2} = O~(1), mu_W = A^k log(2 C^k mn / delta)^k.
  **Tail/max bound only — no Beta/Gaussian coordinate-distribution result is proven.**
- **Theorem 7 (rate-dependent UPPER bound).** With the clamp-aware variant (Algorithm 5,
  stochastic rounding), w.p. >= 1-delta:
  tr((W_hat - W) H (W_hat - W)^T) = O~( (1/(n^2 4^b)) tr(H^{1/2})^2 ||W||_F^2 ).
  So QuIP **does** publish a 4^{-b}-rate result for the Hessian-weighted proxy loss of weights —
  but it is an **upper bound only; no rate-distortion lower bound anywhere**, and no claim of
  optimality of the 4^{-b} constant.
- **Objectives.** The whole paper works in the Hessian-weighted proxy
  l(W_hat) = tr((W_hat - W) H (W_hat - W)^T), explicitly distinct from raw weight MSE
  (incoherence processing is exactly the trick that preserves the proxy quadratic form,
  tr(W~ H~ W~^T) = tr(W H W^T)). Nothing is proven about the *gap* between the two objectives.
- **Unbiased quantization.** Considered: the Q subroutine may be nearest (biased) or standard
  unbiased stochastic rounding (Sec. 2, Supplement C.8). **Empirical finding (Table 15):
  unbiased rounding is consistently *worse* than biased nearest rounding** (perplexity, OPT
  125m-2.7b, large gap at 2-3 bits). No theory of bias accumulation across layers.
- No citation of Shannon rate-distortion theory or minimax lower-bound machinery.

## QuIP# (Tseng, Chee, Sun, Kuleshov, De Sa — arXiv 2402.04396, ICML 2024)

Link: https://arxiv.org/abs/2402.04396

- **RHT incoherence lemma (Lemma 3.1 / "hadincoh", Sec. 3).** Randomized Hadamard Transform
  gives mu_H = sqrt(2 log(2 n^2 / delta)), mu_W = 2 log(4 mn / delta) w.p. >= 1-delta —
  tighter (log vs log^2) than QuIP's Kronecker construction. Again a **max-entry concentration
  bound, sub-Gaussian tails — not a coordinate-distribution theorem**.
- The Gaussian-shape claim is **heuristic**: "the incoherence-processed weights follow a roughly
  ball-shaped sub-Gaussian distribution" (Sec. 3) — used as *motivation* for the E8 lattice
  codebook (E8 cited for optimal 8-dim sphere packing, Viazovska 2017), not proven as a
  distributional limit law.
- **Theorem 4.1 (BlockLDLQ).** E[tr((W_hat - W) H (W_hat - W)^T)] <= (g m mu^2 sigma^2 / n)
  tr(H^{1/2})^2 for g-block LDLQ with a codebook of per-coordinate MSE sigma^2. Upper bound
  only; the bit-rate enters only implicitly through sigma^2 of the chosen codebook.
- **No rate-distortion lower bound, no 4^{-b} expression, no unbiasedness analysis.**
- Distinguishes weight MSE vs Hessian-weighted proxy (inherits QuIP's framing).

## QuaRot (Ashkboos et al. — arXiv 2404.00456, NeurIPS 2024)

Link: https://arxiv.org/abs/2404.00456

- **No original theorems.** Single cited result: computational-invariance theorem
  (Theorem 1 of SliceGPT, Ashkboos et al. 2024) — orthogonal transforms of weights/activations
  leave transformer output unchanged.
- Rotation justification is empirical (outlier/incoherence plots) plus citations to QuIP/QuIP#.
- No error bounds of any kind, no bias discussion. Purely an engineering/systems paper
  (4-bit weights+activations+KV via Hadamard rotations).

## SpinQuant (Liu et al. — arXiv 2405.16406, ICLR 2025)

Link: https://arxiv.org/abs/2405.16406

- **No formal theorems or lemmas.** Evidence is empirical: kurtosis of activations drops from
  >200 to ~3 ("Gaussian-like") after rotation (Fig. 3); quantization error reduced (Figs. 2-3).
- Key empirical observation relevant to us: **performance varies hugely across random rotations**
  (up to 13 points zero-shot accuracy, Fig. 4); random Hadamard better than Haar-random but
  still high-variance — motivating *learned* rotations via Cayley SGD on the Stiefel manifold.
  They attribute Hadamard's edge to QuIP#'s tighter max-value bounds (Tseng et al. 2024).
- No rate bounds, no MSE-vs-output-objective theory, no bias analysis.

## TurboQuant (Zandieh et al. — arXiv 2504.19874, ICLR 2026)

Link: https://arxiv.org/abs/2504.19874

The theory cluster of hypothesis entry 3, almost verbatim — but **for data-oblivious online
vector quantization (KV cache, ANN search), not weights**:

- **Lemma 1.** Coordinates of a Haar-randomly-rotated unit vector are Beta-distributed,
  converging to N(0, 1/d), with near-independence of distinct coordinates in high dimension.
  This IS hypothesis component (a), proven — generically for any unit vector.
- **Theorem 1 (MSE-optimal).** D_mse <= (sqrt(3) pi / 2) * 4^{-b} (refined constants for
  b = 1..4: ~0.36, 0.117, 0.03, 0.009).
- **Theorem 2 (inner-product-optimal).** Two-stage MSE quantizer + 1-bit QJL on the residual
  gives an **unbiased** inner-product estimator with D_prod <= (sqrt(3) pi^2 ||y||^2 / d) 4^{-b}.
- **Theorem 3 (lower bounds).** Via Shannon rate-distortion + Yao's minimax: any (randomized)
  quantizer has D_mse >= 4^{-b} and D_prod >= (1/d) 4^{-b}. TurboQuant within ~2.7x. This IS
  hypothesis component (b), proven, including the MSE-vs-inner-product objective split.
- **Uptake by weight quantization: none found.** Citations and follow-on engineering work
  (vLLM/llama.cpp integrations, Google blog, March 2026) are all KV-cache-side. The paper cites
  QuIP but does not treat weights; no weight-quantization paper found citing it.

## QJL (Zandieh, Daliri, Han — arXiv 2406.03482)

Link: https://arxiv.org/abs/2406.03482

- 1-bit Quantized Johnson-Lindenstrauss: keep the sign of a JL projection; asymmetric estimator
  (quantized key x unquantized query) is **unbiased** for inner products with bounded distortion;
  no quantization constants needed. Hypothesis component (c) — proven, **applied to KV cache
  only**, never to weights.

## The Ordentlich–Polyanskiy line (the biggest overlap, found during the pass)

### Optimal Quantization for Matrix Multiplication (arXiv 2410.13780, 2024)

Link: https://arxiv.org/abs/2410.13780

- Universal nested-lattice quantizer for matmul of arbitrary A, B with explicit error bounds in
  ||A||_F, ||B||_F, ||A^T B||_F; **non-asymptotic information-theoretic LOWER bound**; exact
  **rate-distortion function for matmul of iid Gaussian matrices**, which their scheme achieves.
- Phase transition at R ~= 0.906 bit/entry: below it, JL sketching is *necessary* — i.e. the
  sign-sketch/low-rate regime and the lattice/high-rate regime are provably distinct.

### High-Rate Quantized Matrix Multiplication: Theory and Practice (arXiv 2601.17187, Jan 2026)

Link: https://arxiv.org/abs/2601.17187

- **Theorem 1:** smallest worst-case matmul distortion attainable for all bounded-norm matrices:
  D*_ij = (||a_i||^2 ||b_j||^2 / n) * 2 * 2^{-2R}, achieved by lattice quantizers with **random
  rotation and dither**. **Theorem 2:** matching Gaussian-case lower bound Gamma(R), = 2*2^{-2R}
  above R* ~= 0.906.
- **Part II / Secs. IV-V: weight-only quantization** with activation covariance Sigma_X known at
  the encoder: waterfilling optimum D* = |Sigma_X|^{1/n} sigma_W^2 2^{-2R}; a Sigma_X-oblivious
  decoder (isotropic Gaussian codebook) still attains it at high rate. This is precisely a
  rate-distortion treatment of the *output-distortion* objective for weights, lower bound
  included — the Hessian-weighted-proxy analogue done information-theoretically.
- Explicitly distinguishes entry-MSE from matrix-product distortion; justifies the
  additive-uniform-noise model via **dithered** quantization (unbiasedness machinery present,
  though multi-layer bias accumulation is not studied).
- Benchmarks GPTQ/LDLQ-style schemes against the fundamental limits; does not cite
  QuIP#/QuaRot/SpinQuant/TurboQuant by name.

### NestQuant (Savkin, Kirtas/et al.; arXiv 2502.09720, 2025)

Link: https://arxiv.org/abs/2502.09720

- Applies the O-P optimal-matmul theory (restates the Gamma(R) lower bound,
  2*2^{-2R} - 2^{-4R} for R >= R*) as a **practical LLM quantizer for weights, activations and
  KV cache** (Llama-3-8B at 4 bits, large perplexity-gap reduction vs SpinQuant/QuaRot/QuIP#).
- **Remark 2.1** states exactly the objective-gap point: per-vector-MSE-optimal quantizers are
  *not* optimal for inner products / matrix products.
- Uses Hadamard rotation on weights but proves no new distributional result; no unbiasedness
  analysis.
- **This paper already is "the link" between the rate-distortion theory cluster and practical
  rotation-based weight quantization**, citing and beating QuIP#/QuaRot/SpinQuant.

### PolarQuant (arXiv 2603.29078, March 2026 — WITHDRAWN)

Link: https://arxiv.org/abs/2603.29078

- Claimed: block normalization to the sphere + Hadamard rotation -> approx Gaussian coordinates
  -> Gaussian-matched centroids for weight compression; empirical, no proofs. Withdrawn
  2026-04-20 ("found some errors"). Signal that this exact idea-space is being worked, but no
  standing published claim.

---

## Verdict: what is already published vs genuinely open

**(a) Rotation -> Beta/Gaussian coordinates.**
- *Published for generic vectors:* TurboQuant Lemma 1 (Beta -> N(0,1/d), near-independence).
  Since a weight row is just a vector, the mathematical content applies to weights immediately.
- *Published for weights, weaker form:* QuIP Lemma 5 and QuIP# Lemma 3.1 prove only
  **max-entry/incoherence (sub-Gaussian tail) bounds**, and QuIP#'s "ball-shaped sub-Gaussian"
  remark is explicitly heuristic. QuaRot/SpinQuant: empirical kurtosis only.
- *Open (narrow):* a stated theorem connecting the full coordinate distribution law (not just
  tails) to weight-quantization pipelines — e.g. how Beta-coordinate structure interacts with
  LDLQ-style feedback rounding or codebook design — is not in any of these papers. But as a
  standalone lemma it would be viewed as known (TurboQuant, plus classical sphere-measure
  results).

**(b) MSE vs inner-product objectives, 4^{-b} floors.**
- *Published:* TurboQuant Theorem 3 (both floors, Shannon + Yao, data-oblivious setting);
  Ordentlich-Polyanskiy 2410.13780 (rate-distortion function for Gaussian matmul, matching
  lower bound, JL phase transition); O-P 2601.17187 Theorems 1-2 (worst-case 2*2^{-2R} matmul
  limit) **and a weight-only rate-distortion solution with activation covariance
  (waterfilling, lower bound included)**. NestQuant Remark 2.1 states the objective gap and
  deploys it for weights in practice.
- *Open (narrow):* the floors specialized to the *data-oblivious weight* setting (no Sigma_X /
  Hessian at the encoder, biased adaptive rounding allowed) and a proven gap between
  weight-MSE-optimal and proxy-loss-optimal quantizers under the QuIP proxy specifically.
  Note QuIP Theorem 7 already gives the matching-order 4^{-b} *upper* bound for the proxy loss.

**(c) Unbiased inner-product estimators / bias accumulation.**
- *Published for activations/KV:* QJL and TurboQuant Theorem 2 (unbiased, with distortion
  guarantees).
- *Published for weights, negative-empirical:* QuIP Supplement C.8 — unbiased stochastic
  rounding *hurts* perplexity vs biased nearest rounding; no explanation offered. O-P use
  dither (unbiased noise model) but only single-matmul analysis.
- *Genuinely open:* (i) any theory of **bias accumulation across layers** in quantized networks
  — none of the seven papers above analyzes it; (ii) reconciling QuIP's "unbiased is worse"
  empirical finding with the unbiased-estimator theory (variance-bias tradeoff across depth);
  (iii) bias-corrected sign-sketch (QJL-style) **weight** matvecs — never attempted.

**Recommendation.** Failure mode 1 is **substantially realized for (a) and (b)**: the
Ordentlich–Polyanskiy papers plus NestQuant already supply rate-distortion lower bounds for
matmul/weight quantization and explicitly bridge to QuIP#/QuaRot/SpinQuant, and TurboQuant
supplies the Beta-coordinate lemma and the MSE-vs-inner-product split with 4^{-b} floors in the
data-oblivious vector setting. A contribution framed as "provide the missing theoretical
foundation under rotation-based weight quantization" would not survive review against
NestQuant + O-P 2601.17187. What still tests something novel, and what D2/D3 should be re-aimed
at: the **(c)-cluster for weights** — unbiased (sign-sketch / dithered) weight matvecs, a theory
of bias accumulation vs variance across L layers, and an explanation of QuIP's unbiased-is-worse
anomaly — plus, secondarily, the **data-oblivious weight floor** (no Hessian/Sigma_X at encode
time), which sits in the gap between TurboQuant's vector setting and O-P's known-covariance
setting. D2/D3 experiments remain worthwhile only if re-scoped to those questions; pure
"rotation Gaussianizes weights, hence 4^{-b}" replications are now literature.

---

## Sources

- https://arxiv.org/abs/2307.13304 (QuIP; full text via arXiv PDF and https://ar5iv.labs.arxiv.org/html/2307.13304)
- https://arxiv.org/abs/2402.04396 (QuIP#; full text via https://ar5iv.labs.arxiv.org/html/2402.04396)
- https://arxiv.org/abs/2404.00456 (QuaRot; full text via https://ar5iv.labs.arxiv.org/html/2404.00456)
- https://arxiv.org/abs/2405.16406 (SpinQuant; full text via arXiv PDF https://arxiv.org/pdf/2405.16406)
- https://arxiv.org/abs/2504.19874 (TurboQuant; full text via https://arxiv.org/html/2504.19874v1)
- https://arxiv.org/abs/2406.03482 (QJL)
- https://arxiv.org/abs/2410.13780 (Ordentlich & Polyanskiy, Optimal Quantization for Matrix Multiplication)
- https://arxiv.org/abs/2601.17187 (Ordentlich & Polyanskiy, High-Rate Quantized Matrix Multiplication; full text via https://arxiv.org/html/2601.17187v1)
- https://arxiv.org/abs/2502.09720 (NestQuant; full text via https://arxiv.org/html/2502.09720v3)
- https://arxiv.org/abs/2603.29078 (PolarQuant, withdrawn)
- https://openreview.net/forum?id=tO3ASKZlok (TurboQuant OpenReview / ICLR 2026)
- Web search results on TurboQuant citation uptake (KV-cache-only deployments: vLLM/llama.cpp/Google Research blog coverage, March-April 2026)
