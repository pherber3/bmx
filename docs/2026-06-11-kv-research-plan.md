# KV-cache compression research plan — Tracks K0–K3 (drafted 2026-06-11)

Successor program to the weights arc (BM program + Avenue 1, both closed with
measured reasons). Prime directive unchanged: **kill-or-confirm research; an honest
negative is a valid result.** Don't polish numbers; report them.

## Why KV, and what transfers from the weights program

The decode byte equation: time/token ≈ (weight bytes + KV bytes)/bandwidth. Weight
bytes are a solved-to-within-a-bit problem (rate–distortion floor known, practice
within ~1 bpw of it; structure extraction measured dead at all scales —
`2026-06-11-frontier-breakeven.md`). KV bytes grow with context × batch and equal
the 4-bit weights of a 70B model at ~100k context. The structure that is *absent*
from trained weights (Gaussianized bulk, axis-aligned outliers handled free) lives
in **activations** — which is what the KV cache is.

Transfers, verbatim or nearly: the break-even instrument (now scoring activation
spectra), the basis-decision methodology (rotation vs per-channel scales as
*competing treatments* of axis-aligned outliers — the Avenue 1 lesson, and the KV
literature is genuinely split: KIVI quantizes keys per-channel in the original
basis, QuaRot rotates), matched-total-bits arms with honest metadata accounting
(`quant/arms.py` pattern), null controls (random-sphere vectors as the
theory-setting control vs real caches — the gap between them IS the
marketing-vs-reality measurement), and the free-vs-priced correction taxonomy.

Theory anchors in the vault: [[Vector Quantization Distortion Objectives]] (MSE vs
inner-product distortion; 4^-b floors via Shannon+Yao), [[Quantized
Johnson-Lindenstrauss Transform]] (unbiased 1-bit IP primitive),
[[Two-Stage Quantization for Unbiased Inner Products]] (TurboQuant's composition),
[[Random Rotation Induces Beta-Distributed Coordinates]] (the rotate-then-scalar
reduction). All proved for worst-case unit vectors — none validated on real caches
at scale. That gap is this program's target.

## Falsifiable hypotheses

- **E1 (structure).** Real K/V activations carry structure trained weights lack:
  axis-aligned rogue channels (keys especially), steep per-head spectra, strong
  keys-vs-values asymmetry, RoPE-induced pair structure. Falsified if activation
  break-even margins sit ≈ 0 like weight margins did.
- **E2 (basis decision, round two).** Per-channel scales in the original basis
  capture most of rotation's benefit for key quantization; rotation is
  neutral-to-harmful where the outliers are axis-aligned (dose–response, as in the
  Avenue 1 finding). Falsified if rotated arms dominate per-channel arms at matched
  bits across layers/heads.
- **E3 (TurboQuant kill-or-confirm).** TurboQuant's pipeline (rotate → Lloyd-Max on
  the Beta marginal; prod variant + QJL residual stage) beats the boring baseline
  (KIVI-style per-channel INT4/INT2) at matched total bits *on real caches at
  realistic context*. Null hypothesis to try to confirm: its near-optimality on
  random spheres vanishes on real KV distributions. Either outcome is a result.
- **E4 (unbiasedness — the QuIP anomaly).** Bias in cached-key codes accumulates
  into attention-output drift that grows with context length (aggregation over
  thousands of keys is structural), while weight-side bias does NOT accumulate
  (explaining QuIP Supp C.8's unbiased-is-worse finding). Candidate weight-side
  mechanisms to discriminate: (H1) nearest-rounding errors never align across
  layers, so the 2× local MSE of stochastic rounding decides; (H2) norm layers
  absorb systematic drift; (H3) a crossover exists at extreme depth/low bits.

## Tracks

### K0 — harness (prerequisite, ~1 session)
- `src/bmx/cache/collect.py`: hooked forward passes over a fixed corpus slice;
  store per-layer K and V (pre- and post-RoPE for keys — a deployed design axis),
  plus the matching queries Q, to disk. Models: GPT-2 (cheap iteration, no RoPE)
  and Llama-3.1-8B (RoPE + GQA point). CPU is fine at 2–8k context.
- `src/bmx/cache/metrics.py`: attention-logit distortion (the D_prod objective
  against the *real* Q from the same pass), softmax-output error, plus the
  existing stats (kurtosis, outlier_mass).
- Quantized-cache perplexity eval (sibling of `eval/layer_swap.py`): hook-based
  K/V substitution, ppl on held-out text vs context length.

### K1 — census + break-even on activations (the D1/frontier analog)
Per layer × head: channel outlier structure, kurtosis pre/post rotation, spectra →
break-even margins (the frontier instrument pointed at K/V matrices of shape
(tokens × d_head·heads)), keys vs values, pre vs post RoPE, depth trend, and the
attention-sink/rogue-channel inventory. Output: the structural map that decides
which K2 arms are live.
**Gate G1:** if margins ≈ 0 AND no axis-aligned channel structure → E1 falsified;
drop low-rank arms from K2 and the program narrows to codebook comparison.

### K2 — matched-bits arms on real caches (the Stage-B analog)
Arms (all with honest metadata accounting — scales, codebooks, norms, QJL seeds):
per-channel INT4/INT2 (KIVI-style), groupwise RTN, rotate+RTN (QuaRot-style),
TurboQuant-MSE (rotate + Lloyd-Max on Beta marginal), TurboQuant-prod (two-stage,
+1-bit QJL residual), and — if G1 opens — low-rank-projected KV (post-hoc
MLA-shaped). Controls: the same arms on random unit vectors (the theory setting);
the real-vs-random gap per arm is the headline table.
Metrics: D_mse, logit distortion vs real queries, ppl-with-quantized-cache at
2k/8k (local) and 32k (one VM day if local is prohibitive).
**Gate G2:** any arm beating per-channel INT-k at matched bits by more than the
cost-model slop (≳0.25 effective bits, calibrated as in the frontier doc) on real
caches. The TurboQuant verdict is reported regardless.

### K3 — unbiasedness / depth-coherence (the QuIP anomaly, both sides)
Weight side: nearest vs stochastic rounding at matched bits on GPT-2/Llama-8B;
measure per-layer error coherence (alignment of layer-output error with
accumulated error), ppl vs quantized-depth-prefix, norm ablations → discriminate
H1/H2/H3. KV side: biased vs unbiased key codes; logit drift vs context length.
**Gate G3:** a mechanism account of *when unbiasedness pays its 2× MSE price* —
this decides whether TurboQuant-prod's complexity is ever justified and whether
the 2-bit KV regime needs unbiased codes.

## Sequencing, cost, and kill criteria

K0 → K1 (one cheap local pass) → K2 and K3 in parallel (K3 is independent of G1).
Everything except 32k-context validation runs locally on CPU; budget one optional
VM day at the end. The program dies cleanly if: G1 closes AND K2 shows all
sophisticated arms within slop of per-channel INT-k AND K3 finds unbiasedness
never pays — that combined negative would itself be a strong, publishable-flavor
statement ("KV quantization is also a solved-to-within-slop problem; the boring
baseline is optimal in practice"), mirroring the weights arc.

## Relation to prior results (so nobody re-litigates)

Weights: structure extraction dead at all scales (frontier doc); rotation = free
Gaussianization (D1); additive side-info priced by ε > 1 − 4^(−Δb) (postmortem);
rotation helps iff input not already Gaussian (Stage B dose–response). The KV
program is not a re-run of Avenue 1 on different tensors — it tests the *opposite*
prediction (activations are where the structure lives) with the same instruments,
plus the two questions the weights program could not reach: inner-product-objective
coding and unbiasedness under aggregation.
