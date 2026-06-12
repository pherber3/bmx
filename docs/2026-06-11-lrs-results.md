# Avenue 1 results — low-rank+sparse quantization residual (2026-06-11)

Verdict up front: **honest negative at GPT-2 scale, with the structure confirmed but the
bit-economics against it.** The W = L + S + Q(R) decomposition finds exactly the structure
Stage A predicted it would (the a2 subspace, the d1 outlier channels, Gaussianized
residual), but at honestly-accounted total bits no L+S arm beats plain
rotate-then-RTN anywhere on the tested rate–distortion grid. Avenue 1 in its tested
form is closed; scope conditions and what survives are recorded below.

Runs: `results/lrs_residual/20260611-181438-a1c0ff8` (Stage A),
`results/lrs_residual/20260611-182023-a1c0ff8` (Stage B + figure
`lrs_rate_distortion.png`). Estimator: `bmx.decomp.lrs` two-step hard-threshold +
truncated-SVD (Wainwright §11.4.2 Eq. 11.58, Agarwal et al. 2012), fit in the original
basis per the grounding in `next-avenues-structured-residual.md`.

## Stage A — structural diagnostic (the assumptions hold)

Five GPT-2 matrices (layer-5 attn/MLP quartet + `wpe`), grid r ∈ {8,16,32,64} ×
frac ∈ {1e-4,1e-3,1e-2}, fp64, original basis. At the reference point r=32, frac=1e-3:

| weight | subspace overlap | supp top-10 overlap | spikiness W → L | rel err (L+S) | kurtosis W → R |
|---|---|---|---|---|---|
| attn.c_attn | 0.988 | 0.10 | 10.9 → 10.1 | 0.884 | +0.74 → +0.31 |
| attn.c_proj | 0.921 | 1.00 | 21.2 → 15.5 | 0.862 | +2.02 → +0.04 |
| mlp.c_fc | 0.979 | 0.80 | 15.5 → 8.2 | 0.891 | +0.94 → +0.23 |
| mlp.c_proj | 0.913 | 1.00 | 29.2 → 25.0 | 0.889 | +3.05 → +0.04 |
| wpe | 0.693 | 0.60 | 37.0 → 8.7 | **0.035** | +31.5 → +8.5 |

- **(a) Subspace: pass.** L̂ recovers W's own top-r left singular subspace at 0.91–0.99
  even with S removed — the same subspace-shaped structure a2's Tucker found.
- **(b) Support: pass for projection matrices.** On both c_proj weights supp(Ŝ) hits
  d1's flagged channels perfectly (top-10 overlap 1.0) and residual kurtosis collapses
  to ≈ +0.04 — the "cleaning Gaussianizes the bulk" mechanism, observed directly.
  c_attn/c_fc outliers are more diffuse (overlap 0.1–0.8).
- **(c) Spikiness: ambiguous, not disqualifying.** Calibration matters: the right
  reference for an *incoherent* rank-r matrix at these dims is
  spikiness ≈ √(2·ln(m·p)) ≈ 5.4, not ~1 (a flat matrix). Measured spikiness_L of
  8–25 (falling with r: whole-grid mean 18.7 at r=8 → 10.3 at r=64) is moderately above the
  incoherent reference — identifiability is strained but not void.
- **The load-bearing number: rel_error_LS ≈ 0.86–0.89** — at r=32 + 0.1 % sparsity,
  L+S captures only ~21 % of the Frobenius energy of a matmul weight. The structure is
  real but **thin**, echoing the entire BM program. (`wpe` is the counterexample —
  almost perfectly L+S at rel 0.035 — but it's an embedding, not a matmul weight.)

## Stage B — matched-total-bits compression (the payoff isn't there)

Four arms (`bmx.quant.arms`): plain RTN, rotate→RTN (seed-generated, 0 stored bits),
L+S→RTN(R), L+S→rotate→RTN(R). Accounting counts the bulk ints, fp16 group scales,
fp16 L factors, and fp16+index sparse entries. fp32, group 64, bits ∈ {2,3,4},
distortion = inner-product error on 512 Gaussian probes against the input dim.

Representative (bits=3): baselines sit at 3.25 bpw with ip ≈ 0.250–0.279; the best
L+S points by raw distortion (r=64, frac=1e-2) reach ip ≈ 0.18 but cost 5.3–6.3 bpw —
beaten by the baselines' own bits=4 points (4.25 bpw, ip ≈ 0.107–0.121) at lower cost.

The closest calls are the *cheap* L+S rows, and they still lose:

| weight (bits=3) | rotate_rtn (3.25 bpw) | lrs r=8 frac=1e-3 (~3.5 bpw) | sparse-only r=0 frac=1e-3 (~3.29 bpw) |
|---|---|---|---|
| attn.c_attn | 0.250 | 0.238 | 0.246 |
| attn.c_proj | 0.252 | 0.243 (frac=1e-4) | 0.250 |
| mlp.c_fc | 0.251 | 0.245 | 0.255 |
| mlp.c_proj | 0.252 | 0.237 | 0.248 |

These beat rotate_rtn on *raw* distortion — but not per bit: along the baseline's own
curve distortion roughly halves per added bit (realizable with mixed 3/4-bit groups),
so +0.25 bpw of rotate_rtn buys ip ≈ 0.205 (vs lrs's 0.237–0.245) and even +0.04 bpw
buys ≈ 0.243 (vs sparse-only's 0.246–0.255). **The L+S grid sits above the
interpolated rotate-RTN rate–distortion curve throughout** (the only exceptions are
six near-degenerate r=0/frac=1e-4 points landing 0.2–0.6 % below it — within
single-seed/512-probe noise and with no usable margin). See `lrs_rate_distortion.png`.

Mechanism: cleaning removes only ~21 % of energy, so the quantizer's dynamic range
barely improves, while fp16 side-storage costs real bits (r=64 alone ≈ 1.8 bpw on a
768-dim matrix). The win-per-bit of side information is worse than spending the same
bits on the bulk.

## What survives the negative

1. **Rotation is the free lunch, confirmed again.** rotate_rtn beats rtn on every
   weight at every bit-width for zero stored bits (e.g. 0.279 → 0.252 on attn.c_proj
   at bits=3).
2. **Rotating the residual after L+S extraction helps iff the residual is still
   non-Gaussian — and hurts once it isn't.** Across the 228 matched grid cells the
   pattern is frac-conditional: at frac=1e-2 (aggressive extraction, kurtosis_R ≈ 0)
   plain lrs_rtn wins 60/60; at frac ≤ 1e-4 (little extracted) the rotated-residual
   variant still wins almost everywhere; frac=1e-3 is the crossover (32/60). This is
   the vault incoherence note's failure mode observed as a dose–response curve: RHT's
   benefit vanishes exactly as the extraction Gaussianizes the bulk.
3. **The theoretical takeaway sharpens the AWQ contrast.** Outlier structure is real
   (Stage A), but *additive* side-information (L, S stored at fp16) is the wrong
   currency for it at these sizes — the treatments that win are bit-free
   (basis change / per-channel scales). AWQ works not because it crudely approximates
   L+S, but because its correction costs ~0 bits.

## Scope conditions (what this negative does NOT establish)

- **Scale.** L-factor overhead per weight is 16·r·(1/m + 1/p): ~1.8 bpw at GPT-2's
  768-dim, but ~0.25 bpw at d ≈ 8k for the same r=64. The overhead argument weakens
  ~10× at frontier dims — if Avenue 1 is ever revisited, revisit it directly on a
  large checkpoint, where the gain side would also need re-measuring.
- Quantizer: groupwise symmetric RTN only (no GPTQ-style compensation); side storage
  fixed at fp16 (no int8 factors); ranks ≤ 64; GPT-2 layer 5; Gaussian probes, not
  real activations; no perplexity eval (gated off — with no rate–distortion win,
  perplexity cannot rescue the arm).

## Gate call

Avenue 1 **closes** in its tested form. The fused-kernel byte-accounting step does not
run (nothing to fuse that pays). Infrastructure delivered and reusable: `decomp.lrs`
(registered, planted-recovery-tested), `quant.arms` (+ honest bit accounting),
`eval.layer_swap` (implemented, offline-tested — general LASER-style eval for any
future fit), the two-stage experiment + rate–distortion plot. Next candidates per
`next-avenues-structured-residual.md`: Avenue 2 (uniform-bound sketched matvec, whose
residual question is now empirically characterized by Stage B's kurtosis data) and the
D0-surviving unbiased-matvec cluster.
