# Structured / streamable rotation for eigenbasis water-filling — design (2026-06-21)

## Status

Design approved (mechanics delegated to implementer; user trusts the technical
shape, asked me to lean on the foundational layer for math calls). Follow-up to
`docs/2026-06-21-k2-eigwaterfill-results.md`: the full C×C KLT won 2.24× on logit
but is KILLED-HONEST (+8 bpe rotation cost → 11.6 vs 3.6 bpe). This tests whether a
**cheaper / streamable rotation captures enough of that win to beat uniform at
HONEST bpe**. Offline, on collected caches, builds on the existing rotated-waterfill
core.

## The question and the pass bar

An arm passes (revives the program toward deployment) iff, at matched HONEST bpe
(rotation cost included), `logit_rope(arm) < logit_rope(uniform)` by > ~2 sem on a
clear majority of layers. The full-KLT idealized win (Llama 0.0161) is the ceiling;
uniform (0.0360) is the floor. Each cheaper rotation lands between, at its own
honest cost; the experiment finds whether any lands below uniform.

## What the residual eigenspectrum actually shows (measured before designing)

Inspected `R = M − truncated_svd(M,16)` channel covariance `RᵀR` on Llama layers
0/15/31 (the residual k2b currently quantizes uniformly):

- **No eigengap.** `λ_k/λ_{k+1}` ≈ 1.00–1.05 at every k, every layer — the spectrum
  decays smoothly, no elbow.
- **Energy is spread**, not concentrated: cumulative energy at k=32 ≈ 0.47–0.62, at
  k=128 ≈ 0.80–0.97. Stable rank ≈ 26–37.

Two foundational consequences (primary sources, read at source):

1. **Truncated-KLT has no natural k** (Wainwright, *High-Dimensional Statistics*
   Ex. 8.2: the rank-r projection `V_r V_rᵀ` is MSE-optimal and *unique only under
   an eigengap* `γ_r > γ_{r+1}`). With no gap, any k is an arbitrary energy-fraction
   choice, and cheap k captures only a fraction of the win. We therefore *sweep* k
   to map the cost/capture frontier rather than pick one.
2. **Frozen-prefill will likely drift** (Davis-Kahan, Wainwright §8.1.2 / Vershynin
   Thm 4.1.15: eigenvectors are unstable under perturbation *precisely when there is
   no eigengap*; eigenvalues stay stable via Weyl, eigenvectors do not). A flat
   spectrum is the worst case for a frozen rotation generalizing. This predicts the
   frozen arm's failure mode before the run — which is why the oracle control and a
   logged eigengap are mandatory.

The honest pre-registered expectation: **block-diagonal per-head is the most likely
deployable winner; frozen-prefill probably drifts; truncated-KLT maps the frontier
but likely loses at low (deployable) cost.** Any outcome is a clean, predicted
result.

## Naming (avoid collision with k2b's low-rank)

k2b's `lowrank_rtn_channel` does a low-rank **subtraction** (`L = UsVᵀ` kept in
fp16, residual quantized). These arms do a low-rank / structured **rotation** of the
residual — a change of basis, not a stored signal component. To avoid conflation,
the truncated-KLT arm is named `..._topk` (top-k eigen-directions), never
"low-rank." All rotation arms run *after* k2b's low-rank subtraction, on `R = M−L`.

## Arms

All extend the existing `_lowrank_rotwaterfill_channel` core (rotate residual →
water-fill → unrotate). New rotation modes:

| arm | rotation Q | honest metadata cost (Llama C=1024,S=2048) | role |
|---|---|---|---|
| `lowrank_rtn_channel` | none (uniform) | 0 | baseline floor |
| `lowrank_eigwaterfill_topk` | top-k eigvecs of RᵀR, C×k; rest unrotated | `16·k/S` (k∈{32,64,128} ⇒ +0.25/0.5/1.0) | cost/capture frontier |
| `blockdiag_eigwaterfill` | per-kv-head d×d KLT (block-diagonal) | `16·d/S` ≈ +1.0 | within-head decorrelation (front-runner) |
| `frozen_eigwaterfill` | full C×C KLT fit on first P prefill tokens, frozen | `16·C/S` (amortizes at long ctx) | streamable full rotation |
| `oracle_eigwaterfill` | full C×C KLT refit on the SAME tokens it scores | n/a (control, not deployable) | drift isolator / upper bound |

The existing `lowrank_eigwaterfill_channel` (full C×C, idealized) stays as the
ceiling reference. `lowrank_randwaterfill_channel` stays as the rotation-vs-
eigenstructure control.

### Mechanics per new mode

- **topk:** `eigh(RᵀR)` → take top-k eigvecs `Q_k` (C×k). Form the FULL orthogonal
  transform that rotates the top-k eigen-directions and leaves the complement
  untouched: project `R` onto the eigenbasis, but only the top-k coordinates are
  "rotated" — concretely, complete `Q_k` to a full orthogonal `Q = [Q_k | Q_⊥]`
  (the remaining eigvecs), rotate `R_rot = R @ Q`, water-fill all C rotated columns,
  unrotate `R_hat = R_rot_hat @ Qᵀ`. The "top-k" economy is in STORAGE: only `Q_k`
  (C×k) is stored; `Q_⊥` is *recomputed* at read as the orthogonal complement of the
  stored `Q_k` (any orthonormal completion works because the complement subspace is
  quantized in a basis-agnostic way — water-filling the complement columns uses a
  fixed/recomputable basis, so the stored cost is `16·k/S`, not `16·C/S`). If the
  recompute-complement adds implementation risk, the simpler honest fallback is:
  rotate ONLY the top-k columns (`R_k = R @ Q_k`, S×k), water-fill them, and
  water-fill the original-basis residual `R − (R_k_hat @ Q_kᵀ)` per-channel for the
  complement. Honest bpe adds `16·k/S` either way. The implementer picks the cleaner
  of the two; the test (`topk reduces to full KLT at k=C`) pins correctness.
- **blockdiag:** reshape `R` to per-head `(S, h_kv, d)`; for each head fit a d×d KLT
  on that head's `RᵀR` block; rotate/quantize/unrotate per head. Honest bpe adds
  `16·d/S` (one d×d matrix per head, but `h_kv·d² / (S·C) = d/S` per entry since
  C=h_kv·d). Decorrelates within-head only — no cross-head mixing.
- **frozen:** fit the full C×C KLT on `R[:P]` (first P tokens), apply to all S.
  `charge_rotation` adds `16·C/S`. Config `prefill_fit_len P` (default 512, the K2c
  minimum-prefill guidance).
- **oracle:** identical to the existing full-KLT idealized arm, but reported
  alongside frozen so `frozen_logit / oracle_logit` is the generalization ratio.

## Decisive structure (what makes the verdict conclusive)

- **Pass bar on HONEST bpe**, asserted, per arm. Idealized bpe still logged as the
  mechanism ceiling, but the verdict is honest-bpe.
- **frozen/oracle ratio** separates "rotation drifts" (ratio ≪ 1) from "structure
  absent" (oracle also fails). K2c's frozen-vs-oracle method, on a full rotation.
- **Per-layer residual eigengap** (`λ_top / λ_{P-ish}`, and the smooth-decay flag)
  logged — Davis-Kahan predictor of frozen drift; explains a frozen loss causally.
- **k-sweep frontier**: `logit` and `bpe_honest` vs k for the topk arm — even if no k
  wins, the curve quantifies how much of the 2.2× survives per bit.
- **Dual metric** (logit + rel_fro) and **query_eigen_alignment** carry over.
- Run on GPT-2 (fast) and Llama-3.1-8B (authoritative, post-RoPE, GQA — block-diag
  needs real GQA head structure to be meaningful).

## Test plan (TDD, before implementation)

1. Core rotation modes:
   - **topk orthogonality / partial-rotation correctness:** rotating only the top-k
     columns and leaving C−k unrotated, then reconstructing, is exact (no quant) to
     < 1e-10 — the partial rotate/unrotate must be lossless on the signal.
   - **topk reduces to full KLT at k=C:** with k=C the topk arm matches the existing
     full `lowrank_eigwaterfill_channel` on logit distortion (the whole basis is
     rotated).
   - **blockdiag is block-orthogonal:** each head's d×d Q is orthogonal (QQᵀ=I <
     1e-12); reconstruct exact with no quant; block-diagonal structure verified (no
     cross-head mixing — zeroing one head's residual leaves others unchanged).
   - **frozen vs oracle on a synthetic with KNOWN drift:** construct R whose channel
     covariance rotates partway through the sequence; assert oracle (refit) beats
     frozen (prefill-fit) — proves the ratio detects drift. And on a stationary R
     (no drift), frozen ≈ oracle.
   - **honest bpe per mode:** topk adds exactly `16·k/S`; blockdiag adds `16·d/S`;
     frozen adds `16·C/S` (hand-checked on a small matrix).
   - `S % group == 0` assert; reconstruction shape; dropped-column zeroing.
2. Experiment smoke: extended `k2_waterfill.py` emits all new arms' rows with
   logit+rel_fro, the eigengap and frozen/oracle columns, matched honest-bpe holds
   for the deployable arms (topk/blockdiag/frozen vs uniform — within tolerance once
   the rotation charge is included).

All offline; tiny synthetics / GPT-2 fixture; no downloads.

## Scope / non-goals

- **In scope:** topk (k-sweep), blockdiag, frozen (+ oracle control), all at honest
  bpe; eigengap + frozen/oracle + alignment diagnostics; dual metric; both models.
- **Deferred:** Hadamard-class fixed rotations (zero stored cost, approximate the
  eigenbasis) — a natural next arm if data-derived structured rotations also die on
  cost, but separate. Streaming-cache integration (the rotation would freeze with
  the subspace at first flush) — only if an arm wins honestly.
- **Keys only** (`k_pre`); values have no usable subspace (K1).
- No new model support, no VM-only work. Reuses the rotated-waterfill core,
  `allocate_channel_bits`, `random_orthogonal`; runs on collected caches.

## Run command (target)

```bash
uv run python experiments/k2_waterfill.py \
    --cache-path results/cache/llama-3.1-8b_2048.safetensors \
    --model-label llama-3.1-8b --model-name meta-llama/Llama-3.1-8B \
    --budget-bits 3.0 --rank 16
```
(new arms run automatically; k-sweep and prefill-fit length via Config.)
