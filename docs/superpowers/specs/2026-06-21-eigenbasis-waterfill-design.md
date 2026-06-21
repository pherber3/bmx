# Eigenbasis (KLT) water-filling on key residuals — design (2026-06-21)

## Status

Design approved (mechanics delegated to implementer judgment; user trusts the
technical shape). Kill-or-confirm REVIVAL test for the negative in
`docs/2026-06-21-k2-waterfill-results.md`. Offline, on collected caches, no
streaming-path change. Builds directly on the existing `lowrank_waterfill_channel`
arm.

## The question

The per-channel water-fill was KILLED (uniform wins 32/32 layers) because
Cover–Thomas reverse water-filling allocates over **eigen-directions** `λ_i`, but
we applied it in the raw (correlated) channel basis. This experiment rotates the
residual into its own **eigenbasis** (KLT) first, so water-filling operates on its
native domain, then asks: at matched bpe, does eigenbasis water-fill beat uniform
on logit distortion vs real queries?

## Theory grounding (foundational layer, read at source)

- **The eigenbasis IS the optimal domain for water-filling.** Cover–Thomas
  Eq. 4.1.11 (`foundational/transformer-theory/Principles and Practice of Deep
  Representation Learning.md`): `R_ε(x) = Σ_i ½ log(λ_i / min{κ, λ_i})` is defined
  over the covariance eigenvalues `λ_i`. Decorrelating first is not a hack — it is
  the prerequisite the formula assumes. This is the strongest theoretical case the
  revival has.
- **Orthogonal rotation preserves inner products exactly.** `⟨Qy, Qx⟩ = ⟨y, x⟩`
  for orthogonal `Q`. So the KLT rotation is **logit-neutral** — the attention
  metric sees only the post-rotation quantization error, not the rotation. This is
  the escape from the MSE-vs-inner-product trap that killed the raw-channel version
  (`[[Vector Quantization Distortion Objectives]]`).
- **The honest caveat that keeps this a real bet.** Water-fill's optimality (even
  in the eigenbasis) is for **MSE distortion** — the same objective that proved to
  be the wrong target for the logit metric in the killed experiment. The eigenbasis
  makes the allocation correctly MSE-optimal; it does NOT change which objective is
  optimized. So eigenbasis water-fill will likely beat the raw-channel version on
  MSE, but it beats *uniform* on the **logit** metric only if the high-variance
  (high-`λ`) eigen-directions are also the directions queries read. That alignment
  is an empirical question the theory cannot settle — which is exactly why this is
  worth running.

**Sharpened prediction:** eigenbasis water-fill is MSE-optimal and should beat the
killed raw-channel arm; whether it beats uniform on logit depends on query–
eigendirection alignment (logged as a diagnostic, see below).

## Architecture — one new arm, reuse everything else

A new codec arm `lowrank_eigwaterfill_channel` in `src/bmx/cache/codecs.py`, built
directly on `_lowrank_waterfill_channel` so the diff is small and the comparison is
controlled. Steps:

1. Low-rank `L = U_s Vᵀ`, residual `R = M − L` — identical to the existing arms
   (same SVD path, same fp16 factor roundtrip, accept the same `svd_factors`).
2. **Fit the KLT**: `cov = Rᵀ R` (C×C); `Q = eigenvectors(cov)` via
   `torch.linalg.eigh` (symmetric PSD → real eigvecs), columns ordered by
   descending eigenvalue. `Q` is orthogonal.
3. **Rotate**: `R_rot = R @ Q` — columns are now decorrelated eigen-directions,
   variance maximally concentrated.
4. **Water-fill on `R_rot`** using the EXISTING, already-tested
   `allocate_channel_bits` (no new allocator). Per-eigencolumn RTN at the allocated
   bit-widths; tier-0 eigencolumns dropped (reconstruct from `L` only).
   - Caveat to handle: `R_rot` columns may not satisfy the `S % group == 0` per-
     column RTN grouping the raw arm relied on. The rotation is over the channel
     (column) axis, and RTN here groups along the token axis exactly as
     `_lowrank_waterfill_channel` does — so the grouping constraint is unchanged
     (`S % group == 0`), since S is untouched by a channel-axis rotation. Keep the
     same assertion.
5. **Unrotate**: `R_hat = R_rot_hat @ Qᵀ`, then `M_hat = L + R_hat`. Orthogonality
   of `Q` makes `⟨q, M_hat⟩` distortion equal to the in-eigenbasis quantization
   error — the rotation contributes none.

### Per-layer scope

One C×C rotation per layer, fit from that layer's residual covariance. Maximal
variance concentration. (Global / structured-rotation variants are explicitly out
of scope this round — see Non-goals.)

### Honest-bpe — two-tier (the cost knob)

The C×C rotation is expensive metadata (Llama C=1024 → 2MB fp16/layer). Account in
two passes:

- **Idealized pass (mechanism ceiling):** rotation charged **0 bits**.
  `bpe = mean(b_c) + 16/group + 16·rank·(S+C)/(S·C) + ceil(log2(|tiers|))/S`
  — identical accounting to `_lowrank_waterfill_channel`. Tests whether the
  mechanism helps AT ALL before paying for it.
- **Honest pass (only if idealized wins):** add `+ 16·C/S` for the C×C fp16
  rotation matrix amortized over S tokens. At Llama S=2048, C=1024 this is +8 bpe —
  likely fatal. If it kills here, that is itself the finding: **the eigen-structure
  is real but too expensive to encode** (direct analogue of the weights-program
  frontier law `ε > 1−4^(−Δb)`: side-information that captures real energy but
  doesn't pay for its bits).

The arm returns the **idealized** bpe by default (a `charge_rotation: bool` flag,
default False, switches to honest). The experiment runs the idealized pass first
and only invokes the honest pass on a per-layer win.

## Experiment

Extend `experiments/k2_waterfill.py` (do NOT fork — running all arms on identical
residuals in one pass is the controlled comparison). Add the
`lowrank_eigwaterfill_channel` arm alongside the existing three. Per layer on
`k_pre`:

- existing: `lowrank_rtn_channel` (uniform baseline), `lowrank_waterfill_channel`
  (raw-channel, the killed arm — kept as the within-experiment reference for "did
  the rotation help"), `outlier_two_tier`.
- new: `lowrank_eigwaterfill_channel` at matched **idealized** bpe.

Scoring: same `logit_rope` (RoPE-at-read, real queries, GQA-aware), same
matched-bpe assertion (idealized bpe matched to uniform within 0.05).

New diagnostic column **`query_eigen_alignment`**: the fraction of real-query
energy that projects onto the top-k funded eigen-directions of `R` (k = number of
eigencolumns the allocator gives > 0 bits). This is the quantity that decides
whether a logit win is even possible; log it so the result is explained either way.
Compute as: project queries (in the residual's channel space, GQA-expanded as the
metric does) onto `Q[:, :k]`, ratio of projected energy to total query energy,
mean over query rows.

If the idealized pass shows a per-layer win, the experiment runs the **honest pass**
for those layers (rotation charged) and reports both bpe points.

## Metrics & honesty rules (repo conventions, restated)

- Score on `logit_rope`, never Frobenius as headline (`rel_fro` secondary only).
- Compare on realized bpe, all metadata counted; idealized-vs-honest is the
  explicit two-tier accounting, not a loophole — the idealized number is labelled
  as rotation-free and is a mechanism ceiling, not a deployable compression claim.
- fp64 in tests, fp32 in experiment/codec.
- Matched-bpe enforced by assertion.

## Test plan (TDD, written before implementation)

1. `lowrank_eigwaterfill_channel`:
   - **KLT orthogonality / inner-product preservation:** for a fixed residual and a
     fixed query set, the post-unrotation reconstruction's logit distortion equals
     the in-eigenbasis quantization's logit distortion to tight tolerance — i.e.
     the rotation alone (with NO quantization, tiers=(b,) single uniform tier at
     high b) changes the reconstruction by < 1e-6. Proves `Q` is orthogonal and the
     rotate/unrotate is exact.
   - **Reduces to raw waterfill when Q = I:** construct a residual whose channel
     covariance is already diagonal (independent per-channel Gaussians), so the KLT
     `Q` is the identity (up to sign/permutation). The arm then matches
     `lowrank_waterfill_channel` bit-for-bit (or up to per-column sign, which RTN is
     invariant to — assert via logit distortion equality, not raw tensor equality,
     to sidestep eigvec sign ambiguity). Controlled-difference proof: the arm is
     "raw waterfill + a rotation," nothing else. No production test hook — the
     diagonal-covariance construction forces Q=I naturally.
   - **Honest bpe with rotation charge:** with `charge_rotation=True`, bpe gains
     exactly `16·C/S` over the idealized value on a fixed small matrix (hand-check).
   - **Variance IS concentrated post-rotation:** per-eigencolumn variance CV of
     `R_rot` is strictly greater than per-channel variance CV of `R` on an
     anisotropic synthetic (confirms the KLT did its job — the whole premise).
   - `S % group == 0` assertion fires; reconstruction shape == input; dropped
     eigencolumns contribute zero residual.
2. Experiment smoke: the extended `k2_waterfill.py` runs on the GPT-2 fixture,
   emits the new arm's rows + `query_eigen_alignment` column, matched-bpe holds.

All offline; tiny synthetics / GPT-2 fixture; no downloads.

## Scope / non-goals

- **Per-layer eigenbasis only.** Global (one shared rotation) and structured
  (block-diagonal / Givens) rotations are deferred. If per-layer wins idealized but
  dies on honest cost, the structured variant is the natural follow-up (cheaper
  metadata might survive the honest charge) — but that is a separate spec.
- **Offline only.** No streaming-cache change. If eigenbasis wins honestly, the
  rotation would freeze with the subspace at first flush — a separate promotion.
- **Keys only** (`k_pre`). Values have no usable subspace (K1).
- **No new allocator, no new model support, no VM-only work.** Reuses
  `allocate_channel_bits`; runs on the GPT-2 + Llama-3.1-8B caches already
  collected.

## Run command (target)

```bash
uv run python experiments/k2_waterfill.py \
    --cache-path results/cache/llama-3.1-8b_2048.safetensors \
    --model-label llama-3.1-8b --model-name meta-llama/Llama-3.1-8B \
    --budget-bits 3.0 --rank 16
```
(eigwaterfill arm runs automatically alongside the existing arms.)
