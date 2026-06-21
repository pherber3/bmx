# Eigenbasis (KLT) water-filling on key residuals — design (2026-06-21)

## Status

Design approved with a **maximal-rigor mandate**: "do whatever it takes to give
this the shot it deserves and the scientific rigor to prove or disprove it." That
raises the scope from a single per-layer arm to a full **rotation-scope ablation
with controls and a dual-metric (MSE vs logit) readout**, so the verdict is
decisive, not suggestive. Kill-or-confirm REVIVAL test for the negative in
`docs/2026-06-21-k2-waterfill-results.md`. Offline, on collected caches, no
streaming-path change. Builds on the existing `lowrank_waterfill_channel` arm.

## What "rigorous" means here (the controls that make the verdict decisive)

A single eigenbasis arm could win or lose for the wrong reason. The rigorous test
isolates the cause with three controls:

1. **Random-rotation control** (`lowrank_randwaterfill_channel`): water-fill after a
   random orthogonal rotation instead of the KLT. If eigenbasis beats uniform but
   random-rotation does too, the win is "any rotation helps" (spreads RTN error),
   NOT "the data eigenstructure helps." If eigenbasis beats random, the
   data-derived structure is doing real work. This is the load-bearing control.
2. **Dual-metric readout** (MSE/`rel_fro` AND `logit_rope` for every arm): water-
   fill is provably MSE-optimal in the eigenbasis. If eigenbasis WINS on MSE but
   LOSES on logit, that is direct, measured proof of the objective-mismatch thesis
   (not inferred). If it wins on BOTH, the revival succeeds. Reporting both turns an
   ambiguous logit result into a mechanism diagnosis.
3. **Query–eigendirection alignment** diagnostic: the fraction of real-query energy
   in the funded eigen-directions — the quantity that *predicts* whether a logit win
   is even possible, logged per layer to explain the result causally.

If all three say the same thing, the verdict is decisive either way.

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

## Architecture — two new arms sharing one rotated-waterfill core

Both new arms are the SAME pipeline differing only in how the rotation `Q` is
produced, so they share one implementation and the comparison is perfectly
controlled. A single core `_lowrank_rotwaterfill_channel(M, ..., rotation="klt" |
"random", seed)` in `src/bmx/cache/codecs.py`, exposed as two registered arms:

- `lowrank_eigwaterfill_channel` → `rotation="klt"` (the hypothesis)
- `lowrank_randwaterfill_channel` → `rotation="random"` (the control)

The random rotation reuses the existing `random_orthogonal(C, seed)` from
`bmx.quant.hadamard` (the repo's audited orthogonal sampler — never hand-roll QR,
per the sign-ambiguity pitfall). Steps:

1. Low-rank `L = U_s Vᵀ`, residual `R = M − L` — identical to the existing arms
   (same SVD path, same fp16 factor roundtrip, accept the same `svd_factors`).
2. **Fit the rotation `Q` (C×C, orthogonal)**:
   - `rotation="klt"`: `cov = Rᵀ R`; `Q = eigenvectors(cov)` via
     `torch.linalg.eigh` (symmetric PSD → real eigvecs), columns ordered by
     descending eigenvalue. Variance-concentrating.
   - `rotation="random"`: `Q = random_orthogonal(C, seed)`. Variance-spreading
     (the TurboQuant-style control).
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
two passes, for BOTH new arms:

- **Idealized pass (mechanism ceiling):** rotation charged **0 bits**.
  `bpe = mean(b_c) + 16/group + 16·rank·(S+C)/(S·C) + ceil(log2(|tiers|))/S`
  — identical accounting to `_lowrank_waterfill_channel`. Tests whether the
  mechanism helps AT ALL before paying for it.
  - **Note on the random arm's honest cost:** the random rotation is **seeded**, so
    `Q` regenerates from the seed and costs **0 stored bits** even in the honest
    pass (only the seed, which the repo counts as free, consistent with all
    rotate/sketch arms). So `lowrank_randwaterfill_channel` is honest-by-
    construction — a second, *deployable* control: if random-rotation waterfill
    beats uniform on logit at genuinely matched honest bpe, that alone would revive
    the program (no expensive matrix). The KLT arm must beat THIS to justify a
    data-derived rotation.
- **Honest pass for KLT (only if idealized wins):** add `+ 16·C/S` for the C×C fp16
  rotation matrix amortized over S tokens. At Llama S=2048, C=1024 this is +8 bpe —
  likely fatal. If it kills here, that is itself the finding: **the eigen-structure
  is real but too expensive to encode** (direct analogue of the weights-program
  frontier law `ε > 1−4^(−Δb)`: side-information that captures real energy but
  doesn't pay for its bits). And note: if the random arm already revives the program
  at zero rotation cost, the KLT's matrix cost is moot — the cheap control wins.

The arm returns the **idealized** bpe by default (a `charge_rotation: bool` flag,
default False, switches to honest; ignored for `rotation="random"` which is always
free). The experiment runs the idealized pass first and only invokes the honest KLT
pass on a per-layer idealized win.

## Experiment

Extend `experiments/k2_waterfill.py` (do NOT fork — running all arms on identical
residuals in one pass is the controlled comparison). Five arms per layer on `k_pre`:

- existing: `lowrank_rtn_channel` (uniform baseline), `lowrank_waterfill_channel`
  (raw-channel, the killed arm — the "did rotation help at all" reference),
  `outlier_two_tier`.
- new: `lowrank_eigwaterfill_channel` (KLT, the hypothesis) and
  `lowrank_randwaterfill_channel` (random rotation, the load-bearing control), both
  at matched **idealized** bpe.

**Dual-metric readout — every arm records BOTH `logit_rope` AND `rel_fro` (MSE).**
This is what proves the mechanism: water-fill is MSE-optimal in the eigenbasis, so
the predicted signature of the objective-mismatch thesis is *eigenbasis wins MSE,
loses logit*. Measuring both turns the result from suggestive to diagnostic.

**The decisive comparison matrix** (per layer, then pooled with mean ± sem):

| arm | rotation | expected if revival TRUE | expected if revival FALSE |
|---|---|---|---|
| uniform | none | loses logit | wins logit (the prior) |
| raw waterfill | none | (killed) | loses logit |
| rand waterfill | random | ties/beats uniform logit | loses or ties logit |
| eig waterfill | KLT | **beats uniform AND rand on logit** | wins MSE, loses logit |

Scoring: `logit_rope` (RoPE-at-read, real queries, GQA-aware) + `rel_fro`; matched
idealized bpe asserted to uniform within 0.05.

Diagnostics logged per layer:
- **`query_eigen_alignment`**: fraction of real-query energy projecting onto the
  top-k funded eigen-directions of `R` (k = eigencolumns with > 0 bits). Computed by
  projecting queries (residual channel space, GQA-expanded as the metric does) onto
  `Q[:, :k]`; ratio of projected to total query energy, mean over query rows. The
  quantity that *predicts* a logit win — high alignment + still-losing is itself a
  finding.
- **`resid_stable_rank`** (already present) for the residual context.

Honest pass: if KLT shows a per-layer idealized logit win, re-run those layers with
`charge_rotation=True` and report both bpe points. (Random arm needs no honest pass
— it is already free.)

**Statistical rigor:** per-arm pooled mean ± sem across layers, AND the per-layer
win-rate (how many of N layers each arm beats uniform). A revival claim requires the
KLT arm to beat uniform on logit by > ~2 sem AND on a clear majority of layers AND
to beat the random-rotation control — not a single-number edge. Run on BOTH GPT-2
and Llama-3.1-8B (Llama authoritative: true post-RoPE, strongest rogue-channel
structure, C=1024 gives the eigenstructure the most room to matter).

## Metrics & honesty rules (repo conventions, restated)

- Score on `logit_rope`, never Frobenius as headline (`rel_fro` secondary only).
- Compare on realized bpe, all metadata counted; idealized-vs-honest is the
  explicit two-tier accounting, not a loophole — the idealized number is labelled
  as rotation-free and is a mechanism ceiling, not a deployable compression claim.
- fp64 in tests, fp32 in experiment/codec.
- Matched-bpe enforced by assertion.

## Test plan (TDD, written before implementation)

1. Rotated-waterfill core (`rotation="klt"` unless noted):
   - **Rotation orthogonality / inner-product preservation:** for a fixed residual
     and query set, with NO quantization (single high-bit uniform tier), the
     post-unrotation reconstruction's logit distortion equals the unrotated arm's to
     < 1e-6 — for BOTH `klt` and `random`. Proves `Q` is orthogonal and
     rotate/unrotate is exact for both rotation modes (the whole logit-neutrality
     claim).
   - **Reduces to raw waterfill when Q = I:** construct a residual whose channel
     covariance is already diagonal (independent per-channel Gaussians), so the KLT
     `Q` is identity up to sign/permutation. The KLT arm then matches
     `lowrank_waterfill_channel` — assert via logit-distortion equality (not raw
     tensor equality) to sidestep eigvec sign ambiguity. Controlled-difference
     proof: the arm is "raw waterfill + a rotation," nothing else. No production
     test hook — the diagonal-covariance construction forces Q=I naturally.
   - **Random arm is honest/free:** `lowrank_randwaterfill_channel` bpe equals the
     idealized bpe even with `charge_rotation=True` (seeded rotation costs 0 stored
     bits); and re-running with the same seed is bit-for-bit reproducible.
   - **KLT honest bpe with rotation charge:** `rotation="klt"`,
     `charge_rotation=True` → bpe gains exactly `16·C/S` over idealized on a fixed
     small matrix (hand-check).
   - **Variance IS concentrated by KLT, spread by random:** per-eigencolumn variance
     CV of `R @ Q_klt` is strictly greater than per-channel CV of `R`; per-column CV
     of `R @ Q_random` is strictly LESS (concentration vs spreading — confirms each
     rotation does what it claims, on an anisotropic synthetic).
   - `S % group == 0` assertion fires; reconstruction shape == input; dropped
     eigencolumns contribute zero residual.
2. Experiment: the extended `k2_waterfill.py` smoke-runs on the GPT-2 fixture, emits
   all five arms' rows with BOTH `logit_rope` and `rel_fro`, the
   `query_eigen_alignment` column, and matched idealized bpe holds for both rotated
   arms.

All offline; tiny synthetics / GPT-2 fixture; no downloads.

## Scope / non-goals

- **In scope (the rigor mandate):** per-layer KLT arm, random-rotation control,
  dual-metric (MSE + logit) readout, query–eigendirection alignment diagnostic,
  two-tier (idealized + honest) accounting, per-layer + pooled statistics with the
  random control as a deployable-cost reference. Both GPT-2 and Llama.
- **Deferred (only if per-layer KLT wins idealized but dies on the C×C honest
  cost):** structured rotations (block-diagonal per-head KLT → `16·d/S` metadata
  instead of `16·C/S`; or k Givens rotations) that might survive the honest charge.
  Separate spec, triggered by a positive idealized result. Global (one shared
  rotation across layers) likewise deferred — the per-layer arm upper-bounds it, so
  if per-layer loses idealized, global cannot win.
- **Offline only.** No streaming-cache change. If a rotated arm wins honestly, the
  rotation would freeze with the subspace at first flush — a separate promotion.
- **Keys only** (`k_pre`). Values have no usable subspace (K1).
- **No new allocator, no new model support, no VM-only work.** Reuses
  `allocate_channel_bits` and `random_orthogonal`; runs on the GPT-2 +
  Llama-3.1-8B caches already collected.

## Run command (target)

```bash
uv run python experiments/k2_waterfill.py \
    --cache-path results/cache/llama-3.1-8b_2048.safetensors \
    --model-label llama-3.1-8b --model-name meta-llama/Llama-3.1-8B \
    --budget-bits 3.0 --rank 16
```
(eigwaterfill arm runs automatically alongside the existing arms.)
