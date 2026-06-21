# K2 — water-filling per-channel bit allocation on key residuals (2026-06-21)

Kill-or-confirm for the spec `docs/superpowers/specs/2026-06-21-waterfill-channel-allocation-design.md`.
One question: at matched total bpe, does reverse-water-filling per-channel bits on
the **post-low-rank key residual** beat uniform-per-channel RTN, scored on logit
distortion against real stored queries (`logit_rope`, lower better)? Three arms on
`k_pre` only: `lowrank_rtn_channel` @3b (uniform baseline), `lowrank_waterfill_channel`
(Cover–Thomas `b_c = max(0, ½ log₂(σ²_c/κ))`, κ binary-searched to matched bpe),
`outlier_two_tier` (top-k high-variance channels → fp16, rest low-bits, matched bpe).
Runs: `results/k2_waterfill/20260621-154330-fb3e03c` (gpt2, S=1024),
`results/k2_waterfill/20260621-154628-e418317` (llama-3.1-8b, S=2048, true post-RoPE
scoring). rank=16, group=64, tiers=(0,2,3,4), budget 3.0.

## Headline: uniform wins everywhere at matched bits

| model | arm | bpe | logit_rope ↓ | uniform beats it (per-layer) |
|---|---|---|---|---|
| gpt2 | `lowrank_rtn_channel` (uniform) | 3.833 | **0.0370** | — |
| gpt2 | `lowrank_waterfill_channel` | 3.835 | 0.0406 | 11/12 |
| gpt2 | `outlier_two_tier` | 3.846 | 0.0936 | 12/12 |
| llama-3.1-8b | `lowrank_rtn_channel` (uniform) | 3.625 | **0.0360** | — |
| llama-3.1-8b | `lowrank_waterfill_channel` | 3.626 | 0.0429 | 32/32 |
| llama-3.1-8b | `outlier_two_tier` | 3.628 | 0.0912 | 32/32 |

bpe matched within tolerance (|Δ| < 0.02). Water-fill loses by +10% (gpt2) / +19%
(llama) mean logit distortion and is dominated layer-by-layer (11/12, 32/32). The
two-tier outlier heuristic — the construction the UEP bridge actually critiques —
loses by 2.5×. Clean kill on all three theory-predicted allocations.

## Is the residual flat (Outcome 2) or anisotropic (Outcome 3)?

The spec splits the negative on the residual spectrum. The logged `resid_stable_rank`
(`‖R‖²_F / σ²_max` of the rank-16 residual) is ~29 (gpt2) / ~33 (llama) — but that is
only **2.5–4.1% of the full channel count** (C = 768 gpt2, C = 1024 llama). So the
residual is *not* eigen-isotropic: energy still concentrates in ~30 effective
directions out of ~1000.

But eigen stable-rank is a **rotational** property; water-filling here allocates in the
**raw per-channel** basis. The decisive quantity is the spread of *per-channel*
variances `σ²_c = R.var(dim=tokens)` — that is exactly what `b_c` keys on. Reconstructed
`R = M − truncated_svd(M, 16)` from the caches (`k_pre`, representative layers):

| model | per-chan var CV | max/median | p90/p10 | implied `b_c` std (bits) | eigen stable-rank (% of C) |
|---|---|---|---|---|---|
| gpt2 (L0/5/11) | 0.44 | 3.1× | 3.7× | 0.18 | ~30 (3.9%) |
| llama-3.1-8b (L0/15/31) | 0.81 | 7.7× | 9.8× | 0.33 | ~31 (3.0%) |

The per-channel variances are **clearly spread, not uniform**: llama channels span a
~8× max/median ratio (L0 reaches 15×), and the water-fill formula at κ=median assigns a
real per-channel bit differential (std 0.33 bits, i.e. it genuinely wants to move bits
from low- to high-variance channels). There *was* anisotropy to exploit. Allocating to
it still lost on every layer.

**Classification: Outcome 3 (anisotropic residual, water-fill still loses)**, with a
faint shade of Outcome 2 only on the deepest gpt2 layers (L11 CV drops to 0.23, where
low-rank has flattened the residual most and water-fill ≈ uniform). The deciding number
is the per-channel variance CV of 0.81 on llama (max/median 7.7×): far from the flat
residual Outcome 2 would require, so this is not "low-rank already did the
water-filling." The variance differential is real; spending bits on it is the wrong move.

## Why this is the expected negative

Outcome 3 is the deterministic-rounding / "variance-is-expensive" boundary the spec
flagged as the built-in skeptic, and it lines up with the standing repo prior:

- **Bias cheap, variance expensive.** The k2 bake-off (`docs/2026-06-12-k2-arms-results.md`)
  already found `turboquant_prod` (unbiased two-stage) dominated on single-matrix
  distortion, and the NIAH/LongBench frontier found the randomized/dithered regime is
  the losing one on these tasks. Reverse-water-filling rests on a Gaussian
  rate-distortion (variance/noise) framing; the per-channel variance left after low-rank
  is **incoherent, quantization-friendly residual noise, not retrieval signal**. The
  high-variance channels the allocator funds are not the channels the query inner-product
  reads — so concentrating bits there buys Frobenius accuracy the logit metric does not
  reward, while starving the dropped (tier-0) and 2-bit channels costs logit distortion
  directly. Uniform per-channel RTN — which already spends a per-channel fp16 *scale* on
  every channel — is the better deterministic-rounding operating point.
- **Cover–Thomas assumes eigen-directions of a Gaussian source.** We applied it to the
  raw (correlated) channel basis by design (deployable; per-channel scales already
  exist). The 30-effective-direction eigen concentration sitting underneath ~1000
  weakly-anisotropic raw channels is precisely the mismatch: the structure water-filling
  could exploit lives in a rotation we deliberately did not add.

## Caveats (carried honestly)

- **`outlier_two_tier` bpe is conservatively overstated.** It charges a 16/group fp16
  scale to its fp16 (top-k) channels that those channels do not strictly need. Its
  decisive 2.5× loss is therefore real and, if anything, understated — a tighter
  accounting would not rescue it.
- **gpt2 `logit_rope` is stored-basis, not post-RoPE.** GPT-2's config has no RoPE, so
  the gpt2 column is logit distortion in the stored basis (`model_name=""`, no
  apply-at-read). Only the llama run is true post-RoPE (real post-RoPE keys vs real
  queries). The verdict holds in both, and llama is the authoritative one.

## Gate call: KILLED

Uniform-per-channel RTN on the post-low-rank key residual stands. Reverse-water-filling
does not pay (loses 11/12, 32/32 at matched bpe); the two-tier outlier heuristic loses
decisively (2.5×). The residual is per-channel anisotropic (CV 0.81, max/median 7.7× on
llama), so this is **Outcome 3**, not "low-rank is the water-filling step" — the
deterministic-rounding boundary, consistent with the repo's bias-cheap/variance-expensive
prior. The k2b recipe (`lowrank_rtn_channel` on pre-RoPE keys, uniform residual bits) is
unchanged; no promotion to the streaming codec.

**Deliberately untested escalation (out of scope):** water-fill in the *eigenbasis* after
a decorrelating rotation (the Cover–Thomas formula's actual domain). The 30-effective-
direction eigen concentration means a rotation *could* in principle revive an allocation
win — but that adds a rotation the streaming subspace would have to freeze at first flush,
and the variance-is-expensive evidence predicts it funds the wrong directions for the
logit metric anyway. Not pursued this round.
