# K2 — eigenbasis (KLT) water-filling on key residuals (2026-06-21)

Revival test for the negative in `docs/2026-06-21-k2-waterfill-results.md` (per-
**channel** water-fill KILLED, uniform won 32/32). Cover–Thomas reverse water-
filling is optimal over **eigen-directions**, not raw channels, so this rotates the
key residual `R = M − L` into its own eigenbasis (KLT) before allocating bits, then
unrotates. Spec: `docs/superpowers/specs/2026-06-21-eigenbasis-waterfill-design.md`.
Five arms on `k_pre`, dual metric (logit + MSE), random-rotation control, two-tier
(idealized / honest) accounting. Runs:
`results/k2_waterfill/20260621-165828-16d4df0` (llama, true post-RoPE) and
`results/k2_waterfill/20260621-165332-4c8d8fb` (gpt2). rank=16, group=64,
tiers=(0,2,3,4), budget 3.0.

## Headline: the eigenbasis win is REAL — and KILLED on honest cost

Llama-3.1-8B, mean ± sem over 32 layers (logit_rope = logit distortion vs real
queries, RoPE-at-read; lower better):

| arm | rotation | logit ↓ | MSE (rel_fro) ↓ | bpe (ideal) | bpe (honest) | beats uniform |
|---|---|---|---|---|---|---|
| **`lowrank_eigwaterfill_channel`** | **KLT** | **0.0161 ± 0.0005** | **0.0424** | 3.626 | **11.626** | **32/32** |
| `lowrank_randwaterfill_channel` | random | 0.0401 ± 0.0010 | 0.0884 | 3.626 | (free) | 2/32 |
| `lowrank_rtn_channel` (uniform) | none | 0.0360 ± 0.0010 | 0.0914 | 3.625 | — | — |
| `lowrank_waterfill_channel` (raw, killed) | none | 0.0429 ± 0.0011 | 0.0898 | 3.626 | — | 0/32 |
| `outlier_two_tier` | none | 0.0912 ± 0.0028 | 0.2262 | 3.628 | — | 0/32 |

GPT-2 replicates the ordering: eig 0.0157 ± 0.0020 vs uniform 0.0370 ± 0.0041,
12/12 layers; random control 0.0398 (1/12).

**At matched IDEALIZED bpe, KLT beats uniform by 2.24× on logit (~14 sem of
separation on Llama), on every layer, and wins MSE too.** This is the exact revival
signature the spec predicted — and it inverts the per-channel kill.

## Why this is a real mechanism, not an artifact (falsification survived)

A 2.2× win in a repo whose plan code has had bugs demands skepticism. Every probe
(adversarial review, code re-run on both caches) passed:

1. **bpe-fair.** KLT's realized mean payload is **exactly 3.0** on every probed
   layer — identical to uniform. It drops more channels to tier 0 (L0: 149 vs raw
   waterfill's 5) but funds proportionally more tier-4 channels; the bisection holds
   the mean at budget. The win is not a hidden bit-budget advantage.
2. **Properly scored in the original basis.** The arm unrotates
   (`M_hat = L + R_rot_hat @ Qᵀ`) before scoring — verified byte-exact. If scored
   pre-unrotation, rel_fro vs M would be 0.299 (garbage); unrotated it is 0.027. The
   KLT arm gets **zero** scoring advantage from living in the rotated basis; the
   rotation is logit-neutral (orthogonal), only the post-rotation quantization
   distorts.
3. **Eigenstructure-specific, not "any rotation."** The random-orthogonal control
   (`lowrank_randwaterfill_channel`) **ties/slightly loses** to uniform (0.0401 vs
   0.0360, 2/32 wins). A random rotation spreads variance (TurboQuant-style), giving
   the allocator nothing to bite; the KLT concentrates variance into the directions
   that matter. The control is the load-bearing result: it proves the gain is the
   data eigenstructure, not rotation per se.
4. **Dual metric: KLT wins BOTH logit and MSE.** No eig-wins-MSE-loses-logit
   objective-mismatch signature here. In the eigenbasis the high-variance directions
   the allocator funds ARE directions the queries read (`query_eigen_alignment`
   ≈ 0.59 on Llama: ~59% of query energy lands in the funded eigen-directions). The
   raw-channel kill failed precisely because raw high-variance channels were NOT the
   ones queries read; the KLT fixes that alignment.

## The honest verdict: real but too expensive to encode

The KLT is a per-layer C×C rotation matrix. Charged honestly at `16·C/S` bpe
(C=1024, S=2048 on Llama ⇒ **+8.0 bpe**), the honest bpe is **11.63 vs uniform's
3.63 — a 3.2× cost**. The rotation matrix costs more than 2× the entire 3-bit
payload. As a full stored rotation, the eigenbasis arm is **not deployable**.

This is the **"real structure, doesn't pay for its bits" outcome** — the direct
KV-side analogue of the weights-program frontier law (`ε > 1 − 4^(−Δb)`): the
eigenstructure captures genuine energy in the right directions, but encoding the
basis costs more than the bits it saves on the payload.

## Gate call: MECHANISM CONFIRMED, deployment KILLED-honest — one open follow-up

- The per-channel kill stands as a kill **in the raw basis**. The science is now
  complete: water-filling fails raw-basis because high-variance channels aren't
  query-relevant; it **succeeds in the eigenbasis** because the KLT aligns funded
  directions with query-read directions. The objective-mismatch thesis from the
  prior verdict is refined: it was a *basis* mismatch, not a fundamental
  MSE-vs-logit incompatibility — in the right basis, MSE-optimal allocation IS
  logit-good.
- The k2b recipe (`lowrank_rtn_channel` pre-RoPE keys, uniform residual bits) is
  **unchanged for deployment** — the eigenbasis arm cannot beat it at honest bpe.
- **The one open follow-up the result earns (separate spec, not pursued here):** a
  **structured / streamable rotation** that captures the eigenstructure gain without
  the full C×C charge. Candidates:
  - **block-diagonal per-head KLT** (d×d per kv-head): `16·d/S ≈ +1.0 bpe` instead
    of +8.0 — if even a fraction of the 2.2× win survives at +1 bpe, it could beat
    uniform honestly.
  - **frozen prefill subspace** (fit Q once at prefill, amortize over all decode
    tokens — the K2c streaming pattern): the C×C cost amortizes over the full
    context, so `16·C/S → 16·C/S_total` shrinks at long context.
  - Hadamard-class fixed rotations chosen to approximate the eigenbasis at zero
    stored cost.
  This is the only path from "real mechanism" to "deployable," and it is where the
  next gate should point.

## Caveats (carried honestly)

- **The headline `bpe` column is IDEALIZED (rotation-free).** Any logit-vs-bpe
  Pareto plot MUST use `bpe_honest` for the KLT arm (11.63, not 3.63); plotting the
  idealized column shows a fake Pareto win. (Repo pitfall class: plot scripts must
  account all metadata.)
- The honest-pass gate (`bpe_honest` populated only when KLT beats uniform on logit
  at that layer) is one-directional by design — a NaN there means "not charged,"
  not "free." KLT wins all layers here so it is always charged.
- `outlier_two_tier` bpe remains conservatively overstated (16/group scale on its
  fp16 channels), as in the prior verdict — its decisive loss is real.
- GPT-2 `logit_rope` is stored-basis (no RoPE); Llama is the authoritative
  post-RoPE result.
