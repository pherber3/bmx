# K4 Lloyd payload-quantizer gate — design (K-side codebook swap, pre-rental)

Provenance: pre-rental foundational-textbook deep pass (2026-07-25, personal-brain
vault, MacKay/Lloyd-Max constants + the repo's own measured ĝ table). Finding:
the K-side spectral tiers quantize with uniform RTN (mse_scale) while the
V-side already uses Gaussian Lloyd-Max codebooks; measured ĝ(2)=0.170 vs
Lloyd-Gaussian 0.1177 ⇒ 44%/45%/38%/12% excess payload distortion at tiers
2/3/4/5 — the tiers that decide the 4–8k duel region. The earlier "Lloyd is a
dead end (~1% over uniform)" scoping compared uniform-STEP-tuning, not
uniform-vs-nonuniform codebooks — the wrong constant. User approved this gate
2026-07-25; it is RENTAL-BLOCKING in the same sense the shrinkage gate was:
if it passes, the Llama rotated-W refit ships the winning quantizer.

## Mechanism

- `spectral_quantize` (and `spectral_quantize_packed`) gain
  `quantizer: str = "rtn"` — default bit-exact (pinned). `"lloyd"` quantizes
  each tier's codes against the ANALYTIC Gaussian Lloyd-Max codebook
  (`gaussian_codebook(bits)` — already in codecs.py, the V-side machinery;
  ZERO new metadata) with the group-scale estimated by the same
  alternating-minimization pattern mse_scale uses (assign nearest level ↔
  refit scale minimizing group MSE; deterministic, fp32, fixed iteration
  count). bpe accounting is IDENTICAL by construction: same bits per code,
  same one-fp16-scale-per-group charge — no accounting expression changes.
- `CacheCodecSpec.payload_quant: str = "rtn"` (default-inert, pinned) so the
  streaming/packed spectral write path can ship the winner at the refit;
  recipes suffix `_lq` → `payload_quant="lloyd"` (parser-driven, no table
  changes). Packed sub-byte containers store unsigned level indices — the
  existing tier containers cover the widths; packed-path parity for lloyd is
  exercised at the tiny-fixture level only (the deployment packed run rides
  the rental).

## Pre-registered gate (kill-or-confirm, binding)

Same packs, same allocation, same bpe (identical by construction — a pure
quantizer swap at fixed allocation): on the existing gpt2 packs
(`results/cache/k4_packs_gpt2.safetensors`), heldout caches
(`gpt2_1024.safetensors`, `gpt2_1024_off5120.safetensors`), budgets
{2.2, 2.5}, headline `logit` (gpt2, no RoPE):

**PROMOTE iff heldout win(lloyd)/win(rtn) ≥ 1.02 at BOTH budgets** (the
program's standard real-effect bar) **AND the measured Lloyd ĝ table is
grid-convex** (`_tier_g` assert — required for allocator validity). Fail ⇒
honest negative recorded with the measured per-tier ratios (the theory
predicts 1.2–1.4×; sub-Gaussian top directions and heavy-tailed boundary
directions can shrink it — that is exactly what the measurement decides).

Instruments (reported, never gated):
- Measured ĝ_lloyd table (`k4_g_table` gains the quantizer arm) beside
  ĝ_rtn and the analytic Gaussian constants; per-tier measured/analytic
  ratios.
- Offline certificate: per pack, predicted relative payload-distortion
  reduction Σᵢ λᵢ[ĝ_rtn(bᵢ) − ĝ_lloyd(bᵢ)] / Σᵢ λᵢ ĝ_rtn(bᵢ) — compared to
  the measured win ratio (the cheap-analytic-instrument agreement check,
  same pattern as the int8 certificate).
- High-tier data hygiene: the current measured ĝ(6)/ĝ(8) sit BELOW the
  analytic Lloyd bound (impossible for a real quantizer — sampling-limited
  tail bins); k4_g_table flags tiers where measured < analytic-Lloyd and
  reports the analytic values beside (allocation consumers should prefer
  analytic there; recorded, no allocator default change this cycle).

## Consequences if promoted

The Llama refit fits/ships `payload_quant="lloyd"` (one spec field; the
refit runbook gains one flag) and the refit's g_table measurement uses the
Lloyd arm. If killed: RTN stands, the declination sentence goes in the paper
beside the sign-tier and entropy-coding declinations.

## Non-goals

Empirical (data-fitted) codebooks (metadata charge — needs its own design);
per-tier quantizer mixing; kernel/Triton work (spectral packed dequant is
Torch-side — no kernel blocker; any Phase-B fused-kernel LUT is future
work); no allocator default change (4^{−b} round path untouched; the
lagrange/g_table plumbing already accepts a Lloyd table when wanted).

## Constraints

Repo hard rules (battery before commit; tyro; deterministic; explicit run
selection; tiny offline test models; results-doc commit stops for the USER).
Default-inert everywhere: `quantizer="rtn"`/`payload_quant="rtn"` bit-exact,
pinned. No accounting expression changes anywhere.
