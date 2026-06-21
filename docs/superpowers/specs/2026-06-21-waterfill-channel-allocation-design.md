# Water-filling per-channel bit allocation on key residuals — design (2026-06-21)

## Status

Design approved. Kill-or-confirm experiment. Offline, on collected caches, no
streaming-path change. One science question, decided on logit distortion against
real queries at matched bits.

## The question

The k2b key codec (`lowrank_rtn_channel`) gives **every channel of the
post-low-rank residual the same bit-width** (uniform RTN @3b). A speculative
bridge (UEP ↔ outlier-channel allocation) predicts the optimal allocation is
**reverse water-filling over per-channel variance**. The test:

> At matched total bpe, does variance-allocated mixed-precision on the key
> residual beat uniform-per-channel on logit distortion against real queries?

## Theory grounding (foundational layer)

**Reverse water-filling** — Cover & Thomas Thm 13.3.3, stated in
`D:\Projects\personal-brain\foundational\transformer-theory\Principles and Practice of Deep Representation Learning.md`
Eq. 4.1.11:

```
R_ε(x) = Σ_i ½ log( λ_i / min{κ, λ_i} ),   Σ_i min{κ, λ_i} = ε²
```

The water level `κ` is a **distortion** level, not a rate. Every direction is
coded down to the **same residual variance κ**; directions with `λ_i < κ` are
**dropped entirely** (0 bits). The per-direction allocation is therefore:

```
b_i = max(0, ½ log₂(λ_i / κ))
```

with `κ` the free knob tuned to the bit budget.

**Two assumptions the foundational reading makes load-bearing:**

1. The formula is over **eigen-directions** `λ_i` of a Gaussian source, not raw
   coordinates. Our residual `R = M - L` lives in the raw channel basis (which is
   correlated), so applying water-filling per-channel is an *approximation* of the
   eigenbasis result. We test the per-channel version because that is what is
   cheaply deployable (per-channel scales already exist in `rtn_channel`); the
   eigenbasis version would require an extra rotation we are deliberately not
   adding this round.
2. The low-rank step `L = U_s Vᵀ` **already removes the top eigen-directions** —
   it is itself a crude rank-r water-fill (keep the top directions in fp16
   factors, code the rest). So the sharp hypothesis is: *after low-rank strips the
   dominant directions, is the residual still anisotropic enough across channels
   for a second water-fill to pay?*

This makes the most likely null result (water-fill ≈ uniform) a **clean positive
finding**: low-rank IS the water-filling step, leaving a near-isotropic residual
where uniform-per-channel is already near-optimal. The decisive diagnostic
(residual spectrum flatness) distinguishes that from "water-fill should have
helped but deterministic rounding killed it."

**UEP / deterministic-vs-stochastic boundary** — MacKay Ch. 48 (unequal error
protection) is real but is a *channel-noise* theory; quantization noise is
deterministic rounding. The likely failure boundary of the whole bridge is the
randomized/dithered regime. Our `turboquant_prod` collapse on NIAH/LongBench
(`docs/2026-06-21-niah-longbench-frontier-results.md`) is direct evidence the
randomized regime is the losing one on these tasks — carried as the built-in
skeptic.

## Three possible outcomes, all informative

1. **Water-fill wins** → a new allocation rule for k2b keys; promote to the
   streaming codec next.
2. **Tie + flat residual spectrum** → "low-rank is the water-filling step"; uniform
   residual is near-optimal by construction. A real structural finding.
3. **Tie/loss + still-anisotropic residual** → the deterministic-rounding boundary
   bit; consistent with the variance-is-expensive result. The bridge does not
   transfer to this codec.

The spectrum-flatness diagnostic (logged per layer) is what selects among these.

## Architecture — three pieces, reuse everything else

All three live alongside existing k2 machinery and reuse
`bmx.cache.codecs`, `bmx.cache.collect`, `bmx.cache.metrics`, `bmx.cache.rope`,
`bmx.artifacts`.

### Piece 1 — allocation function (pure, in `bmx/cache/codecs.py`)

```
allocate_channel_bits(
    R: Tensor,            # (S, C) fp32 residual, channel = column
    budget_bits: float,   # target average bits/channel (e.g. 3.0)
    tiers: tuple = (0, 2, 3, 4),
    *, axis: int = 0,     # token axis; variance taken over this axis per channel
) -> Tensor               # (C,) int8 bit-width per channel
```

- Per-channel variance `σ²_c = R.var(dim=axis)` (C values).
- Continuous target `b_c = max(0, ½ log₂(σ²_c / κ))`.
- Binary-search `κ` so `round_to_tiers(b_c).mean() == budget_bits`
  (monotone in κ: larger κ ⇒ fewer bits ⇒ search is well-posed). Fixed iteration
  count, no `while True`. If exact budget unreachable by tiering, land on the
  closest-at-or-below and report the realized mean.
- Round each `b_c` to nearest tier; tier 0 = channel dropped (coded as 0,
  reconstructed as 0.0).
- Deterministic (no RNG). Pure function of `(R, budget_bits, tiers)`.

Returns the per-channel bit array; the **realized** mean bpe is recomputed by the
codec for honest accounting (do not trust the target).

### Piece 2 — codec arm `lowrank_waterfill_channel` (in `bmx/cache/codecs.py`)

Identical to `_lowrank_rtn_channel` except the residual RTN uses **per-channel
bit-widths** from `allocate_channel_bits` instead of a single uniform `bits`:

- Same SVD / low-rank / fp16-factor path (reuse, do not duplicate
  `truncated_svd`; accept the same `svd_factors` param).
- Residual `R = M - L`; allocate `b_c`; quantize each channel-group at its
  assigned bit-width via per-channel RTN (group along token dim, as `rtn_channel`
  does). Channels at tier 0 reconstruct to 0.
- **Honest bpe** (per entry, all metadata counted — a hard repo rule):

  ```
  bpe = mean_c(b_c)              # residual payload, averaged over channels
      + 16 / group               # per-channel-group fp16 scale (one per group, as rtn_channel)
      + 16 * rank * (S + C) / (S * C)   # low-rank fp16 factors (U_s, V), as lowrank_rtn_channel
      + ceil(log2(|tiers|)) / S   # tier map: one tier-index per channel (C indices),
                                  #   ceil(log2|tiers|) bits each, spread over S*C entries
                                  #   => per-channel cost ceil(log2|tiers|)/S per entry
  ```

  Computed exactly in code and returned as the codec's `bpe`; the experiment
  compares on this realized value, never the nominal budget.
- Registered in `CACHE_ARMS` and `S_DIVISIBILITY_ARMS` (it asserts `S % group == 0`
  like its parent).

Add to `quantize_cache` dispatch. `quantize_kv_layout` needs no change (it routes
on `spec.arm`); the new arm flows through the existing `(S,C)` matrix path.

### Piece 3 — experiment `experiments/k2_waterfill.py`

A focused fork of `k2_cache_arms.py` (do NOT extend the big sweep; keep this thin).
Per layer, on `kind = k_pre` only:

- **Baseline**: `lowrank_rtn_channel` @3b (uniform), the rank from config.
- **Water-fill**: `lowrank_waterfill_channel` at **matched bpe** (budget set so the
  realized mean matches the baseline's realized bpe within tolerance; the codec
  reports both, assert |Δbpe| < 0.05).
- **Outlier two-tier** (cheap third arm): top-k highest-variance residual channels
  → fp16, rest → low bits, k and low-bits set to match the same bpe. The specific
  heuristic the bridge critiques.

For each arm, per layer, record:
`model, layer, kind, arm, rank, bpe, rel_fro, logit_rope` (RoPE-at-read, real Q,
GQA-aware — reuse `apply_rope` + `logit_distortion`), plus the **diagnostic**:
`resid_stable_rank` = `(Σλ)/λ_max` of `RᵀR`, and `resid_kept_frac` = fraction of
channels at tier > 0 in the water-fill arm.

Writes parquet via `artifacts.create_run("k2_waterfill", cfg)`; prints a
per-(arm) mean-logit summary and the spectrum-flatness column.

tyro `Config`: `cache_path`, `model_label`, `model_name` (RoPE), `budget_bits=3.0`,
`group=64`, `rank=16`, `tiers=(0,2,3,4)`, `seed=0`. Reuse the layer-key regex and
RoPE-validation block from `k2_cache_arms.py` verbatim.

## Metrics & honesty rules (repo conventions, restated)

- Score on **`logit_rope`** (logit distortion vs real stored queries, post-RoPE
  basis), never Frobenius — Frobenius inverts rankings under rogue channels. Keep
  `rel_fro` as a secondary column only.
- Compare on **realized bpe**, recomputed by the codec with all metadata counted
  (scales, factors, tier indices). Never compare on rank or nominal bits.
- fp64 in tests, fp32 in the experiment.
- Matched-bpe is enforced by assertion, not assumed.

## Test plan (TDD, written before implementation)

1. `allocate_channel_bits`:
   - high-variance channels get ≥ as many bits as low-variance ones (monotone).
   - realized mean ≈ budget for a range of budgets on a known synthetic.
   - sub-κ channels dropped to tier 0 when budget is tight.
   - isotropic input ⇒ ~uniform allocation (the degenerate water-fill).
   - determinism: same input ⇒ same output.
2. `lowrank_waterfill_channel`:
   - reduces to `lowrank_rtn_channel` when `tiers=(b,)` single uniform tier
     (equivalence check vs the existing arm, rel < 1e-6).
   - honest bpe matches a hand-computed value on a small fixed matrix.
   - `S % group == 0` assertion fires.
   - reconstruction shape == input shape; dropped channels are exactly 0.
3. Experiment smoke: runs on the GPT-2 cache fixture, emits parquet with the
   expected columns and a matched-bpe assertion that passes.

All offline; tiny models from `tests/factories.py`; no downloads.

## Scope / non-goals

- **Offline only.** No `StreamingQuantizedCache` change this round. If water-fill
  wins, promoting it to the streaming codec is a separate follow-up (the per-channel
  bit map would freeze with the subspace at first flush).
- **Per-channel, not eigenbasis.** We do not add a rotation to decorrelate before
  allocating; that is the natural next experiment if the per-channel version is
  promising but capped by residual correlation.
- **Keys only** (`k_pre`). Values have no usable subspace (K1); water-filling a
  structureless residual is not motivated.
- No new model support, no VM run. Llama-3.1-8B + GPT-2 caches already collected.

## Run command (target)

```bash
uv run python experiments/k2_waterfill.py \
    --cache-path results/cache/llama-3.1-8b_2048.safetensors \
    --model-label llama-3.1-8b \
    --model-name meta-llama/Llama-3.1-8B \
    --budget-bits 3.0 --rank 16
```
