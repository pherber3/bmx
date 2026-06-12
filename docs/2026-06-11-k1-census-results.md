# K1 — KV-cache activation census results (2026-06-11)

E1 from `2026-06-11-kv-research-plan.md`, tested on GPT-2 (S=1024) and
Llama-3.1-8B (S=2048, bf16 CPU prefill, real WikiText tokens).
Runs: `results/k1_cache_census/20260611-221829-d7b7b37` (gpt2),
`results/k1_cache_census/20260611-224022-d7b7b37` (llama-3.1-8b).
Caches themselves are raw data (`results/cache/`, gitignored, regenerable via
`experiments/collect_cache.py`).

## E1 verdict: CONFIRMED at both scales

The structure that is absent from trained weights lives in the cache. Break-even
margins (same instrument and accounting as `2026-06-11-frontier-breakeven.md`,
where this model class's WEIGHTS scored −0.02…+0.23):

| tensor | gpt2 lr_margin | llama-8b lr_margin | llama chan-norm ratio | llama kurt_ch → rotated |
|---|---|---|---|---|
| K (post-RoPE) | +0.60…+2.46 | +0.46…+0.63 | 4.3…19.7 | up to +3.5 → ≈0 |
| K (pre-RoPE) | (= K; no RoPE) | **+0.90…+2.08** | 5.4…21.2 | up to +5.0 → ≈0 |
| V | +0.07…+0.39 | +0.04…+0.47 | 2.9…10.9 | +0.2…+2.4 → ≈0 |
| Q | +0.38…+1.39 | +0.99…+2.15 | 10.5…23.8 | +0.1…+1.5 → ≈0 |

Validity checks (independent reviewer, GPT-2): census numbers reproduce from the
raw cache to 6 decimals; **mean-centering barely dents K margins** (2.46 → 2.41,
0.90 → 0.68) so keys are genuinely low-rank beyond the trivially-predictable
token mean — while V's small margin mostly IS the mean offset (0.12 → 0.02
centered). Keys and values are different objects and should be treated
differently (as KIVI also concluded empirically).

## The headline: RoPE costs ~1–1.5 bits of key compressibility

Pre-RoPE keys score +0.90…+2.08; the same keys post-RoPE score +0.46…+0.63.
RoPE rotates each position by a different angle, smearing the shared subspace —
the *same mechanism* as the Avenue-1 basis lesson (rotation spreads concentrated
structure), now measured on the serving-relevant object. Design implication,
measured rather than asserted: **compress/store keys pre-RoPE and apply RoPE at
read time** (the MLA-style and decoupled-RoPE design point). This goes straight
into K2's arm design: pre-RoPE arms vs post-RoPE arms at matched bits.

## Other structural facts for K2

- **Rogue channels are real and big in activations**: channel-norm max/median up
  to 23.8 (Q), 21.2 (K_pre), ~11 (V) — vs ≈1–2 for this model's weights. The
  per-channel-scale (KIVI) vs rotation (QuaRot) bake-off has genuine structure to
  fight over; E2 is live.
- **Rotation Gaussianizes activation channels** (kurtosis up to +5.0 → ≈0,
  Hadamard at h·d ∈ {1024, 4096}), but norm heterogeneity is the dominant
  structure, not tail shape — the per-channel treatment targets the actual
  disease; rotation targets a symptom that's partly already mild.
- **Cache low-rank economics improve with context**: factor cost per entry is
  16r(S+C)/(S·C) → 16r/C as S grows, so the margins above (measured at S=1–2k)
  only get better at long context — the opposite of the weight-side economics.
- Q is as structured as K (margins to +2.15) — irrelevant for cache storage (Q is
  not cached) but relevant for sketched-attention ideas (Avenue 2's heir).

## Gate G1: OPEN

Low-rank arms are live for K2 (especially pre-RoPE K), per-channel arms are
mandatory baselines, and the random-sphere-vs-real-cache control will decide the
TurboQuant question (E3). K2 implementation plan is the next build.
