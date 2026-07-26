# K4 local levers — design (tier-gated int8 promotion + determinant-Jensen Gate-A theorem)

Provenance: USER DECISIONS 2026-07-24 — VM Task 8 RELEASED on the int8
certificate (`docs/2026-07-24-k4-math-actions-results.md` §C: blanket int8
rejected offline by its own certificate, 54% vs the 5% line); tier-gated
int8 T=5 PROMOTED as the Lever-2 redesign (certificate passes at 0.021 both
budgets with ~90% of the charge saving surviving); VM rental stays DEFERRED —
all available local work first. Math source: `docs/2026-07-24-k4-math-review.md`
findings #9 (int8) and #6 (determinant-Jensen). Everything here is local
(gpt2 mechanism scale + analytic); the Llama half rides the already-REQUIRED
rotated-W refit.

Two independent parts, one run (shared close-out). Nothing touches shipped
packs, duel parquets, or frozen streaming numerics.

## Part 1 — Tier-gated int8 decoder (Lever-2 redesign)

**The verdict being promoted:** blanket int8 storage of used decoder columns
fails its own distortion certificate because `added` (int8 noise, flat per
column) and `payload` (`lam_i·4^{−b_i}`, shrunk by 4^5–4^8 at top tiers) are
both lam-weighted and dominated by the same top eigendirections. Storing int8
only the columns with `0 < bits_i ≤ T` and fp16 the rest rescues it: at T=5
the certificate reads 0.021 at both budgets (~42% of the 5% line) with
frac_int8 0.916/0.893 and ~90% of the S=4096 charge saving surviving.

**Design key (collapses the feature to one parameter):** blanket int8 IS the
tier gate at `T = 8` — the tier grid is `{0,2,3,4,5,6,8}`, so `bits ≤ 8`
covers every used column, and `int8_decoder_certificate_tiered(pack, 8)`
already reproduces the blanket certificate bit-for-bit. One threshold
parameter therefore expresses fp-only (None), blanket (8), and every gate;
and `mixed_dec_charge` (endpoint-pinned to `skeptic_charge` at
`c_int8 ∈ {0, c_used}`) is the single accounting formula.

### Mechanism

- `int8_decoder_roundtrip(dec, bits_pc, *, tier_threshold: int | None = None)`
  — new keyword, default None = blanket (today's behavior bit-exact; existing
  call sites unchanged). With a threshold, only columns with
  `0 < bits_pc ≤ tier_threshold` are roundtripped; higher-tier used columns
  stay fp32-as-loaded (they ship fp16 — the fp16 half of the mix is already
  measured ≈ 0 cost by the k4_dec_quant fp16 arm, and is charged 16 bits by
  the accounting). `int8_decoder_certificate_tiered` re-expresses its gating
  through this kwarg (dedup; identical numbers — its ddec columns above T are
  zero either way, pinned).
- `CacheCodecSpec.dec_quant` accepts `"int8_t{T}"` (e.g. `"int8_t5"`) beside
  `"fp32"`/`"int8"`. One parser helper in spectral.py,
  `dec_quant_threshold(dec_quant: str) -> int | None` (`"fp32"` → None,
  `"int8"` → 8, `"int8_t5"` → 5; anything else asserts), used by
  `load_packs_for_spec` (roundtrip at materialization with the parsed
  threshold — packed/streaming inherit as today) and by streaming init.
- Accounting at depth: `StreamingQuantizedCache`/`cache_bits_per_entry`
  replace the `skeptic_charge(dec_bits=…)` call with
  `mixed_dec_charge(C, S, tiers, c_used=mean_c_used, c_int8=mean_c_int8)`
  where each layer's `c_int8 = count(0 < pack.bits ≤ T)` (0 when
  threshold is None). Zero numeric change for existing modes — the endpoint
  identities are already pinned by
  `test_mixed_dec_charge_endpoints_match_skeptic_charge`; add an explicit
  streaming-level bit-exactness pin (fp32 and blanket-int8 arms report the
  same bpe before/after the refactor). Linearity in (c_used, c_int8) keeps
  the mean-across-layers trick exact.
- Recipes: the `_dec8` suffix parser gains `_dec8t{T}` → `dec_quant="int8_t{T}"`
  (parse the longer suffix first). No CACHE_ARMS table changes — arms are
  parser-derived, so `k4_b2.2_dec8t5` / `k4_b2.5_dec8t5` become expressible
  immediately; they ship in results only if the measured gate passes.

### Pre-registered gate (kill-or-confirm, binding)

Measured, same harness and axis as the blanket measurement (16.68% at b2.5,
`results/k4_dec_quant/20260723-130005-b32de01`): `experiments/k4_dec_quant.py`
on the existing gpt2 packs (`results/cache/k4_packs_gpt2.safetensors`,
cache `results/cache/gpt2_1024.safetensors`), budgets **{2.2, 2.5}** (the
blanket run measured only 2.5 — both now), dec modes fp32/fp16/int8 +
`int8_t{4,5,6}`.

- **Gate:** `rel_degradation_int8_t5 = 1 − win_int8_t5/win_fp16 < 5%` at BOTH
  budgets. Pass ⇒ tier-gated T=5 is the promoted Lever-2 arm (accounting
  `int8_t5`); fail ⇒ honest negative recorded AND a certificate-vs-measured
  discrepancy diagnosis is mandatory (the certificate said 2.1% — a fail
  means the certificate's un-modeled terms are large; that finding would
  supersede the certificate methodology and go in the doc).
- **Certificate-vs-measured agreement (reported, never gated):** per
  (budget, layer, T ∈ {4,5,6,8=blanket}): certificate
  `implied_rel_degradation` vs measured per-layer
  `1 − win_T(layer)/win_fp16(layer)` on the same packs — the instrument
  validation the "cheap analytic instruments" pattern (§E of the MA results)
  needs. Orderings must agree (T=4 < T=5 < T=6 < blanket measured, as
  certified); report the scatter.
- Deployment view per mode at its OWN bits via `mixed_dec_charge` (own-bits
  wins reported, never gated).

### §3b consequence (honest supersession, not a relabel)

The duel doc's `skeptic-v2-int8` column assumed BLANKET dec_bits=8 — now
rejected by the certificate. The column is superseded:

- `experiments/k4_charge_curve.py` gains `int8_frac_variants` computed through
  `mixed_dec_charge` (NOT eff_dec_bits through `skeptic_charge`, which
  double-counts the fp16-scale term by 16·c_used/(S·C) ≈ 0.003 bits): the
  tier-gated column uses the gpt2-measured frac_int8 band
  **[0.893, 0.916]** applied to the Llama C_used — clearly labeled a
  gpt2-band ESTIMATE (Llama per-tier counts do not exist in any committed
  artifact). Report the new column + crossovers as a band.
- `experiments/k4_fit_packs.py` metrics schema gains per-tier count columns
  (`n_t0…n_t8`) so the rotated-W Llama refit records the exact counts and the
  band collapses to a point on the next rental. Additive, old parquets
  unaffected.
- Duel-doc §3b edit (staged with the results doc, user-approved commit):
  blanket column annotated "rejected by its own certificate — record only";
  tier-gated band column + crossover statement added; the "accounting
  projection pending its own quality gate" caveat replaced by the measured
  gate's outcome.

## Part 2 — Determinant-Jensen Gate-A theorem (math review #6)

**What it anchors:** Gate A's honest negative (corpus-basis retention
0.56–0.69 of per-sequence oracle, flat under corpus tripling) becomes a
predictive, corpus-size-independent statement. Under continuous high-rate
allocation, fixed shared W, pooled fit Σ̄ = E_s[Σ_s]:

- **Identity** (unbiasedness of the pooled water level):
  `E_s[D_pool(s)] = C·κ̄ = C·GM(λ̄)·4^{−B̄}` exactly — pooled-basis
  misalignment costs zero on average; the entire shortfall is on the oracle
  side.
- **Bound** (Minkowski concavity of det^{1/C} on PSD):
  `R = E_s[D_oracle]/E_s[D_pool] = E_s[det^{1/C}(Σ_s)]/det^{1/C}(Σ̄) ≤ 1`,
  the spectral Jensen gap of the sequence-moment mixture — a population
  functional, hence corpus-size-independent (the measured signature).
  W cancels: with a fixed shared W, `det(W^{1/2}Σ_sW^{1/2}) = det(W)·det(Σ_s)`
  in numerator and denominator alike.

### Mechanism

- `spectral.py`: `jensen_gap_report(moments) -> dict` — fp64, per-layer:
  `gm_pool = det^{1/C}(Σ̄)`, `mean_gm_seq = E_s[det^{1/C}(Σ_s)]`,
  `r_pred = mean_gm_seq/gm_pool`, `log_gap = (1/C)(logdet Σ̄ − E_s logdet Σ_s)`
  via `slogdet` with eigenvalue clamp guard (assert min eig > −tol·max eig,
  clamp at tiny positive floor; report n_clamped).
- `experiments/k4_jensen_gap.py` (tyro): per (layer): per-cache W-weighted
  moments Σ_s via the SAME machinery the packs use (`query_position_moment`
  / `corpus_fit_bases` conventions, `w_rope` default frozen), Σ̄ = mean.
  Emits per-layer:
  - `r_pred` (budget-free — the theorem's number);
  - `r_discrete` per budget: E_s[D_oracle_disc]/E_s[D_pool_disc] with the
    ACTUAL tier allocator (`allocate_bits_from_variance`) — oracle = per-cache
    eigenbasis + own allocation on λ(Σ_s); pooled = pooled basis/allocation
    scored on `diag(EᵀΣ_sE)`; D = Σ_i diag_i·4^{−b_i}, all analytic from
    moments (no cache re-quantization);
  - identity check `E_s[D_pool_disc]/(C·κ̄)` (deviation from 1 = the
    discrete/zero-bit regime distance from the theorem's regime, the
    pre-stated gap attribution);
  - corpus-flatness: `r_pred` on the first 3 caches vs all (expect flat —
    the corpus-size-independence signature);
  - cross-domain diagnostic (reported, not gated): `r_pred` of the MIXED
    wiki+code population vs within-domain — the Jensen gap should widen with
    heterogeneity, quantitatively connecting Gate-A retention and the
    corpus-transfer domain-sensitivity verdict to one functional.
- Substrate: the local gpt2 caches (wiki windows
  `gpt2_1024{,_off1024..off5120}`, code `gpt2_1024_code{,_off*}` — the
  corpus-transfer fleet), C=768. Recorded Llama/win-ratio retention numbers
  are CONTEXT in the doc, not a comparison axis (different metric); the
  within-experiment comparison is R_pred vs R_discrete.

### Pre-registered readout (measurement + anchor, not kill-or-confirm)

`|r_pred − r_discrete| ≤ 0.10` absolute at both budgets (layer-mean) ⇒ the
paper claims "the transfer ceiling is the between-sequence spectral
heterogeneity — an intrinsic population quantity no calibration corpus
fixes", theorem-anchored. Larger gap ⇒ the difference is attributed via the
identity check to tier/zero-bit effects and reported as such (also worth
knowing; expect r_pred ≤ r_discrete-side effects since ~20–25% of directions
sit at zero bits at b2.2–2.5, making R_pred an optimistic bound — the review's
stated caveat). Either way the Minkowski bound + unbiasedness identity enter
the paper's theory section with the measured numbers.

## Non-goals

- No per-layer tier threshold (future work; layer-uniform T binds on layer 1
  — recorded), no eigenvalue shrinkage, no order-3 synthesis.
- No new VM tasks beyond the recorded queue; the Llama tier counts +
  per-cache logdet scalars ride the rotated-W refit (schema additions only).
- No CACHE_ARMS/table registration beyond recipe-parser reach; no touching
  shipped packs or committed parquets.

## Constraints

Repo hard rules (battery before commit; tyro CLIs; fp64 moments / fp32
experiment compute; deterministic seeds; explicit run selection; tiny
offline test models; no downloads in tests; results-doc commits stop for the
user). Default-inert everywhere: `dec_quant="fp32"` and `tier_threshold=None`
reproduce today's outputs bit-exactly; the accounting refactor is
zero-numeric-change, pinned at streaming level. The math-review doc is
authoritative for formulas; on any disagreement, STOP and reconcile.
