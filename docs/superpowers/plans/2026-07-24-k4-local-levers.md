# K4 local levers — implementation plan (tier-gated int8 + determinant-Jensen)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the promoted tier-gated int8 decoder (one threshold parameter, measured local gate) and the determinant-Jensen Gate-A anchor (R_pred vs measured retention), all local.

**Architecture:** Part 1 collapses fp32/blanket/tier-gated into one `tier_threshold` (blanket ≡ T=8 on the {0,2,3,4,5,6,8} grid) with `mixed_dec_charge` as the single accounting formula (endpoints already pinned to `skeptic_charge`). Part 2 is fp64 analytics on cached per-sequence moments — no cache re-quantization anywhere.

**Tech stack:** torch fp64/fp32, tyro, pandas/parquet, existing `bmx.cache.spectral` + `experiments/_k4_common` machinery.

## Global constraints

- Spec: `docs/superpowers/specs/2026-07-24-k4-local-levers-design.md` — where this plan and the spec disagree, STOP and reconcile.
- Default-inert: `dec_quant="fp32"` and `tier_threshold=None` bit-exact vs today; the accounting refactor is zero-numeric-change, pinned at streaming level.
- Battery before every commit: `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` (baseline 558 passed / 17 skipped / 1 xfailed).
- Pre-registered gate (binding, Part 1): measured `rel_degradation_int8_t5 < 5%` at budgets {2.2, 2.5}; fail ⇒ honest negative + mandatory certificate-discrepancy diagnosis.
- Pre-registered readout (Part 2): `|r_pred − r_discrete| ≤ 0.10` layer-mean at both budgets ⇒ theorem-anchored ceiling claim; else attribute via the identity check. Not a kill gate.
- fp64 for all moment/determinant math; explicit run selection; no downloads in tests; results-doc commit STOPS for the user.

---

### Task 1: Tier-gated decoder threshold, at depth

**Files:**
- Modify: `src/bmx/cache/spectral.py` (`int8_decoder_roundtrip`, `int8_decoder_certificate_tiered`, new `dec_quant_threshold`, `load_packs_for_spec`, `mixed_dec_charge` type widen)
- Modify: `src/bmx/cache/streaming.py` (`cache_bits_per_entry` + `StreamingQuantizedCache` init assert path), `src/bmx/cache/specs.py` (docstring), `src/bmx/cache/recipes.py` (suffix parser)
- Test: `tests/test_spectral.py`, `tests/test_streaming_spectral.py`

**Interfaces:**
- Produces: `int8_decoder_roundtrip(dec, bits_pc, *, tier_threshold: int | None = None)`; `dec_quant_threshold(dec_quant: str) -> int | None` ("fp32"→None, "int8"→8, "int8_t{T}"→T with 2 ≤ T ≤ 8, else assert); `mixed_dec_charge` accepts float `c_used`/`c_int8` (linearity; assert unchanged). Tasks 2–3 consume all three.

- [ ] **Step 1: failing tests.** `test_roundtrip_tier_threshold_blanket_equiv` (kwarg None == T=8 == current output, exact equality); `test_roundtrip_tier_gated_masks_columns` (bits>T columns untouched, 0<bits≤T columns equal blanket's, zero-bit untouched); `test_dec_quant_threshold_parse` (all three forms + assert on "int8_t1"/"int9"); `test_load_packs_for_spec_tier_gated` (tiny pack file: T=5 spec → high-tier dec columns bit-identical to file, low-tier roundtripped); `test_streaming_bpe_refactor_bitexact` (existing fp32 + blanket-int8 streaming fixtures: bits_per_entry equals `skeptic_charge`-computed expected value EXACTLY after the mixed_dec_charge refactor); `test_streaming_bpe_int8_t5` (tier-gated spec: bpe_k charge equals hand-computed `mixed_dec_charge` with per-layer gate counts); `test_recipe_dec8t_suffix` (`k4_b2.5_dec8t5` → dec_quant "int8_t5"; `_dec8` still "int8"; parse longer suffix first).
- [ ] **Step 2: implement.** Roundtrip: `gate = bits_pc != 0`, then `gate &= bits_pc <= tier_threshold` when not None — rest of body unchanged over `gate`. `int8_decoder_certificate_tiered` re-expresses its ddec through `int8_decoder_roundtrip(..., tier_threshold=tier_threshold)` (drop the manual zeroing; existing certificate tests must pass unchanged — identical numbers). `load_packs_for_spec`: `thr = dec_quant_threshold(k_spec.dec_quant)`; roundtrip with `tier_threshold=thr` when thr is not None (the `"int8"` branch becomes thr==8 — same result, pinned). `cache_bits_per_entry`: replace the `dec_bits` selection with `mixed_dec_charge(C, S, tiers, c_used=mean_c_used, c_int8=mean_c_int8)` where per-layer `c_int8 = 0 if thr is None else int(((p.bits > 0) & (p.bits <= thr)).sum())`. Recipes: `_dec8t(\d+)` suffix parsed before `_dec8`.
- [ ] **Step 3: battery green (expect ~565/17/1); commit** `feat(spectral): tier-gated int8 decoder — one tier_threshold param (blanket==T8), dec_quant int8_t{T} through materialization + streaming accounting via mixed_dec_charge (zero-numeric refactor, pinned)`

### Task 2: Measured local gate — k4_dec_quant tier sweep + RUN

**Files:**
- Modify: `experiments/k4_dec_quant.py`
- Test: `tests/test_k4_experiments.py`

**Interfaces:**
- Consumes: Task 1's roundtrip kwarg + `dec_quant_threshold` + `mixed_dec_charge`.
- Produces: run dir with `metrics.parquet` (dec_mode now includes `int8_t{T}`), new `cert_vs_measured.parquet` (budget, layer, tier_threshold, implied_rel_degradation, measured_rel_deg), extended `dec_quant_verdict.json` (per-mode rel_degradation; binding gate on `int8_t5`; ordering check T4<T5<T6<blanket).

- [ ] **Step 1: failing tests.** Verdict-function unit tests on synthetic frames: gate binds on int8_t5 only; per-layer measured_rel_deg column matches `1 − win_T/win_fp16` hand-computed; blanket row still reported; deploy bpe per mode equals `bpe_model + mixed_dec_charge(...)` with the mode's gate count (endpoints reproduce the old fp32/fp16/int8 values exactly).
- [ ] **Step 2: implement.** `Config.tier_thresholds: tuple[int, ...] = ()`; DEC_MODES per run = ("fp32","fp16","int8") + `int8_t{T}` for each threshold; `_dec_variant` routes tier modes through the roundtrip kwarg; `bpe_skeptic_deploy` per mode via `mixed_dec_charge` (c_int8 from the pack's bits at the mode's threshold; 0 for fp32/fp16; blanket = thr 8). Verdict: wins per mode; `rel_degradation_{mode} = 1 − win_mode/win_fp16`; `gate_pass` binds on int8_t5 (<0.05 both budgets) when present, blanket reported-not-gated; cert-vs-measured rows via `int8_decoder_certificate_tiered(pack, T)` per (budget, layer, T∈thresholds∪{8}).
- [ ] **Step 3: battery green; commit** `feat(exp): k4_dec_quant tier-threshold sweep — int8_t{T} arms, per-layer cert-vs-measured agreement table, gate rebound to tier-gated T=5`
- [ ] **Step 4: RUN** `uv run python experiments/k4_dec_quant.py --pack-path results/cache/k4_packs_gpt2.safetensors --cache-paths results/cache/gpt2_1024.safetensors --model-label gpt2 --budgets 2.2 2.5 --tier-thresholds 4 5 6` — record run id; verdict per the pre-registered gate. Same cache as the blanket 16.68% measurement (comparability); note it is a fit-slice cache (matches the certificate's exact-on-fit-corpus framing).

### Task 3: §3b supersession machinery — charge-curve mixed column + fit-schema tier counts

**Files:**
- Modify: `experiments/k4_charge_curve.py`, `experiments/k4_fit_packs.py`
- Test: `tests/test_k4_experiments.py`

**Interfaces:**
- Consumes: `mixed_dec_charge` (float-widened).
- Produces: charge-curve output gains per-frac tier-gated columns + crossovers (band); k4_fit_packs metrics rows gain `n_t0,n_t2,n_t3,n_t4,n_t5,n_t6,n_t8` per-layer counts (additive schema).

- [ ] **Step 1: failing tests.** Charge-curve: `int8_frac_variants=(1.0,)` column equals the existing dec_bits=8.0 column EXACTLY (blanket endpoint through mixed_dec_charge); frac 0.9 column sits strictly between the 16.0 and 8.0 columns. Fit-packs: tier-count columns sum to C per (layer, budget) and `n_t0 == n_zero_dirs`.
- [ ] **Step 2: implement.** `Config.int8_frac_variants: tuple[float, ...] = ()`; per frac f the corrected column uses `mixed_dec_charge(C, S, tiers, c_used=cu, c_int8=f*cu)` in place of `skeptic_charge(..., dec_bits=db)` inside the same corrected() blend (NOT eff_dec_bits through skeptic_charge — that double-counts the fp16-scale term). Column/crossover labels carry the frac. Fit-packs: count `(bits == b).sum()` per tier at row-emit time.
- [ ] **Step 3: battery green; commit** `feat(exp): charge-curve tier-gated band columns (mixed_dec_charge, gpt2 frac band) + fit-pack per-tier count schema for the rotated-W refit`
- [ ] **Step 4: RUN** the §3b recompute with the duel doc's exact recorded inputs (doc §3 command: nine `results/k3_niah/20260715-*` dirs + `results/k4_fit_packs/20260713-133628-798d0ef/metrics.parquet`, budgets 2.2 2.5) plus `--int8-frac-variants 0.893 0.916` — record run id + the new crossover band vs tq_b3/tq_k3v2.

### Task 4: Determinant-Jensen anchor — jensen_gap_report + k4_jensen_gap RUN

**Files:**
- Modify: `src/bmx/cache/spectral.py` (add `jensen_gap_report`), `experiments/_k4_common.py` (hoist per-cache moment helper out of `corpus_fit_bases`, zero numeric change)
- Create: `experiments/k4_jensen_gap.py`
- Test: `tests/test_spectral.py`, `tests/test_k4_experiments.py`

**Interfaces:**
- Consumes: `corpus_fit_bases`' moment conventions (w_rope frozen default), `allocate_bits_from_variance`.
- Produces: `jensen_gap_report(moments: Sequence[Tensor]) -> dict` (fp64; keys gm_pool, mean_gm_seq, r_pred, log_gap, n_seq, n_clamped); run dir with per-layer parquet + `jensen_verdict.json`.

- [ ] **Step 1: failing tests.** `jensen_gap_report`: identical moments ⇒ r_pred == 1.0 exactly; random PSD mixture ⇒ r_pred ≤ 1 (Minkowski, tol 1e-12); 2-atom diagonal case matches closed form; non-PSD input asserts. Moment-helper hoist: pooled moment == mean of per-cache moments (exact equality pin on the refactor).
- [ ] **Step 2: implement `jensen_gap_report`.** fp64 `eigvalsh`; PSD guard `evals.min() > −1e-10·max` else assert; clamp floor 1e-300 with n_clamped counted; `gm = exp(mean(log evals))`; r_pred = mean_s gm(Σ_s) / gm(Σ̄); `log_gap = log gm(Σ̄) − mean_s log gm(Σ_s)`.
- [ ] **Step 3: implement `k4_jensen_gap.py`.** Config: `cache_paths` (wiki fleet), `cache_paths_alt: tuple = ()` (code fleet, mixed-domain diagnostic), `model_label`, `budgets=(2.2, 2.5)`, `n_flat=3`, `seed=0`, `out_root=""`. Per layer: per-cache W-weighted Σ_s via the hoisted helper (same conventions as pack fitting); Σ̄ = mean. Emit per layer: r_pred (budget-free); per budget r_discrete = Σ_s D_oracle_disc / Σ_s D_pool_disc with the real allocator — oracle: eig(Σ_s)→λ_s, own `allocate_bits_from_variance`, D=Σ λ_s,i·4^{−b_i}; pooled: eig(Σ̄)→(E,λ̄), pooled alloc b̄, D=Σ diag(EᵀΣ_sE)_i·4^{−b̄_i}; identity check `mean_s D_pool_disc / (C·GM(λ̄)·4^{−B̄})`; flatness r_pred@first-n_flat vs @all; mixed-domain r_pred over wiki+code when alt paths given. Verdict json: layer-means, abs gap, `match = gap ≤ 0.10` both budgets, flatness delta, mixed-vs-within r_pred.
- [ ] **Step 4: battery green; commit** `feat(exp): k4_jensen_gap — determinant-Jensen Gate-A anchor (Minkowski bound + water-level identity) on cached per-sequence moments`
- [ ] **Step 5: RUN** `uv run python experiments/k4_jensen_gap.py --cache-paths results/cache/gpt2_1024.safetensors results/cache/gpt2_1024_off1024.safetensors results/cache/gpt2_1024_off2048.safetensors results/cache/gpt2_1024_off3072.safetensors results/cache/gpt2_1024_off4096.safetensors results/cache/gpt2_1024_off5120.safetensors --cache-paths-alt results/cache/gpt2_1024_code.safetensors results/cache/gpt2_1024_code_off1024.safetensors results/cache/gpt2_1024_code_off2048.safetensors results/cache/gpt2_1024_code_off3072.safetensors results/cache/gpt2_1024_code_off4096.safetensors results/cache/gpt2_1024_code_off5120.safetensors --model-label gpt2` — record run id; readout per the pre-registered rule.

### Task 5: Results doc + §3b amendment + close-out

**Files:**
- Create: `docs/2026-07-25-k4-local-levers-results.md`
- Modify: `docs/2026-07-15-k4-duel-results.md` (§3b supersession), ledger, memory

- [ ] **Step 1:** Results doc: Part-1 gate verdict + cert-vs-measured agreement (the instrument-validation figure of merit), new §3b crossover band, Part-2 r_pred/r_discrete table + theorem statement + gap attribution, honest limits, VM-rider lines (rotated-W refit records exact tier counts + per-cache logdet). §3b edit: blanket column annotated "rejected by its own certificate — record only", tier-gated band + crossovers added.
- [ ] **Step 2:** Combined final review (ml-research lens over the run's science, all numbers recomputed from parquets) + simplify pass sized to the diff; fixes applied.
- [ ] **Step 3:** Battery green; STAGE results doc + §3b edit + run artifacts; STOP for user approval of the results commit. Ledger + memory updated.
