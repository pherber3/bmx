# K4 estimation levers — implementation plan (shrinkage gate, per-layer T_ℓ, figure refresh)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide (kill-or-confirm) whether the Llama refit ships with Ledoit–Wolf-shrunk allocation input; decide per-layer int8 thresholds under a materiality bar; restore paper-figure integrity after the blanket-int8 rejection.

**Architecture:** Shrinkage rides the existing `pack_from_basis(lam_alloc=…)` hook — no allocator change, no accounting change, same basis. Per-layer T_ℓ rides the existing tier-threshold plumbing with a pack-derived per-layer threshold. Figures regenerate from committed parquets only.

**Tech stack:** torch fp64, tyro, pandas/parquet, matplotlib (existing figure script), `experiments/_k4_common.py` fit machinery.

## Global constraints

- Spec: `docs/superpowers/specs/2026-07-25-k4-estimation-levers-design.md` — on disagreement, STOP and reconcile.
- Battery before every commit (`uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q`; baseline 590 passed / 17 skipped / 1 xfailed). No default-path numeric change anywhere.
- Part-1 gate (binding): heldout win(lw) ≥ 1.02 × win(plain) at BOTH budgets AND no matched-budget bpe regression > 0.02 ⇒ promote; else honest negative.
- Part-2 rule (binding, both): measured rel_deg(int8_tl) < 5% both budgets AND charge-saving improvement over uniform T=5 ≥ 0.3 bits/token/model at S=4096; else "certified but immaterial" — do not ship.
- Results-doc commit stops for the USER.

---

### Task 1: `shrink_spectrum` (LW from rows + OAS from spectrum)

**Files:**
- Modify: `src/bmx/cache/spectral.py`
- Test: `tests/test_spectral.py`

**Interfaces:**
- Produces: `shrink_spectrum(lam64: Tensor, *, n: int, method: str, rows: Tensor | None = None) -> tuple[Tensor, float]` — returns (shrunk fp64 spectrum, rho). `method="lw"` requires `rows` (the (n, C) fp32/fp64 W-weighted row matrix the moment was built from; standard Ledoit–Wolf intensity ρ̂ = min(1, (1/n²)·Σ_t‖x_t x_tᵀ − Σ̂‖²_F / ‖Σ̂ − μ̂I‖²_F) computed WITHOUT materializing per-row outer products — use the algebraic form with Gram/moment traces); `method="oas"` needs only (lam64, n) (closed form). Both shrink toward μ̂ = mean(lam64): λ' = (1−ρ)λ̂ + ρμ̂. fp64 throughout; assert rows.shape[1] == C for lw; assert 0 ≤ ρ ≤ 1.

- [ ] **Step 1: failing tests.** (a) synthetic Wishart recovery: population Σ = diag(known decaying spectrum), sample n=2·C rows (seeded), LW-shrunk spectrum has strictly smaller MSE to the population spectrum than raw (averaged over ≥20 trials); (b) OAS same property; (c) ρ = 0 reproduces raw exactly; ρ bounds respected on degenerate inputs (n ≫ C ⇒ ρ near 0; n ≈ C tail ⇒ ρ larger — monotone-in-γ sanity on matched synthetic data); (d) LW algebraic form equals the naive per-row-outer-product formula on a tiny case (C=8, n=16) to 1e-10 (pins the trace-trick implementation); (e) shrunk spectrum preserves order (monotone map) and positivity.
- [ ] **Step 2: implement.** LW numerator via traces: (1/n²)Σ_t(‖x_t‖⁴) − (2/n)·tr(Σ̂·Σ̂) + tr(Σ̂²) …use the standard published form b̄² = (1/n²)Σ_t ‖x_t x_tᵀ − Σ̂‖²_F expanded to ‖x_t‖⁴ and tr(Σ̂²) terms; denominator d² = ‖Σ̂ − μ̂I‖²_F = Σ(λ̂−μ̂)². OAS closed form from tr(Σ̂), tr(Σ̂²), n, C.
- [ ] **Step 3: battery; commit** `feat(spectral): shrink_spectrum — Ledoit-Wolf (rows) + OAS (spectrum-only) allocation-input shrinkage for the lam_alloc hook`

### Task 2: `experiments/k4_shrinkage.py` — the gate + RUN

**Files:**
- Create: `experiments/k4_shrinkage.py`
- Modify (only if a small hoist is needed): `experiments/_k4_common.py`
- Test: `tests/test_k4_experiments.py`

**Interfaces:**
- Consumes: `shrink_spectrum`, `corpus_fit_bases`/`per_cache_weighted_moments`, `pack_from_basis(lam_alloc=…)`, `_score_tail`/`_tq_layer_curve`/`_log_interp` (the A-gate matched-bpe win pattern — read `experiments/k4_charge_alloc.py` and mirror its win-at-matched-v2-bpe machinery).
- Produces: run dir with `metrics.parquet` (arm ∈ {plain, lw, oas} × budget × n_fit × layer × cache rows: win, bpe_v2, c_used, rho), `shrinkage_verdict.json` (gate per the Global Constraints rule + diagnostics: n-scaling table, c_used stability, rho per layer, 0↔2 tier-map deltas).

- [ ] **Step 1: failing tests** on the verdict function (synthetic frames): gate logic (both-budgets AND, the 1.02 factor, the 0.02 bpe guard); n-scaling table shape; rho column carried.
- [ ] **Step 2: implement.** Config(cache_paths fit fleet, heldout_cache_paths, model_label, budgets=(2.2, 2.5), n_fits=(768, 1536, 3072, 0)  # 0 = full, methods=("lw","oas"), seed, out_root). Fit bases per n_fit (subsample fit rows with the seeded generator BEFORE moment building; full = today's path), pack per (method: plain uses lam_alloc=None; lw/oas pass shrink_spectrum output), score heldout wins at matched skeptic-v2 bpe via the TQ-curve interpolation. GATE evaluates ONLY (full-n, lw) vs (full-n, plain).
- [ ] **Step 3: battery; commit** `feat(exp): k4_shrinkage — LW/OAS allocation-input gate (matched-bpe heldout wins, n-scaling diagnostic)`
- [ ] **Step 4: RUN** (PYTHONPATH=. prefix) on the wiki fit fleet vs heldout caches — use the SAME fit/heldout split the corpus-transfer run used for wiki (read its config: `results/k4_corpus_transfer/20260723-190823-8dced47/config.json`) so numbers sit on the recorded axis. Record run id + full verdict JSON.

### Task 3: per-layer T_ℓ — analytic sweep, `int8_tl` mode, measured confirm + RUN

**Files:**
- Modify: `src/bmx/cache/spectral.py` (`per_layer_tier_thresholds(packs, bar=0.05) -> dict[int, int]`, `dec_quant_threshold` accepts `"int8_tl"` → sentinel, `load_packs_for_spec` applies per-layer thresholds), `src/bmx/cache/streaming.py` (per-layer thr in the c_int8 computation), `experiments/k4_dec_quant.py` (`int8_tl` mode)
- Test: `tests/test_spectral.py`, `tests/test_streaming_spectral.py`, `tests/test_k4_experiments.py`

**Interfaces:**
- Produces: `per_layer_tier_thresholds(packs: dict[int, SpectralPack], *, bar: float = 0.05) -> dict[int, int]` — per layer the max T ∈ {2,…,6,8} with `int8_decoder_certificate_tiered(pack, T)["implied_rel_degradation"] ≤ bar`; layers where even T=2 fails get 0 (no int8). `"int8_tl"` flows through materialization + accounting with per-layer thresholds.

- [ ] **Step 1: failing tests.** per_layer_tier_thresholds on the tiny fixture pack (monotone: returned T passes bar, T+1 fails or is 8); `int8_tl` materialization applies DIFFERENT thresholds per layer when the certificate says so; streaming bpe with int8_tl equals hand-computed mixed_dec_charge with per-layer c_int8 at the per-layer thresholds; `dec_quant="int8_tl"` recipe suffix `_dec8tl` parses.
- [ ] **Step 2: implement** (per-layer thr threading: `load_packs_for_spec` computes the dict once at materialization; streaming's per-layer c_int8 uses each layer's own threshold — the mean-linearity argument is unchanged).
- [ ] **Step 3: battery; commit** `feat(spectral): int8_tl — certificate-derived per-layer tier thresholds through materialization + accounting`
- [ ] **Step 4: RUN + verdict.** Analytic first: the T_ℓ map + charge saving vs uniform T=5 on the gpt2 packs at both budgets (extend `experiments/k4_int8_certificate.py`'s tier sweep or a small addition there — read it first). Then measured: `k4_dec_quant --tier-thresholds 5 --dec-tl` (add the int8_tl mode to the sweep run; gate = the two-bar rule). Record: per-layer T map, saving delta vs uniform, measured rel_deg. THE MATERIALITY BAR DECIDES SHIPPING — "certified but immaterial" is the expected outcome if the delta < 0.3 bits/token/model; record it as such without shipping the arm into recipes docs headline (the code stays, default-inert).

### Task 4: figure refresh + results doc + close-out

**Files:**
- Modify: `experiments/plot_k4_paper.py`, `results/figures/*` (regenerated)
- Create: `docs/2026-07-25-k4-estimation-levers-results.md`
- Modify: ledger, memory

- [ ] **Step 1:** Figure edits per spec Part 3 (band replaces blanket dashed + resolved caption; new cert-vs-measured scatter from the committed `cert_vs_measured.parquet`). Regenerate committed figures; verify the script names its input runs explicitly.
- [ ] **Step 2:** Results doc (gate verdicts + diagnostics + figure provenance), ledger + memory updates.
- [ ] **Step 3:** Combined final review (ml-research recomputation from artifacts + simplify lenses) sized to the diff; apply fixes; battery; STAGE results doc + figures + run dirs; STOP for user approval.
