# K4 Lloyd payload-quantizer gate — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Decide (kill-or-confirm, local, refit-free) whether the K-side spectral tiers ship Gaussian Lloyd-Max codebooks instead of uniform RTN — before the rental, so the Llama refit fits the winner.

**Architecture:** `quantizer="lloyd"` variant inside the existing tier-quantization loop using the V-side's `gaussian_codebook` (analytic, zero metadata); identical bpe by construction; spec field + recipe suffix for deployment; gate = direct heldout win ratio on the existing gpt2 packs (same packs both arms ⇒ same bpe ⇒ no interpolation across arms).

## Global constraints

- Spec: `docs/superpowers/specs/2026-07-25-k4-lloyd-gate-design.md` — on disagreement STOP and reconcile.
- Battery before every commit (baseline 619 passed / 17 skipped / 1 xfailed). Default-inert: `quantizer="rtn"` / `payload_quant="rtn"` bit-exact, pinned.
- GATE (binding): win(lloyd)/win(rtn) ≥ 1.02 at BOTH budgets on heldout caches AND measured ĝ_lloyd grid-convex. No accounting expression changes.
- Results-doc commit stops for the USER.

---

### Task 1: `quantizer="lloyd"` through the spectral quantize path + spec field

**Files:**
- Modify: `src/bmx/cache/spectral.py` (`spectral_quantize`, `spectral_quantize_packed` tier loop), `src/bmx/cache/specs.py` (`payload_quant` field), `src/bmx/cache/recipes.py` (`_lq` suffix), `src/bmx/cache/streaming.py` + `src/bmx/cache/packed_streaming.py` (thread `payload_quant` to the quantize calls)
- Test: `tests/test_spectral.py`, `tests/test_streaming_spectral.py`, `tests/test_packed_spectral.py`

**Interfaces:**
- Produces: `spectral_quantize(M, pack, *, quantizer="rtn")` (and packed twin) — `"lloyd"` assigns each group's codes to `gaussian_codebook(bits)` levels with an alternating scale fit (assign ↔ least-squares scale, fixed ≥3 iterations, deterministic); returned bpe UNCHANGED (same bits + scale charge). `CacheCodecSpec.payload_quant` default `"rtn"`; recipe `k4_b{x}_lq` (and composable with `_dec8t*`).

- [ ] **Step 1: failing tests.** (a) `quantizer="rtn"` output bit-identical to today (pin on a real tiny pack); (b) lloyd on synthetic Gaussian codes at tier 2/3 beats rtn MSE by ≥ 20% (the textbook gap, generous floor); (c) lloyd bpe == rtn bpe exactly (same pack); (d) packed twin: bitwise-faithful composition quantize→pack→unpack→dequant for lloyd codes at tiers {2,3,4} (unsigned index containers); (e) `payload_quant` default-inert at streaming level (bpe + bytes identical to today); recipe `_lq` parses and composes with `_dec8tl` (`k4_b2.5_lq_dec8tl` or documented ordering — implementer picks and pins ONE canonical suffix order).
- [ ] **Step 2: implement.** Lloyd assign: `codes = argmin_j |x/scale − level_j|` via `torch.bucketize` on level midpoints (deterministic); scale update: least-squares `scale = <x, l(codes)>/<l(codes), l(codes)>` per group, ≥3 alternations from the group-std init; dequant `scale · level[codes]`. Keep fp32; clamp degenerate all-zero groups (scale floor as rtn does).
- [ ] **Step 3: battery; commit** `feat(spectral): lloyd payload quantizer — analytic Gaussian Lloyd-Max codebook arm for K-side tiers (quantizer=/payload_quant=, default-inert rtn bit-exact; identical bpe by construction)`

### Task 2: instruments + the gate RUN

**Files:**
- Modify: `experiments/k4_g_table.py` (quantizer arm + analytic-reference columns + sampling-limited flag)
- Create: `experiments/k4_lloyd_gate.py`
- Test: `tests/test_k4_experiments.py`

**Interfaces:**
- Consumes: Task 1's `quantizer=` kwarg; `load_packs`, `_layer_ctx`/`_score_tail`/`_tq_layer_curve`/`_log_interp` (mirror `k4_w_rope_ab`'s two-variant structure — no causal needed, gpt2).
- Produces: g_table run with `quantizer` column + `analytic_lloyd`/`sampling_limited` columns; gate run dir with `metrics.parquet` (cache, layer, budget, arm ∈ {rtn, lloyd}, dist logit, win, bpe — bpe identical across arms, assert it) + `lloyd_verdict.json` (per-budget win ratio, gate_pass, certificate-predicted reduction vs measured, per-tier measured ĝ ratios).

- [ ] **Step 1: failing tests.** Verdict logic (1.02 both-budgets AND convexity flag); bpe-identical assert; certificate arithmetic on a synthetic frame.
- [ ] **Step 2: implement** both experiments. Certificate: per pack Σᵢ λᵢ[ĝ_rtn(bᵢ) − ĝ_lloyd(bᵢ)] / Σᵢ λᵢ ĝ_rtn(bᵢ) from the MEASURED tables (budget-matched allocation from the pack's own bits).
- [ ] **Step 3: battery; commit** `feat(exp): k4_lloyd_gate + g_table quantizer arm — measured Lloyd ĝ, offline reduction certificate, matched-bpe heldout gate`
- [ ] **Step 4: RUN** (PYTHONPATH=. prefix): (a) `k4_g_table` with `--quantizer lloyd` beside the recorded rtn run inputs (same calibration caches/packs — read the prior run config `results/k4_g_table/20260724-111214-a7370f4/config.json` and reuse verbatim + the arm flag); (b) `k4_lloyd_gate --pack-path results/cache/k4_packs_gpt2.safetensors --cache-paths results/cache/gpt2_1024.safetensors results/cache/gpt2_1024_off5120.safetensors --model-label gpt2 --budgets 2.2 2.5`. Record run ids + THE GATE VERDICT.

### Task 3: results doc + close-out

- [ ] Results doc `docs/2026-07-25-k4-lloyd-gate-results.md` (gate verdict, measured-vs-analytic tables, certificate agreement, high-tier hygiene flags, refit consequence); ml-research review of the science (numbers recomputed from artifacts); ledger + memory; battery; STAGE for user approval; VM-addendum delta (refit flag) recorded.
