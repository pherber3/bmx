# K4 — Spectral Allocation Codec: Design Spec

**Date:** 2026-07-12
**Status:** Design approved by user (this session); implementation plan pending.
**Program tag:** K4 (successor to K2 codec science + K3 streaming; K3 machinery untouched).
**Prior art being unified:** k2t (`lowrank_turboquant`, commit `6f33650`), eig-waterfill
(`k2_blockklt`, commits `e53cf00`/`3d89bcf`), asymmetric turboquant arms (`5720588`).

---

## 1. Problem statement

The b3 verdict (commit `91d7b1d`, 2026-07-08) killed the k2b quality claim:
`turboquant_mse_b3` scores 40.56 LongBench Avg @ 3.21 measured bits vs k2b's 40.62 @ 3.94 —
TurboQuant dominates on the Avg-vs-bits plane, and at the low end b2 (33.31 @ 2.22b) beats
k2b_k2r8 (28.69 @ 2.89b). The user's acceptance bar (2026-07-08, verbatim): **"a better
compression technique than turboquant at matched memory usage"** — and, sharpened this
session: not one operating point but **Pareto-dominance of TurboQuant's quality-vs-bits
frontier wherever quality is winnable**, with matched-quality-at-fewer-bits as the win form
where quality saturates (≥~3.5b everyone ties fp16).

Three post-verdict measured findings point at the same mechanism and have never been cashed
at task level:

1. **k2t real-cache gate positive** (`results/k2d_lrtq_gate/20260708-*`, all 32 Llama-3.1-8B
   layers): lowrank+TQ-residual K @ 2.39 bpe → logit_rope ≈ 0.05 vs turboquant_mse@3b's
   ≈ 0.09 @ 3.02 bpe. Half the distortion at 0.63 fewer bits, on the honest instrument.
2. **eig_full waterfill @2.5b beats uniform@3b 1.9× at matched honest bits** once context
   ≥32k (C×C basis charge amortizes). Per-head KLT keeps only ~20% of the win — the
   structure is cross-head; full-C is required.
3. **Hadamard+uniform LOSES to no-rotation on pre-RoPE keys** (0.86–0.90×) — TurboQuant's
   core mechanism is actively harmful on the structured side.

k2t and eig-waterfill are the same idea at different crudeness: k2t is a two-level bit
allocation {fp16, 2b} over an eigenbasis; waterfilling is the provably optimal allocation.
K4 builds the principled family, measures its full frontier, and duels TurboQuant everywhere.

## 2. The claim (paper spine)

> KV caches are spiked, query-anisotropic sources. Distribution-oblivious quantization
> (random rotation + uniform coding) pays a measurable structure tax — on pre-RoPE keys,
> Hadamard rotation actively hurts (0.86–0.90× vs no rotation). We give the rate-optimal
> alternative: a query-weighted eigenbasis with waterfilling bit allocation across
> directions and layers, calibrated once per model at zero per-sequence cost. It dominates
> TurboQuant's quality-vs-bits frontier at every measured operating point — more quality at
> matched bits below ~3b (where oblivious coding collapses: b2 = 33.31 vs fp16 = 41.81),
> fewer bits at matched quality everywhere.

**Framing anchor (TurboQuant PDF p. 18):** TurboQuant's own 2.5-bit LongBench recipe is not
oblivious — 32 outlier channels @3b + 96 @2b, a data-dependent per-channel allocation. Their
headline numbers already concede that structure exploitation is necessary below 3 bits; K4
is the principled completion of that move. This defuses the "they're calibration-free"
objection up front.

## 3. Codec design

### 3.1 Basis stage — corpus-calibrated, query-weighted KLT

- Metric that scores the task: `E[(qᵀ(k − k̂))²]` with q from the real query distribution
  (GQA-aware, RoPE applied at read — the existing `logit_rope` discipline).
- Optimal basis under that metric (theory brief, Q2): eigenbasis of `W^½ Σ_k W^½` where
  `W = E[qqᵀ]` and `Σ_k = E[kkᵀ]` over the full C = h_kv·d pre-RoPE channels. RoPE sits
  between stored k and the dot product, so W averages the query outer-product over
  positions — computed empirically from collected (q, position) samples (RoPE is block
  rotations; the position average is just an average over the collected sample).
  Status: the W^½ reduction is a one-step corollary of grounded rate-distortion (PPDRL
  eq. 4.1.11) but not a vault-named theorem — flagged extrapolated; Stage 1 measures
  weighted-vs-unweighted basis as an explicit ablation so the claim carries empirical weight.
- **Calibration policy (user decision, this session): corpus-level primary.** Bases fit once
  per model on a calibration corpus (wikitext slices via the existing `collect_cache`
  machinery; held-out sequences for the transfer test), shipped with the model like weights
  (~67 MB for all-layer C×C fp16 bases on Llama-3.1-8B — vs a ~16 GB model). Per-sequence
  prefix calibration (today's k2t/frozen-KLT design) is retained as a measured ablation, not
  the primary. Rationale: the per-sequence design pays two measured taxes — +0.41–0.69
  effective bits of basis drift and +0.5 bpe basis charge at 32k — and the corpus-vs-sequence
  gap is itself a finding ("how much KV structure is model-intrinsic?"; k2c's
  positional-stability result predicts transfer holds).
- Streaming consequence: appends become a fixed matmul against a constant basis — simpler
  than the current frozen-prefix path (no prefix fit, no refresh). Write-once storage
  invariant unchanged.

### 3.2 Within-layer allocation — waterfilling

- Given the weighted spectrum λ̃₁ ≥ … ≥ λ̃_C and budget B bpe: reverse waterfilling
  (PPDRL eq. 4.1.11): direction i gets bᵢ = max(0, ½·log₂(λ̃ᵢ/κ)), water level κ set so
  Σbᵢ = B·C. Implementation quantizes bᵢ to integer levels over direction groups (group
  size a swept parameter; per-direction integer bits is the limit case).
- **Rank is implied, not tuned**: directions below the water level get zero bits — low-rank
  truncation falls out as the κ > λ̃ᵢ region. k2t is the special case {fp16, 2b}; plain
  low-rank is {fp16, 0}.
- **Subspace coefficients are quantized, never fp16** (theory brief, Q1): a single water
  level means spikes deserve only ½·log₂(spike/bulk) more bits than the residual — ~5–9
  bits for typical spiked spectra, never 16. Concrete Pareto move available immediately:
  rank-32 @ 6-bit coefficients costs 0.19 bpe < rank-16 @ fp16's 0.25 bpe while covering
  twice the spike subspace.
- Quantizer within a direction-group: per-channel uniform with **MSE-optimal step** (not
  max-based) — at 2 bits, Lloyd-Max over optimal-step uniform is worth ~1%, so no codebook
  work on the residual; rotation+uniform is near-optimal exactly there (the one place
  TurboQuant's machinery is the right tool). Free early task: audit existing RTN arms'
  step policy; if any use max-based steps, fixing the step is the whole Lloyd win at zero
  cost and improves *every* arm including baselines (fairness requires applying it uniformly).
- Honest bits: ALL metadata counted (scales, allocation tables, basis under the applicable
  accounting mode — §5).

### 3.3 Across-layer allocation — milestone 2

- Measured per-layer sensitivity spread is ~3× (layer 0 pathological in every run; layers
  13–14 cheap) — a fixed per-layer budget is provably wasteful.
- Sensitivity sᵢ per (layer, side∈{K,V}): estimated by injecting quantization one site at a
  time and measuring end-to-end logit error (the K1 census pattern). Precedent: AutoQuantize
  mixed-precision PTQ (vault) — diagonal-Fisher sensitivity + knapsack holds 99.8% of BF16
  at a 4.75-bit budget with 512 calibration samples.
- Each site's distortion-vs-bits curve Dᵢ(Bᵢ) comes from its spectrum (closed form given
  the quantizer model; spot-verified on real caches). Global problem:
  min Σ sᵢ·Dᵢ(Bᵢ) s.t. Σ wᵢBᵢ = B_total (wᵢ = site's share of entries) — Lagrangian
  equalizes marginal distortion per bit; solved numerically offline; the resulting
  per-site bit table is codec metadata (counted, trivially small).
- This **subsumes asymmetric K/V**: "bits belong to K" becomes a solution property, not a
  design choice. V otherwise keeps its proven 2-bit treatment (rotate+Lloyd / TQ-mse
  per-head); whether V has exploitable spectrum is a cheap measured extension, not assumed.

## 4. Evaluation design — offline gauntlet, then one VM trip

No VM is currently rented. Every stage below is local-feasible (pure tensor math on already
collected real caches, gpt2 end-to-end as mechanism check). Every rung gates on the K1
logit-distortion instrument before any VM spend (standing discipline, `publishable-bar`).

### Stage 0 — spectra, sensitivity, transfer, drift (experiments/k4_spectra.py)

Deliverables: per-layer weighted + unweighted spectra; per-site sensitivities sᵢ; and two
diagnostics:

- **Transfer test**: basis fit on corpus split A, distortion on held-out sequences, vs each
  sequence's own oracle basis. **Gate G0: corpus basis retains ≥90% of the oracle-basis win
  over no-basis at matched bits.** Fail → fall back to per-sequence primary (win shrinks by
  the ~1-bit measured tax); the accounting advantage dies early and cheap.
- **Drift decomposition** (theory brief, Q4 — the "refresh recovers ≤23%" anomaly): frozen
  vs oracle-per-block vs oracle-full bases at matched honest bits, reusing blockklt
  machinery. Wainwright Cor. 8.7 puts prefix sampling noise near zero at our n, so the
  +0.41–0.69 bit penalty must be non-stationarity — this diagnostic separates trackable
  drift (block-switched causal bases cost only ~0.008 bpe signaling at 32k blocks and would
  erase the penalty) from misattribution. Informs the per-sequence *ablation* only; the
  corpus-primary path does not pay this tax.

### Stage 1 — within-layer frontier (experiments/k4_frontier.py)

On real Llama-3.1-8B caches (existing `results/cache/llama-3.1-8b_2048.safetensors` +
re-collect longer contexts as needed; 32k matters because basis/metadata charges amortize
with S): distortion-vs-bits curves at B ∈ {1.5, 2, 2.5, 3, 3.5, 4} bpe for:

- spectral (weighted basis, waterfilled, quantized coefficients) — the K4 arm,
- spectral-unweighted (ablation: plain Σ_k KLT),
- turboquant_mse at matched budgets (their exact coder — the incumbent),
- asymmetric turboquant K/V splits (the honest baseline family, arms exist since `5720588`),
- k2t points (rank × residual-bits × coefficient-bits — rung 1 falls out here),
- no-rotation RTN and Hadamard+RTN (the structure-tax exhibits).

Metric: `logit_rope` (query-weighted logit distortion, RoPE at read), per layer and
aggregate. **Gate G1: the spectral curve sits strictly below turboquant_mse's at every
matched budget, aggregate and in ≥90% of layers, under BOTH accounting modes (§5).**
Kill condition: if G1 fails at ≥3b but passes below, the program narrows honestly to the
sub-3b claim (still clears the user's bar); if it fails everywhere, the family is dead and
the negative is written up.

### Stage 2 — global allocation (experiments/k4_alloc.py)

Solve the across-layer/K-V allocation at the Stage-1 winning points. **Gate G2: beats
uniform-per-layer at matched average bits on the instrument**, with a gpt2 quantized-prefill
perplexity mechanism check (local-feasible; 8B ppl deferred to the VM batch). If the win is
real but small (~0.2 bits), it ships as a minor section — reported honestly either way.

### Stage 3 — the VM duel (one batched trip, when a VM is rented)

Final code, checkpointed per-cell (the §3b/no-checkpoint lesson), headroom guards on:

- LongBench full 6-category + NIAH, arms: fp16 anchor, K4 at ~2.2 / ~2.7 / ~3.2 measured
  bits (`kv_size_bits` convention — K+V blended, all metadata counted), turboquant_mse
  b2 / b3 / b4, asymmetric K3V2. Operating points spaced ≥0.5 bits so separation exceeds
  run noise.
- Delta-parity licensing per the anchor-forensics decision (`dd84143`) — deltas from own
  full-cache, not absolute TurboQuant Table-1 parity (43.7-pt chat-wrap sensitivity killed
  absolute parity).
- The NIAH b3 point rides along — settles whether k2b's surviving Synthetic/retrieval edge
  (+2.3) transfers to the structured-codec family.
- Success = the claim in §2 at the measured operating points. This one trip closes the
  paper either way.

## 5. Honest-bits accounting policy

Two reporting modes, both always reported:

- **Model-level mode (primary):** corpus-calibrated bases + allocation tables ship with the
  model (like weights; GPTQ/AWQ precedent) → zero per-sequence charge. Stated explicitly,
  with the artifact size (~67 MB).
- **Skeptic mode:** everything charged per-sequence (the current convention) — at 32k the
  full-C basis adds ~0.5 bpe. G1 must pass under both modes; the paper reports both columns.

Per-sequence-calibrated ablations are always skeptic-mode. All scales, indices, allocation
tables, and norms counted in every mode — no change to the standing rule.

## 6. Scope boundaries

**In scope:** the K4 codec arm (registered in `CACHE_ARMS` + a `k4` recipe), the three
experiments above, the RTN step-policy audit, figures (frontier curves, structure-tax
exhibit, allocation heatmap).

**Out of scope (explicitly):**
- Entropy coding of indices (~0.47 bits available but latency-flagged in the vault) — future
  lever, noted in the paper's discussion only.
- Lloyd/lattice codebooks on residuals (~1% over optimal-step uniform — theory says no).
- Eviction/importance-scoring composition (orthogonal per vault; cite, don't build).
- Kernel/packed-path work for the eigenbasis, and any deployment latency/memory claims —
  deferred until the quality verdict lands (quality is the paper; systems demos at the
  winning point come after, per `publishable-bar`).
- Multi-architecture extension (G9) — decision-deferred as before.
- transformers-side integration beyond what `StreamingQuantizedCache` already provides; the
  corpus-basis path must slot into the existing arm-set gate (`a21f167`) — if it cannot, the
  streaming integration becomes a plan-level task, not silent scope growth.

## 7. Risks and kill conditions

| risk | detection | consequence |
|---|---|---|
| Corpus basis doesn't transfer | G0 (Stage 0, cheap) | fall back to per-sequence primary; lose the accounting edge + drift tax returns; win shrinks ~1 bit |
| Waterfilling win concentrates sub-3b | G1 per-budget breakdown | claim narrows to sub-3b quality + fewer-bits-at-parity above — still clears the bar |
| Across-layer win marginal | G2 | ships as minor section; spine unaffected |
| Weighted basis ≈ unweighted | Stage-1 ablation | drop the W^½ machinery, keep plain KLT; simpler codec, weaker theory section |
| Task-level (LongBench) doesn't follow the instrument | Stage 3 vs Stage 1 comparison | the instrument's validity is itself a finding to report; program stops at the honest negative |
| LongBench Avg noise swamps operating-point gaps | ≥0.5-bit spacing + fp16 anchor per run | duel design, not post-hoc |

## 8. Theory grounding (from the 2026-07-12 vault dig; confidence flagged)

- Waterfilling rule + zero-bit truncation: PPDRL eq. 4.1.11, §4.1.3 — **grounded**.
- Subspace coefficients ~5–9 bits, never fp16; rank implied by water level — **grounded**
  (single-water-level arithmetic).
- AM/GM spectrum ratio as the matched-bits win of a spectrum-adapted coder; Hadamard as
  information-erasure (vault: Random Rotation Induces Beta-Distributed Coordinates) —
  **grounded**; explains the measured 0.86–0.90× and 1.9× results.
- Weighted-metric basis `eig(W^½ Σ_k W^½)` — **extrapolated** (one-step corollary; Stage-1
  ablation carries the empirical weight).
- Lloyd-vs-uniform ≈1% at 2b; ECSQ ~0.47-bit lever — **extrapolated** (standard constants;
  vault holds the √3π/2 side).
- Prefix sampling noise negligible (Wainwright Cor. 8.7) → drift penalty is
  non-stationarity — **grounded**; motivates the Stage-0 decomposition.
- TurboQuant 2.5b uses 32 outlier channels @3b (their PDF p. 18) — **grounded** (primary
  source).
- Sensitivity-knapsack precedent: AutoQuantize (vault) — **grounded**.

## 9. Prior art: the waterfill arc — three kills, one revival, and what K4 changes

Waterfilling has been gated three times (2026-06-21) and partially revived (2026-07-08).
K4 must not re-run any settled question; each verdict below is carried as a constraint.

| # | attempt (doc/run) | verdict | what it settled | how K4 uses it |
|---|---|---|---|---|
| 1 | per-channel waterfill, raw basis (`2026-06-21-k2-waterfill-results.md`) | KILLED 32/32 | high-variance raw channels are not query-read; allocation in the raw basis funds the wrong directions | K4 never allocates in the raw basis; eigenbasis only |
| 2 | eigenbasis (KLT) waterfill (`2026-06-21-k2-eigwaterfill-results.md`) | mechanism REAL (2.24×, all falsification controls passed, query_eigen_alignment 0.59), KILLED-honest at S=2048 (+8 bpe basis charge) | the win exists and is eigenstructure-specific (random-rotation control lost); the kill was the per-sequence C×C charge at short context | the charge is the target: corpus-level mode = 0, skeptic mode amortizes (16·C/S) |
| 3 | structured rotations: top-k / per-head / frozen-prefix (`2026-06-21-k2-structured-rotation-results.md`) | ALL KILLED at matched bpe @S=2048; frozen drifts 1.93× | cheap basis approximations lose ~2× to spending the same bits on uniform precision; frozen RESIDUAL bases drift (residual eigengap 1.13 ≈ none → Davis–Kahan) | K4 does not ship a per-sequence structured rotation; drift threat is scoped to the gapless bulk (see prediction P2) |
| 4 | full-C KLT re-quoted at deploy context (`k2_blockklt`, commit `e53cf00`) | REVIVAL: @32k charge = +0.5 bpe; eig_full@2.5 beats uniform@3b **1.9× at matched honest bits**; per-head keeps only ~20% of the win | kill #2 was regime-specific (short-context accounting); cross-head structure is real — full-C required | this IS K4's skeptic-mode operating point, already measured positive |
| 5 | frozen full-C KLT at 25% prefix (`k2_blockklt_frozen`, commit `3d89bcf`) | +0.41–0.69 eff bits drift; refresh recovers ≤23%; **fit@512 < C is rank-deficient**; longer-fit trend positive | prefix-frozen fits are sample-starved (512 rows for C=1024 dims — half the basis is null-space), so the measured "drift" conflates non-stationarity with estimation failure | corpus fit has n ≫ C — the confound vanishes; Stage-0's drift decomposition separates the two causes cleanly |

**Named predictions this history pre-registers for Stage 0/1** (write the outcomes either way):

- **P1 (transfer):** the corpus-fit full-C basis retains ≥90% of the oracle win — because
  the June drift lived in the *residual* (gapless), while the funded spike directions have
  an eigengap and k2c showed pre-RoPE subspaces are positionally stable, and the
  rank-deficiency confound (row 5) is gone at corpus n.
- **P2 (bulk insensitivity):** basis error in the gapless bulk costs ~nothing, because
  waterfilling assigns the bulk near-flat low bits — any orthogonal basis of the bulk
  subspace is equivalent to the coder. (This is why kill #3's drift number does not
  transfer to K4: drift concentrated exactly where allocation is flat.)
- **P3 (coefficient quantization):** quantizing spike coefficients to ~6b and letting rank
  float dominates every prior rank-16-fp16 arm (theory §8; no prior arm ever tested it —
  kills #1–#3 and the revival all hard-coded lowrank-fp16 + residual treatment).
- **P4 (weighted basis):** the W^½-weighted basis raises the 0.59 query-eigen alignment
  measured in kill #2 and widens the matched-bits win over unweighted KLT.

**Standing methodology carried forward from the arc** (non-negotiable in K4 experiments):
the uniform bit-SWEEP as the frontier baseline (never a single uniform point — the
bits-advantage artifact caught in kill #3); the random-rotation control arm (the
load-bearing eigenstructure proof); the oracle-refit control (drift ceiling); idealized
AND honest bpe columns with the deploy-S amortized quote; region-matched tail scoring for
any frozen/streamed claim; deterministic MSE-optimal rounding (the bias-cheap /
variance-expensive prior — no dithered/unbiased arms).

**Reusable machinery (do not rebuild):** `allocate_channel_bits` (the waterfill
allocator), `_klt_basis`, `_unrotate`, tier/scale/factor bit accounting in
`bmx.cache.codecs`; the `k2_blockklt.py` / `k2_waterfill.py` experiment scaffolds
(Stage 1 extends these); `k2_blockklt_frozen.py`'s region-matched drift methodology
(Stage 0 extends it). Genuinely new in K4: the query-weighted second moment (W^½ basis),
the corpus-level fit + transfer test, full-spectrum waterfill with quantized coefficients
(no fp16 lowrank special case), the sensitivity census + across-layer allocation.

## 10. Success criteria (program-level)

1. G0–G2 pass offline with the gates as stated (or fail with written honest negatives).
2. The Stage-3 duel shows, at ≤2.7 measured bits, K4 Avg ≥ +2 points over the **best**
   TurboQuant-family arm at equal-or-fewer bits (incl. the asymmetric K3V2 baseline — beating
   only b2 while K3V2 holds 40 would not count); AND K4 matches b3's Avg (within noise) at
   ≥0.5 fewer bits — i.e., strictly better quality where quality is winnable, strictly fewer
   bits at parity. NIAH: K4 recall ≥ turboquant_mse at matched bits and no regression vs the
   k2b family's retrieval edge.
3. Every number in the paper traces to a committed parquet under `results/k4_*/<run-id>/`
   with config + env + SHA (standing artifact convention).
