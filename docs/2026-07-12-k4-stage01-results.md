# K4 — Stage 0 / Stage 1 gauntlet: G0 transfer + G1 frontier (2026-07-12)

First real-cache measurement of the K4 spectral-allocation codec (spec
`docs/superpowers/specs/2026-07-12-k4-spectral-codec-design.md`). Stage 0
(`experiments/k4_spectra.py`, gate G0 basis-transfer) and Stage 1
(`experiments/k4_frontier.py`, gate G1 frontier duel vs `turboquant_mse`) on
gpt2 (no-RoPE mechanism check, headline `logit`) and Llama-3.1-8B (headline
`logit_rope`, RoPE-at-read). Every number below was re-derived independently
from the committed parquets (not the runs' own prints).

Runs:
- gpt2 spectra `results/k4_spectra/20260712-171520-478ed98` (corpus mode: 4 offset caches)
- gpt2 frontier `results/k4_frontier/20260712-171559-478ed98` (corpus mode)
- Llama spectra `results/k4_spectra/20260712-171843-478ed98` (oracle + heldout only — no corpus)
- Llama frontier `results/k4_frontier/20260712-172332-478ed98` (oracle + heldout only)

## Headline

**G1 PASSES on the authoritative Llama-3.1-8B post-RoPE instrument, both
accounting modes, all six budgets, 100% of layers — in both the oracle and the
heldout fit mode.** Conservative claim (heldout key basis — fit on rows the
scorer never sees): the weighted spectral codec beats `turboquant_mse` at
matched bits by **6.4× (model accounting) / 4.5× (deploy-skeptic, S=32k) at the
2.5-bpe budget**. The oracle-fit figure — **14.8×/10.5×** — is an upper bound
(basis fit on the scored rows themselves); the clean corpus-basis number, the
deployment-grade figure, is pending the VM (no Llama corpus caches exist
locally). Fully-honest corner (unweighted basis, heldout key fit — no query
info, no key-tail leak): **2.27× deploy / 3.22× model at 2.5 bpe — still >1**;
heldout is rank-deficient so this floor is itself pessimistic. gpt2 replicates
the pass in all three fit modes (oracle/heldout/corpus), 100% of layers.

**G0 FAILS on the only transfer test available at this stage — but the failure
is the pre-registered rank-deficiency confound, not a science kill.** Llama G0
heldout retention = **0.428** (needs ≥0.90); gpt2 heldout **0.312**, corpus
**0.515**. The heldout fit is exactly `C` rows (1024 for Llama, 512 for gpt2 —
half of `C`), i.e. sample-starved at or below the C×C basis dimension — the
identical confound spec §9 row 5 flagged (`k2_blockklt_frozen`, fit@512<C is
rank-deficient). **The clean transfer test (corpus fit, n ≫ C — the primary
calibration policy) could NOT be run on Llama: only one Llama cache exists
locally, so there is no corpus mode.** G0's verdict on the deployment claim is
therefore DEFERRED to the VM, where a proper corpus fit is collectable. The gpt2
corpus retention (0.515, n=4096 ≫ C=768) is the best available proxy and is
itself below gate — flagged honestly below. **This does not gate G1: G1 passes
under the heldout (rank-deficient, worst-case) basis anyway.**

Per spec §7, a G0 fail routes to the per-sequence-primary fallback (lose the
accounting edge, ~1-bit drift tax). But G0 here is not a true "corpus doesn't
transfer" fail — it is "the corpus transfer test wasn't run; the rank-deficient
proxy fails as predicted." The narrowing is: **the model-level accounting mode
is UNVALIDATED pending a VM corpus fit; the G1 pass predicate (win > 1) holds
even under the worst-case rank-deficient heldout basis, but the win MAGNITUDE is
basis-dependent (4.53× heldout → 10.50× oracle, deploy accounting at 2.5 bpe)
and the deployment-grade magnitude awaits the corpus fit.**

## The numbers

### G1 frontier — Llama-3.1-8B (logit_rope, layer-mean, 32 layers)

Weighted spectral codec (oracle basis) vs the incumbent `turboquant_mse` and the
uniform baseline, on pre-RoPE keys `k_pre`, tail-region-scored (rows [S/2:]):

| arm | budget/bits | logit_rope ↓ | bpe_model | bpe_deploy (32k) |
|---|---|---|---|---|
| **spectral (weighted, oracle)** | 1.5 | **0.0158** | 1.750 | 2.250 |
| **spectral (weighted, oracle)** | 2.0 | **0.0113** | 2.250 | 2.750 |
| **spectral (weighted, oracle)** | 2.5 | **0.0084** | 2.750 | 3.250 |
| **spectral (weighted, oracle)** | 3.0 | **0.0064** | 3.250 | 3.750 |
| **spectral (weighted, oracle)** | 3.5 | **0.0050** | 3.750 | 4.250 |
| **spectral (weighted, oracle)** | 4.0 | **0.0041** | 4.250 | 4.750 |
| turboquant_mse (k_pre) | b2 | 0.1946 | 2.016 | 2.016 |
| turboquant_mse (k_pre) | b3 | 0.0947 | 3.016 | 3.016 |
| turboquant_mse (k_pre) | b4 | 0.0493 | 4.016 | 4.016 |
| lowrank_rtn_channel (uniform, r16) | b2 | 0.1052 | 2.625 | 2.625 |
| lowrank_rtn_channel (uniform, r16) | b3 | 0.0383 | 3.625 | 3.625 |
| lowrank_rtn_channel (uniform, r16) | b4 | 0.0164 | 4.625 | 4.625 |

turboquant_mse on post-RoPE `k` scores logit 0.195 / 0.095 / 0.050 at b2/b3/b4 —
no better than pre-RoPE, confirming the pre-RoPE-quantize discipline loses
nothing here.

### G1 win ratios (spectral distortion vs per-layer log-interpolated turboquant_mse curve)

Both accounting modes; win = tq_interp / spectral on layer means; layer-win
fraction = share of 32 layers where the spectral point beats the per-layer
turboquant interpolation. **g1_pass requires win > 1 AND ≥90% layer fraction in
BOTH modes at every budget.**

Llama-3.1-8B, oracle fit:

| budget | win_model | win_deploy | layer-frac (both) | g1_pass |
|---|---|---|---|---|
| 1.5 | 16.59 | 11.55 | 1.00 | ✓ (extrapolated) |
| 2.0 | 16.08 | 11.21 | 1.00 | ✓ |
| 2.5 | 14.82 | 10.50 | 1.00 | ✓ |
| 3.0 | 13.42 | 9.69 | 1.00 | ✓ |
| 3.5 | 12.01 | 8.66 | 1.00 | ✓ (extrapolated) |
| 4.0 | 10.36 | 7.48 | 1.00 | ✓ (extrapolated) |

Llama-3.1-8B, heldout fit (rank-deficient 1024-row basis — worst case):

| budget | win_model | win_deploy | layer-frac (both) | g1_pass |
|---|---|---|---|---|
| 1.5 | 8.77 | 6.10 | 1.00 | ✓ |
| 2.0 | 7.55 | 5.26 | 1.00 | ✓ |
| 2.5 | 6.40 | 4.53 | 1.00 | ✓ |
| 3.0 | 5.48 | 3.95 | 1.00 | ✓ |
| 3.5 | 4.73 | 3.41 | 1.00 | ✓ |
| 4.0 | 4.04 | 2.91 | 1.00 | ✓ |

**Even the rank-deficient heldout basis beats turboquant_mse by ≥2.9× under the
strictest (deploy-skeptic) accounting at every budget, on every layer.** g1_pass
= True for both fit modes. Fully-honest corner (unweighted basis, heldout key
fit — no query info, no key-tail leak): **2.27× deploy / 3.22× model at 2.5 bpe
— still >1**; heldout is rank-deficient so this floor is itself pessimistic. The
`extrapolated=True` flag marks budgets where the spectral bpe falls outside the
turboquant 3-point curve (b∈{2,3,4}) so the comparison is a log-linear
extrapolation — flagged, but the interior budgets (2.0/2.5/3.0) are interpolated
and pass identically.

gpt2 (no-RoPE, `logit`) replicates: oracle b2.5 win_model 23.3 / win_deploy
18.0; corpus b2.5 12.4 / 9.6; heldout b2.5 7.2 / 5.6. g1_pass = True in all three
fit modes.

### G0 retention (Stage 0, budget 2.5)

retention_m = mean over layers of [(ref/spectral_m)/(ref/spectral_oracle)],
headline metric, weighted spectral, ref = lowrank_rtn_channel r16b3.

| model | fit mode | fit rows | retention | g0_pass (≥0.90) |
|---|---|---|---|---|
| Llama-3.1-8B | heldout | 1024 = C | 0.428 | ✗ |
| Llama-3.1-8B | corpus | — | (no cache) | — |
| gpt2 | heldout | 512 = C/2 | 0.312 | ✗ |
| gpt2 | corpus | 4096 = ~5.3·C | 0.515 | ✗ |

The oracle basis itself is strong (Llama mean logit_rope 0.0084 vs reference
0.0383 = 4.6× win); the transfer test measures how much of that survives a basis
fit on data other than the scored sequence. The rank-deficiency signature is
explicit in the spectra parquet: heldout fit yields **332 zero-bit directions**
(vs oracle's 249) and an AM/GM spectral ratio of **3.0e8** (vs oracle 4.1e5) —
the 1024-row fit cannot estimate the C=1024-dim second moment, so the tail
eigenvectors are noise. This is the spec §9 row-5 confound verbatim, and it is
why the primary (corpus, n ≫ C) mode is the one that must be tested — on the VM.

## P1–P4 outcomes (spec §9 pre-registered predictions)

- **P1 (transfer ≥90%): NOT VALIDATED — DEFERRED.** The only transfer test
  runnable locally is the rank-deficient heldout (Llama 0.428, gpt2 corpus 0.515)
  — both below 0.90. P1's mechanism (corpus n ≫ C removes the rank-deficiency
  confound of §9 row 5) is precisely what could not be exercised: no Llama corpus
  cache exists (Task-5 SIGSEGV, below), and gpt2's corpus fit (n=4096) still
  fails. **P1 is UNRESOLVED, not falsified** — the clean test needs a proper
  corpus fit on the VM. Honest read: the fail we have is the confound P1 predicts
  away, so P1 remains open; but gpt2's corpus fail at n=5.3·C is a yellow flag —
  transfer may be weaker than the k2c positional-stability argument predicts.

- **P2 (bulk insensitivity): PARTIALLY CONTRADICTED — the gap GROWS with
  budget.** P2 predicted basis error in the gapless bulk costs ~nothing because
  waterfilling funds the bulk near-flat. Measured heldout/oracle distortion ratio
  (weighted spectral, Llama): **1.91× at b1.5 → 2.36× at b2.5 → 2.71× at b4.0**.
  The gap is smallest at low budget (few directions funded) and widens as the
  budget funds more of the drift-prone bulk — the opposite of "bulk error is
  free." The mechanism direction is consistent with P2 (low budgets are more
  robust to a bad basis), but the claim that bulk error is negligible does NOT
  hold: once funded, a rank-deficient bulk basis costs 2–3×. Note this is measured
  against the rank-deficient heldout basis, so it conflates estimation failure
  with true drift — the corpus test (P1) would separate them.

- **P3 (coefficient-quantization dominates rank-16 fp16): FAILED.** k2t_coeffquant
  (rank-32, 6-bit coefficients) scores logit_rope **0.0478 @ 2.461 bpe** vs
  lowrank_turboquant r16 (fp16 coeffs) **0.0578 @ 2.391 bpe** on Llama.
  coeffquant has *lower distortion* (0.0478 < 0.0578) but *higher bpe* (2.461 >
  2.391), so it does not strictly dominate — `coeffquant_dominates = False`. It is
  a Pareto-comparable point, not a dominating one. Note r32 fp16 lands at exactly
  0.0478 @ 2.766 bpe — coeffquant matches r32's quality at 0.3 fewer bits, which
  is the intended "quantize coeffs, float the rank" win, just not a strict
  domination of the r16 point the predicate tests. gpt2 same verdict (dominates =
  False).

- **P4 (weighted basis > unweighted): PASSED on the measured instrument — but
  UPPER-BOUNDED by query circularity.** Mean unweighted/weighted distortion
  ratio (oracle) on Llama: **1.69–1.73× across budgets** (unweighted KLT is 1.7×
  worse than the W½-weighted basis; peak 1.726× at b2.5). gpt2: 1.26–1.48×.
  **Caveat (query circularity, same plainness as the rank-deficiency confound):
  W is fit from the same 256 probe queries the metric scores, in every fit mode**
  — so the 1.7× weighted increment is measured on the training queries and is an
  UPPER BOUND on the deployable weighting win. The query-clean win is the
  unweighted arm's: **9.04× model / 6.41× deploy at 2.5 bpe** (oracle key fit)
  vs turboquant_mse. A query-heldout arm (W fit on one query set, scored on
  another) is required to license the weighted increment — queued for the VM
  batch.

## mse_scale ablation (the "free win")

`rtn_channel` b3, MSE-optimal step vs max-based step, on real caches (§3.2 free
early task). The step policy is applied uniformly to all arms, so fixing it helps
baselines too — fairness requires it:

| model | mse_scale=False | mse_scale=True | free win |
|---|---|---|---|
| Llama-3.1-8B | 0.1265 | 0.1001 | **−21%** distortion at identical 3.25 bpe |
| gpt2 | 0.0983 | 0.0834 | **−15%** distortion at identical 3.25 bpe |

The MSE-optimal step is a real free win (~15–21% distortion reduction at zero bit
cost) — worth applying uniformly across the arm set, exactly as the spec argues.

## Falsification checks

1. **Independent re-derivation of every verdict number from raw parquets.** A
   standalone pandas script (no experiment-code imports beyond pandas/math)
   reproduced g0_verdict.json and g1_verdict.json to all printed digits: Llama G1
   oracle b2.5 win_model 14.82 / win_deploy 10.50, heldout 6.40 / 4.53; retention
   0.428; P3/P4 exact. gpt2 likewise (oracle b2.5 23.3/18.0). No discrepancy.

2. **Random-basis control LOSES to weighted spectral — the load-bearing
   eigenstructure proof.** The critical check (a tie would invalidate the whole
   result): `spectral_randbasis` / weighted-spectral distortion ratio on Llama =
   **8.6–13.0× across budgets** (12.3× at b2.5); gpt2 **11.0–21.3×**. The random
   orthogonal basis is an order of magnitude worse — the win is the data
   eigenstructure, NOT rotation per se. This reproduces the kill-#2 signature
   (random-rotation control loses) and is nowhere near a tie. **PASS, loudly.**

3. **bpe hand-check.** Spectral b2.5 oracle bpe_model = 2.7498; minus the group
   scale charge (16/64 = 0.25) = 2.4998 ≈ budget 2.5, within tier granularity.
   The payload realizes the budget; the +0.25 is the counted per-group scale. ✓

4. **Skeptic-charge amortization.** Spectral b2.5 bpe_skeptic at S=2048 = 10.75
   (+8 bpe — the historic kill-#2 charge, reproduced exactly), but deploy-skeptic
   at S=32768 = 3.25 (+0.5 bpe). G1 uses the deploy-amortized charge and passes
   10.5× — the C×C charge amortizes at long context exactly as spec §5 states. ✓

5. **fp16-floor / no-degenerate-rows sanity.** No arm has headline distortion
   == 0 (0 rows in either frontier parquet). Reference arm (lowrank_rtn_channel
   r16b3) = **0.0383 @ 3.625 bpe** on Llama vs the June record ~0.036 @ 3.6 bpe —
   essentially exact, well inside the 2× tolerance; the instrument is calibrated.

## Caveats (carried honestly)

- **Single Llama cache, S=2048, tail-scored on 1024 rows.** All Llama numbers are
  one 2048-token sequence, scored on its second half. No cross-sequence variance
  estimate; the VM run must use multiple sequences.
- **Heldout fit = 1024 rows = C (rank-deficiency boundary).** The heldout basis is
  fit at exactly the C×C dimension — sample-starved by construction (spec §9 row
  5). Its 332 zero-bit directions and 3e8 AM/GM ratio confirm the tail is noise.
  G1 passing *under* this basis is a strength (worst-case), but G0's heldout
  retention number is not a clean transfer measurement.
- **No Llama corpus mode locally.** The primary calibration policy (corpus fit,
  n ≫ C, §3.1/§5 model-level accounting) is untested on Llama — the transfer gate
  G0 and prediction P1 are DEFERRED to the VM (VM preamble: collect ≥4 offset
  Llama caches, re-run k4_spectra with `--corpus-cache-paths`).
- **QUERY CIRCULARITY.** The query second moment W is fit from the same 256
  probe queries the logit metric scores — in EVERY fit mode (oracle, heldout,
  corpus alike: queries are always taken from the scored cache). The weighted
  arm's increment over unweighted (P4's 1.7×) is therefore measured on its own
  training queries and is an upper bound. The query-clean numbers are the
  unweighted arm's (9.04× model / 6.41× deploy at 2.5 bpe, oracle key fit;
  3.22×/2.27× at the heldout floor). A query-heldout arm — W fit on one query
  set, metric scored on a disjoint one — is required before the weighted
  increment is paper-licensed; it is on the VM-batch list below.
- **gpt2 is a no-RoPE mechanism check only** (headline `logit`, RoPE identity).
  Its G1 pass and P4 win corroborate the mechanism; Llama post-RoPE `logit_rope`
  is authoritative for all deployment claims.
- **Extrapolated G1 comparisons flagged.** At b∈{1.5, 3.5, 4.0} the spectral bpe
  falls outside the turboquant 3-point curve, so the win is a log-linear
  extrapolation (`extrapolated=True` in the verdict). Interior budgets
  (2.0/2.5/3.0) are interpolated and pass identically — the extrapolation is not
  load-bearing.
- **P2 measured against a rank-deficient basis** conflates estimation failure with
  true non-stationary drift; the growing gap is real but its attribution needs the
  corpus test.

## Contingency record

Task-5 Step 6 (local Llama corpus-cache collection) **SIGSEGV'd** — local Llama
cache collection is RAM-blocked, so no offset Llama caches exist. Consequence:
Llama runs have oracle + heldout fit modes only (no corpus), G0/P1 deferred to the
VM. gpt2 corpus mode (4 offset caches) is intact and stands in as the
mechanism-level transfer proxy.

## Gate calls

- **G1: PASS** (Llama, both fit modes, both accounting modes, all budgets, 100%
  of layers; gpt2 replicates in all three fit modes). The spectral codec's
  frontier sits strictly below turboquant_mse's everywhere measured — the §2
  claim's within-layer half is confirmed on the honest instrument.
- **G0: FAIL on the available (rank-deficient) test; DEFERRED on the primary
  (corpus) test.** Not a science kill — the failing test is the pre-registered
  confound. Model-level accounting is UNVALIDATED pending a VM corpus fit; the
  G1 pass predicate (win > 1) holds even under the worst-case rank-deficient
  heldout basis, but the win MAGNITUDE is basis-dependent (4.53× heldout →
  10.50× oracle at 2.5 bpe, deploy accounting) and the deployment-grade
  magnitude awaits the corpus fit. The program proceeds to Stage 2 on the pass
  predicate while the corpus transfer test is queued for the VM.
- **Paper-grade MAGNITUDE is unlicensed pending two VM measurements:** (a) a
  corpus key basis on Llama (settles which of 4.53×–10.50× the deployable win
  is), and (b) a query-heldout arm validating the W½ weighting (the weighted
  increment is currently measured on its own training queries — the
  query-circularity caveat above). Until both land, the licensable claims are
  the pass predicate itself and the fully-honest floor (2.27× deploy at 2.5
  bpe).
- **Next:** (1) VM: collect ≥4 offset Llama caches, re-run k4_spectra corpus mode
  → settle G0/P1 cleanly. (2) VM: add a query-heldout arm (W fit on a disjoint
  query set) — the load-bearing new experiment to license the weighted (P4)
  increment; it was missing from the original plan and is added here per external
  review. (3) Stage 2 (across-layer allocation, G2) can proceed now on the pass
  predicate. (4) P3's predicate is too strict — coeffquant is Pareto-comparable,
  not dominating; the paper should report the r32-parity framing, not claim
  strict domination of r16.

## External review

An adversarial referee review ran against the raw parquets before this doc was
finalized (verdict: sound with required edits — all incorporated above; every
number reproduced independently). The review's honesty ladder at 2.5 bpe,
deploy-skeptic accounting, Llama, win vs the per-layer turboquant_mse curve:
**A) 10.50× oracle+weighted, B) 4.53× heldout+weighted, C) 6.41×
oracle+unweighted, D) 2.27× heldout+unweighted** — every rung >1, layer-win
fraction 1.00 at every rung (model-accounting rungs: 14.82 / 6.40 / 9.04 /
3.22). Interpolation fairness, tail-scoring symmetry, and bpe accounting were
verified clean; anchor-point comparisons (spectral b2.0 vs the raw turboquant
b2 anchor, no interpolation) give LARGER wins than the interpolated predicate —
**19.05× oracle / 8.96× heldout at b2** — so the log-linear interpolation is
conservative, not flattering.

## Stage 2 — across-layer allocation (G2)

`experiments/k4_alloc.py`, gate G2 (across-layer bit allocation). Part A is a
per-layer sensitivity census; Part B a greedy marginal-upgrade allocation over
the Stage-1 frontier's per-layer distortion-vs-bits curves; Part C the
end-to-end verdict — allocated per-layer `turboquant_mse` K-bits vs a
variance-blind uniform comparator at the SAME mean bits, ppl over a 256-token
continuation conditioned on a 768-token quantized-KV prefill (V held fp16 on
both sides, so the K allocation lever is isolated). **gpt2 only** — Llama
sensitivity is RAM-blocked on this box (Task-5 SIGSEGV, same block as Stage 0/1)
and is VM material. Every number below was re-derived independently from the
run's `metrics.parquet` + `allocation.json` + `g2_verdict.json` (pandas/math
only, no experiment imports) — the greedy allocation reproduced bit-for-bit and
the ppl rows to all printed digits.

Run: `results/k4_alloc/20260712-181302-478ed98` (gpt2, n_prefill=768,
n_cont=256, sens_bits=2, targets {2.5, 3.0}).

### G2 call: **PASS** (both targets)

Allocated ≤ uniform at every target, and the mean-bits match is exact:

| target mean | ppl allocated | ppl uniform | Δppl (unif−alloc) | bpe_k alloc | bpe_k unif | mean-bits check | pass |
|---|---|---|---|---|---|---|---|
| 2.5 | **11.6885** | 12.1542 | +0.4657 | 2.5208 | 2.5208 | ✓ (Δbpe 0.000) | ✓ |
| 3.0 | **11.1910** | 11.7381 | +0.5471 | 3.0208 | 3.0208 | ✓ (Δbpe 0.000) | ✓ |

`g2_pass = True`. The sensitivity-weighted allocation lowers continuation ppl by
**0.47 (b2.5) / 0.55 (b3.0)** over the variance-blind uniform layout at matched
mean bits — a real, non-marginal separation (fp16 floor is 11.09, so the b3.0
allocated ppl 11.19 sits ~0.1 above fp16 while uniform pays ~0.65). Both bpe_k
means land on the 2.5208 / 3.0208 grid identically for allocated and uniform
(the +0.0208 over the nominal target is the per-token turboquant scale charge,
counted on both sides), so the comparison is bit-for-bit fair.

### Sensitivity census (Part A) — the spread is ~35×, NOT ~3×

`s_i = log(ppl_i) − log(ppl_fp16)`: the marginal NLL cost of degrading ONLY
layer i's K to `turboquant_mse @ 2b` (post-RoPE — gpt2 is no-RoPE, so this is a
plain per-channel quant), everything else fp16. fp16 floor ppl = **11.0879**.

| layer | s_i | ppl (this layer @2b) | note |
|---|---|---|---|
| 0 | +0.0277 | 11.399 | expensive (standing "layer-0 pathological" prior — 2nd worst) |
| 1 | +0.0039 | 11.131 | cheap |
| **2** | **+0.1256** | **12.572** | **outlier — the single expensive layer** |
| 3 | +0.0108 | 11.209 | |
| 4 | −0.0015 | 11.071 | noise (quant helped ppl by a hair) |
| 5 | −0.0048 | 11.035 | noise (min) |
| 6 | −0.0011 | 11.076 | noise |
| 7 | +0.0275 | 11.397 | expensive (≈ layer 0) |
| 8 | +0.0105 | 11.205 | |
| 9 | +0.0045 | 11.138 | |
| 10 | +0.0062 | 11.157 | |
| 11 | +0.0036 | 11.128 | |

**Spread: min = −0.0048 (layer 5) → max = +0.1256 (layer 2); max / smallest
positive s_i = 34.8×.** The spec §3.3 prior ("~3× across layers") is a large
underestimate for gpt2 K-sensitivity: **layer 2 is the load-bearing layer** at
~4.5× the next-most-sensitive (layers 0 and 7 tie at ~0.028), and the top-3
{2, 0, 7} carry the entire signal while the bulk sits in ppl noise. The standing
"layer 0 pathological" prior is partially confirmed (layer 0 IS 2nd-worst), but
on gpt2 the true outlier is layer 2, not layer 0. Three layers (4, 5, 6) have
slightly-negative measured s_i — pure ppl noise (single-sequence), exactly the
flat-layer case the allocator's 1e-6 clamp handles so a noise-zero layer stays
eligible for upgrades without inverting its weight.

### Allocation (Part B) — greedy marginal-upgrade over the frontier curves

Per-layer `turboquant_mse` K-bits, greedily buying the cheapest
s_l·ΔD_l per bit until mean bits hits the target (provably optimal for convex
per-layer curves). The sensitivity ranking is visible in the allocation: the
expensive layers {2, 0, 7} get 4 bits first, the noise-floor layers stay at 2.

| target mean | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | realized mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.5 | 4 | 2 | 4 | 2 | 2 | 2 | 2 | 4 | 2 | 2 | 2 | 2 | 2.500 |
| 3.0 | 4 | 2 | 4 | 4 | 2 | 2 | 2 | 4 | 4 | 3 | 3 | 2 | 3.000 |

At b2.5 the three most-sensitive layers {0, 2, 7} take the only three 4-bit
slots (3 × 4 + 9 × 2 = 30 bits / 12 = 2.5); at b3.0 the budget grows to fund the
mid-sensitivity layers {3, 8} to 4 bits and {9, 10} to 3, still starving the
noise-floor layers {4, 5, 6, 11}. Both realized means hit the target exactly.

### Honest notes / caveats

- **This exercises the ALLOCATION lever with EXISTING arms** (`turboquant_mse`
  at the integer 2/3/4-bit grid), NOT the K4 spectral codec. The spectral
  codec's own across-layer allocation — which is where the Stage-0/1 within-layer
  win compounds with a layer budget — is **Stage-3 material** (the deferred
  integration plan). G2 here answers only "does sensitivity-weighted layer
  allocation beat uniform at all, on the honest end-to-end ppl instrument?" —
  and it does.
- **gpt2 only, single sequence.** Llama sensitivity is RAM-blocked locally
  (same SIGSEGV as Stage 0/1) and is VM material; the three negative s_i values
  and the exact layer-2-outlier structure are one 1024-token wikitext-2 sequence
  with no cross-sequence variance estimate. The ~35× spread and the top-3
  {2, 0, 7} ranking should be re-confirmed on Llama post-RoPE before any
  allocation policy is fixed for deployment.
- **The ppl instrument is the warm-prefill continuation ppl**, not the standard
  cold sliding-window wikitext ppl. Conditioning the 256-token continuation on a
  768-token in-distribution context pulls the fp16 floor to 11.09 (well below the
  ~25-37 cold-window gpt2 figure) — this is the same `quantized_prefill_ppl`
  instrument used throughout the K-program and is calibrated; the low absolute
  number is the long warm context, not a bug.
- **The ppl separations are NOT within noise here** — +0.47 / +0.55 over uniform
  are ~4–5× the fp16-to-allocated gap at b3.0, a clean signal on this single
  sequence. (Had they landed within noise, this section would say so plainly and
  the G2 call would read "within-noise / no separation" rather than PASS.)

### G2 re-derivation (falsification)

A standalone pandas/math script (no `k4_alloc` imports) reproduced every
verdict number from the raw artifacts: (1) sensitivity spread −0.0048 → +0.1256,
34.8× positive ratio, top-3 {2, 0, 7}; (2) the greedy allocation reproduced
**bit-for-bit** at both targets from the frontier curves + clamped s (realized
means 2.500 / 3.000); (3) the G2 ppl rows matched `g2_verdict.json` to <1e-3
(ppl_alloc 11.6885 / 11.1910, ppl_unif 12.1542 / 11.7381, Δbpe 0.000 at both);
(4) re-derived `g2_pass = True`. No discrepancy.
