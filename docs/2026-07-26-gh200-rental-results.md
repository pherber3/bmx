# GH200 rental results — the full K4 evidence batch (2026-07-25/26)

One GH200 rental (~29 h traced) executed the entire pre-registered queue of
`docs/2026-07-25-vm-rental-queue.md` (core stages 1–6, both approved riders)
plus a user-approved filler/overnight extension (stages 7–14). **14/14 stages
green.** All artifacts committed per stage on the VM and merged back over
git-bundle transport; branch `feat/triton-decode-kernel` @ `4ac416d` carries
everything. GPU idle audit: compute-busy 88% / model-resident 91% of 60-s
samples, peak 90.3 GiB; ~25 min lost to the one orchestration incident (§9).
Every stage ran ~3–5× under its estimate; the 62–98 GPU-h plan fit in ~29 h.

Full verified tables (every number recomputed from parquets, run-ids cited):
`2026-07-26-gh200-rental-appendix-{niah,longbench,corpus,systems}.md`.

## 0. Stage ledger (wall clock, verdict)

| # | Stage | Start→End (UTC) | Verdict |
|---|---|---|---|
| 1 | Triton decode re-verify | 07-25 04:51→05:00 | GREEN — merge gate passed |
| 2 | Packed-spectral Tasks 7–11 | attempt 04:40 FAIL (§9); rerun 07-25 18:18→07-26 04:30 | GREEN under amended Gate C (§8) |
| 3 | Rotated-W refit + duel + calibration ladder | 07-25 06:50→07:40 | GREEN |
| 4 | LongBench full-suite final recipe | 07-25 07:40→15:08 | GREEN |
| 5 | Corpus-transfer A1/A2 + bigram riders | 07-25 15:08→15:25 | GREEN (checker false-negative overridden, §9c) |
| 6 | Qwen3-8B replication Tasks 7–13 | 07-25 15:25→18:15 | GREEN |
| 7 | NIAH seed replicates (Llama, seeds 1–2 @32k/64k) | 07-26 ~04:40→05:47 | GREEN |
| 8 | Measured dec-quant, rotated Llama pack | 07-26 04:31→~04:40 | GREEN |
| 9 | Same-day LongBench baseline refresh | 07-26 05:47→~07:15 | GREEN |
| 10 | Trigram climb (Llama) | 07-26 ~07:15→~08:05 | GREEN — recipe CONFIRMS |
| 11 | int8_tl replication (Qwen) | 07-26 ~08:05→~08:20 | GREEN |
| 12 | Qwen corpus-transfer matrix | 07-26 ~08:20→08:45 | GREEN — reversal REPLICATES |
| 13 | Qwen calibration ladder | 07-26 08:45→~08:52 | GREEN |
| 14 | NIAH seed-sweep completion | 07-26 ~08:52→09:42 | GREEN |

(Stage numbering is logical queue order; stages 8/7 executed in reverse
numeric order by design — value-per-hour scheduling.)

Enabling code commits (each gauntlet-clean, pushed before use): `6244fbb`
(w_rope CLI pass-through — the locked rotated-W config was not expressible in
the shipped `k4_fit_packs`), `2bb0d6a`+`3bec31d` (Gate-C amendment, §8),
`0c8c9d8` (trigram synthesis arm — order-3 climb licensed by the stage-5 gate).

## 1. Triton merge gate (stage 1)

Full suite on the GH200: **671 passed / 2 skipped / 1 xfailed** — all 16
CUDA-gated fused-decode tests (k2b oracle, RTN + k2b generate-parity, the
no-silent-swallow dispatch gate) unskipped and passed, covering both 2026-07-01
cleanup waves. Real-model acceptance (`profile_decode_ab`, Llama-3.1-8B-
Instruct): `[parity] OK` over 32 greedy tokens; `[path probe]` 96 fused_k2b
calls (= 3·n_layers gate met — no silent chunked fallback); `[logit probe]`
0 argmax flips over 64 teacher-forced steps at 65536 ctx. Recorded (not
gated): max-abs envelope 12.40 at 64k — above the O(0.25–1.45) class the duel
doc cites; §8 re-scopes that class (measurement-basis mismatch, not a
regression). **The branch-merge gate for `feat/triton-decode-kernel` is green;
the merge decision is the user's.**

## 2. The locked refit at Llama scale (stage 3)

Config as locked (never re-litigated): `w_rope="rotated"`, `lam_alloc=None`,
`payload_quant="rtn"`, `dec_quant="int8_tl"`, ridge 1e-3, budgets {2.2, 2.5},
fit = 4×2048-token Instruct caches (offsets 2048–8192), scored = offsets
10240–16384. Pack: `k4_packs_llama31_instruct_rotw.safetensors` (VM-side,
gitignored, deterministic-regenerable; sidecar records w_rope provenance).

int8_tl certificate at scale: T_ℓ maps all-{5,6} across 32 layers, both
budgets. Binding nominal margins: **b2.2 layer 27 = 1.00×** (exactly at the 5%
bar), b2.5 layer 7 = 1.06×. §6 shows the measured-vs-nominal conservatism is
5.3–6.3× — the certificate's conservatism absorbed the nominal razor edge, as
the gpt2-scale conservatism analysis predicted.

Tier counts (the §3b band-collapse deliverable) and ridge flatness: the
funded mass sits at tiers 5+6 (~2.5k/~3.1k of ~24–30k used columns per
budget; only ~300–550 columns at t8); allocation is flat across ridge
{1e-2, 1e-3, 1e-4} (<1% drift between 1e-3 and 1e-4) — gpt2's ridge-flatness
replicates at Llama's n/C. Exact per-layer n_t0…n_t8 in the fit-pack
parquets (appendix-systems §2).

**Scope note:** the rotated-W license came from the *Llama* causal-instrument
A/B; that instrument never ran on Qwen, so the Qwen pack (stage 6) is
correctly frozen-W. Rotated-vs-frozen is a Llama-validated choice, model-
scoped until a Qwen A/B runs.

## 3. Task quality: LongBench at the shipped recipe (stages 4, 9)

Convention: the banked verdict numbers are **macro** (mean of the six
category means) — verified by exactly reproducing the duel doc's §2 values
from the in-tree July-15 runs. All comparisons below are same-convention,
same-tasks, full splits (3750 samples/arm, verified in samples.parquet).
The memory-cited 2026-07-08 verdict runs (b3 40.56, k2b 40.62) are NOT in
the tree; the in-tree July-15 runs are the authoritative comparators.

| comparison | quality (macro) | bits (mean kv) |
|---|---|---|
| **k4_b2.5_dec8tl (final recipe)** | **40.85** | **3.081** |
| vs banked k4_b2.5 fp32/frozen (40.72) | **+0.13** | **−0.72** |
| vs banked tq_b3 (40.37) | **+0.48** | −0.125 |

- **The deployment stack is a free lunch vs its own predecessor**: +0.13
  macro at −0.72 mean bits — every one of the 16 tasks drops bits (int8_tl
  decoder accounting + rotated-W pack).
- **vs TurboQuant's b3**: the +0.48 macro edge is entirely
  synthetic/retrieval (+3.36) and code (+1.40); the four language categories
  slightly favor b3. On the four synthetic+code cells: 4-cell mean quality
  +2.38 at −0.04 mean bits — but NOT cell-wise Pareto: repobench-p loses
  quality (−0.43) and lcc's quality win (+3.23) costs +0.91 bits at short
  sequences (spectral pack charge amortizes against S; the honest bpe
  convention charges it). State the category-mean claim, never "all four
  cells".
- Baseline refresh audit (stage 9): the July-15 fp16/tq_b3/tq_k3v2
  synthetic+code cells reproduce on today's SHA to ≤0.33 (fp16 count noise)
  / ≤0.13 (quantized), kv bits identical — the shared generation path is
  identity-stable across three weeks of branch evolution.
- Correction to the duel doc: it cites "n=2,930"; the parquets say 3750/arm
  (identical banked vs new, so comparability is unaffected).

## 4. NIAH: seeds, models, and the packed path (stages 3, 7, 14, 2, 11, 6)

**Llama (mean ± std over seeds; 5 seeds at 32k/64k, 3 at 4k–16k):**

| arm | 32768 | 65536 |
|---|---|---|
| fp16 | 7.52 (n=1 eff.) | 6.76 (n=1 eff.) |
| k4_b2.5_dec8tl | 7.75 ± 0.76 | 7.70 ± 0.93 |
| k4_b2.2_dec8tl | 7.75 ± 0.57 | 8.27 ± 0.71 |
| turboquant_mse_b3 | 7.60 ± 0.37 | 7.33 ± 0.61 |
| turboquant_mse_k3v2 | 7.47 ± 0.44 | 7.47 ± 0.85 |

**Verdict: an honest null.** Every arm sits within noise of every other at
32k/64k. Two claims die here, honestly: (a) single-cell "k4 beats fp16"
readings are not significant; (b) the mid-batch "tq_k3v2 64k edge is
seed-stable at 8.13 ± 0.70" claim — a 3-seed slice — dissolves at 5 seeds
(7.47 ± 0.85). The surviving claim is the one the duel needs: **quality
parity with fp16 and with TurboQuant's steelman at 0.7–1 fewer bits.**

**Harness finding (limitation to state):** the NIAH `seed` reseeds only the
codec RNG, NOT the needle/haystack draw — fp16 is bit-identical across seeds
(verified across 5 SHAs). Error bars are codec-RNG replication bars on one
fixed needle per length; task-level (needle-resampled) variance is not
measured. A needle-reseeding harness knob is future work.

**Qwen: the TQ-family collapse is real.** At 32768, `turboquant_mse_b3`
falls to **2.60 (3-seed mean; fp16 = 10.00 flat)** and `turboquant_mse_k3v2`
to 5.49, uniformly floored across ALL five depths, onset between 16k and
32k, present in all seeds (codec-RNG draws on the one fixed needle per
length — needle-level replication not yet measured, same limitation as the
Llama bars); `k4_b2.5` and `k4_b2.5_dec8tl` hold fp16 parity throughout. Llama at the same lengths/arms shows zero collapse — this is a
TurboQuant×Qwen interaction, not a length artifact. (LongBench corroboration:
Qwen n=100 probe has tq_b3 synthetic retrieval 19.83 vs k4_b2.5 84.50.)
This is a NEW differentiation result: k4 is cross-model robust where the
steelman baseline is model-fragile. Mechanism unknown — candidate paper
claim only with the caveat n=1 model-pair; a mechanism probe (which Qwen
property breaks turboquant — QK-norm? head_dim 128 scaling?) is future work.

**Packed path**: faithful to streaming at 64k (fp16 exact-equal; k4_b2.5
within one ROUGE bucket, −0.48 at the shared depth); at 131072 (packed-only,
the cells that OOM on dense) k4_b2.5 holds 6.67. Hygiene: the July-15
`f9eeafe` allocation-probe run re-measures duel cells with different values
— excluded from all tables (it is doc §6 material, not duel provenance).
Seed-0 July-25 reruns agree with banked July-15 except two single-bucket
cells and one >1-bucket cell (tq_k3v2@64k: banked 8.38 vs rerun 8.95 —
cross-SHA generation nondeterminism; the tables keep the banked value). The
mid-batch "8.13 ± 0.70" figure used the 8.95 rerun value, so its dissolution
to 7.47 ± 0.85 is partly this seed-0 provenance correction, partly n=3→5.

## 5. Corpus, calibration, and the order ladder (stages 5, 10, 12, 13, 3)

**The reversal (both models, b2.5 shown; b2.2 agrees):**

| D cell | gpt2 (banked) | Llama 8B | Qwen 8B |
|---|---|---|---|
| null→wiki (shuffled order) | 0.094 | **0.282** | **0.249** |
| code→wiki (cross-domain) | 0.45–0.58 band | 0.229 | 0.233 |
| null→code | — | **0.440** | **0.391** |
| wiki→code | — | 0.337 | 0.270 |
| model_intrinsic_flag | True | **False** | **False** |

At 8B the shuffled-order null is WORSE than cross-domain transfer on every
side of both models — the gpt2 token-marginal verdict ("corpus enters via
which-tokens-appear; order second-order") is scale-scoped: word order in
calibration text matters at deployment scale. Simultaneously the
cross-domain penalty roughly halves vs gpt2 — the general-corpus deployment
story strengthens while the token-histogram shortcut weakens.

**The synthesis ladder (recipe: calibrate from n-gram count tables only):**

| model / side | D_uni | D_bi | D_tri | recipe (<0.10) |
|---|---|---|---|---|
| Llama wiki | 0.295 | 0.124 | **0.064** | CONFIRMED |
| Llama code | 0.302 | 0.126 | **0.074** | CONFIRMED |
| Qwen wiki | 0.264 | 0.096 | **0.036** | CONFIRMED |
| Qwen code | 0.301 | 0.112 | **0.064** | CONFIRMED |

(b2.5 shown; b2.2 within 0.01 everywhere.) The ladder self-terminates at
order 3 under the pre-registered both-sides earns-keep rule (order-3 misses
the 50% relative-reduction bar on Llama both sides and Qwen code side) —
order 3 is both sufficient and the licensed stopping point. H3 (hybrid
basis-A+alloc-B) still fails the 0.9 bar on both models (recovery
0.63–0.75) — basis non-transfer replicates; whole-pack per-domain fitting
remains the exploitation lever.

**Calibration ladder (G1 heldout win vs number of fit caches):** G1 passes
at EVERY rung on both models with layer_win_fraction 1.0; Llama mean
win_deploy climbs 5.89 → 6.49 → 7.65 → 8.25 (nc = 1/2/4/8) — monotone, no
saturation at 8, per-cache returns flat through nc=4 and diminishing after. **A single
2048-token general-text cache clears the gate** — calibration cheapness is
a two-model claim. (Llama ladder ran the frozen-W instrument internally —
curve shape is the deliverable; Qwen nc=4 is a b2.5-only point due to a
budget-grid mismatch in the stage-6 main run.)

**Jensen Gate-A at 8B:** Llama debiased r_pred **0.684**, inside the gpt2
band [0.56–0.69], with the debiased identity MATCHING at both budgets
(abs gap 0.040/0.008 — stronger than gpt2, whose debiased match failed).
Layer 0 is the known near-singular pathology (1257 clamped directions) and
is excluded from summaries. The determinant-Jensen transfer-ceiling anchor
is now a two-scale result.

## 6. Deployment: measured int8_tl, census, packed 128k (stages 8, 11, 2)

**Measured decoder distortion (G1-style, scored caches; vs the 5% bar):**

| mode | Llama b2.2 / b2.5 | Qwen b2.2 / b2.5 |
|---|---|---|
| int8 blanket | 17.6% FAIL | 14–20% FAIL |
| int8_t5 (binding gate) | 0.37% / 0.41% | 0.86% / 1.01% |
| **int8_tl (shipped)** | **0.80% / 0.90%** | **0.55% / 0.57%** |

Gate passes both models both budgets; `int8_tl_win_at_own_bits` exceeds the
fp32 win at every cell (net bit win); fp16_shippability_flag False (the
skeptic-v1 fp16 charge was honest). Certificate conservatism quantified:
nominal binding margins 1.00–1.09× vs conservatism ratios (measured damage
vs nominal bound) 6.3×/5.3× (Llama) and 8.3×/8.6× (Qwen); raw clearance
under the 5% bar 5.6–9.1× — the certificate is a sound, ~5–9× conservative
screening instrument (its third and fourth validations).
Caveat: Qwen's b2.2 nominal margin (1.09×, layer 5) is log-cited only (the
committed parquet stores tiers {5,6,8}).

**Census at 128k (measured, GH200 480GB — ≈96 GiB HBM addressable per
nvidia-smi; the 90-GiB gate and June's 94.7-GiB precedent are HBM numbers):** deployment (chunked) path —
**k4_b2.5 = 50.48 GiB, 20% below fp16 dense (63.30)**; k2b chunked 61.89
also undercuts fp16. Projections from 96k matched measured to ~0.15 GiB.
June continuity exact (fp16 128k 63.302 to the GiB; k2b 32k/64k reproduce).
**Correction:** the "expected-OOM k2b sentinel" did NOT occur — k2b 128k
dense measures 83.31 GiB and FITS post-mask-fix; the only real OOM sentinel
is the June pre-fix run. (The stage-2 merge-commit message wording predates
this reading; the parquet is authoritative.)

## 7. Qwen3-8B replication (stages 6, 11, 12, 13)

Architecture gate: `[rope_validation] rel_fro = 0.0005` throughout — the
structural `resolve_qk_capture_modules` fix captures true pre-RoPE normed
keys on the real 8B. Non-thinking protocol held (think_tags=False on every
smoke arm; raw-decode shim); smoke continuations byte-identical across arms.
G1 gates pass with layer_win_fraction 1.0. Replicated on Qwen: the int8_tl
deployment lever (§6), the corpus reversal + trigram recipe (§5), the
calibration ladder (§5), and NIAH fp16-parity for k4 (§4). Qwen pack is
frozen-W by scope (§2 note). The LongBench probe (n=100, synthetic+code,
5 arms) is directional only — full-set Qwen LongBench was not run.

## 8. The Gate-C amendment (pre-registered gate, user-approved change)

Stage 2's first rerun failed its logit probe: 1 argmax flip over 32
teacher-forced steps at 65536 (k4_b2.5 packed vs streaming). Forensics
(deterministic across two full reruns, bit-identical logits): the flip is at
step 0, where the streaming top-2 gap is 0.211 against a step delta of 1.586
(envelope 7.45) on a RANDOM-TOKEN prompt whose near-degenerate argmax makes
a flip a guaranteed casualty of accumulation drift. The top-5 token SET is
identical between paths at every step; the k4 drift class (1.3–7.4) is
statistically identical to the accepted, merge-gate-passed k2b class
(1.2–8.2); torch/CUDA env identical to June. The duel doc had pre-registered
exactly this outcome (§packed parity: 64k greedy parity "diverges
probabilistically", 0 flips held only "on this seed", "merge gate wording
must change accordingly") — the assert had never been updated. Amendment
(`2bb0d6a`, user-approved): a flip FAILS only if drift-INEXPLICABLE
(streaming top-2 gap > that step's max-abs delta) or if flips > N//8;
near-tie flips WARN with gap/delta printed. The rerun reproduced the same
flip, classified it near-tie WARN, and passed with zero hard flips. This
also re-scopes the O(0.25–1.45) envelope class the duel doc cited: both
fused arms measure 7–12 envelopes at 64k with healthy gates on this
hardware; the old class was measured on a different basis.

## 9. Incident record (honest ops)

(a) **Driver `set -e` suppression** (attempt 1): the stage runner used
`if ( set -euo pipefail; …stage… ); then` — bash disables `-e` inside any
compound command whose exit status is tested, so stage bodies ran with
toothless error handling; stage 1 "passed" on its last echo despite its
profile step crashing, and stage 2 plowed past a crashed gate battery into a
census-gate driver pointed at a stale June run dir (`ls -td` clone-mtime
trap). Fixed with a bare subshell + `rc=$?` (verified); a stamp-file
`find -newer` replaced newest-dir discovery. ~25 min GPU lost; contaminated
artifacts removed before relaunch (kept: the 4 corpus caches + frozen pack,
which had legitimately completed and verify at S=2048).
(b) **`sys.path[0]` trap**: `python scripts/x.py` puts `scripts/` first on
sys.path, so `import experiments` fails; fixed driver-wide with
`PYTHONPATH=$HOME/bmx`.
(c) **Stage-5 checker false-negative**: the post-hoc verifier asserted
`gpt2_yellow_flag ∈ {True, False}` but the field is a provenance STRING
(a vestigial gpt2-era schema literal; these 8B runs ARE the replication it
refers to). All pre-registered artifacts verified healthy (2816 matrix rows);
stage marked done with the override reason in the driver ledger.
(d) **Test-debris commits**: `pytest` drops real `results/k4_fit_packs/` run
dirs (tiny-model smoke tests without out_root); the driver's stage-1 commit
captured two before a sweep was added. Repo cleanup candidate: thread
out_root=tmp_path through the remaining smoke tests.

## 10. Supersessions and doc updates required

- `docs/2026-07-15-k4-duel-results.md`: §4 NIAH superseded by §4 here
  (rotated + int8_tl, 5-seed error bars, f9eeafe exclusion); §2 "n=2,930"
  corrected to 3750/arm; §packed-parity envelope class re-scoped (§8).
- `docs/2026-07-23-k4-corpus-transfer-results.md`: gpt2 token-marginal
  verdict is scale-scoped — the two-model 8B reversal (§5) supersedes
  "order/context second-order" for deployment-scale models.
- `docs/2026-07-25-k4-estimation-levers-results.md` §2: Llama/Qwen nominal
  margins + measured clearance (§§2, 6) complete the int8_tl story.
- `docs/2026-07-25-vm-rental-queue.md`: executed in full; both riders run.
- CLAUDE.md research-state: Triton re-verify DONE (merge gate green); K4 VM
  legs complete.

## 11. Caveats the paper must carry

- b2.2 int8_tl: certificate margin nominally 1.00×, measured 6× under bar —
  present both numbers, never the nominal alone.
- NIAH: ~21 samples/cell; seeds reseed codec RNG only (one fixed needle per
  length — state as limitation); no significant arm differences on Llama
  (parity IS the claim); k3v2 = OUR asymmetric steelman (`5720588`), labeled
  as such — TurboQuant published no such configuration.
- The Qwen TQ-collapse (§4): real and replicated within this model pair, but
  n=1 model-pair and mechanism unknown — frame as an observed robustness
  asymmetry, not a general law.
- LongBench: macro convention explicitly; the synthetic+code claim is
  category-mean, not cell-wise Pareto; language categories slightly favor
  b3 at +0.125 bits.
- A2 D cells are the offline logit instrument, not generation tasks — label
  the instrument; stage-5/12 matrices are minutes-scale by design.
- Qwen pack frozen-W (scoped, §2); Qwen LongBench is an n=100 probe.
- gpt2 remains the mechanism-scale model for instrument validations; the 8B
  results are the deployment-scale record.
