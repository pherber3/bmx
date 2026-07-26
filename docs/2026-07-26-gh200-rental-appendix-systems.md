# Consolidation — Deployment / Systems evidence, 2026-07-25/26 GH200 rental

All numbers recomputed from committed parquets/JSONs with pandas (`uv run python`).
Device for every run: **NVIDIA GH200 480 GB**, torch 2.12.0+cu130, transformers 5.11.0
(`env.json` in each run dir). No web access used.

Run-ids consolidated:
- `k4_dec_quant` Llama  = `results/k4_dec_quant/20260726-043633-321f265` (git 321f265)
- `k4_dec_quant` Qwen   = `results/k4_dec_quant/20260726-080912-623978a` (git 623978a)
- `k4_fit_packs` Llama main (rotw, ridge 1e-3) = `results/k4_fit_packs/20260725-065203-32fcdea`
- `k4_fit_packs` Llama ridge 1e-2 = `20260725-065252-32fcdea`; ridge 1e-4 = `20260725-065341-32fcdea`
- `k4_fit_packs` Qwen  main (ridge 1e-3) = `20260725-153022-1c48a9d`
- `k3_kernel_census` stage-1 sweep (32/64/96k) = `20260725-200112-3887193`
- `k3_kernel_census` 128k cells = `202448` (fp16), `202505`/`203008`/`205458` (k2b), `203505`/`204441` (k4_b2.5)

---

## 1. INT8_TL MEASURED — decoder-precision distortion gate (Lever 2)

Headline metric `logit_rope` (RoPE ready both models). Gate binds on
`rel_degradation_int8_t5 < 5%` at **every** budget (pre-registered promotion
criterion); `int8_tl` (per-layer certificate-derived thresholds) reported on the
same axis but never the binding gate mode. `rel_degradation = 1 − win_mode/win_fp16`,
all modes interpolated at the fp16 arm's `bpe_skeptic_deploy` (isolates pure
decoder-precision damage). `*_win_at_own_bits` = win at that mode's OWN
mixed_dec_charge-accounted deploy bits (deployment view, never gated).

### LLAMA-3.1-8B-Instruct — run 20260726-043633-321f265 (n_samples=128 = 4 caches × 32 layers)

| budget | win_fp32 | win_fp16 | rel_deg_int8 (blanket) | rel_deg_int8_t5 (GATE) | rel_deg_int8_t6 | rel_deg_int8_tl | win_int8_tl | int8_tl_win_at_own_bits | gate_pass | ordering_ok |
|--------|---------|---------|------------------------|------------------------|-----------------|-----------------|-------------|-------------------------|-----------|-------------|
| 2.2    | 6.9400  | 6.9393  | 14.459 %               | **0.367 %**            | 1.371 %         | 0.798 %         | 6.8839      | 7.7954                  | **True**  | True        |
| 2.5    | 6.3763  | 6.3752  | 17.596 %               | **0.409 %**            | 1.259 %         | 0.900 %         | 6.3178      | 7.1751                  | **True**  | True        |

`rel_degradation_fp16_vs_fp32` = 1.0e-4 (b2.2), 1.7e-4 (b2.5) — well under the
0.5 % rider ⇒ `fp16_shippability_flag = False` (skeptic-v1 fp16 charge was NOT optimistic).
`int8_win_at_own_bits` (blanket) = 6.785 (b2.2) / 6.035 (b2.5). gate_mode = int8_t5,
gate_pass (overall) = **True**, extrapolated = False.

### QWEN3-8B — run 20260726-080912-623978a (n_samples=144 = 4 caches × 36 layers)

| budget | win_fp32 | win_fp16 | rel_deg_int8 (blanket) | rel_deg_int8_t5 (GATE) | rel_deg_int8_t6 | rel_deg_int8_tl | win_int8_tl | int8_tl_win_at_own_bits | gate_pass | ordering_ok |
|--------|---------|---------|------------------------|------------------------|-----------------|-----------------|-------------|-------------------------|-----------|-------------|
| 2.2    | 5.2470  | 5.2470  | 16.175 %               | **0.861 %**            | 2.428 %         | 0.552 %         | 5.2180      | 5.8368                  | **True**  | True        |
| 2.5    | 5.0824  | 5.0823  | 19.619 %               | **1.006 %**            | 2.578 %         | 0.570 %         | 5.0534      | 5.6930                  | **True**  | True        |

`rel_degradation_fp16_vs_fp32` = 5.2e-6 (b2.2), 1.0e-5 (b2.5) ⇒ `fp16_shippability_flag = False`.
`int8_win_at_own_bits` (blanket) = 4.968 (b2.2) / 4.660 (b2.5). gate_mode = int8_t5,
gate_pass (overall) = **True**, extrapolated = False.

**Verdict:** Lever-2 decoder-int8 gate PASSES on BOTH models at BOTH budgets.
The blanket int8 (14–20 % degradation) is dead as expected; tier-gated int8 rescues it:
int8_t5 clears the 5 % bar with room, and `int8_tl` (the deployment arm — per-layer
thresholds) is even tighter than t5 on Qwen (0.55/0.57 %) and between t5 and t6 on
Llama (0.80/0.90 %). `int8_tl_win_at_own_bits` > win_fp16 at every cell (int8_tl BUYS
bits back — the own-bits deploy view is a net win, not just a wash).

### CERTIFICATE nominal margins vs MEASURED clearance (conservatism)

Certificate = `int8_decoder_certificate_tiered(pack, T)["implied_rel_degradation"]`.
`int8_tl` chooses, per layer, the largest T in grid (2,3,4,5,6,8) whose cert ≤ 0.05 bar;
the **binding layer** is the globally-worst layer that pins the map to the smallest T.
Nominal margin = 0.05 / cert(binding layer at its chosen T). Cert values are
**cache-independent within a budget** (verified: cross-cache spread = 0.0), so the
cert is a clean per-(layer,budget) instrument.

| model | budget | binding layer | chosen T | cert@T | nominal margin | measured rel_deg_int8_tl (mean) | measured clearance under bar | CONSERVATISM (measured/nominal) |
|-------|--------|---------------|----------|--------|----------------|----------------------------------|------------------------------|----------------------------------|
| Llama | 2.2    | **27**        | 6        | 0.04996| **1.001×**     | 0.798 %                          | 6.27×                        | **6.3×**                         |
| Llama | 2.5    | **7**         | 6        | 0.04730| **1.057×**     | 0.900 %                          | 5.56×                        | **5.3×**                         |
| Qwen  | 2.2    | **5**         | (< 5)*   | 0.0459 (log-cited)| **1.090×** | 0.552 %                    | 9.05×                        | **8.3×**                         |
| Qwen  | 2.5    | **19**        | 5        | 0.04903| **1.020×**     | 0.570 %                          | 8.77×                        | **8.6×**                         |

Verification status of the stage-log-cited binding numbers:
- **Llama b2.2 layer 27 = 1.001×** — VERIFIED by recompute from `cert_vs_measured.parquet` (task said 1.00×). ✓
- **Llama b2.5 layer 7 = 1.057×** — VERIFIED (task said 1.06×). ✓
- **Qwen b2.5 layer 19 = 1.020×** — VERIFIED (task said 1.02×). ✓
- **Qwen b2.2 layer 5 = 1.09× — LOG-CITED, could NOT re-derive from the parquet.**
  \* The committed `cert_vs_measured.parquet` only stored tiers {5,6,8}. Layer 5 is the
  globally-worst Qwen layer (cert@t5 = 0.0866, ABOVE the bar), so its int8_tl threshold
  falls to a tier < 5 (T∈{2,3,4}) that was never written to the parquet. The 1.09×
  margin implies cert ≈ 0.0459 at that unstored sub-5 tier. Reviewer correction
  (2026-07-26): among the STORED tiers, layer 0 — not layer 5 — sorts worst
  (t5: 0.1022 vs 0.0866; t6: 0.3828 vs 0.1488; t8: 0.9668 vs 0.8131); layer 0 is
  the known near-singular first-block pathology (excluded from the Jensen
  summaries for the same reason), so layer 5 being the binding layer is
  consistent only under that stated layer-0 exclusion — both the exclusion and
  the exact 1.09× margin are log-cited, not parquet-derivable. All three other
  binding numbers verified exactly from committed data.

**Conservatism ratio: the offline certificate binds tight (nominal 1.00–1.09×, i.e.
sitting right on the 5 % bar) but the actual measured decoder-precision damage is
5–9× smaller** (0.55–0.90 % mean vs the 5 % bar). At the aggregate level the same
pattern holds at every tier — mean IMPLIED (cert) vs mean MEASURED rel_deg:

| model | budget | t5 cert / meas | t6 cert / meas | t8 (blanket) cert / meas |
|-------|--------|----------------|----------------|--------------------------|
| Llama | 2.2    | 2.03 % / 0.35 %| 4.80 % / 0.91 %| 35.37 % / 8.82 %         |
| Llama | 2.5    | 2.31 % / 0.41 %| 5.48 % / 1.03 %| 42.34 % / 11.39 %        |
| Qwen  | 2.2    | 3.78 % / 0.69 %| 9.31 % / 1.76 %| 36.42 % / 9.43 %         |
| Qwen  | 2.5    | 4.58 % / 0.81 %| 11.44 % / 2.06 %| 45.27 % / 12.01 %        |

The certificate is a genuine **conservative upper bound** (~3–6× over-estimate at every
tier) — exactly the "cheap analytic instrument, safe to gate on" property the harness
was built to validate.

---

## 2. TIER MAPS (§3b tier-count table) + ridge stability

`n_tX` = number of decoder columns allocated to tier X (X bits), summed across layers,
from each fit run's `metrics.parquet`. c_used = Σ(n_t2..n_t8) = non-zero-tier columns.
Crossover band = n_t5 + n_t6 (the tiers int8_t5 / int8_t6 gate on).

### Llama-3.1-8B-Instruct main pack (rotated-W, ridge 1e-3, group 64, 32 layers)

| budget | n_t0  | n_t2  | n_t3 | n_t4 | n_t5 | n_t6 | n_t8 | c_used | crossover (t5+t6) |
|--------|-------|-------|------|------|------|------|------|--------|-------------------|
| 2.2    | 8405  | 12237 | 6029 | 3213 | 1460 | 1024 | 400  | 24363  | 2484              |
| 2.5    | 6201  | 11990 | 6903 | 4018 | 1880 | 1228 | 548  | 26567  | 3108              |

### Qwen3-8B main pack (ridge 1e-3, group 64, 36 layers)

| budget | n_t0  | n_t2  | n_t3 | n_t4 | n_t5 | n_t6 | n_t8 | c_used | crossover (t5+t6) |
|--------|-------|-------|------|------|------|------|------|--------|-------------------|
| 2.2    | 8844  | 14063 | 7431 | 3767 | 1549 | 926  | 284  | 28020  | 2475              |
| 2.5    | 6566  | 13287 | 8501 | 4860 | 2074 | 1171 | 405  | 30298  | 3245              |

(The crossover band collapses to exact numbers: only ~2.5k / ~3.1k of the ~24–30k used
columns live in the t5+t6 band, and only ~300–550 need full t8 — the vast majority sit
at t2/t3 where int8 storage is decisive. That is why blanket int8 is fatal — it forces
the tiny t8 minority through the same int8 damage as the t2 majority — while tier-gating
leaves the high-bit columns fp16.)

### Ridge sweep — allocation flatness (Llama, rotated-W, budgets 2.2/2.5)

| ridge | b   | n_t2  | n_t3 | n_t4 | n_t5 | n_t6 | n_t8 | crossover (t5+t6) | c_used |
|-------|-----|-------|------|------|------|------|------|-------------------|--------|
| 1e-2  | 2.2 | 12647 | 6743 | 3189 | 1378 | 832  | 237  | 2210              | 25026  |
| 1e-3  | 2.2 | 12237 | 6029 | 3213 | 1460 | 1024 | 400  | 2484              | 24363  |
| 1e-4  | 2.2 | 12132 | 6019 | 3228 | 1474 | 1028 | 411  | 2502              | 24292  |
| 1e-2  | 2.5 | 12070 | 7710 | 4196 | 1730 | 1077 | 344  | 2807              | 27127  |
| 1e-3  | 2.5 | 11990 | 6903 | 4018 | 1880 | 1228 | 548  | 3108              | 26567  |
| 1e-4  | 2.5 | 11854 | 6885 | 4038 | 1892 | 1239 | 563  | 3131              | 26471  |

**Flatness verdict:** allocation is stable across two ridge decades. 1e-3 and 1e-4 are
nearly identical (crossover 2484 vs 2502 @ b2.2; 3108 vs 3131 @ b2.5 — < 1 % drift).
1e-2 differs a bit more (fewer t8: 237 vs 400 @ b2.2), i.e. heavier ridge pulls a few
columns down out of the top tier, but the overall shape and c_used are unchanged
(±3 %). The int8_tl gate conclusion is not ridge-sensitive.

---

## 3. CENSUS (stage 2) — memory clearing, GH200 480 GB

`k3_kernel_census`, all July-25, git 3887193. Bytes → GiB (÷2^30). `dense_stream` =
resident 2-copy path; `chunked` = packed-codes chunked-dequant at decode (the
memory-critical DEPLOYMENT path). No `oom=True` in ANY July-25 census parquet (see §5).

### Stage-1 sweep 32k / 64k / 96k (run 20260725-200112-3887193)

resident_after_prefill (GiB) [peak_decode GiB]:

| seq  | fp16 dense | k2b dense | k2b chunked | k4_b2.5 dense | k4_b2.5 chunked |
|------|-----------|-----------|-------------|---------------|-----------------|
| 32k  | 27.07 [19.05] | 32.07 [24.05] | 26.72 [18.65] | 32.34 [24.32] | **24.08 [16.01]** |
| 64k  | 39.19 [23.16] | 49.20 [33.17] | 38.48 [22.32] | 49.46 [33.43] | **32.90 [16.75]** |
| 96k  | 51.32 [27.27] | 66.32 [42.28] | 50.23 [26.00] | 66.58 [42.54] | **41.72 [17.49]** |

bpe_k / bpe_v (32k): fp16 16/16; k2b 5.542/2.072; k4_b2.5 3.161/2.072. bpe_k falls with
context (k4_b2.5 96k = 2.855). All cells `oom=False`.

### 128k clearing cells (runs 202448 / 202505 / 203505)

| arm     | path        | resident GiB | peak GiB | oom   | bpe_k  | bpe_v  |
|---------|-------------|--------------|----------|-------|--------|--------|
| fp16    | dense_stream| **63.30**    | 31.24    | False | 16.0   | 16.0   |
| k2b     | dense_stream| **83.31**    | 51.25    | False | 5.5106 | 2.0297 |
| k2b     | chunked     | **61.89**    | 29.58    | False | 5.5106 | 2.0297 |
| k4_b2.5 | dense_stream| **83.56**    | 51.50    | False | 2.8169 | 2.0297 |
| k4_b2.5 | chunked     | **50.48**    | 18.17    | False | 2.8169 | 2.0297 |

vs the projections handed in the task:
- fp16 63.44 projected → **63.30 measured** ✓
- k2b 83.45 (dense resident) → **83.31** ✓; the "+61.99" (chunked) → **61.89** ✓
- k4_b2.5 83.71 (dense) → **83.56** ✓; the "+50.54" (chunked) → **50.48** ✓

**Deployment headline (chunked path):** at 128k, **k2b chunked (61.89 GiB) undercuts
fp16 dense (63.30 GiB)** — the packed cache is already cheaper than fp16 at the memory
frontier — and **k4_b2.5 chunked (50.48 GiB) is 20 % below fp16** with bpe_k 2.82. The
dense 2-copy path costs ~83 GiB for both compression arms (the 2nd dense KV copy the
chunked path elides); chunked is the only arm/path combo that actually saves memory,
and it clears 128k comfortably on 480 GB.

### June continuity anchors

June kernel census (post prefix-mask-fix run `20260623-223357-c1fc279`, the authoritative
one) k2b chunked: 32k = **27.27 GiB**, 64k = **39.54 GiB**. July `200112` k2b chunked:
32k = **26.72**, 64k = **38.48** — REPRODUCES the June anchors within ~0.5–1 GiB, in fact
slightly LOWER (the V-pack / RoPE-sharing optimizations landed since June). fp16 dense and
k2b dense 128k also match June (63.30 / 83.5) to the decimal. Continuity ✓.

---

## 4. STAGE-2 GATE EVIDENCE (amended-gate rerun)

`scripts/profile_decode_ab.py` prints its parity / path-probe / logit-probe verdict to
**stdout only** — it writes NO parquet, so the numeric lines live in the VM `~/queue_logs`
(VM-only, not committed). What DID land in the repo is the code + the merge/commit record:

- Merge `321f265`: "*stage-2 packed-spectral rerun — gates green under amended Gate C
  (near-tie WARN), 96k census + 128k clearing cells + k2b OOM sentinel + 64k/128k packed NIAH*"
- Fix `2bb0d6a` (Gate C amendment): "*near-tie flips (stream top-2 gap ≤ step delta)
  WARN with forensics, flips > N//8 still fail … GH200 forensics 2026-07-25: deterministic
  step-0 flip, gap 0.211 vs delta 1.59, top-5 set identical, k4 drift class == accepted
  k2b class*". Module docstring records drift classes: k4 (1.3–7.4) statistically
  identical to accepted fused-k2b (1.2–8.2).

**Gate verdict lines (LOG-CITED, as pre-registered in the task):**
- **[parity]** streaming == packed over 32 greedy tokens: **OK**
- **[path probe]** chunked path taken **by design, 128 calls** (no fused spectral kernel
  exists yet — Phase A; the inverted assertion confirms the chunked dequant-attention path
  is actually exercised)
- **[logit probe]** **flips = 1, near-tie WARN** (NOT a fail): the single flip is a
  deterministic step-0 argmax flip on the random-token prompt's near-degenerate argmax,
  streaming top-2 **gap = 0.211** which is < that step's max-abs **delta = 1.586** ⇒
  drift-EXPLICABLE (accumulation-order), so WARN not FAIL. `max_abs_envelope = 7.45`
  (recorded, never gated). flips (1) ≤ N//8, hard_flips (drift-inexplicable) = 0 ⇒ gate PASSES.

Gate C was amended precisely because the duel doc (2026-07-15) pre-registered that 64k
packed greedy parity "diverges probabilistically" and the merge-gate wording had to move
from bitwise-parity to drift-inexplicable-flips. The rerun clears the amended gate.

---

## 5. Sanity — missing cells & unexpected OOM

**Grid completeness:** every planned (seq × arm) cell exists. Full grid
{32k,64k,96k,128k} × {fp16,k2b,k4_b2.5} present on `dense_stream`; `chunked` present for
k2b and k4_b2.5 at all four seqs. The four "missing" chunked-fp16 cells are **by design**
— fp16 has no packed-codes chunked-dequant path, so a chunked fp16 row cannot exist.
No true gaps.

**OOM anomaly (flag this):** the merge commit `321f265` message claims a "**k2b OOM
sentinel**", but **NO July-25 census parquet contains any `oom=True` row** — scanned all
seven `20260725-*` census files, every row is `oom=False`. k2b 128k dense sits at 83.31 GiB
resident / 51.25 GiB peak, comfortably inside 480 GB, so it does not OOM on GH200. The only
real OOM sentinel in the record is the **June PRE-FIX** run `20260623-221909-8c16d79`, where
k2b **chunked** OOMed at 32k/64k/128k (the O(S²)-online-softmax-at-prefill bug, resident
recorded as the −9.3e-10 sentinel) — since fixed in `223357-c1fc279`. So the commit
message's "k2b OOM sentinel" is either aspirational or refers to that historical June
sentinel; it is NOT present in the July-25 data. No unexpected OOM occurred; the discrepancy
is purely in the commit-message wording vs the committed parquet.

**Fit-run notes:** the Qwen main pack (`153022-1c48a9d`) was fit with `w_rope=frozen`
and budgets {2.0,2.2,2.5,2.7,3.0,3.2}, whereas the Llama main pack used `w_rope=rotated`
and budgets {2.2,2.5}. Both are ridge 1e-3, group 64. The b2.2/b2.5 slices used above are
directly comparable; noting the w_rope asymmetry for completeness (Llama is the rotated-W
"main pack run" the task named; Qwen's is frozen-W).
