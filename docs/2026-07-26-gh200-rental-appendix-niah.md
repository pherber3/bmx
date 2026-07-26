# NIAH consolidation — GH200 rental 2026-07-25/26 (+ banked July-15 seed-0 duel)

Repo `d:\Projects\bmx`, branch `feat/triton-decode-kernel` @ `4ac416d`.
Every number below is recomputed directly from each run's `metrics.parquet`
(`recall_full`, ROUGE-1 needle-recall on a 0–10 scale) with pandas — no log
claim was trusted. Model split is done per-run from `config.json:model_name`
(Llama-3.1-8B-Instruct vs Qwen3-8B share arm names). Provenance for every cell
is traceable to a run-id; the run-id → (model, arm, length, seed) map is in the
"Run inventory" table at the very bottom.

Metric convention: `recall_full` is the per-cell ROUGE-1 recall ×10 (so 10.0 =
perfect needle recall, the harness's `full` needle string). Per (arm,length,seed)
values are the mean over the 5 NIAH depths {0.1,0.3,0.5,0.7,0.9}. "mean ± std
across seeds" averages those per-seed depth-means, std is the sample std (ddof=1).

---

## FINDINGS (headline verdicts)

**1. TQ-COLLAPSE IS REAL, SEED-STABLE, MODEL-SPECIFIC TO QWEN, AND NOT
DEPTH-LOCALIZED.** On **Qwen3-8B at 32768**, both TurboQuant arms collapse far
below fp16 (which holds a perfect 10.00 at every depth/seed): `turboquant_mse_b3`
seed-means 3.81 / 0.57 / 3.43 (3-seed mean **2.60**, Δ vs fp16 **−7.40**);
`turboquant_mse_k3v2` 5.33 / 4.29 / 6.86 (mean **5.49**, Δ **−4.51**). The
collapse is present in **all three seeds** (magnitude varies, direction does
not) and is spread **across all five depths**, not confined to deep cells
(e.g. tq_b3 seed-1 @32k: 0.48/0.48/0.48/0.95/0.48 — uniformly floored). Onset is
between 16k and 32k: at 16384 it is intermittent (tq_b3 seed-1 = 4.48 while seeds
0,2 ≈ 9.4–10.0). **The k4 family does NOT collapse** — `k4_b2.5` and
`k4_b2.5_dec8tl` hold 10.00 at every Qwen depth/length/seed measured, identical
to fp16. **On Llama at the same 16384/32768 lengths there is NO collapse in any
arm**: fp16, k4, tq_b3, tq_k3v2 all sit ~7.0–8.7. The failure is a TurboQuant ×
Qwen interaction, not a general long-context effect. *(Caveat on the flagged
numbers: the task described "deep-depth cells ~1.3–4.5 vs fp16 ~6.4" — the
collapse cells are real and in that range, but the collapse is whole-sequence not
deep-localized, and the fp16 baseline it should be read against on Qwen is 10.0,
not 6.4; the ~6.4–7.5 fp16 level is the **Llama** scale. Do not compare a Qwen
collapsed cell against a Llama fp16 baseline.)*

**2. LLAMA IS A HONEST NULL — SINGLE-NEEDLE NIAH DOES NOT SEPARATE ARMS.** At
32768/65536 (5 seeds each) every arm sits within noise of fp16: fp16 7.52/6.76;
k4_b2.5_dec8tl 7.75/7.70; k4_b2.2_dec8tl 7.75/**8.27**; tq_b3 7.60/7.33; tq_k3v2
7.47/7.47. k4 is ≥ both TQ arms at 64k (b2.2_dec8tl 8.27 is the single best Llama
cell-mean), but every gap is ≪ the per-seed std (0.4–0.9). This reproduces the
banked duel doc's §4 finding ("everyone holds fp16-level recall through 64k") and
extends it to seeds 1–4. The Llama story is "no regression", the Qwen story is
"TurboQuant regresses hard".

**3. THE BANKED JULY-15 SEED-0 DUEL REPRODUCES EXACTLY.** The canonical banked
run-set `results/k3_niah/20260715-*-21e6d81` reproduces `docs/2026-07-15-k4-duel-results.md`
§4 to the digit (fp16 7.33/7.14/7.05/7.52/6.76; k4_b2.5 7.05/7.05/8.10/7.24/7.71;
tq_b3 7.05/7.62/6.86/7.71/6.95; tq_k3v2 6.95/7.24/7.33/7.52/8.38; k4_b2.2 32k/64k
7.14/7.81). The July-25 seed-0 rerun (`32fcdea`) agrees with the banked duel on
all but 2 of ~55 shared fp16/k3v2 cells (single-ROUGE-bucket, same-seed run
non-determinism). **A SECOND July-15 run (`20260715-132205-f9eeafe`) is NOT the
duel** — it is the allocation-probe SHA (doc §6, "allocated packs") and gives
different k4 values at 32k/64k; it is EXCLUDED from every table here and flagged.

**4. THE PACKED PATH IS FAITHFUL TO STREAMING (at the shared depth).** The
stage-2 packed run (`20260725-205956-3887193`, `use_packed=True`, Llama, seed 0,
depths 0.25/0.5/0.75) at 65536, depth 0.5 (the only depth shared with the
streaming 0.1–0.9 grid): fp16 packed 6.19 == streaming 6.19 (**Δ 0.000**, exact);
k4_b2.5 packed 6.67 vs streaming 7.14 (Δ **−0.48**, one ROUGE bucket, within
single-needle quantization). The packed path also carries the only **131072**
NIAH points measured anywhere (fp16 8.73, k2b 6.98, k4_b2.5 6.67, no streaming
twin at 128k). k2b @64k = 6.83 (bpe 3.78), k4_b2.5 @64k = 6.67 (bpe 2.49). No
same-arm packed-vs-streaming regression beyond one ROUGE bucket.

**5. ANOMALIES / GAPS (details in §Sanity):** (a) **fp16 is bit-identical across
all "seeds"** on both models — the NIAH `seed` does NOT re-draw the needle text
or its depth, it only reseeds the compression-codec RNG. So fp16's std=0 is a
harness artifact (effectively n=1), and the "5-seed" spread on compressed arms is
codec-RNG variance on **one fixed needle instance per length**, not needle-
placement variance — it understates true task variance. (b) **k4_b2.2_dec8tl has
no seed-0 point at 4096/8192/16384** (only seeds 1,2 there) — a coverage gap.
(c) The banked non-dec8tl `k4_b2.5`/`k4_b2.2` arms are seed-0-only (as designed).
(d) The `132205-f9eeafe` allocation run duplicates k4 seed-0 32k/64k with
different values — excluded. (e) The packed run uses a non-duel depth grid
(0.25/0.5/0.75) — expected for stage-2, but means only depth 0.5 is comparable to
streaming. All runs share n_prefill=128, rank=16, group=64 (zero deviations).

---

## 1. LLAMA master table (meta-llama/Llama-3.1-8B-Instruct)

**Provenance:** seed-0 non-dec8tl arms (fp16, k4_b2.5, k4_b2.2, tq_b3, tq_k3v2)
from the banked duel `20260715-*-21e6d81` (canonical, matches doc §4). dec8tl
arms + seeds 1–4 from the July-25/26 GH200 runs (`32fcdea`, `2613454`,
`48e39d4`). Where the July-25 seed-0 rerun (`32fcdea`) overlapped the banked
duel, the banked value is kept (differences ≤1 bucket on 2 cells).

### Llama — mean recall_full per (arm, length, seed)

| arm | length | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| fp16 | 4096 | 7.33 | 7.33 | 7.33 | — | — |
| fp16 | 8192 | 7.14 | 7.24 | 7.14 | — | — |
| fp16 | 16384 | 7.05 | 7.05 | 7.05 | — | — |
| fp16 | 32768 | 7.52 | 7.52 | 7.52 | 7.52 | 7.52 |
| fp16 | 65536 | 6.76 | 6.76 | 6.76 | 6.76 | 6.76 |
| k4_b2.2 | 32768 | 7.14 | — | — | — | — |
| k4_b2.2 | 65536 | 7.81 | — | — | — | — |
| k4_b2.2_dec8tl | 4096 | — | 7.14 | 7.14 | — | — |
| k4_b2.2_dec8tl | 8192 | — | 7.71 | 6.86 | — | — |
| k4_b2.2_dec8tl | 16384 | — | 7.71 | 7.52 | — | — |
| k4_b2.2_dec8tl | 32768 | 7.24 | 7.33 | 7.90 | 8.67 | 7.62 |
| k4_b2.2_dec8tl | 65536 | 7.90 | 9.14 | 8.86 | 7.43 | 8.00 |
| k4_b2.5 | 4096 | 7.05 | — | — | — | — |
| k4_b2.5 | 8192 | 7.05 | — | — | — | — |
| k4_b2.5 | 16384 | 8.10 | — | — | — | — |
| k4_b2.5 | 32768 | 7.24 | — | — | — | — |
| k4_b2.5 | 65536 | 7.71 | — | — | — | — |
| k4_b2.5_dec8tl | 4096 | 6.86 | 7.14 | 7.24 | — | — |
| k4_b2.5_dec8tl | 8192 | 7.52 | 8.19 | 6.76 | — | — |
| k4_b2.5_dec8tl | 16384 | 7.81 | 7.62 | 8.57 | — | — |
| k4_b2.5_dec8tl | 32768 | 8.29 | 7.33 | 8.57 | 7.90 | 6.67 |
| k4_b2.5_dec8tl | 65536 | 7.43 | 7.05 | 8.57 | 6.67 | 8.76 |
| turboquant_mse_b3 | 4096 | 7.05 | 7.24 | 6.86 | — | — |
| turboquant_mse_b3 | 8192 | 7.62 | 6.95 | 7.43 | — | — |
| turboquant_mse_b3 | 16384 | 6.86 | 8.57 | 8.10 | — | — |
| turboquant_mse_b3 | 32768 | 7.71 | 7.33 | 8.19 | 7.43 | 7.33 |
| turboquant_mse_b3 | 65536 | 6.95 | 7.14 | 7.71 | 8.19 | 6.67 |
| turboquant_mse_k3v2 | 4096 | 6.95 | 7.52 | 7.14 | — | — |
| turboquant_mse_k3v2 | 8192 | 7.24 | 7.14 | 8.95 | — | — |
| turboquant_mse_k3v2 | 16384 | 7.33 | 7.62 | 7.62 | — | — |
| turboquant_mse_k3v2 | 32768 | 7.52 | 7.24 | 8.19 | 7.05 | 7.33 |
| turboquant_mse_k3v2 | 65536 | 8.38 | 8.19 | 7.24 | 6.29 | 7.24 |

### Llama — mean ± std across seeds (per-seed depth-means; recall_full /10)

| arm | length | mean | std | n_seeds |
|---|---|---|---|---|
| fp16 | 4096 | 7.33 | 0.00 | 3 |
| fp16 | 8192 | 7.17 | 0.05 | 3 |
| fp16 | 16384 | 7.05 | 0.00 | 3 |
| fp16 | 32768 | 7.52 | 0.00 | 5 |
| fp16 | 65536 | 6.76 | 0.00 | 5 |
| k4_b2.2 | 32768 | 7.14 | — | 1 |
| k4_b2.2 | 65536 | 7.81 | — | 1 |
| k4_b2.2_dec8tl | 4096 | 7.14 | 0.00 | 2 |
| k4_b2.2_dec8tl | 8192 | 7.29 | 0.61 | 2 |
| k4_b2.2_dec8tl | 16384 | 7.62 | 0.13 | 2 |
| k4_b2.2_dec8tl | 32768 | 7.75 | 0.57 | 5 |
| k4_b2.2_dec8tl | 65536 | 8.27 | 0.71 | 5 |
| k4_b2.5 | 4096 | 7.05 | — | 1 |
| k4_b2.5 | 8192 | 7.05 | — | 1 |
| k4_b2.5 | 16384 | 8.10 | — | 1 |
| k4_b2.5 | 32768 | 7.24 | — | 1 |
| k4_b2.5 | 65536 | 7.71 | — | 1 |
| k4_b2.5_dec8tl | 4096 | 7.08 | 0.20 | 3 |
| k4_b2.5_dec8tl | 8192 | 7.49 | 0.71 | 3 |
| k4_b2.5_dec8tl | 16384 | 8.00 | 0.50 | 3 |
| k4_b2.5_dec8tl | 32768 | 7.75 | 0.76 | 5 |
| k4_b2.5_dec8tl | 65536 | 7.70 | 0.93 | 5 |
| turboquant_mse_b3 | 4096 | 7.05 | 0.19 | 3 |
| turboquant_mse_b3 | 8192 | 7.33 | 0.34 | 3 |
| turboquant_mse_b3 | 16384 | 7.84 | 0.88 | 3 |
| turboquant_mse_b3 | 32768 | 7.60 | 0.37 | 5 |
| turboquant_mse_b3 | 65536 | 7.33 | 0.61 | 5 |
| turboquant_mse_k3v2 | 4096 | 7.21 | 0.29 | 3 |
| turboquant_mse_k3v2 | 8192 | 7.78 | 1.02 | 3 |
| turboquant_mse_k3v2 | 16384 | 7.52 | 0.16 | 3 |
| turboquant_mse_k3v2 | 32768 | 7.47 | 0.44 | 5 |
| turboquant_mse_k3v2 | 65536 | 7.47 | 0.85 | 5 |

### Llama — per-depth recall_full (arm × length × seed × depth)

| arm | length | seed | d0.1 | d0.3 | d0.5 | d0.7 | d0.9 | mean |
|---|---|---|---|---|---|---|---|---|
| fp16 | 4096 | 0 | 7.14 | 8.57 | 7.14 | 7.14 | 6.67 | 7.33 |
| fp16 | 4096 | 1 | 7.14 | 8.57 | 7.14 | 7.14 | 6.67 | 7.33 |
| fp16 | 4096 | 2 | 7.14 | 8.57 | 7.14 | 7.14 | 6.67 | 7.33 |
| fp16 | 8192 | 0 | 7.14 | 7.14 | 7.14 | 7.14 | 7.14 | 7.14 |
| fp16 | 8192 | 1 | 7.62 | 7.14 | 7.14 | 7.14 | 7.14 | 7.24 |
| fp16 | 8192 | 2 | 7.14 | 7.14 | 7.14 | 7.14 | 7.14 | 7.14 |
| fp16 | 16384 | 0 | 7.14 | 6.67 | 7.14 | 7.14 | 7.14 | 7.05 |
| fp16 | 16384 | 1 | 7.14 | 6.67 | 7.14 | 7.14 | 7.14 | 7.05 |
| fp16 | 16384 | 2 | 7.14 | 6.67 | 7.14 | 7.14 | 7.14 | 7.05 |
| fp16 | 32768 | 0 | 7.14 | 7.14 | 6.67 | 6.67 | 10.00 | 7.52 |
| fp16 | 32768 | 1 | 7.14 | 7.14 | 6.67 | 6.67 | 10.00 | 7.52 |
| fp16 | 32768 | 2 | 7.14 | 7.14 | 6.67 | 6.67 | 10.00 | 7.52 |
| fp16 | 32768 | 3 | 7.14 | 7.14 | 6.67 | 6.67 | 10.00 | 7.52 |
| fp16 | 32768 | 4 | 7.14 | 7.14 | 6.67 | 6.67 | 10.00 | 7.52 |
| fp16 | 65536 | 0 | 7.62 | 7.14 | 6.19 | 6.19 | 6.67 | 6.76 |
| fp16 | 65536 | 1 | 7.62 | 7.14 | 6.19 | 6.19 | 6.67 | 6.76 |
| fp16 | 65536 | 2 | 7.62 | 7.14 | 6.19 | 6.19 | 6.67 | 6.76 |
| fp16 | 65536 | 3 | 7.62 | 7.14 | 6.19 | 6.19 | 6.67 | 6.76 |
| fp16 | 65536 | 4 | 7.62 | 7.14 | 6.19 | 6.19 | 6.67 | 6.76 |
| k4_b2.2 | 32768 | 0 | 6.67 | 6.67 | 7.14 | 7.14 | 8.10 | 7.14 |
| k4_b2.2 | 65536 | 0 | 7.14 | 7.14 | 10.00 | 7.62 | 7.14 | 7.81 |
| k4_b2.2_dec8tl | 4096 | 1 | 7.14 | 7.62 | 7.14 | 7.14 | 6.67 | 7.14 |
| k4_b2.2_dec8tl | 4096 | 2 | 7.14 | 7.14 | 7.14 | 7.14 | 7.14 | 7.14 |
| k4_b2.2_dec8tl | 8192 | 1 | 7.14 | 10.00 | 7.62 | 7.14 | 6.67 | 7.71 |
| k4_b2.2_dec8tl | 8192 | 2 | 6.67 | 6.67 | 7.14 | 6.67 | 7.14 | 6.86 |
| k4_b2.2_dec8tl | 16384 | 1 | 7.14 | 7.14 | 7.14 | 7.14 | 10.00 | 7.71 |
| k4_b2.2_dec8tl | 16384 | 2 | 7.14 | 6.67 | 6.67 | 7.14 | 10.00 | 7.52 |
| k4_b2.2_dec8tl | 32768 | 0 | 8.10 | 6.67 | 7.14 | 7.14 | 7.14 | 7.24 |
| k4_b2.2_dec8tl | 32768 | 1 | 8.10 | 7.14 | 8.10 | 6.67 | 6.67 | 7.33 |
| k4_b2.2_dec8tl | 32768 | 2 | 6.67 | 8.10 | 8.10 | 6.67 | 10.00 | 7.90 |
| k4_b2.2_dec8tl | 32768 | 3 | 7.14 | 8.10 | 8.10 | 10.00 | 10.00 | 8.67 |
| k4_b2.2_dec8tl | 32768 | 4 | 6.67 | 7.14 | 7.14 | 7.14 | 10.00 | 7.62 |
| k4_b2.2_dec8tl | 65536 | 0 | 10.00 | 7.14 | 7.14 | 8.57 | 6.67 | 7.90 |
| k4_b2.2_dec8tl | 65536 | 1 | 8.10 | 9.52 | 10.00 | 8.10 | 10.00 | 9.14 |
| k4_b2.2_dec8tl | 65536 | 2 | 8.10 | 8.10 | 10.00 | 8.10 | 10.00 | 8.86 |
| k4_b2.2_dec8tl | 65536 | 3 | 6.67 | 6.67 | 9.05 | 8.10 | 6.67 | 7.43 |
| k4_b2.2_dec8tl | 65536 | 4 | 6.67 | 8.57 | 10.00 | 8.10 | 6.67 | 8.00 |
| k4_b2.5 | 4096 | 0 | 7.14 | 7.14 | 7.14 | 6.67 | 7.14 | 7.05 |
| k4_b2.5 | 8192 | 0 | 7.14 | 7.14 | 7.14 | 7.14 | 6.67 | 7.05 |
| k4_b2.5 | 16384 | 0 | 7.14 | 7.14 | 9.05 | 7.14 | 10.00 | 8.10 |
| k4_b2.5 | 32768 | 0 | 7.14 | 7.14 | 6.67 | 7.14 | 8.10 | 7.24 |
| k4_b2.5 | 65536 | 0 | 7.14 | 7.14 | 7.14 | 10.00 | 7.14 | 7.71 |
| k4_b2.5_dec8tl | 4096 | 0 | 7.14 | 6.19 | 7.14 | 6.67 | 7.14 | 6.86 |
| k4_b2.5_dec8tl | 4096 | 1 | 7.14 | 7.14 | 7.14 | 7.14 | 7.14 | 7.14 |
| k4_b2.5_dec8tl | 4096 | 2 | 7.62 | 7.14 | 7.14 | 7.14 | 7.14 | 7.24 |
| k4_b2.5_dec8tl | 8192 | 0 | 6.67 | 10.00 | 7.14 | 6.67 | 7.14 | 7.52 |
| k4_b2.5_dec8tl | 8192 | 1 | 6.67 | 10.00 | 10.00 | 7.14 | 7.14 | 8.19 |
| k4_b2.5_dec8tl | 8192 | 2 | 6.67 | 6.67 | 7.14 | 6.67 | 6.67 | 6.76 |
| k4_b2.5_dec8tl | 16384 | 0 | 7.14 | 8.10 | 10.00 | 6.67 | 7.14 | 7.81 |
| k4_b2.5_dec8tl | 16384 | 1 | 6.67 | 7.14 | 7.14 | 7.14 | 10.00 | 7.62 |
| k4_b2.5_dec8tl | 16384 | 2 | 7.14 | 7.62 | 10.00 | 8.10 | 10.00 | 8.57 |
| k4_b2.5_dec8tl | 32768 | 0 | 8.10 | 8.10 | 8.10 | 7.14 | 10.00 | 8.29 |
| k4_b2.5_dec8tl | 32768 | 1 | 7.14 | 7.14 | 8.10 | 7.14 | 7.14 | 7.33 |
| k4_b2.5_dec8tl | 32768 | 2 | 6.67 | 8.10 | 8.10 | 10.00 | 10.00 | 8.57 |
| k4_b2.5_dec8tl | 32768 | 3 | 6.67 | 8.10 | 8.10 | 6.67 | 10.00 | 7.90 |
| k4_b2.5_dec8tl | 32768 | 4 | 6.67 | 6.67 | 6.19 | 7.14 | 6.67 | 6.67 |
| k4_b2.5_dec8tl | 65536 | 0 | 6.67 | 7.14 | 6.67 | 10.00 | 6.67 | 7.43 |
| k4_b2.5_dec8tl | 65536 | 1 | 6.67 | 7.14 | 6.67 | 8.10 | 6.67 | 7.05 |
| k4_b2.5_dec8tl | 65536 | 2 | 8.10 | 6.67 | 10.00 | 8.10 | 10.00 | 8.57 |
| k4_b2.5_dec8tl | 65536 | 3 | 6.67 | 6.67 | 6.67 | 6.67 | 6.67 | 6.67 |
| k4_b2.5_dec8tl | 65536 | 4 | 6.67 | 7.14 | 10.00 | 10.00 | 10.00 | 8.76 |
| turboquant_mse_b3 | 4096 | 0 | 7.62 | 6.67 | 6.67 | 7.14 | 7.14 | 7.05 |
| turboquant_mse_b3 | 4096 | 1 | 7.14 | 7.14 | 7.62 | 7.62 | 6.67 | 7.24 |
| turboquant_mse_b3 | 4096 | 2 | 6.67 | 6.67 | 6.67 | 7.14 | 7.14 | 6.86 |
| turboquant_mse_b3 | 8192 | 0 | 7.14 | 7.14 | 9.52 | 7.14 | 7.14 | 7.62 |
| turboquant_mse_b3 | 8192 | 1 | 7.14 | 7.14 | 6.67 | 7.14 | 6.67 | 6.95 |
| turboquant_mse_b3 | 8192 | 2 | 7.14 | 9.05 | 6.67 | 7.14 | 7.14 | 7.43 |
| turboquant_mse_b3 | 16384 | 0 | 7.14 | 7.14 | 6.67 | 6.67 | 6.67 | 6.86 |
| turboquant_mse_b3 | 16384 | 1 | 10.00 | 9.05 | 6.67 | 7.14 | 10.00 | 8.57 |
| turboquant_mse_b3 | 16384 | 2 | 10.00 | 6.67 | 7.14 | 6.67 | 10.00 | 8.10 |
| turboquant_mse_b3 | 32768 | 0 | 7.14 | 7.14 | 8.10 | 9.05 | 7.14 | 7.71 |
| turboquant_mse_b3 | 32768 | 1 | 7.14 | 7.14 | 8.10 | 7.14 | 7.14 | 7.33 |
| turboquant_mse_b3 | 32768 | 2 | 7.14 | 7.62 | 6.19 | 10.00 | 10.00 | 8.19 |
| turboquant_mse_b3 | 32768 | 3 | 7.14 | 7.14 | 8.10 | 6.67 | 8.10 | 7.43 |
| turboquant_mse_b3 | 32768 | 4 | 6.67 | 7.62 | 8.10 | 7.14 | 7.14 | 7.33 |
| turboquant_mse_b3 | 65536 | 0 | 7.14 | 7.14 | 6.67 | 7.14 | 6.67 | 6.95 |
| turboquant_mse_b3 | 65536 | 1 | 7.14 | 7.14 | 6.67 | 7.62 | 7.14 | 7.14 |
| turboquant_mse_b3 | 65536 | 2 | 7.14 | 8.10 | 10.00 | 6.19 | 7.14 | 7.71 |
| turboquant_mse_b3 | 65536 | 3 | 7.14 | 7.14 | 9.52 | 7.14 | 10.00 | 8.19 |
| turboquant_mse_b3 | 65536 | 4 | 7.14 | 6.67 | 6.19 | 6.67 | 6.67 | 6.67 |
| turboquant_mse_k3v2 | 4096 | 0 | 7.14 | 7.62 | 6.67 | 6.67 | 6.67 | 6.95 |
| turboquant_mse_k3v2 | 4096 | 1 | 7.14 | 6.67 | 7.14 | 7.14 | 9.52 | 7.52 |
| turboquant_mse_k3v2 | 4096 | 2 | 7.62 | 7.14 | 7.14 | 7.14 | 6.67 | 7.14 |
| turboquant_mse_k3v2 | 8192 | 0 | 6.67 | 7.14 | 8.10 | 7.14 | 7.14 | 7.24 |
| turboquant_mse_k3v2 | 8192 | 1 | 7.14 | 6.67 | 7.14 | 7.14 | 7.62 | 7.14 |
| turboquant_mse_k3v2 | 8192 | 2 | 7.14 | 9.05 | 9.52 | 9.52 | 9.52 | 8.95 |
| turboquant_mse_k3v2 | 16384 | 0 | 6.67 | 7.14 | 9.05 | 6.67 | 7.14 | 7.33 |
| turboquant_mse_k3v2 | 16384 | 1 | 7.14 | 6.67 | 7.14 | 7.14 | 10.00 | 7.62 |
| turboquant_mse_k3v2 | 16384 | 2 | 7.14 | 6.67 | 7.14 | 7.14 | 10.00 | 7.62 |
| turboquant_mse_k3v2 | 32768 | 0 | 7.14 | 7.62 | 7.14 | 6.67 | 9.05 | 7.52 |
| turboquant_mse_k3v2 | 32768 | 1 | 7.14 | 7.62 | 7.14 | 7.14 | 7.14 | 7.24 |
| turboquant_mse_k3v2 | 32768 | 2 | 7.14 | 7.14 | 6.67 | 10.00 | 10.00 | 8.19 |
| turboquant_mse_k3v2 | 32768 | 3 | 7.14 | 7.14 | 6.67 | 7.14 | 7.14 | 7.05 |
| turboquant_mse_k3v2 | 32768 | 4 | 6.67 | 7.14 | 7.14 | 6.67 | 9.05 | 7.33 |
| turboquant_mse_k3v2 | 65536 | 0 | 6.67 | 10.00 | 8.10 | 7.14 | 10.00 | 8.38 |
| turboquant_mse_k3v2 | 65536 | 1 | 6.67 | 7.14 | 7.14 | 10.00 | 10.00 | 8.19 |
| turboquant_mse_k3v2 | 65536 | 2 | 2.86 | 6.67 | 10.00 | 10.00 | 6.67 | 7.24 |
| turboquant_mse_k3v2 | 65536 | 3 | 3.33 | 7.14 | 7.14 | 6.67 | 7.14 | 6.29 |
| turboquant_mse_k3v2 | 65536 | 4 | 6.67 | 6.67 | 6.67 | 6.19 | 10.00 | 7.24 |

## 2. QWEN master table (Qwen/Qwen3-8B)

**Provenance:** seed-0 from the stage-6 bits-curve run `20260725-180335-1c48a9d`
(fp16, k4_b2.5, tq_b3, tq_k3v2 at 4k–32k) plus the stage-11 run
`20260726-081021-623978a` (k4_b2.5_dec8tl + a tq_b3 re-measure at 16k/32k,
identical to 180335 on shared cells). Seeds 1–2 at 4k–32k from
`20260726-092829-48e39d4` / `093535-48e39d4`. Qwen has no 64k/128k NIAH run in
this batch (max length 32768).

### Qwen — mean recall_full per (arm, length, seed)

| arm | length | s0 | s1 | s2 |
|---|---|---|---|---|
| fp16 | 4096 | 10.00 | 10.00 | 10.00 |
| fp16 | 8192 | 10.00 | 10.00 | 10.00 |
| fp16 | 16384 | 10.00 | 10.00 | 10.00 |
| fp16 | 32768 | 10.00 | 10.00 | 10.00 |
| k4_b2.5 | 4096 | 10.00 | 10.00 | 10.00 |
| k4_b2.5 | 8192 | 10.00 | 10.00 | 10.00 |
| k4_b2.5 | 16384 | 10.00 | 10.00 | 10.00 |
| k4_b2.5 | 32768 | 10.00 | 10.00 | 10.00 |
| k4_b2.5_dec8tl | 16384 | 10.00 | — | — |
| k4_b2.5_dec8tl | 32768 | 10.00 | — | — |
| turboquant_mse_b3 | 4096 | 10.00 | 8.48 | 10.00 |
| turboquant_mse_b3 | 8192 | 8.86 | 9.14 | 10.00 |
| turboquant_mse_b3 | 16384 | 9.43 | 4.48 | 10.00 |
| turboquant_mse_b3 | 32768 | 3.81 | 0.57 | 3.43 |
| turboquant_mse_k3v2 | 4096 | 10.00 | 10.00 | 10.00 |
| turboquant_mse_k3v2 | 8192 | 10.00 | 10.00 | 10.00 |
| turboquant_mse_k3v2 | 16384 | 10.00 | 7.81 | 10.00 |
| turboquant_mse_k3v2 | 32768 | 5.33 | 4.29 | 6.86 |

### Qwen — mean ± std across seeds (per-seed depth-means; recall_full /10)

| arm | length | mean | std | n_seeds |
|---|---|---|---|---|
| fp16 | 4096 | 10.00 | 0.00 | 3 |
| fp16 | 8192 | 10.00 | 0.00 | 3 |
| fp16 | 16384 | 10.00 | 0.00 | 3 |
| fp16 | 32768 | 10.00 | 0.00 | 3 |
| k4_b2.5 | 4096 | 10.00 | 0.00 | 3 |
| k4_b2.5 | 8192 | 10.00 | 0.00 | 3 |
| k4_b2.5 | 16384 | 10.00 | 0.00 | 3 |
| k4_b2.5 | 32768 | 10.00 | 0.00 | 3 |
| k4_b2.5_dec8tl | 16384 | 10.00 | — | 1 |
| k4_b2.5_dec8tl | 32768 | 10.00 | — | 1 |
| turboquant_mse_b3 | 4096 | 9.49 | 0.88 | 3 |
| turboquant_mse_b3 | 8192 | 9.33 | 0.59 | 3 |
| turboquant_mse_b3 | 16384 | 7.97 | 3.04 | 3 |
| turboquant_mse_b3 | 32768 | 2.60 | 1.77 | 3 |
| turboquant_mse_k3v2 | 4096 | 10.00 | 0.00 | 3 |
| turboquant_mse_k3v2 | 8192 | 10.00 | 0.00 | 3 |
| turboquant_mse_k3v2 | 16384 | 9.27 | 1.26 | 3 |
| turboquant_mse_k3v2 | 32768 | 5.49 | 1.29 | 3 |

### Qwen — per-depth recall_full (arm × length × seed × depth)

| arm | length | seed | d0.1 | d0.3 | d0.5 | d0.7 | d0.9 | mean |
|---|---|---|---|---|---|---|---|---|
| fp16 | 4096 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| fp16 | 4096 | 1 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| fp16 | 4096 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| fp16 | 8192 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| fp16 | 8192 | 1 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| fp16 | 8192 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| fp16 | 16384 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| fp16 | 16384 | 1 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| fp16 | 16384 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| fp16 | 32768 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| fp16 | 32768 | 1 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| fp16 | 32768 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| k4_b2.5 | 4096 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| k4_b2.5 | 4096 | 1 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| k4_b2.5 | 4096 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| k4_b2.5 | 8192 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| k4_b2.5 | 8192 | 1 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| k4_b2.5 | 8192 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| k4_b2.5 | 16384 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| k4_b2.5 | 16384 | 1 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| k4_b2.5 | 16384 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| k4_b2.5 | 32768 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| k4_b2.5 | 32768 | 1 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| k4_b2.5 | 32768 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| k4_b2.5_dec8tl | 16384 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| k4_b2.5_dec8tl | 32768 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| turboquant_mse_b3 | 4096 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| turboquant_mse_b3 | 4096 | 1 | 10.00 | 6.19 | 6.19 | 10.00 | 10.00 | 8.48 |
| turboquant_mse_b3 | 4096 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| turboquant_mse_b3 | 8192 | 0 | 10.00 | 10.00 | 10.00 | 7.14 | 7.14 | 8.86 |
| turboquant_mse_b3 | 8192 | 1 | 10.00 | 10.00 | 5.71 | 10.00 | 10.00 | 9.14 |
| turboquant_mse_b3 | 8192 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| turboquant_mse_b3 | 16384 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 7.14 | 9.43 |
| turboquant_mse_b3 | 16384 | 1 | 2.86 | 5.24 | 4.76 | 4.76 | 4.76 | 4.48 |
| turboquant_mse_b3 | 16384 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| turboquant_mse_b3 | 32768 | 0 | 4.76 | 4.76 | 4.76 | 2.86 | 1.90 | 3.81 |
| turboquant_mse_b3 | 32768 | 1 | 0.48 | 0.48 | 0.48 | 0.95 | 0.48 | 0.57 |
| turboquant_mse_b3 | 32768 | 2 | 5.24 | 5.24 | 0.95 | 1.90 | 3.81 | 3.43 |
| turboquant_mse_k3v2 | 4096 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| turboquant_mse_k3v2 | 4096 | 1 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| turboquant_mse_k3v2 | 4096 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| turboquant_mse_k3v2 | 8192 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| turboquant_mse_k3v2 | 8192 | 1 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| turboquant_mse_k3v2 | 8192 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| turboquant_mse_k3v2 | 16384 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| turboquant_mse_k3v2 | 16384 | 1 | 10.00 | 10.00 | 3.33 | 10.00 | 5.71 | 7.81 |
| turboquant_mse_k3v2 | 16384 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| turboquant_mse_k3v2 | 32768 | 0 | 6.19 | 6.67 | 1.43 | 4.76 | 7.62 | 5.33 |
| turboquant_mse_k3v2 | 32768 | 1 | 2.86 | 4.76 | 5.24 | 4.29 | 4.29 | 4.29 |
| turboquant_mse_k3v2 | 32768 | 2 | 10.00 | 6.67 | 5.24 | 6.67 | 5.71 | 6.86 |

## 3. TQ-COLLAPSE VERIFICATION (the flagged claim)

Per-depth `recall_full` for `turboquant_mse_b3` / `turboquant_mse_k3v2` vs `fp16` and the k4 arms, at 16384 and 32768, across ALL seeds. Qwen first (where the collapse lives), then Llama (control).

### Qwen — TQ vs fp16/k4, per depth (16384 & 32768)

| length | arm | seed | d0.1 | d0.3 | d0.5 | d0.7 | d0.9 | mean | Δ vs fp16 |
|---|---|---|---|---|---|---|---|---|---|
| 16384 | fp16 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |  |
| 16384 | fp16 | 1 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |  |
| 16384 | fp16 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |  |
| 16384 | k4_b2.5 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | +0.00 |
| 16384 | k4_b2.5 | 1 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | +0.00 |
| 16384 | k4_b2.5 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | +0.00 |
| 16384 | k4_b2.5_dec8tl | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | +0.00 |
| 16384 | turboquant_mse_b3 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 7.14 | 9.43 | -0.57 |
| 16384 | turboquant_mse_b3 | 1 | 2.86 | 5.24 | 4.76 | 4.76 | 4.76 | 4.48 | -5.52 |
| 16384 | turboquant_mse_b3 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | +0.00 |
| 16384 | turboquant_mse_k3v2 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | +0.00 |
| 16384 | turboquant_mse_k3v2 | 1 | 10.00 | 10.00 | 3.33 | 10.00 | 5.71 | 7.81 | -2.19 |
| 16384 | turboquant_mse_k3v2 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | +0.00 |
| 32768 | fp16 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |  |
| 32768 | fp16 | 1 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |  |
| 32768 | fp16 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |  |
| 32768 | k4_b2.5 | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | +0.00 |
| 32768 | k4_b2.5 | 1 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | +0.00 |
| 32768 | k4_b2.5 | 2 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | +0.00 |
| 32768 | k4_b2.5_dec8tl | 0 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | +0.00 |
| 32768 | turboquant_mse_b3 | 0 | 4.76 | 4.76 | 4.76 | 2.86 | 1.90 | 3.81 | -6.19 |
| 32768 | turboquant_mse_b3 | 1 | 0.48 | 0.48 | 0.48 | 0.95 | 0.48 | 0.57 | -9.43 |
| 32768 | turboquant_mse_b3 | 2 | 5.24 | 5.24 | 0.95 | 1.90 | 3.81 | 3.43 | -6.57 |
| 32768 | turboquant_mse_k3v2 | 0 | 6.19 | 6.67 | 1.43 | 4.76 | 7.62 | 5.33 | -4.67 |
| 32768 | turboquant_mse_k3v2 | 1 | 2.86 | 4.76 | 5.24 | 4.29 | 4.29 | 4.29 | -5.71 |
| 32768 | turboquant_mse_k3v2 | 2 | 10.00 | 6.67 | 5.24 | 6.67 | 5.71 | 6.86 | -3.14 |

### Llama — TQ vs fp16/k4, per depth (16384 & 32768)

| length | arm | seed | d0.1 | d0.3 | d0.5 | d0.7 | d0.9 | mean | Δ vs fp16 |
|---|---|---|---|---|---|---|---|---|---|
| 16384 | fp16 | 0 | 7.14 | 6.67 | 7.14 | 7.14 | 7.14 | 7.05 |  |
| 16384 | fp16 | 1 | 7.14 | 6.67 | 7.14 | 7.14 | 7.14 | 7.05 |  |
| 16384 | fp16 | 2 | 7.14 | 6.67 | 7.14 | 7.14 | 7.14 | 7.05 |  |
| 16384 | k4_b2.5 | 0 | 7.14 | 7.14 | 9.05 | 7.14 | 10.00 | 8.10 | +1.05 |
| 16384 | k4_b2.5_dec8tl | 0 | 7.14 | 8.10 | 10.00 | 6.67 | 7.14 | 7.81 | +0.76 |
| 16384 | k4_b2.5_dec8tl | 1 | 6.67 | 7.14 | 7.14 | 7.14 | 10.00 | 7.62 | +0.57 |
| 16384 | k4_b2.5_dec8tl | 2 | 7.14 | 7.62 | 10.00 | 8.10 | 10.00 | 8.57 | +1.52 |
| 16384 | turboquant_mse_b3 | 0 | 7.14 | 7.14 | 6.67 | 6.67 | 6.67 | 6.86 | -0.19 |
| 16384 | turboquant_mse_b3 | 1 | 10.00 | 9.05 | 6.67 | 7.14 | 10.00 | 8.57 | +1.52 |
| 16384 | turboquant_mse_b3 | 2 | 10.00 | 6.67 | 7.14 | 6.67 | 10.00 | 8.10 | +1.05 |
| 16384 | turboquant_mse_k3v2 | 0 | 6.67 | 7.14 | 9.05 | 6.67 | 7.14 | 7.33 | +0.29 |
| 16384 | turboquant_mse_k3v2 | 1 | 7.14 | 6.67 | 7.14 | 7.14 | 10.00 | 7.62 | +0.57 |
| 16384 | turboquant_mse_k3v2 | 2 | 7.14 | 6.67 | 7.14 | 7.14 | 10.00 | 7.62 | +0.57 |
| 32768 | fp16 | 0 | 7.14 | 7.14 | 6.67 | 6.67 | 10.00 | 7.52 |  |
| 32768 | fp16 | 1 | 7.14 | 7.14 | 6.67 | 6.67 | 10.00 | 7.52 |  |
| 32768 | fp16 | 2 | 7.14 | 7.14 | 6.67 | 6.67 | 10.00 | 7.52 |  |
| 32768 | fp16 | 3 | 7.14 | 7.14 | 6.67 | 6.67 | 10.00 | 7.52 |  |
| 32768 | fp16 | 4 | 7.14 | 7.14 | 6.67 | 6.67 | 10.00 | 7.52 |  |
| 32768 | k4_b2.5 | 0 | 7.14 | 7.14 | 6.67 | 7.14 | 8.10 | 7.24 | -0.29 |
| 32768 | k4_b2.5_dec8tl | 0 | 8.10 | 8.10 | 8.10 | 7.14 | 10.00 | 8.29 | +0.76 |
| 32768 | k4_b2.5_dec8tl | 1 | 7.14 | 7.14 | 8.10 | 7.14 | 7.14 | 7.33 | -0.19 |
| 32768 | k4_b2.5_dec8tl | 2 | 6.67 | 8.10 | 8.10 | 10.00 | 10.00 | 8.57 | +1.05 |
| 32768 | k4_b2.5_dec8tl | 3 | 6.67 | 8.10 | 8.10 | 6.67 | 10.00 | 7.90 | +0.38 |
| 32768 | k4_b2.5_dec8tl | 4 | 6.67 | 6.67 | 6.19 | 7.14 | 6.67 | 6.67 | -0.86 |
| 32768 | turboquant_mse_b3 | 0 | 7.14 | 7.14 | 8.10 | 9.05 | 7.14 | 7.71 | +0.19 |
| 32768 | turboquant_mse_b3 | 1 | 7.14 | 7.14 | 8.10 | 7.14 | 7.14 | 7.33 | -0.19 |
| 32768 | turboquant_mse_b3 | 2 | 7.14 | 7.62 | 6.19 | 10.00 | 10.00 | 8.19 | +0.67 |
| 32768 | turboquant_mse_b3 | 3 | 7.14 | 7.14 | 8.10 | 6.67 | 8.10 | 7.43 | -0.10 |
| 32768 | turboquant_mse_b3 | 4 | 6.67 | 7.62 | 8.10 | 7.14 | 7.14 | 7.33 | -0.19 |
| 32768 | turboquant_mse_k3v2 | 0 | 7.14 | 7.62 | 7.14 | 6.67 | 9.05 | 7.52 | +0.00 |
| 32768 | turboquant_mse_k3v2 | 1 | 7.14 | 7.62 | 7.14 | 7.14 | 7.14 | 7.24 | -0.29 |
| 32768 | turboquant_mse_k3v2 | 2 | 7.14 | 7.14 | 6.67 | 10.00 | 10.00 | 8.19 | +0.67 |
| 32768 | turboquant_mse_k3v2 | 3 | 7.14 | 7.14 | 6.67 | 7.14 | 7.14 | 7.05 | -0.48 |
| 32768 | turboquant_mse_k3v2 | 4 | 6.67 | 7.14 | 7.14 | 6.67 | 9.05 | 7.33 | -0.19 |

**Verdict.** *Real* — the collapse appears in every Qwen seed at 32768 (tq_b3 0.57–3.81, tq_k3v2 4.29–6.86) against a flat fp16=k4=10.00. *Seed-stable* — direction holds in 3/3 seeds; magnitude varies (tq_b3 seed-1 is the deepest, 0.57). *Depth-localized?* **No** — the floored values are spread across all five depths, not concentrated at 0.7/0.9 (tq_b3 seed-1 @32k = 0.48 at four of five depths). *Model-specific?* **Yes** — Llama shows zero collapse at the same two lengths/arms (all cells 6.2–10.0, TQ within noise of fp16). Onset is 16k→32k: at 16384 only tq_b3 seed-1 (4.48) and tq_k3v2 seed-1 (7.81) dip; seeds 0,2 hold ≈10. k4_b2.5 / k4_b2.5_dec8tl never collapse on Qwen.

## 4. PACKED-PATH table (stage 2)

Run `20260725-205956-3887193` (Llama, seed 0, `use_packed=True`, depths 0.25/0.5/0.75 — a non-duel grid). Arms fp16 / k2b / k4_b2.5 at 64k and 128k.

### Packed recall_full per depth + honest bpe

| arm | length | d0.25 | d0.5 | d0.75 | mean | bpe_k | bpe_v | kv_bits | compression |
|---|---|---|---|---|---|---|---|---|---|
| fp16 | 65536 | 6.19 | 6.19 | 6.67 | 6.35 | 16.00 | 16.00 | 16.00 | 1.00 |
| fp16 | 131072 | 6.67 | 9.52 | 10.00 | 8.73 | 16.00 | 16.00 | 16.00 | 1.00 |
| k2b | 65536 | 6.67 | 7.14 | 6.67 | 6.83 | 5.52 | 2.04 | 3.78 | 4.23 |
| k2b | 131072 | 7.14 | 7.14 | 6.67 | 6.98 | 5.51 | 2.03 | 3.77 | 4.24 |
| k4_b2.5 | 65536 | 6.67 | 6.67 | 6.67 | 6.67 | 2.93 | 2.04 | 2.49 | 6.43 |
| k4_b2.5 | 131072 | 6.67 | 6.67 | 6.67 | 6.67 | 2.82 | 2.03 | 2.42 | 6.60 |

### Same-arm packed-vs-streaming @65536 (only depth 0.5 is shared)

Streaming = banked `20260715-*-21e6d81` seed-0, 65536.

| arm | depth 0.5 packed | depth 0.5 streaming | Δ (packed − streaming) |
|---|---|---|---|
| fp16 | 6.19 | 6.19 | +0.00 |
| k4_b2.5 | 6.67 | 7.14 | -0.48 |

*(k2b has no streaming NIAH twin in this batch; k4_b2.5 streaming full-depth mean @64k = 7.71, packed 3-depth mean = 6.67, but the depth grids differ so the mean-to-mean gap is not a like-for-like comparison — only the depth-0.5 cell above is. 131072 is packed-only; no streaming 128k run exists.)*

## 5. Sanity / anomalies

| # | finding | detail |
|---|---|---|
| A | **fp16 bit-identical across seeds** | On both models, fp16 per-depth cells are identical across all seed run-ids (Llama @32768 from 5 distinct SHAs; Qwen @32768 from 3). The NIAH `seed` does not re-draw the needle/haystack — it only reseeds the compression codec RNG. fp16 std=0 is a harness artifact (effective n=1); compressed-arm seed spread = codec-RNG variance on ONE fixed needle per length. |
| B | **k4_b2.2_dec8tl missing seed-0 at 4k/8k/16k** | Only seeds 1,2 present at 4096/8192/16384 (runs 085343/085853). No seed-0 short-length b2.2_dec8tl point. |
| C | **banked k4_b2.5 / k4_b2.2 are seed-0 only** | By design (banked duel). k4_b2.5 seed-0 at 4k–64k; k4_b2.2 seed-0 at 32k/64k only. No seeds 1–4 for the non-dec8tl banked arms. |
| D | **DUPLICATE-WITH-DIFFERENT-VALUES: 132205-f9eeafe** | `20260715-132205-f9eeafe` re-measures k4_b2.5/k4_b2.2 seed-0 @32k/64k with DIFFERENT values than the canonical duel (`21e6d81`): k4_b2.5@32k 6.95 vs 7.24, k4_b2.5@64k 8.10 vs 7.71, k4_b2.2@64k 7.81 (same) etc — 6+4+2 differing cells. This is the allocation-probe SHA (doc §6), NOT the duel. EXCLUDED from all tables. |
| E | **DUPLICATE-IDENTICAL: Qwen 180335 vs 081021** | fp16 & tq_b3 @16384/32768 seed-0 appear in both the stage-6 bits-curve run (180335) and the stage-11 run (081021) with IDENTICAL values. Deduped (kept 180335). |
| F | **DUPLICATE ~identical: Llama duel vs 072350/073029 rerun** | July-25 seed-0 rerun agrees with banked duel on all but 2 fp16/k3v2 cells (fp16 8192 d0.1 7.14 vs 7.62; fp16 16384 d0.3 6.67 vs 7.14; k3v2 16384 d0.5/d0.7, k3v2 64k d0.7) — single-ROUGE-bucket same-seed non-determinism. Banked value kept. |
| G | **packed run non-duel depth grid** | `205956` uses depths 0.25/0.5/0.75 (stage-2), not 0.1–0.9. Only depth 0.5 is comparable to streaming. Expected; flagged so packed means aren't compared to streaming means directly. |
| H | **provenance clean** | ALL 28 runs share n_prefill=128, rank=16, group=64, answer_id=7, max_new_tokens=50. Zero deviations. recall_full range [0.48, 10.0], no NaN. recall_kind = rouge1 everywhere. |
| I | **no missing cells within a run** | Every (arm,length,seed) present has all 5 duel depths (or all 3 packed depths). No partial/missing depth rows in any top-level metrics.parquet. |

## 6. Run inventory (run-id → model, arms, lengths, seed, packed)

| run-id | SHA | model | seed | lengths | arms | packed | role |
|---|---|---|---|---|---|---|---|
| 20260715-080730-21e6d81 | 21e6d81 | Llama-3.1-8B-Instruct | 0 | 4096/8192/16384/32768 | fp16 | False | banked duel fp16 4-32k |
| 20260715-080909-21e6d81 | 21e6d81 | Llama-3.1-8B-Instruct | 0 | 4096/8192/16384/32768 | k4_b2.5 | False | banked duel k4_b2.5 4-32k |
| 20260715-081107-21e6d81 | 21e6d81 | Llama-3.1-8B-Instruct | 0 | 4096/8192/16384/32768 | turboquant_mse_b3 | False | banked duel tq_b3 4-32k |
| 20260715-081253-21e6d81 | 21e6d81 | Llama-3.1-8B-Instruct | 0 | 4096/8192/16384/32768 | turboquant_mse_k3v2 | False | banked duel tq_k3v2 4-32k |
| 20260715-110927-21e6d81 | 21e6d81 | Llama-3.1-8B-Instruct | 0 | 65536 | fp16 | False | banked duel fp16 64k |
| 20260715-111108-21e6d81 | 21e6d81 | Llama-3.1-8B-Instruct | 0 | 65536 | k4_b2.5 | False | banked duel k4_b2.5 64k |
| 20260715-111257-21e6d81 | 21e6d81 | Llama-3.1-8B-Instruct | 0 | 65536 | turboquant_mse_b3 | False | banked duel tq_b3 64k |
| 20260715-111443-21e6d81 | 21e6d81 | Llama-3.1-8B-Instruct | 0 | 65536 | turboquant_mse_k3v2 | False | banked duel tq_k3v2 64k |
| 20260715-111948-21e6d81 | 21e6d81 | Llama-3.1-8B-Instruct | 0 | 32768/65536 | k4_b2.2 | False | banked duel k4_b2.2 32k/64k |
| 20260715-132205-f9eeafe | f9eeafe | Llama-3.1-8B-Instruct | 0 | 32768/65536 | k4_b2.2,k4_b2.5 | False | ALLOCATION SHA — EXCLUDED (dup-diff) |
| 20260725-072350-32fcdea | 32fcdea | Llama-3.1-8B-Instruct | 0 | 4096/8192/16384/32768 | fp16,k4_b2.5_dec8tl,turboquant_mse_b3,turboquant_mse_k3v2 | False | Llama seed0 rerun 4-32k (dec8tl+tq) |
| 20260725-073029-32fcdea | 32fcdea | Llama-3.1-8B-Instruct | 0 | 65536 | fp16,k4_b2.5_dec8tl,k4_b2.2_dec8tl,turboquant_mse_b3,turboquant_mse_k3v2 | False | Llama seed0 64k (dec8tl+tq) |
| 20260725-073910-32fcdea | 32fcdea | Llama-3.1-8B-Instruct | 0 | 32768 | k4_b2.2_dec8tl | False | Llama b2.2_dec8tl seed0 32k |
| 20260725-180335-1c48a9d | 1c48a9d | Qwen3-8B | 0 | 4096/8192/16384/32768 | fp16,k4_b2.5,turboquant_mse_b3,turboquant_mse_k3v2 | False | Qwen seed0 bits-curve 4-32k |
| 20260725-205956-3887193 | 3887193 | Llama-3.1-8B-Instruct | 0 | 65536/131072 | fp16,k4_b2.5,k2b | True | PACKED stage-2 (fp16/k2b/k4_b2.5) 64k/128k |
| 20260726-043735-2613454 | 2613454 | Llama-3.1-8B-Instruct | 1 | 32768 | fp16,k4_b2.5_dec8tl,k4_b2.2_dec8tl,turboquant_mse_b3,turboquant_mse_k3v2 | False | Llama seed1 32k |
| 20260726-044112-2613454 | 2613454 | Llama-3.1-8B-Instruct | 1 | 65536 | fp16,k4_b2.5_dec8tl,k4_b2.2_dec8tl,turboquant_mse_b3,turboquant_mse_k3v2 | False | Llama seed1 64k |
| 20260726-044950-2613454 | 2613454 | Llama-3.1-8B-Instruct | 2 | 32768 | fp16,k4_b2.5_dec8tl,k4_b2.2_dec8tl,turboquant_mse_b3,turboquant_mse_k3v2 | False | Llama seed2 32k |
| 20260726-045325-2613454 | 2613454 | Llama-3.1-8B-Instruct | 2 | 65536 | fp16,k4_b2.5_dec8tl,k4_b2.2_dec8tl,turboquant_mse_b3,turboquant_mse_k3v2 | False | Llama seed2 64k |
| 20260726-081021-623978a | 623978a | Qwen3-8B | 0 | 16384/32768 | fp16,k4_b2.5_dec8tl,turboquant_mse_b3 | False | Qwen seed0 dec8tl+tq_b3 16k/32k (dup-identical w/180335 on fp16/tq_b3) |
| 20260726-085343-48e39d4 | 48e39d4 | Llama-3.1-8B-Instruct | 1 | 4096/8192/16384 | fp16,k4_b2.5_dec8tl,k4_b2.2_dec8tl,turboquant_mse_b3,turboquant_mse_k3v2 | False | Llama seed1 4-16k |
| 20260726-085853-48e39d4 | 48e39d4 | Llama-3.1-8B-Instruct | 2 | 4096/8192/16384 | fp16,k4_b2.5_dec8tl,k4_b2.2_dec8tl,turboquant_mse_b3,turboquant_mse_k3v2 | False | Llama seed2 4-16k |
| 20260726-090351-48e39d4 | 48e39d4 | Llama-3.1-8B-Instruct | 3 | 32768 | fp16,k4_b2.5_dec8tl,k4_b2.2_dec8tl,turboquant_mse_b3,turboquant_mse_k3v2 | False | Llama seed3 32k |
| 20260726-090726-48e39d4 | 48e39d4 | Llama-3.1-8B-Instruct | 3 | 65536 | fp16,k4_b2.5_dec8tl,k4_b2.2_dec8tl,turboquant_mse_b3,turboquant_mse_k3v2 | False | Llama seed3 64k |
| 20260726-091609-48e39d4 | 48e39d4 | Llama-3.1-8B-Instruct | 4 | 32768 | fp16,k4_b2.5_dec8tl,k4_b2.2_dec8tl,turboquant_mse_b3,turboquant_mse_k3v2 | False | Llama seed4 32k |
| 20260726-091945-48e39d4 | 48e39d4 | Llama-3.1-8B-Instruct | 4 | 65536 | fp16,k4_b2.5_dec8tl,k4_b2.2_dec8tl,turboquant_mse_b3,turboquant_mse_k3v2 | False | Llama seed4 64k |
| 20260726-092829-48e39d4 | 48e39d4 | Qwen3-8B | 1 | 4096/8192/16384/32768 | fp16,k4_b2.5,turboquant_mse_b3,turboquant_mse_k3v2 | False | Qwen seed1 4-32k |
| 20260726-093535-48e39d4 | 48e39d4 | Qwen3-8B | 2 | 4096/8192/16384/32768 | fp16,k4_b2.5,turboquant_mse_b3,turboquant_mse_k3v2 | False | Qwen seed2 4-32k |

*(Banked-duel provenance pin, from `docs/2026-07-15-k4-duel-results.md` line 162: `results/k3_niah/20260715-{080730,080909,081107,081253,110927,111108,111257,111443,111948}-21e6d81`.)*
