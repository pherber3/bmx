# LongBench evidence consolidation — 2026-07-25/26 GH200 rental

Repo: `d:\Projects\bmx` @ branch `feat/triton-decode-kernel`, HEAD `4ac416d`.
All numbers **recomputed from `metrics.parquet` / `samples.parquet`** — no log claims trusted.
Scores throughout are ×100 of the LongBench fractional metric (`code_sim` column, which
generically holds `count_score` / `retrieval_score` / `qa_f1_score` / `rouge_score` /
`classification_score` per the `metric` column). `kv_size_bits` is the skeptic-at-task-S
blended K/V bits-per-entry; `compression = 16 / kv_size_bits` (verified).

**Macro convention.** The banked July-15 duel doc reports its "AVG" as the **mean of the six
category means** (macro), NOT the flat 16-task mean. Verified: the `21e6d81` k4_b2.5 set gives
category means 21.72/13.24/68.88/27.38/52.68/60.43 whose 6-way mean is **40.72** (= doc §2),
while the flat 16-task mean is 38.74. **All AVG/MACRO numbers below use the macro convention**
so they line up with the banked verdict; flat means are labeled separately where shown.

---

## The runs (July-25/26 batch)

| run-id | model | categories | arms | n_samples cfg |
|---|---|---|---|---|
| 20260725-074007-2a7786b | Llama-3.1-8B-Instruct | code | k4_b2.5_dec8tl | null (full) |
| 20260725-085126-2a7786b | Llama-3.1-8B-Instruct | synthetic | k4_b2.5_dec8tl | null (full) |
| 20260725-091628-2a7786b | Llama-3.1-8B-Instruct | single_qa | k4_b2.5_dec8tl | null (full) |
| 20260725-100400-2a7786b | Llama-3.1-8B-Instruct | multi_qa | k4_b2.5_dec8tl | null (full) |
| 20260725-103729-2a7786b | Llama-3.1-8B-Instruct | few_shot | k4_b2.5_dec8tl | null (full) |
| 20260725-112807-122045a | Llama-3.1-8B-Instruct | summarization | k4_b2.5_dec8tl | null (full) |
| 20260725-160357-1c48a9d | **Qwen3-8B** | synthetic+code | fp16, k4_b2.2, k4_b2.5, tq_b3, tq_k3v2 | **100 (probe)** |
| 20260726-050209-a50b500 | Llama-3.1-8B-Instruct | synthetic | fp16, tq_b3, tq_k3v2 | null (full) |
| 20260726-054753-a50b500 | Llama-3.1-8B-Instruct | code | fp16, tq_b3, tq_k3v2 | null (full) |

**Arm identity.** `k4_b2.5_dec8tl` = budget-2.5 spectral pack + `dec_quant="int8_tl"` (int8
tier-limited decoder, the per-layer certificate-derived decoder PROMOTED in the 2026-07-24
math-actions doc), pack = `results/cache/k4_packs_llama31_instruct_rotw.safetensors`
(**rotated-W**). Recipe parse confirmed in `src/bmx/cache/recipes.py:116-118`.

**Duplicate-config check (batch-internal): NONE.** Each of the 6 dec8tl category runs is
unique; one Qwen probe; two Llama refresh runs. No duplicate/rerun within the July-25/26 batch.

---

## 1. FINAL-RECIPE FULL TABLE — k4_b2.5_dec8tl, Llama-3.1-8B-Instruct, full 16-task English LongBench v1

**n_samples assertion: PASS.** Every metrics `n_samples` equals its `samples.parquet` row
count (per arm×task); all 16 tasks at canonical full-split sizes, total **3750 samples/arm**
(NOT subsampled — the k3_longbench subsampling sentinel would show n<full). Per-task n:
lcc/repobench-p=500, multifieldqa_en=150, all other 14 tasks=200.

> Note: the July-15 duel doc §37 text says "n=2,930". The **parquets are 3750** (both the
> banked July-15 set and this July-25 set — identical per-task counts). The 2930 is a doc-side
> figure that does not match either parquet; the banked-vs-final comparison is nonetheless
> clean because both sides are the same 3750-sample full splits.

| category | task | score | n | metric | kv_size_bits |
|---|---|---:|---:|---|---:|
| single_qa | multifieldqa_en | 25.24 | 150 | qa_f1 | 3.027 |
| single_qa | narrativeqa | 26.45 | 200 | qa_f1 | 2.535 |
| single_qa | qasper | 12.25 | 200 | qa_f1 | 3.441 |
| multi_qa | 2wikimqa | 15.41 | 200 | qa_f1 | 3.096 |
| multi_qa | hotpotqa | 15.39 | 200 | qa_f1 | 2.708 |
| multi_qa | musique | 10.53 | 200 | qa_f1 | 2.623 |
| few_shot | samsum | 42.87 | 200 | rouge | 2.873 |
| few_shot | trec | 71.00 | 200 | classification | 2.996 |
| few_shot | triviaqa | 91.27 | 200 | qa_f1 | 2.698 |
| summarization | gov_report | 32.55 | 200 | rouge | 2.832 |
| summarization | multi_news | 26.22 | 200 | rouge | 4.911 |
| summarization | qmsum | 23.27 | 200 | rouge | 2.706 |
| synthetic | passage_count | 7.98 | 200 | count | 2.672 |
| synthetic | passage_retrieval_en | 97.81 | 200 | retrieval | 2.726 |
| code | lcc | 65.75 | 500 | code_sim | 4.542 |
| code | repobench-p | 57.08 | 500 | code_sim | 2.907 |

**Category means & aggregate:**

| category | mean score |
|---|---:|
| single_qa | 21.31 |
| multi_qa | 13.78 |
| few_shot | 68.38 |
| summarization | 27.34 |
| synthetic | 52.90 |
| code | 61.42 |
| **MACRO (mean of 6)** | **40.85** |
| flat 16-task mean | 38.82 |
| mean kv_size_bits (16 tasks) | **3.081** |

Run-ids for every cell: code=`20260725-074007-2a7786b`, synthetic=`20260725-085126-2a7786b`,
single_qa=`20260725-091628-2a7786b`, multi_qa=`20260725-100400-2a7786b`,
few_shot=`20260725-103729-2a7786b`, summarization=`20260725-112807-122045a`.

---

## 2. VS BANKED

### 2a. Final (dec8tl) vs banked July-15 k4_b2.5 (fp32/frozen), same 16 tasks

Banked set = k4_b2.5 @ SHA `21e6d81` (the duel SHA; reproduces doc §2 category means exactly):
single_qa=`20260714-020840-21e6d81`, multi_qa=`20260714-045630-21e6d81`,
few_shot=`20260714-073836-21e6d81`, summarization=`20260714-160939-21e6d81`,
synthetic=`20260715-022318-21e6d81`, code=`20260715-051537-21e6d81`.
(Both later `d02f1c0`/`455908e`/`6254107` re-runs of the same cells exist and agree to gen-noise
≤0.02 pts; `21e6d81` chosen as the doc-authoritative banked set.)

**Per-task delta (final − banked):**

| category | task | banked (fp32) | final (dec8tl) | Δscore | banked kv | final kv | Δbits |
|---|---|---:|---:|---:|---:|---:|---:|
| code | lcc | 63.92 | 65.75 | **+1.83** | 6.618 | 4.542 | −2.076 |
| code | repobench-p | 56.94 | 57.08 | +0.14 | 3.441 | 2.907 | −0.533 |
| synthetic | passage_count | 6.99 | 7.98 | +0.99 | 2.999 | 2.672 | −0.327 |
| synthetic | passage_retrieval_en | 98.37 | 97.81 | −0.55 | 3.125 | 2.726 | −0.399 |
| single_qa | multifieldqa_en | 26.21 | 25.24 | −0.97 | 3.721 | 3.027 | −0.694 |
| single_qa | narrativeqa | 26.88 | 26.45 | −0.43 | 2.708 | 2.535 | −0.173 |
| single_qa | qasper | 12.07 | 12.25 | +0.18 | 4.457 | 3.441 | −1.016 |
| multi_qa | 2wikimqa | 14.73 | 15.41 | +0.68 | 3.858 | 3.096 | −0.761 |
| multi_qa | hotpotqa | 14.91 | 15.39 | +0.48 | 3.060 | 2.708 | −0.351 |
| multi_qa | musique | 10.06 | 10.53 | +0.46 | 2.937 | 2.623 | −0.314 |
| few_shot | samsum | 42.94 | 42.87 | −0.08 | 3.393 | 2.873 | −0.520 |
| few_shot | trec | 72.50 | 71.00 | **−1.50** | 3.705 | 2.996 | −0.709 |
| few_shot | triviaqa | 91.20 | 91.27 | +0.08 | 3.119 | 2.698 | −0.421 |
| summarization | gov_report | 32.43 | 32.55 | +0.12 | 3.392 | 2.832 | −0.560 |
| summarization | multi_news | 26.49 | 26.22 | −0.27 | 7.226 | 4.911 | −2.316 |
| summarization | qmsum | 23.22 | 23.27 | +0.05 | 3.090 | 2.706 | −0.384 |

**Category & aggregate delta:**

| category | banked k4_b2.5 | final dec8tl | Δ |
|---|---:|---:|---:|
| single_qa | 21.72 | 21.31 | −0.41 |
| multi_qa | 13.24 | 13.78 | +0.54 |
| few_shot | 68.88 | 68.38 | −0.50 |
| summarization | 27.38 | 27.34 | −0.03 |
| synthetic | 52.68 | 52.90 | +0.22 |
| code | 60.43 | 61.42 | +0.99 |
| **MACRO** | **40.72** | **40.85** | **+0.13** |
| mean kv bits | 3.803 | 3.081 | **−0.722** |

Per-task wins final>banked: 10/16.
**Verdict: the int8_tl decoder + rotated-W pack is a strict accounting win — banked-parity
quality (macro +0.13) at −0.72 mean bits.** Every task drops bits (dec8tl kv < banked kv on
all 16, driven by the int8 tier-limited decoder shrinking the K spectral-pack charge; lcc and
multi_news, the two highest-bit tasks, drop the most: −2.08 / −2.32).

### 2b. Final (dec8tl) vs banked July-15 turboquant_mse_b3 (full 16-task)

Banked tq_b3 @ `21e6d81`: single_qa=`20260714-025207`, multi_qa=`20260714-052502`,
few_shot=`20260714-082220`, summarization=`20260714-193616`, synthetic=`20260715-024419`,
code=`20260715-062015` (all `-21e6d81`). Recomputed macro = **40.37** = doc §2 exactly.

| category | dec8tl | banked tq_b3 | Δ |
|---|---:|---:|---:|
| single_qa | 21.31 | 22.39 | −1.07 |
| multi_qa | 13.78 | 14.00 | −0.22 |
| few_shot | 68.38 | 68.40 | −0.02 |
| summarization | 27.34 | 27.89 | −0.55 |
| synthetic | 52.90 | 49.54 | **+3.36** |
| code | 61.42 | 60.02 | **+1.40** |
| **MACRO** | **40.85** | **40.37** | **+0.48** |
| mean kv bits | 3.081 | 3.206 | −0.125 |

Per-task wins dec8tl>tq_b3: 6/16. **The +0.48 macro edge is retrieval/synthetic (+3.36) and
code (+1.40) ONLY; all four language categories favor tq_b3 slightly** — consistent with the
July-15 duel's finding that language categories carry no discriminative power and the K4
quality edge is retrieval-category-only.

### 2c. vs the 2026-07-08 verdict (b3 40.56 @ 3.21b, k2b 40.62 @ 3.94b)

**MEMORY-CITED, NOT IN TREE.** `Glob results/k3_longbench/20260708*` returns nothing; no run
in `results/k3_longbench/2026070*` carries a k2b (`k2b`) LongBench arm. The 07-08 verdict
numbers appear only in the July-15 duel doc §7 as program context. They cannot be reverified
from parquets. (Note: the duel doc §2 reports k4_b2.2=40.56 @ 3.65b and k4_b2.5=40.72 @ 3.80b —
distinct from the 07-08 b3/k2b figures, which the doc explicitly attributes to the prior
verdict.) **Marked memory-cited.**

---

## 3. BASELINE REFRESH AUDIT — July-26 vs July-15, fp16 / tq_b3 / tq_k3v2 (synthetic + code, full)

July-26 refresh: synthetic=`20260726-050209-a50b500`, code=`20260726-054753-a50b500`.
July-15 counterparts (@`21e6d81`, full n): synth fp16=`20260715-014531`, synth tq_b3=`20260715-024419`,
synth tq_k3v2=`20260715-030238`, code fp16=`20260715-032109`, code tq_b3=`20260715-062015`,
code tq_k3v2=`20260715-071320`.

| cat | task | arm | Jul-15 | Jul-26 | Δscore | kv-15 | kv-26 | n |
|---|---|---|---:|---:|---:|---:|---:|---:|
| synthetic | passage_count | fp16 | 7.18 | 6.85 | −0.33 | 16.0 | 16.0 | 200 |
| synthetic | passage_retrieval_en | fp16 | 97.73 | 97.73 | 0.00 | 16.0 | 16.0 | 200 |
| synthetic | passage_count | tq_b3 | 3.10 | 3.10 | 0.00 | 3.098 | 3.098 | 200 |
| synthetic | passage_retrieval_en | tq_b3 | 95.98 | 95.98 | 0.00 | 3.098 | 3.098 | 200 |
| synthetic | passage_count | tq_k3v2 | 2.81 | 2.81 | 0.00 | 2.601 | 2.601 | 200 |
| synthetic | passage_retrieval_en | tq_k3v2 | 95.33 | 95.33 | 0.00 | 2.601 | 2.601 | 200 |
| code | lcc | fp16 | 65.17 | 65.19 | +0.01 | 16.0 | 16.0 | 500 |
| code | repobench-p | fp16 | 58.80 | 58.75 | −0.05 | 16.0 | 16.0 | 500 |
| code | lcc | tq_b3 | 62.51 | 62.52 | +0.01 | 3.635 | 3.635 | 500 |
| code | repobench-p | tq_b3 | 57.52 | 57.51 | −0.01 | 3.175 | 3.175 | 500 |
| code | lcc | tq_k3v2 | 60.75 | 60.62 | −0.13 | 3.159 | 3.159 | 500 |
| code | repobench-p | tq_k3v2 | 54.42 | 54.44 | +0.02 | 2.681 | 2.681 | 500 |

**Verdict: refresh reproduces July-15 to gen-noise.** Max |Δ| = 0.33 (passage_count fp16, a
noisy 0–10 count metric); all quantized cells reproduce to ≤0.13; kv bits bit-identical.
Harness/codec stability across SHA `21e6d81`→`a50b500` is confirmed — the July-26 baselines are
a valid same-config reference for the dec8tl comparison.

---

## 4. THE CATEGORY CLAIM — "k4_b2.5_dec8tl beats tq_b3 on all four synthetic+code cells at fewer bits"

dec8tl cells: lcc/repobench-p=`20260725-074007-2a7786b`, passage_count/retrieval=`20260725-085126-2a7786b`.
tq_b3 same-day (July-26 refresh): lcc/repobench-p=`20260726-054753-a50b500`,
passage_count/retrieval=`20260726-050209-a50b500`.

| cell | dec8tl score | tq_b3 score | Δscore | dec8tl kv | tq_b3 kv | Δbits | quality win? | fewer bits? |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| lcc | 65.75 | 62.52 | **+3.23** | 4.542 | 3.635 | **+0.91** | YES | **NO** |
| repobench-p | 57.08 | 57.51 | **−0.43** | 2.907 | 3.175 | −0.268 | **NO** | YES |
| passage_count | 7.98 | 3.10 | +4.88 | 2.672 | 3.098 | −0.426 | YES | YES |
| passage_retrieval_en | 97.81 | 95.98 | +1.83 | 2.726 | 3.098 | −0.372 | YES | YES |

### ⚠ The claim as literally stated is FALSIFIED — in two independent ways:

1. **lcc wins on quality (+3.23) but costs MORE bits (+0.91).** lcc has the shortest average
   sequence in the code category, so the spectral pack's per-sequence fp16-decoder charge
   (16·C/S) inflates dec8tl's bpe_k to 6.40 (kv 4.54) above tq_b3's flat 3.64. "Fewer bits"
   fails here.
2. **repobench-p loses on quality (−0.43)** though it does win on bits.

**Only 2 of 4 cells (passage_count, passage_retrieval_en) are strict Pareto wins** (better
quality AND fewer bits).

### The aggregate/category picture DOES hold (weaker, true statement):

- **4-cell mean quality: dec8tl 57.16 vs tq_b3 54.78 = +2.38, at −0.04 mean bits** (aggregate
  bit-parity — lcc's high per-sequence charge cancels the other three cells' savings).
- **Category means both favor dec8tl:** synthetic 52.90 vs 49.54 (+3.36); code 61.42 vs 60.01
  (+1.41).
- Defensible restatement: *"k4_b2.5_dec8tl wins both the synthetic and code CATEGORY means over
  tq_b3, at roughly equal aggregate bits (−0.04) — a strict per-cell Pareto win on both synthetic
  cells, a quality win at higher bits on lcc, and a bits win at slightly lower quality on
  repobench-p."*

### bpe accounting (dec8tl, honest kv_size_bits + bpe_k/bpe_v split):

| cell | bpe_k | bpe_v | kv_size_bits | compression |
|---|---:|---:|---:|---:|
| lcc | 6.402 | 2.683 | 4.542 | 3.52× |
| repobench-p | 3.627 | 2.187 | 2.907 | 5.50× |
| passage_count | 3.240 | 2.104 | 2.672 | 5.99× |
| passage_retrieval_en | 3.347 | 2.104 | 2.726 | 5.87× |

tq_b3 is symmetric (bpe_k = bpe_v = kv): lcc 3.635, repobench-p 3.175, passage_count 3.098,
passage_retrieval_en 3.098.

---

## 5. QWEN PROBE — Qwen3-8B, n=100 synthetic+code (5 arms) — `20260725-160357-1c48a9d`

**⚠ CAVEAT: n=100 PROBE, NOT a full-split run. Not headline-eligible. n asserted = 100 for all
20 arm×task cells (samples.parquet).** Note: the k4 arms here are `k4_b2.2` / `k4_b2.5`
(fp32 decoder, pack=`k4_packs_qwen3.safetensors`) — **no `dec8tl` arm was run on Qwen.**

| arm | passage_count | passage_retrieval_en | lcc | repobench-p | 4-mean |
|---|---:|---:|---:|---:|---:|
| fp16 | 7.89 | 94.75 | 70.57 | 62.59 | 58.95 |
| k4_b2.2 | 4.83 | 82.70 | 71.73 | 61.41 | 55.17 |
| k4_b2.5 | 6.11 | 84.50 | 71.87 | 62.77 | 56.31 |
| turboquant_mse_b3 | 2.42 | 19.83 | 65.73 | 53.81 | 35.45 |
| turboquant_mse_k3v2 | 4.00 | 49.33 | 66.49 | 59.22 | 44.76 |

kv_size_bits: k4_b2.2 4-mean 3.451, k4_b2.5 3.688, tq_b3 3.243, tq_k3v2 2.752.

**Probe reading (n=100, directional only):** On Qwen3-8B the TurboQuant baselines COLLAPSE on
synthetic retrieval — tq_b3 passage_retrieval_en = **19.83** vs k4_b2.5's 84.50 and fp16's
94.75 (tq_k3v2 also weak at 49.33). The K4 arms retain most of fp16's synthetic retrieval on
Qwen, a far larger gap than the Llama synthetic edge. Code is near-parity across arms
(k4_b2.5 even edges fp16 on lcc, 71.87 vs 70.57 — within n=100 noise). **This replicates the
Llama "retrieval edge is the K4 story" finding and suggests it is model-transferable, but it is
a 100-sample probe and must be run at full splits before any cross-model claim.**

---

## 6. SANITY SWEEP

- **Missing tasks:** none. dec8tl covers all 16 English LongBench v1 tasks (all 6 categories);
  every task present with correct full-split n.
- **Subsampled cells (Llama full runs):** none. All July-25 dec8tl and July-26 refresh cells
  are at canonical full-split n (metrics n == samples.parquet rows, asserted). Only the Qwen
  run is subsampled (n=100) — by design, a probe.
- **Duplicate runs (July-25/26 batch):** none. 9 distinct configs.
- **Cross-batch duplicates of banked cells** (July-14 vs July-15/16 k4_b2.5) exist and agree to
  gen-noise (≤0.02 pts) — expected reruns across code versions, not anomalies. `21e6d81` used
  as the authoritative banked reference (matches duel doc).
- **Doc/parquet count discrepancy:** duel doc §37 says "n=2,930"; parquets (both banked and
  final) are 3750. Flagged — does not affect banked-vs-final comparability (both are the same
  3750-sample splits).
- **`compression` column** = 16 / kv_size_bits (verified on a row).
- **07-08 verdict runs (b3, k2b):** NOT in the tree — memory-cited only (§2c).

---

## Headline verdicts

1. **FINAL dec8tl full table (Llama, 3750 samples/arm, full splits verified): MACRO 40.85 @
   3.081 mean bits.** Categories: single_qa 21.31 / multi_qa 13.78 / few_shot 68.38 / summ
   27.34 / synthetic 52.90 / code 61.42.
2. **dec8tl vs banked k4_b2.5 (fp32): +0.13 macro at −0.72 mean bits** — the int8_tl decoder +
   rotated-W pack is a near-free ~0.7-bit accounting win at banked-parity quality (all 16 tasks
   drop bits).
3. **dec8tl vs banked tq_b3: +0.48 macro at −0.125 mean bits**, edge entirely from
   synthetic (+3.36) and code (+1.40); language categories favor tq_b3.
4. **⚠ The "beats tq_b3 on all four synthetic+code cells at fewer bits" claim is FALSE as
   stated** — lcc wins quality (+3.23) but at MORE bits (+0.91, short-sequence pack charge);
   repobench-p loses quality (−0.43). Only 2/4 cells (both synthetic) are strict Pareto wins.
   The true statement: dec8tl wins both category means at ~equal aggregate bits (4-cell mean
   +2.38 quality, −0.04 bits).
5. **Baseline refresh: clean** — July-26 reproduces July-15 to ≤0.33 (fp16 count noise), ≤0.13
   quantized; kv bits identical. Harness stable across SHAs.
6. **Qwen probe (n=100, NOT full): TurboQuant collapses on Qwen synthetic** (tq_b3 retrieval
   19.83 vs k4_b2.5 84.50); K4 robust. Directional cross-model support for the retrieval edge —
   needs a full-split rerun before any claim.
