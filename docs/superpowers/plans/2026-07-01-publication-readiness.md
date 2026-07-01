# KV-Cache Compression Paper — Publication-Readiness Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Read the "Nature of this plan" note below first — this is a mixed plan (analysis + experiments + figures), not a pure TDD code plan.**

**Goal:** Take the closed-positive KV-cache compression program to a full-conference-paper-ready state, benchmarked box-for-box against TurboQuant (arXiv 2504.19874), the closest published mirror.

**Architecture:** The paper's spine is task-quality-vs-compression (NIAH + LongBench), matching TurboQuant's presentation format exactly, then adding two contributions TurboQuant does not make: (1) a stronger long-context claim (3-bit keys hold fp16 quality at 32k–128k where TurboQuant's own arm degrades), and (2) a systems result (compression realized in resident memory + a fused decode kernel). All quality arms run through the one fair `StreamingQuantizedCache` / `PackedStreamingCache` path already built.

**Tech Stack:** Python 3.12, PyTorch 2.12, transformers 5.11, Triton 3.7 (VM only), tyro CLIs, parquet artifacts, matplotlib figures. Dev box is AMD 7900 XTX (no CUDA); GPU-authoritative runs go to a rented NVIDIA GH200 VM via git transport.

## Nature of this plan

This is **not** a greenfield TDD build. The science is done and the harnesses exist. This plan has three task *kinds*, flagged per task:

- **[ANALYSIS]** — reads existing artifacts / paper, produces a framing or gap document. No test cycle; deliverable is a committed markdown file whose acceptance criterion is stated inline.
- **[FIGURE/CODE]** — extends existing plot or experiment code. Gets the test-first cadence where a test is meaningful (figure smoke tests, schema tests); otherwise a visual-inspection acceptance criterion.
- **[VM-RUN]** — an authoritative run on the GH200. Deliverable is a committed parquet + a one-paragraph results note. **Blocked on VM access**; the plan assumes access is available (per user decision) and gives exact commands.

Tasks are ordered so that everything doable **without** the VM comes first (framing, figure format, gap-closing), then the VM runs, then the writeup assembly that consumes VM outputs. A worker without a GPU can complete Tasks 1–7 and 12 today.

## Global Constraints

Copied verbatim from `CLAUDE.md` — every task implicitly includes these:

- **NEVER `git commit` without the user's explicit approval.** Stage, propose a message, stop. No "Co-Authored-By" or any AI attribution, ever.
- Before any commit: `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` — all clean, then re-stage. Expected baseline: **271 passed, 17 skipped, 1 xfailed** (342 on the GH200 with Triton — pre-cleanup figure, not re-verified locally; expect roughly +8 adjustment pending the GH200 re-run, see `docs/2026-07-01-kv-code-cleanup-results.md`).
- Dependencies only via `uv add` / `uv add --dev`. Never hand-edit `pyproject.toml`.
- Use the Bash tool (git bash), not PowerShell. Shell cwd resets between turns — `cd /d/Projects/bmx` first in fresh shells.
- Comparisons align on **total bits / `param_count()`** (all metadata counted), never on rank.
- Rank codecs on **inner-product / logit distortion** vs real queries, not Frobenius. Perplexity is end-to-end, too coarse to attribute component choices.
- Plot scripts must **select runs explicitly** (`newest_run_with`) — blind concat double-counts reruns; `bits == -1` marks ablation rows in k2b parquets.
- VM transport is git: push → pull → run → commit parquet back. VM has no push creds (git-bundle transport). Run experiments as modules: `uv run python -m experiments.<name>`.

## The TurboQuant parity checklist (the bar this plan clears)

From reading arXiv 2504.19874 §4. TurboQuant reports **no memory table and no latency/throughput** — their entire memory claim is the "KV Size (bits)" column in Table 1 + a "≥4.5×" prose figure. Their accepted spine is:

| TurboQuant did (§/fig) | Our status | Task |
|---|---|---|
| NIAH, Llama-3.1-8B-Instruct, Fu et al. setup, 4k→104k, recall, length×depth **heatmap** with single aggregate **Score** (Fig 4) | data exists to 128k; heatmap exists but lacks aggregate-score annotation + paper layout | 4, 9 |
| LongBench **-V1**, per-category table incl. Code, **KV Size (bits)** column, baselines (Table 1) | Code-only harness built; need full 6-category table + KV-Size column | 6, 10 |
| Baselines: KIVI, PolarQuant, SnapKV, PyramidKV, Full-Precision | **transitive** — run fp16/turboquant/KIVI as the reproduced ANCHOR; cite TurboQuant's numbers for the rest | 2, 10 |
| Compression stated as ×-factor / bits headline | have honest bpe; need it as a first-class reported column | 5 |
| Distortion-vs-theoretical-bounds plots (Fig 3) | have the `breakeven`/`stats` instrument; not yet plotted vs bounds | 8 |
| Single A100, no memory/latency table | we ADD a resident-memory census + fused kernel (upside) | 11 (writeup) |

**Framing rule for the whole paper (from the user):** the fused-kernel speedup is measured **only against a naive PyTorch chunked baseline** — it is a *systems-feasibility* result (compression is realizable in a single fused decode launch with in-kernel dequant), **not** a competitive-latency claim vs FlashAttention/vLLM. Never state or imply otherwise. It is a "nice-to-have" contribution, not load-bearing.

---

### Task 1: [ANALYSIS] Publication-readiness assessment + claim ledger

**Files:**
- Create: `docs/2026-07-01-publication-readiness-assessment.md`

**Interfaces:**
- Produces: the canonical gap analysis every later task refers to. Downstream tasks cite section anchors from this file (e.g. "closes gap G3").

**Deliverable:** A markdown document with these sections, each grounded in a specific existing artifact or the TurboQuant PDF (cite `results/<run>/` paths and paper figure/table numbers):

1. **The four claims** (from user decision), each with: the exact statement, the artifact(s) that support it today, and what's still missing to make it airtight.
   - C1 — *KV recipe beats TurboQuant at matched compression* (NIAH + LongBench). Spine.
   - C2 — *Long-context edge*: 3-bit keys hold fp16-quality retrieval at 32k–128k where turboquant_mse degrades.
   - C3 — *Runtime memory realized*: census (64.1 GiB @128k ≈ fp16) + OOM-vs-completes A/B.
   - C4 — *Fused kernel feasibility* (softened: speedup vs naive PyTorch only).
2. **Gap register** — table of `G#, claim, what's missing, task that closes it, VM-needed y/n`.
3. **The reframe** — memory must be reported as the **KV-cache slice** (16 GiB fp16 → ~3 GiB k2b, ~5.3×) and the OOM-vs-completes capability result, **not** as total process RSS (where fixed weights + activations mask the win). State this explicitly with the `kv_memory.py` term decomposition (W=14.9, C=16 @128k, A=61.3).
4. **What we claim MORE than TurboQuant** — C2 (long-context) and C3 (realized memory); TurboQuant shows neither.
5. **Honest weaknesses to disclose in the paper** — single-run measurements; census was prefill+4-decode-steps not full generation; cross-model generalization limited to uniform-global-attention models (Qwen3-27B/Gemma-4-31B hybrid attention incompatible); kernel speedup baseline is naive PyTorch.

**Acceptance criterion:** Every one of the four claims maps to ≥1 supporting artifact path OR an explicit gap-task. No claim is left without either evidence or a task that produces it. Reviewer reading this file alone understands what's proven vs pending.

- [ ] **Step 1:** Draft the document with all five sections, pulling numbers from `docs/2026-06-23-kernel-census-results.md`, `docs/2026-06-24-triton-decode-results.md`, `docs/2026-06-21-niah-longbench-frontier-results.md`, and the TurboQuant PDF §4 / Table 1 / Fig 4.
- [ ] **Step 2:** Cross-check every cited number against its source artifact (open the parquet or doc; do not trust memory). Correct any drift.
- [ ] **Step 3:** Stage and propose commit message `docs: publication-readiness assessment + claim ledger vs TurboQuant`. STOP for user approval.

---

### Task 2: [ANALYSIS] Transitive-baseline justification memo

**Files:**
- Create: `docs/2026-07-01-baseline-parity-decision.md`

**Interfaces:**
- Consumes: Task 1's gap register.
- Produces: the baseline strategy referenced by Tasks 9, 10, 12.

**DECISION (locked by user 2026-07-01): baselines are TRANSITIVE. We do NOT implement PolarQuant / SnapKV / PyramidKV.** Rationale: we beat TurboQuant at matched compression; TurboQuant already beat KIVI/PolarQuant/SnapKV/PyramidKV in its own Table 1 / Fig 4; therefore we dominate them transitively. This deletes the former PolarQuant-implementation task (old Task 7) entirely.

**The seam this memo must close:** transitivity only holds if our measurement path is numerically comparable to TurboQuant's. The license for that is the **anchor**: we run `fp16` (Full Cache) and `turboquant_mse` (+`prod`) **on our own `StreamingQuantizedCache` path** and show they reproduce TurboQuant's published Table-1 values (Code ≈46, Avg ≈50). If our reproduced anchor rows land on their numbers, every transitive claim about the un-run baselines is licensed. This is why we KEEP three baselines (fp16, turboquant_mse/prod, KIVI) — they are the anchor, not new work — and drop only the ones that would require fresh implementation.

**Deliverable:** A short memo stating: (1) the transitive argument; (2) the anchor mechanism (reproduce fp16 + turboquant_mse on our path → match Table 1); (3) the three baselines we run and why each is the anchor, not a courtesy; (4) the un-run baselines (PolarQuant/SnapKV/PyramidKV) reported as TurboQuant's published reference rows in T1, clearly attributed. SnapKV/PyramidKV noted as eviction (different mechanism) so their absence is scope, not omission.

**Acceptance criterion:** The memo makes the transitivity airtight by pinning it to the reproduced anchor. It states explicitly that Task 10 MUST verify the anchor rows match (Code ≈46) before any transitive claim is written.

- [ ] **Step 1:** Confirm the current arm inventory in `src/bmx/cache/codecs.py` CACHE_ARMS (fp16 handled by the dense path; turboquant_mse/prod, kivi present).
- [ ] **Step 2:** Write the memo (four points above).
- [ ] **Step 3:** Stage, propose `docs: transitive-baseline justification (anchor = reproduced TurboQuant Table-1 rows)`. STOP for approval.

---

### Task 3: [ANALYSIS] Results-section skeleton in TurboQuant's structure

**Files:**
- Create: `docs/2026-07-01-paper-results-skeleton.md`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: the section/figure/table outline that Tasks 4–11 fill. Each figure/table gets a stable ID (F1, T1, …) reused everywhere downstream.

**Deliverable:** A markdown skeleton mirroring TurboQuant §4, with each element tagged `[HAVE]`, `[NEEDS-FIGURE]`, or `[NEEDS-VM]`:

- **§Empirical validation** — distortion vs bit-width vs theoretical bounds (our Fig **F1** ← Task 8). `[HAVE data, NEEDS-FIGURE]`
- **§NIAH** — recall heatmap per arm + aggregate score (Fig **F2** ← Tasks 4, 9), recall-vs-length line (Fig **F3**). `[NEEDS-VM full sweep]`
- **§LongBench** — per-category table with KV-Size column (Table **T1** ← Tasks 6, 10). `[NEEDS-VM full sets]`
- **§Systems (our addition)** — resident-memory census table (**T2**) + OOM-vs-completes callout + fused-kernel latency table (**T3**, framed as feasibility-vs-naive-baseline). `[HAVE]`
- **§Long-context edge (our addition)** — the 32k–128k crossover where 3-bit-K holds and turboquant_mse degrades (Fig **F4** ← Task 9). `[NEEDS-VM]`

**Acceptance criterion:** Every figure/table in the checklist table (top of plan) appears with an ID and a source task. No orphan claims; no figure without a producing task.

- [ ] **Step 1:** Write the skeleton with element IDs and status tags.
- [ ] **Step 2:** Verify each of the four claims (C1–C4) is visible in at least one section.
- [ ] **Step 3:** Stage, propose `docs: paper results-section skeleton (TurboQuant-parity structure)`. STOP for approval.

---

### Task 4: [FIGURE/CODE] NIAH heatmap — add aggregate-score annotation (TurboQuant Fig 4 parity)

**Files:**
- Modify: `experiments/plots/plot_k3_niah.py:48-79` (the heatmap block)
- Test: `tests/test_niah.py` (add a figure-smoke test) OR `tests/test_k3_niah_experiment.py` if that's where plot tests live — check first.

**Interfaces:**
- Consumes: a NIAH parquet DataFrame with columns `arm, length, depth, recall_full, compression`.
- Produces: `niah_recall_heatmap.png` where each per-arm subplot title includes an aggregate score `Score: X.XX` (TurboQuant's single-number summary — mean of `recall_full/10` over the grid, so it reads 0–1 like theirs), plus the compression factor.

**Context:** The heatmap already exists (`plot_k3_niah.py:48-79`). TurboQuant's Fig 4 signature is the **single aggregate Score above each arm's grid** (e.g. "TurboQuant Score: 0.997", "KIVI Score: 0.981"). We need that annotation to be directly comparable. Their score is a 0–1 mean over the grid; ours is `recall_full` on a 0–10 scale, so divide by 10.

- [ ] **Step 1: Write the failing test.** Confirm which test file holds plot tests (`grep -rl make_figures tests/`), then add:

```python
def test_niah_heatmap_has_aggregate_score(tmp_path):
    import pandas as pd
    from experiments.plots.plot_k3_niah import make_figures
    df = pd.DataFrame(
        {
            "arm": ["k2b"] * 4 + ["fp16"] * 4,
            "length": [4096, 4096, 8192, 8192] * 2,
            "depth": [0.25, 0.75, 0.25, 0.75] * 2,
            "recall_full": [8.0, 8.0, 7.0, 7.0, 9.0, 9.0, 9.0, 9.0],
            "compression": [5.3] * 4 + [1.0] * 4,
        }
    )
    paths = make_figures(df, str(tmp_path))
    # A machine-readable aggregate-score sidecar is emitted for the heatmap.
    scores = {
        p.stem: p for p in paths if p.name == "niah_heatmap_scores.json"
    }
    assert scores, "expected a niah_heatmap_scores.json sidecar"
    import json
    data = json.loads(scores["niah_heatmap_scores"].read_text())
    # k2b grid mean = 7.5/10 = 0.75; fp16 = 9.0/10 = 0.90
    assert abs(data["k2b"] - 0.75) < 1e-6
    assert abs(data["fp16"] - 0.90) < 1e-6
```

- [ ] **Step 2: Run it, verify it fails.** `cd /d/Projects/bmx && uv run pytest tests/<file>::test_niah_heatmap_has_aggregate_score -v` → FAIL (no sidecar / KeyError).

- [ ] **Step 3: Implement.** In the heatmap block, compute per-arm aggregate score `score = nanmean(grid) / 10.0`, add it to each subplot title (`ax.set_title(f"{arm}\nScore: {score:.3f}")`), and write a `niah_heatmap_scores.json` sidecar mapping arm→score into `out_dir`, appending its path to `paths`.

- [ ] **Step 4: Run tests, verify pass.** `uv run pytest tests/<file>::test_niah_heatmap_has_aggregate_score -v` → PASS.

- [ ] **Step 5: Ruff + full suite.** `uv run ruff format . && uv run ruff check . && uv run pytest -q` → clean, 264/17/1.

- [ ] **Step 6: Commit.** Stage, propose `feat(plots): NIAH heatmap aggregate Score annotation (TurboQuant Fig-4 parity)`. STOP for approval.

---

### Task 5: [FIGURE/CODE] Report KV-Size (bits) as a first-class column in NIAH/LongBench parquets

**Files:**
- Modify: `experiments/k3_niah.py` and `experiments/k3_longbench.py` (the per-arm metric row assembly — find where `compression` is written)
- Test: `tests/test_k3_niah_experiment.py`, `tests/test_k3_longbench_experiment.py`

**Interfaces:**
- Consumes: the per-arm `_spec_pair` + `compression_for()` machinery already emitting `compression`.
- Produces: an added `kv_size_bits` column (the honest blended bits-per-entry, K+V averaged) in both experiments' output rows — the exact axis TurboQuant's Table 1 leads with ("KV Size" = 2.5, 3.5, 16).

**Context:** TurboQuant's Table 1 headline axis is **KV Size in bits** (16 for full cache, 2.5/3.5 for their arms). We currently emit `compression` (×-factor). Reviewers of the mirror paper expect the bits column too. `src/bmx/cache/generate.py::compression_for()` already computes `(bpe_k, bpe_v, compression)` — surface `kv_size_bits = (bpe_k + bpe_v) / 2` (or the blended average consistent with how compression is derived; verify the exact convention in `compression_for`).

- [ ] **Step 1: Write the failing test** (NIAH first):

```python
def test_niah_rows_have_kv_size_bits(tmp_path):
    # Use the existing offline synthetic-model fixture path this file already uses.
    # After running the experiment's arm loop over a tiny model, every emitted
    # row must carry a positive, finite kv_size_bits, and fp16 must be ~16.
    df = _run_tiny_niah(tmp_path)  # reuse this file's existing tiny-run helper
    assert "kv_size_bits" in df.columns
    assert (df["kv_size_bits"] > 0).all()
    assert df.loc[df["arm"] == "fp16", "kv_size_bits"].iloc[0] == 16.0
```

(Check the existing test file for the real tiny-run helper name; reuse it rather than inventing `_run_tiny_niah`.)

- [ ] **Step 2: Run, verify fail.** `uv run pytest tests/test_k3_niah_experiment.py::test_niah_rows_have_kv_size_bits -v` → FAIL (no column).

- [ ] **Step 3: Implement in `k3_niah.py`.** Where the row dict is built with `compression`, also compute `kv_size_bits`. For fp16 (no packed form) hardcode `16.0`; for compressed arms derive from `compression_for()`'s `(bpe_k, bpe_v)`. Confirm the exact return signature in `src/bmx/cache/generate.py` before wiring.

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Repeat Steps 1–4 for `k3_longbench.py`** with the analogous test in `tests/test_k3_longbench_experiment.py`.

- [ ] **Step 6: Ruff + full suite** → clean.

- [ ] **Step 7: Commit.** `feat(exp): emit kv_size_bits column in NIAH+LongBench (TurboQuant Table-1 axis)`. STOP for approval.

---

### Task 6: [FIGURE/CODE] LongBench — full 6-category scorers (TurboQuant Table-1 parity)

**Files:**
- Modify: `src/bmx/cache/longbench.py` (task registry + scorers), `experiments/k3_longbench.py`
- Test: `tests/test_longbench.py`, `tests/test_k3_longbench_experiment.py`

**Interfaces:**
- Consumes: the existing `code_sim` scorer + task registry + `generate_through_cache`.
- Produces: a `--categories` option running the six TurboQuant Table-1 columns — SingleQA / MultiQA / Summarization / Fewshot / Synthetic / Code — each with its LongBench-exact scorer.

**DECISION (locked by user 2026-07-01): FULL 6-category table.** No scope gate — we match TurboQuant Table 1 column-for-column. This costs real VM time (Task 10) but produces the un-arguable parity table. Each category uses LongBench's **official** metric, ported verbatim from LongBench's `metrics.py` exactly as `code_sim` was:

- SingleQA / MultiQA → `qa_f1_score` (token-level F1, with the LongBench normalization)
- Summarization → `rouge_score` (ROUGE-L, their variant)
- Fewshot → `classification_score` / `qa_f1` depending on the sub-task (check LongBench's `dataset2metric` map)
- Synthetic → `retrieval_score` / `count_score` (per sub-task)
- Code → `code_sim` (already implemented)

**LongBench variant (locked by user):** target **LongBench-V1** as the parity anchor (their Table 1 header says LongBench-V1); LongBench-E is an **optional add** for the length-vs-compression story (note it in Task 3, don't build it unless the length story needs it). Thread the variant as a `--longbench-version {v1,e}` flag defaulting to `v1`.

- [ ] **Step 1:** Read `src/bmx/cache/longbench.py` for the current registry + `code_sim`; read LongBench's `metrics.py` and `dataset2metric` in the local `LongBench/` clone to get each category's official metric + which datasets map to it.
- [ ] **Step 2: Write failing tests** — one per new scorer (`qa_f1_score`, `rouge_score`, `classification_score`, `retrieval_score`, `count_score`), each asserting the score matches a hand-computed value on a 2-example fixture (verbatim port, same discipline as `code_sim`).
- [ ] **Step 3:** Run, verify fail.
- [ ] **Step 4:** Implement each scorer + register its category with the exact prompt template + `max_gen` from the clone. Wire `--categories` and `--longbench-version` into `k3_longbench.py`.
- [ ] **Step 5:** Run, verify pass. Ruff + full suite.
- [ ] **Step 6: Commit.** `feat(longbench): full 6-category scorers + --categories/--longbench-version (Table-1 parity)`. STOP for approval.

---

*(Former Task 7 — PolarQuant baseline arm — DELETED. Baselines are transitive through the reproduced TurboQuant anchor; see Task 2. Subsequent tasks renumbered.)*

---

### Task 8: [FIGURE/CODE] Distortion-vs-theoretical-bounds figure (TurboQuant Fig 3 parity)

**Files:**
- Create: `experiments/plots/plot_distortion_bounds.py`
- Test: `tests/test_plot_distortion_bounds.py` (new)

**Interfaces:**
- Consumes: `src/bmx/quant/stats.py::sq_floor` (the `4^-b` MSE floor) + `ip_distortion` + the CACHE_ARMS run at b=1..5 on real caches (from an existing k2_cache_arms parquet, or a small offline compute).
- Produces: `distortion_vs_bitwidth.png` — measured `D_mse` and `D_prod` for turboquant_mse / turboquant_prod vs bit-width, overlaid with the paper's upper bound `√3·π/2·4^-b` (MSE) and lower bound `4^-b`, on a log-y axis (their Fig 3 exactly).

**Context:** This is the cheapest high-value parity figure — it's the plot that makes the *stronger-than-TurboQuant* point: their worst-case bounds hold on real caches, but our structure-aware arms sit **below** their curve (2–3× better on the IP metric). The bounds are closed-form; the measured points come from arms we already have.

- [ ] **Step 1: Write failing test:**

```python
def test_distortion_bounds_figure_emitted(tmp_path):
    import pandas as pd
    from experiments.plots.plot_distortion_bounds import make_figures
    # minimal frame: arm, bitwidth, d_mse, d_prod
    df = pd.DataFrame(
        {
            "arm": ["turboquant_mse"] * 4,
            "bitwidth": [1, 2, 3, 4],
            "d_mse": [0.36, 0.117, 0.03, 0.009],
            "d_prod": [1.57, 0.56, 0.18, 0.047],
        }
    )
    paths = make_figures(df, str(tmp_path))
    assert any(p.name == "distortion_vs_bitwidth.png" for p in paths)
```

- [ ] **Step 2:** Run, verify fail (module doesn't exist).
- [ ] **Step 3:** Implement `make_figures(df, out_dir)`: plot measured `d_mse`/`d_prod` per arm as markers; overlay `y = sqrt(3)*pi/2 * 4.0**-b` (upper) and `y = 4.0**-b` (lower) as dashed lines; log-y; legend matching the paper. Import the floor from `bmx.quant.stats` where possible rather than re-hardcoding.
- [ ] **Step 4:** Run, verify pass.
- [ ] **Step 5:** Add a small `experiments/distortion_bounds.py` driver (or extend `k2_cache_arms.py`) that emits the `(arm, bitwidth, d_mse, d_prod)` parquet from real caches, so the figure is reproducible not hand-fed. Ruff + full suite.
- [ ] **Step 6: Commit.** `feat(plots): distortion-vs-bounds figure (TurboQuant Fig-3 parity + our arms beat the curve)`. STOP for approval.

---

### Task 9: [VM-RUN] Authoritative NIAH sweep on the SOTA model (closes C1, C2)

**Files:**
- Create (committed from VM): `results/k3_niah/<run-id>/metrics.parquet` + `docs/2026-07-01-niah-authoritative-results.md`

**Interfaces:**
- Consumes: `experiments/k3_niah.py` with `kv_size_bits` (Task 5) and the aggregate-score plot (Task 4).
- Produces: the paper's NIAH figures F2/F3/F4 source data.

**BLOCKED ON VM ACCESS.** Exact procedure (from `docs/2026-06-20-niah-plan-state.md`, updated with the matched-compression arms from `docs/2026-06-20-longbench-plan-state.md`):

- [ ] **Step 1:** On the GH200: `cd ~/bmx && git pull` (or git-bundle transport per `vm-interaction-guide`), `scripts/vm_setup.sh`, confirm `uv run pytest -q` → 342 passed.
- [ ] **Step 2:** Run the matched-compression sweep spanning short→long context (the C2 crossover lives at 32k–128k):

```bash
uv run python -m experiments.k3_niah \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --arms fp16 k2b k2b_k2r8 turboquant_mse kivi \
  --lengths 4096 8192 16384 32768 65536 131072 \
  --depths 0.1 0.3 0.5 0.7 0.9 \
  --use-packed
```

- [ ] **Step 3: Apply the headroom guard** (from the NIAH plan-state doc, do NOT skip): flag any length where fp16 recall is at floor (non-discriminating) or every arm is at ceiling. The discriminating signal is where fp16≈10 and 2-bit arms drop while k2b holds.
- [ ] **Step 4:** Independently re-derive the headline numbers from the parquet (never trust a green run) — confirm `kv_size_bits` lands near {16, ~3, ~2.x} per arm and compression matches Task 5's expectation.
- [ ] **Step 5:** Generate F2/F3/F4 via `plot_k3_niah.make_figures`. Confirm the C2 crossover is visible (k2b holds ≥fp16 at 32k–128k; turboquant_mse degrades).
- [ ] **Step 6:** Write `docs/2026-07-01-niah-authoritative-results.md` with the recall table (per arm × length), the aggregate scores, and the C2 crossover callout. Commit parquet + figures + doc back. `git add results/ docs/ && ...`. STOP for approval before commit.

---

### Task 10: [VM-RUN] Authoritative LongBench run (closes C1 coding half)

**Files:**
- Create (from VM): `results/k3_longbench/<run-id>/metrics.parquet` + `docs/2026-07-01-longbench-authoritative-results.md`

**Interfaces:**
- Consumes: `experiments/k3_longbench.py` with `kv_size_bits` (Task 5), full categories if Task 6(a) chosen.
- Produces: Table T1 source data.

**BLOCKED ON VM ACCESS.** From `docs/2026-06-20-longbench-plan-state.md`:

- [ ] **Step 1:** VM ready (same as Task 9 Step 1).
- [ ] **Step 2:** Matched-compression, **full 6-category, LongBench-V1** run (the anchor table):

```bash
uv run python -m experiments.k3_longbench \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --arms fp16 k2b k2b_k2r8 turboquant_mse turboquant_prod kivi \
  --categories single_qa multi_qa summarization few_shot synthetic code \
  --longbench-version v1
  # n_samples defaults to None = full sets → Table-1 comparable.
```

Warn: full 6-category sets × 6 arms = many hours / real $. Smoke with `--n-samples 20 --categories code` first, confirm the anchor (Step 4), THEN launch the full run.

- [ ] **Step 3: Apply the headroom guard** (identical discipline to NIAH): flag floor/ceiling non-discriminating tasks per category. Read against TurboQuant Table 1 (Full Cache Avg ≈50.06, Code ≈46.28).
- [ ] **Step 4: ANCHOR VERIFICATION (gates the whole transitive-baseline argument — do NOT skip).** Re-derive per-category scores from the parquet and confirm our **fp16 (Full Cache)** row reproduces TurboQuant Table-1 Full-Cache values (per-category within reasonable tolerance; Avg ≈50, Code ≈46) and **turboquant_mse** reproduces its Table-1 row. If the anchor rows do NOT match, STOP — the transitivity claim (Task 2) is unlicensed and the mismatch must be root-caused before any baseline comparison is written. Confirm `kv_size_bits` column present.
- [ ] **Step 5:** Note the two KNOWN DIVERGENCES from the plan-state doc for the writeup: (a) we do NOT middle-truncate over-long prompts (we compress the full context — intentional); (b) the `length = n_prefill*2` compression proxy under-states compression (lower bound, consistent across arms). Also note the optional LongBench-E length-story run is deferred unless Task 3's skeleton called for it.
- [ ] **Step 6:** Write `docs/2026-07-01-longbench-authoritative-results.md` with T1 (the full 6-category × arm table, our arms + reproduced anchor rows + TurboQuant's published PolarQuant/SnapKV/PyramidKV reference rows clearly attributed). Commit back. STOP for approval.

---

### Task 11: [VM-RUN] Confirm systems numbers are current (closes C3, C4)

**Files:**
- Verify/refresh: `results/k3_kernel_census/<run-id>/`, `results/k3_triton_decode/<run-id>/`
- Create: `docs/2026-07-01-systems-results-final.md`

**Interfaces:**
- Consumes: `experiments/k3_kernel_census.py`, `experiments/k3_triton_decode.py`.
- Produces: Tables T2 (resident-memory census) + T3 (fused-kernel latency, feasibility framing).

**BLOCKED ON VM ACCESS.** The census + kernel numbers already exist (June 23–24) but predate any Task-5 column changes and should be re-confirmed on the current branch head before they go in a paper.

- [ ] **Step 1:** VM ready. `git pull` to the current `feat/triton-decode-kernel` head. `uv run pytest -q` → 342 passed.
- [ ] **Step 2:** Re-run the census to reconfirm the resident table (fp16 / dense_stream / chunked across 4k→128k) and the OOM-vs-completes A/B:

```bash
uv run python -m experiments.k3_kernel_census --model-name meta-llama/Llama-3.1-8B-Instruct
```

- [ ] **Step 3:** Re-run the Triton decode latency bench (RTN + k2b, splits=32) to reconfirm the speedup table vs chunked.
- [ ] **Step 4:** Write `docs/2026-07-01-systems-results-final.md` with **T2 reframed as the KV-cache slice** (16 GiB fp16 → ~3 GiB k2b) alongside the total-resident table (with the explicit note that total is masked by fixed weights+activations), the OOM-vs-completes callout, and **T3 explicitly labeled "speedup vs naive PyTorch chunked baseline — a feasibility result, not a competitive-latency claim."**
- [ ] **Step 5:** Commit back. STOP for approval.

---

### Task 12: [ANALYSIS] Assemble the paper outline + claim-to-evidence traceability matrix

**Files:**
- Create: `docs/2026-07-01-paper-outline.md`

**Interfaces:**
- Consumes: ALL prior tasks (their committed docs, figures, tables).
- Produces: the final pre-writing artifact — a section-by-section outline where every claim links to its evidence file/figure.

**Deliverable:** The assembled outline (Abstract → Intro → Method → §Empirical validation (F1) → §NIAH (F2/F3/F4) → §LongBench (T1) → §Systems (T2/T3) → §Limitations → Conclusion), PLUS a **traceability matrix**: rows = the four claims C1–C4 + every sub-claim; columns = `statement | evidence artifact | figure/table ID | status (proven/pending) | honest caveat`. This is the document that proves, at a glance, that nothing in the paper is unsupported.

**Acceptance criterion:** Zero rows with status "pending" that lack a task reference. Every figure/table ID (F1–F4, T1–T3) resolves to a committed artifact. The limitations section lists every weakness from Task 1 §5.

- [ ] **Step 1:** Assemble the outline, pulling section content pointers from the committed results docs.
- [ ] **Step 2:** Build the traceability matrix; verify every claim resolves.
- [ ] **Step 3:** Final self-check against the TurboQuant parity checklist (top of this plan) — every row either `[HAVE]` with an artifact or explicitly scoped-out with justification.
- [ ] **Step 4:** Stage, propose `docs: paper outline + claim-to-evidence traceability matrix`. STOP for approval.

---

## Self-Review

**Spec coverage** (against the user's decisions + TurboQuant parity checklist):
- Full-conference-paper bar → systems work is first-class (Tasks 11, 12 §Systems), not just bonus. ✓
- Plan assumes VM access → Tasks 9/10/11 give exact commands, sequenced after local prep. ✓
- C1 (recipe beats TurboQuant) → Tasks 9, 10. ✓
- C2 (long-context edge) → Task 9 Steps 2/5 (the 32k–128k crossover). ✓
- C3 (runtime memory realized) → Task 11, reframed as KV-slice (Task 1 §3). ✓
- C4 (fused kernel, softened) → Task 11 Step 4 explicitly labels it feasibility-vs-naive-baseline. ✓
- NIAH heatmap + aggregate score → Task 4. LongBench KV-Size column → Task 5. Full 6-category table (LOCKED, no gate) → Task 6. Baselines transitive via reproduced anchor → Tasks 2, 10-Step-4. Distortion-vs-bounds → Task 8. ✓

**User decisions locked (2026-07-01):** full 6-category LongBench table (not Code-only); LongBench-V1 as parity anchor (E optional); baselines transitive (no PolarQuant/SnapKV/PyramidKV implementation — former Task 7 deleted); anchor rigor = reproduce TurboQuant's fp16 + turboquant_mse rows on our path (Task 10 Step 4 gates the transitive argument).

**Placeholder scan:** Figure/code tasks carry real test code and real file paths. Analysis tasks carry explicit acceptance criteria instead of tests (correct for their kind — flagged in "Nature of this plan"). VM tasks carry exact commands from the existing plan-state docs. No "TBD"/"handle edge cases". ✓

**Type/name consistency:** `kv_size_bits` column name is consistent across Tasks 5, 9, 10. Figure IDs F1–F4 / table IDs T1–T3 are defined in Task 3 and reused in 8/9/10/11/12. `make_figures(df, out_dir)` signature matches the existing `plot_k3_niah.py` contract used in Tasks 4, 8. Task numbering: former Task 7 deleted, but I have NOT renumbered 8–12 in their headers to avoid breaking the many cross-references (F/T IDs and "Task N" mentions); the sequence reads 1–6, then 8–12 with a deletion marker where 7 was. ✓

**Known soft spots (flagged, not hidden):** Task 5 requires confirming the exact return signature of `generate.py::compression_for` before wiring — its Step 3 does that. Task 6 requires reading LongBench's `dataset2metric` map + `metrics.py` from the local clone before porting scorers — its Step 1 does that. Task 10 Step 4 is a hard gate: if the reproduced anchor rows don't match TurboQuant's Table 1, the transitive-baseline argument is void and must be root-caused first.
