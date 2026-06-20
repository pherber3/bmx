# NIAH retrieval metric under KV compression — design (2026-06-20)

A needle-in-a-haystack (NIAH) **recall** metric that runs the existing KV-compression
arms through the **same `StreamingQuantizedCache` path** the K3 ppl sweep uses, following
the TurboQuant / Fu et al. setup: single needle at an arbitrary location, swept across
document lengths, scored by recall. It answers what perplexity cannot — *does the model
still retrieve a planted fact from long pasted context after the KV is compressed?*

This is the first half of the decided next direction (`docs/2026-06-19-k3-handoff-and-workflow.md`
item #0): a **task** metric, not just perplexity. Scope here is **retrieval only**;
code-gen pass@1 is a future, separate spec (we'll do both eventually; retrieval first
because it reuses the needle scaffolding and stresses exactly what compression damages).

## Why (the gap perplexity leaves)

K1→K3 closed the science on quantized-prefill / streaming perplexity: ppl-within-1% of
fp16. But ppl says the *distribution* held, not that the model still *uses* information
stored deep in the cached context. Lossy KV compression is most likely to damage exactly
that — facts in older tokens that have been flushed to quantized storage. The audience is
people running local models for long-context / coding work; "the perplexity barely moved"
does not reassure them that an 8k-token pasted file is still retrievable. A recall metric
under compression does.

## Anchor: TurboQuant / Fu et al. setup

From the TurboQuant paper (their long-context evaluation):

- **Single needle** ("a sentence") planted at an **arbitrary location** in a much larger
  haystack; score = **recall** (did the model retrieve the hidden sentence).
- **Sweep document length** (their axis: 4k → 104k tokens). Length is the primary stressor.
- **Model**: Llama-3.1-8B-Instruct (the Fu et al. standard).
- **Comparison at a fixed memory ratio** (they use 0.25 = 4×).
- Baselines in the paper: PolarQuant, SnapKV, PyramidKV, KIVI. We already have KIVI and
  TurboQuant (mse/prod) as arms; **PolarQuant / SnapKV / PyramidKV are out of scope**
  (token-eviction is a different mechanism we have no implementation of — noted as
  comparison context, not added).

Following their setup keeps our numbers **comparable to a published baseline**, which this
repo prizes (independently re-derived, trustworthy).

## Decisions (locked in brainstorm)

1. **Sequenced scope.** Retrieval now; code-gen pass@1 later (own spec). This spec is
   retrieval-only.
2. **Single needle, recall score, length sweep** — match the paper (not multi-needle).
3. **Run split (the K3 pattern, blessed by the handoff):** local **mechanism gate** at
   small lengths on the tiny offline factory model (CI, no download); headline 4k→32k+
   sweep is a `--model-name` / `--lengths` change on the NVIDIA VM (no CUDA locally).
   The **headroom appears as length grows** — it is earned from the length axis, not from
   artificial difficulty. Short lengths passing for all arms is expected and is the
   mechanism gate, not a failure.
4. **Recall scoring — proxy/real split, mirroring `needle.py`'s existing two signals.**
   The real Fu et al. harness (`eval/needle/needle_in_haystack.py:265`) scores recall as
   **ROUGE-1 F-measure of the needle sentence vs the generated response, ×10 (graded
   0–10)** — greedy decode (`temperature=0`), `generate()`d continuation. We adopt that
   exactly for the headline; it is graded (built-in headroom), is the metric TurboQuant
   reported against, and does not hinge on a single token.
   - **CI/mechanism:** single-token answer, next-token **argmax** match, tokenizer-free at
     id level (built from the existing `_argmax_next_at` helper). Cheap, deterministic,
     no generation loop — proves indexing/no-explosion only.
   - **Headline:** `generate()` the answer through the compressed cache (greedy), score
     **ROUGE-1 F-measure ×10** of the needle sentence vs the response. Requires the
     `rouge-score` dep (add via `uv add`, never hand-edit `pyproject.toml`). VM only.
5. **Fair comparison — honest measured compression, not pinned ratios.** Per the repo's
   hard rule (align on measured bits / `param_count()`, never bend configs to a vanity
   ratio), each arm runs at its natural config and reports its honest `compression` from
   `memory_report`. We plot **recall vs length per arm**, annotate each with its measured
   compression ratio, and draw the paper's **4× line as reference** — reading the result
   against the paper without forcing arms onto an exact 4× target.

## Architecture — reuse the K3 split exactly

Two thin new pieces; no new codec, no new cache class. Everything routes through
`_spec_pair(arm)` → `StreamingQuantizedCache` — the one fair code path.

- **`src/bmx/cache/niah.py`** — metric core, mirroring `needle.py`'s proxy/real split:
  - `niah_recall_argmax(...)` — CI/mechanism gate. Single-token answer, next-token argmax
    match, tokenizer-free. Built from the existing `_argmax_next_at` helper. Proves the
    cache indexes the needle position correctly and does not explode. Tiny offline model.
  - `niah_recall_generate(...)` — headline. Prefills the haystack into
    `StreamingQuantizedCache`, `generate()`s the answer (greedy/deterministic), normalized
    string-match against a multi-token magic-number value. Tokenizer-required, VM only.
- **Haystack builder** (extend `needle.py`'s `build_needle_ids` or a sibling). Two filler
  regimes, matching the run split:
  - **Headline (VM):** real **Paul Graham essays** as filler (shipped in the local clone at
    `Long-Context-Data-Engineering/eval/needle/PaulGrahamEssays/` — vendor or point at
    them), trimmed to the target length, **needle inserted at a sentence boundary** at
    `depth_percent` (their `insert_needle` convention). Default needle/question follow
    theirs: needle *"The best thing to do in San Francisco is eat a sandwich and sit in
    Dolores Park on a sunny day."*, question *"What is the best thing to do in San
    Francisco?"*, prompt-wrapped as `This is a very long story book: <book>{context}</book>.
    Based on the content of the book, Question: {q}\nAnswer:`.
  - **Offline (CI):** synthetic repeated filler (no files, no download), single-token
    answer for the argmax proxy — the existing `build_needle_ids` style.
  Both parameterized by **document length** and **depth**. (Exact template confirmed against
  the local Fu et al. repo — see References.)

## Experiment & data flow

- **`experiments/k3_niah.py`** — thin tyro CLI, same shape as `k3_live_generation.py`.
  Config: `model_name`, `arms`, `lengths: tuple[int,...]` (e.g.
  `(4096, 8192, 16384, 32768)`), `depths: tuple[float,...]` (the depth grid is the primary
  variance control, per the paper), `seed`. (`n_trials` is an optional extra re-plant knob,
  default 1 — only if a single depth grid proves noisy.) Sweeps arms × lengths × depths;
  per cell records `recall` (ROUGE-1 ×10 for the headline; argmax hit for the proxy),
  `bpe_k`, `bpe_v`, honest `compression`, `length`, `depth`. Writes parquet via
  `artifacts.create_run` / `write_metrics`.
- **Offline-test path** (model injected, no `--model-name`): synthetic ids, small lengths,
  exercises argmax recall + parquet schema only. No download, no tokenizer — same airtight
  lazy-import guard as `k3_live_generation.py`.
- **`plots/plot_k3_niah.py`** — reads parquet, two views: (1) the paper's **length × depth
  recall heatmap per arm** (`visualize.py` is the reference) when depths are swept, and
  (2) **recall vs length per arm** with each arm annotated by its measured compression
  ratio and the 4× line as reference. Figures read parquet, never refit. Plot scripts
  select runs explicitly (`newest_run_with` pattern) — never blind-concat a results root.

## Testing & the headroom guard

Mechanism gates (offline, CI), mirroring K3's gate table:

- argmax recall **finite and deterministic** through the streaming cache (no explosion,
  indexing correct) — same family as the existing `needle_retrieved_from_ids` test.
- fp16 arm recalls the planted single-token needle at **100%** on the tiny model (sanity:
  the harness can find a needle when there is no quant error). If fp16 cannot, the test is
  broken, not the codec.
- parquet schema columns present; offline path takes no download (airtight lazy-import
  guard test, copied from `k3_live_generation`).

**Headroom guard (the trap the handoff names):** the metric is only meaningful where
**fp16 recall is high but compressed arms *can* diverge.** The VM run reports fp16 recall
per length; if fp16 itself is at floor (base model can't do the task) or every arm is at
ceiling (task too easy), the writeup **flags that length as non-discriminating** rather
than reporting a vacuous tie. This is an analysis-time honesty check written into the
plot/writeup, not a silent pass.

## Execution model (for the implementation plan)

- **Sub-agent-driven development**, controller as orchestrator/reviewer (the K3 workflow
  the handoff blesses): one fresh sub-agent per task; brief + report as FILES (not pasted
  context); model tiered to task (haiku mechanical, sonnet integration); durable ledger so
  progress survives compaction.
- **First plan task = consult the personal-brain (`personal-brain` skill / `mcp__wiki__*`)
  and DeepWiki on RULER / NIAH conventions BEFORE finalizing the needle template and recall
  scoring.** The #1 K3 process lesson: the vault knows more than our priors and changed the
  K3 design twice. Confirm: exact RULER needle/question template, recall normalization, and
  the depth-jitter convention before baking in our guesses. **Authoritative reference for
  the exact harness TurboQuant followed: the Fu et al. repo, cloned locally at
  `Long-Context-Data-Engineering/`** (NOT indexed on DeepWiki — reference by local
  Read/Grep, not DeepWiki). Key files: `eval/needle/needle_in_haystack.py` (the harness;
  needle/question template, `insert_needle` sentence-boundary logic, ROUGE-1 scorer at
  line 265, length×depth sweep), `eval/needle/PaulGrahamEssays/` (the filler corpus),
  `eval/needle/visualize.py` (the recall heatmap). (vLLM / SGLang via DeepWiki confirmed
  the dequant-on-read contract in K3 and remain the reference for cache-mechanics
  questions.)
- **Whole-branch review that RUNS the code** on the most-capable model
  (`ml-research-reviewer`) is non-negotiable at the end — per-task green reviews
  structurally missed a real bug in K3 (the V-explosion under idempotent-only test V).
- Independently re-derive headline numbers from the artifact; never take them on a
  sub-agent's report.
- Insert tasks mid-stream when a finding warrants it (own gate per task).

## Risks / explicitly deferred

- **Generation variance** — single-needle recall is spiky; the depth sweep (recall at
  multiple depths per length) plus greedy decode (`temperature=0`, deterministic — matches
  the paper) is the mitigation. ROUGE-1 being graded 0–10 already softens the spikiness vs
  a binary hit/miss. (`n_trials` jittered re-plant is available if a single depth grid
  proves too noisy, but the depth grid is the paper's own variance control.)
- **PolarQuant / SnapKV / PyramidKV baselines** — paper comparisons, *out of scope*
  (token-eviction, different mechanism, no implementation).
- **Full 104k length** — harness supports it via `--lengths`; how far we actually run is a
  VM-budget call at run time, not a harness limit. This metric **subsumes the deferred
  "32k-context re-check"** from the handoff.
- **Code-gen pass@1** — the second half of next-direction #0; deliberately a separate
  future spec, not in this one.

## Commit convention (this repo)

Conventional prefixes (`feat:`/`fix:`/`test:`/`refactor:`/`docs:`), imperative, scoped.
**NEVER any `Co-Authored-By` or AI attribution.** Pre-commit gate every time:
`uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q`. Stage and propose;
**never commit without the user's explicit approval.**
