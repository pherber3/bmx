# NIAH Retrieval Metric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A needle-in-a-haystack recall metric that runs the existing KV-compression arms through the same `StreamingQuantizedCache` path the K3 ppl sweep uses, following the TurboQuant / Fu et al. setup (single needle, length×depth sweep, ROUGE-1 recall).

**Architecture:** Two thin new modules — `src/bmx/cache/niah.py` (recall scorers: argmax proxy for CI, ROUGE-1 generate for headline) and a haystack builder (synthetic for CI, real Paul Graham essays for the VM headline) — plus a thin tyro experiment `experiments/k3_niah.py` and a figure script `experiments/plots/plot_k3_niah.py`. Everything routes arms through `_spec_pair(arm)` → `StreamingQuantizedCache`. No new codec, no new cache class.

**Tech Stack:** Python, PyTorch, transformers 5.11, tyro, pandas/parquet, `rouge-score` (new dep), pytest. Offline tiny model from `tests/factories.py`; real model + PG essays on the NVIDIA VM only.

## Global Constraints

- **NEVER `git commit` without the user's explicit approval.** Stage, propose a message, stop. No `Co-Authored-By` or any AI attribution, ever. (This plan's "Commit" steps mean *stage + propose*; the controller gets approval before each commit.)
- Pre-commit gate every time: `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` — all clean. Baseline on main: **129 passed, 1 xfailed**; this plan adds tests, so the count grows.
- Dependencies ONLY via `uv add` / `uv add --dev`. Never hand-edit versions in `pyproject.toml`.
- Use the Bash tool (git bash), `cd /d/Projects/bmx` first in fresh shells (cwd resets between turns).
- No CUDA on this machine (AMD 7900 XTX). Offline tests use the tiny factory model; real-model/long-context numbers are a VM `--model-name` run. `tiny_llama` has `max_position_embeddings=64` — offline tests MUST keep total sequence length ≤ 64.
- Comparisons align on measured bits / honest `compression` (from `memory_report`), NEVER on rank or a pinned ratio. Report each arm's natural config + measured compression; draw the paper's 4× line as reference only.
- Tiny offline test models come from `tests/factories.py` (`tiny_llama`); never download in tests. Offline experiment path injects `model` + `input_ids`/context so `run()` never calls a tokenizer or `load_eval_tokens`.
- dtype: fp32 in experiments/codecs (caches stored fp16). Fail fast: shape asserts at boundaries.

---

## Reference interfaces (read before starting — exact signatures this plan consumes)

From the existing codebase (do not redefine — import and use):

- `bmx.cache.specs.CacheCodecSpec(arm="fp16", bits=3, rank=0, group=64, seed=0, pre_rope=False)`
- `bmx.cache.streaming.StreamingQuantizedCache(model_config, k_spec, v_spec, recent_window=32)`
  with `.attach(model)`, context-manager (`with cache:`), `.bits_per_entry() -> (bpe_k, bpe_v)`,
  `.memory_report(seq_len) -> {"fp16_bytes","packed_bytes","compression"}`.
- `bmx.cache.needle._argmax_next_at(model, input_ids, query_pos, k_spec, v_spec, n_prefill) -> int`
  (re-usable helper; prefills into a fresh `StreamingQuantizedCache`, returns next-token argmax at `query_pos`).
- `bmx.cache.needle.needle_retrieved_from_ids(model, input_ids, query_pos, n_prefill, k_spec, v_spec) -> bool`
- `experiments.k3_live_generation._spec_pair(arm, cfg) -> (k_spec, v_spec)` — the one fair code path
  mapping `"fp16"|"k2b"|"turboquant_mse"|"turboquant_prod"|"kivi"` to spec pairs. **Reuse this**; do not
  re-derive specs in the new experiment.
- `bmx.artifacts.create_run(experiment, config, root="results") -> Path`,
  `write_metrics(run_dir, df, name="metrics") -> Path`.
- Figure entry-point convention: `make_figures(df, out_dir: str) -> list[Path]` (see `experiments/plots/plot_k3.py:13`).
- Offline model: `from factories import tiny_llama` (tests run with `tests/` on `sys.path`).

Local Fu et al. reference (cloned, NOT on DeepWiki — read by Read/Grep):
`Long-Context-Data-Engineering/eval/needle/needle_in_haystack.py` (harness; ROUGE-1 scorer at line 265,
`insert_needle` at ~364, prompt template at ~208), `eval/needle/PaulGrahamEssays/*.txt` (49 filler files),
`eval/needle/visualize.py` (recall heatmap).

---

## Task 0: Lock conventions against the vault + the local Fu et al. harness

**Files:**
- Create: `.git/sdd/niah-conventions.md` (decisions ledger — NOT committed to the repo tree)

**Interfaces:**
- Consumes: nothing.
- Produces: a short written record fixing the exact needle text, question text, prompt wrapper,
  ROUGE variant (`rouge1` fmeasure ×10), and depth/length sweep convention that Tasks 2–4 cite.

This is a research/decision task, not a code task — the #1 K3 process lesson is "consult the brain
and the source before architecting." No TDD cycle; the deliverable is the written ledger.

- [ ] **Step 1: Read the real harness.** Read `Long-Context-Data-Engineering/eval/needle/needle_in_haystack.py`
  in full. Extract verbatim: the default `needle`, `retrieval_question`, the `generate_prompt`
  non-API branch (the `<book>...</book>` wrapper), the `insert_needle` sentence-boundary logic, and
  the scorer line (`scorer.score(self.needle, response)['rouge1'].fmeasure*10`). Read
  `eval/needle/visualize.py` for the heatmap axes (context_length × depth_percent, score 0–10).

- [ ] **Step 2: Consult the personal-brain.** Use the `personal-brain` skill / `mcp__wiki__*` tools to
  check for vault notes on NIAH / RULER recall conventions, ROUGE for retrieval scoring, and
  long-context evaluation pitfalls (depth distribution, sentence-boundary insertion, why ROUGE-1
  over exact-match). Record anything that refines or contradicts the spec.

- [ ] **Step 3: Write the conventions ledger** to `.git/sdd/niah-conventions.md` with these fixed values
  (copy verbatim from Step 1, amend per Step 2 if the vault says otherwise):
  - `NEEDLE_TEXT`, `QUESTION_TEXT`, `PROMPT_TEMPLATE` (the `<book>` wrapper),
  - scorer = `rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)`, recall = `rouge1.fmeasure * 10`,
  - depth convention = percent 0–100, needle snapped backward to the nearest sentence-ending period,
  - length sweep convention = explicit token lengths (`final_context_length_buffer` ≈ 200 reserved for the question/answer).

- [ ] **Step 4: No commit.** This ledger lives under `.git/sdd/` (untracked). Report the fixed
  constants back to the controller; Tasks 2–4 cite them.

---

## Task 1: Add the `rouge-score` dependency and a haystack-corpus path helper

**Files:**
- Modify: `pyproject.toml` (via `uv add` only — do NOT hand-edit)
- Create: `src/bmx/cache/haystack.py`
- Test: `tests/test_haystack.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `bmx.cache.haystack.synthetic_filler(n_repeats: int) -> str` — deterministic repeated filler
    text (no files), used by the CI path.
  - `bmx.cache.haystack.pg_essays_dir() -> pathlib.Path | None` — returns the path to the local
    Paul Graham essays directory if the clone is present, else `None` (VM headline path).
  - `bmx.cache.haystack.read_pg_corpus(essays_dir: pathlib.Path) -> str` — concatenate all `*.txt`.

- [ ] **Step 1: Add the dependency.**

Run: `cd /d/Projects/bmx && uv add rouge-score`
Expected: `pyproject.toml` gains `rouge-score`; `uv.lock` updated; import works:
`uv run python -c "import rouge_score; print('ok')"` prints `ok`.

- [ ] **Step 2: Write the failing test.**

```python
# tests/test_haystack.py
from pathlib import Path

from bmx.cache.haystack import synthetic_filler, pg_essays_dir, read_pg_corpus


def test_synthetic_filler_is_deterministic_and_scales():
    a = synthetic_filler(10)
    b = synthetic_filler(10)
    assert a == b
    assert len(synthetic_filler(20)) > len(a)
    assert isinstance(a, str) and len(a) > 0


def test_pg_essays_dir_returns_path_or_none():
    d = pg_essays_dir()
    # In this repo the clone is present; if absent (CI elsewhere) None is allowed.
    assert d is None or (d.is_dir() and any(d.glob("*.txt")))


def test_read_pg_corpus_concatenates(tmp_path):
    (tmp_path / "a.txt").write_text("alpha ")
    (tmp_path / "b.txt").write_text("beta")
    corpus = read_pg_corpus(tmp_path)
    assert "alpha" in corpus and "beta" in corpus
```

- [ ] **Step 2b: Run it to verify it fails.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_haystack.py -v`
Expected: FAIL — `ModuleNotFoundError: bmx.cache.haystack`.

- [ ] **Step 3: Implement the module.**

```python
# src/bmx/cache/haystack.py
"""Haystack filler for the NIAH retrieval metric.

Two regimes (matching the run split):
  - synthetic_filler: deterministic repeated text, no files — the offline/CI path.
  - Paul Graham essays: real filler from the local Fu et al. clone — the VM headline
    path (max comparability to the TurboQuant / Fu et al. setup).
"""

from __future__ import annotations

from pathlib import Path

_FILLER_SENTENCE = "The grass was green and the sky was blue and the day was calm. "


def synthetic_filler(n_repeats: int) -> str:
    """Deterministic repeated filler (no files). Used by the offline/CI path."""
    assert n_repeats > 0, "n_repeats must be positive"
    return _FILLER_SENTENCE * n_repeats


def pg_essays_dir() -> Path | None:
    """Path to the local Paul Graham essays dir, or None if the clone is absent.

    The Fu et al. repo is cloned at the bmx repo root as a local reference (not
    vendored). Resolve relative to this file: src/bmx/cache/haystack.py -> repo root.
    """
    repo_root = Path(__file__).resolve().parents[3]
    d = repo_root / "Long-Context-Data-Engineering" / "eval" / "needle" / "PaulGrahamEssays"
    return d if d.is_dir() and any(d.glob("*.txt")) else None


def read_pg_corpus(essays_dir: Path) -> str:
    """Concatenate all *.txt files in essays_dir into one filler string."""
    parts = [p.read_text(encoding="utf-8", errors="ignore") for p in sorted(essays_dir.glob("*.txt"))]
    assert parts, f"no *.txt files in {essays_dir}"
    return "\n".join(parts)
```

- [ ] **Step 4: Run the tests to verify they pass.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_haystack.py -v`
Expected: 3 passed.

- [ ] **Step 5: Gate + commit (stage + propose).**

```bash
cd /d/Projects/bmx
uv run ruff format . && uv run ruff check . && uv run pytest -q
git add pyproject.toml uv.lock src/bmx/cache/haystack.py tests/test_haystack.py
# Propose message; get approval before committing:
#   feat: add rouge-score dep + haystack filler (synthetic + PG-essays) for NIAH
```

---

## Task 2: Argmax recall proxy (CI/mechanism gate) in `niah.py`

**Files:**
- Create: `src/bmx/cache/niah.py`
- Test: `tests/test_niah.py`

**Interfaces:**
- Consumes: `bmx.cache.needle._argmax_next_at`, `bmx.cache.specs.CacheCodecSpec`,
  `bmx.cache.streaming.StreamingQuantizedCache` (indirectly via `_argmax_next_at`).
- Produces:
  - `bmx.cache.niah.build_niah_ids_synthetic(vocab: int, n_context: int, depth_frac: float, *, answer_id: int, seed: int) -> torch.Tensor`
    — synthetic id sequence: repeated filler ids with a single distinctive `answer_id` planted at
    `depth_frac`, and the SAME `answer_id` placed as the final token's expected next-token target
    via a short "query" id pattern. Returns `(1, n_context)`. Tokenizer-free.
  - `bmx.cache.niah.niah_recall_argmax(model, input_ids, query_pos, n_prefill, k_spec, v_spec, answer_id) -> bool`
    — True iff the next-token argmax at `query_pos` (through the streaming cache) equals `answer_id`.

This is the offline mechanism gate: it proves the streaming cache indexes the planted position and
returns a finite, deterministic decision. It does NOT measure real recall quality (random weights).
Keep total length ≤ 64 (`tiny_llama` limit).

- [ ] **Step 1: Write the failing test.**

```python
# tests/test_niah.py
import torch

from bmx.cache.niah import build_niah_ids_synthetic, niah_recall_argmax
from bmx.cache.specs import CacheCodecSpec
from factories import tiny_llama


def test_build_niah_ids_shape_and_plant():
    ids = build_niah_ids_synthetic(vocab=97, n_context=40, depth_frac=0.5, answer_id=7, seed=3)
    assert ids.shape == (1, 40)
    # answer_id is planted somewhere in the interior (the needle).
    assert (ids[0] == 7).any()


def test_niah_recall_argmax_returns_bool_fp16():
    model = tiny_llama()
    ids = build_niah_ids_synthetic(vocab=97, n_context=40, depth_frac=0.5, answer_id=7, seed=3)
    fp16 = CacheCodecSpec(arm="fp16")
    got = niah_recall_argmax(
        model, ids, query_pos=39, n_prefill=20, k_spec=fp16, v_spec=fp16, answer_id=7
    )
    assert isinstance(got, bool)
```

- [ ] **Step 2: Run it to verify it fails.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_niah.py -v`
Expected: FAIL — `ModuleNotFoundError: bmx.cache.niah`.

- [ ] **Step 3: Implement the proxy.**

```python
# src/bmx/cache/niah.py
"""NIAH recall metric: argmax proxy (CI) + ROUGE-1 generate (headline).

Mirrors needle.py's proxy/real split. The argmax proxy is the offline mechanism
gate (tokenizer-free, ≤64 tokens for tiny_llama). The generate path is the headline
recall (ROUGE-1 vs the needle sentence), VM/real-model only — added in Task 3.

All arms route through the same StreamingQuantizedCache path used by the ppl sweep.
"""

from __future__ import annotations

import torch

from bmx.cache.needle import _argmax_next_at
from bmx.cache.specs import CacheCodecSpec


def build_niah_ids_synthetic(
    vocab: int,
    n_context: int,
    depth_frac: float,
    *,
    answer_id: int,
    seed: int,
) -> torch.Tensor:
    """Synthetic NIAH id sequence (tokenizer-free, for the offline mechanism gate).

    A repeated-filler id stream with a single distinctive ``answer_id`` planted at
    ``depth_frac`` (the needle). The final positions form a short query so that the
    fp16 model's next-token argmax at the end is a well-defined decision the proxy
    can compare across arms. Returns (1, n_context).
    """
    assert 0 <= depth_frac <= 1, "depth_frac in [0, 1]"
    assert n_context >= 4, "need room for filler + needle + query"
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (1, n_context), generator=g)
    # Plant the needle (answer_id) at depth.
    plant = max(1, min(n_context - 2, int(n_context * depth_frac)))
    ids[0, plant] = answer_id
    # Query tail: make the last token a marker so argmax-at-end is a stable probe.
    ids[0, -1] = answer_id  # last-seen id; mechanism probe only (not a quality claim)
    return ids


def niah_recall_argmax(
    model,
    input_ids: torch.Tensor,
    query_pos: int,
    n_prefill: int,
    k_spec: CacheCodecSpec,
    v_spec: CacheCodecSpec,
    answer_id: int,
) -> bool:
    """True iff the streaming-cache next-token argmax at query_pos equals answer_id.

    Offline mechanism gate: finite, deterministic, indexing-correct. Real recall
    quality is the ROUGE-1 generate path (Task 3), VM only.
    """
    got = _argmax_next_at(model, input_ids, query_pos, k_spec, v_spec, n_prefill)
    return bool(got == answer_id)
```

- [ ] **Step 4: Run the tests to verify they pass.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_niah.py -v`
Expected: 2 passed.

- [ ] **Step 5: Gate + commit (stage + propose).**

```bash
cd /d/Projects/bmx
uv run ruff format . && uv run ruff check . && uv run pytest -q
git add src/bmx/cache/niah.py tests/test_niah.py
#   feat: add NIAH argmax recall proxy + synthetic id builder (offline gate)
```

---

## Task 3: ROUGE-1 generate recall (headline) + real PG-essay needle builder

**Files:**
- Modify: `src/bmx/cache/niah.py`
- Test: `tests/test_niah.py` (add cases)

**Interfaces:**
- Consumes: `bmx.cache.haystack.{synthetic_filler, pg_essays_dir, read_pg_corpus}` (Task 1),
  `bmx.cache.streaming.StreamingQuantizedCache`, `rouge_score.rouge_scorer`. Constants from the
  Task 0 ledger (`NEEDLE_TEXT`, `QUESTION_TEXT`, `PROMPT_TEMPLATE`).
- Produces:
  - `bmx.cache.niah.rouge1_recall(needle_text: str, response_text: str) -> float` — ROUGE-1
    F-measure ×10 (0–10), `use_stemmer=True`. Pure function, tokenizer-free, CI-testable.
  - `bmx.cache.niah.build_niah_prompt(tokenizer, context_length: int, depth_percent: float, *, haystack: str, needle_text: str, question_text: str, buffer: int = 200) -> torch.Tensor`
    — RULER-style prompt ids: filler trimmed to length, needle inserted at the nearest sentence
    boundary at `depth_percent`, wrapped in the `<book>` template + question. Real-tokenizer path
    (VM only; not unit-tested directly).
  - `bmx.cache.niah.niah_recall_generate(model, tokenizer, prompt_ids, n_prefill, k_spec, v_spec, needle_text, max_new_tokens=50) -> float`
    — prefill into `StreamingQuantizedCache`, greedy-generate the answer, return `rouge1_recall`.

The `rouge1_recall` function is the headline scorer and IS unit-testable (no model). The prompt
builder and generate path require a tokenizer/real model → exercised on the VM, guarded out of CI.

- [ ] **Step 1: Write the failing test (the scorer is the CI-testable surface).**

```python
# add to tests/test_niah.py
from bmx.cache.niah import rouge1_recall


def test_rouge1_recall_perfect_and_zero():
    needle = "The best thing to do in San Francisco is eat a sandwich in Dolores Park."
    assert rouge1_recall(needle, needle) == 10.0  # identical => fmeasure 1.0 * 10
    assert rouge1_recall(needle, "completely unrelated zzz qqq") < 2.0  # near-zero overlap


def test_rouge1_recall_partial_is_graded():
    needle = "the magic number is one two three four"
    partial = "the magic number is"
    score = rouge1_recall(needle, partial)
    assert 0.0 < score < 10.0  # graded, not binary
```

- [ ] **Step 2: Run it to verify it fails.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_niah.py::test_rouge1_recall_perfect_and_zero -v`
Expected: FAIL — `ImportError: cannot import name 'rouge1_recall'`.

- [ ] **Step 3: Implement the scorer, prompt builder, and generate path.**

```python
# append to src/bmx/cache/niah.py

from rouge_score import rouge_scorer

from bmx.cache.streaming import StreamingQuantizedCache

# Defaults follow the Fu et al. harness (eval/needle/needle_in_haystack.py); the
# Task 0 ledger is the source of truth if the vault refined these.
NEEDLE_TEXT = (
    "\nThe best thing to do in San Francisco is eat a sandwich and sit in "
    "Dolores Park on a sunny day.\n"
)
QUESTION_TEXT = "What is the best thing to do in San Francisco?"
PROMPT_TEMPLATE = (
    "This is a very long story book: <book> {context} </book>.\n"
    "Based on the content of the book, Question: {question}\nAnswer:"
)

_SCORER = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def rouge1_recall(needle_text: str, response_text: str) -> float:
    """ROUGE-1 F-measure ×10 (0–10) of the needle vs the response — the headline scorer.

    Matches the Fu et al. metric (needle_in_haystack.py:265). Pure function; the
    streaming-cache generate path feeds the model response in.
    """
    return _SCORER.score(needle_text, response_text)["rouge1"].fmeasure * 10.0


def _insert_needle_at_sentence_boundary(
    tokenizer, context_ids: list[int], needle_ids: list[int], depth_percent: float
) -> list[int]:
    """Insert needle_ids into context_ids at depth_percent, snapped back to a period."""
    if depth_percent >= 100:
        return context_ids + needle_ids
    insertion = int(len(context_ids) * (depth_percent / 100.0))
    period_ids = tokenizer.encode(".", add_special_tokens=False)
    head = context_ids[:insertion]
    while head and head[-1] not in period_ids:
        insertion -= 1
        head = context_ids[:insertion]
    return head + needle_ids + context_ids[insertion:]


def build_niah_prompt(
    tokenizer,
    context_length: int,
    depth_percent: float,
    *,
    haystack: str,
    needle_text: str = NEEDLE_TEXT,
    question_text: str = QUESTION_TEXT,
    buffer: int = 200,
) -> torch.Tensor:
    """RULER-style NIAH prompt ids (real-tokenizer path; VM only).

    Trims ``haystack`` to ``context_length - buffer`` tokens, inserts ``needle_text``
    at ``depth_percent`` snapped to a sentence boundary, wraps in PROMPT_TEMPLATE +
    question. Returns (1, L).
    """
    needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)
    ctx_ids = tokenizer.encode(haystack, add_special_tokens=False)
    budget = context_length - buffer
    if len(ctx_ids) + len(needle_ids) > budget:
        ctx_ids = ctx_ids[: budget - len(needle_ids)]
    woven = _insert_needle_at_sentence_boundary(tokenizer, ctx_ids, needle_ids, depth_percent)
    context = tokenizer.decode(woven)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question_text)
    return tokenizer(prompt, return_tensors="pt").input_ids


def niah_recall_generate(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    n_prefill: int,
    k_spec: CacheCodecSpec,
    v_spec: CacheCodecSpec,
    needle_text: str = NEEDLE_TEXT,
    max_new_tokens: int = 50,
) -> float:
    """Prefill the prompt into the streaming cache, greedy-generate, score ROUGE-1.

    Headline recall (VM/real model). n_prefill tokens are quantized on-append; the
    remaining prompt + generation attend to the compressed cache.
    """
    cache = StreamingQuantizedCache(model.config, k_spec=k_spec, v_spec=v_spec)
    cache.attach(model)
    with cache:
        with torch.no_grad():
            # Prefill the leading n_prefill tokens into the streaming cache, then let
            # generate() consume the rest of the prompt + decode the answer.
            model(prompt_ids[:, :n_prefill], past_key_values=cache, use_cache=True)
            out = model.generate(
                prompt_ids,
                past_key_values=cache,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
    response = tokenizer.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True).strip()
    return rouge1_recall(needle_text, response)
```

- [ ] **Step 4: Run the scorer tests to verify they pass.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_niah.py -v`
Expected: all passed (the 2 from Task 2 + the 2 new scorer tests).

- [ ] **Step 5: Gate + commit (stage + propose).**

```bash
cd /d/Projects/bmx
uv run ruff format . && uv run ruff check . && uv run pytest -q
git add src/bmx/cache/niah.py tests/test_niah.py
#   feat: add ROUGE-1 generate recall + PG-essay needle prompt builder (NIAH headline)
```

---

## Task 4: The `k3_niah` experiment (sweep arms × lengths × depths, emit parquet)

**Files:**
- Create: `experiments/k3_niah.py`
- Test: `tests/test_k3_niah_experiment.py`

**Interfaces:**
- Consumes: `experiments.k3_live_generation._spec_pair`, `bmx.cache.niah.*`,
  `bmx.cache.haystack.*`, `bmx.artifacts.{create_run, write_metrics}`,
  `bmx.cache.streaming.StreamingQuantizedCache` (for `memory_report`).
- Produces: `experiments.k3_niah.{Config, run}`. `run(cfg, model=None, root="results")` writes
  `metrics.parquet` with columns: `arm, length, depth, recall, bpe_k, bpe_v, compression, n_prefill`.

Mirror `k3_live_generation.run` exactly: real path (model is None) loads model + tokenizer + PG
essays and scores ROUGE-1; offline path (model injected) uses the synthetic argmax proxy and
small lengths, emits the same schema with `recall` ∈ {0.0, 10.0} (argmax hit ×10 for schema parity).
Airtight lazy-import guard so CI never downloads.

- [ ] **Step 1: Write the failing test (offline parquet-schema mechanics).**

```python
# tests/test_k3_niah_experiment.py
"""k3_niah emits a parquet with the expected schema (tiny_llama, offline, no download)."""

import pandas as pd

from experiments.k3_niah import Config, run
from factories import tiny_llama


def test_k3_niah_run_emits_parquet(tmp_path):
    model = tiny_llama()
    # tiny_llama max_position_embeddings=64 → keep lengths small; group=16 divisibility.
    cfg = Config(
        arms=("fp16", "kivi"),
        lengths=(32, 48),
        depths=(0.25, 0.5),
        n_prefill=16,
        group=16,
        rank=4,
    )
    run_dir = run(cfg, model=model, root=str(tmp_path))
    df = pd.read_parquet(run_dir / "metrics.parquet")
    for col in ("arm", "length", "depth", "recall", "bpe_k", "bpe_v", "compression", "n_prefill"):
        assert col in df.columns, f"missing column: {col}"
    # 2 arms × 2 lengths × 2 depths = 8 rows.
    assert len(df) == 8
    assert set(df["arm"]) <= {"fp16", "k2b", "kivi", "turboquant_mse", "turboquant_prod"}
```

- [ ] **Step 2: Run it to verify it fails.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_k3_niah_experiment.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.k3_niah`.

- [ ] **Step 3: Implement the experiment.**

```python
# experiments/k3_niah.py
"""K3-NIAH — long-context needle-in-a-haystack recall under KV compression.

Sweeps arms × document-lengths × depths on ONE code path (StreamingQuantizedCache),
following the TurboQuant / Fu et al. setup: single needle, ROUGE-1 recall, length
sweep. Reports each arm's honest measured compression (never a pinned ratio).

Real path (model is None, VM run):
  Loads model + tokenizer + Paul Graham essays, builds RULER-style needle prompts,
  greedy-generates, scores ROUGE-1 (niah_recall_generate).

Offline-test path (model injected):
  Synthetic argmax proxy on small lengths (≤64, tiny_llama). recall is the argmax
  hit ×10 for schema parity. No download, no tokenizer.

Model-agnostic: the SOTA VM run is a --model-name change. Figures: plots/plot_k3_niah.py.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.niah import (
    build_niah_ids_synthetic,
    niah_recall_argmax,
)
from bmx.cache.streaming import StreamingQuantizedCache
from experiments.k3_live_generation import _spec_pair


@dataclasses.dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    arms: tuple[str, ...] = ("fp16", "k2b", "turboquant_mse", "turboquant_prod", "kivi")
    lengths: tuple[int, ...] = (4096, 8192, 16384, 32768)
    depths: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    n_prefill: int = 128
    rank: int = 16
    group: int = 64
    seed: int = 0
    answer_id: int = 7
    """Synthetic needle id for the offline argmax proxy (ignored on the real path)."""
    max_new_tokens: int = 50


def _compression_for(model_config, k_spec, v_spec, length: int) -> tuple[float, float, float]:
    """Honest (bpe_k, bpe_v, compression) for an arm at a given length.

    Builds a StreamingQuantizedCache only to read its memory_report accounting; this
    matches the deployable blended-bpe number used by the ppl sweep.
    """
    cache = StreamingQuantizedCache(model_config, k_spec=k_spec, v_spec=v_spec)
    # bits_per_entry reads the codec spec deterministically; memory_report is exact at length.
    mem = cache.memory_report(seq_len=length)
    bpe_k, bpe_v = cache.bits_per_entry()
    return bpe_k, bpe_v, mem["compression"]


def run(cfg: Config, model=None, root: str = "results"):
    tokenizer = None
    haystack = None
    if model is None:
        # Real run (VM): model + tokenizer + Paul Graham essays.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from bmx.cache.haystack import pg_essays_dir, read_pg_corpus
        from bmx.cache.niah import build_niah_prompt, niah_recall_generate

        import torch

        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.float16)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        essays = pg_essays_dir()
        assert essays is not None, "Paul Graham essays not found; clone Fu et al. repo at repo root"
        haystack = read_pg_corpus(essays)

    run_dir = create_run("k3_niah", cfg, root=root)
    rows = []
    for arm in cfg.arms:
        k_spec, v_spec = _spec_pair(arm, cfg)
        for length in cfg.lengths:
            bpe_k, bpe_v, compression = _compression_for(model.config, k_spec, v_spec, length)
            for depth in cfg.depths:
                if tokenizer is None:
                    # Offline: synthetic argmax proxy at this (small) length.
                    ids = build_niah_ids_synthetic(
                        model.config.vocab_size, length, depth, answer_id=cfg.answer_id, seed=cfg.seed
                    )
                    hit = niah_recall_argmax(
                        model, ids, query_pos=length - 1, n_prefill=cfg.n_prefill,
                        k_spec=k_spec, v_spec=v_spec, answer_id=cfg.answer_id,
                    )
                    recall = 10.0 if hit else 0.0
                else:
                    # Real: ROUGE-1 generate recall.
                    prompt_ids = build_niah_prompt(
                        tokenizer, context_length=length, depth_percent=depth * 100.0,
                        haystack=haystack,
                    )
                    recall = niah_recall_generate(
                        model, tokenizer, prompt_ids, cfg.n_prefill, k_spec, v_spec,
                        max_new_tokens=cfg.max_new_tokens,
                    )
                rows.append({
                    "arm": arm, "length": length, "depth": depth, "recall": recall,
                    "bpe_k": bpe_k, "bpe_v": bpe_v, "compression": compression,
                    "n_prefill": cfg.n_prefill,
                })

    write_metrics(run_dir, pd.DataFrame(rows))
    return run_dir


if __name__ == "__main__":
    run(tyro.cli(Config))
```

- [ ] **Step 4: Run the test to verify it passes.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_k3_niah_experiment.py -v`
Expected: 1 passed (8 rows, schema correct, no download).

- [ ] **Step 5: Gate + commit (stage + propose).**

```bash
cd /d/Projects/bmx
uv run ruff format . && uv run ruff check . && uv run pytest -q
git add experiments/k3_niah.py tests/test_k3_niah_experiment.py
#   feat: add k3_niah experiment — NIAH recall sweep on the one fair cache path
```

---

## Task 5: Figures — recall vs length + length×depth heatmap

**Files:**
- Create: `experiments/plots/plot_k3_niah.py`
- Test: `tests/test_k3_niah_experiment.py` (add a plot test)

**Interfaces:**
- Consumes: a metrics DataFrame with `arm, length, depth, recall, compression`.
- Produces: `experiments.plots.plot_k3_niah.make_figures(df, out_dir: str) -> list[Path]` —
  writes (1) recall-vs-length per arm (annotated with each arm's compression, 4× reference line)
  and (2) a length×depth recall heatmap per arm. Returns the PNG paths.

- [ ] **Step 1: Write the failing test.**

```python
# add to tests/test_k3_niah_experiment.py
def test_plot_k3_niah_makes_pngs(tmp_path):
    import pandas as pd
    from experiments.plots.plot_k3_niah import make_figures

    df = pd.DataFrame([
        {"arm": "fp16", "length": 4096, "depth": 0.5, "recall": 10.0, "compression": 1.0},
        {"arm": "fp16", "length": 8192, "depth": 0.5, "recall": 9.0, "compression": 1.0},
        {"arm": "kivi", "length": 4096, "depth": 0.5, "recall": 8.0, "compression": 4.1},
        {"arm": "kivi", "length": 8192, "depth": 0.5, "recall": 6.0, "compression": 4.1},
    ])
    paths = make_figures(df, str(tmp_path))
    assert len(paths) >= 1
    assert all(p.exists() for p in paths)
```

- [ ] **Step 2: Run it to verify it fails.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_k3_niah_experiment.py::test_plot_k3_niah_makes_pngs -v`
Expected: FAIL — `ModuleNotFoundError: experiments.plots.plot_k3_niah`.

- [ ] **Step 3: Implement the figures.**

```python
# experiments/plots/plot_k3_niah.py
"""Figures for k3_niah: recall vs length per arm, and length×depth recall heatmaps.

Reads the parquet, never refits. Select runs explicitly upstream (newest_run_with);
this module only renders a passed-in DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def make_figures(df, out_dir: str) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # --- Figure 1: recall vs length, one line per arm, annotated with compression. ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for arm, g in df.groupby("arm"):
        gl = g.groupby("length")["recall"].mean().sort_index()
        comp = g["compression"].iloc[0]
        ax.plot(gl.index, gl.values, marker="o", label=f"{arm} ({comp:.1f}×)")
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("recall (ROUGE-1 ×10, mean over depth)")
    ax.set_title("NIAH recall vs length under KV compression")
    ax.legend()
    p1 = out / "niah_recall_vs_length.png"
    fig.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p1)

    # --- Figure 2: length×depth recall heatmap per arm (paper's view). ---
    if "depth" in df.columns and df["depth"].nunique() > 1:
        arms = sorted(df["arm"].unique())
        fig, axes = plt.subplots(1, len(arms), figsize=(5 * len(arms), 4), squeeze=False)
        lengths = sorted(df["length"].unique())
        depths = sorted(df["depth"].unique())
        for ax, arm in zip(axes[0], arms):
            g = df[df["arm"] == arm]
            grid = np.full((len(depths), len(lengths)), np.nan)
            for _, r in g.iterrows():
                grid[depths.index(r["depth"]), lengths.index(r["length"])] = r["recall"]
            im = ax.imshow(grid, aspect="auto", vmin=0, vmax=10, origin="lower", cmap="viridis")
            ax.set_xticks(range(len(lengths)), [str(x) for x in lengths])
            ax.set_yticks(range(len(depths)), [f"{d:.0%}" for d in depths])
            ax.set_xlabel("length")
            ax.set_ylabel("depth")
            ax.set_title(arm)
        fig.colorbar(im, ax=axes[0].tolist(), label="recall (0–10)")
        p2 = out / "niah_recall_heatmap.png"
        fig.savefig(p2, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(p2)

    return paths
```

- [ ] **Step 4: Run the test to verify it passes.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_k3_niah_experiment.py -v`
Expected: 2 passed (schema test + plot test).

- [ ] **Step 5: Gate + commit (stage + propose).**

```bash
cd /d/Projects/bmx
uv run ruff format . && uv run ruff check . && uv run pytest -q
git add experiments/plots/plot_k3_niah.py tests/test_k3_niah_experiment.py
#   feat: add k3_niah figures (recall vs length + length×depth heatmap)
```

---

## Task 6: Gitignore the local clone + record research state

**Files:**
- Modify: `.gitignore`
- Modify: `CLAUDE.md` (research-state line)
- Create: `docs/2026-06-20-niah-plan-state.md` (handoff note for the VM run)

**Interfaces:**
- Consumes: nothing.
- Produces: a clean tree (the Fu et al. clone ignored) and a recorded next-step (the VM run).

- [ ] **Step 1: Ignore the local clone.** Add to `.gitignore`:

```
# Local reference clone (Fu et al. NIAH harness) — not vendored
/Long-Context-Data-Engineering/
```

Run: `cd /d/Projects/bmx && git status --short` → `Long-Context-Data-Engineering/` no longer listed.

- [ ] **Step 2: Record research state.** Add one line under the KV-program section of `CLAUDE.md`:

```
- NIAH retrieval metric (task metric #0, retrieval half): ROUGE-1 recall under
  compression on the StreamingQuantizedCache path; offline argmax gate + VM PG-essay
  headline (`experiments/k3_niah.py --model-name`). Spec/plan: docs/superpowers/{specs,plans}/2026-06-20-niah-*.
```

- [ ] **Step 3: Write the VM handoff note** `docs/2026-06-20-niah-plan-state.md`: the exact VM
  command (`uv run python experiments/k3_niah.py --model-name meta-llama/Llama-3.1-8B-Instruct`),
  the headroom guard to apply at analysis time (flag lengths where fp16 recall is at floor or all
  arms at ceiling as non-discriminating), and that code-gen pass@1 remains the deferred second half.

- [ ] **Step 4: Gate + commit (stage + propose).**

```bash
cd /d/Projects/bmx
uv run ruff format . && uv run ruff check . && uv run pytest -q
git add .gitignore CLAUDE.md docs/2026-06-20-niah-plan-state.md docs/superpowers/specs/2026-06-20-niah-retrieval-metric-design.md docs/superpowers/plans/2026-06-20-niah-retrieval-metric.md
#   docs: NIAH retrieval metric spec+plan, research-state, ignore local clone
```

---

## Whole-branch review (non-negotiable — RUNS the code)

After all tasks: dispatch the `ml-research-reviewer` agent on the **full diff**, with the explicit
mandate to RUN the code (not just read it), per the #1 K3 review lesson. Specific probes to demand:

- **Fairness:** confirm every arm routes through `_spec_pair` → `StreamingQuantizedCache` — no arm
  has a private code path. (The K3 V-explosion bug hid because a test used an idempotent-only codec.)
- **Headroom honesty:** verify the offline argmax proxy is mechanism-only and the experiment does
  NOT claim quality on random weights; confirm `recall` semantics differ correctly between the
  offline (argmax ×10) and real (ROUGE-1) paths and that the parquet labels which is which.
- **No-download guarantee:** force-run `uv run pytest -q` with no network and confirm CI never
  hits the tokenizer/model/essay-download paths (lazy imports inside `if model is None`).
- **Compression accounting:** independently re-derive one arm's `compression` from `bits_per_entry`
  and `memory_report`, don't take it on report.
- **Generate-path correctness:** probe that `niah_recall_generate` actually attends to the compressed
  cache (e.g. inject a wrong RoPE/position and confirm recall drops — the "inject a bug to prove the
  test" standard from K3).

Record findings in `.git/sdd/niah-progress.md`; insert fix-tasks with their own gates if any Critical
or Important surfaces. Do NOT merge or push without explicit user approval.
