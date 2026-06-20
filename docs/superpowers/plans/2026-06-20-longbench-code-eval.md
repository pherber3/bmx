# LongBench Code Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce TurboQuant's coding signal (LongBench Table 1 `Code` column) by running `lcc` + `repobench-p` end-to-end through the same `StreamingQuantizedCache` path as NIAH, scored by LongBench's official `code_sim` edit-similarity.

**Architecture:** Extract the generic prefill-then-generate machinery out of `niah_recall_generate` into a shared `generate_through_cache` helper (NIAH and LongBench both call it → one fair path). Add a LongBench-specific module (`code_sim` scorer, dataset loader, task registry) and a thin tyro experiment + plot that mirror the NIAH ones exactly.

**Tech Stack:** Python, PyTorch, transformers 5.11, tyro, pandas/parquet, HF `datasets` 5.0 (LongBench), `fuzzywuzzy`+`python-Levenshtein` (LongBench's official scorer; `rapidfuzz` fallback only if the wheel won't install), pytest. Offline tiny model from `tests/factories.py`; real model + LongBench on the NVIDIA VM only.

## Global Constraints

- **NEVER `git commit` without the user's explicit approval.** Per-task auto-commit is PRE-AUTHORIZED for this plan (NIAH precedent): conventional prefix, imperative, scoped. **NO `Co-Authored-By` or ANY AI attribution, ever** — checked.
- Pre-commit gate every time, in order: `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` — all clean. Baseline at plan start: **164 passed, 1 xfailed** (the xfail `test_cold_start_recovery` is intentional — leave it). This plan adds tests, so the count grows.
- Dependencies ONLY via `uv add` / `uv add --dev`. Never hand-edit `pyproject.toml` versions.
- Use the Bash tool (git bash), NOT PowerShell. cwd resets between turns — `cd /d/Projects/bmx` first in fresh shells.
- No CUDA on this machine (AMD 7900 XTX). Offline tests use `tiny_llama` (max_position_embeddings=64 → offline sequences ≤ 64). Real-model/LongBench numbers are a VM `--model-name` run.
- Comparisons align on measured bits / honest `compression` (from `memory_report`), NEVER a pinned ratio. The fp16 / 4× score is a reference line only.
- Offline experiment path injects `model` (+ a synthetic prompt) so `run()` never calls a tokenizer, `load_dataset`, or downloads. All such imports live INSIDE `if model is None:` (the airtight lazy-import guard, mirrored from `experiments/k3_niah.py`).
- Faithful to LongBench: the scorer, prompt templates, and `max_gen` come VERBATIM from the authoritative LongBench source (Task 0 locks them), not from guesses.

---

## Reference interfaces (read before starting — exact signatures this plan consumes)

From the existing codebase (import and use; do not redefine):

- `bmx.cache.specs.CacheCodecSpec(arm="fp16", bits=3, rank=0, group=64, seed=0, pre_rope=False)`
- `bmx.cache.streaming.StreamingQuantizedCache(model_config, k_spec, v_spec, recent_window=32)`
  with `.attach(model)`, context-manager, `.bits_per_entry()`, `.memory_report(seq_len)`.
- `bmx.cache.niah.rouge1_recall`, `bmx.cache.niah.niah_recall_generate` — the function being
  refactored (Task 2). Current body (the part to extract) is `niah.py:164-181`.
- `experiments.k3_live_generation._spec_pair(arm, cfg) -> (k_spec, v_spec)` — the one fair
  code path mapping `"fp16"|"k2b"|"turboquant_mse"|"turboquant_prod"|"kivi"`. Reuse it.
- `experiments.k3_niah._compression_for(model, k_spec, v_spec, length) -> (bpe_k, bpe_v, compression)`
  — honest measured compression via a calibration prefill. Reuse it (Task 4).
- `bmx.artifacts.create_run(experiment, config, root="results")`, `write_metrics(run_dir, df, name="metrics")`.
- Figure entry-point convention: `make_figures(df, out_dir: str) -> list[Path]` (see `experiments/plots/plot_k3_niah.py`).
- Lazy `load_dataset` precedent: `src/bmx/eval/layer_swap.py:49` (`from datasets import load_dataset`).
- Offline model: `from factories import tiny_llama` (tests run with `tests/` on `sys.path`).

Authoritative LongBench source — **cloned locally** at `LongBench/` (gitignored, like the Fu
et al. NIAH clone). The v1 files live under the nested `LongBench/LongBench/` subdir (the repo
root also holds a v2). Task 0 reads these directly:
- `LongBench/LongBench/metrics.py:80` → `code_sim_score(prediction, ground_truth)` — VERIFIED:
  takes the first prediction line with no `` ` ``/`#`/`//`, then `fuzz.ratio(...) / 100` (range **0–1**).
- `LongBench/LongBench/config/dataset2prompt.json` → VERIFIED templates: `lcc` =
  `"Please complete the code given below. \n{context}Next line of code:\n"`, `repobench-p` =
  `"Please complete the code given below. \n{context}{input}Next line of code:\n"`. (Read with
  `encoding="utf-8"` — the file has non-cp1252 bytes.)
- `LongBench/LongBench/config/dataset2maxlen.json` → VERIFIED: `lcc` = 64, `repobench-p` = 64.
- `LongBench/pred.py` → `build_chat(...)` (the Llama-Instruct prompt wrapper) and `post_process`.

---

## Task 0: Lock LongBench conventions against the authoritative source

**Files:**
- Create: `.git/sdd/longbench-conventions.md` (decisions ledger — NOT committed to the repo tree)

**Interfaces:**
- Consumes: nothing.
- Produces: a written record fixing, VERBATIM: the `lcc` + `repobench-p` prompt templates,
  their `max_gen` values, the exact `code_sim` post-processing + ratio call, the dataset field
  names (`input`/`context`/`answers`/`all_classes`/`length`), and the Llama-Instruct
  `build_chat` wrapper. Tasks 3–5 cite this.

Research/decision task — no TDD cycle; the deliverable is the ledger.

- [ ] **Step 1: Read the LongBench scorer.** Read `LongBench/LongBench/metrics.py` (the local
  clone). Record VERBATIM `code_sim_score` (lines ~80-87): the post-process (first line with no
  `` ` ``/`#`/`//`) and the `fuzz.ratio(...) / 100` normalization (range **0–1**, NOT 0–100).

- [ ] **Step 2: Read the prompt templates + gen lengths.** From
  `LongBench/LongBench/config/dataset2prompt.json` (read with `encoding="utf-8"`) record the
  exact `lcc` and `repobench-p` template strings (preserve the trailing space and note
  `repobench-p`'s `{input}`). From `dataset2maxlen.json` record their `max_gen` (both 64).

- [ ] **Step 3: Read the model-prompt + post-process wrapper.** From `LongBench/pred.py` record
  `build_chat(tokenizer, prompt, model_name)` for Llama-3.x-Instruct (whether it applies the
  chat template / a manual `[INST]` wrapper), and the `post_process` step LongBench applies to
  the raw generation for code tasks (e.g. taking the first line) before scoring.

- [ ] **Step 4: Consult the personal-brain (if connected).** If `mcp__wiki__*` is available,
  check the vault for notes on LongBench / edit-similarity / long-context code eval; record
  anything that refines the above. If not connected, note "vault not reachable" (as in NIAH).

- [ ] **Step 5: Write the ledger** to `.git/sdd/longbench-conventions.md` with every verbatim
  constant from Steps 1–3 and the dataset field names. This is the source of truth Tasks 3–5
  cite. No commit (lives under `.git/sdd/`).

---

## Task 1: Add LongBench dependencies (`fuzzywuzzy` + `python-Levenshtein`)

**Files:**
- Modify: `pyproject.toml` (via `uv add` only)
- Test: `tests/test_longbench_deps.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an importable `fuzzywuzzy.fuzz` (LongBench's exact scorer dependency). `datasets`
  is already installed (5.0.0) — no add needed for it.

- [ ] **Step 1: Add the scorer dependency.**

Run: `cd /d/Projects/bmx && uv add fuzzywuzzy python-Levenshtein`
Expected: both resolve and install; `uv run python -c "from fuzzywuzzy import fuzz; print(fuzz.ratio('abc','abd'))"` prints an integer (67).
**If `python-Levenshtein` fails to build a wheel on this box:** STOP and report BLOCKED to the
controller — the spec says `rapidfuzz` is the documented fallback (`uv add rapidfuzz`), but the
substitution must be recorded by the controller, not chosen silently. Do not proceed past this
step without controller direction.

- [ ] **Step 2: Write the failing test.**

```python
# tests/test_longbench_deps.py
def test_fuzzywuzzy_importable_and_ratio_works():
    from fuzzywuzzy import fuzz

    assert fuzz.ratio("hello world", "hello world") == 100
    assert fuzz.ratio("abc", "xyz") < 50
```

- [ ] **Step 2b: Run it to verify it passes (the add already happened).**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_longbench_deps.py -v`
Expected: PASS (the dependency is installed). If it fails on import, the add did not take — re-run Step 1.

- [ ] **Step 3: Gate + commit (stage + propose).**

```bash
cd /d/Projects/bmx
uv run ruff format . && uv run ruff check . && uv run pytest -q
git add pyproject.toml uv.lock tests/test_longbench_deps.py
#   feat: add fuzzywuzzy + python-Levenshtein (LongBench code_sim scorer dep)
```

---

## Task 2: Extract `generate_through_cache` (shared NIAH + LongBench path)

**Files:**
- Modify: `src/bmx/cache/niah.py` (refactor `niah_recall_generate`, add `generate_through_cache`)
- Test: `tests/test_niah.py` (add one test for the new helper; existing tests must still pass)

**Interfaces:**
- Consumes: `bmx.cache.streaming.StreamingQuantizedCache`, `bmx.cache.specs.CacheCodecSpec`.
- Produces:
  - `bmx.cache.niah.generate_through_cache(model, tokenizer, prompt_ids, n_prefill, k_spec, v_spec, max_new_tokens) -> str`
    — prefill into the streaming cache, greedy-generate the continuation, return the decoded
    NEW tokens (the model's answer string). The continuation-only contract + decode offset
    `out[0, L-n_prefill:]` live here, in one place.
  - `niah_recall_generate` unchanged in signature/behavior, now implemented as
    `generate_through_cache(...)` + `rouge1_recall(needle_text, response)`.

This is a behavior-preserving refactor: `niah_recall_generate` must return the SAME value as
before for the same inputs. The existing NIAH tests are the regression guard.

- [ ] **Step 1: Write the failing test for the new helper.**

```python
# add to tests/test_niah.py
from bmx.cache.niah import generate_through_cache


def test_generate_through_cache_returns_str(tmp_path):
    import torch
    from bmx.cache.specs import CacheCodecSpec
    from factories import tiny_llama

    model = tiny_llama()
    g = torch.Generator().manual_seed(0)
    prompt_ids = torch.randint(0, 97, (1, 24), generator=g)
    fp16 = CacheCodecSpec(arm="fp16")
    out = generate_through_cache(
        model, tokenizer=None, prompt_ids=prompt_ids, n_prefill=12,
        k_spec=fp16, v_spec=fp16, max_new_tokens=4,
    )
    # tokenizer=None path: return the raw new-token ids decoded via the model is not possible,
    # so generate_through_cache must accept a tokenizer; for the offline mechanism test we pass
    # a trivial decode. See Step 3 — the helper requires a tokenizer; this test uses a stub.
    assert isinstance(out, str)
```

NOTE to implementer: `generate_through_cache` decodes with `tokenizer.decode`. For the offline
mechanism test, pass a minimal stub tokenizer exposing `.decode(ids, skip_special_tokens=True)`
returning `" ".join(map(str, ids.tolist()))`. Define the stub inside the test. Adjust the test
to construct and pass that stub rather than `tokenizer=None`.

- [ ] **Step 2: Run it to verify it fails.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_niah.py::test_generate_through_cache_returns_str -v`
Expected: FAIL — `ImportError: cannot import name 'generate_through_cache'`.

- [ ] **Step 3: Implement the extraction.** Replace the body of `niah_recall_generate`
  (`niah.py:164-182`) by extracting the prefill+generate+decode into the new helper:

```python
def generate_through_cache(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    n_prefill: int,
    k_spec: CacheCodecSpec,
    v_spec: CacheCodecSpec,
    max_new_tokens: int,
) -> str:
    """Prefill prompt_ids into a StreamingQuantizedCache, greedy-generate, return the answer.

    The shared headline generate path for every metric that scores a generation through the
    compressed cache (NIAH ROUGE-1, LongBench code_sim, ...). One fair code path.

    Continuation-only contract (mirrors needle.py:21-22):
      Step 1 fills cache positions [0, n_prefill) via a quantize-on-append forward.
      Step 2 feeds ONLY prompt_ids[:, n_prefill:] to generate(); HF returns the supplied
      continuation followed by the newly-decoded tokens, so the new tokens start at index
      (L - n_prefill) in out[0]. Decoding out[0, L - n_prefill :] yields exactly the answer.
    """
    L = prompt_ids.shape[1]
    cont_len = L - n_prefill
    cache = StreamingQuantizedCache(model.config, k_spec=k_spec, v_spec=v_spec)
    cache.attach(model)
    with cache:
        with torch.no_grad():
            model(prompt_ids[:, :n_prefill], past_key_values=cache, use_cache=True)
            out = model.generate(
                prompt_ids[:, n_prefill:],
                past_key_values=cache,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
    return tokenizer.decode(out[0, cont_len:], skip_special_tokens=True).strip()


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

    Headline recall (VM/real model). Now a thin wrapper over generate_through_cache so the
    generate path lives in one place across metrics.
    """
    response = generate_through_cache(
        model, tokenizer, prompt_ids, n_prefill, k_spec, v_spec, max_new_tokens
    )
    return rouge1_recall(needle_text, response)
```

- [ ] **Step 4: Run the niah tests to verify the refactor is behavior-preserving.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_niah.py -v`
Expected: all pass (the new helper test + all prior niah tests unchanged-green).

- [ ] **Step 5: Gate + commit (stage + propose).**

```bash
cd /d/Projects/bmx
uv run ruff format . && uv run ruff check . && uv run pytest -q
git add src/bmx/cache/niah.py tests/test_niah.py
#   refactor: extract generate_through_cache (shared NIAH+LongBench generate path)
```

---

## Task 3: LongBench `code_sim` scorer

**Files:**
- Create: `src/bmx/cache/longbench.py`
- Test: `tests/test_longbench.py`

**Interfaces:**
- Consumes: `fuzzywuzzy.fuzz` (Task 1). Constants from the Task 0 ledger (the post-processing rule).
- Produces:
  - `bmx.cache.longbench.code_sim(prediction: str, ground_truth: str) -> float` — LongBench's
    official code edit-similarity, **range 0–1** (LongBench's `code_sim_score` returns
    `fuzz.ratio(...) / 100`). Post-processing per the ledger, then `fuzz.ratio`.

`code_sim` is the CI-testable surface (pure function, no model). Port it FAITHFULLY from
LongBench's `metrics.py::code_sim_score` (verified verbatim in the Task 0 ledger / the local
clone `LongBench/LongBench/metrics.py:80-87`). Do NOT change the normalization — it returns
0–1, not 0–100.

- [ ] **Step 1: Write the failing test.** (Note the **0–1** range — LongBench divides by 100.)

```python
# tests/test_longbench.py
from bmx.cache.longbench import code_sim


def test_code_sim_identical_is_one():
    # Single clean line (the post-process keeps the first non-comment line); identical => 1.0.
    line = "    return a + b"
    assert code_sim(line, line) == 1.0


def test_code_sim_disjoint_is_low():
    assert code_sim("    return a + b", "xxxxx yyyyy zzzzz") < 0.3


def test_code_sim_partial_is_graded():
    gt = "    return a + b"
    pred = "    return a - b"  # one char off
    s = code_sim(pred, gt)
    assert 0.0 < s < 1.0


def test_code_sim_strips_comment_lines():
    # The post-process skips lines containing `, #, or // and scores the first clean line.
    pred = "# a comment\n    return a + b"
    assert code_sim(pred, "    return a + b") == 1.0
```

- [ ] **Step 2: Run it to verify it fails.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_longbench.py -v`
Expected: FAIL — `ModuleNotFoundError: bmx.cache.longbench`.

- [ ] **Step 3: Implement `code_sim`** — a verbatim port of LongBench's `code_sim_score`
  (`LongBench/LongBench/metrics.py:80-87`): take the first prediction line that contains no
  `` ` ``, `#`, or `//`, then `fuzz.ratio / 100`:

```python
# src/bmx/cache/longbench.py
"""LongBench Code eval: code_sim scorer, dataset loader, task registry.

Faithful port of LongBench's Code-category eval (lcc, repobench-p). The scorer matches
LongBench's metrics.py::code_sim_score exactly (verified in .git/sdd/longbench-conventions.md
against the local clone). The dataset loader + scoring path are VM-only (real model); code_sim
is a pure CI-testable function.
"""

from __future__ import annotations

from fuzzywuzzy import fuzz


def code_sim(prediction: str, ground_truth: str) -> float:
    """LongBench code edit-similarity, range 0–1 (verbatim port of code_sim_score).

    Post-process: strip leading blank lines, then keep the FIRST line that contains no
    backtick / '#' / '//' (LongBench's rule), then fuzz.ratio normalized to [0, 1].
    """
    all_lines = prediction.lstrip("\n").split("\n")
    pred = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            pred = line
            break
    return fuzz.ratio(pred, ground_truth) / 100.0
```

- [ ] **Step 4: Run the tests to verify they pass.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_longbench.py -v`
Expected: 3 passed. (If `test_code_sim_identical_is_100` fails because the post-process strips
the multiline `code` to one line, adjust the test's `ground_truth` to the single expected line
per LongBench's convention — LongBench compares against a single ground-truth line. Match the
ledger; keep the identical-line case at 100.)

- [ ] **Step 5: Gate + commit (stage + propose).**

```bash
cd /d/Projects/bmx
uv run ruff format . && uv run ruff check . && uv run pytest -q
git add src/bmx/cache/longbench.py tests/test_longbench.py
#   feat: add LongBench code_sim scorer (faithful port of metrics.py)
```

---

## Task 4: LongBench task registry, dataset loader, and per-item scorer

**Files:**
- Modify: `src/bmx/cache/longbench.py`
- Test: `tests/test_longbench.py` (add cases)

**Interfaces:**
- Consumes: `bmx.cache.niah.generate_through_cache` (Task 2), `code_sim` (Task 3),
  `CacheCodecSpec`. Constants from the Task 0 ledger (templates, max_gen, field names).
- Produces:
  - `bmx.cache.longbench.LONGBENCH_TASKS: dict[str, dict]` — `{task: {"prompt_template": str,
    "max_gen": int}}` for `lcc` and `repobench-p`, verbatim from the ledger.
  - `bmx.cache.longbench.build_longbench_prompt(tokenizer, item: dict, task: str) -> torch.Tensor`
    — apply the task's prompt template to the item's context, return `(1, L)` ids.
  - `bmx.cache.longbench.load_longbench_task(task: str, n_samples: int | None) -> list[dict]`
    — lazy `load_dataset("THUDM/LongBench", task)`; return up to `n_samples` items (all if
    None), each a dict with the fields the scorer/prompt need (`context`/`input`/`answers`).
    VM-only; not unit-tested (it downloads).
  - `bmx.cache.longbench.longbench_code_score(model, tokenizer, item, task, n_prefill, k_spec, v_spec) -> float`
    — build prompt, `generate_through_cache`, `code_sim` vs the item's ground truth.

`LONGBENCH_TASKS` and `build_longbench_prompt` are CI-testable (no model/download). The loader
and per-item scorer are VM-only — judged by reading.

- [ ] **Step 1: Write the failing test (registry + prompt builder, the CI-testable surface).**

```python
# add to tests/test_longbench.py
from bmx.cache.longbench import LONGBENCH_TASKS, build_longbench_prompt


def test_longbench_tasks_registry():
    assert set(LONGBENCH_TASKS) == {"lcc", "repobench-p"}
    for t in ("lcc", "repobench-p"):
        assert "prompt_template" in LONGBENCH_TASKS[t]
        assert isinstance(LONGBENCH_TASKS[t]["max_gen"], int)
        assert "{context}" in LONGBENCH_TASKS[t]["prompt_template"]


def test_build_longbench_prompt_shapes():
    class StubTok:
        def __call__(self, text, return_tensors=None):
            import torch
            ids = torch.tensor([[ord(c) % 97 for c in text[:40]]])
            return type("E", (), {"input_ids": ids})()

    item = {"context": "def foo():\n    return 1\n", "input": "", "answers": ["    return 1"]}
    ids = build_longbench_prompt(StubTok(), item, "lcc")
    assert ids.shape[0] == 1 and ids.shape[1] > 0
```

- [ ] **Step 2: Run it to verify it fails.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_longbench.py -k "registry or prompt_shapes" -v`
Expected: FAIL — `ImportError: cannot import name 'LONGBENCH_TASKS'`.

- [ ] **Step 3: Implement the registry, prompt builder, loader, and per-item scorer.** The
  `LONGBENCH_TASKS` strings below are the VERIFIED verbatim values (Task 0 ledger / local clone)
  — use them EXACTLY, including the trailing space and `repobench-p`'s `{input}`. Confirm
  against `.git/sdd/longbench-conventions.md` before running:

```python
# append to src/bmx/cache/longbench.py
import torch

from bmx.cache.niah import generate_through_cache
from bmx.cache.specs import CacheCodecSpec

# VERBATIM from LongBench config (verified in .git/sdd/longbench-conventions.md against the
# local clone LongBench/LongBench/config/dataset2prompt.json + dataset2maxlen.json).
# NOTE the exact strings: trailing space after "below. ", and repobench-p has {input}, lcc does
# not. Copy these EXACTLY — do not normalize whitespace.
LONGBENCH_TASKS: dict[str, dict] = {
    "lcc": {
        "prompt_template": "Please complete the code given below. \n{context}Next line of code:\n",
        "max_gen": 64,
    },
    "repobench-p": {
        "prompt_template": "Please complete the code given below. \n{context}{input}Next line of code:\n",
        "max_gen": 64,
    },
}


def build_longbench_prompt(tokenizer, item: dict, task: str) -> torch.Tensor:
    """Apply the task's LongBench prompt template to the item; return (1, L) ids.

    LongBench formats dataset2prompt[task].format(**item); for code tasks the context lives in
    item['context']. (If the ledger records a build_chat wrapper for Llama-Instruct, apply it
    here exactly as LongBench does.)
    """
    template = LONGBENCH_TASKS[task]["prompt_template"]
    prompt = template.format(**item)
    return tokenizer(prompt, return_tensors="pt").input_ids


def load_longbench_task(task: str, n_samples: int | None) -> list[dict]:
    """Lazy-load THUDM/LongBench[task]; return up to n_samples items (all if None). VM-only."""
    from datasets import load_dataset

    ds = load_dataset("THUDM/LongBench", task, split="test")
    items = list(ds if n_samples is None else ds.select(range(min(n_samples, len(ds)))))
    return items


def longbench_code_score(
    model,
    tokenizer,
    item: dict,
    task: str,
    n_prefill: int,
    k_spec: CacheCodecSpec,
    v_spec: CacheCodecSpec,
) -> float:
    """Build the LongBench prompt, generate through the compressed cache, score code_sim.

    Ground truth is item['answers'][0] (LongBench code tasks have a single reference line).
    """
    prompt_ids = build_longbench_prompt(tokenizer, item, task)
    max_gen = LONGBENCH_TASKS[task]["max_gen"]
    response = generate_through_cache(
        model, tokenizer, prompt_ids, n_prefill, k_spec, v_spec, max_gen
    )
    ground_truth = item["answers"][0]
    return code_sim(response, ground_truth)
```

- [ ] **Step 4: Run the CI-testable tests to verify they pass.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_longbench.py -v`
Expected: all pass (code_sim tests + registry + prompt-shape).

- [ ] **Step 5: Gate + commit (stage + propose).**

```bash
cd /d/Projects/bmx
uv run ruff format . && uv run ruff check . && uv run pytest -q
git add src/bmx/cache/longbench.py tests/test_longbench.py
#   feat: add LongBench task registry, loader, and per-item code scorer
```

---

## Task 5: The `k3_longbench` experiment (sweep arms × tasks, emit parquet)

**Files:**
- Create: `experiments/k3_longbench.py`
- Test: `tests/test_k3_longbench_experiment.py`

**Interfaces:**
- Consumes: `experiments.k3_live_generation._spec_pair`, `experiments.k3_niah._compression_for`,
  `bmx.cache.longbench.{LONGBENCH_TASKS, load_longbench_task, longbench_code_score, code_sim, build_longbench_prompt}`,
  `bmx.cache.niah.generate_through_cache`, `bmx.artifacts.{create_run, write_metrics}`.
- Produces: `experiments.k3_longbench.{Config, run}`. `run(cfg, model=None, root="results")`
  writes `metrics.parquet` with columns:
  `arm, task, code_sim, n_samples, bpe_k, bpe_v, compression, n_prefill, score_kind`.

Mirror `experiments/k3_niah.py` exactly: real path (model is None) loads model + tokenizer +
LongBench and scores `code_sim`; offline path (model injected) runs a tiny synthetic code-ish
prompt through `generate_through_cache` + `code_sim` against a known target (mechanism + schema
only), emits `score_kind="code_sim_offline"`. Airtight lazy-import guard.

- [ ] **Step 1: Write the failing test (offline parquet-schema mechanics).**

```python
# tests/test_k3_longbench_experiment.py
"""k3_longbench emits a parquet with the expected schema (tiny_llama, offline, no download)."""

import pandas as pd

from experiments.k3_longbench import Config, run
from factories import tiny_llama


def test_k3_longbench_run_emits_parquet(tmp_path):
    model = tiny_llama()
    # tiny_llama max_position_embeddings=64 → keep prompt small; group=16 divisibility.
    cfg = Config(arms=("fp16", "kivi"), tasks=("lcc", "repobench-p"),
                 n_prefill=16, group=16, rank=4)
    run_dir = run(cfg, model=model, root=str(tmp_path))
    df = pd.read_parquet(run_dir / "metrics.parquet")
    for col in ("arm", "task", "code_sim", "n_samples", "bpe_k", "bpe_v",
                "compression", "n_prefill", "score_kind"):
        assert col in df.columns, f"missing column: {col}"
    # 2 arms × 2 tasks = 4 rows.
    assert len(df) == 4
    assert set(df["arm"]) <= {"fp16", "k2b", "kivi", "turboquant_mse", "turboquant_prod"}
    assert set(df["score_kind"]) == {"code_sim_offline"}
```

- [ ] **Step 2: Run it to verify it fails.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_k3_longbench_experiment.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.k3_longbench`.

- [ ] **Step 3: Implement the experiment** (mirror `experiments/k3_niah.py`):

```python
# experiments/k3_longbench.py
"""K3-LongBench — LongBench Code (lcc, repobench-p) recall under KV compression.

Reproduces TurboQuant's Code-column signal: sweeps arms × tasks on ONE code path
(StreamingQuantizedCache via generate_through_cache), scores LongBench code_sim. Reports each
arm's honest measured compression (never a pinned ratio).

Real path (model is None, VM run): loads model + tokenizer + THUDM/LongBench, generates through
the compressed cache, scores code_sim over n_samples (all if None) per task.

Offline-test path (model injected): a tiny synthetic code-ish prompt through tiny_llama;
code_sim against a known target. Mechanism + parquet schema only; no download, no tokenizer.

Model-agnostic: the SOTA VM run is a --model-name change. Figures: plots/plot_k3_longbench.py.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.longbench import LONGBENCH_TASKS, code_sim
from bmx.cache.niah import generate_through_cache
from experiments.k3_live_generation import _spec_pair
from experiments.k3_niah import _compression_for


@dataclasses.dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    arms: tuple[str, ...] = ("fp16", "k2b", "turboquant_mse", "turboquant_prod", "kivi")
    tasks: tuple[str, ...] = ("lcc", "repobench-p")
    n_samples: int | None = None  # None = full sets (Table-1 comparable); int caps (logged)
    n_prefill: int = 128
    rank: int = 16
    group: int = 64
    seed: int = 0


class _StubTok:
    """Offline decode stub: turns ids into a deterministic 'code-ish' string for code_sim."""

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids.tolist())


def run(cfg: Config, model=None, root: str = "results"):
    tokenizer = None
    if model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from bmx.cache.longbench import load_longbench_task, longbench_code_score

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch.float16
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    if cfg.n_samples is not None:
        print(f"[k3_longbench] SUBSAMPLED n_samples={cfg.n_samples} — NOT comparable to Table 1")

    run_dir = create_run("k3_longbench", cfg, root=root)
    rows = []
    for arm in cfg.arms:
        k_spec, v_spec = _spec_pair(arm, cfg)
        for task in cfg.tasks:
            if tokenizer is None:
                # Offline mechanism: one synthetic prompt through the shared generate path.
                g = torch.Generator().manual_seed(cfg.seed)
                prompt_ids = torch.randint(0, model.config.vocab_size, (1, 32), generator=g)
                resp = generate_through_cache(
                    model, _StubTok(), prompt_ids, cfg.n_prefill, k_spec, v_spec,
                    max_new_tokens=LONGBENCH_TASKS[task]["max_gen"],
                )
                score = code_sim(resp, resp)  # identical => 100; mechanism/schema only
                n_used = 1
                score_kind = "code_sim_offline"
                length = prompt_ids.shape[1]
            else:
                items = load_longbench_task(task, cfg.n_samples)
                scores = [
                    longbench_code_score(
                        model, tokenizer, it, task, cfg.n_prefill, k_spec, v_spec
                    )
                    for it in items
                ]
                score = sum(scores) / len(scores) if scores else float("nan")
                n_used = len(items)
                score_kind = "code_sim"
                length = cfg.n_prefill * 2  # representative length for compression accounting

            bpe_k, bpe_v, compression = _compression_for(model, k_spec, v_spec, length)
            rows.append({
                "arm": arm, "task": task, "code_sim": score, "n_samples": n_used,
                "bpe_k": bpe_k, "bpe_v": bpe_v, "compression": compression,
                "n_prefill": cfg.n_prefill, "score_kind": score_kind,
            })

    write_metrics(run_dir, pd.DataFrame(rows))
    return run_dir


if __name__ == "__main__":
    run(tyro.cli(Config))
```

- [ ] **Step 4: Run the test to verify it passes.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_k3_longbench_experiment.py -v`
Expected: 1 passed (4 rows, schema correct, no download).

- [ ] **Step 5: Gate + commit (stage + propose).**

```bash
cd /d/Projects/bmx
uv run ruff format . && uv run ruff check . && uv run pytest -q
git add experiments/k3_longbench.py tests/test_k3_longbench_experiment.py
#   feat: add k3_longbench experiment — LongBench Code recall on the one fair cache path
```

---

## Task 6: Figure — code_sim per arm per task

**Files:**
- Create: `experiments/plots/plot_k3_longbench.py`
- Test: `tests/test_k3_longbench_experiment.py` (add a plot test)

**Interfaces:**
- Consumes: a metrics DataFrame with `arm, task, code_sim, compression`.
- Produces: `experiments.plots.plot_k3_longbench.make_figures(df, out_dir: str) -> list[Path]`
  — grouped bar chart of mean code_sim per arm per task, each arm annotated with its
  compression; fp16 score drawn as a reference line. Returns the PNG paths.

- [ ] **Step 1: Write the failing test.**

```python
# add to tests/test_k3_longbench_experiment.py
def test_plot_k3_longbench_makes_pngs(tmp_path):
    import pandas as pd
    from experiments.plots.plot_k3_longbench import make_figures

    df = pd.DataFrame([
        {"arm": "fp16", "task": "lcc", "code_sim": 46.0, "compression": 1.0},
        {"arm": "kivi", "task": "lcc", "code_sim": 44.0, "compression": 4.1},
        {"arm": "fp16", "task": "repobench-p", "code_sim": 45.0, "compression": 1.0},
        {"arm": "kivi", "task": "repobench-p", "code_sim": 42.0, "compression": 4.1},
    ])
    paths = make_figures(df, str(tmp_path))
    assert len(paths) >= 1
    assert all(p.exists() for p in paths)
```

- [ ] **Step 2: Run it to verify it fails.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_k3_longbench_experiment.py::test_plot_k3_longbench_makes_pngs -v`
Expected: FAIL — `ModuleNotFoundError: experiments.plots.plot_k3_longbench`.

- [ ] **Step 3: Implement the figure** (mirror `experiments/plots/plot_k3_niah.py`'s headless setup):

```python
# experiments/plots/plot_k3_longbench.py
"""Figure for k3_longbench: mean code_sim per arm per task (vs fp16 reference).

Reads the parquet, never refits. Select runs explicitly upstream.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def make_figures(df, out_dir: str) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tasks = sorted(df["task"].unique())
    arms = sorted(df["arm"].unique())
    x = range(len(tasks))
    width = 0.8 / max(len(arms), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, arm in enumerate(arms):
        g = df[df["arm"] == arm]
        means = [
            g[g["task"] == t]["code_sim"].mean() if not g[g["task"] == t].empty else 0.0
            for t in tasks
        ]
        comp = g["compression"].iloc[0] if not g.empty else 1.0
        offs = [xi + i * width for xi in x]
        ax.bar(offs, means, width=width, label=f"{arm} ({comp:.1f}×)")

    # fp16 reference line (mean across its tasks), if present.
    if "fp16" in arms:
        fp16_mean = df[df["arm"] == "fp16"]["code_sim"].mean()
        ax.axhline(fp16_mean, ls="--", color="gray", lw=1, label="fp16 mean (reference)")

    ax.set_xticks([xi + width * (len(arms) - 1) / 2 for xi in x])
    ax.set_xticklabels(tasks)
    ax.set_ylabel("code_sim (edit-similarity, 0–1)")
    ax.set_title("LongBench Code recall per arm under KV compression")
    ax.legend()
    p = out / "longbench_code_sim.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [p]
```

- [ ] **Step 4: Run the test to verify it passes.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_k3_longbench_experiment.py -v`
Expected: 2 passed (schema + plot).

- [ ] **Step 5: Gate + commit (stage + propose).**

```bash
cd /d/Projects/bmx
uv run ruff format . && uv run ruff check . && uv run pytest -q
git add experiments/plots/plot_k3_longbench.py tests/test_k3_longbench_experiment.py
#   feat: add k3_longbench figure (code_sim per arm per task vs fp16 reference)
```

---

## Task 7: Record research-state + VM handoff

**Files:**
- Modify: `CLAUDE.md` (research-state line)
- Create: `docs/2026-06-20-longbench-plan-state.md` (VM handoff)

**Interfaces:**
- Consumes: nothing.
- Produces: a recorded next-step (the VM run, alongside NIAH) and the headroom guard.

- [ ] **Step 1: Add a research-state line** under the NIAH entry in `CLAUDE.md`:

```
- LongBench Code eval (task metric #0, coding half): TurboQuant Table-1 Code signal
  (lcc + repobench-p) via LongBench code_sim on the SAME StreamingQuantizedCache path
  (shared generate_through_cache); offline mechanism gate + VM full-set headline
  (`experiments/k3_longbench.py --model-name`). Spec/plan:
  docs/superpowers/{specs,plans}/2026-06-20-longbench-*; VM handoff:
  docs/2026-06-20-longbench-plan-state.md.
```

- [ ] **Step 2: Write the VM handoff note** `docs/2026-06-20-longbench-plan-state.md`: the exact
  VM command (`uv run python experiments/k3_longbench.py --model-name meta-llama/Llama-3.1-8B-Instruct`),
  that `n_samples=None` runs the full 500+500 (Table-1 comparable; state expected wall-clock is
  long — `--n-samples N` for a fast look, logged as not-comparable), the headroom guard (flag a
  task where fp16 code_sim is at floor or all arms tie at ceiling as non-discriminating), that
  it shares `generate_through_cache` with NIAH (run both in one VM session), and that the
  scorer is LongBench's exact `code_sim` (fuzzywuzzy) — read against Table 1's Code column.

- [ ] **Step 3: Gate + commit (stage + propose).**

```bash
cd /d/Projects/bmx
uv run ruff format . && uv run ruff check . && uv run pytest -q
git add CLAUDE.md docs/2026-06-20-longbench-plan-state.md docs/superpowers/specs/2026-06-20-longbench-code-eval-design.md docs/superpowers/plans/2026-06-20-longbench-code-eval.md
#   docs: LongBench Code eval spec+plan, research-state, VM handoff
```

---

## Whole-branch review (non-negotiable — RUNS the code)

After all tasks: dispatch the `ml-research-reviewer` agent on the **full diff** for this feature,
with the explicit mandate to RUN the code. Specific probes to demand:

- **Refactor is behavior-preserving:** confirm `niah_recall_generate` returns the same value
  after the `generate_through_cache` extraction (run `tests/test_niah.py`); confirm BOTH NIAH
  and LongBench now route generation through the one `generate_through_cache` function.
- **Fairness:** every arm routes through `_spec_pair` → `generate_through_cache` →
  `StreamingQuantizedCache` — no arm has a private path.
- **Scorer fidelity:** independently check `code_sim` against the Task 0 ledger's LongBench
  post-processing rule and `fuzz.ratio`; verify `code_sim(x, x) == 1.0` (LongBench divides by
  100) and a one-char-off case is graded in (0, 1). Confirm the templates/`max_gen` in
  `LONGBENCH_TASKS` match LongBench verbatim (incl. trailing space + `repobench-p`'s `{input}`).
- **No-download guarantee:** force-run `uv run pytest -q` with no network; confirm CI never hits
  `load_dataset`/tokenizer/model (lazy imports inside `if model is None:`).
- **Honest accounting:** `score_kind` distinguishes `code_sim` (real) from `code_sim_offline`
  (mechanism); `n_samples` records the real count; subsample runs log the not-comparable warning;
  compression is re-derived from `bits_per_entry`/`memory_report` (not taken on report).

Record findings in `.git/sdd/longbench-progress.md`; insert fix-tasks with their own gates for any
Critical/Important. Do NOT merge or push without explicit user approval.
