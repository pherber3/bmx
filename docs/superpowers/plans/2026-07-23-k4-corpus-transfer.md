# K4 Corpus-Transfer Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks 1–6 are local CPU TDD code; Task 7 is the real gpt2 run + results doc; Task 8 is the verification gate. The Llama fit-side replication is a pre-registered VM addendum (unnumbered) at the end — it rides the next rental, NOT this plan's execution.

**Binding spec (read first):** `docs/superpowers/specs/2026-07-23-k4-corpus-transfer-design.md`. Branch `feat/triton-decode-kernel`, plan written at HEAD `023d2e8`.

**Goal:** Measure whether the calibration corpus changes spectral-pack quality (fit ∈ {wikitext, code, shuffled-null} × eval ∈ {wiki-held, code-held} win matrix), decompose WHY (H1 intrinsic-top / H2 tail / H3 basis-transfers-allocation-adapts), and emit a pre-registered §4 verdict — kill-or-confirm at gpt2 mechanism scale.

**Architecture:** Generalize `load_eval_tokens` (dataset id/config/data-dir/split/text-field + seeded post-slice token shuffle; defaults byte-identical) and thread it through `collect_cache.py` with corpus-labeled output names. A new thin tyro harness `experiments/k4_corpus_transfer.py` reuses the existing K4 machinery — `corpus_fit_bases` (hoisted from `k4_fit_packs.py`), `_layer_ctx`/`_score_tail`/`_tq_layer_curve`/`_log_interp` from `experiments/_k4_common.py`, `pack_from_basis` with a new default-inert `lam_alloc` kwarg for the H3 hybrid — and writes `metrics.parquet` (win matrix + hybrid + W-cross), `overlap.parquet` (mechanism diagnostics), and `corpus_transfer_verdict.json` under `results/k4_corpus_transfer/<run-id>/`.

**Tech Stack:** PyTorch (fp32 experiments / fp64 moment math), transformers 5.11, HF `datasets`, tyro, safetensors, pandas/parquet, pytest. All local CPU (gpt2, minutes per step); no CUDA anywhere in this plan.

---

## Global Constraints

**Repo hard rules (CLAUDE.md):**

- **NEVER `git commit` without the user's explicit approval.** Stage, propose the task's exact message, STOP. No AI attribution ever.
- Before any commit: `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` — all clean, then re-stage.
- Use the Bash tool (git bash); `cd /d/Projects/bmx` first in fresh shells. Dependencies only via `uv add` (this plan needs none).
- fp64 in tests, fp32 in experiments/codecs (caches stored fp16). Shape asserts at boundaries. Tiny offline test models/fixtures only in tests — never download in tests. Comparisons align on total bits, never rank. Rank codecs on logit distortion vs real queries, not Frobenius.
- Experiments are thin tyro scripts; artifacts under `results/<exp>/<run-id>/` with config + env + SHA; commit metrics/figures parquet, never raw caches.

**Session-lead binding decisions (verbatim, non-negotiable):**

1. **MATCHED FIT-TOKEN BUDGETS:** every fit corpus uses the SAME total token count and the same number of slices (else corpus-size confounds corpus-content — a reviewer press). Exact counts for this plan: **4 fit slices × 1024 tokens = 4096 fit tokens per corpus** (offsets 1024/2048/3072/4096 in each corpus's own token stream), **2 heldout eval slices × 1024 per natural corpus** (offsets 0 and 5120) — matching the existing gpt2 artifact shape (`results/cache/gpt2_1024_off{1024,2048,3072,4096}.safetensors` are the wikitext fit slices; `gpt2_1024.safetensors` = wiki heldout off0). The harness asserts equal slice counts AND equal total fit rows across corpora.
2. **W-CROSS CELL:** one extra cell separating the two corpus-derived moments — Σ (keys) from corpus A + W (queries) from corpus B, both directions, scored on both eval sides. Free via `fit_spectral_pack`'s separate Wh argument. Report beside the hybrid arm.
3. **The win metric is bits-normalized per pack** (TQ curve interpolated at each pack's OWN skeptic bpe), so cross-fit win ratios are fair even when packs' bpe differ slightly; the harness task states this explicitly so the implementer doesn't "fix" it into a matched-bpe constraint.
4. **Shuffle null:** token-level permutation of the wikitext fit slices, ONE seed (fixed, recorded: **20260723**; per-slice generator seeded `shuffle_seed + token_offset`); shuffle applied AFTER slicing so the null slices cover the same token multiset as the wikitext fit slices.
5. **Code corpus** is `bigcode/the-stack-smol` (Python subset via `data_dir="data/python"`, split `train`, field `content`) — the implementer MUST assert on the actual field name at load (`load_eval_tokens` asserts `text_field` is a column). If the dataset id turns out wrong at implementation time, the recorded fallback is `codeparrot/codeparrot-clean-valid` (split `train`, field `content`, no config name — pass `--dataset-config ""`).
6. **gpt2 scale = mechanism verdicts.** Every table caption carries the gpt2 yellow-flag label ("gpt2 mechanism scale — corpus-W retention ~0.47–0.52, `docs/2026-07-15-k4-duel-results.md`; Llama fit-side replication pre-registered"). Llama fit-side replication is a pre-registered VM addendum task list at the end (NOT numbered local tasks).
7. **Commit discipline:** per-task commit messages proposed in each task; battery (`uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q`, baseline **461 passed / 17 skipped / 1 xfailed** at `023d2e8`) before every commit.
8. **The results doc task pre-drafts BOTH verdict templates** (insensitive / domain-sensitive) and includes a "WHY" section that cites the analytic decomposition results and a vault-grounded prior paragraph (massive activations / attention sinks / rogue channels are input-agnostic model artifacts) — the vault pass is done by the controller at writeup time; the plan leaves the labeled slot, nothing else.
9. Task structure: ~6–8 local tasks, each independently reviewable (realized as Tasks 1–8 below).

**Spec constraints (§6–§7):** no new cache-spec fields, no change to shipped duel packs or old parquets; no task-level headline from gpt2; no third corpus; no web search (HF dataset downloads are fine); `load_eval_tokens` default path stays byte-identical (existing wikitext caches are reused unchanged; new corpora get NEW cache files); deterministic seeds everywhere.

---

## File structure

- Modify `src/bmx/eval/layer_swap.py:48-65` — `load_eval_tokens` gains `dataset_id`, `data_dir`, `split`, `text_field`, `shuffle_seed` keyword params (defaults byte-identical).
- Modify `experiments/collect_cache.py` — corpus passthrough knobs + `corpus_label` in output naming + a non-default-corpus guard; naming factored into testable `_out_path` / `_corpus_is_default` helpers.
- Modify `experiments/_k4_common.py` — receives `_LayerCtx`/`_layer_ctx` (moved verbatim from `k4_dec_quant.py`) and the new `CorpusFit`/`corpus_fit_bases` (extracted from `k4_fit_packs.py`'s per-layer fit loop).
- Modify `experiments/k4_dec_quant.py` — imports `_layer_ctx` from `_k4_common` instead of defining it.
- Modify `experiments/k4_fit_packs.py` — `main` calls `corpus_fit_bases` (byte-identity pinned by the existing `test_k4_fit_packs_default_unchanged`).
- Modify `src/bmx/cache/spectral.py` — `basis_alloc_moment(basis, M_alloc)` (new) + `pack_from_basis(..., lam_alloc=None)` (default-inert). NOT a cache-spec/codec-field change: fitting-path only, mandated by spec §3's hybrid arm.
- Create `experiments/k4_corpus_transfer.py` — the harness (matrix + hybrid + W-cross + diagnostics + verdict).
- Create `docs/2026-07-23-k4-corpus-transfer-results.md` — results doc (Task 7).
- Tests: `tests/test_cache_collect.py` (append, Tasks 1–2), `tests/test_spectral.py` (append, Task 5), `tests/test_k4_experiments.py` (append, Tasks 3–6).
- New local caches (gitignored, regenerable): `results/cache/gpt2_1024_off5120.safetensors`, `gpt2_1024_code{,_off1024,_off2048,_off3072,_off4096,_off5120}.safetensors`, `gpt2_1024_shuf_off{1024,2048,3072,4096}.safetensors`.

---

### Task 1: `load_eval_tokens` generalization (dataset passthrough + seeded post-slice shuffle)

**Files:**
- Modify: `src/bmx/eval/layer_swap.py:48-65` (`load_eval_tokens`)
- Test: `tests/test_cache_collect.py` (append at end)

**Interfaces:**
- Consumes: nothing new (HF `datasets.load_dataset`, `transformers.AutoTokenizer` — already used).
- Produces (Task 2 and the collection commands rely on this exact signature):
  ```python
  def load_eval_tokens(
      model_name: str = "gpt2",
      dataset: str = "wikitext-2-raw-v1",   # HF config name; "" => no config arg
      n_tokens: int = 65536,
      token_offset: int = 0,
      *,
      dataset_id: str = "Salesforce/wikitext",
      data_dir: str = "",                    # when set, load_dataset(dataset_id, data_dir=..., split=...) — the-stack-smol style
      split: str = "test",
      text_field: str = "text",
      shuffle_seed: int = -1,                # >= 0 => permute the RETURNED slice (post-slice token null)
  ) -> torch.Tensor  # 1-D int64, length n_tokens
  ```
  Defaults reproduce today's behavior byte-identically. Shuffle generator: `torch.Generator().manual_seed(shuffle_seed + token_offset)` (binding decision 4).

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_cache_collect.py` (the file already imports `torch`; add `import pytest` at the top of the file if not already present):

```python
# ---------------------------------------------------------------------------
# load_eval_tokens generalization — corpus passthrough + shuffled-token null
# ---------------------------------------------------------------------------


def _patch_eval_tokens_io(monkeypatch):
    """Fake tokenizer (arange ids) + fake dataset with 'text' and 'content'
    columns, so no download happens and byte-identity is checkable."""

    class _FakeTok:
        def __call__(self, text, return_tensors, truncation, max_length):
            import torch

            ids = torch.arange(max_length).unsqueeze(0)
            return type("E", (), {"input_ids": ids})()

    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda *a, **k: {"text": ["x"], "content": ["y"]},
        raising=False,
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: _FakeTok()
    )


def test_load_eval_tokens_generalized_defaults(monkeypatch):
    import bmx.eval.layer_swap as ls

    _patch_eval_tokens_io(monkeypatch)
    base = ls.load_eval_tokens("gpt2", n_tokens=16, token_offset=8)
    explicit = ls.load_eval_tokens(
        "gpt2",
        "wikitext-2-raw-v1",
        n_tokens=16,
        token_offset=8,
        dataset_id="Salesforce/wikitext",
        data_dir="",
        split="test",
        text_field="text",
        shuffle_seed=-1,
    )
    assert torch.equal(base, explicit)
    # pre-change behavior pin: arange slice starting at token_offset
    assert base.shape == (16,) and base[0].item() == 8


def test_load_eval_tokens_shuffle_after_slice(monkeypatch):
    import bmx.eval.layer_swap as ls

    _patch_eval_tokens_io(monkeypatch)
    nat = ls.load_eval_tokens("gpt2", n_tokens=16, token_offset=8)
    shuf1 = ls.load_eval_tokens(
        "gpt2", n_tokens=16, token_offset=8, shuffle_seed=20260723
    )
    shuf2 = ls.load_eval_tokens(
        "gpt2", n_tokens=16, token_offset=8, shuffle_seed=20260723
    )
    assert torch.equal(shuf1, shuf2)  # deterministic under the recorded seed
    assert not torch.equal(shuf1, nat)  # actually permuted
    # shuffle AFTER slicing: same token multiset as the natural slice
    assert torch.equal(shuf1.sort().values, nat.sort().values)


def test_load_eval_tokens_text_field_assert(monkeypatch):
    import bmx.eval.layer_swap as ls

    _patch_eval_tokens_io(monkeypatch)
    # 'content' exists in the fake dataset -> passes; 'nope' must assert.
    ok = ls.load_eval_tokens("gpt2", n_tokens=8, text_field="content")
    assert ok.shape == (8,)
    with pytest.raises(AssertionError, match="text_field"):
        ls.load_eval_tokens("gpt2", n_tokens=8, text_field="nope")
```

- [ ] **Step 2: Run tests to verify they fail.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_cache_collect.py -q -k "generalized or shuffle_after or text_field"`
Expected: 3 FAIL — `TypeError: load_eval_tokens() got an unexpected keyword argument 'dataset_id'` (and the text_field one likewise).

- [ ] **Step 3: Implement.** Replace `load_eval_tokens` in `src/bmx/eval/layer_swap.py` (lines 48–65) with:

```python
def load_eval_tokens(
    model_name: str = "gpt2",
    dataset: str = "wikitext-2-raw-v1",
    n_tokens: int = 65536,
    token_offset: int = 0,
    *,
    dataset_id: str = "Salesforce/wikitext",
    data_dir: str = "",
    split: str = "test",
    text_field: str = "text",
    shuffle_seed: int = -1,
) -> torch.Tensor:
    """Tokenize a corpus slice for eval/calibration. Defaults reproduce the
    original wikitext-test path byte-identically.

    `dataset` is the HF config name ("" = dataset has no named config);
    `data_dir` selects a sub-directory dataset (the-stack-smol style) and
    overrides the config name. `shuffle_seed >= 0` permutes the RETURNED
    slice (shuffle AFTER slicing: the null slice covers the same token
    multiset as the natural slice at this offset); the generator is seeded
    `shuffle_seed + token_offset` so distinct slices get distinct — but fully
    recorded — permutations.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if data_dir:
        ds = load_dataset(dataset_id, data_dir=data_dir, split=split)
    elif dataset:
        ds = load_dataset(dataset_id, dataset, split=split)
    else:
        ds = load_dataset(dataset_id, split=split)
    # Dataset objects expose column_names; test fakes are plain dicts.
    cols = getattr(ds, "column_names", None) or list(ds.keys())
    assert text_field in cols, (
        f"text_field {text_field!r} not a column of {dataset_id}: {cols}"
    )
    text = "\n\n".join(ds[text_field])
    # truncation at the tokenizer avoids encoding the full split
    ids = tok(
        text, return_tensors="pt", truncation=True, max_length=token_offset + n_tokens
    )
    out = ids.input_ids[0][token_offset:]
    if shuffle_seed >= 0:
        g = torch.Generator().manual_seed(shuffle_seed + token_offset)
        out = out[torch.randperm(out.numel(), generator=g)]
    return out
```

Note the ONLY behavioral deltas on the default path: the `cols` assert (wikitext has `text` → passes) and the join reading `ds[text_field]` instead of `load_dataset(...)["text"]` — same value. `swap_and_perplexity` (line 80) still calls `load_eval_tokens(model_name, dataset, n_tokens)` positionally — unchanged.

- [ ] **Step 4: Run tests to verify they pass.**

Run: `uv run pytest tests/test_cache_collect.py -q`
Expected: all pass, including the pre-existing `test_load_eval_tokens_offset` (its dict-based fake must still work — that's what the `cols` fallback is for).

- [ ] **Step 5: Battery + propose commit.**

Run: `uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: clean; `464 passed, 17 skipped, 1 xfailed`.
Stage `src/bmx/eval/layer_swap.py tests/test_cache_collect.py`; propose message:
`feat(eval): load_eval_tokens corpus passthrough (dataset id/config/data-dir/split/field) + seeded post-slice token shuffle; defaults byte-identical`
**STOP for user approval.**

---

### Task 2: `collect_cache.py` corpus passthrough + local cache collection

**Files:**
- Modify: `experiments/collect_cache.py`
- Test: `tests/test_cache_collect.py` (append)

**Interfaces:**
- Consumes: Task 1's `load_eval_tokens(model_name, dataset, n_tokens=, token_offset=, dataset_id=, data_dir=, split=, text_field=, shuffle_seed=)`.
- Produces: `Config` fields `dataset_id: str = "Salesforce/wikitext"`, `dataset_config: str = "wikitext-2-raw-v1"`, `data_dir: str = ""`, `split: str = "test"`, `text_field: str = "text"`, `shuffle_seed: int = -1`, `corpus_label: str = ""`; module helpers `_corpus_is_default(cfg: Config) -> bool` and `_out_path(cfg: Config) -> Path`; output naming `results/cache/<model_short>_<seq_len>[_<corpus_label>][_off<token_offset>].safetensors`. The collected cache files listed in "File structure" are what Task 7 consumes.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_cache_collect.py`:

```python
def test_collect_cache_out_path_and_corpus_guard():
    from experiments.collect_cache import Config, _corpus_is_default, _out_path

    default = Config(model_name="gpt2", seq_len=1024)
    assert _corpus_is_default(default)
    assert _out_path(default).name == "gpt2_1024.safetensors"

    off = Config(model_name="gpt2", seq_len=1024, token_offset=1024)
    # pre-change names unchanged: existing wikitext caches stay reusable as-is
    assert _out_path(off).name == "gpt2_1024_off1024.safetensors"

    code = Config(
        model_name="gpt2",
        seq_len=1024,
        token_offset=1024,
        dataset_id="bigcode/the-stack-smol",
        data_dir="data/python",
        split="train",
        text_field="content",
        corpus_label="code",
    )
    assert not _corpus_is_default(code)
    assert _out_path(code).name == "gpt2_1024_code_off1024.safetensors"

    shuf = Config(
        model_name="gpt2",
        seq_len=1024,
        token_offset=1024,
        shuffle_seed=20260723,
        corpus_label="shuf",
    )
    assert not _corpus_is_default(shuf)
    assert _out_path(shuf).name == "gpt2_1024_shuf_off1024.safetensors"

    # --out still overrides everything
    assert _out_path(
        Config(model_name="gpt2", seq_len=1024, out="x/y.safetensors")
    ) == Path("x/y.safetensors")
```

Add `from pathlib import Path` to the test file's imports if not present.

- [ ] **Step 2: Run to verify failure.**

Run: `uv run pytest tests/test_cache_collect.py::test_collect_cache_out_path_and_corpus_guard -q`
Expected: FAIL — `ImportError: cannot import name '_corpus_is_default'` (or `TypeError` on the new Config fields).

- [ ] **Step 3: Implement.** In `experiments/collect_cache.py`, extend `Config` and factor the naming:

```python
@dataclasses.dataclass
class Config:
    model_name: str = "gpt2"
    seq_len: int = 1024
    n_q_keep: int = 256
    token_offset: int = 0  # calibration-corpus slice offset (0 => leading tokens)
    out: str = ""  # override output path; empty => auto
    # ---- corpus passthrough (K4 corpus-transfer gate) ----------------------
    dataset_id: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"  # "" => dataset has no config name
    data_dir: str = ""  # HF data_dir (the-stack-smol style); overrides config
    split: str = "test"
    text_field: str = "text"
    shuffle_seed: int = -1  # >=0 => seeded post-slice token shuffle (null corpus)
    corpus_label: str = ""  # REQUIRED when any corpus knob above is non-default


_WIKI_DEFAULTS = ("Salesforce/wikitext", "wikitext-2-raw-v1", "", "test", "text", -1)


def _corpus_is_default(cfg: Config) -> bool:
    return (
        cfg.dataset_id,
        cfg.dataset_config,
        cfg.data_dir,
        cfg.split,
        cfg.text_field,
        cfg.shuffle_seed,
    ) == _WIKI_DEFAULTS


def _out_path(cfg: Config) -> Path:
    if cfg.out:
        return Path(cfg.out)
    model_short = cfg.model_name.split("/")[-1].lower()
    label = f"_{cfg.corpus_label}" if cfg.corpus_label else ""
    suffix = f"_off{cfg.token_offset}" if cfg.token_offset else ""
    return (
        Path("results/cache")
        / f"{model_short}_{cfg.seq_len}{label}{suffix}.safetensors"
    )
```

In `main`, replace the output-path block (current lines 39–48) with:

```python
    # Never overwrite wikitext-named caches with a different corpus's content.
    assert _corpus_is_default(cfg) or cfg.corpus_label, (
        "non-default corpus knobs require --corpus-label so the output name "
        "encodes the corpus"
    )
    out_path = _out_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
```

and replace the `load_eval_tokens` call (current lines 60–62) with:

```python
    tokens = load_eval_tokens(
        cfg.model_name,
        cfg.dataset_config,
        n_tokens=cfg.seq_len,
        token_offset=cfg.token_offset,
        dataset_id=cfg.dataset_id,
        data_dir=cfg.data_dir,
        split=cfg.split,
        text_field=cfg.text_field,
        shuffle_seed=cfg.shuffle_seed,
    )
```

Update the module docstring's Usage block with one code-corpus example line (the first command from Step 5).

- [ ] **Step 4: Run tests.**

Run: `uv run pytest tests/test_cache_collect.py -q`
Expected: PASS (all, including pre-existing tests).

- [ ] **Step 5: Collect the new caches (local CPU, network for HF downloads; ~1–2 min each).** From `/d/Projects/bmx`:

```bash
# (a) second wikitext heldout slice (eval side)
uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024 --token-offset 5120

# (b) code corpus: 4 fit + 2 heldout slices, same offsets discipline
for OFF in 0 1024 2048 3072 4096 5120; do
  uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024 \
    --token-offset $OFF \
    --dataset-id bigcode/the-stack-smol --data-dir data/python \
    --split train --text-field content --corpus-label code
done

# (c) shuffled-token null: fit-side only (4 slices), ONE recorded seed
for OFF in 1024 2048 3072 4096; do
  uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024 \
    --token-offset $OFF --shuffle-seed 20260723 --corpus-label shuf
done
```

Expected: 11 new files in `results/cache/` (each ~28 MB, same as the existing gpt2 caches): `gpt2_1024_off5120`, `gpt2_1024_code`, `gpt2_1024_code_off{1024,2048,3072,4096,5120}`, `gpt2_1024_shuf_off{1024,2048,3072,4096}` (all `.safetensors`).

**FALLBACK (binding decision 5):** if `bigcode/the-stack-smol` fails to load or the `text_field` assert fires (field not named `content`), rerun the (b) loop with `--dataset-id codeparrot/codeparrot-clean-valid --dataset-config "" --data-dir "" --split train --text-field content --corpus-label code`, and record which id was used in Task 7's results doc.

- [ ] **Step 6: Sanity-check one cache of each corpus.**

```bash
uv run python - <<'EOF'
from bmx.cache.collect import load_cache
for p in ("results/cache/gpt2_1024_code_off1024.safetensors",
          "results/cache/gpt2_1024_shuf_off1024.safetensors",
          "results/cache/gpt2_1024_off5120.safetensors"):
    c = load_cache(p)
    assert c["layer0.k_pre"].shape == (12, 1024, 64), (p, c["layer0.k_pre"].shape)
    print("OK", p)
EOF
```

Expected: three `OK` lines (gpt2: 12 heads, d=64, S=1024).

- [ ] **Step 7: Battery + propose commit.** (Caches are gitignored — only code + tests get committed.)

Run: `uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: clean; `465 passed, 17 skipped, 1 xfailed`.
Stage `experiments/collect_cache.py tests/test_cache_collect.py`; propose message:
`feat(exp): collect_cache corpus passthrough — dataset knobs + shuffle-seed + corpus-labeled output names (wikitext names unchanged)`
**STOP for user approval.**

---

### Task 3: Shared plumbing — `_layer_ctx` hoist + `corpus_fit_bases` extraction

**Files:**
- Modify: `experiments/_k4_common.py` (add `_LayerCtx`, `_layer_ctx`, `CorpusFit`, `corpus_fit_bases`)
- Modify: `experiments/k4_dec_quant.py:88-137` (delete local `_LayerCtx`/`_layer_ctx`, import instead)
- Modify: `experiments/k4_fit_packs.py:187-231` (per-layer fit loop → `corpus_fit_bases` call)
- Test: `tests/test_k4_experiments.py` (append)

**Interfaces:**
- Consumes: existing `_k4_common` helpers (`corpus_query_moment`, `setup_rope`, `load_layer_keys`), `bmx.cache.spectral.{SpectralBasis, assemble_whitener, fit_spectral_basis, identity_whitener}`, `bmx.cache.collect.to_matrix`.
- Produces (Tasks 4–6 depend on these exact names):
  ```python
  class CorpusFit(NamedTuple):
      bases: dict[int, SpectralBasis]
      M_fits: dict[int, torch.Tensor]  # (S_total, C) fp32 concat of fit caches
      whiteners: dict[int, tuple[torch.Tensor, torch.Tensor]]  # fp64 (Wh, Wh_inv)

  def corpus_fit_bases(
      per_cache_layer_keys: list[dict[int, dict[str, torch.Tensor]]],
      get_cos_sins: list,
      rope_ready: bool,
      layers: list[int],
      *,
      w_source: str,          # "corpus" | "none"
      ridge: float,
      position_stride: int,
  ) -> CorpusFit
  ```
  plus `_LayerCtx` / `_layer_ctx(kinds_map, *, rope_ready, get_cos_sin) -> _LayerCtx` now importable from `experiments._k4_common` (fields unchanged: `k_pre_t, h_kv, S, d, C, Q_fp32, cos_l, sin_l, K_post_true, M_pre, tail`).

- [ ] **Step 1: Write the failing test.** Append to `tests/test_k4_experiments.py`:

```python
def test_corpus_fit_bases_matches_direct_fit(tmp_path):
    from bmx.cache.spectral import fit_spectral_basis, identity_whitener
    from experiments._k4_common import corpus_fit_bases, load_layer_keys, setup_rope

    p1, p2 = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    _tiny_cache(p1, seed=0)
    _tiny_cache(p2, seed=1)
    per_cache = [load_layer_keys(str(p)) for p in (p1, p2)]
    layers = sorted(per_cache[0].keys())
    get_cos_sins = [setup_rope("", lk, layers)[1] for lk in per_cache]

    fit = corpus_fit_bases(
        per_cache, get_cos_sins, False, layers,
        w_source="none", ridge=1e-3, position_stride=8,
    )
    M_ref = torch.cat([to_matrix(lk[0]["k_pre"]) for lk in per_cache], dim=0)
    Ih, Ih_inv = identity_whitener(M_ref.shape[1])
    ref = fit_spectral_basis(M_ref, Ih, Ih_inv)
    assert torch.equal(fit.bases[0].enc, ref.enc)
    assert torch.equal(fit.bases[0].dec, ref.dec)
    assert torch.equal(fit.M_fits[0], M_ref)
    assert torch.equal(fit.whiteners[0][0], Ih)


def test_layer_ctx_importable_from_k4_common(tmp_path):
    from experiments._k4_common import _layer_ctx, load_layer_keys

    p = tmp_path / "c.safetensors"
    _tiny_cache(p, seed=0)
    lk = load_layer_keys(str(p))
    ctx = _layer_ctx(lk[0], rope_ready=False, get_cos_sin=lambda S: None)
    assert ctx.C == ctx.h_kv * ctx.d == 16
    assert ctx.M_pre.shape == (128, 16)
    assert ctx.tail == slice(64, 128)
    assert ctx.K_post_true is None
```

- [ ] **Step 2: Run to verify failure.**

Run: `uv run pytest tests/test_k4_experiments.py -q -k "corpus_fit_bases or layer_ctx_importable"`
Expected: 2 FAIL — `ImportError: cannot import name 'corpus_fit_bases'` / `'_layer_ctx'`.

- [ ] **Step 3: Implement the hoist.** In `experiments/_k4_common.py`:

(a) extend imports:

```python
from typing import Callable, NamedTuple

from bmx.cache.collect import from_matrix, load_cache, to_matrix
from bmx.cache.metrics import logit_distortion, rel_fro
from bmx.cache.rope import apply_rope
from bmx.cache.spectral import (
    SpectralBasis,
    assemble_whitener,
    fit_spectral_basis,
    identity_whitener,
    query_position_moment,
)
```

(b) move `_LayerCtx` and `_layer_ctx` VERBATIM from `experiments/k4_dec_quant.py:88-137` into `_k4_common.py` (below `_score_tail`) — the code is identical to what k4_dec_quant.py currently holds; do not edit it while moving.

(c) add the corpus-fit extraction (below `corpus_query_moment`):

```python
class CorpusFit(NamedTuple):
    """One corpus's pooled per-layer fit: basis + pooled fit matrix + whitener."""

    bases: dict[int, SpectralBasis]
    M_fits: dict[int, torch.Tensor]  # (S_total, C) fp32 — concat of fit caches
    whiteners: dict[int, tuple[torch.Tensor, torch.Tensor]]  # fp64 (Wh, Wh_inv)


def corpus_fit_bases(
    per_cache_layer_keys: list[dict[int, dict[str, torch.Tensor]]],
    get_cos_sins: list,
    rope_ready: bool,
    layers: list[int],
    *,
    w_source: str,
    ridge: float,
    position_stride: int,
) -> CorpusFit:
    """Corpus-pooled per-layer spectral fit, extracted from k4_fit_packs.main:
    Σ_k = concat of ALL corpus caches' k_pre matrices; W = pooled corpus query
    moment ("corpus") or identity ("none"). Byte-identical to the
    pre-extraction k4_fit_packs flow — pinned by
    test_k4_fit_packs_default_unchanged."""
    assert w_source in ("corpus", "none"), f"unknown w_source {w_source!r}"
    bases: dict[int, SpectralBasis] = {}
    M_fits: dict[int, torch.Tensor] = {}
    whiteners: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_i in layers:
        h_kv = d = None
        M_parts = []
        for lk in per_cache_layer_keys:
            k_pre_t = lk[layer_i]["k_pre"]
            this_h_kv, _, this_d = k_pre_t.shape
            if h_kv is None:
                h_kv, d = this_h_kv, this_d
            else:
                assert (this_h_kv, this_d) == (h_kv, d), (
                    f"corpus cache layer{layer_i}.k_pre shape "
                    f"{tuple(k_pre_t.shape)} incompatible with (h_kv={h_kv}, d={d})"
                )
            M_parts.append(to_matrix(k_pre_t))
        M_fit = torch.cat(M_parts, dim=0)
        C = h_kv * d

        if w_source == "corpus":
            W_blocks = corpus_query_moment(
                per_cache_layer_keys,
                get_cos_sins,
                rope_ready,
                layer_i,
                h_kv,
                d,
                position_stride,
            )
            Wh, Wh_inv = assemble_whitener(W_blocks, ridge=ridge)
        else:  # "none"
            Wh, Wh_inv = identity_whitener(C)

        bases[layer_i] = fit_spectral_basis(M_fit, Wh, Wh_inv)
        M_fits[layer_i] = M_fit
        whiteners[layer_i] = (Wh, Wh_inv)
        print(
            f"[layer {layer_i}] (h_kv={h_kv}, d={d}, C={C}, "
            f"S_fit={M_fit.shape[0]}) basis fit",
            flush=True,
        )
    return CorpusFit(bases=bases, M_fits=M_fits, whiteners=whiteners)
```

(d) In `experiments/k4_dec_quant.py`: delete the local `_LayerCtx`/`_layer_ctx` definitions (lines 88–137) and the now-unused `Callable, NamedTuple` imports; add `_layer_ctx` to the existing `from experiments._k4_common import (...)` block.

(e) In `experiments/k4_fit_packs.py`: replace the per-layer fit loop (lines 187–231, from `bases: dict[int, SpectralBasis] = {}` through the end of the `for layer_i in layers:` fit loop) with:

```python
    fit = corpus_fit_bases(
        per_cache_layer_keys,
        get_cos_sins,
        rope_ready,
        layers,
        w_source=cfg.w_source,
        ridge=cfg.ridge,
        position_stride=cfg.position_stride,
    )
    bases = fit.bases
    rows: list[dict] = []
    model_label = cfg.model_label or "unknown"
```

and add `corpus_fit_bases` to the `from experiments._k4_common import (...)` block (the now-unused direct imports of `assemble_whitener`/`fit_spectral_basis`/`identity_whitener`/`to_matrix`/`corpus_query_moment` and `SpectralBasis` may drop from k4_fit_packs if ruff flags them — `SpectralBasis` is still referenced in `_distortion_curves`'s signature, keep it).

- [ ] **Step 4: Run tests — the byte-identity pins are the point.**

Run: `uv run pytest tests/test_k4_experiments.py tests/test_spectral.py -q`
Expected: PASS — especially `test_k4_fit_packs_default_unchanged` (tensor-exact pin on the refactor) and both `k4_dec_quant` smokes.

- [ ] **Step 5: Battery + propose commit.**

Run: `uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: clean; `467 passed, 17 skipped, 1 xfailed`.
Stage `experiments/_k4_common.py experiments/k4_dec_quant.py experiments/k4_fit_packs.py tests/test_k4_experiments.py`; propose message:
`refactor(k4): corpus-pooled fit (corpus_fit_bases) + per-layer setup (_layer_ctx) hoisted to _k4_common — k4_fit_packs/k4_dec_quant byte-identical`
**STOP for user approval.**

---

### Task 4: Harness skeleton — plain 6-cell win matrix + §4 verdict

**Files:**
- Create: `experiments/k4_corpus_transfer.py`
- Test: `tests/test_k4_experiments.py` (append)

**Interfaces:**
- Consumes: Task 3's `corpus_fit_bases`/`CorpusFit`/`_layer_ctx`; existing `_k4_common.{DEPLOY_S, _score_tail, _tq_layer_curve, _log_interp, load_layer_keys, setup_rope}`; `bmx.cache.spectral.{pack_from_basis, spectral_quantize, skeptic_charge}`; `bmx.cache.codecs.quantize_cache`; `bmx.artifacts.{create_run, write_metrics}`.
- Produces: `Config` (fields below), `main(cfg) -> Path` (run dir), `FIT_CORPORA = ("wiki", "code", "null")`, `EVAL_CORPORA = ("wiki", "code")`, `_cell_wins(sub, tq_curves, headline_col) -> tuple[dict[str, float], bool]`, `_transfer_verdict(df, tq_df, headline_col, cfg) -> dict`. Artifacts: `metrics.parquet` (columns `model, kind, fit_corpus, w_corpus, alloc_corpus, eval_corpus, cache, layer, arm, budget, bpe_model, bpe_skeptic_deploy, c_used, rel_fro, logit, logit_rope` — `kind` is always `"k_pre"`; it exists because `_tq_layer_curve` filters on it), `tq_curve.parquet` (same columns), `corpus_transfer_verdict.json`. Tasks 5–6 extend this file; Task 7 runs it.

**BINDING (session-lead decision 3, verbatim intent):** the win metric is bits-normalized PER PACK — each pack's win = TQ curve (per eval cache + layer) log-interpolated at that pack's OWN `bpe_skeptic_deploy` (= `bpe_model + skeptic_charge(C, DEPLOY_S, tiers, c_used=pack.c_used)`), divided by that pack's tail distortion. Cross-fit win ratios are therefore fair even when packs' bpe differ slightly across fit corpora. Do NOT "fix" this into a matched-bpe constraint.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_k4_experiments.py`:

```python
def _tiny_corpus_transfer_cfg(tmp_path, budgets=(2.5,)):
    """Matched fit budgets (binding decision 1): 2 fit slices per corpus,
    1 eval cache per side, all S=128/C=16 tiny caches."""
    from experiments.k4_corpus_transfer import Config

    paths, seed = {}, 0
    for name, n in (("wf", 2), ("cf", 2), ("nf", 2), ("we", 1), ("ce", 1)):
        group = []
        for j in range(n):
            p = tmp_path / f"{name}{j}.safetensors"
            _tiny_cache(p, seed=seed)
            seed += 1
            group.append(str(p))
        paths[name] = tuple(group)
    return Config(
        wiki_fit_paths=paths["wf"],
        code_fit_paths=paths["cf"],
        null_fit_paths=paths["nf"],
        wiki_eval_paths=paths["we"],
        code_eval_paths=paths["ce"],
        model_label="tiny",
        budgets=budgets,
        group=16,
        overlap_ranks=(4, 8),
        out_root=str(tmp_path / "results"),
    )


def test_k4_corpus_transfer_smoke(tmp_path):
    import pandas as pd

    from experiments.k4_corpus_transfer import main

    run_dir = main(_tiny_corpus_transfer_cfg(tmp_path))
    df = pd.read_parquet(run_dir / "metrics.parquet")
    spec = df[df.arm == "spectral"]
    assert set(spec.fit_corpus) == {"wiki", "code", "null"}
    assert set(spec.eval_corpus) == {"wiki", "code"}
    # plain cells: fit==w==alloc corpus
    assert (spec.w_corpus == spec.fit_corpus).all()
    assert (spec.alloc_corpus == spec.fit_corpus).all()
    tq = pd.read_parquet(run_dir / "tq_curve.parquet")
    assert (tq.arm == "turboquant_mse").all()

    v = json.loads((run_dir / "corpus_transfer_verdict.json").read_text())
    assert "gpt2_yellow_flag" in v and "verdict_rule" in v
    pb = v["per_budget"]["2.5"]
    assert set(pb["D"]) == {"code->wiki", "null->wiki", "wiki->code", "null->code"}
    for cell in pb["D"].values():
        assert cell["label"] in ("insensitive", "domain-sensitive", "as-measured")
        assert cell["min"] <= cell["mean"] <= cell["max"]
    assert isinstance(pb["model_intrinsic_flag"], bool)


def test_k4_corpus_transfer_matched_fit_budget_guard(tmp_path):
    import dataclasses

    import pytest

    from experiments.k4_corpus_transfer import main

    cfg = _tiny_corpus_transfer_cfg(tmp_path)
    cfg = dataclasses.replace(cfg, code_fit_paths=cfg.code_fit_paths[:1])
    with pytest.raises(AssertionError, match="matched fit"):
        main(cfg)
```

- [ ] **Step 2: Run to verify failure.**

Run: `uv run pytest tests/test_k4_experiments.py -q -k corpus_transfer`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiments.k4_corpus_transfer'`.

- [ ] **Step 3: Implement `experiments/k4_corpus_transfer.py`.** Complete file (Tasks 5–6 extend it at the marked seams):

```python
"""K4 corpus-transfer gate: fit-corpus × eval-corpus win matrix + mechanism
decomposition (spec: docs/superpowers/specs/2026-07-23-k4-corpus-transfer-design.md).

Cells: plain matrix fit ∈ {wiki, code, null} × eval ∈ {wiki, code} (this
task); hybrid basis-A+alloc-B and W-cross Σ_A×W_B (Task 5); per-rank overlap
/ tier-agreement / spectrum / analytic cross-retention diagnostics in
overlap.parquet (Task 6). "null" = token-shuffled wikitext, fit-side only.

Win metric (BINDING — do not "fix" into a matched-bpe constraint): per
(cell, budget, eval cache, layer), win = TQ curve (turboquant_mse on k_pre,
per eval cache + layer) log-interpolated at the pack's OWN
bpe_skeptic_deploy (bpe_model + skeptic_charge(C, DEPLOY_S, tiers,
c_used=pack.c_used)) ÷ the pack's tail distortion. Bits-normalized PER PACK,
so cross-fit win ratios stay fair even when packs' bpe differ slightly.

Verdict (spec §4): D = 1 − win(fit≠eval)/win(fit=eval), computed per eval
cache (win = mean over layers), then mean/min/max across the eval caches.
D < 10% → corpus-insensitive; D > 25% → domain-sensitive; between → reported
as measured. Null-fit ≈ wikitext-fit on the wiki eval side (D < 10%) raises
model_intrinsic_flag — the stronger "basis is model-intrinsic" claim. All
gpt2-scale numbers are MECHANISM verdicts only (yellow flag in the JSON).
"""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.codecs import quantize_cache
from bmx.cache.spectral import (
    pack_from_basis,
    skeptic_charge,
    spectral_quantize,
)
from experiments._k4_common import (
    DEPLOY_S,
    CorpusFit,
    _layer_ctx,
    _log_interp,
    _score_tail,
    _tq_layer_curve,
    corpus_fit_bases,
    load_layer_keys,
    setup_rope,
)

FIT_CORPORA = ("wiki", "code", "null")
EVAL_CORPORA = ("wiki", "code")


@dataclasses.dataclass
class Config:
    wiki_fit_paths: tuple[str, ...]
    code_fit_paths: tuple[str, ...]
    null_fit_paths: tuple[str, ...]
    wiki_eval_paths: tuple[str, ...]
    code_eval_paths: tuple[str, ...]
    model_label: str = ""
    model_name: str = ""  # HF repo id for RoPE; empty => no-RoPE (gpt2), headline=logit
    budgets: tuple[float, ...] = (2.2, 2.5)
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8)
    group: int = 64
    ridge: float = 1e-3
    position_stride: int = 8
    tq_bits: tuple[int, ...] = (2, 3, 4)
    overlap_ranks: tuple[int, ...] = (8, 16, 32, 64)  # Task 6 diagnostics
    seed: int = 0
    out_root: str = ""


def _load_side(paths: tuple[str, ...], model_name: str):
    """Load caches + per-cache RoPE. Returns (per_cache_layer_keys,
    get_cos_sins, rope_ready, layers)."""
    per_cache = [load_layer_keys(p) for p in paths]
    layers = sorted(per_cache[0].keys())
    for lk in per_cache[1:]:
        assert sorted(lk.keys()) == layers, "caches disagree on layer set"
    rope_ready, get_cos_sins = False, []
    for lk in per_cache:
        ready, gcs = setup_rope(model_name, lk, layers)
        rope_ready = rope_ready or ready
        get_cos_sins.append(gcs)
    return per_cache, get_cos_sins, rope_ready, layers


def _fit_rows(per_cache, layers) -> int:
    return sum(lk[layers[0]]["k_pre"].shape[1] for lk in per_cache)


def main(cfg: Config):
    fit_paths = {
        "wiki": cfg.wiki_fit_paths,
        "code": cfg.code_fit_paths,
        "null": cfg.null_fit_paths,
    }
    for name, paths in fit_paths.items():
        assert paths, f"{name}_fit_paths must be non-empty"
    assert cfg.wiki_eval_paths and cfg.code_eval_paths, "eval paths must be non-empty"

    run = (
        create_run("k4_corpus_transfer", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_corpus_transfer", cfg)
    )
    model_label = cfg.model_label or "unknown"

    # ---- fit side: one CorpusFit per corpus, matched budgets asserted ------
    fits: dict[str, CorpusFit] = {}
    layers = None
    fit_row_counts: dict[str, int] = {}
    for name, paths in fit_paths.items():
        per_cache, get_cos_sins, rope_ready, this_layers = _load_side(
            paths, cfg.model_name
        )
        if layers is None:
            layers = this_layers
        assert this_layers == layers, f"{name} fit caches disagree on layer set"
        fit_row_counts[name] = _fit_rows(per_cache, layers)
        print(f"\n== fitting corpus {name!r} ({len(paths)} caches) ==", flush=True)
        fits[name] = corpus_fit_bases(
            per_cache,
            get_cos_sins,
            rope_ready,
            layers,
            w_source="corpus",
            ridge=cfg.ridge,
            position_stride=cfg.position_stride,
        )
    # Binding decision 1: matched fit-token budgets across corpora.
    assert len({len(p) for p in fit_paths.values()}) == 1, (
        f"matched fit budgets violated: slice counts "
        f"{ {k: len(v) for k, v in fit_paths.items()} }"
    )
    assert len(set(fit_row_counts.values())) == 1, (
        f"matched fit budgets violated: total fit rows {fit_row_counts}"
    )

    # ---- eval side: per-cache layer ctxs (built once, reused by all arms) --
    eval_paths = {"wiki": cfg.wiki_eval_paths, "code": cfg.code_eval_paths}
    # ctxs[(eval_corpus, cache_label)][layer] = _LayerCtx; rope flags per cache
    ctxs: dict[tuple[str, str], dict] = {}
    cache_rope: dict[tuple[str, str], bool] = {}
    any_rope_ready = False
    for eval_c, paths in eval_paths.items():
        for path in paths:
            label = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            layer_keys = load_layer_keys(path)
            assert sorted(layer_keys.keys()) == layers, (
                f"eval cache {label} layer set mismatch"
            )
            rope_ready, get_cos_sin = setup_rope(cfg.model_name, layer_keys, layers)
            any_rope_ready = any_rope_ready or rope_ready
            cache_rope[(eval_c, label)] = rope_ready
            ctxs[(eval_c, label)] = {
                layer_i: _layer_ctx(
                    layer_keys[layer_i],
                    rope_ready=rope_ready,
                    get_cos_sin=get_cos_sin,
                )
                for layer_i in layers
            }
    headline_col = "logit_rope" if any_rope_ready else "logit"

    rows: list[dict] = []
    tq_rows: list[dict] = []

    def emit(dest, **kw):
        base = dict(
            model=model_label,
            kind="k_pre",  # _tq_layer_curve filters on kind == "k_pre"
            fit_corpus="",
            w_corpus="",
            alloc_corpus="",
            eval_corpus="",
            cache="",
            layer=-1,
            arm="",
            budget=float("nan"),
            bpe_model=float("nan"),
            bpe_skeptic_deploy=float("nan"),
            c_used=float("nan"),
            rel_fro=float("nan"),
            logit=float("nan"),
            logit_rope=float("nan"),
        )
        base.update(kw)
        dest.append(base)
        print(
            f"  {base['arm']:16s} fit={base['fit_corpus']:4s} "
            f"eval={base['eval_corpus']:4s} cache={base['cache']:28s} "
            f"layer={base['layer']:2d} budget={base['budget']:5.2f} "
            f"logit={base['logit']:.4f}",
            flush=True,
        )

    def score_pack(pack, eval_c, label, layer_i, *, arm, fit_c, w_c, alloc_c, budget):
        ctx = ctxs[(eval_c, label)][layer_i]
        rope_ready = cache_rope[(eval_c, label)]
        assert pack.enc.shape == (ctx.C, ctx.C), (
            f"pack C mismatch at layer {layer_i}: {pack.enc.shape} vs C={ctx.C}"
        )
        M_hat, bpe_model = spectral_quantize(ctx.M_pre, pack)
        rf, lg, lg_rope = _score_tail(
            M_hat, ctx.h_kv, ctx.tail, ctx.K_post_true, ctx.Q_fp32,
            ctx.cos_l, ctx.sin_l, rope_ready, ctx.k_pre_t, ctx.M_pre,
        )
        bpe_deploy = bpe_model + skeptic_charge(
            ctx.C, DEPLOY_S, cfg.tiers, c_used=pack.c_used
        )
        emit(
            rows,
            fit_corpus=fit_c,
            w_corpus=w_c,
            alloc_corpus=alloc_c,
            eval_corpus=eval_c,
            cache=label,
            layer=layer_i,
            arm=arm,
            budget=float(budget),
            bpe_model=bpe_model,
            bpe_skeptic_deploy=bpe_deploy,
            c_used=float(pack.c_used),
            rel_fro=rf,
            logit=lg,
            logit_rope=lg_rope,
        )

    # ---- TQ baseline curves, per (eval cache, layer), computed ONCE --------
    for (eval_c, label), layer_ctxs in ctxs.items():
        rope_ready = cache_rope[(eval_c, label)]
        for layer_i, ctx in layer_ctxs.items():
            for b in cfg.tq_bits:
                M_hat_tq, bpe_tq = quantize_cache(
                    "turboquant_mse", ctx.M_pre, bits=b, seed=cfg.seed
                )
                rf, lg, lg_rope = _score_tail(
                    M_hat_tq, ctx.h_kv, ctx.tail, ctx.K_post_true, ctx.Q_fp32,
                    ctx.cos_l, ctx.sin_l, rope_ready, ctx.k_pre_t, ctx.M_pre,
                )
                emit(
                    tq_rows,
                    eval_corpus=eval_c,
                    cache=label,
                    layer=layer_i,
                    arm="turboquant_mse",
                    bpe_model=bpe_tq,
                    bpe_skeptic_deploy=bpe_tq,
                    rel_fro=rf,
                    logit=lg,
                    logit_rope=lg_rope,
                )

    # ---- plain matrix: fit ∈ FIT_CORPORA × every eval cache ----------------
    for fit_c in FIT_CORPORA:
        for budget in cfg.budgets:
            for layer_i in layers:
                pack = pack_from_basis(
                    fits[fit_c].bases[layer_i], budget,
                    tiers=cfg.tiers, group=cfg.group,
                )
                for eval_c, paths in eval_paths.items():
                    for path in paths:
                        label = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                        score_pack(
                            pack, eval_c, label, layer_i,
                            arm="spectral", fit_c=fit_c, w_c=fit_c,
                            alloc_c=fit_c, budget=budget,
                        )

    # ---- Task 5 seam: hybrid + W-cross arms appended here ------------------
    # ---- Task 6 seam: overlap.parquet diagnostics appended here ------------

    cols = [
        "model", "kind", "fit_corpus", "w_corpus", "alloc_corpus",
        "eval_corpus", "cache", "layer", "arm", "budget", "bpe_model",
        "bpe_skeptic_deploy", "c_used", "rel_fro", "logit", "logit_rope",
    ]
    df = pd.DataFrame(rows)[cols]
    tq_df = pd.DataFrame(tq_rows)[cols]
    write_metrics(run, df)
    write_metrics(run, tq_df, name="tq_curve")

    verdict = _transfer_verdict(df, tq_df, headline_col, cfg)
    (run / "corpus_transfer_verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n" + "=" * 88)
    print("CORPUS-TRANSFER VERDICT (spec §4)")
    print("=" * 88)
    print(json.dumps(verdict, indent=2))
    print(f"\nTotal rows: {len(df)}")
    print(f"-> {run}")
    return run


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def _cell_wins(
    sub: pd.DataFrame, tq_curves: dict, headline_col: str
) -> tuple[dict[str, float], bool]:
    """Per eval cache: mean over layers of tq_dist(at the pack's OWN
    bpe_skeptic_deploy) / pack distortion (bits-normalized per pack —
    binding decision 3)."""
    wins: dict[str, list[float]] = {}
    extrapolated = False
    for _, row in sub.iterrows():
        pts = tq_curves.get(row.cache, {}).get(int(row.layer))
        if not pts:
            continue
        tq_dist, ex = _log_interp(pts, float(row.bpe_skeptic_deploy))
        extrapolated = extrapolated or ex
        dist = max(float(row[headline_col]), 1e-300)
        wins.setdefault(row.cache, []).append(tq_dist / dist)
    return (
        {c: float(pd.Series(v).mean()) for c, v in wins.items()},
        extrapolated,
    )


def _transfer_verdict(
    df: pd.DataFrame, tq_df: pd.DataFrame, headline_col: str, cfg: Config
) -> dict:
    # TQ curves keyed per cache (never pooled across caches — same reasoning
    # as k4_dec_quant._dec_quant_verdict).
    tq_curves = {
        cache: _tq_layer_curve(g, headline_col) for cache, g in tq_df.groupby("cache")
    }

    per_budget: dict[str, dict] = {}
    for budget in cfg.budgets:
        cells: dict[str, dict] = {}
        for fit_c in FIT_CORPORA:
            for eval_c in EVAL_CORPORA:
                sub = df[
                    (df.arm == "spectral")
                    & (df.budget == float(budget))
                    & (df.fit_corpus == fit_c)
                    & (df.eval_corpus == eval_c)
                ]
                if sub.empty:
                    continue
                wins, ex = _cell_wins(sub, tq_curves, headline_col)
                cells[f"{fit_c}->{eval_c}"] = dict(
                    win_per_cache=wins,
                    win_mean=float(pd.Series(list(wins.values())).mean()),
                    extrapolated=bool(ex),
                )

        D: dict[str, dict] = {}
        for eval_c in EVAL_CORPORA:
            matched = cells.get(f"{eval_c}->{eval_c}")
            if matched is None:
                continue
            for fit_c in FIT_CORPORA:
                if fit_c == eval_c:
                    continue
                cross = cells.get(f"{fit_c}->{eval_c}")
                if cross is None:
                    continue
                ds = [
                    1.0 - cross["win_per_cache"][c] / matched["win_per_cache"][c]
                    for c in matched["win_per_cache"]
                    if c in cross["win_per_cache"]
                ]
                d_mean = float(pd.Series(ds).mean())
                label = (
                    "insensitive"
                    if d_mean < 0.10
                    else "domain-sensitive"
                    if d_mean > 0.25
                    else "as-measured"
                )
                D[f"{fit_c}->{eval_c}"] = dict(
                    mean=d_mean,
                    min=float(min(ds)),
                    max=float(max(ds)),
                    label=label,
                )

        null_wiki = D.get("null->wiki", {}).get("mean")
        per_budget[f"{budget:g}"] = dict(
            cells=cells,
            D=D,
            model_intrinsic_flag=bool(null_wiki is not None and null_wiki < 0.10),
        )
        # Task 5 seam: per_budget[...] gains "hybrid" and "wcross" keys.

    return dict(
        headline_metric=headline_col,
        verdict_rule="D<0.10 insensitive; D>0.25 domain-sensitive; else as-measured",
        gpt2_yellow_flag=(
            "gpt2 scale = mechanism verdict only (corpus-W retention ~0.47-0.52, "
            "docs/2026-07-15-k4-duel-results.md); Llama fit-side replication "
            "pre-registered before any paper claim"
        ),
        per_budget=per_budget,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
```

- [ ] **Step 4: Run tests.**

Run: `uv run pytest tests/test_k4_experiments.py -q -k corpus_transfer`
Expected: 2 PASS (tiny caches: 3 corpora × 2 fit + 2 eval caches, C=16, seconds).

- [ ] **Step 5: Battery + propose commit.**

Run: `uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: clean; `469 passed, 17 skipped, 1 xfailed`.
Stage `experiments/k4_corpus_transfer.py tests/test_k4_experiments.py`; propose message:
`feat(exp): k4_corpus_transfer skeleton — fit×eval win matrix incl. shuffled-null fit, per-pack bits-normalized wins, spec-§4 D verdict`
**STOP for user approval.**

---

### Task 5: Hybrid (basis-A + alloc-B) and W-cross (Σ_A × W_B) arms

**Files:**
- Modify: `src/bmx/cache/spectral.py` (`basis_alloc_moment` new; `pack_from_basis` gains `lam_alloc=None`)
- Modify: `experiments/k4_corpus_transfer.py` (arms at the Task-5 seams)
- Test: `tests/test_spectral.py` (append), `tests/test_k4_experiments.py` (append)

**Interfaces:**
- Consumes: Task 3's `CorpusFit` (`.bases`, `.M_fits`, `.whiteners`), Task 4's `score_pack`/`_cell_wins`/verdict structure; `bmx.cache.spectral.{fit_spectral_basis, key_second_moment}`.
- Produces:
  ```python
  def basis_alloc_moment(basis: SpectralBasis, M_alloc: torch.Tensor) -> torch.Tensor
      # (C,) fp64, diag(encᵀ Σ_alloc enc), clamped >= 0

  def pack_from_basis(basis, budget, *, tiers=(0, 2, 3, 4, 5, 6, 8), group=64,
                      lam_alloc: torch.Tensor | None = None) -> SpectralPack
      # lam_alloc=None reproduces prior behavior bit-exactly
  ```
  metrics.parquet arms `spectral_hybrid` (fit_corpus=basis corpus, alloc_corpus=alloc corpus, w_corpus=basis corpus, eval_corpus=alloc corpus) and `spectral_wcross` (fit_corpus=Σ corpus, w_corpus=W corpus, alloc_corpus=Σ corpus, both eval sides); verdict keys `per_budget[b]["hybrid"]` (with `recovery`, `h3_pass`) and `per_budget[b]["wcross"]`. Task 6's xretention diagnostic reuses `basis_alloc_moment`.

**Mechanism note (resolves the spec's "lam measured on the alloc corpus"):** with the basis fixed from corpus A, the per-direction waterfill input measured on corpus B is the second moment of B's keys in A's encoder coordinates: `lam_B|A = diag(enc_Aᵀ Σ_B enc_A)` — exactly `E[(M_B @ enc_A)_i²]`. `allocate_bits_from_variance` is elementwise over directions (verified: `codecs.py:155-194`, no sort assumption), so an unsorted `lam_alloc` is valid; bits stay index-aligned with `enc`'s columns.

- [ ] **Step 1: Write the failing spectral tests.** Append to `tests/test_spectral.py` (reuse the file's existing imports; add what's missing locally in the tests):

```python
def test_pack_from_basis_lam_alloc_default_unchanged():
    import torch

    from bmx.cache.spectral import (
        fit_spectral_basis,
        identity_whitener,
        pack_from_basis,
    )

    g = torch.Generator().manual_seed(0)
    M = torch.randn(128, 16, generator=g)
    Ih, Ih_inv = identity_whitener(16)
    basis = fit_spectral_basis(M, Ih, Ih_inv)
    default = pack_from_basis(basis, 2.5, group=16)
    explicit_none = pack_from_basis(basis, 2.5, group=16, lam_alloc=None)
    own_lam = pack_from_basis(basis, 2.5, group=16, lam_alloc=basis.lam64)
    assert torch.equal(default.bits, explicit_none.bits)
    assert torch.equal(default.bits, own_lam.bits)


def test_basis_alloc_moment_matches_projection_variance():
    import torch

    from bmx.cache.spectral import (
        basis_alloc_moment,
        fit_spectral_basis,
        identity_whitener,
    )

    g = torch.Generator().manual_seed(0)
    M_fit = torch.randn(64, 8, generator=g)
    M_alloc = torch.randn(96, 8, generator=g)
    Ih, Ih_inv = identity_whitener(8)
    basis = fit_spectral_basis(M_fit, Ih, Ih_inv)
    lam = basis_alloc_moment(basis, M_alloc)
    Y = M_alloc.double() @ basis.enc.double()
    ref = (Y**2).mean(dim=0)
    assert lam.dtype == torch.float64 and lam.shape == (8,)
    assert torch.allclose(lam, ref, rtol=1e-10, atol=1e-12)
    assert (lam >= 0).all()
```

- [ ] **Step 2: Run to verify failure.**

Run: `uv run pytest tests/test_spectral.py -q -k "lam_alloc or alloc_moment"`
Expected: 2 FAIL — `TypeError: pack_from_basis() got an unexpected keyword argument 'lam_alloc'` / `ImportError: basis_alloc_moment`.

- [ ] **Step 3: Implement the spectral additions.** In `src/bmx/cache/spectral.py`, change `pack_from_basis` (lines 187–205) to:

```python
def pack_from_basis(
    basis: SpectralBasis,
    budget: float,
    *,
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8),
    group: int = 64,
    lam_alloc: torch.Tensor | None = None,
) -> SpectralPack:
    """Allocate bits for one budget against an already-fit SpectralBasis.

    `lam_alloc` (fp64, (C,), index-aligned with enc's columns) substitutes the
    waterfill input — the K4 corpus-transfer hybrid path ("basis transfers,
    allocation adapts": basis from corpus A, per-direction variances measured
    on corpus B via basis_alloc_moment). Default None reproduces the prior
    behavior bit-exactly (allocates on basis.lam64).
    """
    assert 1 not in tiers, "symmetric RTN is undefined at 1 bit (qmax=0)"
    alloc_input = basis.lam64 if lam_alloc is None else lam_alloc
    assert alloc_input.shape == basis.lam64.shape, (
        f"lam_alloc shape {tuple(alloc_input.shape)} != {tuple(basis.lam64.shape)}"
    )
    bits = allocate_bits_from_variance(alloc_input, budget, tiers)
    return SpectralPack(
        enc=basis.enc,
        dec=basis.dec,
        lam=basis.lam,
        bits=bits,
        group=group,
        tiers=tuple(tiers),
        budget=float(budget),
    )
```

and add below it:

```python
def basis_alloc_moment(basis: SpectralBasis, M_alloc: torch.Tensor) -> torch.Tensor:
    """Per-direction second moments of M_alloc's rows in `basis`'s coordinate
    system: diag(encᵀ Σ_alloc enc) = E[(M_alloc @ enc)_i²], fp64 (C,), clamped
    ≥ 0. The waterfill input for the H3 hybrid (pack_from_basis lam_alloc)."""
    Sigma = key_second_moment(M_alloc)
    enc64 = basis.enc.double()
    return torch.einsum("ci,cd,di->i", enc64, Sigma, enc64).clamp_min(0.0)
```

(`fit_spectral_pack` needs no change — it delegates to `pack_from_basis` with defaults.)

- [ ] **Step 4: Run spectral tests.**

Run: `uv run pytest tests/test_spectral.py -q`
Expected: PASS (all — the default-unchanged pins in the existing suite are the regression guard).

- [ ] **Step 5: Write the failing harness test.** Append to `tests/test_k4_experiments.py`:

```python
def test_k4_corpus_transfer_hybrid_and_wcross(tmp_path):
    import pandas as pd

    from experiments.k4_corpus_transfer import main

    run_dir = main(_tiny_corpus_transfer_cfg(tmp_path))
    df = pd.read_parquet(run_dir / "metrics.parquet")

    hyb = df[df.arm == "spectral_hybrid"]
    assert set(zip(hyb.fit_corpus, hyb.alloc_corpus, hyb.eval_corpus)) == {
        ("wiki", "code", "code"),
        ("code", "wiki", "wiki"),
    }
    assert (hyb.w_corpus == hyb.fit_corpus).all()

    wx = df[df.arm == "spectral_wcross"]
    assert set(zip(wx.fit_corpus, wx.w_corpus)) == {("wiki", "code"), ("code", "wiki")}
    assert set(wx.eval_corpus) == {"wiki", "code"}
    assert (wx.alloc_corpus == wx.fit_corpus).all()

    v = json.loads((run_dir / "corpus_transfer_verdict.json").read_text())
    pb = v["per_budget"]["2.5"]
    assert set(pb["hybrid"]) == {"basis_wiki_alloc_code", "basis_code_alloc_wiki"}
    for h in pb["hybrid"].values():
        assert "recovery" in h and isinstance(h["h3_pass"], bool)
    assert len(pb["wcross"]) == 4  # 2 directions × 2 eval sides
```

Run: `uv run pytest tests/test_k4_experiments.py::test_k4_corpus_transfer_hybrid_and_wcross -q`
Expected: FAIL — no `spectral_hybrid` rows.

- [ ] **Step 6: Implement the arms.** In `experiments/k4_corpus_transfer.py`:

(a) extend the spectral import block with `basis_alloc_moment, fit_spectral_basis` and add module constants below `EVAL_CORPORA`:

```python
# (basis_corpus, alloc_corpus) — scored on the alloc corpus's eval side (H3).
_HYBRID_CELLS = (("wiki", "code"), ("code", "wiki"))
# (sigma_corpus, w_corpus) — scored on BOTH eval sides (binding decision 2).
_WCROSS_CELLS = (("wiki", "code"), ("code", "wiki"))
```

(b) replace the `# ---- Task 5 seam ...` comment in `main` with:

```python
    # ---- hybrid (H3): basis from A, lam measured on B, waterfill rerun -----
    for basis_c, alloc_c in _HYBRID_CELLS:
        for budget in cfg.budgets:
            for layer_i in layers:
                lam_alloc = basis_alloc_moment(
                    fits[basis_c].bases[layer_i], fits[alloc_c].M_fits[layer_i]
                )
                pack = pack_from_basis(
                    fits[basis_c].bases[layer_i], budget,
                    tiers=cfg.tiers, group=cfg.group, lam_alloc=lam_alloc,
                )
                for path in eval_paths[alloc_c]:
                    label = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                    score_pack(
                        pack, alloc_c, label, layer_i,
                        arm="spectral_hybrid", fit_c=basis_c, w_c=basis_c,
                        alloc_c=alloc_c, budget=budget,
                    )

    # ---- W-cross (binding decision 2): Σ from A, W (whitener) from B -------
    for sigma_c, w_c in _WCROSS_CELLS:
        for layer_i in layers:
            Wh_b, Wh_inv_b = fits[w_c].whiteners[layer_i]
            basis_x = fit_spectral_basis(
                fits[sigma_c].M_fits[layer_i], Wh_b, Wh_inv_b
            )
            for budget in cfg.budgets:
                pack = pack_from_basis(
                    basis_x, budget, tiers=cfg.tiers, group=cfg.group
                )
                for eval_c, paths in eval_paths.items():
                    for path in paths:
                        label = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                        score_pack(
                            pack, eval_c, label, layer_i,
                            arm="spectral_wcross", fit_c=sigma_c, w_c=w_c,
                            alloc_c=sigma_c, budget=budget,
                        )
```

(c) in `_transfer_verdict`, replace the `# Task 5 seam ...` comment with (inside the per-budget loop, before `per_budget[...] = dict(...)`, then add the two keys to that dict):

```python
        hybrid: dict[str, dict] = {}
        for basis_c, alloc_c in _HYBRID_CELLS:
            sub = df[
                (df.arm == "spectral_hybrid")
                & (df.budget == float(budget))
                & (df.fit_corpus == basis_c)
                & (df.alloc_corpus == alloc_c)
            ]
            matched = cells.get(f"{alloc_c}->{alloc_c}")
            if sub.empty or matched is None:
                continue
            wins, ex = _cell_wins(sub, tq_curves, headline_col)
            win_mean = float(pd.Series(list(wins.values())).mean())
            recovery = win_mean / matched["win_mean"]
            hybrid[f"basis_{basis_c}_alloc_{alloc_c}"] = dict(
                win_mean=win_mean,
                recovery=recovery,
                h3_pass=bool(recovery >= 0.9),
                extrapolated=bool(ex),
            )

        wcross: dict[str, dict] = {}
        for sigma_c, w_c in _WCROSS_CELLS:
            for eval_c in EVAL_CORPORA:
                sub = df[
                    (df.arm == "spectral_wcross")
                    & (df.budget == float(budget))
                    & (df.fit_corpus == sigma_c)
                    & (df.w_corpus == w_c)
                    & (df.eval_corpus == eval_c)
                ]
                if sub.empty:
                    continue
                wins, ex = _cell_wins(sub, tq_curves, headline_col)
                wcross[f"sigma_{sigma_c}_W_{w_c}->{eval_c}"] = dict(
                    win_mean=float(pd.Series(list(wins.values())).mean()),
                    extrapolated=bool(ex),
                )
```

and change the per-budget assignment to:

```python
        per_budget[f"{budget:g}"] = dict(
            cells=cells,
            D=D,
            model_intrinsic_flag=bool(null_wiki is not None and null_wiki < 0.10),
            hybrid=hybrid,
            wcross=wcross,
        )
```

- [ ] **Step 7: Run tests.**

Run: `uv run pytest tests/test_k4_experiments.py -q -k corpus_transfer`
Expected: 3 PASS (the Task-4 smoke still passes — its asserts are on the plain cells only).

- [ ] **Step 8: Battery + propose commit.**

Run: `uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: clean; `472 passed, 17 skipped, 1 xfailed`.
Stage `src/bmx/cache/spectral.py experiments/k4_corpus_transfer.py tests/test_spectral.py tests/test_k4_experiments.py`; propose message:
`feat(k4): hybrid basis-A+alloc-B (pack_from_basis lam_alloc + basis_alloc_moment) and W-cross Σ_A×W_B cells + H3 recovery verdict`
**STOP for user approval.**

---

### Task 6: Mechanism diagnostics — `overlap.parquet` (H1/H2 instruments)

**Files:**
- Modify: `experiments/k4_corpus_transfer.py` (diagnostics at the Task-6 seam)
- Test: `tests/test_k4_experiments.py` (append)

**Interfaces:**
- Consumes: Task 3's `CorpusFit`, Task 5's `basis_alloc_moment`; `bmx.census.subspace_overlap(A, U_ref) -> float` (U_ref must be orthonormal), `bmx.quant.hadamard.orthogonalize(M) -> Q` (QR, sign-canonicalized), `bmx.cache.spectral.fit_spectral_basis`.
- Produces: `overlap.parquet` with generic columns `kind, pair, corpus, layer, rank, budget, tier, centered, value` and kinds:
  - `overlap` — per (pair, layer, rank cutoff, centered∈{False,True}): mean squared principal cosine between the two corpora's top-r decoder subspaces (H1: high at top ranks; H2: decays with rank).
  - `tier_agreement` — per (pair, layer, budget, tier): rank-index-aligned fraction of directions assigned tier t by corpus A that corpus B also assigns t.
  - `zero_jaccard` — per (pair, layer, budget): Jaccard of the zero-bit (dropped) direction sets — the 0-vs-2-bit boundary instrument.
  - `spectrum` — per (corpus, layer, rank): eigenvalue lam64[rank] (the overlay).
  - `xretention` — per (src->dst pair, layer, budget): Gate-A-style analytic retention D_own/D_cross with the Gaussian proxy D = Σ_i lam_i·4^(−bits_i) (src basis+alloc evaluated under dst's second moment vs dst's own fit).

**Mechanism notes (resolve two spec ambiguities):** (1) bits vectors live in different bases across corpora, so tier-map agreement is computed RANK-INDEX-ALIGNED — both spectra are descending, so index i means "the i-th most important direction under that corpus's own fit"; this is the H2 instrument, stated as such in the results doc. (2) xretention's cross term is measured in the src basis's (whitened) coordinates while D_own uses dst's — commensurability caveat; it is a diagnostic, never a gate.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_k4_experiments.py`:

```python
def test_k4_corpus_transfer_diagnostics(tmp_path):
    import pandas as pd

    from experiments.k4_corpus_transfer import main

    run_dir = main(_tiny_corpus_transfer_cfg(tmp_path))
    ov = pd.read_parquet(run_dir / "overlap.parquet")
    assert {"overlap", "tier_agreement", "zero_jaccard", "spectrum", "xretention"} <= (
        set(ov.kind)
    )

    o = ov[ov.kind == "overlap"]
    assert set(o.pair) == {"wiki-code", "wiki-null", "code-null"}
    assert set(o["rank"]) == {4, 8}  # cfg.overlap_ranks, both <= C=16
    assert set(o.centered) == {True, False}
    assert ((o.value >= -1e-9) & (o.value <= 1 + 1e-9)).all()

    t = ov[ov.kind == "tier_agreement"]
    assert ((t.value >= 0) & (t.value <= 1)).all()
    assert (t.tier >= 0).all()

    s = ov[ov.kind == "spectrum"]
    assert set(s.corpus) == {"wiki", "code", "null"}
    # descending spectra per (corpus, layer)
    for _, g in s.groupby(["corpus", "layer"]):
        vals = g.sort_values("rank").value.to_numpy()
        assert (vals[:-1] >= vals[1:] - 1e-12).all()

    x = ov[ov.kind == "xretention"]
    # nothing is evaluated UNDER the null's covariance (fit-side only)
    assert all(not p.endswith("->null") for p in x.pair)
    assert (x.value > 0).all()
```

- [ ] **Step 2: Run to verify failure.**

Run: `uv run pytest tests/test_k4_experiments.py::test_k4_corpus_transfer_diagnostics -q`
Expected: FAIL — `FileNotFoundError: overlap.parquet`.

- [ ] **Step 3: Implement.** In `experiments/k4_corpus_transfer.py`:

(a) add imports: `from bmx.census import subspace_overlap` and `from bmx.quant.hadamard import orthogonalize`.

(b) add module-level helpers (below `_WCROSS_CELLS`):

```python
_PAIRS = (("wiki", "code"), ("wiki", "null"), ("code", "null"))


def _rank_overlap(dec_a: torch.Tensor, dec_b: torch.Tensor, r: int) -> float:
    """Mean squared principal cosine between the top-r reconstruction
    subspaces span(dec[:, :r]) of two fits, in [0, 1]."""
    return subspace_overlap(
        dec_a[:, :r].double(), orthogonalize(dec_b[:, :r].double())
    )


def _proxy_distortion(lam64: torch.Tensor, bits: torch.Tensor) -> float:
    """Gaussian rate-distortion proxy Σ_i lam_i · 4^(−bits_i) (same form as
    k4_fit_packs._distortion_curves)."""
    return float((lam64 * torch.pow(4.0, -bits.double())).sum())


def _diagnostics(
    fits: dict[str, CorpusFit], layers: list[int], cfg: Config
) -> pd.DataFrame:
    from bmx.cache.spectral import fit_spectral_basis

    rows: list[dict] = []

    def emit(**kw):
        base = dict(
            kind="", pair="", corpus="", layer=-1, rank=-1,
            budget=float("nan"), tier=-1, centered=False, value=float("nan"),
        )
        base.update(kw)
        rows.append(base)

    C = fits["wiki"].bases[layers[0]].lam64.numel()
    ranks = [r for r in cfg.overlap_ranks if r <= C]
    assert ranks, f"no overlap_ranks <= C={C}"

    # Centered refits (Cov(k) instead of E[kkᵀ]), same whitener — H1 probe.
    centered_bases: dict[str, dict[int, object]] = {}
    for corpus, fit in fits.items():
        centered_bases[corpus] = {}
        for layer_i in layers:
            M = fit.M_fits[layer_i]
            Wh, Wh_inv = fit.whiteners[layer_i]
            centered_bases[corpus][layer_i] = fit_spectral_basis(
                M - M.mean(dim=0, keepdim=True), Wh, Wh_inv
            )

    for a, b in _PAIRS:
        for layer_i in layers:
            for r in ranks:
                emit(
                    kind="overlap", pair=f"{a}-{b}", layer=layer_i, rank=r,
                    centered=False,
                    value=_rank_overlap(
                        fits[a].bases[layer_i].dec, fits[b].bases[layer_i].dec, r
                    ),
                )
                emit(
                    kind="overlap", pair=f"{a}-{b}", layer=layer_i, rank=r,
                    centered=True,
                    value=_rank_overlap(
                        centered_bases[a][layer_i].dec,
                        centered_bases[b][layer_i].dec,
                        r,
                    ),
                )

    # Tier-map agreement, rank-index-aligned (both spectra descending).
    for a, b in _PAIRS:
        for layer_i in layers:
            for budget in cfg.budgets:
                bits_a = pack_from_basis(
                    fits[a].bases[layer_i], budget, tiers=cfg.tiers, group=cfg.group
                ).bits
                bits_b = pack_from_basis(
                    fits[b].bases[layer_i], budget, tiers=cfg.tiers, group=cfg.group
                ).bits
                for tier in cfg.tiers:
                    mask = bits_a == tier
                    if mask.any():
                        emit(
                            kind="tier_agreement", pair=f"{a}-{b}", layer=layer_i,
                            budget=float(budget), tier=int(tier),
                            value=float((bits_b[mask] == tier).float().mean()),
                        )
                za, zb = bits_a == 0, bits_b == 0
                union = int((za | zb).sum())
                emit(
                    kind="zero_jaccard", pair=f"{a}-{b}", layer=layer_i,
                    budget=float(budget),
                    value=float((za & zb).sum()) / union if union else float("nan"),
                )

    # Eigenvalue-spectrum overlay.
    for corpus, fit in fits.items():
        for layer_i in layers:
            lam = fit.bases[layer_i].lam64
            for r in range(lam.numel()):
                emit(
                    kind="spectrum", corpus=corpus, layer=layer_i, rank=r,
                    value=float(lam[r]),
                )

    # Analytic cross-corpus retention (Gate-A machinery pointed across
    # corpora): src basis+alloc under dst's covariance vs dst's own fit.
    # Never INTO null (nothing is evaluated on shuffled text, spec §2).
    for a, b in _PAIRS:
        for src, dst in ((a, b), (b, a)):
            if dst == "null":
                continue
            for layer_i in layers:
                for budget in cfg.budgets:
                    pack_src = pack_from_basis(
                        fits[src].bases[layer_i], budget,
                        tiers=cfg.tiers, group=cfg.group,
                    )
                    lam_dst_given_src = basis_alloc_moment(
                        fits[src].bases[layer_i], fits[dst].M_fits[layer_i]
                    )
                    D_cross = _proxy_distortion(lam_dst_given_src, pack_src.bits)
                    pack_dst = pack_from_basis(
                        fits[dst].bases[layer_i], budget,
                        tiers=cfg.tiers, group=cfg.group,
                    )
                    D_own = _proxy_distortion(
                        fits[dst].bases[layer_i].lam64, pack_dst.bits
                    )
                    emit(
                        kind="xretention", pair=f"{src}->{dst}", layer=layer_i,
                        budget=float(budget),
                        value=D_own / max(D_cross, 1e-300),
                    )

    cols = ["kind", "pair", "corpus", "layer", "rank", "budget", "tier",
            "centered", "value"]
    return pd.DataFrame(rows)[cols]
```

(c) replace the `# ---- Task 6 seam ...` comment in `main` with:

```python
    ov_df = _diagnostics(fits, layers, cfg)
    write_metrics(run, ov_df, name="overlap")
    print(f"overlap.parquet: {len(ov_df)} diagnostic rows", flush=True)
```

- [ ] **Step 4: Run tests.**

Run: `uv run pytest tests/test_k4_experiments.py -q -k corpus_transfer`
Expected: 4 PASS.

- [ ] **Step 5: Battery + propose commit.**

Run: `uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: clean; `473 passed, 17 skipped, 1 xfailed`.
Stage `experiments/k4_corpus_transfer.py tests/test_k4_experiments.py`; propose message:
`feat(exp): k4_corpus_transfer mechanism diagnostics — per-rank subspace overlap (centered/uncentered), rank-aligned tier agreement + zero-set Jaccard, spectrum overlay, analytic cross-corpus retention`
**STOP for user approval.**

---

### Task 7: The real gpt2 run + results doc

**Files:**
- Create: `docs/2026-07-23-k4-corpus-transfer-results.md`
- Commit (new artifacts only): `results/k4_corpus_transfer/<run-id>/{config.json,env.json,metrics.parquet,tq_curve.parquet,overlap.parquet,corpus_transfer_verdict.json}`

**Interfaces:**
- Consumes: Task 2's collected caches; Tasks 4–6's `experiments/k4_corpus_transfer.py`.
- Produces: the committed run dir + the results doc Task 8 verifies.

- [ ] **Step 1: Run the harness (local CPU, ~10–20 min for 12 layers × C=768).** From `/d/Projects/bmx` (tyro tuples are space-separated; `--model-name` stays empty — gpt2 has no RoPE, headline = `logit`):

```bash
uv run python experiments/k4_corpus_transfer.py \
  --wiki-fit-paths results/cache/gpt2_1024_off1024.safetensors results/cache/gpt2_1024_off2048.safetensors results/cache/gpt2_1024_off3072.safetensors results/cache/gpt2_1024_off4096.safetensors \
  --code-fit-paths results/cache/gpt2_1024_code_off1024.safetensors results/cache/gpt2_1024_code_off2048.safetensors results/cache/gpt2_1024_code_off3072.safetensors results/cache/gpt2_1024_code_off4096.safetensors \
  --null-fit-paths results/cache/gpt2_1024_shuf_off1024.safetensors results/cache/gpt2_1024_shuf_off2048.safetensors results/cache/gpt2_1024_shuf_off3072.safetensors results/cache/gpt2_1024_shuf_off4096.safetensors \
  --wiki-eval-paths results/cache/gpt2_1024.safetensors results/cache/gpt2_1024_off5120.safetensors \
  --code-eval-paths results/cache/gpt2_1024_code.safetensors results/cache/gpt2_1024_code_off5120.safetensors \
  --model-label gpt2 --budgets 2.2 2.5
```

Expected: per-corpus `[layer i] ... basis fit` lines, per-row score lines, then the `CORPUS-TRANSFER VERDICT` JSON and `-> results/k4_corpus_transfer/<run-id>`. (Fit-side note: 4×1024 = 4096 fit rows per corpus, matched — the assert passes by construction. Eval sides: 2 caches each.)

- [ ] **Step 2: Sanity-check the verdict against the parquets (independent recomputation of one cell).**

```bash
uv run python - <<'EOF'
import json, sys
from pathlib import Path
import pandas as pd
from experiments._k4_common import _log_interp, _tq_layer_curve

run = sorted(Path("results/k4_corpus_transfer").iterdir())[-1]
v = json.loads((run / "corpus_transfer_verdict.json").read_text())
df = pd.read_parquet(run / "metrics.parquet")
tq = pd.read_parquet(run / "tq_curve.parquet")
hl = v["headline_metric"]
curves = {c: _tq_layer_curve(g, hl) for c, g in tq.groupby("cache")}
sub = df[(df.arm == "spectral") & (df.budget == 2.5)
         & (df.fit_corpus == "code") & (df.eval_corpus == "wiki")]
wins = {}
for _, r in sub.iterrows():
    tqd, _ = _log_interp(curves[r.cache][int(r.layer)], float(r.bpe_skeptic_deploy))
    wins.setdefault(r.cache, []).append(tqd / max(float(r[hl]), 1e-300))
recomputed = sum(sum(v_) / len(v_) for v_ in wins.values()) / len(wins)
stored = v["per_budget"]["2.5"]["cells"]["code->wiki"]["win_mean"]
assert abs(recomputed - stored) < 1e-9, (recomputed, stored)
ex = any(c["extrapolated"] for c in v["per_budget"]["2.5"]["cells"].values())
print("OK  code->wiki win_mean", stored, " extrapolated_anywhere:", ex)
print(json.dumps(v["per_budget"]["2.5"]["D"], indent=2))
EOF
```

Expected: `OK ...` then the D table. If `extrapolated_anywhere: True`, note which cells in the results doc (§4-rule numbers stand, but flag it).

- [ ] **Step 3: Write `docs/2026-07-23-k4-corpus-transfer-results.md`.** Full skeleton (fill every `⟨…⟩` from the verdict JSON / parquets — the `⟨…⟩` markers are number slots in the deliverable doc, not plan placeholders; delete whichever §6 template the measured D rules out):

```markdown
# K4 corpus-transfer gate — results (gpt2 mechanism scale)

Spec: `docs/superpowers/specs/2026-07-23-k4-corpus-transfer-design.md`.
Run: `results/k4_corpus_transfer/⟨run-id⟩` (git SHA ⟨sha⟩). Harness:
`experiments/k4_corpus_transfer.py`. Fit budgets MATCHED by construction:
4 slices × 1024 tokens per corpus (offsets 1024/2048/3072/4096); eval = 2
held-out slices × 1024 per natural corpus (offsets 0, 5120). Shuffle-null
seed 20260723 (post-slice permutation, per-slice generator seed
`20260723 + offset`). Code corpus: ⟨bigcode/the-stack-smol data/python |
codeparrot fallback — record which⟩. Budgets 2.2 / 2.5; headline metric
`logit` (gpt2, no RoPE); win = per-pack bits-normalized TQ-curve ratio at
each pack's OWN bpe_skeptic_deploy.

**YELLOW FLAG (every table below):** gpt2 scale = mechanism verdict only
(corpus-W retention ~0.47–0.52, `docs/2026-07-15-k4-duel-results.md`);
Llama fit-side replication is pre-registered (plan addendum) before any
paper claim.

## 1. Win matrix (win_mean; gpt2 mechanism scale — see yellow flag)

| fit \ eval | wiki-held | code-held |
|---|---|---|
| wiki | ⟨⟩ / ⟨⟩ (b2.2/b2.5) | ⟨⟩ / ⟨⟩ |
| code | ⟨⟩ / ⟨⟩ | ⟨⟩ / ⟨⟩ |
| null (shuffled) | ⟨⟩ / ⟨⟩ | ⟨⟩ / ⟨⟩ |

## 2. Cross-fit degradation D = 1 − win(cross)/win(matched) (§4 rules; gpt2 — see yellow flag)

| cell | b2.2 mean [min,max] | b2.5 mean [min,max] | label |
|---|---|---|---|
| code→wiki | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| wiki→code | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| null→wiki | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| null→code | ⟨⟩ | ⟨⟩ | ⟨⟩ |

Rule: D < 10% → corpus-insensitive; D > 25% → domain-sensitive; between →
as measured. model_intrinsic_flag (null≈wiki on wiki eval): ⟨true/false⟩.

## 3. Hybrid (H3) + W-cross (gpt2 — see yellow flag)

| arm | cell | win_mean | recovery vs matched | h3_pass (≥0.9) |
|---|---|---|---|---|
| hybrid | basis wiki + alloc code → code | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| hybrid | basis code + alloc wiki → wiki | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| W-cross | Σ wiki × W code → wiki / → code | ⟨⟩ / ⟨⟩ | — | — |
| W-cross | Σ code × W wiki → wiki / → code | ⟨⟩ / ⟨⟩ | — | — |

## 4. Mechanism diagnostics (overlap.parquet; gpt2 — see yellow flag)

- Per-rank subspace overlap (pair mean over layers): r=8 ⟨⟩, r=16 ⟨⟩,
  r=32 ⟨⟩, r=64 ⟨⟩ for wiki–code; wiki–null ⟨⟩…; code–null ⟨⟩…
  → H2 check: does divergence grow with rank? ⟨yes/no + numbers⟩
- Centered vs uncentered at r=16 (wiki–code): uncentered ⟨⟩ vs centered ⟨⟩
  → H1 check: agreement drop when the mean/rogue component is removed? ⟨⟩
- Tier agreement: top tiers (≥4 bits) ⟨⟩ vs zero-set Jaccard ⟨⟩ (b2.5,
  layer-mean) → H2's low-tier-boundary prediction: ⟨⟩
- Analytic cross-retention (D_own/D_cross, b2.5 layer-mean): wiki→code ⟨⟩,
  code→wiki ⟨⟩, null→wiki ⟨⟩, null→code ⟨⟩ (diagnostic only — the cross
  term is measured in the src basis's whitened coordinates).

## 5. WHY (mechanism)

⟨Analytic decomposition paragraph: tie §4's overlap-vs-rank, centered-drop,
and tier-boundary numbers to H1/H2/H3 — which hypothesis the data supports
and which it kills, with the specific numbers.⟩

<!-- VAULT PASS (controller, at writeup time): grounded-prior paragraph —
massive activations / attention sinks / rogue channels are input-agnostic
model artifacts, which predicts corpus-insensitive top subspaces. Cite
vault anchors via the personal-brain skill (VQ distortion objectives,
rogue-channel/two-stage-quantization notes). The plan deliberately leaves
this slot empty; do NOT fill it from training data. -->

## 6. Verdict

### Template A — corpus-insensitive (use if all natural-cross D < 10%)
Corpus choice is second-order for spectral-pack fitting at gpt2 scale:
cross-fit costs ⟨D range⟩ (vs the ~40% sequence-level ceiling from Gate A,
`docs/2026-07-12-k4-stage01-results.md`). Null-fit costs ⟨⟩ — ⟨if <10%:
"the basis is (nearly) purely model-intrinsic — token-level + architecture
geometry, not contextual semantics; this is the stronger claim and leads."⟩
Referee answer: one wikitext-fit pack ships for all domains; here is the
matrix. Llama replication pre-registered before the paper states this.

### Template B — domain-sensitive (use if any natural-cross D > 25%)
Fit corpus matters: ⟨cell⟩ degrades ⟨D⟩. The exploitation lever is decided
by H3: ⟨if h3_pass: "hybrid recovery ⟨⟩ ≥ 0.9 — ONE shared basis + per-
domain tier maps (~3·C bits per domain, no basis refit) captures the win;
deployment = shared basis + domain alloc."⟩ ⟨else: "hybrid recovery ⟨⟩ <
0.9 — allocation transfer is insufficient; the lever is whole-pack
per-domain fitting."⟩ W-cross localizes the sensitivity to ⟨Σ|W⟩:
⟨numbers⟩. Llama replication pre-registered before the paper states this.

## 7. VM addendum (pre-registered — rides the next rental)

See the plan's "VM addendum" section
(`docs/superpowers/plans/2026-07-23-k4-corpus-transfer.md`): Llama-Instruct
fit-side replication (same matrix, matched budgets at S=2048), plus the
OPTIONAL LongBench-code probe cell (n=100, paired vs the wikitext-fit arm)
ONLY if H3 confirms here and there.
```

- [ ] **Step 4: Fill every `⟨…⟩` from the run's verdict/parquets; delete the non-applicable §6 template** (keep both ONLY if the D cells straddle the 10–25% band — then retitle §6 "as measured" and report both levers as open).

- [ ] **Step 5: Battery + propose commit.**

Run: `uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: clean; `473 passed, 17 skipped, 1 xfailed` (no code changes in this task).
Stage `docs/2026-07-23-k4-corpus-transfer-results.md results/k4_corpus_transfer/<run-id>/` (parquets + json only — never the raw caches); propose message (pick the measured verdict word):
`docs(k4): corpus-transfer gate results — fit×eval matrix + H1/H2/H3 diagnostics; verdict <insensitive|domain-sensitive|as-measured> at gpt2 mechanism scale`
**STOP for user approval.**

---

### Task 8: Verification gate + push proposal

**Files:** none created — verification only.

**Interfaces:** consumes everything above; produces the go/no-go for pushing the branch.

- [ ] **Step 1: Full battery from clean state.**

Run: `cd /d/Projects/bmx && uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: format makes no changes; check clean; `473 passed, 17 skipped, 1 xfailed` (baseline 461 + 12 new: Task 1 ×3, Task 2 ×1, Task 3 ×2, Task 4 ×2, Task 5 ×3, Task 6 ×1).

- [ ] **Step 2: Confirm nothing outside the plan's file list changed.**

Run: `git status --short` and `git log --oneline 023d2e8..HEAD`
Expected: only the files named in "File structure" + the Task-7 run dir/doc are in the commits; `results/cache/*.safetensors` untracked (gitignored); no edits to shipped pack files, old parquets, or `docs/2026-07-15-k4-duel-results.md`.

- [ ] **Step 3: Spot-verify the byte-identity guarantees one last time.**

Run: `uv run pytest tests/test_k4_experiments.py::test_k4_fit_packs_default_unchanged tests/test_cache_collect.py::test_load_eval_tokens_offset tests/test_spectral.py -q`
Expected: PASS — the Task-3 refactor and Task-1/5 default paths are pinned.

- [ ] **Step 4: Propose the push.**

Propose: `git push origin feat/triton-decode-kernel` — **STOP for user approval** (transport discipline: the VM pulls this branch for the addendum).

---

## VM addendum — Llama fit-side replication (PRE-REGISTERED; unnumbered, rides the next rental; per `vm-interaction-guide` + `vm-longrun-discipline` memories)

Registered NOW so the analysis choices cannot drift after seeing gpt2 numbers:

- **A1 — collect Llama caches** (VM, GH200): `meta-llama/Llama-3.1-8B-Instruct`, `--seq-len 2048`, matched budgets scaled to the existing Llama artifact shape (`results/cache/llama-3.1-8b_2048.safetensors` is off0): per corpus 4 fit slices × 2048 tokens (offsets 2048/4096/6144/8192) + 2 heldout × 2048 (offsets 0, 10240); corpora wiki (defaults), code (same `--dataset-id/--data-dir/--split/--text-field --corpus-label code` as Task 2, or the recorded fallback), shuf (`--shuffle-seed 20260723 --corpus-label shuf`, fit-side only).
- **A2 — run the same matrix**: `experiments/k4_corpus_transfer.py` with the A1 paths, `--model-label llama-3.1-8b-instruct --model-name meta-llama/Llama-3.1-8B-Instruct --budgets 2.2 2.5` (RoPE active → headline `logit_rope`). Same §4 rules, same verdict JSON. No task evals needed.
- **A3 — OPTIONAL LongBench-code probe cell** (ONLY if H3 confirms at BOTH scales): one n=100 paired LongBench-code run, code-fit or hybrid pack vs the shipped wikitext-fit arm — decides whether the domain-alloc lever shows up at task level.
- Transport: git bundle back (VM has no push creds); commit parquets + verdict JSON, never caches; append a Llama section to `docs/2026-07-23-k4-corpus-transfer-results.md` and only then remove the yellow-flag qualifier from any claim the replication supports.

---

## Self-Review (run after writing — findings fixed inline)

**1. Spec coverage (every spec § mapped to a task):**

- §1 hypotheses — H1: Task 6 (centered-vs-uncentered + per-rank overlap); H2: Task 6 (overlap-vs-rank, tier agreement, zero-Jaccard); H3: Task 5 (hybrid arm + recovery verdict). ✓
- §2 corpora — wikitext (existing caches reused unchanged), code (Task 2 collection, binding-5 id + assert + fallback), shuffled null fit-side-only ONE seed post-slice (Tasks 1–2, binding 4); eval = wiki-held + code-held 2×1024 each (Task 2 / Task 7 CLI). ✓
- §3 cells — win matrix: Task 4; hybrid: Task 5; W-cross (binding 2): Task 5; diagnostics (per-rank angles, centered/uncentered, tier agreement + spectrum overlay, cross-corpus retention): Task 6. ✓
- §4 verdict rules — D thresholds 10%/25%, min/max error bars, null-vs-natural comparison + `model_intrinsic_flag`, H3 ≥ 0.9, gpt2 yellow flag in JSON and every doc caption: Tasks 4–5, 7. ✓
- §5 deliverables — (1) `load_eval_tokens` + `collect_cache`: Tasks 1–2; (2) harness + three artifacts: Tasks 4–6; (3) local cache collection, names encode corpus, not committed: Task 2; (4) results doc with both templates + WHY + vault slot: Task 7 (binding 8); (5) VM addendum pre-registered: unnumbered section. ✓
- §6 non-goals — no new cache-spec/codec FIELDS (the `lam_alloc` kwarg + `basis_alloc_moment` are fitting-path additions mandated by §3's hybrid arm, default-inert, pinned by `test_pack_from_basis_lam_alloc_default_unchanged`); no gpt2 task headline; no third corpus; cross-layer questions untouched. ✓
- §7 constraints — battery before every commit (461/17/1 baseline), tyro, fp32/fp64 discipline, explicit run selection (Task 7 Step 2 reads the newest run it just created, by sorted run-id), byte-identical default path (Task 1 pin + Task 2 naming pin + reuse of existing caches), deterministic seeds (shuffle 20260723 recorded; tq seed=cfg.seed). ✓

**2. Placeholder scan:** no TBD/TODO/"add error handling"/"similar to Task N" anywhere; every code step shows the code; the `⟨…⟩` markers appear ONLY inside the Task-7 results-doc skeleton, where they are the deliverable's number slots to be filled from the measured run (explicitly instructed in Task 7 Step 4) — not implementation placeholders. ✓

**3. Type consistency:** `CorpusFit(bases, M_fits, whiteners)` defined in Task 3 = consumed in Tasks 4–6 (`fits[c].bases[layer]`, `.M_fits[layer]`, `.whiteners[layer]` → `(Wh, Wh_inv)` tuple order matches `fit_spectral_basis(M, Wh, Wh_inv)`). `pack_from_basis(..., lam_alloc=None)` (Task 5) matches Task 5/6 call sites; `basis_alloc_moment(basis, M_alloc) -> (C,) fp64` matches both consumers. `_layer_ctx(kinds_map, *, rope_ready, get_cos_sin)` keyword-only signature identical to today's `k4_dec_quant.py` definition (moved verbatim) and to the Task-4/6 call sites. Parquet columns emitted by `emit` == the `cols` list == what tests and Task 7's recomputation read (`kind/fit_corpus/w_corpus/alloc_corpus/eval_corpus/cache/layer/arm/budget/bpe_model/bpe_skeptic_deploy/c_used/rel_fro/logit/logit_rope`). Verdict keys asserted in tests (`per_budget[b]["D"|"cells"|"hybrid"|"wcross"|"model_intrinsic_flag"]`, cell names `fit->eval`, hybrid names `basis_X_alloc_Y`) match `_transfer_verdict`'s construction. Test-count arithmetic re-checked: 461 + 3 + 1 + 2 + 2 + 3 + 1 = 473. ✓

**Issues found & fixed during review:** (a) the pre-existing `test_load_eval_tokens_offset` monkeypatches a plain dict as the dataset — the new column assert therefore reads `getattr(ds, "column_names", None) or list(ds.keys())` (Task 1 Step 3 carries this); (b) tier-map bits vectors live in different bases across corpora — comparison made rank-index-aligned and documented as the H2 instrument (Task 6 mechanism note); (c) `_tiny_corpus_transfer_cfg` passes `overlap_ranks=(4, 8)` because the tiny fixture's C=16 would leave the default (8, 16, 32, 64) half-filtered — the diagnostics test asserts exactly {4, 8}; (d) `_tq_layer_curve` filters on `kind == "k_pre"`, so the harness schema carries a constant `kind="k_pre"` column (Task 4) — without it the verdict would silently interpolate empty curves.

