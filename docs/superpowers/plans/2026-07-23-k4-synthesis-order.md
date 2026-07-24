# K4 Synthesis-Order Extension (Stage 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks 1–3 are local CPU TDD code; Task 4 is the real gpt2 run + results-doc §8 + verification/push. The Llama synthesis rider is an unnumbered VM NOTE at the end — it rides the pre-registered A1/A2 replication, NOT this plan's execution.

**Binding spec (read first):** `docs/superpowers/specs/2026-07-23-k4-corpus-transfer-design.md` **§3b "Synthesis-order addendum"**. Parent plan (Global Constraints inherited): `docs/superpowers/plans/2026-07-23-k4-corpus-transfer.md`. Branch `feat/triton-decode-kernel`, plan written at HEAD `a47aeff`. Stage-1 verdict on record: `docs/2026-07-23-k4-corpus-transfer-results.md` (run `20260723-190823-8dced47`) — D(shuf_wiki→wiki) ≈ 9.4%, `model_intrinsic_flag: true`, wiki↔code domain-sensitive 45–58%, H3 killed.

**Goal:** Measure whether a calibration stream SYNTHESIZED from a traffic token histogram (order 1 = unigram sampling, order 2 = bigram Markov chain) recovers the matched-fit pack win — five new fit-side arms (shuf_code / uni_wiki / uni_code / bi_wiki / bi_code) in the same fit×eval matrix, with the spec §3b pre-registered order-ladder rules (a)/(b) as gates for the RECIPE claim.

**Architecture:** A seeded sampler `synth_stream` (unigram = i.i.d. WITH replacement from the window's empirical histogram; bigram = empirical Markov chain with unigram backoff) is added to `bmx.eval.layer_swap` and wired through `load_eval_tokens` as a default-inert `synth`/`synth_seed` pair — applied AFTER the natural window is sliced, replacing it with a same-length synthetic stream. `collect_cache.py` passes the knobs through with pinned corpus labels. `experiments/k4_corpus_transfer.py` gains five fit-only path tuples (empty = Stage-1 behavior unchanged), scores every present fit arm on both eval sides with the existing per-pack bits-normalized win, and emits a `synthesis` block (rules a/b, pass/fail booleans) in `corpus_transfer_verdict.json`. Task 4 reruns the FULL 8-arm matrix so all cells share one run-id, verifies Stage-1 cells reproduce, and appends §8 to the results doc.

**Tech Stack:** PyTorch (fp32 experiments), transformers 5.11, HF `datasets`, tyro, safetensors, pandas/parquet, pytest. All local CPU (gpt2); no CUDA anywhere in this plan.

---

## Global Constraints

**Repo hard rules (CLAUDE.md):**

- **NEVER `git commit` without the user's explicit approval.** Stage, propose the task's exact message, STOP. No AI attribution ever.
- Before any commit: `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` — all clean, then re-stage. **Battery baseline at `a47aeff`: 480 passed / 17 skipped / 1 xfailed.**
- Use the Bash tool (git bash); `cd /d/Projects/bmx` first in fresh shells. No new dependencies.
- fp32 in experiments (caches fp16); shape asserts at boundaries; tiny offline fixtures only in tests — never download in tests. No web search (HF dataset downloads are fine).
- Artifacts under `results/<exp>/<run-id>/`; commit metrics/verdict parquet+json, never raw caches (`results/cache/*` is gitignored, regenerable).

**§3b binding decisions (verbatim intent, non-negotiable):**

1. **MATCHED FIT-TOKEN BUDGETS:** every new arm uses the Stage-1 shape exactly — **4 fit slices × 1024 tokens (offsets 1024/2048/3072/4096)**. Fit-side ONLY: nothing is ever evaluated on synthetic or shuffled text. The harness's matched-budget guard (slice count + total fit rows) extends over the new arms.
2. **ONE recorded seed for every derived stream: 20260723** — shuffle arms use `shuffle_seed=20260723` (per-slice generator `20260723 + token_offset`, identical to the Stage-1 wiki null), synthesis arms use `synth_seed=20260723` (per-slice generator `20260723 + token_offset`). Deterministic, recorded in config.json by construction.
3. **Per-slice statistics:** each synthetic slice is sampled from the histogram/transitions of exactly the natural slice it replaces (the window at its own offset), mirroring the shuffle null's per-slice permutation — so each arm differs from the natural arm ONLY in the order statistics of the same windows. `shuf_*` = sampling WITHOUT replacement (permutation); `uni_*` = sampling WITH replacement (the traffic-histogram recipe); spec §3b names `uni_*` the recipe estimator and `shuf_*` its control.
4. **Corpus labels pinned:** `shufcode`, `uniwiki`, `unicode`, `biwiki`, `bicode` (no underscores — they compose as `f"uni{eval_side}"` / `f"bi{eval_side}"` in the verdict rules). Cache names follow the existing scheme: `gpt2_1024_<label>_off<N>.safetensors`.
5. **Code source for all code-derived arms = the Stage-1 recorded corpus:** `codeparrot/codeparrot-clean-valid`, `--dataset-config ""`, `--split "train[:200]"`, `--text-field content` (`bigcode/the-stack-smol` is Hub-gated from this environment — results doc, header). The shuf/uni/bi code windows MUST derive from the SAME natural code windows as the Stage-1 code fit slices.
6. **Pre-registered order-ladder rules (spec §3b, verbatim):** for each eval side E ∈ {wiki, code}: **(a) recipe-confirmed for E if D(uni_E→E) < 10%**; **(b) order-2 earns its keep if [D(uni_E→E) − D(bi_E→E)] ≥ ½·D(uni_E→E)** — otherwise unigram is the final recipe and order is dead at 2nd order too (report the privacy note: histograms, not texts). **No higher orders unless (b) passes at BOTH eval sides.** These a/b fields are gates for the RECIPE claim; relative to the parent plan's §4 fields they are additive (report the existing fields unchanged).
7. **gpt2 yellow flag** on every new table/JSON field ("gpt2 mechanism scale — corpus-W retention ~0.47–0.52, `docs/2026-07-15-k4-duel-results.md`; Llama fit-side replication pre-registered").
8. **Defaults byte-identical:** `load_eval_tokens` default path is pinned by the existing recorder/offset tests (`test_load_eval_tokens_generalized_defaults`, `test_load_eval_tokens_offset`, the capped-accumulation pins) — the synth params must be default-inert. The shuffle path is NOT touched.
9. **Stage-1 reproduction check:** the Task-4 full-matrix rerun must reproduce Stage-1 cell values (deterministic pipeline, same caches, same seeds). A mismatch is a STOP-and-diagnose (superpowers:systematic-debugging), never a shrug.

---

## File structure

- Modify `src/bmx/eval/layer_swap.py` — new `synth_stream(window, mode, seed)` + `load_eval_tokens` gains `synth: str = ""`, `synth_seed: int = -1` (default-inert; validation asserts fire before any download).
- Modify `experiments/collect_cache.py` — `Config` gains `synth`/`synth_seed`; `_WIKI_DEFAULTS`/`_corpus_is_default` extended; passthrough to `load_eval_tokens`; docstring example.
- Modify `experiments/k4_corpus_transfer.py` — five fit-only path tuples (default `()`), `SYNTH_FIT_CORPORA`, all-or-nothing + matched-budget guards, plain-matrix loop over all present fit arms, `_transfer_verdict` derives fit corpora from the df and emits the `synthesis` block; `_diagnostics` centered-refit loop restricted to `_PAIRS` corpora (spectrum overlay picks up the new arms automatically; `_PAIRS` overlap/tier/xretention stay 3-corpus — report-only, no §3b requirement).
- Modify `docs/2026-07-23-k4-corpus-transfer-results.md` — §8 "Stage 2 — synthesis order" (Task 4).
- Tests: `tests/test_cache_collect.py` (append, Tasks 1–2), `tests/test_k4_experiments.py` (append + one-line extension of the existing smoke, Task 3).
- New local caches (gitignored, regenerable, Task 2): `results/cache/gpt2_1024_{shufcode,uniwiki,unicode,biwiki,bicode}_off{1024,2048,3072,4096}.safetensors` (20 files).
- Commit (Task 4): `results/k4_corpus_transfer/<new-run-id>/{config.json,env.json,metrics.parquet,tq_curve.parquet,overlap.parquet,corpus_transfer_verdict.json}`.

---

### Task 1: Synthesis sampler — `synth_stream` + `load_eval_tokens` passthrough

**Files:**
- Modify: `src/bmx/eval/layer_swap.py:48-138` (`load_eval_tokens`; add `synth_stream` directly below it)
- Test: `tests/test_cache_collect.py` (append at end of file; `pytest` and `torch` already imported at top)

**Interfaces:**
- Consumes: nothing new.
- Produces (Tasks 2–4 rely on these exact signatures):
  ```python
  def synth_stream(window: torch.Tensor, mode: str, seed: int) -> torch.Tensor
      # 1-D, same shape/dtype as window; mode in ("unigram", "bigram");
      # deterministic in seed; every output token comes from the window.

  def load_eval_tokens(
      ..., *,                      # existing params unchanged
      shuffle_seed: int = -1,      # UNCHANGED
      synth: str = "",             # "" | "unigram" | "bigram"
      synth_seed: int = -1,        # required >= 0 when synth != ""
  ) -> torch.Tensor
      # synth != "" replaces the sliced natural window with
      # synth_stream(window, synth, synth_seed + token_offset);
      # synth and shuffle_seed are mutually exclusive.
  ```

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_cache_collect.py`:

```python
# ---------------------------------------------------------------------------
# §3b synthesis sampler — synth_stream + load_eval_tokens synth passthrough
# (spec: docs/superpowers/specs/2026-07-23-k4-corpus-transfer-design.md §3b)
# ---------------------------------------------------------------------------


def test_synth_stream_unigram_marginal():
    from bmx.eval.layer_swap import synth_stream

    g = torch.Generator().manual_seed(7)
    window = torch.randint(0, 8, (16384,), generator=g)
    s1 = synth_stream(window, "unigram", seed=123)
    s2 = synth_stream(window, "unigram", seed=123)
    s3 = synth_stream(window, "unigram", seed=124)
    assert torch.equal(s1, s2)  # deterministic under the seed
    assert not torch.equal(s1, s3)  # the seed actually enters
    assert s1.shape == window.shape and s1.dtype == window.dtype
    # support: sampled WITH replacement from the window itself
    assert set(s1.tolist()) <= set(window.tolist())
    # unigram marginal preserved within tolerance (empirical L1 vs source)
    src = torch.bincount(window, minlength=8).float() / window.numel()
    smp = torch.bincount(s1, minlength=8).float() / s1.numel()
    assert float((src - smp).abs().sum()) < 0.05


def test_synth_stream_bigram_transitions():
    from bmx.eval.layer_swap import synth_stream

    # Deterministic 3-cycle: every observed transition is t -> (t+1) % 3, and
    # every token appears as a context, so the sampled chain must follow the
    # cycle exactly after the first token.
    window = torch.tensor([0, 1, 2] * 512, dtype=torch.int64)
    out = synth_stream(window, "bigram", seed=5)
    assert out.shape == window.shape and out.dtype == window.dtype
    assert torch.equal(out[1:], (out[:-1] + 1) % 3)
    assert torch.equal(out, synth_stream(window, "bigram", seed=5))

    # Branching chain: from 0 the source goes to 1 three times per rep and to
    # 2 once (P = 0.75 / 0.25); 1 and 2 always return to 0. The sampled
    # conditional frequencies must match the source transition counts.
    window = torch.tensor(([0, 1] * 3 + [0, 2]) * 512, dtype=torch.int64)
    out = synth_stream(window, "bigram", seed=11)
    prev, nxt = out[:-1], out[1:]
    frac1 = float((nxt[prev == 0] == 1).float().mean())
    assert abs(frac1 - 0.75) < 0.05
    assert (nxt[prev == 1] == 0).all()
    assert (nxt[prev == 2] == 0).all()


def test_synth_stream_bigram_unseen_context_backoff():
    from bmx.eval.layer_swap import synth_stream

    # window [5, 9]: succ = {5: [9]}; 9 has NO observed successor. Find a seed
    # whose first (unigram) draw is 9 — the very next step must take the
    # unigram-backoff branch without crashing and stay in the window's support.
    w = torch.tensor([5, 9], dtype=torch.int64)
    hit = None
    for s in range(50):
        o = synth_stream(w, "bigram", seed=s)
        if o[0].item() == 9:
            hit = o
            break
    assert hit is not None, "no seed in 0..49 started at the successorless token"
    assert hit.shape == (2,) and hit[1].item() in (5, 9)


def test_load_eval_tokens_synth_wiring(monkeypatch):
    import bmx.eval.layer_swap as ls

    _patch_eval_tokens_io(monkeypatch)
    nat = ls.load_eval_tokens("gpt2", n_tokens=16, token_offset=8)
    uni1 = ls.load_eval_tokens(
        "gpt2", n_tokens=16, token_offset=8, synth="unigram", synth_seed=20260723
    )
    uni2 = ls.load_eval_tokens(
        "gpt2", n_tokens=16, token_offset=8, synth="unigram", synth_seed=20260723
    )
    bi = ls.load_eval_tokens(
        "gpt2", n_tokens=16, token_offset=8, synth="bigram", synth_seed=20260723
    )
    assert torch.equal(uni1, uni2)  # deterministic under the recorded seed
    assert uni1.shape == bi.shape == nat.shape == (16,)
    # sampled from the window at THIS offset: support subset of the natural slice
    assert set(uni1.tolist()) <= set(nat.tolist())
    assert set(bi.tolist()) <= set(nat.tolist())
    # WITH replacement (unlike the shuffle null): not a permutation of the
    # window — the arange window has 16 distinct tokens, so a permutation
    # would preserve the multiset exactly
    assert not torch.equal(uni1.sort().values, nat.sort().values)


def test_load_eval_tokens_synth_validation(monkeypatch):
    import bmx.eval.layer_swap as ls

    _patch_eval_tokens_io(monkeypatch)
    with pytest.raises(AssertionError, match="synth mode"):
        ls.load_eval_tokens("gpt2", n_tokens=8, synth="trigram", synth_seed=0)
    with pytest.raises(AssertionError, match="synth_seed"):
        ls.load_eval_tokens("gpt2", n_tokens=8, synth="unigram")
    with pytest.raises(AssertionError, match="mutually exclusive"):
        ls.load_eval_tokens(
            "gpt2", n_tokens=8, synth="unigram", synth_seed=0, shuffle_seed=0
        )
```

- [ ] **Step 2: Run tests to verify they fail.**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_cache_collect.py -q -k "synth"`
Expected: 5 FAIL — `ImportError: cannot import name 'synth_stream'` / `TypeError: load_eval_tokens() got an unexpected keyword argument 'synth'`.

- [ ] **Step 3: Implement.** In `src/bmx/eval/layer_swap.py`:

(a) extend the `load_eval_tokens` signature (lines 48–59) — add the two keyword params after `shuffle_seed`:

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
    synth: str = "",
    synth_seed: int = -1,
) -> torch.Tensor:
```

(b) append to the docstring (after the existing shuffle sentence):

```
    `synth` ∈ {"unigram", "bigram"} replaces the RETURNED natural slice with a
    same-length stream sampled from that slice's own n-gram statistics (spec
    §3b — the traffic-histogram calibration recipe at orders 1 and 2); the
    generator is seeded `synth_seed + token_offset` (same per-slice scheme as
    shuffle). Mutually exclusive with shuffle_seed.
```

(c) add validation immediately after the docstring, BEFORE the `datasets`/`transformers` imports (fail fast, no download on bad args):

```python
    assert synth in ("", "unigram", "bigram"), f"unknown synth mode {synth!r}"
    assert not synth or synth_seed >= 0, "synth requires synth_seed >= 0"
    assert not (synth and shuffle_seed >= 0), (
        "synth and shuffle_seed are mutually exclusive (distinct §3b arms)"
    )
```

(d) replace the tail of the function (current lines 134–138) with:

```python
    out = ids.input_ids[0][token_offset:]
    if shuffle_seed >= 0:
        g = torch.Generator().manual_seed(shuffle_seed + token_offset)
        out = out[torch.randperm(out.numel(), generator=g)]
    if synth:
        out = synth_stream(out, synth, synth_seed + token_offset)
    return out
```

(e) add the sampler directly below `load_eval_tokens`:

```python
def synth_stream(window: torch.Tensor, mode: str, seed: int) -> torch.Tensor:
    """Sample a same-length synthetic token stream from `window`'s own n-gram
    statistics (spec §3b: the 'synthesize calibration text from traffic token
    counts' recipe, orders 1 and 2).

    "unigram": i.i.d. WITH replacement from the window's empirical histogram
    (uniform position draws — each token's probability is count/N; contrast
    the shuffle null, which is sampling WITHOUT replacement).
    "bigram": Markov chain with empirical conditionals estimated on the
    window (add-nothing smoothing — a successor is drawn uniformly from the
    multiset of tokens observed after the current context); the first token
    and any context with no observed successor back off to the unigram
    histogram. Deterministic in `seed`; output shape/dtype match `window`.
    """
    assert window.ndim == 1 and window.numel() > 0, "window must be 1-D, non-empty"
    assert mode in ("unigram", "bigram"), f"unknown synth mode {mode!r}"
    n = window.numel()
    g = torch.Generator().manual_seed(seed)

    def uni(k: int) -> torch.Tensor:
        return window[torch.randint(0, n, (k,), generator=g)]

    if mode == "unigram":
        return uni(n)
    succ: dict[int, list[int]] = {}
    for c, nx in zip(window[:-1].tolist(), window[1:].tolist()):
        succ.setdefault(c, []).append(nx)
    succ_t = {c: torch.tensor(v, dtype=window.dtype) for c, v in succ.items()}
    out = torch.empty(n, dtype=window.dtype)
    cur = int(uni(1).item())
    out[0] = cur
    for t in range(1, n):
        choices = succ_t.get(cur)
        if choices is None:  # context never observed with a successor
            cur = int(uni(1).item())
        else:
            j = int(torch.randint(0, choices.numel(), (1,), generator=g).item())
            cur = int(choices[j])
        out[t] = cur
    return out
```

- [ ] **Step 4: Run tests to verify they pass — including the byte-identity pins.**

Run: `uv run pytest tests/test_cache_collect.py -q`
Expected: all pass — the new 5 plus the pre-existing default-path pins (`test_load_eval_tokens_generalized_defaults`, `test_load_eval_tokens_offset`, `test_load_eval_tokens_shuffle_after_slice`, the capped-accumulation pins) prove the defaults and the shuffle path are untouched.

- [ ] **Step 5: Battery + propose commit.**

Run: `uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: clean; `485 passed, 17 skipped, 1 xfailed`.
Stage `src/bmx/eval/layer_swap.py tests/test_cache_collect.py`; propose message:
`feat(eval): §3b synthesis sampler — synth_stream unigram/bigram (seeded, per-window) + load_eval_tokens synth passthrough; defaults and shuffle path byte-identical`
**STOP for user approval.**

---

### Task 2: `collect_cache.py` synth passthrough + the 20 collections

**Files:**
- Modify: `experiments/collect_cache.py` (Config lines 32–46, `_WIKI_DEFAULTS` line 49, `_corpus_is_default` lines 52–60, `load_eval_tokens` call lines 94–104, docstring Usage block)
- Test: `tests/test_cache_collect.py` (append)

**Interfaces:**
- Consumes: Task 1's `load_eval_tokens(..., synth=, synth_seed=)`.
- Produces: `Config` fields `synth: str = ""`, `synth_seed: int = -1` (flags `--synth`, `--synth-seed`); pinned labels `shufcode`/`uniwiki`/`unicode`/`biwiki`/`bicode`; the 20 cache files Task 4 consumes: `results/cache/gpt2_1024_{shufcode,uniwiki,unicode,biwiki,bicode}_off{1024,2048,3072,4096}.safetensors`.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_cache_collect.py`:

```python
def test_collect_cache_synth_labels_and_guard():
    from experiments.collect_cache import Config, _corpus_is_default, _out_path

    uni = Config(
        model_name="gpt2",
        seq_len=1024,
        token_offset=1024,
        synth="unigram",
        synth_seed=20260723,
        corpus_label="uniwiki",
    )
    assert not _corpus_is_default(uni)
    assert _out_path(uni).name == "gpt2_1024_uniwiki_off1024.safetensors"

    bicode = Config(
        model_name="gpt2",
        seq_len=1024,
        token_offset=2048,
        dataset_id="codeparrot/codeparrot-clean-valid",
        dataset_config="",
        split="train[:200]",
        text_field="content",
        synth="bigram",
        synth_seed=20260723,
        corpus_label="bicode",
    )
    assert not _corpus_is_default(bicode)
    assert _out_path(bicode).name == "gpt2_1024_bicode_off2048.safetensors"

    shufcode = Config(
        model_name="gpt2",
        seq_len=1024,
        token_offset=1024,
        dataset_id="codeparrot/codeparrot-clean-valid",
        dataset_config="",
        split="train[:200]",
        text_field="content",
        shuffle_seed=20260723,
        corpus_label="shufcode",
    )
    assert not _corpus_is_default(shufcode)
    assert _out_path(shufcode).name == "gpt2_1024_shufcode_off1024.safetensors"
```

- [ ] **Step 2: Run to verify failure.**

Run: `uv run pytest tests/test_cache_collect.py::test_collect_cache_synth_labels_and_guard -q`
Expected: FAIL — `TypeError: Config.__init__() got an unexpected keyword argument 'synth'`.

- [ ] **Step 3: Implement.** In `experiments/collect_cache.py`:

(a) insert the two fields between `shuffle_seed` and `corpus_label` in `Config`:

```python
    shuffle_seed: int = -1  # >=0 => seeded post-slice token shuffle (null corpus)
    synth: str = ""  # "" | "unigram" | "bigram" — §3b sampled synthetic stream
    synth_seed: int = -1  # required >=0 when synth set; recorded seed: 20260723
    corpus_label: str = ""  # REQUIRED when any corpus knob above is non-default
```

(b) extend the defaults tuple and the guard:

```python
_WIKI_DEFAULTS = (
    "Salesforce/wikitext", "wikitext-2-raw-v1", "", "test", "text", -1, "", -1,
)


def _corpus_is_default(cfg: Config) -> bool:
    return (
        cfg.dataset_id,
        cfg.dataset_config,
        cfg.data_dir,
        cfg.split,
        cfg.text_field,
        cfg.shuffle_seed,
        cfg.synth,
        cfg.synth_seed,
    ) == _WIKI_DEFAULTS
```

(c) add the passthrough to the `load_eval_tokens` call in `main` (after `shuffle_seed=cfg.shuffle_seed,`):

```python
        synth=cfg.synth,
        synth_seed=cfg.synth_seed,
```

(d) add one Usage line to the module docstring after the existing code-corpus example:

```
    uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024 \
        --token-offset 1024 --synth unigram --synth-seed 20260723 --corpus-label uniwiki
```

- [ ] **Step 4: Run tests.**

Run: `uv run pytest tests/test_cache_collect.py -q`
Expected: PASS (all — including the pre-existing `test_collect_cache_out_path_and_corpus_guard` and `test_collect_cache_main_guard_raises_before_model_load`, which pin that the extended defaults tuple stays default-true for synth-free configs).

- [ ] **Step 5: Collect the 20 caches (local CPU; codeparrot is already in the HF cache from Stage 1; ~1 min each, ~20 min total).** From `/d/Projects/bmx`:

```bash
# (a) shuf_code — token-permuted CODE fit slices (same seed scheme as the wiki null)
for OFF in 1024 2048 3072 4096; do
  uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024 \
    --token-offset $OFF \
    --dataset-id codeparrot/codeparrot-clean-valid --dataset-config "" \
    --split "train[:200]" --text-field content \
    --shuffle-seed 20260723 --corpus-label shufcode
done

# (b) uni_wiki / bi_wiki — sampled streams from each wikitext fit slice's statistics
for OFF in 1024 2048 3072 4096; do
  uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024 \
    --token-offset $OFF --synth unigram --synth-seed 20260723 --corpus-label uniwiki
  uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024 \
    --token-offset $OFF --synth bigram --synth-seed 20260723 --corpus-label biwiki
done

# (c) uni_code / bi_code — sampled streams from each code fit slice's statistics
for OFF in 1024 2048 3072 4096; do
  uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024 \
    --token-offset $OFF \
    --dataset-id codeparrot/codeparrot-clean-valid --dataset-config "" \
    --split "train[:200]" --text-field content \
    --synth unigram --synth-seed 20260723 --corpus-label unicode
  uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024 \
    --token-offset $OFF \
    --dataset-id codeparrot/codeparrot-clean-valid --dataset-config "" \
    --split "train[:200]" --text-field content \
    --synth bigram --synth-seed 20260723 --corpus-label bicode
done
```

Expected: 20 new files in `results/cache/` (each ~28 MB): `gpt2_1024_{shufcode,uniwiki,unicode,biwiki,bicode}_off{1024,2048,3072,4096}.safetensors`.

- [ ] **Step 6: Sanity-check one cache per arm.**

```bash
uv run python - <<'EOF'
from bmx.cache.collect import load_cache
for lbl in ("shufcode", "uniwiki", "unicode", "biwiki", "bicode"):
    p = f"results/cache/gpt2_1024_{lbl}_off1024.safetensors"
    c = load_cache(p)
    assert c["layer0.k_pre"].shape == (12, 1024, 64), (p, c["layer0.k_pre"].shape)
    print("OK", p)
EOF
```

Expected: five `OK` lines.

- [ ] **Step 7: Battery + propose commit.** (Caches are gitignored — only code + tests get committed.)

Run: `uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: clean; `486 passed, 17 skipped, 1 xfailed`.
Stage `experiments/collect_cache.py tests/test_cache_collect.py`; propose message:
`feat(exp): collect_cache synth passthrough — §3b sampled-stream arms with pinned labels (shufcode/uniwiki/unicode/biwiki/bicode), seed 20260723`
**STOP for user approval.**

---

### Task 3: Harness extension — five fit-side arms + §3b `synthesis` verdict block

**Files:**
- Modify: `experiments/k4_corpus_transfer.py` (docstring, constants at lines 56–63, `_diagnostics` centered loop at lines 107–115, `Config` at 237–254, `main` fit-path block at 277–284 and plain-matrix loop at 455–478, `_transfer_verdict` at 598–719)
- Test: `tests/test_k4_experiments.py` (append + one-line extension of `test_k4_corpus_transfer_smoke`)

**Interfaces:**
- Consumes: Task 2's cache files (Task 4 only — tests use `_tiny_cache` fixtures); existing `corpus_fit_bases`/`CorpusFit`/`_cell_wins`/verdict machinery (unchanged).
- Produces (Task 4 relies on these exact names):
  ```python
  SYNTH_FIT_CORPORA = ("shufcode", "uniwiki", "unicode", "biwiki", "bicode")

  # Config gains (tyro flags --shufcode-fit-paths etc.; default () = arm absent):
  shufcode_fit_paths: tuple[str, ...] = ()
  uniwiki_fit_paths: tuple[str, ...] = ()
  unicode_fit_paths: tuple[str, ...] = ()
  biwiki_fit_paths: tuple[str, ...] = ()
  bicode_fit_paths: tuple[str, ...] = ()
  ```
  metrics.parquet: `arm == "spectral"` rows with `fit_corpus` ∈ the five new names, scored on BOTH eval sides, `w_corpus == alloc_corpus == fit_corpus`. Verdict JSON: `per_budget[b]["D"]` gains `"<synth>-><eval>"` cells (vs the same matched cell `eval->eval`); `per_budget[b]["synthesis"]` = `{}` when the arms are absent, else:
  ```json
  {"rules": {"wiki": {"D_uni": f, "D_bi": f, "D_shuf": f,
                      "recipe_confirmed": b, "order2_earns_keep": b},
             "code": {...same keys...}},
   "climb_to_order3": b, "note": "..."}
  ```

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_k4_experiments.py`:

```python
def _tiny_corpus_transfer_cfg_synth(tmp_path, budgets=(2.5,)):
    """Stage-1 tiny cfg + the five §3b synthesis fit arms (2 slices each —
    matched with the natural corpora's 2 slices, binding decision 1)."""
    import dataclasses

    cfg = _tiny_corpus_transfer_cfg(tmp_path, budgets=budgets)
    paths, seed = {}, 100
    for name in ("sc", "uw", "uc", "bw", "bc"):
        group = []
        for j in range(2):
            p = tmp_path / f"{name}{j}.safetensors"
            _tiny_cache(p, seed=seed)
            seed += 1
            group.append(str(p))
        paths[name] = tuple(group)
    return dataclasses.replace(
        cfg,
        shufcode_fit_paths=paths["sc"],
        uniwiki_fit_paths=paths["uw"],
        unicode_fit_paths=paths["uc"],
        biwiki_fit_paths=paths["bw"],
        bicode_fit_paths=paths["bc"],
    )


def test_k4_corpus_transfer_synthesis_smoke(tmp_path):
    import pandas as pd

    from experiments.k4_corpus_transfer import main

    run_dir = main(_tiny_corpus_transfer_cfg_synth(tmp_path))
    df = pd.read_parquet(run_dir / "metrics.parquet")
    spec = df[df.arm == "spectral"]
    synth = ("shufcode", "uniwiki", "unicode", "biwiki", "bicode")
    assert set(spec.fit_corpus) == {"wiki", "code", "null", *synth}
    for c in synth:
        sub = spec[spec.fit_corpus == c]
        # fit-side-only arms, scored on BOTH eval sides in the same run
        assert set(sub.eval_corpus) == {"wiki", "code"}
        assert (sub.w_corpus == c).all() and (sub.alloc_corpus == c).all()

    v = json.loads((run_dir / "corpus_transfer_verdict.json").read_text())
    pb = v["per_budget"]["2.5"]
    assert {
        "uniwiki->wiki", "unicode->code", "biwiki->wiki", "bicode->code",
        "shufcode->code", "shufcode->wiki",
    } <= set(pb["D"])
    rules = pb["synthesis"]["rules"]
    assert set(rules) == {"wiki", "code"}
    for eval_c, r in rules.items():
        assert isinstance(r["recipe_confirmed"], bool)
        assert isinstance(r["order2_earns_keep"], bool)
        # pre-registered §3b rules (a)/(b), recomputable from the same JSON
        assert r["recipe_confirmed"] == (r["D_uni"] < 0.10)
        assert r["order2_earns_keep"] == ((r["D_uni"] - r["D_bi"]) >= 0.5 * r["D_uni"])
        assert r["D_shuf"] is not None
    assert pb["synthesis"]["climb_to_order3"] == all(
        r["order2_earns_keep"] for r in rules.values()
    )


def test_k4_corpus_transfer_synthesis_guards(tmp_path):
    import dataclasses

    import pytest

    from experiments.k4_corpus_transfer import main

    cfg = _tiny_corpus_transfer_cfg_synth(tmp_path)
    # partial provision refuses to run (the order ladder needs all five arms)
    with pytest.raises(AssertionError, match="all-or-nothing"):
        main(dataclasses.replace(cfg, bicode_fit_paths=()))
    # the matched-fit-budget guard extends over the synthesis arms
    with pytest.raises(AssertionError, match="matched fit"):
        main(dataclasses.replace(cfg, uniwiki_fit_paths=cfg.uniwiki_fit_paths[:1]))
```

Also extend the EXISTING `test_k4_corpus_transfer_smoke` (after its final `assert isinstance(pb["model_intrinsic_flag"], bool)` line) with one line pinning Stage-1 behavior:

```python
    assert pb["synthesis"] == {}  # no §3b arms provided => empty block
```

- [ ] **Step 2: Run to verify failure.**

Run: `uv run pytest tests/test_k4_experiments.py -q -k "corpus_transfer"`
Expected: the 2 new tests FAIL (`TypeError: ... unexpected keyword argument 'shufcode_fit_paths'`), and `test_k4_corpus_transfer_smoke` FAILS on the new `synthesis` assert (`KeyError`). Pre-existing hybrid/wcross/diagnostics tests still pass.

- [ ] **Step 3: Implement.** In `experiments/k4_corpus_transfer.py`:

(a) append to the module docstring (before the closing `"""`):

```
Synthesis-order addendum (spec §3b): five FIT-SIDE-ONLY arms — shufcode
(token-permuted code slices), uniwiki/unicode (i.i.d. samples from each fit
slice's unigram histogram), biwiki/bicode (bigram-Markov samples) — join the
same matrix at matched budgets. Verdict gains per_budget[b]["synthesis"]
with the pre-registered order-ladder rules: (a) recipe-confirmed for eval
side E if D(uni_E->E) < 10%; (b) order-2 earns its keep if
D(uni_E->E) - D(bi_E->E) >= 0.5 * D(uni_E->E); no higher orders unless (b)
passes on BOTH sides. uni_* is the recipe estimator, shuf_* its
without-replacement control.
```

(b) add the constant below `FIT_CORPORA`/`EVAL_CORPORA` (line 57):

```python
# §3b synthesis-order fit arms (fit-side only; labels compose as
# f"uni{eval_side}" / f"bi{eval_side}" in the order-ladder rules).
SYNTH_FIT_CORPORA = ("shufcode", "uniwiki", "unicode", "biwiki", "bicode")
```

(c) in `_diagnostics`, restrict the centered-refit loop to the `_PAIRS` corpora (the overlap diagnostic never uses synth bases; the spectrum loop below it stays `for corpus, fit in fits.items()` so synth spectra land in overlap.parquet automatically). Replace the loop header at lines 108–110:

```python
    centered_bases: dict[str, dict[int, object]] = {}
    for corpus in sorted({c for p in _PAIRS for c in p}):
        fit = fits[corpus]
        centered_bases[corpus] = {}
```

(d) append the five tuples to `Config` (after `code_eval_paths`):

```python
    # ---- §3b synthesis-order fit arms (fit-side only; all five or none) ----
    shufcode_fit_paths: tuple[str, ...] = ()
    uniwiki_fit_paths: tuple[str, ...] = ()
    unicode_fit_paths: tuple[str, ...] = ()
    biwiki_fit_paths: tuple[str, ...] = ()
    bicode_fit_paths: tuple[str, ...] = ()
```

(e) in `main`, after the `fit_paths = {...}` dict (line 284) and BEFORE the `for name, paths in fit_paths.items(): assert paths` loop, insert:

```python
    synth_paths = {
        "shufcode": cfg.shufcode_fit_paths,
        "uniwiki": cfg.uniwiki_fit_paths,
        "unicode": cfg.unicode_fit_paths,
        "biwiki": cfg.biwiki_fit_paths,
        "bicode": cfg.bicode_fit_paths,
    }
    n_synth = sum(1 for p in synth_paths.values() if p)
    assert n_synth in (0, len(synth_paths)), (
        "§3b synthesis arms are all-or-nothing (the order-ladder rules need "
        f"all five): got {n_synth}/5 non-empty "
        f"{ {k: len(v) for k, v in synth_paths.items()} }"
    )
    if n_synth:
        fit_paths.update(synth_paths)
```

The existing non-empty asserts, `corpus_fit_bases` fit loop, and BOTH matched-budget asserts (slice counts + total fit rows) then iterate over all present arms with no further change.

(f) in `main`, change the plain-matrix loop header (line 456) from `for fit_c in FIT_CORPORA:` to:

```python
    for fit_c in fit_paths:
```

(g) in `_transfer_verdict`, derive the fit-corpus list from the df (insert after the `tq_curves` construction) and use it in BOTH the cells loop and the D loop (replace `for fit_c in FIT_CORPORA:` in each):

```python
    present = set(df.loc[df.arm == "spectral", "fit_corpus"])
    fit_corpora = [c for c in FIT_CORPORA + SYNTH_FIT_CORPORA if c in present]
```

(h) in `_transfer_verdict`'s per-budget loop, after the `wcross` block and before `null_wiki = ...`, insert:

```python
        # §3b synthesis-order rules — gates for the RECIPE claim (spec §3b).
        synthesis: dict = {}
        if any(c in fit_corpora for c in SYNTH_FIT_CORPORA):
            rules: dict[str, dict] = {}
            for eval_c in EVAL_CORPORA:
                d_uni = D.get(f"uni{eval_c}->{eval_c}", {}).get("mean")
                d_bi = D.get(f"bi{eval_c}->{eval_c}", {}).get("mean")
                if d_uni is None or d_bi is None:
                    continue
                shuf_cell = "shufcode->code" if eval_c == "code" else "null->wiki"
                rules[eval_c] = dict(
                    D_uni=d_uni,
                    D_bi=d_bi,
                    D_shuf=D.get(shuf_cell, {}).get("mean"),
                    # rule (a): the sampled-unigram recipe transfers on E
                    recipe_confirmed=bool(d_uni < 0.10),
                    # rule (b): bigram closes >= half the unigram gap on E
                    order2_earns_keep=bool((d_uni - d_bi) >= 0.5 * d_uni),
                )
            synthesis = dict(
                rules=rules,
                climb_to_order3=bool(
                    rules and all(r["order2_earns_keep"] for r in rules.values())
                ),
                note=(
                    "§3b pre-registered gates for the traffic-histogram RECIPE "
                    "claim: uni_* is the recipe estimator, shuf_* its "
                    "without-replacement control (D_shuf for wiki = the Stage-1 "
                    "null->wiki cell); no higher orders unless order2_earns_keep "
                    "on BOTH eval sides"
                ),
            )
```

and add `synthesis=synthesis,` to the `per_budget[f"{budget:g}"] = dict(...)` assignment (after `wcross=wcross,`).

- [ ] **Step 4: Run tests.**

Run: `uv run pytest tests/test_k4_experiments.py -q -k "corpus_transfer"`
Expected: all PASS — the 2 new tests, the extended smoke, and the untouched hybrid/wcross/diagnostics/overlap-pin tests (the tiny synth run is 8 arms × 2 layers × C=16 — seconds).

- [ ] **Step 5: Battery + propose commit.**

Run: `uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: clean; `488 passed, 17 skipped, 1 xfailed`.
Stage `experiments/k4_corpus_transfer.py tests/test_k4_experiments.py`; propose message:
`feat(exp): k4_corpus_transfer §3b synthesis-order arms — five fit-side corpora in the matrix, order-ladder rules a/b as pre-registered verdict gates, all-or-nothing + matched-budget guards`
**STOP for user approval.**

---

### Task 4: The real Stage-2 run (full 8-arm matrix, one run-id) + results-doc §8 + verification/push

**Files:**
- Modify: `docs/2026-07-23-k4-corpus-transfer-results.md` (append §8)
- Commit (new artifacts only): `results/k4_corpus_transfer/<new-run-id>/{config.json,env.json,metrics.parquet,tq_curve.parquet,overlap.parquet,corpus_transfer_verdict.json}`

**Interfaces:**
- Consumes: Task 2's 20 caches + the 16 Stage-1 caches; Task 3's harness.
- Produces: the committed Stage-2 run dir + §8; the go/no-go for the push.

- [ ] **Step 1: Run the FULL matrix — Stage-1 arms INCLUDED so every cell shares one run-id (local CPU, ~30–60 min: 8 corpus fits × 12 layers × C=768 + 8-arm scoring).** From `/d/Projects/bmx` (MODULE form required — script form fails on sys.path; `--model-name` stays empty — gpt2 has no RoPE, headline = `logit`):

```bash
uv run python -m experiments.k4_corpus_transfer \
  --wiki-fit-paths results/cache/gpt2_1024_off1024.safetensors results/cache/gpt2_1024_off2048.safetensors results/cache/gpt2_1024_off3072.safetensors results/cache/gpt2_1024_off4096.safetensors \
  --code-fit-paths results/cache/gpt2_1024_code_off1024.safetensors results/cache/gpt2_1024_code_off2048.safetensors results/cache/gpt2_1024_code_off3072.safetensors results/cache/gpt2_1024_code_off4096.safetensors \
  --null-fit-paths results/cache/gpt2_1024_shuf_off1024.safetensors results/cache/gpt2_1024_shuf_off2048.safetensors results/cache/gpt2_1024_shuf_off3072.safetensors results/cache/gpt2_1024_shuf_off4096.safetensors \
  --shufcode-fit-paths results/cache/gpt2_1024_shufcode_off1024.safetensors results/cache/gpt2_1024_shufcode_off2048.safetensors results/cache/gpt2_1024_shufcode_off3072.safetensors results/cache/gpt2_1024_shufcode_off4096.safetensors \
  --uniwiki-fit-paths results/cache/gpt2_1024_uniwiki_off1024.safetensors results/cache/gpt2_1024_uniwiki_off2048.safetensors results/cache/gpt2_1024_uniwiki_off3072.safetensors results/cache/gpt2_1024_uniwiki_off4096.safetensors \
  --unicode-fit-paths results/cache/gpt2_1024_unicode_off1024.safetensors results/cache/gpt2_1024_unicode_off2048.safetensors results/cache/gpt2_1024_unicode_off3072.safetensors results/cache/gpt2_1024_unicode_off4096.safetensors \
  --biwiki-fit-paths results/cache/gpt2_1024_biwiki_off1024.safetensors results/cache/gpt2_1024_biwiki_off2048.safetensors results/cache/gpt2_1024_biwiki_off3072.safetensors results/cache/gpt2_1024_biwiki_off4096.safetensors \
  --bicode-fit-paths results/cache/gpt2_1024_bicode_off1024.safetensors results/cache/gpt2_1024_bicode_off2048.safetensors results/cache/gpt2_1024_bicode_off3072.safetensors results/cache/gpt2_1024_bicode_off4096.safetensors \
  --wiki-eval-paths results/cache/gpt2_1024.safetensors results/cache/gpt2_1024_off5120.safetensors \
  --code-eval-paths results/cache/gpt2_1024_code.safetensors results/cache/gpt2_1024_code_off5120.safetensors \
  --model-label gpt2 --budgets 2.2 2.5
```

Expected: eight `== fitting corpus '<name>' ==` blocks, the score lines, the `CORPUS-TRANSFER VERDICT` JSON (now with a `synthesis` block per budget), `-> results/k4_corpus_transfer/<new-run-id>`. Spectral rows: 8 fit arms × 2 budgets × 12 layers × 4 eval caches = 768 (+96 hybrid, +192 wcross).

- [ ] **Step 2: Independent verification — recompute one synthesis rule AND the Stage-1 reproduction check.**

```bash
uv run python - <<'EOF'
import json
from pathlib import Path
import pandas as pd
from experiments._k4_common import _log_interp, _tq_layer_curve

run = sorted(Path("results/k4_corpus_transfer").iterdir())[-1]
v = json.loads((run / "corpus_transfer_verdict.json").read_text())
df = pd.read_parquet(run / "metrics.parquet")
tq = pd.read_parquet(run / "tq_curve.parquet")
hl = v["headline_metric"]
curves = {k: _tq_layer_curve(g, hl) for k, g in tq.groupby(["eval_corpus", "cache"])}

def cell_wins(fit_c, eval_c, budget):
    sub = df[(df.arm == "spectral") & (df.budget == budget)
             & (df.fit_corpus == fit_c) & (df.eval_corpus == eval_c)]
    wins = {}
    for _, r in sub.iterrows():
        pts = curves[(r.eval_corpus, r.cache)][int(r.layer)]
        tqd, _ = _log_interp(pts, float(r.bpe_skeptic_deploy))
        wins.setdefault(r.cache, []).append(tqd / max(float(r[hl]), 1e-300))
    return {c: sum(w) / len(w) for c, w in wins.items()}

matched = cell_wins("wiki", "wiki", 2.5)
uni = cell_wins("uniwiki", "wiki", 2.5)
ds = [1.0 - uni[c] / matched[c] for c in matched]
d_mean = sum(ds) / len(ds)
stored = v["per_budget"]["2.5"]["synthesis"]["rules"]["wiki"]["D_uni"]
assert abs(d_mean - stored) < 1e-9, (d_mean, stored)
print("OK  D(uniwiki->wiki) b2.5 =", stored)

# Stage-1 reproduction (Global Constraint 9): deterministic pipeline, same
# caches/seeds => the re-run Stage-1 cells must match the committed run.
s1 = json.loads(Path("results/k4_corpus_transfer/20260723-190823-8dced47/"
                     "corpus_transfer_verdict.json").read_text())
for b in ("2.2", "2.5"):
    for cell in ("code->wiki", "wiki->code", "null->wiki", "null->code"):
        new = v["per_budget"][b]["D"][cell]["mean"]
        old = s1["per_budget"][b]["D"][cell]["mean"]
        assert abs(new - old) < 1e-9, (b, cell, new, old)
print("OK  Stage-1 D cells reproduced at both budgets")
print(json.dumps({b: v["per_budget"][b]["synthesis"] for b in ("2.2", "2.5")},
                 indent=2))
EOF
```

Expected: two `OK` lines then the synthesis blocks. **If the Stage-1 reproduction assert fires: STOP — do not write §8, do not shrug it into the doc; diagnose with superpowers:systematic-debugging (the pipeline is deterministic; a mismatch means a harness regression or a mutated cache).**

- [ ] **Step 3: Append §8 to `docs/2026-07-23-k4-corpus-transfer-results.md`.** Skeleton (fill every `⟨…⟩` from the Step-2 output/verdict JSON; the `⟨…⟩` are number slots in the deliverable, not plan placeholders; keep BOTH pre-drafted outcomes per rule until the numbers pick one, then delete the losing branch — keep both ONLY if the two eval sides split):

```markdown
## 8. Stage 2 — synthesis order (§3b addendum; run `⟨new-run-id⟩`, git SHA ⟨sha⟩)

Motivation: Stage 1 measured D(shuf_wiki→wiki) ≈ 9.4% — the unigram token
histogram carries ~91% of the matched-fit win for English. Stage 2 asks
whether the literal deployment recipe (SAMPLE a calibration stream from a
traffic token histogram) works, on both domains, and whether order 2 buys
anything (spec §3b). Five fit-side arms at matched budgets (4 × 1024 tokens,
offsets 1024/2048/3072/4096; per-slice statistics; seed 20260723 for both
shuffle and synthesis, generator seeded `20260723 + offset`); code source =
the Stage-1 codeparrot fallback (identity with Stage-1 code windows). FULL
matrix rerun — all Stage-1 and Stage-2 cells share this run-id; Stage-1 D
cells reproduced against run `20260723-190823-8dced47` to < 1e-9 (Step-2
check), so §§1–7 above stand unchanged.

**YELLOW FLAG:** gpt2 scale = mechanism verdict only (corpus-W retention
~0.47–0.52, `docs/2026-07-15-k4-duel-results.md`); Llama fit-side
replication pre-registered.

### 8.1 Synthesis-arm win matrix (win_mean, b2.2 / b2.5; gpt2 — see yellow flag)

| fit arm \ eval | wiki-held | code-held |
|---|---|---|
| shuf_code | ⟨⟩ / ⟨⟩ | ⟨⟩ / ⟨⟩ |
| uni_wiki | ⟨⟩ / ⟨⟩ | ⟨⟩ / ⟨⟩ |
| uni_code | ⟨⟩ / ⟨⟩ | ⟨⟩ / ⟨⟩ |
| bi_wiki | ⟨⟩ / ⟨⟩ | ⟨⟩ / ⟨⟩ |
| bi_code | ⟨⟩ / ⟨⟩ | ⟨⟩ / ⟨⟩ |

(Reference matched cells, §1: wiki→wiki 9.736/9.207, code→code 17.488/15.876.)

### 8.2 Matched-side D + order-ladder rules (§3b; gpt2 — see yellow flag)

| eval side | D_shuf (control) | D_uni (recipe) | D_bi | rule (a) D_uni<10% | rule (b) gap-closed ≥ ½ |
|---|---|---|---|---|---|
| wiki | 0.094 (Stage-1 null→wiki) | ⟨⟩ / ⟨⟩ (b2.2/b2.5) | ⟨⟩ / ⟨⟩ | ⟨pass/fail⟩ | ⟨pass/fail⟩ |
| code | ⟨shufcode→code⟩ | ⟨⟩ / ⟨⟩ | ⟨⟩ / ⟨⟩ | ⟨pass/fail⟩ | ⟨pass/fail⟩ |

climb_to_order3: ⟨true/false⟩ (rule b on BOTH sides — the ladder is climbed
one measured rung at a time; no higher orders otherwise).

The shuf-vs-uni gap isolates with-vs-without replacement at matched order-1
statistics: ⟨numbers + one sentence⟩.

### 8.3 Recipe verdict (delete the branch the numbers kill, per rule per side)

**(a) confirmed branch:** the sampled-unigram recipe is deployable for
⟨wiki|code|both⟩ — D(uni→E) ⟨⟩ < 10%: a calibration stream synthesized from
a traffic token HISTOGRAM (no stored user text — the privacy note: ship
histograms, not texts) recovers ⟨⟩% of the matched-fit win at gpt2
mechanism scale.

**(a) killed branch:** the sampled-unigram recipe fails for ⟨side⟩ —
D(uni→E) ⟨⟩ ≥ 10% ⟨vs the shuf control ⟨⟩: if shuf passes where uni fails,
the loss is the with-replacement sampling noise / multiset drift, not the
order statistics⟩. The histogram recipe does not transfer there; per-domain
NATURAL calibration text remains required.

**(b) earns-keep branch:** order 2 closes ⟨⟩ of the unigram gap on ⟨side(s)⟩
(≥ ½ bar) — the bigram chain is the recipe floor there; order 3 is
⟨licensed for the ladder / still gated by the other side⟩.

**(b) dead branch:** order is dead at 2nd order too — bigram closes only
⟨⟩ < ½ of the unigram gap on ⟨side⟩; unigram is the final recipe and the
ladder stops (pre-registered: no higher orders).

### 8.4 VM rider

If rule (a) confirms here, the pre-registered Llama A1/A2 replication
carries the five synthesis arms (exact flags in
`docs/superpowers/plans/2026-07-23-k4-synthesis-order.md`, VM NOTE).
```

- [ ] **Step 4: Battery + propose commit.**

Run: `uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: clean; `488 passed, 17 skipped, 1 xfailed` (no code changes in this task).
Stage `docs/2026-07-23-k4-corpus-transfer-results.md results/k4_corpus_transfer/<new-run-id>/` (parquets + json only — never caches); propose message (pick the measured outcome words):
`results(k4): Stage-2 synthesis order — recipe <confirmed|killed|split> (rule a), order-2 <earns keep|dead> (rule b); full 8-arm matrix rerun, Stage-1 cells reproduced <1e-9, gpt2 mechanism scale`
**STOP for user approval.**

- [ ] **Step 5: Verification gate + push proposal.**

Run: `git status --short` and `git log --oneline a47aeff..HEAD`
Expected: only the files in "File structure" are in the Task 1–4 commits; `results/cache/*.safetensors` untracked (gitignored); no edits to §§1–7 of the results doc beyond the §8 append, no edits to shipped pack files or old parquets.
Then propose: `git push origin feat/triton-decode-kernel` — **STOP for user approval.**

---

## VM NOTE — synthesis arms ride the Llama A1/A2 replication (unnumbered; conditional)

IF the gpt2 Stage-2 run recipe-confirms (rule a) on at least one eval side, the pre-registered Llama A1/A2 replication (parent plan's VM addendum) additionally collects the five synthesis arms at S=2048 — A1 gains 20 collections: the Task-2 commands with `--model-name meta-llama/Llama-3.1-8B-Instruct --seq-len 2048 --token-offset {2048,4096,6144,8192}` and unchanged labels/seeds (`--shuffle-seed 20260723 --corpus-label shufcode` on the code corpus; `--synth unigram|bigram --synth-seed 20260723 --corpus-label uniwiki|biwiki|unicode|bicode`) — and A2's matrix command gains the matching `--shufcode-fit-paths --uniwiki-fit-paths --unicode-fit-paths --biwiki-fit-paths --bicode-fit-paths` flags.

---

## Self-Review (run after writing — findings fixed inline)

**1. Spec coverage (§3b, clause by clause):**

- shuf_code (code permutation, wiki-null seed scheme): Task 2(a) collection + Task 3 arm; D(shuf_code→code) lands in the verdict D table and as `D_shuf` for the code side. ✓
- uni_wiki/uni_code (i.i.d. WITH replacement from the fit-slice histogram — "sampling with replacement is what a traffic histogram supports"): `synth_stream` unigram = uniform position draws (Task 1), per-slice (Global Constraint 3), arms in Tasks 2–3. ✓
- bi_wiki/bi_code (bigram Markov, add-nothing smoothing, unseen-context backoff to unigram, seeded): `synth_stream` bigram (Task 1 — empirical successor multiset = add-nothing smoothing; backoff branch pinned by test). ✓
- Fit-side only: arms exist only as `*_fit_paths`; no synth eval sides anywhere; xretention/`_PAIRS` never point INTO the new arms. ✓
- Matched token budgets: same 4×1024 slice scheme; harness guard (slice count + total rows) covers the merged `fit_paths` dict — pinned by `test_k4_corpus_transfer_synthesis_guards`. ✓
- Same matrix run, both eval sides: Task 3 loop `for fit_c in fit_paths:`; pinned by the synthesis smoke; Task 4 reruns Stage-1 arms in the same run-id. ✓
- Rules (a)/(b) verbatim thresholds (10%, ½-gap), uni as estimator / shuf as control, no-higher-orders-unless-both: `synthesis` block fields `recipe_confirmed`/`order2_earns_keep`/`climb_to_order3` + the note string; §8.2/8.3 pre-draft both outcomes incl. the privacy note ("histograms, not texts"). ✓
- Parent-plan Global Constraints honored: matched budgets (1), seeds recorded (2: 20260723 everywhere, per-slice `+offset`), fit-side-only nulls, battery-before-commit with counts (485/486/488/488 over baseline 480/17/1), gpt2 yellow-flag labeling (§8 captions + verdict JSON already carries it). ✓

**2. Placeholder scan:** no TBD/TODO/"add error handling"/"similar to Task N"; every code step shows the code; `⟨…⟩` appears only inside the Task-4 §8 skeleton where it marks measured-number slots (explicitly instructed). ✓

**3. Type consistency:** `synth_stream(window, mode, seed) -> Tensor` (Task 1) matches the `load_eval_tokens` call site (`synth_stream(out, synth, synth_seed + token_offset)`) and every test import. Config fields `synth`/`synth_seed` (Task 2) match the `load_eval_tokens` kwargs and the `_WIKI_DEFAULTS` 8-tuple order (dataset_id, dataset_config, data_dir, split, text_field, shuffle_seed, synth, synth_seed). `SYNTH_FIT_CORPORA` order (shufcode, uniwiki, unicode, biwiki, bicode) = `synth_paths` dict = Config field names `<label>_fit_paths` = tyro flags in Task 4's command; the label choice makes `f"uni{eval_c}"`/`f"bi{eval_c}"` compose correctly in the rules (uniwiki/unicode/biwiki/bicode). Verdict keys asserted in tests (`per_budget[b]["synthesis"]["rules"][eval_c]{D_uni,D_bi,D_shuf,recipe_confirmed,order2_earns_keep}`, `["climb_to_order3"]`, `synthesis == {}` when absent) match the Task-3(h) construction; Task 4 Step 2 recomputes `D_uni` through the same `_cell_wins` arithmetic (per-cache D, then mean) it stores. Test-count arithmetic: 480 + 5 (T1) + 1 (T2) + 2 (T3) = 488. ✓

**Issues found & fixed during review:** (a) the existing `test_k4_corpus_transfer_smoke` asserts `set(pb["D"])` EXACTLY — synthesis D keys are added only when the arms are present, so the stage-1 tiny cfg (empty tuples) keeps the assertion true; the one-line `synthesis == {}` extension documents this instead of breaking it. (b) `_diagnostics`' centered-refit loop originally iterated all `fits` — with 8 arms that is 5 corpora × 12 layers of wasted 768² eigendecompositions; restricted to the `_PAIRS` corpora (identical set to today's on stage-1 runs, so the existing diagnostics test is unaffected) while the spectrum overlay intentionally keeps all arms. (c) the backoff branch of the bigram sampler cannot be forced by construction on a fixed window (whether the chain reaches the successorless token is seed-dependent), so the test searches seeds 0–49 for a start at the successorless token and pins that seed's output — deterministic once found, P(miss) = 2⁻⁵⁰. (d) `D_shuf` for the wiki side reuses the existing `null->wiki` cell (the Stage-1 wiki shuffle IS the without-replacement control for uni_wiki) rather than collecting a redundant sixth arm.
