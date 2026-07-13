# K4 Stage 3 — Integration + VM Duel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks 1–7 are local code (TDD); Tasks 8–12 are the VM batch (runbook kind, exact commands).

**Goal:** Make the K4 spectral codec runnable end-to-end through the streaming cache and task harnesses, then run the VM batch that (a) licenses the deployment-grade claims (corpus basis, query-heldout) and (b) duels TurboQuant on LongBench/NIAH at matched measured bits.

**Architecture:** The spectral arm joins the codec registry and the `StreamingQuantizedLayer` K-branch, fed by corpus-fitted pack files (basis + per-budget allocations, one file per model) produced by a new `k4_fit_packs` experiment. Recipes expose `k4_b{budget}` arms so `k3_longbench`/`k3_niah` run K4 with zero harness changes. The VM batch runs the referee-mandated validation (corpus G0/G1, query-heldout W) before spending on the duel.

**Tech Stack:** as Stage 0–2 (PyTorch 2.12, transformers 5.11, tyro, safetensors, parquet). VM: rented NVIDIA GH200 per `vm-interaction-guide` memory (git-bundle transport, detached setsid launches, no sudo pip).

## Global Constraints

- **NEVER `git commit` without the user's explicit approval** — stage, propose the task's exact message, STOP (the user may pre-authorize per run, as in Stages 0–2).
- Before any commit: `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` all clean. Baseline at plan time: **422 passed, 17 skipped, 1 xfailed** (branch tip `a8061b8`).
- Honest bits: ALL metadata counted. **The duel's `kv_size_bits` column reports SKEPTIC accounting at the actual S** (per-sequence pack charge `16·C/S + tier_bits`) — conservative and unarguable; the model-level number is derived in the results doc, never silently substituted.
- Deterministic; no dithered arms; fp64 fit math / fp32 codec; tiny offline test models from `tests/factories.py`, never downloads in tests.
- Pack files are BINARY ARTIFACTS (~70 MB/model): **never committed**. They regenerate deterministically from committed config (corpus doc list, seed, ridge); the pack file's JSON sidecar records the git SHA + corpus provenance.
- Referee gates from `docs/2026-07-12-k4-stage01-results.md` §External review bind Task 9: corpus G0 ≥ 0.90 licenses model-level accounting; the query-heldout arm licenses the weighted basis (fallback = unweighted KLT per spec §7).
- VM discipline per memories: don't kill long runs prematurely; per-cell checkpointing on (both k3 harnesses have `partial/` support); headroom guard on task runs; delta-parity licensing (deltas from own fp16, commit `dd84143`), never absolute TurboQuant Table-1 parity.

## File structure

- Modify `src/bmx/cache/spectral.py` — pack persistence (`save_pack_file`/`load_packs`) + corpus-W fitting entry.
- Modify `src/bmx/cache/specs.py` — `CacheCodecSpec.pack_path: str = ""`, `budget: float = 0.0`.
- Modify `src/bmx/cache/codecs.py:42-59` — register `spectral` in `_ARM_TABLE` (`s_divisible=True`).
- Modify `src/bmx/cache/streaming.py` — spectral K-branch in `_quantize_k_block_pre_rope` + pack loading in `StreamingQuantizedCache.__init__` + pack charge in `bits_per_entry()`.
- Modify `src/bmx/cache/recipes.py` — `k4_b{budget}` arms.
- Create `experiments/k4_fit_packs.py` — corpus pack fitting (basis + W policy + per-budget bits).
- Modify `experiments/k4_spectra.py` / `experiments/k4_frontier.py` — `--w-source {scored,corpus}` (the query-heldout arm).
- Tests: `tests/test_spectral.py` (persistence), `tests/test_streaming_spectral.py` (new), `tests/test_cache_specs.py` (spec fields), `tests/test_k4_experiments.py` (fit-packs smoke, w-source smoke).

---

### Task 1: Pack persistence (`save_pack_file` / `load_packs`)

**Files:**
- Modify: `src/bmx/cache/spectral.py`
- Test: `tests/test_spectral.py` (append)

**Interfaces:**
- Consumes: `SpectralBasis(enc, dec, lam)` + `pack_from_basis(basis, budget, *, tiers, group) -> SpectralPack` (post-simplify split, commit `a8061b8`).
- Produces:
  - `save_pack_file(path, bases: dict[int, SpectralBasis], budgets: tuple[float, ...], *, tiers=(0, 2, 3, 4, 5, 6, 8), group=64, meta: dict | None = None) -> None` — one safetensors file: keys `layer{i}.enc|dec|lam` (fp32) + `layer{i}.bits_b{budget:g}` (int64, from `pack_from_basis`); JSON sidecar `<path>.json` with `{tiers, group, budgets, **meta}`.
  - `load_packs(path, budget: float) -> dict[int, SpectralPack]` — reconstructs per-layer `SpectralPack`s for one budget; raises `KeyError` with the available budgets if `budget` isn't in the file.

- [ ] **Step 1: Write the failing test** (append to `tests/test_spectral.py`):

```python
def test_pack_file_roundtrip(tmp_path):
    from bmx.cache.spectral import (
        fit_spectral_basis,
        identity_whitener,
        load_packs,
        pack_from_basis,
        save_pack_file,
    )

    C = 32
    Wh, Wh_inv = identity_whitener(C)
    bases = {}
    for i in range(2):
        M, _ = _spiked_keys(S=256, C=C, seed=i)
        bases[i] = fit_spectral_basis(M, Wh, Wh_inv)
    path = tmp_path / "packs.safetensors"
    save_pack_file(path, bases, budgets=(2.5, 3.0), group=16, meta={"model": "tiny"})

    packs = load_packs(path, 2.5)
    assert set(packs) == {0, 1}
    ref = pack_from_basis(bases[0], 2.5, group=16)
    assert torch.equal(packs[0].bits, ref.bits)
    assert torch.allclose(packs[0].enc, ref.enc)
    assert packs[0].group == 16 and packs[0].budget == 2.5

    import json
    side = json.loads((tmp_path / "packs.safetensors.json").read_text())
    assert side["model"] == "tiny" and 2.5 in side["budgets"]

    import pytest
    with pytest.raises(KeyError):
        load_packs(path, 4.0)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_spectral.py::test_pack_file_roundtrip -v` → ImportError.
- [ ] **Step 3: Implement** in `spectral.py` (safetensors `save_file`/`load_file` as in `bmx.cache.collect`; bits computed via `pack_from_basis` at save time so load never re-runs the allocator; store fp64 lam? No — store the fp32 `basis.lam` AND the fp64 allocation input is only needed at save time, which is where `pack_from_basis` runs; document that reloaded packs are allocation-frozen artifacts).
- [ ] **Step 4: Verify pass**, **Step 5: Full battery + stage + propose commit** `feat(spectral): pack-file persistence — per-layer basis + per-budget allocations in one safetensors artifact`. STOP for approval (or per-run pre-auth).

---

### Task 2: Spec fields + registry entry

**Files:**
- Modify: `src/bmx/cache/specs.py`, `src/bmx/cache/codecs.py:42-59`
- Test: `tests/test_cache_specs.py` (append), `tests/test_cache_codecs.py` (append)

**Interfaces:**
- Produces: `CacheCodecSpec` gains `pack_path: str = ""` and `budget: float = 0.0` (both default-inert — every existing construction unchanged). `"spectral"` appears in `CACHE_ARMS` and `S_DIVISIBILITY_ARMS` (traits `s_divisible=True, packed=False`).
- Note: `quantize_cache`/`quantize_kv_layout` do NOT learn to run spectral (they have no pack); the streaming layer intercepts the arm before those paths (Task 3). `quantize_cache("spectral", ...)` must raise a clear error.

- [ ] **Step 1: Failing tests:**

```python
# tests/test_cache_specs.py
def test_spec_pack_fields_default_inert():
    from bmx.cache.specs import CacheCodecSpec

    s = CacheCodecSpec(arm="rtn_channel", bits=3)
    assert s.pack_path == "" and s.budget == 0.0


# tests/test_cache_codecs.py
def test_spectral_registered_but_not_dispatchable():
    import pytest

    from bmx.cache.codecs import CACHE_ARMS, S_DIVISIBILITY_ARMS, quantize_cache

    assert "spectral" in CACHE_ARMS and "spectral" in S_DIVISIBILITY_ARMS
    M = torch.randn(64, 16)
    with pytest.raises(NotImplementedError, match="pack"):
        quantize_cache("spectral", M, bits=0)
```

- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement.** In `_ARM_TABLE`: `"spectral": _ArmTraits(s_divisible=True)`. In `quantize_cache`, before the `_SPLIT_ARMS` branch: `if arm == "spectral": raise NotImplementedError("spectral requires a fitted pack; it runs through StreamingQuantizedLayer (see bmx.cache.streaming) or spectral_quantize directly")`. Add the two dataclass fields with docstring lines.
- [ ] **Step 4: Verify pass** (plus the existing `test_unknown_arm_raises` and registry tests stay green). **Step 5: battery + stage + propose** `feat(codecs): register spectral arm (pack-gated) + CacheCodecSpec pack_path/budget fields`. STOP.

---

### Task 3: Streaming integration — the spectral K-branch

**Files:**
- Modify: `src/bmx/cache/streaming.py` (`StreamingQuantizedCache.__init__`, `StreamingQuantizedLayer.__init__`/`_quantize_k_block_pre_rope`, `bits_per_entry`)
- Test: `tests/test_streaming_spectral.py` (new)

**Interfaces:**
- Consumes: Task 1 `load_packs`, Task 2 spec fields; existing branch structure at `streaming.py` `_quantize_k_block_pre_rope` (fp16 / `_FROZEN_SUBSPACE_ARMS` / general) and `compute_flush_schedule` alignment (spectral is `s_divisible`, so flush blocks are `group`-aligned — `spectral_quantize`'s `S % group == 0` assert holds by construction).
- Produces: `StreamingQuantizedCache(model_config, k_spec, v_spec, ...)` with `k_spec.arm == "spectral"` loads `load_packs(k_spec.pack_path, k_spec.budget)` once and hands `packs[i]` to layer i; the layer branch runs `spectral_quantize(M, self._pack, mse_scale=True)` and applies RoPE at true positions exactly like the frozen-subspace branch; `bits_per_entry()` adds the per-sequence skeptic charge `skeptic_charge(C, S_committed, pack.tiers)` to the K side when the arm is spectral. `k_spec.pre_rope` MUST be True for spectral (assert at cache init with a clear message).

- [ ] **Step 1: Failing tests** (offline, tiny model; mirror the existing streaming identity-invariant test style — read `tests/test_k3_experiment.py` / the streaming tests for the established fixture pattern before writing):

```python
# tests/test_streaming_spectral.py
import torch

from bmx.cache.specs import CacheCodecSpec
from bmx.cache.spectral import (
    fit_spectral_basis,
    identity_whitener,
    save_pack_file,
    spectral_quantize,
    load_packs,
)
from bmx.cache.streaming import StreamingQuantizedCache
from tests.factories import ids, tiny_llama


def _fit_tiny_packs(model, path, budget=2.5, group=8):
    """Fit per-layer packs from one hooked prefill of the tiny model."""
    from bmx.cache.collect import collect_cache, to_matrix

    cache = collect_cache(model, ids(seq=64))
    n_layer = model.config.num_hidden_layers
    bases = {}
    for i in range(n_layer):
        M = to_matrix(cache[f"layer{i}.k_pre"]).float()
        C = M.shape[1]
        Wh, Wh_inv = identity_whitener(C)
        bases[i] = fit_spectral_basis(M, Wh, Wh_inv)
    save_pack_file(path, bases, budgets=(budget,), group=group, meta={"model": "tiny"})
    return bases


def test_streaming_spectral_matches_reference(tmp_path):
    """Streamed spectral quantization must equal offline spectral_quantize on the
    committed blocks (write-once parity — the K3 invariant)."""
    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    k_spec = CacheCodecSpec(
        arm="spectral", pre_rope=True, group=8, pack_path=path, budget=2.5
    )
    v_spec = CacheCodecSpec(arm="fp16")
    cache = StreamingQuantizedCache(model.config, k_spec=k_spec, v_spec=v_spec)
    cache.attach(model)
    with cache:
        with torch.no_grad():
            model(ids(seq=64), past_key_values=cache, use_cache=True)
    # bpe accounting present and includes the skeptic pack charge
    bpe_k, bpe_v = cache.bits_per_entry()
    assert bpe_v == 16.0
    assert 2.0 < bpe_k < 16.0  # payload + scale + pack charge at tiny S


def test_streaming_spectral_requires_pre_rope(tmp_path):
    import pytest

    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    with pytest.raises(AssertionError, match="pre_rope"):
        StreamingQuantizedCache(
            model.config,
            k_spec=CacheCodecSpec(arm="spectral", pack_path=path, budget=2.5),
            v_spec=CacheCodecSpec(arm="fp16"),
        )
```

(The first test's parity assertion is deliberately the accounting-level check; a byte-exact committed-block comparison requires reaching into `cache.layers[i]` — add it if the existing streaming tests expose committed blocks; read them first and mirror the strongest available invariant, noting which you used.)

- [ ] **Step 2: Verify failure** (spectral arm rejected / AttributeError).
- [ ] **Step 3: Implement.** Cache init: when `k_spec.arm == "spectral"`: `assert k_spec.pre_rope, "spectral quantizes pre-RoPE keys; set pre_rope=True"`; `assert k_spec.pack_path, "spectral requires pack_path"`; `packs = load_packs(k_spec.pack_path, k_spec.budget)`; pass `packs.get(i)` into each layer (assert non-None per layer). Layer branch (between the frozen-subspace and general branches):

```python
        elif spec.arm == "spectral":
            M = to_matrix(k_block_pre)  # (block_len, h*d) fp32
            M_hat, codec_bpe = spectral_quantize(M, self._pack, mse_scale=True)
            k_hat_pre = from_matrix(M_hat, h)
```

`bits_per_entry()`: locate where the K-side blended bpe is finalized; when spectral, add `skeptic_charge(C, committed_total, self.k_spec_pack_tiers)` — thread the pack's tiers/C via the cache (store once at init). Keep the charge OUT of the per-block `codec_bpe` (it amortizes over the whole sequence, matching how the June honest accounting charged `16·C/S` at full-S).
- [ ] **Step 4: Verify pass** + the full existing streaming test files (`uv run pytest tests/ -q -k "streaming or codec_split or k3"`).
- [ ] **Step 5: battery + stage + propose** `feat(streaming): spectral K-branch — corpus packs stream through the write-once path with skeptic pack charge in bits_per_entry`. STOP.

---

### Task 4: Recipes — `k4_b{budget}` arms

**Files:**
- Modify: `src/bmx/cache/recipes.py`
- Test: `tests/test_cache_specs.py` (append; check where spec_pair tests live first — `grep -rl spec_pair tests/`)

**Interfaces:**
- Produces: `spec_pair(arm, *, rank=16, group=64, seed=0, pack_path="")` — new kwarg, default-inert. Arms `"k4_b{budget}"` (e.g. `k4_b2.5`) → `(CacheCodecSpec(arm="spectral", pre_rope=True, group=group, pack_path=pack_path, budget=float(budget)), CacheCodecSpec(arm="turboquant_mse", bits=2, seed=seed))`. V stays turboquant_mse@2 (the proven treatment; spec §3.3). `k4_b*` with empty pack_path raises `ValueError("k4 arms require --pack-path")`.

- [ ] **Step 1: Failing test:**

```python
def test_k4_recipe_spec_pair(tmp_path):
    import pytest

    from bmx.cache.recipes import spec_pair

    k, v = spec_pair("k4_b2.5", pack_path="/some/packs.safetensors")
    assert k.arm == "spectral" and k.pre_rope and k.budget == 2.5
    assert k.pack_path == "/some/packs.safetensors"
    assert v.arm == "turboquant_mse" and v.bits == 2
    with pytest.raises(ValueError, match="pack"):
        spec_pair("k4_b2.5")
```

- [ ] **Steps 2–4: fail → implement (parse `arm[len("k4_b"):]` as float) → pass.** Also thread `--pack-path` into `experiments/k3_niah.py` and `experiments/k3_longbench.py` Configs (one field each, passed to their `spec_pair` calls — find the call sites with grep; both already pass rank/group/seed).
- [ ] **Step 5: battery + stage + propose** `feat(recipes): k4_b{budget} arms — spectral K via corpus packs + turboquant V@2b; --pack-path threaded into k3 harnesses`. STOP.

---

### Task 5: `experiments/k4_fit_packs.py` — corpus pack fitting (+ corpus-W)

**Files:**
- Create: `experiments/k4_fit_packs.py`
- Test: `tests/test_k4_experiments.py` (append)

**Interfaces:**
- Consumes: `_k4_common.load_layer_keys/setup_rope`, `fit_spectral_basis`, `assemble_whitener`, `query_position_moment`, `save_pack_file`.
- Produces: CLI that fits per-layer bases from N corpus caches and writes ONE pack file + a spectra parquet. Config: `corpus_cache_paths: tuple[str, ...]` (required, ≥1), `out_path: str`, `model_label`, `model_name` (RoPE), `budgets=(2.0, 2.2, 2.5, 2.7, 3.0, 3.2)`, `tiers=(0,2,3,4,5,6,8)`, `group=64`, `ridge=1e-3`, `position_stride=8`, `w_source: str = "corpus"` — `"corpus"`: W pooled from the corpus caches' own stored queries (deployment-grade, fixes the referee's circularity); `"none"`: identity whitener (unweighted-KLT fallback pack). Meta records `{model_label, git_sha, corpus_cache_paths, w_source, ridge}`.

- [ ] **Step 1: Failing smoke test:**

```python
def test_k4_fit_packs_smoke(tmp_path):
    from bmx.cache.spectral import load_packs
    from experiments.k4_fit_packs import Config, main

    p1, p2 = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    _tiny_cache(p1, seed=0)
    _tiny_cache(p2, seed=1)
    out = tmp_path / "packs.safetensors"
    cfg = Config(
        corpus_cache_paths=(str(p1), str(p2)),
        out_path=str(out),
        model_label="tiny",
        budgets=(2.5,),
        group=16,
    )
    main(cfg)
    packs = load_packs(out, 2.5)
    assert 0 in packs and packs[0].enc.shape == (16, 16)
    import json
    side = json.loads((str(out) + ".json"))  # noqa: F841 — just check it parses
    side = json.loads(open(str(out) + ".json").read())
    assert side["w_source"] == "corpus"
```

- [ ] **Steps 2–4: fail → implement → pass.** Per layer: concat corpus `k_pre` matrices for Σ; for `w_source="corpus"`, accumulate `query_position_moment` over each corpus cache's `layer{i}.q` (average the W blocks across caches) with each cache's own cos/sin (positions are per-cache); `assemble_whitener`; `fit_spectral_basis`; `save_pack_file` with all budgets. Emit a small parquet (layer, am_gm, top16_energy per budget zero-dirs) via `create_run("k4_fit_packs", cfg)`.
- [ ] **Step 5: battery + stage + propose** `feat(exp): k4_fit_packs — corpus pack fitting with corpus-W (query-circularity fix) or identity-W fallback`. STOP.

---

### Task 6: Query-heldout arm in the offline gauntlet (`--w-source`)

**Files:**
- Modify: `experiments/k4_spectra.py`, `experiments/k4_frontier.py`
- Test: `tests/test_k4_experiments.py` (append)

**Interfaces:**
- Produces: both experiments gain `w_source: str = "scored"` config (`"scored"` = current behavior, byte-identical default; `"corpus"` = W from corpus caches' queries, requires `corpus_cache_paths`). Every emitted row gains a `w_source` column. The referee's decision variable is then measurable: run twice (scored vs corpus) and compare the weighted arm's win — if the corpus-W win ≈ scored-W win, the weighting generalizes; if it collapses to the unweighted arm's win, ship unweighted (spec §7 fallback).

- [ ] **Step 1: Failing test** (smoke: `w_source` column present and `"corpus"` mode runs on the tiny fixtures):

```python
def test_k4_spectra_w_source_corpus(tmp_path):
    import pandas as pd

    from experiments.k4_spectra import Config, main

    main_p, other_p = tmp_path / "m.safetensors", tmp_path / "o.safetensors"
    _tiny_cache(main_p, seed=0)
    _tiny_cache(other_p, seed=1)
    cfg = Config(
        cache_path=str(main_p),
        corpus_cache_paths=(str(other_p),),
        model_label="tiny",
        budgets=(2.5,),
        group=16,
        w_source="corpus",
        out_root=str(tmp_path / "results"),
    )
    run_dir = main(cfg)
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert (df.w_source == "corpus").all()
```

- [ ] **Steps 2–4: fail → implement → pass** (W construction becomes a function of `w_source`; scored path untouched; assert corpus paths present when `w_source="corpus"`). Existing smoke tests must still pass with the default.
- [ ] **Step 5: battery + stage + propose** `feat(exp): --w-source {scored,corpus} — the query-heldout arm for the weighted-basis license`. STOP.

---

### Task 7: Local pre-flight gate + push

**Files:** none new — verification + transport.

- [ ] **Step 1:** Full battery clean; then the gpt2 end-to-end pre-flight (all local, minutes):

```bash
uv run python experiments/k4_fit_packs.py --corpus-cache-paths results/cache/gpt2_1024_off1024.safetensors results/cache/gpt2_1024_off2048.safetensors results/cache/gpt2_1024_off3072.safetensors results/cache/gpt2_1024_off4096.safetensors --out-path results/cache/k4_packs_gpt2.safetensors --model-label gpt2 --group 64
uv run python experiments/k4_spectra.py --cache-path results/cache/gpt2_1024.safetensors --corpus-cache-paths <same 4> --model-label gpt2 --w-source corpus
```

Read the second run's parquet: the corpus-W weighted arm must run and produce finite headline numbers (this is a mechanism pre-flight, not a gate — gpt2 corpus-W retention lands where it lands; record it).
- [ ] **Step 2:** Smoke the streaming path on gpt2 for real: a 3-line scratchpad script building `StreamingQuantizedCache` with `spec_pair("k4_b2.5", pack_path="results/cache/k4_packs_gpt2.safetensors")` on the gpt2 model, one 512-token prefill, print `bits_per_entry()` — expect bpe_k ≈ 2.5 + 0.25 + pack charge at S=512 (16·768/512 = 24 → dominated at tiny S; PRINT it, sanity-check the arithmetic, note that real-S amortization is the VM's job).
- [ ] **Step 3:** STOP — propose pushing `feat/triton-decode-kernel` to origin (the VM transport prerequisite; ~70 commits). User approves the push explicitly.

---

## VM batch (Tasks 8–12) — one rented GH200, ordered, each gated

Transport per `vm-interaction-guide`: push → VM pull (or git bundle), `scripts/vm_setup.sh`, `uv run pytest -q` (expect local 422/17/1 + Triton extras; record the actual count as the new GH200 baseline). Long runs: detached setsid + log under `results/logs/`; NEVER sudo pip.

### Task 8: [VM-RUN] Corpus caches + long-context cache

```bash
for OFF in 2048 4096 6144 8192 10240 12288 14336 16384; do
  uv run python experiments/collect_cache.py --model-name meta-llama/Llama-3.1-8B --seq-len 2048 --token-offset $OFF
done
uv run python experiments/collect_cache.py --model-name meta-llama/Llama-3.1-8B --seq-len 32768   # long-context spectra + real 0.5-bpe amortization point
```

Minutes on GH200. 8 corpus docs (n = 16·C·... = 16384 rows ≫ C=1024 — kills the rank-deficiency confound) + 4 of them reserved as SCORED caches for error bars (fit on the other 4 — document the split in the run log: fit = offsets {2048,4096,6144,8192}, scored = {10240,12288,14336,16384}).

### Task 9: [VM-RUN] The license runs — corpus G0/G1 + query-heldout (GATES; cheap; run BEFORE any duel spend)

```bash
uv run python experiments/k4_fit_packs.py --corpus-cache-paths <the 4 fit caches> \
  --out-path results/cache/k4_packs_llama31.safetensors --model-label llama-3.1-8b \
  --model-name meta-llama/Llama-3.1-8B --w-source corpus
# per scored cache (4x), both w_source values:
uv run python experiments/k4_spectra.py --cache-path <scored cache> --corpus-cache-paths <4 fit caches> \
  --model-label llama-3.1-8b --model-name meta-llama/Llama-3.1-8B --w-source corpus
uv run python experiments/k4_frontier.py --cache-path <scored cache> --corpus-cache-paths <4 fit caches> \
  --model-label llama-3.1-8b --model-name meta-llama/Llama-3.1-8B --w-source corpus
uv run python experiments/k4_frontier.py --cache-path <scored cache> ... --w-source scored   # the comparison pair
```

**Gate A (G0-corpus):** retention ≥ 0.90 with n≫C → model-level accounting licensed. **Gate B (query-heldout):** corpus-W weighted win within ~20% of scored-W weighted win → weighting licensed; if it collapses to the unweighted win, the production pack becomes `w_source="none"` (unweighted KLT) and the paper drops the W^½ claim (spec §7). **Gate C (error bars):** min/max G1 win across the 4 scored caches — the duel proceeds only if the min still clears 1× in both accounting modes (it starts from 2.27× floor, so this should be comfortable; if not, STOP and reassess). Write `docs/<date>-k4-vm-license-results.md` with the three gate calls; commit parquets + doc (bundle back for approval).

### Task 10: [VM-RUN] Llama sensitivity + allocated bonus arm (GPU-fast now)

```bash
uv run python experiments/k4_alloc.py --model-name meta-llama/Llama-3.1-8B \
  --frontier-parquet results/k4_frontier/<corpus-W llama run>/metrics.parquet
```

Produces the Llama s_i table + allocation. Optional duel bonus arm: per-layer-budget K4 (needs a small recipe extension — only build if Gate A–C all pass and time permits; otherwise record allocation for the paper's G2 section and move on).

### Task 11: [VM-RUN] The duel — staged spend

**Stage A probe (hours, ~$):** all candidate arms, 2 categories, n=100, checkpointed:

```bash
uv run python -m experiments.k3_longbench --model-name meta-llama/Llama-3.1-8B-Instruct \
  --arms fp16 k4_b2.2 k4_b2.5 k4_b3.0 turboquant_mse_b3 turboquant_mse_k3v2 turboquant_mse_b2 \
  --pack-path results/cache/k4_packs_llama31_instruct.safetensors \
  --categories synthetic code --n-samples 100
```

**NOTE:** the duel model is the INSTRUCT variant — fit a separate pack file from Instruct-model corpus caches (repeat Task 8/9's fit for `meta-llama/Llama-3.1-8B-Instruct`; base-model packs are NOT assumed transferable — that's a measurable question, record it if probed). Read the probe against the banked b3/b2 numbers (sanity: fp16 Synthetic ≈ 52, Code ≈ 62 on the un-truncated path). **Prune:** keep fp16 + the 2 best K4 points + b3 + K3V2 for the full table.

**Stage B full table (the ~40–60h spend):** survivors × full 6-category LongBench-V1 (`n_samples=None`), headroom guard, delta-parity licensing; plus NIAH:

```bash
uv run python -m experiments.k3_niah --model-name meta-llama/Llama-3.1-8B-Instruct \
  --arms fp16 k4_b2.5 turboquant_mse_b3 turboquant_mse_k3v2 --pack-path <packs> \
  --lengths 4096 8192 16384 32768 --depths 0.1 0.3 0.5 0.7 0.9
```

(NIAH includes the standing b3-retrieval-edge question; 64k/128k long legs only on the fixed streaming path with per-cell checkpoints, per the §3b lesson.)

### Task 12: [VM-RUN] Verdict doc + traceability

Success criteria verbatim from spec §10: at ≤2.7 measured bits, K4 Avg ≥ +2 over the best TurboQuant-family arm at equal-or-fewer bits (incl. K3V2); K4 matches b3's Avg (within noise) at ≥0.5 fewer bits; NIAH recall ≥ turboquant at matched bits. `kv_size_bits` is skeptic-at-actual-S; the model-level number quoted alongside if Gate A passed. Write `docs/<date>-k4-duel-results.md` (kill-or-confirm style, both outcomes pre-drafted as templates in the doc header), update the paper skeleton's C1 row, commit parquets + doc back via bundle. STOP for approval.

---

## Self-Review

**Spec coverage:** spec §4 Stage 3 (VM duel: operating points, baselines incl. K3V2, delta-parity, NIAH b3 rider, checkpointing) → Tasks 11–12. Referee requirements (corpus basis, query-heldout, error bars, real 32k amortization) → Tasks 8–9 + the 32k cache. §5 accounting (skeptic primary in kv_size_bits, model-level licensed by Gate A) → Tasks 3/12. §6 streaming integration through the arm-set gate → Task 3 (new branch beside `_FROZEN_SUBSPACE_ARMS`, packed/fused sites untouched). §7 fallbacks (unweighted pack on Gate-B failure) → Task 9 Gate B. Kernel/latency work: still out of scope (streaming is the duel path — the fastest quantized decode per the standing memory).

**Placeholder scan:** all code steps carry real code; runbook tasks carry exact commands with the one deliberate variable (`<scored cache>`/`<packs>` enumerated by Task 8's recorded split). The Instruct-pack note closes the base-vs-instruct gap explicitly rather than assuming transfer.

**Type consistency:** `save_pack_file(path, bases, budgets, *, tiers, group, meta)` / `load_packs(path, budget)` consistent across Tasks 1/3/5; `CacheCodecSpec.pack_path/budget` across Tasks 2/3/4; `w_source` values `{scored, corpus}` in Task 6 vs `{corpus, none}` in Task 5 — DELIBERATE: experiments compare scored-vs-corpus W; the production fitter never uses scored-W (that's the circularity being eliminated) and instead offers the unweighted fallback. Flagged here so it isn't "fixed" into an inconsistency.

**Known open decision deferred to run time:** whether the duel's K4 operating points are {2.2, 2.5, 3.0} exactly — Task 9's corpus-basis frontier picks the final three (spaced ≥0.5 measured bits, per spec §4 Stage 3).
