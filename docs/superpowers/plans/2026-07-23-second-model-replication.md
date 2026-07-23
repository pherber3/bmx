# Second-Model Replication — Qwen2.5 K4 Gates + Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks 1–6 are local code/verification (TDD, CPU); Tasks 7–13 are the VM batch (runbook kind, exact commands).

**Goal:** Replicate the K4 gates (A/B/C) + the n=100 synthetic+code probe + the bits-vs-context NIAH points (4k–32k) on a second model family — **Qwen2.5-7B(-Instruct)** — so the paper's claim (`docs/2026-07-15-k4-duel-results.md`) is a property of the method, not a Llama-3.1 artifact. Scope is EXACTLY gates + probe + bits curve; a full 6-category Qwen table is a **post-probe decision**, not this plan.

**Architecture:** Zero expected `src/bmx` changes — that is the point. The K4 stack (`hf_compat` introspection → `attach()` k_proj hooks → `StreamingQuantizedCache` write-once path → corpus packs → `k3_longbench`/`k3_niah` harnesses) claims model-family agnosticism; this plan converts that claim from assumption to test. Qwen2 is a Llama-style decoder (`model.model.layers[i].self_attn.k_proj`, GQA h_kv=4, d=128 → C=512 — a power of 2, so every batched-flush license in `streaming._flush_batchable` holds; static RoPE, so the batched RoPE-table extension note at `streaming.py:318-323` is satisfied). Any Llama-ism found becomes a minimal fix + regression test, reported loudly. Fallback model if Task 1 verification fails: `mistralai/Mistral-7B-Instruct-v0.3` (same checks, h_kv=8, d=128 → C=1024).

**Tech Stack:** as the K4 campaign (PyTorch 2.12, transformers 5.11, tyro, safetensors, parquet). VM: rented NVIDIA GH200 per `vm-interaction-guide` memory (git transport, detached setsid launches, no sudo pip). Qwen2.5 checkpoints are public/ungated (no HF token dance, unlike Llama).

## Global Constraints

- **NEVER `git commit` without the user's explicit approval** — stage, propose the task's exact message, STOP (the user may pre-authorize per run, as in prior campaigns).
- Before any commit: `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` all clean. Baseline at plan time: **451 passed, 17 skipped, 1 xfailed** (branch `feat/triton-decode-kernel` @ `11062ff`, 99 s).
- Tiny offline test models from `tests/factories.py`; **never download in tests** (Task 5's 0.5B pre-flight is an experiment run, not a test).
- Pack files are BINARY ARTIFACTS: never committed; they regenerate from committed config; the JSON sidecar records git SHA + corpus provenance.
- Honest bits: `kv_size_bits` in every probe/NIAH parquet is SKEPTIC accounting at actual S (per-sequence pack charge `16·C/S` + tier map). Model-level numbers only if Gate A passes (it is EXPECTED to fail — see Task 9).
- **Delta-parity discipline:** `chat_wrap` stays **False** everywhere (the 43.7-pt chat-wrap sensitivity killed absolute parity; all banked numbers are chat_wrap=False). `max_prompt_tokens` stays at the default 31500.
- **Ops discipline (July campaign, binding):** one 8B process at a time at 31.5k prompts (two → mutual OOM at 96 GB); VM-side chained drivers with per-cell `OK`/`FAILED` lines, never local watchers; verify every detached launch 60–90 s in (log advancing + GPU busy); never kill long runs prematurely; never sudo pip into the venv; per-cell `partial/` shards mean crashes resume via `--resume`.
- Per-sample shards (`write_samples_shard`, already on the harness) mean the probe gets bootstrap CIs for free — no extra instrumentation.

## File structure

- Modify: `tests/factories.py` — `tiny_qwen2()` mirroring `tiny_llama()`.
- Modify: `tests/test_streaming_spectral.py` — parametrize the streaming-spectral tests over `{tiny_llama, tiny_qwen2}`.
- Create: `tests/test_qwen2_compat.py` — hf_compat resolution, k2b streaming smoke, k4 end-to-end generate, EOS-list stop test.
- No `src/bmx` or `experiments/` changes expected (harness `model_name`/`pack_path` are already parameters). Any discovered Llama-ism: minimal fix + regression test in the task that found it.
- Create (VM, Task 13): `docs/2026-07-2X-qwen25-replication-results.md`.

## Naming (derived, not invented)

`collect_cache.py` computes `model_short = model_name.split("/")[-1].lower()`:

| model | model_short | cache file pattern |
|---|---|---|
| `Qwen/Qwen2.5-7B` | `qwen2.5-7b` | `results/cache/qwen2.5-7b_2048_off{N}.safetensors` |
| `Qwen/Qwen2.5-7B-Instruct` | `qwen2.5-7b-instruct` | `results/cache/qwen2.5-7b-instruct_2048_off{N}.safetensors` |
| `Qwen/Qwen2.5-0.5B` (pre-flight) | `qwen2.5-0.5b` | `results/cache/qwen2.5-0.5b_1024*.safetensors` |

Pack files (mirroring `k4_packs_llama31{,_instruct}`): `results/cache/k4_packs_qwen25.safetensors` (base, gates) and `results/cache/k4_packs_qwen25_instruct.safetensors` (Instruct, probe/NIAH). Base packs are NOT assumed transferable to Instruct — same decision as the Llama campaign.

---

### Task 1: Model-choice verification (config-only, local, minutes)

The design decision names Qwen2.5-7B-Instruct **conditional on concrete config checks** — verify, don't assume. This downloads two ~1 KB config.json files, no weights.

**Files:** none committed — scratchpad script; findings recorded in the run log and echoed into Task 13's verdict doc.

- [ ] **Step 1: Run the check script** (scratchpad):

```python
# scratchpad/check_qwen_config.py
from transformers import AutoConfig, GenerationConfig

for name in ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"):
    c = AutoConfig.from_pretrained(name)
    assert not hasattr(c, "text_config") or not hasattr(c.text_config, "num_attention_heads"), (
        "unexpected multimodal wrapper"  # resolve_text_config must return c itself
    )
    d = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    C = c.num_key_value_heads * d
    g = GenerationConfig.from_pretrained(name)
    print(f"{name}: arch={c.architectures} layers={c.num_hidden_layers} "
          f"h_kv={c.num_key_value_heads} d={d} C={C} theta={c.rope_theta} "
          f"rope_scaling={getattr(c, 'rope_scaling', None)} "
          f"max_pos={c.max_position_embeddings} "
          f"sliding={getattr(c, 'use_sliding_window', None)} "
          f"vocab={c.vocab_size} eos={g.eos_token_id}")
    # --- the binding checks ---
    assert c.architectures == ["Qwen2ForCausalLM"], "attach() needs Llama-style k_proj layout"
    assert d & (d - 1) == 0 and C & (C - 1) == 0, "batched-flush fwht licenses need pow2 d and C"
    rs = getattr(c, "rope_scaling", None)
    assert rs is None or (rs.get("rope_type", rs.get("type")) in ("default", "llama3")), (
        f"dynamic/NTK rope_scaling {rs} breaks the batched RoPE-table extension "
        "(streaming.py:318-323) — FALLBACK to Mistral-7B-Instruct-v0.3"
    )
    assert c.max_position_embeddings >= 32768, "32k NIAH point + 31500 LongBench truncation"
    assert not getattr(c, "use_sliding_window", False), (
        "sliding-window attention changes cache semantics"
    )
print("ALL CHECKS PASS")
```

Expected (from model cards, to be CONFIRMED by the run): 28 layers, h_kv=4, d=128, C=512, rope_theta=1e6, rope_scaling=None (static), use_sliding_window=False, vocab 152064; Instruct eos is a **list** `[151645, 151643]` (motivates Task 4); Instruct max_position_embeddings may be 32768 (base may be higher) — 32768 is sufficient: `build_niah_prompt` reserves a 200-token buffer, so the 32k NIAH prompt + template + 50 decode tokens stays inside the window, and LongBench's 31500 truncation + generation fits too. Record the actual values.

- [ ] **Step 2: Gate.** ALL asserts pass → Qwen2.5-7B-Instruct confirmed, proceed. ANY assert fails → rerun the same script for `mistralai/Mistral-7B-Instruct-v0.3` (+ base `mistralai/Mistral-7B-v0.3`; additionally assert `sliding_window in (None,)` for v0.3) and substitute Mistral names/`model_short` (`mistral-7b-v0.3`, `mistral-7b-instruct-v0.3`) throughout Tasks 2–13; the factory in Task 2 becomes `tiny_mistral` (`MistralConfig` — same field names). Record which branch was taken.

---

### Task 2: `tiny_qwen2` factory + hf_compat resolution tests

**Files:**
- Modify: `tests/factories.py`
- Create: `tests/test_qwen2_compat.py`

**Interfaces:**
- Produces: `tiny_qwen2()` — offline `Qwen2Config` model in eval mode, geometry mirroring `tiny_llama` (2 layers, 4 heads, 2 kv heads, hidden 32 → d=8, C=16 pow2, `max_position_embeddings=512` to exceed the 128-token PAGE flush threshold, `vocab_size=97` so `ids()` works unchanged). Qwen2Config defaults give qkv bias=True and no sliding window — deliberately kept (that IS the second family's shape).
- Consumes: `bmx.cache.hf_compat.{model_config_n_layers, resolve_decoder_layers, resolve_text_config}` — the design decision requires these verified against Qwen2 config shape **by test, not assumption**.

- [ ] **Step 1: Write the failing tests** (`tests/test_qwen2_compat.py`):

```python
"""Second-model-family (Qwen2) compatibility: hf_compat resolution, streaming
attach, spectral packs, EOS-list stop. Everything offline via tiny_qwen2."""

import torch

from bmx.cache.hf_compat import (
    model_config_n_layers,
    resolve_decoder_layers,
    resolve_text_config,
    resolve_vocab_size,
)
from tests.factories import ids, tiny_qwen2


def test_hf_compat_resolves_qwen2():
    m = tiny_qwen2()
    assert model_config_n_layers(m) == 2
    layers = resolve_decoder_layers(m)
    assert hasattr(layers[0], "self_attn") and hasattr(layers[0].self_attn, "k_proj")
    tc = resolve_text_config(m.config)
    assert tc is m.config  # no multimodal wrapper on Qwen2ForCausalLM
    assert tc.num_key_value_heads == 2
    assert resolve_vocab_size(m.config) == 97
```

- [ ] **Step 2: Verify failure** — `uv run pytest tests/test_qwen2_compat.py -v` → ImportError (`tiny_qwen2` missing).
- [ ] **Step 3: Implement** in `tests/factories.py` (import `Qwen2Config, Qwen2ForCausalLM` alongside the existing transformers imports):

```python
def tiny_qwen2():
    """Tiny Qwen2 mirroring tiny_llama's geometry — the second-model-family
    fixture. Llama-style layout (model.model.layers[i].self_attn.k_proj) but
    with Qwen2's own defaults (qkv bias=True, no sliding window)."""
    cfg = Qwen2Config(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=32,
        intermediate_size=64,
        vocab_size=97,
        max_position_embeddings=512,  # exceed the 128-token PAGE flush threshold
    )
    torch.manual_seed(3)
    return Qwen2ForCausalLM(cfg).eval()
```

- [ ] **Step 4: Verify pass.** **Step 5: full battery + stage + propose** `test(qwen2): tiny_qwen2 factory + hf_compat resolution on the second family`. STOP.

---

### Task 3: Streaming attach + spectral smoke on `tiny_qwen2`

**Files:**
- Modify: `tests/test_streaming_spectral.py` (parametrize over factories)
- Modify: `tests/test_qwen2_compat.py` (append k2b + k4 end-to-end tests)

**Interfaces:**
- Consumes: `StreamingQuantizedCache.attach()` (hooks `self_attn.k_proj` — the architecture gate at `streaming.py:594-599`), `spec_pair("k2b")`, `spec_pair("k4_b2.5", pack_path=...)`, `_fit_tiny_packs` (already model-parametric — takes `model`), `generate_through_cache`.
- Produces: proof that the write-once streaming path, the spectral K-branch, and the k4 recipe all run on a Qwen2 module tree.

- [ ] **Step 1: Parametrize the spectral streaming tests.** In `tests/test_streaming_spectral.py`, convert `test_streaming_spectral_matches_reference` and `test_streaming_spectral_requires_pre_rope` (and any sibling using `tiny_llama` + `_fit_tiny_packs`) to `@pytest.mark.parametrize("factory", [tiny_llama, tiny_qwen2], ids=["llama", "qwen2"])`, replacing the direct `tiny_llama()` calls with `factory()`. `_fit_tiny_packs` needs no change. **Read the file's header comment first** — the seq=150/recent_window=8 pattern (which actually crosses the PAGE flush threshold) must be preserved verbatim for the new parametrization.
- [ ] **Step 2: Append the k2b + k4 tests** (`tests/test_qwen2_compat.py`):

```python
def test_streaming_k2b_qwen2(tmp_path):
    """The proven k2b arm streams through a Qwen2 module tree: attach() hooks
    fire, pages flush, bpe accounting is real (<16). Mirror the fixture pattern
    of tests/test_streaming_cache.py::test_k2b_pre_rope_streams_token_by_token
    (seq=150, recent_window=8 — read it first, copy its invariant exactly)."""
    from bmx.cache.recipes import spec_pair
    from bmx.cache.streaming import StreamingQuantizedCache

    model = tiny_qwen2()
    k_spec, v_spec = spec_pair("k2b", rank=4, group=8)
    cache = StreamingQuantizedCache(
        model.config, k_spec=k_spec, v_spec=v_spec, recent_window=8
    )
    cache.attach(model)
    with cache:
        with torch.no_grad():
            model(ids(seq=150), past_key_values=cache, use_cache=True)
    bpe_k, bpe_v = cache.bits_per_entry()
    assert bpe_k < 16.0 and bpe_v < 16.0  # at least one page actually flushed


def test_generate_k4_qwen2(tmp_path):
    """k4_b2.5 end-to-end (attach + hooks + spectral flush + greedy decode) on
    Qwen2 — the exact recipe the VM probe runs."""
    from bmx.cache.generate import generate_through_cache
    from bmx.cache.recipes import spec_pair
    from tests.test_streaming_spectral import _fit_tiny_packs

    model = tiny_qwen2()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)  # budget=2.5, group=8

    class _StubTok:
        eos_token_id = None

        def decode(self, t, skip_special_tokens=True):
            return " ".join(map(str, t.tolist() if hasattr(t, "tolist") else t))

    k_spec, v_spec = spec_pair("k4_b2.5", group=8, pack_path=path)
    out = generate_through_cache(
        model, _StubTok(), ids(seq=150), n_prefill=128,
        k_spec=k_spec, v_spec=v_spec, max_new_tokens=4,
    )
    assert isinstance(out, str) and out
```

(If `spec_pair("k2b", ...)`'s signature differs from the assumed kwargs, mirror the actual call sites in `tests/test_streaming_cache.py` — the invariant, not the literals, is binding. Same for the `StreamingQuantizedCache(..., recent_window=8)` kwarg name.)
- [ ] **Step 3: fail → implement (no src change expected) → pass.** If any test fails for a REAL architecture reason (not a fixture literal), that is a finding: fix minimally in `src/bmx`, add the regression test, and flag it in the task report.
- [ ] **Step 4: full battery + stage + propose** `test(qwen2): streaming + spectral + k4 recipe run on the second family (parametrized spectral suite)`. STOP.

---

### Task 4: Harness Llama-ism sweep + EOS-list stop test

**Files:**
- Modify: `tests/test_qwen2_compat.py` (append)
- Possibly modify: whatever the sweep finds (expected: nothing).

**Interfaces:** `generate_through_cache` reads the EOS set as `generation_config.eos_token_id → model.config.eos_token_id → tokenizer.eos_token_id`, and already normalizes list-or-int (`generate.py:66-71`). Qwen2.5-Instruct ships a **list** (`[151645, 151643]`) — pin that this generalizes, since Llama-3.1's list was the only case ever exercised.

- [ ] **Step 1: The sweep.** `grep -rn "Llama\|llama\|128009\|128001\|128008\|<|eot_id|>" src/bmx/cache/ experiments/k3_*.py experiments/_common.py` and audit each hit: default `model_name` strings (fine — overridden by `--model-name`), docstring references (fine), `build_chat`/chat-wrap machinery (dormant — `chat_wrap=False` is a Global Constraint), hard-coded token ids (NOT fine — none expected; any found is a bug to fix). Record the audit result in the task report.
- [ ] **Step 2: Failing EOS test:**

```python
def test_generate_stops_on_any_eos_in_list():
    """Qwen2.5-Instruct's generation_config.eos_token_id is a LIST — the decode
    loop must stop on ANY member, not just the first/tokenizer one."""
    from bmx.cache.generate import generate_through_cache
    from bmx.cache.specs import CacheCodecSpec

    class _StubTok:
        eos_token_id = None

        def decode(self, t, skip_special_tokens=True):
            return " ".join(map(str, t.tolist() if hasattr(t, "tolist") else t))

    model = tiny_qwen2()
    fp16 = CacheCodecSpec(arm="fp16")
    prompt = ids(seq=24)
    # Probe: which token does greedy decode emit first?
    out = generate_through_cache(
        model, _StubTok(), prompt, n_prefill=12,
        k_spec=fp16, v_spec=fp16, max_new_tokens=4, strip=False,
    )
    first = int(out.split()[0])
    # Re-run with an eos LIST containing that token (plus a never-emitted one):
    model.generation_config.eos_token_id = [96, first]
    out2 = generate_through_cache(
        model, _StubTok(), prompt, n_prefill=12,
        k_spec=fp16, v_spec=fp16, max_new_tokens=4, strip=False,
    )
    assert len(out2.split()) == 1  # stopped ON the eos token, immediately
```

(If the loop excludes the EOS token from the emitted string, the assert becomes `out2 == ""` / length 0 — read `generate.py`'s stop semantics at line ~112 and pin the ACTUAL behavior; the invariant is "stops at the first list member", not the off-by-one.)
- [ ] **Step 3: fail → (expected: passes immediately once written correctly, since the set-normalization already exists — then it is a pin, not a fix; run it red-first by asserting the wrong length once to confirm the test bites) → pass.**
- [ ] **Step 4: full battery + stage + propose** `test(generate): EOS-list stop pinned on the second family + harness Llama-ism audit`. STOP.

---

### Task 5: Local real-checkpoint pre-flight — Qwen2.5-0.5B on CPU

The tiny factory proves the module tree; it cannot prove tokenizer/wikitext plumbing, real RoPE (theta=1e6), or the hooked-capture ↔ stored-K consistency on real weights. Qwen2.5-0.5B (h_kv=2, d=64 → C=128, pow2; ~1 GB download) proves all of it locally in minutes — the analogue of the K4 plan's gpt2 pre-flight.

- [ ] **Step 1: Collect two caches** (real Qwen2 checkpoint, CPU, bf16):

```bash
uv run python experiments/collect_cache.py --model-name Qwen/Qwen2.5-0.5B --seq-len 1024
uv run python experiments/collect_cache.py --model-name Qwen/Qwen2.5-0.5B --seq-len 1024 --token-offset 2048
```

- [ ] **Step 2: Fit + spectra, corpus-W:**

```bash
uv run python experiments/k4_fit_packs.py --corpus-cache-paths results/cache/qwen2.5-0.5b_1024_off2048.safetensors --out-path results/cache/k4_packs_qwen25_05b.safetensors --model-label qwen2.5-0.5b --model-name Qwen/Qwen2.5-0.5B --w-source corpus
uv run python experiments/k4_spectra.py --cache-path results/cache/qwen2.5-0.5b_1024.safetensors --corpus-cache-paths results/cache/qwen2.5-0.5b_1024_off2048.safetensors --model-label qwen2.5-0.5b --model-name Qwen/Qwen2.5-0.5B --w-source corpus
```

**The binding check is `setup_rope`'s self-validation line**: `[rope_validation] rel_fro(apply_rope(k_pre), k) < 2e-2` on a real Qwen2 checkpoint — it simultaneously validates `rope_cos_sin` against Qwen2's rope config, the pre-RoPE hook capture (k_proj output INCLUDES Qwen's bias — exactly what apply_rope must reproduce), and the (h,S,d) layout. A failure here is a hard STOP (architecture finding, diagnose before any VM spend). Mechanism read only otherwise: finite headline numbers; 0.5B retention lands where it lands — record it, don't gate on it.
- [ ] **Step 3: Streaming smoke on the real checkpoint** — 3-line scratchpad script: `spec_pair("k4_b2.5", pack_path="results/cache/k4_packs_qwen25_05b.safetensors")`, `StreamingQuantizedCache` on the 0.5B model, one 512-token prefill via `attach()`, print `bits_per_entry()` — expect bpe_k ≈ payload + scales + pack charge (16·128/512 = 4.0 dominates at S=512; PRINT and sanity-check the arithmetic; real-S amortization is the VM's job).

---

### Task 6: Local gate + push

- [ ] **Step 1:** `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` — expect 451+new passed / 17 skipped / 1 xfailed. Stage everything from Tasks 2–4; propose the combined message if not already committed per-task. STOP for approval.
- [ ] **Step 2:** STOP — propose pushing `feat/triton-decode-kernel` to origin (VM transport prerequisite). User approves the push explicitly.

---

## VM batch (Tasks 7–13) — one rented GH200, ordered, each gated

Transport per `vm-interaction-guide`: push → VM pull (or git bundle), `bash scripts/vm_setup.sh`, `uv run pytest -q` (record the actual GH200 count as the new baseline — local 451/17/1 plus CUDA/Triton extras). Long runs: detached `setsid` + log under `results/logs/`; verify 60–90 s in; VM has no push creds — results come back by git bundle.

**Cost estimate (pre-registered so overruns are visible):** caches ~30 min; gates <1 h; Instruct packs ~10 min; probe 3–5 h; NIAH 2–4 h. Total ≈ one GPU-day.

### Task 7: [VM-RUN] Setup + baseline

```bash
bash scripts/vm_setup.sh
uv run pytest -q            # record GH200 baseline (incl. CUDA batched-flush A/B: the licenses re-pin on CUDA here)
```

`tests/test_streaming_batched_flush.py`'s cuda parametrization must be green — it is the bitwise license the probe's speed depends on. Qwen weights download on first use (public, no token).

### Task 8: [VM-RUN] Corpus caches — base (gates) + Instruct (probe packs)

```bash
for OFF in 2048 4096 6144 8192 10240 12288 14336 16384; do
  uv run python experiments/collect_cache.py --model-name Qwen/Qwen2.5-7B --seq-len 2048 --token-offset $OFF
done
for OFF in 2048 4096 6144 8192; do
  uv run python experiments/collect_cache.py --model-name Qwen/Qwen2.5-7B-Instruct --seq-len 2048 --token-offset $OFF
done
```

Minutes on GH200. **Split (document in the run log, mirroring the Llama protocol):** fit = offsets {2048, 4096, 6144, 8192}, scored = {10240, 12288, 14336, 16384}. 4 fit caches × 2048 rows = 8192 rows ≫ C=512 (16× — an even better rank-deficiency margin than Llama's 8× at C=1024).

### Task 9: [VM-RUN] License gates A/B/C (GATES; cheap; run BEFORE any probe spend)

```bash
FIT="results/cache/qwen2.5-7b_2048_off2048.safetensors results/cache/qwen2.5-7b_2048_off4096.safetensors results/cache/qwen2.5-7b_2048_off6144.safetensors results/cache/qwen2.5-7b_2048_off8192.safetensors"

uv run python experiments/k4_fit_packs.py --corpus-cache-paths $FIT \
  --out-path results/cache/k4_packs_qwen25.safetensors --model-label qwen2.5-7b \
  --model-name Qwen/Qwen2.5-7B --w-source corpus

for S in 10240 12288 14336 16384; do
  SC=results/cache/qwen2.5-7b_2048_off${S}.safetensors
  uv run python experiments/k4_spectra.py  --cache-path $SC --corpus-cache-paths $FIT \
    --model-label qwen2.5-7b --model-name Qwen/Qwen2.5-7B --w-source corpus
  uv run python experiments/k4_spectra.py  --cache-path $SC \
    --model-label qwen2.5-7b --model-name Qwen/Qwen2.5-7B --w-source scored
  uv run python experiments/k4_frontier.py --cache-path $SC --corpus-cache-paths $FIT \
    --model-label qwen2.5-7b --model-name Qwen/Qwen2.5-7B --w-source corpus
  uv run python experiments/k4_frontier.py --cache-path $SC \
    --model-label qwen2.5-7b --model-name Qwen/Qwen2.5-7B --w-source scored
done
```

Gate reads (same metric names as the Llama campaign — `retention_corpus`/`g0_pass_corpus` from k4_spectra's verdict block; `win_model`, `win_skeptic_deploy`, `layer_win_fraction_*` from k4_frontier's summary rows at budget 2.5):

- **Gate A (G0-corpus):** retention ≥ 0.90 licenses model-level accounting. **PRE-REGISTERED EXPECTATION: FAIL** — Llama measured 0.56–0.64 and the corpus-scale ablation proved the ceiling structural. Record the Qwen range either way. A Qwen fail in a similar band REPLICATES the structural-transfer-ceiling finding on a second family (a paper point). A PASS would be a genuine surprise (licenses model-level accounting on Qwen — report it as such, don't average it away). Gate A's outcome is NOT part of the replicates/does-not-replicate verdict.
- **Gate B (query-heldout):** the weighted arm's increment under corpus-W within ~20% of its scored-W increment (Llama: 1.54–1.59× vs ≈1.7×, transfer ratio 0.85–0.88 — PASS). PASS → corpus-W weighted packs stand. FAIL → refit both pack files with `--w-source none` (unweighted-KLT fallback, spec §7) and record that Qwen drops the W^½ claim.
- **Gate C (error bars — THE GO/NO-GO):** min over the 4 scored caches of `win_model` AND `win_skeptic_deploy` at budget 2.5 > 1×, with `layer_win_fraction ≥ 0.9` (frontier `g1_pass`) on every cache. Llama's floor was 6.19× — anything clearing 1× on both accountings proceeds. **FAIL → STOP: no probe spend**; write the does-not-replicate verdict at the gate layer (Task 13 template B) with the measured wins.

### Task 10: [VM-RUN] Instruct packs

```bash
FIT_I="results/cache/qwen2.5-7b-instruct_2048_off2048.safetensors results/cache/qwen2.5-7b-instruct_2048_off4096.safetensors results/cache/qwen2.5-7b-instruct_2048_off6144.safetensors results/cache/qwen2.5-7b-instruct_2048_off8192.safetensors"
uv run python experiments/k4_fit_packs.py --corpus-cache-paths $FIT_I \
  --out-path results/cache/k4_packs_qwen25_instruct.safetensors --model-label qwen2.5-7b-instruct \
  --model-name Qwen/Qwen2.5-7B-Instruct --w-source corpus   # 'none' if Gate B failed
```

Default budgets (2.0, 2.2, 2.5, 2.7, 3.0, 3.2) cover the probe arms (2.2, 2.5). Base packs are not reused for the Instruct model (measured Llama-campaign decision, kept).

### Task 11: [VM-RUN] The probe — n=100 synthetic+code, 5 arms

**Chained VM-side driver** (per ops discipline — probe then NIAH sequentially, one 8B process at a time, per-cell OK/FAILED lines):

```bash
mkdir -p results/logs
cat > /tmp/qwen_batch.sh <<'SH'
set -u
cd "$HOME/bmx"
run() { NAME=$1; shift; echo "=== $NAME START $(date -u +%F' '%T)";
       "$@" && echo "=== $NAME OK" || echo "=== $NAME FAILED"; }
run probe uv run python -m experiments.k3_longbench \
  --model-name Qwen/Qwen2.5-7B-Instruct --device cuda \
  --arms fp16 k4_b2.2 k4_b2.5 turboquant_mse_b3 turboquant_mse_k3v2 \
  --pack-path results/cache/k4_packs_qwen25_instruct.safetensors \
  --categories synthetic code --n-samples 100
run niah uv run python -m experiments.k3_niah \
  --model-name Qwen/Qwen2.5-7B-Instruct --device cuda \
  --arms fp16 k4_b2.5 turboquant_mse_b3 turboquant_mse_k3v2 \
  --pack-path results/cache/k4_packs_qwen25_instruct.safetensors \
  --lengths 4096 8192 16384 32768
SH
setsid bash /tmp/qwen_batch.sh > results/logs/qwen_batch.log 2>&1 &
sleep 75 && tail -5 results/logs/qwen_batch.log && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

Probe cell count: 5 arms × 4 datasets (synthetic → `passage_count`, `passage_retrieval_en`; code → `lcc`, `repobench-p`) × 100 samples = 2000 generations ≈ 3–5 h at the post-batched-flush rate. `chat_wrap` and `max_prompt_tokens` stay at defaults (False / 31500). Per-sample shards land under `partial/` → bootstrap CIs computed exactly as the code-CI cells were (10k resamples over per-sample shards; reuse that procedure/script from the `fdb5e6b` results commit). Crash → relaunch the same command with `--resume <run_dir>`.

### Task 12: [VM-RUN] NIAH bits-curve points (runs inside the Task-11 chain)

Already chained above. 4 arms × 4 lengths × 5 default depths = 80 cells ≈ 2–4 h. The bits criterion reads the parquet's measured `kv_size_bits` (skeptic at actual S) per (arm, length) — NOT a computed prediction. Note for the read: with C=512 the per-sequence pack charge is HALF Llama's (16·512/S vs 16·1024/S), so the crossover vs b3 should land at-or-earlier than Llama's ~16k; the pre-registered window (below) is deliberately the same 8k–32k. The 32k point is in-window for a 32768 max_position model (200-token buffer in `build_niah_prompt` covers template + 50 decode tokens); the harness prints prompt lengths — verify ≤ 32768 − 50 in the log. Recall read: single-needle NIAH was a NULL on Llama (everyone ≈ fp16) — the expectation here is no-regression, not separation.

### Task 13: [VM-RUN] Verdict doc + traceability

Write `docs/2026-07-2X-qwen25-replication-results.md` (kill-or-confirm style). Evaluate the pre-registered criteria VERBATIM (below), fill ONE of the two templates, include the Task 1 config record, Gate A/B/C table (Llama column beside Qwen for every gate), probe table with CIs, NIAH bits+recall table. Commit parquets (`results/k4_*/`, `results/k3_longbench/`, `results/k3_niah/`) + doc, bundle back. STOP for approval.

---

## Pre-registered success criteria (binding; written BEFORE any VM run)

**Gate layer:** Gate C passes (min `win_model` > 1 AND min `win_skeptic_deploy` > 1 at budget 2.5 across all 4 scored caches, `layer_win_fraction ≥ 0.9`). Gate C failure = does-not-replicate at the gate layer; no probe spend.

**Probe layer — the claim SHAPE must match Llama (all three):**
1. **Parity:** best-K4 (the better of k4_b2.2/k4_b2.5 by pooled synthetic+code avg) is ≥ turboquant_mse_b3 − noise: the bootstrap CI of (best-K4 − b3) pooled delta must NOT be entirely below 0.
2. **Retrieval edge:** (K4 − b3) synthetic-category delta point estimate > 0 with P(Δ>0) ≥ 0.75 at n=100 (Llama full-set: +3.25 [+1.02, +5.56], P=0.9975; n=100 CIs are wide — sign + P≥0.75 is the pre-committed shape test; a tighter full-set CI is the post-probe decision).
3. **Bits crossover:** measured skeptic `kv_size_bits` for k4_b2.5 > b3 at 4096 and < b3 at 32768, with the crossover therefore between 8k and 32k (Llama: between 8k and 16k).

**Secondary (report, not gate):** NIAH recall no-regression (k4_b2.5 mean within noise of fp16 — Llama measured a null); Gate A outcome vs the Llama band 0.56–0.69 (replicating the structural ceiling); Gate B transfer ratio vs Llama's 0.85–0.88.

### Verdict template 1 — REPLICATES

> On Qwen2.5-7B(-Instruct), Gate C passed (min G1 win {X}×/{Y}× model/deploy), and the n=100 probe reproduced the Llama claim shape: best-K4 pooled delta vs b3 = {Δ} [{CI}], synthetic delta = {Δs} (P(Δ>0)={p}), skeptic bits crossover vs b3 at ~{N}k (k4_b2.5 {a} vs b3 {b} at 32k). NIAH recall held fp16-level. The K4 result is a property of the method, not a Llama-3.1 artifact. Gate A measured {r_lo}–{r_hi} ({replicating/breaking} the structural transfer ceiling). Post-probe decision now open: full 6-category Qwen table (≈40–60 h) or ship the two-family evidence as-is.

### Verdict template 2 — DOES NOT REPLICATE

> On Qwen2.5-7B(-Instruct), the replication broke at {Gate C | criterion 1/2/3}: measured {numbers} vs pre-registered {rule}. The paper's claim is scoped to Llama-3.1 unless the mechanism difference is diagnosed; candidate suspects, in order: C=512 vs 1024 spectral-concentration difference (fewer directions to waterfill), Qwen's qkv-bias key statistics shifting the pre-RoPE spectrum, rope_theta=1e6 position mixing, and the smaller per-sequence pack charge masking/revealing the bits edge. No full-table spend. Honest negative; the gates did their job.

---

## Self-Review

**Design-decision coverage:** Model choice + concrete config verification (incl. static-rope and batched-flush pow2 licenses, sliding-window check) → Task 1 with the Mistral fallback branch spelled out. Local-first: tiny_qwen2 factory mirroring tiny_llama (offline, no downloads) → Task 2; attach/streaming smoke with k2b + spectral via `_fit_tiny_packs` → Task 3 (parametrizes the existing spectral suite rather than duplicating it); hf_compat helpers verified by test → Task 2; EOS-list generalization + Llama-ism sweep + chat_wrap discipline → Task 4 (+ Global Constraints). VM runbook mirrors the K4 plan's Tasks 8–12: corpus caches (base 8 offsets + Instruct 4, seq 2048, split documented) → Task 8; corpus-W fit_packs → Task 9/10; spectra+frontier × 4 scored × both w-sources → Task 9 with the three gate rules and Gate A's pre-registered expected-fail recorded as replication evidence; probe with the exact 5 arms/2 categories/n=100 → Task 11; NIAH 4k–32k with the exact 4 arms → Task 12; verdict + both templates → Task 13. Ops discipline is stated as constraints (one-8B rule, chained driver with OK/FAILED lines, 60–90 s launch verification, per-sample shards → free CIs). `model_short` derived from `collect_cache.py`'s actual code, not invented.

**Placeholder scan:** every command carries real paths/model ids; the only deliberate variables are `{...}` slots inside the two verdict templates (to be filled by measurement) and the Gate-B-conditional `--w-source none` refit. Task 3's code notes the two kwarg names (`spec_pair("k2b", ...)`, `recent_window`) that must be mirrored from the existing tests rather than trusted from this plan — flagged in-place.

**Type/name consistency:** pack files `k4_packs_qwen25{,_instruct}.safetensors` used identically in Tasks 9/10/11/12; cache filename pattern `qwen2.5-7b{,-instruct}_2048_off{N}.safetensors` matches `collect_cache.py`'s `model_short` rule and is used identically in Tasks 8/9/10; gate metric names (`retention_corpus`, `win_model`, `win_skeptic_deploy`, `layer_win_fraction_*`) match `k4_spectra.py:304-323` / `k4_frontier.py:592-623`; probe/NIAH arm names (`k4_b2.2`, `k4_b2.5`, `turboquant_mse_b3`, `turboquant_mse_k3v2`) match `recipes.py`'s parsers (`k4_b{budget}` at line 88, `_b{bits}` at 115, `_k{kb}v{vb}` at 119).

**Known open decisions deferred to run time:** (a) Gate B failure flips both pack fits to `--w-source none` — the probe arms are unchanged either way; (b) whether to add k4_b2.2 NIAH points at 32k (only if the chain finishes early — NOT pre-registered, reported separately if run); (c) the full 6-category Qwen table is explicitly post-probe, per the binding scope.

**Deliberate deviations from the format model:** no new `src/bmx` code tasks (the replication's null hypothesis is "no code needed"), and Task 12 rides Task 11's chained driver instead of a separate launch — both are consequences of the one-8B-process rule and the replication-not-development scope.
