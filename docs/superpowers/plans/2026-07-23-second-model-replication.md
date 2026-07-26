# Second-Model Replication — Qwen3-8B K4 Gates + Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks 1–6 are local code/verification (TDD, CPU); Tasks 7–13 are the VM batch (runbook kind, exact commands).

**Goal:** Replicate the K4 gates (A/B/C) + the n=100 synthetic+code probe + the bits-vs-context NIAH points (4k–32k) on a second model family — **Qwen/Qwen3-8B** (post-trained) — so the paper's claim (`docs/2026-07-15-k4-duel-results.md`) is a property of the method, not a Llama-3.1 artifact. Scope is EXACTLY gates + probe + bits curve; a full 6-category Qwen table is a **post-probe decision**, not this plan.

**Architecture:** ONE pre-scoped `src/bmx` change, then the stack is expected model-agnostic. Qwen3 attention computes `k_proj → view(b,S,h_kv,d) → per-head k_norm (RMSNorm) → RoPE` (verified in the installed transformers 5.11.0 `modeling_qwen3.py:248-249,263-268`), so "pre-RoPE keys" MUST be captured at the **k_norm OUTPUT** — hooking `k_proj` (today's capture point in `collect.py:106`, `streaming.py:628`, `packed_streaming.py:993`) silently captures un-normed keys and breaks the `k == RoPE(k_pre)` identity the whole rope-at-read path relies on. **Today's code does NOT raise on Qwen3** (it has a `k_proj`, so the `streaming.py:613-621` guard passes) — the failure would be silent wrongness, which is why Task 3 adds a structural capture-module dispatch (`hf_compat.resolve_qk_capture_modules`: prefer `{q,k}_norm` when present, else `{q,k}_proj`) wired into all three hook sites and pinned by tests. Everything else (write-once streaming, spectral packs, `k3_longbench`/`k3_niah` harnesses) is expected unchanged: Qwen3-8B is otherwise a Llama-style decoder (`model.model.layers[i].self_attn.{q,k,v}_proj`, GQA h_kv=8, d=128 → C=1024 pow2 so every batched-flush license in `streaming._flush_batchable` holds; static RoPE theta=1e6; no sliding window — `layer_types` all `full_attention`; `attention_bias=False`, one less delta from Llama than Qwen2's qkv bias). Any further Llama-ism found becomes a minimal fix + regression test, reported loudly.

**Tech Stack:** as the K4 campaign (PyTorch 2.12, transformers 5.11, tyro, safetensors, parquet). VM: rented NVIDIA GH200 per `vm-interaction-guide` memory (git transport, detached setsid launches, no sudo pip). Qwen3 checkpoints are public/ungated (no HF token dance, unlike Llama).

## Global Constraints

- **NEVER `git commit` without the user's explicit approval** — stage, propose the task's exact message, STOP (the user may pre-authorize per run, as in prior campaigns).
- Before any commit: `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` all clean. Baseline at plan time: **489 passed, 17 skipped, 1 xfailed** (branch `feat/triton-decode-kernel` @ `602936e`).
- **Model: `Qwen/Qwen3-8B` (post-trained). ALL generation evals run in NON-THINKING mode (`enable_thinking=False` in the chat template) — pre-registered: short-answer eval protocol, parity with the Llama duel (no CoT); record this beside every generation task.** Concretely for this harness: `chat_wrap=False` (below) means `apply_chat_template` is never called (`longbench.py:162-163` returns raw template-fill ids), so no thinking scaffold is ever injected; `enable_thinking=False` binds any future template-applying parity run, and think-tag emission on raw completions is scanned and recorded (Task 10 smoke, echoed in Task 13).
- **Pre-RoPE capture point (binding):** on Qwen3, `k_pre` = the `k_norm` OUTPUT (`k_proj → reshape heads → per-head k_norm → RoPE`), so the invariant `k = RoPE(k_pre)` holds — the streaming path's rope-at-read identity depends on it. collect + attach gain a structural dispatch (hook `self_attn.k_norm` forward output instead of `self_attn.k_proj`; Task 3). **Task 1 VERIFIES this from the installed transformers 5.11 Qwen3 modeling source (read the attention forward, confirm norm-then-rope order and module names) before any code — if the order differs, STOP and report.**
- **C: verify from config (expect `num_key_value_heads=8 × head_dim=128 → C=1024` — the SAME C as Llama-3.1-8B).** The old plan's "C=512 halves the pack charge" note is WRONG for Qwen3-8B — removed. The charge story is now skeptic-v2 (c_used-based, landed 2026-07-23) and applies automatically via `bits_per_entry` (spectral branch adds `skeptic_charge(C, S, tiers, c_used=mean_c_used, ...)` — `streaming.py:653-714`).
- **CPU pre-flight model: `Qwen/Qwen3-0.6B`** (replaces Qwen2.5-0.5B); the `setup_rope` rel_fro < 2e-2 gate (`experiments/_k4_common.py:110-116`) and the hook-capture invariant test (`k == RoPE(k_pre)` on a real small-model forward) stay. Note: Qwen3-0.6B has explicit `head_dim=128` decoupled from `hidden//heads` (1024/16=64), so its C is ALSO 1024 — a stronger pre-flight than the old 0.5B (C=128).
- **Fallback (recorded): if Task 1's architecture verification or the 0.6B pre-flight fails structurally, fall back to `Qwen/Qwen2.5-7B-Instruct` (the original plan's model, zero-adaptation) — one line, not a parallel track.**
- **`tiny_qwen3` factory in `tests/factories.py`** (`Qwen3Config`, tiny dims, qk-norm ON — the factory must exercise the k_norm hook path; never download in tests). `head_dim=8` must be explicit: Qwen3Config's default is 128, decoupled from `hidden_size // num_attention_heads`.
- **Landed since the original plan (all binding here):** skeptic-v2 accounting (`bits_per_entry` charges `16·c_used/S` automatically — no manual charge math anywhere in this plan); the corpus-transfer harness + synthesis arms (`experiments/k4_corpus_transfer.py` — OPTIONAL one-line VM rider in Task 10, not gated); the **module-form launch convention (`uv run python -m experiments....` everywhere)**; the loader's prefix-row materialization (`load_eval_tokens` materializes only the needed row prefix, commit `1c6a068` — already safe for big corpora).
- Tiny offline test models from `tests/factories.py`; **never download in tests** (Task 5's 0.6B pre-flight is an experiment run, not a test).
- Pack files are BINARY ARTIFACTS: never committed; they regenerate from committed config; the JSON sidecar records git SHA + corpus provenance.
- Honest bits: `kv_size_bits` in every probe/NIAH parquet is SKEPTIC-v2 accounting at actual S (per-sequence pack charge `16·c_used/S` + tier map, `c_used` read from the pack). Model-level numbers only if Gate A passes (it is EXPECTED to fail — see Task 9).
- **Delta-parity discipline:** `chat_wrap` stays **False** everywhere (the 43.7-pt chat-wrap sensitivity killed absolute parity; all banked numbers are chat_wrap=False). `max_prompt_tokens` stays at the default 31500.
- **Ops discipline (July campaign, binding):** one 8B process at a time at 31.5k prompts (two → mutual OOM at 96 GB); VM-side chained drivers with per-cell `OK`/`FAILED` lines, never local watchers; verify every detached launch 60–90 s in (log advancing + GPU busy); never kill long runs prematurely; never sudo pip into the venv; per-cell `partial/` shards mean crashes resume via `--resume`.
- Per-sample shards (`write_samples_shard`, already on the harness) mean the probe gets bootstrap CIs for free — no extra instrumentation. (Shards store per-sample SCORES, not prediction text — the think-tag scan surface is Task 10's smoke, where the generated strings are in hand.)

## File structure

- Modify: `src/bmx/cache/hf_compat.py` — `resolve_qk_capture_modules(self_attn)` (layer-0, no bmx.cache imports — keeps the file's contract).
- Modify: `src/bmx/cache/collect.py` — `_register_qkproj_hooks` registers on the resolved capture modules; `reshape_heads` docstring covers the 4-D norm-output shape.
- Modify: `src/bmx/cache/streaming.py:623-628` — `attach()` registers on the resolved k module.
- Modify: `src/bmx/cache/packed_streaming.py:988-995` — same one-line substitution (packed path is out of replication scope, but a silently-wrong hook point must not ship).
- Modify: `tests/factories.py` — `tiny_qwen3()` mirroring `tiny_llama()`.
- Modify: `tests/test_streaming_spectral.py` — parametrize the streaming-spectral tests over `{tiny_llama, tiny_qwen3}`.
- Create: `tests/test_qwen3_compat.py` — hf_compat resolution, capture-point invariants (collect + streaming + packed), k2b streaming smoke, k4 end-to-end generate, EOS-list stop test.
- No `experiments/` changes expected (harness `model_name`/`pack_path` are already parameters). Any further discovered Llama-ism: minimal fix + regression test in the task that found it.
- Create (VM, Task 13): `docs/2026-07-2X-qwen3-replication-results.md`.

## Naming (derived, not invented)

`experiments/collect_cache.py` computes `model_short = model_name.split("/")[-1].lower()`:

| model | model_short | cache file pattern |
|---|---|---|
| `Qwen/Qwen3-8B` | `qwen3-8b` | `results/cache/qwen3-8b_2048_off{N}.safetensors` |
| `Qwen/Qwen3-0.6B` (pre-flight) | `qwen3-0.6b` | `results/cache/qwen3-0.6b_2048*.safetensors` |

Pack files: `results/cache/k4_packs_qwen3.safetensors` (the 8B campaign pack) and `results/cache/k4_packs_qwen3_06b.safetensors` (pre-flight). ONE pack file, not two: the binding model choice runs gates AND probe on the same post-trained `Qwen/Qwen3-8B` checkpoint, so the Llama campaign's base-vs-instruct pack split does not arise (`Qwen/Qwen3-8B-Base` exists but is out of scope).

---

### Task 1: Architecture + model-choice verification (source + config, local, minutes)

Two verifications, in order, BEFORE any code. Part A reads the installed transformers 5.11 Qwen3 attention source; Part B downloads two ~1 KB config.json files, no weights.

**Files:** none committed — scratchpad scripts; findings recorded in the run log and echoed into Task 13's verdict doc.

- [ ] **Step 1: Part A — verify norm-then-rope order and module names from the INSTALLED source** (binding; not from memory, not from this plan):

```bash
uv run python - <<'PY'
import inspect
from transformers.models.qwen3 import modeling_qwen3 as m
src = inspect.getsource(m.Qwen3Attention)
print(src)
PY
```

Confirm all four, quoting the lines in the task report:
1. `self.k_norm = Qwen3RMSNorm(self.head_dim, eps=...)` — a PER-HEAD norm module named exactly `k_norm` (and `q_norm`) on the attention module.
2. Forward order: `key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))` **then** `apply_rotary_pos_emb(query_states, key_states, cos, sin)` — norm BEFORE RoPE. (At plan-writing time this is `modeling_qwen3.py:263-268`.)
3. `k_norm`'s forward OUTPUT shape is `(b, S, h_kv, d)` — the `.transpose(1, 2)` happens on the norm's output, outside the module, so a forward hook on `k_norm` sees the already-headed 4-D tensor. `bmx.cache.collect.reshape_heads` covers this via its numel-equal reshape (Task 3 updates its docstring to say so).
4. Queries mirror keys: `q_norm(q_proj(...).view(...))` — so the query the attention ACTUALLY uses is the q_norm output (this is why Task 3 moves the q hook too: the K4 W-moment statistics must see real queries).

**If the order differs (RoPE before norm, fused norm, different module names): STOP and report — do not adapt on the fly.** Fallback per Global Constraints: `Qwen/Qwen2.5-7B-Instruct`.

- [ ] **Step 2: Part B — config checks** (scratchpad):

```python
# scratchpad/check_qwen3_config.py
from transformers import AutoConfig, GenerationConfig

for name in ("Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B"):
    c = AutoConfig.from_pretrained(name)
    assert not hasattr(c, "text_config") or not hasattr(c.text_config, "num_attention_heads"), (
        "unexpected multimodal wrapper"  # resolve_text_config must return c itself
    )
    d = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    C = c.num_key_value_heads * d
    g = GenerationConfig.from_pretrained(name)
    lt = getattr(c, "layer_types", None)
    print(f"{name}: arch={c.architectures} layers={c.num_hidden_layers} "
          f"h_kv={c.num_key_value_heads} d={d} C={C} theta={c.rope_theta} "
          f"rope_scaling={getattr(c, 'rope_scaling', None)} "
          f"max_pos={c.max_position_embeddings} "
          f"layer_types={set(lt) if lt else None} "
          f"sliding={getattr(c, 'sliding_window', None)} "
          f"vocab={c.vocab_size} eos={g.eos_token_id}")
    # --- the binding checks ---
    assert c.architectures == ["Qwen3ForCausalLM"], "qk-norm dispatch targets Qwen3Attention"
    assert d & (d - 1) == 0 and C & (C - 1) == 0, "batched-flush fwht licenses need pow2 d and C"
    rs = getattr(c, "rope_scaling", None)
    assert rs is None or (rs.get("rope_type", rs.get("type")) in ("default", "llama3")), (
        f"dynamic/NTK rope_scaling {rs} breaks the batched RoPE-table extension "
        "(streaming's one growing cos/sin table) — FALLBACK to Qwen2.5-7B-Instruct"
    )
    assert c.max_position_embeddings >= 32768, "32k NIAH point + 31500 LongBench truncation"
    assert lt is None or set(lt) == {"full_attention"}, (
        "sliding-window layers change cache semantics"
    )

c8 = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
d8 = getattr(c8, "head_dim", None) or c8.hidden_size // c8.num_attention_heads
assert c8.num_key_value_heads * d8 == 1024, "expected C=1024 (8 x 128) on Qwen3-8B"
print("ALL CHECKS PASS")
```

Expected (from model cards, to be CONFIRMED by the run): 8B — 36 layers, h_kv=8, explicit head_dim=128 → C=1024 (same as Llama-3.1-8B), rope_theta=1e6, rope_scaling=None (static), max_position_embeddings=40960 (32k NIAH point fits with ~8k margin), vocab 151936, eos a **list** `[151645, 151643]` (motivates Task 4's EOS test); 0.6B — 28 layers, h_kv=8, explicit head_dim=128 → C=1024 (SAME C as the 8B). Record the actual values.

- [ ] **Step 3: Gate.** Part A order confirmed AND all Part B asserts pass → Qwen3-8B confirmed, proceed. ANY failure → STOP, report, and fall back to `Qwen/Qwen2.5-7B-Instruct` per Global Constraints (that model needs zero adaptation — the pre-amendment Qwen2.5 version of this plan file is in git history, `git log --follow docs/superpowers/plans/2026-07-23-second-model-replication.md`, and applies verbatim). Record which branch was taken.

---

### Task 2: `tiny_qwen3` factory + hf_compat resolution tests

**Files:**
- Modify: `tests/factories.py`
- Create: `tests/test_qwen3_compat.py`

**Interfaces:**
- Produces: `tiny_qwen3()` — offline `Qwen3Config` model in eval mode, geometry mirroring `tiny_llama` (2 layers, 4 heads, 2 kv heads, hidden 32, **explicit head_dim=8** → C=16 pow2, `max_position_embeddings=512` to exceed the 128-token PAGE flush threshold, `vocab_size=97` so `ids()` works unchanged). Qwen3's qk-norm is unconditional — the factory exercises the k_norm capture path by construction.
- Consumes: `bmx.cache.hf_compat.{model_config_n_layers, resolve_decoder_layers, resolve_text_config, resolve_vocab_size}` — verified against Qwen3 config shape **by test, not assumption**.

- [ ] **Step 1: Write the failing tests** (`tests/test_qwen3_compat.py`):

```python
"""Second-model-family (Qwen3) compatibility: hf_compat resolution, the qk-norm
pre-RoPE capture point, streaming attach, spectral packs, EOS-list stop.
Everything offline via tiny_qwen3."""

import torch

from bmx.cache.hf_compat import (
    model_config_n_layers,
    resolve_decoder_layers,
    resolve_text_config,
    resolve_vocab_size,
)
from tests.factories import ids, tiny_llama, tiny_qwen3


def test_hf_compat_resolves_qwen3():
    m = tiny_qwen3()
    assert model_config_n_layers(m) == 2
    layers = resolve_decoder_layers(m)
    sa = layers[0].self_attn
    assert hasattr(sa, "q_proj") and hasattr(sa, "k_proj")
    assert hasattr(sa, "q_norm") and hasattr(sa, "k_norm")  # the qk-norm family marker
    tc = resolve_text_config(m.config)
    assert tc is m.config  # no multimodal wrapper on Qwen3ForCausalLM
    assert tc.num_key_value_heads == 2
    d = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
    assert d == 8  # explicit head_dim; Qwen3Config's default (128) must not leak in
    assert resolve_vocab_size(m.config) == 97
```

- [ ] **Step 2: Verify failure** — `uv run pytest tests/test_qwen3_compat.py -v` → ImportError (`tiny_qwen3` missing).
- [ ] **Step 3: Implement** in `tests/factories.py` (extend the existing transformers import line with `Qwen3Config, Qwen3ForCausalLM`):

```python
def tiny_qwen3():
    """Tiny Qwen3 mirroring tiny_llama's geometry — the second-model-family
    fixture. Llama-style layer tree (model.model.layers[i].self_attn) PLUS
    Qwen3's per-head q_norm/k_norm between projection and RoPE (qk-norm is
    unconditional in Qwen3, so this factory exercises the k_norm capture path
    by construction). head_dim=8 must be explicit: Qwen3Config's default is
    128, decoupled from hidden_size // num_attention_heads."""
    cfg = Qwen3Config(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=32,
        intermediate_size=64,
        vocab_size=97,
        head_dim=8,
        max_position_embeddings=512,  # exceed the 128-token PAGE flush threshold
    )
    torch.manual_seed(3)
    return Qwen3ForCausalLM(cfg).eval()
```

- [ ] **Step 4: Verify pass.** **Step 5: full battery + stage + propose** `test(qwen3): tiny_qwen3 factory + hf_compat resolution on the second family`. STOP.

---

### Task 3: qk-norm pre-RoPE capture dispatch (the one expected src change)

Today all three hook sites register on `k_proj`; on Qwen3 that captures UN-NORMED keys — no error is raised (`Qwen3` has a `k_proj`, so `streaming.py:613-621`'s guard passes) and every downstream rope-at-read consumer is silently wrong. This task makes the capture point structural.

**Files:**
- Modify: `src/bmx/cache/hf_compat.py` (append helper)
- Modify: `src/bmx/cache/collect.py:61-66` (reshape_heads docstring), `:87-107` (`_register_qkproj_hooks`)
- Modify: `src/bmx/cache/streaming.py:234-239` (stash_pre_rope docstring), `:623-628` (attach hook registration)
- Modify: `src/bmx/cache/packed_streaming.py:988-995` (attach hook registration)
- Test: `tests/test_qwen3_compat.py` (append)

**Interfaces:**
- Produces: `resolve_qk_capture_modules(self_attn) -> tuple[nn.Module, nn.Module]` — `(q_module, k_module)` whose forward OUTPUT is the pre-RoPE q/k. Later tasks (and the whole VM batch) rely on `collect_cache` and `attach()` capturing post-k_norm tensors on Qwen3 with NO call-site changes.
- Consumes: `tiny_qwen3` (Task 2), `reshape_heads` (already shape-agnostic: `(1,S,h*d)` and `(1,S,h,d)` are numel-equal, so its one reshape covers both).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_qwen3_compat.py`):

```python
def test_collect_k_pre_is_post_knorm_qwen3():
    """THE capture-point invariant: k == RoPE(k_pre) on a real Qwen3 module
    tree. Qwen3 applies per-head RMSNorm (k_norm) between k_proj and RoPE, so
    k_pre MUST be captured at the k_norm OUTPUT — hooking k_proj (the Llama
    point) breaks this identity and every rope-at-read consumer downstream.
    Mirrors tests/test_cache_rope.py::test_apply_rope_matches_collect_cache."""
    from bmx.cache.collect import collect_cache
    from bmx.cache.rope import apply_rope, rope_cos_sin

    model = tiny_qwen3()
    input_ids = ids(seq=16)
    cache = collect_cache(model, input_ids, n_q_keep=256)
    S = input_ids.shape[1]
    cos, sin = rope_cos_sin(model.config, S)
    for i in range(model.config.num_hidden_layers):
        k_pre = cache[f"layer{i}.k_pre"].float()
        k = cache[f"layer{i}.k"].float()
        rel = (apply_rope(k_pre, cos.float(), sin.float()) - k).norm() / k.norm()
        assert rel < 1e-2, f"layer{i}: rel_fro {rel:.4e} — k_pre not post-k_norm?"


def test_collect_hooks_land_on_qk_norm_qwen3():
    """Structural pin for BOTH capture hooks (the k identity above cannot see
    q): on a qk-norm family the q/k hooks must hang on q_norm/k_norm, not the
    projections — the K4 W-moment statistics must see the query attention
    actually uses (q_norm output), not the raw q_proj output."""
    from bmx.cache.collect import register_hooks

    model = tiny_qwen3()
    store: dict = {}
    handles, n_layer = register_hooks(model, store, 8)
    try:
        sa = model.model.layers[0].self_attn
        assert len(sa.q_norm._forward_hooks) == 1
        assert len(sa.k_norm._forward_hooks) == 1
        assert len(sa.q_proj._forward_hooks) == 0
        assert len(sa.k_proj._forward_hooks) == 0
    finally:
        for h in handles:
            h.remove()
    assert n_layer == 2


def test_streaming_prerope_roundtrip_qwen3():
    """attach() + rope-at-read reproduce the true post-RoPE keys through a
    Qwen3 tree (fp16 K isolates capture+RoPE plumbing from quant error —
    the tests/test_streaming_cache.py::test_prerope_key_capture_and_rope_at_read
    pattern). seq=150 with recent_window=8 crosses the PAGE(128) flush
    threshold, so the committed region definitely derives from the
    hook-captured k_pre, not the pristine fp16 tail."""
    from bmx.cache.specs import CacheCodecSpec
    from bmx.cache.streaming import StreamingQuantizedCache

    model = tiny_qwen3()
    input_ids = ids(seq=150, seed=7)
    cache = StreamingQuantizedCache(
        model.config,
        k_spec=CacheCodecSpec(arm="fp16", pre_rope=True),
        v_spec=CacheCodecSpec(arm="fp16"),
        recent_window=8,
    )
    cache.attach(model)
    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True)
    cache.detach()
    k_post, _ = cache.reconstruct_layer(0)

    ref = StreamingQuantizedCache(
        model.config,
        k_spec=CacheCodecSpec(arm="fp16"),
        v_spec=CacheCodecSpec(arm="fp16"),
    )
    with torch.no_grad():
        model(input_ids, past_key_values=ref, use_cache=True)
    k_true = ref.layers[0].keys
    rel = (k_post.float() - k_true.float()).norm() / k_true.float().norm().clamp_min(
        1e-6
    )
    assert rel < 1e-2


def test_packed_attach_hooks_k_norm_qwen3():
    """PackedStreamingCache.attach shares the capture dispatch (structural pin
    only — the packed path is out of replication scope, but a silently-wrong
    hook point must not ship). Mirror the smallest attach fixture in
    tests/test_packed_streaming.py if the constructor kwargs differ."""
    from bmx.cache.packed_streaming import PackedStreamingCache
    from bmx.cache.specs import CacheCodecSpec

    model = tiny_qwen3()
    cache = PackedStreamingCache(
        model.config,
        k_spec=CacheCodecSpec(arm="rtn_token", bits=4, group=8, pre_rope=True),
        v_spec=CacheCodecSpec(arm="rtn_token", bits=4, group=8),
    )
    cache.attach(model)
    try:
        sa = model.model.layers[0].self_attn
        assert len(sa.k_norm._forward_hooks) == 1
        assert len(sa.k_proj._forward_hooks) == 0
    finally:
        cache.detach()
```

- [ ] **Step 2: Verify failure** — `uv run pytest tests/test_qwen3_compat.py -v`: the two collect tests and the streaming roundtrip FAIL (k_pre is un-normed today; hooks land on the projections); the packed pin FAILS. `test_hf_compat_resolves_qwen3` stays green.
- [ ] **Step 3: Implement.** (a) Append to `src/bmx/cache/hf_compat.py`:

```python
def resolve_qk_capture_modules(self_attn):
    """(q_module, k_module) whose forward OUTPUT is the pre-RoPE query/key.

    Llama-family attention is {q,k}_proj -> reshape heads -> RoPE, so the
    projection output IS the pre-RoPE tensor. Qwen3/Gemma3-style attention
    inserts a per-head RMSNorm between projection and RoPE
    (k_proj -> view(b, S, h, d) -> k_norm -> RoPE): capturing at k_proj there
    breaks the k == RoPE(k_pre) identity every rope-at-read consumer relies
    on, and q_proj output is not the query attention actually uses (W-moment
    statistics must see the q_norm output). Probe structurally: prefer
    {q,k}_norm when both are present, else the plain projections. The norm
    modules emit the already-headed (b, S, h, d) shape; collect.reshape_heads
    covers both layouts with one numel-equal reshape.
    """
    if hasattr(self_attn, "q_norm") and hasattr(self_attn, "k_norm"):
        return self_attn.q_norm, self_attn.k_norm
    return self_attn.q_proj, self_attn.k_proj
```

(b) In `src/bmx/cache/collect.py`: extend the hf_compat import to `from bmx.cache.hf_compat import (resolve_decoder_layers, resolve_qk_capture_modules, resolve_text_config)`; update `reshape_heads`'s docstring to:

```python
def reshape_heads(out: torch.Tensor, n_head: int, d: int) -> torch.Tensor:
    """(1, S, n_head*d) or (1, S, n_head, d) projection/norm output ->
    (n_head, S, d), fp16 contiguous. The two input shapes are numel-equal, so
    one reshape covers both (Qwen3's q_norm/k_norm emit the 4-D headed form)."""
```

and in `_register_qkproj_hooks`, replace the two registration lines at the bottom of the loop with the dispatch (hook bodies unchanged):

```python
        q_mod, k_mod = resolve_qk_capture_modules(layer.self_attn)
        handles.append(q_mod.register_forward_hook(q_hook))
        handles.append(k_mod.register_forward_hook(k_hook))
```

Also update `_register_qkproj_hooks`'s docstring first line to: `"""Hooks on the pre-RoPE q/k capture modules ({q,k}_proj on Llama-family, {q,k}_norm on Qwen3-style qk-norm attention); returns (handles, n_layer)."""`

(c) In `src/bmx/cache/streaming.py`: add `resolve_qk_capture_modules` to the existing hf_compat import; in `attach()` replace the registration loop's last line so it reads:

```python
        for i, mlayer in enumerate(decoder_layers):

            def k_hook(module, inp, out, i=i):
                self.layers[i].stash_pre_rope(out)

            _, k_mod = resolve_qk_capture_modules(mlayer.self_attn)
            self._handles.append(k_mod.register_forward_hook(k_hook))
        return self
```

Keep the existing `self_attn.k_proj` presence guard above it unchanged (it excludes GPT-2-style fused attention; Qwen3 passes it and then dispatches to `k_norm`). Update `stash_pre_rope`'s docstring shape line to `out: (1, T, h_kv*d) (k_proj) or (1, T, h_kv, d) (k_norm) -> reshaped to (h_kv, T, d) fp16, ...`.

(d) In `src/bmx/cache/packed_streaming.py`: add `resolve_qk_capture_modules` to the hf_compat import; in `attach()`'s pre_rope branch replace `mlayer.self_attn.k_proj.register_forward_hook(k_hook)` with:

```python
                _, k_mod = resolve_qk_capture_modules(mlayer.self_attn)
                self._handles.append(k_mod.register_forward_hook(k_hook))
```

- [ ] **Step 4: Verify pass** — `uv run pytest tests/test_qwen3_compat.py -v` all green; then the FULL battery (`uv run pytest -q`) to prove the Llama fallback branch changed nothing (the existing prerope/rope-identity tests on `tiny_llama` are the regression pins).
- [ ] **Step 5: Stage + propose** `feat(cache): qk-norm capture dispatch — pre-RoPE q/k hook {q,k}_norm output on Qwen3-family (collect + streaming + packed attach)`. STOP.

---

### Task 4: Streaming/spectral parametrization + k4 end-to-end + EOS-list + Llama-ism sweep

**Files:**
- Modify: `tests/test_streaming_spectral.py` (parametrize over factories)
- Modify: `tests/test_qwen3_compat.py` (append k2b, k4, EOS tests)
- Possibly modify: whatever the sweep finds (expected: nothing).

**Interfaces:**
- Consumes: `StreamingQuantizedCache.attach()` (now qk-norm-aware), `spec_pair("k2b", ...)`, `spec_pair("k4_b2.5", ..., pack_path=...)` (`recipes.py:13`; the `k4_b{budget}` parser at `recipes.py:88-114`), `_fit_tiny_packs` (`tests/test_streaming_spectral.py:29-42` — already model-parametric, takes `model`; internally calls `collect_cache`, so it exercises the Task-3 dispatch), `generate_through_cache` (`generate.py`; its EOS set is `generation_config.eos_token_id → model.config.eos_token_id → tokenizer.eos_token_id`, normalized list-or-int at `generate.py:66-71`).
- Produces: proof that the write-once streaming path, the spectral K-branch, the k4 recipe, and list-EOS stopping all run on a Qwen3 module tree.

- [ ] **Step 1: Parametrize the spectral streaming tests.** In `tests/test_streaming_spectral.py`, convert `test_streaming_spectral_matches_reference`, `test_streaming_spectral_requires_pre_rope`, and `test_streaming_spectral_committed_block_matches_offline_and_frozen` to `@pytest.mark.parametrize("factory", [tiny_llama, tiny_qwen3], ids=["llama", "qwen3"])`, replacing the direct `tiny_llama()` calls with `factory()` (import `tiny_qwen3` alongside `tiny_llama`). `_fit_tiny_packs` needs no change. **Read the file's header comment first** — the seq=150/recent_window=8 pattern (which actually crosses the PAGE flush threshold) must be preserved verbatim for the new parametrization. Note the committed-parity test compares streamed capture vs collect capture — with Task 3 landed both are post-k_norm, so it doubles as an attach↔collect agreement pin on Qwen3.
- [ ] **Step 2: Append the k2b + k4 tests** (`tests/test_qwen3_compat.py`):

```python
def test_streaming_k2b_qwen3():
    """The proven k2b arm streams through a Qwen3 module tree: attach() hooks
    fire, pages flush, bpe accounting is real (<16). Mirror the fixture pattern
    of tests/test_streaming_cache.py::test_k2b_pre_rope_streams_token_by_token
    (seq=150, recent_window=8 — read it first, copy its invariant exactly)."""
    from bmx.cache.recipes import spec_pair
    from bmx.cache.streaming import StreamingQuantizedCache

    model = tiny_qwen3()
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


def test_generate_k4_qwen3(tmp_path):
    """k4_b2.5 end-to-end (attach + hooks + spectral flush + greedy decode) on
    Qwen3 — the exact recipe the VM probe runs."""
    from bmx.cache.generate import generate_through_cache
    from bmx.cache.recipes import spec_pair
    from tests.test_streaming_spectral import _fit_tiny_packs

    model = tiny_qwen3()
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


def test_generate_stops_on_any_eos_in_list():
    """Qwen3's generation_config.eos_token_id is a LIST ([151645, 151643] on
    the real model) — the decode loop must stop on ANY member. Pinned here on
    tiny_qwen3 because Llama-3.1's list was the only case ever exercised.
    Verified stop semantics (generate.py:105-113): the EOS token IS appended
    to new_ids before the break, so the stub-decoded output has length 1 when
    the first decode token is an eos member."""
    from bmx.cache.generate import generate_through_cache
    from bmx.cache.specs import CacheCodecSpec

    class _StubTok:
        eos_token_id = None

        def decode(self, t, skip_special_tokens=True):
            return " ".join(map(str, t.tolist() if hasattr(t, "tolist") else t))

    model = tiny_qwen3()
    fp16 = CacheCodecSpec(arm="fp16")
    prompt = ids(seq=24)
    # Probe: which token does greedy decode emit SECOND? (the first decode
    # token is emitted before the loop's eos check ever runs — generate.py:103)
    out = generate_through_cache(
        model, _StubTok(), prompt, n_prefill=12,
        k_spec=fp16, v_spec=fp16, max_new_tokens=4, strip=False,
    )
    toks = out.split()
    assert len(toks) == 4  # no eos configured => full budget
    second = int(toks[1])
    # Re-run with an eos LIST containing that token (plus a never-emitted one):
    model.generation_config.eos_token_id = [96, second]
    out2 = generate_through_cache(
        model, _StubTok(), prompt, n_prefill=12,
        k_spec=fp16, v_spec=fp16, max_new_tokens=4, strip=False,
    )
    assert len(out2.split()) == 2  # stopped ON the second token, immediately
```

(If `spec_pair("k2b", ...)`'s kwargs or `recent_window` differ from the assumed names, mirror the actual call sites in `tests/test_streaming_cache.py` — the invariant, not the literals, is binding. If greedy decode happens to emit token 96 or repeat, adjust the seed in `ids(...)`, not the invariant.)
- [ ] **Step 3: The Llama-ism sweep.** `grep -rn "Llama\|llama\|128009\|128001\|128008\|<|eot_id|>" src/bmx/cache/ experiments/k3_*.py experiments/_common.py` and audit each hit: default `model_name` strings (fine — overridden by `--model-name`), docstring references (fine), `build_chat`/chat-wrap machinery (dormant — `chat_wrap=False` is a Global Constraint), hard-coded token ids (NOT fine — none expected; any found is a bug to fix). Record the audit result in the task report.
- [ ] **Step 4: fail → implement (no further src change expected) → pass.** If any test fails for a REAL architecture reason (not a fixture literal), that is a finding: fix minimally in `src/bmx`, add the regression test, and flag it in the task report.
- [ ] **Step 5: full battery + stage + propose** `test(qwen3): streaming + spectral + k4 recipe + EOS-list on the second family (parametrized spectral suite)`. STOP.

---

### Task 5: Local real-checkpoint pre-flight — Qwen3-0.6B on CPU

The tiny factory proves the module tree; it cannot prove tokenizer/wikitext plumbing, real RoPE (theta=1e6), or the hooked-capture ↔ stored-K consistency on real weights — in particular that the k_norm capture point holds on the REAL Qwen3 attention, not just the tiny replica. Qwen3-0.6B (h_kv=8, explicit d=128 → C=1024, pow2, the SAME C as the 8B; ~1.5 GB download) proves all of it locally in minutes.

- [ ] **Step 1: Collect three caches** (real Qwen3 checkpoint, CPU, bf16; module-form launches):

```bash
uv run python -m experiments.collect_cache --model-name Qwen/Qwen3-0.6B --seq-len 2048
uv run python -m experiments.collect_cache --model-name Qwen/Qwen3-0.6B --seq-len 2048 --token-offset 2048
uv run python -m experiments.collect_cache --model-name Qwen/Qwen3-0.6B --seq-len 2048 --token-offset 4096
```

Split: fit = offsets {2048, 4096} (2 × 2048 = 4096 rows = 4×C — rank-sufficient for a mechanism read), scored = the offset-0 cache.

- [ ] **Step 2: Fit + spectra, corpus-W:**

```bash
uv run python -m experiments.k4_fit_packs --corpus-cache-paths results/cache/qwen3-0.6b_2048_off2048.safetensors results/cache/qwen3-0.6b_2048_off4096.safetensors --out-path results/cache/k4_packs_qwen3_06b.safetensors --model-label qwen3-0.6b --model-name Qwen/Qwen3-0.6B --w-source corpus
uv run python -m experiments.k4_spectra --cache-path results/cache/qwen3-0.6b_2048.safetensors --corpus-cache-paths results/cache/qwen3-0.6b_2048_off2048.safetensors results/cache/qwen3-0.6b_2048_off4096.safetensors --model-label qwen3-0.6b --model-name Qwen/Qwen3-0.6B --w-source corpus
```

**The binding check is `setup_rope`'s self-validation line** (`experiments/_k4_common.py:110-116`): `[rope_validation] rel_fro(apply_rope(k_pre), k) < 2e-2` on a real Qwen3 checkpoint — it simultaneously validates `rope_cos_sin` against Qwen3's rope config, the post-k_norm hook capture (exactly what apply_rope must reproduce), and the (h,S,d) layout. A failure here is a hard STOP (architecture finding, diagnose before any VM spend; structural failure → the Qwen2.5-7B-Instruct fallback per Global Constraints). Mechanism read only otherwise: finite headline numbers; 0.6B retention lands where it lands — record it, don't gate on it.

- [ ] **Step 3: Streaming smoke on the real checkpoint** (scratchpad):

```python
# scratchpad/qwen3_06b_stream_smoke.py
import torch
from transformers import AutoModelForCausalLM

from bmx.cache.recipes import spec_pair
from bmx.cache.streaming import StreamingQuantizedCache
from bmx.eval.layer_swap import load_eval_tokens

MODEL = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).eval()
k_spec, v_spec = spec_pair(
    "k4_b2.5", rank=16, group=64, seed=0,
    pack_path="results/cache/k4_packs_qwen3_06b.safetensors",
)
cache = StreamingQuantizedCache(model.config, k_spec=k_spec, v_spec=v_spec)
cache.attach(model)
input_ids = load_eval_tokens(MODEL, "wikitext-2-raw-v1", n_tokens=512).unsqueeze(0)
with cache, torch.no_grad():
    model(input_ids, past_key_values=cache, use_cache=True)
bpe_k, bpe_v = cache.bits_per_entry()
print(f"bpe_k={bpe_k:.3f} bpe_v={bpe_v:.3f}")
```

Expect bpe_k ≈ payload + scales + skeptic-v2 pack charge, which at S=512 is `16·c_used/512` with `c_used` read from the pack (≤ C=1024) — the charge legitimately dominates at S=512; PRINT and sanity-check the arithmetic against the pack sidecar's c_used; real-S amortization is the VM's job. There is NO manual `16·C/S` arithmetic anywhere — `bits_per_entry` applies skeptic-v2 itself.

---

### Task 6: Local gate + push

- [ ] **Step 1:** `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` — expect 489+new passed / 17 skipped / 1 xfailed. Stage everything from Tasks 2–4; propose the combined message if not already committed per-task. STOP for approval.
- [ ] **Step 2:** STOP — propose pushing `feat/triton-decode-kernel` to origin (VM transport prerequisite). User approves the push explicitly.

---

## VM batch (Tasks 7–13) — one rented GH200, ordered, each gated

Transport per `vm-interaction-guide`: push → VM pull (or git bundle), `bash scripts/vm_setup.sh`, `uv run pytest -q` (record the actual GH200 count as the new baseline — local 489/17/1 plus CUDA/Triton extras). Long runs: detached `setsid` + log under `results/logs/`; verify 60–90 s in; VM has no push creds — results come back by git bundle.

**Cost estimate (pre-registered so overruns are visible):** caches ~30 min; gates <1 h; smoke ~10 min; probe 3–5 h; NIAH 2–4 h. Total ≈ one GPU-day. (Qwen3-8B is 36 layers vs Llama's 32 — the post-batched-flush per-sample rate estimate carries with ~10% headroom.)

### Task 7: [VM-RUN] Setup + baseline

```bash
bash scripts/vm_setup.sh
uv run pytest -q            # record GH200 baseline (incl. CUDA batched-flush A/B: the licenses re-pin on CUDA here)
```

`tests/test_streaming_batched_flush.py`'s cuda parametrization must be green — it is the bitwise license the probe's speed depends on. Qwen3 weights download on first use (public, no token).

### Task 8: [VM-RUN] Corpus caches — one model, eight offsets

```bash
for OFF in 2048 4096 6144 8192 10240 12288 14336 16384; do
  uv run python -m experiments.collect_cache --model-name Qwen/Qwen3-8B --seq-len 2048 --token-offset $OFF
done
```

Minutes on GH200. **Split (document in the run log, mirroring the Llama protocol):** fit = offsets {2048, 4096, 6144, 8192}, scored = {10240, 12288, 14336, 16384}. 4 fit caches × 2048 rows = 8192 rows vs C=1024 → 8× rank-deficiency margin (the same margin the Llama campaign had at C=1024). One model for gates AND probe (the binding Qwen3-8B choice) — no separate base/Instruct cache sets. `load_eval_tokens` materializes only the needed row prefix (commit `1c6a068`), so the wikitext slices are safe at any offset.

### Task 9: [VM-RUN] License gates A/B/C (GATES; cheap; run BEFORE any probe spend)

```bash
FIT="results/cache/qwen3-8b_2048_off2048.safetensors results/cache/qwen3-8b_2048_off4096.safetensors results/cache/qwen3-8b_2048_off6144.safetensors results/cache/qwen3-8b_2048_off8192.safetensors"

uv run python -m experiments.k4_fit_packs --corpus-cache-paths $FIT \
  --out-path results/cache/k4_packs_qwen3.safetensors --model-label qwen3-8b \
  --model-name Qwen/Qwen3-8B --w-source corpus

for S in 10240 12288 14336 16384; do
  SC=results/cache/qwen3-8b_2048_off${S}.safetensors
  uv run python -m experiments.k4_spectra  --cache-path $SC --corpus-cache-paths $FIT \
    --model-label qwen3-8b --model-name Qwen/Qwen3-8B --w-source corpus
  uv run python -m experiments.k4_spectra  --cache-path $SC \
    --model-label qwen3-8b --model-name Qwen/Qwen3-8B --w-source scored
  uv run python -m experiments.k4_frontier --cache-path $SC --corpus-cache-paths $FIT \
    --model-label qwen3-8b --model-name Qwen/Qwen3-8B --w-source corpus
  uv run python -m experiments.k4_frontier --cache-path $SC \
    --model-label qwen3-8b --model-name Qwen/Qwen3-8B --w-source scored
done
```

The `[rope_validation]` line fires inside every fit/spectra call — on the 8B it re-verifies the k_norm capture on the deployment model itself. Gate reads (same metric names as the Llama campaign — `retention_corpus`/`g0_pass_corpus` from k4_spectra's G0 verdict block at `k4_spectra.py:305-338`; `win_model`, `win_skeptic_deploy`, `layer_win_fraction_{model,deploy}`, `g1_pass` from k4_frontier's summary at `k4_frontier.py:610-631`, at budget 2.5):

- **Gate A (G0-corpus):** retention ≥ 0.90 licenses model-level accounting. **PRE-REGISTERED EXPECTATION: FAIL** — Llama measured 0.56–0.64 and the corpus-scale ablation proved the ceiling structural. Record the Qwen3 range either way. A Qwen3 fail in a similar band REPLICATES the structural-transfer-ceiling finding on a second family (a paper point). A PASS would be a genuine surprise (licenses model-level accounting on Qwen3 — report it as such, don't average it away). Gate A's outcome is NOT part of the replicates/does-not-replicate verdict.
- **Gate B (query-heldout):** the weighted arm's increment under corpus-W within ~20% of its scored-W increment (Llama: 1.54–1.59× vs ≈1.7×, transfer ratio 0.85–0.88 — PASS). PASS → corpus-W weighted packs stand. FAIL → refit the pack file with `--w-source none` (unweighted-KLT fallback, spec §7) and record that Qwen3 drops the W^½ claim. NOTE the Qwen3-specific stake: W is a QUERY moment, and on Qwen3 the captured queries are q_norm outputs (Task 3) — Gate B is also the first end-to-end evidence that the post-q_norm W statistics transfer.
- **Gate C (error bars — THE GO/NO-GO):** min over the 4 scored caches of `win_model` AND `win_skeptic_deploy` at budget 2.5 > 1×, with `layer_win_fraction ≥ 0.9` (frontier `g1_pass`) on every cache. Llama's floor was 6.19×. skeptic-v2 (c_used-based) is the deploy accounting on both sides of the comparison — automatic, no plan-side arithmetic. **FAIL → STOP: no probe spend**; write the does-not-replicate verdict at the gate layer (Task 13 template B) with the measured wins.

### Task 10: [VM-RUN] Pre-probe smoke — one real generation per arm + protocol scan

**Protocol (recorded): NON-THINKING — `enable_thinking=False` pre-registered for any template-applying path; this harness runs `chat_wrap=False` (no template at all), parity with the Llama duel, no CoT.**

Minutes before hours: catches pack/arm wiring, bpe accounting, and think-tag emission BEFORE the 3–5 h probe. The raw-decode shim is deliberate — Qwen3's `<think>`/`</think>` are special tokens that `skip_special_tokens=True` would strip, hiding an emission.

- [ ] **Step 1:** run (scratchpad, foreground — it's minutes):

```python
# scratchpad/qwen3_smoke.py — one real generation per arm, think-tag scan,
# skeptic-v2 bits sanity.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bmx.cache.generate import compression_for, generate_through_cache
from bmx.cache.recipes import spec_pair
from bmx.eval.layer_swap import load_eval_tokens

MODEL = "Qwen/Qwen3-8B"
PACK = "results/cache/k4_packs_qwen3.safetensors"


class _RawTok:
    """Decode WITHOUT skip_special_tokens so <think> emission is visible."""

    def __init__(self, tok):
        self._tok = tok
        self.eos_token_id = tok.eos_token_id

    def decode(self, t, skip_special_tokens=True):
        return self._tok.decode(t, skip_special_tokens=False)


tok = AutoTokenizer.from_pretrained(MODEL)
model = (
    AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    .to("cuda")
    .eval()
)
prompt_ids = load_eval_tokens(MODEL, "wikitext-2-raw-v1", n_tokens=512).unsqueeze(0)
for arm in ("fp16", "k4_b2.2", "k4_b2.5"):
    k_spec, v_spec = spec_pair(arm, rank=16, group=64, seed=0, pack_path=PACK)
    out = generate_through_cache(
        model, _RawTok(tok), prompt_ids, n_prefill=128,
        k_spec=k_spec, v_spec=v_spec, max_new_tokens=64,
    )
    think = ("<think>" in out) or ("</think>" in out)
    print(f"[smoke] arm={arm} think_tags={think} out={out[:120]!r}")
    bpe_k, bpe_v, comp = compression_for(model, k_spec, v_spec, 2048)
    print(f"[smoke] arm={arm} S=2048 bpe_k={bpe_k:.3f} bpe_v={bpe_v:.3f} x{comp:.2f}")
```

Reads: constructing the k4 arms asserts the pack covers budgets 2.2/2.5; `think_tags` expected **False** on raw completions (a True is a FINDING — record it and the affected fraction in Task 13, it does not auto-stop the probe but must be reported beside the headline); k4 bpe at S=2048 must be finite, < 16, and consistent with payload + skeptic-v2 charge (`16·c_used/2048` — read c_used off the pack sidecar).
- [ ] **Step 2 (OPTIONAL rider, not gated):** if the rental has slack after Task 12, run the Qwen3 fit-side corpus matrix A1/A2-style via `experiments.k4_corpus_transfer` on the qwen3-8b caches (per its module docstring + `docs/superpowers/specs/2026-07-23-k4-corpus-transfer-design.md`) — one command, reported separately from the replication verdict.

### Task 11: [VM-RUN] The probe — n=100 synthetic+code, 5 arms

**Protocol (recorded): NON-THINKING — `enable_thinking=False` pre-registered; `chat_wrap=False` (no template applied), parity with the Llama duel, no CoT.**

**Chained VM-side driver** (per ops discipline — probe then NIAH sequentially, one 8B process at a time, per-cell OK/FAILED lines):

```bash
mkdir -p results/logs
cat > /tmp/qwen3_batch.sh <<'SH'
set -u
cd "$HOME/bmx"
run() { NAME=$1; shift; echo "=== $NAME START $(date -u +%F' '%T)";
       "$@" && echo "=== $NAME OK" || echo "=== $NAME FAILED"; }
run probe uv run python -m experiments.k3_longbench \
  --model-name Qwen/Qwen3-8B --device cuda \
  --arms fp16 k4_b2.2 k4_b2.5 turboquant_mse_b3 turboquant_mse_k3v2 \
  --pack-path results/cache/k4_packs_qwen3.safetensors \
  --categories synthetic code --n-samples 100
run niah uv run python -m experiments.k3_niah \
  --model-name Qwen/Qwen3-8B --device cuda \
  --arms fp16 k4_b2.5 turboquant_mse_b3 turboquant_mse_k3v2 \
  --pack-path results/cache/k4_packs_qwen3.safetensors \
  --lengths 4096 8192 16384 32768
SH
setsid bash /tmp/qwen3_batch.sh > results/logs/qwen3_batch.log 2>&1 &
sleep 75 && tail -5 results/logs/qwen3_batch.log && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

Probe cell count: 5 arms × 4 datasets (synthetic → `passage_count`, `passage_retrieval_en`; code → `lcc`, `repobench-p`) × 100 samples = 2000 generations ≈ 3–5 h at the post-batched-flush rate. `chat_wrap` and `max_prompt_tokens` stay at defaults (False / 31500). Per-sample shards land under `partial/` → bootstrap CIs computed exactly as the code-CI cells were (10k resamples over per-sample shards; reuse that procedure/script from the `fdb5e6b` results commit). Crash → relaunch the same command with `--resume <run_dir>`.

### Task 12: [VM-RUN] NIAH bits-curve points (runs inside the Task-11 chain)

**Protocol (recorded): NON-THINKING — `enable_thinking=False` pre-registered; `chat_wrap=False` (no template applied), parity with the Llama duel, no CoT.**

Already chained above. 4 arms × 4 lengths × 5 default depths = 80 cells ≈ 2–4 h. The bits criterion reads the parquet's measured `kv_size_bits` (skeptic-v2 at actual S, `16·c_used/S` pack charge applied automatically by `bits_per_entry`) per (arm, length) — NOT a computed prediction. Geometry note for the read: C=1024 is IDENTICAL to Llama-3.1-8B, so no charge-scale shift is expected from C; the only accounting delta vs the Llama campaign's banked numbers is skeptic-v2's c_used < C, which CHEAPENS the pack charge on both models symmetrically. The pre-registered crossover window (below) is deliberately the same 8k–32k. The 32k point sits well inside Qwen3-8B's 40960-position window; the harness prints prompt lengths — verify ≤ 40960 − 50 in the log. Recall read: single-needle NIAH was a NULL on Llama (everyone ≈ fp16) — the expectation here is no-regression, not separation.

### Task 13: [VM-RUN] Verdict doc + traceability

Write `docs/2026-07-2X-qwen3-replication-results.md` (kill-or-confirm style). Evaluate the pre-registered criteria VERBATIM (below), fill ONE of the two templates, include the Task 1 source+config record (norm-then-rope quote included), Gate A/B/C table (Llama column beside Qwen3 for every gate), probe table with CIs, NIAH bits+recall table, the Task 10 think-tag scan result, and the non-thinking protocol line. Commit parquets (`results/k4_*/`, `results/k3_longbench/`, `results/k3_niah/`) + doc, bundle back. STOP for approval.

---

## Pre-registered success criteria (binding; written BEFORE any VM run)

**Gate layer:** Gate C passes (min `win_model` > 1 AND min `win_skeptic_deploy` > 1 at budget 2.5 across all 4 scored caches, `layer_win_fraction ≥ 0.9`). Gate C failure = does-not-replicate at the gate layer; no probe spend.

**Probe layer — the claim SHAPE must match Llama (all three):**
1. **Parity:** best-K4 (the better of k4_b2.2/k4_b2.5 by pooled synthetic+code avg) is ≥ turboquant_mse_b3 − noise: the bootstrap CI of (best-K4 − b3) pooled delta must NOT be entirely below 0.
2. **Retrieval edge:** (K4 − b3) synthetic-category delta point estimate > 0 with P(Δ>0) ≥ 0.75 at n=100 (Llama full-set: +3.25 [+1.02, +5.56], P=0.9975; n=100 CIs are wide — sign + P≥0.75 is the pre-committed shape test; a tighter full-set CI is the post-probe decision).
3. **Bits crossover:** measured skeptic-v2 `kv_size_bits` for k4_b2.5 > b3 at 4096 and < b3 at 32768, with the crossover therefore between 8k and 32k (Llama: between 8k and 16k; with identical C=1024 geometry the Llama window itself is the central expectation, but the pre-registered window stays 8k–32k).

**Secondary (report, not gate):** NIAH recall no-regression (k4_b2.5 mean within noise of fp16 — Llama measured a null); Gate A outcome vs the Llama band 0.56–0.69 (replicating the structural ceiling); Gate B transfer ratio vs Llama's 0.85–0.88; think-tag emission count from the Task 10 scan (expected 0).

### Verdict template 1 — REPLICATES

> On Qwen3-8B (post-trained, non-thinking protocol, chat_wrap=False), Gate C passed (min G1 win {X}×/{Y}× model/deploy), and the n=100 probe reproduced the Llama claim shape: best-K4 pooled delta vs b3 = {Δ} [{CI}], synthetic delta = {Δs} (P(Δ>0)={p}), skeptic-v2 bits crossover vs b3 at ~{N}k (k4_b2.5 {a} vs b3 {b} at 32k). NIAH recall held fp16-level. The K4 result is a property of the method, not a Llama-3.1 artifact — and it survived a qk-norm architecture whose pre-RoPE capture point differs from Llama's. Gate A measured {r_lo}–{r_hi} ({replicating/breaking} the structural transfer ceiling). Post-probe decision now open: full 6-category Qwen3 table (≈40–60 h) or ship the two-family evidence as-is.

### Verdict template 2 — DOES NOT REPLICATE

> On Qwen3-8B (post-trained, non-thinking protocol, chat_wrap=False), the replication broke at {Gate C | criterion 1/2/3}: measured {numbers} vs pre-registered {rule}. The paper's claim is scoped to Llama-3.1 unless the mechanism difference is diagnosed; candidate suspects, in order: the per-head qk-norm reshaping the pre-RoPE key statistics (RMS-equalized rows can flatten the spectrum the waterfill exploits), post-q_norm query statistics shifting the W moment, rope_theta=1e6 position mixing, the post-trained-only fit corpus (packs fit through a post-trained checkpoint on wikitext), and the 36-layer depth profile shifting the allocation. No full-table spend. Honest negative; the gates did their job.

---

## Self-Review

**Amendment coverage:** (1) Qwen/Qwen3-8B + non-thinking protocol → Global Constraints, recorded lines on Tasks 10/11/12, both templates; the chat_wrap=False ⇒ no-template mechanism is spelled out so nobody hunts for an `enable_thinking` call site that this harness never reaches, and the think-tag scan uses a raw-decode shim because `<think>` is a special token that the harness's `skip_special_tokens=True` decode would hide. (2) k_norm capture point → Architecture, Global Constraints, Task 1 Part A (installed-source verification with STOP rule), Task 3 (dispatch + four pinning tests); resolved beyond the amendment's letter: (a) today's `attach()` does NOT raise on Qwen3 — Qwen3 has a `k_proj`, so the failure mode is silent wrongness, which the structural dispatch + tests close; (b) the q hook moves to `q_norm` output too (the K4 W moment is a QUERY statistic — pre-norm queries would corrupt it), pinned by the hooks-land-on-qk-norm test and flagged at Gate B; (c) `packed_streaming.attach` is a third k_proj site — same one-line fix + structural pin, scope-flagged. (3) C=1024 verified in Task 1 (hard assert), C=512 note removed, skeptic-v2 noted as automatic in Tasks 5/9/10/12. (4) Qwen3-0.6B pre-flight → Task 5 with both retained gates; upgraded to seq 2048 × 2 fit offsets because the 0.6B's C is 1024 (not the old 0.5B's 128) and one 1024-row cache would be exactly rank-C. (5) Fallback → one Global-Constraints line, referenced at Tasks 1/5, no parallel track. (6) tiny_qwen3 → Task 2, explicit head_dim=8 (Qwen3Config defaults head_dim=128 — verified in the installed configuration_qwen3.py), qk-norm unconditional. (7) skeptic-v2 / corpus-transfer rider (Task 10 Step 2, one line, optional) / module-form launches (every command in Tasks 5, 8–11 uses `python -m experiments....`) / prefix-row loader (noted at Task 8). (8) claim-shape criteria + both templates kept, names/numbers updated; battery baseline 489/17/1 in Global Constraints and Tasks 6/7.

**Placeholder scan:** every command carries real paths/model ids; the only deliberate variables are `{...}` slots inside the two verdict templates (to be filled by measurement) and the Gate-B-conditional `--w-source none` refit. Task 4 flags the two kwarg names (`spec_pair("k2b", ...)`, `recent_window`) and the greedy-token seed sensitivity that must be mirrored from the existing tests rather than trusted from this plan; Task 3's packed test flags the constructor-kwargs mirror.

**Type/name consistency:** pack file `k4_packs_qwen3.safetensors` used identically in Tasks 9/10/11/12 (and `k4_packs_qwen3_06b.safetensors` only in Task 5); cache pattern `qwen3-8b_2048_off{N}.safetensors` matches `collect_cache.py:85-91`'s `model_short` rule and is used identically in Tasks 8/9; `resolve_qk_capture_modules` has one signature everywhere (Task 3 code, Task 2/3 tests, Architecture); gate metric names (`retention_corpus`, `g0_pass_corpus`, `win_model`, `win_skeptic_deploy`, `layer_win_fraction_{model,deploy}`, `g1_pass`) match `k4_spectra.py:305-338` / `k4_frontier.py:610-631` as re-verified on this checkout; probe/NIAH arm names (`k4_b2.2`, `k4_b2.5`, `turboquant_mse_b3`, `turboquant_mse_k3v2`) match `recipes.py`'s parsers (`k4_b{budget}` at `recipes.py:88`, `_b{bits}` and `_k{kb}v{vb}` in the same parser block); the harness `spec_pair(arm, rank=cfg.rank, group=cfg.group, seed=cfg.seed, pack_path=cfg.pack_path)` call (`k3_niah.py:105-107`, `k3_longbench.py:166-168`) is what Task 10's smoke mirrors (`rank=16, group=64, seed=0` are those Configs' defaults).

**Known open decisions deferred to run time:** (a) Gate B failure flips the pack fit to `--w-source none` — the probe arms are unchanged either way; (b) whether to add k4_b2.2 NIAH points at 32k (only if the chain finishes early — NOT pre-registered, reported separately if run); (c) the Task 10 corpus-transfer rider (optional, slack-dependent, separately reported); (d) the full 6-category Qwen3 table is explicitly post-probe, per the binding scope.

**Deliberate deviations from the format model:** Task 3 is a real `src/bmx` task (the original plan's "zero src changes" null hypothesis is dead on Qwen3 — the qk-norm capture point is a verified architectural fact, not a suspicion), and Task 12 rides Task 11's chained driver instead of a separate launch — a consequence of the one-8B-process rule and the replication-not-development scope. Task 10 replaces the original's separate-Instruct-packs task: one post-trained model means one pack file, and the freed slot funds a cheap pre-probe smoke that the non-thinking protocol scan genuinely needs.
