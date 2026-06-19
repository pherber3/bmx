# K3 — Live Streaming KV-Compression Cache + In-Practice Verdict

**Date:** 2026-06-19
**Status:** Approved design (pending user spec review)
**Scope:** First implementation cycle — local gates T0–T6 plus model-agnostic experiment
script (T7) and plot (T8). The authoritative SOTA/VM run is an explicit follow-up,
not in this spec.

## Purpose

The KV program (K1→K2c) closed positive on **quantized-prefill perplexity**: prefill N
tokens, quantize the *whole* cache at once, teacher-force a continuation, read NLL. That
is a clean scientific gate but **not real usage**. Real usage is (a) quantize-on-append —
each decoded token's K/V compressed as written; (b) autoregressive generation where the
model attends to its own *compressed* cache, so errors compound over hundreds of steps;
(c) the K2c streaming design (freeze the pre-RoPE key subspace at prefill, project each
appended key) running in a live loop. None of (a)–(c) has been measured.

This cycle builds the **quantize-on-append cache class** named as open engineering work in
CLAUDE.md/README, and uses it to answer one practical question end-to-end:

> Does the K2b structure-aware recipe (fit a per-layer pre-RoPE low-rank subspace) hold up
> in **live generation** — on both **quality and real GPU memory** — against the
> data-oblivious near-optimal baseline (TurboQuant) on **identical machinery**?

The contest is the substantive one the vault records: TurboQuant is *data-oblivious,
worst-case-optimal* (Haar rotation + Lloyd-Max, no per-model fitting); K2b is
*structure-aware*. The K2 program already found worst-case-optimal coding concedes 2–3× to
structure-aware coding on real keys. This cycle tests whether that margin survives **live,
on-append, autoregressive** decode — the regime no prior gate touched.

### Origin note (the "beat TurboQuant" framing)

This work was prompted by a public KV-compression repo (cloned in, inspected, deleted) that
claimed "TurboQuant is king." That repo's TurboQuant was a stub (a global `×0.707` in place
of a random rotation) and its quality benchmark never ran compressed inference
(`tq_ppl = fp16_ppl`). There is no real method there to beat. The defensible result is not
"beat that code" but "test our recipe in practice against a **faithful** TurboQuant on real
end-to-end metrics" — which this spec does.

## Decisions (settled with user)

- **Success bar:** quality **and** memory, end-to-end. A drop-in cache that quantizes on
  append, runs live `generate()`, holds quality within target, and shows *measured* memory
  reduction. Not a notional bpe number.
- **Kernels: out of scope.** No Triton / fused dequant-attention. Plain-PyTorch
  dequant-then-attend. Decode may be slower; wall-clock is *recorded, not optimized*
  (premature optimization explicitly rejected). The fused kernel stays a separate future
  item gated by the Track B byte model.
- **Integration (revised after reading transformers 5.11 source + vault):** mirror HF's own
  `QuantizedCache` pattern. transformers 5.x splits the cache into a `Cache` *container* + a
  per-layer `CacheLayerMixin`; the load-bearing contract is **`layer.update(k, v) -> (k, v)`
  returns DEQUANTIZED tensors for attention while storing only the COMPRESSED form**. The
  model never holds a dense cache — the dequant is the transient return value, discarded after
  attention. So real memory is structural, via the official API, NOT bespoke surgery. We build
  a `DynamicLayer` subclass (`StreamingQuantizedLayer`) whose `_quantize`/`_dequantize` carry
  our codec + frozen pre-RoPE subspace, replicated across layers by a thin `Cache` container
  (`StreamingQuantizedCache`). This resolves the earlier memory-mechanism fork entirely:
  HF's `QuantizedCache` IS the dequant-on-read pattern, already shipping. (Rejected: per-step
  manual loop — notional memory, not drop-in. Rejected: attention monkeypatch — unnecessary;
  the layer contract already gives read-scoped dequant.)
  Confirmed against `transformers/cache_utils.py` (5.11.0): `CacheLayerMixin` abstract methods
  `lazy_initialization, update, get_seq_length, get_mask_sizes, get_max_cache_shape`;
  `QuantizedLayer.update` re-quantizes full history on each `residual_length` flush and
  dequants full history each step — so our "re-quantize the slab" path is production-faithful,
  not a shortcut. The O(S)/flush cost is latency (kernels, deferred), not correctness.
- **Recipe architecture: Approach A — faithful streaming cache.** Freeze the pre-RoPE key
  subspace and the V rotation at prefill; project each appended key onto the frozen subspace
  (r dot products), quantize residual per-channel; small fp16 recent window; RoPE applied at
  read at each token's true position. This *is* K2c made live, so it tests the published
  streaming claim directly. (Rejected: B block-recompute — refits per block, the thing K2c
  showed is unnecessary, measures a weaker claim. C slow-cadence K refit — kept as an
  **unbuilt** fallback only if long-context drift bites at the VM run; K2c found drift-flat
  so building it now is premature.)
- **Codec pluggability:** the live cache accepts any `CacheCodecSpec` for K and V, so K2b,
  TurboQuant, and KIVI run as **swappable arms on one code path**. Fairness is structural,
  not promised.
- **Faithful baselines already exist** (verified against the personal-brain vault):
  - TurboQuant_mse = Haar rotation → Lloyd-Max per coord = existing `_turboquant_mse`.
  - TurboQuant_prod = MSE at (b−1) + 1-bit QJL on residual = existing `_turboquant_prod`
    (honest bpe `(b-1)+1+32/C`, matching the vault's two-fp16-norms accounting).
  - KIVI = per-channel K / per-token V RTN = existing `rtn_channel` / `rtn_token`.
  No new codec arm is written. A pin-test (T0) asserts these stay paper-faithful.
- **K2b headline arm (fixed point, no sweep):** K = pre-RoPE `lowrank_rtn_channel`, rank 16,
  3 bits, per-channel residual; V = `rotate`/Lloyd, 2 bits. ≈3.0 bpe — the K2c-validated
  streaming point the docs already publish.
- **Models:** factory `tiny_llama` (GQA+RoPE, offline) for correctness/plumbing tests; a
  tiny *real* GQA model — **default Llama-3.2-1B** (Qwen2.5-1.5B as fallback if gating/
  download blocks it) — for local quality/memory numbers.
  The authoritative run targets a SOTA GQA model (Qwen3 / Gemma-class, ~27–32B) on the
  NVIDIA VM — **follow-up spec**; the experiment script is written model-agnostic so that run
  is a config change, not new code.
- **Validate locally, measure authoritatively remotely:** `resident_kv_bytes()` sums the
  *compressed* layer state (quantized indices + scales + frozen subspace + fp16 residual
  window) — a real packed footprint by construction (the layer never persists the dense
  dequant). Locally it proves the mechanism (resident = compressed, no dense persistence);
  the process-level peak-memory 5× and absolute numbers come from the CUDA VM run.
- **Quality metric:** two-tier. (1) generated-continuation perplexity vs fp16 at every gate;
  (2) **needle-in-a-haystack** retrieval accuracy at long context as the headline probe —
  TurboQuant's *own* paper benchmark (Full-Attention parity at 4× on Llama-3.1-8B), so
  matching/beating it there is the most credible "in practice" claim and stresses exactly the
  inner-product/attention-score regime where structure-aware key coding should win.

## Architecture

Four units. The only genuinely new code is the cache (T2) and the needle harness (T6); the
rest is a mechanical lift, a thin script, and a plot — all along existing repo patterns.

```
src/bmx/cache/
├── specs.py        # NEW: CacheCodecSpec lifted out of ppl_eval (shared, no cycle)
├── streaming.py    # NEW: StreamingQuantizedLayer(DynamicLayer) + StreamingQuantizedCache
│                   #      (Cache container) — mirrors HF QuantizedCache/QuantizedLayer split
├── live_eval.py    # NEW: live_generation_ppl through the streaming cache
├── needle.py       # NEW: needle-in-a-haystack retrieval probe
├── codecs.py       # unchanged (faithful TurboQuant/KIVI already here)
├── ppl_eval.py     # re-imports CacheCodecSpec from specs.py (no behavior change)
├── collect.py rope.py metrics.py   # unchanged
experiments/
├── k3_live_generation.py           # NEW: thin tyro; sweep arms × context; emit parquet
└── plots/plot_k3.py                # NEW: read parquet, never refit
tests/
├── test_streaming_cache.py         # NEW: plumbing/quality/memory gates
├── test_live_eval.py test_needle.py test_k3_experiment.py   # NEW
└── test_turboquant_faithful.py     # NEW: T0 baseline-fidelity pin
```

### The cache class pair (mirrors HF `QuantizedCache`)

`StreamingQuantizedLayer(DynamicLayer)` — the per-layer unit. Implements the `CacheLayerMixin`
contract: `lazy_initialization` (fit/freeze the pre-RoPE K subspace + V rotation on the first,
prefill `update`), `update(k, v, *args, **kwargs) -> (k_deq, v_deq)` (store compressed, return
dequantized for attention), `get_seq_length`, `get_mask_sizes`, `get_max_cache_shape`. Codec
driven by the layer's K-spec/V-spec. Owns a pre-RoPE key hook (keys arrive post-RoPE in
`update`, so pre-RoPE capture needs the k_proj hook, fitted into layer state). Stores
`_q_keys`/`_q_values` (compressed) + fp16 residual window; never persists the dense dequant.

`StreamingQuantizedCache(Cache)` — thin container replicating the layer across the model's
layers (via `layer_class_to_replicate` or explicit `layers=[...]`), passed as
`past_key_values=` to `model.generate()`. Exposes `attach(model)`/`detach()` for the pre-RoPE
hooks, `bits_per_entry()`, and `resident_kv_bytes()` (sum of compressed layer state).

### `cache/specs.py` (T1) — mechanical lift

Move the `CacheCodecSpec` dataclass verbatim from `ppl_eval.py` into `cache/specs.py`, and
re-export it from `ppl_eval` (`from bmx.cache.specs import CacheCodecSpec`) so existing
imports keep working. Rationale: `streaming.py` needs `CacheCodecSpec`; if it imported the
type from `ppl_eval`, and `ppl_eval` later needs anything from `streaming`, that is a cycle.
A neutral `specs.py` both modules import prevents it. No field changes; behavior identical
(guarded by the existing `ppl_eval` tests).

### `cache/streaming.py` (T2) — `StreamingQuantizedCache(DynamicCache)`

The one new stateful unit. Subclasses transformers 5.x `DynamicCache`. Driven by a K-spec
and a V-spec (`CacheCodecSpec`).

**Per-layer resident state (the compressed representation — nothing fp16-cache-sized
persists):**
- K (when spec is `lowrank_rtn_channel` + `pre_rope`): frozen subspace basis `V:(C,r)` fp16,
  per-token coeffs `(S,r)` fp16, per-channel-quantized residual (packed) + scales fp16.
- V (when spec is `rotate`/Lloyd): frozen rotation **seed** (0 stored bytes — regenerated via
  `quant.hadamard`), quantized V (packed) + scales.
- fp16 recent window `(h_kv, W, d)` for the last W tokens (both K and V), W from spec.
- For non-lowrank specs (TurboQuant/KIVI/rtn): per-append quantize via `quantize_cache` with
  no frozen subspace; same window + packed-state discipline.

**Lifecycle:**
- *Prefill / freeze:* at the end of the prefill block, fit the pre-RoPE K subspace
  (`truncated_svd`) and V rotation, quantize the prefill block, store frozen factors. Mirrors
  K2c freeze.
- *Append (`update`):* project the appended pre-RoPE key onto frozen `V` (r dot products),
  quantize the residual per-channel; quantize V into the frozen rotation; push token into the
  fp16 window; flush the oldest window token into packed state when the window fills.
- *Read:* reconstruct K (`V @ coeffs.T + dequant(residual)`), `apply_rope` at each token's
  **true position**, dequant V, concat the fp16 window. **Read-scoped:** the reconstructed
  fp16 tensor is local, handed to attention, and dropped — never cached on the object. Peak
  transient = one layer's cache (the profile `collect.py` already relies on).

**Layout discipline:** the `(h,S,d) ↔ (S,h·d)` mapping goes through
`collect.to_matrix`/`from_matrix` only — never hand-rolled. RoPE through `rope.apply_rope`
with `rope_cos_sin(config, S)`; never re-derive frequencies. Shape asserts at every boundary;
no silent dtype coercion. Honest bpe via the codec's own returned `bpe`.

### `experiments/k3_live_generation.py` (T7)

Thin tyro CLI. Args: model name, context/continuation lengths, arm set, recent-window W,
seed, output dir. Builds `StreamingQuantizedCache` from each arm's spec, runs real
`model.generate()`, and for each arm records: generated-continuation perplexity, measured
peak resident KV memory, honest bpe (K and V), wall-clock decode (recorded, not optimized),
config+env+git SHA via `artifacts.py`. Arms: `{fp16, k2b, turboquant_mse, turboquant_prod,
kivi}`. Model-agnostic (structural dispatch) so the VM/SOTA run is a config change.

### `experiments/plots/plot_k3.py` (T8)

Reads the committed parquet, never refits. Two figures: quality-vs-bpe (ppl gap per arm at
its honest bpe) and memory-vs-context. Explicit run selection (no blind concat).

## Test ladder (acceptance gates, cheapest-and-most-diagnostic first)

- **T0 — baseline-fidelity pin** (`test_turboquant_faithful.py`, offline). Assert
  `turboquant_mse` = rotation→Lloyd, `turboquant_prod` = (b−1) MSE + 1-bit QJL with bpe
  `(b-1)+1+32/C`, KIVI = `rtn_channel`/`rtn_token`. Cheap insurance the head-to-head is fair.
  *Independent; can run first.*
- **T3 — plumbing gate** (`tiny_llama`, in suite). With a **no-op fp16 codec**,
  `StreamingQuantizedCache` produces **bit-identical** logits to plain `DynamicCache` over a
  real `generate()`. Isolates cache-surgery correctness before quantization. Hard equality.
- **T4 — quality gate + head-to-head** (tiny real GQA, local). Live `generate()` with the
  K2b spec; generated-continuation ppl gap vs fp16 within target. **Threshold:** K2b ≤ +1%
  ppl vs fp16 (the K2b/K2c docs report +0.5% at quantized-prefill; +1% gives live-decode
  error-compounding headroom — if K2b exceeds +1%, that is itself a reportable finding that
  on-append compounding costs more than batch quantize, not a silent pass). Same loop and
  threshold reported for TurboQuant_mse/prod and KIVI — the head-to-head on one code path;
  the comparison is lower ppl at equal-or-lower honest bpe.
- **T5 — memory gate** (local proxy → VM-authoritative). Assert peak resident KV ≈ predicted
  compressed bytes (within one-layer transient slack) **and** meaningfully below the fp16
  baseline. Fails loudly if a stray reference holds the fp16 dequant. This is what makes "it
  works" mean *both* quality and memory.
- **T6 — needle harness** (`test_needle.py` + harness used by T7). Insert a known fact at a
  controlled depth in long context; assert the model retrieves it under each arm. Headline
  probe; matches TurboQuant's paper benchmark.

## Subagent task decomposition (for subagent-driven development)

Each task is independently testable with an explicit acceptance gate — a subagent cannot
"mostly" pass a bit-identical assertion. Dependency structure:

```
T0 (baseline pin) ── independent, run first/parallel
T1 (specs lift)  ── independent, mechanical
T2 (cache)       ── depends on T1; TDD against tiny_llama; the one hard unit
T3 (plumbing)    ── depends on T2
T4 (quality)     ── depends on T2 (+ tiny real model)
T5 (memory)      ── depends on T2
T6 (needle)      ── depends on T2
T7 (experiment)  ── depends on T2, T4, T6
T8 (plot)        ── depends on T7 parquet
```

T3–T6 are largely parallel once T2 lands. T2 is the critical path and gets built test-first
against the factory model before any real-model run.

## Conventions honored (per CLAUDE.md)

- Honest bpe, ALL metadata counted; comparisons align on total bits, never rank.
- Distortion/perplexity metrics, not Frobenius. Perplexity is end-to-end verdict.
- `to_matrix`/`from_matrix` is the only K1-layout site; RoPE via `apply_rope`; keys quantized
  PRE-RoPE.
- fp32 in experiments/codecs, fp16 cache storage, fp64 only in numeric tests. Fail fast.
- Tiny offline models from `tests/factories.py`; never download in tests.
- Experiments are thin tyro scripts; figures read parquet, never refit; commit metrics/
  figures, never checkpoints or raw caches.
- No commit without explicit user approval; `ruff format` → `ruff check` → `pytest -q` before
  any staging.

## Execution model

- **Subagent-driven with supervision.** Each task (T0–T8) is dispatched to a subagent built
  test-first. The supervising agent (not the subagent) reviews each deliverable against its
  acceptance gate before accepting — bit-identical / threshold / byte-assertion gates are
  hard, not "looks done." Failed gates are returned with specifics, not patched over.
- **Final simplify pass.** After all gates are green, run the `simplify` skill over the new
  code (streaming.py, specs.py, the experiment + plot + tests) to refactor for reuse,
  altitude, and clarity — quality-only, no behavior change — then re-run
  `ruff format` → `ruff check` → `pytest -q` to confirm still-green before proposing a commit.

## Out of scope (this cycle)

- Triton / fused dequant-attention kernel (separate item; gated by Track B byte model).
- The authoritative SOTA/VM run (follow-up spec; experiment script is already model-agnostic).
- Approach C (slow-cadence K-subspace refit) — unbuilt fallback only if VM long-context drift
  contradicts K2c.
- 32k-context drift re-check (belongs to the VM follow-up).
```
