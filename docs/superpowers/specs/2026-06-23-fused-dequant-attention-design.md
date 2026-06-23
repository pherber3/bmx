# Fused dequant-attention kernel — design (2026-06-23)

Closes the "Open (engineering, not science)" item in CLAUDE.md: the fused
dequant-attention kernel that makes the bpe compression *real at runtime* and
unblocks a batched 128k-context NIAH/LongBench sweep on the GH200.

The science is settled and committed (`docs/2026-06-21-niah-longbench-frontier-results.md`):
at matched ~7× compression k2b beats turboquant on both NIAH and LongBench Code,
and the key finding is the **key-bits ↔ context tradeoff** (3-bit-key k2b holds
fp16-quality retrieval to 128k). This spec is engineering only — no new science.

## The reframe (the crux this design rests on)

The 128k OOM is **not** dominated by transient codec scratch (~7 GB), as the
results doc originally framed it. Reading `src/bmx/cache/streaming.py`, the
`StreamingQuantizedCache` decode path materializes, every step:

1. `_q_prefix_k` / `_q_prefix_v` — the **dequantized** frozen prefix, full
   `(h_kv, S, d)` fp16 (`streaming.py:312-316`).
2. `k_hat` / `v_hat = cat(prefix, fp16 tail)` — a **second** full dense fp16
   slab, stored as `self.keys`/`self.values` for stock attention to read
   (`streaming.py:344-362`).
3. Per-flush codec scratch (Hadamard / SVD / Lloyd / QJL working tensors).

So at 128k the cache holds **~2 full fp16 copies of K and V** (~32 GB) plus the
transient scratch. The bpe compression is an **accounting fiction at runtime** on
this path — `memory_report()` itself admits it (`streaming.py:460-463`): "the
literal 5× is the fused-kernel/paged-store VM measurement." This is deliberate:
`StreamingQuantizedCache` is a correctness/quality vehicle (does on-append quant +
frozen subspace hold accuracy? what is the honest bpe?), never a memory vehicle.

**The kernel's real target is the resident double-copy** (terms 1+2), not the
transient scratch. A path that keeps packed codes resident and dequantizes K/V
block-by-block inside attention — never building the dense slab — makes bpe real
at runtime and subsumes the transient-scratch problem (no big scratch tensors
either).

### Grounding (Personal Brain vault)

The resident-cost arithmetic is the canonical KV-cache formula, not a guess:

- KV resident size `= 2 · L · n_kv_heads · head_dim · seq_len · bytes`
  (*The Physics of LLM Inference* ~line 778; *AI Systems Performance Engineering
  Textbook* ~line 27986; wiki note **Walk me through what happens to the KV cache
  during a 1000-token generation on a 7B model**). For Llama-3.1-8B
  (L=32, h_kv=8, d=128): 4 KB/token/layer → 128 KB/token → **16 GB at 128k fp16
  per copy**. Two copies ≈ 32 GB — matches the doc's "16 GB full-fp16 KV" term.
- Inference memory budget = weights (fixed) + KV (∝ b·s) + activations (∝ b·s
  prefill, b·1 decode) + attention scores `O(s²)` "this is the big one… in
  inference we can free it immediately" (*Physics* ~line 813). The chunked path
  replaces the `O(h·s)` SDPA score row with an `O(h·block)` tile.
- Online-softmax tiled attention is mathematically exact (no approximation):
  `online_attention_update` (*Physics* ~line 1931); wiki **How does the online
  softmax trick enable tiled attention computation**. The dequant-in-attention
  contraction order is the **FlashTPA** pattern (wiki **Tensor Product
  Attention**, §5 / Algorithm 2): a blocked online-softmax kernel that
  reconstructs K/V inside the loop without materializing K/V or the score matrix.

## Scope, phases, and "done"

Three sequenced phases. **This spec commits to Phases 1 + 2.** Phase 3 (Triton)
is gated behind a written decision rule and is a separate spec when triggered.

| Phase | What | Runs where | Gate to next |
|---|---|---|---|
| **1. Byte-ledger + census** | Analytic memory ledger (`bench/`-style honest bytes) for resident-packed + per-block scratch + attention working set; a CUDA memory-census instrument measuring the real resident-vs-transient split & peak per arm. | Ledger: local (pure Python). Census: VM (CUDA). | Ledger predicts whether chunked-PyTorch clears 94.5 GB at 128k. |
| **2. Chunked-PyTorch dequant-attention** | Reference dequant-attention: packed codes resident, online-softmax over per-block dequant, free each block. Pure PyTorch — **testable on AMD locally**. Bit-for-bit gated vs `StreamingQuantizedCache`. | Local (AMD) for correctness + memory-model validation; VM for the real 128k peak. | Census shows whether chunked's `peak_decode` clears the ceiling. |
| **3. Triton fused kernel** | Real fused dequant-attention kernel. **Gated** (see rule below). | VM only (CUDA). | — |

**"Done" for this spec (Phases 1 + 2):** a validated byte-ledger that predicts
the 128k peak; a census instrument; a correctness-proven chunked-PyTorch
dequant-attention path that (per ledger + a VM measurement) clears the 128k
batched-sweep ceiling — **or** an honest finding that it does not and Triton is
required, with the numbers to justify it. The concrete payoff is unblocking the
batched 128k sweep.

**Honesty caveats baked in:** (a) AMD-local Phase-2 numbers validate
*correctness + the memory model*, not the literal process-RSS / latency a VM
Triton kernel delivers (PyTorch allocator + no true fusion) — the deployment "5×"
headline still needs a VM run; (b) the census must report resident-vs-transient
explicitly so nobody over-claims from local numbers.

## Component 1 — byte-ledger (`src/bmx/bench/kv_memory.py`)

Pure-Python honest bytes ledger, same spirit as `bench/factored_matvec.py`: no
allocation, no CUDA, just arithmetic. "Use the Track B byte model to predict
before building" (CLAUDE.md) means **adopt its methodology** — an honest bytes
accounting, correctness-gated before any measurement — not literally reuse
`factored_matvec` (which models a different computation and has no cache/peak
notion).

**`KVMemCase` (dataclass):** `seq_len S`, `n_layer L`, `h_kv`, `d_head d`,
`bpe_k`, `bpe_v`, `block`, `recent_window W`, `path ∈ {dense_stream, chunked}`,
plus model-level fixed terms `weights_bytes`, `act_bytes`, and a `logits_bytes`
parameter (the `logits_to_keep=1` fix means ≈1 position; carried explicitly so we
can sanity-check the historical 62.6 GB all-position term).

**Three cache terms, per path:**

| Term | `dense_stream` (today) | `chunked` (Phase 2) |
|---|---|---|
| Resident cache | `2 · L · h_kv · S · d · 2` (dense `k_hat`+`v_hat`) **+** dense `_q_prefix_k/v` ≈ **2 full fp16 KV copies** | packed codes `L · h_kv · S · d · (bpe_k+bpe_v)/8` — bpe finally resident |
| Per-block transient | per-flush scratch sized to whole committed prefix (worst arm) | one block: `~ h_kv · block · d · 2` × small const |
| Attention working set | full `(h_kv, S)` score row (stock SDPA) | online-softmax `(h_kv, block)` tile + `(h_kv, d)` accumulator |

**Output:** `predict_peak(case) -> dict` →
`{resident_bytes, transient_bytes, attn_bytes, weights_bytes, act_bytes,
logits_bytes, predicted_peak_bytes}` plus a `compression_at_runtime` ratio
(resident dense_stream ÷ resident chunked) — the number that quantifies "bpe
becomes real."

**Headline prediction it must produce:** Llama-3.1-8B geometry (L=32, h_kv=8,
d=128), S=131072, k2b bpe, GH200 weights/act terms from the measured 92 GB fp16
decomposition → predicted peak for `dense_stream` (should reproduce the observed
~99–100 GB OOM, validating the ledger) vs `chunked` (should land well under
94.5 GB if the thesis holds). **If the ledger says chunked still OOMs, Phase 3 is
mandatory and we know it before writing the kernel.**

**Correctness gate (`bench/` discipline):** the `dense_stream` prediction is
validated against the census measurement (Phase 1 VM) before the `chunked`
prediction is trusted — the same "assert correctness before you believe the
model" rule `bench/harness.py` applies to factored_matvec.

## Component 2 — chunked dequant-attention (`src/bmx/cache/chunked_attention.py`)

`chunked_dequant_attention(q, packed_k_blocks, packed_v_blocks, k_meta, v_meta,
...) -> attn_out` — computes one layer's attention directly from packed cache
state, block by block, via online softmax:

```
acc = 0; m = -inf; l = 0                     # running state, per (h_kv, n_q, d)
for blk in blocks:                           # committed-prefix blocks
    K_blk = dequant_block(packed_k_blocks[blk], k_meta)   # (h_kv, block, d) transient
    V_blk = dequant_block(packed_v_blocks[blk], v_meta)
    S = q @ K_blk.transpose                   # (h_kv, n_q, block) score tile
    m, l, acc = online_update(acc, m, l, S, V_blk)        # Physics ~line 1931
    del K_blk, V_blk                          # freed before next block
# then the fp16 recent-window tail, same online update
return acc / l
```

- `dequant_block` reuses the **existing per-arm dequant math**, but this
  requires a **codec split** (see "Codec split" below) — it is not a pure storage
  change. Today every codec returns `(M_hat, bpe)` with `M_hat` already
  *dequantized*; the packed representation (RTN indices + per-group scales, Lloyd
  codebook assignments + per-token norms, low-rank factors, QJL signs) is a local
  variable and **discarded** (`rtn.py:13` returns `(Q*scale)`; `_turboquant_mse`,
  `_lowrank_rtn_channel`, `qjl_reconstruct` likewise). To store packed codes and
  dequantize at read, each arm splits into `quantize → packed` and
  `packed → dequant_block`. `dequant_block` is then called one block at a time
  **at read**. The math is identical; what changes is that the packed codes are
  *kept* (not the dequant) and dequant happens at read.
- **RoPE in the loop:** RoPE is applied to each dequantized K block **before**
  the `q @ K_blk^T` contraction, at the block's absolute positions
  `[block_start, block_end)` — identical to the current
  `_quantize_k_block_pre_rope` (`streaming.py:176-193`), just per-block inside
  the attention loop. The growing `_rope_cos`/`_rope_sin` table is shared state.
  (Implementation note: apply RoPE *before* the contraction, never after.)

### Codec split (`src/bmx/cache/codecs.py` — edited, not read-only)

Each arm is split into two functions sharing the same math:

- `quantize_packed(arm, M, ...) -> (packed, bpe)` where `packed` is an
  arm-specific bundle of indices/scales/factors/codebook-assignments/norms.
- `dequant_packed(arm, packed, ...) -> M_hat`.

The existing `quantize_cache(...) -> (M_hat, bpe)` becomes
`dequant_packed(*quantize_packed(...))` and is kept for the current
`StreamingQuantizedCache` path (so that path and all existing tests are
unchanged). The split is mechanical (each arm's body becomes two functions) and
the bit-for-bit gate tests it directly: `dequant_packed(quantize_packed(M))` must
equal the current `quantize_cache(M)` `M_hat` exactly, per arm.
- **GQA-aware**: `h_kv` broadcast to query heads (`repeat_interleave`), since
  Llama-3.1 is GQA (vault GQA note).
- Online softmax is mathematically exact → output equals full-prefix attention
  (vault).

**Golden reference oracle (the quality yardstick).** Alongside
`chunked_dequant_attention`, the same module ships `naive_dense_attention` — the
slowest, most-obviously-correct path: dequant *everything*, a single full
softmax, no online trick, no chunking. It is the **single named ground truth**
that every faster path is diffed against (online-softmax, chunked, packed, and —
critically — the future Phase-3 Triton kernel), via an `attention_diff(a, b) ->
{max_abs, max_rel, mean_abs}` helper that *quantifies* drift. This is the
discipline that prevents chasing kernel performance with no measure of quality:
no faster path is accepted on speed alone, only against a quantified diff from the
oracle. The oracle shares the least code with the thing under test, so it is a
trustworthy ground truth in a way "compare against the chunked path" (itself an
optimization) is not.

## Component 3 — `PackedStreamingCache` (`src/bmx/cache/packed_streaming.py`)

Sibling of `StreamingQuantizedCache`. The layer stores only **packed codes +
metadata** for the committed prefix (the bpe footprint, resident) plus the fp16
recent window. It **never** builds `_q_prefix_k/v` dense or the `k_hat`/`v_hat`
slab.

**Packed layer state inventory (per layer):**
- per-block packed K codes + per-block packed V codes (the resident bpe
  footprint) for the committed prefix;
- the fp16 recent window (un-flushed tail);
- `_committed_S_q`, `_k_pre`/`_k_pre_offset`, the growing `_rope_cos`/`_rope_sin`
  table — as today;
- **the frozen subspace `_frozen_svd = (Us, V)`** fitted at first flush
  (`streaming.py:149-151`). This is *required at read time* — each block's
  dequant uses the per-block projection `Us = M_block @ V_frozen` — so it is part
  of the packed state, not transient. (Called out because the bit-for-bit gate
  fails silently if it drifts from the current path.)

**Decode-time flush (the new write path).** Prefill uses the existing path, so
the decode flush is the new code. `PackedStreamingLayer.update()`, when a block
is ready (same schedule as below), does the **quantize-to-packed** step
(`quantize_packed` from the codec split) instead of the current
quantize-to-dequant: (a) quantize the ready block to packed codes, (b) append to
the packed prefix, (c) keep the fp16 tail. No dense prefix is ever built.

**Attention integration (the one real fork in Phase 2).** Stock HF attention
demands a dense K/V tensor from the cache, so this path must route attention
through `chunked_dequant_attention`:

| Option | Mechanism | Verdict |
|---|---|---|
| **A. Attention-function registry (chosen)** | Register a custom fn via `ALL_ATTENTION_FUNCTIONS.register("chunked_dequant", fn)` (transformers 5.x `AttentionInterface`, **verified present in the pinned version**) and set `model.config._attn_implementation = "chunked_dequant"`. The official extension point — survives transformers patches better than a `forward` override. | Pure PyTorch, AMD-testable; slower than SDPA but correctness + memory are what we validate. |
| B. `forward` monkeypatch | Brittle across 5.x point releases. | Rejected. |
| C. Custom Cache returning a lazy view | Keep stock SDPA, return a dense tensor — rematerializes, defeats the purpose. | Rejected. |

**Integration subtlety to resolve in the plan:** the registered fn's signature is
`(module, query, key, value, attention_mask, scaling, dropout, **kwargs)`, where
`key`/`value` are the dense tensors a normal cache returns from `update()`. The
packed path has no dense `key`/`value` — so `PackedStreamingCache.update()` must
pass the packed state through (e.g. via `kwargs` carrying a handle to the layer,
or by returning a sentinel that the registered fn recognizes and then reads the
packed state off the cache). The plan must pick one; the parity test pins it.

**Shared block schedule (anti-drift).** The committed-block schedule
`new_S_q = ((S - W) // g) * g` (`streaming.py:223`) must be **identical** between
`StreamingQuantizedCache` and `PackedStreamingCache` or the bit-for-bit gate
fails. Extract it as a shared helper `compute_flush_schedule(S, W, g) -> int` in
`streaming.py` and have both layers call it — do not duplicate the expression.
(This makes `streaming.py` a small *edit*, not purely read-only.)

**Scope guard / YAGNI:** Phase 2 targets the **decode-path** read (the
128k-sweep blocker). Prefill uses the existing path. No paging, CPU offload,
NVMe tiers, or block-size autotuning.

## Component 4 — census instrument (`experiments/k3_kernel_census.py`)

Thin tyro CLI (repo convention). Measures the actual memory split per arm per
context length so the ledger is validated and the Phase-2/3 decision is made from
data:

- Wraps a single generation with `torch.cuda.reset_peak_memory_stats()` /
  `max_memory_allocated()` around prefill and decode separately.
- Reports per arm × context length, **distinguishing total from incremental** so
  the ledger's term-by-term decomposition is directly checkable (note
  `max_memory_allocated()` is cumulative — it includes already-resident weights):
  - `resident_after_prefill` — total allocated after prefill (incl. weights);
  - `peak_decode` — total peak during decode (the OOM number);
  - `peak_decode_incremental = peak_decode − resident_after_prefill` — the
    transient / working-set delta.
  The ledger's `resident + transient` then validates against
  `resident_after_prefill + peak_decode_incremental` directly, rather than a
  term-by-term model compared to a single cumulative number. Run for both
  `StreamingQuantizedCache` (today) and `PackedStreamingCache` (Phase 2).
- Writes parquet to `results/k3_kernel_census/<run-id>/` with config + env + SHA
  (`artifacts.py`). Figures read parquet, never refit.
- **This instrument confirms or refutes the code-read** that resident
  double-copy dominates. If the census shows transient actually dominates, the
  framing flips and we revisit — the honest-negative escape hatch.

## Phase 3 — Triton fused kernel (gated, NOT built in this spec)

Decision rule:

> Build Triton **only if** (a) the Phase-1 ledger predicts *or* the Phase-2
> census measures that chunked-PyTorch's `peak_decode` at 128k still exceeds
> 94.5 GB; **or** (b) a deployment claim needs the literal process-RSS / speed
> win (the "5×") that PyTorch's allocator + lack of true fusion can't deliver.

If built, the Triton kernel is validated against **`naive_dense_attention` (the
oracle)** via `attention_diff` — NOT against the chunked path (which is itself an
optimization sharing too much code to be a trustworthy ground truth). The
algorithm is already proven by Phase 2 (FA1→FA2→FA3 keep the same online-softmax +
tiling structure, changing only work partitioning — vault), so Triton only
changes *where* it runs; the oracle is the constant quality yardstick across all
three phases. VM-only, separate spec when triggered.

## File inventory

| File | New/Edit | Purpose |
|---|---|---|
| `src/bmx/cache/codecs.py` | **edit** | Split each arm into `quantize_packed`/`dequant_packed`; `quantize_cache` becomes their composition (current path + tests unchanged). |
| `src/bmx/quant/rtn.py` | **edit** | Expose a packed (indices, scales) form of `rtn_quantize` for the split (current dequant-returning fn kept). |
| `src/bmx/cache/streaming.py` | **edit (small)** | Extract `compute_flush_schedule(S,W,g)`; reuse codec dequant math. No behavior change to the current path. |
| `src/bmx/bench/kv_memory.py` | new | Byte-ledger: `KVMemCase`, `predict_peak()`. Pure Python, AMD-local. |
| `src/bmx/cache/chunked_attention.py` | new | `chunked_dequant_attention()` — online softmax over per-block dequant. |
| `src/bmx/cache/packed_streaming.py` | new | `PackedStreamingCache`/`Layer` — packed codes + frozen subspace + fp16 window only; registry-based attention routing. |
| `experiments/k3_kernel_census.py` | new | tyro CLI; per-arm resident/peak/incremental → parquet. VM. |
| `tests/test_kv_memory.py` | new | Ledger arithmetic vs hand-computed Llama-3.1 numbers. |
| `tests/test_codec_split.py` | new | Per arm: `dequant_packed(quantize_packed(M))` == current `quantize_cache(M)` `M_hat` exactly. |
| `tests/test_chunked_attention.py` | new | **Bit-for-bit** vs `StreamingQuantizedCache`; fp64; tiny `factories.py` model. |

## Testing strategy (all local on AMD — discipline gate before any VM time)

1. **Ledger:** `dense_stream` prediction reproduces the hand-derived Llama-3.1 KV
   numbers (16 GB/copy @128k, ~2 copies) and the observed ~99–100 GB OOM —
   validates the model against a known point before trusting `chunked`.
1b. **Codec split:** per arm, `dequant_packed(quantize_packed(M))` equals the
   current `quantize_cache(M)` `M_hat` **exactly** (fp64). This is the foundation
   the chunked path's correctness rests on — test it first.
2. **Chunked attention correctness:** the load-bearing test, fp64, every arm
   (at least fp16, k2b, turboquant_mse). **Primary gate: bit-for-bit** vs the
   current path, which is achievable *iff* the chunked path uses the identical
   committed-block schedule + identical per-block dequant (online softmax is
   exact). Only if bit-for-bit proves structurally impossible do we fall back to
   a tight numerical tolerance, and the test must document the specific reason
   (see risks). Do not start from the tolerance fallback — earn it.
3. **`PackedStreamingCache` parity:** end-to-end generation on a tiny model
   matches `StreamingQuantizedCache` token-for-token (same philosophy as the
   existing identity-invariant cache test).
4. **Full suite green** (`uv run ruff format .` → `uv run ruff check .` →
   `uv run pytest -q`, expect 220 passed / 1 xfailed + new) before proposing any
   commit. **Never commit without the user's approval** (CLAUDE.md hard rule).

## Risks & mitigations

- *Eager-attention monkeypatch brittle across transformers 5.x* → isolate the
  patch in `packed_streaming.py`'s `attach()`, pin with a parity test;
  `resolve_decoder_layers` already handles layer-structure variation.
- *Bit-for-bit may not hold* if block boundaries differ between paths → the
  chunked path must use the **identical** committed-block schedule as
  `StreamingQuantizedCache` (`new_S_q` logic); test enforces it. If exactness is
  impossible, fall back to a tight numerical tolerance and document why (the
  results doc already notes quant-noise-level drift is a property of block-flush
  timing, not a bug).
- *Ledger mispredicts via allocator fragmentation* → census measures real peak;
  ledger is validated against it, not trusted blind. Fragmentation called out as
  an unmodeled term with a measured fudge factor.
- *Chunked PyTorch too slow to run 128k locally* → correctness on tiny models
  locally; the 128k peak measurement is VM (expected). We validate the *model*,
  not the wall-clock, locally.

## Explicitly out of scope (YAGNI)

No Triton (gated to Phase 3), no paging / CPU-offload / NVMe tiers, no
prefill-path rewrite, no multi-model port (Gemma4 / Qwen3.6 sliding-window is
separately scoped per the 2026-06-21 results doc), no block-size autotuning.

## Execution model

Subagent-driven development (independent tasks dispatched to subagents in this
session), with a `simplify` skill pass at the end once the components are
finished. No commit for this spec.
