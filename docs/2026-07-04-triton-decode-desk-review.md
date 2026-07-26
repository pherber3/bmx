# Triton decode path — desk review: why end-to-end generation can lose to the PyTorch cache

**Date:** 2026-07-04. **Scope:** static review only — the GH200 is occupied by the live
LongBench run and every number measured beside it is contaminated (see the Wave-3 plan,
Task 5). This doc explains the *mechanism*; magnitudes come from the isolated profile
(`scripts/profile_decode_ab.py`) after the VM frees up.

## Correcting the record first

Two different claims are floating around and they are about different things:

1. "The Triton kernel is ~1.6× slower than PyTorch" — that was the **old per-block
   kernel** (one launch per KV block per head), deleted in the Phase-3 rewrite. The
   fused split-KV kernel that replaced it measured **44–2624× faster than
   `chunked_dequant_attention`** at the decode microbench
   (`docs/2026-06-24-triton-decode-results.md`).
2. "The Triton generation path for the streaming cache is slower than the PyTorch
   version" — this is about **end-to-end generation**:
   `PackedStreamingCache` (fused kernel) vs `StreamingQuantizedCache` (whose committed
   prefix is stored **dequantized fp16** and attends via one dense SDPA call). Both
   claims can be true at once, because the microbench and the generation path measure
   different code.

Two structural reasons the microbench does not predict generation:

- **The microbench never measures the tail merge.** `experiments/k3_triton_decode.py`
  passes `k_tail=None` (deliberately, to time the kernel in isolation). Real generation
  ALWAYS has a tail — see F1.
- **The microbench baseline is `chunked_dequant_attention`** (re-dequantizes every page
  every step; 12–2000 ms at these contexts). The generation-path baseline is dense
  flash SDPA over an fp16-resident slab (~0.05–0.5 ms). Beating chunked by 2624× and
  losing to dense SDPA end-to-end are perfectly consistent.

## Findings (file:line, ranked by expected cost)

### F0 — the observed "12× slower" A/B never ran the Triton kernel (arm-coverage gap + silent-slow fallback) — RESOLVES the reported contradiction

The one in-generation A/B on record (2026-07-03-ish, contaminated, other session) ran
`use_packed=True` with the **`turboquant_mse` arm**. `spec_pair("turboquant_mse")`
returns K=V=`turboquant_mse` (`src/bmx/cache/recipes.py:63-65`), which fails BOTH fused
predicates in `PackedStreamingLayer.attend` (`packed_streaming.py:551-559,614-622` —
fused packed needs `rtn_token`/`rtn_token`; fused k2b needs
`lowrank_rtn_channel`/`turboquant_mse_perhead`). Every decode step therefore fell
through to `chunked_dequant_attention`, which re-dequantizes **every committed page,
every layer, every token** — the O(S)/step path the fused kernel exists to replace.

Magnitude check: the microbench's chunked cost at 8k is ~50–130 ms per layer-call
(`docs/2026-06-24-triton-decode-results.md`); × 32 layers ≈ 1.6–4.1 s/token. The
contaminated observation was 5.16 s/token (with ~⅓-GPU contention on top). The numbers
match; no kernel-wrapper mystery is needed to explain that A/B.

Two corollaries:

- The session's leading hypothesis — "`_PagedStacks` rebuilds stacks every step" — is
  wrong twice over: `_PagedStacks` has been incremental since the I3 fix (117ab1b),
  and no stacks were ever built because no fused path fired.
- The fallback is **correct but silently ~30–70× slower** at decode on CUDA. As of this
  commit, `attend` emits a one-time `warnings.warn` when a CUDA decode lands on the
  chunked fallback, naming the uncovered arm pair — so no future benchmark can
  attribute chunked's cost to the Triton path again. F1–F3 below still stand for the
  arms the fused kernels DO cover (`rtn_token`, `k2b_ph`); they are the *next* layer of
  overhead once the right arm is used, and the isolated profile should measure them.

### F1 — the fp16 tail merge runs in PyTorch on EVERY decode step; the GPU merge kernel is dead code during generation

`_finalize_decode` (`src/bmx/cache/triton_dequant_attention.py:342`) takes the fast
GPU-merge path only when `k_tail is None or k_tail.shape[1] == 0`. Under the streaming
schedule the tail is **never empty**: `compute_flush_schedule` keeps
`tail_len ∈ [recent_window, recent_window + PAGE)` = [32, 160) with the defaults. So
100% of real decode steps take the fallback that:

- builds `3 × num_splits` Python tensor views (96 at the tuned `num_splits=32`)
  (`:357-359`);
- upcasts and `repeat_interleave`s the tail **allocating (n_q_heads, tail_len, d) fp32
  copies each step** (`:362-363`);
- runs ~12 more small CUDA ops (two einsums, amax/exp/sum, then `_merge_partials`'
  three `torch.stack`s over 33 tensors + 6 reduction ops, `:153-170`).

Per layer per token that is ~150 Python dispatches + ~15 extra kernel launches. Across
32 layers: **thousands of small ops per generated token**, vs ~5 per layer for the
dense-SDPA cache. At ~5–15 µs per dispatched op this alone plausibly costs
O(5–15 ms)/token — larger than the fused kernel's own 0.07–6 ms. This is the prime
suspect and the first thing the isolated profile should confirm.

### F2 — k2b constants are re-uploaded to the GPU every call

`fused_decode_attention_k2b` (`:950,955`) calls
`gaussian_codebook(vbits).to(q.device)` and `_hadamard_signs(d, v_seed).to(q.device)`
per invocation. Both helpers are `lru_cache`d **on CPU** (`codecs.py:106,197`), so the
compute is cached but the **host→device copy + allocation happen per layer per step**
(64 small H2D transfers per token at 32 layers). `_hadamard_matrix` (`:49`) already
shows the correct pattern — cached per `(d, device, dtype)`.

### F3 — `uniform_blk` rebuilds a Python set over all committed blocks every step

`PackedStreamingLayer.attend` (`src/bmx/cache/packed_streaming.py:549`):
`len({e - s for _, s, e in blocks}) == 1` is O(n_blocks) pure-Python work per layer per
step, growing with context (at 128k: 1024 blocks × 32 layers = 32k tuple unpacks per
token). Should be maintained incrementally in `update()` where pages are appended.

### F4 — per-step launch/dispatch floor

Even on the fused path, each layer-step pays: Triton dispatch (autotune key lookup +
launcher, ~10–50 µs — noticeably heavier than a raw CUDA launch), the partials
`torch.empty`s (`:959-967`), and (F1) the merge fallback. The deleted CUDA-graph decode
path was the standard cure for exactly this launch-overhead-bound regime; it is
recoverable — rationale and exact `git checkout 93751eb -- …` steps in
`docs/2026-06-24-decode-path-debloat-removal.md`. Gated: only worth it for a deployment
speed claim.

### F5 — minor / verify-on-VM

- `rope_cos.to(q.device, torch.float16).contiguous()` (`:969-974`) — should be a no-op
  if the tables are stored fp16-contiguous on device; verify, don't assume.
- `q.squeeze(1).view(...).contiguous()` (`:958`) — no-op for HF's layout; verify.

## What a fix buys (and what it can't)

Per the standing prediction ([[triton-decode-win-prediction]]): decode wall-clock is
KV-fraction-bounded. At LongBench contexts (5–30k) attention is a small slice of the
~70 ms/token step (the MLP/linears dominate), so even a zero-overhead packed path
roughly TIES the dense cache on speed. **The packed path's honest claim is resident
memory (the batched 128k sweeps that OOM dense) at speed parity — not a speedup
headline.** The fixes below aim at parity, i.e. removing the *regression*, not
manufacturing a win.

Ranked fix plan (post-VM-confirmation, in order of ROI):

1. **Fold the tail into the split-KV merge.** Either (a) treat the fp16 tail as one
   extra virtual split whose partial is computed by a tiny third kernel (or even by the
   main kernel's split 0 with a masked extra tile), then ALWAYS take the existing GPU
   `_fused_merge_kernel` path; or (b) minimally, compute the tail partial with the
   current torch ops but append it into the partials tensors and call the GPU merge —
   killing the 96-view + 3-stack Python merge. (a) removes ~all of F1; (b) removes ~⅔.
2. **Device-cache the k2b constants** — `lru_cache` keyed on `(bits/seed, device)`
   exactly like `_hadamard_matrix`. Trivial, removes F2.
3. **Incremental `uniform_blk`/`blk`** maintained at flush time in `update()`. Trivial,
   removes F3.
4. **CUDA-graph the whole decode step** (recover the deleted graphable path) — only if
   a deployment latency claim becomes a paper/demo requirement after 1–3 land.

## Measurement protocol (Task 5 companion, GH200, AFTER pid 10858 completes)

`scripts/profile_decode_ab.py` (this commit):

- **Mode A (end-to-end):** real model (`--model-name`), same prompt, three caches —
  `DynamicCache` (dense fp16), `StreamingQuantizedCache`, `PackedStreamingCache` — at
  several contexts. Greedy-token parity asserted between the two quantized caches
  before any timing is trusted (oracle-gated per [[oracle-gated-perf-work]]).
  Per-token decode ms = (t[1+N tokens] − t[1 token]) / N.
- **Mode B (attribution):** `torch.profiler` over N packed decode steps; top ops by
  CUDA time, self-CPU time, and **call count** (the launch-count signature of F1/F2 is
  unmistakable: repeat_interleave/stack/einsum × layers × steps).

Success criterion for the fixes: packed per-token decode within ~10% of
`StreamingQuantizedCache` at 8k, and the op-count per step collapsing from thousands
to ~hundreds.
