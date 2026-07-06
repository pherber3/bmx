# Wave 5 — Realize the resident-memory win (and diagnose, not guess, the packed latency)

**Status:** ACTIVE. Subagent-driven; per-task auto-commit pre-authorized (user 2026-07-04,
carried forward). NO VM CONTACT by any agent — all work local; every GPU step is user-launched.
Baseline at start: 333 passed / 8 skipped / 1 xfailed @ 84bb1f8 (pushed).

## Why (the T4 mechanism, fully derived from code — supersedes the doc's "C3 reversed" framing)

T4 measured packed at 4.17 GiB marginal per 32k sequence — worse than dense fp16's 4.0. That is
NOT "compression isn't realizable"; it is two implementation gaps:

1. **Containers were never bit-packed.** V codes are int16 indices (`codecs.py:475,491`) — 16 bits
   per 2-bit code, i.e. ZERO in-memory compression for V (2.15 GiB/seq, same as dense V). K
   residual is int8 for 3-bit codes (2× only, 1.07 GiB/seq). The honest-bpe LEDGER is correct as
   an accounting statement; the runtime tensors just don't use that representation.
2. **Double-buffering.** `_PagedStacks` copies every field the block list already holds; after the
   first decode step a sequence carries BOTH (~3.6 GiB blocks + ~3.3 GiB stacks ≈ 6.9 GiB/seq,
   worse than dense).

Arithmetic (32k seq, Llama-3.1-8B, 32L, h_kv=8, d=128, C=1024; k2b_ph = K lowrank 3b r16 +
V turboquant_perhead 2b). T4's 4.17 marginal matches the block-list line because
`batch_oom_sweep.py` records marks post-prefill, pre-first-decode (stacks not yet built):

| state | GiB/seq @32k | vs dense 4.30 |
|---|---|---|
| today, post-prefill (blocks only) | ~3.6 | 1.2× |
| today, steady-state (blocks + stacks) | ~6.9 | 0.6× (WORSE) |
| **W5-1** single storage | ~3.6 | 1.2× |
| **W5-1 + W5-2** (V 2b packed) | ~1.7 | 2.5× |
| **W5-1 + W5-2 + W5-3** (K 3b packed) | ~1.1 | **4.0× ≈ honest bpe** |

## Task 1 (W5-1): single-storage pages — stacks become THE storage, blocks become views

`src/bmx/cache/packed_streaming.py`. After `_PagedStacks` absorbs page i, free the block-list
tensors and re-point block i's dict entries at VIEWS into the stack row (`buf[field][i]`) — same
values, same shapes per block, so `chunked_dequant_attention` (prefill + fallback + parity tests)
consumes them unchanged. Constraints:

- Gate on the fused-capable arm pairs only (stacks exist only for rtn/rtn and k2b_ph); other arms
  keep today's behavior exactly.
- `_PagedStacks._grow` reallocates buffers → the cache must RE-POINT all block views after a grow
  (make view-refresh a helper owned by the cache; `_PagedStacks` returns which rows are live).
- Prefill-time attends happen before stacks exist — unchanged. The memory drop lands at the first
  decode attend (v1; eager-stack-at-flush is a possible v2, only if census says prefill peak needs it).
- CPU tests: (a) chunked attend over block-views == chunked attend over original blocks
  (torch.equal) before AND after a forced `_grow`; (b) memory proxy: after stacking, the block
  dicts' tensors share storage with the stack buffers (`t.untyped_storage().data_ptr()` equality
  or `t._base is buf`), i.e. no duplicate allocation survives; (c) existing packed-vs-streaming
  CPU parity tests unchanged.
- Commit: `perf(packed): single-storage pages — block dicts become views into _PagedStacks after stacking (kills the double-buffer; T4 mechanism 2)`

## Task 2 (W5-2): bit-pack V indices in the stacks (2-bit → 4 codes/int8 byte), flag-gated

The 8× container waste. Scope: the STACKED layout + kernel read only — `quantize_packed`/
`dequant_packed`/the block dicts keep int16 (parity: the reference/chunked path stays
bit-identical; transient pre-stack pages are negligible after W5-1).

- `build_kv_stacked_k2b(..., pack_v: bool = False)`: when True, v_idx is stored
  `(max_blocks, blk, C//4) uint8`, 4 consecutive-channel 2-bit codes per byte
  (code k at channel c lives in byte c//4, bits 2*(c%4)). Pure-torch pack + unpack reference
  functions with a CPU roundtrip-identity test vs the int16 layout.
- `_fused_decode_k2b_kernel` gains `V_PACKED: tl.constexpr`; unpack in-kernel:
  `byte = load(v_idx_ptr + row*C//4 + c//4); idx = (byte >> (2*(c%4))) & 3` — index math only,
  the surrounding gather/dequant is untouched. This IS a @triton.jit edit — the ONLY sanctioned
  one this wave, and it ships **default OFF** (`pack_v=False` everywhere) so current behavior is
  byte-identical until the GH200 oracle passes with the flag on.
- Only bits=2 (and 4) need support; assert loudly otherwise.
- GH200 acceptance (user-launched, listed in report): k2b oracle tests + generate-parity with
  pack_v=True monkeypatched/parametrized, then flip the default in a follow-up commit ON THE VM
  EVIDENCE, not before.
- Commit: `feat(kernel): flag-gated 2-bit packing for stacked V indices (4 codes/byte; T4 mechanism 1, V half — default OFF until GH200 oracle)`

## Task 3 (W5-3, GATED on W5-2's measured win): pack the K residual (3-bit)

3-bit is not byte-aligned (candidate: 8 codes per 3 bytes, or 10 per int32). Decide AFTER the VM
confirms W5-1+W5-2 land ~1.7 GiB/seq and the kernel unpack cost is negligible. Do not start until
then — the win is 0.6 GiB/seq vs W5-2's 1.9.

## Task 4 (W5-4): T3b latency — evidence first, fixes second

The doc's "_PagedStacks per-step rebuild dominates" diagnosis names a mechanism that does not
exist post-I3 (incremental append, CPU-equality-tested). Do NOT implement fixes for it. Required
evidence before any latency work (user-launched on idle GPU):
1. `profile_decode_ab.py` raw stdout incl. Mode B (`--profile-steps 20`) top-25-by-count and
   by-CUDA-time tables at 64k+;
2. the same after W5-1/W5-2 (they change the memory traffic);
3. the dense-vs-streaming inversion (34.5 flat < 47.7–60.4) must reproduce under Mode A's
   warmup discipline before it is treated as real — it is physically suspect (streaming does a
   strict superset of dense's per-step work).
Pre-registered candidates if packed slope reproduces: rope-table `.to().contiguous()` per call;
autotune dispatch; allocator churn from the 2.15 GiB int16 v_idx traffic (W5-2 shrinks it 8×);
per-flush `build_kv_stacked_*` H2D. Findings go in a results doc, fixes become W5-5+.

## Sequencing

W5-1 → W5-2 (same files, serial) → user's VM window: GH200 suite + profile A/B + oracle w/
pack_v=True + batch_oom_sweep rerun (expect ~1.7 GiB/seq marginal, max co-resident ≈ 3× dense's
16 minus weights headroom) → W5-3/W5-4 decisions on that evidence. The truncated LongBench-E
parity rerun does NOT wait for this wave (it runs on the streaming path).
