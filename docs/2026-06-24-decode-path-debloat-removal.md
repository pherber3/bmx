# Decode-path debloat — what was removed, how to recover it

Date: 2026-06-24. Branch: `feat/triton-decode-kernel`.
Removal commit: **`7b07552`** ("post-review debloat …", −1641/+82).
Last commit that still CONTAINS all the removed code: **`93751eb`** (the parent of `7b07552`).

A pre-merge external audit confirmed the deployment decode path
(`fused_decode_attention_packed`, `fused_decode_attention_k2b`, the per-block
`triton_decode_attention` fallback, `_PagedStacks`) is correct and complete. Three
chunks of code were never on that path and were deleted to cut bloat. **A future
reader looking at the clean tree has no way to know these existed — this file is the
pointer.** Everything is recoverable from git history (the removal only deleted; it
did not rewrite history).

## What was removed (and why it was dead)

1. **Dense `fused_decode_attention`** (`src/bmx/cache/triton_dequant_attention.py`,
   ~250 lines). A fused decode kernel that consumed DENSE fp16 KV (no in-kernel
   dequant). Benchmark/reference only — it had **zero callers** in `src/`, `tests/`,
   or `experiments/`, and no test. The packed variant
   (`fused_decode_attention_packed`) superseded it for every real use (it dequants
   int8 codes in-kernel, preserving compression).

2. **The graphable CUDA-graph decode path** (~500 src + 635 test lines):
   `build_kv_stacked`, `triton_decode_attention_graphable`,
   `_graphable_decode_kernel`, `_graphable_reduce`, and
   `tests/test_triton_cudagraph.py`. A working CUDA-graph capture-safety
   implementation (device-tensor seq_len so graph replay uses the live length). It
   was **never wired into `PackedStreamingLayer.attend`** — only its own CUDA-gated
   test exercised it. It is also **RTN-only and built for the pre-paging block
   layout**; the deployment cache now uses the uniform PAGE=128 paged layout + the
   k2b recipe, so real CUDA-graph serving would be re-implemented against
   `_PagedStacks`/the k2b path anyway — this code would not be reused as-is.

3. **`hadamard_kernel_ref.py` + `tests/test_hadamard_kernel_ref.py`** (~386 lines).
   A CPU "port-faithful" butterfly-FWHT reference that de-risked a planned Triton
   *in-kernel FWHT* for the turboquant-V unrotate. That in-kernel FWHT was **never
   built** — we shipped the per-head Hadamard codec instead (`turboquant_mse_perhead`
   + the cached `_hadamard_matrix` `(d,d)` matmul in the k2b kernel). The de-risk
   artifact's purpose was fully discharged, so it and its test were obsolete.

## When you would want to recover something

- **Recover the graphable path** if you do a real serving integration that wants
  CUDA-graph capture for decode latency — BUT treat it as a *reference*, not
  drop-in: it must be re-targeted to the paged `_PagedStacks` layout and the k2b
  arm (the original is RTN-only + pre-paging). Often a rewrite using the old code as
  a spec is cleaner than restoring it.
- **Recover `hadamard_kernel_ref`** only if someone revisits an *in-kernel* FWHT
  for V (e.g. to shave the per-head `(d,d)` matmul). The current per-head design
  makes this unnecessary; don't restore it without that specific goal.
- **The dense `fused_decode_attention`** is unlikely to be worth recovering — the
  packed kernel is strictly better. Restore only if you need a dense-KV A/B
  microbenchmark baseline.

## How to recover

The code lives intact in `93751eb` (and every earlier commit on this branch).
The deleted files come back whole; the in-file deletions (dense kernel, graphable
funcs inside `triton_dequant_attention.py`) come back as a diff to apply.

```bash
# See exactly what 7b07552 removed (the full deleted code):
git show 7b07552            # the deletion diff
git show 93751eb:src/bmx/cache/hadamard_kernel_ref.py        # a whole deleted file
git show 93751eb:tests/test_triton_cudagraph.py

# Restore a whole deleted FILE as-is:
git checkout 93751eb -- src/bmx/cache/hadamard_kernel_ref.py \
                        tests/test_hadamard_kernel_ref.py \
                        tests/test_triton_cudagraph.py

# The dense kernel + graphable funcs were deleted from WITHIN
# triton_dequant_attention.py — extract them from the parent and re-integrate by hand
# (do NOT `git checkout 93751eb -- triton_dequant_attention.py`, that would also
# revert the audit fixes and _PagedStacks that came after):
git show 93751eb:src/bmx/cache/triton_dequant_attention.py > /tmp/old_kernel.py
#   then copy out fused_decode_attention / build_kv_stacked /
#   triton_decode_attention_graphable / _graphable_decode_kernel / _graphable_reduce.
```

If `93751eb` has been squashed/lost by a later history rewrite, search any commit on
the branch before `7b07552` — `git log --oneline --all -- src/bmx/cache/hadamard_kernel_ref.py`
finds commits that still had a deleted file.

## Addendum (2026-07-01)

The original debloat removed the dense path's launcher (`fused_decode_attention`) and
builder (`build_kv_stacked`) but missed the kernel body itself. `_fused_decode_kernel`
(~165 lines) was deleted in the follow-up cleanup. Recovery: same as above — parent
commit `93751eb` contains the full dense path including this kernel.

Note for anyone recovering: code taken from `93751eb` predates the 2026-07-01 cleanup
(`query_abs_start` → `is_prefill`, module extractions, and the per-block decode path
removal) — reconcile signatures against current `chunked_attention.py` /
`packed_streaming.py` before wiring it back in.

2026-07-01 Wave 2: the per-block launch path (_online_softmax_block_kernel,
_online_block_kernel_launch, _partition_blocks, triton_decode_attention) was also
removed — the fused kernels + chunked fallback cover all configs. Recover from the
parent of the Wave-2 removal commit if ever needed.
