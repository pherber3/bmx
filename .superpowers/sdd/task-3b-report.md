# Task 3b Report — Split-KV decode parallelism + autotune (stage 3b)

Status: **DONE_WITH_CONCERNS** (kernel unverified on hardware — VM verification pending Task 6 batch)

---

## What was added

`src/bmx/cache/triton_dequant_attention.py` gains:
- `num_splits: int = 1` keyword-only arg on `triton_decode_attention` (back-compatible default)
- `_partition_blocks(k_blocks, v_blocks, num_splits)` — ceiling-division partition of block list
- `_merge_partials(partial_accs, partial_ms, partial_lses)` — PyTorch reduction (base-e)
- `@triton.autotune` + `@triton.jit(do_not_specialize=["n_blocks_in_split"])` on the kernel
- `_AUTOTUNE_CONFIGS` (4 combos: BLK∈{64,128} × num_warps∈{4,8}, num_stages=2)

`tests/test_triton_decode_rtn.py` gains:
- `test_triton_split_kv_matches_oracle` — parametrized `num_splits ∈ {1,2,4,8}` vs oracle (4 CUDA-gated skips locally)
- `test_triton_split_kv_num_splits_1_bit_identical_to_3a` — determinism assertion on the serial path (1 CUDA-gated skip locally)

---

## Reduction kernel body (`_merge_partials`)

```python
accs  = torch.stack([a.float()     for a     in partial_accs],  dim=0)  # (S, H, 1, d)
ms    = torch.stack([m.float()     for m     in partial_ms],    dim=0)  # (S, H, 1, 1)
lses  = torch.stack([lse_t.float() for lse_t in partial_lses], dim=0)  # (S, H, 1, 1)

m_global   = ms.amax(dim=0, keepdim=True)          # (1, H, 1, 1)
scales     = torch.exp(ms - m_global)              # (S, H, 1, 1) — base-e
l_merged   = (lses * scales).sum(dim=0)            # (H, 1, 1)
acc_merged = (accs * scales).sum(dim=0)            # (H, 1, d)
return (acc_merged / l_merged).to(torch.float16)
```

This is the standard online-softmax combine applied across the split axis.

---

## Split partition logic

`_partition_blocks` uses ceiling-division to assign `ceil(n / num_splits)` blocks per split.
All packed-block processing is identical to the 3a serial loop (`_dequant_block` + `_online_block_kernel_launch`).
The tail block (`k_tail`/`v_tail`) is always assigned to split 0 to avoid double-counting.
Empty splits (when `n_blocks < num_splits`) are skipped rather than contributing degenerate `lse=0` partials.
When only one non-empty split results (tail-only or single block), the merge is short-circuited to direct `acc / lse`.

---

## Autotune config + do_not_specialize

```python
from triton import Config as _TritonConfig

_AUTOTUNE_CONFIGS = [
    _TritonConfig({"BLK": 64},  num_warps=4, num_stages=2),
    _TritonConfig({"BLK": 64},  num_warps=8, num_stages=2),
    _TritonConfig({"BLK": 128}, num_warps=4, num_stages=2),
    _TritonConfig({"BLK": 128}, num_warps=8, num_stages=2),
]

@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["d", "n_q_groups"])
@triton.jit(do_not_specialize=["n_blocks_in_split"])
def _online_softmax_block_kernel(..., n_blocks_in_split, d: tl.constexpr, n_q_groups: tl.constexpr, BLK: tl.constexpr):
```

- `key=["d", "n_q_groups"]`: autotuner specializes on head dim + GQA groups — hardware-characteristic, stable across decode steps.
- `do_not_specialize=["n_blocks_in_split"]`: block count changes every decode step as the KV cache grows; specializing would recompile per step (the AWS 10× TTFT regression). This arg is NOT `tl.constexpr` — a param cannot be both.
- `BLK` is now a `tl.constexpr` injected by autotune (not passed by the caller).

---

## Merge formula: why it reduces to identity at num_splits=1

At `num_splits=1` there is exactly one partial `(acc_0, m_0, lse_0)`:
```
m_global = m_0
scales   = exp(m_0 - m_0) = 1
l_merged = lse_0 * 1 = lse_0
out      = acc_0 * 1 / lse_0 = acc_0 / lse_0
```
This is bit-identical to the 3a final line `return acc / lse.to(q.dtype)`.

The `num_splits=1` code path short-circuits the merge entirely:
```python
if num_splits == 1:
    acc, _m, lse = _run_split(k_blocks, v_blocks, with_tail=True)
    return acc / lse.to(q.dtype)   # exactly 3a
```
So the bit-identity holds by construction (same call, same kernel, same division).

At `num_splits > 1` the combine is online_softmax_update applied across the split axis,
giving the same result as processing all blocks serially (the associativity of the online-softmax combine).

BASE-E throughout: the 3a kernel stores `lse` as the raw unnormalized sum of `exp(score - m)` (base-e).
The merge uses `torch.exp` (base-e). Mixing base-2 LSE with base-e merge would be a silent trap — avoided.

---

## Test (`tests/test_triton_decode_rtn.py`)

| Test | Guard | n_splits | Local |
|---|---|---|---|
| `test_triton_module_imports_with_available_flag` | none | — | PASSED |
| `test_require_triton_raises_without_cuda` | none | — | PASSED |
| `test_triton_rtn_decode_matches_oracle` | `@cuda` | — | SKIPPED |
| `test_triton_rtn_decode_matches_oracle_prerope` | `@cuda` | — | SKIPPED |
| `test_triton_decode_asserts_n_q_eq_1` | `@cuda` | — | SKIPPED |
| `test_triton_split_kv_matches_oracle[1]` | `@cuda` | 1 | SKIPPED |
| `test_triton_split_kv_matches_oracle[2]` | `@cuda` | 2 | SKIPPED |
| `test_triton_split_kv_matches_oracle[4]` | `@cuda` | 4 | SKIPPED |
| `test_triton_split_kv_matches_oracle[8]` | `@cuda` | 8 | SKIPPED |
| `test_triton_split_kv_num_splits_1_bit_identical_to_3a` | `@cuda` | 1 | SKIPPED |

Oracle test: 16 blocks (n_blocks=16 divides evenly for all four split counts), `n_q_heads=8`, `n_q_groups=4`, `d=64`, `blk=64`.
Tolerance: `max_abs < 1e-2`. Expect near fp16 rounding (~2e-4). Do NOT loosen.

---

## Local evidence

```
uv run ruff format . && uv run ruff check .   →  All checks passed!
uv run pytest -q                              →  259 passed, 8 skipped, 1 xfailed
TRITON_AVAILABLE = False  (AMD/no-CUDA dev box, confirmed)
_require_triton() raises RuntimeError("...TRITON_AVAILABLE...")  ✓
```

---

## Hardware-verify risks (prioritized)

1. **LSE merge correctness at num_splits > 1** (#1 risk — the whole point of 3b).
   The merge formula is correct by construction (reduces to identity at 1 split, online-softmax combine at N).
   But it has NEVER run on CUDA. A wrong rescaling factor would produce O(1) error caught by
   `test_triton_split_kv_matches_oracle[2/4/8]`.  VM MUST run this parametrized test.

2. **Autotune config validity**: The 4 configs assume BLK ∈ {64, 128} are valid for the tl.dot constraint (BLK ≥ 16 ✓).
   The autotuner will select the best; if all 4 fail (wrong num_warps for the GPU), add configs.
   Key: "d" and "n_q_groups" must remain in the kernel signature as `tl.constexpr` args.

3. **do_not_specialize interaction with autotune**: `@triton.autotune` wraps `@triton.jit`.
   The order (autotune outer, jit inner) is the standard Triton pattern. Confirm the
   `do_not_specialize` argument is honoured through the autotune wrapper on the VM's Triton version.

4. **Carried 3a in-place store aliasing** (3a concern #2/#3): acc/m/lse buffer slice aliasing
   still applies — same concern as 3a. If oracle tests fail with all-zero output, add explicit
   `.copy_()` after the kernel call.

5. **Carried 3a tail-block BLK < 16** (3a concern #4): the tail block (`k_tail`) may have fewer
   than 16 tokens, which violates the `tl.dot ≥ 16` constraint. The tail goes to `_run_split`
   which calls `_online_block_kernel_launch` → the Triton kernel. If tail_len < 16, this errors.
   Mitigation: pad tail or bypass Triton for tail (use PyTorch online_softmax_update).
   Not fixed in 3b; tracked for 3c.

6. **`tl.reshape` availability**: still requires Triton ≥ 2.1 (3a carry-forward).

---

Files:
- `src/bmx/cache/triton_dequant_attention.py` — extended (split-KV + reduction + autotune)
- `tests/test_triton_decode_rtn.py` — extended (5 new CUDA-gated tests: 4 parametrized + 1 bit-identity)
