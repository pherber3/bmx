# Fused Triton decode kernel — results (Phase 3a, the deployment kernel)

**Branch:** `feat/triton-decode-kernel`. **Hardware:** NVIDIA GH200 480GB
(132 SMs, HBM3e ~4 TB/s). **Status:** both deployment paths complete, wired into
the live cache, generate-parity verified; full suite 342 passed / 1 xfailed on the
GH200.

## What this is

A single-launch, split-KV Triton **decode**-attention kernel that dequantizes the
packed KV codes **in-kernel** — so the resident KV stays compressed (no dense fp16
copy) and one kernel launch per decode step does the whole attention. Two arms:

- **RTN** (`rtn_token` K and V): int8 codes + per-group scales dequanted in-register.
- **k2b — the real recipe** (`lowrank_rtn_channel` K @3b + `turboquant_mse_perhead`
  V @2b): in-kernel lowrank-K reconstruction (`Us @ Vfac.T` + RTN residual) +
  in-kernel RoPE on the pre-RoPE keys + per-head turboquant-V dequant (codebook
  gather + an in-kernel per-head `d`-point Hadamard unrotate + per-head norms).

Both are wired into `PackedStreamingCache.attend` (decode path) and produce
generate-parity with `StreamingQuantizedCache`.

## The story (per-block launch → fused → split-KV → tl.dot)

The starting point (the previous leg's kernel) launched **one Triton kernel per KV
block per KV head** with the online-softmax carry threaded through PyTorch between
launches. Measured baseline: **~1.6× slower than chunked PyTorch** — launch
overhead dominated (~10ms/step from per-block launches at 128k). The rewrite:

1. **Fuse into one launch** (internal KV-block loop, register `(m,lse,acc)` carry,
   GQA group-fused so the KV tile is loaded once per head): ~36× vs chunked, but
   only **0.6% of HBM peak** — grid `(h_kv,)` = 8 programs on 132 SMs (SM-starved).
2. **Split-KV** (grid `(h_kv, num_splits)` + a tiny merge kernel — Triton has no
   global barrier, so the merge must be a second launch; the vLLM "3D kernel"
   pattern): 540× vs chunked, ~10% peak — then **plateaued at 16 splits**. A
   split-sweep proved the wall was per-program, not SM count (more splits stopped
   helping).
3. **`tl.dot` contraction** (pad the GQA group dim to 16; the broadcast cube
   `q[:,None,:]*k[None,:,:]` was register-spilling and capping bandwidth):
   **10% → 54% of HBM peak**. This was the key perf fix.

## Measured decode latency (GH200, Llama dims d=128, h_kv=8, n_q_groups=4, blk=64)

**RTN packed** (int8 codes resident, dequant-in-kernel), num_splits=32:

| ctx     | chunked (ms) | fused packed (ms) | speedup | % HBM peak |
|---------|-------------:|------------------:|--------:|-----------:|
| 2048    | 12.3         | 0.073             | 168×    | 1.5%       |
| 8192    | 49.1         | 0.075             | 652×    | 5.7%       |
| 32768   | 213.7        | 0.090             | 2368×   | 19.2%      |
| 131072  | 812.1        | 0.310             | 2624×   | 22.4%      |

(Packed latency ≈ the dense-fp16 fused kernel's, at **half the resident memory** —
int8 codes vs fp16. The dense variant hit 54% of HBM peak; the % above is on the
smaller int8 byte count.)

**k2b — the real recipe** (lowrank K @3b + per-head turboquant V @2b, all
dequant-in-kernel), num_splits=32:

| ctx     | chunked (ms) | fused k2b (ms) | speedup |
|---------|-------------:|---------------:|--------:|
| 2048    | 31.0         | 0.71           | 44×     |
| 8192    | 128.4        | 0.95           | 136×    |
| 32768   | 526.0        | 2.10           | 251×    |
| 131072  | 2065.6       | 6.41           | 322×    |

k2b is slower than RTN (it does in-kernel lowrank reconstruction + RoPE + a
per-head Hadamard per block — more compute per token) and is now **compute-bound**,
not memory-bound. It is still **~322× faster than the chunked k2b path** and runs
the full quality recipe end-to-end. Further k2b speedup is a compute-optimization
opportunity, not a correctness gap.

## Why per-head Hadamard for V (the one real design decision)

`turboquant_mse` rotates the full `C = h_kv·d` row with one Hadamard, which couples
all heads. Under GQA each query head has its own softmax, so that cross-head
rotation **cannot** be fused into a per-head decode kernel and **cannot** be folded
into `o_proj` (dimension mismatch + per-head-`p` commutation failure — worked out
rigorously, and it's why QuaRot/SpinQuant use a per-head Hadamard). The fix is the
production-standard one: rotate each `d_head` block independently
(`turboquant_mse_perhead`, per-head norms). This is **quality-equivalent** — the
turboquant distortion bound `√3·π/2·4^−b` is dimension-independent in the constant
and the Beta→Gaussian concentration is already excellent at d=128 (measured: per-head
rel-MSE 0.116 vs full-C 0.117, ratio 0.986). With per-head rotation, V dequant is
fully per-head: a `d`-point Hadamard done in-kernel as a `(d,d)` matmul, no
cross-head traffic, **no `o_proj` surgery, no weight mutation** — the rotation stays
entirely cache-side.

## Correctness

- Per-component oracle isolation (RTN 10/10, k2b 6/6 vs `naive_dense_attention`,
  all ≤ ~1e-3 vs the fp16/tf32 bar of 1e-2).
- The FWHT-defer identity and per-head independence verified in pure torch.
- Live generate-parity through `PackedStreamingCache` vs `StreamingQuantizedCache`
  for both arms (`test_fused_packed_generate_matches_streaming_cuda`,
  `test_fused_k2b_generate_matches_streaming_cuda`).
- Full suite **342 passed / 1 xfailed** on the GH200.

## How to deploy (the integrator's recipe)

The kernel consumes exactly a **paged-KV block-table layout**: int8/packed codes
stacked `(max_blocks, h_kv, blk, d)` + scales/factors, which serving engines
(vLLM/SGLang) already maintain. The entry points:

- `fused_decode_attention_packed(q, k_codes, v_codes, k_scales, v_scales, seq_len, …)`
  — RTN.
- `fused_decode_attention_k2b(q, stacks, seq_len, …)` — the real recipe; `stacks`
  from `build_kv_stacked_k2b`.

`PackedStreamingCache` shows the wiring: it dispatches decode to the fused kernel
when on CUDA with a compatible arm (RTN, or k2b with d≥16/pow2 and rank≥16), folds
the fp16 recent-window tail via an online-softmax merge, and otherwise falls back to
the per-block Triton / chunked path (fail-loud, no silent error-swallow). A
production integrator maintains the code stacks incrementally in their paged-KV
block table instead of rebuilding per step.

## Triton 3.7 / GH200 facts learned (for future kernel work)

- No `tl.cat`/`tl.join`/2D-slice; no Python list/tuple carry inside `@jit` (use a 2D
  tensor carry). rotate_half via a `(d,d)` perm matrix applied with `tl.dot` (the
  broadcast-cube form spills SMEM).
- `tl.dot` needs M,N,K ≥ 16 → pad the GQA group to 16 for q·k / p·v, or GEMV on the
  group axis; use `tl.dot` for the big all-≥16 contractions (lowrank, rotate,
  Hadamard) — the broadcast cubes blow out shared memory.
- `tl.arange` ranges must be powers of 2 → 2D `(BLK_POW2, d)` loads with row masking
  handle non-pow2 flush blocks.
- Split-KV needs a second merge kernel (no global barrier); oversubscribe SMs ~2×
  (`pick_num_splits`, 32 optimal here).
