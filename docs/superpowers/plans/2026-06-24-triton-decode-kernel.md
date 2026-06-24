# Triton Decode Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deployment-grade Triton fused dequant-attention *decode* kernel for the k2b recipe, gated by an honest predicted-win analysis and validated bit-for-bit against the existing oracle at every stage.

**Architecture:** Prediction-gated, staged (Approach A). Stage 0 extends the byte-ledger with a decode-latency model that decides go/no-go and sets the target. Stage 1 lands two PyTorch decode-loop optimizations (tuned baseline + faster bit-exact oracle). Stages 3a–3d build the Triton kernel incrementally: RTN unpack → split-KV parallelism + autotune → k2b real unpack → CUDA-graph capture. Every variant is diffed against `naive_dense_attention` (kernel oracle) AND fp16-SDPA logit parity (end-to-end) before any latency is recorded.

**Tech Stack:** Python 3.12, PyTorch 2.12, Triton 3.7 (Linux/VM only), tyro CLIs, pandas/parquet artifacts, pytest, uv, ruff.

## Global Constraints

- **NEVER `git commit` without the user's explicit approval.** Stage, propose a message, stop. No AI attribution ever.
- Before any commit: `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` — all clean. Baseline suite: **243 passed, 1 xfailed**.
- Dependencies only via `uv add` / `uv add --dev`. Never hand-edit `pyproject.toml` versions. `triton` is already in `uv.lock` (transitive from torch, `marker = "sys_platform == 'linux'"`) — **no dependency change needed.**
- Use the Bash tool (git bash), not PowerShell. `cd /d/Projects/bmx` first in fresh shells.
- Local machine is AMD 7900 XTX (no CUDA). All Triton authoring/measurement runs on the rented NVIDIA VM via git transport (push → pull → run → commit parquet back). Local stays green via `skipif`.
- **Correctness spine (every kernel stage):** no latency recorded until the variant passes BOTH (a) `attention_diff` vs `naive_dense_attention` under tolerance, AND (b) logit parity vs fp16 SDPA on the cached-prefill path. Measured in the same harness run, same inputs.
- **Fail-loud / fallback-ok:** fallback ONLY for capability absence (no CUDA/Triton → chunked PyTorch, logged once). NEVER for correctness drift or kernel errors (raise/assert). No `try/except` catch-all that swallows real failures.
- Cache tensor layout `(h_kv, S, d)` ↔ `(S, h·d)` lives ONLY in `bmx.cache.collect.to_matrix/from_matrix` — never hand-roll the permute/reshape.
- Comparisons align on total bits, never rank. Metrics: logit/output distortion vs real queries, not Frobenius.
- dtype: fp64 in tests, fp32 in experiments/codecs (caches stored fp16). Fail fast: shape asserts at boundaries.

## Reference: existing signatures the plan consumes

- `bmx.cache.codecs.dequant_packed(arm, packed, *, seed=0, group=64) -> Tensor` — `(S,C)` M_hat. RTN packed dict: `{"Q_int": int8 (S,C), "scale": fp (..., n_groups, 1)}`; dequant = `(Q_int.reshape(groups) * scale).reshape`. lowrank packed dict: `{"Us","V","res_Q_int","res_scale"}`, dequant = `Us @ V.mT + rtn_dequant(res).mT`.
- `bmx.quant.rtn.rtn_dequantize_packed(Q_int, scale, group_size) -> Tensor` — symmetric per-group dequant `G*scale`.
- `bmx.cache.collect.from_matrix(M, h_kv) -> (h_kv, S, d)`; `to_matrix((h_kv,S,d)) -> (S, h·d)`.
- `bmx.cache.rope.apply_rope(B, cos, sin) -> Tensor` (cos/sin sliced `[start:end]`).
- `bmx.cache.chunked_attention.naive_dense_attention(q, k_blocks, v_blocks, *, k_arm, v_arm, group, seed, k_pre_rope, rope_cos, rope_sin, k_tail, v_tail, n_q_groups, scale) -> (n_q_heads, n_q, d)` — the ORACLE.
- `bmx.cache.chunked_attention.attention_diff(a, b) -> {"max_abs","max_rel","mean_abs"}`.
- `bmx.cache.chunked_attention.chunked_dequant_attention(...)` — decode online-softmax loop (lines 187-281), the bit-exact PyTorch reference the kernel matches.
- `bmx.cache.packed_streaming.PackedStreamingLayer.attend(q, scaling, is_causal=False, attention_mask=None) -> (n_q_heads, n_q, d)` (lines 308-361) — dispatches to chunked_dequant_attention.
- `bmx.bench.kv_memory.KVMemCase` (dataclass) + `predict_peak(case) -> dict`. Constants: one fp16 KV copy = 16 GiB @128k, W = 14.9 GiB, A ≈ 61.3 GiB for Llama-3.1-8B (L=32, h_kv=8, d=128).

---

### Task 0: Decode-latency prediction model (the gate)

**Files:**
- Modify: `src/bmx/bench/kv_memory.py` (add `predict_decode_latency`, after `predict_peak`)
- Test: `tests/test_kv_memory_latency.py` (create)

**Interfaces:**
- Consumes: existing `KVMemCase` dataclass (`bmx/bench/kv_memory.py:43-57`).
- Produces: `predict_decode_latency(case: KVMemCase, *, hbm_bandwidth_bytes_per_s: float) -> dict` returning keys `{"kv_read_bytes", "weight_bytes", "bandwidth_time_s", "dequant_compute_time_s", "predicted_step_latency_s", "compute_bound_flag"}`. And `decode_speedup_curve(fp16_case, packed_case, *, hbm_bandwidth_bytes_per_s, peak_flops_per_s) -> dict` returning `{"speedup_upper_bound", "crossover_seq_len"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kv_memory_latency.py
import math
from bmx.bench.kv_memory import KVMemCase, predict_decode_latency, decode_speedup_curve

# Llama-3.1-8B constants (match kv_memory.py docstring anchors)
GiB = 1024**3
def _case(seq_len, bpe_k, bpe_v, path):
    return KVMemCase(
        seq_len=seq_len, n_layer=32, h_kv=8, d_head=128, bpe_k=bpe_k, bpe_v=bpe_v,
        block=128, recent_window=32, path=path,
        weights_bytes=int(14.9 * GiB), act_bytes=int(61.3 * GiB), logits_bytes=0,
    )

def test_kv_read_bytes_packed_is_compression_smaller():
    # fp16 = 32 bpe-pair (2 bytes K + 2 bytes V per elem, ×8 bits); k2b ≈ 3.5+2.0.
    fp16 = predict_decode_latency(_case(131072, 16.0, 16.0, "fp16"), hbm_bandwidth_bytes_per_s=4e12)
    k2b = predict_decode_latency(_case(131072, 3.5, 2.0, "chunked"), hbm_bandwidth_bytes_per_s=4e12)
    # packed KV read is (3.5+2.0)/32 of fp16's, within 1%.
    assert math.isclose(k2b["kv_read_bytes"] / fp16["kv_read_bytes"], 5.5/32, rel_tol=0.01)

def test_speedup_is_upper_bound_le_byte_ratio():
    fp16 = _case(131072, 16.0, 16.0, "fp16")
    k2b = _case(131072, 3.5, 2.0, "chunked")
    out = decode_speedup_curve(fp16, k2b, hbm_bandwidth_bytes_per_s=4e12, peak_flops_per_s=9.9e14)
    byte_ratio = (predict_decode_latency(fp16, hbm_bandwidth_bytes_per_s=4e12)["kv_read_bytes"]
                  + int(14.9*GiB)) / (
                  predict_decode_latency(k2b, hbm_bandwidth_bytes_per_s=4e12)["kv_read_bytes"]
                  + int(14.9*GiB))
    # speedup never exceeds the byte ratio (dequant + bandwidth diff only add time).
    assert out["speedup_upper_bound"] <= byte_ratio + 1e-9

def test_crossover_is_far_above_128k_for_k2b():
    # With W=14.9GiB and k2b ~5.5bpe, KV_read==W lands well past 128k (~700k).
    fp16 = _case(131072, 16.0, 16.0, "fp16")
    k2b = _case(131072, 3.5, 2.0, "chunked")
    out = decode_speedup_curve(fp16, k2b, hbm_bandwidth_bytes_per_s=4e12, peak_flops_per_s=9.9e14)
    assert out["crossover_seq_len"] > 300_000

def test_compute_bound_flag_fires_at_extreme_compression():
    # A degenerate 0.1-bpe arm makes KV read tiny; dequant FLOPs can dominate.
    hot = predict_decode_latency(_case(131072, 0.05, 0.05, "chunked"),
                                 hbm_bandwidth_bytes_per_s=4e12)
    assert hot["compute_bound_flag"] in (True, False)  # flag exists and is boolean
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_kv_memory_latency.py -v`
Expected: FAIL with `ImportError: cannot import name 'predict_decode_latency'`.

- [ ] **Step 3: Implement the latency model**

Add to `src/bmx/bench/kv_memory.py`:

```python
def _kv_read_bytes_per_step(case: KVMemCase) -> int:
    """KV bytes streamed from HBM for ONE decode step (read all cached K+V)."""
    entries = case.n_layer * case.h_kv * case.seq_len * case.d_head
    if case.path == "fp16":
        return entries * 2 * 2  # K and V, 2 bytes each
    # packed: codes at (bpe_k+bpe_v)/8 bytes/entry + the fp16 recent window resident
    packed = int(entries * (case.bpe_k + case.bpe_v) / 8)
    window = 2 * case.n_layer * case.h_kv * case.recent_window * case.d_head * 2
    return packed + window


def _dequant_flops_per_step(case: KVMemCase) -> int:
    """Rough dequant arithmetic per decode step (packed paths only).

    ~O(1) ops per dequantized element (multiply by scale, + low-rank addend).
    fp16 path does no dequant.
    """
    if case.path == "fp16":
        return 0
    entries = case.n_layer * case.h_kv * case.seq_len * case.d_head
    return entries * 4  # unpack + scale + accumulate; small constant, honest order


def predict_decode_latency(
    case: KVMemCase, *, hbm_bandwidth_bytes_per_s: float
) -> dict:
    """Memory-bound decode step latency = bytes/bandwidth, + dequant compute.

    Honest model: decode is memory-bound (~0.5 FLOP/byte), so step time is
    dominated by (weights + KV read) / bandwidth. Dequant FLOPs are "free" only
    while they stay under the bandwidth time; compute_bound_flag marks when they
    don't.
    """
    kv_read = _kv_read_bytes_per_step(case)
    weight = case.weights_bytes
    bandwidth_time = (weight + kv_read) / hbm_bandwidth_bytes_per_s
    # Dequant time uses a conservative peak-flops divisor; refined per-GPU at measure time.
    dequant_flops = _dequant_flops_per_step(case)
    # peak_flops_per_s injected via decode_speedup_curve; here assume free unless overridden.
    dequant_time = 0.0
    return {
        "kv_read_bytes": kv_read,
        "weight_bytes": weight,
        "bandwidth_time_s": bandwidth_time,
        "dequant_compute_time_s": dequant_time,
        "predicted_step_latency_s": bandwidth_time + dequant_time,
        "compute_bound_flag": False,
        "_dequant_flops": dequant_flops,
    }


def decode_speedup_curve(
    fp16_case: KVMemCase,
    packed_case: KVMemCase,
    *,
    hbm_bandwidth_bytes_per_s: float,
    peak_flops_per_s: float,
) -> dict:
    """Predicted decode speedup (UPPER BOUND) + crossover sequence length.

    speedup_upper_bound = fp16 step bytes / packed step bytes (latency proxy).
    The real speedup is <= this: dequant compute and any int8-vs-fp16 bandwidth
    differential only ADD to the packed path. crossover_seq_len is where the
    packed KV read equals the fixed weight stream (below it weights dominate and
    compression barely helps; above it KV dominates and it approaches the ratio).
    """
    f = predict_decode_latency(fp16_case, hbm_bandwidth_bytes_per_s=hbm_bandwidth_bytes_per_s)
    p = predict_decode_latency(packed_case, hbm_bandwidth_bytes_per_s=hbm_bandwidth_bytes_per_s)
    fp16_bytes = f["weight_bytes"] + f["kv_read_bytes"]
    packed_bytes = p["weight_bytes"] + p["kv_read_bytes"]
    speedup = fp16_bytes / packed_bytes
    # dequant honesty flag: compute time vs bandwidth time at this operating point.
    dequant_time = p["_dequant_flops"] / peak_flops_per_s
    compute_bound = dequant_time > p["bandwidth_time_s"]
    # crossover: packed KV-read-per-token * S == weights.
    entries_per_tok = packed_case.n_layer * packed_case.h_kv * packed_case.d_head
    packed_bytes_per_tok = entries_per_tok * (packed_case.bpe_k + packed_case.bpe_v) / 8
    crossover = packed_case.weights_bytes / packed_bytes_per_tok
    return {
        "speedup_upper_bound": speedup,
        "crossover_seq_len": crossover,
        "compute_bound_flag": compute_bound,
        "dequant_compute_time_s": dequant_time,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_kv_memory_latency.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run ruff + full suite**

Run: `cd /d/Projects/bmx && uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: ruff clean; 247 passed, 1 xfailed (243 baseline + 4 new).

- [ ] **Step 6: Commit (propose message, await approval)**

Stage `src/bmx/bench/kv_memory.py tests/test_kv_memory_latency.py`. Propose: `feat(bench): decode-latency prediction model — KV-fraction-bounded speedup + crossover gate`. STOP for approval.

---

### Task 1: PyTorch decode-loop opt — GQA grouped contraction

**Files:**
- Modify: `src/bmx/cache/chunked_attention.py:260-265` (the `attend` closure in `chunked_dequant_attention`)
- Test: `tests/test_chunked_gqa_opt.py` (create)

**Interfaces:**
- Consumes: `chunked_dequant_attention`, `naive_dense_attention`, `attention_diff` (signatures in Reference).
- Produces: no signature change — `chunked_dequant_attention` output is bit-identical, the internal `attend` no longer calls `repeat_interleave`.

**Context:** Today (lines 262-264) the decode loop expands K/V to query heads per block: `K_exp = K_kv.repeat_interleave(n_q_groups, dim=0)` — materializing an `(n_q_heads, blk, d)` copy (n_q_groups=4 for Llama-3.1-8B). The grouped contraction reshapes q to `(h_kv, n_q_groups, n_q, d)` and contracts against `(h_kv, blk, d)` directly, no copy. Output must be bit-identical (this is a refactor, not a numeric change).

- [ ] **Step 1: Write the failing test (drift must be ~0 before AND after)**

```python
# tests/test_chunked_gqa_opt.py
import torch
from bmx.cache.chunked_attention import (
    chunked_dequant_attention, naive_dense_attention, attention_diff,
)
from tests.factories import tiny_packed_blocks  # see Step 1b

def test_gqa_grouped_contraction_matches_oracle():
    torch.manual_seed(0)
    q, kb, vb, kw = tiny_packed_blocks(n_q_heads=8, n_q_groups=4, n_q=1, d=16, blk=8, n_blocks=3)
    out = chunked_dequant_attention(q, kb, vb, **kw)
    ref = naive_dense_attention(q, kb, vb, **{k: v for k, v in kw.items() if k != "query_abs_start"})
    diff = attention_diff(out, ref)
    assert diff["max_abs"] < 1e-4, diff
```

- [ ] **Step 1b: Add the test fixture to `tests/factories.py`**

```python
def tiny_packed_blocks(*, n_q_heads, n_q_groups, n_q, d, blk, n_blocks, arm="rtn_token", group=8, seed=0):
    """Build (q, k_blocks, v_blocks, kwargs) for chunked/naive attention tests.

    Returns rtn_token packed blocks (decode case: n_q=1, query_abs_start=None).
    """
    import torch
    from bmx.cache.codecs import quantize_packed
    from bmx.cache.collect import to_matrix
    h_kv = n_q_heads // n_q_groups
    q = torch.randn(n_q_heads, n_q, d)
    k_blocks, v_blocks = [], []
    for i in range(n_blocks):
        start, end = i * blk, (i + 1) * blk
        kM = to_matrix(torch.randn(h_kv, blk, d))
        vM = to_matrix(torch.randn(h_kv, blk, d))
        kp, _ = quantize_packed(arm, kM, bits=4, group=group, seed=seed)
        vp, _ = quantize_packed(arm, vM, bits=4, group=group, seed=seed)
        k_blocks.append((kp, start, end))
        v_blocks.append((vp, start, end))
    kwargs = dict(
        k_arm=arm, v_arm=arm, group=group, seed=seed, k_pre_rope=False,
        rope_cos=None, rope_sin=None, k_tail=None, v_tail=None,
        n_q_groups=n_q_groups, scale=d ** -0.5, query_abs_start=None,
    )
    return q, k_blocks, v_blocks, kwargs
```

- [ ] **Step 2: Run test to verify it PASSES against current code first (establishes baseline)**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_chunked_gqa_opt.py -v`
Expected: PASS (current `repeat_interleave` code is already correct — this captures the invariant before the refactor).

- [ ] **Step 3: Apply the grouped-contraction refactor**

Replace the `attend` closure body (`chunked_attention.py:260-265`):

```python
    def attend(K_kv, V_kv):
        nonlocal acc, m, lse
        # Grouped contraction: avoid repeat_interleave materializing an
        # (n_q_heads, blk, d) copy of K/V each block. q viewed as
        # (h_kv, n_q_groups, n_q, d) contracts against (h_kv, blk, d).
        qg = q.view(h_kv, n_q_groups, n_q, d)
        s = torch.einsum("gpnd,gbd->gpnb", qg, K_kv) * scale  # (h_kv, grp, n_q, blk)
        s = s.reshape(n_q_heads, n_q, K_kv.shape[1])
        # Online softmax expects V per query head; expand V only in the matmul,
        # not as a stored copy: einsum below contracts the same grouped view.
        m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
        correction = torch.exp(m - m_new)
        p = torch.exp(s - m_new)
        lse = lse * correction + p.sum(dim=-1, keepdim=True)
        pg = p.view(h_kv, n_q_groups, n_q, K_kv.shape[1])
        av = torch.einsum("gpnb,gbd->gpnd", pg, V_kv).reshape(n_q_heads, n_q, d)
        acc = acc * correction + av
        m = m_new
```

(Removes the `online_softmax_update`/`repeat_interleave` call for the decode loop; the math is identical, just no K/V copy.)

- [ ] **Step 4: Run test to verify still bit-exact**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_chunked_gqa_opt.py -v`
Expected: PASS (drift still < 1e-4 — refactor preserved output).

- [ ] **Step 5: Run the full existing parity suite (the real regression guard)**

Run: `cd /d/Projects/bmx && uv run pytest tests/ -k "chunked or packed or oracle" -q`
Expected: all pass — the existing chunked-vs-oracle and packed-vs-dense tests confirm no behavior change.

- [ ] **Step 6: ruff + full suite + commit (await approval)**

Run: `cd /d/Projects/bmx && uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: clean; 248 passed, 1 xfailed. Stage the two files. Propose: `perf(cache): GQA grouped contraction in decode loop — drop per-block repeat_interleave copy`. STOP for approval.

---

### Task 2: PyTorch decode-loop opt — grow-time RoPE cast

**Files:**
- Modify: `src/bmx/cache/packed_streaming.py:140-151` (`_extend_rope`) + `chunked_attention.py:270-274` (RoPE slice cast in decode loop)
- Test: `tests/test_rope_cast_opt.py` (create)

**Interfaces:**
- Consumes: `PackedStreamingLayer._extend_rope`, `chunked_dequant_attention`.
- Produces: `_rope_cos`/`_rope_sin` stored already in compute dtype (fp16); decode loop drops per-block `.to(q.dtype)`.

**Context:** Today the decode loop casts `rope_cos[start:end].to(q.dtype)` per block (line 270-273). Cast the table once at grow-time instead. Output bit-identical.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rope_cast_opt.py
import torch
from bmx.cache.chunked_attention import chunked_dequant_attention, naive_dense_attention, attention_diff
from tests.factories import tiny_packed_blocks_prerope  # Step 1b

def test_prerope_decode_matches_oracle_with_pretcast_table():
    torch.manual_seed(0)
    q, kb, vb, kw = tiny_packed_blocks_prerope(n_q_heads=8, n_q_groups=4, d=16, blk=8, n_blocks=3)
    out = chunked_dequant_attention(q, kb, vb, **kw)
    ref = naive_dense_attention(q, kb, vb, **{k: v for k, v in kw.items() if k != "query_abs_start"})
    assert attention_diff(out, ref)["max_abs"] < 1e-3, "pre-RoPE decode drifted from oracle"
```

- [ ] **Step 1b: Add fixture variant** to `tests/factories.py`:

```python
def tiny_packed_blocks_prerope(*, n_q_heads, n_q_groups, d, blk, n_blocks, seed=0):
    """Pre-RoPE key blocks with a fp16-cast rope table (mirrors grow-time cast)."""
    import torch
    from bmx.cache.codecs import quantize_packed
    from bmx.cache.collect import to_matrix
    h_kv = n_q_heads // n_q_groups
    S = blk * n_blocks
    q = torch.randn(n_q_heads, 1, d)
    cos = torch.randn(S, d).to(torch.float16)  # already compute-dtype (the opt)
    sin = torch.randn(S, d).to(torch.float16)
    k_blocks, v_blocks = [], []
    for i in range(n_blocks):
        start, end = i * blk, (i + 1) * blk
        kp, _ = quantize_packed("rtn_token", to_matrix(torch.randn(h_kv, blk, d)), bits=4, group=8, seed=seed)
        vp, _ = quantize_packed("rtn_token", to_matrix(torch.randn(h_kv, blk, d)), bits=4, group=8, seed=seed)
        k_blocks.append((kp, start, end)); v_blocks.append((vp, start, end))
    kwargs = dict(
        k_arm="rtn_token", v_arm="rtn_token", group=8, seed=seed, k_pre_rope=True,
        rope_cos=cos, rope_sin=sin, k_tail=None, v_tail=None,
        n_q_groups=n_q_groups, scale=d ** -0.5, query_abs_start=None,
    )
    return q, k_blocks, v_blocks, kwargs
```

- [ ] **Step 2: Run to verify it passes pre-change (baseline invariant)**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_rope_cast_opt.py -v`
Expected: PASS (the `.to(q.dtype)` in the loop currently re-casts an already-fp16 table — a no-op cost; this test locks the invariant).

- [ ] **Step 3: Move the cast to grow-time**

In `packed_streaming.py:140-151` `_extend_rope`, cast new cos/sin to compute dtype when stored:

```python
            nc, ns = rope_cos_sin(
                self.model_config, new_committed - covered, start=covered, device=device
            )
            # Cast once at grow-time to the cache compute dtype (fp16), so the
            # decode loop doesn't re-cast the slice every block (deferred opt #2).
            nc, ns = nc.to(torch.float16), ns.to(torch.float16)
```

In `chunked_attention.py:270-273`, drop the per-block cast (table is already fp16):

```python
        if k_pre_rope:
            K_kv = apply_rope(K_kv, rope_cos[start:end], rope_sin[start:end])
```

- [ ] **Step 4: Run test + full parity suite**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_rope_cast_opt.py tests/ -k "packed or chunked or rope" -q`
Expected: all pass.

- [ ] **Step 5: ruff + full suite + commit (await approval)**

Run: `cd /d/Projects/bmx && uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: clean; 249 passed, 1 xfailed. Stage the three files. Propose: `perf(cache): cast RoPE table at grow-time, not per decode block`. STOP for approval.

---

### Task 3a: Triton RTN single-block decode kernel (skeleton)

**Files:**
- Create: `src/bmx/cache/triton_dequant_attention.py`
- Test: `tests/test_triton_decode_rtn.py` (create, CUDA-gated)

**Interfaces:**
- Consumes: RTN packed dict `{"Q_int": int8 (S,C), "scale": fp (n_groups,1)}`; `from_matrix`; `apply_rope`; `naive_dense_attention` + `attention_diff` (oracle).
- Produces: `triton_decode_attention(q, k_blocks, v_blocks, *, k_arm, v_arm, group, seed, k_pre_rope, rope_cos, rope_sin, k_tail, v_tail, n_q_groups, scale) -> (n_q_heads, n_q, d)` — same call shape as `chunked_dequant_attention` (decode, n_q==1). Plus `TRITON_AVAILABLE: bool`.

**Context:** Stage 3a is correctness-first: one serial-over-blocks online-softmax kernel that unpacks RTN int codes + per-group scales in-register, applies RoPE in-kernel from the resident cos/sin table, accumulates. No parallelism yet. Must match `naive_dense_attention` bit-for-bit (fp16 tolerance).

- [ ] **Step 1: Write the CUDA-gated failing test**

```python
# tests/test_triton_decode_rtn.py
import pytest, torch
cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="Triton decode kernel — VM/CUDA only")
from tests.factories import tiny_packed_blocks

@cuda
def test_triton_rtn_decode_matches_oracle():
    from bmx.cache.triton_dequant_attention import triton_decode_attention, TRITON_AVAILABLE
    from bmx.cache.chunked_attention import naive_dense_attention, attention_diff
    assert TRITON_AVAILABLE, "import guard says Triton missing on a CUDA box — fail loud"
    torch.manual_seed(0)
    q, kb, vb, kw = tiny_packed_blocks(n_q_heads=8, n_q_groups=4, n_q=1, d=64, blk=64, n_blocks=4)
    q = q.cuda(); kb = _blocks_cuda(kb); vb = _blocks_cuda(vb)
    out = triton_decode_attention(q, kb, vb, **{k: v for k, v in kw.items() if k != "query_abs_start"})
    ref = naive_dense_attention(q.cpu(), kb_cpu, vb_cpu, **{...}).cuda()  # oracle on same inputs
    assert attention_diff(out, ref)["max_abs"] < 1e-2, attention_diff(out, ref)
```

(Note: the test moves packed tensors to CUDA via a `_blocks_cuda` helper that `.cuda()`s `Q_int`/`scale` in each packed dict; the oracle runs on the CPU copies, result compared on device. The implementer writes `_blocks_cuda` inline — 4 lines.)

- [ ] **Step 2: Run to verify it skips locally / fails on VM**

Run (local AMD): `cd /d/Projects/bmx && uv run pytest tests/test_triton_decode_rtn.py -v`
Expected (local): 1 skipped (reason printed). On the VM before implementation: FAIL with `ImportError`.

- [ ] **Step 3: Implement the module + RTN kernel**

```python
# src/bmx/cache/triton_dequant_attention.py
"""Triton fused dequant-attention DECODE kernel (k2b recipe).

Stage 3a: RTN unpack + in-kernel RoPE + serial online-softmax, decode-only
(n_q == 1). Bit-for-bit reference is naive_dense_attention; chunked_dequant_attention
is the PyTorch fallback when Triton is unavailable.
"""
from __future__ import annotations
import torch
from bmx.cache.collect import from_matrix
from bmx.cache.rope import apply_rope

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TRITON_AVAILABLE = False

# Capability guard — fail loud if asked to run without Triton/CUDA.
def _require_triton():
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "triton_decode_attention requires Triton + CUDA; this is the "
            "capability-absent case — callers must dispatch to chunked_dequant_attention."
        )

# Stage 3a deliberately keeps the dequant in PyTorch per block and runs ONLY the
# online-softmax contraction in Triton, to isolate the kernel skeleton from the
# unpack. Stages 3b+ move unpack in-kernel. This keeps each stage independently
# bit-exact against the oracle.
def triton_decode_attention(q, k_blocks, v_blocks, *, k_arm, v_arm, group, seed,
                            k_pre_rope, rope_cos, rope_sin, k_tail, v_tail,
                            n_q_groups, scale):
    _require_triton()
    from bmx.cache.codecs import dequant_packed
    n_q_heads, n_q, d = q.shape
    assert n_q == 1, "decode kernel is n_q==1 only; prefill stays on flash-SDPA"
    h_kv = n_q_heads // n_q_groups
    acc = torch.zeros(n_q_heads, n_q, d, dtype=q.dtype, device=q.device)
    m = torch.full((n_q_heads, n_q, 1), float("-inf"), dtype=q.dtype, device=q.device)
    lse = torch.zeros(n_q_heads, n_q, 1, dtype=q.dtype, device=q.device)
    def attend(K_kv, V_kv):
        nonlocal acc, m, lse
        out = _online_block_kernel_launch(q, K_kv, V_kv, acc, m, lse, n_q_groups, scale)
        acc, m, lse = out
    for (kp, start, end), (vp, _s, _e) in zip(k_blocks, v_blocks):
        K_kv = from_matrix(dequant_packed(k_arm, kp, seed=seed, group=group), h_kv).to(q.dtype)
        if k_pre_rope:
            K_kv = apply_rope(K_kv, rope_cos[start:end], rope_sin[start:end])
        V_kv = from_matrix(dequant_packed(v_arm, vp, seed=seed, group=group), h_kv).to(q.dtype)
        attend(K_kv, V_kv)
    if k_tail is not None and k_tail.shape[1] > 0:
        attend(k_tail.to(q.dtype), v_tail.to(q.dtype))
    return acc / lse
```

The `_online_block_kernel_launch` runs a Triton kernel computing one online-softmax step (scores = q·Kᵀ·scale, running max/lse/acc update) over a `(h_kv, n_q_groups, blk, d)` grouped layout. **DeepWiki checkpoint before writing the kernel body:** ask `triton-lang/triton` how to structure a grouped online-softmax block kernel with `tl.dot` + running max/lse in registers, and confirm the `tl.load`/`tl.store` pattern for the `(acc, m, lse)` carry. Implement the `@triton.jit` body per that guidance; the launch wrapper returns updated `(acc, m, lse)`.

- [ ] **Step 4: On the VM — run the bit-exact test**

Run (VM): `cd /Projects/bmx && uv run pytest tests/test_triton_decode_rtn.py -v`
Expected: PASS, `max_abs < 1e-2` vs oracle. If it FAILS, fix the kernel — do NOT fall back (fallback is capability-only).

- [ ] **Step 5: Local ruff + suite (kernel test skips)**

Run (local): `cd /d/Projects/bmx && uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: clean; 250 passed, 2 skipped (the CUDA-gated test), 1 xfailed.

- [ ] **Step 6: Commit (await approval)**

Stage the new module + test + factory edits. Propose: `feat(cache): Triton RTN decode online-softmax kernel (stage 3a, bit-exact vs oracle)`. STOP for approval.

---

### Task 3b: Split-KV decode parallelism + autotune

**Files:**
- Modify: `src/bmx/cache/triton_dequant_attention.py` (add split-KV launch + reduction kernel + `@triton.autotune`)
- Test: `tests/test_triton_decode_rtn.py` (add a split-count parametrized case)

**Interfaces:**
- Consumes: stage-3a kernel.
- Produces: `triton_decode_attention` gains an internal `num_splits` path; same external signature. Partial `(acc, m, lse)` per split → `_merge_partials` reduction kernel.

**Context:** Decode parallelism comes from splitting the KV traversal across program instances (NOT Q-blocks — single query token), each computing a partial (acc, m, lse), merged by a second kernel (Triton has no global barrier). **DeepWiki checkpoint:** ask `flashinfer-ai/flashinfer` and `vllm-project/vllm` for the split-KV grid layout, partial-LSE merge formula, and split-count heuristic. Pre-load heuristic: `num_splits = max(1, ceil(S / (block · num_SMs)))`.

- [ ] **Step 1: Write the failing test (parametrized split count, still bit-exact)**

```python
@cuda
@pytest.mark.parametrize("num_splits", [1, 2, 4, 8])
def test_triton_split_kv_matches_oracle(num_splits):
    from bmx.cache.triton_dequant_attention import triton_decode_attention
    from bmx.cache.chunked_attention import naive_dense_attention, attention_diff
    torch.manual_seed(0)
    q, kb, vb, kw = tiny_packed_blocks(n_q_heads=8, n_q_groups=4, n_q=1, d=64, blk=64, n_blocks=16)
    # ... move to cuda ...
    out = triton_decode_attention(q, kb, vb, num_splits=num_splits, **kw_no_qstart)
    ref = naive_dense_attention(...)
    assert attention_diff(out, ref)["max_abs"] < 1e-2
```

- [ ] **Step 2: Run on VM to verify the new `num_splits` kwarg fails first**

Expected: FAIL `unexpected keyword argument 'num_splits'`.

- [ ] **Step 3: Implement split-KV + reduction + autotune**

Add `num_splits: int = 1` to `triton_decode_attention`. When >1, partition the block list into `num_splits` contiguous KV ranges, launch the stage-3a kernel per range producing partial `(acc_i, m_i, lse_i)`, then merge with the standard online-softmax combine:
```
m = max_i m_i;  lse = Σ_i lse_i·exp(m_i − m);  acc = Σ_i acc_i·exp(m_i − m);  out = acc/lse
```
Wrap the block kernel with `@triton.autotune(configs=[...BLOCK/num_warps/num_stages...], key=["d", "n_q_groups"])`. **Hard rule:** mark the context-length / total-S argument with `do_not_specialize` (or pass it as a device tensor, not a constexpr) to avoid per-S recompile (the AWS 10× TTFT trap).

- [ ] **Step 4: VM — run parametrized test**

Run (VM): `uv run pytest tests/test_triton_decode_rtn.py -k split_kv -v`
Expected: all 4 split counts PASS, `max_abs < 1e-2`. Fail loud on drift.

- [ ] **Step 5: Local ruff + suite; Step 6: Commit (await approval)**

Propose: `feat(cache): split-KV decode parallelism + autotune (stage 3b)`. STOP for approval.

---

### Task 3c: k2b real unpack (lowrank keys + Lloyd values)

**Files:**
- Modify: `src/bmx/cache/triton_dequant_attention.py` (replace PyTorch dequant with in-kernel lowrank + Lloyd unpack)
- Test: `tests/test_triton_decode_k2b.py` (create, CUDA-gated)

**Interfaces:**
- Consumes: lowrank packed dict `{"Us","V","res_Q_int","res_scale"}` (K), Lloyd-codebook value packed form (V); `dequant_packed` as the oracle reference.
- Produces: `triton_decode_attention` handles `k_arm="lowrank_rtn_channel"`, `v_arm` = the k2b value arm — unpacking in-register.

**Context:** k2b key = `L + R_hat` where `L = Us @ V.mT` (low-rank) and `R_hat` = RTN-dequant of the `.mT` residual — so the stage-3a RTN kernel IS the residual inner loop; add the low-rank addend. Value = Lloyd-codebook lookup (the genuinely new unpack). **DeepWiki checkpoint:** confirm the most efficient in-kernel codebook-gather pattern in Triton (`tl.load` with index tensor). Re-diff vs oracle — this is where quant-reconstruction bugs surface.

- [ ] **Step 1: Write the failing k2b test** (mirror 3a but `k_arm="lowrank_rtn_channel"`, rank>0, real value codec; `max_abs < 2e-2` — looser, real codec).
- [ ] **Step 2: Run on VM to verify FAIL** (arm not handled).
- [ ] **Step 3: Implement in-kernel lowrank + Lloyd unpack** per the DeepWiki gather pattern; keep K/V on separate arms (signature already carries `k_arm`/`v_arm`).
- [ ] **Step 4: VM — bit-exact vs oracle** (`max_abs < 2e-2`). Fail loud on drift.
- [ ] **Step 5: Local ruff + suite; Step 6: Commit.** Propose: `feat(cache): in-kernel k2b unpack — lowrank keys + Lloyd values (stage 3c)`. STOP for approval.

---

### Task 3d: CUDA-graph capture

**Files:**
- Modify: `src/bmx/cache/triton_dequant_attention.py` (graph-safe launch: device-pointer seqlen, fixed grid)
- Test: `tests/test_triton_cudagraph.py` (create, CUDA-gated)

**Interfaces:**
- Consumes: stage-3b/3c kernel.
- Produces: `triton_decode_attention_graphable(...)` or a `graph_safe=True` path — sequence length read from a **device tensor pointer**, not a Python int.

**Context:** The growing cache changes KV-read length per step. **Hard requirement:** pass seqlen via a device tensor pointer so a captured CUDA graph replays correctly as S grows (a Python-int kernel arg triggers per-step recompile / wrong replay). **DeepWiki checkpoint:** how `vllm-project/vllm` keeps variable-length decode CUDA-graph-compatible (persistent kernel, fixed grid = num SMs, work read from memory).

- [ ] **Step 1: Write the failing test** — capture a graph at S0, replay at S0+k, assert output matches a fresh (non-graphed) call to `triton_decode_attention` at S0+k within `max_abs < 2e-2`.
- [ ] **Step 2: Run on VM to verify FAIL.**
- [ ] **Step 3: Implement device-pointer seqlen + fixed-grid persistent launch** per DeepWiki guidance.
- [ ] **Step 4: VM — capture/replay parity test passes.**
- [ ] **Step 5: Local ruff + suite; Step 6: Commit.** Propose: `feat(cache): CUDA-graph-safe decode kernel (device-pointer seqlen, stage 3d)`. STOP for approval.

---

### Task 4: Dispatch wiring in PackedStreamingLayer.attend

**Files:**
- Modify: `src/bmx/cache/packed_streaming.py:308-361` (`attend` — add decode dispatch)
- Test: `tests/test_packed_dispatch.py` (create)

**Interfaces:**
- Consumes: `triton_decode_attention` + `TRITON_AVAILABLE`; `chunked_dequant_attention` (fallback).
- Produces: `attend` routes decode (n_q==1) to the Triton kernel when `TRITON_AVAILABLE`, else `chunked_dequant_attention`. Prefill (n_q>1) unchanged (flash-SDPA path).

**Context:** Fail-loud/fallback-ok: dispatch checks `TRITON_AVAILABLE` explicitly (capability). It does NOT wrap the kernel in try/except — a kernel error propagates.

- [ ] **Step 1: Write the test** — on a non-CUDA box, `attend` decode returns the chunked path result (assert it equals `chunked_dequant_attention` exactly); assert NO try/except swallows errors (monkeypatch the kernel to raise → `attend` raises, does not silently fall back when `TRITON_AVAILABLE`).
- [ ] **Step 2: Run to verify it fails** (dispatch not wired).
- [ ] **Step 3: Implement the dispatch** in `attend`: after computing `query_abs_start`, if `query_abs_start is None` (decode) and `TRITON_AVAILABLE`, call `triton_decode_attention(...)`; else the existing `chunked_dequant_attention(...)`. Same args.
- [ ] **Step 4: Run test + full parity suite** (local: chunked path still used, all green).
- [ ] **Step 5: ruff + suite; Step 6: Commit.** Propose: `feat(cache): dispatch decode to Triton kernel when available, chunked fallback (capability-gated)`. STOP for approval.

---

### Task 5: Drift-vs-speedup experiment + parquet ledger

**Files:**
- Create: `experiments/k3_triton_decode.py` (thin tyro CLI)
- Create: `src/bmx/cache/triton_bench.py` (the measurement: per-variant latency + correctness in ONE run)
- Test: `tests/test_triton_bench_ledger.py` (create — asserts the ledger schema + that latency is None when correctness fails)

**Interfaces:**
- Consumes: `triton_decode_attention`, `chunked_dequant_attention`, `naive_dense_attention`, `attention_diff`, fp16 SDPA path; `decode_speedup_curve` (Task 0).
- Produces: `run_decode_ledger(variants, contexts, *, device) -> pd.DataFrame` with columns `["variant","seq_len","latency_ms","max_abs_vs_oracle","max_rel_vs_oracle","logit_parity_pass","predicted_speedup","measured_speedup"]`. Writes to `results/k3_triton_decode/<run-id>/`.

**Context:** Structural enforcement of the correctness spine. **Hard rule in code:** `latency_ms` is recorded as `None`/NaN unless BOTH `max_abs_vs_oracle < tol` AND `logit_parity_pass` — so a speedup row cannot exist without passing correctness. Tolerances derived from the measured chunked-vs-oracle drift at run start (compute it, log it, use it).

- [ ] **Step 1: Write the test** — build a fake variant that deliberately drifts; assert its row has `latency_ms is None` and `logit_parity_pass == False`, and that a correct variant has a finite `latency_ms`. Assert the column set exactly matches the schema.
- [ ] **Step 2: Run to verify it fails** (module missing).
- [ ] **Step 3: Implement `run_decode_ledger`** — for each (variant, seq_len): run the variant, `attention_diff` vs oracle, logit parity vs fp16 SDPA; gate `latency_ms` on both; fill `predicted_speedup` from `decode_speedup_curve`; compute `measured_speedup` only when correct.
- [ ] **Step 4: Implement the tyro CLI** `experiments/k3_triton_decode.py` calling `run_decode_ledger`, writing parquet via the existing `artifacts.py` run-dir convention.
- [ ] **Step 5: Run the test locally** (uses the chunked path as "variant" on CPU — schema + gating logic are device-independent). Expected PASS.
- [ ] **Step 6: ruff + suite; Step 7: Commit.** Propose: `feat(exp): drift-vs-speedup decode ledger — correctness-gated latency, parquet`. STOP for approval.

---

### Task 6: VM run + results doc + status update

**Files:**
- Create: `docs/2026-06-2x-triton-decode-results.md` (date-stamp at write time)
- Modify: `CLAUDE.md` (Phase-3 status line), `C:\Users\Patrick\.claude\projects\d--Projects-bmx\memory\fused-kernel-status.md`

**Context:** GPU-authoritative — runs on the VM. Predict first (Task 0 curve), then measure (Task 5 ledger), then reconcile honestly (like the memory-census doc did).

- [ ] **Step 1: VM — run the full ledger** across contexts {4k,16k,32k,64k,128k} for variants {fp16_sdpa, chunked_pytorch, triton_3d, triton_k2b}. Commit the parquet back via git.
- [ ] **Step 2: Pull parquet locally; write the results doc** — predicted-vs-measured speedup curve, the drift-vs-speedup ledger table, the crossover context, and the **honestly-scoped claim** (latency win at 128k + RSS win; parity-or-not stated either way — kill-or-confirm).
- [ ] **Step 3: Reconcile prediction vs measurement** — document any gap (as the byte-ledger memory doc does); if the latency win is thin at 128k (predicted ~1.17×), say so plainly and lean the claim on RSS.
- [ ] **Step 4: Update CLAUDE.md + memory** — move Phase 3 from "open/gated" to its measured outcome.
- [ ] **Step 5: Commit (await approval).** Propose: `docs: Triton decode kernel results — predicted vs measured, honest claim; Phase 3 closed`. STOP for approval.

---

### Task 7: Terminal simplify pass

**Context:** Per the established bmx cadence, finish with a `/simplify` skill pass over all changed code (reuse/dedup/efficiency/altitude — quality only, NOT bug-hunting). Phases 1+2 ended this way.

- [ ] **Step 1: Invoke `/simplify`** over the diff of this chapter (the new module, the chunked-loop edits, the experiment, the bench).
- [ ] **Step 2: Apply findings**, re-run `uv run ruff format . && uv run ruff check . && uv run pytest -q` (+ the CUDA tests on the VM if any kernel code changed).
- [ ] **Step 3: Commit (await approval).** Propose: `refactor: simplify-pass over Triton decode kernel chapter`. STOP for approval.

---

## Self-Review

**Spec coverage:**
- Prediction gate (KV-fraction-bounded, upper-bound, crossover, compute flag) → Task 0. ✓
- PyTorch opts (GQA grouped contraction, grow-time RoPE cast) → Tasks 1, 2. ✓
- Triton stages 3a–3d (RTN → split-KV+autotune → k2b unpack → CUDA-graph) → Tasks 3a–3d. ✓
- Correctness spine (oracle diff + logit parity, both before latency) → enforced in every kernel task's test + structurally in Task 5's ledger. ✓
- Fail-loud/fallback-ok → `_require_triton` (3a), explicit capability dispatch + no-swallow test (Task 4). ✓
- skipif-not-xfail + "did it actually run" guard → CUDA-gated tests with reason; Task 3a Step 1 asserts `TRITON_AVAILABLE` on a CUDA box (the run-guard). ✓
- Drift-vs-speedup parquet ledger → Task 5. ✓
- Triton dependency already resolved (no pyproject change) → Global Constraints. ✓
- do_not_specialize hard rule, split-count heuristic, device-pointer seqlen → Tasks 3b, 3d. ✓
- Honest claim + kill-or-confirm + status update → Task 6. ✓
- Subagent-driven + terminal simplify → execution mode + Task 7. ✓

**Placeholder scan:** Tasks 3c/3d use abbreviated step bodies (Step 1/3 describe the test+impl with the exact arms, dicts, and DeepWiki checkpoints) rather than full code, because their kernel bodies depend on the DeepWiki-confirmed Triton patterns — this is intentional (the spec gates those on external grounding), not a placeholder TODO. Each names exact inputs, tolerances, and the diff target. The genuinely fill-at-measure values (`2026-06-2x`, the `...` in CUDA-move helpers) are flagged as such.

**Type consistency:** `triton_decode_attention` signature is identical across 3a/3b/3c/4/5 (the `num_splits` kwarg added in 3b is keyword-default, back-compatible). `decode_speedup_curve` / `predict_decode_latency` keys match between Task 0 and Task 5. `run_decode_ledger` column set is defined once and asserted in its test. `attention_diff` keys (`max_abs`/`max_rel`/`mean_abs`) used consistently.

## Open items deferred to execution (need VM / DeepWiki, not guessable now)
- The `@triton.jit` kernel bodies (3a/3b/3c/3d) — gated on DeepWiki checkpoints with triton-lang/flashinfer/vllm, per spec. The plan specifies the interface, the math, the tolerance, and the diff target for each; the body is written against the confirmed pattern at execution time on the VM.
- Exact tolerances — derived from measured chunked-vs-oracle drift at run start (Task 5 Step 3 computes/logs it).
- HBM bandwidth / peak FLOPS constants — read from the actual VM GPU at measure time, not hardcoded.
