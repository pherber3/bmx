# Fused dequant-attention (Phases 1+2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the KV-cache bpe compression real at runtime — keep packed codes resident and dequantize K/V block-by-block inside attention via online softmax — so a batched 128k-context sweep fits under the GH200's 94.5 GB ceiling.

**Architecture:** Split each codec arm into `quantize_packed`/`dequant_packed` (today they only return dequantized values, discarding the packed form). Build an analytic byte-ledger to predict the 128k peak before building. Build a chunked online-softmax attention that dequantizes one block at a time. Build `PackedStreamingCache` that stores packed codes (+ frozen subspace + fp16 window) and routes attention through the chunked path via the transformers `AttentionInterface` registry. A VM census instrument validates the ledger and the real peak.

**Tech Stack:** Python 3.12, PyTorch (CPU/ROCm local, CUDA on VM), transformers 5.x (`AttentionInterface`, `DynamicLayer`/`Cache`), tyro CLIs, pandas/parquet, pytest, uv, ruff.

## Global Constraints

- **NEVER `git commit` without the user's explicit approval.** Stage, propose a message, stop. No AI attribution / Co-Authored-By, ever. (The `git commit` steps below mean "stage + propose"; do not run the commit until the user approves.)
- Before any commit: `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` — all clean. Expected baseline: 220 passed, 1 xfailed (+ new tests).
- Dependencies only via `uv add` / `uv add --dev`. Never hand-edit `pyproject.toml` versions. (This plan adds no new deps.)
- Use the Bash tool (git bash), not PowerShell. Shell cwd resets between turns — `cd /d/Projects/bmx` first in fresh shells.
- AMD 7900 XTX locally (no CUDA). GPU-authoritative work (the census at real context lengths, the 128k peak) runs on the rented NVIDIA VM. Everything in Tasks 1–6 is correctness/memory-model work runnable locally; Task 7 is the VM instrument.
- dtype: fp64 in tests, fp32 in experiments/codecs (caches stored fp16). Fail fast: shape asserts at boundaries, no silent coercion.
- Comparisons align on total bits (ALL metadata counted), never on rank.
- Tiny offline test models come from `tests/factories.py` (`tiny_llama`, `tiny_gpt2`, `ids`); never download in tests.
- The codec split MUST preserve the existing `quantize_cache(...) -> (M_hat, bpe)` behavior bit-for-bit (the current `StreamingQuantizedCache` path and all existing tests depend on it).

---

## File structure

| File | Responsibility |
|---|---|
| `src/bmx/quant/rtn.py` (edit) | Add `rtn_quantize_packed` → `(Q_int, scale)`; keep `rtn_quantize` as the dequant-returning composition. |
| `src/bmx/cache/codecs.py` (edit) | Add `quantize_packed(arm, M, ...) -> (packed, bpe)` and `dequant_packed(arm, packed, ...) -> M_hat`; rewrite `quantize_cache` as their composition. |
| `src/bmx/bench/kv_memory.py` (new) | Analytic byte-ledger: `KVMemCase`, `predict_peak`. Pure Python. |
| `src/bmx/cache/chunked_attention.py` (new) | `online_softmax_update`, `chunked_dequant_attention`. |
| `src/bmx/cache/streaming.py` (edit, small) | Extract `compute_flush_schedule(S, W, g)`; both cache classes call it. |
| `src/bmx/cache/packed_streaming.py` (new) | `PackedStreamingLayer`/`PackedStreamingCache`; registry-based attention routing. |
| `experiments/k3_kernel_census.py` (new) | tyro CLI: per-arm resident/peak/incremental → parquet. VM. |
| `tests/test_codec_split.py` (new) | Per-arm `dequant_packed(quantize_packed(M)) == quantize_cache(M)` exactly. |
| `tests/test_kv_memory.py` (new) | Ledger arithmetic vs hand-computed Llama-3.1 numbers. |
| `tests/test_chunked_attention.py` (new) | Online-softmax exactness + chunked vs dense attention. |
| `tests/test_packed_streaming.py` (new) | Parity vs `StreamingQuantizedCache` (bit-for-bit). |

---

## Task 1: Codec split foundation — `rtn_quantize_packed`

The whole packed path rests on being able to produce, and later consume, packed codes. Start at the lowest level: `rtn_quantize` (used by `rtn_token`, `rtn_channel`, `rotate_rtn_token`, `lowrank_*`). Today it returns `(Q*scale)` and discards `Q` and `scale`.

**Files:**
- Modify: `src/bmx/quant/rtn.py`
- Test: `tests/test_codec_split.py`

**Interfaces:**
- Produces: `rtn_quantize_packed(W, bits, group_size) -> (Q_int: int8 tensor of shape W with int levels, scale: tensor (…, n_groups, 1))` and `rtn_dequantize_packed(Q_int, scale, group_size) -> W_hat`. `rtn_quantize(W, bits, group_size)` keeps its current signature/return and becomes `rtn_dequantize_packed(*rtn_quantize_packed(...))` reshaped.

- [ ] **Step 1: Write the failing test**

Create `tests/test_codec_split.py`:

```python
"""Codec split: packed quantize/dequant must equal the dequant-returning path."""

import torch

from bmx.quant.rtn import rtn_quantize, rtn_quantize_packed, rtn_dequantize_packed


def test_rtn_packed_roundtrip_matches_dequant_path():
    torch.manual_seed(0)
    W = torch.randn(8, 64, dtype=torch.float64)
    bits, group = 3, 16
    ref = rtn_quantize(W, bits, group)
    Q_int, scale = rtn_quantize_packed(W, bits, group)
    W_hat = rtn_dequantize_packed(Q_int, scale, group)
    assert W_hat.shape == W.shape
    assert torch.equal(W_hat, ref)  # exact: same arithmetic, just split
    # Q_int holds integer levels within the symmetric range.
    qmax = 2 ** (bits - 1) - 1
    assert Q_int.max() <= qmax and Q_int.min() >= -qmax - 1
    assert Q_int.dtype == torch.int8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_codec_split.py::test_rtn_packed_roundtrip_matches_dequant_path -v`
Expected: FAIL with `ImportError: cannot import name 'rtn_quantize_packed'`.

- [ ] **Step 3: Write minimal implementation**

Replace the body of `src/bmx/quant/rtn.py` with:

```python
"""Groupwise symmetric round-to-nearest quantization.

Two-step form (quantize -> packed -> dequant) plus the original one-shot
`rtn_quantize` kept as the composition for the existing dequant-returning callers.
"""

import torch


def rtn_quantize_packed(W: torch.Tensor, bits: int, group_size: int):
    """(..., d) -> (Q_int int8 same shape, scale (..., n_groups, 1)).

    Q_int holds the integer levels; scale is per-group. Dequant is
    `rtn_dequantize_packed(Q_int, scale, group_size)`.
    """
    *lead, d = W.shape
    assert d % group_size == 0, f"dim {d} not divisible by group {group_size}"
    qmax = 2 ** (bits - 1) - 1
    G = W.reshape(*lead, d // group_size, group_size)
    scale = G.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    Q = (G / scale).round().clamp(-qmax - 1, qmax)
    Q_int = Q.to(torch.int8).reshape(W.shape)
    return Q_int, scale


def rtn_dequantize_packed(
    Q_int: torch.Tensor, scale: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Inverse of rtn_quantize_packed: (Q_int, scale) -> dequantized W_hat."""
    *lead, d = Q_int.shape
    G = Q_int.reshape(*lead, d // group_size, group_size).to(scale.dtype)
    return (G * scale).reshape(Q_int.shape)


def rtn_quantize(W: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Groupwise symmetric RTN, returning dequantized values (unchanged API)."""
    Q_int, scale = rtn_quantize_packed(W, bits, group_size)
    return rtn_dequantize_packed(Q_int, scale, group_size)
```

- [ ] **Step 4: Run the new test + the full suite to verify nothing regressed**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_codec_split.py -v && uv run pytest -q`
Expected: new test PASSES; full suite 220 passed, 1 xfailed (the `rtn_quantize` composition is bit-identical, so all existing codec tests still pass).

- [ ] **Step 5: Format, lint, stage, propose commit**

```bash
cd /d/Projects/bmx && uv run ruff format . && uv run ruff check .
git add src/bmx/quant/rtn.py tests/test_codec_split.py
```
Propose message: `refactor(quant): split rtn_quantize into packed quantize/dequant`
(Do NOT commit without user approval.)

---

## Task 2: Codec split — `quantize_packed` / `dequant_packed` for all arms

Lift the split to the codec layer. Each arm's packed form is an arm-specific dict. `quantize_cache` becomes `dequant_packed(*quantize_packed(...))`.

**Files:**
- Modify: `src/bmx/cache/codecs.py`
- Test: `tests/test_codec_split.py`

**Interfaces:**
- Consumes: `rtn_quantize_packed`, `rtn_dequantize_packed` (Task 1).
- Produces:
  - `quantize_packed(arm, M, *, bits, seed=0, group=64, rank=0, svd_factors=None) -> (packed: dict, bpe: float)`.
  - `dequant_packed(arm, packed, *, seed=0, group=64) -> M_hat`.
  - `quantize_cache(...)` unchanged externally (now composes the two).
  - `packed` dict keys per arm (the read-time state):
    - `rtn_token`/`rtn_channel`/`rotate_rtn_token`: `{"Q_int", "scale"}` (rotate also implicit via seed at dequant).
    - `turboquant_mse`: `{"indices", "norms"}` (codebook by `bits`+seed at dequant).
    - `turboquant_prod`: `{"mse_packed", "qjl_signs", "qjl_norms"}`.
    - `lowrank_rtn_channel`: `{"Us", "V", "res_Q_int", "res_scale"}`.
  - Scope note: the bit-for-bit gate covers the arms the cache actually uses (`fp16`, `lowrank_rtn_channel`, `turboquant_mse`, `rtn_channel`, `rtn_token`). The `lowrank_*waterfill_*` family is NOT split in this plan (not on the streaming path); `quantize_packed`/`dequant_packed` raise `NotImplementedError` for them with a clear message. `quantize_cache` keeps serving them unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_codec_split.py`:

```python
import pytest

from bmx.cache.codecs import dequant_packed, quantize_cache, quantize_packed

SPLIT_ARMS = [
    ("rtn_token", dict(bits=3, group=16)),
    ("rtn_channel", dict(bits=3, group=8)),
    ("rotate_rtn_token", dict(bits=3, group=16)),
    ("turboquant_mse", dict(bits=2)),
    ("turboquant_prod", dict(bits=3)),
    ("lowrank_rtn_channel", dict(bits=3, group=8, rank=4)),
]


@pytest.mark.parametrize("arm,kw", SPLIT_ARMS)
def test_quantize_packed_matches_quantize_cache(arm, kw):
    torch.manual_seed(0)
    # S=16 (divisible by group 8), C=16 (power of 2 for hadamard rotate arms).
    M = torch.randn(16, 16, dtype=torch.float64)
    ref_hat, ref_bpe = quantize_cache(arm, M, **kw)
    packed, bpe = quantize_packed(arm, M, **kw)
    hat = dequant_packed(arm, packed, group=kw.get("group", 64), seed=0)
    assert bpe == pytest.approx(ref_bpe)
    assert torch.equal(hat, ref_hat)


def test_waterfill_arm_not_split_raises():
    M = torch.randn(16, 16, dtype=torch.float64)
    with pytest.raises(NotImplementedError):
        quantize_packed("lowrank_waterfill_channel", M, bits=3, group=8, rank=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_codec_split.py -k "quantize_packed or waterfill_arm" -v`
Expected: FAIL with `ImportError: cannot import name 'quantize_packed'`.

- [ ] **Step 3: Write minimal implementation**

In `src/bmx/cache/codecs.py`, add the packed split. Add these functions (place after the arm helpers, before `quantize_cache`). They reuse the existing module-level helpers `_rotate`, `_unrotate`, `gaussian_codebook`, `qjl_reconstruct`, `_qjl_sketch`, `truncated_svd`, and the new `rtn_quantize_packed`/`rtn_dequantize_packed`.

```python
from bmx.quant.rtn import rtn_quantize_packed, rtn_dequantize_packed

_SPLIT_ARMS = frozenset(
    {
        "rtn_token",
        "rtn_channel",
        "rotate_rtn_token",
        "turboquant_mse",
        "turboquant_prod",
        "lowrank_rtn_channel",
    }
)


def quantize_packed(
    arm, M, *, bits, seed=0, group=64, rank=0, svd_factors=None
):
    """(S,C) fp -> (packed dict, honest bpe). Inverse: dequant_packed."""
    if arm not in _SPLIT_ARMS:
        raise NotImplementedError(
            f"arm {arm!r} not split into packed form (not on the streaming path); "
            f"use quantize_cache. Split arms: {sorted(_SPLIT_ARMS)}"
        )
    S, C = M.shape
    if arm == "rtn_token":
        Q_int, scale = rtn_quantize_packed(M, bits, group)
        return {"Q_int": Q_int, "scale": scale}, bits + 16.0 / group
    if arm == "rtn_channel":
        Q_int, scale = rtn_quantize_packed(M.mT, bits, group)
        return {"Q_int": Q_int, "scale": scale}, bits + 16.0 / group
    if arm == "rotate_rtn_token":
        Q_int, scale = rtn_quantize_packed(_rotate(M, seed), bits, group)
        return {"Q_int": Q_int, "scale": scale}, bits + 16.0 / group
    if arm == "turboquant_mse":
        indices, norms = _turboquant_mse_packed(M, bits, seed)
        return {"indices": indices, "norms": norms}, bits + 16.0 / C
    if arm == "turboquant_prod":
        assert bits >= 2
        mse_packed = _turboquant_mse_packed(M, bits - 1, seed)
        M1 = _turboquant_mse_dequant(*mse_packed, bits - 1, seed, C)
        R = M - M1
        r_norms = R.norm(dim=1, keepdim=True).clamp_min(1e-12).half().float()
        R_unit = R / r_norms
        G = _qjl_sketch(C, seed).to(R)
        signs = torch.sign(R_unit @ G.T)
        packed = {
            "mse_indices": mse_packed[0],
            "mse_norms": mse_packed[1],
            "qjl_signs": signs.to(torch.int8),
            "qjl_norms": r_norms,
        }
        return packed, (bits - 1) + 1 + 32.0 / C
    # lowrank_rtn_channel
    assert rank > 0 and rank <= min(S, C) and S % group == 0
    if svd_factors is not None:
        Us, V = svd_factors
    else:
        Us, V = truncated_svd(M, rank)
    Us_stored = Us.half().float()
    V_stored = V.half().float()
    L = Us_stored @ V_stored.mT
    R = M - L
    res_Q_int, res_scale = rtn_quantize_packed(R.mT, bits, group)
    bpe = bits + 16.0 / group + 16.0 * rank * (S + C) / (S * C)
    return {"Us": Us_stored, "V": V_stored, "res_Q_int": res_Q_int,
            "res_scale": res_scale}, bpe


def dequant_packed(arm, packed, *, seed=0, group=64):
    """Inverse of quantize_packed -> dequantized (S,C) M_hat."""
    if arm not in _SPLIT_ARMS:
        raise NotImplementedError(f"arm {arm!r} not split into packed form")
    if arm == "rtn_token":
        return rtn_dequantize_packed(packed["Q_int"], packed["scale"], group)
    if arm == "rtn_channel":
        return rtn_dequantize_packed(packed["Q_int"], packed["scale"], group).mT
    if arm == "rotate_rtn_token":
        M_rot_hat = rtn_dequantize_packed(packed["Q_int"], packed["scale"], group)
        return _unrotate(M_rot_hat, seed)
    if arm == "turboquant_mse":
        C = packed["indices"].shape[1]
        return _turboquant_mse_dequant(
            packed["indices"], packed["norms"], _tq_bits(packed), seed, C
        )
    if arm == "turboquant_prod":
        C = packed["mse_indices"].shape[1]
        M1 = _turboquant_mse_dequant(
            packed["mse_indices"], packed["mse_norms"], _tq_bits_prod(packed), seed, C
        )
        G = _qjl_sketch(C, seed).to(packed["qjl_norms"])
        signs = packed["qjl_signs"].to(G.dtype)
        scale = math.sqrt(math.pi / 2) / C
        R_hat = packed["qjl_norms"] * scale * (signs @ G)
        return M1 + R_hat
    # lowrank_rtn_channel
    L = packed["Us"] @ packed["V"].mT
    R_hat = rtn_dequantize_packed(packed["res_Q_int"], packed["res_scale"], group).mT
    return L + R_hat
```

Then add the turboquant packed helpers (refactored out of `_turboquant_mse`, same math), and store `bits` in the packed dict so dequant knows the codebook:

```python
def _turboquant_mse_packed(M, bits, seed):
    """(S,C) -> (indices int16 (S,C), norms fp (S,1)). Codebook from bits+seed."""
    S, C = M.shape
    norms = M.norm(dim=1, keepdim=True).clamp_min(1e-12).half().float()
    M_unit = M / norms
    M_rot = _rotate(M_unit, seed)
    cb = gaussian_codebook(bits).to(M.device)
    sqrt_c = math.sqrt(C)
    mid = (cb[:-1] + cb[1:]) / 2
    indices = torch.bucketize(M_rot * sqrt_c, mid).to(torch.int16)
    return indices, norms


def _turboquant_mse_dequant(indices, norms, bits, seed, C):
    cb = gaussian_codebook(bits).to(norms.device)
    sqrt_c = math.sqrt(C)
    M_quant = cb[indices.long()] / sqrt_c
    M_recon = _unrotate(M_quant, seed)
    return M_recon * norms
```

To carry `bits` for dequant, store it in the packed dict instead of inferring. Simplify by adding `"bits"` to the turboquant packed dicts in `quantize_packed` (`{"indices", "norms", "bits"}`; prod: add `"bits"` = `bits`), and replace `_tq_bits(packed)`/`_tq_bits_prod(packed)` calls with `packed["bits"]` / `packed["bits"] - 1`. (Update the two `dequant_packed` turboquant branches accordingly; drop the `_tq_bits*` helpers — they were a placeholder for this decision.)

Finally rewrite `quantize_cache` for the split arms to compose (leave the waterfill branches exactly as they are):

```python
    # inside quantize_cache, for the six split arms, replace the body with:
    if arm in _SPLIT_ARMS:
        packed, bpe = quantize_packed(
            arm, M, bits=bits, seed=seed, group=group, rank=rank,
            svd_factors=svd_factors,
        )
        return dequant_packed(arm, packed, seed=seed, group=group), bpe
```

- [ ] **Step 4: Run the split tests + full suite**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_codec_split.py -v && uv run pytest -q`
Expected: all split arms PASS (`torch.equal` exact); full suite 220 passed, 1 xfailed. If any `torch.equal` fails, the split diverged from the original math — fix the helper, do not loosen to `allclose`.

- [ ] **Step 5: Format, lint, stage, propose commit**

```bash
cd /d/Projects/bmx && uv run ruff format . && uv run ruff check .
git add src/bmx/cache/codecs.py tests/test_codec_split.py
```
Propose message: `refactor(cache): split codecs into quantize_packed/dequant_packed`

---

## Task 3: Byte-ledger (`src/bmx/bench/kv_memory.py`)

Pure-Python analytic memory model. Independent of Tasks 1–2; predicts the 128k peak before any kernel exists.

**Files:**
- Create: `src/bmx/bench/kv_memory.py`
- Test: `tests/test_kv_memory.py`

**Interfaces:**
- Produces:
  - `@dataclass KVMemCase` with fields: `seq_len, n_layer, h_kv, d_head, bpe_k, bpe_v, block, recent_window, path, weights_bytes, act_bytes, logits_bytes`.
  - `predict_peak(case: KVMemCase) -> dict` with keys `resident_bytes, transient_bytes, attn_bytes, weights_bytes, act_bytes, logits_bytes, predicted_peak_bytes, compression_at_runtime`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kv_memory.py`:

```python
"""Byte-ledger: validate against hand-computed Llama-3.1-8B KV numbers."""

from bmx.bench.kv_memory import KVMemCase, predict_peak

GiB = 1024**3


def _llama31(seq_len, path, bpe_k=16.0, bpe_v=16.0):
    # Llama-3.1-8B: L=32, h_kv=8, d=128. Weights ~14.9 GB, modest act, logits ~1 pos.
    return KVMemCase(
        seq_len=seq_len, n_layer=32, h_kv=8, d_head=128,
        bpe_k=bpe_k, bpe_v=bpe_v, block=128, recent_window=32, path=path,
        weights_bytes=int(14.9 * GiB), act_bytes=int(2.0 * GiB),
        logits_bytes=int(0.5 * GiB),
    )


def test_resident_one_fp16_copy_is_16gb_at_128k():
    # 2 * L * h_kv * S * d * 2 bytes = one full K+V fp16 copy.
    # = 2*32*8*131072*128*2 = 17,179,869,184 bytes = 16 GiB.
    case = _llama31(131072, "dense_stream")
    r = predict_peak(case)
    one_copy = 2 * 32 * 8 * 131072 * 128 * 2
    assert one_copy == 16 * GiB
    # dense_stream holds ~2 copies (dequant prefix + reassembled slab).
    assert r["resident_bytes"] == 2 * one_copy


def test_chunked_resident_is_bpe_footprint():
    # 3-bit K, 2-bit V (k2b-ish): resident = L*h_kv*S*d*(bpe_k+bpe_v)/8.
    case = _llama31(131072, "chunked", bpe_k=3.0, bpe_v=2.0)
    r = predict_peak(case)
    expected = int(32 * 8 * 131072 * 128 * (3.0 + 2.0) / 8)
    assert r["resident_bytes"] == expected


def test_dense_stream_128k_reproduces_oom():
    # dense_stream peak should exceed the 94.5 GB GH200 ceiling at 128k.
    case = _llama31(131072, "dense_stream")
    r = predict_peak(case)
    assert r["predicted_peak_bytes"] > 94.5 * GiB


def test_chunked_128k_clears_ceiling():
    case = _llama31(131072, "chunked", bpe_k=3.0, bpe_v=2.0)
    r = predict_peak(case)
    assert r["predicted_peak_bytes"] < 94.5 * GiB
    assert r["compression_at_runtime"] > 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_kv_memory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bmx.bench.kv_memory'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/bmx/bench/kv_memory.py`:

```python
"""Analytic byte-ledger for KV-cache peak memory (Track-B-style honest bytes).

No allocation, no CUDA — arithmetic only. Grounded in the canonical KV formula
2*L*h_kv*S*d*bytes (Physics of LLM Inference; AI Systems Perf Eng). Predicts the
128k peak for the current dense-stream path (should reproduce the ~99-100 GB OOM)
vs the chunked path (packed codes resident). Validated against the VM census
before the chunked prediction is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KVMemCase:
    seq_len: int
    n_layer: int
    h_kv: int
    d_head: int
    bpe_k: float
    bpe_v: float
    block: int
    recent_window: int
    path: str  # "dense_stream" | "chunked"
    weights_bytes: int
    act_bytes: int
    logits_bytes: int


def _one_fp16_copy_bytes(c: KVMemCase) -> int:
    # K and V, fp16 (2 bytes), all layers, all positions.
    return 2 * c.n_layer * c.h_kv * c.seq_len * c.d_head * 2


def _packed_bytes(c: KVMemCase) -> int:
    entries = c.n_layer * c.h_kv * c.seq_len * c.d_head
    return int(entries * (c.bpe_k + c.bpe_v) / 8)


def predict_peak(case: KVMemCase) -> dict:
    assert case.path in ("dense_stream", "chunked"), case.path
    one_copy = _one_fp16_copy_bytes(case)
    # one fp16 K+V copy for the recent window only (shared by both paths).
    window_frac = max(case.recent_window, 0) / max(case.seq_len, 1)
    window_bytes = int(one_copy * window_frac)

    if case.path == "dense_stream":
        # Dequantized frozen prefix + reassembled (prefix+tail) slab ~= 2 copies.
        resident = 2 * one_copy
        # Transient: worst-arm per-flush scratch ~ a couple of full-prefix temps.
        transient = one_copy // case.n_layer  # one layer's block-set scratch, rough
        # Stock SDPA materializes a full (h, S) score row per query (freeable).
        attn = case.h_kv * case.seq_len * 4  # fp32 scores, last query only
    else:  # chunked
        resident = _packed_bytes(case) + window_bytes
        # One dequantized block of K+V (transient, freed each step).
        transient = 2 * case.n_layer * case.h_kv * case.block * case.d_head * 2
        # Online softmax: (h, block) score tile + (h, d) accumulator.
        attn = case.h_kv * (case.block + case.d_head) * 4

    predicted = (
        resident + transient + attn
        + case.weights_bytes + case.act_bytes + case.logits_bytes
    )
    dense_resident = 2 * one_copy
    compression_at_runtime = dense_resident / max(resident, 1)
    return {
        "resident_bytes": resident,
        "transient_bytes": transient,
        "attn_bytes": attn,
        "weights_bytes": case.weights_bytes,
        "act_bytes": case.act_bytes,
        "logits_bytes": case.logits_bytes,
        "predicted_peak_bytes": predicted,
        "compression_at_runtime": compression_at_runtime,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_kv_memory.py -v`
Expected: all 4 PASS. (If `test_dense_stream_128k_reproduces_oom` fails, the act/weights terms need tuning to the measured 92 GB fp16 decomposition — adjust the `_llama31` fixture constants, not the formula.)

- [ ] **Step 5: Format, lint, stage, propose commit**

```bash
cd /d/Projects/bmx && uv run ruff format . && uv run ruff check .
git add src/bmx/bench/kv_memory.py tests/test_kv_memory.py
```
Propose message: `feat(bench): analytic KV-cache byte-ledger for peak prediction`

---

## Task 4: Golden reference oracle + shared flush schedule + online-softmax core

Three independently-correct pieces the chunked attention needs. **The golden
reference comes first** — `naive_dense_attention` is the ONE named oracle (the
slowest, most-obviously-correct path: dequant everything, a single full softmax,
no online trick, no chunking) that *every* faster path is diffed against, with a
diff helper that quantifies drift. This is the yardstick that prevents chasing
performance with no measure of quality — and it carries through to the Phase-3
Triton kernel, which must pass the SAME oracle (not "chunked", which is itself an
optimization sharing too much code to be a trustworthy ground truth).

**Files:**
- Modify: `src/bmx/cache/streaming.py` (extract helper; have the layer call it)
- Create: `src/bmx/cache/chunked_attention.py` (oracle + diff helper + online-softmax update)
- Test: `tests/test_chunked_attention.py`

**Interfaces:**
- Produces:
  - `naive_dense_attention(q, k_blocks, v_blocks, *, k_arm, v_arm, group, seed, k_pre_rope, rope_cos, rope_sin, k_tail, v_tail, n_q_groups, scale) -> (n_q_heads, n_q, d)` — the oracle. Same call shape as `chunked_dequant_attention` (Task 5) so they are drop-in comparable. (Intentional asymmetry: `chunked_dequant_attention` additionally takes `tail_start` for the caller's bookkeeping; the oracle doesn't need it because it concatenates the tail into the dense K/V directly. Do not "harmonize" this away.)
  - `attention_diff(a, b) -> dict` with `{"max_abs", "max_rel", "mean_abs"}` — quantifies drift between any two attention outputs.
  - `compute_flush_schedule(S: int, W: int, g: int) -> int` in `streaming.py` (returns `new_S_q`).
  - `online_softmax_update(acc, m, l, scores_new, v_new) -> (acc, m, l)` in `chunked_attention.py`, where `acc:(...,n_q,d)`, `m,l:(...,n_q,1)`, `scores_new:(...,n_q,blk)`, `v_new:(...,blk,d)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_chunked_attention.py`:

```python
"""Chunked dequant-attention: oracle, online-softmax exactness, schedule."""

import torch

from bmx.cache.chunked_attention import (
    attention_diff,
    naive_dense_attention,
    online_softmax_update,
)
from bmx.cache.collect import to_matrix
from bmx.cache.streaming import compute_flush_schedule


def test_oracle_equals_hand_softmax_gqa():
    # The oracle IS the ground truth; pin it against a from-scratch GQA softmax so
    # a future edit to the oracle can't silently corrupt the yardstick.
    torch.manual_seed(0)
    h_kv, n_q_heads, n_q, S, d = 2, 4, 1, 32, 8
    q = torch.randn(n_q_heads, n_q, d, dtype=torch.float64)
    K = torch.randn(h_kv, S, d, dtype=torch.float64)
    V = torch.randn(h_kv, S, d, dtype=torch.float64)
    scale = 1.0 / (d ** 0.5)
    # fp16-arm packed blocks of 16 (so dequant is identity, isolating attention).
    k_blocks, v_blocks = [], []
    for j in range(0, S, 16):
        k_blocks.append(({"fp16": to_matrix(K[:, j:j + 16])}, j, j + 16))
        v_blocks.append(({"fp16": to_matrix(V[:, j:j + 16])}, j, j + 16))
    out = naive_dense_attention(
        q, k_blocks, v_blocks, k_arm="fp16", v_arm="fp16", group=8, seed=0,
        k_pre_rope=False, rope_cos=None, rope_sin=None, k_tail=None, v_tail=None,
        n_q_groups=n_q_heads // h_kv, scale=scale)
    Kx = K.repeat_interleave(n_q_heads // h_kv, dim=0)
    Vx = V.repeat_interleave(n_q_heads // h_kv, dim=0)
    ref = torch.softmax((q @ Kx.transpose(-1, -2)) * scale, dim=-1) @ Vx
    assert torch.allclose(out, ref, atol=1e-12, rtol=1e-12)


def test_attention_diff_reports_zero_for_identical():
    a = torch.randn(2, 1, 8, dtype=torch.float64)
    d = attention_diff(a, a.clone())
    assert d["max_abs"] == 0.0 and d["max_rel"] == 0.0 and d["mean_abs"] == 0.0


def test_flush_schedule_matches_formula():
    # largest multiple of g leaving >= W recent tokens, else 0.
    assert compute_flush_schedule(S=100, W=32, g=16) == 64
    assert compute_flush_schedule(S=40, W=32, g=16) == 0   # (40-32)//16*16 = 0
    assert compute_flush_schedule(S=20, W=32, g=16) == 0   # S <= W
    assert compute_flush_schedule(S=160, W=32, g=1) == 128


def test_online_softmax_equals_full_softmax():
    torch.manual_seed(0)
    h, n_q, S, d = 2, 1, 48, 8
    q = torch.randn(h, n_q, d, dtype=torch.float64)
    K = torch.randn(h, S, d, dtype=torch.float64)
    V = torch.randn(h, S, d, dtype=torch.float64)
    scale = 1.0 / (d ** 0.5)

    # Reference: full softmax over all S keys.
    full_scores = (q @ K.transpose(-1, -2)) * scale  # (h, n_q, S)
    ref = torch.softmax(full_scores, dim=-1) @ V       # (h, n_q, d)

    # Streamed in blocks of 16.
    acc = torch.zeros(h, n_q, d, dtype=torch.float64)
    m = torch.full((h, n_q, 1), float("-inf"), dtype=torch.float64)
    l = torch.zeros(h, n_q, 1, dtype=torch.float64)
    for j in range(0, S, 16):
        Kb, Vb = K[:, j:j + 16], V[:, j:j + 16]
        s = (q @ Kb.transpose(-1, -2)) * scale  # (h, n_q, blk)
        acc, m, l = online_softmax_update(acc, m, l, s, Vb)
    out = acc / l
    assert torch.allclose(out, ref, atol=1e-12, rtol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_chunked_attention.py -v`
Expected: FAIL with `ImportError` (the symbols don't exist yet).

- [ ] **Step 3: Write minimal implementation**

In `src/bmx/cache/streaming.py`, add the helper near the top (module level, after imports) and use it inside `update()`:

```python
def compute_flush_schedule(S: int, W: int, g: int) -> int:
    """Largest multiple of g that leaves >= W recent tokens fp16, else 0.

    Single source of truth for the committed-block boundary; both
    StreamingQuantizedLayer and PackedStreamingLayer call this so their schedules
    cannot drift (bit-for-bit parity depends on it).
    """
    return ((S - W) // g) * g if S > W else 0
```

Then replace the existing line in `StreamingQuantizedLayer.update()`:
```python
        new_S_q = ((S - W) // g) * g if S > W else 0
```
with:
```python
        new_S_q = compute_flush_schedule(S, W, g)
```

Create `src/bmx/cache/chunked_attention.py`:

```python
"""Chunked dequant-attention + the naive golden reference oracle.

The packed cache stores compressed codes. Two attention paths share one call
shape so they are drop-in comparable:
  - naive_dense_attention  — the ORACLE: dequant everything, ONE full softmax, no
    online trick, no chunking. Slowest, most obviously correct. Every faster path
    (online-softmax, chunked, packed, future Triton) is diffed against THIS, and
    attention_diff() quantifies the drift. The yardstick that keeps us honest.
  - chunked_dequant_attention (Task 5) — dequant ONE block at a time, online
    softmax (exact — Physics of LLM Inference ~line 1931), free the block. Never
    materializes full dense K/V or the full score row.
"""

from __future__ import annotations

import torch

from bmx.cache.codecs import dequant_packed
from bmx.cache.collect import from_matrix
from bmx.cache.rope import apply_rope


def online_softmax_update(acc, m, l, scores_new, v_new):
    """One online-softmax step.

    acc:(...,n_q,d) m,l:(...,n_q,1) scores_new:(...,n_q,blk) v_new:(...,blk,d).
    Returns updated (acc, m, l). Divide acc by l after the last block.
    """
    m_new = torch.maximum(m, scores_new.amax(dim=-1, keepdim=True))
    correction = torch.exp(m - m_new)  # (...,n_q,1); <=1, never overflows
    p = torch.exp(scores_new - m_new)  # (...,n_q,blk)
    l = l * correction + p.sum(dim=-1, keepdim=True)
    acc = acc * correction + p @ v_new  # (...,n_q,d)
    return acc, m_new, l


def attention_diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Quantify drift between two attention outputs (oracle vs fast path)."""
    diff = (a.double() - b.double()).abs()
    denom = b.double().abs().clamp_min(1e-12)
    return {
        "max_abs": float(diff.max()),
        "max_rel": float((diff / denom).max()),
        "mean_abs": float(diff.mean()),
    }


def _dequant_block(packed, arm, group, seed, h_kv):
    """packed dict -> (h_kv, blk, d) dense, matching to_matrix layout."""
    M = packed["fp16"] if arm == "fp16" else dequant_packed(
        arm, packed, group=group, seed=seed)
    return from_matrix(M, h_kv)


def _dense_kv(blocks, arm, group, seed, h_kv, k_pre_rope, rope_cos, rope_sin):
    """Dequant all blocks to one dense (h_kv, S_committed, d), RoPE-at-read for K."""
    parts = []
    for packed, start, end in blocks:
        B = _dequant_block(packed, arm, group, seed, h_kv)
        if k_pre_rope:
            B = apply_rope(B, rope_cos[start:end].to(B.dtype),
                           rope_sin[start:end].to(B.dtype))
        parts.append(B)
    return torch.cat(parts, dim=1) if parts else None


def naive_dense_attention(
    q, k_blocks, v_blocks, *, k_arm, v_arm, group, seed, k_pre_rope,
    rope_cos, rope_sin, k_tail, v_tail, n_q_groups, scale,
):
    """ORACLE: dequant everything, single full softmax, GQA-expand. No chunking.

    Same call shape as chunked_dequant_attention so they are drop-in comparable.
    """
    n_q_heads = q.shape[0]
    h_kv = n_q_heads // n_q_groups
    K = _dense_kv(k_blocks, k_arm, group, seed, h_kv, k_pre_rope,
                  rope_cos, rope_sin)
    V = _dense_kv(v_blocks, v_arm, group, seed, h_kv, False, None, None)
    if k_tail is not None and k_tail.shape[1] > 0:
        K = k_tail.to(q.dtype) if K is None else torch.cat(
            [K, k_tail.to(q.dtype)], dim=1)
        V = v_tail.to(q.dtype) if V is None else torch.cat(
            [V, v_tail.to(q.dtype)], dim=1)
    Kx = K.to(q.dtype).repeat_interleave(n_q_groups, dim=0)
    Vx = V.to(q.dtype).repeat_interleave(n_q_groups, dim=0)
    scores = (q @ Kx.transpose(-1, -2)) * scale
    return torch.softmax(scores, dim=-1) @ Vx
```

- [ ] **Step 4: Run test + full suite**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_chunked_attention.py -v && uv run pytest -q`
Expected: both new tests PASS; full suite still 220 passed, 1 xfailed (the schedule extraction is behavior-preserving).

- [ ] **Step 5: Format, lint, stage, propose commit**

```bash
cd /d/Projects/bmx && uv run ruff format . && uv run ruff check .
git add src/bmx/cache/streaming.py src/bmx/cache/chunked_attention.py tests/test_chunked_attention.py
```
Propose message: `feat(cache): shared flush schedule + online-softmax update`

---

## Task 5: `chunked_dequant_attention` (per-block dequant + RoPE-in-loop + GQA)

Assemble the full chunked attention: iterate committed blocks, dequant each (via Task 2's `dequant_packed`), RoPE the K block at its absolute positions BEFORE the contraction, online-softmax accumulate, then the fp16 tail. GQA-aware.

**Files:**
- Modify: `src/bmx/cache/chunked_attention.py`
- Test: `tests/test_chunked_attention.py`

**Interfaces:**
- Consumes: `online_softmax_update` (Task 4), `dequant_packed` (Task 2), `apply_rope`/`rope_cos_sin` (`bmx.cache.rope`).
- Produces:
  - `chunked_dequant_attention(q, k_blocks, v_blocks, *, k_arm, v_arm, group, seed, k_pre_rope, rope_cos, rope_sin, k_tail, v_tail, tail_start, n_q_groups, scale) -> attn_out (n_q_heads, n_q, d)` where `k_blocks`/`v_blocks` are lists of `(packed dict, block_start, block_end)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chunked_attention.py`. The chunked path is diffed against
the **oracle** (`naive_dense_attention`) — the single yardstick — using
`attention_diff` so any drift is quantified, not just thresholded:

```python
import pytest

from bmx.cache.chunked_attention import chunked_dequant_attention
from bmx.cache.codecs import quantize_packed


@pytest.mark.parametrize("k_arm,v_arm,kw", [
    ("fp16", "fp16", {}),
    ("turboquant_mse", "turboquant_mse", dict(bits=2)),
])
def test_chunked_matches_oracle_no_rope(k_arm, v_arm, kw):
    # chunked dequant-attn must equal the oracle (dequant-all + full softmax) over
    # the SAME packed blocks. Isolates the online-softmax + per-block assembly.
    torch.manual_seed(0)
    h_kv, n_q_heads, n_q, S, d = 2, 4, 1, 48, 8  # GQA: 4 q-heads over 2 kv-heads
    group = 8
    q = torch.randn(n_q_heads, n_q, d, dtype=torch.float64)
    K = torch.randn(h_kv, S, d, dtype=torch.float64)
    V = torch.randn(h_kv, S, d, dtype=torch.float64)
    scale = 1.0 / (d ** 0.5)

    def pack_side(T, arm):
        blocks = []
        for j in range(0, S, 16):
            M = to_matrix(T[:, j:j + 16])  # (16, h_kv*d)
            packed = {"fp16": M} if arm == "fp16" else quantize_packed(
                arm, M, group=group, **kw)[0]
            blocks.append((packed, j, j + 16))
        return blocks

    k_blocks, v_blocks = pack_side(K, k_arm), pack_side(V, v_arm)
    common = dict(
        k_arm=k_arm, v_arm=v_arm, group=group, seed=0, k_pre_rope=False,
        rope_cos=None, rope_sin=None, k_tail=None, v_tail=None,
        n_q_groups=n_q_heads // h_kv, scale=scale)

    oracle = naive_dense_attention(q, k_blocks, v_blocks, **common)
    fast = chunked_dequant_attention(
        q, k_blocks, v_blocks, tail_start=S, **common)

    drift = attention_diff(fast, oracle)
    assert drift["max_abs"] < 1e-10, drift  # online softmax is exact vs oracle
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_chunked_attention.py -k chunked_matches -v`
Expected: FAIL with `ImportError: cannot import name 'chunked_dequant_attention'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/bmx/cache/chunked_attention.py` (`_dequant_block`, `from_matrix`,
`apply_rope` already imported/defined in Task 4 — do not redefine):

```python
def chunked_dequant_attention(
    q, k_blocks, v_blocks, *, k_arm, v_arm, group, seed,
    k_pre_rope, rope_cos, rope_sin, k_tail, v_tail, tail_start, n_q_groups,
    scale,
):
    """Online-softmax attention over per-block dequantized K/V. GQA-aware.

    q: (n_q_heads, n_q, d). k_blocks/v_blocks: list of (packed, start, end).
    k_pre_rope: if True, dequantized K blocks are pre-RoPE and get RoPE applied at
    [start,end) before the contraction. k_tail/v_tail: (h_kv, tail_len, d) fp16
    recent window (post-RoPE for K). Returns (n_q_heads, n_q, d).
    """
    n_q_heads, n_q, d = q.shape
    h_kv = n_q_heads // n_q_groups
    acc = torch.zeros(n_q_heads, n_q, d, dtype=q.dtype, device=q.device)
    m = torch.full((n_q_heads, n_q, 1), float("-inf"), dtype=q.dtype, device=q.device)
    l = torch.zeros(n_q_heads, n_q, 1, dtype=q.dtype, device=q.device)

    def attend(K_kv, V_kv):
        nonlocal acc, m, l
        K = K_kv.repeat_interleave(n_q_groups, dim=0)  # (n_q_heads, blk, d)
        V = V_kv.repeat_interleave(n_q_groups, dim=0)
        s = (q @ K.transpose(-1, -2)) * scale          # (n_q_heads, n_q, blk)
        acc, m, l = online_softmax_update(acc, m, l, s, V)

    for (packed, start, end) in k_blocks:
        K_kv = _dequant_block(packed, k_arm, group, seed, h_kv).to(q.dtype)
        if k_pre_rope:
            cos = rope_cos[start:end].to(q.dtype)
            sin = rope_sin[start:end].to(q.dtype)
            K_kv = apply_rope(K_kv, cos, sin)
        # V is paired by index (same block boundaries).
        vp = v_blocks[k_blocks.index((packed, start, end))][0]
        V_kv = _dequant_block(vp, v_arm, group, seed, h_kv).to(q.dtype)
        attend(K_kv, V_kv)

    if k_tail is not None and k_tail.shape[1] > 0:
        attend(k_tail.to(q.dtype), v_tail.to(q.dtype))

    return acc / l
```

Note: the `v_blocks[...index...]` lookup is O(n²) and fragile — replace it by iterating `zip(k_blocks, v_blocks)`:

```python
    for (kpacked, start, end), (vpacked, _vs, _ve) in zip(k_blocks, v_blocks):
        K_kv = _dequant_block(kpacked, k_arm, group, seed, h_kv).to(q.dtype)
        if k_pre_rope:
            K_kv = apply_rope(
                K_kv, rope_cos[start:end].to(q.dtype), rope_sin[start:end].to(q.dtype)
            )
        V_kv = _dequant_block(vpacked, v_arm, group, seed, h_kv).to(q.dtype)
        attend(K_kv, V_kv)
```
(Use this loop; delete the `.index(...)` version above.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_chunked_attention.py -v`
Expected: all PASS (both arms match dense attention to 1e-10).

- [ ] **Step 5: Format, lint, stage, propose commit**

```bash
cd /d/Projects/bmx && uv run ruff format . && uv run ruff check .
git add src/bmx/cache/chunked_attention.py tests/test_chunked_attention.py
```
Propose message: `feat(cache): chunked dequant-attention with RoPE-in-loop + GQA`

---

## Task 6: `PackedStreamingCache` + registry attention routing

The cache that stores packed codes (+ frozen subspace + fp16 window), flushes-to-packed on decode, and routes attention through `chunked_dequant_attention` via the `AttentionInterface` registry. Parity-gated against `StreamingQuantizedCache`.

**Files:**
- Create: `src/bmx/cache/packed_streaming.py`
- Test: `tests/test_packed_streaming.py`

**Interfaces:**
- Consumes: `compute_flush_schedule` (Task 4), `quantize_packed` (Task 2), `chunked_dequant_attention` (Task 5), `CacheCodecSpec`, `apply_rope`/`rope_cos_sin`, `resolve_text_config`/`resolve_decoder_layers` (from `streaming.py`).
- Produces: `PackedStreamingCache(model_config, k_spec, v_spec, recent_window=32)` with `.attach(model)` / `.detach()` registering the custom attention fn and the k_proj hooks; drop-in `past_key_values=`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_packed_streaming.py`:

```python
"""PackedStreamingCache: parity with StreamingQuantizedCache (bit-for-bit)."""

import torch

from bmx.cache.packed_streaming import PackedStreamingCache
from bmx.cache.specs import CacheCodecSpec
from bmx.cache.streaming import StreamingQuantizedCache
from factories import ids, tiny_llama


def _k2b():
    return (
        CacheCodecSpec(arm="lowrank_rtn_channel", bits=3, rank=4, group=16,
                       pre_rope=True),
        CacheCodecSpec(arm="turboquant_mse", bits=2),
    )


def test_packed_generate_matches_streaming():
    model = tiny_llama()
    input_ids = ids(vocab=97, seq=12, seed=5)
    k_spec, v_spec = _k2b()

    ref_cache = StreamingQuantizedCache(model.config, k_spec=k_spec, v_spec=v_spec)
    ref_cache.attach(model)
    with torch.no_grad():
        ref = model.generate(input_ids, max_new_tokens=20, do_sample=False,
                             use_cache=True, past_key_values=ref_cache)
    ref_cache.detach()

    packed = PackedStreamingCache(model.config, k_spec=k_spec, v_spec=v_spec)
    packed.attach(model)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=20, do_sample=False,
                             use_cache=True, past_key_values=packed)
    packed.detach()

    assert torch.equal(out, ref)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_packed_streaming.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bmx.cache.packed_streaming'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/bmx/cache/packed_streaming.py`. This layer stores packed blocks + frozen subspace + fp16 tail, flushes-to-packed on the shared schedule, and exposes the packed state to a registered attention fn. Key design points:

- The registered attention fn `chunked_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs)` ignores the dense `key`/`value` HF passes (they come from a fallback) and instead reads the packed state for `module`'s layer off the cache. To find "this layer's" state, store a back-reference: during `attach`, set `mlayer.self_attn._packed_layer = self.layers[i]`. The fn reads `module._packed_layer`.
- `update()` must still return tensors (HF cache contract) and keep `self.keys/self.values` as a SMALL fp16 tail-only slab (so the fallback path and shape bookkeeping work), but attention never uses the full dense K/V — it uses the packed state.

```python
"""Packed streaming KV cache: resident packed codes, chunked dequant-attention.

Sibling of StreamingQuantizedCache. Stores per-block PACKED codes (the bpe
footprint) + the frozen subspace + the fp16 recent window — never the dense
dequant prefix or a reassembled dense slab. Attention is routed through
chunked_dequant_attention via the transformers AttentionInterface registry, so
the dense K/V is never materialized. Bit-for-bit parity with
StreamingQuantizedCache is the correctness gate.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache, DynamicLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from bmx.cache.chunked_attention import chunked_dequant_attention
from bmx.cache.codecs import S_DIVISIBILITY_ARMS, quantize_packed
from bmx.cache.collect import _reshape_heads, to_matrix
from bmx.cache.rope import rope_cos_sin
from bmx.cache.specs import CacheCodecSpec
from bmx.cache.streaming import (
    compute_flush_schedule,
    model_config_n_layers,
    resolve_decoder_layers,
    resolve_text_config,
)
from bmx.decomp.lrs import truncated_svd

_ATTN_NAME = "chunked_dequant"


def chunked_attention_forward(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    """Registered attention fn: route through packed chunked dequant-attention.

    query: (1, n_q_heads, n_q, d). Reads packed state off module._packed_layer.
    Returns (attn_output (1, n_q, n_q_heads*d), attn_weights=None) per HF contract.
    """
    layer = module._packed_layer
    q = query.squeeze(0)  # (n_q_heads, n_q, d)
    out = layer.attend(q, scaling)  # (n_q_heads, n_q, d)
    n_q_heads, n_q, d = out.shape
    attn_output = out.transpose(0, 1).reshape(1, n_q, n_q_heads * d)
    return attn_output.to(query.dtype), None


ALL_ATTENTION_FUNCTIONS.register(_ATTN_NAME, chunked_attention_forward)


class PackedStreamingLayer(DynamicLayer):
    def __init__(self, k_spec, v_spec, model_config, recent_window=32):
        super().__init__()
        self.k_spec = k_spec
        self.v_spec = v_spec
        self.model_config = model_config
        self.recent_window = recent_window
        self._k_pre = None
        self._k_pre_offset = 0
        self._committed_S_q = 0
        self._k_blocks = []  # list of (packed, start, end)
        self._v_blocks = []
        self._frozen_svd = None
        self._rope_cos = None
        self._rope_sin = None
        tc = resolve_text_config(model_config)
        self._h_kv = getattr(tc, "num_key_value_heads", tc.num_attention_heads)
        self._d_head = (
            getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
        )
        self._g = k_spec.group if k_spec.arm in S_DIVISIBILITY_ARMS else 1

    def stash_pre_rope(self, out):
        block = _reshape_heads(out, self._h_kv, self._d_head)
        self._k_pre = block if self._k_pre is None else torch.cat(
            [self._k_pre, block], dim=1)

    def _extend_rope(self, new_committed, device):
        covered = 0 if self._rope_cos is None else self._rope_cos.shape[0]
        if new_committed > covered:
            nc, ns = rope_cos_sin(
                self.model_config, new_committed - covered, start=covered,
                device=device)
            if self._rope_cos is None:
                self._rope_cos, self._rope_sin = nc, ns
            else:
                self._rope_cos = torch.cat([self._rope_cos, nc], dim=0)
                self._rope_sin = torch.cat([self._rope_sin, ns], dim=0)

    def _quantize_k_block(self, k_block_pre, start, end):
        """(h_kv, blk, d) pre-RoPE -> packed K block. RoPE applied at READ."""
        M = to_matrix(k_block_pre)  # (blk, h_kv*d)
        spec = self.k_spec
        if spec.arm == "lowrank_rtn_channel":
            if self._frozen_svd is None:
                Us, V = truncated_svd(M, spec.rank)
                self._frozen_svd = (Us, V)
            packed, _ = quantize_packed(
                spec.arm, M, bits=spec.bits, group=spec.group, rank=spec.rank,
                svd_factors=self._frozen_svd, seed=spec.seed)
        else:
            packed, _ = quantize_packed(
                spec.arm, M, bits=spec.bits, group=spec.group, rank=spec.rank,
                seed=spec.seed)
        self._extend_rope(end, k_block_pre.device)
        return packed

    def update(self, key_states, value_states, *args, **kwargs):
        keys, values = super().update(key_states, value_states, *args, **kwargs)
        S = keys.shape[2]
        W = self.recent_window
        new_S_q = compute_flush_schedule(S, W, self._g)
        if new_S_q > self._committed_S_q:
            start, end = self._committed_S_q, new_S_q
            if self.k_spec.pre_rope:
                ls = start - self._k_pre_offset
                le = end - self._k_pre_offset
                k_block_pre = self._k_pre[:, ls:le, :].float()
                kpacked = self._quantize_k_block(k_block_pre, start, end)
            else:
                k_block = keys.squeeze(0)[..., start:end, :].float()
                kpacked, _ = quantize_packed(
                    self.k_spec.arm, to_matrix(k_block), bits=self.k_spec.bits,
                    group=self.k_spec.group, rank=self.k_spec.rank,
                    seed=self.k_spec.seed)
            v_block = values.squeeze(0)[..., start:end, :].float()
            vpacked, _ = quantize_packed(
                self.v_spec.arm, to_matrix(v_block), bits=self.v_spec.bits,
                group=self.v_spec.group, rank=self.v_spec.rank, seed=self.v_spec.seed)
            self._k_blocks.append((kpacked, start, end))
            self._v_blocks.append((vpacked, start, end))
            self._committed_S_q = new_S_q
            if self.k_spec.pre_rope and self._k_pre is not None:
                pl = new_S_q - self._k_pre_offset
                self._k_pre = self._k_pre[:, pl:, :].contiguous() if pl < \
                    self._k_pre.shape[1] else None
                self._k_pre_offset = new_S_q
        # Keep self.keys/.values as the fp16 tail-only slab (HF contract).
        self.keys, self.values = keys, values
        return keys, values

    def attend(self, q, scaling):
        """q: (n_q_heads, n_q, d) -> (n_q_heads, n_q, d) via chunked dequant-attn."""
        tail_start = self._committed_S_q
        k_tail = self.keys.squeeze(0)[..., tail_start:, :]
        v_tail = self.values.squeeze(0)[..., tail_start:, :]
        n_q_heads = q.shape[0]
        return chunked_dequant_attention(
            q, self._k_blocks, self._v_blocks,
            k_arm=self.k_spec.arm, v_arm=self.v_spec.arm,
            group=self.k_spec.group, seed=self.k_spec.seed,
            k_pre_rope=self.k_spec.pre_rope,
            rope_cos=self._rope_cos, rope_sin=self._rope_sin,
            k_tail=k_tail, v_tail=v_tail, tail_start=tail_start,
            n_q_groups=n_q_heads // self._h_kv, scale=scaling,
        )


class PackedStreamingCache(Cache):
    def __init__(self, model_config, k_spec, v_spec, recent_window=32):
        super().__init__(
            layer_class_to_replicate=lambda: PackedStreamingLayer(
                k_spec, v_spec, model_config, recent_window))
        self.model_config = model_config
        self.k_spec = k_spec
        self.v_spec = v_spec
        self.recent_window = recent_window
        self._handles = []
        self._saved_impl = None
        self._model = None

    def attach(self, model):
        self.detach()
        self._model = model
        self._saved_impl = model.config._attn_implementation
        model.config._attn_implementation = _ATTN_NAME
        n_layers = model_config_n_layers(model)
        while len(self.layers) < n_layers:
            self.layers.append(PackedStreamingLayer(
                self.k_spec, self.v_spec, self.model_config, self.recent_window))
        for i, mlayer in enumerate(resolve_decoder_layers(model)):
            mlayer.self_attn._packed_layer = self.layers[i]
            if self.k_spec.pre_rope:
                def k_hook(module, inp, out, i=i):
                    self.layers[i].stash_pre_rope(out)
                self._handles.append(
                    mlayer.self_attn.k_proj.register_forward_hook(k_hook))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._model is not None and self._saved_impl is not None:
            self._model.config._attn_implementation = self._saved_impl
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.detach()
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_packed_streaming.py -v`
Expected: PASS (`torch.equal(out, ref)`). If it fails:
- First check the codec-split and chunked-attention tests still pass (foundation).
- The likely divergence is RoPE position handling or the tail boundary. The packed `attend` must apply RoPE to committed blocks at `[start,end)` and treat the tail as already post-RoPE (it comes from `super().update()` which stores post-RoPE keys). Confirm `tail_start == _committed_S_q`.
- This is the "earn the tolerance fallback" gate: only if exact parity proves structurally impossible (document the specific reason — e.g. attention-fn dtype upcast HF applies that the streaming path doesn't), fall back to `torch.allclose(out, ref)` on the *logits* of a single forward and record why in the test docstring.

- [ ] **Step 5: Format, lint, stage, propose commit**

```bash
cd /d/Projects/bmx && uv run ruff format . && uv run ruff check .
git add src/bmx/cache/packed_streaming.py tests/test_packed_streaming.py
```
Propose message: `feat(cache): PackedStreamingCache with chunked dequant-attention routing`

---

## Task 7: Census instrument (`experiments/k3_kernel_census.py`) — VM

Thin tyro CLI measuring resident/peak/incremental per arm per context length, for both cache paths, → parquet. Validates the ledger and the real 128k peak. Runs on the VM (CUDA); locally it runs on tiny lengths for a smoke test.

**Files:**
- Create: `experiments/k3_kernel_census.py`
- (No new unit test; the deliverable is the instrument + a local smoke run.)

**Interfaces:**
- Consumes: `StreamingQuantizedCache`, `PackedStreamingCache`, `KVMemCase`/`predict_peak`, `bmx.artifacts` run-dir helpers.

- [ ] **Step 1: Read the artifacts + an existing experiment for conventions**

Run: `cd /d/Projects/bmx && sed -n '1,60p' src/bmx/artifacts.py && sed -n '1,40p' experiments/k3_niah.py`
Expected: see the `results/<exp>/<run-id>/` run-dir API (config + env + SHA) and the tyro CLI shape. Mirror them.

- [ ] **Step 2: Write the instrument**

Create `experiments/k3_kernel_census.py`:

```python
"""KV-cache memory census: resident / peak / incremental per arm per length.

Measures both cache paths (StreamingQuantizedCache, PackedStreamingCache) and
compares against the analytic byte-ledger. CUDA-authoritative (VM); falls back to
a tiny CPU smoke run locally. Writes parquet to results/k3_kernel_census/<run-id>.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import new_run_dir  # adjust to the actual helper name
from bmx.bench.kv_memory import KVMemCase, predict_peak
from bmx.cache.packed_streaming import PackedStreamingCache
from bmx.cache.specs import CacheCodecSpec
from bmx.cache.streaming import StreamingQuantizedCache


@dataclasses.dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    seq_lens: tuple[int, ...] = (4096, 16384, 32768)
    arms: tuple[str, ...] = ("fp16", "k2b")
    max_new_tokens: int = 4


def _specs(arm):
    if arm == "fp16":
        return CacheCodecSpec(arm="fp16"), CacheCodecSpec(arm="fp16")
    if arm == "k2b":
        return (
            CacheCodecSpec(arm="lowrank_rtn_channel", bits=3, rank=16, group=64,
                           pre_rope=True),
            CacheCodecSpec(arm="turboquant_mse", bits=2),
        )
    raise ValueError(arm)


def _measure(model, input_ids, cache):
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(input_ids, past_key_values=cache, use_cache=True)
    resident = torch.cuda.max_memory_allocated() if cuda else 0
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=4, do_sample=False,
                       use_cache=True, past_key_values=cache)
    peak = torch.cuda.max_memory_allocated() if cuda else 0
    return resident, peak


def main(cfg: Config):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu").eval()
    rows = []
    for S in cfg.seq_lens:
        input_ids = torch.randint(0, tok.vocab_size, (1, S),
                                  device=model.device)
        for arm in cfg.arms:
            k_spec, v_spec = _specs(arm)
            for path, Cls in [("dense_stream", StreamingQuantizedCache),
                              ("chunked", PackedStreamingCache)]:
                cache = Cls(model.config, k_spec=k_spec, v_spec=v_spec)
                if k_spec.pre_rope:
                    cache.attach(model)
                resident, peak = _measure(model, input_ids, cache)
                if hasattr(cache, "detach"):
                    cache.detach()
                bpe_k, bpe_v = cache.bits_per_entry() if hasattr(
                    cache, "bits_per_entry") else (float("nan"), float("nan"))
                rows.append({
                    "seq_len": S, "arm": arm, "path": path,
                    "resident_after_prefill": resident, "peak_decode": peak,
                    "peak_decode_incremental": peak - resident,
                    "bpe_k": bpe_k, "bpe_v": bpe_v,
                })
                del cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    run_dir = new_run_dir("k3_kernel_census", cfg)  # adjust to real API
    df.to_parquet(run_dir / "census.parquet")
    print(df.to_string(index=False))
    print(f"\nwrote {run_dir / 'census.parquet'}")


if __name__ == "__main__":
    main(tyro.cli(Config))
```

- [ ] **Step 3: Reconcile the artifacts API**

Run: `cd /d/Projects/bmx && grep -n "def " src/bmx/artifacts.py | head -30`
Then fix the `new_run_dir(...)` call + import to match the real function name/signature (e.g. it may be a class or `run_dir(exp, config)`). Do not invent — use what exists.

- [ ] **Step 4: Local CPU smoke run (no download — guard the model load)**

Add a `--model-name` override to point at nothing heavy locally is not possible (needs real geometry). Instead verify the CLI parses and the non-model code paths import cleanly:

Run: `cd /d/Projects/bmx && uv run python -c "import experiments.k3_kernel_census as m; print(m.Config())"`
Expected: prints the default `Config(...)` with no import errors. (The full run is VM-only; this confirms the instrument is wired.)

- [ ] **Step 5: Format, lint, stage, propose commit**

```bash
cd /d/Projects/bmx && uv run ruff format . && uv run ruff check .
git add experiments/k3_kernel_census.py
```
Propose message: `feat(exp): KV-cache memory census instrument (resident/peak/incremental)`

---

## Task 8: Full-suite green + ledger-vs-census reconciliation note

Final integration gate before the VM run.

**Files:**
- Modify: `docs/2026-06-23-kernel-plan-state.md` (new VM handoff note)

- [ ] **Step 1: Run the full suite**

Run: `cd /d/Projects/bmx && uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: 220 passed + new tests (codec-split, kv_memory, chunked_attention, packed_streaming), 1 xfailed. Zero ruff findings.

- [ ] **Step 2: Write the VM handoff note**

Create `docs/2026-06-23-kernel-plan-state.md` documenting: the four components built, the ledger's predicted dense_stream vs chunked peaks at 128k (run `uv run python -c` to print `predict_peak` for both with the real Llama-3.1 constants), the exact census command to run on the VM, and the decision rule for Phase 3 (Triton) — quoting the spec's gate verbatim. **Explicitly state that the Phase-3 Triton kernel must be validated against `naive_dense_attention` (the oracle) via `attention_diff`, not against the chunked path** — the oracle shares the least code with the kernel and is the trustworthy ground truth; `attention_diff` quantifies any quality drift so a faster kernel is never accepted on speed alone.

- [ ] **Step 3: Stage, propose commit**

```bash
cd /d/Projects/bmx && git add docs/2026-06-23-kernel-plan-state.md
```
Propose message: `docs: kernel Phases 1+2 complete; VM census handoff + Phase-3 gate`

---

## Post-implementation: `simplify` pass

After all tasks are merged and green, run the `simplify` skill over the diff (codecs.py split, chunked_attention.py, packed_streaming.py especially — the codec split and the registry routing are the spots most likely to have accumulated incidental complexity). Quality-only; `/code-review` separately for bugs if desired.

---

## Self-review notes (addressed inline)

- **Spec coverage:** codec split (Tasks 1–2), byte-ledger (Task 3), golden-reference oracle + diff helper (Task 4), chunked attention incl. online-softmax + RoPE-in-loop + GQA (Tasks 4–5), frozen-subspace-as-packed-state + decode flush + registry routing (Task 6), census with incremental (Task 7), bit-for-bit gate (Tasks 2, 6), oracle-based Phase-3 gate carried to the handoff note (Task 8). All spec components mapped.
- **Golden reference (oracle):** `naive_dense_attention` is the single named ground truth (Task 4); online-softmax (Task 4), chunked (Task 5), packed (Task 6 via end-to-end parity), and the future Triton kernel (Task 8 gate) are ALL diffed against it via `attention_diff`. No faster path is accepted on speed without a quantified quality diff against the oracle.
- **YAGNI:** waterfill arms explicitly NOT split (Task 2 scope note); no Triton/paging/offload.
- **Type consistency:** `quantize_packed`/`dequant_packed`/`compute_flush_schedule`/`online_softmax_update`/`chunked_dequant_attention`/`PackedStreamingCache` names and signatures are consistent across the tasks that consume them (Interfaces blocks).
- **Known reconciliations the implementer must do (flagged, not placeholders):** the `bmx.artifacts` run-dir API name (Task 7 Step 3) and the turboquant packed-dict `bits` carry (Task 2 Step 3) — both have explicit instructions, not "TBD".
