# K4 Spectral Allocation Codec — Implementation Plan (Stages 0–2, offline gauntlet)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and gate the K4 spectral codec (query-weighted eigenbasis + waterfilling) offline on real caches — kill-or-confirm gates G0/G1/G2 from `docs/superpowers/specs/2026-07-12-k4-spectral-codec-design.md`, no VM required.

**Architecture:** A new `src/bmx/cache/spectral.py` module holds the K4 core (query-weighted whitener, corpus-fit eigenbasis pack, waterfilled quantizer) reusing the existing waterfill allocator and RTN machinery. Three thin tyro experiments (`k4_spectra`, `k4_frontier`, `k4_alloc`) produce the Stage-0/1/2 parquets and machine-readable gate verdicts. **Deliberately out of scope for this plan** (its own plan after the G1 verdict): CACHE_ARMS registration, recipes, streaming-cache integration, and the Stage-3 VM duel — building deployment plumbing before the kill-or-confirm verdict is waste if G1 kills.

**Tech Stack:** Python 3.12, PyTorch 2.12 (CPU — dev box has no CUDA), safetensors, pandas/parquet, tyro CLIs, pytest.

## Global Constraints

Copied from `CLAUDE.md` / the spec — every task implicitly includes these:

- **NEVER `git commit` without the user's explicit approval.** Every "Commit" step below means: `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` (baseline **397 passed, 17 skipped, 1 xfailed** as of `089c528`; grows as tasks add tests) → `git add <files>` → propose the message → **STOP for approval**. No AI attribution ever.
- Dependencies only via `uv add`. (This plan needs no new dependencies.)
- Use the Bash tool (git bash); `cd /d/Projects/bmx` first in fresh shells.
- Honest bits: ALL metadata counted (scales, tier maps, bases per accounting mode). Two accounting modes per spec §5: **model-level** (corpus artifacts ship with the model, zero per-sequence charge) and **skeptic** (per-sequence charge `16·C/S` for the decoder matrix + tier map), skeptic quoted at both the cache's S and `deploy_s=32768`.
- Metrics: rank codecs on logit distortion vs real queries (`logit_rope` = RoPE-at-read for `k_pre` arms), never Frobenius alone.
- dtype: fp64 for moment/eig math, fp32 in codec paths, caches stored fp16. Shape asserts at boundaries.
- Tiny offline test models come from synthetic tensors / `tests/factories.py`; never download in tests.
- Carried methodology from spec §9 (non-negotiable): uniform bit-SWEEP baseline, random-basis control arm, oracle-refit control, idealized+honest bpe columns with deploy-S quote, region-matched tail scoring for any frozen/fit-on-prefix claim, deterministic MSE-optimal rounding (no dithered arms).
- Runtime note: everything here is CPU-feasible. gpt2 work = minutes. Llama-3.1-8B *cache-tensor* work (frontier, spectra) = minutes-to-~1h (pure tensor math on the existing `results/cache/llama-3.1-8b_2048.safetensors`). Llama *model-forward* work (Task 5 collection, Task 10 sensitivity) = ~3–10 min per 2048-token prefill on CPU; tasks flag it and bound it.

**Pre-registered predictions being tested (spec §9):** P1 transfer ≥90%, P2 bulk-insensitivity, P3 coefficient-quantization dominates fp16 subspaces, P4 weighted > unweighted basis. Write outcomes either way.

---

### Task 1: MSE-optimal RTN scale (opt-in `mse_scale`)

The theory brief (spec §3.2): at 2 bits, Lloyd-Max over **optimal-step** uniform is worth ~1%, but `rtn_quantize_packed` currently uses **max-based** scales (`abs().amax()/qmax`) — the gap between max-based and MSE-optimal step is the free win. Add an opt-in flag; **default stays `False`** so every existing arm/test/banked number is bit-identical.

**Files:**
- Modify: `src/bmx/quant/rtn.py`
- Test: `tests/test_rtn_mse_scale.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `rtn_quantize_packed(W, bits, group_size, mse_scale: bool = False)` and `rtn_quantize(W, bits, group_size, mse_scale: bool = False)` — same return types as today. Tasks 4 and 7 pass `mse_scale=True`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rtn_mse_scale.py
import torch

from bmx.quant.rtn import rtn_quantize, rtn_quantize_packed


def _gaussian(S=256, C=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(S, C, generator=g)


def test_default_unchanged():
    """mse_scale defaults False and is bit-identical to the historical behavior."""
    W = _gaussian()
    q_old, s_old = rtn_quantize_packed(W, 3, 64)
    q_new, s_new = rtn_quantize_packed(W, 3, 64, mse_scale=False)
    assert torch.equal(q_old, q_new)
    assert torch.equal(s_old, s_new)


def test_mse_scale_lowers_mse():
    """MSE-optimal step strictly beats max-based step on Gaussian data at 2 and 3 bits."""
    W = _gaussian()
    for bits in (2, 3):
        err_max = (rtn_quantize(W, bits, 64) - W).pow(2).mean()
        err_mse = (rtn_quantize(W, bits, 64, mse_scale=True) - W).pow(2).mean()
        assert err_mse < err_max, f"bits={bits}: {err_mse} !< {err_max}"


def test_mse_scale_deterministic():
    W = _gaussian(seed=3)
    a = rtn_quantize(W, 2, 64, mse_scale=True)
    b = rtn_quantize(W, 2, 64, mse_scale=True)
    assert torch.equal(a, b)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_rtn_mse_scale.py -v`
Expected: FAIL — `TypeError: rtn_quantize_packed() got an unexpected keyword argument 'mse_scale'`.

- [ ] **Step 3: Implement**

In `src/bmx/quant/rtn.py`, add the refinement helper and thread the flag (alternating minimization: given codes, the MSE-optimal scale is `<G,Q>/<Q,Q>`; given scale, optimal codes are round+clamp — monotone non-increasing MSE, deterministic):

```python
def _mse_refine_scale(
    G: torch.Tensor, scale: torch.Tensor, qmax: int, n_iter: int = 10
) -> torch.Tensor:
    """Alternating-minimization refinement of the per-group scale (Lloyd on the
    step size only; codebook stays uniform). Deterministic, monotone in MSE."""
    for _ in range(n_iter):
        Q = (G / scale).round().clamp(-qmax - 1, qmax)
        num = (G * Q).sum(dim=-1, keepdim=True)
        den = (Q * Q).sum(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = (num / den).clamp_min(1e-12)
    return scale


def rtn_quantize_packed(
    W: torch.Tensor, bits: int, group_size: int, mse_scale: bool = False
):
    ...  # existing body unchanged through the max-based `scale =` line, then:
    if mse_scale:
        scale = _mse_refine_scale(G, scale, qmax)
    Q = (G / scale).round().clamp(-qmax - 1, qmax)
    ...


def rtn_quantize(
    W: torch.Tensor, bits: int, group_size: int, mse_scale: bool = False
) -> torch.Tensor:
    Q_int, scale = rtn_quantize_packed(W, bits, group_size, mse_scale=mse_scale)
    return rtn_dequantize_packed(Q_int, scale, group_size)
```

(Keep the existing docstrings; add one line each noting the flag and that the default preserves historical behavior.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_rtn_mse_scale.py -v` → 3 PASS.

- [ ] **Step 5: Full battery + commit**

`uv run ruff format . && uv run ruff check . && uv run pytest -q` → clean (400/17/1).
Stage `src/bmx/quant/rtn.py tests/test_rtn_mse_scale.py`; propose:
`feat(quant): opt-in MSE-optimal RTN step (mse_scale) — default preserves max-based behavior bit-identically`
**STOP for user approval.**

---

### Task 2: Extract `allocate_bits_from_variance` from `allocate_channel_bits`

K4 allocates bits from a **corpus spectrum** (eigenvalues), not from the scored matrix's own per-channel variance. Extract the variance→bits body so both callers share one bisection; `allocate_channel_bits` becomes a thin wrapper with identical behavior (pinned by the existing allocator tests in `tests/test_cache_codecs.py`).

**Files:**
- Modify: `src/bmx/cache/codecs.py:149-191`
- Test: `tests/test_cache_codecs.py` (append)

**Interfaces:**
- Produces: `allocate_bits_from_variance(var: torch.Tensor (C,), budget_bits: float, tiers: tuple[int, ...] = (0, 2, 3, 4), *, n_search: int = 40) -> torch.Tensor (C,) int64`. Consumed by Task 4's `fit_spectral_pack` and by the existing `allocate_channel_bits`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_cache_codecs.py` next to the existing `test_allocate_*` functions):

```python
def test_allocate_from_variance_matches_channel_allocator():
    from bmx.cache.codecs import allocate_bits_from_variance, allocate_channel_bits

    g = torch.Generator().manual_seed(0)
    R = torch.randn(256, 64, generator=g) * torch.linspace(0.1, 4.0, 64)
    a = allocate_channel_bits(R, 3.0, tiers=(0, 2, 3, 4))
    b = allocate_bits_from_variance(
        R.var(dim=0, unbiased=False), 3.0, tiers=(0, 2, 3, 4)
    )
    assert torch.equal(a, b)


def test_allocate_from_variance_rich_tiers():
    from bmx.cache.codecs import allocate_bits_from_variance

    # Spiked spectrum: 4 large eigenvalues over a flat bulk — the K4 tier set
    # must fund the spikes at 5-8 bits and the bulk at 0-2.
    var = torch.cat([torch.tensor([1e4, 1e4, 1e3, 1e3]), torch.ones(60)])
    bits = allocate_bits_from_variance(var, 2.5, tiers=(0, 2, 3, 4, 5, 6, 8))
    assert bits[:4].min() >= 5, f"spikes underfunded: {bits[:4]}"
    assert bits[4:].max() <= 3
    assert bits.float().mean().item() <= 2.5 + 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cache_codecs.py::test_allocate_from_variance_matches_channel_allocator -v`
Expected: FAIL — ImportError (`allocate_bits_from_variance` not defined).

- [ ] **Step 3: Implement** — in `codecs.py`, move the body of `allocate_channel_bits` (everything after the `var =` line) into the new function verbatim; the wrapper computes `var` and delegates:

```python
def allocate_bits_from_variance(
    var: torch.Tensor,
    budget_bits: float,
    tiers: tuple[int, ...] = (0, 2, 3, 4),
    *,
    n_search: int = 40,
) -> torch.Tensor:
    """Reverse-water-filling bit allocation from a per-direction variance vector.

    Same bisection as allocate_channel_bits; factored out so corpus spectra
    (K4 spectral packs) and per-matrix variances share one implementation.
    Returns (C,) int64 bit-widths, each a member of `tiers`.
    """
    assert var.dim() == 1, f"var must be 1-D (C,); got {tuple(var.shape)}"
    var = var.double().clamp_min(1e-30)
    # ... existing body verbatim from the current allocate_channel_bits,
    #     from `tiers_t = ...` through `return best.to(torch.int64)` ...


def allocate_channel_bits(
    R: torch.Tensor,
    budget_bits: float,
    tiers: tuple[int, ...] = (0, 2, 3, 4),
    *,
    axis: int = 0,
    n_search: int = 40,
) -> torch.Tensor:
    """Reverse-water-filling per-channel bit allocation (Cover-Thomas Thm 13.3.3).
    Thin wrapper over allocate_bits_from_variance on R.var(dim=axis)."""
    assert R.dim() == 2, f"R must be 2-D (S, C); got {tuple(R.shape)}"
    return allocate_bits_from_variance(
        R.var(dim=axis, unbiased=False), budget_bits, tiers, n_search=n_search
    )
```

- [ ] **Step 4: Run to verify pass** — the two new tests AND all existing `test_allocate_*` + waterfill tests:

Run: `uv run pytest tests/test_cache_codecs.py tests/test_k2_waterfill.py -q` → all PASS.

- [ ] **Step 5: Full battery + commit**

Propose: `refactor(codecs): extract allocate_bits_from_variance — corpus-spectrum allocation shares the channel allocator's bisection`
**STOP for user approval.**

---

### Task 3: `spectral.py` part 1 — query-weighted moments + whitener

The math (spec §3.1 + theory brief Q2): the metric is `E[(qᵀ R_p e)²] = eᵀ W e` with `W = E_{q,p}[R_pᵀ q qᵀ R_p]` — **block-diagonal per kv-head** (logits never couple heads), estimated from stored probe queries with the RoPE rotation averaged over sampled key positions. `R_pᵀ q` is RoPE applied with negated sin.

**Files:**
- Create: `src/bmx/cache/spectral.py`
- Test: `tests/test_spectral.py` (new)

**Interfaces:**
- Consumes: `bmx.cache.rope._rotate_half`, `bmx.cache.collect.to_matrix` layout (channel c = head·d + dim, head-major — per-head blocks are contiguous column slices).
- Produces (all fp64):
  - `key_second_moment(M: (S, C) fp32) -> (C, C)` — uncentered `MᵀM/S`.
  - `query_position_moment(q: (h, T, d), cos: (S, d), sin: (S, d), h_kv: int, *, position_stride: int = 8) -> (h_kv, d, d)` — the W blocks, GQA-pooled.
  - `assemble_whitener(W_blocks: (h_kv, d, d), *, ridge: float = 1e-3) -> (Wh (C, C), Wh_inv (C, C))` — symmetric block sqrt / inverse-sqrt with per-block eigenvalue floor `ridge·λ_max`.
  - `identity_whitener(C: int) -> (Wh, Wh_inv)` — both identity (the unweighted-KLT ablation path).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_spectral.py
import torch

from bmx.cache.spectral import (
    assemble_whitener,
    identity_whitener,
    key_second_moment,
    query_position_moment,
)


def test_key_second_moment_shape_and_value():
    g = torch.Generator().manual_seed(0)
    M = torch.randn(128, 16, generator=g)
    Sigma = key_second_moment(M)
    assert Sigma.shape == (16, 16) and Sigma.dtype == torch.float64
    expected = (M.double().mT @ M.double()) / 128
    assert torch.allclose(Sigma, expected)


def test_query_moment_identity_rope_matches_plain_outer_product():
    """With cos=1, sin=0 (R_p = I), W_j must equal the plain GQA-pooled query
    second moment."""
    g = torch.Generator().manual_seed(1)
    h, T, d, h_kv = 4, 32, 8, 2
    q = torch.randn(h, T, d, generator=g)
    S = 64
    cos, sin = torch.ones(S, d), torch.zeros(S, d)
    W = query_position_moment(q, cos, sin, h_kv, position_stride=16)
    grp = h // h_kv
    for j in range(h_kv):
        qj = q[j * grp : (j + 1) * grp].reshape(-1, d).double()
        expected = qj.mT @ qj / qj.shape[0]
        assert torch.allclose(W[j], expected, atol=1e-10), f"head {j}"


def test_query_moment_is_symmetric_psd():
    g = torch.Generator().manual_seed(2)
    q = torch.randn(8, 16, 8, generator=g)
    # A real-ish RoPE table: interleave some rotation
    S = 32
    theta = torch.linspace(0, 3.0, S).unsqueeze(1) * torch.ones(1, 8)
    W = query_position_moment(q, theta.cos(), theta.sin(), h_kv=4)
    for j in range(4):
        assert torch.allclose(W[j], W[j].mT, atol=1e-12)
        assert torch.linalg.eigvalsh(W[j]).min() > -1e-10


def test_whitener_squares_to_w():
    g = torch.Generator().manual_seed(3)
    A = torch.randn(2, 8, 8, generator=g).double()
    W_blocks = A @ A.mT / 8 + 0.1 * torch.eye(8)
    Wh, Wh_inv = assemble_whitener(W_blocks, ridge=0.0)
    C = 16
    W_dense = torch.zeros(C, C, dtype=torch.float64)
    W_dense[:8, :8], W_dense[8:, 8:] = W_blocks[0], W_blocks[1]
    assert torch.allclose(Wh @ Wh, W_dense, atol=1e-8)
    assert torch.allclose(Wh @ Wh_inv, torch.eye(C, dtype=torch.float64), atol=1e-8)


def test_identity_whitener():
    Wh, Wh_inv = identity_whitener(12)
    assert torch.equal(Wh, torch.eye(12, dtype=torch.float64))
    assert torch.equal(Wh_inv, torch.eye(12, dtype=torch.float64))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_spectral.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bmx.cache.spectral'`.

- [ ] **Step 3: Implement `src/bmx/cache/spectral.py`**

```python
"""K4 spectral codec: query-weighted eigenbasis + waterfilled bit allocation.

The metric that scores the task is E[(qᵀ R_p (k - k_hat))²] where R_p is the
per-position RoPE rotation. That equals eᵀ W e with W = E[R_pᵀ q qᵀ R_p] —
block-diagonal per kv-head (attention logits never couple heads). Substituting
u = W^{1/2} k reduces weighted rate-distortion to plain MSE on covariance
W^{1/2} Σ_k W^{1/2}: the optimal basis is that matrix's eigenbasis and bits
waterfill on its eigenvalues (spec §3, theory grounding §8 of
docs/superpowers/specs/2026-07-12-k4-spectral-codec-design.md).

All moment/eig math is fp64; codec application is fp32.
"""

from __future__ import annotations

import dataclasses

import torch

from bmx.cache.codecs import allocate_bits_from_variance, scale_bits, tier_bits
from bmx.cache.rope import _rotate_half
from bmx.quant.rtn import rtn_quantize


def key_second_moment(M: torch.Tensor) -> torch.Tensor:
    """Uncentered per-channel second moment MᵀM/S of an (S, C) fp matrix, fp64."""
    assert M.dim() == 2, f"M must be (S, C); got {tuple(M.shape)}"
    Md = M.double()
    return Md.mT @ Md / M.shape[0]


def query_position_moment(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    h_kv: int,
    *,
    position_stride: int = 8,
) -> torch.Tensor:
    """W blocks (h_kv, d, d): E over probe queries and sampled key positions of
    (R_pᵀ q)(R_pᵀ q)ᵀ, pooled over each kv-head's GQA query group.

    R_pᵀ q is RoPE at position p with negated sin (inverse rotation). cos/sin
    are (S, d) tables from rope_cos_sin; pass cos=ones/sin=zeros for no-RoPE
    models (gpt2) — then W is the plain pooled query second moment.
    """
    h, T, d = q.shape
    assert h % h_kv == 0, f"h={h} not divisible by h_kv={h_kv}"
    grp = h // h_kv
    S = cos.shape[0]
    q64 = q.double()
    W = torch.zeros(h_kv, d, d, dtype=torch.float64)
    positions = list(range(0, S, position_stride))
    for p in positions:
        cp = cos[p].double().view(1, 1, d)
        sp = sin[p].double().view(1, 1, d)
        q_rot = q64 * cp + _rotate_half(q64) * (-sp)  # (h, T, d) = R_pᵀ q
        for j in range(h_kv):
            qj = q_rot[j * grp : (j + 1) * grp].reshape(-1, d)  # (grp*T, d)
            W[j] += qj.mT @ qj
    W /= len(positions) * grp * T
    return W


def assemble_whitener(
    W_blocks: torch.Tensor, *, ridge: float = 1e-3
) -> tuple[torch.Tensor, torch.Tensor]:
    """(h_kv, d, d) fp64 blocks -> dense (C, C) fp64 (W^{1/2}, W^{-1/2}).

    Per-block symmetric eigendecomposition; eigenvalues floored at
    ridge·λ_max(block) so near-null query directions can't explode W^{-1/2}.
    Blocks land on the diagonal in to_matrix's head-major channel order.
    """
    h_kv, d, _ = W_blocks.shape
    C = h_kv * d
    Wh = torch.zeros(C, C, dtype=torch.float64)
    Wh_inv = torch.zeros(C, C, dtype=torch.float64)
    for j in range(h_kv):
        Wj = 0.5 * (W_blocks[j] + W_blocks[j].mT)
        lam, E = torch.linalg.eigh(Wj)
        lam = lam.clamp_min(ridge * lam.max().clamp_min(1e-30))
        sl = slice(j * d, (j + 1) * d)
        Wh[sl, sl] = E @ torch.diag(lam.sqrt()) @ E.mT
        Wh_inv[sl, sl] = E @ torch.diag(lam.rsqrt()) @ E.mT
    return Wh, Wh_inv


def identity_whitener(C: int) -> tuple[torch.Tensor, torch.Tensor]:
    """W = I: the unweighted-KLT ablation path (plain Σ_k eigenbasis)."""
    eye = torch.eye(C, dtype=torch.float64)
    return eye, eye.clone()
```

(`SpectralPack`/`fit_spectral_pack`/`spectral_quantize` are Task 4 — same file.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_spectral.py -v` → 5 PASS.

- [ ] **Step 5: Full battery + commit**

Propose: `feat(spectral): query-weighted moment estimation + per-head whitener (K4 basis stage groundwork)`
**STOP for user approval.**

---

### Task 4: `spectral.py` part 2 — pack fit + waterfilled quantizer + accounting

**Files:**
- Modify: `src/bmx/cache/spectral.py`
- Test: `tests/test_spectral.py` (append)

**Interfaces:**
- Consumes: Task 2's `allocate_bits_from_variance`, Task 1's `rtn_quantize(..., mse_scale=True)`, Task 3's whiteners.
- Produces:
  - `SpectralPack` dataclass: `enc (C,C) fp32`, `dec (C,C) fp32`, `lam (C,) fp32`, `bits (C,) int64`, `group: int`, `tiers: tuple[int, ...]`, `budget: float`. Encoding: `Y = M @ enc`; decoding: `M_hat = Y_hat @ dec.mT`. Identity: `enc @ dec.mT == I`.
  - `fit_spectral_pack(M_fit, Wh, Wh_inv, budget: float, *, tiers=(0, 2, 3, 4, 5, 6, 8), group: int = 64) -> SpectralPack`
  - `spectral_quantize(M, pack, *, mse_scale: bool = True) -> tuple[torch.Tensor, float]` — `(M_hat, bpe_model)` where `bpe_model = mean(bits) + scale_bits(group)` (model-level accounting).
  - `skeptic_charge(C: int, S: int, tiers: tuple[int, ...]) -> float` — `16·C/S + tier_bits(tiers, S)` (per-sequence decoder matrix + per-direction bit map).

Tier note: tiers exclude 1 (symmetric RTN has qmax=0 at 1 bit) and cap at 8 (int8 codes) — the theory-predicted 5–9-bit spike range is covered by {5, 6, 8}.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_spectral.py`):

```python
from bmx.cache.codecs import scale_bits, tier_bits
from bmx.cache.spectral import (
    SpectralPack,
    fit_spectral_pack,
    skeptic_charge,
    spectral_quantize,
)


def _spiked_keys(S=512, C=64, seed=0, spike_dirs=None, spike_std=(30.0, 30.0)):
    """Keys with two planted spike directions over unit noise. Returns (M, dirs)."""
    g = torch.Generator().manual_seed(seed)
    if spike_dirs is None:
        raw = torch.randn(C, 2, generator=g)
        spike_dirs, _ = torch.linalg.qr(raw)  # (C, 2) orthonormal
    z = torch.randn(S, 2, generator=g) * torch.tensor(spike_std)
    noise = torch.randn(S, C, generator=g)
    return z @ spike_dirs.mT + noise, spike_dirs


def test_pack_roundtrip_identity():
    from bmx.cache.spectral import identity_whitener

    M, _ = _spiked_keys()
    Wh, Wh_inv = identity_whitener(64)
    pack = fit_spectral_pack(M, Wh, Wh_inv, budget=8.0, tiers=(8,), group=64)
    # enc @ dec.T must be the identity (basis is invertible by construction)
    eye = pack.enc.double() @ pack.dec.double().mT
    assert torch.allclose(eye, torch.eye(64, dtype=torch.float64), atol=1e-4)
    # At a uniform 8-bit allocation the codec is near-lossless
    M_hat, bpe = spectral_quantize(M, pack)
    assert (M_hat - M).norm() / M.norm() < 0.02
    assert abs(bpe - (8.0 + scale_bits(64))) < 1e-9


def test_waterfill_funds_spikes_drops_bulk():
    from bmx.cache.spectral import identity_whitener

    M, _ = _spiked_keys()
    Wh, Wh_inv = identity_whitener(64)
    pack = fit_spectral_pack(M, Wh, Wh_inv, budget=2.0)
    assert pack.bits[:2].min() >= 5, f"spike dirs underfunded: {pack.bits[:4]}"
    assert (pack.bits == 0).sum() > 0, "tight budget must drop bulk directions"
    assert pack.bits.float().mean().item() <= 2.0 + 1e-9


def test_weighted_basis_beats_unweighted_on_query_skewed_source():
    """P4 mechanism test: two equal-variance key spikes, queries read only one.
    The W-weighted basis funds the query-read spike and wins on logit distortion
    at the same budget."""
    from bmx.cache.collect import from_matrix
    from bmx.cache.metrics import logit_distortion
    from bmx.cache.spectral import (
        assemble_whitener,
        identity_whitener,
        query_position_moment,
    )

    C, S, T = 64, 512, 64
    M, dirs = _spiked_keys(S=S, C=C, seed=0)
    g = torch.Generator().manual_seed(7)
    # Queries aligned with spike 1 only (plus small noise); h = h_kv = 1 head.
    q = (
        torch.randn(T, 1, generator=g) * dirs[:, 1].unsqueeze(0)
        + 0.05 * torch.randn(T, C, generator=g)
    ).unsqueeze(0)  # (1, T, C)
    cos, sin = torch.ones(S, C), torch.zeros(S, C)  # no RoPE in this synthetic

    W = query_position_moment(q, cos, sin, h_kv=1, position_stride=64)
    Wh, Wh_inv = assemble_whitener(W)
    eWh, eWh_inv = identity_whitener(C)

    budget = 2.0
    p_w = fit_spectral_pack(M, Wh, Wh_inv, budget)
    p_u = fit_spectral_pack(M, eWh, eWh_inv, budget)
    K = from_matrix(M, 1)
    d_w = logit_distortion(K, from_matrix(spectral_quantize(M, p_w)[0], 1), q)
    d_u = logit_distortion(K, from_matrix(spectral_quantize(M, p_u)[0], 1), q)
    assert d_w < d_u, f"weighted {d_w} !< unweighted {d_u}"


def test_spectral_deterministic():
    from bmx.cache.spectral import identity_whitener

    M, _ = _spiked_keys(seed=5)
    Wh, Wh_inv = identity_whitener(64)
    p1 = fit_spectral_pack(M, Wh, Wh_inv, 2.5)
    p2 = fit_spectral_pack(M, Wh, Wh_inv, 2.5)
    assert torch.equal(p1.bits, p2.bits)
    a, _ = spectral_quantize(M, p1)
    b, _ = spectral_quantize(M, p2)
    assert torch.equal(a, b)


def test_skeptic_charge_formula():
    assert abs(
        skeptic_charge(1024, 32768, (0, 2, 3, 4, 5, 6, 8))
        - (16.0 * 1024 / 32768 + tier_bits((0, 2, 3, 4, 5, 6, 8), 32768))
    ) < 1e-12
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_spectral.py -v`
Expected: new tests FAIL with ImportError (`SpectralPack` etc. not defined); Task-3 tests still PASS.

- [ ] **Step 3: Implement** (append to `spectral.py`):

```python
@dataclasses.dataclass
class SpectralPack:
    """Corpus-fit spectral codec for one (layer, side): basis + bit allocation.

    Y = M @ enc (encode); M_hat = Y_hat @ dec.mT (decode); enc @ dec.mT = I.
    enc = W^{1/2} E, dec = W^{-1/2} E where E is the eigenbasis of
    W^{1/2} Σ_k W^{1/2} (descending eigenvalues lam). bits waterfills lam.
    Model-level accounting: the pack ships with the model (zero per-sequence
    bits); skeptic-mode per-sequence charge is skeptic_charge(C, S, tiers).
    """

    enc: torch.Tensor  # (C, C) fp32
    dec: torch.Tensor  # (C, C) fp32
    lam: torch.Tensor  # (C,) fp32, descending
    bits: torch.Tensor  # (C,) int64, members of tiers
    group: int
    tiers: tuple[int, ...]
    budget: float


def fit_spectral_pack(
    M_fit: torch.Tensor,
    Wh: torch.Tensor,
    Wh_inv: torch.Tensor,
    budget: float,
    *,
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8),
    group: int = 64,
) -> SpectralPack:
    """Fit basis + allocation on M_fit (the calibration matrix). fp64 internally."""
    assert 1 not in tiers, "symmetric RTN is undefined at 1 bit (qmax=0)"
    Sigma = key_second_moment(M_fit)
    T = Wh @ Sigma @ Wh
    lam, E = torch.linalg.eigh(0.5 * (T + T.mT))
    lam = lam.flip(0).clamp_min(0.0)  # descending
    E = E.flip(1)
    bits = allocate_bits_from_variance(lam, budget, tiers)
    return SpectralPack(
        enc=(Wh @ E).float(),
        dec=(Wh_inv @ E).float(),
        lam=lam.float(),
        bits=bits,
        group=group,
        tiers=tuple(tiers),
        budget=float(budget),
    )


def spectral_quantize(
    M: torch.Tensor, pack: SpectralPack, *, mse_scale: bool = True
) -> tuple[torch.Tensor, float]:
    """Quantize (S, C) M with a fitted pack. Returns (M_hat, bpe_model).

    bpe_model = mean payload + groupwise-scale term (model-level accounting —
    the pack itself ships with the model). Add skeptic_charge(C, S, tiers) for
    the per-sequence-charged view.
    """
    S, C = M.shape
    assert pack.enc.shape == (C, C), f"pack C mismatch: {pack.enc.shape} vs C={C}"
    assert S % pack.group == 0, f"S={S} not divisible by group={pack.group}"
    Y = M @ pack.enc.to(M.dtype)
    Y_hat = torch.zeros_like(Y)
    for b in sorted(set(int(x) for x in pack.bits.tolist())):
        if b == 0:
            continue
        cols = (pack.bits == b).nonzero(as_tuple=True)[0]
        Y_hat[:, cols] = rtn_quantize(
            Y[:, cols].mT, b, pack.group, mse_scale=mse_scale
        ).mT
    M_hat = Y_hat @ pack.dec.mT.to(M.dtype)
    bpe = float(pack.bits.float().mean().item()) + scale_bits(pack.group)
    return M_hat, bpe


def skeptic_charge(C: int, S: int, tiers: tuple[int, ...]) -> float:
    """Per-sequence charge when the pack is NOT granted model-level status:
    one fp16 C×C decoder matrix (16·C/S) + the per-direction bit map."""
    return 16.0 * C / S + tier_bits(tiers, S)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_spectral.py -v` → 10 PASS. The P4 mechanism test (`test_weighted_basis_beats_unweighted_on_query_skewed_source`) is the load-bearing one — if it fails, STOP and re-derive the enc/dec orientation before proceeding (do not weaken the test).

- [ ] **Step 5: Full battery + commit**

Propose: `feat(spectral): SpectralPack fit + waterfilled quantizer — weighted basis wins the P4 mechanism test on planted query-skewed source`
**STOP for user approval.**

---

### Task 5: Calibration corpus — `token_offset` in cache collection + local corpus caches

Corpus-vs-heldout transfer (G0) needs caches from **distinct documents**. `load_eval_tokens` always takes the leading n_tokens of wikitext-2-test; add an offset so different slices act as different documents.

**Files:**
- Modify: `src/bmx/eval/layer_swap.py:48-62` (`load_eval_tokens`), `experiments/collect_cache.py`
- Test: `tests/test_cache_collect.py` (append one test)

**Interfaces:**
- Produces: `load_eval_tokens(model_name, dataset, n_tokens, token_offset: int = 0)`; `collect_cache.py --token-offset N` writing `results/cache/<model>_<S>_off<N>.safetensors` when N>0. Consumed by Tasks 6–7 as `corpus_cache_paths`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_cache_collect.py`; no downloads — monkeypatch the dataset):

```python
def test_load_eval_tokens_offset(monkeypatch):
    import bmx.eval.layer_swap as ls

    class _FakeTok:
        def __call__(self, text, return_tensors, truncation, max_length):
            import torch

            ids = torch.arange(max_length).unsqueeze(0)
            return type("E", (), {"input_ids": ids})()

    monkeypatch.setattr(
        ls, "load_dataset", lambda *a, **k: {"text": ["x"]}, raising=False
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: _FakeTok()
    )
    base = ls.load_eval_tokens("gpt2", n_tokens=16)
    off = ls.load_eval_tokens("gpt2", n_tokens=16, token_offset=8)
    assert base.shape == off.shape == (16,)
    assert off[0].item() == base[0].item() + 8
```

(Check how `load_dataset` is imported inside the function — it is a local import, so patch `datasets.load_dataset` instead if the module-level monkeypatch doesn't bite; adjust the patch target after reading the function, keeping the assertion block unchanged.)

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_cache_collect.py::test_load_eval_tokens_offset -v` → FAIL (`unexpected keyword argument 'token_offset'`).

- [ ] **Step 3: Implement.** In `load_eval_tokens`, add `token_offset: int = 0`, tokenize with `max_length=token_offset + n_tokens`, return `ids.input_ids[0][token_offset:]`. In `collect_cache.py` `Config`, add `token_offset: int = 0`; pass it through; auto filename gains `_off{cfg.token_offset}` suffix when nonzero.

- [ ] **Step 4: Run to verify pass**, then full battery.

- [ ] **Step 5: Collect the local gpt2 corpus** (runs, not tests — minutes total):

```bash
uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024
uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024 --token-offset 1024
uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024 --token-offset 2048
uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024 --token-offset 3072
uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024 --token-offset 4096
```

Expected: 5 files under `results/cache/` (gitignored — local artifacts only).

- [ ] **Step 6 (bounded contingency): attempt ONE extra Llama cache locally.**

```bash
uv run python experiments/collect_cache.py --model-name meta-llama/Llama-3.1-8B --seq-len 2048 --token-offset 2048
```

CPU bf16 prefill of 2048 tokens on an 8B model ≈ 3–10 min if RAM allows (~16 GB for weights). **Bound: if it OOMs or exceeds ~30 min, abort, record the fact in the Task-11 results doc, and the Llama cross-document corpus point moves to the Stage-3 VM batch preamble** — the Llama oracle/heldout modes (Tasks 6–7) proceed regardless on the existing `llama-3.1-8b_2048.safetensors`. If it succeeds, repeat for offsets 4096 and 6144 (3 extra docs total).

- [ ] **Step 7: Commit** (code + test only; caches are gitignored).

Propose: `feat(collect): token_offset for multi-document calibration corpora (K4 Stage-0 transfer test)`
**STOP for user approval.**

---

### Task 6: `experiments/k4_spectra.py` — Stage 0 (spectra, transfer, drift decomposition)

**Files:**
- Create: `experiments/k4_spectra.py`
- Test: `tests/test_k4_experiments.py` (new)

**Interfaces:**
- Consumes: Tasks 3–4 (`spectral.py` public API), cache files, `bmx.artifacts.create_run/write_metrics`.
- Produces: `results/k4_spectra/<run-id>/metrics.parquet` with columns `(model, layer, weighted, fit_mode, budget, bpe_model, bpe_skeptic, bpe_skeptic_deploy, rel_fro, logit, logit_rope, am_gm, top16_energy, n_zero_dirs)` + `g0_verdict.json` sidecar. `fit_mode ∈ {oracle, heldout, corpus}` (corpus rows only when `corpus_cache_paths` given).

Semantics (spec §4 Stage 0, operationalized):
- **oracle** — pack fit on all S rows of the scored cache; scored on the tail half (region-matched, so all modes score the SAME rows).
- **heldout** — pack fit on rows `[:S//2]`, scored on rows `[S//2:]` (the frozen-prefix analogue; at S=2048, a 1024-row fit for C=1024 is at the rank-deficiency boundary — exactly the confound spec §9 row 5 names; the corpus mode is the fix).
- **corpus** — pack fit on the concatenation of OTHER documents' caches for the same layer; scored on this cache's tail half.
- **G0 (per spec):** retention = win(fit_mode)/win(oracle) where win = logit_rope(uniform baseline)/logit_rope(spectral), at the reference budget 2.5, uniform baseline = `lowrank_rtn_channel` r16 b3 scored on the same tail (the standing k2b-K reference). Gate: mean retention ≥ 0.9 for the mode that will ship (corpus where available, else heldout as the honest lower bound).

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_k4_experiments.py
import json

import torch

from bmx.cache.collect import save_cache


def _tiny_cache(path, S=128, C=16, h_kv=2, T=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = C // h_kv
    raw = torch.randn(C, 3, generator=g)
    dirs, _ = torch.linalg.qr(raw)
    z = torch.randn(S, 3, generator=g) * torch.tensor([20.0, 12.0, 8.0])
    M = (z @ dirs.mT + torch.randn(S, C, generator=g)).half()
    K = M.reshape(S, h_kv, d).permute(1, 0, 2)  # from_matrix layout
    tensors = {}
    for i in range(2):  # 2 layers
        tensors[f"layer{i}.k_pre"] = K.contiguous()
        tensors[f"layer{i}.k"] = K.contiguous()
        tensors[f"layer{i}.v"] = torch.randn(h_kv, S, d, generator=g).half()
        tensors[f"layer{i}.q"] = torch.randn(h_kv * 2, T, d, generator=g).half()
    save_cache(tensors, path)


def test_k4_spectra_smoke(tmp_path):
    import pandas as pd

    from experiments.k4_spectra import Config, main

    main_path = tmp_path / "main.safetensors"
    other_path = tmp_path / "other.safetensors"
    _tiny_cache(main_path, seed=0)
    _tiny_cache(other_path, seed=1)
    cfg = Config(
        cache_path=str(main_path),
        corpus_cache_paths=(str(other_path),),
        model_label="tiny",
        budgets=(2.5,),
        group=16,
        out_root=str(tmp_path / "results"),
    )
    run_dir = main(cfg)
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert set(df.fit_mode.unique()) == {"oracle", "heldout", "corpus"}
    assert {"am_gm", "logit", "bpe_model", "bpe_skeptic"} <= set(df.columns)
    verdict = json.loads((run_dir / "g0_verdict.json").read_text())
    assert "retention_heldout" in verdict and "retention_corpus" in verdict
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_k4_experiments.py -v` → FAIL (no module `experiments.k4_spectra`).

- [ ] **Step 3: Implement `experiments/k4_spectra.py`.** Follow `k2d_lrtq_gate.py`'s skeleton exactly (layer regex, RoPE setup + self-validation, emit/print pattern). Core structure:

```python
"""K4 Stage 0: weighted/unweighted spectra + basis-transfer gate (G0).

Per layer, fits SpectralPacks under three fit modes (oracle / heldout /
corpus), scores them REGION-MATCHED on the tail half of the cache against the
uniform k2b-K reference, and emits per-layer spectra stats + the G0 retention
verdict. See docs/superpowers/specs/2026-07-12-k4-spectral-codec-design.md §4.
"""

from __future__ import annotations

import dataclasses
import json
import re

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.codecs import quantize_cache
from bmx.cache.collect import from_matrix, load_cache, to_matrix
from bmx.cache.metrics import logit_distortion, rel_fro
from bmx.cache.rope import apply_rope
from bmx.cache.spectral import (
    assemble_whitener,
    fit_spectral_pack,
    identity_whitener,
    key_second_moment,
    query_position_moment,
    skeptic_charge,
    spectral_quantize,
)

_LAYER_RE = re.compile(r"^layer(\d+)\.(k|v|q|k_pre)$")
DEPLOY_S = 32768


@dataclasses.dataclass
class Config:
    cache_path: str
    corpus_cache_paths: tuple[str, ...] = ()
    model_label: str = ""
    model_name: str = ""  # HF repo id for RoPE; empty => stored-basis logit only
    budgets: tuple[float, ...] = (2.5,)
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8)
    group: int = 64
    ridge: float = 1e-3
    position_stride: int = 8
    ref_rank: int = 16  # the uniform k2b-K reference arm
    ref_bits: int = 3
    seed: int = 0
    max_layers: int = 0
    out_root: str = ""
```

`main(cfg) -> Path` (returns the run dir for the smoke test). Per layer:
1. Build `M` (S, C) from `k_pre`, queries `Q`, RoPE tables as in `k2d_lrtq_gate.py` (including the layer-0 RoPE self-validation assert). For no-RoPE runs use `cos = torch.ones(S, d)`, `sin = torch.zeros(S, d)` so `query_position_moment` degrades to the plain pooled moment.
2. Whiteners: `W = query_position_moment(q, cos, sin, h_kv, position_stride=cfg.position_stride)`; `Wh, Wh_inv = assemble_whitener(W, ridge=cfg.ridge)`; unweighted pair from `identity_whitener(C)`.
3. Fit matrices per mode: oracle `M`, heldout `M[: S // 2]`, corpus `torch.cat([to_matrix(other[f"layer{i}.k_pre"]).float() for other in corpus_caches])` (skip mode if no corpus paths).
4. Tail scoring region `tail = slice(S // 2, S)`; scoring helper (all arms scored on the SAME tail rows, RoPE positions kept absolute):

```python
def _score_tail(M_hat, h_kv, tail, K_post_true, Q, cos, sin, rope_ready, k_pre_t, M):
    K_hat = from_matrix(M_hat, h_kv)[:, tail, :].float()
    rf = rel_fro(M_hat[tail], M[tail])
    if rope_ready:
        K_hat_rope = apply_rope(K_hat, cos[tail], sin[tail])
        lg_rope = logit_distortion(K_post_true[:, tail], K_hat_rope, Q)
        lg = logit_distortion(k_pre_t.float()[:, tail], K_hat, Q)
    else:
        lg = logit_distortion(k_pre_t.float()[:, tail], K_hat, Q)
        lg_rope = float("nan")
    return rf, lg, lg_rope
```

5. Rows: per (weighted ∈ {True, False}, fit_mode, budget): fit pack on the mode's fit matrix (with the mode's OWN whitener — corpus mode also estimates W from... **the scored cache's queries** — queries are probe-side and known at read time in every mode; only the KEY statistics vary by fit mode. State this in the docstring), `spectral_quantize(M, pack)`, tail-score, emit with `bpe_skeptic = bpe + skeptic_charge(C, S, tiers)` and `bpe_skeptic_deploy = bpe + skeptic_charge(C, DEPLOY_S, tiers)`, plus spectra stats from the oracle-weighted pack: `am_gm = lam.mean()/lam.clamp_min(1e-12).log().mean().exp()`, `top16_energy = lam[:16].sum()/lam.sum()`, `n_zero_dirs = int((pack.bits == 0).sum())`.
6. Reference row per layer: `quantize_cache("lowrank_rtn_channel", M, bits=cfg.ref_bits, rank=cfg.ref_rank, group=cfg.group)`, tail-scored, `fit_mode="reference"`.
7. `g0_verdict.json`: for the headline metric (`logit_rope` when RoPE, else `logit`), per fit_mode m ∈ {heldout, corpus}: `retention_m = mean over layers of [ (ref/spectral_m) / (ref/spectral_oracle) ]` at budget 2.5, weighted basis; plus `g0_pass_m = retention_m >= 0.9`. Write and print.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_k4_experiments.py -v` → PASS.

- [ ] **Step 5: Full battery + commit**

Propose: `feat(exp): k4_spectra Stage-0 gate — spectra census + oracle/heldout/corpus basis-transfer retention (G0)`
**STOP for user approval.**

---

### Task 7: `experiments/k4_frontier.py` — Stage 1 (the frontier duel, gate G1)

**Files:**
- Create: `experiments/k4_frontier.py`
- Test: `tests/test_k4_experiments.py` (append)

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: `results/k4_frontier/<run-id>/metrics.parquet`, columns `(model, layer, kind, arm, fit_mode, weighted, budget, bits, rank, mse_scale, bpe_model, bpe_skeptic, bpe_skeptic_deploy, rel_fro, logit, logit_rope)` + `g1_verdict.json`. Task 8 plots from this parquet; Task 10 reads its per-layer distortion-vs-budget curves.

Arm roster (all on `k_pre`, tail-region-scored like Task 6 so spectral fit modes stay honest; budgets `(1.5, 2.0, 2.5, 3.0, 3.5, 4.0)`):

| arm | what it is | why it's here |
|---|---|---|
| `spectral` (weighted × {oracle, heldout, corpus?}) | the K4 codec | the candidate |
| `spectral_unweighted` (same fit modes) | W = I ablation | P4 test |
| `spectral_randbasis` | enc/dec = seeded random orthogonal, allocation from the fit-region variances of `Y = M @ Q` | the load-bearing eigenstructure control (spec §9) |
| `turboquant_mse` b ∈ {2, 3, 4}, kinds `k_pre` and `k` | the incumbent | the duel target |
| `lowrank_rtn_channel` r16, b ∈ {2, 3, 4, 5} | the uniform bit-sweep frontier | the June-mandated baseline family |
| `lowrank_turboquant` r ∈ {16, 32}, b=2 | k2t | rung 1, continuity with k2d |
| `k2t_coeffquant` r=32, coeff_bits=6, b=2 (experiment-local) | quantized subspace coefficients | P3 test |
| `rtn_channel` b3 × mse_scale ∈ {False, True} (experiment-local) | step-policy ablation | Task-1 win quantified on real caches |

Experiment-local helpers (in `k4_frontier.py`, not the codec registry):

```python
def _rtn_channel_arm(M, bits, group, mse_scale):
    from bmx.cache.codecs import scale_bits
    from bmx.quant.rtn import rtn_quantize

    M_hat = rtn_quantize(M.mT, bits, group, mse_scale=mse_scale).mT
    return M_hat, bits + scale_bits(group)


def _k2t_coeffquant_arm(M, rank, coeff_bits, res_bits, group, seed, factors):
    """P3: quantize the low-rank coefficients (Us) instead of storing fp16.
    bpe: residual turboquant payload+norm, Us at coeff_bits (+ its group
    scales), V still fp16 (16·r/S per entry)."""
    from bmx.cache.codecs import norm_bits, quantize_cache, scale_bits
    from bmx.quant.rtn import rtn_quantize

    S, C = M.shape
    Us, V = factors
    Us_q = rtn_quantize(Us.mT, coeff_bits, group, mse_scale=True).mT
    V_st = V.half().float()
    L = Us_q @ V_st.mT
    R_hat, _ = quantize_cache("turboquant_mse", M - L, bits=res_bits, seed=seed)
    bpe = (
        res_bits
        + norm_bits(1, C)                      # residual per-row norm
        + coeff_bits * rank / C                # Us payload
        + (16.0 / group) * rank / C            # Us group scales
        + 16.0 * rank / S                      # V fp16
    )
    return L + R_hat, bpe
```

`g1_verdict.json` (the gate, spec §4 Stage 1): for each spectral point (weighted, shipping fit mode), interpolate the `turboquant_mse` k_pre curve (log-distortion vs bpe, linear interp between its three points) at the spectral point's `bpe_model` AND at its `bpe_skeptic_deploy`; record `win_model = d_tq_interp/d_spectral`, `win_skeptic_deploy` likewise, and per-budget per-layer win fractions. `g1_pass` = spectral strictly better (win > 1) at every budget in both accounting modes AND in ≥90% of layers. Also emit `p3_verdict` (k2t_coeffquant vs lowrank_turboquant r16@2 at their measured bpes) and `p4_verdict` (weighted vs unweighted mean ratio).

- [ ] **Step 1: Write the failing smoke test** (append to `tests/test_k4_experiments.py`):

```python
def test_k4_frontier_smoke(tmp_path):
    import json

    import pandas as pd

    from experiments.k4_frontier import Config, main

    main_path = tmp_path / "main.safetensors"
    _tiny_cache(main_path, seed=0)
    cfg = Config(
        cache_path=str(main_path),
        model_label="tiny",
        budgets=(2.0, 3.0),
        group=16,
        ranks=(4,),
        coeffquant_rank=4,
        out_root=str(tmp_path / "results"),
    )
    run_dir = main(cfg)
    df = pd.read_parquet(run_dir / "metrics.parquet")
    for arm in ("spectral", "spectral_unweighted", "spectral_randbasis",
                "turboquant_mse", "lowrank_rtn_channel", "k2t_coeffquant",
                "rtn_channel"):
        assert (df.arm == arm).any(), f"missing arm {arm}"
    v = json.loads((run_dir / "g1_verdict.json").read_text())
    assert "g1_pass" in v and "p3_verdict" in v and "p4_verdict" in v
```

- [ ] **Step 2: Run to verify failure** → ModuleNotFoundError.

- [ ] **Step 3: Implement** following the `k2d_lrtq_gate.py` job-loop skeleton: same Config fields as Task 6 plus `budgets`, `ranks: tuple[int, ...] = (16, 32)`, `coeffquant_rank: int = 32`, `coeffquant_bits: int = 6`, `tq_bits: tuple[int, ...] = (2, 3, 4)`, `uniform_bits: tuple[int, ...] = (2, 3, 4, 5)`. Reuse the `_score_tail` helper (lift it into a small shared `experiments/_k4_common.py` together with `_LAYER_RE`, the cache-loading loop, and the RoPE block — both k4 experiments import from it; keep it <80 lines). SVD factors cached per (layer, rank) exactly as `k2d_lrtq_gate.get_svd`. Spectral packs fit per (layer, weighted, fit_mode) ONCE and reused across budgets (refit per budget is only the allocation — `fit_spectral_pack` is cheap; simplicity wins, fit per budget). Print the running summary in the k2d `emit` style; end with the G1 verdict block.

- [ ] **Step 4: Run to verify pass** — both smoke tests.

- [ ] **Step 5: Full battery + commit**

Propose: `feat(exp): k4_frontier Stage-1 duel — spectral vs turboquant frontier with G1/P3/P4 machine-readable verdicts`
**STOP for user approval.**

---

### Task 8: `experiments/plots/plot_k4_frontier.py`

**Files:**
- Create: `experiments/plots/plot_k4_frontier.py`
- Test: `tests/test_k4_experiments.py` (append)

**Interfaces:**
- Consumes: Task 7's parquet schema.
- Produces: `make_figures(df: pd.DataFrame, out_dir: str) -> list[Path]` (the repo's plot contract) emitting `k4_frontier_model.png` (logit_rope vs bpe_model, log-y, one line per arm, layer-mean ± sem) and `k4_frontier_skeptic.png` (same vs `bpe_skeptic_deploy`), plus `k4_structure_tax.png` (bar: turboquant_mse vs rtn_channel vs spectral at matched ~3 bpe — the Hadamard-tax exhibit).

- [ ] **Step 1: Failing test** (append):

```python
def test_k4_frontier_figures(tmp_path):
    import pandas as pd

    from experiments.plots.plot_k4_frontier import make_figures

    rows = []
    for arm, base in (("spectral", 0.03), ("turboquant_mse", 0.09),
                      ("rtn_channel", 0.12)):
        for i, bpe in enumerate((2.0, 3.0, 4.0)):
            rows.append(dict(model="tiny", layer=i % 2, kind="k_pre", arm=arm,
                             fit_mode="oracle", weighted=True, budget=bpe,
                             bits=int(bpe), rank=0, mse_scale=False,
                             bpe_model=bpe, bpe_skeptic=bpe + 8.0,
                             bpe_skeptic_deploy=bpe + 0.5,
                             rel_fro=base, logit=base, logit_rope=base * (4.0 ** -i)))
    paths = make_figures(pd.DataFrame(rows), str(tmp_path))
    names = {p.name for p in paths}
    assert {"k4_frontier_model.png", "k4_frontier_skeptic.png",
            "k4_structure_tax.png"} <= names
```

- [ ] **Step 2: Verify failure**, **Step 3: implement** (matplotlib, no seaborn; select rows explicitly by arm — never blind-concat, per the repo pitfall), **Step 4: verify pass**, **Step 5: full battery + commit**.

Propose: `feat(plots): K4 frontier figures — model/skeptic accounting curves + structure-tax exhibit`
**STOP for user approval.**

---

### Task 9: Per-layer codec specs in `quantized_prefill_ppl`

Stage 2's G2 check applies a **different bit-width per layer**. Extend `quantized_prefill_ppl` with optional per-layer spec lists; default path stays byte-identical (pinned by existing ppl tests).

**Files:**
- Modify: `src/bmx/cache/ppl_eval.py`
- Test: `tests/test_ppl_per_layer.py` (new; use `tests/factories.py`'s tiny offline model — check its factory name first, it is the one `tests/test_k3_experiment.py` uses)

**Interfaces:**
- Produces: `quantized_prefill_ppl(model, input_ids, n_prefill, k_spec, v_spec, state=None, k_specs: list[CacheCodecSpec] | None = None, v_specs: list[CacheCodecSpec] | None = None)` — when given, `k_specs[i]`/`v_specs[i]` override `k_spec`/`v_spec` for layer i (`len == n_layer` asserted; `bpe_k`/`bpe_v` returned as the mean over layers). Constraint: mixed `pre_rope` values across `k_specs` are rejected (one RoPE regime per run) — assert all equal `k_spec.pre_rope`.

- [ ] **Step 1: Failing test:**

```python
# tests/test_ppl_per_layer.py
import torch

from bmx.cache.ppl_eval import CacheCodecSpec, quantized_prefill_ppl
from tests.factories import tiny_llama  # adjust to the factory's real name


def _ids(model, n=96):
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, model.config.vocab_size, (1, n), generator=g)


def test_per_layer_specs_identity_invariant():
    """A per-layer list of identical specs must reproduce the single-spec result."""
    model = tiny_llama()
    ids = _ids(model)
    spec = CacheCodecSpec(arm="rtn_channel", bits=3, group=8)
    v = CacheCodecSpec(arm="fp16")
    n_layer = model.config.num_hidden_layers
    a = quantized_prefill_ppl(model, ids, 64, spec, v)
    b = quantized_prefill_ppl(
        model, ids, 64, spec, v, k_specs=[spec] * n_layer, v_specs=[v] * n_layer
    )
    assert abs(a["ppl"] - b["ppl"]) < 1e-6
    assert abs(a["bpe_k"] - b["bpe_k"]) < 1e-9


def test_per_layer_specs_mixed_bits_runs():
    model = tiny_llama()
    ids = _ids(model)
    n_layer = model.config.num_hidden_layers
    k_specs = [
        CacheCodecSpec(arm="rtn_channel", bits=2 + (i % 2), group=8)
        for i in range(n_layer)
    ]
    out = quantized_prefill_ppl(
        model, ids, 64, k_specs[0], CacheCodecSpec(arm="fp16"), k_specs=k_specs
    )
    assert out["ppl"] > 0 and 2.0 < out["bpe_k"] < 3.5
```

(Before writing, open `tests/factories.py` and use its actual tiny-Llama factory name and config; if only a GPT-2-style factory exists, use it with `arm="rtn_channel"`, `pre_rope=False` — the invariant is architecture-agnostic.)

- [ ] **Step 2: Verify failure** (`unexpected keyword argument 'k_specs'`).

- [ ] **Step 3: Implement.** In the layer loop, resolve `k_spec_i = k_specs[i] if k_specs is not None else k_spec` (same for V); accumulate `bpe_k`/`bpe_v` into running means instead of plain overwrite when lists are given. Asserts: list lengths equal `n_layer`; `all(s.pre_rope == k_spec.pre_rope for s in k_specs)`; `not any(s.pre_rope for s in v_specs or [])`.

- [ ] **Step 4: Verify pass** + existing ppl tests still green: `uv run pytest tests/test_ppl_per_layer.py tests/ -q -k "ppl"`.

- [ ] **Step 5: Full battery + commit**

Propose: `feat(ppl): optional per-layer codec specs in quantized_prefill_ppl (K4 across-layer allocation harness)`
**STOP for user approval.**

---

### Task 10: `experiments/k4_alloc.py` — Stage 2 (sensitivity + across-layer allocation, gate G2)

**Files:**
- Create: `experiments/k4_alloc.py`
- Test: `tests/test_k4_experiments.py` (append — the allocator is pure math, unit-testable without a model)

**Interfaces:**
- Consumes: Task 9's per-layer ppl; Task 7's parquet (per-layer distortion-vs-budget curves); `run_prefill` for a reusable prefill state.
- Produces: `results/k4_alloc/<run-id>/metrics.parquet` (sensitivity rows + ppl rows) + `g2_verdict.json` + `allocation.json` (per-layer bit table). Public function `greedy_layer_allocation(curves: dict[int, dict[float, float]], s: dict[int, float], budgets: tuple[float, ...], target_mean: float) -> dict[int, float]`.

Three parts:
- **A — sensitivity census** (model forwards; gpt2 default, `--model-name meta-llama/...` flagged SLOW ~32 prefills × minutes): one `run_prefill` state, then per layer i: `k_specs` = fp16 everywhere except layer i gets `CacheCodecSpec(arm="turboquant_mse", bits=2)`; `s_i = log(ppl_i) − log(ppl_fp16)` (NLL delta). Emit `(layer, kind="sensitivity", s_i, ppl)` rows.
- **B — allocation**: greedy marginal upgrade, provably optimal for convex curves:

```python
def greedy_layer_allocation(curves, s, budgets, target_mean):
    """curves[layer][budget] = distortion; start every layer at min(budgets),
    repeatedly upgrade the layer with the largest s[l]*(D[cur]-D[next]) per
    budget-unit until the mean budget reaches target_mean. Deterministic
    (ties broken by layer index)."""
    grid = sorted(budgets)
    cur = {l: 0 for l in curves}  # index into grid
    def mean_b():
        return sum(grid[i] for i in cur.values()) / len(cur)
    while mean_b() < target_mean - 1e-9:
        best, best_gain = None, -1.0
        for l in sorted(curves):
            i = cur[l]
            if i + 1 >= len(grid):
                continue
            gain = s[l] * (curves[l][grid[i]] - curves[l][grid[i + 1]])
            gain /= grid[i + 1] - grid[i]
            if gain > best_gain:
                best, best_gain = l, gain
        if best is None:
            break
        cur[best] += 1
    return {l: grid[i] for l, i in cur.items()}
```

- **C — G2 verdict** (gpt2 end-to-end): with the Task-7 gpt2 frontier parquet supplying `curves` (arm=`turboquant_mse`, kind=`k_pre`, per-layer `logit` at each bits) and Part A's `s`, build per-layer `k_specs` (`turboquant_mse` at the allocated bits) at `target_mean ∈ {2.5, 3.0}`; compare ppl vs uniform `turboquant_mse` at the same mean bits (verify realized mean via returned `bpe_k` within 0.1). `g2_pass` = allocated ppl ≤ uniform ppl at both targets. Note in the doc: this exercises the ALLOCATION lever with existing arms — the spectral codec's own across-layer allocation lands with the Stage-3 integration plan.

- [ ] **Step 1: Failing unit test for the allocator** (append to `tests/test_k4_experiments.py`):

```python
def test_greedy_layer_allocation_prefers_sensitive_steep_layers():
    from experiments.k4_alloc import greedy_layer_allocation

    grid = (2.0, 3.0, 4.0)
    # Layer 0: sensitive + steep curve; layer 1: insensitive + flat.
    curves = {0: {2.0: 0.4, 3.0: 0.1, 4.0: 0.02}, 1: {2.0: 0.05, 3.0: 0.04, 4.0: 0.039}}
    s = {0: 1.0, 1: 0.05}
    alloc = greedy_layer_allocation(curves, s, grid, target_mean=3.0)
    assert alloc[0] == 4.0 and alloc[1] == 2.0
    assert sum(alloc.values()) / 2 == 3.0


def test_greedy_layer_allocation_uniform_when_symmetric():
    from experiments.k4_alloc import greedy_layer_allocation

    grid = (2.0, 3.0, 4.0)
    curves = {l: {2.0: 0.4, 3.0: 0.1, 4.0: 0.02} for l in range(4)}
    s = {l: 1.0 for l in range(4)}
    alloc = greedy_layer_allocation(curves, s, grid, target_mean=3.0)
    assert all(v == 3.0 for v in alloc.values())
```

- [ ] **Step 2: Verify failure**, **Step 3: implement** (`Config`: `model_name="gpt2"`, `frontier_parquet: str` [required — path to the Task-7/11 gpt2 run], `n_prefill=768`, `n_cont=256`, `target_means=(2.5, 3.0)`, `sens_bits=2`, `out_root=""`), **Step 4: verify pass**, **Step 5: full battery + commit**.

Propose: `feat(exp): k4_alloc Stage-2 — sensitivity census + greedy across-layer allocation with G2 end-to-end verdict`
**STOP for user approval.**

---

### Task 11: [RUN] Stage 0 + Stage 1 on real caches; G0/G1 verdicts doc

**Files:**
- Create: `docs/2026-07-XX-k4-stage01-results.md` (use the actual date)
- Committed artifacts: the two run dirs' `metrics.parquet` + verdict JSONs + figures (results parquets ARE committed in this repo — see `results/k2d_lrtq_gate/`).

- [ ] **Step 1:** gpt2 (corpus mode included — minutes):

```bash
uv run python experiments/k4_spectra.py --cache-path results/cache/gpt2_1024.safetensors \
  --corpus-cache-paths results/cache/gpt2_1024_off1024.safetensors results/cache/gpt2_1024_off2048.safetensors results/cache/gpt2_1024_off3072.safetensors results/cache/gpt2_1024_off4096.safetensors \
  --model-label gpt2
uv run python experiments/k4_frontier.py --cache-path results/cache/gpt2_1024.safetensors \
  --corpus-cache-paths <same> --model-label gpt2
```

- [ ] **Step 2:** Llama (oracle + heldout always; corpus if Task-5 Step 6 succeeded):

```bash
uv run python experiments/k4_spectra.py --cache-path results/cache/llama-3.1-8b_2048.safetensors \
  --model-label llama-3.1-8b --model-name meta-llama/Llama-3.1-8B
uv run python experiments/k4_frontier.py --cache-path results/cache/llama-3.1-8b_2048.safetensors \
  --model-label llama-3.1-8b --model-name meta-llama/Llama-3.1-8B
```

- [ ] **Step 3:** Generate figures: load the frontier parquet, call `make_figures`, save into the run dir.
- [ ] **Step 4:** **Independently re-derive the headline numbers from the parquets** (open them fresh; never trust the run's own prints): G0 retentions, G1 win ratios per budget in both accounting modes, P3/P4 verdicts. Cross-check one spectral bpe by hand (`mean(bits) + 16/group`).
- [ ] **Step 5:** Write the results doc in the repo's verdict style (headline → falsification checks → gate calls → caveats). **The gate calls are kill-or-confirm: write KILLED honestly if G1 fails** (spec §7 defines the narrowed fallbacks). Record P1–P4 outcomes explicitly against spec §9's predictions, and the Task-5-Step-6 contingency outcome.
- [ ] **Step 6:** Stage parquets + verdicts + figures + doc; propose `results(k4): Stage-0/1 gauntlet — G0 transfer + G1 frontier verdicts on gpt2 + Llama-3.1-8B [outcome summary in one line]`. **STOP for user approval.**

---

### Task 12: [RUN] Stage 2 + G2 verdict doc

- [ ] **Step 1:** `uv run python experiments/k4_alloc.py --model-name gpt2 --frontier-parquet results/k4_frontier/<gpt2-run-id>/metrics.parquet` (minutes). Optionally launch the Llama sensitivity overnight (`--model-name meta-llama/Llama-3.1-8B`, ~2–5 h CPU) — background it and note the run id.
- [ ] **Step 2:** Re-derive G2 from the parquet: allocated-vs-uniform ppl at both targets, sensitivity spread (report the measured s_i range vs the ~3× prior from spec §3.3).
- [ ] **Step 3:** Append a Stage-2 section to the Task-11 doc (or a new dated doc if days apart): G2 call, the per-layer allocation table, honest note that the spectral-codec-specific across-layer run is Stage-3 material.
- [ ] **Step 4:** Stage + propose `results(k4): Stage-2 across-layer allocation — sensitivity census + G2 end-to-end verdict`. **STOP for user approval.**

---

## After this plan

**Stage-3 plan (write only after the G1 verdict, per the kill-or-confirm discipline):** CACHE_ARMS registration + `k4` recipe + pack persistence (safetensors + JSON sidecar) + `StreamingQuantizedCache` integration through the arm-set gate (`a21f167`) + the VM batch (corpus cache collection preamble, LongBench/NIAH duel at matched operating points vs b2/b3/b4/asymmetric-K3V2, NIAH b3 rider, per-cell checkpointing, delta-parity licensing). If G1 kills, that plan is never written — the honest negative is Task 11's doc.

## Self-Review

**Spec coverage:** §3.1 basis/whitener → Tasks 3–4; §3.2 waterfill/coefficient-quantization/step-policy → Tasks 1, 2, 4, 7 (P3 arm); §3.3 across-layer → Tasks 9–10; §4 Stage 0 → Task 6 (G0), Stage 1 → Task 7 (G1), Stage 2 → Task 10 (G2), Stage 3 → explicitly deferred (After-this-plan section); §5 accounting → both modes in every spectral row + `skeptic_charge`; §6 scope: no entropy coding, no Lloyd residual codebooks (mse_scale is a step fix, not a codebook), no kernel work; §9 methodology: uniform sweep (Task 7), random-basis control (Task 7), oracle control (fit modes), region-matched scoring (`_score_tail`), deterministic rounding throughout; predictions P1–P4 each have a named verdict output. Gaps: none found; the spec's "sensitivity in Stage 0" lives in Task 10 (k4_alloc) instead of k4_spectra — same deliverable, consumed where it's used; flagged here deliberately.

**Placeholder scan:** every code step carries real code; the two "existing body verbatim" markers (Tasks 1–2) reference exact current line ranges of code that must move unchanged — that is an instruction, not a placeholder. Task 9/11 contain two "check the real name first" notes (test factory name, monkeypatch target) — deliberate read-before-write guards on files this plan doesn't rewrite, each with the invariant fully specified.

**Type consistency:** `allocate_bits_from_variance(var, budget_bits, tiers, *, n_search)` matches Tasks 2/4; `SpectralPack` fields match between Tasks 4/6/7; `fit_mode` values {oracle, heldout, corpus, reference} consistent across Tasks 6/7/8; parquet columns in Task 8's test match Task 7's schema; `greedy_layer_allocation` signature matches its tests; `mse_scale` kwarg spelled identically in Tasks 1/4/7.
