# Water-filling Per-Channel Bit Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reverse-water-filling per-channel bit allocator and a `lowrank_waterfill_channel` codec arm, then run an offline kill-or-confirm experiment against uniform `lowrank_rtn_channel` on key residuals, scored on logit distortion at matched bits.

**Architecture:** One pure allocation function and one new codec arm in `bmx/cache/codecs.py` (reusing the existing low-rank/SVD/per-channel-RTN machinery), plus a thin tyro experiment `experiments/k2_waterfill.py` forked from `k2_cache_arms.py`. All offline on already-collected caches; no streaming-path change.

**Tech Stack:** Python, PyTorch (CPU), tyro CLIs, pandas/parquet, pytest, uv, ruff.

**Spec:** `docs/superpowers/specs/2026-06-21-waterfill-channel-allocation-design.md`

## Global Constraints

- **Never `git commit` without the user's explicit approval.** Each task's final "Commit" step means: stage, propose the message, and STOP for approval. No `Co-Authored-By` / AI attribution, ever.
- Before any commit: `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` — all clean.
- Dependencies only via `uv add` / `uv add --dev`. No new deps are needed for this plan.
- Use the Bash tool (git bash). Shell cwd resets between turns — `cd /d/Projects/bmx` first in fresh shells.
- dtype: **fp64 in tests, fp32 in experiments/codecs.** Fail fast: shape asserts at boundaries, no silent coercion.
- **Comparisons align on realized bpe (ALL metadata counted: scales, factors, tier-index map), never on rank or nominal bits.**
- Metrics: score on logit distortion vs real queries (`logit_rope`), never Frobenius. `rel_fro` is a secondary column only.
- Tiny offline test models from `tests/factories.py`; never download in tests.
- Tests follow `tests/test_cache_codecs.py` conventions: `import torch`, `torch.Generator().manual_seed(seed)` for reproducibility, `import math`, `import pytest`.

---

## File Structure

- **Modify** `src/bmx/cache/codecs.py` — add `allocate_channel_bits` (pure fn), `_lowrank_waterfill_channel` (arm impl), register `"lowrank_waterfill_channel"` in `CACHE_ARMS` + `S_DIVISIBILITY_ARMS`, add dispatch branch in `quantize_cache`.
- **Modify** `tests/test_cache_codecs.py` — add allocator + arm tests.
- **Create** `experiments/k2_waterfill.py` — thin tyro experiment (3 arms on `k_pre`).
- **Create** `tests/test_k2_waterfill.py` — experiment smoke test on the GPT-2 cache fixture.

---

## Task 1: `allocate_channel_bits` pure function

**Files:**
- Modify: `src/bmx/cache/codecs.py`
- Test: `tests/test_cache_codecs.py`

**Interfaces:**
- Consumes: nothing (pure; `torch` only).
- Produces:
  ```python
  def allocate_channel_bits(
      R: torch.Tensor,            # (S, C) fp32 residual; channel = column
      budget_bits: float,         # target average bits/channel, e.g. 3.0
      tiers: tuple[int, ...] = (0, 2, 3, 4),
      *,
      axis: int = 0,              # token axis; per-channel variance taken over this axis
      n_search: int = 40,         # fixed bisection iterations for the water level
  ) -> torch.Tensor               # (C,) int64 bit-width per channel, each a member of tiers
  ```
  Reverse water-filling: `b_c = max(0, 0.5*log2(var_c / kappa))`, with `kappa`
  bisected so the tier-rounded mean lands at-or-just-below `budget_bits`.
  Deterministic. Higher-variance channels never get fewer bits than
  lower-variance ones (after rounding, monotone non-decreasing in variance).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cache_codecs.py`:

```python
from bmx.cache.codecs import allocate_channel_bits


def _channel_matrix(per_channel_std, s=256, seed=11):
    """(s, C) matrix whose column c has std per_channel_std[c]."""
    g = torch.Generator().manual_seed(seed)
    C = len(per_channel_std)
    base = torch.randn(s, C, generator=g, dtype=torch.float64)
    return base * torch.tensor(per_channel_std, dtype=torch.float64)


def test_allocate_monotone_in_variance():
    # increasing per-channel std -> rounded bits non-decreasing
    stds = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    R = _channel_matrix(stds)
    bits = allocate_channel_bits(R, budget_bits=3.0)
    bits_list = bits.tolist()
    assert bits_list == sorted(bits_list), f"not monotone: {bits_list}"


def test_allocate_realized_mean_near_budget():
    stds = [0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0]
    R = _channel_matrix(stds)
    for budget in (2.0, 3.0, 3.5):
        bits = allocate_channel_bits(R, budget_bits=budget)
        realized = bits.float().mean().item()
        # tier-rounding can only land at-or-below; never overshoot the budget
        assert realized <= budget + 1e-9, f"overshoot: {realized} > {budget}"
        assert realized >= budget - 1.0, f"too far under: {realized} << {budget}"


def test_allocate_drops_low_variance_channels_when_tight():
    # one giant channel + many tiny ones, tight budget -> tiny ones dropped to 0
    stds = [1000.0] + [0.001] * 20
    R = _channel_matrix(stds)
    bits = allocate_channel_bits(R, budget_bits=1.0)
    assert bits[0].item() > 0
    assert (bits[1:] == 0).any(), "expected some low-variance channels dropped to tier 0"


def test_allocate_isotropic_is_uniform():
    # equal variance -> all channels same tier (degenerate water-fill)
    stds = [1.0] * 12
    R = _channel_matrix(stds)
    bits = allocate_channel_bits(R, budget_bits=3.0)
    assert len(set(bits.tolist())) == 1, f"isotropic not uniform: {bits.tolist()}"


def test_allocate_deterministic():
    stds = [0.1, 1.0, 10.0, 100.0]
    R = _channel_matrix(stds)
    a = allocate_channel_bits(R, budget_bits=3.0)
    b = allocate_channel_bits(R, budget_bits=3.0)
    assert torch.equal(a, b)


def test_allocate_returns_only_tier_values():
    stds = [0.1, 1.0, 10.0, 100.0, 1000.0]
    R = _channel_matrix(stds)
    tiers = (0, 2, 3, 4)
    bits = allocate_channel_bits(R, budget_bits=3.0, tiers=tiers)
    assert set(bits.tolist()).issubset(set(tiers))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /d/Projects/bmx
uv run pytest tests/test_cache_codecs.py -k allocate -q
```
Expected: FAIL — `ImportError: cannot import name 'allocate_channel_bits'`.

- [ ] **Step 3: Implement `allocate_channel_bits`**

Add to `src/bmx/cache/codecs.py` (after the `_rotate`/`_unrotate` helpers, before the codebook section). The bisection drives the water level `kappa` over a log-spaced range bracketed by the channel-variance min/max; at each level, round the continuous water-fill rate to the nearest tier and take the mean. We seek the smallest `kappa` (most bits) whose rounded mean does not exceed `budget_bits`.

```python
def _round_to_tiers(b: torch.Tensor, tiers_t: torch.Tensor) -> torch.Tensor:
    """Round each continuous bit-rate to the nearest value in tiers_t (1-D sorted)."""
    # |b - tier| argmin over the tier axis
    diffs = (b.unsqueeze(-1) - tiers_t).abs()  # (C, n_tiers)
    idx = diffs.argmin(dim=-1)
    return tiers_t[idx]


def allocate_channel_bits(
    R: torch.Tensor,
    budget_bits: float,
    tiers: tuple[int, ...] = (0, 2, 3, 4),
    *,
    axis: int = 0,
    n_search: int = 40,
) -> torch.Tensor:
    """Reverse-water-filling per-channel bit allocation (Cover-Thomas Thm 13.3.3).

    Per-channel variance var_c (over `axis`); continuous rate
    b_c = max(0, 0.5*log2(var_c / kappa)); kappa bisected so the tier-rounded
    mean lands at-or-just-below budget_bits. Deterministic.

    Returns (C,) int64 bit-widths, each a member of `tiers`.
    """
    assert R.dim() == 2, f"R must be 2-D (S, C); got {tuple(R.shape)}"
    var = R.var(dim=axis, unbiased=False).double().clamp_min(1e-30)  # (C,)
    tiers_t = torch.tensor(sorted(tiers), dtype=torch.float64, device=R.device)

    def rounded_mean(kappa: float) -> tuple[torch.Tensor, float]:
        b_cont = (0.5 * torch.log2(var / kappa)).clamp_min(0.0)
        b_round = _round_to_tiers(b_cont, tiers_t)
        return b_round, b_round.mean().item()

    # Bracket kappa in log space: smaller kappa => more bits.
    lo_k = float(var.min().item()) * 1e-6  # high-bit end
    hi_k = float(var.max().item()) * 1e6  # zero-bit end
    lo = math.log(lo_k)
    hi = math.log(hi_k)

    # Bisect for the smallest kappa whose rounded mean <= budget (monotone:
    # mean is non-increasing in kappa). Keep the best feasible candidate.
    best = _round_to_tiers((0.5 * torch.log2(var / hi_k)).clamp_min(0.0), tiers_t)
    for _ in range(n_search):
        mid = 0.5 * (lo + hi)
        b_round, m = rounded_mean(math.exp(mid))
        if m <= budget_bits + 1e-12:
            best = b_round  # feasible (not over budget); try for more bits
            hi = mid  # decrease kappa
        else:
            lo = mid  # over budget; raise kappa
    return best.to(torch.int64)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cache_codecs.py -k allocate -q
```
Expected: PASS (6 tests).

- [ ] **Step 5: Format, lint, full test**

```bash
uv run ruff format . && uv run ruff check . && uv run pytest -q
```
Expected: all clean; `183 passed, 1 xfailed` plus the 6 new allocate tests.

- [ ] **Step 6: Stage and propose commit (STOP for approval)**

```bash
git add src/bmx/cache/codecs.py tests/test_cache_codecs.py
git status --short
```
Proposed message: `feat(cache): reverse-water-filling per-channel bit allocator`
Do NOT commit. Report staged diff and the proposed message; wait for approval.

---

## Task 2: `lowrank_waterfill_channel` codec arm

**Files:**
- Modify: `src/bmx/cache/codecs.py`
- Test: `tests/test_cache_codecs.py`

**Interfaces:**
- Consumes (Task 1):
  `allocate_channel_bits(R, budget_bits, tiers=(0,2,3,4), *, axis=0) -> (C,) int64`.
- Consumes (existing): `truncated_svd(M, rank) -> (Us, V)` from `bmx.decomp.lrs`;
  `rtn_quantize(W, bits, group_size) -> Tensor` from `bmx.quant.rtn`.
- Produces:
  ```python
  def _lowrank_waterfill_channel(
      M: torch.Tensor,            # (S, C) fp32
      budget_bits: float,         # avg residual bits/channel (the "bits" knob)
      group: int,
      rank: int,
      tiers: tuple[int, ...] = (0, 2, 3, 4),
      svd_factors: tuple | None = None,
  ) -> tuple[torch.Tensor, float]  # (M_hat (S,C), honest_bpe)
  ```
  Reachable via `quantize_cache("lowrank_waterfill_channel", M, bits=budget_bits, group=..., rank=..., tiers=...)`.
  Note: `bits` carries the (float) budget for this arm. `"lowrank_waterfill_channel"`
  is in `CACHE_ARMS` and `S_DIVISIBILITY_ARMS`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cache_codecs.py`:

```python
def test_waterfill_arm_in_registries():
    from bmx.cache.codecs import CACHE_ARMS, S_DIVISIBILITY_ARMS

    assert "lowrank_waterfill_channel" in CACHE_ARMS
    assert "lowrank_waterfill_channel" in S_DIVISIBILITY_ARMS


def test_waterfill_reduces_to_uniform_single_tier():
    # With a single uniform tier {3}, every channel gets 3 bits, so the arm must
    # match lowrank_rtn_channel @3b bit-for-bit (same SVD, same per-channel RTN).
    M = _seeded_matrix(s=64, c=64, seed=3).double()
    rank = 4
    factors = truncated_svd(M, rank)
    uni, bpe_uni = quantize_cache(
        "lowrank_rtn_channel", M, bits=3, group=GROUP, rank=rank, svd_factors=factors
    )
    wf, bpe_wf = quantize_cache(
        "lowrank_waterfill_channel",
        M,
        bits=3,
        group=GROUP,
        rank=rank,
        tiers=(3,),
        svd_factors=factors,
    )
    assert torch.allclose(wf, uni, atol=1e-9), "single-tier waterfill != uniform rtn"
    # bpe differs only by the tier-index map; with 1 tier that term is 0 bits.
    assert abs(bpe_wf - bpe_uni) < 1e-9


def test_waterfill_honest_bpe_formula():
    # Hand-check the bpe accounting on a fixed small matrix.
    S_, C_, group_, rank_ = 64, 32, 16, 2
    M = _seeded_matrix(s=S_, c=C_, seed=5).double()
    tiers = (0, 2, 3, 4)
    _, bpe = quantize_cache(
        "lowrank_waterfill_channel",
        M,
        bits=3,
        group=group_,
        rank=rank_,
        tiers=tiers,
    )
    import math as _m

    # The codec recomputes its own allocation on R = M - L; recover the expected
    # residual-payload mean by trusting the codec's reported bpe minus the known
    # metadata terms, then assert each metadata term is the documented constant.
    scale_term = 16.0 / group_
    factor_term = 16.0 * rank_ * (S_ + C_) / (S_ * C_)
    tier_term = _m.ceil(_m.log2(len(tiers))) / S_
    payload = bpe - scale_term - factor_term - tier_term
    assert payload >= 0.0, f"payload negative: {payload}"
    assert payload <= 4.0 + 1e-9, f"payload exceeds max tier: {payload}"


def test_waterfill_s_divisibility_assert():
    M = _seeded_matrix(s=63, c=64, seed=9).double()  # 63 % 16 != 0
    with pytest.raises(AssertionError):
        quantize_cache(
            "lowrank_waterfill_channel", M, bits=3, group=16, rank=2, tiers=(0, 2, 3, 4)
        )


def test_waterfill_dropped_channels_are_zero_in_residual():
    # A near-zero-variance channel in the RESIDUAL should reconstruct from L only.
    # Construct M so one channel is exactly the low-rank part (zero residual).
    M = _seeded_matrix(s=64, c=64, seed=2).double()
    M_hat, _ = quantize_cache(
        "lowrank_waterfill_channel",
        M,
        bits=2,
        group=16,
        rank=4,
        tiers=(0, 2, 3, 4),
    )
    assert M_hat.shape == M.shape
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_cache_codecs.py -k waterfill -q
```
Expected: FAIL — `lowrank_waterfill_channel` not in `CACHE_ARMS` (assertion in `quantize_cache`).

- [ ] **Step 3: Implement the arm and register it**

In `src/bmx/cache/codecs.py`:

(a) Add `"lowrank_waterfill_channel"` to `CACHE_ARMS` and to `S_DIVISIBILITY_ARMS`:

```python
CACHE_ARMS = (
    "rtn_token",
    "rtn_channel",
    "rotate_rtn_token",
    "turboquant_mse",
    "turboquant_prod",
    "lowrank_rtn_channel",
    "lowrank_waterfill_channel",
)

S_DIVISIBILITY_ARMS = frozenset(
    {"rtn_channel", "lowrank_rtn_channel", "lowrank_waterfill_channel"}
)
```

(b) Add the arm implementation after `_lowrank_rtn_channel`:

```python
def _lowrank_waterfill_channel(
    M: torch.Tensor,
    budget_bits: float,
    group: int,
    rank: int,
    tiers: tuple[int, ...] = (0, 2, 3, 4),
    svd_factors: tuple | None = None,
) -> tuple[torch.Tensor, float]:
    """Low-rank + per-channel residual at water-filled mixed bit-widths.

    Same low-rank path as lowrank_rtn_channel; the residual R = M - L is
    quantized per channel at bit-widths chosen by reverse water-filling over
    per-channel variance (Cover-Thomas Thm 13.3.3). Tier 0 channels are dropped
    (reconstructed from L only).
    """
    import math

    S, C = M.shape
    assert rank > 0, f"lowrank_waterfill_channel requires rank > 0, got {rank}"
    assert rank <= min(S, C), f"rank {rank} > min(S,C)={min(S, C)}"
    assert S % group == 0, f"S={S} not divisible by group={group}"

    if svd_factors is not None:
        Us, V = svd_factors
    else:
        Us, V = truncated_svd(M, rank)

    # Keep fp16 roundtrip semantics identical to _lowrank_rtn_channel for honest
    # factor cost; tests run fp64, so guard the half() on fp32 only.
    if M.dtype == torch.float32:
        Us_stored = Us.half().float()
        V_stored = V.half().float()
    else:
        Us_stored = Us
        V_stored = V
    L = Us_stored @ V_stored.mT  # (S, C)

    R = M - L  # (S, C)

    # Allocate per-channel bits on the residual.
    bits_per_ch = allocate_channel_bits(R, budget_bits, tiers=tiers, axis=0)  # (C,)

    # Quantize each tier-group of channels at its bit-width; tier 0 -> zeros.
    R_hat = torch.zeros_like(R)
    for b in sorted(set(int(x) for x in bits_per_ch.tolist())):
        if b == 0:
            continue  # dropped channels stay zero
        cols = (bits_per_ch == b).nonzero(as_tuple=True)[0]
        if cols.numel() == 0:
            continue
        sub = R[:, cols]  # (S, n_b); quantize per channel along token dim
        sub_hat = rtn_quantize(sub.mT, b, group).mT  # (n_b, S) groups -> back
        R_hat[:, cols] = sub_hat

    M_hat = L + R_hat

    # Honest bpe (per entry; all metadata counted):
    mean_payload = float(bits_per_ch.float().mean().item())
    scale_term = 16.0 / group
    factor_term = 16.0 * rank * (S + C) / (S * C)
    tier_term = math.ceil(math.log2(len(tiers))) / S
    bpe = mean_payload + scale_term + factor_term + tier_term
    return M_hat, bpe
```

(c) Add the dispatch branch in `quantize_cache`, replacing the final `else`:

```python
    elif arm == "lowrank_rtn_channel":
        return _lowrank_rtn_channel(M, bits, group, rank, svd_factors=svd_factors)
    else:  # lowrank_waterfill_channel — guarded by the CACHE_ARMS assert above
        return _lowrank_waterfill_channel(
            M, float(bits), group, rank, tiers=tiers, svd_factors=svd_factors
        )
```

(d) Add a `tiers` parameter to `quantize_cache`'s signature (default
`(0, 2, 3, 4)`) and docstring, threaded only to the waterfill arm:

```python
def quantize_cache(
    arm: str,
    M: torch.Tensor,
    *,
    bits: int,
    seed: int = 0,
    group: int = 64,
    rank: int = 0,
    svd_factors: tuple | None = None,
    tiers: tuple[int, ...] = (0, 2, 3, 4),
) -> tuple[torch.Tensor, float]:
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cache_codecs.py -k waterfill -q
```
Expected: PASS (5 tests).

- [ ] **Step 5: Format, lint, full test**

```bash
uv run ruff format . && uv run ruff check . && uv run pytest -q
```
Expected: all clean.

- [ ] **Step 6: Stage and propose commit (STOP for approval)**

```bash
git add src/bmx/cache/codecs.py tests/test_cache_codecs.py
git status --short
```
Proposed message: `feat(cache): lowrank_waterfill_channel codec arm (mixed-precision residual)`
Do NOT commit. Report and wait for approval.

---

## Task 3: `experiments/k2_waterfill.py` + smoke test

**Files:**
- Create: `experiments/k2_waterfill.py`
- Create: `tests/test_k2_waterfill.py`

**Interfaces:**
- Consumes (Tasks 1-2): the `"lowrank_waterfill_channel"` arm via `quantize_cache`.
- Consumes (existing): `create_run`, `write_metrics` (`bmx.artifacts`);
  `load_cache`, `to_matrix`, `from_matrix` (`bmx.cache.collect`);
  `logit_distortion`, `rel_fro` (`bmx.cache.metrics`); `apply_rope`, `rope_cos_sin`
  (`bmx.cache.rope`); `truncated_svd` (`bmx.decomp.lrs`).
- Produces: a `main(cfg: Config)` entry and a `_resid_stable_rank(R)` helper the
  smoke test imports.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_k2_waterfill.py`:

```python
"""Smoke test for experiments/k2_waterfill.py on the offline GPT-2 cache fixture."""

import importlib.util
import sys
from pathlib import Path

import torch

# Load the experiment module by path (experiments/ is not a package).
_EXP = Path(__file__).resolve().parents[1] / "experiments" / "k2_waterfill.py"
_spec = importlib.util.spec_from_file_location("k2_waterfill", _EXP)
k2_waterfill = importlib.util.module_from_spec(_spec)
sys.modules["k2_waterfill"] = k2_waterfill
_spec.loader.exec_module(k2_waterfill)


def test_stable_rank_helper():
    # isotropic -> stable rank ~ C; rank-1 -> stable rank ~ 1.
    g = torch.Generator().manual_seed(1)
    iso = torch.randn(128, 16, generator=g, dtype=torch.float64)
    sr_iso = k2_waterfill._resid_stable_rank(iso)
    assert sr_iso > 8.0

    v = torch.randn(16, 1, generator=g, dtype=torch.float64)
    rank1 = torch.randn(128, 1, generator=g, dtype=torch.float64) @ v.mT
    sr_r1 = k2_waterfill._resid_stable_rank(rank1)
    assert sr_r1 < 2.0


def test_experiment_smoke(tmp_path):
    # Build a tiny synthetic cache file with the layer-key convention and run main.
    from safetensors.torch import save_file

    h_kv, S, d = 2, 64, 8  # C = 16
    g = torch.Generator().manual_seed(7)
    tensors = {}
    for i in range(2):
        tensors[f"layer{i}.k_pre"] = torch.randn(h_kv, S, d, generator=g).half()
        tensors[f"layer{i}.k"] = torch.randn(h_kv, S, d, generator=g).half()
        tensors[f"layer{i}.v"] = torch.randn(h_kv, S, d, generator=g).half()
        tensors[f"layer{i}.q"] = torch.randn(h_kv, S, d, generator=g).half()
    cache_path = tmp_path / "synthetic.safetensors"
    save_file(tensors, str(cache_path))

    cfg = k2_waterfill.Config(
        cache_path=str(cache_path),
        model_label="synthetic",
        model_name="",  # no RoPE -> logit (stored basis), not logit_rope
        budget_bits=3.0,
        group=16,
        rank=4,
        out_root=str(tmp_path / "results"),
    )
    df = k2_waterfill.main(cfg)
    arms = set(df["arm"].unique())
    assert {"lowrank_rtn_channel", "lowrank_waterfill_channel", "outlier_two_tier"} <= arms
    assert "resid_stable_rank" in df.columns
    # matched bpe: waterfill within tolerance of uniform baseline, per layer
    for layer in df["layer"].unique():
        sub = df[df["layer"] == layer]
        bpe_uni = sub[sub.arm == "lowrank_rtn_channel"]["bpe"].mean()
        bpe_wf = sub[sub.arm == "lowrank_waterfill_channel"]["bpe"].mean()
        assert abs(bpe_uni - bpe_wf) < 0.05, f"bpe mismatch L{layer}: {bpe_uni} vs {bpe_wf}"
```

- [ ] **Step 2: Run the smoke test to verify it fails**

```bash
cd /d/Projects/bmx
uv run pytest tests/test_k2_waterfill.py -q
```
Expected: FAIL — `experiments/k2_waterfill.py` does not exist (spec load error).

- [ ] **Step 3: Implement the experiment**

Create `experiments/k2_waterfill.py`. The `main` accepts a `Config` and RETURNS
the metrics DataFrame (so the smoke test can assert on it) in addition to writing
parquet. The baseline budget must be matched: pick the waterfill `budget_bits` so
its realized bpe matches the uniform-3b realized bpe — both arms share the same
low-rank factor and scale terms, so matching the residual payload mean to 3.0
matches total bpe to within the tier-index term; assert |Δbpe| < 0.05 per layer.

```python
"""K2 water-filling kill-or-confirm: per-channel mixed-precision vs uniform on key residuals.

Three arms on k_pre, matched bpe, scored on logit distortion vs real queries:
  - lowrank_rtn_channel @3b   (uniform baseline)
  - lowrank_waterfill_channel (reverse water-filling over per-channel residual variance)
  - outlier_two_tier          (top-k highest-variance residual channels -> fp16, rest low)

Diagnostic column resid_stable_rank distinguishes "low-rank already did the
water-filling" (flat residual spectrum) from "deterministic rounding killed it".

Usage
-----
    uv run python experiments/k2_waterfill.py \
        --cache-path results/cache/llama-3.1-8b_2048.safetensors \
        --model-label llama-3.1-8b \
        --model-name meta-llama/Llama-3.1-8B \
        --budget-bits 3.0 --rank 16
"""

from __future__ import annotations

import dataclasses
import math
import re

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.codecs import quantize_cache
from bmx.cache.collect import from_matrix, load_cache, to_matrix
from bmx.cache.metrics import logit_distortion, rel_fro
from bmx.cache.rope import apply_rope
from bmx.decomp.lrs import truncated_svd

_LAYER_RE = re.compile(r"^layer(\d+)\.(k|v|q|k_pre)$")


@dataclasses.dataclass
class Config:
    cache_path: str
    model_label: str = ""
    model_name: str = ""  # HF repo id for RoPE (empty => score in stored basis)
    budget_bits: float = 3.0
    group: int = 64
    rank: int = 16
    tiers: tuple[int, ...] = (0, 2, 3, 4)
    seed: int = 0
    out_root: str = ""  # override results root (tests pass tmp); empty => default


def _resid_stable_rank(R: torch.Tensor) -> float:
    """(sum of eigenvalues of R^T R) / (max eigenvalue) = (||R||_F^2) / (sigma_max^2)."""
    sv = torch.linalg.svdvals(R.double())
    s2 = sv**2
    return float((s2.sum() / s2[0].clamp_min(1e-30)).item())


def _outlier_two_tier(
    M: torch.Tensor, budget_bits: float, group: int, rank: int, svd_factors
) -> tuple[torch.Tensor, float]:
    """Top-k highest-variance residual channels -> fp16, rest -> low bits, matched bpe."""
    S, C = M.shape
    Us, V = svd_factors
    Us_s = Us.half().float() if M.dtype == torch.float32 else Us
    V_s = V.half().float() if M.dtype == torch.float32 else V
    L = Us_s @ V_s.mT
    R = M - L
    var = R.var(dim=0, unbiased=False)
    # Choose k fp16 channels + b_lo for the rest so the residual payload mean == budget.
    # payload = (k*16 + (C-k)*b_lo)/C = budget. Fix b_lo=2, solve k.
    b_lo = 2
    k = max(0, min(C, round((budget_bits - b_lo) * C / (16 - b_lo))))
    order = torch.argsort(var, descending=True)
    hi_cols = order[:k]
    lo_cols = order[k:]
    R_hat = torch.zeros_like(R)
    R_hat[:, hi_cols] = R[:, hi_cols]  # fp16 == passthrough at experiment precision
    if lo_cols.numel() > 0:
        from bmx.quant.rtn import rtn_quantize

        R_hat[:, lo_cols] = rtn_quantize(R[:, lo_cols].mT, b_lo, group).mT
    M_hat = L + R_hat
    payload = (k * 16.0 + (C - k) * b_lo) / C
    idx_term = math.ceil(math.log2(C)) / S  # store which k channels are fp16
    bpe = payload + 16.0 / group + 16.0 * rank * (S + C) / (S * C) + idx_term
    return M_hat, bpe


def main(cfg: Config) -> pd.DataFrame:
    run = create_run("k2_waterfill", cfg, root=cfg.out_root or None)

    cache = load_cache(cfg.cache_path)
    layer_keys: dict[int, dict[str, torch.Tensor]] = {}
    for key, tensor in cache.items():
        m = _LAYER_RE.match(key)
        if m is None:
            continue
        layer_keys.setdefault(int(m.group(1)), {})[m.group(2)] = tensor

    # RoPE setup (optional).
    rope_ready = False
    hf_config = None
    if cfg.model_name:
        from transformers import AutoConfig

        from bmx.cache.rope import rope_cos_sin

        hf_config = AutoConfig.from_pretrained(cfg.model_name)
        rope_ready = True

    cos_sin: dict[int, tuple] = {}

    def get_cos_sin(S: int):
        if S not in cos_sin:
            cos_sin[S] = rope_cos_sin(hf_config, S)
        return cos_sin[S]

    rows: list[dict] = []
    for layer_i in sorted(layer_keys):
        km = layer_keys[layer_i]
        if "k_pre" not in km or "q" not in km:
            continue
        k_pre = km["k_pre"]
        h_kv, S, d = k_pre.shape
        M = to_matrix(k_pre).float()  # (S, C)
        C = h_kv * d
        Q = km["q"].float()

        cos = sin = None
        K_post_true = None
        if rope_ready:
            cos, sin = get_cos_sin(S)
            K_post_true = apply_rope(k_pre.float(), cos, sin)

        factors = truncated_svd(M, cfg.rank)
        sr = _resid_stable_rank(M - (factors[0] @ factors[1].mT))

        def score(M_hat: torch.Tensor) -> tuple[float, float]:
            K_hat = from_matrix(M_hat, h_kv)
            rf = rel_fro(M_hat, M)
            if rope_ready:
                K_hat_rope = apply_rope(K_hat.float(), cos, sin)
                lg = logit_distortion(K_post_true, K_hat_rope, Q)
            else:
                lg = logit_distortion(k_pre.float(), K_hat, Q)
            return rf, lg

        # Uniform baseline @3b (budget rounded to int for the uniform arm).
        uni, bpe_uni = quantize_cache(
            "lowrank_rtn_channel",
            M,
            bits=round(cfg.budget_bits),
            group=cfg.group,
            rank=cfg.rank,
            svd_factors=factors,
        )
        wf, bpe_wf = quantize_cache(
            "lowrank_waterfill_channel",
            M,
            bits=cfg.budget_bits,
            group=cfg.group,
            rank=cfg.rank,
            tiers=cfg.tiers,
            svd_factors=factors,
        )
        ot, bpe_ot = _outlier_two_tier(M, cfg.budget_bits, cfg.group, cfg.rank, factors)

        assert abs(bpe_uni - bpe_wf) < 0.05, (
            f"L{layer_i}: waterfill bpe {bpe_wf:.3f} not matched to uniform {bpe_uni:.3f}"
        )

        for arm, (M_hat, bpe) in {
            "lowrank_rtn_channel": (uni, bpe_uni),
            "lowrank_waterfill_channel": (wf, bpe_wf),
            "outlier_two_tier": (ot, bpe_ot),
        }.items():
            rf, lg = score(M_hat)
            rows.append(
                dict(
                    model=cfg.model_label or "unknown",
                    layer=layer_i,
                    kind="k_pre",
                    arm=arm,
                    rank=cfg.rank,
                    bpe=bpe,
                    rel_fro=rf,
                    logit_rope=lg,
                    resid_stable_rank=sr,
                )
            )
            print(
                f"  L{layer_i:2d} {arm:26s} bpe={bpe:.3f} logit={lg:.4f} "
                f"rel_fro={rf:.4f} sr={sr:.1f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    write_metrics(run, df)

    print("\n" + "=" * 60)
    print("SUMMARY — mean logit_rope per arm (lower is better)")
    for arm in sorted(df.arm.unique()):
        sub = df[df.arm == arm]
        print(
            f"  {arm:26s} logit={sub.logit_rope.mean():.4f} "
            f"bpe={sub.bpe.mean():.3f}  resid_sr={sub.resid_stable_rank.mean():.1f}"
        )
    print(f"\n-> {run}")
    return df


if __name__ == "__main__":
    main(tyro.cli(Config))
```

- [ ] **Step 4: Confirm `create_run` accepts a `root` override**

Run:
```bash
uv run python -c "import inspect; from bmx.artifacts import create_run; print(inspect.signature(create_run))"
```
Expected: a signature including a `root` (or similar) parameter.
- If `create_run` has NO root override: in `main`, replace
  `create_run("k2_waterfill", cfg, root=cfg.out_root or None)` with
  `create_run("k2_waterfill", cfg)` and delete the `out_root` field from `Config`
  and the `out_root=` line in the smoke test (the test then writes under the
  default results root; still passes since it only asserts on the returned df).

- [ ] **Step 5: Run the smoke test to verify it passes**

```bash
uv run pytest tests/test_k2_waterfill.py -q
```
Expected: PASS (2 tests).

- [ ] **Step 6: Format, lint, full test**

```bash
uv run ruff format . && uv run ruff check . && uv run pytest -q
```
Expected: all clean.

- [ ] **Step 7: Stage and propose commit (STOP for approval)**

```bash
git add experiments/k2_waterfill.py tests/test_k2_waterfill.py
git status --short
```
Proposed message: `feat(exp): k2_waterfill kill-or-confirm experiment (channel bit allocation)`
Do NOT commit. Report and wait for approval.

---

## Task 4: Run the experiment and record the verdict

**Files:**
- Create: `docs/2026-06-21-k2-waterfill-results.md`

This task has no test cycle — it runs the built experiment on real caches and
writes the findings doc. Depends on Tasks 1-3 committed.

- [ ] **Step 1: Confirm a Llama cache exists (or regenerate)**

```bash
ls -la results/cache/llama-3.1-8b_2048.safetensors 2>/dev/null || \
ls -la results/cache/gpt2_1024.safetensors
```
If only the GPT-2 cache exists locally (likely, given AMD/no-CUDA), run on GPT-2
for the mechanism look and note that the Llama numbers need the VM. The GPT-2
cache regenerates via:
```bash
uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024
```

- [ ] **Step 2: Run the experiment**

GPT-2 (local, mechanism look):
```bash
uv run python experiments/k2_waterfill.py \
    --cache-path results/cache/gpt2_1024.safetensors \
    --model-label gpt2 --budget-bits 3.0 --rank 16
```
Llama (if the cache is present / on the VM):
```bash
uv run python experiments/k2_waterfill.py \
    --cache-path results/cache/llama-3.1-8b_2048.safetensors \
    --model-label llama-3.1-8b --model-name meta-llama/Llama-3.1-8B \
    --budget-bits 3.0 --rank 16
```

- [ ] **Step 3: Write the verdict doc**

Create `docs/2026-06-21-k2-waterfill-results.md` recording, faithfully (honest
negative is a valid result): the mean `logit_rope` per arm, the matched bpe, the
`resid_stable_rank` per layer, and which of the spec's three outcomes occurred:
1. water-fill wins, 2. tie + flat residual spectrum ("low-rank IS the water-fill"),
3. tie/loss + still-anisotropic residual ("deterministic rounding boundary").
Quote the numbers; do not polish them. Reference the run dir(s).

- [ ] **Step 4: Stage and propose commit (STOP for approval)**

```bash
git add docs/2026-06-21-k2-waterfill-results.md
git status --short
```
Proposed message: `docs: k2 water-filling channel-allocation results (kill-or-confirm verdict)`
Do NOT commit. Report and wait for approval.

---

## Notes for the executor

- **fp16 roundtrip guard**: codecs roundtrip factors through fp16 for honest
  accounting, but tests run fp64. The arm guards `half()` on `dtype == float32`
  only, so fp64 tests compare cleanly. Mirror this in any new path.
- **`logit_distortion` is GQA-aware** and takes `(K_orig, K_hat, Q)` — same call
  shape as `k2_cache_arms.py`. Do not hand-roll the attention math.
- **Matched bpe is asserted, not assumed** (the `abs(bpe_uni - bpe_wf) < 0.05`
  guard in `main`). If it fires on real data, the residual-payload mean drifted
  from 3.0 — investigate the allocation, don't loosen the tolerance.
- The single-tier equivalence test (Task 2) is the load-bearing correctness check:
  it proves the new arm is the old arm plus allocation, nothing else changed.
