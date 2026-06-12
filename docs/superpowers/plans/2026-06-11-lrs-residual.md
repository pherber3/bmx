# Avenue 1 — Low-Rank+Sparse Quantization Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate the two-step hard-threshold + truncated-SVD L+S estimator (`W = L + S + Q(R)`) as a quantization pre-conditioner, per the grounded design in `docs/next-avenues-structured-residual.md` Avenue 1.

**Architecture:** A new `lrs` decomposition behind the existing `Decomposition` registry (operates on single 2-D matrices, unlike the 3-D stack methods); reusable compression "arms" (plain RTN / rotate-RTN / LRS-RTN / LRS-rotate-RTN) with explicit total-bit accounting in `bmx.quant.arms`; a two-stage experiment (Stage A structural diagnostic → Stage B matched-bits compression) and a gated layer-swap perplexity eval.

**Tech Stack:** torch (SVD/topk), existing `bmx.quant` (rtn, hadamard, stats), `bmx.artifacts` (parquet runs), tyro CLIs, transformers GPT-2, `datasets` (new dep, layer-swap only).

---

## NON-NEGOTIABLE PROJECT RULES (override anything below that conflicts)

- **NEVER `git commit` without the user's explicit approval.** Every "Commit" step below means: run `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` (expect all green), `git add` the listed files, **propose the commit message, and STOP until the user approves.** No AI-attribution trailers.
- Use the Bash tool (git bash). Fresh shells: `cd /d/Projects/bmx` first.
- Dependencies only via `uv add`. fp64 in tests, fp32 in experiments.
- Baseline suite status before this plan: **53 passed, 1 xfailed** (`test_cold_start_recovery` xfail is intentional — leave it).

## Design decisions locked by the grounding (do not "improve" these)

1. **HARD thresholding**: `T_ν(v) = v·𝟙[|v| > ν]` keeps the value verbatim (Wainwright §11.4.2 Eq. 11.58). Soft-thresholding (shrinking by ν) is WRONG here.
2. **Fit L+S in the ORIGINAL basis.** Rotation provably spreads the concentrated mass S needs (Beta-coordinate theorem). Rotation appears only as a *competing arm* and as a *post-L+S residual treatment*.
3. **Comparisons at matched TOTAL bits** — L factors, S values, AND S index bits counted. Rotations are seed-generated (0 stored bits).
4. **Estimator first, convex program never** (in this plan): the two-step estimator is the workhorse; Eq. 10.53 is a later referee, out of scope.

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `src/bmx/decomp/lrs.py` | Create | hard_threshold, topk_sparse, truncated_svd, two_step_lrs, spikiness_ratio, `LRSFit`, `@register("lrs")` |
| `src/bmx/decomp/__init__.py` | Modify | import `lrs` so registration happens |
| `tests/test_lrs.py` | Create | operator semantics + planted L+S recovery |
| `src/bmx/quant/arms.py` | Create | `reconstruct_arm` (the 4 compression pipelines), `total_bits` accounting |
| `tests/test_arms.py` | Create | bit accounting + arm correctness/determinism |
| `experiments/lrs_residual.py` | Create | Stage A diagnostic + Stage B matched-bits sweep (tyro CLI, parquet via artifacts) |
| `experiments/plots/plot_lrs.py` | Create | rate–distortion figure from Stage B parquet |
| `src/bmx/eval/layer_swap.py` | Modify (replace stub) | `set_weight`, `perplexity`, `swap_and_perplexity` |
| `tests/test_layer_swap.py` | Create | offline tests on a tiny random GPT-2 (no download) |
| `experiments/lrs_layer_swap.py` | Create | gated perplexity comparison of the arms |

---

### Task 1: The two-step L+S estimator (`lrs.py` core)

**Files:**
- Create: `src/bmx/decomp/lrs.py`
- Create: `tests/test_lrs.py`
- Modify: `src/bmx/decomp/__init__.py`

- [ ] **Step 1.1: Write the failing tests**

Create `tests/test_lrs.py`:

```python
import pytest
import torch

from bmx.decomp.lrs import (
    fit_lrs,
    hard_threshold,
    spikiness_ratio,
    topk_sparse,
    two_step_lrs,
)
from bmx.quant.hadamard import orthogonalize


def _planted(m=64, p=48, r=3, k=20, seed=0, spike=1.0, scale=0.01):
    """L: incoherent rank-r with entries ~scale; S: k spikes of magnitude `spike`.

    Separation spike/scale ~ 100x makes support identification immediate and
    alternation convergence geometric.
    """
    g = torch.Generator().manual_seed(seed)
    U = orthogonalize(torch.randn(m, r, generator=g, dtype=torch.float64))
    V = orthogonalize(torch.randn(p, r, generator=g, dtype=torch.float64))
    s = torch.linspace(1.0, 0.5, r, dtype=torch.float64) * scale * (m * p) ** 0.5
    L = (U * s) @ V.mT
    idx = torch.randperm(m * p, generator=g)[:k]
    S = torch.zeros(m * p, dtype=torch.float64)
    signs = (torch.rand(k, generator=g, dtype=torch.float64) > 0.5).double() * 2 - 1
    S[idx] = spike * signs
    return L, S.view(m, p)


def test_hard_threshold_is_hard_not_soft():
    # Eq. 11.58: T_nu keeps the VALUE; soft-thresholding would shrink by nu.
    x = torch.tensor([0.5, -2.0, 1.1, 1.0])
    out = hard_threshold(x, 1.0)
    assert torch.equal(out, torch.tensor([0.0, -2.0, 1.1, 0.0]))


def test_topk_sparse_exact_count_and_values():
    g = torch.Generator().manual_seed(3)
    W = torch.randn(10, 7, generator=g, dtype=torch.float64)
    S = topk_sparse(W, 5)
    nz = S != 0
    assert nz.sum().item() == 5
    assert torch.equal(S[nz], W[nz])  # kept verbatim
    # agrees with hard_threshold at any nu between the 5th and 6th magnitude
    mags = W.abs().flatten().sort(descending=True).values
    nu = (mags[4] + mags[5]).item() / 2
    assert torch.equal(S, hard_threshold(W, nu))
    assert torch.equal(topk_sparse(W, 0), torch.zeros_like(W))


def test_planted_recovery():
    L, S = _planted()
    W = L + S
    Us, V, S_hat = two_step_lrs(W, r=3, k=20, n_alternations=10)
    # exact support recovery
    assert torch.equal(S_hat != 0, S != 0)
    rec = Us @ V.mT + S_hat
    assert (rec - W).norm() / W.norm() < 1e-5
    assert (Us @ V.mT - L).norm() / L.norm() < 1e-4


def test_fit_lrs_edges_and_param_count():
    L, S = _planted(m=16, p=12, r=2, k=4)
    W = L + S
    full = fit_lrs(W, rank=(12, 0))
    assert full.relative_error(W) < 1e-10  # full rank, no sparsity: exact
    fit = fit_lrs(W, rank=(2, 4))
    assert fit.param_count() == 2 * (16 + 12) + 4
    zero = fit_lrs(W, rank=(0, 0))
    assert abs(zero.relative_error(W) - 1.0) < 1e-12


def test_spikiness_ratio():
    flat = torch.ones(8, 8)
    assert abs(spikiness_ratio(flat) - 1.0) < 1e-6
    spiky = torch.zeros(8, 8)
    spiky[0, 0] = 1.0
    assert abs(spikiness_ratio(spiky) - 8.0) < 1e-6  # max*sqrt(64)/fro = 8


def test_registered_and_rejects_3d():
    from bmx.decomp.base import available_methods

    assert "lrs" in available_methods()
    with pytest.raises(AssertionError):
        fit_lrs(torch.zeros(4, 4, 4), rank=(2, 2))
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_lrs.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'bmx.decomp.lrs'`

- [ ] **Step 1.3: Write the implementation**

Create `src/bmx/decomp/lrs.py`:

```python
"""Low-rank-plus-sparse two-step estimator (Avenue 1).

Hard-threshold-then-truncated-SVD per Wainwright HDS §11.4.2 (Eq. 11.58 /
Prop. 11.19; direct method due to Agarwal et al. 2012): S = T_nu(W) keeps
large entries VERBATIM (hard threshold — soft-thresholding is a different,
wrong operator here), L = truncated SVD of W - S, optionally alternated.

Operates on a single 2-D weight matrix — unlike the 3-D stack methods — and
is parameterized by budget (r, k) rather than threshold nu, so matched-bit
sweeps are direct. Fit in the ORIGINAL basis: rotation provably spreads the
concentrated mass S needs (see docs/next-avenues-structured-residual.md).
"""

import torch

from bmx.decomp.base import FitResult, register


def hard_threshold(W: torch.Tensor, nu: float) -> torch.Tensor:
    """T_nu(v) = v * 1[|v| > nu] — keeps the value, no shrinkage (Eq. 11.58)."""
    return W * (W.abs() > nu)


def topk_sparse(W: torch.Tensor, k: int) -> torch.Tensor:
    """Hard threshold parameterized by budget: exactly the k largest |entries|."""
    if k == 0:
        return torch.zeros_like(W)
    flat = W.flatten()
    idx = flat.abs().topk(k).indices
    S = torch.zeros_like(flat)
    S[idx] = flat[idx]
    return S.view_as(W)


def truncated_svd(W: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Best rank-r approximation as (U*s, V): W_r = Us @ V.mT."""
    m, p = W.shape
    if r == 0:
        return W.new_zeros(m, 0), W.new_zeros(p, 0)
    U, s, Vh = torch.linalg.svd(W, full_matrices=False)
    return U[:, :r] * s[:r], Vh[:r, :].mT.contiguous()


def two_step_lrs(
    W: torch.Tensor, r: int, k: int, n_alternations: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (Us, V, S) with L = Us @ V.mT; W ≈ L + S."""
    assert W.ndim == 2, f"two_step_lrs operates on a matrix, got ndim={W.ndim}"
    S = topk_sparse(W, k)
    Us, V = truncated_svd(W - S, r)
    for _ in range(n_alternations):
        S = topk_sparse(W - Us @ V.mT, k)
        Us, V = truncated_svd(W - S, r)
    return Us, V, S


def spikiness_ratio(M: torch.Tensor) -> float:
    """alpha_hat = ||M||_max * sqrt(d1*d2) / ||M||_F (Wainwright's spikiness,
    normalized so a flat matrix scores ~1 and a single spike scores sqrt(d1*d2))."""
    return (M.abs().max() * M.numel() ** 0.5 / M.norm()).item()


class LRSFit(FitResult):
    def __init__(self, Us: torch.Tensor, V: torch.Tensor, S: torch.Tensor):
        k = int((S != 0).sum())
        super().__init__(method="lrs", rank=(Us.shape[1], k), loss_history=[])
        self.Us, self.V, self.S = Us, V, S

    def reconstruct(self) -> torch.Tensor:
        return self.Us @ self.V.mT + self.S

    def param_count(self) -> int:
        # stored NUMBERS only (r*(m+p) factor entries + k sparse values);
        # sparse index bits are storage, not parameters — counted by
        # bmx.quant.arms.total_bits where bit budgets are compared.
        r, k = self.rank
        return r * (self.Us.shape[0] + self.V.shape[0]) + k


@register("lrs")
def fit_lrs(W: torch.Tensor, rank, *, n_alternations: int = 2) -> LRSFit:
    r, k = (int(x) for x in rank)
    assert W.ndim == 2, f"lrs operates on a single matrix, got ndim={W.ndim}"
    m, p = W.shape
    assert 0 <= r <= min(m, p), f"rank {r} > min(m,p)={min(m, p)}"
    assert 0 <= k <= W.numel(), f"sparse budget {k} > numel={W.numel()}"
    Us, V, S = two_step_lrs(W, r, k, n_alternations=n_alternations)
    fit = LRSFit(Us, V, S)
    fit.loss_history = [fit.relative_error(W)]
    return fit
```

Modify `src/bmx/decomp/__init__.py` to:

```python
from bmx.decomp import baselines as _baselines  # noqa: F401
from bmx.decomp import bmd_rals as _bmd_rals  # noqa: F401
from bmx.decomp import lrs as _lrs  # noqa: F401
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `uv run pytest tests/test_lrs.py -v`
Expected: 6 passed. If `test_planted_recovery` fails on the 1e-5 tolerance, report the actual number to the user rather than silently loosening — the separation in `_planted` is 100×, so failure means a bug, not a tolerance problem.

- [ ] **Step 1.5: Full gate + propose commit (USER APPROVAL REQUIRED)**

Run: `uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: 59 passed, 1 xfailed.
Stage `src/bmx/decomp/lrs.py src/bmx/decomp/__init__.py tests/test_lrs.py`, propose:
`feat: two-step hard-threshold L+S estimator, registered as "lrs"`
**Stop and wait for user approval before committing.**

---

### Task 2: Compression arms + total-bit accounting (`quant/arms.py`)

**Files:**
- Create: `src/bmx/quant/arms.py`
- Create: `tests/test_arms.py`

- [ ] **Step 2.1: Write the failing tests**

Create `tests/test_arms.py`:

```python
import pytest
import torch

from bmx.quant.arms import ARMS, reconstruct_arm, total_bits


def _W(m=32, p=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, p, generator=g, dtype=torch.float64)


def test_total_bits_formula():
    # bulk ints + fp16 group scales + fp16 L factors + (fp16 + index) sparse
    m, p, bits, gs, r, k = 32, 64, 4, 32, 2, 5
    idx_bits = (m * p - 1).bit_length()  # ceil(log2(2048)) = 11
    expected = (
        m * p * bits
        + (m * p // gs) * 16
        + r * (m + p) * 16
        + k * (16 + idx_bits)
    )
    assert total_bits(m, p, bits=bits, group_size=gs, r=r, k=k) == expected
    # rotation arms store nothing extra: r=k=0 accounting
    assert total_bits(m, p, bits=4, group_size=32, r=0, k=0) == m * p * 4 + 64 * 16


def test_plain_arm_matches_rtn():
    from bmx.quant.rtn import rtn_quantize

    W = _W()
    rec, r, k = reconstruct_arm("rtn", W, bits=4, group_size=32, r=8, k=10, seed=0)
    assert torch.equal(rec, rtn_quantize(W, 4, 32))
    assert (r, k) == (0, 0)  # plain arm stores no L/S regardless of request


def test_lrs_arm_exact_at_full_rank():
    W = _W()
    rec, r, k = reconstruct_arm("lrs_rtn", W, bits=2, group_size=32, r=32, k=0, seed=0)
    # full-rank L makes R = 0; quantization of zeros is exact
    assert (rec - W).norm() / W.norm() < 1e-10
    assert (r, k) == (32, 0)


def test_rotate_arms_deterministic_in_seed():
    W = _W()
    a1, _, _ = reconstruct_arm("rotate_rtn", W, bits=4, group_size=32, r=0, k=0, seed=7)
    a2, _, _ = reconstruct_arm("rotate_rtn", W, bits=4, group_size=32, r=0, k=0, seed=7)
    a3, _, _ = reconstruct_arm("rotate_rtn", W, bits=4, group_size=32, r=0, k=0, seed=8)
    assert torch.equal(a1, a2)
    assert not torch.equal(a1, a3)


def test_all_arms_run_and_unknown_raises():
    W = _W()
    for arm in ARMS:
        rec, _, _ = reconstruct_arm(arm, W, bits=4, group_size=32, r=4, k=8, seed=0)
        assert rec.shape == W.shape
        assert (rec - W).norm() / W.norm() < 0.5
    with pytest.raises(AssertionError):
        reconstruct_arm("nope", W, bits=4, group_size=32, r=0, k=0, seed=0)
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `uv run pytest tests/test_arms.py -v`
Expected: `ModuleNotFoundError: No module named 'bmx.quant.arms'`

- [ ] **Step 2.3: Write the implementation**

Create `src/bmx/quant/arms.py`:

```python
"""Compression arms for the Avenue 1 comparison, plus honest bit accounting.

Each arm maps W -> reconstructed W-hat. The L+S arms fit on CLEAN W in the
ORIGINAL basis (load-bearing — rotation spreads the mass S needs; see
docs/next-avenues-structured-residual.md), then quantize R = W - L - S.
Rotations are generated from a seed, so they cost 0 stored bits.
"""

import torch

from bmx.decomp.lrs import two_step_lrs
from bmx.quant.hadamard import random_orthogonal
from bmx.quant.rtn import rtn_quantize

FP_BITS = 16  # storage precision for L factors, S values, and group scales

ARMS = ("rtn", "rotate_rtn", "lrs_rtn", "lrs_rotate_rtn")


def total_bits(m: int, p: int, *, bits: int, group_size: int, r: int, k: int) -> int:
    """Total stored bits: bulk ints + group scales + L factors + S (values+indices)."""
    bulk = m * p * bits + (m * p // group_size) * FP_BITS
    idx_bits = (m * p - 1).bit_length()
    return bulk + r * (m + p) * FP_BITS + k * (FP_BITS + idx_bits)


def _rotate_rtn(W: torch.Tensor, bits: int, group_size: int, seed: int) -> torch.Tensor:
    Q = random_orthogonal(W.shape[-1], seed=seed, dtype=W.dtype)
    return rtn_quantize(W @ Q.mT, bits, group_size) @ Q


def reconstruct_arm(
    arm: str,
    W: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    r: int,
    k: int,
    seed: int,
    n_alternations: int = 2,
) -> tuple[torch.Tensor, int, int]:
    """Returns (W_hat, r_stored, k_stored); pass r/k stored to total_bits."""
    assert arm in ARMS, f"unknown arm {arm!r}; available: {ARMS}"
    if arm == "rtn":
        return rtn_quantize(W, bits, group_size), 0, 0
    if arm == "rotate_rtn":
        return _rotate_rtn(W, bits, group_size, seed), 0, 0
    Us, V, S = two_step_lrs(W, r, k, n_alternations=n_alternations)
    L = Us @ V.mT
    R = W - L - S
    if arm == "lrs_rtn":
        Rq = rtn_quantize(R, bits, group_size)
    else:  # lrs_rotate_rtn: rotate only the residual, after L+S extraction
        Rq = _rotate_rtn(R, bits, group_size, seed)
    return L + S + Rq, r, k
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `uv run pytest tests/test_arms.py -v`
Expected: 5 passed.

- [ ] **Step 2.5: Full gate + propose commit (USER APPROVAL REQUIRED)**

Run: `uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: 64 passed, 1 xfailed.
Stage `src/bmx/quant/arms.py tests/test_arms.py`, propose:
`feat: compression arms (rtn/rotate/lrs variants) with total-bit accounting`
**Stop and wait for user approval before committing.**

---

### Task 3: Stage A diagnostic experiment

**Files:**
- Create: `experiments/lrs_residual.py` (Stage A part; Stage B added in Task 4)

No unit tests for experiment scripts (project convention: experiments are thin tyro CLIs over tested library code). The verification step is running it.

- [ ] **Step 3.1: Write the Stage A script**

Create `experiments/lrs_residual.py`:

```python
"""Avenue 1: low-rank+sparse quantization residual, two stages in order.

Stage A — structural diagnostic (original basis, NO quantization). One
two-step estimator call per (W, r, k) tests three assumptions at once:
(a) subspace match: does L-hat's column space align with W's own top-r
    left singular subspace (the structure a2 found via Tucker)?
(b) support match: does supp(S-hat) concentrate on the channels d1's
    outlier_mass flags? (Spearman rank corr + top-10 channel overlap.)
(c) spikiness go/no-go: ||L||_max <= alpha/sqrt(d1 d2) — reported as
    spikiness_ratio(L) (alpha-hat); if L-hat is itself spiky the L/S split
    is ill-posed (Wainwright §10.7) and the avenue narrows.

Stage B — compression at matched TOTAL bits (Task 4).
"""

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.decomp.lrs import spikiness_ratio, two_step_lrs
from bmx.quant.stats import kurtosis, outlier_mass
from bmx.stacks.gpt2 import load_gpt2_state


@dataclasses.dataclass
class Config:
    stage: str = "both"  # "a", "b", or "both"
    model_name: str = "gpt2"
    weights: tuple[str, ...] = (
        "transformer.h.5.attn.c_attn.weight",
        "transformer.h.5.attn.c_proj.weight",
        "transformer.h.5.mlp.c_fc.weight",
        "transformer.h.5.mlp.c_proj.weight",
    )
    # d1's worst structured offender; diagnostic-only (not a matmul weight)
    stage_a_extra: tuple[str, ...] = ("transformer.wpe.weight",)
    ranks: tuple[int, ...] = (0, 8, 16, 32, 64)
    sparse_fracs: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2)
    bits: tuple[int, ...] = (2, 3, 4)
    group_size: int = 64
    n_alternations: int = 2
    n_probes: int = 512
    seed: int = 0


def _subspace_overlap(Us: torch.Tensor, U_ref: torch.Tensor) -> float:
    """Mean squared cosine between span(Us) and span(U_ref), in [0, 1]."""
    Q, _ = torch.linalg.qr(Us)
    return ((U_ref.mT @ Q) ** 2).sum().item() / U_ref.shape[1]


def _top_overlap(a: torch.Tensor, b: torch.Tensor, n: int = 10) -> float:
    ia = set(a.topk(n).indices.tolist())
    ib = set(b.topk(n).indices.tolist())
    return len(ia & ib) / n


def stage_a(cfg: Config, sd: dict) -> pd.DataFrame:
    rows = []
    for name in cfg.weights + cfg.stage_a_extra:
        W = sd[name].to(torch.float64)
        m, p = W.shape
        U_W, _, _ = torch.linalg.svd(W, full_matrices=False)
        om = outlier_mass(W)
        for r in cfg.ranks:
            for frac in cfg.sparse_fracs:
                k = int(frac * m * p)
                if r == 0 or k == 0:
                    continue  # diagnostics need both parts present
                Us, V, S = two_step_lrs(W, r, k, cfg.n_alternations)
                L = Us @ V.mT
                R = W - L - S
                supp_frac = (S != 0).to(torch.float64).mean(dim=0)
                rows.append(
                    {
                        "weight": name,
                        "shape": str((m, p)),
                        "r": r,
                        "sparse_frac": frac,
                        "k": k,
                        "subspace_overlap": _subspace_overlap(Us, U_W[:, :r]),
                        "supp_spearman": pd.Series(supp_frac.numpy()).corr(
                            pd.Series(om.numpy()), method="spearman"
                        ),
                        "supp_top10_overlap": _top_overlap(supp_frac, om),
                        "spikiness_W": spikiness_ratio(W),
                        "spikiness_L": spikiness_ratio(L),
                        "spikiness_S": spikiness_ratio(S),  # k > 0 in this loop
                        "rel_error_LS": ((R).norm() / W.norm()).item(),
                        "kurtosis_W": kurtosis(W, dim=-1).mean().item(),
                        "kurtosis_R": kurtosis(R, dim=-1).mean().item(),
                    }
                )
                row = rows[-1]
                print(
                    f"[A] {name} r={r} frac={frac:g}: overlap={row['subspace_overlap']:.3f} "
                    f"spearman={row['supp_spearman']:.3f} spike_L={row['spikiness_L']:.1f} "
                    f"kurt {row['kurtosis_W']:+.2f}->{row['kurtosis_R']:+.2f}"
                )
    return pd.DataFrame(rows)


def main(cfg: Config) -> None:
    sd, _ = load_gpt2_state(cfg.model_name)
    run = create_run("lrs_residual", cfg)
    if cfg.stage in ("a", "both"):
        write_metrics(run, stage_a(cfg, sd), name="stage_a")
    if cfg.stage in ("b", "both"):
        raise NotImplementedError("Stage B lands in Task 4; run with --stage a")
    print(f"-> {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
```

NOTE for the implementing engineer: the `stage b` branch above is intentionally a
hard failure placeholder for ONE task only — Task 4 replaces it. If running Stage A
in this task, use `--stage a`.

- [ ] **Step 3.2: Run Stage A locally (CPU is fine; GPT-2 downloads on first run)**

Run: `uv run python experiments/lrs_residual.py --stage a`
Expected: prints `[A] ...` rows; a `results/lrs_residual/<run-id>/stage_a.parquet` appears. Runtime ~2–5 min on CPU (fp64 SVDs of ≤1024×3072 matrices × 5 weights × 12 grid points × 3 alternations).

- [ ] **Step 3.3: Sanity-read the output with the user's gate criteria**

Report to the user (this is a diagnostic — findings, not a pass/fail test):
- (a) `subspace_overlap` high (≳0.8) at r=32 would mean L-hat finds W's own dominant subspace even with S removed — the a2-consistent reading.
- (b) `supp_spearman` and `supp_top10_overlap` — does sparsity land on d1's channels?
- (c) `spikiness_L` close to `spikiness_W`-after-cleaning vs huge — the identifiability go/no-go.
- `kurtosis_R` vs `kurtosis_W` — the "cleaning Gaussianizes the bulk" claim, pre-quantization.

- [ ] **Step 3.4: Propose commit of the script only (USER APPROVAL REQUIRED)**

`results/` parquet: commit per project convention (metrics/figures yes, checkpoints never).
Run the full gate, stage `experiments/lrs_residual.py results/lrs_residual/`, propose:
`feat: Avenue 1 Stage A structural diagnostic (subspace/support/spikiness)`
**Stop and wait for user approval. Flag that Stage B lands in the next commit, so this commit contains the intentional placeholder import guard.**

---

### Task 4: Stage B matched-bits compression sweep + plot

**Files:**
- Modify: `experiments/lrs_residual.py` (replace the Stage B placeholder)
- Create: `experiments/plots/plot_lrs.py`

- [ ] **Step 4.1: Implement Stage B**

In `experiments/lrs_residual.py`, add imports at top:

```python
from bmx.quant.arms import ARMS, reconstruct_arm, total_bits
from bmx.quant.stats import ip_distortion
```

Add the Stage B function (after `stage_a`):

```python
def stage_b(cfg: Config, sd: dict) -> pd.DataFrame:
    g = torch.Generator().manual_seed(cfg.seed)
    rows = []
    for name in cfg.weights:
        W = sd[name].to(torch.float32)
        m, p = W.shape
        # GPT-2 Conv1D computes y = x @ W: probes live on the INPUT dim m,
        # so distortion is measured on W.mT (rows = output features).
        X = torch.randn(cfg.n_probes, m, generator=g, dtype=torch.float32)
        for bits in cfg.bits:
            for arm in ARMS:
                # plain/rotate arms have no (r, k) grid; lrs arms sweep it
                grid = (
                    [(0, 0.0)]
                    if arm in ("rtn", "rotate_rtn")
                    else [
                        (r, frac)
                        for r in cfg.ranks
                        for frac in cfg.sparse_fracs
                        if r > 0 or frac > 0
                    ]
                )
                for r, frac in grid:
                    k = int(frac * m * p)
                    rec, r_st, k_st = reconstruct_arm(
                        arm,
                        W,
                        bits=bits,
                        group_size=cfg.group_size,
                        r=r,
                        k=k,
                        seed=cfg.seed,
                        n_alternations=cfg.n_alternations,
                    )
                    tb = total_bits(
                        m, p, bits=bits, group_size=cfg.group_size, r=r_st, k=k_st
                    )
                    rows.append(
                        {
                            "weight": name,
                            "arm": arm,
                            "bits": bits,
                            "r": r_st,
                            "sparse_frac": frac,
                            "k": k_st,
                            "total_bits": tb,
                            "bits_per_weight": tb / (m * p),
                            "rel_error": ((rec - W).norm() / W.norm()).item(),
                            "ip_distortion": ip_distortion(W.mT, rec.mT, X),
                        }
                    )
                    row = rows[-1]
                    print(
                        f"[B] {name} {arm} b={bits} r={r_st} frac={frac:g}: "
                        f"{row['bits_per_weight']:.2f} bpw  "
                        f"rel={row['rel_error']:.4f} ip={row['ip_distortion']:.4f}"
                    )
    return pd.DataFrame(rows)
```

Replace the placeholder branch in `main` with:

```python
    if cfg.stage in ("b", "both"):
        write_metrics(run, stage_b(cfg, sd), name="stage_b")
```

- [ ] **Step 4.2: Run Stage B locally**

Run: `uv run python experiments/lrs_residual.py --stage b`
Expected: `[B]` rows; `results/lrs_residual/<run-id>/stage_b.parquet`. Runtime ~10–20 min CPU (the lrs arms refit per grid point; 4 weights × 3 bits × 2 lrs-arms × 19 grid points + cheap baselines). If painful, trim with `--ranks 0 16 32 --sparse-fracs 0.0 1e-3`.

- [ ] **Step 4.3: Write the rate–distortion plot**

Create `experiments/plots/plot_lrs.py` (follow `plot_b1.py`'s read-parquet-never-refit pattern):

```python
"""Rate–distortion view of Stage B: ip_distortion vs bits-per-weight, one
panel per weight, arms as colors. Reads the newest stage_b.parquet."""

import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tyro


@dataclasses.dataclass
class Config:
    run_dir: str = ""  # default: newest results/lrs_residual run with stage_b


def newest_run(root="results/lrs_residual") -> Path:
    runs = sorted(p for p in Path(root).iterdir() if (p / "stage_b.parquet").exists())
    assert runs, f"no stage_b.parquet under {root}"
    return runs[-1]


def main(cfg: Config) -> None:
    run = Path(cfg.run_dir) if cfg.run_dir else newest_run()
    df = pd.read_parquet(run / "stage_b.parquet")
    weights = sorted(df["weight"].unique())
    fig, axes = plt.subplots(1, len(weights), figsize=(5 * len(weights), 4), sharey=True)
    for ax, w in zip(axes, weights):
        sub = df[df["weight"] == w]
        for arm, marker in zip(sorted(sub["arm"].unique()), "o^sx"):
            a = sub[sub["arm"] == arm].sort_values("bits_per_weight")
            ax.scatter(a["bits_per_weight"], a["ip_distortion"], label=arm, marker=marker, s=18)
        ax.set_title(w.removeprefix("transformer."), fontsize=9)
        ax.set_xlabel("bits / weight (total, incl. L+S storage)")
        ax.set_yscale("log")
    axes[0].set_ylabel("inner-product distortion")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = run / "lrs_rate_distortion.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
```

- [ ] **Step 4.4: Generate the figure and read the result**

Run: `uv run python experiments/plots/plot_lrs.py`
Expected: PNG in the run dir. **The Avenue 1 question, answered visually:** at fixed total bits-per-weight, do `lrs_rtn` / `lrs_rotate_rtn` points sit below `rtn` and `rotate_rtn`? Report the answer to the user either way — an honest negative is a valid result.

- [ ] **Step 4.5: Full gate + propose commit (USER APPROVAL REQUIRED)**

Run the full gate, stage `experiments/lrs_residual.py experiments/plots/plot_lrs.py results/lrs_residual/`, propose:
`feat: Avenue 1 Stage B matched-bits compression sweep + rate-distortion figure`
**Stop and wait for user approval before committing.**

---

### Task 5: Implement `eval/layer_swap.py` (offline-testable)

**Files:**
- Modify: `src/bmx/eval/layer_swap.py` (replace stub)
- Create: `tests/test_layer_swap.py`

- [ ] **Step 5.1: Add the `datasets` dependency**

Run: `uv add datasets`
Expected: resolves cleanly. (Used only by the convenience loader; the core functions stay dependency-light and offline-testable.)

- [ ] **Step 5.2: Write the failing tests**

Create `tests/test_layer_swap.py` — tests run on a tiny RANDOM GPT-2 (no download, no network):

```python
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from bmx.eval.layer_swap import perplexity, set_weight


def _tiny():
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=97, n_positions=64)
    torch.manual_seed(0)
    return GPT2LMHeadModel(cfg)


def test_set_weight_replaces_and_changes_logits():
    model = _tiny()
    ids = torch.randint(0, 97, (1, 16), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        before = model(ids).logits.clone()
    W = model.transformer.h[0].attn.c_attn.weight
    set_weight(model, 0, "attn.c_attn", torch.zeros_like(W))
    assert model.transformer.h[0].attn.c_attn.weight.abs().sum() == 0
    with torch.no_grad():
        after = model(ids).logits
    assert not torch.allclose(before, after)


def test_set_weight_validates():
    model = _tiny()
    with pytest.raises(AssertionError):
        set_weight(model, 0, "attn.c_attn", torch.zeros(3, 3))
    with pytest.raises(AssertionError):
        set_weight(model, 0, "not.a.module", torch.zeros(3, 3))


def test_perplexity_finite_and_self_swap_invariant():
    model = _tiny()
    ids = torch.randint(0, 97, (256,), generator=torch.Generator().manual_seed(2))
    p1 = perplexity(model, ids, block=64)
    W = model.transformer.h[1].mlp.c_fc.weight.detach().clone()
    set_weight(model, 1, "mlp.c_fc", W)  # replace with itself
    p2 = perplexity(model, ids, block=64)
    assert p1 > 0 and torch.isfinite(torch.tensor(p1))
    assert abs(p1 - p2) < 1e-4
```

- [ ] **Step 5.3: Run tests to verify they fail**

Run: `uv run pytest tests/test_layer_swap.py -v`
Expected: `ImportError: cannot import name 'perplexity'` (the stub only defines `swap_and_perplexity`).

- [ ] **Step 5.4: Write the implementation**

Replace `src/bmx/eval/layer_swap.py` entirely with:

```python
"""LASER-style layer-selective weight replacement + perplexity.

Originally gated on Track A's A4 decision (which closed negative); now
serving Avenue 1 step 3: the functional metric for structured-residual
quantization. set_weight/perplexity are offline-testable; the
swap_and_perplexity convenience wrapper downloads GPT-2 + WikiText.
"""

import torch

from bmx.decomp.base import FitResult

OBJECTS = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")


def set_weight(model, layer: int, object_name: str, W: torch.Tensor) -> None:
    """Replace transformer.h[layer].<object_name>.weight with W, in place."""
    assert object_name in OBJECTS, f"object must be one of {OBJECTS}"
    module = model.transformer.h[layer]
    for part in object_name.split("."):
        module = getattr(module, part)
    assert module.weight.shape == W.shape, (
        f"shape mismatch: module {tuple(module.weight.shape)} vs W {tuple(W.shape)}"
    )
    with torch.no_grad():
        module.weight.copy_(W.to(module.weight.dtype))


@torch.no_grad()
def perplexity(model, input_ids: torch.Tensor, block: int = 512) -> float:
    """exp(mean NLL) over non-overlapping blocks of a 1-D token stream."""
    assert input_ids.ndim == 1 and input_ids.numel() >= block
    model.eval()
    n = (input_ids.numel() // block) * block
    blocks = input_ids[:n].view(-1, block)
    nll, n_tok = 0.0, 0
    for row in blocks:
        out = model(row.unsqueeze(0), labels=row.unsqueeze(0))
        nll += out.loss.item() * (row.numel() - 1)  # loss is mean over block
        n_tok += row.numel() - 1
    return float(torch.exp(torch.tensor(nll / n_tok)))


def load_eval_tokens(
    model_name: str = "gpt2",
    dataset: str = "wikitext-2-raw-v1",
    n_tokens: int = 65536,
) -> torch.Tensor:
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained(model_name)
    text = "\n\n".join(load_dataset("wikitext", dataset, split="test")["text"])
    return tok(text, return_tensors="pt").input_ids[0][:n_tokens]


def swap_and_perplexity(
    model_name: str,
    layer: int,
    object_name: str,
    fit: FitResult,
    dataset: str = "wikitext-2-raw-v1",
    n_tokens: int = 65536,
) -> tuple[float, float]:
    """One-shot convenience: returns (ppl_base, ppl_swapped). Downloads."""
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(model_name)
    ids = load_eval_tokens(model_name, dataset, n_tokens)
    base = perplexity(model, ids)
    set_weight(model, layer, object_name, fit.reconstruct())
    return base, perplexity(model, ids)
```

Two intentional deviations from the old stub, surface them in the commit message: the return type is now `(base, swapped)` not a single delta (callers need both), and the default dataset is `wikitext-2-raw-v1` not 103 (`load_dataset` pulls the whole config; 103's train split is a multi-GB download for the same-sized test split).

- [ ] **Step 5.5: Run tests to verify they pass**

Run: `uv run pytest tests/test_layer_swap.py -v`
Expected: 3 passed, no network access.

- [ ] **Step 5.6: Full gate + propose commit (USER APPROVAL REQUIRED)**

Run the full gate (expect 67 passed, 1 xfailed), stage
`src/bmx/eval/layer_swap.py tests/test_layer_swap.py pyproject.toml uv.lock`, propose:
`feat: implement layer-swap perplexity eval (set_weight/perplexity offline-testable)`
**Stop and wait for user approval before committing.**

---

### Task 6 (GATED on Stage A/B reading): layer-swap experiment

**Gate:** run this task only after the user has seen Stage A/B results and agrees the arms are worth a functional eval. If Stage B shows no lrs advantage anywhere on the rate–distortion curve, stop here and write the honest negative instead.

**Files:**
- Create: `experiments/lrs_layer_swap.py`

- [ ] **Step 6.1: Write the runner**

Create `experiments/lrs_layer_swap.py`. The `(r, sparse_frac, bits)` defaults below are placeholders in the CLI sense only — the executor sets them from the best Stage B config via flags at run time:

```python
"""Avenue 1 functional eval: WikiText perplexity after swapping ONE GPT-2
weight matrix for each arm's reconstruction, at the Stage-B-chosen config.
Downloads GPT-2 + WikiText-2 on first run; CPU runtime a few minutes per arm
at the default 64k eval tokens."""

import dataclasses

import pandas as pd
import torch
import tyro
from transformers import GPT2LMHeadModel

from bmx.artifacts import create_run, write_metrics
from bmx.eval.layer_swap import load_eval_tokens, perplexity, set_weight
from bmx.quant.arms import ARMS, reconstruct_arm, total_bits


@dataclasses.dataclass
class Config:
    model_name: str = "gpt2"
    layer: int = 5
    object_name: str = "attn.c_proj"  # one of bmx.eval.layer_swap.OBJECTS
    r: int = 32
    sparse_frac: float = 1e-3
    bits: int = 4
    group_size: int = 64
    n_alternations: int = 2
    seed: int = 0
    n_tokens: int = 65536


def main(cfg: Config) -> None:
    run = create_run("lrs_layer_swap", cfg)
    model = GPT2LMHeadModel.from_pretrained(cfg.model_name)
    name = f"transformer.h.{cfg.layer}.{cfg.object_name}.weight"
    W = model.state_dict()[name].detach().clone().to(torch.float32)
    m, p = W.shape
    k = int(cfg.sparse_frac * m * p)
    ids = load_eval_tokens(cfg.model_name, n_tokens=cfg.n_tokens)
    base = perplexity(model, ids)
    rows = [{"arm": "fp32_base", "bits_per_weight": 32.0, "ppl": base}]
    print(f"base ppl = {base:.3f}")
    for arm in ARMS:
        rec, r_st, k_st = reconstruct_arm(
            arm,
            W,
            bits=cfg.bits,
            group_size=cfg.group_size,
            r=cfg.r,
            k=k,
            seed=cfg.seed,
            n_alternations=cfg.n_alternations,
        )
        set_weight(model, cfg.layer, cfg.object_name, rec)
        ppl = perplexity(model, ids)
        set_weight(model, cfg.layer, cfg.object_name, W)  # restore
        tb = total_bits(m, p, bits=cfg.bits, group_size=cfg.group_size, r=r_st, k=k_st)
        rows.append({"arm": arm, "bits_per_weight": tb / (m * p), "ppl": ppl})
        print(f"{arm}: ppl = {ppl:.3f} (+{ppl - base:.3f}) at {tb / (m * p):.2f} bpw")
    write_metrics(run, pd.DataFrame(rows))
    print(f"-> {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
```

- [ ] **Step 6.2: Run with the Stage-B-chosen config**

Run (example; substitute the winning config): `uv run python experiments/lrs_layer_swap.py --r 32 --sparse-frac 1e-3 --bits 4`
Expected: base ppl ≈ 25–30 for GPT-2 on WikiText-2 at 64k tokens; per-arm deltas printed; parquet written. NOTE: per-arm bits-per-weight differ here (lrs arms carry L/S storage) — the honest comparison is ppl-vs-bpw, both recorded.

- [ ] **Step 6.3: Full gate + propose commit (USER APPROVAL REQUIRED)**

Stage `experiments/lrs_layer_swap.py results/lrs_layer_swap/`, propose:
`feat: Avenue 1 layer-swap perplexity comparison across compression arms`
**Stop and wait for user approval before committing.**

---

### Task 7: Record results + update project docs

**Files:**
- Create: `docs/2026-06-11-lrs-results.md` (content written from actual run outputs — structure below)
- Modify: `CLAUDE.md` (Research state section), `docs/HANDOFF.md` (mark Avenue 1 items done/decided)

- [ ] **Step 7.1: Write the results doc**

`docs/2026-06-11-lrs-results.md`, structured like `2026-06-10-h100-session-results.md`:
- Stage A verdict per diagnostic (a)/(b)/(c) with the measured numbers and run-id.
- Stage B verdict: does L+S-then-quantize beat plain/rotate RTN at matched total bits, where on the (bits, r, k) grid, and by how much (cite the figure).
- Layer-swap ppl table if Task 6 ran.
- Explicit gate call: does Avenue 1 proceed to the fused-kernel byte-accounting step (VM session), narrow, or close? An honest negative is a valid result — say which assumption failed.

- [ ] **Step 7.2: Update CLAUDE.md and HANDOFF.md**

In `CLAUDE.md` "Research state": add one Avenue 1 line with the verdict and a pointer to the results doc. In `docs/HANDOFF.md`: update the "Avenue 1 first build" section to reflect what is now built and what the measured gate call was.

- [ ] **Step 7.3: Full gate + propose commit (USER APPROVAL REQUIRED)**

Stage the three docs, propose:
`docs: Avenue 1 Stage A/B results and gate call`
**Stop and wait for user approval before committing.**

---

## Self-review notes (done at plan-writing time)

- **Spec coverage:** HANDOFF item 1 (lrs.py, hard threshold, register, planted tests) → Task 1. Item 2 Stage A → Task 3; Stage B with matched total bits and all four arms → Tasks 2+4. Item 3 (layer_swap, then kernel byte accounting) → Tasks 5–6; the fused-kernel VM step is intentionally OUT of this plan (it needs a rented GPU and only happens if the gate opens — Task 7 records that call). Stage-3 unbiased-QJL extension: out of scope by design (next plan if B clears).
- **Types:** `two_step_lrs` returns `(Us, V, S)` everywhere; `reconstruct_arm` returns `(rec, r_stored, k_stored)` everywhere; `perplexity(model, ids_1d, block)` consistent between tests and experiments.
- **Known judgment calls** (flag to user, don't silently change): `param_count = r(m+p)+k` excludes index bits (bits live in `total_bits`); fp16 storage assumption (`FP_BITS = 16`) for factors/scales/values; wikitext-2 default instead of the stub's wikitext-103.
