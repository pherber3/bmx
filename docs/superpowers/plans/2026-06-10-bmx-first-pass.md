# bmx First Implementation Pass

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the `bmx` research framework and implement Phase 0 (BMD-RALS solver + validation gate), the Track B factored-matvec bench, Track A stack builders + baselines + a2/a3 experiments, and Track D quant utilities + d1 experiment, per `docs/superpowers/specs/2026-06-10-bmx-framework-design.md`.

**Architecture:** Two layers — a framework library (`src/bmx/`: decompositions behind a protocol + registry, stack builders returning tensors-with-metadata, bench/quant utilities, experiment-agnostic artifact IO) and an instance layer (`experiments/`: thin tyro-config scripts per research-plan item). Tests are the permanent Phase 0 validation gate.

**Tech Stack:** Python 3.12, uv (all deps via `uv add`, never hand-edit pins), PyTorch (CPU local / CUDA on VM), tensorly (CP/Tucker baselines), transformers+safetensors (weight extraction), pandas+pyarrow (metrics), tyro (configs), pytest.

**COMMIT POLICY (overrides everything below): NEVER run `git commit` without the user's explicit approval, and NEVER add "Co-Authored-By" or any AI attribution to commit messages. Every "Commit" step in this plan means: stage the listed files with `git add`, present the proposed commit message to the user, and STOP until they approve. Batching several tasks' staged work into one approved commit is fine if the user prefers.**

**Conventions used throughout (memorize before starting):**
- BMD stack axis is **mode 3**: a stack is `T ∈ (n1, n2, h)`, slices are `T[:, :, k]`.
- Factors: `A (n1, ℓ, h)` output gains, `B (n1, n2, ℓ)` **shared templates**, `C (ℓ, n2, h)` input gains.
- `bmp(A,B,C)[i,j,k] = Σ_t A[i,t,k]·B[i,j,t]·C[t,j,k]`, i.e. slice k = `Σ_t diag(A[:,t,k]) @ B[:,:,t] @ diag(C[t,:,k])`.
- Cyclic transpose `cyc(X) = X.permute(1, 2, 0)`; it has order 3; inverse is `X.permute(2, 0, 1)`.
- fp64 in tests, fp32 default in experiments. Shell is bash (git bash). Run everything with `uv run`.

---

### Task 1: Scaffold the project

**Files:**
- Create: `pyproject.toml`, `src/bmx/__init__.py`, `.python-version` (via `uv init`)
- Create: `.gitignore`, `README.md`
- Already present: `docs/` (specs + this plan), `.git/`

- [ ] **Step 1: uv init and python pin**

```bash
cd /d/Projects/bmx
uv init --lib --name bmx --python 3.12 .
```

Note: `uv init` may complain the directory is non-empty; it isn't a problem (docs/ and .git/ coexist fine). If it created `src/bmx/py.typed` and a sample `__init__.py`, keep them but replace `__init__.py` content with just a docstring + `__version__ = "0.1.0"`.

- [ ] **Step 2: Add dependencies (never hand-edit versions)**

```bash
uv add torch numpy tensorly transformers safetensors accelerate datasets einops scipy pandas pyarrow matplotlib tyro
uv add --dev pytest ruff
```

Expected: resolves and creates `uv.lock`, `.venv/`. Torch on Windows = CPU wheel (~200MB+), this is correct per spec.

- [ ] **Step 3: Write .gitignore**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
# large artifacts never committed; parquet metrics and figures ARE committed
results/**/*.safetensors
results/**/*.bin
results/**/*.pt
hf_cache/
```

- [ ] **Step 4: Write minimal README.md**

```markdown
# bmx

Research framework for tensor-decomposition weight compression (BMD / hypermatrix
algebra applied to LLM weights). See `docs/superpowers/specs/2026-06-10-bmx-framework-design.md`
for the design and the three hypotheses under test.

## Quickstart

    uv sync
    uv run pytest -q                 # Phase 0 validation gate
    uv run python experiments/a2_matched_param.py --help

## Layout

- `src/bmx/` — framework: decomp methods (registry), stacks, bench, quant, eval, artifacts
- `experiments/` — thin scripts per research-plan item (a2, a3, b1, d1, ...)
- `results/` — committed metrics/figures (config + git SHA captured per run)
- `scripts/` — NVIDIA-VM workflow (setup, Nsight wrappers)
- `tests/` — permanent Phase 0 validation gate
```

- [ ] **Step 5: Sanity check and commit**

```bash
uv run python -c "import torch; print(torch.__version__)"
mkdir -p tests experiments scripts results
git add -A && git commit -m "chore: scaffold bmx project with uv"
```

---

### Task 2: BM product core ops

**Files:**
- Create: `src/bmx/decomp/__init__.py`, `src/bmx/decomp/ops.py`
- Test: `tests/test_ops.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_ops.py`:

```python
import torch

from bmx.decomp.ops import bmp, cyclic_transpose, cyclic_transpose_inv


def _factors(m=4, p=3, n=5, ell=2, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(m, ell, n, generator=g, dtype=dtype)
    B = torch.randn(m, p, ell, generator=g, dtype=dtype)
    C = torch.randn(ell, p, n, generator=g, dtype=dtype)
    return A, B, C


def test_bmp_shape():
    A, B, C = _factors()
    assert bmp(A, B, C).shape == (4, 3, 5)


def test_bmp_matches_diag_template_slices():
    A, B, C = _factors()
    T = bmp(A, B, C)
    m, p, n = T.shape
    ell = A.shape[1]
    for k in range(n):
        slice_k = sum(
            torch.diag(A[:, t, k]) @ B[:, :, t] @ torch.diag(C[t, :, k])
            for t in range(ell)
        )
        torch.testing.assert_close(T[:, :, k], slice_k)


def test_transpose_identity():
    A, B, C = _factors()
    lhs = cyclic_transpose(bmp(A, B, C))
    rhs = bmp(cyclic_transpose(B), cyclic_transpose(C), cyclic_transpose(A))
    torch.testing.assert_close(lhs, rhs)


def test_cyclic_transpose_order_three():
    T = torch.randn(4, 3, 5, dtype=torch.float64)
    torch.testing.assert_close(
        cyclic_transpose(cyclic_transpose(cyclic_transpose(T))), T
    )
    torch.testing.assert_close(cyclic_transpose_inv(cyclic_transpose(T)), T)
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_ops.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bmx.decomp'`

- [ ] **Step 3: Implement**

`src/bmx/decomp/__init__.py`: empty (or re-export later).

`src/bmx/decomp/ops.py`:

```python
"""Core Bhattacharya-Mesner tensor operations.

Conventions (fixed across the codebase):
    stack tensor  T : (n1, n2, h)   -- slice/stack axis is mode 3
    factor A : (n1, ell, h)         -- per-slice output gains
    factor B : (n1, n2, ell)        -- shared templates
    factor C : (ell, n2, h)         -- per-slice input gains
"""

import torch


def bmp(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """BM product: out[i,j,k] = sum_t A[i,t,k] * B[i,j,t] * C[t,j,k]."""
    m, ell, n = A.shape
    mb, p, lb = B.shape
    lc, pc, nc = C.shape
    assert (m, ell) == (mb, lb) and (ell, p, n) == (lc, pc, nc), (
        f"incompatible BMD factor shapes A={tuple(A.shape)} "
        f"B={tuple(B.shape)} C={tuple(C.shape)}"
    )
    return torch.einsum("itk,ijt,tjk->ijk", A, B, C)


def cyclic_transpose(T: torch.Tensor) -> torch.Tensor:
    """X^T in the BM sense: 1-based permute [2,3,1]. Order 3."""
    return T.permute(1, 2, 0)


def cyclic_transpose_inv(T: torch.Tensor) -> torch.Tensor:
    """Inverse of cyclic_transpose (= applying it twice)."""
    return T.permute(2, 0, 1)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_ops.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/bmx/decomp tests/test_ops.py
git commit -m "feat: BM product and cyclic transpose with diag-template property tests"
```

---

### Task 3: FitResult protocol and method registry

**Files:**
- Create: `src/bmx/decomp/base.py`
- Test: `tests/test_base.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_base.py`:

```python
import pytest
import torch

from bmx.decomp.base import FitResult, available_methods, get_method, register


class _DummyFit(FitResult):
    def __init__(self, T):
        super().__init__(method="dummy", rank=1, loss_history=[1.0, 0.5])
        self._T = T

    def reconstruct(self):
        return torch.zeros_like(self._T)

    def param_count(self):
        return 7


def test_registry_roundtrip():
    @register("dummy")
    def fit_dummy(T, rank):
        return _DummyFit(T)

    assert "dummy" in available_methods()
    fit = get_method("dummy")(torch.ones(2, 2, 2), rank=1)
    assert fit.param_count() == 7
    assert fit.loss_history[-1] == 0.5


def test_relative_error():
    T = torch.ones(2, 2, 2, dtype=torch.float64)
    fit = _DummyFit(T)
    assert fit.relative_error(T) == pytest.approx(1.0)


def test_unknown_method_raises():
    with pytest.raises(KeyError):
        get_method("does-not-exist")
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_base.py -v`
Expected: FAIL with `ImportError` (no `bmx.decomp.base`)

- [ ] **Step 3: Implement**

`src/bmx/decomp/base.py`:

```python
"""Decomposition protocol and method registry (framework extension point #1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch


class FitResult(ABC):
    """A fitted decomposition. param_count() is first-class: all cross-method
    comparisons align on parameters, never on rank."""

    def __init__(self, method: str, rank: Any, loss_history: list[float]):
        self.method = method
        self.rank = rank
        self.loss_history = loss_history

    @abstractmethod
    def reconstruct(self) -> torch.Tensor: ...

    @abstractmethod
    def param_count(self) -> int: ...

    def relative_error(self, T: torch.Tensor) -> float:
        return (
            torch.linalg.norm(self.reconstruct() - T) / torch.linalg.norm(T)
        ).item()


_REGISTRY: dict[str, Callable[..., FitResult]] = {}


def register(name: str):
    def deco(fn: Callable[..., FitResult]):
        _REGISTRY[name] = fn
        return fn

    return deco


def get_method(name: str) -> Callable[..., FitResult]:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown method {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def available_methods() -> list[str]:
    return sorted(_REGISTRY)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_base.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/bmx/decomp/base.py tests/test_base.py
git commit -m "feat: FitResult protocol and decomposition registry"
```

---

### Task 4: Synthetic generators

**Files:**
- Create: `src/bmx/stacks/__init__.py`, `src/bmx/stacks/synthetic.py`
- Test: `tests/test_synthetic.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_synthetic.py`:

```python
import torch

from bmx.decomp.ops import bmp
from bmx.stacks.synthetic import bm_rank_tensor, random_bmd_factors


def test_random_factors_shapes_and_determinism():
    A1, B1, C1 = random_bmd_factors(6, 5, 4, ell=2, seed=3)
    A2, B2, C2 = random_bmd_factors(6, 5, 4, ell=2, seed=3)
    assert A1.shape == (6, 2, 4) and B1.shape == (6, 5, 2) and C1.shape == (2, 5, 4)
    torch.testing.assert_close(A1, A2)
    torch.testing.assert_close(B1, B2)
    torch.testing.assert_close(C1, C2)


def test_bm_rank_tensor_is_bmp_of_factors():
    T, (A, B, C) = bm_rank_tensor(6, 5, 4, ell=2, seed=0)
    torch.testing.assert_close(T, bmp(A, B, C))


def test_slices_are_generically_full_rank():
    # The whole point: low BM-rank does NOT mean low slice rank.
    T, _ = bm_rank_tensor(8, 8, 4, ell=2, seed=0)
    for k in range(4):
        assert torch.linalg.matrix_rank(T[:, :, k]).item() == 8
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_synthetic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bmx.stacks'`

- [ ] **Step 3: Implement**

`src/bmx/stacks/__init__.py`: empty.

`src/bmx/stacks/synthetic.py`:

```python
"""Known-answer tensors for solver validation and random factors for bench shapes."""

import torch

from bmx.decomp.ops import bmp


def random_bmd_factors(
    m: int,
    p: int,
    n: int,
    ell: int,
    seed: int,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(m, ell, n, generator=g, dtype=dtype)
    B = torch.randn(m, p, ell, generator=g, dtype=dtype)
    C = torch.randn(ell, p, n, generator=g, dtype=dtype)
    return A.to(device), B.to(device), C.to(device)


def bm_rank_tensor(
    m: int,
    p: int,
    n: int,
    ell: int,
    seed: int,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
):
    """Exact BM-rank<=ell tensor with known generating factors."""
    A, B, C = random_bmd_factors(m, p, n, ell, seed, dtype=dtype, device=device)
    return bmp(A, B, C), (A, B, C)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_synthetic.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/bmx/stacks tests/test_synthetic.py
git commit -m "feat: synthetic BM-rank tensor generators"
```

---

### Task 5: Constructive initializations (Theorems 3.3 and 3.1)

**Files:**
- Create: `src/bmx/decomp/init.py`
- Test: `tests/test_init.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_init.py`:

```python
import torch

from bmx.decomp.init import mode1_init, ss_svd_init
from bmx.decomp.ops import bmp


def test_ss_svd_init_equals_per_slice_truncated_svd():
    """At zero ALS sweeps, SS-SVD init IS the per-slice truncated SVD baseline."""
    T = torch.randn(8, 6, 5, dtype=torch.float64)
    ell = 3
    A, B, C = ss_svd_init(T, ell)
    rec = bmp(A, B, C)
    for k in range(T.shape[2]):
        U, S, Vh = torch.linalg.svd(T[:, :, k], full_matrices=False)
        trunc = U[:, :ell] @ torch.diag(S[:ell]) @ Vh[:ell, :]
        torch.testing.assert_close(rec[:, :, k], trunc)


def test_ss_svd_exact_when_ell_ge_max_slice_rank():
    T = torch.randn(4, 6, 5, dtype=torch.float64)  # slice rank <= 4
    A, B, C = ss_svd_init(T, ell=4)
    torch.testing.assert_close(bmp(A, B, C), T)


def test_mode1_init_exact_at_unfolding_rank():
    T = torch.randn(3, 4, 5, dtype=torch.float64)  # mode-1 unfolding (15, 4): rank 4
    A, B, C = mode1_init(T, ell=4)
    torch.testing.assert_close(bmp(A, B, C), T)


def test_init_factor_shapes():
    T = torch.randn(8, 6, 5, dtype=torch.float64)
    for init in (ss_svd_init, mode1_init):
        A, B, C = init(T, ell=2)
        assert A.shape == (8, 2, 5)
        assert B.shape == (8, 6, 2)
        assert C.shape == (2, 6, 5)
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_init.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

`src/bmx/decomp/init.py`:

```python
"""Constructive BMD initializations from BM-rank upper bounds.

ss_svd_init  -- Tian-Kilmer Thm 3.3: per-frontal-slice truncated SVD.
mode1_init   -- Tian-Kilmer Thm 3.1: truncated SVD of the mode-1 unfolding.

Both set the template factor B to all-ones, which specializes the BM product
to slice-wise matrix products: bmp(A, 1, C)[:, :, k] = A[:, :, k] @ C[:, :, k].
"""

import torch


def ss_svd_init(T: torch.Tensor, ell: int):
    m, p, n = T.shape
    assert ell <= min(m, p), f"ss_svd_init needs ell <= min(m,p)={min(m, p)}"
    U, S, Vh = torch.linalg.svd(T.permute(2, 0, 1), full_matrices=False)
    A = (U[:, :, :ell] * S[:, None, :ell]).permute(1, 2, 0)  # (m, ell, n)
    C = Vh[:, :ell, :].permute(1, 2, 0)                      # (ell, p, n)
    B = torch.ones(m, p, ell, dtype=T.dtype, device=T.device)
    return A, B, C


def mode1_init(T: torch.Tensor, ell: int):
    m, p, n = T.shape
    X = T.permute(0, 2, 1).reshape(m * n, p)  # X[i*n + k, j] = T[i, j, k]
    assert ell <= min(X.shape), f"mode1_init needs ell <= {min(X.shape)}"
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    Us = U[:, :ell] * S[:ell]
    A = Us.reshape(m, n, ell).permute(0, 2, 1)               # (m, ell, n)
    C = Vh[:ell].unsqueeze(-1).expand(ell, p, n).contiguous()  # same V^T every slice
    B = torch.ones(m, p, ell, dtype=T.dtype, device=T.device)
    return A, B, C
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_init.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/bmx/decomp/init.py tests/test_init.py
git commit -m "feat: SS-SVD and mode-1 constructive BMD initializations"
```

---

### Task 6: BMD-RALS solver (Phase 0 core)

**Files:**
- Create: `src/bmx/decomp/bmd_rals.py`
- Modify: `src/bmx/decomp/__init__.py` (import for registry side-effects)
- Test: `tests/test_bmd_rals.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_bmd_rals.py`:

```python
import torch

from bmx.decomp.bmd_rals import fit_bmd_rals
from bmx.decomp.init import ss_svd_init
from bmx.decomp.ops import bmp
from bmx.stacks.synthetic import bm_rank_tensor


def test_recovers_exact_bm_rank_tensor():
    """Phase 0 gate: generative BM-rank-2 tensors recovered at RE <= 1e-3.

    ALS is non-convex; we require recovery on at least one of three seeds and
    improvement-over-init on all of them.
    """
    finals = []
    for seed in (0, 1, 2):
        T, _ = bm_rank_tensor(16, 16, 8, ell=2, seed=seed)
        fit = fit_bmd_rals(T, rank=2, n_iters=500, tol=1e-12)
        A0, B0, C0 = ss_svd_init(T, 2)
        init_re = (torch.linalg.norm(bmp(A0, B0, C0) - T) / torch.linalg.norm(T)).item()
        assert fit.loss_history[-1] < init_re, "ALS must improve on its init"
        finals.append(fit.loss_history[-1])
    assert min(finals) < 1e-3, f"no seed recovered: {finals}"


def test_loss_monotone_nonincreasing():
    T, _ = bm_rank_tensor(10, 9, 6, ell=2, seed=0)
    fit = fit_bmd_rals(T, rank=2, n_iters=50)
    hist = torch.tensor(fit.loss_history)
    assert (hist[1:] <= hist[:-1] + 1e-12).all(), "ALS loss must not increase"


def test_param_count():
    T, _ = bm_rank_tensor(8, 7, 5, ell=2, seed=0)
    fit = fit_bmd_rals(T, rank=3, n_iters=2)
    # ell * (n1*n2 + n1*h + n2*h)
    assert fit.param_count() == 3 * (8 * 7 + 8 * 5 + 7 * 5)


def test_tikhonov_runs_and_reconstructs():
    T, _ = bm_rank_tensor(8, 8, 4, ell=2, seed=1)
    fit = fit_bmd_rals(T, rank=2, n_iters=50, lam=1e-6)
    assert fit.relative_error(T) < 0.5
    assert fit.reconstruct().shape == T.shape


def test_registered():
    from bmx.decomp.base import get_method

    assert get_method("bmd_rals") is fit_bmd_rals
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_bmd_rals.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

`src/bmx/decomp/bmd_rals.py`:

```python
"""Tian-Kilmer RALS for the BM decomposition.

Each factor update is mp independent least-squares of size (n x ell)
(paper Eqs. 6.5/6.7/6.9), batched as one torch.linalg.lstsq call. The cyclic
transpose identity bmp(A,B,C)^T = bmp(B^T, C^T, A^T) lets a single middle-slot
solver serve all three factor updates.
"""

import torch

from bmx.decomp.base import FitResult, register
from bmx.decomp.init import mode1_init, ss_svd_init
from bmx.decomp.ops import bmp, cyclic_transpose
from bmx.stacks.synthetic import random_bmd_factors


class BMDFit(FitResult):
    def __init__(self, A, B, C, loss_history):
        super().__init__(method="bmd_rals", rank=A.shape[1], loss_history=loss_history)
        self.A, self.B, self.C = A, B, C

    def reconstruct(self) -> torch.Tensor:
        return bmp(self.A, self.B, self.C)

    def param_count(self) -> int:
        m, ell, n = self.A.shape
        p = self.B.shape[1]
        return ell * (m * p + m * n + p * n)


def _solve_middle(T, F1, F3, lam: float):
    """min_F2 ||T - bmp(F1, F2, F3)||_F^2, decoupled over (i, j)."""
    m, p, n = T.shape
    ell = F1.shape[1]
    H = torch.einsum("itk,tjk->ijkt", F1, F3).reshape(m * p, n, ell)
    y = T.reshape(m * p, n, 1)
    if lam == 0.0:
        sol = torch.linalg.lstsq(H, y).solution
    else:
        G = H.mT @ H + lam * torch.eye(ell, dtype=T.dtype, device=T.device)
        sol = torch.linalg.solve(G, H.mT @ y)
    return sol.reshape(m, p, ell)


@register("bmd_rals")
def fit_bmd_rals(
    T: torch.Tensor,
    rank: int,
    *,
    n_iters: int = 200,
    tol: float = 1e-9,
    init: str = "ss_svd",
    lam: float = 0.0,
    seed: int = 0,
) -> BMDFit:
    ell = int(rank)
    if init == "ss_svd":
        A, B, C = ss_svd_init(T, ell)
    elif init == "mode1":
        A, B, C = mode1_init(T, ell)
    elif init == "random":
        m, p, n = T.shape
        A, B, C = random_bmd_factors(
            m, p, n, ell, seed, dtype=T.dtype, device=str(T.device)
        )
    else:
        raise ValueError(f"unknown init {init!r}")

    norm_T = torch.linalg.norm(T)
    Tt = cyclic_transpose(T).contiguous()
    Ttt = cyclic_transpose(Tt).contiguous()
    cyc = cyclic_transpose

    history: list[float] = []
    for _ in range(n_iters):
        B = _solve_middle(T, A, C, lam)
        # C sits in the middle slot of the once-transposed problem.
        Ct = _solve_middle(Tt, cyc(B).contiguous(), cyc(A).contiguous(), lam)
        C = Ct.permute(2, 0, 1).contiguous()
        # A sits in the middle slot of the twice-transposed problem.
        Att = _solve_middle(
            Ttt, cyc(cyc(C)).contiguous(), cyc(cyc(B)).contiguous(), lam
        )
        A = Att.permute(1, 2, 0).contiguous()

        re = (torch.linalg.norm(bmp(A, B, C) - T) / norm_T).item()
        history.append(re)
        if len(history) >= 2 and abs(history[-2] - history[-1]) < tol:
            break

    return BMDFit(A, B, C, history)
```

`src/bmx/decomp/__init__.py` (registry side-effect imports):

```python
from bmx.decomp import bmd_rals as _bmd_rals  # noqa: F401  (registers "bmd_rals")
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_bmd_rals.py -v`
Expected: 5 passed. If `test_recovers_exact_bm_rank_tensor` fails the 1e-3 bar,
first try more iters (1000) and more seeds before touching the solver — ALS
swamps are a known failure mode; the test asserts min-over-seeds for this reason.
Note: with `lam=0` the monotonicity test relies on exact LS solves; if it
flickers at 1e-12 tolerance on your platform, loosen to 1e-10, not more.

- [ ] **Step 5: Run the full suite, verify nothing broke**

Run: `uv run pytest -q`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/bmx/decomp tests/test_bmd_rals.py
git commit -m "feat: BMD-RALS solver with batched decoupled least-squares"
```

---

### Task 7: SageMath agreement fixtures

**Files:**
- Create: `tests/fixtures/README.md`
- Test: `tests/test_sagemath_agreement.py`

- [ ] **Step 1: Write the fixture-format doc**

`tests/fixtures/README.md`:

```markdown
# SageMath BM-ALS agreement fixtures

Export from the SageMath BM-ALS runs (n = 8, 16) into `sagemath_bmals.json`:

```json
{
  "cases": [
    {
      "name": "n8_ell2_seed0",
      "ell": 2,
      "tensor": [[[ ... ]]],
      "sage_rel_error": 0.0123
    }
  ]
}
```

`tensor` is the full nested-list tensor (axes n1, n2, h) the SageMath solver was
run on; `sage_rel_error` is its final relative Frobenius error at rank `ell`.
The pytest in `tests/test_sagemath_agreement.py` auto-skips while this file is
absent. Agreement criterion: bmx RE <= 1.1 * sage RE + 1e-9 (we may do better;
we must not do meaningfully worse).
```

- [ ] **Step 2: Write the (skipping) test**

`tests/test_sagemath_agreement.py`:

```python
import json
from pathlib import Path

import pytest
import torch

from bmx.decomp.bmd_rals import fit_bmd_rals

FIXTURE = Path(__file__).parent / "fixtures" / "sagemath_bmals.json"


@pytest.mark.skipif(not FIXTURE.exists(), reason="SageMath fixture not exported yet")
def test_agreement_with_sagemath():
    cases = json.loads(FIXTURE.read_text())["cases"]
    assert cases, "fixture exists but has no cases"
    for case in cases:
        T = torch.tensor(case["tensor"], dtype=torch.float64)
        fit = fit_bmd_rals(T, rank=case["ell"], n_iters=500, tol=1e-12)
        ours = fit.loss_history[-1]
        theirs = case["sage_rel_error"]
        assert ours <= 1.1 * theirs + 1e-9, (
            f"{case['name']}: bmx RE {ours:.3e} vs sage {theirs:.3e}"
        )
```

- [ ] **Step 3: Run, verify it skips cleanly**

Run: `uv run pytest tests/test_sagemath_agreement.py -v`
Expected: 1 skipped ("SageMath fixture not exported yet")

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/README.md tests/test_sagemath_agreement.py
git commit -m "test: SageMath agreement harness (skips until fixture exported)"
```

---

### Task 8: Baseline decompositions

**Files:**
- Create: `src/bmx/decomp/baselines.py`
- Modify: `src/bmx/decomp/__init__.py`
- Test: `tests/test_baselines.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_baselines.py`:

```python
import torch

from bmx.decomp.baselines import fit_cp, fit_shared_tucker, fit_slice_svd, fit_tucker


def _T(m=8, p=6, n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, p, n, generator=g, dtype=torch.float64)


def test_slice_svd_exact_at_full_rank_and_params():
    T = _T()
    fit = fit_slice_svd(T, rank=6)
    assert fit.relative_error(T) < 1e-10
    fit2 = fit_slice_svd(T, rank=2)
    assert fit2.param_count() == 5 * 2 * (8 + 6)  # h * r * (m + p)


def test_cp_param_count_and_error_decreases():
    T = _T()
    f_small = fit_cp(T, rank=2, seed=0)
    f_big = fit_cp(T, rank=30, seed=0)
    assert f_small.param_count() == 2 * (8 + 6 + 5)
    assert f_big.relative_error(T) < f_small.relative_error(T)


def test_tucker_exact_at_full_rank_and_params():
    T = _T()
    fit = fit_tucker(T, rank=(8, 6, 5))
    assert fit.relative_error(T) < 1e-8
    f2 = fit_tucker(T, rank=(2, 3, 4))
    assert f2.param_count() == 8 * 2 + 6 * 3 + 5 * 4 + 2 * 3 * 4


def test_shared_tucker_exact_at_full_rank_and_params():
    T = _T()
    fit = fit_shared_tucker(T, rank=(8, 6))
    assert fit.relative_error(T) < 1e-8
    f2 = fit_shared_tucker(T, rank=(3, 2))
    # n1*R1 + n2*R2 + h*R1*R2 : per-slice cores, shared factors
    assert f2.param_count() == 8 * 3 + 6 * 2 + 5 * 3 * 2


def test_all_registered():
    from bmx.decomp.base import available_methods

    for name in ("slice_svd", "cp", "tucker", "shared_tucker"):
        assert name in available_methods()
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

`src/bmx/decomp/baselines.py`:

```python
"""Baseline decompositions: per-slice SVD, CP-ALS, Tucker/HOOI, shared-factor
Tucker (TensorLLM's operator: shared mode-0/1 factors, per-slice cores)."""

import numpy as np
import tensorly as tl
import torch
from tensorly.decomposition import parafac, partial_tucker, tucker
from tensorly.tenalg import multi_mode_dot

from bmx.decomp.base import FitResult, register

tl.set_backend("numpy")


def _np(T: torch.Tensor) -> np.ndarray:
    return T.detach().cpu().numpy()


def _back(x: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


class SliceSVDFit(FitResult):
    def __init__(self, W, V, rank, T_shape):
        super().__init__(method="slice_svd", rank=rank, loss_history=[])
        self.W, self.V = W, V  # W: (n, m, r), V: (n, p, r)
        self._shape = T_shape

    def reconstruct(self):
        return torch.einsum("kir,kjr->ijk", self.W, self.V)

    def param_count(self):
        m, p, n = self._shape
        return n * self.rank * (m + p)


@register("slice_svd")
def fit_slice_svd(T: torch.Tensor, rank: int) -> SliceSVDFit:
    r = int(rank)
    U, S, Vh = torch.linalg.svd(T.permute(2, 0, 1), full_matrices=False)
    W = U[:, :, :r] * S[:, None, :r]      # (n, m, r)
    V = Vh[:, :r, :].mT.contiguous()      # (n, p, r)
    fit = SliceSVDFit(W, V, r, tuple(T.shape))
    fit.loss_history = [fit.relative_error(T)]
    return fit


class DenseFit(FitResult):
    """Generic fit storing a dense reconstruction + explicit param count."""

    def __init__(self, method, rank, rec, n_params):
        super().__init__(method=method, rank=rank, loss_history=[])
        self._rec = rec
        self._n_params = n_params

    def reconstruct(self):
        return self._rec

    def param_count(self):
        return self._n_params


@register("cp")
def fit_cp(T: torch.Tensor, rank: int, *, n_iter_max: int = 500, seed: int = 0):
    m, p, n = T.shape
    r = int(rank)
    cp = parafac(
        _np(T), rank=r, n_iter_max=n_iter_max, init="random", random_state=seed
    )
    rec = _back(tl.cp_to_tensor(cp), T)
    fit = DenseFit("cp", r, rec, r * (m + p + n))
    fit.loss_history = [fit.relative_error(T)]
    return fit


@register("tucker")
def fit_tucker(T: torch.Tensor, rank, *, n_iter_max: int = 200, seed: int = 0):
    m, p, n = T.shape
    R1, R2, R3 = (int(x) for x in rank)
    core, factors = tucker(
        _np(T), rank=[R1, R2, R3], n_iter_max=n_iter_max, init="svd"
    )
    rec = _back(tl.tucker_to_tensor((core, factors)), T)
    n_params = m * R1 + p * R2 + n * R3 + R1 * R2 * R3
    fit = DenseFit("tucker", (R1, R2, R3), rec, n_params)
    fit.loss_history = [fit.relative_error(T)]
    return fit


@register("shared_tucker")
def fit_shared_tucker(T: torch.Tensor, rank, *, n_iter_max: int = 200):
    """Tucker with mode-3 factor pinned to identity: shared U1, U2; per-slice cores."""
    m, p, n = T.shape
    R1, R2 = (int(x) for x in rank)
    core, factors = partial_tucker(
        _np(T), rank=[R1, R2], modes=[0, 1], n_iter_max=n_iter_max, init="svd"
    )
    rec = _back(multi_mode_dot(core, factors, modes=[0, 1]), T)
    n_params = m * R1 + p * R2 + n * R1 * R2
    fit = DenseFit("shared_tucker", (R1, R2), rec, n_params)
    fit.loss_history = [fit.relative_error(T)]
    return fit
```

`src/bmx/decomp/__init__.py` becomes:

```python
from bmx.decomp import bmd_rals as _bmd_rals  # noqa: F401
from bmx.decomp import baselines as _baselines  # noqa: F401
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: 5 passed.
Known wobble: tensorly's `partial_tucker` return type has shifted across
versions — if unpacking fails, print the return value; recent versions return
`(core, factors)`; some return a `TuckerTensor` you can unpack the same way.
Fix the unpack here, not in callers.

- [ ] **Step 5: Commit**

```bash
git add src/bmx/decomp tests/test_baselines.py
git commit -m "feat: slice-SVD, CP, Tucker, shared-factor Tucker baselines"
```

---

### Task 9: Factored matvec (Track B kernel, eager + compile)

**Files:**
- Create: `src/bmx/bench/__init__.py`, `src/bmx/bench/factored_matvec.py`
- Test: `tests/test_factored_matvec.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_factored_matvec.py`:

```python
import torch

from bmx.bench.factored_matvec import (
    dense_from_factors,
    dense_slice_matvec,
    factored_matvec,
)
from bmx.stacks.synthetic import random_bmd_factors


def test_factored_matches_dense():
    A, B, C = random_bmd_factors(16, 12, 4, ell=3, seed=0)
    x = torch.randn(5, 12, dtype=torch.float64)  # batch 5
    W = dense_from_factors(A, B, C)              # (h, m, p)
    assert W.shape == (4, 16, 12)
    y_dense = dense_slice_matvec(W, x)           # (h, b, m)
    y_fact = factored_matvec(A, B, C, x)
    assert y_fact.shape == (4, 5, 16)
    torch.testing.assert_close(y_fact, y_dense)


def test_single_slice_is_diag_template_matvec():
    A, B, C = random_bmd_factors(8, 8, 3, ell=2, seed=1)
    x = torch.randn(1, 8, dtype=torch.float64)
    y = factored_matvec(A, B, C, x)
    k = 1
    manual = sum(
        A[:, t, k] * (B[:, :, t] @ (C[t, :, k] * x[0]))
        for t in range(2)
    )
    torch.testing.assert_close(y[k, 0], manual)
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_factored_matvec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bmx.bench'`

- [ ] **Step 3: Implement**

`src/bmx/bench/__init__.py`: empty.

`src/bmx/bench/factored_matvec.py`:

```python
"""The Track B kernel: y_k = sum_t u_t^k * (V_t (w_t^k * x)) vs dense per-slice GEMV.

Bytes story: dense reads h*m*p weights per token; factored reads ell*m*p template
weights (reused across all h slices) + 2*ell*(m+p)*h gain entries. FLOPs inflate
by ~ell. In the memory-bound decode regime bytes are latency.
"""

import torch

from bmx.decomp.ops import bmp


def dense_from_factors(A, B, C) -> torch.Tensor:
    """Materialize the stacked weights W: (h, m, p), W[k] = slice k of bmp."""
    return bmp(A, B, C).permute(2, 0, 1).contiguous()


def dense_slice_matvec(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """W: (h, m, p), x: (b, p) -> y: (h, b, m). The baseline that reads h*m*p bytes."""
    return torch.einsum("kij,bj->kbi", W, x)


def factored_matvec(A, B, C, x: torch.Tensor) -> torch.Tensor:
    """A: (m, ell, h), B: (m, p, ell), C: (ell, p, h), x: (b, p) -> (h, b, m)."""
    xs = torch.einsum("bj,tjk->tkbj", x, C)    # input gains applied
    ys = torch.einsum("ijt,tkbj->tkbi", B, xs)  # template GEMMs (the bulk)
    return torch.einsum("itk,tkbi->kbi", A, ys)  # output gains + sum over t


_compiled = None


def factored_matvec_compiled(A, B, C, x):
    """torch.compile variant; compiles lazily on first call (CUDA recommended)."""
    global _compiled
    if _compiled is None:
        _compiled = torch.compile(factored_matvec)
    return _compiled(A, B, C, x)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_factored_matvec.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/bmx/bench tests/test_factored_matvec.py
git commit -m "feat: factored diag-template matvec with dense baseline"
```

---

### Task 10: Artifacts module

**Files:**
- Create: `src/bmx/artifacts.py`
- Test: `tests/test_artifacts.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_artifacts.py`:

```python
import dataclasses
import json

import pandas as pd

from bmx.artifacts import create_run, write_metrics


@dataclasses.dataclass
class _Cfg:
    layers: tuple[int, ...] = (0, 1)
    device: str = "cpu"


def test_create_run_writes_config_and_env(tmp_path):
    run = create_run("unit_test_exp", _Cfg(), root=tmp_path)
    assert run.is_dir()
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["device"] == "cpu" and cfg["layers"] == [0, 1]
    env = json.loads((run / "env.json").read_text())
    assert "torch" in env and "git_sha" in env


def test_write_metrics_roundtrip(tmp_path):
    run = create_run("unit_test_exp", _Cfg(), root=tmp_path)
    df = pd.DataFrame([{"layer": 0, "method": "bmd_rals", "rel_error": 0.5}])
    write_metrics(run, df)
    back = pd.read_parquet(run / "metrics.parquet")
    assert back.iloc[0]["method"] == "bmd_rals"
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_artifacts.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

`src/bmx/artifacts.py`:

```python
"""Run-directory management: every experiment writes config + env + parquet
metrics under results/<experiment>/<run-id>/ (framework extension point #3)."""

import dataclasses
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch


def git_sha() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path(__file__).resolve().parent,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def create_run(experiment: str, config, root="results") -> Path:
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{git_sha()}"
    run = Path(root) / experiment / run_id
    run.mkdir(parents=True, exist_ok=False)

    if dataclasses.is_dataclass(config):
        cfg = dataclasses.asdict(config)
    elif isinstance(config, dict):
        cfg = config
    else:
        cfg = vars(config)
    (run / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    env = {
        "git_sha": git_sha(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "platform": platform.platform(),
    }
    (run / "env.json").write_text(json.dumps(env, indent=2))
    return run


def write_metrics(run_dir: Path, df: pd.DataFrame, name: str = "metrics") -> Path:
    out = Path(run_dir) / f"{name}.parquet"
    df.to_parquet(out, index=False)
    return out
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_artifacts.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/bmx/artifacts.py tests/test_artifacts.py
git commit -m "feat: experiment-agnostic run artifacts (config, env, parquet metrics)"
```

---

### Task 11: Bench harness + b1 experiment

**Files:**
- Create: `src/bmx/bench/harness.py`
- Create: `experiments/b1_kernel_bench.py`
- Test: `tests/test_harness.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_harness.py`:

```python
import pandas as pd

from bmx.bench.harness import BenchCase, run_cases


def test_run_cases_smoke_cpu():
    cases = [
        BenchCase(m=32, p=32, h=4, ell=2, batch=2, impl="dense"),
        BenchCase(m=32, p=32, h=4, ell=2, batch=2, impl="eager"),
    ]
    df = run_cases(cases, device="cpu", warmup=1, iters=3)
    assert isinstance(df, pd.DataFrame) and len(df) == 2
    row = df[df.impl == "eager"].iloc[0]
    assert row.ms > 0
    assert row.model_bytes_factored < row.model_bytes_dense
    assert {"m", "p", "h", "ell", "batch", "impl", "ms", "flops"} <= set(df.columns)
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_harness.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

`src/bmx/bench/harness.py`:

```python
"""Timing harness for Track B. Correctness is asserted before anything is timed."""

import time
from dataclasses import asdict, dataclass

import pandas as pd
import torch

from bmx.bench.factored_matvec import (
    dense_from_factors,
    dense_slice_matvec,
    factored_matvec,
    factored_matvec_compiled,
)
from bmx.stacks.synthetic import random_bmd_factors


@dataclass
class BenchCase:
    m: int
    p: int
    h: int
    ell: int
    batch: int
    impl: str  # dense | eager | compiled
    dtype: str = "float32"


def _time_callable(fn, args, device: str, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(*args)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(*args)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    return (time.perf_counter() - t0) * 1e3 / iters


def run_cases(
    cases: list[BenchCase], device: str = "cpu", warmup: int = 10, iters: int = 50
) -> pd.DataFrame:
    rows = []
    for case in cases:
        dtype = getattr(torch, case.dtype)
        A, B, C = random_bmd_factors(
            case.m, case.p, case.h, case.ell, seed=0, dtype=dtype, device=device
        )
        x = torch.randn(case.batch, case.p, dtype=dtype, device=device)
        W = dense_from_factors(A, B, C)

        # correctness gate before timing (fp32 tolerance)
        torch.testing.assert_close(
            factored_matvec(A, B, C, x),
            dense_slice_matvec(W, x),
            rtol=1e-3,
            atol=1e-4,
        )

        if case.impl == "dense":
            fn, args = dense_slice_matvec, (W, x)
        elif case.impl == "eager":
            fn, args = factored_matvec, (A, B, C, x)
        elif case.impl == "compiled":
            fn, args = factored_matvec_compiled, (A, B, C, x)
        else:
            raise ValueError(f"unknown impl {case.impl!r}")

        ms = _time_callable(fn, args, device, warmup, iters)

        esize = torch.tensor([], dtype=dtype).element_size()
        dense_flops = 2 * case.h * case.m * case.p * case.batch
        rows.append(
            asdict(case)
            | {
                "ms": ms,
                "device": device,
                "model_bytes_dense": case.h * case.m * case.p * esize,
                "model_bytes_factored": (
                    case.ell * case.m * case.p
                    + case.ell * (case.m + case.p) * case.h
                )
                * esize,
                "flops": (
                    dense_flops if case.impl == "dense" else case.ell * dense_flops
                ),
            }
        )
    return pd.DataFrame(rows)
```

`experiments/b1_kernel_bench.py`:

```python
"""Track B: price the diag-template factored matvec against dense per-slice GEMV.

Local (CPU): correctness + rough numbers. Authoritative numbers: NVIDIA VM,
see scripts/nsight_b1.sh. Prediction under test: wall-time ratio -> h/ell in the
memory-bound regime; report the curve over batch, not a point.
"""

import dataclasses
import itertools

import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.bench.harness import BenchCase, run_cases


@dataclasses.dataclass
class Config:
    d: tuple[int, ...] = (768, 2048, 4096)
    h: tuple[int, ...] = (8, 12, 32, 64)
    ell: tuple[int, ...] = (1, 2, 4, 8)
    batch: tuple[int, ...] = (1, 4, 16, 32)
    impls: tuple[str, ...] = ("dense", "eager", "compiled")
    dtype: str = "float32"
    device: str = "cpu"
    warmup: int = 10
    iters: int = 50


def main(cfg: Config) -> None:
    cases = [
        BenchCase(m=d, p=d, h=h, ell=ell, batch=b, impl=impl, dtype=cfg.dtype)
        for d, h, ell, b, impl in itertools.product(
            cfg.d, cfg.h, cfg.ell, cfg.batch, cfg.impls
        )
        if ell < h  # the claim only matters when templates < slices
    ]
    run = create_run("b1_kernel_bench", cfg)
    df = run_cases(cases, device=cfg.device, warmup=cfg.warmup, iters=cfg.iters)
    write_metrics(run, df)
    print(f"{len(df)} cases -> {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
```

- [ ] **Step 4: Run tests + smoke the experiment**

Run: `uv run pytest tests/test_harness.py -v`
Expected: 1 passed

Run: `uv run python experiments/b1_kernel_bench.py --d 64 --h 4 --ell 2 --batch 1 --impls dense eager --iters 3 --warmup 1`
Expected: prints `8 cases -> results/b1_kernel_bench/<run-id>` and the parquet exists.

- [ ] **Step 5: Commit**

```bash
git add src/bmx/bench experiments/b1_kernel_bench.py tests/test_harness.py
git commit -m "feat: bench harness and b1 kernel benchmark experiment"
```

---

### Task 12: Stack dataclass, GPT-2 builders, permutation null

**Files:**
- Create: `src/bmx/stacks/base.py`, `src/bmx/stacks/gpt2.py`, `src/bmx/stacks/null.py`
- Test: `tests/test_gpt2_stacks.py`, `tests/test_null.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_gpt2_stacks.py`:

```python
import torch

from bmx.stacks.gpt2 import circuit_stack, raw_stack, w_all_4d


def _fake_sd(d=8, n_head=2, n_layer=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for layer in range(n_layer):
        sd[f"transformer.h.{layer}.attn.c_attn.weight"] = torch.randn(
            d, 3 * d, generator=g, dtype=torch.float64
        )
        sd[f"transformer.h.{layer}.attn.c_proj.weight"] = torch.randn(
            d, d, generator=g, dtype=torch.float64
        )
    return sd


def test_raw_stack_shapes_and_values():
    d, n_head = 8, 2
    dh = d // n_head
    sd = _fake_sd(d, n_head)
    q = raw_stack(sd, layer=0, n_head=n_head, which="q")
    assert q.tensor.shape == (d, dh, n_head)
    Wq = sd["transformer.h.0.attn.c_attn.weight"][:, :d]
    torch.testing.assert_close(q.tensor[:, :, 1], Wq[:, dh : 2 * dh])
    assert q.object_name == "raw_q" and q.layer == 0


def test_circuit_stack_wqk_matches_manual():
    d, n_head = 8, 2
    dh = d // n_head
    sd = _fake_sd(d, n_head)
    W = sd["transformer.h.0.attn.c_attn.weight"]
    Wq, Wk = W[:, :d], W[:, d : 2 * d]
    wqk = circuit_stack(sd, layer=0, n_head=n_head, kind="wqk")
    assert wqk.tensor.shape == (d, d, n_head)
    h = 1
    manual = Wq[:, h * dh : (h + 1) * dh] @ Wk[:, h * dh : (h + 1) * dh].T
    torch.testing.assert_close(wqk.tensor[:, :, h], manual)


def test_circuit_stack_wov_matches_manual():
    d, n_head = 8, 2
    dh = d // n_head
    sd = _fake_sd(d, n_head)
    Wv = sd["transformer.h.0.attn.c_attn.weight"][:, 2 * d :]
    Wo = sd["transformer.h.0.attn.c_proj.weight"]
    wov = circuit_stack(sd, layer=0, n_head=n_head, kind="wov")
    h = 0
    manual = Wv[:, h * dh : (h + 1) * dh] @ Wo[h * dh : (h + 1) * dh, :]
    torch.testing.assert_close(wov.tensor[:, :, h], manual)


def test_w_all_4d_shape():
    sd = _fake_sd(8, 2)
    s = w_all_4d(sd, layer=0, n_head=2)
    assert s.tensor.shape == (8, 4, 4, 2)  # (d, d_head, matrix-type, head)
```

`tests/test_null.py`:

```python
import torch

from bmx.stacks.null import permutation_null


def test_null_preserves_per_slice_spectra_as_multiset():
    T = torch.randn(8, 6, 5, dtype=torch.float64)
    Tn, transform = permutation_null(T, seed=0)
    assert Tn.shape == T.shape
    orig = torch.linalg.svdvals(T.permute(2, 0, 1))
    new = torch.linalg.svdvals(Tn.permute(2, 0, 1))
    # slice k of Tn is a two-sided rotation of slice perm[k] of T
    torch.testing.assert_close(new, orig[transform.perm], rtol=1e-10, atol=1e-10)


def test_null_is_seeded_and_changes_tensor():
    T = torch.randn(8, 6, 5, dtype=torch.float64)
    T1, _ = permutation_null(T, seed=1)
    T2, _ = permutation_null(T, seed=1)
    T3, _ = permutation_null(T, seed=2)
    torch.testing.assert_close(T1, T2)
    assert not torch.allclose(T1, T3)
    assert not torch.allclose(T1, T)
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_gpt2_stacks.py tests/test_null.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

`src/bmx/stacks/base.py`:

```python
"""Stack: a tensor plus the metadata that keys every downstream metric row
(framework extension point #2 — any weight-object source implements a builder)."""

from dataclasses import dataclass

import torch


@dataclass
class Stack:
    tensor: torch.Tensor
    model: str
    layer: int
    object_name: str
    axes: tuple[str, ...]
```

`src/bmx/stacks/gpt2.py`:

```python
"""GPT-2 attention weight stacks. Builders take a state-dict mapping so unit
tests can inject small fakes; load_gpt2_state fetches the real checkpoint.

GPT-2 uses Conv1D: weight (in_features, out_features), y = x @ W + b.
c_attn.weight (d, 3d) packs Q|K|V column blocks; per-head columns are
head-major. c_proj.weight (d, d) rows are head-major.
"""

import torch

from bmx.stacks.base import Stack


def load_gpt2_state(model_name: str = "gpt2"):
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(model_name)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    meta = {
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "d": model.config.n_embd,
    }
    return sd, meta


def _per_head(sd: dict, layer: int, n_head: int):
    W = sd[f"transformer.h.{layer}.attn.c_attn.weight"]
    d = W.shape[0]
    dh = d // n_head
    Wq, Wk, Wv = W[:, :d], W[:, d : 2 * d], W[:, 2 * d :]
    # (d, d) column blocks -> (d, d_head, n_head)
    q = Wq.reshape(d, n_head, dh).permute(0, 2, 1)
    k = Wk.reshape(d, n_head, dh).permute(0, 2, 1)
    v = Wv.reshape(d, n_head, dh).permute(0, 2, 1)
    Wo = sd[f"transformer.h.{layer}.attn.c_proj.weight"]
    o = Wo.reshape(n_head, dh, d)  # o[h] = W_O^h : (d_head, d)
    return q, k, v, o, d, dh


def raw_stack(sd: dict, layer: int, n_head: int, which: str, model="gpt2") -> Stack:
    q, k, v, o, d, dh = _per_head(sd, layer, n_head)
    tensors = {"q": q, "k": k, "v": v, "o": o.permute(2, 1, 0)}  # o -> (d, dh, h)
    assert which in tensors, f"which must be one of {sorted(tensors)}"
    return Stack(
        tensors[which].contiguous(), model, layer, f"raw_{which}",
        ("d_model", "d_head", "head"),
    )


def circuit_stack(sd: dict, layer: int, n_head: int, kind: str, model="gpt2") -> Stack:
    q, k, v, o, d, dh = _per_head(sd, layer, n_head)
    if kind == "wqk":
        T = torch.einsum("ich,jch->ijh", q, k)  # W_Q^h @ W_K^h.T
    elif kind == "wov":
        T = torch.einsum("ich,hcj->ijh", v, o)  # W_V^h @ W_O^h
    else:
        raise ValueError(f"kind must be 'wqk' or 'wov', got {kind!r}")
    return Stack(
        T.contiguous(), model, layer, kind, ("d_model", "d_model", "head")
    )


def w_all_4d(sd: dict, layer: int, n_head: int, model="gpt2") -> Stack:
    """TensorLLM's object: (d_model, d_head, matrix-type[Q,K,V,O^T], head)."""
    q, k, v, o, d, dh = _per_head(sd, layer, n_head)
    T = torch.stack([q, k, v, o.permute(2, 1, 0)], dim=2)
    return Stack(
        T.contiguous(), model, layer, "w_all",
        ("d_model", "d_head", "matrix_type", "head"),
    )
```

`src/bmx/stacks/null.py`:

```python
"""The permutation null (A3): destroys cross-slice alignment, preserves
per-slice spectra. The per-slice two-sided orthogonal rotations are the
load-bearing part; the slice shuffle alone is absorbed into every method's
slice-mode factor."""

from dataclasses import dataclass

import torch


@dataclass
class NullTransform:
    seed: int
    perm: torch.Tensor   # (h,)
    Q: torch.Tensor      # (h, m, m) left rotations
    R: torch.Tensor      # (h, p, p) right rotations


def _random_orthogonal_batch(count: int, dim: int, g, dtype):
    M = torch.randn(count, dim, dim, generator=g, dtype=dtype)
    Q, _ = torch.linalg.qr(M)
    return Q


def permutation_null(T: torch.Tensor, seed: int):
    m, p, n = T.shape
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    Q = _random_orthogonal_batch(n, m, g, T.dtype)
    R = _random_orthogonal_batch(n, p, g, T.dtype)
    X = T[:, :, perm].permute(2, 0, 1)        # (n, m, p)
    Y = Q @ X @ R.mT                          # slice k -> Q_k T[:,:,perm_k] R_k^T
    return Y.permute(1, 2, 0).contiguous(), NullTransform(seed, perm, Q, R)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_gpt2_stacks.py tests/test_null.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/bmx/stacks tests/test_gpt2_stacks.py tests/test_null.py
git commit -m "feat: GPT-2 attention stacks and permutation null control"
```

---

### Task 13: Quant module (Track D utilities)

**Files:**
- Create: `src/bmx/quant/__init__.py`, `src/bmx/quant/hadamard.py`, `src/bmx/quant/rtn.py`, `src/bmx/quant/stats.py`
- Test: `tests/test_quant.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_quant.py`:

```python
import scipy.linalg
import torch

from bmx.quant.hadamard import fwht, random_orthogonal, randomized_hadamard
from bmx.quant.rtn import rtn_quantize
from bmx.quant.stats import ip_distortion, kurtosis, outlier_mass, sq_floor


def test_fwht_matches_scipy_hadamard():
    d = 16
    H = torch.tensor(scipy.linalg.hadamard(d), dtype=torch.float64) / d**0.5
    X = torch.eye(d, dtype=torch.float64)
    torch.testing.assert_close(fwht(X), H)  # rows of identity -> rows of H


def test_fwht_is_involution_and_isometry():
    x = torch.randn(3, 32, dtype=torch.float64)
    torch.testing.assert_close(fwht(fwht(x)), x)
    torch.testing.assert_close(
        x.norm(dim=-1), fwht(x).norm(dim=-1)
    )


def test_randomized_hadamard_and_orthogonal_are_isometries():
    x = torch.randn(5, 64, dtype=torch.float64)
    y = randomized_hadamard(x, seed=0)
    assert not torch.allclose(y, fwht(x))
    torch.testing.assert_close(x.norm(dim=-1), y.norm(dim=-1))
    Q = random_orthogonal(48, seed=0, dtype=torch.float64)
    torch.testing.assert_close(Q @ Q.T, torch.eye(48, dtype=torch.float64))


def test_rtn_error_decreases_with_bits():
    W = torch.randn(16, 128, dtype=torch.float64)
    errs = [
        (rtn_quantize(W, bits=b, group_size=32) - W).norm() / W.norm()
        for b in (2, 3, 4, 8)
    ]
    assert errs[0] > errs[1] > errs[2] > errs[3]
    assert errs[3] < 0.01


def test_gaussianization_kurtosis_drops():
    """Heavy-tailed rows become near-Gaussian under random rotation."""
    g = torch.Generator().manual_seed(0)
    W = torch.distributions.StudentT(4.0).sample((64, 256))  # excess kurtosis >> 0
    before = kurtosis(W, dim=-1).mean()
    Q = random_orthogonal(256, seed=1, dtype=W.dtype)
    after = kurtosis(W @ Q.T, dim=-1).mean()
    assert after < before / 2


def test_outlier_mass_and_floor():
    W = torch.randn(32, 64, dtype=torch.float64)
    mass = outlier_mass(W, k_sigma=3.0)
    assert mass.shape == (64,)
    assert 0 <= mass.min() and mass.max() <= 1
    assert sq_floor(2) == 4.0**-2


def test_ip_distortion_zero_for_exact():
    W = torch.randn(8, 32, dtype=torch.float64)
    X = torch.randn(16, 32, dtype=torch.float64)
    assert ip_distortion(W, W, X) == 0
    assert ip_distortion(W, rtn_quantize(W, 4, 32), X) > 0
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_quant.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bmx.quant'`

- [ ] **Step 3: Implement**

`src/bmx/quant/__init__.py`: empty.

`src/bmx/quant/hadamard.py`:

```python
"""Data-oblivious rotations: fast Walsh-Hadamard (power-of-2 dims, Sylvester
ordering) and QR random orthogonal (any dim, the Haar reference)."""

import torch


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal FWHT over the last dim (must be a power of 2)."""
    d = x.shape[-1]
    assert d & (d - 1) == 0 and d > 0, f"fwht requires power-of-2 dim, got {d}"
    orig_shape = x.shape
    y = x.reshape(-1, d).clone()
    h = 1
    while h < d:
        y = y.view(-1, d // (2 * h), 2, h)
        pos = y[:, :, 0, :] + y[:, :, 1, :]
        neg = y[:, :, 0, :] - y[:, :, 1, :]
        y = torch.stack((pos, neg), dim=2).view(-1, d)
        h *= 2
    return (y / d**0.5).view(orig_shape)


def randomized_hadamard(x: torch.Tensor, seed: int) -> torch.Tensor:
    """H @ diag(signs) @ x rows — the standard randomized Hadamard rotation."""
    d = x.shape[-1]
    g = torch.Generator().manual_seed(seed)
    signs = (torch.randint(0, 2, (d,), generator=g) * 2 - 1).to(x.dtype)
    return fwht(x * signs)


def random_orthogonal(
    d: int, seed: int, dtype=torch.float32, device="cpu"
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(d, d, generator=g, dtype=dtype)
    Q, _ = torch.linalg.qr(M)
    return Q.to(device)
```

`src/bmx/quant/rtn.py`:

```python
"""Groupwise symmetric round-to-nearest quantization (returns dequantized values)."""

import torch


def rtn_quantize(W: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    *lead, d = W.shape
    assert d % group_size == 0, f"dim {d} not divisible by group {group_size}"
    qmax = 2 ** (bits - 1) - 1
    G = W.reshape(*lead, d // group_size, group_size)
    scale = G.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    Q = (G / scale).round().clamp(-qmax - 1, qmax)
    return (Q * scale).reshape(W.shape)
```

`src/bmx/quant/stats.py`:

```python
"""Distribution diagnostics for D1 and the distortion-floor machinery for D3."""

import torch


def kurtosis(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Fisher excess kurtosis along dim (0 for Gaussian)."""
    mu = x.mean(dim=dim, keepdim=True)
    var = x.var(dim=dim, unbiased=False, keepdim=True)
    return ((x - mu) ** 4).mean(dim=dim) / var.squeeze(dim) ** 2 - 3.0


def outlier_mass(W: torch.Tensor, k_sigma: float = 3.0) -> torch.Tensor:
    """Per-channel (last-dim column) fraction of entries beyond k_sigma * global std."""
    thresh = k_sigma * W.std()
    return (W.abs() > thresh).to(torch.float64).mean(dim=0)


def ip_distortion(W: torch.Tensor, Wq: torch.Tensor, X: torch.Tensor) -> float:
    """Relative inner-product distortion ||W X^T - Wq X^T||_F / ||W X^T||_F."""
    ref = W @ X.mT
    return ((Wq @ X.mT - ref).norm() / ref.norm()).item()


def sq_floor(bits: int) -> float:
    """Worst-case MSE rate floor 4^-b (Shannon + Yao, TurboQuant §3.3)."""
    return 4.0 ** (-bits)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_quant.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/bmx/quant tests/test_quant.py
git commit -m "feat: rotation, RTN quantization, and distribution-stats utilities"
```

---

### Task 14: Sweep helper + a2 / a3 / d1 experiments

**Files:**
- Create: `src/bmx/sweep.py`
- Create: `experiments/a2_matched_param.py`, `experiments/a3_permutation_null.py`, `experiments/d1_gaussianization.py`
- Test: `tests/test_sweep.py`

- [ ] **Step 1: Write the failing test**

`tests/test_sweep.py`:

```python
import torch

from bmx.stacks.base import Stack
from bmx.sweep import decomp_sweep


def test_decomp_sweep_rows_and_keys():
    T = torch.randn(8, 8, 4, dtype=torch.float64)
    stack = Stack(T, model="test", layer=3, object_name="wqk", axes=("a", "b", "h"))
    plan = {"bmd_rals": [1, 2], "slice_svd": [2], "shared_tucker": [(4, 4)]}
    df = decomp_sweep(stack, plan, fit_opts={"bmd_rals": {"n_iters": 5}})
    assert len(df) == 4
    assert {"model", "layer", "object", "method", "rank", "params", "rel_error",
            "seconds"} <= set(df.columns)
    assert (df[df.method == "slice_svd"].params == 4 * 2 * (8 + 8)).all()
    assert (df.rel_error >= 0).all() and (df.rel_error <= 1.5).all()
```

- [ ] **Step 2: Run test, verify failure**

Run: `uv run pytest tests/test_sweep.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

`src/bmx/sweep.py`:

```python
"""Matched-parameter decomposition sweep: the shared engine of a2/a3/c2.

Output rows are keyed by (model, layer, object, method, rank, params) so
distribution-over-layers reporting is the default downstream."""

import time

import pandas as pd

from bmx.decomp.base import get_method
from bmx.stacks.base import Stack


def decomp_sweep(
    stack: Stack,
    plan: dict[str, list],
    fit_opts: dict[str, dict] | None = None,
    extra_cols: dict | None = None,
) -> pd.DataFrame:
    fit_opts = fit_opts or {}
    rows = []
    for method, ranks in plan.items():
        fn = get_method(method)
        for rank in ranks:
            t0 = time.perf_counter()
            fit = fn(stack.tensor, rank, **fit_opts.get(method, {}))
            seconds = time.perf_counter() - t0
            rows.append(
                {
                    "model": stack.model,
                    "layer": stack.layer,
                    "object": stack.object_name,
                    "method": method,
                    "rank": str(rank),
                    "params": fit.param_count(),
                    "rel_error": fit.relative_error(stack.tensor),
                    "seconds": seconds,
                    "n_iters": len(fit.loss_history),
                }
                | (extra_cols or {})
            )
    return pd.DataFrame(rows)
```

`experiments/a2_matched_param.py`:

```python
"""A2: matched-parameter comparison on GPT-2 attention stacks.

Fits BMD-RALS vs slice-SVD vs CP vs Tucker vs shared-factor Tucker across
layers and objects; the load-bearing axis downstream is error vs param_count.
"""

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.stacks.gpt2 import circuit_stack, load_gpt2_state, raw_stack
from bmx.stacks.null import permutation_null
from bmx.sweep import decomp_sweep


@dataclasses.dataclass
class Config:
    model_name: str = "gpt2"
    layers: tuple[int, ...] = tuple(range(12))
    objects: tuple[str, ...] = ("wqk", "wov")  # also: raw_q raw_k raw_v raw_o
    dtype: str = "float32"
    null_seed: int = -1  # >= 0 applies the permutation null (used by a3)
    bmd_iters: int = 200
    experiment: str = "a2_matched_param"


# rank grids chosen to span comparable param ranges per method on (768, 768, 12)
PLAN = {
    "bmd_rals": [1, 2, 4, 8],
    "slice_svd": [1, 2, 4, 8, 16, 32, 64],
    "cp": [8, 32, 128, 512],
    "tucker": [(32, 32, 4), (64, 64, 8), (128, 128, 12), (256, 256, 12)],
    "shared_tucker": [(16, 16), (32, 32), (64, 64), (128, 128)],
}


def build_stack(sd, meta, layer: int, obj: str, dtype):
    if obj.startswith("raw_"):
        s = raw_stack(sd, layer, meta["n_head"], which=obj.removeprefix("raw_"))
    else:
        s = circuit_stack(sd, layer, meta["n_head"], kind=obj)
    s.tensor = s.tensor.to(getattr(torch, dtype))
    return s


def main(cfg: Config) -> None:
    sd, meta = load_gpt2_state(cfg.model_name)
    run = create_run(cfg.experiment, cfg)
    frames = []
    for layer in cfg.layers:
        for obj in cfg.objects:
            stack = build_stack(sd, meta, layer, obj, cfg.dtype)
            extra = {}
            if cfg.null_seed >= 0:
                stack.tensor, _ = permutation_null(
                    stack.tensor, seed=cfg.null_seed + layer
                )
                extra = {"null_seed": cfg.null_seed + layer}
            df = decomp_sweep(
                stack,
                PLAN,
                fit_opts={"bmd_rals": {"n_iters": cfg.bmd_iters}},
                extra_cols=extra,
            )
            frames.append(df)
            print(f"layer {layer} {obj}: {len(df)} fits done")
    write_metrics(run, pd.concat(frames, ignore_index=True))
    print(f"-> {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
```

`experiments/a3_permutation_null.py`:

```python
"""A3: the permutation null. Same sweep as A2 on alignment-destroyed stacks.

If BMD's advantage survives this control, the advantage is expressivity
(parameters per component), not cross-slice structure — and the diag-template
hypothesis is dead regardless of raw numbers."""

import dataclasses

import tyro

from a2_matched_param import Config, main


@dataclasses.dataclass
class NullConfig(Config):
    null_seed: int = 0
    experiment: str = "a3_permutation_null"


if __name__ == "__main__":
    main(tyro.cli(NullConfig))
```

`experiments/d1_gaussianization.py`:

```python
"""D1: do trained weight rows Gaussianize under data-oblivious rotation?

Tests entry 3's failure mode 2 directly: weights are trained, correlated
objects — does the random-vector Gaussianization argument survive contact
with them? Reports per-matrix kurtosis and outlier mass, before vs after."""

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.quant.hadamard import random_orthogonal
from bmx.quant.stats import kurtosis, outlier_mass
from bmx.stacks.gpt2 import load_gpt2_state


@dataclasses.dataclass
class Config:
    model_name: str = "gpt2"
    seed: int = 0
    min_dim: int = 64  # skip tiny weights (layernorms, biases)


def main(cfg: Config) -> None:
    sd, _ = load_gpt2_state(cfg.model_name)
    run = create_run("d1_gaussianization", cfg)
    rows = []
    for name, W in sd.items():
        if W.ndim != 2 or min(W.shape) < cfg.min_dim:
            continue
        W = W.to(torch.float64)
        d = W.shape[-1]
        Q = random_orthogonal(d, seed=cfg.seed, dtype=torch.float64)
        Wr = W @ Q.T  # rotate rows
        rows.append(
            {
                "weight": name,
                "shape": str(tuple(W.shape)),
                "kurtosis_before": kurtosis(W, dim=-1).mean().item(),
                "kurtosis_after": kurtosis(Wr, dim=-1).mean().item(),
                "outlier_mass_before": outlier_mass(W).mean().item(),
                "outlier_mass_after": outlier_mass(Wr).mean().item(),
                "outlier_mass_max_before": outlier_mass(W).max().item(),
                "outlier_mass_max_after": outlier_mass(Wr).max().item(),
            }
        )
        print(f"{name}: kurtosis {rows[-1]['kurtosis_before']:+.3f} -> "
              f"{rows[-1]['kurtosis_after']:+.3f}")
    write_metrics(run, pd.DataFrame(rows))
    print(f"-> {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
```

- [ ] **Step 4: Run tests + smoke a2 on one layer**

Run: `uv run pytest tests/test_sweep.py -v`
Expected: 1 passed

Run: `uv run python experiments/a2_matched_param.py --layers 0 --objects wqk --bmd-iters 5`
Expected: downloads GPT-2 on first run (~500MB), prints `layer 0 wqk: ... fits done`, writes parquet. (CP at rank 512 on 768×768×12 takes a few minutes on CPU — acceptable for the smoke test; Ctrl-C and rerun with a smaller plan is also fine.)

- [ ] **Step 5: Commit**

```bash
git add src/bmx/sweep.py experiments/ tests/test_sweep.py
git commit -m "feat: decomposition sweep engine and a2/a3/d1 experiments"
```

---

### Task 15: VM scripts, eval stubs, final pass

**Files:**
- Create: `scripts/vm_setup.sh`, `scripts/nsight_b1.sh`
- Create: `src/bmx/eval/__init__.py`, `src/bmx/eval/layer_swap.py`, `src/bmx/eval/expert_error.py`

- [ ] **Step 1: Write VM scripts**

`scripts/vm_setup.sh`:

```bash
#!/usr/bin/env bash
# One-time setup on a fresh NVIDIA VM (Ubuntu). Run from anywhere.
set -euo pipefail

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd "$(dirname "$0")/.."
uv sync
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.get_device_name(0))"
uv run pytest -q
echo "VM ready."
```

`scripts/nsight_b1.sh`:

```bash
#!/usr/bin/env bash
# Track B authoritative measurement: bytes from DRAM + achieved bandwidth via
# Nsight Compute, around the b1 kernel bench. Run on the NVIDIA VM only.
# Usage: scripts/nsight_b1.sh [extra b1 args...]
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="results/b1_kernel_bench/ncu-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --csv --log-file "$OUT/ncu.csv" \
    uv run python experiments/b1_kernel_bench.py \
      --device cuda --impls dense eager compiled --iters 10 --warmup 3 "$@"

echo "ncu metrics -> $OUT/ncu.csv (commit this with the run's parquet)"
```

```bash
chmod +x scripts/vm_setup.sh scripts/nsight_b1.sh
```

- [ ] **Step 2: Write eval stubs (shaped signatures, explicit NotImplementedError)**

`src/bmx/eval/__init__.py`: empty.

`src/bmx/eval/layer_swap.py`:

```python
"""A5 (gated on the A4 decision): LASER-style layer-selective weight replacement
and WikiText-103 perplexity. Implemented when Track A's gate opens."""

import torch

from bmx.decomp.base import FitResult


def swap_and_perplexity(
    model_name: str,
    layer: int,
    object_name: str,
    fit: FitResult,
    dataset: str = "wikitext-103-raw-v1",
) -> float:
    """Replace one layer's object with fit.reconstruct(), return perplexity delta."""
    raise NotImplementedError("gated on A4 decision; see research plan Track A5")
```

`src/bmx/eval/expert_error.py`:

```python
"""C3 (gated on C1/C2): routed-token capture and expert-output relative error.
Implemented when Track C opens."""

import torch


def capture_routed_activations(model_name: str, n_tokens: int, layer: int):
    """Forward hooks collecting per-expert input activations on calibration text."""
    raise NotImplementedError("gated on C1 census; see research plan Track C3")


def expert_output_error(
    W_true: torch.Tensor, W_rec: torch.Tensor, X: torch.Tensor
) -> float:
    """Relative L2 of expert outputs under reconstructed weights, given captured X."""
    raise NotImplementedError("gated on C1 census; see research plan Track C3")
```

- [ ] **Step 2b: Write the Track C stack stub**

`src/bmx/stacks/moe.py`:

```python
"""C-track (gated on C1 census): stacked expert matrices from fine-grained MoE
checkpoints, loaded from safetensors shards without instantiating the model.
Implemented when Track C opens; shaped now so a2-style sweeps port directly."""

from bmx.stacks.base import Stack


def expert_stack(
    checkpoint_dir: str, layer: int, which: str, model: str = ""
) -> Stack:
    """Stack of per-expert FFN matrices, which in {gate, up, down} -> (d, d_ff, E)."""
    raise NotImplementedError("gated on C1 census; see research plan Track C")
```

- [ ] **Step 3: Full suite + lint**

Run: `uv run pytest -q`
Expected: all passed, 1 skipped (sagemath fixture)

Run: `uv run ruff check .`
Expected: clean (fix anything it flags)

- [ ] **Step 4: Update README with VM workflow section**

Append to `README.md`:

```markdown
## NVIDIA VM workflow (Track B authoritative numbers)

1. Push your branch; on the VM: `git clone <repo> && cd bmx && scripts/vm_setup.sh`
2. `scripts/nsight_b1.sh` (wraps `experiments/b1_kernel_bench.py --device cuda` in ncu)
3. `git add results/ && git commit && git push` — metrics come home as parquet + csv

## SageMath fixtures

Export per `tests/fixtures/README.md` to activate the agreement test.
```

- [ ] **Step 5: Commit**

```bash
git add scripts/ src/bmx/eval README.md
git commit -m "feat: VM scripts, eval stubs, README workflow docs"
```

---

## Post-plan notes for the executor

- **Phase 0 validation is the pytest suite**, not a script: the spec's `p0_validate` experiment slot is intentionally fulfilled by `tests/` (run `uv run pytest -q`), per the spec's "Tests = Phase 0 gate" section.
- **Order matters** up to Task 8 (each task imports the previous); Tasks 9–13 are independent of each other and can be done in any order; Task 14 needs 8+10+12; Task 15 is last.
- **Known risk spots:** tensorly `partial_tucker` unpacking (Task 8 Step 4 note); ALS recovery threshold (Task 6 Step 4 note); `torch.compile` is unavailable/slow on Windows CPU — never required by tests (only the `compiled` impl path uses it, and only b1 on the VM exercises it by default).
- **Out of scope for this plan** (per spec): Triton kernel (VM follow-up), Track C beyond stubs, AWQ/GPTQ integrations, KD recovery, CI.
