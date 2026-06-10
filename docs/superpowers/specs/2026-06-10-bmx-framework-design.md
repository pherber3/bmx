# bmx — Tensor-Decomposition Weight-Compression Research Framework

**Date:** 2026-06-10
**Status:** Approved design
**Scope:** Project scaffold + framework layer + first implementation pass (Phase 0, Track B bench, Track A stacks, Track D quant utilities)

## Purpose

`bmx` is a research codebase for kill-or-confirm testing of three speculative hypotheses about hypermatrix (Bhattacharya–Mesner) algebra applied to LLM weight compression and inference bandwidth:

1. **Diag-template factored matvec** — BM rank-ℓ structure on stacked weight matrices gives an h/ℓ bandwidth amplification in memory-bound decode (`wiki/.speculative/2026-06-09 BM Diag-Template Slices as Bandwidth-Amplifying Factored Matvec.md`).
2. **BMD expert streaming** — resident shared templates + per-expert gain vectors collapse PCIe transfer cost for fine-grained MoE offload (`wiki/.speculative/2026-06-09 BMD Expert Streaming for Fine-Grained MoE Offload.md`).
3. **VQ theory transfer** — the vault's data-oblivious VQ theory (rotation Gaussianization, MSE-vs-IP distortion, 4^-b floors) grounds rotation-based weight quantization (`wiki/.speculative/2026-06-09 Vault VQ Theory as the Missing Foundation for Weight Quantization.md`).

The experimental program is the Tracks A–D plan (cheap diagnostics gate expensive work; math and systems claims tested independently). The repo is also a **reusable framework**: the library layer is decomposition-method-agnostic and experiment-agnostic so future related research plugs in without restructuring.

## Decisions (settled with user)

- **Location/name:** `D:\Projects\bmx`, package `bmx`.
- **Environment:** uv-managed; `uv init --lib` src layout; Python pinned 3.12; **all dependencies added via `uv add`, pyproject never hand-edited for versions.** Plain `uv add torch` serves both targets (Windows PyPI wheels = CPU; Linux PyPI wheels = CUDA-bundled).
- **Hardware:** hybrid. Local machine (Windows, AMD 7900 XTX) runs CPU-tractable math tracks; an NVIDIA cloud VM is the measurement bench for Track B (Triton + Nsight Compute) and large sweeps. Core code is device-agnostic (`device` is a config string; ROCm and CUDA both present as `torch.cuda`).
- **Shell:** bash (git bash) for all commands.
- **HF auth:** read from `HF_TOKEN` env var / `hf auth login` cache. Never committed.
- **Initial dependencies:** `torch numpy tensorly transformers safetensors accelerate datasets einops scipy pandas pyarrow matplotlib tyro`; dev group `pytest ruff`.

## Architecture

Two layers:

- **Framework layer** (`src/bmx/`): reusable, tested, experiment-agnostic. Decomposition methods behind one protocol + registry; stack builders producing typed tensors-with-metadata; bench/quant/eval utilities; artifact IO.
- **Instance layer** (`experiments/`, `docs/`): thin scripts implementing the current research plan's items, plus notes. Disposable without touching the framework.

```
bmx/
├── pyproject.toml              # uv-managed only
├── src/bmx/
│   ├── decomp/
│   │   ├── base.py             # Decomposition protocol, FitResult, registry
│   │   ├── bmd_rals.py         # Tian–Kilmer RALS (batched lstsq, transpose identity)
│   │   ├── init.py             # SS-SVD (Thm 3.3) and mode-1 unfolding (Thm 3.1) inits
│   │   └── baselines.py        # slice-SVD, CP-ALS, Tucker/HOOI, shared-factor Tucker
│   ├── stacks/
│   │   ├── base.py             # Stack dataclass: tensor + metadata (model, layer, object)
│   │   ├── gpt2.py             # W_QK / W_OV circuit stacks, raw head stacks, 4D W_all
│   │   ├── moe.py              # expert stacks from safetensors shards (no model instantiation)
│   │   ├── synthetic.py        # generative BM-rank-r known-answer tensors; bench shapes
│   │   └── null.py             # permutation null (seeded slice shuffle + per-slice orthogonals)
│   ├── bench/
│   │   ├── factored_matvec.py  # eager / torch.compile / triton (guarded import) + dense GEMV
│   │   └── harness.py          # CUDA-event timing, shape sweeps, correctness-before-timing
│   ├── quant/
│   │   ├── hadamard.py         # fast Walsh–Hadamard, padding/blocking
│   │   ├── rtn.py              # groupwise round-to-nearest W4/W3/W2
│   │   └── stats.py            # kurtosis, QQ vs N(0,1/d), outlier mass, IP-distortion vs 4^-b
│   ├── eval/                   # stubs in v0
│   │   ├── layer_swap.py       # LASER-style replacement + WikiText-103 perplexity
│   │   └── expert_error.py     # routed-activation capture; expert output relative L2
│   └── artifacts.py            # run dirs, config/git-SHA/seed capture, parquet metrics
├── experiments/                # p0_validate, a2_matched_param, a3_permutation_null,
│   │                           # b1_kernel_bench, c1_redundancy_census, d1_gaussianization, ...
│   └── plots/                  # figures read parquet; never refit to restyle
├── scripts/                    # vm_setup.sh, nsight_b1.sh (VM-side ncu wrapper)
├── tests/                      # Phase 0 validation gate, permanent
│   └── fixtures/               # SageMath BM-ALS agreement numbers (n=8,16)
├── docs/                       # design specs, D0 literature notes
└── results/                    # committed metrics/figures; never checkpoints
```

## Key contracts

**Decomposition protocol** (`decomp/base.py`): `fit(T, rank, **opts) -> FitResult` with `FitResult.factors`, `.reconstruct()`, `.param_count()`, `.loss_history`. `rank` is method-interpreted (int ℓ for BMD/CP/slice-SVD; tuple of multilinear ranks for Tucker variants) — cross-method comparisons align on `param_count()`, never on rank. `param_count()` is first-class because every Track A/C comparison is error-vs-parameters; the accounting comes from the fit object, never recomputed in plotting code. Methods self-register by name (`@register("bmd_rals")`) so experiments select methods from config strings; adding a method = implement protocol + register (framework extension point #1).

**BMD conventions** (`bmd_rals.py`): stack axis is mode 3 — a stack is `(n1, n2, h)` and slice k of a rank-ℓ BMD is `Σ_t diag(A[:,t,k]) · B[:,:,t] · diag(C[t,:,k])`: **B holds the ℓ shared templates, A/C the per-slice output/input gains.** Factor updates are the decoupled block least-squares (Tian–Kilmer Eqs. 6.5/6.7/6.9) as one batched `torch.linalg.lstsq` per factor; the cyclic transpose identity makes one update routine serve all three factors. Optional Tikhonov regularization. fp64 for validation, fp32 default for fits.

**Stack dataclass** (`stacks/base.py`): tensor + metadata `(model, layer, object_name, axes)`. All builders return it; downstream metrics are keyed by metadata so layers-as-replicates is the default reporting unit (framework extension point #2 — any future weight-object source implements a builder).

**Permutation null** (`stacks/null.py`): destroys cross-slice alignment while preserving per-slice spectra — seeded shuffle of slice order plus independent random orthogonal rotation of each slice. The rotations are the load-bearing part (a pure shuffle is absorbed into every method's slice-mode factor). Transform object is returned for logging/invertibility.

**Artifacts** (`artifacts.py`): `run_dir(experiment)` → `results/<exp>/<run-id>/` containing `config.json` (tyro dataclass dump), git SHA, seeds, package versions, and `metrics.parquet` keyed by `(layer, object, method, rank, params)`. Experiment-agnostic (framework extension point #3).

## Testing = Phase 0 gate

`tests/` is the BMD-ALS port validation, kept green permanently:

- Generative BM-rank-3 tensor recovered at relative error ~1e-3.
- Transpose-identity property test (`bmp(A,B,C)^T == bmp(B^T, C^T, A^T)`).
- SS-SVD init at zero ALS sweeps equals the per-slice truncated SVD baseline exactly.
- Parameter-count accounting unit tests per method.
- Agreement with SageMath BM-ALS fixtures at n=8,16 (user exports numbers once into `tests/fixtures/`).

All CPU-fast; no GPU in CI path.

## VM workflow

Transport is git: push branch → pull on VM → `scripts/vm_setup.sh` (uv sync + sanity) → run bench/Nsight → commit results parquet back. No rsync state. Triton arrives automatically with Linux torch; the Triton kernel module is import-guarded so the package imports cleanly where triton is absent.

## Error-handling philosophy

Research code fails fast: shape asserts at module boundaries, no silent dtype/device coercion, convergence histories always recorded, correctness asserts before any timing run.

## Build order (first implementation pass)

1. Scaffold: `uv init --lib`, python pin, deps, gitignore, pytest skeleton.
2. `decomp.bmd_rals` + `init` + `stacks.synthetic` + full Phase 0 test suite.
3. `bench.factored_matvec` + `harness` (correct on CPU, timed on VM).
4. `stacks.gpt2` + `decomp.baselines` + `a2`/`a3` experiments.
5. `quant` + `d1_gaussianization`. Track C modules stubbed with shaped signatures.

## Out of scope (v0)

- Track C implementation beyond stubs (gated on Track A/B outcomes per the research plan).
- KD recovery (C5) and any multi-node work.
- AWQ/GPTQ third-party integrations for D2 (decision deferred to when D1 results exist).
- CI infrastructure (local pytest only until the repo earns it).
