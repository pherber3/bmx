# CLAUDE.md — bmx

Research framework for testing whether Bhattacharya–Mesner (BM / hypermatrix)
decomposition compresses LLM weights usefully — bandwidth-amplified decode,
MoE expert streaming, and rotation-based quantization theory. This is
**kill-or-confirm research code**: experiments exist to close gates, and an
honest negative is a valid result. Don't polish numbers; report them.

Read before substantial work:
- `docs/superpowers/specs/2026-06-10-bmx-framework-design.md` — architecture + contracts
- `docs/superpowers/plans/2026-06-10-bmx-first-pass.md` — what was built and why (post-plan notes list intentional deviations)
- The three hypotheses live in the personal-brain vault:
  `D:\Projects\personal-brain\wiki\.speculative\2026-06-09 *.md` (diag-template
  matvec, BMD expert streaming, VQ theory transfer). Theory anchors are wiki
  notes there ([[BM-Decomposition Model]], [[BMD-ALS Algorithm]],
  [[Interaction Tensor as Pairwise-Coupled Structural Object]], etc.) — use the
  `personal-brain` skill / `mcp__wiki__*` tools when theory questions come up.

## Hard rules

- **NEVER `git commit` without the user's explicit approval.** Stage, propose a
  message, stop. No "Co-Authored-By" or any AI attribution, ever.
- Before any commit: `uv run ruff format .` then `uv run ruff check .` then
  `uv run pytest -q` — all clean, then re-stage.
- Dependencies only via `uv add` / `uv add --dev`. Never hand-edit versions in
  `pyproject.toml`.
- Use the Bash tool (git bash), not PowerShell. The shell cwd resets between
  turns — `cd /d/Projects/bmx` first in fresh shells.
- This machine has an AMD 7900 XTX (no CUDA). GPU-authoritative work (Track B
  Nsight numbers, big sweeps) runs on a rented NVIDIA VM via `scripts/`
  (transport is git: push → pull on VM → run → commit parquet back).

## Commands

```bash
uv run pytest -q                      # full suite ≈ 30 s; this IS the Phase 0 gate
uv run pytest tests/test_bmd_rals.py -v   # solver-only
uv run python experiments/a2_matched_param.py --help   # tyro CLIs; tuples are space-separated
uv run python experiments/d1_gaussianization.py        # cheap (~30 s + GPT-2 download)
```

Expected suite status: **46 passed, 1 xfailed**. The xfail is intentional
(see Research state). The SageMath agreement fixture is generated, not
hand-written — regenerate with
`uv run python scripts/export_sagemath_fixture.py` (reads
`D:\Projects\Hypermatrix ML`; self-validates BM-product conventions before
writing).

## The math conventions (memorize; everything assumes them)

- Stack tensor `T : (n1, n2, h)` — **slice/stack axis is mode 3**, slices `T[:, :, k]`.
- Factors: `A (n1, ell, h)` output gains · `B (n1, n2, ell)` **shared templates** · `C (ell, n2, h)` input gains.
- `bmp(A,B,C)[i,j,k] = Σ_t A[i,t,k]·B[i,j,t]·C[t,j,k]`, i.e. slice k =
  `Σ_t diag(A[:,t,k]) @ B[:,:,t] @ diag(C[t,:,k])` — the diag-template reading.
- `cyclic_transpose = permute(1,2,0)`, order 3; identity
  `cyc(bmp(A,B,C)) = bmp(cyc(B), cyc(C), cyc(A))` drives all three RALS updates
  through one middle-slot solver (`decomp/bmd_rals.py`).
- **Cross-method comparisons align on `param_count()`, never on rank.** Rank is
  method-interpreted (int for BMD/CP/slice-SVD, tuple for Tucker variants).
- dtype: fp64 in tests, fp32 in experiments. Fail fast: shape asserts at
  boundaries, no silent coercion.

## Architecture (one line each)

`src/bmx/decomp/` — methods behind `@register(name)` → `FitResult` (importing
`bmx.decomp` registers all: bmd_rals, slice_svd, cp, tucker, shared_tucker).
`src/bmx/stacks/` — weight-stack builders returning `Stack` (tensor +
metadata); `gpt2.stack_by_name` is the name→object dispatch; `null.py` is the
A3 control (seeded slice shuffle + per-slice orthogonal rotations — the
rotations are the load-bearing part). `src/bmx/bench/` — Track B factored
matvec + timing harness (correctness asserted before timing). `src/bmx/quant/`
— Track D rotations/RTN/stats. `src/bmx/sweep.py` — the shared
matched-parameter sweep engine; rows keyed by (model, layer, object, method,
rank, params, solver, null_seed) so layers-as-replicates is the default.
`src/bmx/artifacts.py` — `results/<exp>/<run-id>/` with config + env + git SHA
+ `metrics.parquet`. Experiments in `experiments/` are thin tyro scripts;
figures read parquet, never refit. Commit metrics/figures, never checkpoints.

## Research state & gates (as of 2026-06)

- **Track A** (is attention-stack structure template-shaped?): a2/a3 ready,
  not yet run at scale. Gate A4: CP≈Tucker floor with BMD below it *on real
  stacks but not on the a3 null* → template-shaped. BMD advantage surviving
  the null = expressivity artifact → hypothesis dead.
- **Track B** (does the h/ℓ bandwidth win materialize?): b1 + Nsight scripts
  ready; needs the NVIDIA VM. Report the curve over batch, not a point.
- **Track C** (MoE expert streaming): gated on C1 redundancy census; stubs in
  `stacks/moe.py`, `eval/expert_error.py` raise NotImplementedError with the
  gate named.
- **Track D** (VQ theory transfer): d1 ready and cheap. Early smoke result:
  rotation crushes GPT-2 weight kurtosis (+30…+48 → ≈0) — entry 3's mechanism
  survives trained weights.
- **Known empirical fact, not a bug:** plain RALS cold-starts (SS-SVD or
  random) swamp at RE ≈ 0.21 on random-dense-factor BM-rank-2 synthetics;
  near-truth inits converge below 1e-8 (update equations verified exact).
  Kept visible as the xfailed `test_cold_start_recovery`.
- **But the swamp is specific to that synthetic family.** On the 10 real
  interaction tensors from the user's BM-ALS paper (Hypermatrix ML repo,
  n=4/8, estimated from dynamical systems), cold-start RALS reaches
  1e-13…1e-5 RE — beating the paper's greedy-deflation SageMath BM-ALS
  (2.7e-4…9.8e-3) by 3-10 orders of magnitude on every case
  (`tests/test_sagemath_agreement.py`, fixture committed). Two consequences:
  (a) Track A fits on real weight stacks are less optimizer-limited than the
  swamp finding alone suggested — keep the perturbation-init disambiguation
  protocol anyway; (b) the paper's reported BM-ALS numbers understate BMD on
  its own benchmark — RALS would strengthen those results.

## Pitfalls already hit (don't rediscover)

- tensorly 0.9 `partial_tucker` returns `(decomposition, errors)` — unpack
  `result[0]`. Handled in `baselines.py`; beware on version bumps.
- `torch.linalg.lstsq` on CUDA uses the full-rank 'gels' driver → garbage on
  rank-deficient blocks. `fit_bmd_rals` resolves a solver policy per device
  and records it as `fit.solver` (lands in metrics — check it when CPU and VM
  numbers disagree).
- torch QR leaves sign ambiguity; all orthogonal sampling goes through
  `quant.hadamard.orthogonalize` (sign-canonicalized). Don't hand-roll QR.
- CP-ALS at rank 512 on (768,768,12) is minutes-per-fit on CPU; a full a2
  sweep is an overnight job locally. Trim `--layers` or PLAN for quick looks;
  use `bmd_check_every` (already defaulted in a2) for ALS speed.
- `torch.compile` on Windows CPU is unreliable — the "compiled" bench impl is
  VM-only in practice; never required by tests.
- A slice-order shuffle alone is a no-op control for every method here
  (absorbed into the slice-mode factor); the per-slice rotations are what make
  the a3 null real.
