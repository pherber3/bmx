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

Expected suite status: **68 passed, 1 xfailed**. The xfail is intentional
(see Research state). The SageMath agreement fixture is generated, not
hand-written — regenerate with
`uv run python scripts/export_sagemath_fixture.py` (reads
`D:\Projects\Hypermatrix ML`; self-validates BM-product conventions before
writing).

**Research state (start here if continuing the science):** the BM program is
concluded (all entries discarded/re-scoped), and **Avenue 1 (low-rank+sparse
quantization residual) is now also CLOSED** — built, measured, honest negative;
see `docs/2026-06-11-lrs-results.md` for the verdict, then
`docs/2026-06-11-frontier-breakeven.md`: the scale loophole is resolved —
transform weights hug the Shannon break-even line at 768→8192 (stable rank
grows with width at the canceling rate); the only payers are table-like
objects (wpe, routers, layer-0 rogue-channel readers), all axis-aligned or
negligible. **Current program: KV-cache compression** — `docs/2026-06-11-kv-research-plan.md`.
K1 (census): E1 confirmed, cache carries the structure weights lack; RoPE costs
~1–1.5 bits of key compressibility (`docs/2026-06-11-k1-census-results.md`).
K2 (codec bake-off): **low-rank on pre-RoPE keys wins at matched bits (2–3×
lower logit distortion); values want turboquant-style per-token Lloyd; KIVI's
key/value asymmetry reproduced; TurboQuant's bounds replicate but worst-case
optimality concedes to structure** (`docs/2026-06-12-k2-arms-results.md`).
K2b (perplexity round): **G2b closed positive — deployable recipe: keys
pre-RoPE lowrank+per-channel @3b, values rotate+Lloyd @2b ⇒ ~3.0 bpe avg,
+0.5% ppl, 5.3× KV memory vs fp16; bits belong to K (2× more sensitive);
unbiased coding dominated everywhere (K3 largely answered: bias is cheap,
variance is expensive)** (`docs/2026-06-12-k2b-ppl-results.md`). K2c: **streaming validated — prefill-frozen pre-RoPE subspaces generalize
(eps_ratio 0.94, drift-flat to 2k tokens; post-RoPE drifts monotonically =
the RoPE mechanism a 4th way); rank 16 is the validated streaming point**
(`docs/2026-06-12-k2c-results.md`). Remaining: engineering only —
quantize-on-append cache class, 32k re-check, fused dequant-attention
kernel (Track B byte model predicts it first). The machinery below is the
substrate (registry, sweep, quant/arms, cache, artifacts).

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
matvec + timing harness (correctness asserted before timing); the `bmm` impl
(pre-transposed templates) is the one that realizes the byte win at ell>=2.
`src/bmx/quant/` — rotations (`hadamard`), groupwise RTN (`rtn`), distribution
stats (`stats`: kurtosis, outlier_mass, ip_distortion, sq_floor). `src/bmx/census.py`
— pairwise expert-similarity metrics (cos/CKA/subspace + participation ratio).
`src/bmx/sweep.py` — the shared matched-parameter sweep engine; rows keyed by
(model, layer, object, method, rank, params, solver, null_seed) so
layers-as-replicates is the default. `src/bmx/artifacts.py` —
`results/<exp>/<run-id>/` with config + env + git SHA + `metrics.parquet`.
Experiments in `experiments/` are thin tyro scripts; figures read parquet,
never refit. Commit metrics/figures, never checkpoints.

## Research state — BM program CONCLUDED (2026-06-10 H100 session)

Full results: `docs/2026-06-10-h100-session-results.md`. Forward avenues:
`docs/next-avenues-structured-residual.md`. One-line verdict per track:

- **Track A — entry 1 DISCARDED.** 12-layer a2/a3 sweeps: BMD worst at matched
  params; real-vs-null gap ≈0 for BMD while Tucker keeps 0.06–0.10 → attention
  structure is **subspace-shaped, not template-shaped**.
- **Track B — mechanism CONFIRMED, kernel-limited.** ncu DRAM counters match
  the byte model exactly (dense h·m·p vs factored ell·m·p, 32× at h=64 ell=2);
  wall-clock speedup tracks bytes at batch 1, decays with batch; ~9× headroom
  for a fused kernel. Reusable result if a template-shaped object is ever found.
- **Track C — entry 2 FALSIFIED on OLMoE.** C1 census (3 checkpoints): experts
  orthogonal-as-vectors but share ~10 global second-moment modes (global, not
  clustered). C2 discriminator: that structure is too thin — BMD never separates
  from Tucker/slice-SVD; at ell=E/8, RE ≈ 0.87.
- **Track D — D1 strong, entry 3 RE-SCOPED.** Rotation crushes GPT-2 kurtosis
  (median +2.0 / max +47.9 → ≈0). D0 lit pass: the Gaussianization + 4^-b floor
  theory is already published (NestQuant, Ordentlich–Polyanskiy); the open edge
  is unbiased weight matvecs + structured residuals → **Avenue 1**.
- **Avenue 1 (L+S residual) — CLOSED 2026-06-11, honest negative.** Stage A:
  the structure exists (subspace overlap 0.91–0.99; supp(Ŝ) hits d1's channels,
  top-10 overlap 1.0 on c_proj; residual kurtosis → ≈0) but is thin (L+S
  captures ~21% of energy at r=32+0.1%). Stage B: at honestly-accounted total
  bits every L+S point sits above the interpolated rotate-RTN rate–distortion
  curve — additive fp16 side-information is the wrong currency at GPT-2 dims.
  Confirmed twice more: rotation is free win; rotating the post-extraction
  residual helps only while it's still non-Gaussian and hurts once extraction
  Gaussianizes it (dose–response in frac). `docs/2026-06-11-lrs-results.md`.
- **Solver is sound (not the cause of any negative):** plain RALS swamps at
  RE ≈ 0.21 only on random-dense synthetics; near-truth inits hit <1e-8, and on
  the BM-ALS paper's own 10 tensors RALS beats the paper's solver by 3–10 orders
  of magnitude (`tests/test_sagemath_agreement.py`). Kept visible as the xfailed
  `test_cold_start_recovery`.

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
