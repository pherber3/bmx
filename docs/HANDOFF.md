# Handoff — bmx, end of 2026-06-10

> **UPDATE 2026-06-11:** the Avenue 1 build described below was executed the next day
> (plan: `docs/superpowers/plans/2026-06-11-lrs-residual.md`) and **closed with an
> honest negative** — structure confirmed, bit-economics against it. Verdict, scope
> conditions, and what survives: `docs/2026-06-11-lrs-results.md`. All infrastructure
> items below now exist (`decomp/lrs.py`, `quant/arms.py`, `eval/layer_swap.py`
> implemented + offline-tested; suite 68 passed, 1 xfailed). Remaining live leads are
> Avenue 2 and the unbiased-matvec cluster. The build-steps section is kept for the
> record of what was planned; it is DONE, not a to-do.

State at handoff for a fresh session. The original BM/hypermatrix research program is
**complete and concluded** (all three speculative entries discarded or re-scoped with
measured reasons). The framework is healthy and the next work is a new direction:
**structured-residual (low-rank + sparse) quantization**, Avenue 1 of
`docs/next-avenues-structured-residual.md`.

## Read these first (in order)
1. `CLAUDE.md` — conventions, hard rules, commands, research state.
2. `docs/2026-06-10-h100-session-results.md` — what the BM program found and why each
   entry was killed (A4 gate, Track B bytes, C1/C2 MoE, D0/D1).
3. `docs/next-avenues-structured-residual.md` — the three forward avenues with foundational
   grounding; **Avenue 1 is the recommended next build.**
4. `docs/d0-literature-notes.md` — what quantization theory is already published (so we
   don't re-derive it).

## Repo health
- `uv run pytest -q` → **53 passed, 1 xfailed** (the xfail is the intentional cold-start
  ALS swamp; documented). `uv run ruff check .` clean.
- Latest commit on `main` at handoff: the C1/C2 results + this avenues writeup. Remote:
  github.com/pherber3/bmx (private).
- The Lambda H100 instance from this session is **terminated** — no live cloud state.

## Hard rules (do not violate — see CLAUDE.md for full list)
- **Never `git commit` without the user's explicit approval. No AI-attribution trailers.**
- Before any commit: `uv run ruff format .` → `ruff check` → `pytest` → re-stage.
- Deps only via `uv add`. Bash (git bash), not PowerShell. `cd /d/Projects/bmx` in fresh shells.

## What exists and is reusable for Avenue 1
The framework is deliberately built so a new decomposition is a drop-in:
- `src/bmx/decomp/base.py` — `Decomposition` protocol + `@register(name)` registry. A new
  method (e.g. low-rank+sparse) implements `fit → FitResult` with `reconstruct()` and
  `param_count()`, registers, and immediately works in the sweep engine.
- `src/bmx/quant/` — `rtn` (groupwise RTN), `hadamard` (rotation/orthogonalize), `stats`
  (kurtosis, outlier_mass, **ip_distortion**, sq_floor). Avenue 1's residual quantization
  and metrics are already here.
- `src/bmx/sweep.py` — matched-parameter sweep; rows keyed by
  (model, layer, object, method, rank, params, solver). The comparison engine for "does
  structured-residual beat plain RTN/AWQ at matched bits."
- `src/bmx/stacks/gpt2.py` (`stack_by_name`), `stacks/moe.py` (lazy expert loading),
  `artifacts.py` (run dirs + parquet), `experiments/plots/` (read parquet, never refit).
- `src/bmx/eval/layer_swap.py` — **stub** (NotImplementedError); implementing it (LASER-style
  weight replacement + WikiText perplexity) is the functional-eval step Avenue 1 step 4 needs.

## Avenue 1 first build (concrete — read the corrected design in
## `docs/next-avenues-structured-residual.md` Avenue 1 first; it is grounded against
## the foundational texts and supersedes any earlier sketch)
1. New module `src/bmx/decomp/lrs.py`: the two-step estimator — **hard**-threshold
   `Ŝ = T_ν(W)` with `T_ν(v) = v·𝟙[|v| > ν]` (Wainwright §11.4.2 Eq. 11.58 — NOT
   soft-thresholding), then `L̂ = truncated-SVD(W − Ŝ)`, optionally alternated.
   Register as `"lrs"`. Unit-test on planted L+S tensors (recovery of planted support
   and rank). The convex program (10.53) is a later referee, not the first build.
2. Experiment `experiments/lrs_residual.py`, two stages **in this order**:
   - **Stage A diagnostic** (one matrix, original basis — NOT post-rotation; rotation
     provably spreads the sparse mass L+S needs concentrated): does rank(L̂) match a2's
     Tucker subspace, does supp(Ŝ) match d1's flagged channels, does spikiness
     `‖L̂‖_max ≤ α/√(d₁d₂)` hold? Three assumptions, one estimator call.
   - **Stage B compression**: fit on clean W, quantize `R = W − L̂ − Ŝ` via `quant.rtn`,
     compare at **matched total bits (count L̂/Ŝ storage)** against plain RTN,
     rotate-then-RTN, and L+S-then-rotate-residual. Metrics: `ip_distortion`, kurtosis
     of R, then layer-swap perplexity.
3. If it clears: implement `eval/layer_swap.py`, measure perplexity delta; then the
   fused-kernel byte accounting (needs a VM again — same workflow as this session,
   `scripts/vm_setup.sh` + the bench harness; reuse the bmm/fused-matvec path in
   `bmx.bench.factored_matvec`). Stage-3 extension hook: unbiased QJL on the quant
   residual (TurboQuant's composition) ties into the D0-surviving (c)-cluster.

## Loose ends (none blocking)
- **Vault `.speculative` status updates are written and user-approved** (entry 1 → discarded,
  entry 2 → discarded, entry 3 → re-scoped, with dated evidence-citing verdicts in
  `D:\Projects\personal-brain\wiki\.speculative\2026-06-09 *.md`) but the **vault repo
  commit is still pending** — the user commits in that repo themselves.
- `docs/friday-brief-tensor-compression.md` — a meeting brief, not research; ignore for
  implementation.
- The unweight/Omni Cloudflare notes in the vault are good context for the
  fused-dequant-kernel framing if Avenue 1 reaches the kernel stage.
