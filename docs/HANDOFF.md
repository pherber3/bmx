# Handoff — bmx, end of 2026-06-10

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

## Avenue 1 first build (concrete)
Per `docs/next-avenues-structured-residual.md`:
1. New module `src/bmx/decomp/lrs.py`: low-rank+sparse fit. Start with the cheap two-step
   estimator (soft-threshold S, residual truncated-SVD L — Wainwright §11.4.2 / Prop 11.19),
   not the full convex program. Register as `"lrs"`. Unit-test on planted L+S tensors.
2. Experiment `experiments/lr_sparse_residual.py`: on GPT-2 / Llama-1B matrices, fit L+S at
   several (rank, nnz) budgets; **first check the spikiness condition `‖L‖_max ≤ α/√(d₁d₂)`
   empirically — that's the go/no-go**; then quantize the residual with `quant.rtn` and
   measure `ip_distortion` + kurtosis vs quantizing W directly, at matched total bits.
3. If it clears: implement `eval/layer_swap.py`, measure perplexity delta; then the
   fused-kernel byte accounting (needs a VM again — same workflow as this session,
   `scripts/vm_setup.sh` + the bench harness; reuse the bmm/fused-matvec path in
   `bmx.bench.factored_matvec`).

## Loose ends (none blocking)
- **Vault `.speculative` status updates not yet written** (user deferred). When the user
  approves: entry 1 → discarded (subspace-shaped, real-vs-null gap ≈0); entry 2 → discarded
  (C2: expert structure too thin, falsified on OLMoE); entry 3 → re-scoped to the
  unbiased-matvec / structured-residual cluster (a/b already published per D0). These are
  edits to `D:\Projects\personal-brain\wiki\.speculative\2026-06-09 *.md` and must be done
  with user approval per vault rules.
- `docs/friday-brief-tensor-compression.md` — a meeting brief, not research; ignore for
  implementation.
- The unweight/Omni Cloudflare notes in the vault are good context for the
  fused-dequant-kernel framing if Avenue 1 reaches the kernel stage.
