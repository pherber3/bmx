# bmx

Research framework for structured weight compression of LLMs, originally built to test
whether Bhattacharya–Mesner (hypermatrix) algebra gives a bandwidth-amplifying weight
decomposition. **That program is concluded** (the diag-template prior does not describe
trained weights — see `docs/2026-06-10-h100-session-results.md`); the framework now
carries the follow-on work on **structured-residual quantization**
(`docs/next-avenues-structured-residual.md`).

**New here?** Read `docs/HANDOFF.md` first, then `CLAUDE.md`. The framework — a
registry of matched-parameter decomposition methods, weight-stack builders, a Track-B
GPU kernel bench, quantization utilities, and a reproducible artifact harness — is
reusable for any "compress this weight object, measure error vs parameters/bytes" task.

## Quickstart

    uv sync
    uv run pytest -q                 # 53 passed, 1 xfailed (intentional)
    uv run python experiments/a2_matched_param.py --help

## Layout

- `src/bmx/` — framework: decomp methods (registry), stacks, bench, quant, census, eval, artifacts
- `experiments/` — thin scripts per research item (a2/a3 attention, b1 kernel, c1/c2 MoE, d1 quant)
- `results/` — committed metrics/figures (config + git SHA captured per run)
- `scripts/` — NVIDIA-VM workflow (setup, Nsight wrappers), SageMath fixture exporter
- `tests/` — validation suite (BM solver gate + module unit tests)
- `docs/` — `HANDOFF.md`, session results, forward avenues, D0 lit notes, design spec

## NVIDIA VM workflow (Track B authoritative numbers)

1. Push your branch; on the VM: `git clone <repo> && cd bmx && scripts/vm_setup.sh`
2. `scripts/nsight_b1.sh` (wraps `experiments/b1_kernel_bench.py --device cuda` in ncu)
3. `git add results/ && git commit && git push` — metrics come home as parquet + csv

## SageMath fixtures

Export per `tests/fixtures/README.md` to activate the agreement test.
