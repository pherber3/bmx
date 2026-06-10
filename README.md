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

## NVIDIA VM workflow (Track B authoritative numbers)

1. Push your branch; on the VM: `git clone <repo> && cd bmx && scripts/vm_setup.sh`
2. `scripts/nsight_b1.sh` (wraps `experiments/b1_kernel_bench.py --device cuda` in ncu)
3. `git add results/ && git commit && git push` — metrics come home as parquet + csv

## SageMath fixtures

Export per `tests/fixtures/README.md` to activate the agreement test.
