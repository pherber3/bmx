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
