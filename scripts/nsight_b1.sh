#!/usr/bin/env bash
# Track B authoritative measurement: bytes from DRAM, achieved bandwidth, and
# L2 behavior via Nsight Compute, around the b1 kernel bench. NVIDIA VM only.
#
# Metric set follows the roofline recipe in the AI Systems Performance
# Engineering textbook (ncu DRAM + LTS/L2 pct-of-peak + occupancy): dram bytes
# decide whether the factored kernel actually reads ~ell/h of the dense bytes;
# the lts (L2) counters diagnose template residency/self-eviction (templates
# exceed H100's 50MB L2 at d=4096 ell>=2, so a slice-looped kernel would show
# near-peak DRAM through nominally cached reads).
#
# Run on a SMALL case set (a handful of shapes): ncu replays every kernel,
# so profiling the full grid takes hours and adds nothing.
# Usage: scripts/nsight_b1.sh --d 4096 --h 64 --ell 2 --batch 1 [more b1 args]
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="results/b1_kernel_bench/ncu-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active \
    --csv --log-file "$OUT/ncu.csv" \
    uv run python experiments/b1_kernel_bench.py \
      --device cuda --iters 5 --warmup 2 "$@"

echo "ncu metrics -> $OUT/ncu.csv (commit this with the run's parquet)"
