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
                    case.ell * case.m * case.p + case.ell * (case.m + case.p) * case.h
                )
                * esize,
                "flops": (
                    dense_flops if case.impl == "dense" else case.ell * dense_flops
                ),
            }
        )
    return pd.DataFrame(rows)
