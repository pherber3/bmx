"""Timing harness for Track B. Correctness is asserted before anything is timed."""

import time
from dataclasses import asdict, dataclass

import pandas as pd
import torch

from bmx.bench.factored_matvec import (
    dense_from_factors,
    dense_slice_matvec,
    factored_matvec,
    factored_matvec_bmm,
    factored_matvec_compiled,
    templates_to_bmm_layout,
)
from bmx.decomp.ops import bmd_param_count
from bmx.stacks.synthetic import random_bmd_factors


@dataclass
class BenchCase:
    m: int
    p: int
    h: int
    ell: int
    batch: int
    impl: str  # dense | eager | compiled | bmm
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

        # Correctness gate before timing. Norm-based, not elementwise: cuBLAS
        # and einsum reduce in different orders, so single elements can show
        # ~1e-4 cancellation noise in fp32 while the kernels agree. A real
        # indexing bug produces O(1) relative error, far above these gates.
        y_ref = dense_slice_matvec(W, x)
        rel = ((factored_matvec(A, B, C, x) - y_ref).norm() / y_ref.norm()).item()
        tol = {torch.float64: 1e-10, torch.float32: 1e-4}.get(dtype, 1e-2)
        assert rel < tol, (
            f"factored vs dense mismatch: rel Frobenius {rel:.3e} (tol {tol})"
        )

        if case.impl == "dense":
            fn, args = dense_slice_matvec, (W, x)
        elif case.impl == "eager":
            fn, args = factored_matvec, (A, B, C, x)
        elif case.impl == "compiled":
            fn, args = factored_matvec_compiled, (A, B, C, x)
        elif case.impl == "bmm":
            # Template relayout happens once, outside timing: a deployment
            # stores templates in bmm order to begin with.
            Bt = templates_to_bmm_layout(B)
            rel_bmm = (
                (factored_matvec_bmm(A, Bt, C, x) - y_ref).norm() / y_ref.norm()
            ).item()
            assert rel_bmm < tol, f"bmm vs dense mismatch: {rel_bmm:.3e}"
            fn, args = factored_matvec_bmm, (A, Bt, C, x)
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
                    bmd_param_count(case.m, case.p, case.h, case.ell) * esize
                ),
                "flops": (
                    dense_flops if case.impl == "dense" else case.ell * dense_flops
                ),
            }
        )
        if device.startswith("cuda"):
            # ~500 shape variants in a full sweep; without this the allocator
            # fragments across cases and large late cases OOM.
            del A, B, C, x, W, y_ref
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)
