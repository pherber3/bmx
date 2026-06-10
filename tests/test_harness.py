import pandas as pd

from bmx.bench.harness import BenchCase, run_cases


def test_run_cases_smoke_cpu():
    cases = [
        BenchCase(m=32, p=32, h=4, ell=2, batch=2, impl="dense"),
        BenchCase(m=32, p=32, h=4, ell=2, batch=2, impl="eager"),
    ]
    df = run_cases(cases, device="cpu", warmup=1, iters=3)
    assert isinstance(df, pd.DataFrame) and len(df) == 2
    row = df[df.impl == "eager"].iloc[0]
    assert row.ms > 0
    assert row.model_bytes_factored < row.model_bytes_dense
    assert {"m", "p", "h", "ell", "batch", "impl", "ms", "flops"} <= set(df.columns)
