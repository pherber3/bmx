"""Track B: price the diag-template factored matvec against dense per-slice GEMV.

Local (CPU): correctness + rough numbers. Authoritative numbers: NVIDIA VM,
see scripts/nsight_b1.sh. Prediction under test: wall-time ratio -> h/ell in the
memory-bound regime; report the curve over batch, not a point.
"""

import dataclasses
import itertools

import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.bench.harness import BenchCase, run_cases


@dataclasses.dataclass
class Config:
    d: tuple[int, ...] = (768, 2048, 4096)
    h: tuple[int, ...] = (8, 12, 32, 64)
    ell: tuple[int, ...] = (1, 2, 4, 8)
    batch: tuple[int, ...] = (1, 4, 16, 32)
    impls: tuple[str, ...] = ("dense", "eager", "compiled", "bmm")
    dtype: str = "float32"
    device: str = "cpu"
    warmup: int = 10
    iters: int = 50


def main(cfg: Config) -> None:
    cases = [
        BenchCase(m=d, p=d, h=h, ell=ell, batch=b, impl=impl, dtype=cfg.dtype)
        for d, h, ell, b, impl in itertools.product(
            cfg.d, cfg.h, cfg.ell, cfg.batch, cfg.impls
        )
        if ell < h  # the claim only matters when templates < slices
    ]
    run = create_run("b1_kernel_bench", cfg)
    df = run_cases(cases, device=cfg.device, warmup=cfg.warmup, iters=cfg.iters)
    write_metrics(run, df)
    print(f"{len(df)} cases -> {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
