"""Export the SageMath BM-ALS agreement fixture from the Hypermatrix ML repo.

Reads the per-seed serialized tensors and BM-ALS results
(stage1/data/reviewer/multi_seed_n{4,8,16}/seed*/) and writes
tests/fixtures/sagemath_bmals.json in the format test_sagemath_agreement.py
expects: {"cases": [{"name", "ell", "tensor", "sage_rel_error"}]}.

Self-validation: the source repo's BM rank-1 term is
R[i,j,k] = U[i,k] * V[i,j] * W[j,k] (Gnang / Tian-Kilmer standard, identical
to bmx's). For every case we rebuild the SageMath factors in bmx's (A, B, C)
layout, reconstruct with bmx's own bmp, and require the resulting relative
error to match the stored final_error — a convention mismatch fails loudly
instead of producing an apples-to-oranges fixture.

Usage (from the bmx repo root):
    uv run python scripts/export_sagemath_fixture.py \
        --source "D:/Projects/Hypermatrix ML"
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from bmx.decomp.ops import bmp


def case_from_seed_dir(seed_dir: Path, n: int, seed: int) -> dict:
    A_path = seed_dir / f"A_n{n}_seed{seed}.npy"
    npz_path = seed_dir / f"bm_als_n{n}_seed{seed}.npz"
    T = np.load(A_path)
    bm = np.load(npz_path)

    eigs = bm["eigenvalues"]
    U, V, W = bm["U_factors"], bm["V_factors"], bm["W_factors"]
    ell = int(bm["rank"])
    sage_re = float(bm["final_error"])

    # Rebuild in bmx layout: A[i,t,k] = eig_t * U_t[i,k]; B[i,j,t] = V_t[i,j];
    # C[t,j,k] = W_t[j,k]  (component axis t = their list index r).
    A_ours = torch.from_numpy((eigs[:, None, None] * U).transpose(1, 0, 2)).double()
    B_ours = torch.from_numpy(V.transpose(1, 2, 0)).double()
    C_ours = torch.from_numpy(np.ascontiguousarray(W)).double()
    T_t = torch.from_numpy(T).double()

    re_check = (
        torch.linalg.norm(bmp(A_ours, B_ours, C_ours) - T_t) / torch.linalg.norm(T_t)
    ).item()
    if not math.isclose(re_check, sage_re, rel_tol=1e-5, abs_tol=1e-9):
        raise SystemExit(
            f"convention mismatch for {npz_path.name}: bmx reconstruction of the "
            f"SageMath factors gives RE {re_check:.6e}, stored final_error is "
            f"{sage_re:.6e}. Do not write a fixture from mismatched conventions."
        )

    return {
        "name": f"n{n}_seed{seed}_ell{ell}",
        "ell": ell,
        "tensor": T.tolist(),  # axes (i, j, k); bmx stacks on mode 3
        "sage_rel_error": sage_re,
        "convention_check_rel_error": re_check,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="D:/Projects/Hypermatrix ML")
    parser.add_argument(
        "--out",
        default=str(
            Path(__file__).parent.parent / "tests/fixtures/sagemath_bmals.json"
        ),
    )
    args = parser.parse_args()

    source = Path(args.source)
    cases = []
    for n_dir in sorted(source.glob("stage1/data/reviewer/multi_seed_n*")):
        n = int(n_dir.name.removeprefix("multi_seed_n"))
        for seed_dir in sorted(n_dir.glob("seed*")):
            seed = int(seed_dir.name.removeprefix("seed"))
            cases.append(case_from_seed_dir(seed_dir, n, seed))
            print(
                f"  {cases[-1]['name']}: sage RE {cases[-1]['sage_rel_error']:.4e} "
                f"(convention check {cases[-1]['convention_check_rel_error']:.4e})"
            )

    if not cases:
        raise SystemExit(f"no multi_seed data found under {source}")

    out = Path(args.out)
    out.write_text(json.dumps({"source": str(source), "cases": cases}, indent=1))
    print(f"{len(cases)} cases -> {out}")


if __name__ == "__main__":
    main()
