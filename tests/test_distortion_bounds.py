"""F1 distortion-vs-bit-width figure (TurboQuant Fig-3 parity).

Correctness anchor: TurboQuant Fig 3 plots ABSOLUTE distortions on UNIT-NORM
vectors (D_mse = ||x - x_hat||^2 for ||x||=1; D_prod = E|<y,x> - <y,x_hat>|^2),
NOT the relative Frobenius error. The measured turboquant_mse D_mse must land
near its own closed-form MSE bounds [4^-b, sqrt(3)*pi/2 * 4^-b] — if it does
not, the measurement is wrong.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from experiments.distortion_bounds import measure_distortions
from experiments.plots import plot_distortion_bounds


def test_measure_distortions_lands_within_mse_bounds():
    df = measure_distortions(
        source="sphere",
        arms=("turboquant_mse",),
        bit_list=(2, 3),
        d=128,
        n_vectors=256,
        seed=0,
    )
    assert set(df.columns) >= {"arm", "bitwidth", "d_mse", "d_prod"}

    for b in (2, 3):
        row = df[(df.arm == "turboquant_mse") & (df.bitwidth == b)]
        assert len(row) == 1, f"expected one row for turboquant_mse @ b={b}"
        d_mse = float(row["d_mse"].iloc[0])
        lower = 4.0 ** (-b)
        upper = math.sqrt(3) * math.pi / 2 * 4.0 ** (-b)
        # finite-sample tolerance around the closed-form bounds
        assert 0.5 * lower <= d_mse <= 2.0 * upper, (
            f"turboquant_mse d_mse={d_mse:.5g} @ b={b} outside "
            f"[{0.5 * lower:.5g}, {2.0 * upper:.5g}] "
            f"(bounds [{lower:.5g}, {upper:.5g}])"
        )


def test_distortion_bounds_figure_emitted(tmp_path):
    import pandas as pd

    df = pd.DataFrame(
        [
            {"arm": "turboquant_mse", "bitwidth": 2, "d_mse": 0.05, "d_prod": 4e-4},
            {"arm": "turboquant_mse", "bitwidth": 3, "d_mse": 0.012, "d_prod": 1e-4},
            {"arm": "turboquant_prod", "bitwidth": 2, "d_mse": 0.08, "d_prod": 3e-4},
            {"arm": "turboquant_prod", "bitwidth": 3, "d_mse": 0.02, "d_prod": 8e-5},
        ]
    )
    out = plot_distortion_bounds.make_figures(df, tmp_path)
    assert isinstance(out, list) and len(out) >= 1
    png = tmp_path / "distortion_vs_bitwidth.png"
    assert png in out
    assert png.exists()
    assert isinstance(png, Path)
