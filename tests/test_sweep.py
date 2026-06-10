import torch

from bmx.stacks.base import Stack
from bmx.sweep import decomp_sweep


def test_decomp_sweep_rows_and_keys():
    T = torch.randn(8, 8, 4, dtype=torch.float64)
    stack = Stack(T, model="test", layer=3, object_name="wqk", axes=("a", "b", "h"))
    plan = {"bmd_rals": [1, 2], "slice_svd": [2], "shared_tucker": [(4, 4)]}
    df = decomp_sweep(stack, plan, fit_opts={"bmd_rals": {"n_iters": 5}})
    assert len(df) == 4
    assert {
        "model",
        "layer",
        "object",
        "method",
        "rank",
        "params",
        "rel_error",
        "seconds",
    } <= set(df.columns)
    assert (df[df.method == "slice_svd"].params == 4 * 2 * (8 + 8)).all()
    assert (df.rel_error >= 0).all() and (df.rel_error <= 1.5).all()
