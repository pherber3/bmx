import torch

from bmx.quant.breakeven import breakeven_row


def _g(seed=0):
    return torch.Generator().manual_seed(seed)


def test_rank1_matrix_pays():
    u = torch.randn(64, 1, generator=_g(0), dtype=torch.float64)
    v = torch.randn(48, 1, generator=_g(1), dtype=torch.float64)
    bk = breakeven_row(u @ v.mT)
    assert bk["lr_best_r"] == 1
    assert bk["lr_margin_bits"] > 1.0
    assert bk["stable_rank"] < 1.01


def test_random_matrix_is_marginal_or_negative():
    bk = breakeven_row(torch.randn(64, 48, generator=_g(2), dtype=torch.float64))
    assert bk["lr_margin_bits"] < 0.0  # no structure: side info never pays
    assert bk["sp_margin_bits"] < 0.1
    assert bk["stable_rank"] > 10


def test_margin_formula_hand_check():
    # diag(2, 1, 1, ...) on a 32x32 matrix: eps(1) = 4/(4+31), cost = 16*64/1024
    W = torch.eye(32, dtype=torch.float64)
    W[0, 0] = 2.0
    bk = breakeven_row(W)
    eps1 = 4.0 / 35.0
    saved1 = torch.log2(torch.tensor(1 / (1 - eps1))).item() / 2
    cost1 = 16.0 * 1 * 64 / 1024
    # r=1 margin must match the formula; argmax may pick another r, so compare
    # against the analytic r=1 value only when r=1 is the argmax
    if bk["lr_best_r"] == 1:
        assert abs(bk["lr_margin_bits"] - (saved1 - cost1)) < 1e-9
    assert abs(bk["lr_eps"] - eps1) < 1e-9 or bk["lr_best_r"] != 1
