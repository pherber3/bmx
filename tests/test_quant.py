import scipy.linalg
import torch

from bmx.quant.hadamard import fwht, random_orthogonal, randomized_hadamard
from bmx.quant.rtn import rtn_quantize
from bmx.quant.stats import ip_distortion, kurtosis, outlier_mass, sq_floor


def test_fwht_matches_scipy_hadamard():
    d = 16
    H = torch.tensor(scipy.linalg.hadamard(d), dtype=torch.float64) / d**0.5
    X = torch.eye(d, dtype=torch.float64)
    torch.testing.assert_close(fwht(X), H)  # rows of identity -> rows of H


def test_fwht_is_involution_and_isometry():
    x = torch.randn(3, 32, dtype=torch.float64)
    torch.testing.assert_close(fwht(fwht(x)), x)
    torch.testing.assert_close(x.norm(dim=-1), fwht(x).norm(dim=-1))


def test_randomized_hadamard_and_orthogonal_are_isometries():
    x = torch.randn(5, 64, dtype=torch.float64)
    y = randomized_hadamard(x, seed=0)
    assert not torch.allclose(y, fwht(x))
    torch.testing.assert_close(x.norm(dim=-1), y.norm(dim=-1))
    Q = random_orthogonal(48, seed=0, dtype=torch.float64)
    torch.testing.assert_close(Q @ Q.T, torch.eye(48, dtype=torch.float64))


def test_rtn_error_decreases_with_bits():
    W = torch.randn(16, 128, dtype=torch.float64)
    errs = [
        (rtn_quantize(W, bits=b, group_size=32) - W).norm() / W.norm()
        for b in (2, 3, 4, 8)
    ]
    assert errs[0] > errs[1] > errs[2] > errs[3]
    assert errs[3] < 0.01


def test_gaussianization_kurtosis_drops():
    """Heavy-tailed rows become near-Gaussian under random rotation."""
    torch.manual_seed(0)  # StudentT.sample draws from the global RNG
    W = torch.distributions.StudentT(4.0).sample((64, 256))  # excess kurtosis >> 0
    before = kurtosis(W, dim=-1).mean()
    Q = random_orthogonal(256, seed=1, dtype=W.dtype)
    after = kurtosis(W @ Q.T, dim=-1).mean()
    assert after < before / 2


def test_outlier_mass_and_floor():
    W = torch.randn(32, 64, dtype=torch.float64)
    mass = outlier_mass(W, k_sigma=3.0)
    assert mass.shape == (64,)
    assert 0 <= mass.min() and mass.max() <= 1
    assert sq_floor(2) == 4.0**-2


def test_ip_distortion_zero_for_exact():
    W = torch.randn(8, 32, dtype=torch.float64)
    X = torch.randn(16, 32, dtype=torch.float64)
    assert ip_distortion(W, W, X) == 0
    assert ip_distortion(W, rtn_quantize(W, 4, 32), X) > 0
