import pytest
import torch

from bmx.decomp.base import (
    FitResult,
    _REGISTRY,
    available_methods,
    get_method,
    register,
)


class _DummyFit(FitResult):
    def __init__(self, T):
        super().__init__(method="dummy", rank=1, loss_history=[1.0, 0.5])
        self._T = T

    def reconstruct(self):
        return torch.zeros_like(self._T)

    def param_count(self):
        return 7


def test_registry_roundtrip():
    @register("dummy")
    def fit_dummy(T, rank):
        return _DummyFit(T)

    try:
        assert "dummy" in available_methods()
        fit = get_method("dummy")(torch.ones(2, 2, 2), rank=1)
        assert fit.param_count() == 7
        assert fit.loss_history[-1] == 0.5
    finally:
        _REGISTRY.pop("dummy", None)


def test_relative_error():
    T = torch.ones(2, 2, 2, dtype=torch.float64)
    fit = _DummyFit(T)
    assert fit.relative_error(T) == pytest.approx(1.0)


def test_unknown_method_raises():
    with pytest.raises(KeyError):
        get_method("does-not-exist")
