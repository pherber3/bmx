"""Decomposition protocol and method registry (framework extension point #1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch


class FitResult(ABC):
    """A fitted decomposition. param_count() is first-class: all cross-method
    comparisons align on parameters, never on rank."""

    def __init__(self, method: str, rank: Any, loss_history: list[float]):
        self.method = method
        self.rank = rank
        self.loss_history = loss_history

    @abstractmethod
    def reconstruct(self) -> torch.Tensor: ...

    @abstractmethod
    def param_count(self) -> int: ...

    def relative_error(self, T: torch.Tensor) -> float:
        return (torch.linalg.norm(self.reconstruct() - T) / torch.linalg.norm(T)).item()


_REGISTRY: dict[str, Callable[..., FitResult]] = {}


def register(name: str):
    def deco(fn: Callable[..., FitResult]):
        _REGISTRY[name] = fn
        return fn

    return deco


def get_method(name: str) -> Callable[..., FitResult]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown method {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_methods() -> list[str]:
    return sorted(_REGISTRY)
