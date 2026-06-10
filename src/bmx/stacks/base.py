"""Stack: a tensor plus the metadata that keys every downstream metric row
(framework extension point #2 — any weight-object source implements a builder)."""

from dataclasses import dataclass

import torch


@dataclass
class Stack:
    tensor: torch.Tensor
    model: str
    layer: int
    object_name: str
    axes: tuple[str, ...]
