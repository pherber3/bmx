"""Stacked expert matrices from fine-grained MoE checkpoints (Track C).

Loads per-layer expert weights directly from safetensors shards via the
checkpoint's weight-map index -- the model is never instantiated, so a 30 GB
checkpoint costs disk plus one layer of tensors at a time.

Key pattern (OLMoE / Qwen1.5-MoE / DeepSeek-V2-Lite all match):
    model.layers.{layer}.mlp.experts.{expert}.{which}_proj.weight
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from bmx.stacks.base import Stack


def _weight_map(checkpoint_dir: Path) -> dict[str, str]:
    """Tensor name -> shard filename (single-file checkpoints map to itself)."""
    index = checkpoint_dir / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    single = checkpoint_dir / "model.safetensors"
    assert single.exists(), f"no safetensors index or file in {checkpoint_dir}"
    with safe_open(single, framework="pt") as f:
        return {k: single.name for k in f.keys()}


def moe_layers(checkpoint_dir: str | Path) -> list[int]:
    """Layer indices that have routed experts (dense layers excluded)."""
    pattern = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.\d+\.")
    layers = {
        int(m.group(1))
        for k in _weight_map(Path(checkpoint_dir))
        if (m := pattern.match(k))
    }
    return sorted(layers)


def expert_stack(
    checkpoint_dir: str | Path, layer: int, which: str, model: str = ""
) -> Stack:
    """Stack of per-expert matrices, which in {gate, up, down} -> (out, in, E).

    gate/up experts are (d_ff, d_model); down is (d_model, d_ff). Slices are
    stacked on the LAST axis per the project convention.
    """
    assert which in ("gate", "up", "down"), f"which must be gate/up/down, got {which!r}"
    checkpoint_dir = Path(checkpoint_dir)
    wmap = _weight_map(checkpoint_dir)

    expert_re = re.compile(
        rf"model\.layers\.{layer}\.mlp\.experts\.(\d+)\.{which}_proj\.weight"
    )
    keys = {int(m.group(1)): k for k in wmap if (m := expert_re.fullmatch(k))}
    assert keys, f"no {which} experts found for layer {layer} in {checkpoint_dir}"
    n_experts = max(keys) + 1
    assert sorted(keys) == list(range(n_experts)), "non-contiguous expert indices"

    by_shard: dict[str, list[int]] = defaultdict(list)
    for e in range(n_experts):
        by_shard[wmap[keys[e]]].append(e)

    tensors: list[torch.Tensor | None] = [None] * n_experts
    for shard, experts in by_shard.items():
        with safe_open(checkpoint_dir / shard, framework="pt") as f:
            for e in experts:
                tensors[e] = f.get_tensor(keys[e])

    T = torch.stack(tensors, dim=-1)  # type: ignore[arg-type]
    axes = (
        ("d_ff", "d_model", "expert")
        if which != "down"
        else ("d_model", "d_ff", "expert")
    )
    return Stack(
        T.contiguous(), model or checkpoint_dir.name, layer, f"expert_{which}", axes
    )
