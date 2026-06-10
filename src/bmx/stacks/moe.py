"""C-track (gated on C1 census): stacked expert matrices from fine-grained MoE
checkpoints, loaded from safetensors shards without instantiating the model.
Implemented when Track C opens; shaped now so a2-style sweeps port directly."""

from bmx.stacks.base import Stack


def expert_stack(checkpoint_dir: str, layer: int, which: str, model: str = "") -> Stack:
    """Stack of per-expert FFN matrices, which in {gate, up, down} -> (d, d_ff, E)."""
    raise NotImplementedError("gated on C1 census; see research plan Track C")
