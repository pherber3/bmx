"""C3 (gated on C1/C2): routed-token capture and expert-output relative error.
Implemented when Track C opens."""

import torch


def capture_routed_activations(model_name: str, n_tokens: int, layer: int):
    """Forward hooks collecting per-expert input activations on calibration text."""
    raise NotImplementedError("gated on C1 census; see research plan Track C3")


def expert_output_error(
    W_true: torch.Tensor, W_rec: torch.Tensor, X: torch.Tensor
) -> float:
    """Relative L2 of expert outputs under reconstructed weights, given captured X."""
    raise NotImplementedError("gated on C1 census; see research plan Track C3")
