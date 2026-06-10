"""A5 (gated on the A4 decision): LASER-style layer-selective weight replacement
and WikiText-103 perplexity. Implemented when Track A's gate opens."""

from bmx.decomp.base import FitResult


def swap_and_perplexity(
    model_name: str,
    layer: int,
    object_name: str,
    fit: FitResult,
    dataset: str = "wikitext-103-raw-v1",
) -> float:
    """Replace one layer's object with fit.reconstruct(), return perplexity delta."""
    raise NotImplementedError("gated on A4 decision; see research plan Track A5")
