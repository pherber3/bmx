"""The Track B kernel: y_k = sum_t u_t^k * (V_t (w_t^k * x)) vs dense per-slice GEMV.

Bytes story: dense reads h*m*p weights per token; factored reads ell*m*p template
weights (reused across all h slices) + ell*(m+p)*h gain entries. FLOPs inflate
by ~ell. In the memory-bound decode regime bytes are latency.
"""

import torch

from bmx.decomp.ops import bmp


def dense_from_factors(A, B, C) -> torch.Tensor:
    """Materialize the stacked weights W: (h, m, p), W[k] = slice k of bmp."""
    return bmp(A, B, C).permute(2, 0, 1).contiguous()


def dense_slice_matvec(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """W: (h, m, p), x: (b, p) -> y: (h, b, m). The baseline that reads h*m*p bytes."""
    return torch.einsum("kij,bj->kbi", W, x)


def factored_matvec(A, B, C, x: torch.Tensor) -> torch.Tensor:
    """A: (m, ell, h), B: (m, p, ell), C: (ell, p, h), x: (b, p) -> (h, b, m)."""
    xs = torch.einsum("bj,tjk->tkbj", x, C)  # input gains applied
    ys = torch.einsum("ijt,tkbj->tkbi", B, xs)  # template GEMMs (the bulk)
    return torch.einsum("itk,tkbi->kbi", A, ys)  # output gains + sum over t


_compiled = None


def factored_matvec_compiled(A, B, C, x):
    """torch.compile variant; compiles lazily on first call (CUDA recommended)."""
    global _compiled
    if _compiled is None:
        _compiled = torch.compile(factored_matvec)
    return _compiled(A, B, C, x)
