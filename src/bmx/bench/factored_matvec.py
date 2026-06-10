"""The Track B kernel: y_k = sum_t u_t^k * (V_t (w_t^k * x)) vs dense per-slice GEMV.

Bytes story: dense reads h*m*p weights per token; factored reads ell*m*p template
weights (reused across all h slices) + ell*(m+p)*h gain entries. FLOPs inflate
by ~ell. In the memory-bound decode regime bytes are latency.
"""

import torch


def dense_from_factors(A, B, C) -> torch.Tensor:
    """Materialize the stacked weights W: (h, m, p), W[k] = slice k of bmp.

    Built term-by-term over the ell axis: bmp's single einsum materializes an
    (m, p, ell, h) intermediate, which is 32 GiB at (4096, 4096, 8, 64) fp32.
    """
    m, ell, h = A.shape
    p = B.shape[1]
    W = torch.zeros(h, m, p, dtype=A.dtype, device=A.device)
    for t in range(ell):
        W += (
            A[:, t, :].T[:, :, None]  # A[i,t,k] at [k,i,.]
            * B[:, :, t][None, :, :]  # B[i,j,t] at [.,i,j]
            * C[t, :, :].T[:, None, :]  # C[t,j,k] at [k,.,j]
        )
    return W


def dense_slice_matvec(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """W: (h, m, p), x: (b, p) -> y: (h, b, m). The baseline that reads h*m*p bytes."""
    return torch.einsum("kij,bj->kbi", W, x)


def factored_matvec(A, B, C, x: torch.Tensor) -> torch.Tensor:
    """A: (m, ell, h), B: (m, p, ell), C: (ell, p, h), x: (b, p) -> (h, b, m)."""
    xs = torch.einsum("bj,tjk->tkbj", x, C)  # input gains applied
    ys = torch.einsum("ijt,tkbj->tkbi", B, xs)  # template GEMMs (the bulk)
    return torch.einsum("itk,tkbi->kbi", A, ys)  # output gains + sum over t


def templates_to_bmm_layout(B: torch.Tensor) -> torch.Tensor:
    """One-time relayout of templates (m, p, ell) -> (ell, m, p) contiguous.

    A deployment stores templates in this layout. The einsum path instead
    re-copies B into bmm order on every call, which is what destroys the
    bandwidth win at ell >= 2 (measured on H100: ~8x jump from ell=1 to 2).
    """
    return B.permute(2, 0, 1).contiguous()


def factored_matvec_bmm(A, Bt, C, x: torch.Tensor) -> torch.Tensor:
    """Pre-transposed-template variant: the template read is one clean bmm.

    A: (m, ell, h), Bt: (ell, m, p) from templates_to_bmm_layout, C: (ell, p, h),
    x: (b, p) -> (h, b, m).
    """
    ell, m, p = Bt.shape
    h = A.shape[2]
    b = x.shape[0]
    xs = torch.einsum("bj,tjk->tjkb", x, C).reshape(ell, p, h * b)
    ys = torch.bmm(Bt, xs).view(ell, m, h, b)  # reads ell*m*p template bytes
    return torch.einsum("itk,tikb->kbi", A, ys)


_compiled = None


def factored_matvec_compiled(A, B, C, x):
    """torch.compile variant; compiles lazily on first call (CUDA recommended)."""
    global _compiled
    if _compiled is None:
        # One graph per shape in a sweep: the default recompile limit (8)
        # silently falls back to eager beyond it, corrupting 'compiled' rows.
        torch._dynamo.config.recompile_limit = 4096
        _compiled = torch.compile(factored_matvec, dynamic=False)
    return _compiled(A, B, C, x)
