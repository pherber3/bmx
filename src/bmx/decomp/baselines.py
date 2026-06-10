"""Baseline decompositions: per-slice SVD, CP-ALS, Tucker/HOOI, shared-factor
Tucker (TensorLLM's operator: shared mode-0/1 factors, per-slice cores)."""

import tensorly as tl
import torch
from tensorly.decomposition import parafac, partial_tucker, tucker
from tensorly.tenalg import multi_mode_dot

from bmx.decomp.base import FitResult, register

# PyTorch backend: tensorly fits run on whatever device/dtype T lives on,
# so CP/Tucker baselines ride the GPU alongside BMD-RALS on VM sweeps.
tl.set_backend("pytorch")


class SliceSVDFit(FitResult):
    def __init__(self, W, V, rank, T_shape):
        super().__init__(method="slice_svd", rank=rank, loss_history=[])
        self.W, self.V = W, V  # W: (n, m, r), V: (n, p, r)
        self._shape = T_shape

    def reconstruct(self):
        return torch.einsum("kir,kjr->ijk", self.W, self.V)

    def param_count(self):
        m, p, n = self._shape
        return n * self.rank * (m + p)


@register("slice_svd")
def fit_slice_svd(T: torch.Tensor, rank: int) -> SliceSVDFit:
    r = int(rank)
    m, p, n = T.shape
    assert r <= min(m, p), f"slice_svd rank {r} > min(m,p)={min(m, p)}"
    U, S, Vh = torch.linalg.svd(T.permute(2, 0, 1), full_matrices=False)
    W = U[:, :, :r] * S[:, None, :r]  # (n, m, r)
    V = Vh[:, :r, :].mT.contiguous()  # (n, p, r)
    fit = SliceSVDFit(W, V, r, tuple(T.shape))
    fit.loss_history = [fit.relative_error(T)]
    return fit


class DenseFit(FitResult):
    """Generic fit storing a dense reconstruction + explicit param count."""

    def __init__(self, method, rank, rec, n_params):
        super().__init__(method=method, rank=rank, loss_history=[])
        self._rec = rec
        self._n_params = n_params

    def reconstruct(self):
        return self._rec

    def param_count(self):
        return self._n_params


@register("cp")
def fit_cp(T: torch.Tensor, rank: int, *, n_iter_max: int = 500, seed: int = 0):
    m, p, n = T.shape
    r = int(rank)
    cp = parafac(T, rank=r, n_iter_max=n_iter_max, init="random", random_state=seed)
    rec = tl.cp_to_tensor(cp)
    fit = DenseFit("cp", r, rec, r * (m + p + n))
    fit.loss_history = [fit.relative_error(T)]
    return fit


@register("tucker")
def fit_tucker(T: torch.Tensor, rank, *, n_iter_max: int = 200, seed: int = 0):
    m, p, n = T.shape
    R1, R2, R3 = (int(x) for x in rank)
    assert R1 <= m and R2 <= p and R3 <= n, (
        f"tucker rank {(R1, R2, R3)} exceeds dims {(m, p, n)}"
    )
    # tucker() returns a TuckerTensor (namedtuple-like): TuckerTensor[0]=core,
    # TuckerTensor[1]=factors — standard (core, factors) unpack works correctly.
    core, factors = tucker(T, rank=[R1, R2, R3], n_iter_max=n_iter_max, init="svd")
    rec = tl.tucker_to_tensor((core, factors))
    n_params = m * R1 + p * R2 + n * R3 + R1 * R2 * R3
    fit = DenseFit("tucker", (R1, R2, R3), rec, n_params)
    fit.loss_history = [fit.relative_error(T)]
    return fit


@register("shared_tucker")
def fit_shared_tucker(T: torch.Tensor, rank, *, n_iter_max: int = 200):
    """Tucker with mode-3 factor pinned to identity: shared U1, U2; per-slice cores."""
    m, p, n = T.shape
    R1, R2 = (int(x) for x in rank)
    assert R1 <= m and R2 <= p, f"shared_tucker rank {(R1, R2)} exceeds dims {(m, p)}"
    # In tensorly 0.9.0 partial_tucker returns a plain tuple of length 2:
    #   result[0] = (core, factors)   — the decomposition
    #   result[1] = [err0, err1, ...]  — per-iteration reconstruction errors
    # Direct `core, factors = partial_tucker(...)` would incorrectly assign
    # core=(core_array, factors_list) and factors=[err_scalars].
    # Correct unpack: unwrap result[0] to get core and factors.
    result = partial_tucker(
        T, rank=[R1, R2], modes=[0, 1], n_iter_max=n_iter_max, init="svd"
    )
    core, factors = result[0]
    rec = multi_mode_dot(core, factors, modes=[0, 1])
    n_params = m * R1 + p * R2 + n * R1 * R2
    fit = DenseFit("shared_tucker", (R1, R2), rec, n_params)
    fit.loss_history = [fit.relative_error(T)]
    return fit
