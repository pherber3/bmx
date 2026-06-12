"""Pairwise expert-similarity metrics for the C1 redundancy census.

Tests entry 2's failure mode 2: is cross-expert redundancy global (shared
structure a joint decomposition can exploit) or concentrated in a few
mergeable clusters? Three complementary metrics over the experts' SHARED
dimension (d_model -- the input side for gate/up, the output side for down):

- cos:  cosine similarity of flattened weights (what merging papers use).
- cka:  linear CKA between centered Grams over the shared dimension
        (G_e = S_e^T S_e): "do experts weight the same shared directions?".
- sub:  principal-subspace overlap ||V_i V_j^T||_F^2 / r of the top-r right
        singular subspaces: basis-invariant subspace agreement.
"""

import torch

from bmx.quant.hadamard import orthogonalize
from bmx.stacks.base import Stack


def subspace_overlap(A: torch.Tensor, U_ref: torch.Tensor) -> float:
    """Mean squared cosine between span(A) and span(U_ref), in [0, 1].

    A is any full-column-rank basis (orthonormalized here — the QR is
    load-bearing when A's columns are scaled, e.g. SVD factors carrying
    singular values); U_ref must already have orthonormal columns.
    Single-pair counterpart of pairwise_similarities' "sub" metric.
    """
    Q = orthogonalize(A)
    return ((U_ref.mT @ Q) ** 2).sum().item() / U_ref.shape[1]


def experts_shared_last(stack: Stack) -> torch.Tensor:
    """(out, in, E) expert stack -> (E, private, shared) with d_model last."""
    if stack.axes[0] == "d_ff":  # gate/up: (d_ff, d_model, E)
        return stack.tensor.permute(2, 0, 1)
    return stack.tensor.permute(2, 1, 0)  # down: (d_model, d_ff, E)


def pairwise_similarities(S: torch.Tensor, top_r: int = 32) -> dict[str, torch.Tensor]:
    """S: (E, private, shared) -> {metric: (E, E) similarity matrix}."""
    E, p, d = S.shape

    flat = S.reshape(E, -1)
    flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos = flat @ flat.T

    G = torch.einsum("epd,epc->edc", S, S)  # (E, d, d) shared-dim Grams
    G = G - G.mean(dim=1, keepdim=True)
    G = G - G.mean(dim=2, keepdim=True)  # double centering
    g = G.reshape(E, -1)
    g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cka = g @ g.T

    r = min(top_r, p, d)
    Vh = torch.linalg.svd(S, full_matrices=False).Vh[:, :r, :]  # (E, r, d)
    M = Vh.reshape(E * r, d) @ Vh.reshape(E * r, d).T
    sub = M.reshape(E, r, E, r).pow(2).sum(dim=(1, 3)) / r

    return {"cos": cos, "cka": cka, "sub": sub}


def similarity_summary(sim: torch.Tensor) -> dict[str, float]:
    """Off-diagonal stats + participation ratio (global-vs-clustered signal).

    PR = (sum lambda)^2 / sum lambda^2 of the PSD similarity matrix, in [1, E]:
    ~1 means one global shared mode dominates; ~E means experts are mutually
    orthogonal (no redundancy). Reported normalized as pr_frac = PR / E.
    """
    E = sim.shape[0]
    off = sim[~torch.eye(E, dtype=torch.bool, device=sim.device)]
    lam = torch.linalg.eigvalsh(sim.double()).clamp_min(0)
    pr = (lam.sum() ** 2 / (lam.pow(2).sum())).item()
    return {
        "off_mean": off.mean().item(),
        "off_median": off.median().item(),
        "off_max": off.max().item(),
        "frac_gt_05": (off > 0.5).double().mean().item(),
        "frac_gt_09": (off > 0.9).double().mean().item(),
        "pr_frac": pr / E,
        "n_experts": E,
    }
