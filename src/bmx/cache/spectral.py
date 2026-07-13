"""K4 spectral codec: query-weighted eigenbasis + waterfilled bit allocation.

The metric that scores the task is E[(qᵀ R_p (k - k_hat))²] where R_p is the
per-position RoPE rotation. That equals eᵀ W e with W = E[R_pᵀ q qᵀ R_p] —
block-diagonal per kv-head (attention logits never couple heads). Substituting
u = W^{1/2} k reduces weighted rate-distortion to plain MSE on covariance
W^{1/2} Σ_k W^{1/2}: the optimal basis is that matrix's eigenbasis and bits
waterfill on its eigenvalues (spec §3, theory grounding §8 of
docs/superpowers/specs/2026-07-12-k4-spectral-codec-design.md).

All moment/eig math is fp64; codec application is fp32.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from bmx.cache.codecs import (
    allocate_bits_from_variance,
    quantize_by_bits,
    scale_bits,
    tier_bits,
)
from bmx.cache.rope import _rotate_half


def key_second_moment(M: torch.Tensor) -> torch.Tensor:
    """Uncentered per-channel second moment MᵀM/S of an (S, C) fp matrix, fp64."""
    assert M.dim() == 2, f"M must be (S, C); got {tuple(M.shape)}"
    Md = M.double()
    return Md.mT @ Md / M.shape[0]


def query_position_moment(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    h_kv: int,
    *,
    position_stride: int = 8,
) -> torch.Tensor:
    """W blocks (h_kv, d, d): E over probe queries and sampled key positions of
    (R_pᵀ q)(R_pᵀ q)ᵀ, pooled over each kv-head's GQA query group.

    R_pᵀ q is RoPE at position p with negated sin (inverse rotation). cos/sin
    are (S, d) tables from rope_cos_sin; pass cos=ones/sin=zeros for no-RoPE
    models (gpt2) — then W is the plain pooled query second moment.
    """
    h, T, d = q.shape
    assert h % h_kv == 0, f"h={h} not divisible by h_kv={h_kv}"
    grp = h // h_kv
    S = cos.shape[0]
    q64 = q.double()
    W = torch.zeros(h_kv, d, d, dtype=torch.float64)
    positions = list(range(0, S, position_stride))
    for p in positions:
        cp = cos[p].double().view(1, 1, d)
        sp = sin[p].double().view(1, 1, d)
        q_rot = q64 * cp + _rotate_half(q64) * (-sp)  # (h, T, d) = R_pᵀ q
        for j in range(h_kv):
            qj = q_rot[j * grp : (j + 1) * grp].reshape(-1, d)  # (grp*T, d)
            W[j] += qj.mT @ qj
    W /= len(positions) * grp * T
    return W


def assemble_whitener(
    W_blocks: torch.Tensor, *, ridge: float = 1e-3
) -> tuple[torch.Tensor, torch.Tensor]:
    """(h_kv, d, d) fp64 blocks -> dense (C, C) fp64 (W^{1/2}, W^{-1/2}).

    Per-block symmetric eigendecomposition; eigenvalues floored at
    ridge·λ_max(block) so near-null query directions can't explode W^{-1/2}.
    Blocks land on the diagonal in to_matrix's head-major channel order.
    """
    h_kv, d, _ = W_blocks.shape
    C = h_kv * d
    Wh = torch.zeros(C, C, dtype=torch.float64)
    Wh_inv = torch.zeros(C, C, dtype=torch.float64)
    for j in range(h_kv):
        Wj = 0.5 * (W_blocks[j] + W_blocks[j].mT)
        lam, E = torch.linalg.eigh(Wj)
        lam = lam.clamp_min(ridge * lam.max().clamp_min(1e-30))
        sl = slice(j * d, (j + 1) * d)
        Wh[sl, sl] = E @ torch.diag(lam.sqrt()) @ E.mT
        Wh_inv[sl, sl] = E @ torch.diag(lam.rsqrt()) @ E.mT
    return Wh, Wh_inv


def identity_whitener(C: int) -> tuple[torch.Tensor, torch.Tensor]:
    """W = I: the unweighted-KLT ablation path (plain Σ_k eigenbasis)."""
    eye = torch.eye(C, dtype=torch.float64)
    return eye, eye.clone()


@dataclasses.dataclass
class SpectralPack:
    """Corpus-fit spectral codec for one (layer, side): basis + bit allocation.

    Y = M @ enc (encode); M_hat = Y_hat @ dec.mT (decode); enc @ dec.mT = I.
    enc = W^{1/2} E, dec = W^{-1/2} E where E is the eigenbasis of
    W^{1/2} Σ_k W^{1/2} (descending eigenvalues lam). bits waterfills lam.
    Model-level accounting: the pack ships with the model (zero per-sequence
    bits); skeptic-mode per-sequence charge is skeptic_charge(C, S, tiers).
    """

    enc: torch.Tensor  # (C, C) fp32
    dec: torch.Tensor  # (C, C) fp32
    lam: torch.Tensor  # (C,) fp32, descending
    bits: torch.Tensor  # (C,) int64, members of tiers
    group: int
    tiers: tuple[int, ...]
    budget: float


@dataclasses.dataclass
class SpectralBasis:
    """Budget-independent part of a spectral fit: basis + spectrum.

    `lam64` retains the fp64 eigenvalues (the exact tensor
    `allocate_bits_from_variance` is called with at pack time); `lam` is the
    fp32-rounded copy stored on the resulting SpectralPack.
    """

    enc: torch.Tensor  # (C, C) fp32
    dec: torch.Tensor  # (C, C) fp32
    lam: torch.Tensor  # (C,) fp32, descending
    lam64: torch.Tensor  # (C,) fp64, descending — private, allocation input


def fit_spectral_basis(
    M_fit: torch.Tensor,
    Wh: torch.Tensor,
    Wh_inv: torch.Tensor,
) -> SpectralBasis:
    """Fit the budget-independent basis + spectrum on M_fit. fp64 internally."""
    Sigma = key_second_moment(M_fit)
    T = Wh @ Sigma @ Wh
    lam, E = torch.linalg.eigh(0.5 * (T + T.mT))
    lam = lam.flip(0).clamp_min(0.0)  # descending
    E = E.flip(1)
    return SpectralBasis(
        enc=(Wh @ E).float(),
        dec=(Wh_inv @ E).float(),
        lam=lam.float(),
        lam64=lam,
    )


def pack_from_basis(
    basis: SpectralBasis,
    budget: float,
    *,
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8),
    group: int = 64,
) -> SpectralPack:
    """Allocate bits for one budget against an already-fit SpectralBasis."""
    assert 1 not in tiers, "symmetric RTN is undefined at 1 bit (qmax=0)"
    bits = allocate_bits_from_variance(basis.lam64, budget, tiers)
    return SpectralPack(
        enc=basis.enc,
        dec=basis.dec,
        lam=basis.lam,
        bits=bits,
        group=group,
        tiers=tuple(tiers),
        budget=float(budget),
    )


def fit_spectral_pack(
    M_fit: torch.Tensor,
    Wh: torch.Tensor,
    Wh_inv: torch.Tensor,
    budget: float,
    *,
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8),
    group: int = 64,
) -> SpectralPack:
    """Fit basis + allocation on M_fit (the calibration matrix). fp64 internally."""
    return pack_from_basis(
        fit_spectral_basis(M_fit, Wh, Wh_inv), budget, tiers=tiers, group=group
    )


def spectral_quantize(
    M: torch.Tensor, pack: SpectralPack, *, mse_scale: bool = True
) -> tuple[torch.Tensor, float]:
    """Quantize (S, C) M with a fitted pack. Returns (M_hat, bpe_model).

    bpe_model = mean payload + groupwise-scale term (model-level accounting —
    the pack itself ships with the model). Add skeptic_charge(C, S, tiers) for
    the per-sequence-charged view.
    """
    S, C = M.shape
    assert pack.enc.shape == (C, C), f"pack C mismatch: {pack.enc.shape} vs C={C}"
    assert S % pack.group == 0, f"S={S} not divisible by group={pack.group}"
    Y = M @ pack.enc.to(M.dtype)
    Y_hat = quantize_by_bits(Y, pack.bits, pack.group, mse_scale=mse_scale)
    M_hat = Y_hat @ pack.dec.mT.to(M.dtype)
    bpe = float(pack.bits.float().mean().item()) + scale_bits(pack.group)
    return M_hat, bpe


def skeptic_charge(C: int, S: int, tiers: tuple[int, ...]) -> float:
    """Per-sequence charge when the pack is NOT granted model-level status:
    one fp16 C×C decoder matrix (16·C/S) + the per-direction bit map."""
    return 16.0 * C / S + tier_bits(tiers, S)


def save_pack_file(
    path: str | Path,
    bases: dict[int, SpectralBasis],
    budgets: tuple[float, ...],
    *,
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8),
    group: int = 64,
    meta: dict | None = None,
) -> None:
    """Save per-layer bases + per-budget bit allocations to one safetensors file.

    Bits are computed via `pack_from_basis` at save time (the fp64 allocation
    input `basis.lam64` is only needed here) so `load_packs` never re-runs the
    allocator — reloaded packs are allocation-frozen artifacts, not re-fit.
    A JSON sidecar at `<path>.json` carries tiers/group/budgets/meta since
    safetensors only stores tensors.
    """
    tensors: dict[str, torch.Tensor] = {}
    for i, basis in bases.items():
        tensors[f"layer{i}.enc"] = basis.enc
        tensors[f"layer{i}.dec"] = basis.dec
        tensors[f"layer{i}.lam"] = basis.lam
        for budget in budgets:
            pack = pack_from_basis(basis, budget, tiers=tiers, group=group)
            tensors[f"layer{i}.bits_b{budget:g}"] = pack.bits
    save_file(tensors, str(path))

    sidecar = {"tiers": list(tiers), "group": group, "budgets": list(budgets)}
    if meta:
        sidecar.update(meta)
    Path(str(path) + ".json").write_text(json.dumps(sidecar))


def load_packs(path: str | Path, budget: float) -> dict[int, SpectralPack]:
    """Reconstruct per-layer SpectralPacks for one budget from a saved file.

    Raises KeyError (with the available budgets) if `budget` isn't in the file.
    """
    sidecar = json.loads(Path(str(path) + ".json").read_text())
    budgets = sidecar["budgets"]
    if budget not in budgets:
        raise KeyError(f"budget {budget} not in pack file; available: {budgets}")
    tiers = tuple(sidecar["tiers"])
    group = sidecar["group"]

    tensors = load_file(str(path))
    layers = sorted(
        {int(k.split(".")[0].removeprefix("layer")) for k in tensors if "." in k}
    )
    packs: dict[int, SpectralPack] = {}
    for i in layers:
        packs[i] = SpectralPack(
            enc=tensors[f"layer{i}.enc"],
            dec=tensors[f"layer{i}.dec"],
            lam=tensors[f"layer{i}.lam"],
            bits=tensors[f"layer{i}.bits_b{budget:g}"],
            group=group,
            tiers=tiers,
            budget=float(budget),
        )
    return packs
