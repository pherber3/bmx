import torch

from bmx.cache.codecs import scale_bits, tier_bits
from bmx.cache.spectral import (
    SpectralPack,  # noqa: F401  (documents the fitted-pack API this module tests)
    assemble_whitener,
    fit_spectral_pack,
    identity_whitener,
    key_second_moment,
    query_position_moment,
    skeptic_charge,
    spectral_quantize,
)


def test_key_second_moment_shape_and_value():
    g = torch.Generator().manual_seed(0)
    M = torch.randn(128, 16, generator=g)
    Sigma = key_second_moment(M)
    assert Sigma.shape == (16, 16) and Sigma.dtype == torch.float64
    expected = (M.double().mT @ M.double()) / 128
    assert torch.allclose(Sigma, expected)


def test_query_moment_identity_rope_matches_plain_outer_product():
    """With cos=1, sin=0 (R_p = I), W_j must equal the plain GQA-pooled query
    second moment."""
    g = torch.Generator().manual_seed(1)
    h, T, d, h_kv = 4, 32, 8, 2
    q = torch.randn(h, T, d, generator=g)
    S = 64
    cos, sin = torch.ones(S, d), torch.zeros(S, d)
    W = query_position_moment(q, cos, sin, h_kv, position_stride=16)
    grp = h // h_kv
    for j in range(h_kv):
        qj = q[j * grp : (j + 1) * grp].reshape(-1, d).double()
        expected = qj.mT @ qj / qj.shape[0]
        assert torch.allclose(W[j], expected, atol=1e-10), f"head {j}"


def test_query_moment_is_symmetric_psd():
    g = torch.Generator().manual_seed(2)
    q = torch.randn(8, 16, 8, generator=g)
    # A real-ish RoPE table: interleave some rotation
    S = 32
    theta = torch.linspace(0, 3.0, S).unsqueeze(1) * torch.ones(1, 8)
    W = query_position_moment(q, theta.cos(), theta.sin(), h_kv=4)
    for j in range(4):
        assert torch.allclose(W[j], W[j].mT, atol=1e-12)
        assert torch.linalg.eigvalsh(W[j]).min() > -1e-10


def test_whitener_squares_to_w():
    g = torch.Generator().manual_seed(3)
    A = torch.randn(2, 8, 8, generator=g).double()
    W_blocks = A @ A.mT / 8 + 0.1 * torch.eye(8)
    Wh, Wh_inv = assemble_whitener(W_blocks, ridge=0.0)
    C = 16
    W_dense = torch.zeros(C, C, dtype=torch.float64)
    W_dense[:8, :8], W_dense[8:, 8:] = W_blocks[0], W_blocks[1]
    assert torch.allclose(Wh @ Wh, W_dense, atol=1e-8)
    assert torch.allclose(Wh @ Wh_inv, torch.eye(C, dtype=torch.float64), atol=1e-8)


def test_identity_whitener():
    Wh, Wh_inv = identity_whitener(12)
    assert torch.equal(Wh, torch.eye(12, dtype=torch.float64))
    assert torch.equal(Wh_inv, torch.eye(12, dtype=torch.float64))


def test_query_moment_matches_explicit_rotation_matrices():
    """Value-pin with sin != 0: W must equal mean_p R_pT (pooled qqT) R_p where
    R_p is built explicitly from the FORWARD rotation only — columns are
    apply_rope(e_i) — and transposed as a matrix. No negated sin appears in the
    ground truth, so it derives the adjoint independently rather than assuming
    it; a sign flip in the production adjoint would fail this (margin ~0.7 vs
    atol 1e-10). The cos/sin table uses the duplicated-half structure
    (cat(freqs, freqs)) of real RoPE tables — the structure under which the
    negated-sin expression IS the matrix transpose."""
    from bmx.cache.rope import apply_rope

    g = torch.Generator().manual_seed(4)
    h, T, d, h_kv, S = 2, 16, 4, 1, 8
    q = torch.randn(h, T, d, generator=g)
    # Real RoPE tables duplicate halves: cos/sin[:, j] == cos/sin[:, j + d/2].
    freqs = torch.linspace(0.5, 1.0, d // 2)
    theta = torch.linspace(0.3, 2.0, S).unsqueeze(1) * torch.cat(
        [freqs, freqs]
    ).unsqueeze(0)
    cos, sin = theta.cos(), theta.sin()

    stride = 2
    W = query_position_moment(q, cos, sin, h_kv, position_stride=stride)

    # GQA-pooled query second moment E[qqT] (h_kv=1 pools all heads), once.
    q_flat = q.double().reshape(-1, d)  # (h*T, d)
    pooled = q_flat.mT @ q_flat / (h * T)  # (d, d)

    W_expected = torch.zeros(d, d, dtype=torch.float64)
    positions = list(range(0, S, stride))
    for p in positions:
        # Explicit R_p: columns are apply_rope(e_i) at position p (FORWARD rotation).
        basis = torch.eye(d).double().unsqueeze(1)  # (d, 1, d): d vectors, 1 position
        Rp_cols = apply_rope(basis, cos[p : p + 1].double(), sin[p : p + 1].double())
        Rp = Rp_cols.squeeze(1).mT  # (d, d), column i = R_p e_i
        W_expected += Rp.mT @ pooled @ Rp  # R_pT E[qqT] R_p via MATRIX transpose
    W_expected /= len(positions)
    assert torch.allclose(W[0], W_expected, atol=1e-10)


def _spiked_keys(S=512, C=64, seed=0, spike_dirs=None, spike_std=(30.0, 30.0)):
    """Keys with two planted spike directions over unit noise. Returns (M, dirs)."""
    g = torch.Generator().manual_seed(seed)
    if spike_dirs is None:
        raw = torch.randn(C, 2, generator=g)
        spike_dirs, _ = torch.linalg.qr(raw)  # (C, 2) orthonormal
    z = torch.randn(S, 2, generator=g) * torch.tensor(spike_std)
    noise = torch.randn(S, C, generator=g)
    return z @ spike_dirs.mT + noise, spike_dirs


def test_pack_roundtrip_identity():
    from bmx.cache.spectral import identity_whitener

    M, _ = _spiked_keys()
    Wh, Wh_inv = identity_whitener(64)
    pack = fit_spectral_pack(M, Wh, Wh_inv, budget=8.0, tiers=(8,), group=64)
    # enc @ dec.T must be the identity (basis is invertible by construction)
    eye = pack.enc.double() @ pack.dec.double().mT
    assert torch.allclose(eye, torch.eye(64, dtype=torch.float64), atol=1e-4)
    # At a uniform 8-bit allocation the codec is near-lossless
    M_hat, bpe = spectral_quantize(M, pack)
    assert (M_hat - M).norm() / M.norm() < 0.02
    assert abs(bpe - (8.0 + scale_bits(64))) < 1e-9


def test_waterfill_funds_spikes_drops_bulk():
    from bmx.cache.spectral import identity_whitener

    M, _ = _spiked_keys()
    Wh, Wh_inv = identity_whitener(64)
    pack = fit_spectral_pack(M, Wh, Wh_inv, budget=2.0)
    assert pack.bits[:2].min() >= 5, f"spike dirs underfunded: {pack.bits[:4]}"
    assert (pack.bits == 0).sum() > 0, "tight budget must drop bulk directions"
    assert pack.bits.float().mean().item() <= 2.0 + 1e-9


def test_weighted_basis_beats_unweighted_on_query_skewed_source():
    """P4 mechanism test: two equal-variance key spikes, queries read only one.
    The W-weighted basis funds the query-read spike and wins on logit distortion
    at the same budget."""
    from bmx.cache.collect import from_matrix
    from bmx.cache.metrics import logit_distortion
    from bmx.cache.spectral import (
        assemble_whitener,
        identity_whitener,
        query_position_moment,
    )

    C, S, T = 64, 512, 64
    M, dirs = _spiked_keys(S=S, C=C, seed=0)
    g = torch.Generator().manual_seed(7)
    # Queries aligned with spike 1 only (plus small noise); h = h_kv = 1 head.
    q = (
        torch.randn(T, 1, generator=g) * dirs[:, 1].unsqueeze(0)
        + 0.05 * torch.randn(T, C, generator=g)
    ).unsqueeze(0)  # (1, T, C)
    cos, sin = torch.ones(S, C), torch.zeros(S, C)  # no RoPE in this synthetic

    W = query_position_moment(q, cos, sin, h_kv=1, position_stride=64)
    Wh, Wh_inv = assemble_whitener(W)
    eWh, eWh_inv = identity_whitener(C)

    budget = 2.0
    p_w = fit_spectral_pack(M, Wh, Wh_inv, budget)
    p_u = fit_spectral_pack(M, eWh, eWh_inv, budget)
    K = from_matrix(M, 1)
    d_w = logit_distortion(K, from_matrix(spectral_quantize(M, p_w)[0], 1), q)
    d_u = logit_distortion(K, from_matrix(spectral_quantize(M, p_u)[0], 1), q)
    assert d_w < d_u, f"weighted {d_w} !< unweighted {d_u}"


def test_spectral_deterministic():
    from bmx.cache.spectral import identity_whitener

    M, _ = _spiked_keys(seed=5)
    Wh, Wh_inv = identity_whitener(64)
    p1 = fit_spectral_pack(M, Wh, Wh_inv, 2.5)
    p2 = fit_spectral_pack(M, Wh, Wh_inv, 2.5)
    assert torch.equal(p1.bits, p2.bits)
    a, _ = spectral_quantize(M, p1)
    b, _ = spectral_quantize(M, p2)
    assert torch.equal(a, b)


def test_skeptic_charge_formula():
    assert (
        abs(
            skeptic_charge(1024, 32768, (0, 2, 3, 4, 5, 6, 8))
            - (16.0 * 1024 / 32768 + tier_bits((0, 2, 3, 4, 5, 6, 8), 32768))
        )
        < 1e-12
    )


def test_pack_file_roundtrip(tmp_path):
    from bmx.cache.spectral import (
        fit_spectral_basis,
        identity_whitener,
        load_packs,
        pack_from_basis,
        save_pack_file,
    )

    C = 32
    Wh, Wh_inv = identity_whitener(C)
    bases = {}
    for i in range(2):
        M, _ = _spiked_keys(S=256, C=C, seed=i)
        bases[i] = fit_spectral_basis(M, Wh, Wh_inv)
    path = tmp_path / "packs.safetensors"
    save_pack_file(path, bases, budgets=(2.5, 3.0), group=16, meta={"model": "tiny"})

    packs = load_packs(path, 2.5)
    assert set(packs) == {0, 1}
    ref = pack_from_basis(bases[0], 2.5, group=16)
    assert torch.equal(packs[0].bits, ref.bits)
    assert torch.allclose(packs[0].enc, ref.enc)
    assert packs[0].group == 16 and packs[0].budget == 2.5

    import json

    side = json.loads((tmp_path / "packs.safetensors.json").read_text())
    assert side["model"] == "tiny" and 2.5 in side["budgets"]

    import pytest

    with pytest.raises(KeyError):
        load_packs(path, 4.0)
