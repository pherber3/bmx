import dataclasses

import pytest
import torch

from bmx.cache.codecs import scale_bits, tier_bits
from bmx.cache.spectral import (
    SpectralPack,  # noqa: F401  (documents the fitted-pack API this module tests)
    assemble_whitener,
    fit_spectral_pack,
    identity_whitener,
    jensen_gap_report,
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


def test_load_packs_for_spec_matches_load_packs_and_owns_dec_quant(tmp_path):
    """FIX 2 license: load_packs_for_spec(k_spec) is the hoisted altitude both
    StreamingQuantizedCache.__init__ and PackedStreamingCache.__init__ now
    delegate to — pack materialization owns the dec_quant decision.

    Non-spectral arm -> {} (nothing to load). Spectral arm with dec_quant="fp32"
    -> bitwise matches a direct load_packs() call. dec_quant="int8" ->
    matches int8_decoder_roundtrip applied by hand. Invalid dec_quant / missing
    pre_rope / missing pack_path all raise, exactly as the pre-hoist inline
    blocks did."""
    from bmx.cache.spectral import (
        fit_spectral_basis,
        identity_whitener,
        int8_decoder_roundtrip,
        load_packs,
        load_packs_for_spec,
        save_pack_file,
    )
    from bmx.cache.specs import CacheCodecSpec

    C = 32
    Wh, Wh_inv = identity_whitener(C)
    bases = {}
    for i in range(2):
        M, _ = _spiked_keys(S=256, C=C, seed=i)
        bases[i] = fit_spectral_basis(M, Wh, Wh_inv)
    path = tmp_path / "packs.safetensors"
    save_pack_file(path, bases, budgets=(2.5,), group=16, meta={"model": "tiny"})

    # Non-spectral arm: nothing to load.
    fp16_spec = CacheCodecSpec(arm="fp16")
    assert load_packs_for_spec(fp16_spec) == {}

    # Spectral, dec_quant="fp32" (default): bitwise matches load_packs directly.
    fp32_spec = CacheCodecSpec(
        arm="spectral", pre_rope=True, pack_path=str(path), budget=2.5
    )
    via_spec = load_packs_for_spec(fp32_spec)
    via_direct = load_packs(path, 2.5)
    assert set(via_spec) == set(via_direct) == {0, 1}
    for i in via_direct:
        assert torch.equal(via_spec[i].dec, via_direct[i].dec)
        assert torch.equal(via_spec[i].bits, via_direct[i].bits)

    # Spectral, dec_quant="int8": matches int8_decoder_roundtrip applied by hand.
    int8_spec = CacheCodecSpec(
        arm="spectral",
        pre_rope=True,
        pack_path=str(path),
        budget=2.5,
        dec_quant="int8",
    )
    via_int8 = load_packs_for_spec(int8_spec)
    for i in via_direct:
        expected_dec = int8_decoder_roundtrip(via_direct[i].dec, via_direct[i].bits)
        assert torch.equal(via_int8[i].dec, expected_dec)
        assert via_int8[i].dec.dtype == via_direct[i].dec.dtype

    # Guard preservation: invalid dec_quant / missing pre_rope / missing pack_path.
    with pytest.raises(AssertionError, match="dec_quant"):
        load_packs_for_spec(
            CacheCodecSpec(
                arm="spectral",
                pre_rope=True,
                pack_path=str(path),
                budget=2.5,
                dec_quant="bf16",
            )
        )
    with pytest.raises(AssertionError, match="pre_rope"):
        load_packs_for_spec(
            CacheCodecSpec(arm="spectral", pack_path=str(path), budget=2.5)
        )
    with pytest.raises(AssertionError, match="pack_path"):
        load_packs_for_spec(CacheCodecSpec(arm="spectral", pre_rope=True))


def test_skeptic_charge_v2_hand_computed():
    """v2 arithmetic pinned exactly; defaults reproduce v1 bit-exactly."""
    tiers7 = (0, 2, 3, 4, 5, 6, 8)
    # c_used == C reproduces the old value exactly (continuity pin).
    assert skeptic_charge(1024, 32768, tiers7, c_used=1024) == skeptic_charge(
        1024, 32768, tiers7
    )
    assert skeptic_charge(1024, 32768, tiers7) == 0.500091552734375  # v1, hand
    # Hand-computed v2 fp16 case: 16*400/8192 + 3/8192.
    assert skeptic_charge(1024, 8192, tiers7, c_used=400) == 0.7816162109375
    # Hand-computed v2 int8 case: 8*400/8192 + 16*400/(8192*1024) + 3/8192.
    assert (
        skeptic_charge(1024, 8192, tiers7, c_used=400, dec_bits=8.0)
        == 0.391754150390625
    )


def test_spectral_payload_bpe_v2():
    import dataclasses as _dc

    from bmx.cache.spectral import spectral_payload_bpe

    M, _ = _spiked_keys(S=256, C=8, seed=0)
    Wh, Wh_inv = identity_whitener(8)
    pack = fit_spectral_pack(M, Wh, Wh_inv, 2.0, group=4)
    pack = _dc.replace(pack, bits=torch.tensor([8, 4, 2, 0, 0, 0, 0, 0]))
    assert pack.c_used == 3
    # v2 = mean(bits) + scale_bits(4) * 3/8 = 1.75 + 4.0*0.375 = 3.25 (v1 was 5.75).
    assert spectral_payload_bpe(pack) == 3.25
    # spectral_quantize returns the same payload-v2 number.
    _, bpe = spectral_quantize(M, pack)
    assert bpe == 3.25


def test_dropped_decoder_columns_never_read():
    """THE license for charging C×C_used: mutating dec columns of zero-bit dirs
    must not change the reconstruction by a single bit."""
    M, _ = _spiked_keys(S=256, C=64, seed=1)
    Wh, Wh_inv = identity_whitener(64)
    pack = fit_spectral_pack(M, Wh, Wh_inv, 1.5, group=16)  # low budget => dropped dirs
    assert pack.c_used < 64, "fixture must produce zero-bit dirs"
    M_hat_ref, _ = spectral_quantize(M, pack)
    import dataclasses as _dc

    dec_mut = pack.dec.clone()
    dec_mut[:, pack.bits == 0] = torch.randn_like(dec_mut[:, pack.bits == 0])
    M_hat_mut, _ = spectral_quantize(M, _dc.replace(pack, dec=dec_mut))
    assert torch.equal(M_hat_ref, M_hat_mut)


def test_int8_decoder_roundtrip():
    from bmx.cache.spectral import int8_decoder_roundtrip

    torch.manual_seed(0)
    dec = torch.randn(32, 32)
    bits = torch.tensor([3] * 20 + [0] * 12)
    dec_rt = int8_decoder_roundtrip(dec, bits)
    used = bits != 0
    # Unused columns untouched; used columns within one int8 step of source.
    assert torch.equal(dec_rt[:, ~used], dec[:, ~used])
    step = dec[:, used].abs().amax(dim=0) / 127.0
    assert (dec_rt[:, used] - dec[:, used]).abs().amax(dim=0).le(step + 1e-6).all()
    assert torch.equal(dec_rt, int8_decoder_roundtrip(dec, bits))  # deterministic


def test_pack_from_basis_lam_alloc_default_unchanged():
    import torch

    from bmx.cache.spectral import (
        fit_spectral_basis,
        identity_whitener,
        pack_from_basis,
    )

    g = torch.Generator().manual_seed(0)
    M = torch.randn(128, 16, generator=g)
    Ih, Ih_inv = identity_whitener(16)
    basis = fit_spectral_basis(M, Ih, Ih_inv)
    default = pack_from_basis(basis, 2.5, group=16)
    explicit_none = pack_from_basis(basis, 2.5, group=16, lam_alloc=None)
    own_lam = pack_from_basis(basis, 2.5, group=16, lam_alloc=basis.lam64)
    assert torch.equal(default.bits, explicit_none.bits)
    assert torch.equal(default.bits, own_lam.bits)


def test_basis_alloc_moment_matches_projection_variance():
    import torch

    from bmx.cache.spectral import (
        basis_alloc_moment,
        fit_spectral_basis,
        identity_whitener,
    )

    g = torch.Generator().manual_seed(0)
    M_fit = torch.randn(64, 8, generator=g)
    M_alloc = torch.randn(96, 8, generator=g)
    Ih, Ih_inv = identity_whitener(8)
    basis = fit_spectral_basis(M_fit, Ih, Ih_inv)
    lam = basis_alloc_moment(basis, M_alloc)
    Y = M_alloc.double() @ basis.enc.double()
    ref = (Y**2).mean(dim=0)
    assert lam.dtype == torch.float64 and lam.shape == (8,)
    assert torch.allclose(lam, ref, rtol=1e-10, atol=1e-12)
    assert (lam >= 0).all()


@pytest.mark.parametrize("b", [2, 3, 4, 5, 6, 8])
def test_tier_container_roundtrip(b):
    from bmx.cache.spectral import _pack_tier_codes, _unpack_tier_codes

    qmax = 2 ** (b - 1) - 1
    codes = torch.randint(-qmax - 1, qmax + 1, (7, 128), dtype=torch.int8)
    packed = _pack_tier_codes(codes, b)
    assert packed.dtype in (torch.uint8, torch.int8)
    assert torch.equal(_unpack_tier_codes(packed, b, 128), codes)


def test_spectral_packed_bitwise_matches_spectral_quantize():
    from bmx.cache.spectral import (
        fit_spectral_basis,
        identity_whitener,
        pack_from_basis,
        spectral_dequant_packed,
        spectral_quantize,
        spectral_quantize_packed,
    )

    C, S = 32, 128
    Wh, Wh_inv = identity_whitener(C)
    M, _ = _spiked_keys(S=S, C=C, seed=0)
    basis = fit_spectral_basis(M, Wh, Wh_inv)
    pack = pack_from_basis(basis, 2.5, group=16)
    ref, ref_bpe = spectral_quantize(M, pack, mse_scale=True)
    packed, bpe = spectral_quantize_packed(M, pack, mse_scale=True)
    assert bpe == ref_bpe
    # Container discipline (the T4 pin at codec level): codes uint8/int8, scales fp32.
    for k, t in packed.items():
        if k.endswith("_codes"):
            assert t.dtype in (torch.uint8, torch.int8), (k, t.dtype)
        else:
            assert k.endswith("_scale") and t.dtype == torch.float32, (k, t.dtype)
    M_hat = spectral_dequant_packed(packed, pack)
    assert torch.equal(M_hat, ref)  # BITWISE — the codec-level parity anchor


def test_pack_from_basis_s_ref_default_inert():
    """s_ref=None / g_table=None reproduce today's allocation bit-exactly
    (the default-inert pin idiom), on both pack_from_basis and
    fit_spectral_pack."""
    from bmx.cache.spectral import fit_spectral_basis, pack_from_basis

    M, _ = _spiked_keys(S=256, C=64, seed=3)
    Wh, Wh_inv = identity_whitener(64)
    basis = fit_spectral_basis(M, Wh, Wh_inv)
    default = pack_from_basis(basis, 2.5)
    explicit = pack_from_basis(basis, 2.5, s_ref=None, g_table=None)
    assert torch.equal(default.bits, explicit.bits)
    p1 = fit_spectral_pack(M, Wh, Wh_inv, 2.5)
    p2 = fit_spectral_pack(M, Wh, Wh_inv, 2.5, s_ref=None, g_table=None)
    assert torch.equal(p1.bits, p2.bits)


def test_pack_from_basis_s_ref_hand_case():
    """Hand-computable charge-aware allocation (math review #2 mechanism).
    lam=(256,16,1,1e-8), tiers (0,2,4), group=16, C=4, s_ref=64 =>
    s = 16/16 + 16*4/64 = 2.0. At budget 3.0 (mean TOTAL charge):
      (4,4,0,0): charge (6+6)/4 = 3.0, D ~ 2.06  <- Lagrangian pick
      (4,4,2,0): charge 4.0 -- infeasible
    Plain at budget 3.0 (mean payload bits) opens THREE directions
    (4,4,4,0). Charge-aware closes the marginal direction: c_used 2 vs 3 --
    the 0<->2 boundary movement the math doc predicts."""
    from bmx.cache.spectral import SpectralBasis, pack_from_basis

    lam64 = torch.tensor([256.0, 16.0, 1.0, 1e-8], dtype=torch.float64)
    eye = torch.eye(4, dtype=torch.float32)
    basis = SpectralBasis(enc=eye, dec=eye.clone(), lam=lam64.float(), lam64=lam64)
    ca = pack_from_basis(basis, 3.0, tiers=(0, 2, 4), group=16, s_ref=64)
    plain = pack_from_basis(basis, 3.0, tiers=(0, 2, 4), group=16)
    assert torch.equal(ca.bits, torch.tensor([4, 4, 0, 0], dtype=torch.int64))
    assert torch.equal(plain.bits, torch.tensor([4, 4, 4, 0], dtype=torch.int64))
    assert ca.c_used == 2 and plain.c_used == 3


def test_s_ref_c_used_monotone():
    """Diagnostic invariant the A-gate pre-registers: c_used decreases as the
    deployment context shortens (bigger per-direction charge)."""
    from bmx.cache.spectral import fit_spectral_basis, pack_from_basis

    M, _ = _spiked_keys(S=512, C=64, seed=6)
    Wh, Wh_inv = identity_whitener(64)
    basis = fit_spectral_basis(M, Wh, Wh_inv)
    c_none = pack_from_basis(basis, 2.5, group=16).c_used
    c_8k = pack_from_basis(basis, 2.5, group=16, s_ref=8192).c_used
    c_1k = pack_from_basis(basis, 2.5, group=16, s_ref=1024).c_used
    assert c_1k <= c_8k <= c_none
    assert c_1k < c_none, "fixture must actually exercise the charge"


def test_query_moment_w_rope_default_inert():
    """w_rope='frozen' default == bare call, bit-exact (the default-inert pin)."""
    g = torch.Generator().manual_seed(11)
    q = torch.randn(4, 16, 8, generator=g)
    S = 32
    theta = torch.linspace(0, 2.0, S).unsqueeze(1) * torch.ones(1, 8)
    cos, sin = theta.cos(), theta.sin()
    bare = query_position_moment(q, cos, sin, h_kv=2, position_stride=8)
    frozen = query_position_moment(
        q, cos, sin, h_kv=2, position_stride=8, w_rope="frozen"
    )
    assert torch.equal(bare, frozen)


def test_query_moment_rotated_matches_explicit_forward_rotation():
    """Value-pin for w_rope='rotated' (math review #3(b)): W must equal
    sum_p (S-p) * R_p (pooled qq^T) R_p^T / sum_p (S-p) with R_p the FORWARD
    rotation (columns apply_rope(e_i)) — no transpose, unlike the frozen
    path's ground-truth test. Duplicated-half cos/sin structure as in real
    RoPE tables."""
    from bmx.cache.rope import apply_rope

    g = torch.Generator().manual_seed(12)
    h, T, d, h_kv, S = 2, 16, 4, 1, 8
    q = torch.randn(h, T, d, generator=g)
    freqs = torch.linspace(0.5, 1.0, d // 2)
    theta = torch.linspace(0.3, 2.0, S).unsqueeze(1) * torch.cat(
        [freqs, freqs]
    ).unsqueeze(0)
    cos, sin = theta.cos(), theta.sin()

    stride = 2
    W = query_position_moment(
        q, cos, sin, h_kv, position_stride=stride, w_rope="rotated"
    )

    q_flat = q.double().reshape(-1, d)
    pooled = q_flat.mT @ q_flat / (h * T)
    positions = list(range(0, S, stride))
    W_expected = torch.zeros(d, d, dtype=torch.float64)
    total = 0.0
    for p in positions:
        basis = torch.eye(d).double().unsqueeze(1)
        Rp_cols = apply_rope(basis, cos[p : p + 1].double(), sin[p : p + 1].double())
        Rp = Rp_cols.squeeze(1).mT  # (d, d), column i = R_p e_i (FORWARD)
        wt = float(S - p)  # triangular: #pairs at offset p ~ S - p
        W_expected += wt * (Rp @ pooled @ Rp.mT)
        total += wt
    W_expected /= total
    assert torch.allclose(W[0], W_expected, atol=1e-10)


def test_query_moment_rotated_null_on_no_rope():
    """gpt2 null control: with sin == 0 the odd (sign-flipping) term vanishes
    and every position contributes the identical qq^T, so frozen == rotated
    up to fp summation order (the reason the A/B must run on a RoPE model)."""
    g = torch.Generator().manual_seed(13)
    q = torch.randn(4, 16, 8, generator=g)
    S = 64
    cos, sin = torch.ones(S, 8), torch.zeros(S, 8)
    frozen = query_position_moment(q, cos, sin, h_kv=2, position_stride=8)
    rotated = query_position_moment(
        q, cos, sin, h_kv=2, position_stride=8, w_rope="rotated"
    )
    assert torch.allclose(frozen, rotated, atol=1e-10)


def test_int8_certificate_analytic_identity():
    """Fit-cache-free identity: on synthetic rows whose code second moment is
    EXACTLY diag(lam), the closed form sum_i lam_i*||enc^T ddec[:,i]||^2 must
    equal the brute-force mean weighted row norm mean_s ||enc^T ddec y_s||^2.
    Non-identity diagonal whitener so enc != dec (the mutant-visible regime,
    same rationale as test_rank_overlap_pinned_to_dec_not_enc)."""
    import math as _math

    from bmx.cache.spectral import (
        fit_spectral_pack,
        int8_decoder_certificate,
        int8_decoder_roundtrip,
    )

    C, S = 16, 200
    scales = torch.linspace(0.2, 5.0, C, dtype=torch.float64)
    Wh, Wh_inv = torch.diag(scales), torch.diag(1.0 / scales)
    g = torch.Generator().manual_seed(0)
    M = torch.randn(S, C, generator=g)
    pack = fit_spectral_pack(M, Wh, Wh_inv, 2.5, tiers=(0, 2, 4), group=8)
    assert 0 < pack.c_used < C  # fixture must have both used and dropped dirs

    cert = int8_decoder_certificate(pack)
    for key in ("added", "payload", "noise_to_signal", "implied_rel_degradation"):
        assert key in cert and cert[key] >= 0.0

    # Brute force on rows with exact code moment diag(lam):
    lam = pack.lam.double()
    G = torch.randn(4 * C, C, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(G)  # (4C, C), Q^T Q = I
    Z = _math.sqrt(4 * C) * Q @ torch.diag(lam.clamp_min(0).sqrt())
    ddec = int8_decoder_roundtrip(pack.dec, pack.bits).double() - pack.dec.double()
    brute = float(((Z @ ddec.mT @ pack.enc.double()) ** 2).sum() / (4 * C))
    assert abs(cert["added"] - brute) < 1e-9 * max(brute, 1e-12)

    # Payload closed form pinned against the independent expression.
    payload = float((lam * torch.pow(4.0, -pack.bits.double())).sum())
    assert abs(cert["payload"] - payload) < 1e-12
    assert (
        abs(
            cert["implied_rel_degradation"]
            - cert["added"] / (cert["payload"] + cert["added"])
        )
        < 1e-15
    )


def test_int8_certificate_dropped_columns_and_determinism():
    """Mutating dropped dec columns cannot change the certificate (the
    test_dropped_decoder_columns_never_read license extends to it), and the
    certificate is deterministic."""
    import dataclasses as _dc

    from bmx.cache.spectral import fit_spectral_pack, int8_decoder_certificate

    M, _ = _spiked_keys(S=256, C=64, seed=1)
    Wh, Wh_inv = identity_whitener(64)
    pack = fit_spectral_pack(M, Wh, Wh_inv, 1.5, group=16)
    assert pack.c_used < 64
    c1 = int8_decoder_certificate(pack)
    dec_mut = pack.dec.clone()
    dec_mut[:, pack.bits == 0] = torch.randn_like(dec_mut[:, pack.bits == 0])
    c2 = int8_decoder_certificate(_dc.replace(pack, dec=dec_mut))
    assert c1 == c2 == int8_decoder_certificate(pack)


def test_int8_certificate_tiered_endpoints_and_charge():
    """K4 math-actions 6b: int8_decoder_certificate_tiered's two boundary
    cases, plus mixed_dec_charge/effective_dec_bits pinned against
    skeptic_charge at both endpoints of the int8 coverage fraction.

    tier_threshold >= max(pack.bits) must reproduce the blanket certificate
    bit-for-bit (nothing gets zeroed -- same ddec, same projection).
    tier_threshold below every used tier must give added == 0.0 exactly
    (every used column zeroed before the enc^T ddec projection)."""
    from bmx.cache.spectral import (
        effective_dec_bits,
        fit_spectral_pack,
        int8_decoder_certificate,
        int8_decoder_certificate_tiered,
        mixed_dec_charge,
        skeptic_charge,
    )

    C, S = 16, 200
    scales = torch.linspace(0.2, 5.0, C, dtype=torch.float64)
    Wh, Wh_inv = torch.diag(scales), torch.diag(1.0 / scales)
    g = torch.Generator().manual_seed(0)
    M = torch.randn(S, C, generator=g)
    pack = fit_spectral_pack(M, Wh, Wh_inv, 2.5, tiers=(0, 2, 3, 4, 5, 6, 8), group=8)
    used_tiers = sorted(set(int(b) for b in pack.bits.tolist() if b != 0))
    assert len(used_tiers) >= 2  # fixture must span more than one used tier

    blanket = int8_decoder_certificate(pack)

    # T = max(bits): reproduces the blanket certificate exactly.
    top = int8_decoder_certificate_tiered(pack, max(used_tiers))
    assert top["c_int8"] == top["c_used"] == pack.c_used
    assert top["frac_int8"] == 1.0
    for key in ("added", "payload", "noise_to_signal", "implied_rel_degradation"):
        assert abs(top[key] - blanket[key]) < 1e-12 * max(abs(blanket[key]), 1e-12)

    # T below the lowest used tier: zero added distortion, zero int8 coverage.
    bottom = int8_decoder_certificate_tiered(pack, used_tiers[0] - 1)
    assert bottom["added"] == 0.0
    assert bottom["c_int8"] == 0
    assert bottom["frac_int8"] == 0.0
    assert bottom["noise_to_signal"] == 0.0
    assert bottom["implied_rel_degradation"] == 0.0
    assert bottom["payload"] == blanket["payload"]  # payload never depends on ddec

    # mixed_dec_charge/effective_dec_bits pinned at both coverage endpoints.
    S_ref = 4096
    cu = pack.c_used
    sc_fp16 = skeptic_charge(C, S_ref, pack.tiers, c_used=cu, dec_bits=16.0)
    sc_int8 = skeptic_charge(C, S_ref, pack.tiers, c_used=cu, dec_bits=8.0)
    mc_none = mixed_dec_charge(C, S_ref, pack.tiers, c_used=cu, c_int8=0)
    mc_all = mixed_dec_charge(C, S_ref, pack.tiers, c_used=cu, c_int8=cu)
    assert abs(mc_none - sc_fp16) < 1e-15
    assert abs(mc_all - sc_int8) < 1e-15
    assert effective_dec_bits(C, cu, 0) == 16.0
    assert abs(effective_dec_bits(C, cu, cu) - (8.0 + 16.0 / C)) < 1e-12


def test_roundtrip_tier_threshold_blanket_equiv():
    """K4 local-levers Task 1: tier_threshold=None (default) and tier_threshold
    == max used tier (blanket, here 8, the top of the standard 7-tier grid)
    must both reproduce today's un-gated int8_decoder_roundtrip bit-for-bit."""
    from bmx.cache.spectral import int8_decoder_roundtrip

    torch.manual_seed(0)
    dec = torch.randn(32, 32)
    bits = torch.tensor([2, 3, 4, 5, 6, 8] * 5 + [0, 0])
    blanket = int8_decoder_roundtrip(dec, bits)
    default_none = int8_decoder_roundtrip(dec, bits, tier_threshold=None)
    thr_8 = int8_decoder_roundtrip(dec, bits, tier_threshold=8)
    assert torch.equal(blanket, default_none)
    assert torch.equal(blanket, thr_8)


def test_roundtrip_tier_gated_masks_columns():
    """Columns with bits > T stay fp32-as-loaded (untouched); columns with
    0 < bits <= T match the blanket roundtrip exactly for those columns;
    zero-bit columns are always untouched."""
    from bmx.cache.spectral import int8_decoder_roundtrip

    torch.manual_seed(1)
    dec = torch.randn(16, 16)
    bits = torch.tensor([2, 3, 4, 5, 6, 8, 0, 0, 2, 3, 4, 5, 6, 8, 0, 0])
    T = 5
    gated = int8_decoder_roundtrip(dec, bits, tier_threshold=T)
    blanket = int8_decoder_roundtrip(dec, bits)

    above_t = (bits > T) & (bits != 0)
    at_or_below_t = (bits != 0) & (bits <= T)
    zero_bit = bits == 0

    assert torch.equal(gated[:, above_t], dec[:, above_t])  # untouched (fp32-as-loaded)
    assert torch.equal(gated[:, at_or_below_t], blanket[:, at_or_below_t])
    assert torch.equal(gated[:, zero_bit], dec[:, zero_bit])


def test_dec_quant_threshold_parse():
    """dec_quant_threshold: 'fp32'->None, 'int8'->8, 'int8_t{T}'->T (2<=T<=8);
    anything else (including out-of-range T) asserts."""
    from bmx.cache.spectral import dec_quant_threshold

    assert dec_quant_threshold("fp32") is None
    assert dec_quant_threshold("int8") == 8
    assert dec_quant_threshold("int8_t5") == 5
    assert dec_quant_threshold("int8_t2") == 2
    assert dec_quant_threshold("int8_t8") == 8

    with pytest.raises(AssertionError):
        dec_quant_threshold("int8_t1")
    with pytest.raises(AssertionError):
        dec_quant_threshold("int9")
    with pytest.raises(AssertionError):
        dec_quant_threshold("bf16")


def test_load_packs_for_spec_tier_gated(tmp_path):
    """A tier-gated dec_quant ('int8_t{T}') spec: columns with bits > T stay
    bit-identical to the file; columns with 0 < bits <= T are roundtripped
    (matching int8_decoder_roundtrip with tier_threshold=T applied by hand)."""
    from bmx.cache.spectral import (
        fit_spectral_basis,
        identity_whitener,
        int8_decoder_roundtrip,
        load_packs,
        save_pack_file,
    )
    from bmx.cache.specs import CacheCodecSpec
    from bmx.cache.spectral import load_packs_for_spec

    C = 32
    Wh, Wh_inv = identity_whitener(C)
    M, _ = _spiked_keys(S=256, C=C, seed=0)
    bases = {0: fit_spectral_basis(M, Wh, Wh_inv)}
    path = tmp_path / "packs.safetensors"
    save_pack_file(path, bases, budgets=(2.5,), group=16, meta={"model": "tiny"})

    via_direct = load_packs(path, 2.5)
    pack = via_direct[0]
    used_tiers = sorted(set(int(b) for b in pack.bits.tolist() if b != 0))
    assert len(used_tiers) >= 2, "fixture must span more than one used tier"
    T = used_tiers[0]  # a real, low threshold so both sides of the gate exercise

    tiered_spec = CacheCodecSpec(
        arm="spectral",
        pre_rope=True,
        pack_path=str(path),
        budget=2.5,
        dec_quant=f"int8_t{T}",
    )
    via_tiered = load_packs_for_spec(tiered_spec)
    expected_dec = int8_decoder_roundtrip(pack.dec, pack.bits, tier_threshold=T)
    assert torch.equal(via_tiered[0].dec, expected_dec)

    above_t = (pack.bits > T) & (pack.bits != 0)
    assert above_t.any(), "fixture must have columns above the threshold"
    assert torch.equal(via_tiered[0].dec[:, above_t], pack.dec[:, above_t])


# ---------------------------------------------------------------------------
# per_layer_tier_thresholds / int8_tl (K4 estimation-levers Task 3)
# ---------------------------------------------------------------------------


def _two_layer_packs_distinct_spectra():
    """Two packs deliberately built so their certificates diverge across the
    tier grid: layer 0's spikes are far stronger (steeper spectrum decay ->
    the certificate's added/payload ratio is smaller at any given T, so it
    should certify a HIGHER T_ℓ) than layer 1's near-flat spectrum (weak
    spikes barely above noise -> more of the certificate mass sits in the
    low tiers, so it should certify a LOWER T_ℓ, mirroring the uniform-T=5
    "layer 1 is the worst layer" finding this task is about)."""
    C = 32
    Wh, Wh_inv = identity_whitener(C)
    strong, _ = _spiked_keys(S=512, C=C, seed=0, spike_std=(80.0, 80.0))
    weak, _ = _spiked_keys(S=512, C=C, seed=1, spike_std=(3.0, 3.0))
    pack_strong = fit_spectral_pack(strong, Wh, Wh_inv, budget=2.5, group=8)
    pack_weak = fit_spectral_pack(weak, Wh, Wh_inv, budget=2.5, group=8)
    return {0: pack_strong, 1: pack_weak}


def test_per_layer_tier_thresholds_each_passes_bar_next_tier_fails_or_max():
    """Each returned T_ℓ must itself pass the bar; the next grid tier up must
    either fail the bar or T_ℓ must already be the grid max (8) -- i.e. the
    returned T really is the LARGEST passing tier, not just A passing tier."""
    from bmx.cache.spectral import (
        TIER_THRESHOLD_GRID,
        int8_decoder_certificate_tiered,
        per_layer_tier_thresholds,
    )

    packs = _two_layer_packs_distinct_spectra()
    bar = 0.05
    t_map = per_layer_tier_thresholds(packs, bar=bar)
    assert set(t_map) == set(packs)

    for layer_i, T in t_map.items():
        pack = packs[layer_i]
        if T == 0:
            # Even the smallest grid tier must fail (or the layer has no used
            # tiers at all -- not the case for this fixture's budget).
            assert (
                int8_decoder_certificate_tiered(pack, TIER_THRESHOLD_GRID[0])[
                    "implied_rel_degradation"
                ]
                > bar
            )
            continue
        cert_here = int8_decoder_certificate_tiered(pack, T)
        assert cert_here["implied_rel_degradation"] <= bar
        idx = TIER_THRESHOLD_GRID.index(T)
        if idx + 1 < len(TIER_THRESHOLD_GRID):
            next_t = TIER_THRESHOLD_GRID[idx + 1]
            cert_next = int8_decoder_certificate_tiered(pack, next_t)
            assert cert_next["implied_rel_degradation"] > bar


def test_per_layer_tier_thresholds_fixture_forces_different_t_per_layer():
    """The fixture is deliberately built so the two layers' certified T_ℓ
    differ -- the case this task exists to handle (uniform T=5 binds on the
    single worst layer; per-layer should NOT collapse to one shared value)."""
    from bmx.cache.spectral import per_layer_tier_thresholds

    packs = _two_layer_packs_distinct_spectra()
    t_map = per_layer_tier_thresholds(packs)
    assert t_map[0] != t_map[1], f"fixture must force distinct T_ℓ; got {t_map}"


def test_per_layer_tier_thresholds_monotone_in_bar():
    """A larger bar can only relax the certificate, never tighten it: T_ℓ at
    a larger bar must be pointwise >= T_ℓ at a smaller bar, for every layer."""
    from bmx.cache.spectral import per_layer_tier_thresholds

    packs = _two_layer_packs_distinct_spectra()
    tight = per_layer_tier_thresholds(packs, bar=0.01)
    loose = per_layer_tier_thresholds(packs, bar=0.20)
    for layer_i in packs:
        assert loose[layer_i] >= tight[layer_i]


def test_per_layer_tier_thresholds_zero_when_even_smallest_tier_fails():
    """An unreasonably strict bar (effectively zero) must drive every layer's
    T_ℓ to 0 -- no int8 for that layer -- since no nonzero-tier certificate
    can be exactly zero on a fixture with real spikes."""
    from bmx.cache.spectral import per_layer_tier_thresholds

    packs = _two_layer_packs_distinct_spectra()
    t_map = per_layer_tier_thresholds(packs, bar=0.0)
    assert all(T == 0 for T in t_map.values())


def test_dec_quant_threshold_int8_tl_is_sentinel_and_others_unchanged():
    """dec_quant_threshold('int8_tl') == PER_LAYER_TIER_SENTINEL (-1); every
    prior form ('fp32'/'int8'/'int8_t{T}') is byte-for-byte unchanged."""
    from bmx.cache.spectral import PER_LAYER_TIER_SENTINEL, dec_quant_threshold

    assert PER_LAYER_TIER_SENTINEL == -1
    assert dec_quant_threshold("int8_tl") == -1
    assert dec_quant_threshold("fp32") is None
    assert dec_quant_threshold("int8") == 8
    assert dec_quant_threshold("int8_t5") == 5
    assert dec_quant_threshold("int8_t2") == 2
    assert dec_quant_threshold("int8_t8") == 8
    with pytest.raises(AssertionError):
        dec_quant_threshold("int8_t1")
    with pytest.raises(AssertionError):
        dec_quant_threshold("int9")


def test_load_packs_for_spec_int8_tl_applies_per_layer_thresholds(tmp_path):
    """dec_quant='int8_tl' materialization: each layer's decoder is
    roundtripped at ITS OWN T_ℓ (per_layer_tier_thresholds applied to the
    just-loaded packs), not one shared threshold -- verified against the
    fixture built to force different T_ℓ per layer, and hand-computed via
    int8_decoder_roundtrip at each layer's own T_ℓ."""
    from bmx.cache.spectral import (
        int8_decoder_roundtrip,
        load_packs,
        load_packs_for_spec,
        per_layer_tier_thresholds,
        save_pack_file,
        fit_spectral_basis,
    )
    from bmx.cache.specs import CacheCodecSpec

    C = 32
    Wh, Wh_inv = identity_whitener(C)
    strong, _ = _spiked_keys(S=512, C=C, seed=0, spike_std=(80.0, 80.0))
    weak, _ = _spiked_keys(S=512, C=C, seed=1, spike_std=(3.0, 3.0))
    bases = {
        0: fit_spectral_basis(strong, Wh, Wh_inv),
        1: fit_spectral_basis(weak, Wh, Wh_inv),
    }
    path = tmp_path / "packs_tl.safetensors"
    save_pack_file(path, bases, budgets=(2.5,), group=8, meta={"model": "tiny"})

    raw_packs = load_packs(path, 2.5)
    t_map = per_layer_tier_thresholds(raw_packs)
    assert t_map[0] != t_map[1], "fixture must force distinct T_ℓ (see module test)"

    tl_spec = CacheCodecSpec(
        arm="spectral",
        pre_rope=True,
        pack_path=str(path),
        budget=2.5,
        dec_quant="int8_tl",
    )
    via_tl = load_packs_for_spec(tl_spec)

    for layer_i, pack in raw_packs.items():
        T = t_map[layer_i]
        if T == 0:
            assert torch.equal(via_tl[layer_i].dec, pack.dec)
        else:
            expected = int8_decoder_roundtrip(pack.dec, pack.bits, tier_threshold=T)
            assert torch.equal(via_tl[layer_i].dec, expected)
    # And the two layers' materialized decoders must actually differ in how
    # they were gated (not both landing on the same threshold by accident).
    assert not torch.equal(
        int8_decoder_roundtrip(
            raw_packs[0].dec, raw_packs[0].bits, tier_threshold=t_map[0]
        ),
        int8_decoder_roundtrip(
            raw_packs[0].dec, raw_packs[0].bits, tier_threshold=t_map[1]
        ),
    )


def _spiked_matrix(S, C, seed, spike_std):
    """Minimal spiked-key generator with an explicit (S, C, seed, spike_std)
    signature (unlike _spiked_keys, whose spike_std is a fixed pair) --
    needed to hunt a certificate BOUNDARY case below."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(C, 2, generator=g)
    dirs, _ = torch.linalg.qr(raw)
    z = torch.randn(S, 2, generator=g) * torch.tensor([spike_std, spike_std])
    noise = torch.randn(S, C, generator=g)
    return z @ dirs.mT + noise


def _pristine_vs_postroundtrip_divergent_packs():
    """A REAL boundary case (2026-07-24 fix-wave regression fixture, found by
    grid search over C/budget/spike/seed): layer 0's PRISTINE certificate at
    T=8 sits at implied_rel_degradation ~5.01% (just OVER the 5% bar, so
    per_layer_tier_thresholds certifies T=6 for it) -- but re-running the
    SAME certificate on layer 0's decoder AFTER it has been int8-roundtripped
    at T=6 gives ~4.93% at T=8 (just UNDER the bar), because
    int8_decoder_roundtrip is near-idempotent (re-roundtripping an
    already-int8-stored column perturbs it negligibly further, shrinking
    ddec and hence the certificate's `added` term). A naive re-derivation of
    per_layer_tier_thresholds from the POST-roundtrip pack would silently
    promote layer 0 from T=6 to T=8 -- exactly the bug this fix wave closes.
    Layer 1 is an unrelated well-behaved control (T=8 either way, sanity
    check that the fixture isn't accidentally forcing divergence everywhere).
    """
    C = 64
    Wh, Wh_inv = identity_whitener(C)
    M0 = _spiked_matrix(300, C, seed=4, spike_std=40.0)  # the boundary layer
    M1 = _spiked_matrix(300, C, seed=1, spike_std=10.0)  # control layer
    pack0 = fit_spectral_pack(M0, Wh, Wh_inv, budget=2.5, group=8)
    pack1 = fit_spectral_pack(M1, Wh, Wh_inv, budget=2.5, group=8)
    return {0: pack0, 1: pack1}


def test_pristine_vs_postroundtrip_certificate_diverges_fixture_sanity():
    """Sanity-pin the fixture itself (independent of any load_packs_for_spec
    code path): pristine layer 0 certifies T=6; re-deriving the map from a
    pack ALREADY roundtripped at T=6 wrongly promotes it to T=8. If this
    test ever fails, the fixture stopped exhibiting the boundary case and
    the regression tests below would silently stop covering the bug."""
    from bmx.cache.spectral import (
        int8_decoder_roundtrip,
        per_layer_tier_thresholds,
    )

    packs = _pristine_vs_postroundtrip_divergent_packs()
    t_map_pristine = per_layer_tier_thresholds(packs)
    assert t_map_pristine[0] == 6, t_map_pristine
    assert t_map_pristine[1] == 8, t_map_pristine

    pack0 = packs[0]
    rt_dec = int8_decoder_roundtrip(pack0.dec, pack0.bits, tier_threshold=6)
    rt_pack0 = dataclasses.replace(pack0, dec=rt_dec)
    t_map_post = per_layer_tier_thresholds({0: rt_pack0, 1: packs[1]})
    assert t_map_post[0] == 8, (
        f"fixture must exhibit the near-idempotency divergence; got {t_map_post}"
    )
    assert t_map_post[0] != t_map_pristine[0]


def test_load_packs_for_spec_int8_tl_sets_dec_tier_from_pristine_map(tmp_path):
    """K4 estimation-levers Task 3 fix wave (P1 bug): dec_quant='int8_tl'
    materialization must set each layer's SpectralPack.dec_tier to the
    PRISTINE per_layer_tier_thresholds value -- on the boundary fixture,
    layer 0's dec_tier must be 6, never 8 (the value a re-derivation from the
    already-roundtripped decoder would wrongly produce). Also pins dec_tier
    for 'int8' (8 on every layer) and 'fp32' (None on every layer)."""
    from bmx.cache.spectral import (
        fit_spectral_basis,
        int8_decoder_roundtrip,
        load_packs_for_spec,
        per_layer_tier_thresholds,
        save_pack_file,
    )
    from bmx.cache.specs import CacheCodecSpec

    C = 64
    Wh, Wh_inv = identity_whitener(C)
    M0 = _spiked_matrix(300, C, seed=4, spike_std=40.0)
    M1 = _spiked_matrix(300, C, seed=1, spike_std=10.0)
    bases = {
        0: fit_spectral_basis(M0, Wh, Wh_inv),
        1: fit_spectral_basis(M1, Wh, Wh_inv),
    }
    path = tmp_path / "boundary_packs.safetensors"
    save_pack_file(path, bases, budgets=(2.5,), group=8, meta={"model": "boundary"})

    from bmx.cache.spectral import load_packs

    pristine = load_packs(path, 2.5)
    t_map = per_layer_tier_thresholds(pristine)
    assert t_map == {0: 6, 1: 8}, "boundary fixture drifted; re-tune spike/seed"

    tl_spec = CacheCodecSpec(
        arm="spectral",
        pre_rope=True,
        pack_path=str(path),
        budget=2.5,
        dec_quant="int8_tl",
    )
    via_tl = load_packs_for_spec(tl_spec)
    assert via_tl[0].dec_tier == 6  # NOT 8 -- the bug this test guards against
    assert via_tl[1].dec_tier == 8
    # And the materialized decoder itself must match the PRISTINE-T6 roundtrip,
    # not a T8 roundtrip (dec_tier is not just a label -- it must agree with
    # what was actually written into .dec).
    expected_dec0 = int8_decoder_roundtrip(
        pristine[0].dec, pristine[0].bits, tier_threshold=6
    )
    assert torch.equal(via_tl[0].dec, expected_dec0)

    # 'int8' (blanket): dec_tier == 8 on every layer.
    blanket_spec = CacheCodecSpec(
        arm="spectral",
        pre_rope=True,
        pack_path=str(path),
        budget=2.5,
        dec_quant="int8",
    )
    via_blanket = load_packs_for_spec(blanket_spec)
    assert via_blanket[0].dec_tier == 8 and via_blanket[1].dec_tier == 8

    # 'int8_t5' (single shared threshold): dec_tier == 5 on every layer.
    t5_spec = CacheCodecSpec(
        arm="spectral",
        pre_rope=True,
        pack_path=str(path),
        budget=2.5,
        dec_quant="int8_t5",
    )
    via_t5 = load_packs_for_spec(t5_spec)
    assert via_t5[0].dec_tier == 5 and via_t5[1].dec_tier == 5

    # 'fp32': dec_tier is None on every layer (no roundtrip at all).
    fp32_spec = CacheCodecSpec(
        arm="spectral",
        pre_rope=True,
        pack_path=str(path),
        budget=2.5,
        dec_quant="fp32",
    )
    via_fp32 = load_packs_for_spec(fp32_spec)
    assert via_fp32[0].dec_tier is None and via_fp32[1].dec_tier is None

    # And a freshly-loaded pack (never through load_packs_for_spec) also
    # defaults to dec_tier=None -- pristine packs are never "secretly" int8.
    assert pristine[0].dec_tier is None and pristine[1].dec_tier is None


# ---------------------------------------------------------------------------
# jensen_gap_report (K4 local-levers Task 4 — determinant-Jensen Gate-A anchor)
# ---------------------------------------------------------------------------


def _random_psd(C: int, generator: torch.Generator) -> torch.Tensor:
    A = torch.randn(C, C, generator=generator, dtype=torch.float64)
    return A @ A.mT + 1e-6 * torch.eye(C, dtype=torch.float64)


def test_jensen_gap_identical_moments_r_pred_one():
    """Identical moments: no heterogeneity, the Jensen gap must vanish
    exactly (up to fp round-off)."""
    g = torch.Generator().manual_seed(0)
    M = _random_psd(12, g)
    report = jensen_gap_report([M, M.clone(), M.clone()])
    assert abs(report["r_pred"] - 1.0) < 1e-12
    assert abs(report["log_gap"]) < 1e-12
    assert report["n_seq"] == 3
    assert report["n_clamped"] == 0


def test_jensen_gap_random_psd_mixture_minkowski_bound():
    """A genuinely heterogeneous PSD mixture must satisfy the Minkowski bound
    r_pred <= 1 (with a tight numerical tolerance, not a loose sanity check)."""
    g = torch.Generator().manual_seed(1)
    moments = [_random_psd(10, g) for _ in range(8)]
    report = jensen_gap_report(moments)
    assert report["r_pred"] <= 1.0 + 1e-12
    # A genuinely heterogeneous mixture should show a real (not vanishing) gap.
    assert report["r_pred"] < 1.0 - 1e-6
    assert report["log_gap"] > 0.0


def test_jensen_gap_two_atom_diagonal_closed_form():
    """Hand-computable closed form: two diagonal moments diag(a), diag(b).
    gm(diag(a)) = GM(a) exactly (eigenvalues of a diagonal matrix are its
    entries), so r_pred = mean(GM(a), GM(b)) / GM((a+b)/2) by direct
    substitution into the definition — computed independently here via
    Python floats, not by calling back into the geometric-mean machinery."""
    import math

    a = torch.tensor([1.0, 4.0, 9.0, 16.0], dtype=torch.float64)
    b = torch.tensor([2.0, 2.0, 50.0, 0.5], dtype=torch.float64)
    Ma, Mb = torch.diag(a), torch.diag(b)
    report = jensen_gap_report([Ma, Mb])

    def gm(vals):
        return math.exp(sum(math.log(v) for v in vals) / len(vals))

    gm_a, gm_b = gm(a.tolist()), gm(b.tolist())
    pooled = [(ai + bi) / 2 for ai, bi in zip(a.tolist(), b.tolist())]
    expected_r_pred = ((gm_a + gm_b) / 2) / gm(pooled)
    expected_log_gap = math.log(gm(pooled)) - (math.log(gm_a) + math.log(gm_b)) / 2

    assert abs(report["r_pred"] - expected_r_pred) < 1e-10
    assert abs(report["log_gap"] - expected_log_gap) < 1e-10
    assert abs(report["gm_pool"] - gm(pooled)) < 1e-10
    assert abs(report["mean_gm_seq"] - (gm_a + gm_b) / 2) < 1e-10


def test_jensen_gap_non_psd_input_asserts():
    """A matrix with a genuinely negative eigenvalue (beyond the PSD-guard's
    fp-round-off slack) must assert, not silently clamp."""
    M = torch.diag(torch.tensor([1.0, -1.0, 2.0], dtype=torch.float64))
    with pytest.raises(AssertionError):
        jensen_gap_report([M])


def test_jensen_gap_clamps_near_zero_eigenvalues_and_counts_them():
    """A moment with a (numerically) zero eigenvalue is clamped to the tiny
    floor rather than rejected (it's within the PSD-guard's tolerance), and
    the clamp is counted in n_clamped -- once per per-sequence pass the zero
    eigenvalue is seen in, plus once more for the pooled moment (which is
    itself singular here since the single input moment has a zero eigenvalue
    exactly on that axis)."""
    import math

    M = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 0.0], dtype=torch.float64))
    report = jensen_gap_report([M])
    assert report["n_clamped"] == 2
    # gm must not be NaN/inf: the clamp floor keeps the log finite.
    assert report["gm_pool"] > 0.0 and math.isfinite(report["gm_pool"])


def test_jensen_gap_requires_square_and_matching_shapes():
    with pytest.raises(AssertionError):
        jensen_gap_report([torch.eye(3, 4, dtype=torch.float64)])
    with pytest.raises(AssertionError):
        jensen_gap_report(
            [torch.eye(3, dtype=torch.float64), torch.eye(4, dtype=torch.float64)]
        )


def test_jensen_gap_requires_nonempty():
    with pytest.raises(AssertionError):
        jensen_gap_report([])


# ---------------------------------------------------------------------------
# jensen_gap_report Wishart log-det debiasing (n_rows-aware follow-up)
# ---------------------------------------------------------------------------


def test_jensen_gap_n_rows_none_bit_identical_regression_pin():
    """Default n_rows=None must reproduce EXACTLY the pre-follow-up return
    dict: the same 6 keys (gm_pool, mean_gm_seq, r_pred, log_gap, n_seq,
    n_clamped), none of the new debiasing keys, and the SAME values a fixed
    seed produced before this follow-up (hand-verified against the identical
    -moments closed form: r_pred == 1.0, log_gap == 0.0)."""
    pre_follow_up_keys = {
        "gm_pool",
        "mean_gm_seq",
        "r_pred",
        "log_gap",
        "n_seq",
        "n_clamped",
    }
    g = torch.Generator().manual_seed(3)
    M = _random_psd(9, g)
    report = jensen_gap_report([M, M.clone(), M.clone()])
    assert set(report) == pre_follow_up_keys
    assert abs(report["r_pred"] - 1.0) < 1e-12
    assert abs(report["log_gap"]) < 1e-12


def test_jensen_gap_debiasing_closed_form_wishart_correction():
    """Synthetic iid Gaussian rows, known population covariance = identity
    (population gm = 1 exactly). At n=48, C=32 (n < 2C, a real small-sample
    regime) the RAW sample gm is a substantial underestimate of the
    population gm; averaging the DEBIASED gm over many trials must land much
    closer to 1.0 than the raw estimate does. Seeded, generous tolerance,
    200 trials (fast: C=32 eigh is cheap)."""
    import math

    torch.manual_seed(11)
    C, n, trials = 32, 48, 200
    raw_gms = []
    for _ in range(trials):
        X = torch.randn(n, C, dtype=torch.float64)
        S = X.mT @ X / n
        report = jensen_gap_report([S], n_rows=[n])
        raw_gms.append(report["mean_gm_seq"])
    raw_mean = sum(raw_gms) / trials
    assert abs(raw_mean - 1.0) > 0.2, "fixture must show a REAL raw bias to debias away"

    # Closed-form Bartlett/digamma correction, computed independently here.
    i = torch.arange(1, C + 1, dtype=torch.float64)
    psi = torch.special.digamma((n - i + 1) / 2.0)
    b_expected = float((psi + math.log(2.0 / n)).mean())
    bias_factor_expected = math.exp(b_expected)

    # bias_factor_seq from the report (single-moment case: mean_s b_s == b_s).
    single_report = jensen_gap_report([torch.eye(C, dtype=torch.float64)], n_rows=[n])
    assert abs(single_report["bias_factor_seq"] - bias_factor_expected) < 1e-9

    # The debiased trial-mean must land far closer to the true population
    # gm (1.0) than the raw trial-mean does.
    debiased_trial_means = [g / bias_factor_expected for g in raw_gms]
    debiased_mean = sum(debiased_trial_means) / trials
    assert abs(debiased_mean - 1.0) < abs(raw_mean - 1.0) / 3


def test_jensen_gap_debiasing_equal_n_algebraic_identity():
    """Equal n_rows for every moment, pooled n = sum(n_rows): the DEBIASED
    r_pred reduces to raw_r_pred * exp(b_pool - b_seq) exactly (b_seq is the
    common per-moment bias since every moment shares the same n; algebraic
    identity, no synthetic-data randomness needed to prove it)."""
    import math

    g = torch.Generator().manual_seed(4)
    moments = [_random_psd(10, g) for _ in range(4)]
    n_each = 40
    n_rows = [n_each] * len(moments)

    report = jensen_gap_report(moments, n_rows=n_rows)
    raw = jensen_gap_report(moments)  # no n_rows -> raw r_pred, for cross-check

    assert abs(report["r_pred"] - raw["r_pred"]) < 1e-12  # raw field untouched

    C = moments[0].shape[0]

    def b(n):
        idx = torch.arange(1, C + 1, dtype=torch.float64)
        psi = torch.special.digamma((n - idx + 1) / 2.0)
        return float((psi + math.log(2.0 / n)).mean())

    b_seq = b(n_each)
    b_pool = b(n_each * len(moments))
    expected_r_pred_debiased = raw["r_pred"] * math.exp(b_pool - b_seq)

    assert abs(report["r_pred_debiased"] - expected_r_pred_debiased) < 1e-9
    assert abs(report["bias_factor_seq"] - math.exp(b_seq)) < 1e-9
    assert abs(report["bias_factor_pool"] - math.exp(b_pool)) < 1e-9


def _wishart_rows(
    pop_spectrum: torch.Tensor, n: int, generator: torch.Generator
) -> torch.Tensor:
    """n iid rows ~ N(0, diag(pop_spectrum)), fp64."""
    C = pop_spectrum.shape[0]
    z = torch.randn(n, C, generator=generator, dtype=torch.float64)
    return z * pop_spectrum.sqrt().unsqueeze(0)


def test_shrink_spectrum_lw_reduces_mse_to_population():
    """Wishart recovery: population = known decaying diag spectrum, n=64
    Gaussian rows, C=32 (gamma=0.5, a real small-sample regime). Averaged
    over 20 seeded trials, LW-shrunk spectrum has strictly smaller MSE to
    the population spectrum than the raw sample spectrum."""
    from bmx.cache.spectral import shrink_spectrum

    C, n, trials = 32, 64, 20
    pop = torch.logspace(2, -2, C, dtype=torch.float64)
    raw_mse_total, lw_mse_total = 0.0, 0.0
    for trial in range(trials):
        g = torch.Generator().manual_seed(1000 + trial)
        rows = _wishart_rows(pop, n, g)
        Sigma = rows.mT @ rows / n
        lam_raw = torch.linalg.eigvalsh(
            Sigma
        )  # ascending, matches population sort order
        lam_shrunk, rho = shrink_spectrum(lam_raw, n=n, method="lw", rows=rows)
        assert 0.0 <= rho <= 1.0
        pop_sorted = pop.sort().values
        raw_mse_total += float(((lam_raw - pop_sorted) ** 2).mean())
        lw_mse_total += float(((lam_shrunk - pop_sorted) ** 2).mean())
    assert lw_mse_total < raw_mse_total, (
        f"LW mean MSE {lw_mse_total / trials:.6g} must beat raw {raw_mse_total / trials:.6g}"
    )


def test_shrink_spectrum_oas_reduces_mse_to_population():
    """Same Wishart-recovery property for OAS (spectrum-only, no rows)."""
    from bmx.cache.spectral import shrink_spectrum

    C, n, trials = 32, 64, 20
    pop = torch.logspace(2, -2, C, dtype=torch.float64)
    raw_mse_total, oas_mse_total = 0.0, 0.0
    for trial in range(trials):
        g = torch.Generator().manual_seed(2000 + trial)
        rows = _wishart_rows(pop, n, g)
        Sigma = rows.mT @ rows / n
        lam_raw = torch.linalg.eigvalsh(Sigma)
        lam_shrunk, rho = shrink_spectrum(lam_raw, n=n, method="oas")
        assert 0.0 <= rho <= 1.0
        pop_sorted = pop.sort().values
        raw_mse_total += float(((lam_raw - pop_sorted) ** 2).mean())
        oas_mse_total += float(((lam_shrunk - pop_sorted) ** 2).mean())
    assert oas_mse_total < raw_mse_total, (
        f"OAS mean MSE {oas_mse_total / trials:.6g} must beat raw {raw_mse_total / trials:.6g}"
    )


def test_shrink_spectrum_large_n_rho_small_and_shrunk_near_raw():
    """n = 200*C: the estimate is precise, both methods must report a small
    intensity. "Near raw" is checked via the exact affine map (not a
    tolerance-based allclose): the spectrum spans 4 decades here, so even a
    tiny rho produces a large RELATIVE shift on the smallest eigenvalues
    (they get pulled toward mu_hat, which is order-of-magnitude larger) --
    that is correct shrinkage behavior, not a bug. The exact-formula check
    is scale-independent and pins the map itself."""
    from bmx.cache.spectral import shrink_spectrum

    C = 16
    n = 200 * C
    pop = torch.logspace(2, -2, C, dtype=torch.float64)
    g = torch.Generator().manual_seed(3)
    rows = _wishart_rows(pop, n, g)
    Sigma = rows.mT @ rows / n
    lam_raw = torch.linalg.eigvalsh(Sigma)
    mu = lam_raw.mean()

    lam_lw, rho_lw = shrink_spectrum(lam_raw, n=n, method="lw", rows=rows)
    assert rho_lw < 0.05
    assert torch.allclose(lam_lw, (1.0 - rho_lw) * lam_raw + rho_lw * mu, rtol=1e-9)

    lam_oas, rho_oas = shrink_spectrum(lam_raw, n=n, method="oas")
    assert rho_oas < 0.05
    assert torch.allclose(lam_oas, (1.0 - rho_oas) * lam_raw + rho_oas * mu, rtol=1e-9)


def test_shrink_spectrum_rho_monotone_in_gamma():
    """Same population spectrum, matched-seed sampling: shrinkage intensity
    rho must be LARGER at small n (near-degenerate, big gamma=C/n) than at
    large n, for both LW and OAS. Monotone-in-gamma sanity, not a fixed
    numeric bound."""
    from bmx.cache.spectral import shrink_spectrum

    C = 16
    pop = torch.logspace(2, -2, C, dtype=torch.float64)

    g_small = torch.Generator().manual_seed(4)
    rows_small = _wishart_rows(pop, 48, g_small)
    Sigma_small = rows_small.mT @ rows_small / 48
    lam_small = torch.linalg.eigvalsh(Sigma_small)

    g_large = torch.Generator().manual_seed(4)
    rows_large = _wishart_rows(pop, 2048, g_large)
    Sigma_large = rows_large.mT @ rows_large / 2048
    lam_large = torch.linalg.eigvalsh(Sigma_large)

    _, rho_lw_small = shrink_spectrum(lam_small, n=48, method="lw", rows=rows_small)
    _, rho_lw_large = shrink_spectrum(lam_large, n=2048, method="lw", rows=rows_large)
    assert rho_lw_small > rho_lw_large

    _, rho_oas_small = shrink_spectrum(lam_small, n=48, method="oas")
    _, rho_oas_large = shrink_spectrum(lam_large, n=2048, method="oas")
    assert rho_oas_small > rho_oas_large


def test_shrink_spectrum_lw_trace_trick_matches_naive_double_loop():
    """Pins the algebra: the trace-trick LW numerator must equal the naive
    per-row-outer-product double loop, on a tiny case (C=8, n=16), to 1e-10.
    Both derivations are computed independently here (not by calling
    shrink_spectrum's internals), then checked against the same rho it
    returns."""
    from bmx.cache.spectral import shrink_spectrum

    C, n = 8, 16
    g = torch.Generator().manual_seed(5)
    rows = torch.randn(n, C, generator=g, dtype=torch.float64)
    Sigma = rows.mT @ rows / n
    lam = torch.linalg.eigvalsh(Sigma)
    mu = lam.mean()

    d2 = float(((lam - mu) ** 2).sum())

    # Naive double loop: sum_t || x_t x_t^T - Sigma ||_F^2 / n^2.
    naive_sum = 0.0
    for t in range(n):
        x_t = rows[t]
        outer = torch.outer(x_t, x_t)
        naive_sum += float(((outer - Sigma) ** 2).sum())
    naive_bbar2 = min(d2, naive_sum / (n**2))
    naive_rho = naive_bbar2 / d2

    # Trace-trick form, independently re-derived here (not shrink_spectrum's code):
    # sum_t ||x_t x_t^T - Sigma||_F^2 = sum_t ||x_t||^4 - n*tr(Sigma^2).
    norm4_sum = float((rows.pow(2).sum(dim=1) ** 2).sum())
    trace_sq = float((Sigma * Sigma).sum())  # tr(Sigma^2) == ||Sigma||_F^2 (symmetric)
    trace_sum = norm4_sum - n * trace_sq
    trace_bbar2 = min(d2, trace_sum / (n**2))
    trace_rho = trace_bbar2 / d2

    assert abs(trace_sum - naive_sum) < 1e-10 * max(1.0, abs(naive_sum))
    assert abs(trace_rho - naive_rho) < 1e-10

    _, rho = shrink_spectrum(lam, n=n, method="lw", rows=rows)
    assert abs(rho - naive_rho) < 1e-10


def test_shrink_spectrum_order_and_positivity_preserved():
    """The shrink map is affine per-entry toward mu: order (sortedness) and
    positivity of the input spectrum are preserved in the output."""
    from bmx.cache.spectral import shrink_spectrum

    C, n = 12, 40
    g = torch.Generator().manual_seed(6)
    rows = torch.randn(n, C, generator=g, dtype=torch.float64)
    Sigma = rows.mT @ rows / n
    lam = torch.linalg.eigvalsh(Sigma)  # ascending, all > 0 a.s.
    assert (lam > 0).all()

    lam_lw, _ = shrink_spectrum(lam, n=n, method="lw", rows=rows)
    assert (lam_lw[1:] >= lam_lw[:-1] - 1e-12).all(), "order must be preserved"
    assert (lam_lw > 0).all(), "positivity must be preserved"

    lam_oas, _ = shrink_spectrum(lam, n=n, method="oas")
    assert (lam_oas[1:] >= lam_oas[:-1] - 1e-12).all()
    assert (lam_oas > 0).all()


def test_shrink_spectrum_preserves_mean():
    """The affine shrink map lambda' = (1-rho)*lambda + rho*mu preserves the
    mean of the spectrum exactly, for any rho in [0, 1] (mean of a constant
    shift toward the mean is a no-op on the mean)."""
    from bmx.cache.spectral import shrink_spectrum

    C, n = 20, 30
    g = torch.Generator().manual_seed(7)
    rows = torch.randn(n, C, generator=g, dtype=torch.float64)
    Sigma = rows.mT @ rows / n
    lam = torch.linalg.eigvalsh(Sigma)

    lam_lw, _ = shrink_spectrum(lam, n=n, method="lw", rows=rows)
    assert abs(float(lam_lw.mean()) - float(lam.mean())) < 1e-12

    lam_oas, _ = shrink_spectrum(lam, n=n, method="oas")
    assert abs(float(lam_oas.mean()) - float(lam.mean())) < 1e-12


def test_shrink_spectrum_zero_d2_forces_rho_one():
    """Degenerate case: Sigma already proportional to I (d^2 == 0) forces
    rho = 1 exactly (already at target; also guards a 0/0 division)."""
    from bmx.cache.spectral import shrink_spectrum

    C = 8
    lam = torch.full((C,), 3.0, dtype=torch.float64)
    # Construct rows (n=C) whose sample covariance rows^T @ rows / n is
    # exactly 3*I: a diagonal design with entries sqrt(3*C).
    rows = torch.zeros(C, C, dtype=torch.float64)
    rows[range(C), range(C)] = (3.0 * C) ** 0.5
    Sigma_check = rows.mT @ rows / C
    assert torch.allclose(Sigma_check, 3.0 * torch.eye(C, dtype=torch.float64))

    lam_lw, rho_lw = shrink_spectrum(lam, n=C, method="lw", rows=rows)
    assert rho_lw == 1.0
    assert torch.allclose(lam_lw, lam)

    lam_oas, rho_oas = shrink_spectrum(lam, n=100, method="oas")
    assert rho_oas == 1.0
    assert torch.allclose(lam_oas, lam)


def test_shrink_spectrum_rejects_bad_method_and_missing_rows():
    from bmx.cache.spectral import shrink_spectrum

    lam = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    with pytest.raises(AssertionError):
        shrink_spectrum(lam, n=10, method="bogus")
    with pytest.raises(AssertionError):
        shrink_spectrum(lam, n=10, method="lw")  # rows required
    with pytest.raises(AssertionError):
        shrink_spectrum(
            lam, n=10, method="lw", rows=torch.randn(10, 5)
        )  # shape mismatch
