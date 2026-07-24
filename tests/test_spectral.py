import pytest
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
