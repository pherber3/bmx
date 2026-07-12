import torch

from bmx.cache.spectral import (
    assemble_whitener,
    identity_whitener,
    key_second_moment,
    query_position_moment,
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
