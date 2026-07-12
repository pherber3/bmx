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


def test_query_moment_matches_explicit_rotation_matrices():
    """Value-pin with sin != 0: W must equal mean_p R_pT (pooled qqT) R_p where
    R_p^T is the inverse rotation (negated sin). A sign flip in the adjoint
    (_rotate_half(q) * sin instead of * (-sin)) would produce a different result."""
    from bmx.cache.rope import apply_rope

    g = torch.Generator().manual_seed(4)
    h, T, d, h_kv, S = 2, 16, 4, 1, 8
    q = torch.randn(h, T, d, generator=g)
    theta = torch.linspace(0.3, 2.0, S).unsqueeze(1) * torch.linspace(
        0.5, 1.0, d
    ).unsqueeze(0)
    cos, sin = theta.cos(), theta.sin()

    stride = 2
    W = query_position_moment(q, cos, sin, h_kv, position_stride=stride)

    # Ground truth: build W independently using apply_rope for the inverse rotation.
    # apply_rope(q, cos, -sin) implements q * cos + rotate_half(q) * (-sin) = R_p^T @ q
    # (where apply_rope uses cos/sin in the RoPE sense, not matrix form).
    q64 = q.double()

    W_expected = torch.zeros(d, d, dtype=torch.float64)
    positions = list(range(0, S, stride))

    for p in positions:
        # Compute the inverse rotation by negating sin: R_p^T @ q
        cos_p = cos[p].double()
        sin_p = sin[p].double()
        # q_rot = q64 * cos_p + _rotate_half(q64) * (-sin_p)
        q_rot_inv = apply_rope(
            q64, cos_p.unsqueeze(0), -sin_p.unsqueeze(0)
        )  # (h, T, d)

        # Pool over all h and T: sum (R_p^T @ q)^T @ (R_p^T @ q)
        q_rot_pooled = q_rot_inv.reshape(-1, d)  # (h*T, d)
        W_expected += q_rot_pooled.mT @ q_rot_pooled

    W_expected /= len(positions) * h * T
    assert torch.allclose(W[0], W_expected, atol=1e-10)
