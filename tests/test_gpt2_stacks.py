import torch

from bmx.stacks.gpt2 import circuit_stack, raw_stack, w_all_4d


def _fake_sd(d=8, n_head=2, n_layer=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for layer in range(n_layer):
        sd[f"transformer.h.{layer}.attn.c_attn.weight"] = torch.randn(
            d, 3 * d, generator=g, dtype=torch.float64
        )
        sd[f"transformer.h.{layer}.attn.c_proj.weight"] = torch.randn(
            d, d, generator=g, dtype=torch.float64
        )
    return sd


def test_raw_stack_shapes_and_values():
    d, n_head = 8, 2
    dh = d // n_head
    sd = _fake_sd(d, n_head)
    q = raw_stack(sd, layer=0, n_head=n_head, which="q")
    assert q.tensor.shape == (d, dh, n_head)
    Wq = sd["transformer.h.0.attn.c_attn.weight"][:, :d]
    torch.testing.assert_close(q.tensor[:, :, 1], Wq[:, dh : 2 * dh])
    assert q.object_name == "raw_q" and q.layer == 0


def test_circuit_stack_wqk_matches_manual():
    d, n_head = 8, 2
    dh = d // n_head
    sd = _fake_sd(d, n_head)
    W = sd["transformer.h.0.attn.c_attn.weight"]
    Wq, Wk = W[:, :d], W[:, d : 2 * d]
    wqk = circuit_stack(sd, layer=0, n_head=n_head, kind="wqk")
    assert wqk.tensor.shape == (d, d, n_head)
    h = 1
    manual = Wq[:, h * dh : (h + 1) * dh] @ Wk[:, h * dh : (h + 1) * dh].T
    torch.testing.assert_close(wqk.tensor[:, :, h], manual)


def test_circuit_stack_wov_matches_manual():
    d, n_head = 8, 2
    dh = d // n_head
    sd = _fake_sd(d, n_head)
    Wv = sd["transformer.h.0.attn.c_attn.weight"][:, 2 * d :]
    Wo = sd["transformer.h.0.attn.c_proj.weight"]
    wov = circuit_stack(sd, layer=0, n_head=n_head, kind="wov")
    h = 0
    manual = Wv[:, h * dh : (h + 1) * dh] @ Wo[h * dh : (h + 1) * dh, :]
    torch.testing.assert_close(wov.tensor[:, :, h], manual)


def test_w_all_4d_shape():
    sd = _fake_sd(8, 2)
    s = w_all_4d(sd, layer=0, n_head=2)
    assert s.tensor.shape == (8, 4, 4, 2)  # (d, d_head, matrix-type, head)
