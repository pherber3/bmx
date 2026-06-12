import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from bmx.eval.layer_swap import perplexity, set_weight


def _tiny():
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=97, n_positions=64)
    torch.manual_seed(0)
    return GPT2LMHeadModel(cfg)


def test_set_weight_replaces_and_changes_logits():
    model = _tiny()
    ids = torch.randint(0, 97, (1, 16), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        before = model(ids).logits.clone()
    W = model.transformer.h[0].attn.c_attn.weight
    set_weight(model, 0, "attn.c_attn", torch.zeros_like(W))
    assert model.transformer.h[0].attn.c_attn.weight.abs().sum() == 0
    with torch.no_grad():
        after = model(ids).logits
    assert not torch.allclose(before, after)


def test_set_weight_validates():
    model = _tiny()
    with pytest.raises(AssertionError):
        set_weight(model, 0, "attn.c_attn", torch.zeros(3, 3))
    with pytest.raises(AssertionError):
        set_weight(model, 0, "not.a.module", torch.zeros(3, 3))


def test_perplexity_finite_and_self_swap_invariant():
    model = _tiny()
    ids = torch.randint(0, 97, (256,), generator=torch.Generator().manual_seed(2))
    p1 = perplexity(model, ids, block=64)
    W = model.transformer.h[1].mlp.c_fc.weight.detach().clone()
    set_weight(model, 1, "mlp.c_fc", W)  # replace with itself
    p2 = perplexity(model, ids, block=64)
    assert p1 > 0 and torch.isfinite(torch.tensor(p1))
    assert abs(p1 - p2) < 1e-4
