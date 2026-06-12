"""GPT-2 attention weight stacks. Builders take a state-dict mapping so unit
tests can inject small fakes; load_gpt2_state fetches the real checkpoint.

GPT-2 uses Conv1D: weight (in_features, out_features), y = x @ W + b.
c_attn.weight (d, 3d) packs Q|K|V column blocks; per-head columns are
head-major. c_proj.weight (d, d) rows are head-major.
"""

import torch

from bmx.stacks.base import Stack


def load_gpt2_state(model_name: str = "gpt2"):
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(model_name)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    meta = {
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "d": model.config.n_embd,
    }
    return sd, meta


def _per_head(sd: dict, layer: int, n_head: int):
    W = sd[f"transformer.h.{layer}.attn.c_attn.weight"]
    d = W.shape[0]
    dh = d // n_head
    Wq, Wk, Wv = W[:, :d], W[:, d : 2 * d], W[:, 2 * d :]
    # (d, d) column blocks -> (d, d_head, n_head)
    q = Wq.reshape(d, n_head, dh).permute(0, 2, 1)
    k = Wk.reshape(d, n_head, dh).permute(0, 2, 1)
    v = Wv.reshape(d, n_head, dh).permute(0, 2, 1)
    Wo = sd[f"transformer.h.{layer}.attn.c_proj.weight"]
    o = Wo.reshape(n_head, dh, d)  # o[h] = W_O^h : (d_head, d)
    return q, k, v, o


def raw_stack(sd: dict, layer: int, n_head: int, which: str, model="gpt2") -> Stack:
    """Per-head projection stack, common shape (d_model, d_head, head).

    Note: which="o" stores W_O^h TRANSPOSED (slice h is W_O^h.T, (d, d_head))
    so all four stacks share one shape; transpose back before using as W_O.
    """
    q, k, v, o = _per_head(sd, layer, n_head)
    tensors = {"q": q, "k": k, "v": v, "o": o.permute(2, 1, 0)}  # o -> (d, dh, h)
    assert which in tensors, f"which must be one of {sorted(tensors)}"
    return Stack(
        tensors[which].contiguous(),
        model,
        layer,
        f"raw_{which}",
        ("d_model", "d_head", "head"),
    )


def circuit_stack(sd: dict, layer: int, n_head: int, kind: str, model="gpt2") -> Stack:
    q, k, v, o = _per_head(sd, layer, n_head)
    if kind == "wqk":
        T = torch.einsum("ich,jch->ijh", q, k)  # W_Q^h @ W_K^h.T
    elif kind == "wov":
        T = torch.einsum("ich,hcj->ijh", v, o)  # W_V^h @ W_O^h
    else:
        raise ValueError(f"kind must be 'wqk' or 'wov', got {kind!r}")
    return Stack(T.contiguous(), model, layer, kind, ("d_model", "d_model", "head"))


def w_all_4d(sd: dict, layer: int, n_head: int, model="gpt2") -> Stack:
    """TensorLLM's object: (d_model, d_head, matrix-type[Q,K,V,O^T], head)."""
    q, k, v, o = _per_head(sd, layer, n_head)
    T = torch.stack([q, k, v, o.permute(2, 1, 0)], dim=2)
    return Stack(
        T.contiguous(),
        model,
        layer,
        "w_all",
        ("d_model", "d_head", "matrix_type", "head"),
    )


def stack_by_name(
    sd: dict, layer: int, n_head: int, obj: str, model: str = "gpt2"
) -> Stack:
    """Object-name dispatch shared by the stack experiments (a2/a3).

    Accepts 'wqk', 'wov', 'w_all', or 'raw_q'/'raw_k'/'raw_v'/'raw_o'.
    """
    if obj.startswith("raw_"):
        return raw_stack(sd, layer, n_head, which=obj.removeprefix("raw_"), model=model)
    if obj in ("wqk", "wov"):
        return circuit_stack(sd, layer, n_head, kind=obj, model=model)
    if obj == "w_all":
        return w_all_4d(sd, layer, n_head, model=model)
    raise ValueError(f"unknown stack object {obj!r}")
