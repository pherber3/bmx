import json

import pytest
import torch
from safetensors.torch import save_file

from bmx.census import experts_shared_last, pairwise_similarities, similarity_summary
from bmx.stacks.moe import expert_stack, moe_layers


@pytest.fixture
def fake_checkpoint(tmp_path):
    """Two shards; layer 0 dense, layer 1 has E=3 experts (gate/up/down)."""
    d, d_ff, n_e = 6, 4, 3

    def key(layer, e, which):
        return f"model.layers.{layer}.mlp.experts.{e}.{which}_proj.weight"

    tensors = {"model.layers.0.mlp.gate_proj.weight": torch.randn(d_ff, d)}
    for e in range(n_e):
        tensors[key(1, e, "gate")] = torch.full((d_ff, d), float(e))
        tensors[key(1, e, "up")] = torch.randn(d_ff, d)
        tensors[key(1, e, "down")] = torch.randn(d, d_ff)

    names = sorted(tensors)
    shard_a = {k: tensors[k] for k in names[: len(names) // 2]}
    shard_b = {k: tensors[k] for k in names[len(names) // 2 :]}
    save_file(shard_a, str(tmp_path / "model-00001.safetensors"))
    save_file(shard_b, str(tmp_path / "model-00002.safetensors"))
    weight_map = {k: "model-00001.safetensors" for k in shard_a}
    weight_map |= {k: "model-00002.safetensors" for k in shard_b}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return tmp_path


def test_moe_layers_excludes_dense(fake_checkpoint):
    assert moe_layers(fake_checkpoint) == [1]


def test_expert_stack_shapes_and_values(fake_checkpoint):
    gate = expert_stack(fake_checkpoint, layer=1, which="gate")
    assert gate.tensor.shape == (4, 6, 3)  # (d_ff, d_model, E)
    for e in range(3):
        assert (gate.tensor[:, :, e] == float(e)).all()
    down = expert_stack(fake_checkpoint, layer=1, which="down")
    assert down.tensor.shape == (6, 4, 3)  # (d_model, d_ff, E)
    assert gate.axes == ("d_ff", "d_model", "expert")
    assert down.axes == ("d_model", "d_ff", "expert")
    with pytest.raises(AssertionError):
        expert_stack(fake_checkpoint, layer=1, which="qkv")


def test_shared_last_orientation(fake_checkpoint):
    gate = expert_stack(fake_checkpoint, layer=1, which="gate")
    down = expert_stack(fake_checkpoint, layer=1, which="down")
    assert experts_shared_last(gate).shape == (3, 4, 6)  # (E, d_ff, d_model)
    assert experts_shared_last(down).shape == (3, 4, 6)


def test_metrics_on_planted_structure():
    torch.manual_seed(0)
    E, p, d = 6, 8, 16
    base = torch.randn(p, d, dtype=torch.float64)
    S = torch.randn(E, p, d, dtype=torch.float64)
    S[1] = S[0]  # exact duplicate pair
    S[2] = 2.0 * base
    S[3] = -0.5 * base  # same subspace, different scale/sign
    sims = pairwise_similarities(S, top_r=4)
    for m in ("cos", "cka", "sub"):
        assert sims[m].shape == (E, E)
        assert torch.allclose(sims[m].diagonal(), torch.ones(E, dtype=torch.float64))
        assert sims[m][0, 1] > 0.999, f"{m}: duplicate pair not detected"
    # scaled copies of one template: identical Grams/subspaces, opposite cosines
    assert sims["cka"][2, 3] > 0.999
    assert sims["sub"][2, 3] > 0.999
    assert sims["cos"][2, 3] < -0.999


def test_summary_pr_extremes():
    E, p, d = 8, 6, 12
    same = torch.randn(1, p, d, dtype=torch.float64).expand(E, p, d).contiguous()
    pr_same = similarity_summary(pairwise_similarities(same)["cos"])["pr_frac"]
    assert pr_same < 0.2  # one global mode -> PR ~ 1/E

    torch.manual_seed(1)
    rand = torch.randn(E, p, d, dtype=torch.float64)
    pr_rand = similarity_summary(pairwise_similarities(rand)["cos"])["pr_frac"]
    assert pr_rand > 0.7  # near-orthogonal experts -> PR ~ E
