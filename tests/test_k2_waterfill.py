"""Smoke test for experiments/k2_waterfill.py on the offline GPT-2 cache fixture."""

import importlib.util
import sys
from pathlib import Path

import torch

# Load the experiment module by path (experiments/ is not a package).
_EXP = Path(__file__).resolve().parents[1] / "experiments" / "k2_waterfill.py"
_spec = importlib.util.spec_from_file_location("k2_waterfill", _EXP)
k2_waterfill = importlib.util.module_from_spec(_spec)
sys.modules["k2_waterfill"] = k2_waterfill
_spec.loader.exec_module(k2_waterfill)


def test_stable_rank_helper():
    # isotropic -> stable rank ~ C; rank-1 -> stable rank ~ 1.
    g = torch.Generator().manual_seed(1)
    iso = torch.randn(128, 16, generator=g, dtype=torch.float64)
    sr_iso = k2_waterfill._resid_stable_rank(iso)
    assert sr_iso > 8.0

    v = torch.randn(16, 1, generator=g, dtype=torch.float64)
    rank1 = torch.randn(128, 1, generator=g, dtype=torch.float64) @ v.mT
    sr_r1 = k2_waterfill._resid_stable_rank(rank1)
    assert sr_r1 < 2.0


def _build_synthetic_cache(tmp_path: Path) -> Path:
    """Write a tiny 2-layer synthetic cache and return its path."""
    from safetensors.torch import save_file

    h_kv, S, d = 2, 64, 8  # C = 16
    g = torch.Generator().manual_seed(7)
    tensors = {}
    for i in range(2):
        tensors[f"layer{i}.k_pre"] = torch.randn(h_kv, S, d, generator=g).half()
        tensors[f"layer{i}.k"] = torch.randn(h_kv, S, d, generator=g).half()
        tensors[f"layer{i}.v"] = torch.randn(h_kv, S, d, generator=g).half()
        tensors[f"layer{i}.q"] = torch.randn(h_kv, S, d, generator=g).half()
    cache_path = tmp_path / "synthetic.safetensors"
    save_file(tensors, str(cache_path))
    return cache_path


def test_experiment_smoke(tmp_path):
    # Build a tiny synthetic cache file with the layer-key convention and run main.
    cache_path = _build_synthetic_cache(tmp_path)

    cfg = k2_waterfill.Config(
        cache_path=str(cache_path),
        model_label="synthetic",
        model_name="",  # no RoPE -> logit (stored basis), not logit_rope
        budget_bits=3.0,
        group=16,
        rank=4,
        out_root=str(tmp_path / "results"),
    )
    df = k2_waterfill.main(cfg)
    arms = set(df["arm"].unique())
    assert {
        "lowrank_rtn_channel",
        "lowrank_waterfill_channel",
        "lowrank_eigwaterfill_channel",
        "lowrank_randwaterfill_channel",
        "outlier_two_tier",
    } <= arms
    assert "resid_stable_rank" in df.columns
    # matched bpe: waterfill within tolerance of uniform baseline, per layer
    for layer in df["layer"].unique():
        sub = df[df["layer"] == layer]
        bpe_uni = sub[sub.arm == "lowrank_rtn_channel"]["bpe"].mean()
        bpe_wf = sub[sub.arm == "lowrank_waterfill_channel"]["bpe"].mean()
        assert abs(bpe_uni - bpe_wf) < 0.05, (
            f"bpe mismatch L{layer}: {bpe_uni} vs {bpe_wf}"
        )


def test_default_root_no_out_root(tmp_path, monkeypatch):
    """Regression: out_root='' must not crash (was passing None to Path())."""
    # Redirect cwd to tmp_path so create_run's default 'results' root lands there.
    monkeypatch.chdir(tmp_path)
    cache_path = _build_synthetic_cache(tmp_path)

    cfg = k2_waterfill.Config(
        cache_path=str(cache_path),
        model_label="synthetic",
        model_name="",
        budget_bits=3.0,
        group=16,
        rank=4,
        # out_root intentionally left as "" (the default) to exercise the default path
    )
    df = k2_waterfill.main(cfg)
    assert len(df) >= 5 * 2, "expected at least 5 arms × 2 layers"
    assert {
        "lowrank_rtn_channel",
        "lowrank_waterfill_channel",
        "lowrank_eigwaterfill_channel",
        "lowrank_randwaterfill_channel",
        "outlier_two_tier",
    } <= set(df["arm"].unique())
    # run dir must be under tmp_path, not the real repo results/
    results_dir = tmp_path / "results" / "k2_waterfill"
    assert results_dir.exists(), f"expected run dir under tmp_path: {results_dir}"


def test_experiment_has_structured_arms(tmp_path):
    from safetensors.torch import save_file

    h_kv, S, d = 2, 64, 8  # C=16
    g = torch.Generator().manual_seed(7)
    tensors = {}
    for i in range(2):
        for nm in ("k_pre", "k", "v", "q"):
            tensors[f"layer{i}.{nm}"] = torch.randn(h_kv, S, d, generator=g).half()
    cache_path = tmp_path / "synthetic.safetensors"
    save_file(tensors, str(cache_path))

    cfg = k2_waterfill.Config(
        cache_path=str(cache_path),
        model_label="synthetic",
        model_name="",
        budget_bits=3.0,
        group=16,
        rank=4,
        topk_ks=(8,),
        prefill_fit_len=32,
        out_root=str(tmp_path / "results"),
    )
    df = k2_waterfill.main(cfg)
    arms = set(df["arm"].unique())
    # block-diag, frozen, oracle present; topk present (named with its k)
    assert "lowrank_blockdiagwaterfill_channel" in arms
    assert "lowrank_frozenwaterfill_channel" in arms
    assert "lowrank_oraclewaterfill_channel" in arms
    assert any(a.startswith("lowrank_topkwaterfill_channel") for a in arms)
    # diagnostics
    assert "resid_eigengap" in df.columns
    assert "frozen_oracle_ratio" in df.columns
    fr = df[df.arm == "lowrank_frozenwaterfill_channel"]
    assert fr["frozen_oracle_ratio"].notna().all()
    # honest bpe present for the structured deployable arms
    assert "bpe_honest" in df.columns
    bd = df[df.arm == "lowrank_blockdiagwaterfill_channel"]
    assert bd["bpe_honest"].notna().all()


def test_experiment_has_rotated_arms_and_dual_metric(tmp_path):
    from safetensors.torch import save_file

    h_kv, S, d = 2, 64, 8
    g = torch.Generator().manual_seed(7)
    tensors = {}
    for i in range(2):
        tensors[f"layer{i}.k_pre"] = torch.randn(h_kv, S, d, generator=g).half()
        tensors[f"layer{i}.k"] = torch.randn(h_kv, S, d, generator=g).half()
        tensors[f"layer{i}.v"] = torch.randn(h_kv, S, d, generator=g).half()
        tensors[f"layer{i}.q"] = torch.randn(h_kv, S, d, generator=g).half()
    cache_path = tmp_path / "synthetic.safetensors"
    save_file(tensors, str(cache_path))

    cfg = k2_waterfill.Config(
        cache_path=str(cache_path),
        model_label="synthetic",
        model_name="",
        budget_bits=3.0,
        group=16,
        rank=4,
        out_root=str(tmp_path / "results"),
    )
    df = k2_waterfill.main(cfg)
    arms = set(df["arm"].unique())
    assert {"lowrank_eigwaterfill_channel", "lowrank_randwaterfill_channel"} <= arms
    # dual metric present and populated for every arm
    assert "rel_fro" in df.columns and "logit_rope" in df.columns
    assert df["rel_fro"].notna().all() and df["logit_rope"].notna().all()
    # alignment diagnostic present for the rotated arms
    assert "query_eigen_alignment" in df.columns
    eig = df[df.arm == "lowrank_eigwaterfill_channel"]
    assert eig["query_eigen_alignment"].notna().all()
    assert (
        (eig["query_eigen_alignment"] >= 0) & (eig["query_eigen_alignment"] <= 1.0001)
    ).all()
    # matched idealized bpe: both rotated arms within tol of uniform, per layer
    for layer in df["layer"].unique():
        sub = df[df["layer"] == layer]
        bpe_uni = sub[sub.arm == "lowrank_rtn_channel"]["bpe"].mean()
        for arm in ("lowrank_eigwaterfill_channel", "lowrank_randwaterfill_channel"):
            bpe_arm = sub[sub.arm == arm]["bpe"].mean()
            assert abs(bpe_uni - bpe_arm) < 0.05, (
                f"{arm} L{layer} bpe {bpe_arm} vs {bpe_uni}"
            )
