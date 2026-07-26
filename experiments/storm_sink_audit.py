"""Storm Task-3 — sink carve-out audit (plan
`docs/superpowers/plans/2026-07-26-storm-gates.md` Task 3).

Question: what fraction of the K4 codec's QUERY-WEIGHTED distortion budget
    Σ_s e_sᵀ W e_s     (e_s = k_s - k_hat_s, the per-position K-reconstruction
error; W the per-kv-head query second moment that defines the whitener)
lands on the SINK positions 0..3, and does exempting them reclaim bits? A
companion number — the fraction of query attention MASS those positions
receive — lets the verdict distinguish the two mechanisms:
  * "codec wastes budget on sinks"  (high distortion share)  ⇒ carve-out.
  * "sinks get mass but the codec already reconstructs them fine"
    (low distortion share, high mass)  ⇒ the W-weighting already prices them.

Pre-registered gate (verbatim, plan Task 3):
    sink share >= 5% of the weighted-distortion budget at EITHER budget
    ⇒ CONFIRM carve-out (spec the recipe change). < 5% at BOTH budgets
    ⇒ honest null: the W-weighting already prices sinks correctly — record
    as a VALIDATION of the metric.

Mechanism scale, no VM, no web. Reuses the banked corpus packs
(`results/cache/k4_packs_<model>.safetensors` + sidecar — the deployment-grade
codec: per-layer basis fit on a corpus, per-budget waterfilled bits) and scores
the position decomposition on a HELD-OUT (offset-0) cache distinct from the
corpus the pack was fit on. If a pack file is absent it is fit on the fly from
the sidecar-named corpus caches via the shared `corpus_fit_bases` machinery.

The key identity (verified to fp64 round-off in
`test_storm_sink_audit.py`): the pack's encoder satisfies W = enc @ encᵀ
exactly (enc = W^{1/2} E, E orthogonal), so the query-weighted per-position
distortion is simply the row-wise squared norm of the error projected through
enc — `||encᵀ e_s||²` — needing only the pack itself, no W refit. Positions are
the S axis; (h,S,d) ↔ (S, h·d) layout goes through `to_matrix` only.

fp32 experiment path (caches fp16, moment/enc math the codec already carries in
fp32); the position-decomposition core keeps an fp64 cross-check available to
the test.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics
from bmx.cache.collect import to_matrix
from bmx.cache.metrics import _expand_kv
from bmx.cache.rope import apply_rope
from bmx.cache.spectral import (
    SpectralPack,
    load_packs,
    save_pack_file,
    spectral_quantize,
)
from experiments._k4_common import (
    corpus_fit_bases,
    load_layer_keys,
    setup_rope,
)

# The pre-registered budgets and sink window (plan Task 3 / the K4 recipe's
# operating band). N_SINKS = 4 ⇒ positions {0, 1, 2, 3}. GATE_PCT is plan-locked.
BUDGETS = (2.2, 2.5)
N_SINKS = 4
GATE_PCT = 5.0


# ---------------------------------------------------------------------------
# Position-decomposition core (pure; pinned by the offline test)
# ---------------------------------------------------------------------------


def position_weighted_distortion(M: torch.Tensor, pack: SpectralPack) -> torch.Tensor:
    """Per-position query-weighted distortion d_s = e_sᵀ W e_s for one
    (layer, side) pack, where e_s = M[s] - M_hat[s] is the K-reconstruction
    error row and W = pack.enc @ pack.encᵀ is the per-kv-head query second
    moment the whitener encodes (enc = W^{1/2} E, E orthogonal ⇒ enc encᵀ = W
    exactly — verified in the test to fp64 round-off).

    Because W = enc encᵀ, e_sᵀ W e_s = ||encᵀ e_s||², so this is the row-wise
    squared norm of the error projected through enc — the same quantity the
    codec's waterfill minimizes, decomposed by KEY POSITION with no W refit.

    M: (S, C) fp32 pre-RoPE key matrix (`to_matrix(k_pre)`). Returns (S,) fp32,
    d_s >= 0, summing to the total query-weighted reconstruction distortion.
    """
    S, C = M.shape
    assert pack.enc.shape == (C, C), f"pack C mismatch: {pack.enc.shape} vs C={C}"
    M_hat, _bpe = spectral_quantize(M, pack)
    err = M.float() - M_hat.float()  # (S, C)
    proj = err @ pack.enc.float()  # (S, C): row s = encᵀ e_s
    return (proj * proj).sum(dim=1)  # (S,)


def sink_share(d_pos: torch.Tensor, n_sinks: int = N_SINKS) -> float:
    """Fraction of the total per-position weighted distortion attributable to
    positions 0..n_sinks-1. `d_pos`: the (S,) output of
    `position_weighted_distortion`."""
    assert d_pos.dim() == 1, f"d_pos must be (S,); got {tuple(d_pos.shape)}"
    assert n_sinks < d_pos.shape[0], f"n_sinks {n_sinks} >= S {d_pos.shape[0]}"
    total = float(d_pos.sum())
    if total <= 0.0:
        return 0.0
    return float(d_pos[:n_sinks].sum()) / total


def attention_mass_by_position(
    q: torch.Tensor,
    k_post: torch.Tensor,
    *,
    n_sinks: int = N_SINKS,
) -> tuple[torch.Tensor, float]:
    """Query-weighted attention mass each source position receives, from the
    stored probe queries reading the (post-RoPE) key cache. q: (h, T, d) stored
    queries — the last T=n_q_keep positions, so absolute positions [S-T, S).
    k_post: (h_kv, S, d) post-RoPE keys. GQA-expanded to h heads.

    Softmax over the FULL sequence with a causal mask (query row i at absolute
    position S-T+i may only attend to source s <= S-T+i). Mass is averaged over
    heads and query rows, giving a (S,) distribution that sums to 1. Returns
    (mass_per_pos (S,), sink_mass = Σ_{s<n_sinks} mass). This is the honest
    "how attended-to are the sinks" number that distinguishes the two verdict
    mechanisms — it is NOT the distortion budget, which W already prices.
    """
    q = q.float()
    k_post = k_post.float()
    h, T, d = q.shape
    S = k_post.shape[1]
    k_exp = _expand_kv(k_post, h)  # (h, S, d)
    logits = q @ k_exp.transpose(-1, -2) / (d**0.5)  # (h, T, S)
    q_pos = torch.arange(S - T, S).view(T, 1)
    s_pos = torch.arange(S).view(1, S)
    causal = (s_pos <= q_pos).view(1, T, S)  # query i sees s <= S-T+i
    logits = logits.masked_fill(~causal, float("-inf"))
    attn = torch.softmax(logits, dim=-1)  # (h, T, S)
    mass = attn.mean(dim=(0, 1))  # (S,), sums to 1
    return mass, float(mass[:n_sinks].sum())


# ---------------------------------------------------------------------------
# Pack loading / on-the-fly fit
# ---------------------------------------------------------------------------


def _fit_packs_from_sidecar_corpus(
    sidecar_path: str, out_path: str, model_name: str, budgets: tuple[float, ...]
) -> None:
    """Fit a pack file on the fly from the corpus caches named in an existing
    sidecar (or, more generally, from any corpus caches) via the SHARED
    `corpus_fit_bases` machinery k4_fit_packs uses — deployment-grade W (pooled
    corpus queries), pre-RoPE keys. Only invoked when the banked pack file is
    absent; the reuse path is the norm."""
    sidecar = json.loads(Path(sidecar_path).read_text())
    corpus_paths = sidecar["corpus_cache_paths"]
    group = sidecar.get("group", 64)
    tiers = tuple(sidecar.get("tiers", (0, 2, 3, 4, 5, 6, 8)))
    ridge = sidecar.get("ridge", 1e-3)
    w_source = sidecar.get("w_source", "corpus")

    per_cache_layer_keys = [load_layer_keys(p) for p in corpus_paths]
    layers = sorted(per_cache_layer_keys[0].keys())
    rope_ready = False
    get_cos_sins = []
    for lk in per_cache_layer_keys:
        ready, gcs = setup_rope(model_name, lk, layers)
        rope_ready = rope_ready or ready
        get_cos_sins.append(gcs)
    fit = corpus_fit_bases(
        per_cache_layer_keys,
        get_cos_sins,
        rope_ready,
        layers,
        w_source=w_source,
        ridge=ridge,
        position_stride=sidecar.get("position_stride", 8),
    )
    save_pack_file(out_path, fit.bases, budgets, tiers=tiers, group=group)


def _resolve_packs(
    pack_path: str, budget: float, model_name: str
) -> dict[int, SpectralPack]:
    """Load the banked pack for `budget`, fitting the pack file on the fly from
    its sidecar's corpus caches if the .safetensors is missing but the sidecar
    is present (the plan's "fit on the fly ... or reuse ... if present")."""
    if not Path(pack_path).exists():
        sidecar_path = pack_path + ".json"
        assert Path(sidecar_path).exists(), (
            f"neither pack {pack_path!r} nor sidecar {sidecar_path!r} present — "
            "cannot fit on the fly without the corpus cache manifest"
        )
        _fit_packs_from_sidecar_corpus(
            sidecar_path, pack_path, model_name, tuple(BUDGETS)
        )
    return load_packs(pack_path, budget)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Config:
    # Held-out (offset-0) cache the position decomposition is scored on — MUST
    # be distinct from the corpus caches the pack was fit on.
    score_cache_path: str = "results/cache/gpt2_1024.safetensors"
    # Banked corpus pack (+ .json sidecar). Reused if present; else fit on the
    # fly from the sidecar's corpus caches.
    pack_path: str = "results/cache/k4_packs_gpt2.safetensors"
    model_label: str = "gpt2"
    model_name: str = ""  # HF repo id for RoPE; empty => no-RoPE (gpt2) tables
    budgets: tuple[float, ...] = BUDGETS
    n_sinks: int = N_SINKS
    out_root: str = ""


def _audit_one_budget(
    packs: dict[int, SpectralPack],
    layer_keys: dict[int, dict[str, torch.Tensor]],
    layers: list[int],
    rope_ready: bool,
    get_cos_sin,
    n_sinks: int,
) -> tuple[list[dict], dict]:
    """Per-layer position rows + the pooled (corpus-total) verdict block for
    one budget. The pooled sink DISTORTION share — Σ_layers Σ_{s<n} d_s over
    Σ_layers Σ_s d_s — is the gate quantity (the total weighted-distortion
    budget, raw-summed across layers, is the codec's actual bit allocation)."""
    rows: list[dict] = []
    total_dist = 0.0
    sink_dist = 0.0
    # Attention mass is per-layer a distribution summing to 1; the pooled sink
    # mass is the mean of per-layer sink masses (each layer contributes equally
    # — every layer reads the same S positions with its own queries).
    sink_mass_layers: list[float] = []

    for li in layers:
        kinds = layer_keys[li]
        k_pre = kinds["k_pre"]  # (h_kv, S, d) fp16 pre-RoPE
        h_kv, S, d = k_pre.shape
        M = to_matrix(k_pre)  # (S, C) fp32
        pack = packs[li]

        d_pos = position_weighted_distortion(M, pack)  # (S,)
        layer_total = float(d_pos.sum())
        layer_sink = float(d_pos[:n_sinks].sum())
        total_dist += layer_total
        sink_dist += layer_sink

        # Post-RoPE keys + forward-rotated stored queries for the attention mass.
        q = kinds["q"].float()  # (h, T, d)
        if rope_ready:
            cos, sin = get_cos_sin(S)
            k_post = apply_rope(k_pre.float(), cos, sin)
            T = q.shape[1]
            q_positions = torch.arange(S - T, S)
            q_read = apply_rope(q, cos[q_positions], sin[q_positions])
        else:
            k_post = kinds["k"].float()  # already == pre-RoPE for gpt2
            q_read = q
        _mass, layer_sink_mass = attention_mass_by_position(
            q_read, k_post, n_sinks=n_sinks
        )
        sink_mass_layers.append(layer_sink_mass)

        rows.append(
            dict(
                layer=li,
                S=S,
                sink_distortion=layer_sink,
                total_distortion=layer_total,
                sink_dist_share=(layer_sink / layer_total if layer_total > 0 else 0.0),
                sink_attn_mass=layer_sink_mass,
            )
        )

    pooled_share = sink_dist / total_dist if total_dist > 0 else 0.0
    pooled_mass = sum(sink_mass_layers) / len(sink_mass_layers)
    per_layer_shares = [r["sink_dist_share"] for r in rows]
    verdict = dict(
        sink_distortion=sink_dist,
        total_distortion=total_dist,
        sink_dist_share=pooled_share,
        sink_dist_share_pct=100.0 * pooled_share,
        sink_attn_mass=pooled_mass,
        sink_attn_mass_pct=100.0 * pooled_mass,
        max_layer_sink_dist_share=max(per_layer_shares),
        min_layer_sink_dist_share=min(per_layer_shares),
        mean_layer_sink_dist_share=sum(per_layer_shares) / len(per_layer_shares),
    )
    return rows, verdict


def main(cfg: Config):
    assert cfg.score_cache_path, "score_cache_path required"
    assert cfg.pack_path, "pack_path required"

    run = (
        create_run("storm_sink_audit", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("storm_sink_audit", cfg)
    )

    layer_keys = load_layer_keys(cfg.score_cache_path)
    layers = sorted(layer_keys.keys())
    rope_ready, get_cos_sin = setup_rope(cfg.model_name, layer_keys, layers)

    all_rows: list[dict] = []
    per_budget: dict[str, dict] = {}
    for budget in cfg.budgets:
        packs = _resolve_packs(cfg.pack_path, budget, cfg.model_name)
        assert sorted(packs.keys()) == layers, (
            f"pack layers {sorted(packs.keys())} != score-cache layers {layers}"
        )
        rows, verdict = _audit_one_budget(
            packs, layer_keys, layers, rope_ready, get_cos_sin, cfg.n_sinks
        )
        for r in rows:
            r["model"] = cfg.model_label
            r["budget"] = float(budget)
            all_rows.append(r)
        per_budget[f"{budget:g}"] = verdict
        print(
            f"[budget {budget:g}] pooled sink DISTORTION share = "
            f"{verdict['sink_dist_share_pct']:.3f}%  |  sink ATTENTION mass = "
            f"{verdict['sink_attn_mass_pct']:.3f}%",
            flush=True,
        )

    df = pd.DataFrame(all_rows)[
        [
            "model",
            "budget",
            "layer",
            "S",
            "sink_distortion",
            "total_distortion",
            "sink_dist_share",
            "sink_attn_mass",
        ]
    ]
    write_metrics(run, df)

    # ---- pre-registered gate: >=5% at EITHER budget ⇒ CONFIRM carve-out -----
    confirm = any(v["sink_dist_share_pct"] >= GATE_PCT for v in per_budget.values())
    verdict = dict(
        task="storm Task-3 sink carve-out audit",
        model=cfg.model_label,
        n_sinks=cfg.n_sinks,
        gate_pct=GATE_PCT,
        per_budget=per_budget,
        gate_outcome=(
            "CONFIRM carve-out" if confirm else "honest null (W-metric validated)"
        ),
        carve_out_confirmed=bool(confirm),
        git_sha=git_sha(),
    )
    (run / "verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n" + "=" * 78)
    print("STORM TASK-3 VERDICT — sink carve-out audit")
    print("=" * 78)
    print(json.dumps(verdict, indent=2))
    print(f"\n-> {run}")
    return run


if __name__ == "__main__":
    main(tyro.cli(Config))
