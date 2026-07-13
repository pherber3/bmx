"""K4: corpus pack fitting — fit per-layer SpectralBasis from N corpus caches
and write ONE pack file (+ a small spectra parquet).

Σ_k for each layer is the concat of ALL corpus caches' k_pre matrices (the
deployment-realistic fit: one basis per layer, calibrated on a corpus, not on
the sequence being scored — see k4_spectra.py's "corpus" fit_mode).

W (the query second moment that defines the whitener) has two sources:
  - "corpus": pooled from the corpus caches' OWN stored queries — each cache
    contributes its own query_position_moment (using its own cos/sin, since S
    may differ per cache), averaged across caches. This is the deployment-
    grade choice: at pack-fit time we only ever have corpus-side queries, not
    the queries of the sequence the pack will later score (fixing the
    referee's circularity concern from k4_spectra's oracle/heldout modes).
  - "none": identity_whitener — the unweighted-KLT fallback pack.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics
from bmx.cache.collect import to_matrix
from bmx.cache.spectral import (
    SpectralBasis,
    assemble_whitener,
    fit_spectral_basis,
    identity_whitener,
    pack_from_basis,
    save_pack_file,
)
from experiments._k4_common import corpus_query_moment, load_layer_keys, setup_rope

_W_SOURCES = {"corpus", "none"}


@dataclasses.dataclass
class Config:
    corpus_cache_paths: tuple[str, ...]
    out_path: str
    model_label: str = ""
    model_name: str = ""  # HF repo id for RoPE; empty => no-RoPE (ones/zeros) tables
    budgets: tuple[float, ...] = (2.0, 2.2, 2.5, 2.7, 3.0, 3.2)
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8)
    group: int = 64
    ridge: float = 1e-3
    position_stride: int = 8
    w_source: str = "corpus"
    out_root: str = ""


def main(cfg: Config):
    assert cfg.w_source in _W_SOURCES, f"w_source={cfg.w_source!r} not in {_W_SOURCES}"
    assert len(cfg.corpus_cache_paths) >= 1, "corpus_cache_paths must be non-empty"

    run = (
        create_run("k4_fit_packs", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_fit_packs", cfg)
    )

    per_cache_layer_keys = [load_layer_keys(p) for p in cfg.corpus_cache_paths]

    layers = sorted(per_cache_layer_keys[0].keys())
    for lk in per_cache_layer_keys[1:]:
        assert sorted(lk.keys()) == layers, "corpus caches disagree on layer set"

    # RoPE setup, per corpus cache (S may differ per cache).
    rope_ready = False
    get_cos_sins = []
    for lk in per_cache_layer_keys:
        ready, get_cos_sin = setup_rope(cfg.model_name, lk, layers)
        rope_ready = rope_ready or ready
        get_cos_sins.append(get_cos_sin)

    bases: dict[int, SpectralBasis] = {}
    rows: list[dict] = []
    model_label = cfg.model_label or "unknown"

    for layer_i in layers:
        # ---- Σ_k: concat of ALL corpus caches' k_pre matrices -------------
        h_kv = d = None
        M_parts = []
        for lk in per_cache_layer_keys:
            k_pre_t = lk[layer_i]["k_pre"]
            this_h_kv, S_c, this_d = k_pre_t.shape
            if h_kv is None:
                h_kv, d = this_h_kv, this_d
            else:
                assert (this_h_kv, this_d) == (h_kv, d), (
                    f"corpus cache layer{layer_i}.k_pre shape "
                    f"{tuple(k_pre_t.shape)} incompatible with (h_kv={h_kv}, d={d})"
                )
            M_parts.append(to_matrix(k_pre_t))
        M_fit = torch.cat(M_parts, dim=0)
        C = h_kv * d

        # ---- W: pooled query second moment, or identity --------------------
        if cfg.w_source == "corpus":
            W_blocks = corpus_query_moment(
                per_cache_layer_keys,
                get_cos_sins,
                rope_ready,
                layer_i,
                h_kv,
                d,
                cfg.position_stride,
            )
            Wh, Wh_inv = assemble_whitener(W_blocks, ridge=cfg.ridge)
        else:  # "none"
            Wh, Wh_inv = identity_whitener(C)

        basis = fit_spectral_basis(M_fit, Wh, Wh_inv)
        bases[layer_i] = basis

        print(
            f"[layer {layer_i}] (h_kv={h_kv}, d={d}, C={C}, "
            f"S_fit={M_fit.shape[0]}) basis fit",
            flush=True,
        )

        for budget in cfg.budgets:
            pack = pack_from_basis(basis, budget, tiers=cfg.tiers, group=cfg.group)
            lam = pack.lam
            am_gm = (lam.mean() / lam.clamp_min(1e-12).log().mean().exp()).item()
            top16_energy = (lam[:16].sum() / lam.sum().clamp_min(1e-12)).item()
            n_zero_dirs = int((pack.bits == 0).sum())
            rows.append(
                dict(
                    model=model_label,
                    layer=layer_i,
                    budget=float(budget),
                    am_gm=am_gm,
                    top16_energy=top16_energy,
                    n_zero_dirs=n_zero_dirs,
                )
            )

    save_pack_file(
        cfg.out_path,
        bases,
        cfg.budgets,
        tiers=cfg.tiers,
        group=cfg.group,
        meta={
            "model_label": model_label,
            "git_sha": git_sha(),
            "corpus_cache_paths": list(cfg.corpus_cache_paths),
            "w_source": cfg.w_source,
            "ridge": cfg.ridge,
        },
    )

    df = pd.DataFrame(rows)
    write_metrics(run, df)

    print(f"\nWrote pack file -> {cfg.out_path}")
    print(f"-> {run}")

    return run


if __name__ == "__main__":
    main(tyro.cli(Config))
