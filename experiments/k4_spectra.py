"""K4 Stage 0: weighted/unweighted spectra + basis-transfer gate (G0).

Per layer, fits SpectralPacks under three fit modes (oracle / heldout /
corpus), scores them REGION-MATCHED on the tail half of the cache against the
uniform k2b-K reference, and emits per-layer spectra stats + the G0 retention
verdict. See docs/superpowers/specs/2026-07-12-k4-spectral-codec-design.md §4.

Fit-mode scope: only the KEY statistics (the second moment Σ_k the basis is
fit on) vary by fit_mode (oracle: all S rows of the scored cache; heldout:
rows [:S//2] of the scored cache; corpus: concat of OTHER caches' same-layer
k_pre matrices). The query moment W — and hence the whitener — is ALWAYS
estimated from the SCORED cache's own queries in every mode: queries are
probe-side information, known at read time regardless of which corpus the
key basis was fit on.

The `lowrank_rtn_channel` reference arm is fit and scored per layer as a
uniform-codec baseline for the G0 retention ratio; its rows live in the SAME
metrics.parquet with fit_mode="reference" (weighted=False, budget/spectra
stats NaN — no spectral pack exists for it; its bpe already counts all
metadata per-sequence, so bpe_model == bpe_skeptic == bpe_skeptic_deploy).
"""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.codecs import quantize_cache
from bmx.cache.collect import load_cache, to_matrix
from bmx.cache.rope import apply_rope
from bmx.cache.spectral import (
    assemble_whitener,
    fit_spectral_basis,
    identity_whitener,
    pack_from_basis,
    query_position_moment,
    skeptic_charge,
    spectral_quantize,
)
from experiments._k4_common import DEPLOY_S, _score_tail, load_layer_keys, setup_rope


@dataclasses.dataclass
class Config:
    cache_path: str
    corpus_cache_paths: tuple[str, ...] = ()
    model_label: str = ""
    model_name: str = ""  # HF repo id for RoPE; empty => stored-basis logit only
    budgets: tuple[float, ...] = (2.5,)
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8)
    group: int = 64
    ridge: float = 1e-3
    position_stride: int = 8
    ref_rank: int = 16  # the uniform k2b-K reference arm
    ref_bits: int = 3
    seed: int = 0
    max_layers: int = 0
    out_root: str = ""


def main(cfg: Config):
    run = (
        create_run("k4_spectra", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_spectra", cfg)
    )

    corpus_caches = [load_cache(p) for p in cfg.corpus_cache_paths]

    layer_keys = load_layer_keys(cfg.cache_path)

    layers = sorted(layer_keys.keys())
    if cfg.max_layers > 0:
        layers = layers[: cfg.max_layers]

    # ---- RoPE setup (same pattern as k2d_lrtq_gate.py) ---------------------
    rope_ready, get_cos_sin = setup_rope(cfg.model_name, layer_keys, layers)

    rows: list[dict] = []
    model_label = cfg.model_label or "unknown"
    headline_col = "logit_rope" if rope_ready else "logit"

    def emit(row: dict) -> None:
        rows.append(row)
        print(
            f"  layer={row['layer']:2d} weighted={row['weighted']!s:5s} "
            f"fit_mode={row['fit_mode']:8s} budget={row['budget']:.2f}  "
            f"bpe_model={row['bpe_model']:.3f}  rel_fro={row['rel_fro']:.4f}  "
            f"logit={row['logit']:.4f}  logit_rope={row['logit_rope']:.4f}",
            flush=True,
        )

    for layer_i in layers:
        kinds_map = layer_keys[layer_i]
        k_pre_t = kinds_map["k_pre"]  # (h_kv, S, d) fp16, pre-RoPE
        q_t = kinds_map["q"]  # (h, T, d) fp16
        h_kv, S, d = k_pre_t.shape
        C = h_kv * d
        Q_fp32 = q_t.float()

        if rope_ready:
            cos_l, sin_l = get_cos_sin(S)
        else:
            cos_l = torch.ones(S, d)
            sin_l = torch.zeros(S, d)
        K_post_true = apply_rope(k_pre_t.float(), cos_l, sin_l) if rope_ready else None

        M = to_matrix(k_pre_t)  # (S, C) fp32
        tail = slice(S // 2, S)

        print(f"\n[layer {layer_i}] (h_kv={h_kv}, S={S}, d={d}, C={C})", flush=True)

        # Query second moment: always from THIS cache's own queries.
        W_blocks = query_position_moment(
            q_t.float(), cos_l, sin_l, h_kv, position_stride=cfg.position_stride
        )
        Wh, Wh_inv = assemble_whitener(W_blocks, ridge=cfg.ridge)
        Ih, Ih_inv = identity_whitener(C)

        # Fit matrices per mode.
        fit_matrices: dict[str, torch.Tensor] = {
            "oracle": M,
            "heldout": M[: S // 2],
        }
        if corpus_caches:
            others = []
            for other in corpus_caches:
                other_k = other[f"layer{layer_i}.k_pre"]
                assert other_k.shape[0] == h_kv and other_k.shape[2] == d, (
                    f"corpus cache layer{layer_i}.k_pre shape {tuple(other_k.shape)} "
                    f"incompatible with (h_kv={h_kv}, d={d})"
                )
                others.append(to_matrix(other_k))
            fit_matrices["corpus"] = torch.cat(others, dim=0)

        for weighted, (basis_h, basis_h_inv) in (
            (True, (Wh, Wh_inv)),
            (False, (Ih, Ih_inv)),
        ):
            for fit_mode, M_fit in fit_matrices.items():
                basis = fit_spectral_basis(M_fit, basis_h, basis_h_inv)
                for budget in cfg.budgets:
                    pack = pack_from_basis(
                        basis, budget, tiers=cfg.tiers, group=cfg.group
                    )
                    M_hat, bpe_model = spectral_quantize(M, pack)
                    rf, lg, lg_rope = _score_tail(
                        M_hat,
                        h_kv,
                        tail,
                        K_post_true,
                        Q_fp32,
                        cos_l,
                        sin_l,
                        rope_ready,
                        k_pre_t,
                        M,
                    )
                    bpe_skeptic = bpe_model + skeptic_charge(C, S, cfg.tiers)
                    bpe_skeptic_deploy = bpe_model + skeptic_charge(
                        C, DEPLOY_S, cfg.tiers
                    )

                    lam = pack.lam
                    am_gm = (
                        lam.mean() / lam.clamp_min(1e-12).log().mean().exp()
                    ).item()
                    top16_energy = (lam[:16].sum() / lam.sum().clamp_min(1e-12)).item()
                    n_zero_dirs = int((pack.bits == 0).sum())

                    emit(
                        dict(
                            model=model_label,
                            layer=layer_i,
                            weighted=weighted,
                            fit_mode=fit_mode,
                            budget=float(budget),
                            bpe_model=bpe_model,
                            bpe_skeptic=bpe_skeptic,
                            bpe_skeptic_deploy=bpe_skeptic_deploy,
                            rel_fro=rf,
                            logit=lg,
                            logit_rope=lg_rope,
                            am_gm=am_gm,
                            top16_energy=top16_energy,
                            n_zero_dirs=n_zero_dirs,
                        )
                    )

        # Reference arm: lowrank_rtn_channel @ ref_bits, ref_rank, tail-scored.
        M_hat_ref, bpe_ref = quantize_cache(
            "lowrank_rtn_channel",
            M,
            bits=cfg.ref_bits,
            rank=cfg.ref_rank,
            group=cfg.group,
            seed=cfg.seed,
        )
        rf_ref, lg_ref, lg_rope_ref = _score_tail(
            M_hat_ref,
            h_kv,
            tail,
            K_post_true,
            Q_fp32,
            cos_l,
            sin_l,
            rope_ready,
            k_pre_t,
            M,
        )
        # Reference rows share the full schema: no spectral pack exists, so
        # budget/spectra stats are NaN; lowrank_rtn_channel's bpe already
        # counts all metadata per-sequence, so all three bpe views coincide.
        emit(
            dict(
                model=model_label,
                layer=layer_i,
                weighted=False,
                fit_mode="reference",
                budget=float("nan"),
                bpe_model=bpe_ref,
                bpe_skeptic=bpe_ref,
                bpe_skeptic_deploy=bpe_ref,
                rel_fro=rf_ref,
                logit=lg_ref,
                logit_rope=lg_rope_ref,
                am_gm=float("nan"),
                top16_energy=float("nan"),
                n_zero_dirs=float("nan"),
            )
        )

    df = pd.DataFrame(rows)
    write_metrics(run, df)

    # ---- G0 verdict: transfer retention at the first (reference) budget ----
    ref_budget = cfg.budgets[0]
    ref_by_layer = df[df.fit_mode == "reference"].set_index("layer")[headline_col]

    oracle_sub = df[
        (df.fit_mode == "oracle") & (df.weighted) & (df.budget == ref_budget)
    ].set_index("layer")[headline_col]

    verdict: dict = {"headline_metric": headline_col, "ref_budget": ref_budget}
    if not oracle_sub.empty:
        verdict["mean_headline_oracle"] = float(oracle_sub.mean())
    for mode in ("heldout", "corpus"):
        sub = df[(df.fit_mode == mode) & (df.weighted) & (df.budget == ref_budget)]
        if sub.empty:
            continue
        retentions = []
        for _, r in sub.iterrows():
            ref_val = ref_by_layer.loc[r.layer]
            spectral_m = r[headline_col]
            spectral_oracle = oracle_sub.loc[r.layer]
            win_m = ref_val / spectral_m
            win_oracle = ref_val / spectral_oracle
            retentions.append(win_m / win_oracle)
        retention = float(pd.Series(retentions).mean())
        verdict[f"retention_{mode}"] = retention
        verdict[f"g0_pass_{mode}"] = bool(retention >= 0.9)
        verdict[f"mean_headline_{mode}"] = float(sub[headline_col].mean())

    (run / "g0_verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n" + "=" * 88)
    print("G0 VERDICT — basis-transfer retention at budget", ref_budget)
    print("=" * 88)
    print(json.dumps(verdict, indent=2))
    n_ref = int((df.fit_mode == "reference").sum())
    print(f"\nTotal rows: {len(df)}  (including {n_ref} reference rows)")
    print(f"-> {run}")

    return run


if __name__ == "__main__":
    main(tyro.cli(Config))
