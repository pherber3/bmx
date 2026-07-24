"""K4 local-levers Task 4: determinant-Jensen Gate-A anchor.

Gate A's honest negative — a corpus-fitted spectral basis retains only
~0.5-0.7 of the per-sequence oracle's win, FLAT under corpus tripling — has a
two-line theory anchor (math review 2026-07-24 §6, spec Part 2). Under
continuous high-rate allocation, fixed shared W, pooled fit Σ̄ = E_s[Σ_s]:

    Identity: E_s[D_pool(s)] = C·GM(λ̄)·4^{-B̄}    exactly.
    Bound:    R = E_s[D_oracle]/E_s[D_pool]
                = E_s[det^{1/C}(Σ_s)] / det^{1/C}(Σ̄)  <= 1   (Minkowski)

R is a POPULATION functional — corpus-size-independent, matching the measured
flatness signature. This experiment computes R_pred (`jensen_gap_report`,
budget-free) from cached per-sequence moments and R_discrete (the SAME
moments run through the real tier allocator, `allocate_bits_from_variance`)
at each budget, then reports the pre-registered agreement readout.

All moment/eig math is fp64 (this experiment is pure analysis on cached
moments — no cache re-quantization, no GPU).
"""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.codecs import allocate_bits_from_variance
from bmx.cache.spectral import jensen_gap_report, key_second_moment
from experiments._k4_common import (
    load_layer_keys,
    per_cache_weighted_moments,
    setup_rope,
)

_TIERS = (0, 2, 3, 4, 5, 6, 8)  # the standard grid pack fitting allocates on


@dataclasses.dataclass
class Config:
    cache_paths: tuple[str, ...]  # within-domain (wiki) fleet
    cache_paths_alt: tuple[str, ...] = ()  # cross-domain (code) fleet, optional
    model_label: str = "gpt2"
    model_name: str = ""  # HF repo id for RoPE; empty => no-RoPE (ones/zeros) tables
    budgets: tuple[float, ...] = (2.2, 2.5)
    w_source: str = "corpus"
    ridge: float = 1e-3
    position_stride: int = 8
    n_flat: int = 3  # corpus-flatness probe: first n_flat caches vs all
    seed: int = 0
    out_root: str = ""


def _whitened_moments(
    layer_keys_list: list[dict[int, dict[str, torch.Tensor]]],
    get_cos_sins: list,
    rope_ready: bool,
    layer_i: int,
    cfg: Config,
) -> tuple[list[torch.Tensor], list[int], torch.Tensor, torch.Tensor]:
    """Per-cache whitened moments T_s = Wh @ Σ_s @ Wh (fp64) for one layer,
    under the IDENTICAL conventions pack fitting uses (per_cache_weighted_moments),
    plus the per-cache token-row counts (n_rows[s] = M_parts[s].shape[0], read
    off the actual data — never hardcoded) and the shared (Wh, Wh_inv)."""
    pcm = per_cache_weighted_moments(
        layer_keys_list,
        get_cos_sins,
        rope_ready,
        layer_i,
        w_source=cfg.w_source,
        ridge=cfg.ridge,
        position_stride=cfg.position_stride,
    )
    T_list = []
    n_rows = []
    for M in pcm.M_parts:
        Sigma = key_second_moment(M)  # fp64 (C, C), raw per-cache moment
        T = pcm.Wh @ Sigma @ pcm.Wh
        T_list.append(T)
        n_rows.append(int(M.shape[0]))
    return T_list, n_rows, pcm.Wh, pcm.Wh_inv


def _gm(evals: torch.Tensor) -> float:
    """det^{1/C}(M) from its (clamped >=0) eigenvalues, overflow-safe."""
    return float(torch.exp(evals.clamp_min(1e-300).log().mean()))


def _discrete_readout(T_list: list[torch.Tensor], budget: float) -> dict:
    """r_discrete + identity_check for one layer at one budget, per the spec's
    real-allocator recipe (Part 2 / math review #6):

    oracle: per-cache eig(T_s) -> lam_s; own allocation b_s = allocate_bits_
    from_variance(lam_s, budget, TIERS); D_oracle_disc(s) = sum_i lam_s,i * 4^-b_i.

    pooled: (E, lam_bar) = eig(mean_s T_s); pooled allocation b_bar computed
    ONCE from lam_bar; per cache D_pool_disc(s) = sum_i diag(E^T T_s E)_i * 4^-b_bar_i.

    identity_check = mean_s[D_pool_disc(s)] / (C * GM(lam_bar) * 4^-budget) —
    deviation from 1.0 measures how far the discrete/zero-bit regime is from
    the theorem's continuous-allocation regime (pre-stated gap attribution).
    """
    C = T_list[0].shape[0]
    pooled = sum(T_list) / len(T_list)
    pooled = 0.5 * (pooled + pooled.mT)
    lam_bar, E = torch.linalg.eigh(pooled)
    lam_bar = lam_bar.clamp_min(0.0)
    b_bar = allocate_bits_from_variance(lam_bar, budget, _TIERS).double()
    charge_bar = torch.pow(4.0, -b_bar)  # (C,)

    d_oracle_list, d_pool_list = [], []
    for T_s in T_list:
        lam_s = torch.linalg.eigvalsh(0.5 * (T_s + T_s.mT)).clamp_min(0.0)
        b_s = allocate_bits_from_variance(lam_s, budget, _TIERS).double()
        d_oracle_list.append(float((lam_s * torch.pow(4.0, -b_s)).sum()))

        diag_pooled_basis = torch.einsum("ci,cd,di->i", E, T_s, E).clamp_min(0.0)
        d_pool_list.append(float((diag_pooled_basis * charge_bar).sum()))

    mean_d_oracle = sum(d_oracle_list) / len(d_oracle_list)
    mean_d_pool = sum(d_pool_list) / len(d_pool_list)
    r_discrete = mean_d_oracle / mean_d_pool

    gm_lam_bar = _gm(lam_bar)
    continuous_pool = C * gm_lam_bar * (4.0**-budget)
    identity_check = mean_d_pool / continuous_pool

    return dict(
        r_discrete=r_discrete,
        identity_check=identity_check,
        mean_d_oracle=mean_d_oracle,
        mean_d_pool=mean_d_pool,
    )


def main(cfg: Config):
    assert cfg.cache_paths, "cache_paths must be non-empty"
    assert cfg.n_flat >= 1, f"n_flat must be >= 1; got {cfg.n_flat}"
    assert cfg.n_flat <= len(cfg.cache_paths), (
        f"n_flat={cfg.n_flat} exceeds len(cache_paths)={len(cfg.cache_paths)}"
    )

    run = (
        create_run("k4_jensen_gap", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_jensen_gap", cfg)
    )
    model_label = cfg.model_label or "unknown"

    layer_keys_list = [load_layer_keys(p) for p in cfg.cache_paths]
    layers = sorted(layer_keys_list[0].keys())
    for lk in layer_keys_list[1:]:
        assert sorted(lk.keys()) == layers, "cache_paths disagree on layer set"

    rope_ready = False
    get_cos_sins = []
    for lk in layer_keys_list:
        ready, get_cos_sin = setup_rope(cfg.model_name, lk, layers)
        rope_ready = rope_ready or ready
        get_cos_sins.append(get_cos_sin)

    alt_layer_keys_list: list = []
    alt_get_cos_sins: list = []
    if cfg.cache_paths_alt:
        alt_layer_keys_list = [load_layer_keys(p) for p in cfg.cache_paths_alt]
        for lk in alt_layer_keys_list:
            assert sorted(lk.keys()) == layers, (
                "cache_paths_alt disagree on layer set with cache_paths"
            )
        for lk in alt_layer_keys_list:
            ready, get_cos_sin = setup_rope(cfg.model_name, lk, layers)
            alt_get_cos_sins.append(get_cos_sin)

    rows: list[dict] = []
    per_layer_verdict: dict[int, dict] = {}

    for layer_i in layers:
        T_list, n_rows, _Wh, _Wh_inv = _whitened_moments(
            layer_keys_list, get_cos_sins, rope_ready, layer_i, cfg
        )
        # The discrete readout pools per-cache moments UNWEIGHTED while the
        # debias treats the pool's sample size as sum(n_rows) (concatenation
        # semantics) — the two agree only for equal per-cache row counts.
        # Fail fast rather than silently disagree on a future unequal fleet.
        assert len(set(n_rows)) == 1, (
            f"per-cache row counts must be equal (unweighted pooled mean vs "
            f"sum-n debias semantics); got {sorted(set(n_rows))}"
        )

        report_all = jensen_gap_report(T_list, n_rows=n_rows)
        report_flat = jensen_gap_report(
            T_list[: cfg.n_flat], n_rows=n_rows[: cfg.n_flat]
        )
        flatness_delta = abs(report_flat["r_pred"] - report_all["r_pred"])
        flatness_delta_debiased = abs(
            report_flat["r_pred_debiased"] - report_all["r_pred_debiased"]
        )

        per_budget: dict[str, dict] = {}
        for budget in cfg.budgets:
            disc = _discrete_readout(T_list, budget)
            per_budget[f"{budget:g}"] = disc

        mixed_r_pred = None
        within_r_pred = report_all["r_pred"]
        mixed_r_pred_debiased = None
        within_r_pred_debiased = report_all["r_pred_debiased"]
        if alt_layer_keys_list:
            T_alt, n_rows_alt, _, _ = _whitened_moments(
                alt_layer_keys_list, alt_get_cos_sins, rope_ready, layer_i, cfg
            )
            # Mixed-domain diagnostic: pool wiki+code moments under ONE shared
            # frame. The alt fleet's own whitener differs from the primary
            # fleet's (each is its own corpus-pooled W); re-whiten the alt
            # fleet's raw per-cache Sigma with the PRIMARY fleet's Wh so both
            # populations share one frame (the theorem's "fixed shared W"
            # precondition) — recompute T_alt against _Wh directly rather
            # than reusing alt's own whitener.
            pcm_alt = per_cache_weighted_moments(
                alt_layer_keys_list,
                alt_get_cos_sins,
                rope_ready,
                layer_i,
                w_source=cfg.w_source,
                ridge=cfg.ridge,
                position_stride=cfg.position_stride,
            )
            T_alt_shared = [_Wh @ key_second_moment(M) @ _Wh for M in pcm_alt.M_parts]
            report_mixed = jensen_gap_report(
                T_list + T_alt_shared, n_rows=n_rows + n_rows_alt
            )
            mixed_r_pred = report_mixed["r_pred"]
            mixed_r_pred_debiased = report_mixed["r_pred_debiased"]

        row = dict(
            model=model_label,
            layer=layer_i,
            gm_pool=report_all["gm_pool"],
            mean_gm_seq=report_all["mean_gm_seq"],
            r_pred=report_all["r_pred"],
            log_gap=report_all["log_gap"],
            n_seq=report_all["n_seq"],
            n_clamped=report_all["n_clamped"],
            r_pred_flat=report_flat["r_pred"],
            flatness_delta=flatness_delta,
            mixed_r_pred=mixed_r_pred if mixed_r_pred is not None else float("nan"),
            within_r_pred=within_r_pred,
            r_pred_debiased=report_all["r_pred_debiased"],
            bias_factor_seq=report_all["bias_factor_seq"],
            bias_factor_pool=report_all["bias_factor_pool"],
            r_pred_flat_debiased=report_flat["r_pred_debiased"],
            flatness_delta_debiased=flatness_delta_debiased,
            mixed_r_pred_debiased=(
                mixed_r_pred_debiased
                if mixed_r_pred_debiased is not None
                else float("nan")
            ),
            within_r_pred_debiased=within_r_pred_debiased,
        )
        for budget in cfg.budgets:
            disc = per_budget[f"{budget:g}"]
            row[f"r_discrete_b{budget:g}"] = disc["r_discrete"]
            row[f"identity_check_b{budget:g}"] = disc["identity_check"]
            row[f"abs_gap_b{budget:g}"] = abs(report_all["r_pred"] - disc["r_discrete"])
            row[f"abs_gap_debiased_b{budget:g}"] = abs(
                report_all["r_pred_debiased"] - disc["r_discrete"]
            )
        rows.append(row)
        per_layer_verdict[layer_i] = dict(row)

        print(
            f"[layer {layer_i:2d}] r_pred={report_all['r_pred']:.4f} "
            f"r_pred_debiased={report_all['r_pred_debiased']:.4f} "
            f"log_gap={report_all['log_gap']:.4f} "
            + " ".join(
                f"r_disc@{b:g}={per_budget[f'{b:g}']['r_discrete']:.4f}"
                for b in cfg.budgets
            ),
            flush=True,
        )

    df = pd.DataFrame(rows)
    write_metrics(run, df)

    verdict = _jensen_verdict(df, cfg)
    (run / "jensen_verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n" + "=" * 88)
    print("JENSEN-GAP VERDICT — determinant-Jensen Gate-A anchor")
    print("=" * 88)
    print(json.dumps(verdict, indent=2))
    print(f"\nTotal rows: {len(df)}")
    print(f"-> {run}")

    return run


def _jensen_verdict(df: pd.DataFrame, cfg: Config) -> dict:
    r_pred_mean = float(df.r_pred.mean())
    r_pred_flat_mean = float(df.r_pred_flat.mean())
    flatness_delta_mean = float(df.flatness_delta.mean())

    r_pred_debiased_mean = float(df.r_pred_debiased.mean())
    bias_factor_seq_mean = float(df.bias_factor_seq.mean())
    bias_factor_pool_mean = float(df.bias_factor_pool.mean())
    r_pred_flat_debiased_mean = float(df.r_pred_flat_debiased.mean())
    flatness_delta_debiased_mean = float(df.flatness_delta_debiased.mean())

    per_budget: dict[str, dict] = {}
    match_all = True
    match_debiased_all = True
    for budget in cfg.budgets:
        b = f"{budget:g}"
        r_discrete_mean = float(df[f"r_discrete_b{b}"].mean())
        identity_check_mean = float(df[f"identity_check_b{b}"].mean())
        abs_gap_mean = abs(r_pred_mean - r_discrete_mean)
        match = bool(abs_gap_mean <= 0.10)
        match_all = match_all and match

        abs_gap_debiased_mean = abs(r_pred_debiased_mean - r_discrete_mean)
        match_debiased = bool(abs_gap_debiased_mean <= 0.10)
        match_debiased_all = match_debiased_all and match_debiased

        per_budget[b] = dict(
            r_discrete=r_discrete_mean,
            identity_check=identity_check_mean,
            abs_gap=abs_gap_mean,
            match=match,
            abs_gap_debiased=abs_gap_debiased_mean,
            match_debiased=match_debiased,
        )

    mixed_r_pred_mean = float(df.mixed_r_pred.mean()) if cfg.cache_paths_alt else None
    within_r_pred_mean = float(df.within_r_pred.mean())
    mixed_r_pred_debiased_mean = (
        float(df.mixed_r_pred_debiased.mean()) if cfg.cache_paths_alt else None
    )
    within_r_pred_debiased_mean = float(df.within_r_pred_debiased.mean())

    return dict(
        r_pred=r_pred_mean,
        per_budget=per_budget,
        match=match_all,
        flatness=dict(
            n_flat=cfg.n_flat,
            r_pred_at_n_flat=r_pred_flat_mean,
            r_pred_at_all=r_pred_mean,
            delta=flatness_delta_mean,
            r_pred_at_n_flat_debiased=r_pred_flat_debiased_mean,
            r_pred_at_all_debiased=r_pred_debiased_mean,
            delta_debiased=flatness_delta_debiased_mean,
        ),
        mixed_domain=dict(
            mixed_r_pred=mixed_r_pred_mean,
            within_r_pred=within_r_pred_mean,
            mixed_r_pred_debiased=mixed_r_pred_debiased_mean,
            within_r_pred_debiased=within_r_pred_debiased_mean,
        )
        if cfg.cache_paths_alt
        else None,
        n_layers=int(df.shape[0]),
        wishart_debiasing=dict(
            post_hoc=True,
            note=(
                "r_pred_debiased/match_debiased are a LABELED POST-HOC analysis "
                "(closed-form Bartlett/digamma Wishart log-det debiasing, "
                "math review finding #7 connection) — the PRE-REGISTERED "
                "readout is `match` on raw r_pred above; match_debiased is "
                "reported alongside it, not in place of it."
            ),
            r_pred_debiased=r_pred_debiased_mean,
            bias_factor_seq=bias_factor_seq_mean,
            bias_factor_pool=bias_factor_pool_mean,
            match_debiased=match_debiased_all,
        ),
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
