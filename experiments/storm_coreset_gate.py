"""Storm Task-2 — leverage-score token coreset vs uniform quantization (plan
`docs/superpowers/plans/2026-07-26-storm-gates.md` Task 2; briefing Tier-1 #2 —
the one unplayed axis: token/row SELECTION).

Question: at matched TOTAL bits, does keeping the top-k tokens by leverage
score (of pre-RoPE K's top-r left-singular subspace; sinks 0..3 ALWAYS kept)
EXACTLY in fp16 and dropping the rest entirely beat quantizing all tokens
uniformly — on the query-weighted logit instrument with real stored queries
and RoPE-at-read?

Pre-registered gate (verbatim, plan Task 2):
    coreset (or the mixed arm) beats the best uniform arm at matched bpe on
    >= 90% of layers at >= 1 budget  =>  CONFIRM (opens the selection axis;
    Llama rides next rental). Loses everywhere  =>  honest negative with the
    mechanism (expected failure mode: softmax-denominator bias / worst-case
    needle loss — report which).

Coreset bit accounting (plan-locked; pinned by test_coreset_bit_accounting)
---------------------------------------------------------------------------
Retained tokens are stored at full fp16 over all C = h_kv*d coordinates plus
an explicit position index of ceil(log2(S)) bits each, amortized over ALL S
tokens of the cache:

    total_bits(k)   = k * (16*C + ceil(log2 S))
    bits/token      = 16*k*C/S + ceil(log2 S)*k/S          (the plan's formula)
    bpe             = bits/token / C

Matching rule (no interpolation needed — stated per the task brief): the
coreset's k knob is fine-grained (one token = 1/S of kept fraction), so for
every uniform arm we match DOWNWARD, k = floor(bpe_uniform * S*C / (16*C +
ceil(log2 S))). The coreset never receives more bits than the uniform arm it
challenges; the deficit is < one token's cost (< 0.016 bpe at gpt2 geometry,
< 0.008 at qwen3-0.6b) — conservative AGAINST the uniform arm never, against
the coreset by at most that sliver.

Dropping convention: a dropped token contributes NOTHING to the stored cache;
in the shipped logit instrument this is embedded as a zero key row, so the
instrument charges the FULL logit q·k_s as error for every dropped position s
(kept rows are exact — the caches are stored fp16, so a kept row round-trips
bit-exactly). The gate metric is the shipped `logit_distortion` (post-RoPE
keys reconstructed via apply-RoPE-at-read, stored pre-RoPE probe queries,
GQA-aware) over the FULL sequence — the K4 tail-slice convention would exempt
first-half drops from scoring entirely, which is meaningless for an eviction
gate.

Mixed arm split rule (stated per the task brief): the RECENT WINDOW = the
last W = floor(S/4) positions, quantized with turboquant_mse (the strongest
shipped streaming uniform codec) at b_w = max(2, floor(budget_level)) bits;
the DISTANT TAIL = positions [0, S-W), kept exactly via the leverage coreset
(sinks always included, leverage rank r = mixed_r) with k_tail rows chosen to
fill the remaining budget:
    k_tail = floor((bpe_target*S*C - W*C*bpe_w - ceil(log2 S)) /
                   (16*C + ceil(log2 S)))
(the lone extra ceil(log2 S) charges the window-boundary integer). If the
budget cannot cover window + sinks, W halves until it can.

Diagnostics reported alongside the gate metric (the two pre-registered
failure-mode fingerprints):
  * drop-mass — the fraction of TRUE causal attention mass (stored queries
    forward-rotated at their true positions [S-T, S), softmax over the full
    causal row) landing on dropped positions: the softmax-DENOMINATOR-bias
    number (that mass has nowhere to go once its tokens are evicted).
  * worst-case — max over (head, query, dropped causal s) of the scaled true
    logit |q_t·k_s|/sqrt(d), vs the max over ALL causal pairs: the needle-loss
    signature (under the zero-row embedding the dropped logit IS the error).

Uniform arms at each budget level: rtn_token and turboquant_mse at bits
{2,3,4} plus the banked K4 spectral packs at budgets {2.2, 2.5, 3.0}
(model-level bpe accounting, the shipped K4 convention — the pack ships with
the model; the coreset's whole charge is per-sequence, which only handicaps
the coreset). Substrate: held-out offset-0 caches (gpt2 S=1024,
qwen3-0.6b S=2048 — regenerate via `experiments/collect_cache.py` if absent);
packs are the banked corpus packs, fit on distinct offset caches.

fp32 experiment path per repo convention (caches fp16, SVD/metrics fp32).
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics
from bmx.cache.codecs import quantize_cache
from bmx.cache.collect import from_matrix, to_matrix
from bmx.cache.metrics import _expand_kv, logit_distortion
from bmx.cache.rope import apply_rope
from bmx.cache.spectral import load_packs, spectral_quantize
from experiments._k4_common import load_layer_keys, setup_rope

N_SINKS = 4  # positions 0..3, the plan's always-resident sink set
GATE_WIN_FRAC = 0.9  # ">= 90% of layers" (pre-registered)
R_GRID = (16, 32, 64)  # leverage subspace ranks (plan: "a small grid")


# ---------------------------------------------------------------------------
# Leverage + selection core (pure; pinned by the offline tests)
# ---------------------------------------------------------------------------


def leverage_scores(
    M: torch.Tensor, r: int, U: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-token leverage scores of the top-r left-singular subspace of M.

    M: (S, C) fp32. lev[s] = ||U_r[s, :]||^2, the squared row norm of the top-r
    left singular vectors — the standard statistical-leverage definition (how
    much of direction s's identity lives in the dominant row space). Values in
    [0, 1], summing to r. Pass a precomputed U (from
    torch.linalg.svd(M, full_matrices=False)) to amortize the SVD across the
    r grid.
    """
    if U is None:
        U, _, _ = torch.linalg.svd(M.float(), full_matrices=False)
    r_eff = min(r, U.shape[1])
    Ur = U[:, :r_eff]
    return (Ur * Ur).sum(dim=1)


def stable_rank(svals: torch.Tensor) -> float:
    """||M||_F^2 / sigma_1^2 from the singular values of M."""
    s = svals.float()
    return float((s * s).sum() / (s[0] * s[0]).clamp_min(1e-30))


def select_coreset(lev: torch.Tensor, k: int, n_sinks: int = N_SINKS) -> torch.Tensor:
    """Keep set: sinks {0..n_sinks-1} PLUS the top-(k - n_sinks) non-sink
    positions by leverage. Returns sorted int64 indices, exactly k of them.
    Deterministic (stable argsort — ties resolve to the earlier position)."""
    S = lev.shape[0]
    assert n_sinks <= k <= S, f"need n_sinks={n_sinks} <= k={k} <= S={S}"
    sinks = torch.arange(n_sinks, dtype=torch.int64)
    rest = lev.clone().float()
    rest[:n_sinks] = float("-inf")  # sinks are unconditionally in; rank the rest
    order = torch.argsort(rest, descending=True, stable=True)
    keep = torch.cat([sinks, order[: k - n_sinks]])
    keep = keep.sort().values
    assert keep.unique().numel() == k
    return keep


def coreset_reconstruct(M: torch.Tensor, keep_idx: torch.Tensor) -> torch.Tensor:
    """Kept rows exact, dropped rows zero — the instrument's embedding of
    eviction (the dropped token's key error is the full key)."""
    M_hat = torch.zeros_like(M)
    M_hat[keep_idx] = M[keep_idx]
    return M_hat


# ---------------------------------------------------------------------------
# Bit accounting (plan-locked; pinned by test_coreset_bit_accounting)
# ---------------------------------------------------------------------------


def token_index_bits(S: int) -> int:
    """ceil(log2(S)) bits to name one retained position out of S."""
    return int(math.ceil(math.log2(S)))


def coreset_total_bits(k: int, S: int, C: int) -> int:
    """Total stored bits for a k-token coreset: 16 bits/coordinate over the C
    columns of each retained token, plus one ceil(log2 S)-bit index each."""
    return k * (16 * C + token_index_bits(S))


def coreset_bpe(k: int, S: int, C: int) -> float:
    """Honest bpe: total coreset bits amortized over ALL S*C cache entries."""
    return coreset_total_bits(k, S, C) / (S * C)


def matched_coreset_k(target_bpe: float, S: int, C: int, n_sinks: int = N_SINKS) -> int:
    """Largest k whose coreset bpe does not exceed target_bpe (floor matching —
    the coreset never gets more bits than the uniform arm), clamped to
    [n_sinks, S]."""
    k = int(math.floor(target_bpe * S * C / (16 * C + token_index_bits(S))))
    return max(n_sinks, min(k, S))


def mixed_arm(
    M: torch.Tensor,
    target_bpe: float,
    lev: torch.Tensor,
    *,
    level_bits: float,
    n_sinks: int = N_SINKS,
    window_frac: float = 0.25,
    seed: int = 0,
) -> tuple[torch.Tensor, float, int, int]:
    """Mixed arm: quantized recent window + exact leverage coreset on the
    distant tail (split rule in the module docstring). Returns
    (M_hat, bpe, k_tail, W)."""
    S, C = M.shape
    idxb = token_index_bits(S)
    row_cost = 16 * C + idxb
    budget_total = target_bpe * S * C
    W = max(1, int(S * window_frac))
    b_w = max(2, int(math.floor(level_bits)))
    while True:
        Mw_hat, bpe_w = quantize_cache(
            "turboquant_mse", M[S - W :], bits=b_w, seed=seed
        )
        window_bits = W * C * bpe_w
        k_tail = int(math.floor((budget_total - window_bits - idxb) / row_cost))
        if k_tail >= n_sinks or W <= 8:
            break
        W = W // 2  # budget can't cover window + sinks: shrink the window
    k_tail = max(n_sinks, min(k_tail, S - W))
    keep_tail = select_coreset(lev[: S - W], k_tail, n_sinks=n_sinks)
    M_hat = torch.zeros_like(M)
    M_hat[S - W :] = Mw_hat
    M_hat[keep_tail] = M[keep_tail]
    bpe = (window_bits + k_tail * row_cost + idxb) / (S * C)
    return M_hat, bpe, k_tail, W


# ---------------------------------------------------------------------------
# Diagnostics: drop-mass (softmax-denominator bias) + worst-case needle loss
# ---------------------------------------------------------------------------


def true_causal_logits(
    q_read: torch.Tensor, k_post: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled TRUE logits q_t·k_s/sqrt(d) under the read convention
    (queries already forward-rotated at their true absolute positions for RoPE
    models). q_read: (h, T, d); k_post: (h_kv, S, d), GQA-expanded. Returns
    (logits (h, T, S) fp32, causal (T, S) bool) — query row i at absolute
    position S-T+i sees source s <= S-T+i."""
    q_read = q_read.float()
    k_post = k_post.float()
    h, T, d = q_read.shape
    S = k_post.shape[1]
    k_exp = _expand_kv(k_post, h)
    logits = q_read @ k_exp.transpose(-1, -2) / (d**0.5)  # (h, T, S)
    q_pos = torch.arange(S - T, S).view(T, 1)
    s_pos = torch.arange(S).view(1, S)
    causal = s_pos <= q_pos  # (T, S)
    return logits, causal


def drop_diagnostics(
    logits: torch.Tensor, causal: torch.Tensor, retained_idx: torch.Tensor
) -> tuple[float, float, float]:
    """(drop_mass, worst_dropped_logit, max_true_logit) for one retained set.

    drop_mass: fraction of the true causal attention mass (softmax per (head,
    query) over the full causal row, then averaged over heads and query rows)
    landing on positions NOT in retained_idx — the softmax-denominator-bias
    account. worst_dropped_logit: max |scaled logit| over (head, query,
    dropped causal s) — under the zero-row embedding this IS the max per-query
    logit error induced by dropping. max_true_logit: same max over ALL causal
    pairs, the scale reference."""
    h, T, S = logits.shape
    masked = logits.masked_fill(~causal.view(1, T, S), float("-inf"))
    attn = torch.softmax(masked, dim=-1)  # (h, T, S)
    mass = attn.mean(dim=(0, 1))  # (S,), sums to 1
    retained = torch.zeros(S, dtype=torch.bool)
    retained[retained_idx] = True
    drop_mass = float(mass[~retained].sum())
    abs_l = logits.abs().masked_fill(~causal.view(1, T, S), 0.0)
    max_true = float(abs_l.max())
    dropped_cols = abs_l[:, :, ~retained]
    worst = float(dropped_cols.max()) if dropped_cols.numel() else 0.0
    return drop_mass, worst, max_true


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Config:
    # Held-out (offset-0) cache — distinct from the pack-fit corpus caches.
    cache_path: str = "results/cache/gpt2_1024.safetensors"
    # Banked K4 spectral packs (uniform-arm family #3); "" skips spectral arms.
    pack_path: str = "results/cache/k4_packs_gpt2.safetensors"
    model_label: str = "gpt2"
    model_name: str = ""  # HF repo id for RoPE tables; "" => no-RoPE (gpt2)
    rtn_bits: tuple[int, ...] = (2, 3, 4)
    tq_bits: tuple[int, ...] = (2, 3, 4)
    spectral_budgets: tuple[float, ...] = (2.2, 2.5, 3.0)
    r_grid: tuple[int, ...] = R_GRID
    use_stable_rank_r: bool = True  # add the r = round(stable rank) variant
    mixed_r: int = 32  # leverage rank the mixed arm's tail selection uses
    mixed_window_frac: float = 0.25
    group: int = 64  # rtn_token group (C % group == 0)
    n_sinks: int = N_SINKS
    seed: int = 0
    out_root: str = ""


def _uniform_arms(cfg: Config) -> list[tuple[str, str, float, dict]]:
    """[(arm_id, codec, level, kwargs)] — every uniform comparison point."""
    arms: list[tuple[str, str, float, dict]] = []
    for b in cfg.rtn_bits:
        arms.append((f"rtn_token@{b}", "rtn_token", float(b), dict(bits=int(b))))
    for b in cfg.tq_bits:
        arms.append(
            (f"turboquant_mse@{b}", "turboquant_mse", float(b), dict(bits=int(b)))
        )
    for bud in cfg.spectral_budgets:
        arms.append(
            (f"spectral@{bud:g}", "spectral", float(bud), dict(budget=float(bud)))
        )
    return arms


def _score(
    M_hat: torch.Tensor,
    K_true_read: torch.Tensor,
    Q: torch.Tensor,
    h_kv: int,
    rope_ready: bool,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
) -> float:
    """The shipped instrument: logit_distortion on the read-convention keys
    (apply-RoPE-at-read for RoPE models), stored queries, full sequence."""
    K_hat = from_matrix(M_hat, h_kv).float()
    if rope_ready:
        K_hat = apply_rope(K_hat, cos, sin)
    return logit_distortion(K_true_read, K_hat, Q)


def main(cfg: Config):
    assert Path(cfg.cache_path).exists(), (
        f"cache {cfg.cache_path!r} absent — regenerate via "
        "`uv run python experiments/collect_cache.py --model-name <model> ...`"
    )
    run = (
        create_run("storm_coreset_gate", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("storm_coreset_gate", cfg)
    )

    layer_keys = load_layer_keys(cfg.cache_path)
    layers = sorted(layer_keys.keys())
    rope_ready, get_cos_sin = setup_rope(cfg.model_name, layer_keys, layers)

    spectral_packs: dict[float, dict] = {}
    for bud in cfg.spectral_budgets:
        assert cfg.pack_path and Path(cfg.pack_path).exists(), (
            f"spectral budgets requested but pack {cfg.pack_path!r} absent"
        )
        spectral_packs[float(bud)] = load_packs(cfg.pack_path, bud)

    arms = _uniform_arms(cfg)
    rows: list[dict] = []

    for li in layers:
        kinds = layer_keys[li]
        k_pre = kinds["k_pre"]  # (h_kv, S, d) fp16 pre-RoPE
        h_kv, S, d = k_pre.shape
        C = h_kv * d
        M = to_matrix(k_pre)  # (S, C) fp32
        Q = kinds["q"].float()  # (h, T, d) stored pre-RoPE probe queries
        T = Q.shape[1]

        if rope_ready:
            cos, sin = get_cos_sin(S)
            K_true_read = apply_rope(k_pre.float(), cos, sin)
            q_positions = torch.arange(S - T, S)
            q_read = apply_rope(Q, cos[q_positions], sin[q_positions])
        else:
            cos = sin = None
            K_true_read = k_pre.float()
            q_read = Q

        # One SVD per layer, shared across the r grid + stable-rank variant.
        U, svals, _ = torch.linalg.svd(M, full_matrices=False)
        sr = stable_rank(svals)
        r_variants: dict[str, int] = {f"cs_r{r}": int(r) for r in cfg.r_grid}
        if cfg.use_stable_rank_r:
            r_variants["cs_rstable"] = max(1, min(round(sr), min(S, C)))
        lev_by_variant = {
            name: leverage_scores(M, r, U=U) for name, r in r_variants.items()
        }
        lev_mixed = leverage_scores(M, cfg.mixed_r, U=U)

        # True read-convention logits, once per layer (diagnostics substrate).
        logits_true, causal = true_causal_logits(q_read, K_true_read)
        diag_cache: dict[tuple, tuple[float, float, float]] = {}

        # ---- uniform arms -------------------------------------------------
        uniform_rows: list[dict] = []
        for arm_id, codec, level, kw in arms:
            if codec == "spectral":
                pack = spectral_packs[kw["budget"]][li]
                M_hat, bpe = spectral_quantize(M, pack)
            else:
                M_hat, bpe = quantize_cache(
                    codec, M, seed=cfg.seed, group=cfg.group, **kw
                )
            dist = _score(M_hat, K_true_read, Q, h_kv, rope_ready, cos, sin)
            row = dict(
                model=cfg.model_label,
                level=level,
                kind="uniform",
                arm=arm_id,
                matched_arm="",
                layer=li,
                S=S,
                C=C,
                r=-1,
                bpe_target=float("nan"),
                bpe=bpe,
                k_kept=-1,
                kept_frac=float("nan"),
                dist=dist,
                drop_mass=float("nan"),
                worst_dropped_logit=float("nan"),
                max_true_logit=float("nan"),
                worst_ratio=float("nan"),
            )
            uniform_rows.append(row)
            rows.append(row)

        # ---- coreset + mixed arms, matched per uniform arm ----------------
        for u in uniform_rows:
            target = u["bpe"]
            level = u["level"]
            for name, r in r_variants.items():
                k = matched_coreset_k(target, S, C, n_sinks=cfg.n_sinks)
                keep = select_coreset(lev_by_variant[name], k, n_sinks=cfg.n_sinks)
                M_hat = coreset_reconstruct(M, keep)
                dist = _score(M_hat, K_true_read, Q, h_kv, rope_ready, cos, sin)
                key = (name, k)
                if key not in diag_cache:
                    diag_cache[key] = drop_diagnostics(logits_true, causal, keep)
                dm, worst, max_true = diag_cache[key]
                rows.append(
                    dict(
                        model=cfg.model_label,
                        level=level,
                        kind="coreset",
                        arm=name,
                        matched_arm=u["arm"],
                        layer=li,
                        S=S,
                        C=C,
                        r=r,
                        bpe_target=target,
                        bpe=coreset_bpe(k, S, C),
                        k_kept=k,
                        kept_frac=k / S,
                        dist=dist,
                        drop_mass=dm,
                        worst_dropped_logit=worst,
                        max_true_logit=max_true,
                        worst_ratio=worst / max(max_true, 1e-12),
                    )
                )
            # Mixed arm (split rule in the module docstring).
            M_hat, bpe_m, k_tail, W = mixed_arm(
                M,
                target,
                lev_mixed,
                level_bits=level,
                n_sinks=cfg.n_sinks,
                window_frac=cfg.mixed_window_frac,
                seed=cfg.seed,
            )
            dist = _score(M_hat, K_true_read, Q, h_kv, rope_ready, cos, sin)
            keep_tail = select_coreset(lev_mixed[: S - W], k_tail, n_sinks=cfg.n_sinks)
            retained = torch.cat([keep_tail, torch.arange(S - W, S)])
            key = ("mixed", W, k_tail)
            if key not in diag_cache:
                diag_cache[key] = drop_diagnostics(logits_true, causal, retained)
            dm, worst, max_true = diag_cache[key]
            rows.append(
                dict(
                    model=cfg.model_label,
                    level=level,
                    kind="mixed",
                    arm=f"mixed_r{cfg.mixed_r}",
                    matched_arm=u["arm"],
                    layer=li,
                    S=S,
                    C=C,
                    r=cfg.mixed_r,
                    bpe_target=target,
                    bpe=bpe_m,
                    k_kept=k_tail,
                    kept_frac=(k_tail + W) / S,
                    dist=dist,
                    drop_mass=dm,
                    worst_dropped_logit=worst,
                    max_true_logit=max_true,
                    worst_ratio=worst / max(max_true, 1e-12),
                )
            )
        print(
            f"[layer {li}] stable_rank={sr:.1f} scored "
            f"{len(arms)} uniform + {len(r_variants) + 1} selection variants",
            flush=True,
        )

    df = pd.DataFrame(rows)
    write_metrics(run, df)

    # ---- pre-registered gate, evaluated verbatim --------------------------
    verdict = evaluate_gate(df, cfg)
    (run / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\n" + "=" * 78)
    print("STORM TASK-2 VERDICT — leverage-score token coreset gate")
    print("=" * 78)
    print(json.dumps(verdict, indent=2))
    print(f"\n-> {run}")
    return run


def evaluate_gate(df: pd.DataFrame, cfg: Config) -> dict:
    """Per budget level: per layer, the BEST uniform arm (min distortion); a
    selection variant wins that layer iff its distortion, matched to that best
    arm's bpe, is strictly lower. Gate: any variant reaching >= 90% of layers
    at >= 1 budget level => CONFIRM; else honest negative with the mechanism
    named from the diagnostics."""
    sel = df[df.kind.isin(["coreset", "mixed"])]
    variants = sorted(sel.arm.unique())
    per_level: dict[str, dict] = {}
    confirmed = False
    for level in sorted(df.level.unique()):
        uni = df[(df.kind == "uniform") & (df.level == level)]
        layers = sorted(uni.layer.unique())
        best_by_layer = {}
        for li in layers:
            g = uni[uni.layer == li]
            i = g["dist"].idxmin()
            best_by_layer[li] = (g.loc[i, "arm"], float(g.loc[i, "dist"]))
        stats: dict[str, dict] = {}
        for v in variants:
            wins, dists, dms, ratios, keeps = 0, [], [], [], []
            for li in layers:
                best_arm, best_dist = best_by_layer[li]
                m = sel[
                    (sel.arm == v) & (sel.layer == li) & (sel.matched_arm == best_arm)
                ]
                assert len(m) == 1, f"expected 1 row for {v}/{best_arm}/layer{li}"
                dist_v = float(m["dist"].iloc[0])
                wins += int(dist_v < best_dist)
                dists.append(dist_v)
                dms.append(float(m["drop_mass"].iloc[0]))
                ratios.append(float(m["worst_ratio"].iloc[0]))
                keeps.append(float(m["kept_frac"].iloc[0]))
            stats[v] = dict(
                layer_win_frac=wins / len(layers),
                mean_dist=sum(dists) / len(dists),
                mean_drop_mass=sum(dms) / len(dms),
                max_worst_ratio=max(ratios),
                mean_kept_frac=sum(keeps) / len(keeps),
            )
        best_variant = max(stats, key=lambda v: stats[v]["layer_win_frac"])
        arm_modes = pd.Series([a for a, _ in best_by_layer.values()]).mode()
        level_block = dict(
            best_uniform_arm_mode=str(arm_modes.iloc[0]),
            mean_best_uniform_dist=sum(d for _, d in best_by_layer.values())
            / len(layers),
            n_layers=len(layers),
            per_variant=stats,
            best_variant=best_variant,
            best_variant_win_frac=stats[best_variant]["layer_win_frac"],
        )
        confirmed = confirmed or (level_block["best_variant_win_frac"] >= GATE_WIN_FRAC)
        per_level[f"{level:g}"] = level_block

    # Mechanism naming (pre-registered failure modes), from the diagnostics of
    # the best-matched selection rows across all levels.
    best_rows = sel  # all selection rows carry the diagnostics
    dm_mean = float(best_rows.drop_mass.mean())
    dm_max = float(best_rows.drop_mass.max())
    ratio_max = float(best_rows.worst_ratio.max())
    mechanisms = []
    if dm_mean >= 0.05:
        mechanisms.append(
            f"softmax-denominator bias ({100 * dm_mean:.1f}% of true causal "
            "attention mass lands on dropped tokens, mean over selection rows; "
            f"max {100 * dm_max:.1f}%)"
        )
    if ratio_max >= 0.5:
        mechanisms.append(
            "worst-case needle loss (a dropped token carries a logit up to "
            f"{ratio_max:.2f}x the max true causal logit — under eviction that "
            "logit error is the full logit)"
        )
    return dict(
        task="storm Task-2 leverage-score token coreset gate",
        model=cfg.model_label,
        n_sinks=cfg.n_sinks,
        gate_win_frac=GATE_WIN_FRAC,
        per_level=per_level,
        gate_outcome=(
            "CONFIRM (selection axis opens; Llama rides next rental)"
            if confirmed
            else "honest negative — coreset loses everywhere at matched bits"
        ),
        confirmed=bool(confirmed),
        mechanism=(mechanisms if mechanisms else ["none triggered"]),
        drop_mass_mean=dm_mean,
        drop_mass_max=dm_max,
        worst_ratio_max=ratio_max,
        git_sha=git_sha(),
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
