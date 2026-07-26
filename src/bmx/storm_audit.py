"""Storm Task-1 audit core: load banked task/frontier parquets, split by model,
exclude the non-duel run, dedup by newest, and compute the pre-registered
Spearman gate + the Qwen per-event case study.

Kept out of the thin experiment script so the model-split, exclusion, dedup,
Spearman, and gate logic are unit-testable on tiny synthetic parquets (offline).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

# ---------------------------------------------------------------------------
# Pre-registered gate (plan-locked — DO NOT change the threshold).
GATE_THRESHOLD = 0.8

# LongBench category → English datasets (mirror of bmx.cache.longbench).
CATEGORY2DATASETS = {
    "single_qa": ["narrativeqa", "qasper", "multifieldqa_en"],
    "multi_qa": ["hotpotqa", "2wikimqa", "musique"],
    "summarization": ["gov_report", "qmsum", "multi_news"],
    "few_shot": ["trec", "triviaqa", "samsum"],
    "synthetic": ["passage_count", "passage_retrieval_en"],
    "code": ["lcc", "repobench-p"],
}
_DATASET2CATEGORY = {d: c for c, ds in CATEGORY2DATASETS.items() for d in ds}

# Task-arm → frontier K-codec (arm, key). `budget` for spectral, `bits` for tq.
# `turboquant_mse_b3` and `_k3v2` share the SAME K codec (tq@3b) → identical logit
# point; they are recorded as a tie on the logit axis (the audit's blind-spot finding).
ARM_TO_FRONTIER = {
    "k4_b2.5": ("spectral", "budget", 2.5),
    "k4_b2.5_dec8tl": (
        "spectral",
        "budget",
        2.5,
    ),  # dec8 changes V/decoder, not K basis
    "k4_b2.2": ("spectral", "budget", 2.2),
    "k4_b2.2_dec8tl": ("spectral", "budget", 2.2),
    "turboquant_mse_b3": ("turboquant_mse", "bits", 3),
    "turboquant_mse_k3v2": ("turboquant_mse", "bits", 3),
}

# The duel arm set (task arms carrying both sides). fp16 is the win/tie/loss reference,
# not a ranked codec.
DUEL_TASK_ARMS = [
    "k4_b2.5",
    "k4_b2.2",
    "turboquant_mse_b3",
    "turboquant_mse_k3v2",
]


def _canon_model(name: str) -> str:
    """Coarse model family key from a HF model_name in a run's config.json."""
    n = (name or "").lower()
    if "qwen" in n:
        return "Qwen"
    if "llama" in n:
        return "Llama"
    return name or "unknown"


# ---------------------------------------------------------------------------
# Loading + model-split + exclusion + dedup.


def _load_runs(
    root: str,
    exclude_run_substr: str,
    extra_files: tuple[str, ...] = (),
) -> tuple[list[dict], list[str]]:
    """Return (records, excluded_notes). Each record: dict with model, run_id, mtime,
    and one DataFrame per requested file name ('metrics' always; 'samples' if asked).

    A run is skipped (noted) if its run-id contains `exclude_run_substr`.
    """
    records: list[dict] = []
    excluded: list[str] = []
    rootp = Path(root)
    if not rootp.exists():
        return records, excluded
    for d in sorted(rootp.iterdir()):
        if not d.is_dir():
            continue
        cfgp, mp = d / "config.json", d / "metrics.parquet"
        if not cfgp.exists() or not mp.exists():
            continue
        run_id = d.name
        if exclude_run_substr and exclude_run_substr in run_id:
            excluded.append(f"{root}/{run_id} (matched '{exclude_run_substr}')")
            continue
        cfg = json.loads(cfgp.read_text())
        rec = {
            "model": _canon_model(cfg.get("model_name", "")),
            "model_name": cfg.get("model_name", ""),
            "run_id": run_id,
            # run-id timestamp prefix (YYYYmmdd-HHMMSS) sorts as newest-last.
            "stamp": run_id[:15],
            "metrics": pd.read_parquet(mp),
        }
        for f in extra_files:
            fp = d / f"{f}.parquet"
            rec[f] = pd.read_parquet(fp) if fp.exists() else None
        records.append(rec)
    return records, excluded


def _dedup_newest(df: pd.DataFrame, cell_cols: list[str]) -> tuple[pd.DataFrame, int]:
    """Keep the newest run per (cell_cols) cell. `stamp` marks recency; ties broken by
    run_id. Returns (deduped, n_dropped)."""
    if df.empty:
        return df, 0
    ordered = df.sort_values(["stamp", "run_id"])
    kept = ordered.drop_duplicates(subset=cell_cols, keep="last")
    return kept.reset_index(drop=True), len(df) - len(kept)


# ---------------------------------------------------------------------------
# NIAH task-side ranking (mean recall per arm×length; NO per-sample rows).


def load_niah(root: str, exclude_run_substr: str) -> tuple[pd.DataFrame, list[str]]:
    recs, excluded = _load_runs(root, exclude_run_substr)
    frames = []
    for r in recs:
        m = r["metrics"].copy()
        m["model"] = r["model"]
        m["run_id"] = r["run_id"]
        m["stamp"] = r["stamp"]
        frames.append(m)
    if not frames:
        return pd.DataFrame(), excluded
    df = pd.concat(frames, ignore_index=True)
    # A cell is (model, arm, length, depth, seed?) — dedup by newest run.
    seed_col = ["seed"] if "seed" in df.columns else []
    cell = ["model", "arm", "length", "depth"] + seed_col
    df, n_drop = _dedup_newest(df, cell)
    notes = list(excluded)
    if n_drop:
        notes.append(f"NIAH: deduped {n_drop} repeated (model,arm,length,depth) cells")
    return df, notes


# ---------------------------------------------------------------------------
# LongBench task-side (macro + per-category from metrics; per-event from samples).


def load_longbench(
    root: str, exclude_run_substr: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    recs, excluded = _load_runs(root, exclude_run_substr, extra_files=("samples",))
    mframes, sframes = [], []
    for r in recs:
        m = r["metrics"].copy()
        m["model"] = r["model"]
        m["run_id"] = r["run_id"]
        m["stamp"] = r["stamp"]
        # `code_sim` is the score column for code tasks; other tasks store the same
        # aggregate under `code_sim` too (the k3_longbench schema uses one score col).
        mframes.append(m)
        if r["samples"] is not None:
            s = r["samples"].copy()
            s["model"] = r["model"]
            s["run_id"] = r["run_id"]
            s["stamp"] = r["stamp"]
            sframes.append(s)
    metrics = pd.concat(mframes, ignore_index=True) if mframes else pd.DataFrame()
    samples = pd.concat(sframes, ignore_index=True) if sframes else pd.DataFrame()
    notes = list(excluded)
    if not metrics.empty:
        metrics, n_drop = _dedup_newest(metrics, ["model", "arm", "task"])
        if n_drop:
            notes.append(f"LongBench metrics: deduped {n_drop} (model,arm,task) cells")
    if not samples.empty:
        # A sample cell is (model, arm, task, sample_idx) — newest run wins.
        samples, n_drop = _dedup_newest(samples, ["model", "arm", "task", "sample_idx"])
        if n_drop:
            notes.append(
                f"LongBench samples: deduped {n_drop} (model,arm,task,sample_idx) cells"
            )
    return metrics, samples, notes


# ---------------------------------------------------------------------------
# Frontier logit-instrument ranking.


def load_frontier(
    root: str, exclude_run_substr: str, spectral_fit_mode: str
) -> tuple[pd.DataFrame, list[str]]:
    """Per-layer K-tensor logit distortion, pooled + deduped by newest run.

    Returns rows keyed by (model, arm, budget|bits, kind, fit_mode) with per-layer
    `logit`/`logit_rope`. Model here is the frontier's own model string (e.g.
    'llama-3.1-8b-instruct'); mapped to a family later.
    """
    recs, excluded = _load_runs(root, exclude_run_substr)
    frames = []
    for r in recs:
        m = r["metrics"].copy()
        m["run_id"] = r["run_id"]
        m["stamp"] = r["stamp"]
        frames.append(m)
    if not frames:
        return pd.DataFrame(), excluded
    df = pd.concat(frames, ignore_index=True)
    # Family key from the frontier model string.
    df["family"] = df["model"].map(_canon_model)
    # Dedup by newest per (model, arm, budget, bits, kind, fit_mode, weighted, layer).
    df, n_drop = _dedup_newest(
        df,
        ["model", "arm", "budget", "bits", "kind", "fit_mode", "weighted", "layer"],
    )
    notes = list(excluded)
    if n_drop:
        notes.append(f"frontier: deduped {n_drop} per-layer cells")
    return df, notes


def frontier_scalar(
    frontier: pd.DataFrame,
    model: str,
    arm: str,
    key: str,
    value,
    *,
    spectral_fit_mode: str,
    kind: str = "k_pre",
    logit_col: str = "logit_rope",
    reducer: str = "mean",
) -> float | None:
    """Reduce the frontier's per-layer distortion to one scalar for a codec.

    `model` is the frontier's own model string (e.g. 'llama-3.1-8b-instruct') — matched
    exactly so the ladder-nc replicate rows never pool into the ranking scalar.

    `reducer` ∈ {'mean', 'p90', 'p95', 'p99', 'max'} — mean is the ranking statistic;
    the tail quantiles feed the Qwen 'would a tail statistic have predicted it' probe.
    Layer 0 (near-singular pathology, per the rental doc) is excluded from summaries.
    """
    sub = frontier[
        (frontier["model"] == model)
        & (frontier["arm"] == arm)
        & (frontier["kind"] == kind)
    ]
    if arm == "spectral":
        sub = sub[(sub["budget"] == value) & (sub["fit_mode"] == spectral_fit_mode)]
    else:  # turboquant_mse: keyed by integer bits, baseline fit
        sub = sub[sub["bits"] == value]
    sub = sub[sub["layer"] != 0]
    if sub.empty:
        return None
    vals = sub[logit_col].to_numpy(dtype=float)
    if reducer == "mean":
        return float(np.mean(vals))
    if reducer == "max":
        return float(np.max(vals))
    if reducer.startswith("p"):
        q = int(reducer[1:]) / 100.0
        return float(np.quantile(vals, q))
    raise ValueError(f"unknown reducer {reducer!r}")


# ---------------------------------------------------------------------------
# Spearman between the logit ranking and each task ranking.


def _spearman(logit_scores: dict, task_scores: dict, task_higher_is_better: bool):
    """Spearman between the logit-distortion ranking and the task ranking over the
    arms present in BOTH dicts.

    Logit distortion: lower = better. Task metric: `task_higher_is_better` sets the
    sense. We correlate distortion vs task with the task NEGATED when higher-is-better,
    so a POSITIVE rho means 'the instrument agrees' (a good codec on the instrument is a
    good codec on the task). A negative rho is a SIGN INVERSION.
    """
    arms = [a for a in logit_scores if a in task_scores]
    arms = [
        a
        for a in arms
        if logit_scores[a] is not None
        and task_scores[a] is not None
        and not (isinstance(task_scores[a], float) and np.isnan(task_scores[a]))
    ]
    if len(arms) < 2:
        return None, len(arms), arms
    x = np.array([logit_scores[a] for a in arms], dtype=float)  # distortion
    y = np.array([task_scores[a] for a in arms], dtype=float)  # task metric
    y_aligned = (
        -y if task_higher_is_better else y
    )  # -> lower is better, like distortion
    if np.allclose(x, x[0]) or np.allclose(y_aligned, y_aligned[0]):
        # No variance on one axis (e.g. b3/k3v2 tie on the K-logit) — rho undefined.
        return None, len(arms), arms
    rho, _ = spearmanr(x, y_aligned)
    # spearmanr on distortion-vs-(-task): positive rho = agreement.
    return float(rho), len(arms), arms


# ---------------------------------------------------------------------------
# Per-event (LongBench): win/tie/loss vs fp16, per sample, per arm.


def per_event_winrate(
    samples: pd.DataFrame, model: str, task: str, tie_eps: float = 1e-9
) -> pd.DataFrame | None:
    """Per-arm win/tie/loss vs fp16 across per-sample scores for one (model, task).

    A sample is a WIN if arm_score > fp16_score + eps, LOSS if < fp16_score - eps, TIE
    otherwise. Returns per-arm win/tie/loss counts + win_rate + mean_score, or None if
    fp16 or the arm samples are absent.
    """
    if samples.empty:
        return None
    s = samples[(samples["model"] == model) & (samples["task"] == task)]
    fp = s[s["arm"] == "fp16"][["sample_idx", "score"]].rename(
        columns={"score": "fp16"}
    )
    if fp.empty:
        return None
    rows = []
    for arm in sorted(s["arm"].unique()):
        if arm == "fp16":
            continue
        a = s[s["arm"] == arm][["sample_idx", "score"]].merge(fp, on="sample_idx")
        if a.empty:
            continue
        d = a["score"] - a["fp16"]
        wins = int((d > tie_eps).sum())
        losses = int((d < -tie_eps).sum())
        ties = int(len(a) - wins - losses)
        # total-failure events: arm scored ~0 while fp16 got meaningful credit.
        total_fail = int(((a["score"] <= tie_eps) & (a["fp16"] > 0.1)).sum())
        rows.append(
            {
                "model": model,
                "task": task,
                "arm": arm,
                "n": len(a),
                "win": wins,
                "tie": ties,
                "loss": losses,
                "win_rate": wins / len(a),
                "loss_rate": losses / len(a),
                "total_fail_rate": total_fail / len(a),
                "mean_score": float(a["score"].mean()),
                "fp16_mean": float(a["fp16"].mean()),
            }
        )
    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------
# The Qwen TQ-collapse case study.


def qwen_case_study(
    niah: pd.DataFrame,
    lb_samples: pd.DataFrame,
    frontier: pd.DataFrame,
    *,
    spectral_fit_mode: str,
) -> dict:
    """Where does the Qwen TQ collapse live (uniform vs event-concentrated), and would a
    distortion tail statistic have separated Qwen-TQ from Llama-TQ where the mean did
    not?
    """
    notes: list[str] = []
    out: dict = {"notes": notes}

    # (1) NIAH: the collapse profile per arm/length (recall_full, 0-10 scale).
    if not niah.empty:
        q = niah[niah["model"] == "Qwen"]
        collapse_arms = ["turboquant_mse_b3", "turboquant_mse_k3v2", "k4_b2.5"]
        prof = (
            q[q["arm"].isin(collapse_arms)]
            .groupby(["arm", "length"])["recall_full"]
            .mean()
            .round(2)
        )
        out["niah_collapse_profile"] = {
            f"{a}@{int(length)}": float(v) for (a, length), v in prof.items()
        }
        # onset: lengths where tq falls below half of k4.
        for a in ("turboquant_mse_b3", "turboquant_mse_k3v2"):
            fell = []
            for length in sorted(q["length"].unique()):
                ka = q[(q["arm"] == "k4_b2.5") & (q["length"] == length)][
                    "recall_full"
                ].mean()
                ta = q[(q["arm"] == a) & (q["length"] == length)]["recall_full"].mean()
                if pd.notna(ka) and pd.notna(ta) and ka > 0 and ta < 0.5 * ka:
                    fell.append(int(length))
            if fell:
                notes.append(f"Qwen NIAH {a}: <50% of k4_b2.5 at lengths {fell}")

    # (2) Per-event: is the collapse uniform degradation or a mass of total-failure
    # events? Use LongBench samples on Qwen where the full duel set + fp16 exist.
    if not lb_samples.empty:
        qs = lb_samples[lb_samples["model"] == "Qwen"]
        pe_rows = []
        for task in sorted(qs["task"].unique()):
            pe = per_event_winrate(qs, "Qwen", task)
            if pe is not None:
                pe_rows.append(pe)
        if pe_rows:
            pe_all = pd.concat(pe_rows, ignore_index=True)
            out["qwen_per_event"] = pe_all.to_dict("records")
            for _, r in pe_all.iterrows():
                if r["arm"] in ("turboquant_mse_b3", "turboquant_mse_k3v2"):
                    notes.append(
                        f"Qwen LB {r['task']} {r['arm']}: loss_rate={r['loss_rate']:.2f} "
                        f"total_fail_rate={r['total_fail_rate']:.2f} "
                        f"(mean {r['mean_score']:.2f} vs fp16 {r['fp16_mean']:.2f})"
                    )

    # (3) Distortion tail: does a tail quantile of the frontier K-logit separate Qwen-TQ
    # from Llama-TQ where the mean did not? Compare tq@3b K_pre across the two families.
    tail = {}
    for fam, fam_key in (("Qwen", "qwen3-8b"), ("Llama", "llama-3.1-8b-instruct")):
        row = {}
        for red in ("mean", "p90", "p95", "p99", "max"):
            row[red] = frontier_scalar(
                frontier,
                fam_key,
                "turboquant_mse",
                "bits",
                3,
                spectral_fit_mode=spectral_fit_mode,
                reducer=red,
            )
        # spectral b2.5 for contrast (the arm that HOLDS on Qwen).
        for red in ("mean", "p99"):
            row[f"spectral_b2.5_{red}"] = frontier_scalar(
                frontier,
                fam_key,
                "spectral",
                "budget",
                2.5,
                spectral_fit_mode=spectral_fit_mode,
                reducer=red,
            )
        tail[fam] = row
    out["distortion_tail"] = tail
    # Does any tail statistic separate Qwen-TQ from Llama-TQ where the mean is close?
    qm, lm = tail.get("Qwen", {}), tail.get("Llama", {})
    if qm.get("mean") is not None and lm.get("mean") is not None:
        mean_ratio = qm["mean"] / lm["mean"] if lm["mean"] else float("nan")
        notes.append(
            f"tq@3b K_pre mean-logit: Qwen {qm['mean']:.4f} vs Llama {lm['mean']:.4f} "
            f"(ratio {mean_ratio:.2f})"
        )
        for red in ("p99", "max"):
            if qm.get(red) is not None and lm.get(red) is not None and lm[red]:
                notes.append(
                    f"tq@3b K_pre {red}-logit: Qwen {qm[red]:.4f} vs Llama "
                    f"{lm[red]:.4f} (ratio {qm[red] / lm[red]:.2f})"
                )
    return out


# ---------------------------------------------------------------------------
# The top-level audit.


def run_audit(
    *,
    niah_root: str,
    longbench_root: str,
    frontier_root: str,
    exclude_run_substr: str,
    spectral_fit_mode: str,
    gate_threshold: float,
) -> tuple[dict, pd.DataFrame]:
    """Run the full audit. Returns (verdict_dict, metrics_dataframe)."""
    niah, niah_notes = load_niah(niah_root, exclude_run_substr)
    lb_metrics, lb_samples, lb_notes = load_longbench(
        longbench_root, exclude_run_substr
    )
    frontier, fr_notes = load_frontier(
        frontier_root, exclude_run_substr, spectral_fit_mode
    )

    data_notes: list[str] = []
    # Honest per-sample granularity statement.
    if niah.empty:
        data_notes.append("NIAH: no runs found.")
    else:
        data_notes.append(
            "NIAH parquets carry NO per-sample rows (only mean recall per "
            "arm/length/depth) — the per-event 0/1 view is LongBench-only; NIAH "
            "contributes rank correlation on mean recall."
        )
    if lb_samples.empty:
        data_notes.append("LongBench: NO per-sample (samples.parquet) rows found.")
    else:
        by_model = lb_samples.groupby("model")["arm"].agg(lambda s: sorted(set(s)))
        for model, arms in by_model.items():
            tasks = sorted(lb_samples[lb_samples["model"] == model]["task"].unique())
            data_notes.append(
                f"LongBench per-sample present: {model} arms={arms} tasks={tasks}"
            )

    provenance = list(niah_notes) + list(lb_notes) + list(fr_notes)

    # --- Spearman rows -----------------------------------------------------
    spearman_rows: list[dict] = []
    metrics_records: list[dict] = []

    # Build the logit scalar per task-arm per family (mean over layers, spectral_fit_mode).
    def logit_scalars(family_key: str) -> dict:
        d = {}
        for arm, (fr_arm, key, val) in ARM_TO_FRONTIER.items():
            if arm not in DUEL_TASK_ARMS:
                continue
            d[arm] = frontier_scalar(
                frontier,
                family_key,
                fr_arm,
                key,
                val,
                spectral_fit_mode=spectral_fit_mode,
            )
        return d

    fam_keys = {"Llama": "llama-3.1-8b-instruct", "Qwen": "qwen3-8b"}

    # NIAH axis: per (model, length), rank arms by mean recall_full.
    if not niah.empty:
        for model, fam_key in fam_keys.items():
            lscores = logit_scalars(fam_key)
            q = niah[niah["model"] == model]
            for length in sorted(q["length"].unique()):
                tscores = {}
                for arm in DUEL_TASK_ARMS:
                    v = q[(q["arm"] == arm) & (q["length"] == length)][
                        "recall_full"
                    ].mean()
                    tscores[arm] = float(v) if pd.notna(v) else np.nan
                rho, n, arms_used = _spearman(
                    lscores, tscores, task_higher_is_better=True
                )
                spearman_rows.append(
                    _mk_spearman_row(
                        model,
                        f"NIAH recall@{int(length)}",
                        rho,
                        n,
                        gate_threshold,
                        arms_used,
                    )
                )
                for arm in arms_used:
                    metrics_records.append(
                        {
                            "model": model,
                            "task_axis": f"NIAH recall@{int(length)}",
                            "arm": arm,
                            "logit_distortion": lscores.get(arm),
                            "task_metric": tscores.get(arm),
                            "metric_kind": "niah_recall_full",
                        }
                    )

    # LongBench axes: macro + per-category, per model.
    if not lb_metrics.empty:
        for model, fam_key in fam_keys.items():
            lscores = logit_scalars(fam_key)
            mm = lb_metrics[lb_metrics["model"] == model]
            if mm.empty:
                continue
            score_col = "code_sim" if "code_sim" in mm.columns else "score"
            # per-category means, then macro = mean of category means.
            mm = mm.copy()
            mm["category"] = mm["task"].map(_DATASET2CATEGORY)
            # per-category axis
            cat_axes = {}
            for cat in sorted(mm["category"].dropna().unique()):
                cat_axes[f"LB {cat}"] = (
                    mm[mm["category"] == cat].groupby("arm")[score_col].mean().to_dict()
                )
            # macro axis: mean over category means per arm.
            cat_means = (
                mm.dropna(subset=["category"])
                .groupby(["arm", "category"])[score_col]
                .mean()
                .reset_index()
            )
            macro = cat_means.groupby("arm")[score_col].mean().to_dict()
            cat_axes["LB macro"] = macro
            for axis, arm_map in cat_axes.items():
                tscores = {a: arm_map.get(a, np.nan) for a in DUEL_TASK_ARMS}
                rho, n, arms_used = _spearman(
                    lscores, tscores, task_higher_is_better=True
                )
                spearman_rows.append(
                    _mk_spearman_row(model, axis, rho, n, gate_threshold, arms_used)
                )
                for arm in arms_used:
                    metrics_records.append(
                        {
                            "model": model,
                            "task_axis": axis,
                            "arm": arm,
                            "logit_distortion": lscores.get(arm),
                            "task_metric": tscores.get(arm),
                            "metric_kind": "longbench_score",
                        }
                    )

    # --- Per-event ranking-agreement (LongBench) ---------------------------
    per_event_rows: list[dict] = []
    if not lb_samples.empty:
        for model in sorted(lb_samples["model"].unique()):
            ms = lb_samples[lb_samples["model"] == model]
            for task in sorted(ms["task"].unique()):
                pe = per_event_winrate(ms, model, task)
                if pe is None or len(pe) < 2:
                    continue
                # does per-event win-rate ranking agree with mean-score ranking?
                pe_sorted = pe.sort_values("arm")
                tau, _ = kendalltau(pe_sorted["win_rate"], pe_sorted["mean_score"])
                per_event_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "kendall_tau": None if pd.isna(tau) else round(float(tau), 3),
                        "rank_agrees_with_mean": bool(pd.notna(tau) and tau > 0),
                        "arms": sorted(pe["arm"].tolist()),
                    }
                )
                for _, r in pe.iterrows():
                    metrics_records.append(
                        {
                            "model": model,
                            "task_axis": f"LB per-event {task}",
                            "arm": r["arm"],
                            "logit_distortion": None,
                            "task_metric": r["win_rate"],
                            "metric_kind": "per_event_win_rate",
                            "win": r["win"],
                            "tie": r["tie"],
                            "loss": r["loss"],
                            "total_fail_rate": r["total_fail_rate"],
                            "mean_score": r["mean_score"],
                        }
                    )

    # --- Qwen case study ---------------------------------------------------
    qwen = qwen_case_study(
        niah, lb_samples, frontier, spectral_fit_mode=spectral_fit_mode
    )

    # --- Gate (pre-registered, evaluated EXACTLY as the plan states) -------
    gate = _evaluate_gate(spearman_rows, gate_threshold)

    verdict = {
        "gate_threshold": gate_threshold,
        "spectral_fit_mode": spectral_fit_mode,
        "data_availability": data_notes,
        "provenance": provenance,
        "spearman": spearman_rows,
        "per_event": per_event_rows,
        "qwen_case_study": qwen,
        "gate": gate,
    }
    metrics_df = pd.DataFrame(metrics_records)
    return verdict, metrics_df


def _mk_spearman_row(model, axis, rho, n, threshold, arms_used):
    below = rho is not None and rho < threshold
    inversion = rho is not None and rho < 0
    return {
        "model": model,
        "task_axis": axis,
        "rho": None if rho is None else round(rho, 4),
        "n_pairs": n,
        "arms": arms_used,
        "below_threshold": bool(below),
        "sign_inversion": bool(inversion),
    }


def _evaluate_gate(spearman_rows: list[dict], threshold: float) -> dict:
    """Gate verbatim: Spearman >= threshold on Llama across the duel arms ⇒ SAFE PROXY.
    Spearman < threshold ANYWHERE, or a sign inversion on any arm pair ⇒ FLAG.
    """
    # Rows with a defined rho (>= 2 varying pairs).
    defined = [r for r in spearman_rows if r["rho"] is not None]
    llama_defined = [r for r in defined if r["model"] == "Llama"]
    any_below = any(r["below_threshold"] for r in defined)
    any_inversion = any(r["sign_inversion"] for r in defined)

    # The instrument's structural blind-spot: b3/k3v2 collapse to one logit point, so no
    # Llama axis has a defined rho spanning a tq-internal contrast. Record it.
    llama_has_defined = len(llama_defined) > 0
    llama_all_pass = llama_has_defined and all(
        not r["below_threshold"] and not r["sign_inversion"] for r in llama_defined
    )

    if any_below or any_inversion:
        outcome = "FLAG"
        offenders = [
            f"{r['model']}/{r['task_axis']} rho={r['rho']}"
            for r in defined
            if r["below_threshold"] or r["sign_inversion"]
        ]
        reason = (
            "Spearman < threshold or a sign inversion occurred: "
            + "; ".join(offenders)
            + ". The paper's quality tables must carry the per-event column alongside "
            "macro."
        )
    elif llama_all_pass:
        outcome = "SAFE_PROXY"
        reason = (
            f"All defined Llama duel-arm axes clear Spearman >= {threshold} with no "
            "sign inversion — the logit instrument is a safe proxy; keep the shipped "
            "metric."
        )
    else:
        # No Llama axis produced a defined rho (the b3/k3v2 logit tie + duel-arm NIAH
        # parity leave every Llama axis rank-degenerate) — the gate cannot be cleared as
        # a proxy; FLAG conservatively and record why.
        outcome = "FLAG"
        reason = (
            "No Llama duel-arm axis yields a defined Spearman rho: the two TQ arms "
            "(b3, k3v2) share one K-logit point (V-budget invisible to the K "
            "instrument) and the spectral arms sit far from them, so within-family "
            "rank has no variance while cross-family rank has only one effective "
            "contrast. The instrument cannot be certified a safe proxy across the duel "
            "arms — FLAG: carry the per-event column."
        )
    return {
        "outcome": outcome,
        "reason": reason,
        "any_below_threshold": any_below,
        "any_sign_inversion": any_inversion,
        "n_defined_axes": len(defined),
        "n_llama_defined_axes": len(llama_defined),
    }
