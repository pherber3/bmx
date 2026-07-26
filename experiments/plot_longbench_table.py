"""Publication Table-1 generator: k3_longbench metrics.parquet -> TurboQuant-format table.

Reads a k3_longbench run's metrics.parquet and never refits (repo plot-discipline rule,
CLAUDE.md): all scores here are exactly the per-item metric values k3_longbench already
computed. Runs are selected EXPLICITLY (mirrors experiments/plots/plot_k2.py's
`newest_run_with` pattern) — never a blind concat of a results root, which would double-count
reruns.

Aggregation-ambiguity note (why we report BOTH averages): TurboQuant's own published Average
column is NOT reproducible as the mean of its published per-category values. For "Full Cache"
the six published categories (45.29, 45.16, 26.55, 68.38, 59.54, 46.28) mean to 48.53, not the
published 50.06 — so TurboQuant's exact aggregation (presumably a per-task or per-sample-count
weighted mean over the raw LongBench task list, not a flat mean of the six category numbers) is
not recoverable from the paper alone. Rather than silently picking an aggregation and calling it
"the" average, this script reports both candidates for every row:
  - "Average (categories)": mean of the 6 category means (equal category weight).
  - "Average (all tasks)": mean over every task-level score actually present (equal task weight).
Because the anchor-delta gate compares PER-CATEGORY (not the Average column), this ambiguity
does not block the gate — it is surfaced for transparency only.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import tyro

from bmx.cache.longbench import CATEGORY2DATASETS

# TurboQuant (arXiv 2504.19874), Table 1, Llama-3.1-8B-Instruct rows. Columns:
# KV Size (bits), SingleQA, MultiQA, Summarization, Few shot, Synthetic, Code, Avg.
# (Ministral rows in the same table are out of scope here.)
PUBLISHED: dict[str, dict[str, float]] = {
    "Full Cache (published)": {
        "KV Size (bits)": 16,
        "SingleQA": 45.29,
        "MultiQA": 45.16,
        "Summarization": 26.55,
        "Few shot": 68.38,
        "Synthetic": 59.54,
        "Code": 46.28,
        "Avg": 50.06,
    },
    "KIVI (KV 3) (published)": {
        "KV Size (bits)": 3,
        "SingleQA": 43.38,
        "MultiQA": 37.99,
        "Summarization": 27.16,
        "Few shot": 68.38,
        "Synthetic": 59.50,
        "Code": 44.68,
        "Avg": 48.50,
    },
    "KIVI (KV 5) (published)": {
        "KV Size (bits)": 5,
        "SingleQA": 45.04,
        "MultiQA": 45.70,
        "Summarization": 26.47,
        "Few shot": 68.57,
        "Synthetic": 59.55,
        "Code": 46.41,
        "Avg": 50.16,
    },
    "PolarQuant (KV 3.9) (published)": {
        "KV Size (bits)": 3.9,
        "SingleQA": 45.18,
        "MultiQA": 44.48,
        "Summarization": 26.23,
        "Few shot": 68.25,
        "Synthetic": 60.07,
        "Code": 45.24,
        "Avg": 49.78,
    },
    "TurboQuant (2.5) (published)": {
        "KV Size (bits)": 2.5,
        "SingleQA": 44.16,
        "MultiQA": 44.96,
        "Summarization": 24.80,
        "Few shot": 68.01,
        "Synthetic": 59.65,
        "Code": 45.76,
        "Avg": 49.44,
    },
    "TurboQuant (3.5) (published)": {
        "KV Size (bits)": 3.5,
        "SingleQA": 45.01,
        "MultiQA": 45.31,
        "Summarization": 26.00,
        "Few shot": 68.63,
        "Synthetic": 59.95,
        "Code": 46.17,
        "Avg": 50.06,
    },
}

CATEGORY_LABELS: dict[str, str] = {
    "single_qa": "SingleQA",
    "multi_qa": "MultiQA",
    "summarization": "Summarization",
    "few_shot": "Few shot",
    "synthetic": "Synthetic",
    "code": "Code",
}
CATEGORY_ORDER = list(CATEGORY_LABELS.values())
COLUMNS = [
    "KV Size (bits)",
    *CATEGORY_ORDER,
    "Average (categories)",
    "Average (all tasks)",
]

# Anchor pairs for the delta-gate readout: our measured arm -> the published row it should
# track. Code is the gating category for the first rerun phase (per the task brief).
ANCHOR_PAIRS: dict[str, str] = {
    "fp16 (measured)": "Full Cache (published)",
    "turboquant_mse (measured)": "TurboQuant (2.5) (published)",
}

# Our former 'kivi' arm is a symmetric-RTN strawman, not real KIVI (docs/2026-07-04-kivi-arm-
# diagnosis.md) — exclude it from the table entirely rather than mislabel it against the
# published KIVI rows.
KIVI_DIAGNOSIS_DOC = "docs/2026-07-04-kivi-arm-diagnosis.md"
EXCLUDED_ARMS = {"kivi"}
# Arms that are honestly RTN-family (not the paper's schemes) get an explicit "rtn2" tag
# rather than implying parity with anything published.
RTN_STYLE_ARMS = {"rtn_channel", "rtn_token"}


def newest_run_with(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Rows matching mask from the NEWEST run that has any such rows (mirrors plot_k2.py)."""
    hits = df[mask]
    assert not hits.empty, "no run matches the selection"
    return hits[hits.run == hits.run.max()]


def load_runs(root: str) -> pd.DataFrame:
    """All k3_longbench runs under root, tagged with a `run` id column (never blind-concat)."""
    runs = sorted(Path(root).glob("*/metrics.parquet"))
    assert runs, f"no metrics.parquet under {root}"
    dfs = []
    for p in runs:
        df = pd.read_parquet(p)
        df["run"] = p.parent.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _task_to_category(task: str) -> str | None:
    for cat, tasks in CATEGORY2DATASETS.items():
        if task in tasks:
            return cat
    return None


def _method_label(arm: str) -> str:
    if arm in RTN_STYLE_ARMS:
        return f"{arm} (measured, rtn2)"
    return f"{arm} (measured)"


@dataclasses.dataclass
class TableResult:
    rows_by_method: dict[str, dict]
    footnotes: list[str]
    anchor_deltas: dict[str, dict[str, dict]]
    anchor_source: dict[str, str]
    overall_verdict: str
    caption: str
    parity_warning: str | None
    method_order: list[str]

    def _row_cells(self, method: str) -> list[str]:
        row = self.rows_by_method[method]
        cells = [method]
        for col in COLUMNS:
            v = row.get(col, "n/a")
            if isinstance(v, float):
                cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        return cells

    def to_markdown(self) -> str:
        header = ["Method", *COLUMNS]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
        for method in self.method_order:
            lines.append("| " + " | ".join(self._row_cells(method)) + " |")
        out = [self.caption, "", *lines, ""]
        if self.footnotes:
            out.append("Footnotes:")
            out.extend(f"- {f}" for f in self.footnotes)
            out.append("")
        out.append(
            "Anchor-delta gate (ours - published, |delta| <= tolerance => PASS):"
        )
        for method, cats in self.anchor_deltas.items():
            src = self.anchor_source[method]
            out.append(f"- {method} vs {src}:")
            for cat, v in cats.items():
                out.append(
                    f"  - {cat}: delta={v['delta']:+.2f} -> {v['verdict']}"
                    if v["delta"] is not None
                    else f"  - {cat}: n/a (missing category)"
                )
        out.append(f"Overall verdict: {self.overall_verdict}")
        return "\n".join(out)

    def to_latex(self) -> str:
        header = ["Method", *COLUMNS]
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\begin{tabular}{l" + "r" * len(COLUMNS) + "}",
            "\\toprule",
            " & ".join(header) + " \\\\",
            "\\midrule",
        ]
        for method in self.method_order:
            lines.append(" & ".join(self._row_cells(method)) + " \\\\")
        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{self.caption}}}",
            "\\end{table}",
        ]
        return "\n".join(lines)


def build_table(
    df: pd.DataFrame, run_id: str, anchor_tolerance: float = 2.0
) -> TableResult:
    """Turn a k3_longbench metrics frame into the publication Table-1 layout.

    `df` is one k3_longbench run's rows (arm, task, code_sim, kv_size_bits, ...). Category
    score = mean of that category's present task scores (asterisked + footnoted if a category's
    task list is only partially covered). fp16 is always KV Size 16 (uncompressed passthrough);
    other measured arms use the mean of the parquet's `kv_size_bits` column across tasks.
    """
    footnotes: list[str] = []

    if (df["arm"] == "kivi").any():
        footnotes.append(
            f"arm 'kivi' EXCLUDED: diagnosed as a symmetric-RTN strawman, not real KIVI "
            f"(see {KIVI_DIAGNOSIS_DOC})."
        )
    df = df[~df["arm"].isin(EXCLUDED_ARMS)].copy()

    df["category"] = df["task"].map(_task_to_category)
    unknown = df[df["category"].isna()]
    if not unknown.empty:
        footnotes.append(
            "tasks with no known category (excluded from category means): "
            + ", ".join(sorted(unknown["task"].unique()))
        )
    df = df[df["category"].notna()].copy()

    rows_by_method: dict[str, dict] = {}
    method_order: list[str] = []

    for arm, g in df.groupby("arm", sort=False):
        method = _method_label(arm)
        method_order.append(method)
        row: dict = {}
        row["KV Size (bits)"] = (
            16.0 if arm == "fp16" else float(g["kv_size_bits"].mean())
        )

        cat_means = {}
        for cat, label in CATEGORY_LABELS.items():
            expected = set(CATEGORY2DATASETS[cat])
            present = set(g.loc[g["category"] == cat, "task"].unique())
            if not present:
                row[label] = "n/a"
                continue
            mean_score = g.loc[g["category"] == cat, "code_sim"].mean() * 100.0
            missing = expected - present
            if missing:
                row[label] = f"{mean_score:.2f}*"
                footnotes.append(
                    f"{method} {label}: partial category, missing task(s) "
                    f"{', '.join(sorted(missing))} — mean is over {sorted(present)} only."
                )
            else:
                row[label] = mean_score
            cat_means[label] = mean_score

        row["Average (categories)"] = (
            sum(cat_means.values()) / len(cat_means) if cat_means else float("nan")
        )
        row["Average (all tasks)"] = float(g["code_sim"].mean() * 100.0)
        rows_by_method[method] = row

    # Published (transitive) block — clearly separated, appended after measured rows.
    for method, vals in PUBLISHED.items():
        method_order.append(method)
        row = {col: vals[col] for col in CATEGORY_ORDER}
        row["KV Size (bits)"] = vals["KV Size (bits)"]
        cat_avg = sum(vals[c] for c in CATEGORY_ORDER) / len(CATEGORY_ORDER)
        row["Average (categories)"] = cat_avg
        row["Average (all tasks)"] = vals[
            "Avg"
        ]  # only the paper's own Avg is available
        rows_by_method[method] = row
    footnotes.append(
        "published 'Average (all tasks)' is the paper's own reported Avg column (task-level "
        "scores are not published), NOT independently recomputed — see module docstring for "
        "why the category-mean and published Avg disagree (48.53 vs 50.06 for Full Cache)."
    )

    # --- Anchor-delta gate ---
    anchor_deltas: dict[str, dict[str, dict]] = {}
    anchor_source: dict[str, str] = {}
    any_fail = False
    any_compared = False
    for measured_method, published_method in ANCHOR_PAIRS.items():
        if measured_method not in rows_by_method:
            continue
        anchor_source[measured_method] = published_method
        cats: dict[str, dict] = {}
        for label in CATEGORY_ORDER:
            m_val = rows_by_method[measured_method].get(label, "n/a")
            if isinstance(m_val, str):
                # strip trailing '*' (partial-category marker) or treat 'n/a' as missing.
                if m_val == "n/a":
                    cats[label] = {"delta": None, "verdict": "N/A"}
                    continue
                m_val = float(m_val.rstrip("*"))
            p_val = PUBLISHED[published_method][label]
            delta = m_val - p_val
            verdict = "PASS" if abs(delta) <= anchor_tolerance else "FAIL"
            any_compared = True
            if verdict == "FAIL":
                any_fail = True
            cats[label] = {"delta": delta, "verdict": verdict}
        anchor_deltas[measured_method] = cats
    overall_verdict = "FAIL" if any_fail else ("PASS" if any_compared else "N/A")

    # --- Provenance / caption ---
    # Defensive: longbench_version/max_prompt_tokens are absent in pre-W3 parquets (older code
    # vintage predates these columns entirely). Report "unknown (pre-W3 parquet)" instead of
    # crashing with KeyError — this is a read-only provenance/caption concern, not a scoring
    # change. n_samples predates W3 too but was already present in every observed parquet;
    # guarded the same way for symmetry/future-proofing.
    UNKNOWN_PRE_W3 = "unknown (pre-W3 parquet)"
    if "longbench_version" in df.columns and len(df):
        versions = sorted(df["longbench_version"].dropna().unique())
    else:
        versions = []
    if "max_prompt_tokens" in df.columns and len(df):
        max_prompt_tokens_vals = sorted(df["max_prompt_tokens"].dropna().unique())
    else:
        max_prompt_tokens_vals = []
    if "n_samples" in df.columns and len(df):
        n_samples_vals = sorted(df["n_samples"].dropna().unique())
    else:
        n_samples_vals = []

    versions_display = versions if versions else UNKNOWN_PRE_W3
    max_prompt_tokens_display = (
        max_prompt_tokens_vals if max_prompt_tokens_vals else UNKNOWN_PRE_W3
    )
    n_samples_display = n_samples_vals if n_samples_vals else UNKNOWN_PRE_W3
    caption = (
        f"run_id={run_id}; longbench_version={versions_display}; "
        f"max_prompt_tokens={max_prompt_tokens_display}; n_samples={n_samples_display}"
    )

    parity_warning = None
    non_parity_version = any(v != "v1_e" for v in versions)
    sentinel_truncation = any(v == -1 for v in max_prompt_tokens_vals)
    if non_parity_version or sentinel_truncation:
        reasons = []
        if non_parity_version:
            reasons.append(f"longbench_version={versions} != 'v1_e'")
        if sentinel_truncation:
            reasons.append("max_prompt_tokens contains the -1 (no-truncation) sentinel")
        parity_warning = "NOT TurboQuant-input-parity: " + "; ".join(reasons)
        caption = f"*** {parity_warning} *** | {caption}"

    return TableResult(
        rows_by_method=rows_by_method,
        footnotes=footnotes,
        anchor_deltas=anchor_deltas,
        anchor_source=anchor_source,
        overall_verdict=overall_verdict,
        caption=caption,
        parity_warning=parity_warning,
        method_order=method_order,
    )


@dataclasses.dataclass
class Config:
    run_dir: str  # explicit path to a k3_longbench run dir (contains metrics.parquet)
    anchor_tolerance: float = 2.0
    out_prefix: str = "table1"  # writes <run_dir>/<out_prefix>.md and .tex


def main(cfg: Config) -> None:
    run_path = Path(cfg.run_dir)
    df = pd.read_parquet(run_path / "metrics.parquet")
    result = build_table(
        df, run_id=run_path.name, anchor_tolerance=cfg.anchor_tolerance
    )

    md = result.to_markdown()
    tex = result.to_latex()
    print(md)

    md_path = run_path / f"{cfg.out_prefix}.md"
    tex_path = run_path / f"{cfg.out_prefix}.tex"
    md_path.write_text(md)
    tex_path.write_text(tex)
    print(f"\n-> {md_path}")
    print(f"-> {tex_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
