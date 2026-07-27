"""Storm Task-8 — attention-predictability measurement, the Tier-3 prefetch
gatekeeper (plan `docs/superpowers/plans/2026-07-26-storm-gates.md` Task 8).

Question: is the top-k attention read set one-step predictable? This is the
life-or-death number for the briefing's two-tier mechanism (cold-2bit-HBM +
exact-host speculative prefetch,
docs/2026-07-26-storm-kv-mechanisms-briefing.md Tier 3): a prefetcher that
cannot predict step t+1's heavy reads from step t's observed attention drowns
in prefetch pollution before any engineering starts.

Substrate: gpt2 + qwen3-0.6b, real wikitext continuation (load_eval_tokens,
token_offset=0), TEACHER-FORCED decode: prefill a prompt, then feed the TRUE
next token one step at a time with output_attentions=True (eager attention,
fp32, CPU), collecting each step's per-head attention row over the growing
cache. qwen3-0.6b: prompt_len=1024, decode_steps=256. gpt2's absolute context
ceiling is n_positions=1024, so its prompt clamps to 1024−256=768 (recorded
in the verdict — a >=1k prompt plus a decode window is physically unreachable
inside gpt2's positional budget).

PRE-REGISTERED (fixed BEFORE running; the gate is binding — plan verbatim:
"one-step-ahead predictor recall@k >= 0.9 for tokens carrying > X% attention
mass => the two-tier mechanism graduates to a spec. < 0.9 => killed by
prefetch-pollution before any engineering."):

- X = 1.0 (percent, per-row mass): a realized read event at step t+1 is a
  token carrying > 1% of that head's (or head-pooled row's) attention mass —
  the per-head-mass convention (a token below 1% of one head's read is
  negligible to that head's output).
- k = ceil(0.05 * S_{t+1}): 5% of the current cache length — the LOW end of
  the plausible 5–10%-of-S prefetch budget (smaller k is the conservative
  choice: it can only lower recall).
- Predictor family (cheap, deployment-plausible, O(k) bookkeeping per step;
  NO learned predictors):
    identity          — top-k of a_t (step t's observed attention row);
    identity_recency  — the n_recent=16 newest cache positions at t+1
                        (always including the token appended between t and
                        t+1) UNION top-(k−16) of a_t over the remaining
                        positions — total budget exactly k;
    ema               — top-k of ema_t = (1−λ)·ema_{t−1} + λ·a_t (λ = 0.5,
                        ema_0 = a_0; new positions enter with zero history);
    ema_recency       — identity_recency with ema_t in place of a_t.
- GATE QUANTITY: per model, the MAX over the four pre-registered predictors
  of MICRO-pooled recall@k over ALL (layer, head, step) realized events at
  scope="head" (the mechanism ships its best cheap predictor); the gate
  passes only if EVERY model clears >= 0.9 (threshold plan-locked verbatim).
- Grid (anti-brittleness, reported NOT gated): X in {0.5, 1.0, 2.0} percent
  x k in {0.025, 0.05, 0.10}·S — the gate cell is evaluated verbatim at
  (X=1.0, k=0.05·S); the 3x3 grid shows whether the verdict is
  threshold-brittle.

Metrics (per model, scope="head" per (layer, head) and scope="layer_pooled"
= head-mean row per layer, the per-layer KV-fetch-granularity companion):
recall@k (micro: total hits / total realized events); mass coverage —
fraction of step t+1's realized attention mass carried by the predicted set
(the number that actually governs prefetch quality; the > X% threshold does
NOT enter it); stability of the per-step pooled recall series (mean/std/min
+ fraction of steps >= 0.9); and a sink/recent/content decomposition — every
realized event and every unit of read mass is classed sink (positions 0–3,
the program's sink convention) / recent (the 16 newest positions — the same
window the recency predictors get for free) / content (everything else),
with per-class recall + mass coverage, so the verdict distinguishes
"predictable because static (sinks + recency)" from "predictable content
reads".

Mechanism scale, CPU-only, no VM, no web. Teacher-forced (the plan's
harness); free-running drift is out of scope and recorded as a caveat.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics

PREDICTORS = ("identity", "identity_recency", "ema", "ema_recency")
CLASSES = ("sink", "recent", "content")
GATE_THRESHOLD = 0.9  # plan-locked verbatim


@dataclasses.dataclass
class Config:
    models: tuple[str, ...] = ("gpt2", "Qwen/Qwen3-0.6B")
    labels: tuple[str, ...] = ("gpt2", "qwen3-0.6b")
    prompt_len: int = 1024  # clamps to max_positions - decode_steps (gpt2: 768)
    decode_steps: int = 256
    token_offset: int = 0
    # ---- pre-registered gate cell + anti-brittleness grid ------------------
    x_pct_grid: tuple[float, ...] = (0.5, 1.0, 2.0)
    k_frac_grid: tuple[float, ...] = (0.025, 0.05, 0.10)
    gate_x_pct: float = 1.0  # pre-registered: realized event = > 1% row mass
    gate_k_frac: float = 0.05  # pre-registered: k = ceil(0.05 * S_{t+1})
    # ---- pre-registered predictor/decomposition conventions ----------------
    n_recent: int = 16  # recency window (predictor budget AND "recent" class)
    n_sink: int = 4  # sink class = positions 0..3 (program convention)
    ema_lambda: float = 0.5
    seed: int = 0
    out_root: str = ""


# ---------------------------------------------------------------------------
# Pure core (pinned by tests/test_storm_attn_predict.py)
# ---------------------------------------------------------------------------


def class_masks(S_next: int, n_sink: int, n_recent: int) -> dict[str, torch.Tensor]:
    """Partition positions [0, S_next) into sink / recent / content boolean
    masks. Fails loudly if sink and recent would overlap (degenerate S)."""
    assert n_sink + n_recent <= S_next, (
        f"sink({n_sink}) + recent({n_recent}) overlap at S={S_next}"
    )
    idx = torch.arange(S_next)
    sink = idx < n_sink
    recent = idx >= S_next - n_recent
    return {"sink": sink, "recent": recent, "content": ~(sink | recent)}


def predictor_sets(
    a_prev: torch.Tensor, ema_prev: torch.Tensor, k: int, n_recent: int
) -> dict[str, torch.Tensor]:
    """The four pre-registered predicted sets for step t+1, from step t's
    observed rows. a_prev/ema_prev: (R, S_prev) rows over the cache at step t;
    the cache at t+1 has S_next = S_prev + 1 positions (one new token).
    Returns {name: (R, k) int64 indices into [0, S_next)} — every predictor
    spends exactly the same budget k. Recency variants take the n_recent
    newest positions at t+1 (indices S_next-n_recent .. S_next-1, including
    the brand-new position S_prev) plus top-(k - n_recent) of the row over
    the remaining (non-recent) positions."""
    R, S_prev = a_prev.shape
    assert ema_prev.shape == a_prev.shape
    assert 0 < n_recent < k <= S_prev, f"need 0 < {n_recent=} < {k=} <= {S_prev=}"
    S_next = S_prev + 1
    ridx = torch.arange(S_next - n_recent, S_next)

    def with_recency(base: torch.Tensor) -> torch.Tensor:
        masked = base.clone()
        # Exclude positions already covered by the recency window (the new
        # position S_prev is not in the row at all).
        masked[:, S_next - n_recent :] = float("-inf")
        top = masked.topk(k - n_recent, dim=-1).indices
        return torch.cat([top, ridx.expand(R, -1)], dim=-1)

    return {
        "identity": a_prev.topk(k, dim=-1).indices,
        "identity_recency": with_recency(a_prev),
        "ema": ema_prev.topk(k, dim=-1).indices,
        "ema_recency": with_recency(ema_prev),
    }


def set_mask(pred_idx: torch.Tensor, S_next: int) -> torch.Tensor:
    """(R, k) predicted indices -> (R, S_next) boolean membership mask."""
    R = pred_idx.shape[0]
    assert int(pred_idx.max()) < S_next
    m = torch.zeros(R, S_next, dtype=torch.bool)
    m.scatter_(1, pred_idx, True)
    return m


def score_sets(
    pred_mask: torch.Tensor,
    a_next: torch.Tensor,
    x_frac: float,
    masks: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Score one step: realized events = tokens with a_next > x_frac (row
    mass fraction); recall counts hits inside the predicted set; mass_cov =
    attention mass at t+1 covered by the predicted set (threshold-free — the
    prefetch-quality number). Per-class variants for sink/recent/content.
    All outputs are (R,) tensors."""
    assert pred_mask.shape == a_next.shape
    realized = a_next > x_frac
    hit = pred_mask & realized
    out = {
        "n_realized": realized.sum(-1),
        "n_hit": hit.sum(-1),
        "mass_cov": (a_next * pred_mask).sum(-1),
    }
    for cls in CLASSES:
        m = masks[cls]
        out[f"n_realized_{cls}"] = (realized & m).sum(-1)
        out[f"n_hit_{cls}"] = (hit & m).sum(-1)
        out[f"mass_{cls}"] = (a_next * m).sum(-1)
        out[f"mass_cov_{cls}"] = (a_next * (pred_mask & m)).sum(-1)
    return out


def set_mask_batch(idx: torch.Tensor, S_next: int) -> torch.Tensor:
    """(P, R, k) predicted indices -> (P, R, S_next) boolean membership masks
    (one scatter for the whole predictor stack)."""
    P, R, k = idx.shape
    assert int(idx.max()) < S_next
    m = torch.zeros(P * R, S_next, dtype=torch.bool)
    m.scatter_(1, idx.reshape(P * R, k), True)
    return m.view(P, R, S_next)


def score_sets_batch(
    pred_masks: torch.Tensor,
    a_next: torch.Tensor,
    x_fracs: list[float],
    masks: dict[str, torch.Tensor],
) -> dict[float, dict[str, torch.Tensor]]:
    """Batched score_sets over a stacked predictor axis and the X grid —
    numerically identical to the reference `score_sets` (parity pinned by
    tests). pred_masks: (P, R, S); returns {x_frac: {field: (P, R)}}.
    Mass fields are X-independent and shared across the returned cells."""
    P = pred_masks.shape[0]
    assert pred_masks.shape[1:] == a_next.shape
    a = a_next.unsqueeze(0)
    base = {"mass_cov": (a * pred_masks).sum(-1)}
    for cls in CLASSES:
        m = masks[cls]
        base[f"mass_{cls}"] = (a_next * m).sum(-1).unsqueeze(0).expand(P, -1)
        base[f"mass_cov_{cls}"] = (a * (pred_masks & m)).sum(-1)
    out: dict[float, dict[str, torch.Tensor]] = {}
    for xf in x_fracs:
        realized = a_next > xf
        hit = pred_masks & realized
        d = dict(base)
        d["n_realized"] = realized.sum(-1).unsqueeze(0).expand(P, -1)
        d["n_hit"] = hit.sum(-1)
        for cls in CLASSES:
            m = masks[cls]
            d[f"n_realized_{cls}"] = (realized & m).sum(-1).unsqueeze(0).expand(P, -1)
            d[f"n_hit_{cls}"] = (hit & m).sum(-1)
        out[xf] = d
    return out


def ema_update(
    ema_prev: torch.Tensor, a_next: torch.Tensor, lam: float
) -> torch.Tensor:
    """ema_t = (1-lam) * pad(ema_{t-1}) + lam * a_t over the grown support
    (the new position enters with zero history)."""
    R, S_prev = ema_prev.shape
    assert a_next.shape == (R, S_prev + 1)
    pad = torch.zeros(R, S_prev + 1, dtype=ema_prev.dtype)
    pad[:, :S_prev] = ema_prev
    return (1 - lam) * pad + lam * a_next


# ---------------------------------------------------------------------------
# The measurement loop (model + tokens injectable for offline tiny-factory tests)
# ---------------------------------------------------------------------------


def _max_positions(config) -> int:
    for attr in ("n_positions", "max_position_embeddings"):
        v = getattr(config, attr, None)
        if v:
            return int(v)
    return 1 << 30


def run_for_model(
    cfg: Config, model_name: str, label: str, model=None, tokens=None
) -> tuple[list[dict], list[dict], dict]:
    """Teacher-forced decode with attention capture; returns (metric rows,
    per-step rows for the gate cell, meta). `model`/`tokens` are injectable
    so offline tests run tiny factories — None loads the real model (eager
    attention, fp32 on CPU) and the standard wikitext eval slice."""
    if model is None:
        from transformers import AutoModelForCausalLM

        print(f"[{label}] loading {model_name} (eager, fp32)", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, attn_implementation="eager"
        ).eval()
    # output_attentions needs eager attention (sdpa returns no weights) —
    # force it for injected models too.
    model.set_attn_implementation("eager")
    max_pos = _max_positions(model.config)
    eff_prompt = min(cfg.prompt_len, max_pos - cfg.decode_steps)
    assert eff_prompt > cfg.n_sink + cfg.n_recent, "prompt too short for classes"
    total = eff_prompt + cfg.decode_steps
    if tokens is None:
        from bmx.eval.layer_swap import load_eval_tokens

        tokens = load_eval_tokens(
            model_name, n_tokens=total, token_offset=cfg.token_offset
        )
    tokens = tokens.flatten()
    assert tokens.numel() >= total, f"need {total} tokens, got {tokens.numel()}"
    ids = tokens[:total].view(1, total)

    acc: dict[tuple, dict[str, torch.Tensor]] = {}
    step_rows: list[dict] = []
    prev_rows: list[torch.Tensor] | None = None
    ema_rows: list[torch.Tensor] | None = None
    n_pairs = 0
    n_layers = n_heads = 0
    t0 = time.time()
    with torch.no_grad():
        out = model(ids[:, :eff_prompt], use_cache=True)
        cache = out.past_key_values
        print(f"[{label}] prefill {eff_prompt} done ({time.time() - t0:.1f}s)")
        for i in range(cfg.decode_steps):
            pos = eff_prompt + i
            out = model(
                ids[:, pos : pos + 1],
                past_key_values=cache,
                use_cache=True,
                output_attentions=True,
            )
            cache = out.past_key_values
            assert out.attentions, "no attention weights returned (need eager)"
            # Per layer: (H, S_cur) q-head rows + the head-mean pooled row.
            rows = []
            for att in out.attentions:
                a = att[0, :, 0, :].float()
                rows.append(torch.cat([a, a.mean(0, keepdim=True)], dim=0))
            n_layers, n_heads = len(rows), rows[0].shape[0] - 1
            S_next = rows[0].shape[-1]
            assert S_next == pos + 1, f"attention row length {S_next} != {pos + 1}"
            assert torch.isfinite(rows[0]).all()
            assert torch.allclose(
                rows[0].sum(-1), torch.ones(rows[0].shape[0]), atol=1e-3
            ), "attention rows must be normalized"

            if prev_rows is not None:
                masks = class_masks(S_next, cfg.n_sink, cfg.n_recent)
                x_fracs = [x / 100.0 for x in cfg.x_pct_grid]
                tally = {
                    p: {s: [0.0, 0.0, 0.0, 0] for s in ("head", "layer_pooled")}
                    for p in PREDICTORS
                }
                for layer, (a_prev, ema_prev, a_next) in enumerate(
                    zip(prev_rows, ema_rows, rows)
                ):
                    for kf in cfg.k_frac_grid:
                        k = max(1, math.ceil(kf * S_next))
                        sets = predictor_sets(a_prev, ema_prev, k, cfg.n_recent)
                        pred_masks = set_mask_batch(
                            torch.stack([sets[p] for p in PREDICTORS]), S_next
                        )
                        scored = score_sets_batch(pred_masks, a_next, x_fracs, masks)
                        for x, xf in zip(cfg.x_pct_grid, x_fracs):
                            s = scored[xf]
                            slot = acc.setdefault((layer, x, kf), {})
                            for f, v in s.items():
                                v64 = v.to(torch.float64)
                                slot[f] = slot[f] + v64 if f in slot else v64.clone()
                            if x == cfg.gate_x_pct and kf == cfg.gate_k_frac:
                                for pi, pred in enumerate(PREDICTORS):
                                    for scope, sl in (
                                        ("head", slice(0, n_heads)),
                                        ("layer_pooled", slice(n_heads, n_heads + 1)),
                                    ):
                                        t = tally[pred][scope]
                                        t[0] += float(s["n_hit"][pi, sl].sum())
                                        t[1] += float(s["n_realized"][pi, sl].sum())
                                        t[2] += float(s["mass_cov"][pi, sl].sum())
                                        t[3] += int(sl.stop - sl.start)
                n_pairs += 1
                for pred in PREDICTORS:
                    for scope in ("head", "layer_pooled"):
                        nh, nr, mc, n_rows = tally[pred][scope]
                        step_rows.append(
                            dict(
                                model=label,
                                step=i,
                                S=S_next,
                                predictor=pred,
                                scope=scope,
                                n_hit=nh,
                                n_realized=nr,
                                recall=(nh / nr if nr > 0 else float("nan")),
                                mass_cov_sum=mc,
                                n_rows=n_rows,
                            )
                        )
            ema_rows = (
                rows
                if ema_rows is None
                else [ema_update(e, r, cfg.ema_lambda) for e, r in zip(ema_rows, rows)]
            )
            prev_rows = rows
            if (i + 1) % 32 == 0:
                print(
                    f"[{label}] step {i + 1}/{cfg.decode_steps} "
                    f"(S={S_next}, {time.time() - t0:.1f}s)",
                    flush=True,
                )

    assert n_pairs == cfg.decode_steps - 1
    R = n_heads + 1
    rows_out: list[dict] = []
    for (layer, x, kf), s in acc.items():
        for pi, pred in enumerate(PREDICTORS):
            for r in range(R):
                nr = float(s["n_realized"][pi, r])
                nh = float(s["n_hit"][pi, r])
                row = dict(
                    model=label,
                    layer=layer,
                    head=(r if r < n_heads else -1),
                    scope=("head" if r < n_heads else "layer_pooled"),
                    predictor=pred,
                    x_pct=x,
                    k_frac=kf,
                    n_steps=n_pairs,
                    n_realized=nr,
                    n_hit=nh,
                    recall=(nh / nr if nr > 0 else float("nan")),
                    mass_cov_sum=float(s["mass_cov"][pi, r]),
                    mass_cov_mean=float(s["mass_cov"][pi, r]) / n_pairs,
                )
                for cls in CLASSES:
                    row[f"n_realized_{cls}"] = float(s[f"n_realized_{cls}"][pi, r])
                    row[f"n_hit_{cls}"] = float(s[f"n_hit_{cls}"][pi, r])
                    row[f"mass_{cls}_sum"] = float(s[f"mass_{cls}"][pi, r])
                    row[f"mass_cov_{cls}_sum"] = float(s[f"mass_cov_{cls}"][pi, r])
                rows_out.append(row)
    meta = dict(
        label=label,
        model_name=model_name,
        n_layers=n_layers,
        n_heads=n_heads,
        eff_prompt=eff_prompt,
        S_first=eff_prompt + 2,
        S_last=eff_prompt + cfg.decode_steps,
        n_step_pairs=n_pairs,
    )
    return rows_out, step_rows, meta


# ---------------------------------------------------------------------------
# Aggregation + gate (pure over the metrics DataFrames; pinned by tests)
# ---------------------------------------------------------------------------


def agg_micro(sel: pd.DataFrame) -> dict:
    """Micro-pooled recall + mass coverage over the selected rows: recall =
    total hits / total realized events; mass_coverage = mean over (row, step)
    of the covered attention mass (each row-step's full read mass is 1)."""
    assert not sel.empty
    nr, nh = float(sel.n_realized.sum()), float(sel.n_hit.sum())
    row_steps = float(sel.n_steps.sum())  # rows x steps each
    return dict(
        recall=(nh / nr if nr > 0 else float("nan")),
        mass_coverage=float(sel.mass_cov_sum.sum()) / row_steps,
        n_realized=nr,
    )


def model_cell(
    df: pd.DataFrame, label: str, x: float, kf: float, scope: str
) -> dict[str, dict]:
    """Per-predictor micro aggregates at one (X, k_frac) grid cell."""
    sel = df[
        (df.model == label) & (df.x_pct == x) & (df.k_frac == kf) & (df.scope == scope)
    ]
    return {p: agg_micro(sel[sel.predictor == p]) for p in PREDICTORS}


def best_of(cell: dict[str, dict]) -> tuple[str, dict]:
    """The gate's max-over-family: the predictor with the highest micro
    recall (the mechanism ships its best cheap predictor)."""
    pred = max(cell, key=lambda p: cell[p]["recall"])
    return pred, cell[pred]


def grid_best(
    df: pd.DataFrame,
    label: str,
    scope: str,
    x_grid: tuple[float, ...],
    kf_grid: tuple[float, ...],
) -> dict[str, dict]:
    out = {}
    for x in x_grid:
        for kf in kf_grid:
            pred, best = best_of(model_cell(df, label, x, kf, scope))
            out[f"X={x:g}%|k={kf:g}S"] = dict(
                predictor=pred,
                recall=best["recall"],
                mass_coverage=best["mass_coverage"],
            )
    return out


def decomposition(df: pd.DataFrame, label: str, pred: str, x: float, kf: float) -> dict:
    """Static-vs-content decomposition at the gate cell (scope=head): where
    do realized events / read mass live (sink / recent / content), and how
    predictable is each class."""
    sel = df[
        (df.model == label)
        & (df.x_pct == x)
        & (df.k_frac == kf)
        & (df.scope == "head")
        & (df.predictor == pred)
    ]
    assert not sel.empty
    total_real = float(sel.n_realized.sum())
    total_mass = float(sel.n_steps.sum())  # each row-step carries mass 1
    out: dict = {}
    for cls in CLASSES:
        nr, nh = float(sel[f"n_realized_{cls}"].sum()), float(sel[f"n_hit_{cls}"].sum())
        m, mc = (
            float(sel[f"mass_{cls}_sum"].sum()),
            float(sel[f"mass_cov_{cls}_sum"].sum()),
        )
        out[cls] = dict(
            realized_event_share=(nr / total_real if total_real else float("nan")),
            realized_mass_share=m / total_mass,
            recall=(nh / nr if nr > 0 else float("nan")),
            mass_coverage=(mc / m if m > 0 else float("nan")),
        )
    out["static_realized_mass_share"] = (
        out["sink"]["realized_mass_share"] + out["recent"]["realized_mass_share"]
    )
    return out


def worst_head(df: pd.DataFrame, label: str, pred: str, x: float, kf: float) -> dict:
    """The worst (layer, head) by micro recall at the gate cell — heads with
    zero realized events carry no recall and are excluded (counted)."""
    sel = df[
        (df.model == label)
        & (df.x_pct == x)
        & (df.k_frac == kf)
        & (df.scope == "head")
        & (df.predictor == pred)
    ]
    assert not sel.empty
    n_empty = int((sel.n_realized == 0).sum())
    live = sel[sel.n_realized > 0]
    row = live.loc[live.recall.idxmin()]
    return dict(
        layer=int(row["layer"]),
        head=int(row["head"]),  # NOT row.head — that's the pandas method
        recall=float(row.recall),
        mass_coverage=float(row.mass_cov_mean),
        n_realized=float(row.n_realized),
        n_heads_without_realized_events=n_empty,
    )


def stability(
    steps_df: pd.DataFrame,
    label: str,
    pred: str,
    scope: str = "head",
    threshold: float = GATE_THRESHOLD,
) -> dict:
    """Per-step pooled recall series at the gate cell: is predictability
    stable across the decode window?"""
    sel = steps_df[
        (steps_df.model == label)
        & (steps_df.predictor == pred)
        & (steps_df.scope == scope)
    ]
    r = sel.recall.dropna()
    assert len(r) > 0
    return dict(
        n_steps=int(len(r)),
        mean=float(r.mean()),
        std=float(r.std()),
        min=float(r.min()),
        frac_steps_ge_gate=float((r >= threshold).mean()),
    )


def gate_eval(
    recall_by_model: dict[str, float], threshold: float = GATE_THRESHOLD
) -> dict:
    """PRE-REGISTERED gate: every model's best-predictor micro recall@k at
    (X=1%, k=0.05S, scope=head) must be >= 0.9 (boundary inclusive)."""
    assert recall_by_model, "no models to gate"
    min_model = min(recall_by_model, key=recall_by_model.get)  # type: ignore[arg-type]
    gate_pass = all(v >= threshold for v in recall_by_model.values())
    return dict(
        threshold=threshold,
        per_model_recall=dict(recall_by_model),
        min_model=min_model,
        min_recall=recall_by_model[min_model],
        gate_pass=gate_pass,
        gate_outcome=(
            "CONFIRM — one-step-ahead recall@k >= 0.9 for tokens carrying > X% "
            "attention mass on every model: the two-tier (cold-2bit-HBM + "
            "exact-host speculative prefetch) mechanism graduates to a spec"
            if gate_pass
            else "KILL — one-step-ahead recall@k < 0.9: the two-tier prefetch "
            "mechanism is killed by prefetch pollution before any engineering"
        ),
    )


def assemble_verdict(
    df: pd.DataFrame, steps_df: pd.DataFrame, cfg: Config, metas: list[dict]
) -> dict:
    per_model: dict = {}
    recalls: dict[str, float] = {}
    for meta in metas:
        label = meta["label"]
        cell = model_cell(df, label, cfg.gate_x_pct, cfg.gate_k_frac, "head")
        pred, best = best_of(cell)
        pool_pred, pool_best = best_of(
            model_cell(df, label, cfg.gate_x_pct, cfg.gate_k_frac, "layer_pooled")
        )
        recalls[label] = best["recall"]
        per_model[label] = dict(
            **meta,
            per_predictor=cell,
            best_predictor=pred,
            best=best,
            layer_pooled_companion=dict(best_predictor=pool_pred, **pool_best),
            worst_head=worst_head(df, label, pred, cfg.gate_x_pct, cfg.gate_k_frac),
            decomposition=decomposition(
                df, label, pred, cfg.gate_x_pct, cfg.gate_k_frac
            ),
            stability=stability(steps_df, label, pred),
            grid=grid_best(df, label, "head", cfg.x_pct_grid, cfg.k_frac_grid),
        )
    return dict(
        task="storm Task-8 attention predictability (Tier-3 prefetch gatekeeper)",
        pre_registered=dict(
            gate_x_pct=cfg.gate_x_pct,
            gate_k_frac=cfg.gate_k_frac,
            threshold=GATE_THRESHOLD,
            gate_quantity=(
                "per model: MAX over the 4 pre-registered cheap predictors of "
                "MICRO-pooled recall@k over ALL (layer, head, step) realized "
                "events (> X% per-head mass) at scope=head; gate passes only "
                "if EVERY model clears >= 0.9"
            ),
            predictors=list(PREDICTORS),
            n_recent=cfg.n_recent,
            n_sink=cfg.n_sink,
            ema_lambda=cfg.ema_lambda,
        ),
        per_model=per_model,
        gate=gate_eval(recalls),
        caveats=[
            "teacher-forced decode (the plan's harness); free-running drift "
            "is unmeasured",
            "mechanism scale (124M/0.6B); Llama-scale rides the next rental "
            "only if this gate survives",
            "gpt2 prompt clamped to n_positions - decode_steps = 768 (its "
            "absolute context ceiling is 1024)",
            "the recent class (newest 16 positions) is trivially resident in "
            "any deployment — the decomposition separates it so content "
            "predictability is read directly",
            "attention rows are q-head-level; GQA KV-fetch granularity "
            "(union over q-heads sharing a KV head) lies between scope=head "
            "and scope=layer_pooled",
        ],
        git_sha=git_sha(),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Config):
    assert len(cfg.models) == len(cfg.labels) >= 1
    assert cfg.gate_x_pct in cfg.x_pct_grid, "gate X must be a grid point"
    assert cfg.gate_k_frac in cfg.k_frac_grid, "gate k_frac must be a grid point"
    assert cfg.decode_steps >= 2, "need at least one (t, t+1) pair"
    torch.manual_seed(cfg.seed)

    run = (
        create_run("storm_attn_predict", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("storm_attn_predict", cfg)
    )

    all_rows: list[dict] = []
    all_steps: list[dict] = []
    metas: list[dict] = []
    for model_name, label in zip(cfg.models, cfg.labels):
        rows, steps, meta = run_for_model(cfg, model_name, label)
        all_rows.extend(rows)
        all_steps.extend(steps)
        metas.append(meta)

    df = pd.DataFrame(all_rows)
    steps_df = pd.DataFrame(all_steps)
    write_metrics(run, df)
    write_metrics(run, steps_df, name="steps")

    verdict = assemble_verdict(df, steps_df, cfg, metas)
    (run / "verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n" + "=" * 78)
    print("STORM TASK-8 VERDICT — attention predictability (Tier-3 gatekeeper)")
    print("=" * 78)
    print(json.dumps(verdict, indent=2))
    print(f"\n-> {run}")
    return run


if __name__ == "__main__":
    main(tyro.cli(Config))
