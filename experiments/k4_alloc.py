"""K4 Stage 2: sensitivity census + across-layer allocation (gate G2).

Three parts:

A — sensitivity census: one shared `run_prefill` state, then per layer i,
    swap ONLY that layer's keys to `turboquant_mse @ sens_bits` (post-RoPE,
    pre_rope=False — keeps gpt2/llama uniform) while every other layer stays
    fp16 (both K and V). `s_i = log(ppl_i) - log(ppl_fp16)`: the marginal NLL
    cost of degrading layer i alone, holding everything else exact.

B — allocation: `greedy_layer_allocation` (public, module-level) does a
    greedy marginal-upgrade walk over the Task-7 frontier's per-layer
    distortion-vs-bits curves (`turboquant_mse`, kind="k_pre"), weighted by
    Part A's sensitivities. Provably optimal for convex per-layer curves:
    at each step it buys the cheapest-per-bit distortion reduction available
    anywhere, so the final allocation minimizes sum_l s_l * D_l(bits_l)
    subject to mean(bits) == target_mean (a discrete water-filling argument).

C — G2 verdict: gpt2 end-to-end. Allocated per-layer `turboquant_mse` bits
    (from B) vs a variance-blind uniform comparator at the SAME mean bits
    (integer target -> constant; fractional target -> deterministic
    alternating floor/ceil over layer index, mirroring
    `experiments/k2_blockklt.py`'s `_uniform_bits_vector` design but indexed
    by layer instead of channel). V stays fp16 on both sides so only the K
    allocation lever is being isolated. g2_pass requires allocated ppl <=
    uniform ppl at every target_mean.

Note: this exercises the ALLOCATION lever with EXISTING arms (turboquant_mse
at integer bit grids); the spectral codec's own across-layer allocation
lands with the Stage-3 integration plan.
"""

from __future__ import annotations

import dataclasses
import json
import math

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, write_metrics
from bmx.cache.ppl_eval import CacheCodecSpec, quantized_prefill_ppl, run_prefill
from bmx.eval.layer_swap import load_eval_tokens


@dataclasses.dataclass
class Config:
    model_name: str = "gpt2"
    frontier_parquet: str = ""  # REQUIRED: path to the Task-7/11 gpt2 frontier run
    n_prefill: int = 768
    n_cont: int = 256
    target_means: tuple[float, ...] = (2.5, 3.0)
    sens_bits: int = 2
    out_root: str = ""


# ---------------------------------------------------------------------------
# Part B: greedy across-layer allocation (verbatim from the task brief).
# ---------------------------------------------------------------------------


def greedy_layer_allocation(
    curves: dict[int, dict[float, float]],
    s: dict[int, float],
    budgets: tuple[float, ...],
    target_mean: float,
) -> dict[int, float]:
    """curves[layer][budget] = distortion; start every layer at min(budgets),
    repeatedly upgrade the layer with the largest s[l]*(D[cur]-D[next]) per
    budget-unit until the mean budget reaches target_mean. Deterministic
    (ties broken by layer index)."""
    grid = sorted(budgets)
    cur = {l: 0 for l in curves}  # noqa: E741 (index into grid)

    def mean_b():
        return sum(grid[i] for i in cur.values()) / len(cur)

    while mean_b() < target_mean - 1e-9:
        best, best_gain = None, -1.0
        for l in sorted(curves):  # noqa: E741
            i = cur[l]
            if i + 1 >= len(grid):
                continue
            gain = s[l] * (curves[l][grid[i]] - curves[l][grid[i + 1]])
            gain /= grid[i + 1] - grid[i]
            if gain > best_gain:
                best, best_gain = l, gain
        if best is None:
            break
        cur[best] += 1
    return {l: grid[i] for l, i in cur.items()}  # noqa: E741


# ---------------------------------------------------------------------------
# Uniform comparator: deterministic alternating floor/ceil over layer index,
# mirroring k2_blockklt.py's _uniform_bits_vector (there: channel index).
# ---------------------------------------------------------------------------


def _uniform_bits_by_layer(n_layer: int, target_mean: float) -> dict[int, int]:
    """Variance-blind per-layer bits with mean == target_mean.

    Integer target -> constant. Fractional -> floor everywhere, ceil on
    round(frac*n_layer) evenly-spread layer indices (deterministic,
    allocation-free baseline at fractional mean bits)."""
    lo = int(math.floor(target_mean))
    n_hi = round((target_mean - lo) * n_layer)
    bits = {layer: lo for layer in range(n_layer)}
    if n_hi > 0:
        idx = (torch.arange(n_hi).double() * (n_layer / n_hi)).long()
        for i in idx.tolist():
            bits[i] = lo + 1
    realized_mean = sum(bits.values()) / n_layer
    assert abs(realized_mean - target_mean) < 1e-6
    return bits


# ---------------------------------------------------------------------------
# Part A helpers
# ---------------------------------------------------------------------------


def _tq_spec(bits: int) -> CacheCodecSpec:
    return CacheCodecSpec(arm="turboquant_mse", bits=bits, pre_rope=False)


def _fp16_spec() -> CacheCodecSpec:
    return CacheCodecSpec(arm="fp16")


def _sensitivity_census(model, ids, cfg: Config, state, n_layer: int, ppl_fp16: float):
    """Part A: per-layer marginal NLL cost of degrading ONLY that layer's K."""
    rows: list[dict] = []
    s: dict[int, float] = {}
    log_ppl_fp16 = math.log(ppl_fp16)
    for i in range(n_layer):
        k_specs = [_fp16_spec()] * n_layer
        k_specs[i] = _tq_spec(cfg.sens_bits)
        v_specs = [_fp16_spec()] * n_layer
        result = quantized_prefill_ppl(
            model,
            ids,
            cfg.n_prefill,
            k_spec=_fp16_spec(),
            v_spec=_fp16_spec(),
            state=state,
            k_specs=k_specs,
            v_specs=v_specs,
        )
        s_i = math.log(result["ppl"]) - log_ppl_fp16
        s[i] = s_i
        rows.append(
            dict(
                model=cfg.model_name,
                layer=i,
                kind="sensitivity",
                s_i=s_i,
                ppl=result["ppl"],
                sens_bits=cfg.sens_bits,
                n_prefill=cfg.n_prefill,
            )
        )
        print(f"  layer={i:2d}  ppl={result['ppl']:.4f}  s_i={s_i:.6f}", flush=True)
    return rows, s


# ---------------------------------------------------------------------------
# Part C helpers
# ---------------------------------------------------------------------------


def _load_curves(frontier_parquet: str) -> dict[int, dict[float, float]]:
    """Per-layer distortion-vs-bits curves from the Task-7 frontier parquet:
    rows arm=="turboquant_mse", kind=="k_pre"; headline = logit_rope where
    non-NaN else logit (gpt2 frontier runs have model_name="" -> logit only)."""
    df = pd.read_parquet(frontier_parquet)
    sub = df[(df.arm == "turboquant_mse") & (df.kind == "k_pre")].copy()
    assert not sub.empty, (
        f"no turboquant_mse/k_pre rows in {frontier_parquet!r} — wrong parquet?"
    )
    headline = sub["logit_rope"].where(sub["logit_rope"].notna(), sub["logit"])
    sub = sub.assign(headline=headline)
    curves: dict[int, dict[float, float]] = {}
    for layer, g in sub.groupby("layer"):
        curves[int(layer)] = {
            float(b): float(v) for b, v in zip(g.bits.tolist(), g.headline.tolist())
        }
    return curves


def main(cfg: Config):
    assert cfg.frontier_parquet, "frontier_parquet is required (Task-7 gpt2 run)"

    run = (
        create_run("k4_alloc", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_alloc", cfg)
    )

    from transformers import AutoModelForCausalLM

    print(f"Loading model: {cfg.model_name}")
    dtype = torch.float32 if "gpt2" in cfg.model_name.lower() else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype)
    model.eval()

    n_tokens = cfg.n_prefill + cfg.n_cont
    print(f"Loading {n_tokens} eval tokens...")
    tokens_1d = load_eval_tokens(cfg.model_name, n_tokens=n_tokens)
    ids = tokens_1d.unsqueeze(0)  # (1, n_tokens)

    print(f"Prefilling {cfg.n_prefill} tokens (shared state, post-RoPE arms)...")
    state = run_prefill(model, ids, cfg.n_prefill, capture_pre_rope=False)
    n_layer = len(state.cache.layers)

    print("Baseline fp16 ppl...")
    fp16_spec = _fp16_spec()
    base = quantized_prefill_ppl(
        model, ids, cfg.n_prefill, fp16_spec, fp16_spec, state=state
    )
    ppl_fp16 = base["ppl"]
    print(f"  ppl_fp16={ppl_fp16:.4f}")

    # ---- Part A: sensitivity census -----------------------------------
    print("\n=== Part A: sensitivity census ===")
    sens_rows, s_raw = _sensitivity_census(model, ids, cfg, state, n_layer, ppl_fp16)
    # Clamp s_i to a small positive floor: ppl noise can push a truly-flat
    # layer's measured s_i to ~0 or slightly negative, which would zero out
    # (or invert) its weight in the greedy allocator. A floor keeps every
    # layer eligible for upgrades without materially changing the ranking
    # of genuinely sensitive layers (whose s_i is orders of magnitude larger).
    s = {layer: max(v, 1e-6) for layer, v in s_raw.items()}

    # ---- Part B: greedy allocation over the frontier curves -------------
    print("\n=== Part B: greedy allocation ===")
    curves = _load_curves(cfg.frontier_parquet)
    missing = set(range(n_layer)) - set(curves.keys())
    assert not missing, f"frontier parquet missing layers {sorted(missing)}"
    budgets = tuple(sorted({b for c in curves.values() for b in c.keys()}))

    allocations: dict[float, dict[int, float]] = {}
    for target_mean in cfg.target_means:
        alloc = greedy_layer_allocation(curves, s, budgets, target_mean)
        allocations[target_mean] = alloc
        print(f"  target_mean={target_mean}: {alloc}")

    # ---- Part C: G2 verdict — allocated vs uniform at matched mean bits -
    print("\n=== Part C: G2 verdict ===")
    ppl_rows: list[dict] = []
    g2_targets: dict[str, dict] = {}
    g2_pass = True

    for target_mean in cfg.target_means:
        alloc = allocations[target_mean]
        k_specs_alloc = [
            CacheCodecSpec(arm="turboquant_mse", bits=int(alloc[i]), pre_rope=False)
            for i in range(n_layer)
        ]
        v_specs_fp16 = [_fp16_spec()] * n_layer
        result_alloc = quantized_prefill_ppl(
            model,
            ids,
            cfg.n_prefill,
            k_spec=_fp16_spec(),
            v_spec=_fp16_spec(),
            state=state,
            k_specs=k_specs_alloc,
            v_specs=v_specs_fp16,
        )

        uniform_bits = _uniform_bits_by_layer(n_layer, target_mean)
        k_specs_unif = [
            CacheCodecSpec(arm="turboquant_mse", bits=uniform_bits[i], pre_rope=False)
            for i in range(n_layer)
        ]
        result_unif = quantized_prefill_ppl(
            model,
            ids,
            cfg.n_prefill,
            k_spec=_fp16_spec(),
            v_spec=_fp16_spec(),
            state=state,
            k_specs=k_specs_unif,
            v_specs=v_specs_fp16,
        )

        bpe_delta = abs(result_alloc["bpe_k"] - result_unif["bpe_k"])
        mean_bits_check = bpe_delta < 0.1
        this_pass = bool(result_alloc["ppl"] <= result_unif["ppl"])
        g2_pass = g2_pass and this_pass

        ppl_rows.append(
            dict(
                model=cfg.model_name,
                kind="ppl_allocated",
                target_mean=target_mean,
                ppl=result_alloc["ppl"],
                bpe_k=result_alloc["bpe_k"],
                bpe_v=result_alloc["bpe_v"],
            )
        )
        ppl_rows.append(
            dict(
                model=cfg.model_name,
                kind="ppl_uniform",
                target_mean=target_mean,
                ppl=result_unif["ppl"],
                bpe_k=result_unif["bpe_k"],
                bpe_v=result_unif["bpe_v"],
            )
        )

        g2_targets[str(target_mean)] = dict(
            ppl_allocated=result_alloc["ppl"],
            ppl_uniform=result_unif["ppl"],
            bpe_k_allocated=result_alloc["bpe_k"],
            bpe_k_uniform=result_unif["bpe_k"],
            mean_bits_check_pass=mean_bits_check,
            pass_=this_pass,
        )
        print(
            f"  target_mean={target_mean}: "
            f"ppl_alloc={result_alloc['ppl']:.4f} (bpe_k={result_alloc['bpe_k']:.3f})  "
            f"ppl_unif={result_unif['ppl']:.4f} (bpe_k={result_unif['bpe_k']:.3f})  "
            f"mean_bits_check={mean_bits_check}  pass={this_pass}"
        )
        if not mean_bits_check:
            print(
                f"    WARNING: realized bpe_k mismatch "
                f"({result_alloc['bpe_k']:.3f} vs {result_unif['bpe_k']:.3f}) "
                f"exceeds 0.1 tolerance"
            )

    verdict = dict(
        g2_pass=bool(g2_pass),
        ppl_fp16=ppl_fp16,
        sens_bits=cfg.sens_bits,
        targets=g2_targets,
        note=(
            "Exercises the ALLOCATION lever with existing arms (turboquant_mse "
            "at integer bit grids); the spectral codec's own across-layer "
            "allocation lands with the Stage-3 integration plan."
        ),
    )
    (run / "g2_verdict.json").write_text(json.dumps(verdict, indent=2))
    (run / "allocation.json").write_text(
        json.dumps(
            {
                str(tm): {str(layer): b for layer, b in a.items()}
                for tm, a in allocations.items()
            },
            indent=2,
        )
    )

    df = pd.DataFrame(sens_rows + ppl_rows)
    write_metrics(run, df)

    print("\n" + "=" * 88)
    print("G2 VERDICT — allocated vs uniform turboquant_mse at matched mean bits")
    print("=" * 88)
    print(json.dumps(verdict, indent=2))
    print(f"\nTotal rows: {len(df)}")
    print(f"-> {run}")

    return run


if __name__ == "__main__":
    main(tyro.cli(Config))
