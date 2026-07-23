"""Shared pieces for the K4 experiment family (k4_spectra, k4_frontier,
k4_alloc, k4_fit_packs): layer-keyed cache loading, RoPE setup +
self-validation, tail scoring, and the across-layer greedy allocator.
"""

from __future__ import annotations

import math
import re

import pandas as pd
import torch

from bmx.cache.collect import from_matrix, load_cache
from bmx.cache.metrics import logit_distortion, rel_fro
from bmx.cache.rope import apply_rope
from bmx.cache.spectral import query_position_moment

_LAYER_RE = re.compile(r"^layer(\d+)\.(k|v|q|k_pre)$")
DEPLOY_S = 32768

# Floor for measured per-layer sensitivities s_i: ppl noise can push a
# truly-flat layer's measured s_i to ~0 or slightly negative, which would zero
# out (or invert) its weight in the greedy allocator. A small positive floor
# keeps every layer eligible for upgrades without materially changing the
# ranking of genuinely sensitive layers (whose s_i is orders of magnitude
# larger).
SENS_FLOOR = 1e-6


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


def bucket_layer_keys(cache: dict) -> dict[int, dict[str, torch.Tensor]]:
    """Bucket an already-loaded cache dict's tensors by layer index."""
    layer_keys: dict[int, dict[str, torch.Tensor]] = {}
    for key, tensor in cache.items():
        m = _LAYER_RE.match(key)
        if m is None:
            continue
        layer_keys.setdefault(int(m.group(1)), {})[m.group(2)] = tensor
    return layer_keys


def load_layer_keys(cache_path: str) -> dict[int, dict[str, torch.Tensor]]:
    """Load a cache file and bucket its tensors by layer index."""
    return bucket_layer_keys(load_cache(cache_path))


def setup_rope(model_name: str, layer_keys: dict[int, dict[str, torch.Tensor]], layers):
    """Load RoPE config (if given) + self-validate on layer 0. Returns
    (rope_ready, get_cos_sin) where get_cos_sin(S) -> (cos, sin)."""
    rope_ready = False
    hf_config = None
    cos_sin_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    if model_name:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_name)
        rope_ready = True
        print(f"RoPE config loaded from {model_name}", flush=True)

    def get_cos_sin(S: int):
        from bmx.cache.rope import rope_cos_sin

        if S not in cos_sin_cache:
            cos_sin_cache[S] = rope_cos_sin(hf_config, S)
        return cos_sin_cache[S]

    if rope_ready and layers:
        l0 = layer_keys[layers[0]]
        k_pre0, k0 = l0["k_pre"].float(), l0["k"].float()
        cos0, sin0 = get_cos_sin(k_pre0.shape[1])
        rel = (
            (apply_rope(k_pre0, cos0, sin0) - k0).norm() / k0.norm().clamp_min(1e-12)
        ).item()
        print(
            f"[rope_validation] rel_fro(apply_rope(k_pre), k) = {rel:.4f}", flush=True
        )
        assert rel < 2e-2, f"RoPE self-validation FAILED: {rel:.4f} >= 2e-2"

    return rope_ready, get_cos_sin


def corpus_query_moment(
    corpus_layer_keys,
    corpus_get_cos_sins,
    rope_ready,
    layer_i,
    h_kv,
    d,
    position_stride,
):
    """Equal-weight per-cache average of query_position_moment over corpus caches.

    The deployment-grade W (query-heldout): each cache contributes its own stored
    queries with its own RoPE tables; no scored-cache query information enters.
    """
    W_sum = torch.zeros(h_kv, d, d, dtype=torch.float64)
    for lk, get_cs in zip(corpus_layer_keys, corpus_get_cos_sins):
        c_q_t = lk[layer_i]["q"]
        c_S = lk[layer_i]["k_pre"].shape[1]
        if rope_ready:
            c_cos, c_sin = get_cs(c_S)
        else:
            c_cos, c_sin = torch.ones(c_S, d), torch.zeros(c_S, d)
        W_sum += query_position_moment(
            c_q_t.float(), c_cos, c_sin, h_kv, position_stride=position_stride
        )
    return W_sum / len(corpus_layer_keys)


def _score_tail(M_hat, h_kv, tail, K_post_true, Q, cos, sin, rope_ready, k_true_t, M):
    K_hat = from_matrix(M_hat, h_kv)[:, tail, :].float()
    rf = rel_fro(M_hat[tail], M[tail])
    if rope_ready:
        K_hat_rope = apply_rope(K_hat, cos[tail], sin[tail])
        lg_rope = logit_distortion(K_post_true[:, tail], K_hat_rope, Q)
        lg = logit_distortion(k_true_t.float()[:, tail], K_hat, Q)
    else:
        lg = logit_distortion(k_true_t.float()[:, tail], K_hat, Q)
        lg_rope = float("nan")
    return rf, lg, lg_rope


def _tq_layer_curve(
    df: pd.DataFrame, headline_col: str
) -> dict[int, list[tuple[float, float]]]:
    """Per-layer turboquant_mse k_pre curve: sorted [(bpe, distortion), ...]."""
    sub = df[(df.arm == "turboquant_mse") & (df.kind == "k_pre")]
    curves: dict[int, list[tuple[float, float]]] = {}
    for layer, g in sub.groupby("layer"):
        pts = sorted(zip(g.bpe_model.tolist(), g[headline_col].tolist()))
        curves[int(layer)] = pts
    return curves


def _log_interp(pts: list[tuple[float, float]], x: float) -> tuple[float, bool]:
    """Interpolate log(y) linearly in x over sorted pts=[(x,y),...].
    Extrapolates log-linearly from the nearest two points when x is outside
    the range; returns (y, extrapolated)."""
    xs = [p[0] for p in pts]
    ys = [math.log(max(p[1], 1e-300)) for p in pts]
    if x <= xs[0]:
        if x == xs[0]:
            return math.exp(ys[0]), False
        x0, x1, y0, y1 = xs[0], xs[1], ys[0], ys[1]
        t = (x - x0) / (x1 - x0)
        return math.exp(y0 + t * (y1 - y0)), True
    if x >= xs[-1]:
        if x == xs[-1]:
            return math.exp(ys[-1]), False
        x0, x1, y0, y1 = xs[-2], xs[-1], ys[-2], ys[-1]
        t = (x - x0) / (x1 - x0)
        return math.exp(y0 + t * (y1 - y0)), True
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return math.exp(ys[i] + t * (ys[i + 1] - ys[i])), False
    raise AssertionError("unreachable")
