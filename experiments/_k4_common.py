"""Shared pieces for the K4 experiment family (k4_spectra, k4_frontier):
layer-keyed cache loading, RoPE setup + self-validation, and tail scoring.
"""

from __future__ import annotations

import re

import torch

from bmx.cache.collect import from_matrix, load_cache
from bmx.cache.metrics import logit_distortion, rel_fro
from bmx.cache.rope import apply_rope
from bmx.cache.spectral import query_position_moment

_LAYER_RE = re.compile(r"^layer(\d+)\.(k|v|q|k_pre)$")
DEPLOY_S = 32768


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
