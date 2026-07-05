"""Isolated decode-path A/B profile: packed (Triton) vs streaming (PyTorch) vs dense.

GH200 companion to docs/2026-07-04-triton-decode-desk-review.md and Wave-3 Task 5.
Run ONLY on an otherwise-idle GPU (numbers measured beside another live run are
contaminated — the whole reason this script exists).

Mode A (default): end-to-end per-token decode latency for three caches on the SAME
real model + prompt — DynamicCache (dense fp16 baseline), StreamingQuantizedCache
(PyTorch; committed prefix resident dequantized, dense SDPA), PackedStreamingCache
(packed-resident, fused Triton decode). Greedy-token parity between the two quantized
caches is asserted BEFORE any timing is trusted (oracle-gated perf discipline).
Per-token decode ms = (t(1+n_decode tokens) - t(1 token)) / n_decode, which cancels
prefill and includes every host-side cost the desk review indicts (tail merge, stack
maintenance, launch overhead).

Mode B (--profile-steps N): torch.profiler over N packed decode steps; prints top ops
by CUDA time and by CALL COUNT — the launch-count signature of desk-review findings
F1 (PyTorch tail merge every step) and F2 (per-call H2D constant uploads).

Usage (VM):
  uv run python scripts/profile_decode_ab.py --model-name meta-llama/Llama-3.1-8B-Instruct
  uv run python scripts/profile_decode_ab.py --model-name ... --ctx-lens 4096 16384 --profile-steps 20
"""

from __future__ import annotations

import dataclasses
import time

import torch
import tyro

from bmx.cache.packed_streaming import PackedStreamingCache
from bmx.cache.recipes import spec_pair
from bmx.cache.streaming import StreamingQuantizedCache


@dataclasses.dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    # k2b_ph = the fused-kernel-compatible real recipe (lowrank K + per-head turboquant V).
    arm: str = "k2b_ph"
    ctx_lens: tuple[int, ...] = (4096, 16384, 65536)
    n_decode: int = 64  # decode tokens per timed generate
    n_parity: int = 32  # greedy tokens compared between the two quantized caches
    rank: int = 16
    group: int = 64
    seed: int = 0
    profile_steps: int = 0  # >0: also run Mode B on the packed cache at ctx_lens[0]
    skip_dense: bool = False  # dense fp16 OOMs first at long ctx; skip to keep sweeping


def _prompt_ids(tokenizer, ctx_len: int, device) -> torch.Tensor:
    torch.manual_seed(1234)
    vocab = int(tokenizer.vocab_size)
    return torch.randint(low=10, high=vocab - 10, size=(1, ctx_len), device=device)


def _timed_generate(
    model, cache, input_ids, max_new_tokens: int
) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            past_key_values=cache,
        )
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def _fresh_cache(kind: str, model, cfg: Config):
    if kind == "dense":
        from transformers import DynamicCache

        return DynamicCache()
    k_spec, v_spec = spec_pair(cfg.arm, rank=cfg.rank, group=cfg.group, seed=cfg.seed)
    cls = StreamingQuantizedCache if kind == "streaming" else PackedStreamingCache
    return cls(model.config, k_spec=k_spec, v_spec=v_spec)


def _run_one(model, tokenizer, cfg: Config, kind: str, ctx_len: int) -> dict:
    """One (cache-kind, ctx) cell: warm 1-token generate, then the timed pair."""
    ids_ = _prompt_ids(tokenizer, ctx_len, model.device)

    def gen(max_new: int):
        cache = _fresh_cache(kind, model, cfg)
        attach = getattr(cache, "attach", None)
        if attach is not None:
            attach(model)
        out, dt = _timed_generate(model, cache, ids_, max_new)
        detach = getattr(cache, "detach", None)
        if detach is not None:
            detach()
        return out, dt

    gen(1)  # warmup: autotune, allocator, codebook caches
    out_1, t_1 = gen(1)
    out_n, t_n = gen(1 + cfg.n_decode)
    ms_per_token = (t_n - t_1) * 1000.0 / cfg.n_decode
    peak = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    return {
        "cache": kind,
        "ctx": ctx_len,
        "ms_per_decode_token": round(ms_per_token, 3),
        "prefill_plus1_s": round(t_1, 3),
        "peak_gib_so_far": round(peak, 2),
        "tokens": out_n[0, ctx_len:].tolist()[:8],
    }


def main(cfg: Config) -> None:
    assert torch.cuda.is_available(), "GH200-only script; refuses to mis-measure on CPU"
    from experiments._common import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(cfg.model_name, "cuda")

    # --- Parity gate (oracle-gated perf: no timing is reported without it) --------
    ids_p = _prompt_ids(tokenizer, min(cfg.ctx_lens), model.device)
    outs = {}
    for kind in ("streaming", "packed"):
        cache = _fresh_cache(kind, model, cfg)
        cache.attach(model)
        with torch.no_grad():
            outs[kind] = model.generate(
                ids_p,
                max_new_tokens=cfg.n_parity,
                do_sample=False,
                use_cache=True,
                past_key_values=cache,
            )
        cache.detach()
    assert torch.equal(outs["streaming"], outs["packed"]), (
        "PARITY FAIL: packed and streaming greedy tokens diverge — fix correctness "
        "before believing any latency number below"
    )
    print(f"[parity] streaming == packed over {cfg.n_parity} greedy tokens: OK")

    # --- Path probe: PROVE which decode kernel fires for this arm ------------------
    # The 2026-07-03 "packed 12x slower" A/B ran turboquant_mse, which routes to
    # NEITHER fused kernel (chunked fallback) — desk-review finding F0. Counting the
    # actual calls makes that mistake impossible to repeat silently.
    import bmx.cache.packed_streaming as _ps

    counts = {"fused_packed": 0, "fused_k2b": 0, "chunked": 0}

    def _wrap(name, fn):
        def inner(*a, **k):
            counts[name] += 1
            return fn(*a, **k)

        return inner

    originals = (
        _ps.fused_decode_attention_packed,
        _ps.fused_decode_attention_k2b,
        _ps.chunked_dequant_attention,
    )
    _ps.fused_decode_attention_packed = _wrap("fused_packed", originals[0])
    _ps.fused_decode_attention_k2b = _wrap("fused_k2b", originals[1])
    _ps.chunked_dequant_attention = _wrap("chunked", originals[2])
    try:
        cache = _fresh_cache("packed", model, cfg)
        cache.attach(model)
        with torch.no_grad():
            model.generate(
                ids_p,
                max_new_tokens=4,
                do_sample=False,
                use_cache=True,
                past_key_values=cache,
            )
        cache.detach()
    finally:
        (
            _ps.fused_decode_attention_packed,
            _ps.fused_decode_attention_k2b,
            _ps.chunked_dequant_attention,
        ) = originals
    print(f"[path probe] arm={cfg.arm} decode attend calls: {counts}")
    n_layers = model.config.num_hidden_layers
    fused_total = counts["fused_packed"] + counts["fused_k2b"]
    assert fused_total >= 3 * n_layers, (
        f"arm {cfg.arm!r} is NOT hitting a fused kernel at decode "
        f"(counts={counts}) — this A/B would measure the chunked fallback, not the "
        f"Triton path (the exact F0 mistake). Use k2b_ph or an rtn arm."
    )

    # --- Mode A: the A/B/C table ---------------------------------------------------
    rows = []
    kinds = (["dense"] if not cfg.skip_dense else []) + ["streaming", "packed"]
    for ctx in cfg.ctx_lens:
        for kind in kinds:
            try:
                row = _run_one(model, tokenizer, cfg, kind, ctx)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row = {"cache": kind, "ctx": ctx, "ms_per_decode_token": float("nan")}
                print(f"[oom] {kind} @ {ctx} — recorded as NaN")
            rows.append(row)
            print(row)

    import pandas as pd

    df = pd.DataFrame(rows)
    print("\n=== per-token decode latency (ms) ===")
    print(
        df.pivot_table(
            index="ctx", columns="cache", values="ms_per_decode_token"
        ).to_string()
    )

    # --- Mode B: op-level attribution on the packed path ---------------------------
    if cfg.profile_steps > 0:
        # Fresh cache; the profiled window includes ONE prefill (identifiable by its
        # flash-SDPA/is_prefill ops) followed by profile_steps decode steps whose
        # repeated per-step signature dominates the call counts.
        ctx = cfg.ctx_lens[0]
        ids_ = _prompt_ids(tokenizer, ctx, model.device)
        cache = _fresh_cache("packed", model, cfg)
        cache.attach(model)
        from torch.profiler import ProfilerActivity, profile

        with torch.no_grad():
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                model.generate(
                    ids_,
                    max_new_tokens=cfg.profile_steps,
                    do_sample=False,
                    use_cache=True,
                    past_key_values=cache,
                )
        cache.detach()
        print("\n=== top 25 by CUDA time ===")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
        print("\n=== top 25 by call count (the F1/F2 signature) ===")
        print(prof.key_averages().table(sort_by="count", row_limit=25))


if __name__ == "__main__":
    main(tyro.cli(Config))
