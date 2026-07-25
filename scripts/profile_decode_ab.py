"""Isolated decode-path A/B profile: packed (Triton/chunked) vs streaming (PyTorch)
vs dense.

GH200 companion to docs/2026-07-04-triton-decode-desk-review.md and Wave-3 Task 5.
Run ONLY on an otherwise-idle GPU (numbers measured beside another live run are
contaminated — the whole reason this script exists).

Mode A (default): end-to-end per-token decode latency for three caches on the SAME
real model + prompt — DynamicCache (dense fp16 baseline), StreamingQuantizedCache
(PyTorch; committed prefix resident dequantized, dense SDPA), PackedStreamingCache
(packed-resident; fused Triton decode for rtn_token/k2b_ph, chunked dequant decode
for every other arm INCLUDING spectral — spectral has a packed twin as of Task 2 but
no fused spectral kernel exists yet, Phase A). Greedy-token parity between the two
quantized caches is asserted BEFORE any timing is trusted (oracle-gated perf
discipline) — this now runs for spectral (pack-gated) arms too, since spectral has a
packed twin to be at parity with.
Per-token decode ms = (t(1+n_decode tokens) - t(1 token)) / n_decode, which cancels
prefill and includes every host-side cost the desk review indicts (tail merge, stack
maintenance, launch overhead).

Mode B (--profile-steps N): torch.profiler over N packed decode steps; prints top ops
by CUDA time and by CALL COUNT — the launch-count signature of desk-review findings
F1 (PyTorch tail merge every step) and F2 (per-call H2D constant uploads).

Mode C (--logit-probe N): numerical divergence between the packed and streaming
caches over N teacher-forced decode steps at max(ctx_lens) — both caches fed
streaming's greedy token at each step (so trajectories stay comparable even if they'd
otherwise diverge), recording max|logits_packed - logits_streaming| and the argmax
flip count per step. Gate (amended 2026-07-25): FAIL on drift-INEXPLICABLE flips
(streaming top-2 gap > that step's max-abs delta) or flips > N//8; near-tie flips
WARN with forensics — see _logit_probe's docstring. The max-abs envelope is
RECORDED, never gated bitwise — long-context accumulation-order drift across many
committed pages is expected and pre-existing, not a correctness bug.

Usage (VM):
  uv run python scripts/profile_decode_ab.py --model-name meta-llama/Llama-3.1-8B-Instruct
  uv run python scripts/profile_decode_ab.py --model-name ... --ctx-lens 4096 16384 --profile-steps 20
  uv run python scripts/profile_decode_ab.py --model-name ... --ctx-lens 65536 --logit-probe 32
"""

from __future__ import annotations

import dataclasses
import time

import torch
import tyro

from bmx.cache.codecs import PACK_GATED_ARMS
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
    logit_probe: int = 0
    """>0: also run Mode C — N teacher-forced decode steps at max(ctx_lens), recording
    per-step max|logits_packed - logits_streaming|, the streaming top-2 gap, and the
    argmax flip count. Gate: fail on drift-inexplicable flips (gap > step delta) or
    flips > N//8; near-tie flips warn; the max-abs envelope is recorded, never gated
    bitwise (see _logit_probe)."""
    skip_dense: bool = False  # dense fp16 OOMs first at long ctx; skip to keep sweeping
    pack_path: str = ""
    """Fitted spectral pack file — required for k4_* arms. Pack-gated (spectral) arms
    route through PackedStreamingCache same as any other arm as of Task 2 (chunked
    decode, no fused kernel) — the parity gate, path probe (inverted assertion), and
    packed column all run for them too."""


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


def _fresh_cache(kind: str, model, k_spec, v_spec):
    if kind == "dense":
        from transformers import DynamicCache

        return DynamicCache()
    cls = StreamingQuantizedCache if kind == "streaming" else PackedStreamingCache
    return cls(model.config, k_spec=k_spec, v_spec=v_spec)


def _run_one(model, tokenizer, cfg: Config, k_spec, v_spec, kind: str, ctx_len: int):
    """One (cache-kind, ctx) cell: warm 1-token generate, then the timed pair."""
    ids_ = _prompt_ids(tokenizer, ctx_len, model.device)

    def gen(max_new: int):
        cache = _fresh_cache(kind, model, k_spec, v_spec)
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
    k_spec, v_spec = spec_pair(
        cfg.arm, rank=cfg.rank, group=cfg.group, seed=cfg.seed, pack_path=cfg.pack_path
    )

    # Pack-gated arms (spectral K) now DO have a packed twin (Task 2): PackedStreamingCache
    # routes spectral through the chunked path by design (no fused spectral kernel exists
    # yet, Phase A). The parity gate and path probe below run for every arm, spectral
    # included — the short-ctx greedy parity gate binds gate 2 at real-model scale.

    # --- Parity gate (oracle-gated perf: no timing is reported without it) --------
    ids_p = _prompt_ids(tokenizer, min(cfg.ctx_lens), model.device)
    outs = {}
    for kind in ("streaming", "packed"):
        cache = _fresh_cache(kind, model, k_spec, v_spec)
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
        cache = _fresh_cache("packed", model, k_spec, v_spec)
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
    if k_spec.arm in PACK_GATED_ARMS:
        # Spectral has NO fused kernel (Phase A: chunked dequant is the licensed decode
        # path by design — see PackedStreamingCache's decode-time warning). A fused count
        # here would mean a kernel was silently wired without its oracle gates: invert
        # the assertion so THAT would fail loudly instead of being read as a win.
        assert counts["chunked"] > 0 and fused_total == 0, (
            f"arm {cfg.arm!r} is pack-gated (spectral) and expected to hit ONLY the "
            f"chunked path at decode (counts={counts}) — a nonzero fused count means a "
            f"kernel fired without its oracle gates; a zero chunked count means decode "
            f"never ran the attend path at all."
        )
        print(f"[path probe] arm={cfg.arm} chunked-by-design: OK (counts={counts})")
    else:
        assert fused_total >= 3 * n_layers, (
            f"arm {cfg.arm!r} is NOT hitting a fused kernel at decode "
            f"(counts={counts}) — this A/B would measure the chunked fallback, not the "
            f"Triton path (the exact F0 mistake). Use k2b_ph or an rtn arm."
        )

    _mode_a(model, tokenizer, cfg, k_spec, v_spec)

    if cfg.logit_probe > 0:
        _logit_probe(model, tokenizer, cfg, k_spec, v_spec)


def _mode_a(
    model,
    tokenizer,
    cfg: Config,
    k_spec,
    v_spec,
    kinds: tuple[str, ...] = ("dense", "streaming", "packed"),
) -> None:
    # --- Mode A: the A/B/C table ---------------------------------------------------
    if cfg.skip_dense:
        kinds = tuple(k for k in kinds if k != "dense")
    rows = []
    for ctx in cfg.ctx_lens:
        for kind in kinds:
            try:
                row = _run_one(model, tokenizer, cfg, k_spec, v_spec, kind, ctx)
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
        if "packed" not in kinds:
            print(
                f"[mode b] skipped: pack-gated arm {cfg.arm!r} has no packed cache "
                "to profile despite --profile-steps"
            )
            return
        # Fresh cache; the profiled window includes ONE prefill (identifiable by its
        # flash-SDPA/is_prefill ops) followed by profile_steps decode steps whose
        # repeated per-step signature dominates the call counts.
        ctx = cfg.ctx_lens[0]
        ids_ = _prompt_ids(tokenizer, ctx, model.device)
        cache = _fresh_cache("packed", model, k_spec, v_spec)
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


def _logit_probe(model, tokenizer, cfg: Config, k_spec, v_spec) -> None:
    """Mode C: numerical divergence between packed and streaming over N teacher-forced
    decode steps at max(ctx_lens).

    Both caches are prefilled on the SAME prompt, then advanced token-by-token with
    streaming's own greedy token fed to BOTH caches (teacher forcing) — this keeps the
    two trajectories comparable even if packed's own argmax would otherwise diverge
    the token stream, isolating the per-step logit delta from compounding drift.

    Gate (amended 2026-07-25, user-approved): a flip FAILS only when it is
    drift-INEXPLICABLE — the streaming top-2 gap at that step exceeds the step's
    max-abs delta, so accumulation drift cannot account for it — or when flips
    exceed N//8 (a path that flips >12.5% of steps is suspect regardless of
    ties). Near-tie flips (gap <= delta) are WARN + forensics, not FAIL: the
    duel doc pre-registered exactly this phenomenon (2026-07-15 §packed parity:
    64k greedy parity "diverges probabilistically", 0-flips held only "on this
    seed", "merge gate wording must change accordingly"), and the 2026-07-25
    GH200 forensics confirmed it — a deterministic step-0 flip at stream gap
    0.211 vs delta 1.59 on the random-token prompt's near-degenerate argmax,
    with the top-5 token SET identical between paths at every step and the k4
    drift class (1.3-7.4) statistically identical to the accepted fused-k2b
    class (1.2-8.2). The max-abs delta itself stays RECORDED, never gated.
    """
    ctx = max(cfg.ctx_lens)
    N = cfg.logit_probe
    ids_ = _prompt_ids(tokenizer, ctx, model.device)

    caches = {}
    logits = {}
    for kind in ("streaming", "packed"):
        cache = _fresh_cache(kind, model, k_spec, v_spec)
        cache.attach(model)
        with torch.no_grad():
            out = model(
                ids_,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        caches[kind] = cache
        logits[kind] = out.logits[:, -1, :].float()

    rows = []
    n_flips = 0
    n_hard_flips = 0
    max_abs_overall = 0.0
    for step in range(N):
        # Teacher force: BOTH caches advance on streaming's greedy token, so a
        # diverging packed argmax never lets the two trajectories walk apart.
        next_tok = logits["streaming"].argmax(dim=-1, keepdim=True)
        pos = torch.tensor([ctx + step], device=model.device)
        step_logits = {}
        for kind in ("streaming", "packed"):
            with torch.no_grad():
                out = model(
                    next_tok,
                    past_key_values=caches[kind],
                    use_cache=True,
                    cache_position=pos,
                )
            step_logits[kind] = out.logits[:, -1, :].float()
        logits = step_logits

        diff = (logits["packed"] - logits["streaming"]).abs()
        max_abs = diff.max().item()
        max_abs_overall = max(max_abs_overall, max_abs)
        top2 = logits["streaming"].topk(2, dim=-1).values
        gap = (top2[..., 0] - top2[..., 1]).item()
        flip = (
            logits["packed"].argmax(dim=-1) != logits["streaming"].argmax(dim=-1)
        ).item()
        hard = bool(flip) and gap > max_abs
        n_flips += int(flip)
        n_hard_flips += int(hard)
        rows.append(
            {
                "step": step,
                "max_abs_delta": round(max_abs, 6),
                "stream_top2_gap": round(gap, 6),
                "flip": flip,
                "hard": hard,
            }
        )
        print(
            f"[logit probe] step={step:3d} max_abs_delta={max_abs:.6f} "
            f"stream_top2_gap={gap:.6f} flip={flip}"
            + (
                " HARD (drift-inexplicable)"
                if hard
                else (" (near-tie WARN)" if flip else "")
            ),
            flush=True,
        )

    for kind in ("streaming", "packed"):
        detach = getattr(caches[kind], "detach", None)
        if detach is not None:
            detach()

    import pandas as pd

    print("\n=== logit probe per-step table ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        f"\n[logit probe] arm={cfg.arm} ctx={ctx} N={N} "
        f"flips={n_flips} max_abs_envelope={max_abs_overall:.6f}"
    )
    assert n_hard_flips == 0, (
        f"LOGIT PROBE FAIL: {n_hard_flips} drift-INEXPLICABLE argmax flip(s) over {N} "
        f"teacher-forced steps at ctx={ctx} (max_abs_envelope={max_abs_overall:.6f}) — "
        f"a flip whose streaming top-2 gap exceeds that step's max-abs delta cannot be "
        f"accumulation-order drift; this is a real divergence."
    )
    assert n_flips <= N // 8, (
        f"LOGIT PROBE FAIL: {n_flips} flips over {N} steps (> N//8 = {N // 8}) — even "
        f"near-tie flips at this rate indicate the packed path's drift class has "
        f"changed, not seed luck; investigate before trusting the path."
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
