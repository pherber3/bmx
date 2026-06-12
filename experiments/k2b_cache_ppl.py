"""K2b: Quantized-prefill perplexity sweep for KV-cache codec arms.

Evaluates five arm combinations at bits ∈ {2, 3} against an fp16 baseline by
prefilling N tokens, quantizing the full KV cache, then measuring teacher-forced
continuation perplexity.  Produces a parquet under results/k2b_cache_ppl/.

Usage
-----
    uv run python experiments/k2b_cache_ppl.py --model-name gpt2 \\
        --n-prefill 768 --n-eval 256

    uv run python experiments/k2b_cache_ppl.py --model-name gpt2 \\
        --n-prefill 768 --n-eval 256 --contexts 256 512

    uv run python experiments/k2b_cache_ppl.py \\
        --model-name meta-llama/Llama-3.1-8B \\
        --n-prefill 1792 --n-eval 256 --contexts 512 1024
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import torch
import tyro
from transformers import AutoModelForCausalLM

from bmx.artifacts import create_run, write_metrics
from bmx.cache.ppl_eval import CacheCodecSpec, quantized_prefill_ppl
from bmx.eval.layer_swap import load_eval_tokens


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Config:
    model_name: str = "gpt2"
    n_prefill: int = 768
    n_eval: int = 256
    bits: tuple[int, ...] = (2, 3)
    rank: int = 32
    group: int = 64
    seed: int = 0
    contexts: tuple[int, ...] = ()  # optional extra n_prefill values
    # sensitivity ablation: K-only / V-only / asymmetric-bits rows for the
    # finalist combo instead of the standard arms table
    ablation: bool = False


# ---------------------------------------------------------------------------
# Arms table builder
# Each entry: (arm_k_label, arm_v_label, k_spec_fn, v_spec_fn, is_baseline)
# k_spec_fn / v_spec_fn are callables (cfg, bits) -> CacheCodecSpec.
# The fp16 baseline row uses bits=0 (dummy) and is evaluated once.
# ---------------------------------------------------------------------------


def _has_rope(model: torch.nn.Module) -> bool:
    """Return True if the model's config supports rotary embeddings."""
    return hasattr(model.config, "rope_parameters")


def _build_arms_table(cfg: Config, model: torch.nn.Module):
    """Return the arms table with pre_rope resolved for this model."""
    pre_rope = _has_rope(model)

    def fp16(bits):
        return CacheCodecSpec(arm="fp16")

    def rtn_channel(bits):
        return CacheCodecSpec(
            arm="rtn_channel", bits=bits, group=cfg.group, seed=cfg.seed
        )

    def lowrank_rtn_channel(bits):
        return CacheCodecSpec(
            arm="lowrank_rtn_channel",
            bits=bits,
            rank=cfg.rank,
            group=cfg.group,
            seed=cfg.seed,
            pre_rope=pre_rope,
        )

    def turboquant_mse(bits):
        return CacheCodecSpec(
            arm="turboquant_mse", bits=bits, group=cfg.group, seed=cfg.seed
        )

    def turboquant_prod(bits):
        return CacheCodecSpec(
            arm="turboquant_prod", bits=bits, group=cfg.group, seed=cfg.seed
        )

    return [
        ("fp16", "fp16", fp16, fp16, True),
        ("rtn_channel", "rtn_channel", rtn_channel, rtn_channel, False),
        (
            "lowrank_rtn_channel",
            "turboquant_mse",
            lowrank_rtn_channel,
            turboquant_mse,
            False,
        ),
        ("turboquant_mse", "turboquant_mse", turboquant_mse, turboquant_mse, False),
        ("turboquant_prod", "turboquant_prod", turboquant_prod, turboquant_prod, False),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_model(model_name: str) -> torch.nn.Module:
    """Load model in fp32 for gpt2, bf16 otherwise."""
    if "gpt2" in model_name.lower():
        dtype = torch.float32
    else:
        dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.eval()
    return model


def _short_label(arm_k: str, arm_v: str) -> str:
    if arm_k == arm_v:
        return arm_k
    return f"{arm_k}/{arm_v}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Config) -> None:
    run = create_run("k2b_cache_ppl", cfg)
    print(f"Run dir: {run}")

    # All n_prefill values to evaluate: primary + any extras
    all_contexts: list[int] = [cfg.n_prefill] + [c for c in cfg.contexts]
    n_tokens_needed = max(all_contexts) + cfg.n_eval

    print(f"Loading model: {cfg.model_name}")
    model = _load_model(cfg.model_name)

    arms_table = _build_arms_table(cfg, model)
    has_rope = _has_rope(model)
    print(
        f"RoPE detected: {has_rope}  (lowrank pre_rope={'True' if has_rope else 'False (no RoPE in model)'})"
    )

    print(f"Loading {n_tokens_needed} eval tokens...")
    tokens_1d = load_eval_tokens(cfg.model_name, n_tokens=n_tokens_needed)
    # ppl_eval expects (1, N)
    input_ids = tokens_1d.unsqueeze(0)  # (1, n_tokens_needed)

    rows: list[dict] = []

    # fp16 baselines keyed by n_prefill
    fp16_ppl: dict[int, float] = {}

    # Evaluate each n_prefill context separately
    for n_pref in all_contexts:
        n_total = n_pref + cfg.n_eval
        ids = input_ids[:, :n_total]  # (1, n_pref + n_eval)

        print(f"\n--- n_prefill={n_pref} ---")

        # ------------------------------------------------------------------
        # 1.  fp16 baseline (bits-independent, evaluated once per context)
        # ------------------------------------------------------------------
        arm_k_lbl, arm_v_lbl, fn_k, fn_v, _ = arms_table[0]
        k_spec = fn_k(0)
        v_spec = fn_v(0)
        result = quantized_prefill_ppl(model, ids, n_pref, k_spec, v_spec)
        ppl_base = result["ppl"]
        fp16_ppl[n_pref] = ppl_base
        row = dict(
            model=cfg.model_name,
            arm_k=arm_k_lbl,
            arm_v=arm_v_lbl,
            bits=16,
            bits_k=16,
            bits_v=16,
            rank=0,
            pre_rope=False,
            n_prefill=n_pref,
            bpe_k=result["bpe_k"],
            bpe_v=result["bpe_v"],
            ppl=ppl_base,
            dppl_pct=0.0,
        )
        rows.append(row)
        print(
            f"  fp16 baseline: ppl={ppl_base:.4f}  "
            f"bpe_k={result['bpe_k']:.2f}  bpe_v={result['bpe_v']:.2f}"
        )

        # ------------------------------------------------------------------
        # 2.  Quantized arms: standard table × bits, or the sensitivity
        #     ablation (K-only / V-only / asymmetric bits on the finalists)
        # ------------------------------------------------------------------
        lowrank_fn = arms_table[2][2]
        tq_fn = arms_table[2][3]
        fp16_spec = CacheCodecSpec(arm="fp16")
        if cfg.ablation:
            b_lo, b_hi = min(cfg.bits), max(cfg.bits)
            entries = [
                ("lowrank_rtn_channel", "fp16", lowrank_fn(b_hi), fp16_spec, b_hi, 16),
                ("lowrank_rtn_channel", "fp16", lowrank_fn(b_lo), fp16_spec, b_lo, 16),
                ("fp16", "turboquant_mse", fp16_spec, tq_fn(b_hi), 16, b_hi),
                ("fp16", "turboquant_mse", fp16_spec, tq_fn(b_lo), 16, b_lo),
                (
                    "lowrank_rtn_channel",
                    "turboquant_mse",
                    lowrank_fn(b_hi),
                    tq_fn(b_lo),
                    b_hi,
                    b_lo,
                ),
                (
                    "lowrank_rtn_channel",
                    "turboquant_mse",
                    lowrank_fn(b_lo),
                    tq_fn(b_hi),
                    b_lo,
                    b_hi,
                ),
            ]
        else:
            entries = [
                (ak, av, fn_k(b), fn_v(b), b, b)
                for ak, av, fn_k, fn_v, _ in arms_table[1:]
                for b in sorted(cfg.bits)
            ]

        for arm_k_lbl, arm_v_lbl, k_spec, v_spec, bits_k, bits_v in entries:
            result = quantized_prefill_ppl(model, ids, n_pref, k_spec, v_spec)
            ppl_q = result["ppl"]
            dppl = 100.0 * (ppl_q / ppl_base - 1.0)
            row = dict(
                model=cfg.model_name,
                arm_k=arm_k_lbl,
                arm_v=arm_v_lbl,
                bits=bits_k if bits_k == bits_v else -1,
                bits_k=bits_k,
                bits_v=bits_v,
                rank=cfg.rank if arm_k_lbl == "lowrank_rtn_channel" else 0,
                pre_rope=(arm_k_lbl == "lowrank_rtn_channel" and has_rope),
                n_prefill=n_pref,
                bpe_k=result["bpe_k"],
                bpe_v=result["bpe_v"],
                ppl=ppl_q,
                dppl_pct=dppl,
            )
            rows.append(row)
            print(
                f"  {arm_k_lbl}/{arm_v_lbl} k@{bits_k}b v@{bits_v}b: "
                f"ppl={ppl_q:.4f}  dppl={dppl:+.2f}%  "
                f"bpe_k={result['bpe_k']:.2f}  bpe_v={result['bpe_v']:.2f}"
            )

    # ------------------------------------------------------------------
    # Write parquet
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows)
    write_metrics(run, df)
    print(f"\nParquet written to {run}/metrics.parquet")

    # ------------------------------------------------------------------
    # End summary: sorted by dppl_pct per (n_prefill, bits)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY — sorted by dppl_pct per (n_prefill, bits)")
    print("=" * 72)

    quant_df = df[df.bits != 16].copy()
    for n_pref in sorted(quant_df.n_prefill.unique()):
        pbase = fp16_ppl[n_pref]
        print(f"\nn_prefill={n_pref}  fp16_baseline_ppl={pbase:.4f}")
        for bits in sorted(quant_df.bits.unique()):
            sub = quant_df[
                (quant_df.n_prefill == n_pref) & (quant_df.bits == bits)
            ].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("dppl_pct")
            print(f"\n  bits={bits}")
            print(
                f"  {'arm_k/arm_v':<30} {'bpe_k':>6} {'bpe_v':>6} "
                f"{'ppl':>8} {'dppl%':>8}"
            )
            print("  " + "-" * 62)
            for _, r in sub.iterrows():
                lbl = _short_label(r.arm_k, r.arm_v)
                if r.bits == -1:  # ablation rows: asymmetric bits
                    lbl += f" [k@{int(r.bits_k)}|v@{int(r.bits_v)}]"
                print(
                    f"  {lbl:<30} {r.bpe_k:>6.2f} {r.bpe_v:>6.2f} "
                    f"{r.ppl:>8.4f} {r.dppl_pct:>+8.2f}%"
                )

    print(f"\nTotal rows: {len(df)}")
    print(f"Run dir: {run}")


if __name__ == "__main__":
    main(tyro.cli(Config))
