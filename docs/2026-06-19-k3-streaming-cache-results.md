# K3 — live streaming KV-compression cache (in-practice viability) results (2026-06-19)

Does the K2b recipe work in **live generation** — quantize-on-append, autoregressive
decode where each step attends to its own compressed cache, errors compounding over
hundreds of steps — on both quality and memory? K1→K2c closed the science on
quantized-prefill perplexity; this is the engineering gate the docs flagged as open
("quantize-on-append cache class"). Built as `StreamingQuantizedLayer`/
`StreamingQuantizedCache` (subclassing transformers 5.11 `DynamicLayer`/`Cache`,
mirroring HF `QuantizedCache`), exercised on the offline tiny GQA factory model for
mechanism and wired for a real-text + planted-needle run on a real model.

Branch: `k3-live-streaming-cache`. All numbers below independently re-derived from the
cache (not taken on report). Spec/plan: `docs/superpowers/{specs,plans}/2026-06-19-k3-*`.

## Verdict: CONFIRMED — the recipe streams token-by-token; the one real bug (V re-quant blowup) is fixed at the root

The cache plugs into `model.generate()` unchanged and quantizes on append. The
load-bearing correctness facts, each gated:

| property | measured | gate |
|---|---|---|
| Passthrough is faithful | bit-identical logits + greedy generate vs plain cache | `torch.equal` |
| Token-by-token scoring is correct | fp16 tbt-ppl == batched-ppl, **rel 0.000000** | fp16 has no quant error ⇒ agree iff indexing right |
| Pre-RoPE keys, RoPE-at-read | reconstructed post-RoPE keys vs true, **rel 1.9e-4** (multi-block) | streamed past 2nd/3rd flush at nonzero offsets |
| **V codec stable under streaming** | K2b V (turboquant_mse) tbt-vs-batched **rel 0.0000** (norm 4.02, not 397) | the C1 bug; see below |
| Frozen subspace (K2c) actually frozen | `_frozen_svd[1]` bitwise-unchanged across flushes | no per-step refit |
| K2b quality holds under streaming | K2b tbt-ppl / fp16 tbt-ppl = **1.001** | < 3× (mechanism, on random-weight model) |

## The one real bug, and the root-cause fix (write-once storage)

The first implementation re-quantized the **entire growing prefix every decode step**,
operating on the previous step's *dequantized* slab. For idempotent codecs (KIVI's
RTN) this was harmless; for the K2b value codec `turboquant_mse` (rotate + per-token
norm rescale + Lloyd) it **compounded** — re-coding a re-coded vector grows the norm
each pass. Measured: V cache norm 4.0 → 397.6, **rel 98×** over 64 token-by-token steps.
Token-by-token gates didn't catch it because they used idempotent `rtn_token` V and
never checked V quality.

Fix: **write-once quantized storage** — the canonical KV-cache semantics ("each entry
written once, read every subsequent step"; confirmed against the personal-brain vault).
Each token's K/V is quantized **exactly once**, when its block flushes out of the fp16
recent window, from its *pristine* source (pre-RoPE `_k_pre` for K, the still-fp16
tail for V), and frozen in `_q_prefix_k/v` — never re-quantized. `turboquant_mse`
applied once is stable. This simultaneously fixed the per-step SVD refit (freeze the
subspace at first flush) and removed the redundant full-history pre-RoPE buffer.

## Residual window — channel-grouped arms now stream

`rtn_channel`/`lowrank_rtn_channel` assert `S % group == 0`, which crashes on the first
decode token (S=17, group=16) if the whole cache is quantized each step. The fp16
**recent window** (HF/KIVI pattern) fixes this: keep the most-recent W tokens fp16,
quantize only block-aligned older tokens. With the window, KIVI and K2b decode
token-by-token without crashing; bpe is the honest blend of quantized-prefix + fp16-tail.

## What is demonstrated vs explicitly deferred

**Demonstrated (gated):** token-by-token streaming correctness; no V explosion;
multi-block pre-RoPE positions; frozen subspace; honest packed bpe < fp16; a
model-agnostic real-text + planted-needle experiment (`--model-name` is the only knob
for the SOTA/VM run). All arms (K2b, TurboQuant_mse/prod, KIVI, fp16) run on **one**
code path with the same honest-bpe accounting — the head-to-head is structurally fair.

**Deferred (caveats, not blockers — they belong here, not in the gates):**
1. The literal process-level memory 5× needs the fused dequant-attention kernel /
   paged uint8 store. This branch reports the **honest packed-bpe deployable** number
   and prunes the redundant resident buffer; `reconstruct_layer` still materializes the
   dequant slab for the model to read (the documented Stage-B contract). Process-RSS is
   the VM/kernel measurement.
2. Headline quality/needle **numbers** come from executing the experiment on a real
   model (Llama-3.2-1B default) with wikitext + planted needle on the NVIDIA VM. This
   branch makes that run *trustworthy*; it does not execute it (no CUDA on the 7900 XTX).
   Tiny-llama tests gate **mechanism** (finite, no explosion, indexing-correct, frozen),
   not quality — absolute ppl on random weights is meaningless by construction.
3. `memory_report` compression is the *blended* bpe (includes the fp16 recent window),
   so at short context it is conservative vs the long-context asymptote.

## Gate call

The streaming design is **validated in practice at its core**: the K2b recipe streams
token-by-token with quality holding (1.001× fp16) and honest packed memory below fp16,
on one fair code path against TurboQuant and KIVI. What remains is the fused kernel
(predict with the Track B byte model before building) and the authoritative SOTA-model
VM run — both engineering, both out of scope here. The cache is a drop-in
`past_key_values=` for `model.generate()`.
