# Multi-architecture KV-compression extension — design (2026-07-01)

**Status: SCOPED, decision-deferred.** This spec exists so the work is ready to dispatch; the ship/delay call (does the paper wait for a second architecture, or ship Llama-only with this as follow-on?) is made *after* the Llama authoritative runs land (publication plan Tasks 9–10). Chosen path (user, 2026-07-01): "scope it now, decide later," targets **Gemma + Qwen3**.

## Why (the paper-strengthening thesis)

The KV program's two load-bearing findings — **"bits belong to K"** (asymmetric key/value bit allocation) and **"quantize pre-RoPE keys"** (the frozen low-rank subspace is a position-independent model property only *before* RoPE) — are validated on **Llama-family GQA only**: uniform global attention, full RoPE. A reviewer's fair question is: *are these Llama artifacts, or general properties of trained attention?* Answering it on a second (and third) architecture converts the paper from "a recipe for Llama" to "a characterization of KV structure across modern attention." That is a first-class generalization contribution, not a courtesy check.

**This is genuinely falsifiable, both ways** (kill-or-confirm, per the repo's prime directive):
- If the recipe holds on Gemma/Qwen → strong generalization claim.
- If it breaks (e.g. partial RoPE destroys the pre-RoPE subspace argument; sliding windows break the frozen-prefix assumption) → an honest, publishable **boundary finding** ("the recipe is specific to global-attention/full-RoPE stacks; here is why").

Either outcome is a result. Do not tune to force a positive.

## Correct scoping (corrects the earlier "Qwen has no KV cache" error)

Per vLLM's TurboQuant scoping (the mirror method's own boundary): the recipe applies to **any standard full-attention layer with a KV cache**. There is no "Qwen-only cache format." The only special case is **hybrid** models (attention + Mamba/linear-attention layers): the linear-attention layers have no standard KV cache and are **skipped** (boundary protection disabled, full-attention layers must be identifiable) — exactly as TurboQuant handles them. So:

- **Qwen3 standard decoder** → recipe applies to every layer, same as Llama.
- **Qwen3 hybrid variant** → recipe applies to the full-attention layers; linear-attention layers deferred/skipped. Report per-layer coverage honestly.

## Falsifiable hypotheses

- **A1 (pre-RoPE survives partial RoPE — Gemma).** Gemma applies RoPE to only a fraction of head dims. Hypothesis: the "quantize pre-RoPE keys, rotate at read" trick still wins on the rotated subset, and the un-rotated dims are quantized directly (no pre/post distinction). Falsified if the pre-RoPE low-rank subspace on the rotated dims drifts with position the way full-RoPE post-RoPE keys do (K2c's failure mode), i.e. partial RoPE contaminates the frozen subspace.
- **A2 (frozen subspace survives sliding-window — Gemma).** Gemma alternates sliding-window and full-attention layers. Hypothesis: the frozen-prefill subspace argument holds per-layer for the *full-attention* layers; for *sliding-window* layers the "cache" is a bounded window so the compression target shrinks but the codec still applies to the window contents. Falsified if the sliding-window layers' key structure is not low-rank (the window is too short for the subspace to be stable).
- **A3 ("bits belong to K" is architecture-general — Qwen3 + Gemma).** Hypothesis: the K-is-2×-more-sensitive-than-V asymmetry (K2b sensitivity ablation) reproduces on both. Falsified if a symmetric or V-heavy allocation matches/beats the K-heavy one at matched bits on either model.
- **A4 (TurboQuant-parity holds — both).** Hypothesis: at matched compression, our recipe still beats `turboquant_mse` on NIAH + LongBench for Gemma and Qwen3 (TurboQuant used Ministral as their 2nd model; Qwen3 is a stronger, actively-maintained target). Falsified if turboquant_mse matches/wins on either.

## What has to change (engineering surface)

The science hypotheses above are gated on cache-class work. Grounding: `src/bmx/cache/streaming.py` `StreamingQuantizedLayer`/`Cache` assume (a) one global-attention KV per layer, (b) full RoPE via `rope.py` cos/sin from config, (c) a frozen pre-RoPE subspace fit once at prefill.

1. **Partial RoPE (`rope.py` + `streaming.py`).** `rope_cos_sin` derives full-head cos/sin from config. Gemma applies RoPE to a dim slice. Need: read the partial-RoPE dim split from config (`LlamaRotaryEmbedding(config=...)` inherits scaling — the analog for Gemma must inherit its partial spec, never re-derive), split keys into rotated/un-rotated, apply the pre-RoPE-freeze trick to the rotated slice only, quantize the un-rotated slice directly.
2. **Sliding-window / alternating (`streaming.py` + `hf_compat.py`).** The cache must know, per layer, whether it is sliding or full attention, and bound the frozen prefix to the window for sliding layers. `hf_compat.py` model introspection (`resolve_*`) extends to report the per-layer attention type + window size.
3. **Hybrid-layer skip (Qwen3 hybrid, if targeted).** Identify full-attention layers; the cache applies the codec only to them and passes linear-attention layers through uncompressed. Report per-layer coverage (fraction of KV bytes actually compressed).
4. **Introspection resolvers (`hf_compat.py`).** Head dims, GQA groups, RoPE spec, attention-type-per-layer, window sizes — all read from config, never hardcoded. The `factories.py` tiny-model set gains a partial-RoPE + sliding-window toy model for offline tests (never download in tests).

## Metrics / gates (reuse the existing instruments)

Same rigor as the Llama program — no new metrics:
- Per-layer break-even margins + kurtosis pre/post (partial-)rotation (K1 census, pointed at Gemma/Qwen caches) → decides which arms are live per architecture.
- Logit distortion vs real queries + quantized-prefill ppl (K2/K2b) at matched bits.
- Frozen-vs-oracle subspace ratio + drift (K2c) — the direct test of A1/A2.
- NIAH recall + LongBench (the paper's spine) at matched compression vs turboquant_mse (A4).

**Kill criterion:** if A1 AND A2 both falsify on Gemma (partial RoPE + sliding windows both break the frozen-subspace argument) AND A3 falsifies, the honest finding is "the recipe is specific to global-attention/full-RoPE stacks." That is reported as a boundary, and the paper ships Llama-only with this documented — not buried.

## Sequencing / cost

- **Prerequisite:** the Llama authoritative runs (publication plan Tasks 9–10) land first — they set the bar this extension is measured against, and the ship/delay decision reads their results.
- **Order:** Qwen3-standard first (lowest cache-class delta — it's Llama-like global attention + full RoPE; mostly an introspection + tiny-model-fixture task, tests A3/A4 cheaply), THEN Gemma (the hard one — partial RoPE + sliding windows, tests A1/A2). This front-loads the cheap generalization signal before committing to the Gemma engineering.
- **Cost:** Qwen3-standard ≈ a few tasks (introspection + fixtures + reuse the whole pipeline). Gemma ≈ a real engineering leg (partial-RoPE + sliding-window cache-class changes) + full census/K2/K2c/task re-runs. Both need VM for the authoritative numbers.
- **Decision point:** after Qwen3 lands, re-assess whether Gemma is in-paper or follow-on based on (a) whether Qwen3 already gives a convincing generalization story and (b) timeline.

## Relation to prior results (so nobody re-litigates)

This is not a re-run of the Llama program on new tensors for its own sake — it tests whether the *mechanism* (K-sensitivity asymmetry; pre-RoPE subspace stability) is architecture-general or Llama-specific, using the same instruments. It also matches TurboQuant's own multi-model presentation (they reported Llama + Ministral); Qwen3 is a stronger 2nd model, Gemma a genuinely harder 3rd that probes the method's boundary.
