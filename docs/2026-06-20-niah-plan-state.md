# NIAH retrieval metric — state + VM handoff (2026-06-20)

The retrieval half of next-direction #0 (a coding/long-context **task** metric, not just
perplexity). Harness is **built, reviewed, and merged-ready**; what remains is the
authoritative **VM run** on a real model — no new code, just `--model-name` + a GPU.

Spec: `docs/superpowers/specs/2026-06-20-niah-retrieval-metric-design.md`
Plan + per-task ledger: `docs/superpowers/plans/2026-06-20-niah-retrieval-metric.md`,
`.superpowers/sdd/progress.md`.

## What was built (one paragraph)

A needle-in-a-haystack **recall** metric following the Fu et al. / TurboQuant setup
(single needle, length×depth sweep, ROUGE-1 recall), running the existing compression
arms through the **same `StreamingQuantizedCache` path** the K3 ppl sweep uses. New code:
`src/bmx/cache/haystack.py` (synthetic CI filler + real Paul Graham essays),
`src/bmx/cache/niah.py` (argmax recall proxy for the offline mechanism gate; `rouge1_recall`
+ PG-essay prompt builder + `niah_recall_generate` for the headline),
`experiments/k3_niah.py` (sweep arms × lengths × depths → parquet, honest measured
compression per arm), `experiments/plots/plot_k3_niah.py` (recall-vs-length +
length×depth heatmap). All arms route through `_spec_pair` — structurally fair.

## The VM run (the open work — engineering, not science)

```bash
cd /d/Projects/bmx   # on the NVIDIA VM (no CUDA on the 7900 XTX dev box)
uv run python experiments/k3_niah.py \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --lengths 4096 8192 16384 32768 \
  --depths 0.1 0.3 0.5 0.7 0.9
# extensible toward 104k via --lengths; validate on the 8B first, then a 27-31B SOTA model.
```

**No git clone required on the VM.** The Paul Graham haystack now self-downloads from the
HuggingFace dataset `sgoel9/paul_graham_essays` via `haystack.load_pg_corpus()` (lazy
`load_dataset`, 215 essays, `text` column). Same self-download path as LongBench
(`THUDM/LongBench`) — both metrics need only `uv` deps + HF dataset access, no repo clones.
Figures: `experiments/plots/plot_k3_niah.py::make_figures(df, out_dir)` over the emitted
parquet.

## Headroom guard — apply at analysis time (do NOT skip)

The metric is only meaningful where **fp16 recall is high but compressed arms *can* diverge.**
When reading the VM parquet:
- If **fp16 recall is at floor** at some length (the base model itself can't do the task),
  that length is **non-discriminating** — flag it, don't report a vacuous tie.
- If **every arm is at ceiling** (task too easy at that length), likewise non-discriminating.
- The discriminating signal lives where fp16 ≈ 10 and the 2-bit arms (KIVI, TurboQuant) start
  to drop while K2b holds — that's where compression plausibly breaks retrieval. Expect the
  gap to open as length grows (older tokens flushed to quantized storage).

## Two real bugs the kill-or-confirm review loop caught (don't reintroduce)

The plan's own example code carried two correctness bugs that green per-task schema tests did
NOT catch — both fixed and verified:
1. **Double-prefill** (`niah_recall_generate`): passed the FULL prompt to `model.generate()`
   while the cache already held `[:n_prefill]`, re-processing the prefill with wrong positions.
   Fixed to feed the continuation only (`prompt_ids[:, n_prefill:]`), decode slice
   `out[0, L-n_prefill:]` — matches the proven `needle.py:21-22` pattern.
2. **nan compression** (`_compression_for`): read `bits_per_entry()` from a cache that had
   never run a forward pass → bpe `nan` → compression `1.0` for **every** arm. Fixed with a
   seeded calibration prefill before reading bpe (verified kivi@48 → 1.371×, fp16@48 → 1.0×).

Lesson (same as K3): the headline numbers must be **independently re-derived from the
artifact**, never taken on a green schema test.

## Deferred (not blockers)

- **Code-gen pass@1 under compression** — the *second half* of next-direction #0; its own
  spec/plan later. This doc is retrieval only.
- Vault (personal-brain) was offline this session (wiki-mcp not connected); conventions were
  locked from the authoritative local Fu et al. source instead — re-check the vault if it
  helps when it's back.
- Minor nits recorded in `.superpowers/sdd/progress.md` (test-strength gaps, undocumented
  fallbacks) — for the writeup, not blockers.
