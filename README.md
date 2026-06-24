# bmx

Kill-or-confirm research on LLM tensor compression: matched-budget experiments,
permutation/random controls, honest bit accounting, every result a committed
parquet. Two completed programs:

**1. Weights (closed, negative-with-a-law).** Started as a test of
Bhattacharya–Mesner (hypermatrix) decomposition for bandwidth-amplified decode;
the diag-template prior does not describe trained weights
(`docs/2026-06-10-h100-session-results.md`). Generalizing the failure produced a
break-even inequality — side-information costing Δb bits/weight pays iff it
removes energy fraction ε > 1 − 4^(−Δb) — and measuring it from GPT-2 to
Llama-70B shows transform weights sit *on* the break-even line at every width
(stable rank grows with width at the canceling rate). Lossy structural weight
compression is scale-invariantly marginal; the only payers are table-like
objects (position embeddings, MoE routers, layer-0 rogue-channel readers — all
axis-aligned, i.e. absorbed free by per-channel scales).
Docs: `2026-06-11-lrs-results.md` (L+S negative + theory postmortem),
`2026-06-11-frontier-breakeven.md` (the law).

**2. KV cache (closed, positive).** The same instrument scores cache
activations at +0.5–2.5 bits of margin where weights scored ≈0. End state, all
measured end-to-end on Llama-3.1-8B: **keys pre-RoPE low-rank(r≈16–32) +
per-channel residual @3b, values rotate+Lloyd @2b ⇒ ~3.0 bits/entry,
+0.5% perplexity, 5.3× KV memory vs fp16**; bits belong to K (2× more
sensitive than V); RoPE costs ~1–1.5 bits of key compressibility (store keys
pre-RoPE, rotate at read); prefill-frozen subspaces generalize to later tokens
(0.94 of oracle, drift-flat), so the recipe streams. TurboQuant's bounds
replicate exactly on real caches but worst-case-optimal coding concedes 2–3×
to structure-aware coding on keys; unbiased coding is dominated everywhere.
Docs, in order: `2026-06-11-kv-research-plan.md` → `2026-06-11-k1-census-results.md`
→ `2026-06-12-k2-arms-results.md` → `2026-06-12-k2b-ppl-results.md` →
`2026-06-12-k2c-results.md`. Headline figure:
`results/k2_cache_arms/k2_headline.png`.

**3. Streaming cache (K3, closed positive).** The quantize-on-append cache class is
built and validated: `StreamingQuantized{Layer,Cache}` streams token-by-token under
real `generate()` — write-once quantized storage (each token quantized once from its
pristine source; this fixed a real bug where the value codec's norm exploded 98× under
naive re-quantization), frozen pre-RoPE subspace, fp16 residual window. Quality holds
(1.001× fp16 on token-by-token ppl), packed bpe < fp16, all arms (K2b/TurboQuant/KIVI/
fp16) on one fair code path — `docs/2026-06-19-k3-streaming-cache-results.md`.

Remaining work is engineering, not science: the fused dequant-attention kernel (for the
literal process-RSS win; the Track B byte model in `src/bmx/bench/` predicts kernel wins
before any CUDA is written), the authoritative SOTA-model VM run, and a 32k-context
re-check.

## Quickstart

    uv sync
    uv run pytest -q                      # 243 passed, 1 xfailed (intentional)
    uv run python experiments/k1_cache_census.py --help   # tyro CLIs everywhere

Experiments run on CPU except where noted (this repo was developed against an
AMD GPU; NVIDIA-authoritative numbers come from a rented VM — see below).
Raw caches (`results/cache/`, gitignored) regenerate via
`experiments/collect_cache.py`.

## Layout

- `src/bmx/` — the framework: `decomp/` (registered methods incl. the BM-RALS
  solver, which beats the BM-ALS paper's own solver by 3–10 orders of
  magnitude), `cache/` (KV collection, codecs, RoPE, distortion metrics,
  quantized-prefill ppl eval), `quant/` (rotations, RTN, break-even
  instrument, stats), `stacks/`, `bench/`, `sweep.py`, `artifacts.py`
- `experiments/` — thin tyro scripts, one per research item; `plots/` read
  parquet, never refit
- `results/` — committed metrics + figures (config + env + git SHA per run)
- `docs/` — results docs (the program record), research plans,
  `superpowers/` (implementation plans/specs)
- `scripts/` — NVIDIA-VM setup + Nsight wrappers, SageMath fixture exporter
- `tests/` — 129 tests; agents: see `CLAUDE.md` for conventions and pitfalls

## NVIDIA VM workflow (GPU-authoritative numbers)

1. Push; on the VM: `git clone <repo> && cd bmx && scripts/vm_setup.sh`
2. Run the experiment (Nsight wrapper: `scripts/nsight_b1.sh`)
3. `git add results/ && git commit && git push` — metrics come home as parquet
