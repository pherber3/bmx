# bmx

Kill-or-confirm research on LLM tensor compression: matched-budget experiments,
permutation/random controls, honest bit accounting, every result a committed
parquet. Five programs, all closed:

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

**4. Fused kernels (closed, positive).** `PackedStreamingCache` keeps packed
codes resident with chunked dequant-attention at decode (merged to main); the
Phase-3 Triton split-KV decode kernel dequants in-kernel (RTN + the real k2b
recipe with in-kernel low-rank K + RoPE + per-head Hadamard V), uniform paged
layout, GH200-re-verified 2026-07-25 (full CUDA suite green, real-model parity,
fused-path probe, 0 argmax flips over 64 steps at 64k) — **merged to main
2026-07-26**. Docs:
`2026-06-23-kernel-census-results.md`, `2026-06-24-triton-decode-results.md`.

**5. K4 spectral codec (closed, positive — the headline program).**
Corpus-calibrated query-weighted KLT over key coordinates + reverse-waterfill
bit allocation, values rotate+Lloyd @2b, tier-gated int8 decoders. Shipped
recipe `k4_b2.5_dec8tl` (rotated-W calibration, per-layer int8_tl), all
measured on a GH200 across Llama-3.1-8B-Instruct AND Qwen3-8B
(`docs/2026-07-26-gh200-rental-results.md` + four verified appendices):
LongBench macro **40.85 @ 3.081 mean bits** (+0.48 over the strongest
TurboQuant baseline at −0.125 bits; +0.13 over its own fp32 decoders at
−0.72 bits), NIAH parity with fp16 at 4k–128k (5-seed), measured int8
decoder cost 0.55–0.90% vs a 5% bar on both models, 128k resident memory
50.5 GiB vs fp16's 63.3. Side findings with their own legs: calibration
needs only ~2k tokens of general text (gate passes from one cache, both
models); packs can be synthesized from trigram count tables alone (D < 0.10,
both models, ladder self-terminates at order 3); the gpt2-scale
"token-marginal" calibration claim reverses at 8B (word order matters);
TurboQuant-family arms collapse on Qwen at 32k while K4 holds (mechanism
open). Full theory + declination ledger: `docs/2026-07-25-k4-paper-shelf.md`.

Remaining open items are engineering: a wider batched-128k co-residency sweep,
a needle-reseeding NIAH harness knob, the Qwen TurboQuant-collapse mechanism
probe, and a fused spectral decode kernel (gated on a deployment-latency claim
the science does not need).

## Quickstart

    uv sync
    uv run pytest -q                      # 651 passed, 17 skipped, 1 xfailed (intentional)
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
- `tests/` — 600+ tests across 63 files; agents: see `CLAUDE.md` for
  conventions and pitfalls

## NVIDIA VM workflow (GPU-authoritative numbers)

1. Push; on the VM: `git clone <repo> && cd bmx && scripts/vm_setup.sh`
2. Run the experiment (Nsight wrapper: `scripts/nsight_b1.sh`)
3. `git add results/ && git commit && git push` — metrics come home as parquet
