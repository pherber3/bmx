# Tensor compression for LLM inference — where I went, what I found, what actually works

One-page brief. Context: a conversation with a Cloudflare engineer about systems-side
inference efficiency sent me down a tensor-compression path. Here's the honest arc
and the leads worth pursuing.

---

## The bet I tested

Could **hypermatrix (Bhattacharya–Mesner) algebra** give a *structural* compression of
LLM weights — not quantization, but exploiting shared structure across heads / experts
for a bandwidth win in memory-bound decode? Concretely: if a stack of weight matrices is
"a few shared templates under per-slice diagonal gains," you read `ℓ/h` of the weight
bytes per token. Bytes are latency in decode, so that's a real speedup mechanism.

## What I did — and the result

Built a small research framework, ported the solver properly (it beats the prior
published solver on its own benchmark by 3–10 orders of magnitude — so the method, not
the math, was always on trial), and ran a **kill-or-confirm program on one H100 for ~$17**.

**The structural bet is wrong, and I can show *why* with a clean measured control:**

- **Attention heads share a subspace, not templates.** A permutation-null control
  (destroy cross-slice alignment, preserve per-slice spectra) isolates it: Tucker keeps
  a real 0.06–0.10 reconstruction-error advantage on real weights that vanishes on the
  null — genuine structure, *subspace-shaped*. The hypermatrix prior shows **zero**
  real-vs-null gap. Structure exists; it's just not the shape I bet on.
- **MoE experts are mostly private content.** Census across OLMoE / Qwen1.5-MoE /
  DeepSeek-V2-Lite: experts are *orthogonal* as weight vectors (zero mergeable pairs)
  but share ~10 global second-moment modes — agreement on *which directions matter*, not
  on the functions computed. Too thin to compress; at the streaming-relevant budget every
  method (not just mine) sits at >0.9 error.

The selling point isn't the idea — it's the discipline: a falsifiable hypothesis, killed
cheaply, reported with the measurement that makes the "no" trustworthy.

## The transferable result (decoupled from the bet failing)

Confirmed with **Nsight DRAM byte counters on H100**: in memory-bound decode,
**bytes-read-from-HBM is latency.** A kernel reading `ℓ/h` of the weight bytes gets a
near-proportional speedup at batch 1 (measured 3–19×), decaying toward 1× as batch grows
and the dense path amortizes its own reads. The byte model held to the counter. The gap
to ideal is *kernel utilization* (skinny GEMM ≈ 10% of peak bandwidth) — i.e. the win is
real but needs a **fused kernel** to collect, which is exactly the design pattern below.

**This is the same lesson Cloudflare's own Unweight reached from the other direction.**

## What actually moves the needle (mapped to the bottleneck it attacks)

1. **Weight quantization — the workhorse.** AWQ (salient-channel scaling) as production
   default; GPTQ for domain-calibrated; **W4A8 on TensorRT-LLM** as the near-lossless
   upgrade. Genuinely-open research edge (I checked the theory literature): the
   rotation/Gaussianization theory under QuIP#/QuaRot/NestQuant is *settled*; what's open
   is **unbiased quantized matvecs and depth-wise bias accumulation** — a clean problem,
   not hand-waving.

2. **KV cache — the decode memory wall.** Caps batch size and context, not weights.
   Levers: GQA/MLA (shrink per-token footprint), INT8/FP8 KV quant (composes with weight
   quant), PagedAttention (fragmentation). Often the highest-ROI lever at a serving
   provider: it buys batch size → throughput → $/token directly.

3. **MoE expert streaming — the systems sibling of my failed idea, and it's real.** PCIe
   is ~100× slower than HBM, so offload lives or dies on prefetch hit-rate and per-miss
   ms. Compression composes: INT4 experts are 4× smaller → cheaper misses → offload
   viable at lower hit-rates. **Ties directly to Cloudflare Omni** (whole-model
   unified-memory paging, 13 models / 400% over-commit on one GPU) — same HBM↔DRAM bus
   arithmetic at a coarser granularity.

4. **Disaggregated prefill/decode — the architectural lever.** Prefill is compute-bound,
   decode bandwidth-bound; splitting onto matched hardware pools gave Splitwise ~2.35× RPS
   at equal cost. Bottleneck-matched hardware, the cleanest systems win of the four.

## The Cloudflare connection (why this conversation, specifically)

Cloudflare Research's **Unweight** (April 2026) is the same target I was chasing, done
right: lossless, bit-exact compression that Huffman-codes the redundant BF16 *exponent*
bytes (~30% MLP compression) and — the part that matters — **decompresses inside a fused
warp-specialized matmul in shared memory, so reconstructed weights never round-trip
through HBM.** Their durable finding and my measured one are the same insight: the bus is
the bottleneck, and fusion is how you avoid paying it twice. Two notes I'd raise with the
VP:

- Their honest accounting (30–40% throughput overhead today for ~13% memory saving — a
  *capacity* play, not yet a *speed* play) matches what I saw: the byte savings are real,
  realizing them as latency is a kernel-engineering problem. There's shared ground on
  where the next kernel wins come from.
- **Omni × Unweight compound** (smaller footprints × over-commit → models-per-GPU), and
  my MoE-streaming + quantization analysis slots into the same models-per-GPU axis.

## One forward lead

Everything I tested was *post-hoc* — decomposing weights trained without the structure.
A model **trained** in a bandwidth-friendly factored form from the start (design-time, the
way Tensor-Product-Attention did for activations) is the untested door, and it inherits
the fused-kernel result for free. Speculative, but it's the one direction the negative
results don't close.

---

*Backing detail: full results in `docs/2026-06-10-h100-session-results.md`; quantization
theory pass in `docs/d0-literature-notes.md`; all runs + figures committed under
`results/`.*
