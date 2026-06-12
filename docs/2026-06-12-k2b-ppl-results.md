# K2b — quantized-prefill perplexity results (2026-06-12)

The functional round for K2's finalists (`2026-06-12-k2-arms-results.md`), in the
quantized-prefill setting: prefill N tokens, quantize the full cache with the
codec pair, teacher-force the next 256 tokens, compare NLL to the fp16-cache
baseline. Eval validity is anchored by an exact identity invariant (fp16 no-op
path == full-forward continuation NLL to machine epsilon, three-way verified on
both architectures). Runs: `results/k2b_cache_ppl/20260612-052754-d3329ab`
(gpt2), `20260612-053057-d3329ab` (llama, contexts 512/1024/1792),
`20260612-084509`/`20260612-084556-d3329ab` (sensitivity ablations).

## G2b verdict: the K2 ranking holds end-to-end, and the recipe is deployable

Llama-3.1-8B, n_prefill=1792 (fp16 baseline ppl 6.272):

| arm (K / V) | bpe K/V | Δppl b=3 | Δppl b=2 |
|---|---|---|---|
| **lowrank+rtn_channel pre-RoPE / turboquant_mse** | 4.04/3.02 (b=3) | **+0.31%** | +2.48% |
| turboquant_mse / turboquant_mse | 3.02/3.02 | +1.00% | +5.58% |
| turboquant_prod / turboquant_prod | 3.03/3.03 | +5.41% | +196.6% |
| rtn_channel / rtn_channel (symmetric KIVI-style control) | 3.25/3.25 | +9.17% | **+3160%** |

- The finalist combo is the best arm at every bit-width and every context
  (512/1024/1792), and its degradation is context-STABLE (+2.5% flat at b=2
  while the control worsens 1808% → 3160% with length).
- The symmetric per-channel control collapses at 2 bits. Caveat recorded: real
  KIVI is asymmetric (zero-points) with a fp16 residual window — exactly the
  machinery that prevents this failure; our control is the symmetric variant,
  so the right reading is "symmetric per-channel needs ≥3 bits," not "KIVI fails."
- GPT-2 reproduces the ordering (combo +2% at b=3 vs +25% control), with one
  instructive swap vs K2's per-matrix ranking: once K AND V are quantized
  together, V's codec dominates the outcome, and turboquant_mse overtakes
  rtn_channel — single-tensor distortion rankings don't compose; the pair does.

## Sensitivity ablation: the bits belong to K

K-only / V-only / asymmetric-bits on the finalist codecs (Llama, n_prefill=1792):

| configuration | bpe K/V | Δppl |
|---|---|---|
| V-only @3 (K fp16) | 16/3.02 | +0.18% |
| K-only @3 (V fp16) | 4.04/16 | +0.39% |
| **K@3 / V@2 (the recipe)** | **4.04/2.02** | **+0.51%** |
| V-only @2 | 16/2.02 | +1.24% |
| K-only @2 | 3.04/16 | +2.38% |
| K@2 / V@3 (wrong way) | 3.04/3.02 | +2.71% |

K-only damage at 2 bits is ~2× V-only damage (GPT-2 agrees: +11.4% vs +6.6%),
and the asymmetric pair confirms the direction both ways: K@3/V@2 ≈ the sum of
its parts (+0.51%), K@2/V@3 is the worst configuration at its budget. **The
deployable recipe: keys pre-RoPE low-rank(r=32)+per-channel @3b (≈4.0 bpe with
factors), values rotate+Lloyd @2b (≈2.0 bpe) — ~3.0 bpe average, +0.5% ppl,
a 5.3× KV-memory reduction vs fp16.** The low-rank factor cost amortizes
further at long context, so 4.0 bpe on keys is the short-context worst case.

## The unbiasedness question (K3), largely answered for free

turboquant_prod — the unbiased two-stage codec — is dominated at every bits,
context, and model: its 2× variance premium is never repaid by aggregation over
1–2k cached tokens and 32 layers; at 2 bits it amplifies catastrophically
(+197% vs +5.6% for its own biased MSE stage). Together with QuIP's weight-side
finding, the picture is now consistent rather than anomalous: in cache/weight
coding, bias is cheap (errors stay incoherent through the network) and variance
is expensive. K3's remaining open sliver: a regime where the estimate is
aggregated MANY more times than it is paid for (e.g. 32k+ contexts) — testable
on a VM if ever needed; nothing in these data motivates it.

## Caveats (by design this round)

Symmetric quantization only; full-sequence per-channel scales (offline prefill
quantization, not streaming decode-time appends); contexts ≤ 2k (CPU); the
low-rank arm's streaming variant (frozen channel-subspace fit on prefill,
per-token coefficients thereafter — the MLA-shaped deployment) is the natural
K2c build if this goes further toward systems work.

## Gate call

G2b CLOSES POSITIVE with a quantified, attributed, deployable recipe. The KV
program's remaining tracks: K3 narrows to the long-context aggregation sliver
(low priority given the above); the systems-facing next step, if any, is the
streaming low-rank codec + a fused dequant-attention kernel byte model (Track B
machinery applies). The science story for external audiences is complete:
census → structure → matched-bits codecs → end-to-end perplexity, with every
step's instrument published in-repo.
