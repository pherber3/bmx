# K2 — matched-bits cache-codec bake-off results (2026-06-12)

E2/E3 from `2026-06-11-kv-research-plan.md`. Six codecs (`bmx.cache.codecs`), honest
bit accounting (every scale/norm/factor counted; seeded rotations/sketches free),
real caches from K1 (GPT-2 S=1024; Llama-3.1-8B S=2048) + random-sphere controls.
Primary metric: attention-logit distortion against the layer's real stored queries
(per-head, GQA-expanded); kind v uses attention-output distortion. Pre-RoPE arms
evaluated honestly in the post-RoPE basis via `bmx.cache.rope` (apply-at-read;
self-validated against collected k/k_pre pairs, rel < 2e-2 asserted in-run).
Runs: `results/k2_cache_arms/20260611-234123-b7f1b90` (gpt2, 894 rows),
`20260611-235445-b7f1b90` (llama-3.1-8b, 2334 rows).

## Headline: the cache's low-rank structure pays at MATCHED bits — and pre-RoPE is where it pays most

Llama-8B keys, final post-RoPE logit distortion, mean over 32 layers (bpe = total
bits/entry incl. metadata):

| budget | best baseline (any scalar codec) | lowrank_rtn_channel on k_pre |
|---|---|---|
| ~3.0 bpe | turboquant_mse 3.02 → 0.090 | **r=32 @ b=2, 3.00 → 0.080** |
| ~3.4–3.6 bpe | (none between 3.25 and 4.0) | **r=16 @ b=3, 3.63 → 0.036** |
| ~4.0–4.25 bpe | turboquant_mse 4.02 → 0.047; rtn_channel 4.25 → 0.049 | **r=32 @ b=3, 4.00 → 0.030** |

At every budget the low-rank points sit below every scalar codec — **2–3× lower
logit distortion at equal bits, or ~1 bit saved at equal distortion**. This is the
opposite of the Avenue 1 weights verdict, exactly as K1's break-even margins
predicted, and it survives honest accounting (fp16 factors counted; factors
fp16-roundtripped in reconstruction).

**Pre-RoPE is a free further win for the low-rank arm only**: at b=3/r=32,
compress-pre-RoPE-then-rope-at-read scores 0.0297 vs 0.0425 for the same codec on
post-RoPE keys (−30%). Elementwise quantizers don't care (rtn_channel 0.1202 pre vs
0.1211 post) — confirming K1's mechanism: RoPE smears the *subspace*, not the
per-entry distribution. Deployable recipe: store keys pre-RoPE (the MLA-style
decoupled design point), low-rank + per-channel residual.

## Keys and values want different codecs (KIVI's asymmetry, reproduced from first principles)

Values are the mirror image: no usable subspace (K1: V margins were mean-dominated),
and per-channel scales are the WORST real arm (b=3 output distortion 0.337 vs
rtn_token 0.159), while turboquant_mse (rotate + per-token Lloyd-Max) wins the
whole V curve (0.294/0.152/0.079 at b=2/3/4). KIVI's empirical keys-per-channel /
values-per-token split falls out of our instruments without citing KIVI.

A metric lesson that generalizes (also visible in the codec unit tests): under
rogue channels, **Frobenius error inverts the ranking** — on Llama keys rtn_channel
wins rel_fro (0.181 vs rotation's 0.249) but loses logit distortion (0.121 vs
0.114). The vault's MSE-vs-IP distinction is not pedantry; it flips conclusions.

## E3 — the TurboQuant verdict: claims replicate, objective is wrong

Measured real-vs-random gap (rel_fro, b=3, Llama keys): turboquant_mse real 0.1852
vs random-sphere 0.1858 — **its near-optimal worst-case rate holds exactly on real
data** (no degradation; the GPT-2 run showed the same at +0.176 gap in its favor
vs its own theory floor being beaten by structure-aware arms). So: not marketing in
the false-claims sense — the bounds are real and it is the best scalar codec for
values. But worst-case-optimal is the wrong objective for keys, where real caches
are far from adversarial: structure-blind optimality concedes 2–3× to a
structure-aware codec at matched bits. **turboquant_prod (the unbiased two-stage)
loses everywhere on distortion** (b=3 keys: 0.197 vs 0.090 for its own MSE stage)
— its 2× variance premium can only pay through aggregation effects, which is
precisely K3's unbiasedness question and the perplexity follow-up's job; on
single-matrix distortion it is dominated, full stop.

E2 verdict, nuanced by scale: on GPT-2 keys per-channel beat rotation outright; on
Llama keys (rogue ratios ~20×) rotation edges per-channel on logit while losing
Frobenius — but both are dominated by lowrank, so the basis war among scalar
codecs is a fight for second place.

## Gate G2: OPEN — three finalists to the perplexity round

1. **lowrank_rtn_channel on pre-RoPE keys** (the matched-bits winner),
2. **turboquant_mse for values** (best V codec at every budget),
3. **rtn_channel** (the boring KIVI-style control — the bar to clear).

Follow-up plan (K2b): quantized-cache perplexity eval wiring these three end-to-end
(keys + values together), context-length sweep, and the turboquant_prod aggregation
question folded into K3. The skipped-by-design caveats stand: symmetric quantization
only; full-S per-channel scales (offline science, not streaming); S=1–2k contexts
(low-rank factor cost amortizes further at long context, so margins here are
conservative for the lowrank arm).
