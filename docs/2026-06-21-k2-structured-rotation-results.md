# K2 — structured / streamable rotation for eigenbasis water-filling (2026-06-21)

Follow-up to `docs/2026-06-21-k2-eigwaterfill-results.md`: the full C×C KLT rotation
on the key residual won 2.24× on logit but is KILLED-HONEST (the C×C matrix costs
+8 bpe). This tests whether a **cheaper / streamable rotation captures enough of the
win to beat uniform at HONEST bpe** — read as a logit-vs-honest-bpe **Pareto
frontier** against a uniform bit-sweep. Arms: top-k truncated KLT (k∈{32,64,128}),
block-diagonal per-head KLT, frozen-prefill full KLT, oracle (refit control). Runs:
`results/k2_waterfill/20260621-175904-6027e56` (llama, post-RoPE) and
`results/k2_waterfill/20260621-175657-de07b6d` (gpt2). rank=16, group=64,
tiers=(0,2,3,4), budget 3.0, prefill_fit_len=512.

## Headline: NO structured rotation beats uniform at matched bpe — killed

The deployable verdict is the Pareto comparison: each structured arm's logit at its
HONEST bpe vs the **uniform bit-sweep** at the same bpe. Llama-3.1-8B, mean ± sem
over 32 layers (logit_rope, lower better):

| arm | logit ↓ | honest bpe | vs uniform at same bpe |
|---|---|---|---|
| **uniform @2b** | 0.0978 ± 0.0030 | 2.625 | (frontier) |
| **uniform @3b** | 0.0360 ± 0.0010 | 3.625 | (frontier) |
| **uniform @4b** | **0.0154 ± 0.0004** | 4.625 | (frontier) |
| **uniform @5b** | 0.0072 ± 0.0002 | 5.625 | (frontier) |
| blockdiag per-head KLT | 0.0309 ± 0.0010 | 4.626 | **2.0× WORSE** than uniform@4b (0.0154) |
| topk k=128 | 0.0344 ± 0.0009 | 4.626 | **2.2× WORSE** than uniform@4b |
| topk k=64 | 0.0386 ± 0.0011 | 4.126 | worse than uniform interpolated @4.1b |
| topk k=32 | 0.0393 ± 0.0011 | 3.876 | worse than uniform@3b–4b |
| frozen-prefill full KLT | 0.0310 ± 0.0009 | 11.626 | not competitive (and drifts; see below) |
| oracle (control, uncharged) | 0.0161 ± 0.0005 | — | = full-KLT ceiling |

GPT-2 replicates exactly: uniform@4b 0.0159 vs blockdiag 0.0336 @4.835 (2.1× worse),
topk_k128 0.0331 @5.835 vs uniform@5b 0.0074 (4.5× worse).

**At every matched bpe, uniform precision wins — decisively (~2× on the closest
arm).** The eigenbasis win is real (oracle reproduces 0.0161) but cannot be encoded
cheaply enough: a structured rotation that costs ~1 extra bit delivers ~2× the logit
distortion of just spending that bit on uniform precision.

## Every pre-registered prediction held (foundation-grounded)

The spec measured the residual eigenspectrum first (no eigengap, energy spread over
100+ directions) and pre-registered three predictions from primary sources. All
three confirmed:

1. **block-diagonal is the best cheap arm but still loses** (0.0309 — lowest of the
   structured arms, yet 2× worse than uniform@matched-bpe). Within-head decorrelation
   captures the most per bit of the cheap options, not enough to pay.
2. **frozen-prefill DRIFTS** — frozen/oracle ratio **1.93** on Llama (1.32 on GPT-2):
   frozen is ~2× worse than the oracle that refits on the scored tokens. The measured
   residual eigengap is **1.13** (≈1.0 = no gap). This is exactly Davis-Kahan
   (Wainwright §8.1.2 / Vershynin Thm 4.1.15): eigenvectors are unstable without an
   eigengap, so a prefill-fit rotation rotates away from the live subspace. The
   no-gap spectrum *predicted* the drift before the run. (Frozen is also killed
   independently by its 11.6 honest bpe — the full C×C matrix.)
3. **topk loses at low k, needs near-full k** — k=32 (0.0393) and k=64 (0.0386) barely
   beat even uniform@3b; only k=128 approaches the structured pack, at which point it
   is not cheap. With no eigengap (Wainwright Ex 8.2), there is no natural cutoff and
   cheap k captures only a fraction (k=32 ≈ 47% of residual energy).

## Why this is the expected negative — the break-even law, KV side

This confirms rather than revives. The prior verdict established the eigenbasis win
is **real but too expensive to encode** as a full rotation; this shows the same holds
for every *structured* rotation tested. It is the KV-side instance of the program's
weight-side frontier law (`ε > 1 − 4^(−Δb)`, `docs/2026-06-11-frontier-breakeven.md`):
side-information (here, a stored rotation basis) pays only if the energy it removes
exceeds what the same bits buy the bulk quantizer. One extra bit of uniform precision
divides residual distortion by ~4 (the `4^{-b}` rate); a +1-bpe rotation removes far
less, so it sits above the break-even line. The eigenstructure is genuine but its
description cost dominates its benefit at every granularity from per-head (d×d) to
full (C×C).

## Gate call: KILLED — uniform recipe stands

No structured rotation makes the eigenbasis water-fill deployable. The k2b recipe
(`lowrank_rtn_channel` on pre-RoPE keys, **uniform** residual bits) is unchanged and
remains the operating point: at any bpe, uniform precision is the better spend than a
data-derived rotation plus its storage. The full arc is now closed:

- per-channel water-fill (raw basis) → killed (wrong basis)
- eigenbasis water-fill (full C×C) → real 2.2× win, killed-honest (rotation too big)
- structured rotations (topk / block-diag / frozen) → killed at matched bpe (rotation
  description cost dominates the gain at every granularity)

**Deferred (last avenue, not pursued):** Hadamard-class FIXED rotations cost **zero**
stored bits (seeded/structured, like the existing `random` arm). The random-rotation
control already loses (it spreads variance, the opposite of what helps), but a
*fixed structured* rotation chosen to approximate the average eigenbasis is the only
remaining zero-cost option. The evidence is not encouraging — the win is data-derived
and per-layer, which a fixed transform cannot track — but it is the one rotation that
escapes the break-even charge entirely. Separate gate if ever pursued.

## Caveats (carried honestly)

- **The verdict is the Pareto comparison against the uniform bit-sweep**, NOT the
  `lower_logit_than_uniform_at_3b` win-count (which the SUMMARY explicitly labels
  "NOT matched-bpe"). An earlier version of this experiment compared structured arms
  only to uniform@3b — a bits-advantage artifact that flattered them; the uniform
  sweep (uniform@{2,3,4,5}) fixes it and is the honest baseline.
- frozen's two failure modes are distinct and both real: it **drifts** (fo 1.93,
  scored uncharged) AND is **too expensive** (11.6 honest bpe). Don't conflate.
- GPT-2 `logit_rope` is stored-basis (no RoPE); Llama is the authoritative post-RoPE
  result. Block-diagonal needs real GQA head structure — Llama (h_kv=8, d=128) is the
  meaningful test; both agree.
- `oracle` is a non-deployable control (refits on the tokens it scores); it exists
  only to measure frozen's drift and to reproduce the full-KLT ceiling (0.0161 ✓).
