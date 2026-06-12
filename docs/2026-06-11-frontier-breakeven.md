# Frontier break-even pre-test — the scale question, resolved (2026-06-11)

Question (left open by `2026-06-11-lrs-results.md`): the GPT-2 negative left a scale
loophole — fp16 low-rank side-information costs Δb = 16r(1/m+1/p) bpw, which shrinks
~10× at frontier dims, so the break-even condition `ε > 1 − 4^(−Δb)` drops from
"capture 92% of energy" to "capture 29%". Does Avenue 1 revive at d = 8192?

**Answer: no — and the reason is a law, not a sample.** Trained *transform* weights
track the Shannon break-even line across three decades of width because their stable
rank grows with width at almost exactly the canceling rate. The only matrices that
sit above the line are *table-like* objects (position embeddings, MoE routers, and —
growing with model scale — the rogue-channel structure of layer-0 input-readers),
and every one of those is either negligible in parameters or axis-aligned, i.e.
already absorbed for zero bits by per-channel scaling. **Avenue 1 closes at all
scales.**

Method: `experiments/frontier_breakeven.py` streams HF shards one at a time (peak
disk ≈ one shard), takes singular values only, and reports per matrix the best
margin in effective bits/weight:
`lr_margin = max_r [log₄(1/(1−ε(r))) − 16r(m+p)/(mp)]` (sparse analog with
fp16+index cost). Positive ⇒ the structure pays. Calibration on GPT-2: the metric
gives `wpe` +4.0 bits (the one known true L+S object) and scores every matmul
weight ≈ 0, matching the Stage B empirical near-tie-but-lose. Working threshold for
"interesting": ≳ +0.25 bits (cost-model slop eats anything smaller — exactly the
+0.1-class rows that lost empirically in Stage B).

Runs: `results/frontier_breakeven/20260611-19{3752,3901,4609,5021}-2291d69`
(gpt2; Llama-3.1-8B; Qwen3-30B-A3B-Base; Llama-3.1-70B, 3 layers sampled).
Figure: `results/frontier_breakeven/frontier_margin_vs_width.png`.

## The law: transforms hug the line at every width

| model | width(s) | transform lr_margins | typical stable rank |
|---|---|---|---|
| GPT-2 | 768 | −0.02 … +0.13 | 18–131 |
| Llama-3.1-8B | 4096–14336 | −0.02 … +0.23 | 34–584 |
| Qwen3-30B-A3B (96 expert mats) | 768–4096 | −0.03 … +0.04 | 15–182 |
| Llama-3.1-70B (layers 40, 79) | 8192–28672 | −0.02 … +0.06 | 14–1132 |

At small budgets the break-even condition reduces to `stable_rank ≲ d_h/22`
(d_h = harmonic mean dimension): the threshold *loosens* linearly with width — but
measured stable ranks grow roughly linearly with width too (e.g. up_proj:
27 → 255–584 → 671 from GPT-2 to 8B to 70B mid-stack). The two effects cancel, so
the margin is pinned near zero at 768, 4096, and 8192 alike. The scale loophole was
real arithmetic about the *cost* side; the *energy* side moves in lockstep. The
spectra of trained transform weights are, in this accounting, approximately
self-similar across scale — marginal compressibility is scale-invariant.

Sparse side: dead everywhere at scale (max +0.04 outside layer 0); index bits grow
with log(mp), so spike economics worsen with size.

## The exceptions are a taxonomy, not noise

1. **Tables pay.** `wpe` +4.0 (stable rank 3); Qwen MoE *routers*
   (`mlp.gate.weight`, 128×2048) +0.34/+0.49 with stable rank 2. Genuinely low-rank
   functional objects — and ~0.1% of parameters.
2. **Layer-0 input-readers at scale (the new finding).** Margins grow with model
   size: GPT-2 layer 0 ≈ 0; 8B layer 0 q_proj +0.23 (stable rank 6); 70B layer 0
   q +4.17, k +3.46, v +2.28, gate +2.82, up +2.81 — five wpe-class payers, while
   layer-0 *output-side* matrices (o_proj, down_proj) stay normal (+0.01).
   **Verified by shard re-download:** the top singular values ARE column norms
   (615.7 ↔ 615.3, 266 ↔ 258, …) — the structure is a handful of giant *input
   columns*, max/median column norm 1844× (q_proj), 665× (gate_proj, where ONE
   column of a 28672×8192 matrix holds 97% of the energy). It decays into the bulk
   over the first layers (q_proj stable rank 1.8 → 3.6 → 9.6 at layers 0/1/2).
   This is the massive-activations / rogue-embedding-channel phenomenon read
   directly from weight spectra — input-readers of layer 0 carry enormous weights
   on the rogue channels, increasingly so with model scale.

## Why even the exceptions don't reopen Avenue 1

Giant columns are **axis-aligned** structure. A per-input-channel scale (the AWQ
move) absorbs a column's magnitude *exactly*, multiplicatively, for ~0 stored bits —
no low-rank factors needed. So the complete account is:

- transform weights → ON the line → rotation + RTN (free currency), nothing to
  extract;
- rogue channels (layer-0 readers) → ABOVE the line but axis-aligned → per-channel
  scales (free currency);
- non-axis-aligned true tables (wpe, routers) → the only honest L+S customers →
  ~0.1% of parameters.

There is no weight class left for additive fp16 side-information to serve. This also
*explains practice*: the first layer is precisely the one quantization deployments
special-case (higher precision or channel protection), and the measured 665–1844×
column ratios are the mechanism; the measured growth of this structure with model
scale predicts the special-casing matters more, not less, at larger models.

## Status

Avenue 1: **closed at all scales** (supersedes the "revisit at frontier dims" scope
condition in `2026-06-11-lrs-results.md`). The instrument (`frontier_breakeven.py`)
is general: one cheap streaming pass scores any future checkpoint's weight classes
against the side-information break-even line. Next live lead per the postmortem:
the unbiased-rounding depth-coherence study (QuIP Supp C.8 anomaly), which is
orthogonal to this result and unaffected by it.
