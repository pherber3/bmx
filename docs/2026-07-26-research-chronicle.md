# The bmx research chronicle — from hypermatrix hypothesis to shipped KV codec (2026-06-10 → 2026-07-26)

This is the capstone narrative of the program: every junction in sequence —
what was believed, what was measured, why the direction changed, and the
mathematics that licensed each decision. It is assembled from and pointered
into the primary record (46 dated results/decision docs, 46 specs/plans,
per-run config+env+SHA provenance); where this doc and a results doc
disagree, the results doc wins. Companion documents: `README.md` (the
2-minute arc), `2026-07-25-k4-paper-shelf.md` (paper material by section),
`2026-07-26-gh200-rental-results.md` (the final evidence batch).

The one-sentence arc: **a falsified hypothesis about weight structure was
generalized into a break-even instrument; the instrument said the structure
lives in the KV cache; six weeks of kill-or-confirm gates turned that into a
corpus-calibrated spectral codec that beats the strongest published baseline
on quality-per-bit, with every intermediate negative documented and most of
them load-bearing.**

---

## Part I — The question and the instrument (2026-06-10/11)

**The starting hypothesis.** The project began as a test of
Bhattacharya–Mesner (hypermatrix) decomposition on trained transformer
weights: model weight *stacks* (e.g. all 12 attention circuits of GPT-2,
768×768×12) as **template-shaped** — one shared diagonal template reused
across slices via the BM product
`bmp(A,B,C)[i,j,k] = Σ_t A[i,t,k]·B[i,j,t]·C[t,j,k]`, with a
bandwidth-amplified-decode payoff: decode is byte-bound, and a factored
matvec reads `ℓ·m·p` bytes where dense reads `h·m·p` — if ℓ ≪ h, decode
speeds up proportionally. Three program entries: diag-template matvec, BMD
expert streaming for MoE, VQ theory (`2026-06-10-h100-session-results.md`).

**How it died — with a discriminator, not an opinion.** One H100 session ran
matched-parameter sweeps with permutation-null controls. Entry 1: BMD was
the *worst* method at every matched budget (median rel_error 0.33 vs
Tucker's 0.127 at comparable params; slice-SVD exact because slices have
rank ≤ d_head), and the null control localized the reason — Tucker beats its
permutation-null by 0.06–0.10 (real cross-slice structure exists, and it is
**subspace-shaped**), while BMD's real-vs-null gap is ≈0 (no diag-template
alignment beyond chance). Entry 2 falsified on OLMoE: expert redundancy is
global (~10 shared modes) but too thin to exploit — at matched 18M params
BMD 0.874 vs Tucker 0.878, indistinguishable. Entry 3 re-scoped (the
Gaussianization and 4^-b floor hypotheses were already published). The
solver itself was vindicated — the repo's RALS beats the BM-ALS paper's own
solver by 3–10 orders of magnitude on the paper's own tensors (pinned by
`tests/test_sagemath_agreement.py`; the project's one intentional xfail
records the flip side). The method was sound; the hypothesis about weights
was wrong.

**The generalization — the break-even inequality.** The follow-up (L+S
residual, `2026-06-11-lrs-results.md`) confirmed weight structure *exists*
(subspace overlap 0.91–0.99 with the Tucker subspaces; sparse support hits
the known rogue channels) but captures only ~21% of a matmul weight's
energy. The postmortem turned that into the program's central instrument.
From the b-bit quantization floor `D ∝ 4^{−b}` (each bit quarters MSE),
spending Δb bits/weight on side-information instead of on the bulk quantizer
pays iff the energy fraction ε it removes satisfies

> **ε > 1 − 4^(−Δb)**,  with Δb = 16·r·(1/m + 1/p) for fp16 rank-r factors.

At GPT-2 scale, r=64 costs Δb ≈ 1.78 ⇒ break-even ε = 91.6%; measured
ε ≈ 22–26% — "a 4× shortfall on the wrong side of an exponential." Two
structural reasons the baselines were unbeatable: rotations and per-channel
scales are *multiplicative/basis* corrections that never face the 4^-b
hurdle ("a different, free currency"), and groupwise RTN had already
quarantined outliers locally.

**The frontier law.** The inequality left a scale loophole (Δb shrinks with
width; at d≈8k break-even ε drops to 29%). Measuring `lr_margin` from GPT-2
to Llama-3.1-70B closed it as a law (`2026-06-11-frontier-breakeven.md`):
transform weights sit ON the break-even line at every width (margins
−0.03…+0.23), because break-even reduces to `stable_rank ≲ d_h/22` while
measured stable ranks grow ~linearly with width (up_proj 27 → 671) — **the
two effects cancel; marginal weight compressibility is scale-invariant.**
The only payers form a taxonomy: true tables (wpe +4.0 bits, MoE routers),
and layer-0 rogue-channel input-readers that *grow* with scale (70B layer-0
q_proj +4.17; one gate_proj column holding 97% of the matrix energy,
1844× column-norm ratios) — but those are **axis-aligned**, absorbed free by
per-channel scales. No weight class is left for additive side-information to
serve. Avenue 1 closed at all scales.

---

## Part II — The pivot (2026-06-11)

The KV research plan (`2026-06-11-kv-research-plan.md`) states the decision
in one equation and one sentence. Decode `time/token ≈ (weight bytes + KV
bytes)/bandwidth`; weight bytes are solved-to-within-a-bit; **KV bytes grow
with context × batch and equal the 4-bit weights of a 70B model at ~100k
context.** And: *"the structure that is absent from trained weights
(Gaussianized bulk, axis-aligned outliers handled free) lives in
activations — which is what the KV cache is."*

Everything transferred verbatim: the break-even instrument (now scoring
activation spectra — the K1 census literally reuses
`frontier_breakeven.py`'s `matrix_row`), the basis-decision methodology
(per-channel scales vs rotation as *competing treatments* — the KIVI/QuaRot
split in the literature), matched-total-bits arms with all-metadata
accounting, and null controls (random-sphere matrices as the codec-theory
control; "the gap between them IS the marketing-vs-reality measurement").

The bet paid immediately: where this model class's weights scored
−0.02…+0.23 bits of margin, the cache scored **K up to +2.46 (gpt2), K
pre-RoPE +0.90…+2.08 (Llama), V +0.04…+0.47** on the same instrument.

---

## Part III — The KV program: K1 → K2c and the June allocation trilogy (2026-06-11/12, 06-21)

**K1 census — the three doctrines** (`2026-06-11-k1-census-results.md`):
(1) **Bits belong to K** — keys carry the compressible structure; the 2×
sensitivity number lands in K2b (K-only@2b costs +2.38% ppl vs V-only@2b
+1.24%). (2) **Store keys pre-RoPE** — pre-RoPE margins +0.90…+2.08 vs
post-RoPE +0.46…+0.63 ("RoPE costs ~1–1.5 bits of key compressibility"):
position-dependent rotation smears the shared subspace — the same mechanism
as the weights-era basis lesson, now measured on the serving object.
(3) **Rogue channels are real in activations** — channel-norm max/median up
to 23.8 vs ≈1–2 for weights; the basis war has real structure to fight over.
Validity: mean-centering barely dents K margins (2.46→2.41) — keys are
genuinely low-rank; V's small margin mostly IS the token mean.

**The metric doctrine.** Under rogue channels Frobenius error *inverts*
codec rankings — measured directly (rtn_channel wins rel_fro 0.181 vs 0.249
yet loses logit distortion 0.121 vs 0.114 against rotation on the same
keys). Hence the repo-wide rule: rank codecs on attention-logit distortion
against the layer's *real stored queries*; perplexity is the end-to-end
verdict but too coarse to attribute component choices.

**K2 bake-off → the recipe** (`2026-06-12-k2-arms-results.md`, `-k2b-`,
`-k2c-`): at every matched budget, low-rank on pre-RoPE keys sits 2–3×
below every scalar codec (e.g. r=32@b3: 0.030 vs turboquant_mse 0.047 at
~4 bpe); pre-RoPE is a further −30% for the low-rank arm only (elementwise
codecs don't care — confirming the subspace mechanism). Values are the
mirror image: no usable subspace, per-channel scales are the *worst* V arm,
rotate+Lloyd (turboquant_mse) wins the whole V curve — KIVI's empirical
split falls out of the instruments without citing KIVI. TurboQuant's bounds
**replicate exactly on real caches** (real 0.1852 vs random-sphere 0.1858)
— but worst-case-optimal is the wrong objective for keys: structure-blind
coding concedes 2–3× to structure-aware coding at matched bits, and
unbiased coding (turboquant_prod) is dominated everywhere. End-to-end
(quantized-prefill ppl, anchored by an exact fp16 no-op identity):
**keys pre-RoPE lowrank+per-channel @3b, values rotate+Lloyd @2b ⇒ ~3.0
bpe, +0.5% ppl, 5.3× vs fp16**, degradation context-stable while the
symmetric per-channel control collapses (+3160% at 2 bits). K2c licensed
streaming: the prefill-frozen pre-RoPE subspace holds **0.942 of the oracle,
drift-flat over 1200 tokens** (post-RoPE decays monotonically — the fourth
independent line on the same mechanism).

**The June allocation trilogy — the seed of K4** (the three 2026-06-21
docs; runs never committed, docs are the surviving record). Raw-basis
per-channel water-fill: KILLED (uniform wins 32/32 layers — the funded
high-variance channels are not the ones queries read). Eigenbasis (KLT)
water-fill: **mechanism CONFIRMED, deployment KILLED-HONEST** — the KLT wins
2.24× (32/32 layers; the random-rotation control ties/loses, proving it's
data eigenstructure; funded directions ARE query-read,
query_eigen_alignment ≈ 0.59) but the C×C basis charged per-sequence costs
+8.0 bpe at S=2048 — the KV-side instance of ε > 1−4^(−Δb). Structured/
streamable rotations: KILLED at matched bpe (block-diagonal 2.0× worse;
frozen-prefill rotation drifts — residual eigengap 1.13 ≈ none, so
Davis–Kahan gives no stability). **The trilogy defined exactly which lever a
revival had to pull: keep the eigenbasis win, move the basis cost off the
sequence.** K4 is that revival — corpus-calibrated *model-level* bases with
zero per-sequence charge.

---

## Part IV — Making it real: K3, task metrics, kernels, hygiene (2026-06-19 → 07-06)

**K3 streaming cache** (`2026-06-19-k3-streaming-cache-results.md`). The
recipe had to survive live generation. One real bug found and fixed at the
root: the first implementation re-quantized the growing prefix each step
from the previous step's *dequantized* slab — harmless for idempotent RTN,
catastrophic for the Lloyd V codec (**norm 4.0 → 397.6, a 98× explosion over
64 steps**). The fix is the design: **write-once quantized storage** — each
token quantized exactly once, at block flush, from its pristine source; a
fp16 recent window lets channel-grouped arms stream. Verdict: K2b quality
holds under streaming (tbt-ppl 1.001× fp16), honest packed bpe < fp16, and
all arms (K2b/TurboQuant/KIVI-style/fp16) run on **one fair code path**.

**Task metrics arrive** (`2026-06-21-niah-longbench-frontier-results.md`).
The paper bar is TurboQuant's table, so the program built both halves of it
on the live cache: NIAH ROUGE-1 needle recall and LongBench Code
`code_sim`. First findings: at matched ~7×, the compression-matched
k2b_k2r8 variant beats turboquant_mse on both metrics (NIAH 8.47 vs 7.88;
Code 50.3 vs 46.0 — and our turboquant_mse
reproduces the paper's Code ≈ 46 exactly, the comparability datum); the
**key-bits ↔ context tradeoff** appears (3-bit keys hold fp16 retrieval to
64k; 2-bit keys die first at long context); turboquant_prod's collapse is
verified to be the method, not a bug (unbiased per-key IP noise as large as
the IP itself, exposed to softmax each step).

**The kernel program** (`2026-06-23-kernel-census-results.md`,
`2026-06-24-triton-decode-results.md`). Two real CUDA-only bugs on first
GPU contact: the chunked path's online-softmax materialized O(S²) score
tiles at *prefill* (it was designed for decode; fix = dispatch on n_q —
flash SDPA at prefill, chunked at decode); and a custom AttentionInterface
ran **maskless** because no AttentionMaskInterface mask fn was registered —
transformers silently passed `attention_mask=None`, prefill fell back to
`is_causal=True`, which is wrong for the cached two-block prefill (0.40
attention divergence → garbage logits). Census verdict: at 128k, chunked
k2b is resident-neutral with fp16 (64.1 vs 63.3 GiB) while dense-streaming
balloons to 83.5 — compression became resident, not accounting fiction; the
batched 128k NIAH that OOMs on the dense path completes on the packed path
(k2b recall 7.94, healthy). Phase 3 built the single-launch split-KV Triton
decode kernel that dequants **in-kernel** (low-rank K reconstruction + RoPE
+ per-head turboquant-V Hadamard): 2624× (RTN) / 322× (k2b) over chunked at
128k. The one real design decision: **per-head Hadamard for V** — the full-C
rotation couples heads and provably cannot fuse into a per-head decode
kernel or fold into o_proj; the per-head variant is quality-equivalent
(rel-MSE ratio 0.986; the TurboQuant constant is dimension-independent).
Dead paths were deleted with a recovery doc
(`2026-06-24-decode-path-debloat-removal.md`) — the deletion-with-pointer
discipline.

**Hygiene junctions** (07-01 → 07-06). Baselines locked TRANSITIVE
(TurboQuant already benchmarks KIVI/PolarQuant/SnapKV/PyramidKV on the same
tables; we run fp16 + turboquant arms + the rtn2 control ourselves). The
"kivi" arm was diagnosed as a symmetric-RTN strawman (no zero-point; 33–38×
MSE penalty off-center) and renamed honestly. The Triton desk review found
F0–F3 without touching the GPU — F0 being the classic: the "12× slower"
A/B had benchmarked an arm that fires **no** fused kernel (chunked fallback
by construction; the warn now ships). And **anchor forensics** killed
absolute cross-harness parity for good: the same items, same generation,
raw-template vs chat-wrapped prompts move LongBench code_sim by **43.7
points** — an order of magnitude larger than any quantization effect in the
program. Decision that licenses the paper's comparisons: **delta-parity** —
compare each method's delta from its own full-cache baseline on its own
path (theirs: TurboQuant-2.5 −0.62; ours: k2b −1.19 at 4.07×). The C3
memory saga ran negative (07-05: packed fits *half* as many sequences as
fp16 — ~4 GiB/seq of non-code scratch) then resolved positive (07-06:
pack_v + shared RoPE + the sweep-design artifacts accounted; **packed
marginal 2.258 GiB/seq vs dense 4.008 @32k, zero residual mystery**) — an
honest reversal-of-a-reversal, fully decomposed.

---

## Part V — The crisis junction (2026-07-05/08)

The authoritative VM run (`2026-07-05-authoritative-vm-results.md`) moved
three claims from "pending" to "measured negative" in one weekend: the
absolute LongBench anchor missed by ~8 points (later explained by prompt
policy, not truncation — truncated rerun moved it <0.1); the runtime-memory
claim reversed on the measured path of the day; and the 60-hour eval had
run the wrong arm (`k2b` full-C V routes to the chunked fallback *by
construction* — only `k2b_ph` fires the kernel), so "the kernel is slow"
had never actually measured the kernel. Then the LongBench verdict landed:
**turboquant_mse@3b ("b3") reaches the same Avg at fewer bits than k2b
(40.56 @ 3.21 vs 40.62 @ 3.94) — the k2b LongBench quality claim was
dead**, surviving only in Synthetic/retrieval (+2.3). (Provenance note: the
07-08 verdict runs are not in the git tree; the numbers are pinned by the
duel doc §7 and the K4 spec §1, and the July-15 in-tree runs are the
authoritative comparators — as the rental doc also records.)

What made this a junction rather than a defeat: the quality/compression
*science* held (k2b ≈ 97% of fp16 at 4.07×, beats the turboquant arm by
+7.3 Avg), and the program's own June trilogy had already identified the
unplayed lever. If a structure-blind scalar codec at 3 bits matches a
structure-aware codec at 3.9, the structure must be cashed *more
efficiently* — per-direction, per-layer, with the basis cost amortized off
the sequence. That is the K4 spectral codec: **query-weighted KLT +
waterfill across directions and layers, corpus-calibrated** — explicitly
"the post-b3-kill attack," unifying the k2t and eig-waterfill ideas the
June kills had scoped.

---

## Part VI — K4: design, gauntlet, duel (2026-07-12 → 07-15)

**The design and its exactness results**
(`docs/superpowers/specs/2026-07-12-k4-spectral-codec-design.md`). The
metric is `E[(qᵀ(k−k̂))²]` with real queries, RoPE applied at read. The
optimal transform under it is the eigenbasis of `W^{1/2} Σ_k W^{1/2}` with
`W = E[qqᵀ]` — and the load-bearing algebra is **exact**: with
`enc = W̃^{1/2}E`, `dec = W̃^{−1/2}E` (W̃ ridge-floored, E orthogonal),
`‖W̃^{1/2}(k−k̂)‖ = ‖y−ŷ‖` identically, so per-direction MSE in code space
IS the weighted metric; the measured logit distortion equals
`Σ_s e_sᵀW e_s` with no cross-terms (queries enter only through their
second moment); GQA pooling and per-head block-diagonality are exact.
Allocation is reverse-waterfill `b_i = max(0, ½log₂(λ̃_i/κ))` over tiers
{0,2,3,4,5,6,8} with the Gaussian proxy `D = Σ λ_i 4^{−b_i}`; rank is not
tuned — truncation falls out as the zero-bit region; spike coefficients get
`½log₂(spike/bulk)` extra bits, never fp16. The pre-registered gauntlet:
G0 (corpus basis retains ≥90% of the oracle win), G1 (strictly below
turboquant_mse at every budget, ≥90% of layers, under BOTH accounting
modes), G2 (across-layer allocation beats uniform at matched mean bits).

**Stage 01** (`2026-07-12-k4-stage01-results.md`): G1 PASSED at every
budget, 100% of layers, both accountings — even the worst-case
rank-deficient heldout basis beats turboquant_mse ≥2.9× under
deploy-skeptic, and the fully-honest corner (unweighted, heldout) still
clears at 2.27×. G0 failed only on its pre-registered rank-deficiency
confound (the clean n≫C corpus test needed the VM). G2 passed on gpt2 with
a surprise: per-layer sensitivity spread ~35×, not the spec's ~3× prior.

**The duel and the accounting** (`2026-07-15-k4-duel-results.md`). The
gates first: Gate A (corpus retention ≥0.90) **FAILED at 0.56–0.64 and
tripling the corpus moved it only to 0.61–0.69** — the transfer ceiling is
structural (this number becomes a theorem in Part VII); consequence:
skeptic accounting is primary. Gate B (query-heldout) passed (weighting
transfers at 0.85–0.88 of its scored-W win); Gate C error bars passed.
Then the two central constructions:

- *Accounting modes, exactly:* skeptic-v1 charges the full C×C fp16 decoder
  per sequence (`16·C/S + tier_bits`); skeptic-v2 charges only the columns
  actually read (`16·C_used/S`; measured C̄_used 829.7 @b2.5, not the ~400
  back-of-envelope); v2-int8 halves the decoder charge again (later
  superseded by the *certified tier-gated* decoder — Part VII).
- *The crossover construction:* a pack codec's bpe **decreases with
  context** (the pack charge amortizes: ~8 bpe at 2k → ~0.5 at 32k) while
  packless baselines are flat — so the honest comparison is a
  bits-vs-context curve with crossovers, not a single number. As finally
  certified: k4_b2.5 crosses below tq_b3 at **~5.6–5.7k context** and below
  the K3/V2 steelman at ~23k; k4_b2.2 sits below both at every measured
  length.

On quality the duel found parity-with-retrieval-edge (LongBench uniform
k4_b2.5 vs b3: Wilcoxon p=0.74; synthetic +3.25 [CI +1.02, +5.56]) — the
pre-registered effect-size targets were NOT met and the doc says so up
front; the real result was the crossover frontier plus the machinery to
defend it.

---

## Part VII — The hardening: review, certificates, honest negatives (2026-07-23 → 07-25)

**The math review** (`2026-07-24-k4-math-review.md`) — ten numbered
findings, each dispositioned by measurement, three of which shaped
everything after:

- **#3, the instrument circularity (the crux).** W as implemented is exact
  *for the instrument* but time-reversed against the true causal logit: the
  odd term `sinφcosφ·(JM−MJ)` of the per-plane rotation average flips sign
  under time reversal, and for low-frequency rotary planes (≈29 of Llama's
  64, whose wavelength exceeds 64k on its rope base — a derived count, per
  the paper shelf) the phase never averages out — first-order, not noise. The naive A/B was then caught being **circular** (it scored
  frozen-W against frozen-W's own objective, and was suspended, explicitly);
  the third instrument — true masked causal logit error, pinned by the RoPE
  composition identity `(R_t q)ᵀ(R_s k) = (R_{t−s} q)ᵀ k` — returned
  **rotated-W better by 1.20× (min 1.13), consistently in sign**. "Rotated
  REQUIRED" for the deployment refit rests on that non-circular number, and
  all `logit_rope` figures are scoped program-wide as frozen-instrument
  (fair codec-vs-codec, not the deployment metric).
- **#9, the int8 decoder certificate.** The int8 roundtrip perturbation
  `Δdec` is deterministic, so the added weighted distortion is an **exact
  offline certificate**, not a bound:
  `added = Σ_i λ_i‖encᵀΔdec[:,i]‖²` against `payload = Σ_i λ_i 4^{−b_i}`.
  It killed blanket int8 (implied degradation to 54%; measured 13.5–16.7%
  vs the 5% gate) and rescued it tier-gated: int8 only for columns with
  bits ≤ T. T=5 certifies at both budgets; the per-layer variant
  (`int8_tl`) ships. Measured across the program: certificate conservatism
  5.3–8.6×; T-*ordering* exact (0/96 ordering violations); conservative as
  a bound on 87/96 cells (layer 2 runs anti-conservative up to ~3×,
  absorbed by the 5–10× shipped-threshold margin and stated wherever the
  certificate is used) — repeated validations of the "cheap analytic
  instruments" pattern (settle distortion questions on pack-side algebra;
  gate GPU spend on the answer).
- **#6, the transfer ceiling as a theorem.** Gate A's structural 0.56–0.64
  became the determinant–Jensen anchor: the pooled water level is unbiased
  (`E_s[D_pool] = C·GM(λ̄)·4^{−B̄}`, exact — basis misalignment costs zero
  in expectation), so retention is exactly
  `R = E_s[det^{1/C}(Σ_s)]/det^{1/C}(Σ̄) ≤ 1` (Minkowski concavity) — a
  population functional, hence corpus-size-independent, exactly the
  measured signature. Debiased for the asymmetric Wishart log-det bias
  (digamma closed form), gpt2 lands at 0.586 inside the predicted band; the
  residual identity gap decomposes exactly into the AM/GM gap of the tier
  grid. The honest negative became a predictive quantity.

**The honest negatives of this era, each with a mechanism** — charge-aware
allocation (the plain budget knob already walks the same (c_used, bpe)
locus); LW/OAS shrinkage (additive shrink-toward-mean lifts near-zero tails
by orders *relative to themselves*, exploding c_used in the log-scale
waterfill; no small-n rescue); the Gaussian-Lloyd K-side codebook (blanket
fails 0.25/0.22 vs the 1.02 bar — at ≥5 bits per-group MSE-scale RTN beats
a fixed codebook even on pure N(0,1), and the metric-dominant top directions
are sub-Gaussian, kurtosis 2.58; the certified per-tier mix is confined to
~0.02% of the λ-metric — immaterial). Two brain-dig rounds killed the
remaining candidate levers with numbers (sign/ternary tiers, W re-weighting,
truncation, entropy coding — the last foreclosed by paged random access,
with the codebook fix then measured and declined). The corpus-transfer gate
at gpt2 scale delivered its H2/H3 verdicts (basis does NOT transfer across
domains — hybrid recovery 0.43–0.57; the exploitation lever is whole-pack
per-domain fitting) and the then-current token-marginal reading — later
scale-scoped by the rental (Part VIII).

The era closed with the **locked refit config** — `w_rope="rotated"`,
`lam_alloc=None`, `payload_quant="rtn"`, `dec_quant="int8_tl"` — and a
pre-registered rental queue (`2026-07-25-vm-rental-queue.md`): "every
pre-rental science question is CLOSED."

---

## Part VIII — The rental: the evidence batch (2026-07-25/26)

One GH200 rental executed the entire queue plus a user-approved overnight
extension — 14/14 stages green, ~29 h, GPU busy 88% of samples. The full
record with verified appendices is `2026-07-26-gh200-rental-results.md`;
the junction-level outcomes:

- **The Triton merge gate passed** (full CUDA suite green; real-model
  parity; fused-path probe; 0 argmax flips over 64 steps at 64k) — with one
  pre-registered gate amended mid-rental: the packed-vs-streaming logit
  probe tripped on a *deterministic near-tie flip* (streaming top-2 gap
  0.211 vs drift delta 1.59 on a random-token prompt) that the duel doc had
  itself predicted ("diverges probabilistically… merge gate wording must
  change"); the amendment (fail only on drift-inexplicable flips, or >N/8)
  was forensically justified, user-approved, and the rerun passed with the
  same flip correctly classified WARN.
- **The shipped recipe is quality-neutral and cheaper than its own
  predecessor**: LongBench macro 40.85 @ 3.081 mean bits — +0.13 over the
  fp32/frozen pack at **−0.72 bits** (the int8_tl accounting win as one
  number) and **+0.48 over tq_b3 at −0.125 bits**, the edge living in
  synthetic (+3.36) and code (+1.40).
- **NIAH is an honest null on Llama** (all arms within codec-RNG noise of
  fp16 at 32k/64k over 5 seeds — parity at fewer bits IS the claim), with
  a real harness finding: the seed reseeds only codec RNG, not the needle
  (fp16 bit-identical across seeds) — stated as a limitation.
- **Two-model replication across the board**: int8_tl certifies and
  measures safe on Qwen3-8B (0.55–0.90% vs the 5% bar, both models;
  certificate conservatism 5.3–8.6×); the Jensen anchor lands debiased
  0.684 *inside* the gpt2 band with the identity matching; the calibration
  ladder passes G1 from **a single 2048-token cache** through nc=8 on both
  models.
- **The corpus story sharpened in both directions**: the gpt2
  token-marginal claim REVERSES at 8B on both models (shuffled-order
  calibration is *worse* than cross-domain — word order matters at scale)
  while the cross-domain penalty roughly halves; and the **trigram
  count-table recipe CONFIRMS on both models** (D_tri 0.036–0.074 < 0.10),
  with the ladder self-terminating at order 3 under its own pre-registered
  both-sides rule. Packs can be calibrated from n-gram statistics alone.
- **Systems, measured**: 128k resident 50.48 GiB (k4 chunked) vs 63.30
  (fp16), packed NIAH healthy at 128k; k2b 128k dense *fits* post-mask-fix
  (83.31 — correcting a stale expected-OOM).
- **A candidate differentiator, properly hedged**: the TurboQuant-family
  arms collapse on Qwen at 32k (tq_b3 → 2.60 vs fp16's flat 10.00, uniform
  across depths, all seeds) while k4 holds parity — real within this model
  pair, mechanism open, n=1 pair.
- **Ops honesty**: two orchestration bugs (bash's `set -e` suppression
  inside tested compound commands; `sys.path[0]` vs `import experiments`)
  cost ~25 minutes and are documented with fixes; one stage checker
  false-negative was overridden with evidence; every result was committed
  per-stage and bundle-merged with full provenance.

---

## Part IX — The theory spine, assembled

The mathematical through-line, in dependency order — each item measured or
proved within the program, each carrying its pointer:

1. **The 4^-b floor** (rate-distortion of b-bit quantization) → the
   **break-even inequality ε > 1 − 4^(−Δb)** → the **frontier law** (stable
   rank ∝ width cancels the loosening threshold; weights compressibility
   scale-invariant; payers = axis-aligned, absorbed free). [Part I docs]
2. **The margin transfer**: the same instrument scores cache K at
   +0.9–2.5 bits — structure lives in activations. Pre-RoPE storage because
   rotation smears subspace structure; bits belong to K (2× sensitivity);
   logit-metric doctrine because Frobenius inverts under rogue channels.
   [K1/K2/K2b]
3. **The per-sequence basis charge** as the KV-side break-even: eigenbasis
   wins 2.24× but +8 bpe kills it; therefore **amortize the basis to the
   model level** — the K4 move. [June trilogy → K4 spec]
4. **The whitening identity** `‖W̃^{1/2}(k−k̂)‖ = ‖y−ŷ‖` (exact) makes
   weighted-KLT + reverse-waterfill the textbook-optimal
   transform+allocation for the measured metric; GQA pooling and per-head
   block-diagonality exact; allocation optimality via Everett/Lagrangian
   with duality gap ≤ κ·Δmax/C ≈ 0.002 bpe. [K4 spec, math review #1]
5. **The instrument-vs-deployment scoping**: the frozen-W metric is exact
   for itself and provably time-reversed against the causal logit; the odd
   rotary term is first-order on low-frequency planes (≈29/64 at 64k);
   the non-circular causal instrument decides — rotated-W by 1.20×.
   [math review #3, actions §B]
6. **The certificate calculus**: deterministic perturbations make added
   weighted distortion exactly computable offline
   (`added = Σ λ_i‖encᵀΔdec[:,i]‖²`) — kill blanket int8, ship tier-gated
   int8; conservatism 5.3–8.6×, ordering exact. The pattern generalizes:
   settle distortion questions on pack algebra, gate GPU spend on the
   answer. [math review #9, estimation levers, rental §6]
7. **The transfer ceiling as geometry**: retention =
   `E_s[det^{1/C}Σ_s]/det^{1/C}Σ̄ ≤ 1` (Minkowski) — corpus-size-
   independent, Wishart-debiased into its band at both scales, residual =
   the tier grid's AM/GM gap, exactly decomposed. Related: (1/C)·logdet
   gaps bound a sequence-identity mutual information — the ceiling is
   intrinsic. [math review #6, local levers, rental appendix-corpus]
8. **The declination ledger** — every alternative killed with a number and
   a mechanism: unbiased coding (aggregation never repays 2× variance in
   this regime), Lloyd codebooks (shape-matched RTN wins where the metric
   lives), shrinkage (log-scale waterfill amplifies relative tail lifts),
   charge-aware allocation (budget knob walks the same locus), entropy
   coding (random access forecloses it), sign/ternary tiers, W
   re-weighting (triangular is AT the measured peak). [Parts III, VII]
9. **The calibration theory**: corpus enters through second moments;
   at 8B, word order matters (the token-marginal reversal, two models);
   n-gram sufficiency climbs a pre-registered ladder that self-terminates
   at order 3 under the insensitivity bar; estimation is not the bottleneck
   (nc=1 suffices; shrinkage has nothing to rescue). [corpus-transfer,
   trigram climb, calibration ladders]

---

## Part X — Where it stands (2026-07-26)

**The shipped artifact**: `k4_b2.5_dec8tl` — corpus-calibrated rotated-W
spectral keys with tier-gated int8 decoders + per-head rotate/Lloyd values
@2b — measured on two 8B models: LongBench macro 40.85 @ 3.081 bits (above
the strongest TurboQuant baseline at fewer bits), NIAH parity with fp16
through 128k on the packed path, decoder cost <1% of the distortion budget,
128k resident memory 20% under fp16, calibratable from one 2048-token
general-text cache or from trigram count tables alone.

**The claims discipline that carries it**: matched-bits accounting with
every metadata bit priced; context-dependent bpe with certified crossovers;
delta-parity licensing for cross-harness comparison; the strongest version
of the competing method (our own asymmetric K3/V2 steelman) in every table;
seed bars labeled for what they are; every superseded number carrying its
banner.

**Open, deliberately**: the branch merge (gate green, decision with the
user); paper drafting (all results in; shelf + chronicle + rental doc are
the inputs); the Qwen TQ-collapse mechanism; a needle-reseeding NIAH knob;
the wider batched-128k co-residency sweep; hybrid-attention architectures
as a scoped future project (linear layers have no growing KV; unified-KV
dissolves the K/V split — the trend that *motivates* KV compression also
bounds where this codec applies).

The program's method, in one line: **pre-register the gate, measure the
kill, keep the instrument** — five programs in, every pivot in this
chronicle was forced by a number that is still in the repo.
