# STORM briefing — alternate KV-compression mechanisms (2026-07-26)

Six-lens grounded brainstorm (STORM protocol) over the personal-brain vault
(foundational texts primary) plus the bmx measured record. Lenses: information
theory, circuits/function, high-dimensional geometry, systems/memory-hierarchy,
baseline spine, adversarial skeptic. Web forbidden; every claim cited to
vault file:line or the bmx record; unanswerable questions recorded as gaps.
This is a brainstorm record, NOT a results doc — nothing here is measured
beyond what the cited record already contains.

## The convergent verdicts

1. **The admissibility law** (all lenses, independently): any new mechanism
   must (i) price side-information to the MODEL level (zero per-sequence
   charge — the eigenbasis +8 bpe lesson), (ii) preserve paged random access
   (stream/context-dependent codes foreclosed — "metadata you must move is
   not compression"), (iii) bound error in the ATTENTION OUTPUT, never
   inject per-key noise into softmax (the unbiased-sketch lesson). The
   declination ledger is this law applied.
2. **The one unplayed axis is token/row SELECTION** (baseline: the only
   untouched family, composition blank in both records; systems:
   per-cache-type policy is the dominant capacity lever; geometry: measured
   low stable rank puts keys in the regime where the effective-rank phase
   transition makes row-dropping cheap, and a coreset carries NO basis
   charge). Constraint from circuits: sinks always-resident; role-aware
   importance.
3. **Twin unification** (circuits + info-theory): the program's statistical
   doctrines have exact mechanistic/information-theoretic identities —
   bits-belong-to-K = K feeds only the QK circuit (rank ≤ d_head);
   the logit metric = the Information Bottleneck task term; the weighted
   KLT = the Gaussian-IB closed form. Paper-theory strengthening; zero bits.
4. **Per-entry coding is at its floor** (systems, info-theory, geometry):
   decoder side information is worth the ~2–3× already measured and is
   otherwise spent/harvested/foreclosed; sketching is pre-killed by the
   random-rotation-ties control; the leverage moved to placement,
   recomputation, and selection.
5. **The sharpest surviving challenge (skeptic): the metric question.** The
   objective is squared logit reconstruction; the task is a 0/1 retrieval
   event. Three of our own anomalies are fingerprints of the gap (near-tie
   NIAH flips; the Qwen TQ-collapse hiding behind sane logit metrics;
   "ppl too coarse"). Cheaply testable on banked parquets.

## Ranked mechanism portfolio

Tier 1 — offline, GPU-free gates on banked artifacts:
1. Per-event retrieval metric audit: re-rank all arms by argmax-flip /
   needle-event error; Spearman vs the logit-MSE ranking. Could change what
   "quality" means in every table; may explain the Qwen collapse.
2. Leverage-score token coreset (+ sinks) vs uniform quantization at
   matched bits on the existing logit instrument (≥90%-of-layers rule).
3. Sink carve-out audit: distortion-budget share of positions 0–3 (we
   handle sinks nowhere).
4. Head-role regression: does weight-readable role explain the 35×
   per-layer sensitivity spread?
5. Prompt-policy pack robustness: cross-score packs across the
   anchor-forensics raw/chat cache sets (the 43.7-pt axis).

Tier 2 — cheap algebra, likely honest nulls worth closing: oracle-vs-average
query gap (second-moment sufficiency predicts ≈0); residual predictor after
KLT; h-cache / MLA-latent accounting spreadsheet (challenge mostly dissolved
on pre-RoPE+GQA arithmetic; no MLA number held); vMF radial codec check (its
banked value: the tangent-normal decomposition explains the Lloyd-gate
negative — the sphere Gaussianizes the tangential bulk); W_OV read-subspace
V coding (predicted GQA-null); break-even blind-spot audit (the instrument's
taxonomy was reopened twice — multiplicative scales, then the deterministic
int8 perturbation; enumerate remaining unpriced correction classes).

Tier 3 — big swings, gate hard: two-tier cold-2bit-HBM + exact-host
speculative prefetch (gated on one-step attention predictability, measurable
offline); superposed/consolidation probe (MacKay Hopfield capacity 0.138·N
with the N·log N ORDER cost — which the 8B order-reversal independently
re-derived; consolidate one flat-spectrum layer at matched bits vs needle
recall); computed-shortcut replacement of convolution-like heads (highest
ceiling, highest retrieval risk); sequence-axis stacking (breaks the
frozen-model contract; belongs with the hybrid-attention future project).

## Self-critique (per the STORM protocol)

Single-source leans: the head-role machinery rests on one WIKI note
(Q-K Concentration) — compiled-layer, re-derive from weights before shipping
anything; systems capacity data (Mooncake/DeepSeek) single-wiki-note; the
tangent-normal Lloyd explanation single-foundational-source (but matches our
measured kurtosis independently). Analogical transfers, not vault theorems:
MacKay's per-event p_b rate-distortion (channel-coding machinery) → the
0/1-retrieval objective; the effective-rank phase transition → token
sampling (the vault has no coreset/leverage theorem). Missing lens:
training/optimization (compression-aware finetuning). Convergence-bias
check: all lenses read the chronicle, so "selection is open" partly
inherits from it — but three lenses justified it on independent grounds;
rated genuine. Vault ingestion candidates: Wyner–Ziv; coresets/leverage
scores; MLA/latent caches; a Hopfield↔KV bridge; per-event distortion;
tier-miss latency vs the ~20 ms decode budget. One vault-note correction:
"KV errors don't accumulate" is implementation-conditional (the measured
98× blowup; true only under write-once storage).

## Takeaway

Per-entry coding is at its floor and defended by an explicit admissibility
law. The remaining order of magnitude lives on two never-priced axes —
WHICH tokens to keep, and WHAT quality means (per-event retrieval vs
squared logits) — and the second gates the first. Both have offline,
GPU-free kill-or-confirm gates reusing shipped instruments.
