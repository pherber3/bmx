# The program map — the view from altitude (2026-07-26)

Prompted by the user's observation that the kill-or-confirm cadence is
depth-first by construction: every gate opens the next gate on the SAME
branch, and nothing in the method ever forces a breadth review. This doc is
that review — the full territory given everything the program now holds,
including branches never opened. It is a decision map, not a results doc.

## 0. What we actually hold (the assets, ranked by durability)

1. **A validated methodology** — kill-or-confirm gates, matched-bit honest
   accounting, cheap analytic certificates gating GPU spend, pre-registered
   verdicts, the declination ledger. More durable than any single result;
   transfers to any tensor-compression question.
2. **General instruments** — the break-even inequality ε > 1−4^(−Δb); the
   query-weighted logit metric + whitening identity; the certificate
   calculus (deterministic-perturbation distortion, exact offline); the
   determinant-Jensen transfer-ceiling anchor.
3. **A publishable headline result** — k4_b2.5_dec8tl: above the strongest
   TurboQuant baseline at fewer bits, two models, 128k systems numbers,
   full provenance (the rental batch + chronicle).
4. **Scientific findings that transcend the codec** — the frontier law
   (weights compressibility is scale-invariant); the order-reversal at 8B
   (calibration is order-sensitive at scale, two models); TurboQuant-family
   architecture-fragility (Qwen collapse) vs K4 robustness; the trigram
   calibration sufficiency (packs from count tables).
5. **Infrastructure** — streaming/packed caches, the Triton decode kernel,
   NIAH/LongBench harnesses, the VM discipline. All merged to main.

## 1. The branches (the full map)

**A. PUBLISH (the current branch's endpoint).** The K4 paper from the shelf
+ rental doc + chronicle. Value: high and TIME-SENSITIVE (the only branch
with decay — the field moves; priority matters for every other branch too,
since published results anchor them). Effort: known, all inputs in hand.
Risk: low. One live dependency: branch C's metric audit could change what
the quality tables mean — it costs hours and should run BEFORE the tables
are finalized, not after.

**B. DEEPEN THE CODEC** (the storm's output — still the same hill). The
selection axis + Tier-1 gates. Value: a second result or a stronger
camera-ready; every gate is offline/GPU-free. Risk: tangent-pull — this is
exactly the depth habit; timebox it to the gate battery, don't let it delay A.

**C. THE METRIC QUESTION as its own contribution.** Per-event (0/1
retrieval) evaluation of KV codecs vs the field's reconstruction-style
metrics. The skeptic's strongest survivor; our own anomalies (near-tie
flips, the Qwen collapse hiding behind sane logit numbers) are its
motivating evidence. Value: potentially field-level (an evaluation-critique
paper that also strengthens A); cheap to scope. This is the one branch that
FEEDS BACK into A and should be partially executed first (the audit, not
the paper).

**D. THE DEPLOYMENT ARTIFACT.** vLLM/sglang integration; shipped
general-text packs; the calibration-as-a-service story (trigram recipe).
Value: adoption/impact, answers the "not everyone wants to build this"
question. Effort: engineering-heavy, maintenance treadmill. Not a paper.
Post-A.

**E. HYBRID ATTENTION (already scoped as a separate project).** K4 on
DeltaNet/sliding/unified-KV architectures — needs a capture-layer redesign
spec. Value: forward relevance (the architecture trend both motivates KV
compression and bounds this codec; our own cross-model finding proves
codec×architecture interaction is real). Post-A; spec-first.

**F. THE INSTRUMENT EXPORTED (the genuinely separate paths).** The
byte-equation predicate that selected the KV cache — large, growing,
byte-bound, structured — is satisfied by other objects the program has
never priced:
   - **Cross-request prefix/KV reuse** (the serving industry's live
     problem; our packs + selection machinery apply directly to shared-
     prefix stores — Mooncake-class systems);
   - **Vector databases / retrieval indexes** (the coreset/consolidation
     axis IS vector-DB science; the logit-metric doctrine maps to
     retrieval-event fidelity);
   - **Optimizer states & gradient communication** (training-side, 2×
     weights, the break-even instrument applies verbatim);
   - **Activation checkpoints** (the recompute-vs-store frontier we
     touched in Part IV);
   - **Agent long-term memory** (the superposed-storage/Hopfield thread —
     order-costly but reachable; our order-reversal finding is directly
     relevant);
   - **Multimodal/diffusion caches** (unpriced entirely).
   Value: possibly the largest long-term branch — "a general lossy-
   compression science for ML tensors" is a research program, not a paper.
   Each sub-branch needs its own K1-style census before commitment.

**G. THE SCIENCE FINDINGS STANDALONE.** Short notes/papers: the
order-reversal (calibration theory at scale); TQ architecture-fragility
(quantizer robustness across model families — needs the mechanism probe
first). Cheap, post-A or folded into A.

**H. STOP/CONSOLIDATE.** The record is complete, licensed, share-ready.
Harvesting (A) and choosing deliberately among B–G is itself a valid
endpoint; more depth on any branch without choosing is the failure mode
this map exists to prevent.

## 2. The decision structure

- **A gates everything** and is the only time-decaying branch: publish
  first. Priority anchors B–G.
- **C's audit (hours, banked parquets) runs BEFORE A's tables freeze** —
  it is the single cheap item that could change the paper's meaning.
- **B is timeboxed to its offline gate battery**, runnable in parallel with
  drafting A without GPU or scope creep; its results land in A's
  future-work or become the next cycle AFTER A ships.
- **D, E, F, G are post-paper portfolio choices** — to be picked
  deliberately, from this map, not entered by momentum.

## 3. The honest self-observation

The program's method optimizes descent: every negative sharpened the same
attack. That was correct while the hill was unclimbed; the hill is now
climbed (A is in hand). The next unit of effort buys more on breadth —
publishing, the metric question, or an F-branch census — than on another
codec lever. This map should be re-drawn whenever a branch completes, and
any new "one more cycle" proposal should name which branch it serves.
