# K4 corpus-transfer gate — design (mechanism + exploitation)

**Question (from `docs/user-notes.md`, 2026-07-23, verbatim):** "does the corpora
used affect the accuracy of the pack file? does using a coding corpus make it
better at coding tasks, or will using an english text one make it worse?
basically, are these moments computed independent of the tokens it sees?" Plus
the follow-up: **why** does the observed (in)sensitivity occur, so it can be
exploited — the program goal is the optimal packing, not just a robustness
checkbox.

**Status of the question today:** the calibration corpus is a single hardcoded
choice (`load_eval_tokens` → wikitext test split, offsets as pseudo-documents);
every shipped pack is wikitext-fit. Existing indirect evidence: (a) wikitext-fit
packs reach LongBench **code parity** with b3 (60.43 vs 60.02, CI [−2.21,+3.06])
— packs never saw code and code didn't collapse; (b) Gate A: within-corpus
per-sequence retention 0.56–0.64, structural (3× corpus → 0.61–0.69) — sequence
shift already costs ~40% of oracle win and the pooled basis still wins 6.4×/4.5×
heldout; (c) Gate B: query-weighting transfers at 0.85–0.88. Missing: the
explicit fit-corpus × eval-corpus measurement and the mechanism decomposition.

**Kill-or-confirm framing:** confirm-insensitive = cross-fit costs little → the
referee answer with numbers ("corpus choice is second-order; here is the
matrix"). Kill = strong domain sensitivity → not a loss: it converts to the
domain-adapted-packs lever (per-domain packing beats one-size). Either verdict
feeds the paper; the hybrid arm tells us WHICH lever to pull.

## 1. Where the corpus enters the math

The pack pipeline per layer: `Σ = E[k_pre k_pre^T]` (key second moment, corpus
average), `W` (query second moment → weighting), basis = `eig(W^½ Σ W^½)`,
`lam` (eigenvalues) → reverse-waterfill over dirs+layers → tier map. The corpus
enters ONLY through the two second moments. Decomposition hypotheses
(pre-registered):

- **H1 (model-intrinsic top):** the retained top-tier subspace is dominated by
  input-agnostic structure — the mean/rogue-channel/attention-sink component of
  `Σ = μμ^T + Cov(k)` — so top-rank subspace overlap across corpora is high.
  (Prior: massive-activation / attention-sink literature — these are
  weight/norm artifacts, present for any input; vault pass in the writeup.)
- **H2 (tail sensitivity):** the corpus signal concentrates in the
  low-eigenvalue tail → cross-corpus divergence grows with eigen-rank, and
  tier-map disagreement concentrates in the LOW tiers (0/2-bit boundary), not
  the top tiers.
- **H3 (exploitation — "basis transfers, allocation adapts"):** basis from
  corpus A + `lam`/waterfill recomputed on corpus B recovers ≥0.9 of the
  full-B-fit win. If true, optimal packing = ONE shared basis + per-domain tier
  maps (a per-domain artifact of ~3·C bits, no basis refit) — the deployment
  lever. If false and full-B-fit wins big, the lever is whole-pack
  domain-fitting instead.

## 2. Corpora (3 fit sources, 2 eval sides)

- **wikitext** (`Salesforce/wikitext`, wikitext-2-raw-v1 — today's default;
  fit and heldout slices by token offset, disjoint, seeded).
- **code**: `bigcode/the-stack-smol` (Python subset), deliberately NOT
  LongBench text — no eval circularity. Same slicing discipline.
- **shuffled-token null**: wikitext token stream randomly permuted (seeded).
  Preserves unigram/embedding statistics, destroys syntax/semantics/context.
  Separates "any plausible tokens through the model" from "structured content":
  if shuffled-fit ≈ wikitext-fit, `Σ` is dominated by token-level +
  model-intrinsic geometry, not contextualization. Fit-side only (nothing is
  evaluated ON shuffled text).

Eval side: held-out wikitext slices and held-out code slices.

## 3. Cells and diagnostics

**Win matrix (primary):** fit ∈ {wiki, code, null} × eval ∈ {wiki-held,
code-held} → G1-style win (tail logit distortion vs the per-layer
`turboquant_mse` curve interpolated at the SAME skeptic bpe — the existing
`_k4_common` machinery, no new metric). Matched cells (fit==eval domain) are
the reference for each column.

**Hybrid arm (H3):** basis(wiki) + alloc(code) scored on code-held, and
basis(code) + alloc(wiki) scored on wiki-held — via the `pack_from_basis` path
(basis fixed, `lam` measured on the alloc corpus, waterfill rerun). Compare to
the full-fit matched cell.

**Mechanism diagnostics (H1/H2):**
- Per-rank subspace overlap: principal angles between fit-A and fit-B retained
  subspaces as a function of rank cutoff (top-8, top-16, … C_used), per layer.
- Centered vs uncentered: repeat the basis fit on `Cov(k)` (mean removed) vs
  `E[kk^T]`; if cross-corpus agreement drops when centered, the shared part is
  the mean/rogue structure (H1 sharpened).
- Tier-map agreement per tier level (agreement among top-tier dirs vs the
  0-vs-2-bit boundary), plus eigenvalue-spectrum overlay per layer.
- Cross-corpus retention: Gate-A retention machinery pointed across corpora
  (A-basis quality evaluated under B's covariance).

## 4. Pre-registered verdict rules

- Primary: relative cross-fit degradation `D = 1 − win(fit≠eval)/win(fit=eval)`
  per (fit, eval) cell, mean over heldout caches; min/max as error bars.
  **D < 10% → corpus-insensitive; D > 25% → domain-sensitive; between →
  reported as measured.** (Scale context: Gate A's sequence-level ceiling
  already costs ~40% of oracle — corpus shifts below that are second-order.)
- Null-fit expected to degrade substantially MORE than natural cross-fit; if
  null-fit ≈ wikitext-fit, the basis is (nearly) purely model-intrinsic — a
  stronger claim than insensitivity, report prominently.
- H3: hybrid recovery `win(hybrid)/win(full matched fit) ≥ 0.9` at the same
  budget confirms "basis transfers, allocation adapts".
- All at gpt2 scale = MECHANISM verdicts (gpt2 yellow flag: corpus-W retention
  ~0.47–0.52 — stated in every table caption). Llama fit-side replication rides
  the next rental before any paper claim.

## 5. Deliverables

1. `load_eval_tokens` generalized (dataset id/config/split/text-field params;
   defaults reproduce today's behavior byte-exactly; shuffle as a seeded flag
   at the token level). `collect_cache.py` grows `--dataset`/`--shuffle-seed`
   passthrough + output naming that encodes the corpus.
2. `experiments/k4_corpus_transfer.py` — thin tyro harness: loads/fits packs
   per corpus (or loads prefit pack files), emits `metrics.parquet` (win
   matrix + hybrid arm), `overlap.parquet` (per-rank/per-layer diagnostics),
   `corpus_transfer_verdict.json` (the §4 rules). Artifacts under
   `results/k4_corpus_transfer/<run-id>/` with config+env+SHA.
3. gpt2 cache collection for code + shuffled corpora (local CPU, minutes) —
   `results/cache/` names encode corpus; NOT committed (regenerable).
4. Results doc `docs/2026-07-<day>-k4-corpus-transfer-results.md`: the matrix,
   the diagnostics figures/tables, the WHY section (analytic decomposition +
   vault-grounded prior on massive activations / attention sinks / rogue
   channels), both verdict templates pre-drafted.
5. VM addendum (pre-registered now, rides the next rental): Llama-Instruct
   fit-side replication (fit code/null packs from VM-collected caches, same
   matrix, no task evals needed) + OPTIONAL one LongBench-code probe cell with
   a code-fit or hybrid pack if H3 confirms (n=100, paired vs the wikitext-fit
   arm).

## 6. Non-goals

- No new codec or spec fields; no change to shipped duel packs or parquets.
- No task-level headline from gpt2 (mechanism scale only).
- No third natural corpus, no multilingual sweep (YAGNI until the 2-corpus
  matrix motivates it).
- Cross-layer/shared-basis questions stay in the pack-charge plan's Lever-3
  future-work paragraph — this experiment shares machinery but answers a
  different question (across corpora, not across layers).

## 7. Constraints

- All the repo's hard rules (commit gates, battery, tyro conventions, explicit
  run selection, fp32 experiments, deterministic seeds).
- No web search (HF dataset downloads are fine); tiny offline models only in
  tests — the experiment itself uses real gpt2 locally (existing artifacts).
- `load_eval_tokens` default path must stay byte-identical (pinned by reusing
  existing collected caches unchanged; new corpora get NEW cache files).
