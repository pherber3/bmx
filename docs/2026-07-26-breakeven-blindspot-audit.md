# Break-even blind-spot audit — is the correction-class taxonomy closed? (2026-07-26)

Storm-gate Task 7 (`docs/superpowers/plans/2026-07-26-storm-gates.md` §Task 7).
A **theory page**: no experiments, no code. The question is whether the
break-even instrument's own taxonomy of "corrections" is complete — the
instrument prices exactly one currency (additive fp16 side-information), and
the program has twice discovered a correction class it does *not* price
("multiplicative scales, then the deterministic int8 perturbation" —
`docs/2026-07-26-storm-kv-mechanisms-briefing.md:64-66`). This page enumerates
every shipped class, then every unshipped candidate sharing the
outside-the-instrument reasons, and gates on whether any untested free-currency
class survives.

## 1. The instrument, exactly

`ε > 1 − 4^(−Δb)` — spending Δb bits/entry on **additive** side-information
pays iff the energy fraction ε it removes clears that bar; derived from the
b-bit quantization floor `D ∝ 4^{−b}` (each bit quarters MSE), so a bit spent
on side-info competes against the same bit spent on the bulk quantizer
(`docs/2026-06-11-lrs-results.md:146-156`; `src/bmx/quant/breakeven.py:1-8,24-28`).
What it prices: an **additive** correction term charged in **stored bits**
against the `4^{−b}` bulk alternative — for fp16 rank-r factors
Δb = 16·r·(1/m + 1/p) (`breakeven.py:40-43`); for a sparse spike list
Δb = k·(16 + ⌈log₂n⌉)/n (`breakeven.py:47-55`). What it is **silent** about,
by construction: any correction that (i) is not additive-in-the-reconstruction,
or (ii) costs zero stored bits, or (iii) is deterministic and does not obey the
`4^{−b}` law. Every "reopening" below is a class that trips one of (i)–(iii).

## 2. Shipped correction classes — priced-natively vs outside-the-instrument

| # | class | shipped where | verdict | one-line reason |
|---|---|---|---|---|
| a | additive low-rank factors (L) | `codecs.py:89-91` `factor_bits`; `_lowrank_split` `codecs.py:677-695`; lowrank_rtn_channel/k2b | **priced natively** | the instrument's original subject — additive term, fp16-charged at 16·r·(S+C)/(SC); faces `4^{−b}` directly (`lrs-results.md:146-156`) |
| b | multiplicative per-channel scales (AWQ move) | groupwise RTN scale `codecs.py:79-81`; per-row/per-head norms `codecs.py:84-86` | **outside** | absorbed into scale params that already exist (or into `o_proj`), **zero marginal bits** — the original "free currency" discovery (`frontier-breakeven.md:72-81`; `lrs-results.md:157-166`) |
| c | basis rotations, seeded | `_rotate`/`_unrotate` `codecs.py:104-139`; rotate_rtn / turboquant Hadamard | **outside** | orthogonal ⇒ inner-product-neutral; seed-generated ⇒ **zero stored bits**; never faces the hurdle (`lrs-results.md:80-82`; `frontier-breakeven.md:76-79`) |
| d | corpus-amortized model-level bases (K4 enc/dec) | `SpectralPack` `spectral.py:174-206`; `skeptic_charge` `spectral.py:879-910` | **partially priced** | per-**sequence** charge IS priced (16·C/S amortization, `spectral.py:20-22`); per-**model** charge priced by nothing — legitimate stance, argued in §2d |
| e | deterministic int8 decoder perturbation + certificate | `int8_decoder_roundtrip` `spectral.py:913-943`; `int8_decoder_certificate` `spectral.py:962-992` | **outside** | a deterministic, **non-`4^{−b}`** distortion channel; needed its own exact calculus `added = Σᵢ λᵢ‖encᵀΔdec[:,i]‖²` — the existence proof the taxonomy was not closed (`chronicle` Part VII #9, l.369-378) |
| f | mean-centering / per-direction bias | math review #10a `2026-07-24-k4-math-review.md:407-415` (NOT shipped as a distortion lever) | **outside** | affine, model-level: a per-direction mean stored in the pack is 16·C bits total = **zero in every accounting mode** (`math-review:410-412`) |
| g | tier-gating maps / allocation tables | `tier_bits` `codecs.py:94-96`; payload-v2 `spectral.py:602-615`; mixed_dec_charge `spectral.py:1036+` | **priced natively** | metadata (⌈log₂n_tiers⌉/S per channel), counted in payload-v2 and the skeptic modes — a bit is a bit |

### 2d — the per-model basis charge: gap or stance?

The K4 decoder is charged per **sequence** at `16·C/S` (skeptic-v1) or
`16·c_used/S` (skeptic-v2), which amortizes to ≈0 at long context
(`spectral.py:18-31`; the crossover construction, `chronicle` l.310-316). The
one-time **per-model** cost of shipping the C×C basis with the checkpoint is
charged to **nothing**. This is the same accounting stance the whole
free-currency story rests on — a seed-generated rotation (row c) also "costs
zero bits" only because the *seed* ships with the model. The stance is
**legitimate, not a gap**, for one reason with a theorem behind it: the basis
is corpus-level, not sequence-level, and the transfer-ceiling theorem
(`chronicle` Part VII #6, l.361-368; retention = `E_s[det^{1/C}Σ_s]/det^{1/C}Σ̄
≤ 1`, Minkowski) proves the model-level basis is a **population functional** —
its cost does not scale with the served workload. A model-level artifact
amortized over the model's entire deployment lifetime is priced at zero on the
same principle that weights themselves are not re-charged per token. The honest
edge: this only holds because the basis genuinely transfers (Gate B passed at
0.85–0.88, `chronicle` l.303-305); if per-tenant refitting were required the
charge would move back onto a per-deployment axis. Stated as a scope condition,
not a free lunch.

## 3. Unshipped candidate classes sharing the outside-the-instrument reasons

Each candidate is a correction that trips (i) non-additive, (ii) zero-bit, or
(iii) deterministic-non-`4^{−b}` — i.e. would be invisible to
`ε > 1 − 4^(−Δb)`. Marked TESTED (with the kill/confirm cite) or UNTESTED (with
the cheapest pack-algebra test that would settle it).

| candidate | reason it's outside | status | cite / cheapest test |
|---|---|---|---|
| per-token magnitude / norm codes (multiplicative per-row) | multiplicative, but **is charged** | **TESTED — it is the V-side turboquant_mse codec** | `_turboquant_mse_perhead_packed` stores per-(row,head) norms fp16, charged `norm_bits(h,C)=16h/C` (`codecs.py:616-636,84-86,731-745`). NOT free currency — a per-token norm is S values, not absorbable into a fixed param, so it pays rent and IS priced. Confirmed the V winner (`chronicle` Part III, l.135-141). Blind spot closed by inclusion. |
| learned additive bias per direction (model-level affine) | affine, model-level, zero-bit | **UNTESTED as a distortion lever** | This is row (f)'s mean-centering, generalized. Measured only for *basis overlap* (K1 margin 2.46→2.41, `chronicle` l.120; corpus overlap −0.011 ≈2%, `2026-07-23-k4-corpus-transfer-results.md:93-95`); the **G1 distortion sweep row is explicitly deferred future-work** (`2026-07-24-k4-math-actions-design.md:104`; `math-review:415` "do it only as a sweep row"). **Cheapest test:** one G1 row — subtract per-direction pack mean μ before waterfill, feed σ² (not μ²+σ²) as `allocate_bits_from_variance` input, add 16·C to the model-level (zero-charge) side; compare weighted distortion at matched bpe. ~30 lines, no GPU, on cached moments. Prior says small (mean concentrates in top dirs that get 8 bits anyway, `math-review:412-414`) — but small ≠ measured. |
| per-position scale schedules (position-dependent multiplicative) | multiplicative, position-indexed | **UNTESTED** | No per-position scale has been fit; only per-position *rotation* (RoPE) is studied, and it smears rather than helps (`2026-06-12-k2c-results.md:22`). A position-indexed scale s(p) shared model-level (a fixed schedule, not per-sequence) is zero-bit. **Cheapest test:** fit s(p) = RMS of code-space rows at position p on the corpus, divide it out before quant, multiply back at decode; score `Σ_s e_sᵀW e_s` delta at matched bpe on cached moments. Expected null (autocorrelated tokens ⇒ near-flat s(p)), but the null is the deliverable and one pack-algebra pass settles it. |
| quantizer-step schedules across context | changes the quantizer, not additive | **PARTIALLY TESTED (adjacent), lever UNTESTED** | The *bit* schedule across context is the crossover story (bpe amortizes, `chronicle` l.310-316); the *step-size* schedule (finer steps for recent tokens at fixed bits) is not. Charge-aware allocation — the closest lever — was **killed**: the budget knob walks the same (c_used, bpe) locus (`k4-paper-shelf.md:71-72`). **Cheapest test:** subsumed by that kill for the bit axis; for a pure step-size schedule, one G1 row varying `mse_scale`/group by position band — but the group-scale is already per-group MSE-refined (`codecs.py:300`), so the expected margin is ~0. Low priority; note as covered-by-adjacent. |
| sign / permutation symmetries (seed-generated relabeling) | orthogonal/permutation, zero-bit | **PARTIALLY TESTED** | Rotation (a superset of signed-permutation) is the confirmed free-currency win (row c). The *1-bit sign tier* (adding a sign level to the grid) was killed — it raises c_used +0.059 bpe, never selected (`k4-paper-shelf.md:69-70`); but that is a *tier-grid* question (priced natively), not a symmetry. A pure zero-bit sign/permutation relabel to reduce distortion beyond what Hadamard already achieves has **no residual fuel**: Hadamard already equalizes per-coordinate variance (turboquant mechanism), so a further free sign flip is distortion-neutral by construction. **Judged closed by the rotation result** — no cheap test needed; the mechanism argument is dispositive. |
| low-precision ACCUMULATION formats (compute-side "free currency") | compute-side, not a stored-bits correction at all | **TESTED — and it is NOT free** | The fp16-dots lesson: k2b kernel dots use fp16 **operands** with **fp32 accumulate** (`2026-07-06-overnight-kernel-results.md:27-30`). Accumulation stayed fp32 deliberately; dropping it to fp16 is a *distortion* channel, not a currency — it would be a deterministic-non-`4^{−b}` perturbation exactly like the int8 decoder (row e), and would need the **same certificate calculus** to price, not the break-even instrument. **Not a break-even blind spot** — it is a compute-precision knob whose distortion is certifiable by the row-(e) machinery, already the shipped pattern (settle on pack algebra, `chronicle` #6 l.375-378). Cheapest test if ever pursued: reuse `int8_decoder_certificate`'s closed form with Δ = the fp16-accumulate rounding op. |
| implicit free parameters in the shipped codecs (audit sweep) | various | **checked — none unpriced found** | Read of `codecs.py`/`spectral.py`: seed (zero-bit, row c), group size (fixes scale_bits, priced), ridge floor (`spectral.py:145-165`, deterministic conditioning knob — a *whitener* param, model-level, its distortion effect is the row-(e)-style certifiable kind and bounded κ(W̃)≤ridge^{−1/2}, `k4-paper-shelf.md:90-91`), mse_scale (per-group MSE-optimal step, priced via the scale it produces). No free additive term escapes accounting. |

## 4. Gate verdict

**Gate (pre-registered):** any untested free-currency correction class found ⇒
spec its cheap test; none ⇒ the "taxonomy closed" claim gets its first positive
argument.

**Verdict: ONE untested free-currency class survives with a cheap test spec'd —
the learned additive bias / mean-centering lever (row f / candidate 2).** It is
affine, model-level, zero-bit in every accounting mode, and its distortion
effect on the *shipped* metric has been measured only for basis overlap, never
as a matched-bpe G1 distortion row (`2026-07-24-k4-math-actions-design.md:104`
lists it as deferred future-work; `math-review:415`). Its cheapest test is the
~30-line offline G1 row in §3. The taxonomy is therefore **not yet closed** —
but the gap is narrow and low-expected-effect (prior: mean concentrates in the
8-bit top directions), so this is a *close-the-ledger* row, not a live avenue.

Every other candidate is disposed: per-token norms are shipped-and-priced (V
codec); per-position scales and step schedules are UNTESTED but expected-null
and one pack-algebra pass away; sign/permutation symmetry is closed by the
rotation result's mechanism; low-precision accumulation is a certifiable
distortion channel (row-e machinery), not a currency the break-even instrument
was ever meant to price. The instrument's silence has been mapped to its three
exact causes (non-additive / zero-bit / deterministic-non-`4^{−b}`), and only
one class in that shadow remains unmeasured.

## 5. Self-check — grounding of every row

Every class row in §2 and §3 cites a doc-line or file-line for its
shipped/priced/tested status. Rows I could not ground to a *measurement* (as
opposed to a mechanism argument) are flagged here honestly:

- **§2f mean-centering "outside"** — grounded to the *lever's non-charge*
  (`math-review:410-412`), not to a shipped codec (it is not shipped); correct
  by construction but flagged as **not-yet-measured as a distortion lever** (the
  §4 verdict turns on exactly this).
- **§3 sign/permutation "closed by mechanism"** — grounded to a *mechanism
  argument* (Hadamard already equalizes variance), NOT to a direct experiment.
  Flagged: this is a reasoned closure, falsifiable by the same G1 pack-algebra
  row if a skeptic demands the number.
- **§3 per-position scale + step-schedule "expected null"** — the *nulls are
  predicted, not measured*; flagged as UNTESTED with the cheap test given. The
  step-schedule row leans on the charge-aware-allocation kill
  (`k4-paper-shelf.md:71-72`) for the bit axis, which IS measured; the pure
  step-size axis is argument-only.

No row is ungrounded in the sense of citing nothing; three rows are grounded to
mechanism/non-charge rather than to a distortion measurement, and each is
flagged above with its cheapest falsifier.
