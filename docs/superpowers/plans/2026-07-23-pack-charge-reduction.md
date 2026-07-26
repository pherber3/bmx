# K4 Pack-Charge Reduction — Implementation Plan (skeptic accounting v2 + int8 decoder)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks 1–7 are local CPU code (TDD); Tasks 8–9 are the ONE short VM batch; Task 10 is the verdict doc. Everything local is done, committed, and pushed BEFORE any VM need.

**Goal:** Shrink the per-sequence skeptic pack charge that makes K4 more expensive than TurboQuant below ~8k context (b3 crossover currently ~13–16k, `docs/2026-07-15-k4-duel-results.md` §3). Two levers: (1) charge only the decoder columns the dequant provably reads — a free, honest accounting CORRECTION; (2) int8-quantize the decoder — a gated codec change (refit nothing). Target: move the b3 crossover left of ~8k. Lever 3 (cross-layer shared basis) is explicitly OUT of scope (one paragraph, §Lever 3 below).

**Architecture:** `skeptic_charge` in `src/bmx/cache/spectral.py` gains `c_used` / `dec_bits` keyword args (defaults reproduce the v1 expression bit-exactly); `spectral_quantize`'s payload bpe stops charging group scales on zero-bit dirs (the recorded minor, same principle); `StreamingQuantizedCache.bits_per_entry` and the k4 experiments thread the pack's realized C_used. Old parquets are NEVER edited — the results doc gets a clearly-labeled charge-corrected companion curve computed ANALYTICALLY from stored S and the committed fit-pack allocation stats. Lever 2 adds `int8_decoder_roundtrip` + a `dec_quant` spec field + `k4_b{budget}_dec8` recipe aliases, gated by a G1-style distortion measurement plus one n=100 probe A/B cell on the VM.

**Tech Stack:** as Stage 0–3 (PyTorch 2.12, transformers 5.11, tyro, safetensors, parquet). VM: rented NVIDIA GH200 per `vm-interaction-guide` memory — the VM tail here is ~1–2 h total and should be batched at the START of the second-model-replication rental (plans sequencing: this plan first, entirely local except Tasks 8–9).

---

## The arithmetic (measured inputs — read before implementing)

**Current (skeptic-v1) charge**, per K-entry, per layer (`spectral.py:skeptic_charge`):

```
charge_v1(S) = 16·C/S + tier_bits(tiers, S)          # one fp16 C×C decoder + 3-bit/dir tier map
             = (16·1024 + 3)/S            for C=1024, 7 tiers
```

`kv_size_bits` blends K and V equally (`generate.avg_bpe`), so the blended table pays `charge/2 = 8193.5/S`: 2.0 bits at 4k, 0.5 at 16k, 0.125 at 64k.

**The correction (Lever 1):** `quantize_by_bits` skips zero-bit tiers, so `Y_hat[:, bits==0] == 0`, and `M_hat = Y_hat @ dec.mT` multiplies those decoder columns by exact zeros — the decoder columns for dropped dirs are provably never read (Task 1 pins this with a mutation test). The honest decoder artifact is `C×C_used` fp16; the tier map stays full-C (it IS the encoding of which dirs are dropped). Likewise zero-bit dirs store no RTN groups, hence no group scales — the payload's `scale_bits(group)` term over-charges by `scale_bits(group)·(C−C_used)/C` (the recorded minor "spectral bpe over-charges group scale on zero-bit dirs"; conservative in all shipped parquets).

```
charge_v2(S)      = 16·C_used/S + tier_bits(tiers, S)                       # Lever 1
charge_v2_int8(S) =  8·C_used/S + 16·C_used/(S·C) + tier_bits(tiers, S)     # Lever 2 (int8 + fp16 per-column scales)
payload_v2        = mean(bits) + scale_bits(group)·C_used/C                  # replaces payload_v1 = mean(bits) + scale_bits(group)
```

**MEASURED C_used (⚠ diagnosed discrepancy):** the program lead's estimate assumed C_used ≈ 400 ("~300–500 of 1024 at budget 2.5"). The committed fit-pack parquet for the duel packs (`results/k4_fit_packs/20260713-133628-798d0ef/metrics.parquet`, model=llama-3.1-8b-instruct, uniform) says otherwise:

| budget | mean n_zero_dirs | per-layer range | **mean C_used** | decoder-charge reduction |
|---|---|---|---|---|
| 2.2 | 263.4 | 210–475 | **760.6** | 1.35× |
| 2.5 | 194.3 | 139–423 | **829.7** | 1.23× |

(The ~300–500 figure matches the ZERO-dir counts at low budgets / the stage-0 gpt2 heldout fit's 332-of-768, not the used-dir counts of the Llama corpus packs — corpus-pooled covariances have fuller tails.) Consequence, applying `corrected(S) = measured(S) − [16·(C−C̄_used)/(2S)] − [scale_bits(64)·(1−C̄_used/C)/2]` to the §3 measured rows (k4_b2.5, C̄_used = 829.7):

| accounting | 4k | 8k | 16k | 32k | 64k | b3 crossover | k3v2 crossover |
|---|---|---|---|---|---|---|---|
| v1 as measured | 4.81 | 3.60 | 2.99 | 2.69 | 2.53 | ~13k | 32–64k |
| v2, C_used=400 (lead's assumption) | 3.51 | 2.91 | 2.65 | 2.54 | 2.48 | **~5–7k** | ~16–24k |
| **v2, C_used=829.7 (measured)** | 4.41 | 3.39 | 2.87 | 2.62 | 2.48 | **~10–11k** | ~32k |
| **v2 + int8, C_used=829.7 (measured)** | 3.60 | 2.98 | 2.67 | 2.52 | 2.43 | **~5–6k** | ~20–24k |

(b3 reference: 3.42/3.22/3.12/3.07/3.04; k3v2: 2.94/2.73/2.62/2.57/2.54. k4_b2.2 at 64k improves 2.38 → 2.32 (v2) → ~2.27 (v2+int8), extending the cheapest-arm Pareto win.)

**Implication (binding for how the verdict is written):** with the MEASURED C_used, Lever 1 alone moves the b3 crossover to ~10–11k, not left of 8k; the <8k target rides on Lever 2 passing its gate. Structure stays as the lead set it: Lever 1 first (free + honest, ships regardless), Lever 2 gated (if it fails, ship Lever 1 alone and the crossover claim is stated as ~10–11k). The projections above are estimates from the headline table; Task 4 recomputes them EXACTLY per row from the stored parquets — those exact numbers, not this table, go in the results doc.

---

## Global Constraints

- **NEVER `git commit` without the user's explicit approval** — stage, propose the task's exact message, STOP (the user may pre-authorize per run, as in Stages 0–3).
- Before any commit: `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` all clean. Baseline at plan time: **451 passed, 17 skipped, 1 xfailed** (branch tip `11062ff`).
- **Honest-bpe expressions are the scientific record.** Every accounting change is a NEW NAMED MODE with the old expression documented beside it: `skeptic-v1 (full-C fp16)` vs `skeptic-v2 (used-columns)` vs `skeptic-v2-int8`; `payload-v1` vs `payload-v2`. `skeptic_charge`'s DEFAULT arguments reproduce v1 bit-exactly forever (pinned test). New experiment parquets emit the v1 value in companion `*_fullc` columns beside the v2 primary — no silent substitution anywhere.
- **Past parquets stay as-measured** (conservative, noted as such). The results doc gets a companion charge-corrected table computed analytically; never edit old parquets, never re-run old cells to "fix" their accounting.
- Comparisons stay skeptic-primary; model-level accounting remains SECONDARY (Gate A failed structurally — `docs/2026-07-15-k4-duel-results.md` §1).
- Lever 2 refits NOTHING: the int8 decoder is a roundtrip of the EXISTING pack's `dec`; allocation, basis, and encoder untouched.
- Numerics discipline: the default streaming path must stay byte-identical (the GH200 bitwise gates on `tests/test_streaming_batched_flush.py` pin codec outputs) — `dec_quant` defaults to `"fp32"` (today's compute path) and only the ACCOUNTING changes by default.
- Deterministic; tiny offline test models from `tests/factories.py`; never download in tests. gpt2 pre-flight artifacts already exist locally (`results/cache/gpt2_1024*.safetensors`, `results/cache/k4_packs_gpt2.safetensors`).
- VM discipline per memories: detached setsid launches, never sudo pip, don't kill long runs prematurely; batch the VM tail with the next rental.

## File structure

- Modify `src/bmx/cache/spectral.py` — `skeptic_charge(C, S, tiers, *, c_used=None, dec_bits=16.0)`; `spectral_payload_bpe(pack)`; `SpectralPack.c_used` property; `int8_decoder_roundtrip`; module-docstring "Accounting modes" section naming v1/v2.
- Modify `src/bmx/cache/streaming.py` — `bits_per_entry()` threads per-layer `c_used` (mean across layers) + `dec_bits`; cache init applies `dec_quant`.
- Modify `src/bmx/cache/specs.py` — `dec_quant: str = "fp32"` (default-inert).
- Modify `src/bmx/cache/recipes.py` — optional `_dec8` suffix on `k4_b{budget}` arms.
- Modify `experiments/k4_frontier.py`, `experiments/k4_spectra.py` — pass `c_used` from the pack; emit `bpe_skeptic_fullc` / `bpe_skeptic_deploy_fullc` companion columns.
- Modify `experiments/_k4_common.py` — host `_log_interp` + `_tq_layer_curve` (moved from k4_frontier, shared with the new gate harness).
- Create `experiments/k4_dec_quant.py` — the Lever-2 G1-style gate harness (dec modes fp32/fp16/int8).
- Create `experiments/k4_charge_curve.py` — analytic charge-corrected companion curve (reads NIAH parquets + fit-pack allocation stats; writes a markdown table; touches no old parquet).
- Modify `docs/2026-07-15-k4-duel-results.md` — §3b companion table (clearly labeled).
- Tests: `tests/test_spectral.py`, `tests/test_streaming_spectral.py`, `tests/test_cache_specs.py`, `tests/test_k4_experiments.py` (all append).

---

### Task 1: Accounting v2 in `spectral.py` — charge arithmetic + the never-read license test

**Files:**
- Modify: `src/bmx/cache/spectral.py`
- Test: `tests/test_spectral.py` (append)

**Interfaces:**
- `skeptic_charge(C, S, tiers, *, c_used: float | None = None, dec_bits: float = 16.0) -> float` — `c_used=None` means `c_used=C`; expression `dec_bits·c_used/S + (16·c_used/(S·C) if dec_bits < 16 else 0.0) + tier_bits(tiers, S)`. Assert `0 < c_used <= C` and `dec_bits in (8.0, 16.0)`. Docstring documents v1 beside v2 verbatim.
- `spectral_payload_bpe(pack: SpectralPack) -> float` — `mean(bits) + scale_bits(group)·(c_used/C)`; docstring records payload-v1 (`mean(bits) + scale_bits(group)`) and the over-count it corrects. `spectral_quantize` switches its returned bpe to this (payload-v2).
- `SpectralPack.c_used` property — `int((self.bits != 0).sum())`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_spectral.py`):

```python
def test_skeptic_charge_v2_hand_computed():
    """v2 arithmetic pinned exactly; defaults reproduce v1 bit-exactly."""
    tiers7 = (0, 2, 3, 4, 5, 6, 8)
    # c_used == C reproduces the old value exactly (continuity pin).
    assert skeptic_charge(1024, 32768, tiers7, c_used=1024) == skeptic_charge(
        1024, 32768, tiers7
    )
    assert skeptic_charge(1024, 32768, tiers7) == 0.500091552734375  # v1, hand
    # Hand-computed v2 fp16 case: 16*400/8192 + 3/8192.
    assert skeptic_charge(1024, 8192, tiers7, c_used=400) == 0.7816162109375
    # Hand-computed v2 int8 case: 8*400/8192 + 16*400/(8192*1024) + 3/8192.
    assert (
        skeptic_charge(1024, 8192, tiers7, c_used=400, dec_bits=8.0)
        == 0.391754150390625
    )


def test_spectral_payload_bpe_v2():
    import dataclasses as _dc

    from bmx.cache.spectral import spectral_payload_bpe

    M, _ = _spiked_keys(S=256, C=8, seed=0)
    Wh, Wh_inv = identity_whitener(8)
    pack = fit_spectral_pack(M, Wh, Wh_inv, 2.0, group=4)
    pack = _dc.replace(pack, bits=torch.tensor([8, 4, 2, 0, 0, 0, 0, 0]))
    assert pack.c_used == 3
    # v2 = mean(bits) + scale_bits(4) * 3/8 = 1.75 + 4.0*0.375 = 3.25 (v1 was 5.75).
    assert spectral_payload_bpe(pack) == 3.25
    # spectral_quantize returns the same payload-v2 number.
    _, bpe = spectral_quantize(M, pack)
    assert bpe == 3.25


def test_dropped_decoder_columns_never_read():
    """THE license for charging C×C_used: mutating dec columns of zero-bit dirs
    must not change the reconstruction by a single bit."""
    M, _ = _spiked_keys(S=256, C=64, seed=1)
    Wh, Wh_inv = identity_whitener(64)
    pack = fit_spectral_pack(M, Wh, Wh_inv, 1.5, group=16)  # low budget => dropped dirs
    assert pack.c_used < 64, "fixture must produce zero-bit dirs"
    M_hat_ref, _ = spectral_quantize(M, pack)
    import dataclasses as _dc

    dec_mut = pack.dec.clone()
    dec_mut[:, pack.bits == 0] = torch.randn_like(dec_mut[:, pack.bits == 0])
    M_hat_mut, _ = spectral_quantize(M, _dc.replace(pack, dec=dec_mut))
    assert torch.equal(M_hat_ref, M_hat_mut)
```

- [ ] **Step 2: Verify failure** — `uv run pytest tests/test_spectral.py -q -k "v2 or never_read"` → TypeError / AttributeError / value mismatch.
- [ ] **Step 3: Implement.** In `spectral.py`: extend `skeptic_charge` (keyword-only new args; keep the two-line docstring plus the mode-naming block); add `spectral_payload_bpe` and switch `spectral_quantize`'s bpe line to call it; add the `c_used` property on `SpectralPack`. Add an "Accounting modes" section to the module docstring: `skeptic-v1 (full-C fp16) = 16·C/S + tier_bits` (the expression every parquet before 2026-07-23 measured), `skeptic-v2 (used-columns)`, `skeptic-v2-int8`, `payload-v1`, `payload-v2` — expressions verbatim.
- [ ] **Step 4: Verify pass**, and that the EXISTING `test_skeptic_charge_formula` stays green untouched (defaults are v1).
- [ ] **Step 5: Full battery** (expect the streaming blended tests to still pass — nothing threads c_used yet; `spectral_quantize`'s payload change shifts only spectral-arm bpe values, which no test pins to v1 constants — if one does, update it citing payload-v2 and note it in the commit). Stage + propose `feat(spectral): skeptic-v2 accounting — charge only used decoder columns; payload-v2 drops phantom group scales; dropped-column license test`. STOP for approval.

---

### Task 2: Thread v2 through `bits_per_entry` (blended tests updated additively)

**Files:**
- Modify: `src/bmx/cache/streaming.py`
- Test: `tests/test_streaming_spectral.py` (update expected-value arithmetic + append)

**Interfaces:**
- `bits_per_entry()` — spectral branch becomes: `mean_c_used = sum(layer._pack.c_used for layer in self.layers) / len(self.layers)`; `bpe_k += skeptic_charge(C, S, tiers, c_used=mean_c_used, dec_bits=self._dec_bits)` where `self._dec_bits` is 8.0 iff `k_spec.dec_quant == "int8"` (field lands in Task 5; until then hardcode 16.0 with a comment). The charge is linear in c_used, so the across-layer mean of per-layer charges equals the charge at mean c_used — allocated packs (per-layer C_used differs, range 139–423 zero-dirs at b2.5) are exact under this. Update the docstring's charge equation and note v1 beside it.

- [ ] **Step 1: Update + append the failing tests.** In `tests/test_streaming_spectral.py`, both blended-accounting tests (the uniform one and the allocated `layer_budgets` one around line 230) currently compute `expected = mean(layer.bpe_k) + skeptic_charge(C, S, pack.tiers)`. Update each to v2 and ADD the direction assertion (additive — old structure kept, new invariant appended):

```python
    mean_c_used = sum(l._pack.c_used for l in cache.layers) / len(cache.layers)
    expected = sum(per_layer) / len(per_layer) + skeptic_charge(
        C, S, cache.layers[-1]._pack.tiers, c_used=mean_c_used
    )
    assert abs(bpe_k - expected) < 1e-9
    # v2 must charge strictly less than v1 whenever dirs were dropped (additive pin).
    expected_v1 = sum(per_layer) / len(per_layer) + skeptic_charge(
        C, S, cache.layers[-1]._pack.tiers
    )
    assert mean_c_used < C and bpe_k < expected_v1
```

  (The tiny fixture drops ~3 of 16 dirs at budget 2.5 per the fit-packs smoke parquet, so `mean_c_used < C` holds; if a fixture ever allocates all dirs, lower its budget.)
- [ ] **Step 2: Verify failure** (cache still charges v1 → `bpe_k == expected_v1`).
- [ ] **Step 3: Implement** in `bits_per_entry` per the interface above.
- [ ] **Step 4: Verify pass** + `uv run pytest tests/ -q -k "streaming or spectral"`.
- [ ] **Step 5: Battery + stage + propose** `feat(streaming): bits_per_entry charges skeptic-v2 (per-pack used columns, exact for allocated packs)`. STOP.

---

### Task 3: Thread v2 through the k4 experiments (+ companion v1 columns)

**Files:**
- Modify: `experiments/k4_frontier.py`, `experiments/k4_spectra.py`
- Test: `tests/test_k4_experiments.py` (append)

**Interfaces:**
- Both experiments' spectral rows change from `bpe_model + skeptic_charge(C, S, cfg.tiers)` to `bpe_model + skeptic_charge(C, S, cfg.tiers, c_used=pack.c_used)` (same for `DEPLOY_S`), and EVERY spectral row gains companions `bpe_skeptic_fullc` / `bpe_skeptic_deploy_fullc` carrying the v1 values (baseline arms: set the `_fullc` columns equal to their `bpe_skeptic` — no pack, no divergence). `bpe_model` is payload-v2 automatically via Task 1.
- The G1 verdict machinery keeps using `bpe_skeptic_deploy` (now v2) — that IS the correction taking effect; the `_fullc` columns keep v1 readable in-data for any cross-run comparison.

- [ ] **Step 1: Failing test** (append; follow the existing `_tiny_cache` smoke pattern in `tests/test_k4_experiments.py`):

```python
def test_k4_frontier_emits_v2_and_fullc(tmp_path):
    import pandas as pd

    from experiments.k4_frontier import Config, main

    p = tmp_path / "m.safetensors"
    _tiny_cache(p, seed=0)
    cfg = Config(
        cache_path=str(p), model_label="tiny", budgets=(1.5,),
        group=16, max_layers=1, out_root=str(tmp_path / "results"),
    )
    run_dir = main(cfg)
    df = pd.read_parquet(run_dir / "metrics.parquet")
    spec = df[(df.arm == "spectral") & (df.fit_mode == "oracle")]
    assert {"bpe_skeptic_fullc", "bpe_skeptic_deploy_fullc"} <= set(df.columns)
    # v2 <= v1 always; strict where the allocation dropped dirs.
    assert (spec.bpe_skeptic <= spec.bpe_skeptic_fullc + 1e-12).all()
    assert (spec.bpe_skeptic < spec.bpe_skeptic_fullc).any()
```

- [ ] **Step 2: Verify failure. Step 3: Implement** (both experiments; `pack.c_used` is available at every emit site — the randbasis/unweighted arms use their own packs' c_used). Keep column order explicit in the final `df[...]` select.
- [ ] **Step 4: Verify pass** + the existing k4 experiment smokes stay green.
- [ ] **Step 5: Battery + stage + propose** `feat(exp): k4 frontier/spectra emit skeptic-v2 primary + full-C v1 companion columns`. STOP.

---

### Task 4: The charge-corrected companion curve (analytic; old parquets untouched)

**Files:**
- Create: `experiments/k4_charge_curve.py`
- Modify: `docs/2026-07-15-k4-duel-results.md` (append §3b)
- Test: `tests/test_k4_experiments.py` (append)

**Interfaces:**
- CLI (tyro): `niah_run_dirs: tuple[str, ...]` (explicit run selection — plot-script discipline, never blind-concat a results root), `fit_packs_parquet: str` (the run that fit THE duel packs: `results/k4_fit_packs/20260713-133628-798d0ef/metrics.parquet`), `budgets=(2.2, 2.5)`, `C: int = 1024`, `group: int = 64`, `tiers=(0,2,3,4,5,6,8)`, `dec_bits_variants=(16.0, 8.0)`, `out_path: str = ""` (markdown table; stdout if empty).
- Logic: for each NIAH parquet row of a k4 arm, with its recorded sequence length S (the calibration length `compression_for` ran at) and the fit-pack parquet's per-layer `n_zero_dirs` at that budget (mean over layers → C̄_used): `corrected = measured_kv_bits − [skeptic_charge(C, S, tiers) − skeptic_charge(C, S, tiers, c_used=C̄_used, dec_bits=db)]/2 − [scale_bits(group)·(1 − C̄_used/C)]/2` for each `db` in `dec_bits_variants` (the `/2` is the K/V blend, `generate.avg_bpe`). TQ baseline rows pass through unchanged. Emits one markdown table: arm × context × {v1 as-measured, v2, v2-int8} + the recomputed b3/k3v2 crossover contexts (linear interpolation in 1/S), each column labeled with its mode name.
- LongBench task-S numbers: correct via per-sample lengths from the run's `samples.parquet` shards where present; if absent for a run, print the formula evaluated at the run's mean S and mark the cell `≈` — never silently mix the two.

- [ ] **Step 1: Failing smoke test** (synthetic parquets in tmp_path):

```python
def test_k4_charge_curve_smoke(tmp_path):
    import pandas as pd

    from experiments.k4_charge_curve import Config, main

    run = tmp_path / "niah" / "r1"
    run.mkdir(parents=True)
    pd.DataFrame(
        {"arm": ["k4_b2.5", "k4_b2.5"], "length": [8192, 32768],
         "kv_size_bits": [3.60, 2.69]}
    ).to_parquet(run / "metrics.parquet")
    fp = tmp_path / "fit.parquet"
    pd.DataFrame(
        {"model": ["m"] * 2, "layer": [0, 1], "budget": [2.5] * 2,
         "n_zero_dirs": [190, 198]}
    ).to_parquet(fp)
    out = tmp_path / "table.md"
    main(Config(niah_run_dirs=(str(run),), fit_packs_parquet=str(fp),
                budgets=(2.5,), out_path=str(out)))
    text = out.read_text()
    assert "skeptic-v2" in text and "as-measured" in text
    # 8k row, mean n_zero = 194 => C_used = 830:
    # v2 = 3.60 - 16*194/(2*8192) - 0.25*(194/1024)/2 = 3.60 - 0.18945 - 0.02368 = 3.3869
    assert "3.39" in text
```

  (Belt-and-braces: also compute the expected value INSIDE the test from `skeptic_charge`/`scale_bits` themselves and assert the rendered digits match that — the hardcoded literal is a hand-check, the computed one is the regression pin; if they disagree at implementation time, the formula transcription is wrong — diagnose, don't adjust the literal.)
- [ ] **Step 2: Verify failure. Step 3: Implement** (thin: pandas + the Task-1 functions; no torch needed beyond imports it drags in).
- [ ] **Step 4: Run it for real** against the committed duel parquets (`results/k3_niah/20260715-*`, fit-packs run `20260713-133628-798d0ef`) and append the output to `docs/2026-07-15-k4-duel-results.md` as **§3b "Charge-corrected companion curve (skeptic-v2, computed 2026-07-23 — analytic re-derivation; §3 stays as-measured)"** with: the mode definitions, the C̄_used table (incl. the diagnosed ~400-vs-830 discrepancy note), the corrected curve, and the new crossover statements. Sanity: the v2 row must land within ±0.05 of this plan's projection table; if not, STOP and diagnose (per-row S vs nominal is the first suspect).
- [ ] **Step 5: Battery + stage + propose** `feat(exp)+docs(k4): charge-corrected companion curve — skeptic-v2 re-derivation of the bits-vs-context table (old parquets untouched)`. STOP.

---

### Task 5: Lever-2 codec piece — int8 decoder (local, gated USE, code lands now)

**Files:**
- Modify: `src/bmx/cache/spectral.py` (`int8_decoder_roundtrip`), `src/bmx/cache/specs.py` (`dec_quant`), `src/bmx/cache/recipes.py` (`_dec8` suffix), `src/bmx/cache/streaming.py` (init + `_dec_bits`)
- Test: `tests/test_spectral.py`, `tests/test_cache_specs.py`, `tests/test_streaming_spectral.py` (append)

**Interfaces:**
- `int8_decoder_roundtrip(dec: torch.Tensor, bits_pc: torch.Tensor) -> torch.Tensor` — per-column symmetric absmax int8 over USED columns only (`bits_pc != 0`): `scale = absmax/127` cast fp16 (the shipped scale dtype) then back to fp32 BEFORE dequant; unused columns returned untouched (never read — Task 1's license test covers them). Deterministic, fp32 in/out.
- `CacheCodecSpec.dec_quant: str = "fp32"` — default-inert (today's byte-identical compute path); `"int8"` = roundtrip the loaded pack's `dec` at cache init and charge `dec_bits=8`.
- `recipes.spec_pair`: `k4_b{budget}_dec8` → same as `k4_b{budget}` with `dec_quant="int8"` (parse the suffix before the float).
- `StreamingQuantizedCache.__init__`: when `k_spec.arm == "spectral"` and `k_spec.dec_quant == "int8"`, replace each layer pack's `dec` via `int8_decoder_roundtrip` (dataclasses.replace, once, at init); set `self._dec_bits = 8.0` else `16.0`; `bits_per_entry` uses it (Task 2 left the hook).

- [ ] **Step 1: Failing tests:**

```python
# tests/test_spectral.py
def test_int8_decoder_roundtrip():
    from bmx.cache.spectral import int8_decoder_roundtrip

    torch.manual_seed(0)
    dec = torch.randn(32, 32)
    bits = torch.tensor([3] * 20 + [0] * 12)
    dec_rt = int8_decoder_roundtrip(dec, bits)
    used = bits != 0
    # Unused columns untouched; used columns within one int8 step of source.
    assert torch.equal(dec_rt[:, ~used], dec[:, ~used])
    step = dec[:, used].abs().amax(dim=0) / 127.0
    assert (dec_rt[:, used] - dec[:, used]).abs().amax(dim=0).le(step + 1e-6).all()
    assert torch.equal(dec_rt, int8_decoder_roundtrip(dec, bits))  # deterministic


# tests/test_cache_specs.py
def test_dec_quant_default_inert_and_dec8_recipe(tmp_path):
    from bmx.cache.recipes import spec_pair
    from bmx.cache.specs import CacheCodecSpec

    assert CacheCodecSpec(arm="rtn_channel", bits=3).dec_quant == "fp32"
    k, v = spec_pair("k4_b2.5_dec8", pack_path="/p/packs.safetensors")
    assert k.arm == "spectral" and k.budget == 2.5 and k.dec_quant == "int8"
    assert v.arm == "turboquant_mse" and v.bits == 2


# tests/test_streaming_spectral.py — append; reuse the module's _fit_tiny_packs fixture
def test_streaming_dec8_charges_int8_and_degrades_gracefully(tmp_path):
    path = str(tmp_path / "packs.safetensors")
    model = tiny_llama()
    _fit_tiny_packs(model, path)
    caches = {}
    for dq in ("fp32", "int8"):
        k_spec = CacheCodecSpec(arm="spectral", pre_rope=True, group=8,
                                pack_path=path, budget=2.5, dec_quant=dq)
        cache = StreamingQuantizedCache(
            model.config, k_spec=k_spec,
            v_spec=CacheCodecSpec(arm="fp16"), recent_window=8,
        )
        cache.attach(model)
        with cache:
            with torch.no_grad():
                model(ids(seq=150), past_key_values=cache, use_cache=True)
        caches[dq] = cache
    bpe_fp32, _ = caches["fp32"].bits_per_entry()
    bpe_int8, _ = caches["int8"].bits_per_entry()
    assert bpe_int8 < bpe_fp32  # 8-bit decoder charge strictly cheaper
    # And the delta equals the charge arithmetic exactly (payloads identical).
    layer = caches["fp32"].layers[-1]
    C = layer._h_kv * layer._d_head
    S = layer.get_seq_length()
    mc = sum(l._pack.c_used for l in caches["fp32"].layers) / len(caches["fp32"].layers)
    t = layer._pack.tiers
    expected_delta = skeptic_charge(C, S, t, c_used=mc) - skeptic_charge(
        C, S, t, c_used=mc, dec_bits=8.0
    )
    assert abs((bpe_fp32 - bpe_int8) - expected_delta) < 1e-9
```

  (Payload bpe is identical between the two arms — the decoder roundtrip changes M_hat VALUES, not the bit allocation — so the bpe delta is pure charge arithmetic. The QUALITY effect is Lever 2's gate, Tasks 6/8.)
- [ ] **Step 2: Verify failure. Step 3: Implement** the four small pieces. `dec_quant` values validated at cache init (`assert k_spec.dec_quant in ("fp32", "int8")`).
- [ ] **Step 4: Verify pass** + full streaming suite (bitwise-pinned tests must be untouched — default path identical).
- [ ] **Step 5: Battery + stage + propose** `feat(spectral): int8 decoder roundtrip + dec_quant spec field + k4_b*_dec8 arms (accounting dec_bits=8; use gated on Task-8 measurement)`. STOP.

---

### Task 6: The Lever-2 gate harness (`k4_dec_quant.py`) + gpt2 local pre-flight

**Files:**
- Create: `experiments/k4_dec_quant.py`
- Modify: `experiments/_k4_common.py` (move `_log_interp` + `_tq_layer_curve` from `k4_frontier.py`; frontier imports them — pure moves, zero behavior change)
- Test: `tests/test_k4_experiments.py` (append)

**Interfaces:**
- Config: `pack_path: str`, `cache_paths: tuple[str, ...]` (scored caches), `model_label: str`, `model_name: str = ""`, `budgets: tuple[float, ...] = (2.2, 2.5)`, `tq_bits=(2, 3, 4)`, `seed=0`, `out_root=""`. Loads packs via `load_packs` (REFITS NOTHING — this is the "quantize dec of the EXISTING pack" rule made structural).
- Per (cache, layer, budget): score `spectral_quantize(M_pre, pack_variant)` tail-region (`_score_tail`, headline `logit_rope` when RoPE available, else `logit` — mirror frontier's `headline_col` logic) for dec modes `fp32` (as-run today), `fp16` (`dec.half().float()` — what skeptic-v1 charges; the shippability check), `int8` (`int8_decoder_roundtrip`); plus the per-layer `turboquant_mse` k_pre curve at `tq_bits` for the win interpolation.
- Verdict JSON (`dec_quant_verdict.json`): per budget, mean-over-(caches×layers) of per-layer win = tq_interp(at the FP16 arm's `bpe_skeptic_deploy`, i.e. the SAME bpe point for all dec modes — isolates pure quality damage; the bits benefit is reported separately, never gated on) / dist. Fields: `win_fp32`, `win_fp16`, `win_int8`, `rel_degradation_int8 = 1 − win_int8/win_fp16`, `rel_degradation_fp16_vs_fp32 = 1 − win_fp16/win_fp32`, `int8_win_at_own_bits` (deployment view, `dec_bits=8` + c_used accounting), and `gate_pass = all(rel_degradation_int8 < 0.05 per budget)`. **Acceptance (pre-registered, binding): `rel_degradation_int8 < 5%` relative vs the fp16-decoder arm at every budget.** Rider: if `rel_degradation_fp16_vs_fp32 > 0.5%`, flag it in the verdict — that would mean skeptic-v1's fp16-decoder charge was optimistic and must be re-examined (expected ≈ 0; this closes the fp32-stored/fp16-charged loophole with a measurement).

- [ ] **Step 1: Failing smoke test:**

```python
def test_k4_dec_quant_smoke(tmp_path):
    import json

    import pandas as pd

    from experiments.k4_dec_quant import Config, main

    fit, scored = tmp_path / "f.safetensors", tmp_path / "s.safetensors"
    _tiny_cache(fit, seed=0)
    _tiny_cache(scored, seed=1)
    packs_path = tmp_path / "packs.safetensors"
    _fit_tiny_pack_file(fit, packs_path, budgets=(2.5,), group=16)  # helper mirroring k4_fit_packs
    run_dir = main(Config(pack_path=str(packs_path), cache_paths=(str(scored),),
                          model_label="tiny", budgets=(2.5,),
                          out_root=str(tmp_path / "results")))
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert set(df.dec_mode) == {"fp32", "fp16", "int8"}
    v = json.loads((run_dir / "dec_quant_verdict.json").read_text())
    assert "gate_pass" in v and "rel_degradation_int8" in str(v)
```

- [ ] **Step 2: Verify failure. Step 3: Implement** (reuse `load_layer_keys`/`setup_rope`/`_score_tail`; the moved `_log_interp`/`_tq_layer_curve`; ~200 lines).
- [ ] **Step 4: gpt2 mechanism pre-flight (local, minutes, artifacts already on disk):**

```bash
uv run python experiments/k4_dec_quant.py --pack-path results/cache/k4_packs_gpt2.safetensors \
  --cache-paths results/cache/gpt2_1024.safetensors --model-label gpt2 --budgets 2.5
```

  Read the verdict: all three dec modes finite, fp16≈fp32, int8 degradation lands where it lands (gpt2 is a mechanism check, not the gate — RECORD the number; if gpt2 int8 degradation is already >20% that's an early red flag worth reporting before the VM spend).
- [ ] **Step 5: Battery + stage + propose** `feat(exp): k4_dec_quant — G1-style int8/fp16 decoder distortion gate on existing packs (refit nothing)`. STOP.

---

### Task 7: Local pre-flight gate + push (the VM transport prerequisite)

**Files:** none new — verification + transport.

- [ ] **Step 1:** Full battery clean (`uv run ruff format .` → `check` → `pytest -q`; expect 451+new/17/1 profile). Then the end-to-end local smoke: rerun the Task-4 script against the real duel parquets and eyeball the §3b table one more time; run one streaming gpt2 sanity with `spec_pair("k4_b2.5_dec8", pack_path="results/cache/k4_packs_gpt2.safetensors")` (3-line scratchpad script), print `bits_per_entry()` and check the dec8-vs-fp32 delta against `skeptic_charge` by hand.
- [ ] **Step 2:** Confirm every task's commit landed (user-approved), branch clean.
- [ ] **Step 3:** STOP — propose pushing `feat/triton-decode-kernel` to origin. User approves the push explicitly. (If the second-model-replication rental is imminent, Tasks 8–9 ride at the start of that VM session.)

---

## VM batch (Tasks 8–9) — one GH200 session, ~1–2 h total, ordered, gated

Transport per `vm-interaction-guide`: push → VM pull, `scripts/vm_setup.sh`, `uv run pytest -q` (record the GH200 count). Pack files and scored caches are NOT committed artifacts — regenerate on a fresh rental (deterministic from committed config; ~30–40 min GPU):

```bash
# Only if absent on the VM (fresh rental). Instruct model — the deployment/duel arm.
for OFF in 2048 4096 6144 8192 10240 12288 14336 16384; do
  uv run python experiments/collect_cache.py --model-name meta-llama/Llama-3.1-8B-Instruct \
    --seq-len 2048 --token-offset $OFF
done
uv run python experiments/k4_fit_packs.py --corpus-cache-paths <the 4 fit caches: offsets 2048..8192> \
  --out-path results/cache/k4_packs_llama31_instruct.safetensors --model-label llama-3.1-8b-instruct \
  --model-name meta-llama/Llama-3.1-8B-Instruct --w-source corpus
```

(Fit/scored split verbatim from the Stage-3 plan: fit = offsets {2048..8192}, scored = {10240..16384}. NOTE the lead specified "the 4 scored caches" — those are model-matched Instruct caches here; pack and caches must come from the SAME model. If the old rental persists with base-model scored caches intact, run the base pair too as a free replication row, clearly labeled.)

### Task 8: [VM-RUN] The Lever-2 gate measurement (BEFORE any probe spend)

```bash
for SC in <the 4 scored caches: offsets 10240..16384>; do
  uv run python experiments/k4_dec_quant.py --pack-path results/cache/k4_packs_llama31_instruct.safetensors \
    --cache-paths $SC --model-label llama-3.1-8b-instruct \
    --model-name meta-llama/Llama-3.1-8B-Instruct --budgets 2.2 2.5
done
```

Minutes per cache (CPU-heavy matrix math; GPU idle is fine). **Gate call (pre-registered):** `rel_degradation_int8 < 5%` at BOTH budgets, mean over the 4 scored caches, min/max reported as error bars. Also record `rel_degradation_fp16_vs_fp32` (expected ≈ 0; >0.5% triggers the skeptic-v1 re-examination flag). **If the gate FAILS:** skip Task 9 entirely, ship Lever 1 alone, and the verdict doc records the honest negative + the measured degradation number — the crossover claim is then ~10–11k, not <8k.

### Task 9: [VM-RUN] The probe A/B cell (ONLY if Task 8 passes; ~30 min GPU)

```bash
uv run python -m experiments.k3_longbench --model-name meta-llama/Llama-3.1-8B-Instruct \
  --arms k4_b2.5 k4_b2.5_dec8 \
  --pack-path results/cache/k4_packs_llama31_instruct.safetensors \
  --categories synthetic --n-samples 100
```

Same run = same samples = paired per-sample deltas. **Acceptance: scores within noise** — operationalized: paired bootstrap (10k resamples on the per-sample shards, the existing CI machinery from the code-category work) 95% CI of (dec8 − fp32-dec) synthetic score contains 0, AND |point estimate| ≤ 1.0. Also sanity-read the run's `kv_size_bits` for the dec8 arm: it must sit BELOW the k4_b2.5 arm by exactly the charge arithmetic at the run's calibration S (compute the expected delta by hand in the run log). Commit parquets back via bundle. If the CI excludes 0 with |Δ| > 1.0: int8 fails end-to-end despite passing the distortion gate — record both numbers, ship Lever 1 alone, and flag the gate-vs-task discrepancy as a finding (that would itself be publishable honesty).

---

### Task 10: Verdict doc + traceability

**Files:**
- Create: `docs/2026-07-<day>-k4-pack-charge-results.md`
- Modify: `docs/2026-07-15-k4-duel-results.md` (§3b already landed in Task 4; add one line pointing at the new doc)

Kill-or-confirm style, both outcomes pre-drafted as templates in the doc header. Contents: (1) the accounting-mode definitions (v1/v2/v2-int8, expressions verbatim); (2) the measured C̄_used table + the ~400-vs-830 diagnosis; (3) the exact charge-corrected curve (from Task 4, extended with the v2-int8 column if Lever 2 shipped) + new crossover statements vs b3 and k3v2; (4) the Lever-2 gate numbers (distortion degradation, fp16-shippability rider, probe A/B CI) and the ship/no-ship call; (5) explicitly: which claims changed (crossover context, 64k cheapest-arm bits) and which did NOT (all quality numbers — no score moved; this plan changed accounting and, if Lever 2 shipped, decoder precision only). Every number traceable to parquets under `results/k4_dec_quant/`, `results/k3_longbench/`, `results/k4_fit_packs/20260713-133628-798d0ef/` with config + env + SHA. Update the paper skeleton's bits-vs-context table reference. Stage + propose; STOP.

---

## Lever 3 — explicitly OUT of scope (recorded as future work, no tasks)

Cross-layer shared basis: one decoder shared by all 32 layers would divide the decoder charge by n_layer (~0.008 bits at 4k after Levers 1+2 — the charge would effectively vanish at all contexts). It is NOT planned because basis transfer across layers is an unproven research question, not an accounting fact: the per-layer spectra differ enough that the sensitivity census shows a ~9× s_i spread across layers, and Gate A already demonstrated that this basis family's transfer (across sequences) hits structural ceilings. If pursued, it needs its own G1-style gauntlet (shared-basis retention per layer vs per-layer bases at matched bits) — record in the paper's future-work section and stop there.

---

## Self-Review

**Binding-decision coverage:** Lever 1 (charge only used columns; skeptic_charge gains c_used; bits_per_entry threads it; doc discipline old-parquets-as-measured + analytic companion curve; arithmetic shown) → Tasks 1–4 + the arithmetic section. Lever 2 (int8 decoder, refit-nothing, G1-style <5% gate on 4 scored caches with logit_rope, one n=100 probe A/B, ship-Lever-1-alone fallback) → Tasks 5–6 (code, local) + 8–9 (measurements). Lever 3 → one paragraph, no tasks. Hard rules: named modes with old expressions beside (Task 1 docstrings + `_fullc` columns + §3b labels), skeptic-primary unchanged, tests pin charge arithmetic exactly (hand-computed incl. C_used==C), int8 roundtrip test, blended test updated additively. Sequencing: Tasks 1–7 local/CPU, committed and pushed before Tasks 8–9 (the only VM need; batchable with the plan-3 rental).

**Deliberate deviation from the lead's brief, flagged not buried:** the brief's estimate (C_used≈400 → ~2.5× → crossover ~6–7k) is contradicted by the committed fit-pack parquet for the actual duel packs (mean C_used 829.7 at b2.5 → 1.23× → crossover ~10–11k for Lever 1 alone). The plan implements exactly the levers the lead ordered, but the projection table shows BOTH rows and the verdict doc must use the measured one — with the consequence stated plainly: the <8k target needs Lever 2 to pass its gate. Task 4's exact per-row recomputation is the authority; this table is the estimate.

**Placeholder scan:** all code steps carry real code; the one deliberately-flagged illustrative constant (Task 4 Step 1's `"3.49"`) is marked for derivation-in-test at implementation time, with instructions. VM runbook commands are exact with the one variable (`<scored caches>` = the recorded offsets split).

**Type consistency:** `skeptic_charge(C, S, tiers, *, c_used=None, dec_bits=16.0)` used identically in Tasks 1/2/3/4/5 (float c_used allowed — across-layer means are exact by linearity); `dec_quant ∈ {"fp32","int8"}` in specs/streaming vs `dec_mode ∈ {fp32, fp16, int8}` in the gate harness — DELIBERATE: fp16 is a measurement arm (the shippability check for what v1 charges), never a streaming path; flagged here so it isn't "unified" into a streaming mode nobody runs.

**Known open decisions deferred to run time:** whether the base-model replication rows in Task 8 run (only if the old rental's artifacts survive); the exact §3b LongBench task-S treatment (per-sample lengths if shards carry them, else labeled mean-S approximation); the day-stamp of the verdict doc.
