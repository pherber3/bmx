# KIVI arm diagnosis — symmetric-RTN strawman vs real KIVI

**Date:** 2026-07-04
**Trigger:** the live GH200 LongBench run (`results/k3_longbench/20260702-164100-46b9579`)
shows the `kivi` arm scoring ~0 across tasks (narrativeqa F1 0.007; qasper mostly literal
zeros) while the fp16 anchor is sane (narrativeqa 0.3189). TurboQuant's Table-1 published
KIVI rows average ~48–50. This doc answers: is our "kivi" arm a fair implementation of KIVI,
or a strawman that collapses at 2 bits for reasons unrelated to KIVI's real behavior?

## Verdict (code-reading, not speculation)

**Our `kivi` arm is 2-bit, scale-only SYMMETRIC groupwise RTN — no zero-point.** It is not a
faithful implementation of KIVI's asymmetric scheme. The near-zero LongBench scores are
consistent with a known symmetric-quantization failure mode (collapse on off-center /
one-sided per-group distributions), not a bug in the eval harness. **Recommendation: drop
`kivi` from the truncated parity rerun; cite TurboQuant's published KIVI-3 row transitively;
relabel our arm `rtn2` wherever it would appear in print.**

## Evidence: what the code actually does

### 1. The kivi arm's spec pair

`src/bmx/cache/recipes.py:66-70`:

```python
if arm == "kivi":
    return (
        CacheCodecSpec(arm="rtn_channel", bits=2, group=group, seed=seed),
        CacheCodecSpec(arm="rtn_token", bits=2, group=group, seed=seed),
    )
```

`group` defaults to 64 (`spec_pair(..., group: int = 64, ...)`,
`src/bmx/cache/recipes.py:14`). So: **K uses `rtn_channel`, V uses `rtn_token`, both at
2 bits, group size 64.** This part is a reasonable structural match to real KIVI (per-channel
grouping for keys, per-token grouping for values is exactly KIVI's design) — the grouping
axis choice is right. The bit-width (2) also matches our stated arm name (we called it
"kivi" implying the paper's 2-bit configuration). The problem is the underlying quantizer,
not the grouping.

### 2. `rtn_channel` / `rtn_token` both bottom out in the same symmetric primitive

`src/bmx/cache/codecs.py:555-560` (the packed streaming-path dispatch used by
`quantize_packed`):

```python
if arm == "rtn_token":
    Q_int, scale = rtn_quantize_packed(M, bits, group)
    return {"Q_int": Q_int, "scale": scale}, bits + scale_bits(group)
if arm == "rtn_channel":
    Q_int, scale = rtn_quantize_packed(M.mT, bits, group)
    return {"Q_int": Q_int, "scale": scale}, bits + scale_bits(group)
```

Both arms differ only in whether they quantize `M` or `M.mT` (i.e., which axis is grouped —
token-major vs channel-major). Both call the same `rtn_quantize_packed`.

### 3. `rtn_quantize_packed` is scale-only symmetric — no zero-point, by construction

`src/bmx/quant/rtn.py:1-24` (module docstring: *"Groupwise symmetric round-to-nearest
quantization"*):

```python
def rtn_quantize_packed(W: torch.Tensor, bits: int, group_size: int):
    *lead, d = W.shape
    assert d % group_size == 0, f"dim {d} not divisible by group {group_size}"
    assert bits <= 8, f"rtn_quantize_packed: int8 codes require bits <= 8, got {bits}"
    qmax = 2 ** (bits - 1) - 1
    G = W.reshape(*lead, d // group_size, group_size)
    scale = G.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    Q = (G / scale).round().clamp(-qmax - 1, qmax)
    Q_int = Q.to(torch.int8).reshape(W.shape)
    return Q_int, scale
```

There is exactly one stored parameter per group — `scale` — derived from
`abs().amax()`. There is no `zero_point` / `offset` term anywhere in the function, the
returned dict (`{"Q_int": Q_int, "scale": scale}`), or the dequant path
(`src/bmx/quant/rtn.py:27-39`, `rtn_dequantize_packed(Q_int, scale, group_size) = G * scale`,
purely linear through the origin). **This is scale-only symmetric quantization: the
representable levels are `{-qmax-1, ..., -1, 0, 1, ..., qmax} * scale`, always centered
at zero.**

At `bits=2`: `qmax = 2**(2-1) - 1 = 1`. The signed int8 code is clamped to
`[-qmax-1, qmax] = [-2, 1]`, i.e. **4 integer levels: {-2, -1, 0, 1}**, all multiples of one
`scale` value, centered on 0. For a per-group distribution that is NOT centered at zero
(e.g. shifted by an outlier channel mean, or genuinely one-sided), most of that 4-level
budget is wasted on the unused side of the range — exactly the failure mode the task brief
predicted.

### 4. Confirms: no other code path adds a zero-point for these arms

`grep`-ing `src/bmx/` for `zero_point`, `zp`, or asymmetric quantization finds nothing in the
RTN or codecs modules — `rtn_channel`/`rtn_token`/`rtn_quantize`/`rtn_quantize_packed` are the
only functions behind the `kivi` arm, and all four are symmetric by the same construction
above (`rtn_quantize` at `src/bmx/quant/rtn.py:36-39` is just
`rtn_dequantize_packed(*rtn_quantize_packed(...))`, so it inherits the same behavior).

## KIVI's actual scheme (as characterized from the task brief + TurboQuant's usage; NOT re-verified against the KIVI paper)

> **Caveat:** WebSearch/WebFetch were unavailable for this task; the KIVI paper itself was not
> consulted. This section is a characterization from (a) the controller-pinned TurboQuant
> context and (b) the standard/well-known description of KIVI's design as summarized in the
> task brief — treat it as background, not a verified citation. The **verdict above rests
> entirely on our own code**, which was read directly.

- KIVI is described as using **asymmetric** per-group quantization: each group stores both a
  scale AND a zero-point, so the representable range can sit anywhere (not forced through
  zero). This directly addresses the collapse mode symmetric quantization hits on off-center
  distributions.
- Grouping: **per-channel** for keys, **per-token** for values — which our arm already
  matches structurally (`rtn_channel` for K, `rtn_token` for V).
- KIVI also keeps a **fp16 residual window** (most-recent tokens stored unquantized) rather
  than quantizing the entire cache uniformly — our `kivi` arm quantizes everything at 2 bits
  with no residual window, which is a second (unaccounted for) divergence from the real
  scheme, independent of the symmetric/asymmetric issue.
- Honest bpe accounting for a faithful reproduction would need to charge for the zero-point
  (an additional value per group, typically fp16 like the scale — i.e. `+scale_bits(group)`
  again, doubling the metadata term) — this is called out in the task brief as "+16/group more
  metadata bits per the honest rule." Our current `bits + scale_bits(group)` bpe formula
  (`src/bmx/cache/codecs.py:557,560`) does not include this term, so it could not currently
  report an honest bpe for asymmetric KIVI even if the arm were switched to one.

## TurboQuant's own published KIVI configuration (pinned by controller, ground truth for this task)

TurboQuant Table 1 (Llama-3.1-8B-Instruct) publishes KIVI at **KV Size 3 bits and 5 bits**
(averages 48.50 and 50.16), **not 2 bits**. Our `kivi` arm was run at 2 bits — a bit-width
TurboQuant did not even evaluate KIVI at in the cited table. This is a second, independent
reason the ~0 scores are not informative about "how KIVI compares": even a faithful
asymmetric implementation at 2 bits would be extrapolating past the paper's tested range.

TurboQuant §4.3 also states that KIVI (and PolarQuant) "leave generated tokens unquantized" —
our harness quantizes tokens during generation on the streaming path (this is deliberate,
matches how the rest of our arms are measured, and is consistent with TurboQuant's own
harness per the controller's brief). This is an additional comparability caveat worth noting
for any transitive KIVI citation: our arms (including any future faithful KIVI arm) quantize
generated tokens; the literal KIVI method as described does not. This affects any of our arms
compared against a transitively-cited KIVI number, not just this one.

## Corroboration (CPU, throwaway script — not proof, evidence only)

Script: `kivi_corroboration.py` (scratchpad, not committed — throwaway per task
instructions). Compares our actual `rtn_quantize` (2-bit, group=64) against a 10-line
min/max-zero-point asymmetric quantizer, by reconstruction MSE.

**Synthetic distributions** (group=64, bits=2):

| distribution | symmetric MSE | asymmetric MSE | ratio (sym/asym) |
|---|---|---|---|
| shifted gaussian (mean=3, std=0.5) | 1.652e+00 | 5.003e-02 | **33.0x** |
| one-sided uniform [5, 7] | 1.260e+00 | 3.343e-02 | **37.7x** |
| zero-mean gaussian (control) | 5.177e-01 | 2.032e-01 | 2.55x |

The zero-mean control still favors asymmetric (4 evenly-placed levels beat 4 levels forced
through the origin even when centered, because clamping at `qmax=1` wastes a code on the
sign-asymmetric int8 range `[-2, 1]`), but the gap explodes 13–15x further once the
distribution is off-center — exactly the mechanism the task brief predicted.

**Real K/V tensors** (`tests.factories.tiny_llama`, seeded, via `collect_cache`; group=8
since the tiny model's `d_head=8` doesn't divide 64 — same qualitative regime as production
`d_head=128`/group=64):

| tensor | symmetric MSE | asymmetric MSE | ratio |
|---|---|---|---|
| K (layer0, post-RoPE) | 3.552e-03 | 9.017e-04 | 3.94x |
| K_pre (layer0, pre-RoPE) | 3.618e-03 | 8.863e-04 | 4.08x |
| V (layer0) | 3.138e-03 | 8.196e-04 | 3.83x |
| K (layer1, post-RoPE) | 3.242e-03 | 8.009e-04 | 4.05x |
| K_pre (layer1, pre-RoPE) | 3.146e-03 | 8.199e-04 | 3.84x |
| V (layer1) | 3.172e-03 | 7.781e-04 | 4.08x |

This tiny untrained model's K/V channels happen to be roughly zero-mean (random init, no
learned outlier structure), so the real-tensor gap here (~3.8–4.1x) lands closer to the
synthetic zero-mean control than the shifted case — consistent with the theory: **the
asymmetric quantizer is never worse, and the gap widens sharply exactly when per-group
distributions are off-center**, which is the well-documented behavior of real trained-model
KV channels (per-channel outlier offsets are the standard motivation cited for asymmetric /
smoothing-based KV quantizers). A ~4x MSE floor even in the favorable zero-mean case, on top
of a real model's per-channel offsets pushing well past 4x, is consistent with — though does
not by itself prove — 2-bit symmetric RTN collapsing to near-random output on
LongBench-scale sequences.

## Recommendation (as pre-approved by the controller; user may override)

1. **Drop `kivi` from the truncated parity rerun.** It does not implement asymmetric
   quantization, does not keep a residual window, and was run at a bit-width TurboQuant
   itself did not publish for KIVI. Running it further would only reproduce the same
   near-zero collapse without telling us anything about KIVI's real quality.
2. **Cite TurboQuant's published KIVI-3 row (KV Size 3, avg 48.50) transitively** in any
   Table-1-style comparison — the locked transitive-baseline strategy explicitly licenses
   this, and it is the closest published KIVI operating point to our 2-bit target (KIVI-5
   at avg 50.16 is the other option if a higher bit-width comparison point is wanted).
3. **Relabel our existing arm `rtn2`** anywhere it appears in print (parquet `arm` column,
   plots, tables) — it is exactly what it says: 2-bit symmetric groupwise RTN, no more, no
   less. It remains a legitimate baseline arm under its honest name; it is only mislabeled
   as "kivi."
4. **Faithful asymmetric KIVI is out of scope for this paper** unless the user asks for it
   now. It would require: (a) a new packed quantizer with a stored zero-point per group,
   (b) an updated honest bpe formula (`bits + 2 * scale_bits(group)` to also charge the
   zero-point, per the task brief's "+16/group" rule), and (c) a residual fp16 window to
   match KIVI's actual design — none of which exist in the current codebase and none of
   which this task modifies.

## Files read (no source/test files modified by this task)

- `src/bmx/quant/rtn.py` (full file, 40 lines)
- `src/bmx/cache/recipes.py:1-71` (full file)
- `src/bmx/cache/codecs.py:1-60` (arm registry), `:320-390` (waterfill usage of
  `rtn_quantize`), `:540-620` (`quantize_packed` dispatch for `rtn_token`/`rtn_channel`),
  `:620-640` (`dequant_packed` dispatch)
- `tests/factories.py:1-56` (`tiny_llama`, `ids`)
- `src/bmx/cache/collect.py:126-161` (`collect_cache`)
