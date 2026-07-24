# Packed Spectral Path — Phase A Implementation Plan (resident-memory claim + 128k evidence)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks 1–6 are local code (TDD); Tasks 7–11 are the VM batch (runbook kind, exact commands). Task 12 is a SKETCH ONLY (Phase B, gated).

**Goal:** Give the K4 spectral arm a packed-resident storage path so it gains a resident-MEMORY claim and 128k evidence. Today `StreamingQuantizedCache` keeps the dequantized fp16 prefix resident — the measured 1.5×-vs-dense decode result (`docs/2026-07-15-k4-duel-results.md` §6b) is a LATENCY claim only, and the streaming path cannot produce 128k evidence (it OOMs there by construction — the June census shows the dense-resident path at 83.5 GiB @128k vs packed 64.1, `docs/2026-06-23-kernel-census-results.md`). The packed path is how k2b_ph ran 128k; this plan extends it to spectral.

**Phase A scope (this plan's core): packed storage + CHUNKED dequant-attention. NO new Triton kernel.** The memory claim and the 128k table need residency, not kernel speed — the chunked path decoded k2b at ~60 ms/tok at 64k (acceptable). A fused spectral kernel is Phase B, explicitly GATED on Phase-A latency measurements showing need, and only sketched here (Task 12), not planned.

**Architecture:** `PackedStreamingLayer` gains a spectral K write-branch that stores per-page, per-TIER packed code segments (spectral codes are per-direction ints at VARIABLE widths — tiers {0,2,3,4,5,6,8} from the pack's per-direction bits); `chunked_dequant_attention` gains the matching read-branch (unpack tier segments → assemble Y_hat → `Y_hat @ dec.mT` → RoPE at true positions). V stays `turboquant_mse@2` — **the packed path already supports it functionally** (`quantize_packed`/`dequant_packed`/chunked all handle `turboquant_mse`; verified in `src/bmx/cache/codecs.py:622-626,708-711` and `chunked_attention.py:_dequant_block`) **but its container is int16 `(S, C)` indices = 16 bits per 2-bit code** — exactly the T4 unpacked-container failure mode — so Phase A must pack the V containers too or the V side resides at fp16 size and the memory table is fake. No `_PagedStacks`/stacked-buffer work at all (that machinery serves the fused kernels; avoiding it also sidesteps the stacks double-buffer history entirely).

**Tech Stack:** as Stage 3 (PyTorch 2.12, transformers 5.11, tyro, safetensors, parquet). VM: rented NVIDIA GH200 per `vm-interaction-guide` memory (git transport, detached setsid launches, no sudo pip).

## Global Constraints

- **NEVER `git commit` without the user's explicit approval** — stage, propose the task's exact message, STOP (the user may pre-authorize per run, as in prior batches).
- Before any commit: `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` all clean. Baseline at plan time: **451 passed, 17 skipped, 1 xfailed** (branch `feat/triton-decode-kernel` tip `11062ff`, 69 s).
- **The streaming path's numerics are FROZEN.** The duel results (2026-07-15/16) are licensed on `StreamingQuantizedCache` at bitwise-gated SHAs. Every change here is additive on the packed side; `spectral_quantize`'s refactor in Task 1 must be provably bitwise-neutral (the composition argument + the existing `test_batched_flush_bitwise_matches_reference_spectral` pin). Existing packed arms (k2b/k2b_ph/rtn) keep byte-identical block dicts and behavior (regression-pinned by `tests/test_packed_streaming.py`'s 29 tests).
- **Parity gates (binding, learned 2026-07-15):** (1) committed-block **BITWISE** parity vs `StreamingQuantizedCache` — same `compute_flush_schedule`, same PAGE=128 pages (the shared-schedule idiom); (2) **greedy-token parity at SHORT context (4k)**; (3) **logit-TOLERANCE parity at 64k** — bitwise greedy at 64k is **unattainable** (accumulation-order drift across 512 pages produces O(0.25–1.45) logit deltas with no token flip; `docs/2026-07-15-k4-duel-results.md` appendix). **Do not add, run, or gate on a bitwise 64k comparison.**
- **Container discipline (the T4 lesson):** committed storage holds packed dtypes ONLY — uint8/int8 code containers, fp16 norms; the sole exception is fp32 RTN scales, kept deliberately for bitwise parity (see Task 1 note) and disclosed. A test walks every block dict and asserts dtypes (the regression pin).
- Honest bits: ALL metadata counted; `bits_per_entry` on the packed cache must equal streaming's number on the same schedule (Task 4 pins it). Skeptic accounting primary (Gate A failed structurally; settled 2026-07-16). Memory deliverables are MEASURED resident GiB via the census instrument, delta-parity for quality (deltas vs own fp16, never absolute TurboQuant parity).
- Pack files are binary artifacts, **never committed**; regenerate deterministically (sidecar records SHA + corpus provenance).
- VM discipline per memories: don't kill long runs prematurely; per-cell checkpoints (`k3_niah` has `partial/` resume; the census rewrites its parquet after every cell); the 128k census task must clear the 96k headroom gate FIRST (the ~94.7 GiB prefill-mask peak precedent leaves thin margin under the 95.6 GiB HBM ceiling).

## File structure

- Modify `src/bmx/cache/spectral.py` — `spectral_quantize_packed` / `spectral_dequant_packed` + tier-segment containers; `spectral_quantize` re-expressed as their composition.
- Modify `src/bmx/cache/packed_streaming.py` — spectral K write-branch + per-layer pack loading (weakref `_make_layer`), V container packing (spectral path only), fp32 shared-RoPE variant, attend() threading + honest warn, bpe accumulation.
- Modify `src/bmx/cache/chunked_attention.py` — spectral read-branch (`k_pack` threading, fp32-RoPE-then-cast order), `turboquant_mse` transient-unpack branch.
- Modify `src/bmx/cache/streaming.py` — extract `cache_bits_per_entry` / `kv_memory_report` free functions (shared accounting, zero numeric change).
- Modify `experiments/k3_kernel_census.py` — `pack_path` threading.
- Modify `scripts/profile_decode_ab.py` — spectral 3-way (drop the pack-gated skip), inverted path-probe assertion, `--logit-probe` mode.
- Tests: `tests/test_spectral.py` (append codec tests), `tests/test_packed_spectral.py` (new — write path, read path, parity ladder, accounting).

---

### Task 1: Spectral packed codec — `spectral_quantize_packed` / `spectral_dequant_packed` + tier containers

**Files:**
- Modify: `src/bmx/cache/spectral.py`
- Test: `tests/test_spectral.py` (append)

**Interfaces:**
- Consumes: `rtn_quantize_packed`/`rtn_dequantize_packed` (`bmx.quant.rtn` — note: **what `quantize_packed` produces for rtn arms is int8 code containers + fp32 group scales, NOT sub-byte storage**); the true sub-byte utilities are `pack_codes`/`unpack_codes` (unsigned, bits ∈ {2,4}, 8//bits codes per byte) and `pack_signed_codes`/`unpack_signed_codes` (signed 4-bit two's-complement nibbles, bits ≤ 4) in `bmx.cache.triton_dequant_attention` (import is CPU-safe; no cycle: codecs ← triton_dequant_attention ← spectral).
- Produces:
  - `tier_columns(bits: torch.Tensor) -> dict[int, torch.Tensor]` — ascending column indices per nonzero tier, iterating `sorted(set(bits.tolist()))` exactly as `quantize_by_bits` does (this IS the sort-by-tier permutation; scattering into these columns at dequant IS the inverse permutation).
  - `spectral_quantize_packed(M, pack, *, mse_scale=True, cols_by_tier=None) -> (dict, float)` — flat-key dict `{f"t{b}_codes": container, f"t{b}_scale": fp32 (n_b, S//group, 1)}`.
  - `spectral_dequant_packed(packed, pack, *, cols_by_tier=None) -> torch.Tensor` — fp32 `(S, C)` M_hat, **bitwise-equal** to `spectral_quantize(M, pack)[0]`.
  - `spectral_quantize` re-expressed as `spectral_dequant_packed(spectral_quantize_packed(...))` — bitwise-neutral by construction (`rtn_quantize` is literally `rtn_dequantize_packed(rtn_quantize_packed(...))`, `bmx/quant/rtn.py:56-65`, so per-tier compose == `quantize_by_bits`'s per-tier call; same enc/dec matmuls, same order, same device/dtype).
- **Container policy** (reuse-only, no new bit-twiddling): tier 2 and 4 → offset to unsigned (`+2^(b-1)`, RTN codes span `[-2^(b-1), 2^(b-1)-1]`) then `pack_codes` (exact width: 4 and 2 codes/byte); tier 3 → `pack_signed_codes` nibbles (4-bit container, +1 bit/code resident); tiers 5, 6, 8 → int8 as `rtn_quantize_packed` produced (exact at 8; +3/+2 resident at 5/6). Packing runs along the S axis (`(n_b, PAGE)`, PAGE=128 divisible by every per_byte). Resident-vs-accounted deltas are bounded and disclosed in the results doc; spike tiers (5/6/8) are few directions by construction (spiked spectra), so the overhead is small — and the census measures real bytes regardless.
- **fp32-scale decision (deliberate, disclosed):** the streaming spectral quantizer never fp16-roundtrips its RTN scales (`quantize_by_bits` → `rtn_quantize_packed` keeps fp32); storing fp16 scales here would break committed-block BITWISE parity, and changing streaming to roundtrip would invalidate the licensed duel numbers. So packed stores the exact fp32 scales, `bits_per_entry` accounts them at fp16 per the standing `scale_bits` convention (same accounting streaming reports), and the +16/group resident-vs-accounted delta (+0.25 bpe at group 64) is disclosed next to the measured GiB table. V norms are the opposite case: they ARE fp16-roundtripped at quantize time (`.half().float()`), so storing them as fp16 is exact.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_spectral.py`; `_spiked_keys` already exists there):

```python
@pytest.mark.parametrize("b", [2, 3, 4, 5, 6, 8])
def test_tier_container_roundtrip(b):
    from bmx.cache.spectral import _pack_tier_codes, _unpack_tier_codes

    qmax = 2 ** (b - 1) - 1
    codes = torch.randint(-qmax - 1, qmax + 1, (7, 128), dtype=torch.int8)
    packed = _pack_tier_codes(codes, b)
    assert packed.dtype in (torch.uint8, torch.int8)
    assert torch.equal(_unpack_tier_codes(packed, b, 128), codes)


def test_spectral_packed_bitwise_matches_spectral_quantize():
    from bmx.cache.spectral import (
        fit_spectral_basis,
        identity_whitener,
        pack_from_basis,
        spectral_dequant_packed,
        spectral_quantize,
        spectral_quantize_packed,
    )

    C, S = 32, 128
    Wh, Wh_inv = identity_whitener(C)
    M, _ = _spiked_keys(S=S, C=C, seed=0)
    basis = fit_spectral_basis(M, Wh, Wh_inv)
    pack = pack_from_basis(basis, 2.5, group=16)
    ref, ref_bpe = spectral_quantize(M, pack, mse_scale=True)
    packed, bpe = spectral_quantize_packed(M, pack, mse_scale=True)
    assert bpe == ref_bpe
    # Container discipline (the T4 pin at codec level): codes uint8/int8, scales fp32.
    for k, t in packed.items():
        if k.endswith("_codes"):
            assert t.dtype in (torch.uint8, torch.int8), (k, t.dtype)
        else:
            assert k.endswith("_scale") and t.dtype == torch.float32, (k, t.dtype)
    M_hat = spectral_dequant_packed(packed, pack)
    assert torch.equal(M_hat, ref)  # BITWISE — the codec-level parity anchor
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_spectral.py -k "tier_container or packed_bitwise" -v` → ImportError.
- [ ] **Step 3: Implement** in `spectral.py` (order matters — write the bitwise test against the CURRENT `spectral_quantize`, get the packed pair passing, THEN refactor `spectral_quantize`'s body to the composition so the test keeps pinning both):

```python
_SUBBYTE_TIERS = frozenset({2, 4})  # offset-binary via pack_codes (exact width)
_NIBBLE_TIERS = frozenset({3})      # signed nibbles via pack_signed_codes (4-bit container)


def _pack_tier_codes(Q_int: torch.Tensor, b: int) -> torch.Tensor:
    if b in _SUBBYTE_TIERS:
        return pack_codes(Q_int.to(torch.int16) + 2 ** (b - 1), b)
    if b in _NIBBLE_TIERS:
        return pack_signed_codes(Q_int, b)
    return Q_int  # int8 container (tiers 5, 6, 8)


def _unpack_tier_codes(t: torch.Tensor, b: int, S: int) -> torch.Tensor:
    if b in _SUBBYTE_TIERS:
        return (unpack_codes(t, b, S) - 2 ** (b - 1)).to(torch.int8)
    if b in _NIBBLE_TIERS:
        return unpack_signed_codes(t, b, S)
    return t


def spectral_quantize_packed(M, pack, *, mse_scale=True, cols_by_tier=None):
    S, C = M.shape
    assert pack.enc.shape == (C, C), f"pack C mismatch: {pack.enc.shape} vs C={C}"
    assert S % pack.group == 0, f"S={S} not divisible by group={pack.group}"
    cols_by_tier = cols_by_tier if cols_by_tier is not None else tier_columns(pack.bits)
    assert cols_by_tier, "pack allocates zero bits everywhere; nothing to store"
    Y = M @ pack.enc.to(M.dtype)
    packed: dict[str, torch.Tensor] = {}
    for b, cols in cols_by_tier.items():
        Q_int, scale = rtn_quantize_packed(Y[:, cols].mT, b, pack.group, mse_scale=mse_scale)
        packed[f"t{b}_codes"] = _pack_tier_codes(Q_int, b)
        packed[f"t{b}_scale"] = scale
    return packed, float(pack.bits.float().mean().item()) + scale_bits(pack.group)


def spectral_dequant_packed(packed, pack, *, cols_by_tier=None):
    cols_by_tier = cols_by_tier if cols_by_tier is not None else tier_columns(pack.bits)
    first_b = next(iter(cols_by_tier))
    S = packed[f"t{first_b}_scale"].shape[1] * pack.group  # scale is (n_b, S//group, 1)
    C = pack.enc.shape[0]
    Y_hat = torch.zeros(S, C, dtype=pack.dec.dtype, device=pack.dec.device)
    for b, cols in cols_by_tier.items():
        Q_int = _unpack_tier_codes(packed[f"t{b}_codes"], b, S)
        Y_hat[:, cols] = rtn_dequantize_packed(Q_int, packed[f"t{b}_scale"], pack.group).mT
    return Y_hat @ pack.dec.mT
```

`tier_columns` iterates `sorted(set(int(x) for x in bits.tolist()))` skipping 0 — byte-for-byte the `quantize_by_bits` loop. `cols_by_tier` is an optional precomputed arg because the read path calls this **per committed page per decode step per layer** (512 pages × 32 layers at 64k) — a `.tolist()` over C=1024 per call is decode-loop overhead; the layer computes it once (Task 2).
- [ ] **Step 4: Verify pass**, then refactor `spectral_quantize` to the composition and confirm the whole file's tests plus `uv run pytest tests/test_streaming_batched_flush.py -q` (the spectral bitwise A/B pin) stay green.
- [ ] **Step 5: Full battery + `ruff format`/`check` + stage + propose commit** `feat(spectral): packed codec form — per-tier sub-byte/int8 containers, bitwise-faithful to spectral_quantize`. STOP for approval.

---

### Task 2: PackedStreaming write path — spectral K-branch, pack loading, V containers, the guard move

**Files:**
- Modify: `src/bmx/cache/packed_streaming.py`
- Test: `tests/test_packed_spectral.py` (new)

**Interfaces:**
- Consumes: Task 1 codec; `load_packs`/`SpectralPack` (`spectral.py`); `pack_codes` (V containers); the existing per-page flush loop (`PackedStreamingLayer.update` — untouched; only `_pack_k_block`/`_pack_v_block` grow branches).
- Produces:
  - `PackedStreamingCache(model_config, k_spec, v_spec, ...)` with `k_spec.arm == "spectral"` loads packs ONCE at init and hands `packs[i]` to layer i. **The guard moves, not deleted:** today spectral through this cache dies at first flush with `quantize_packed`'s `NotImplementedError` (`codecs.py:607-611`); after this task, spectral is intercepted BEFORE `quantize_packed` and the guard becomes init-time asserts (`pre_rope`, `pack_path`, per-layer pack present) — fail at construction with a clear message. `quantize_packed`'s own NotImplementedError is UNTOUCHED (it still guards any other pack-gated/misrouted arm reaching it).
  - Per-layer pack handoff uses the **weakref `_make_layer` closure pattern from streaming.py** (`streaming.py:543-555`) — the closure needs `len(self.layers)`, and a strong `self` capture is the exact self→closure→self cycle that produced the 88 GiB gen-2-gc OOM (commit `562d696`, pinned by `test_cache_freed_by_refcount_after_generation`). `attach()`'s pre-size loop switches to `self._make_layer()` (one construction path, matching streaming).
  - `PackedStreamingLayer(..., pack: SpectralPack | None = None)`: asserts `self._page % pack.group == 0` (PAGE alignment is against the PACK's sidecar group — what `spectral_quantize_packed` asserts along S — not `spec.group`); precomputes `self._tier_cols = tier_columns(pack.bits)` once (CPU, before any device move); one-time device move of pack tensors AND `_tier_cols` at first flush (mirroring `streaming.py:304-311`'s `dataclasses.replace` idiom — index tensors must ride along or CUDA indexing takes the slow cross-device path).
  - `_pack_k_block` spectral branch (before the `quantize_packed` call), returning `(packed, bpe)` — the signature change also returns bpe for the existing arms (Task 4 consumes it; today the bpe is discarded at the `quantize_packed` call sites).
  - `_pack_v_block` V-container packing, **scoped to the spectral path only** so every existing arm's block dicts stay byte-identical: when `self.k_spec.arm == "spectral"` and the V arm is turboquant-family with `8 % bits == 0 and bits < 8`, replace `"indices"` (int16) with `"indices_packed" = pack_codes(indices, bits)` (uint8, 4 codes/byte at 2b) and store `"norms"` as `.half()` (exact — already fp16-roundtripped). This is the established W5-2 replace-key idiom (`_repoint_k2b_blocks` docstring); consumers go through the transient unpack (Task 3), never store back.

- [ ] **Step 1: Failing tests** (new file `tests/test_packed_spectral.py`; reuse `_fit_tiny_packs` from `tests/test_streaming_spectral.py` — group=8, so tiny PAGE=128 % 8 == 0 holds):

```python
import torch

from bmx.cache.specs import CacheCodecSpec
from bmx.cache.packed_streaming import PackedStreamingCache
from tests.factories import ids, tiny_llama
from tests.test_streaming_spectral import _fit_tiny_packs


def _k4_specs(path, group=8, budget=2.5):
    return (
        CacheCodecSpec(arm="spectral", pre_rope=True, group=group, pack_path=path, budget=budget),
        CacheCodecSpec(arm="turboquant_mse", bits=2, seed=0),
    )


def test_packed_spectral_guards(tmp_path):
    import pytest

    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    k, v = _k4_specs(path)
    with pytest.raises(AssertionError, match="pre_rope"):
        PackedStreamingCache(
            model.config,
            k_spec=CacheCodecSpec(arm="spectral", pack_path=path, budget=2.5),
            v_spec=v,
        )
    with pytest.raises(AssertionError, match="pack_path"):
        PackedStreamingCache(
            model.config,
            k_spec=CacheCodecSpec(arm="spectral", pre_rope=True),
            v_spec=v,
        )


def test_packed_spectral_container_discipline(tmp_path):
    """The T4 pin: committed pages hold packed dtypes ONLY — no int16 indices,
    no fp32/fp16 dense codes. seq=300, W=8 flushes two 128-token pages."""
    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    k, v = _k4_specs(path)
    cache = PackedStreamingCache(model.config, k_spec=k, v_spec=v, recent_window=8)
    cache.attach(model)
    with cache:
        with torch.no_grad():
            model(ids(seq=300), past_key_values=cache, use_cache=True)
    layer = cache.layers[0]
    assert len(layer._k_blocks) == 2 and layer._committed_S_q == 256
    for kp, _s, _e in layer._k_blocks:
        for key, t in kp.items():
            if key.endswith("_codes"):
                assert t.dtype in (torch.uint8, torch.int8), (key, t.dtype)
            else:
                assert key.endswith("_scale") and t.dtype == torch.float32
    for vp, _s, _e in layer._v_blocks:
        assert "indices" not in vp and vp["indices_packed"].dtype == torch.uint8
        assert vp["norms"].dtype == torch.float16
```

- [ ] **Step 2: Verify failure** (today: `NotImplementedError: arm 'spectral' not split into packed form` out of the first flush — the guard this task moves to init).
- [ ] **Step 3: Implement** per the interfaces above. The K branch body:

```python
        if spec.arm == "spectral":
            if self._pack.enc.device != M.device:  # one-time device placement
                self._pack = dataclasses.replace(
                    self._pack,
                    enc=self._pack.enc.to(M.device), dec=self._pack.dec.to(M.device),
                    lam=self._pack.lam.to(M.device), bits=self._pack.bits.to(M.device),
                )
                self._tier_cols = {b: c.to(M.device) for b, c in self._tier_cols.items()}
            packed, bpe = spectral_quantize_packed(
                M, self._pack, mse_scale=True, cols_by_tier=self._tier_cols
            )
```

(`_extend_rope(block_end, device)` stays exactly where it is in `_pack_k_block`.)
- [ ] **Step 4: Verify pass** + `uv run pytest tests/test_packed_streaming.py tests/test_streaming_spectral.py -q` (existing arms untouched, streaming untouched).
- [ ] **Step 5: battery + stage + propose** `feat(packed): spectral K write-branch — per-tier packed pages + packed V containers; pack-gated guard moved to cache init`. STOP.

---

### Task 3: Read path — chunked spectral branch, fp32 RoPE-at-read, the parity ladder

**Files:**
- Modify: `src/bmx/cache/chunked_attention.py`, `src/bmx/cache/packed_streaming.py` (attend threading, `_shared_rope` dtype, warn message)
- Test: `tests/test_packed_spectral.py` (append)

**Interfaces:**
- Consumes: Task 1 `spectral_dequant_packed`; the existing `k_pre_rope` RoPE-at-read structure in `chunked_attention.py` (the streaming K-branch precedent for table handling is `streaming.py:324-339`: fp32 tables from `rope_cos_sin`, sliced at TRUE positions `[start:end)`, applied in fp32, THEN cast to fp16).
- Produces:
  - `_shared_rope(config, upto, device, dtype=torch.float16)` — dtype joins the key. Spectral layers request **fp32** (`self._rope_dtype`); everything else keeps fp16 (byte-identical behavior). Why: streaming's committed prefix is `apply_rope(M_hat_fp32, cos_fp32, sin_fp32).to(fp16)`; committed-block BITWISE parity is only reachable if the packed read replays that exact op order — fp16 tables (or rope-after-fp16-cast) both break it. Cost: one shared fp32 pair per process, ~134 MB at 128k (vs 67 MB fp16) — still the shared design that killed the 0.5 GiB/cache duplication, and only paid when a spectral cache exists.
  - `chunked_dequant_attention(..., k_pack: SpectralPack | None = None)` + `_dequant_block(..., pack=None)` + threading through `_dense_kv`/`_assemble_dense_kv`/`_prefill_dense_attention`/`naive_dense_attention` (the ORACLE must cover the new arm — oracle-gated discipline).
  - `_dequant_block` gains two branches: (a) `arm == "spectral"` → `from_matrix(spectral_dequant_packed(packed, pack), h_kv)` (fp32 out); (b) the existing W5-2 transient-unpack condition widens from `arm == "turboquant_mse_perhead"` to `arm in ("turboquant_mse", "turboquant_mse_perhead")`, and restores `"norms": packed["norms"].float()` (a no-op for the existing fp32-norm dicts, exact for Task 2's fp16-stored norms). Transient only — never stored back (the `block_v_indices` contract).
  - Decode-loop order branch, spectral only (k2b keeps today's cast-then-rope, pinned by its parity tests):

```python
        if k_arm == "spectral":
            K_kv = _dequant_block(kpacked, k_arm, group, seed, h_kv, pack=k_pack)  # fp32
            K_kv = apply_rope(K_kv, rope_cos[start:end], rope_sin[start:end])       # fp32 tables
            K_kv = K_kv.to(q.dtype)  # fp32-rope-then-fp16-cast: streaming's exact order
        else:
            K_kv = _dequant_block(kpacked, k_arm, group, seed, h_kv).to(q.dtype)
            if k_pre_rope:
                K_kv = apply_rope(K_kv, rope_cos[start:end], rope_sin[start:end])
```

  (`_dense_kv` — the prefill/oracle path — already dequants fp32, ropes with `rope_cos[start:end].to(B.dtype)`, and casts later, so with fp32 tables it is bitwise-correct without a branch.)
  - `attend()` threads `k_pack=self._pack`; the F0-lesson CUDA-decode warn is split so benchmarks can't misattribute cost: for spectral, warn "spectral decode runs the CHUNKED path BY DESIGN (Phase A resident-memory path, no fused spectral kernel); expect chunked-class decode latency" — the existing "use rtn/k2b_ph or accept the cost" warn stays verbatim for genuinely misrouted arms.

- [ ] **Step 1: Failing tests** (append to `tests/test_packed_spectral.py`) — the parity ladder, strongest first:

```python
def _run_pair(model, path, seq, group=8):
    from bmx.cache.streaming import StreamingQuantizedCache

    k, v = _k4_specs(path, group=group)
    caches = []
    for Cls in (StreamingQuantizedCache, PackedStreamingCache):
        cache = Cls(model.config, k_spec=k, v_spec=v, recent_window=8)
        cache.attach(model)
        with cache:
            with torch.no_grad():
                model(ids(seq=seq), past_key_values=cache, use_cache=True)
        caches.append(cache)
    return caches


def test_committed_blocks_bitwise_match_streaming(tmp_path):
    """BINDING GATE 1: same compute_flush_schedule, same PAGE=128 pages — the
    packed read-path reconstruction of every committed page (dequant -> fp32
    RoPE at true positions -> fp16 cast) equals streaming's frozen prefix
    bit-for-bit; V pages likewise."""
    from bmx.cache.chunked_attention import _dequant_block
    from bmx.cache.rope import apply_rope

    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    stream, packed = _run_pair(model, path, seq=300)
    for i, (sl, pl) in enumerate(zip(stream.layers, packed.layers)):
        assert sl._committed_S_q == pl._committed_S_q == 256  # shared schedule
        for (kp, s, e), (vp, _s, _e) in zip(pl._k_blocks, pl._v_blocks):
            K = _dequant_block(kp, "spectral", 8, 0, pl._h_kv, pack=pl._pack)
            K = apply_rope(K, pl._rope_cos[s:e], pl._rope_sin[s:e]).to(torch.float16)
            assert torch.equal(K, sl._q_prefix_k[:, s:e, :]), f"K layer {i} [{s}:{e})"
            V = _dequant_block(vp, "turboquant_mse", 8, 0, pl._h_kv).to(torch.float16)
            assert torch.equal(V, sl._q_prefix_v[:, s:e, :]), f"V layer {i} [{s}:{e})"


def test_packed_spectral_decode_matches_oracle(tmp_path):
    """Chunked online-softmax decode vs naive_dense_attention on the same
    committed blocks (the standing oracle gate; mirror test_chunked_attention's
    tolerance conventions)."""
    ...


def test_packed_spectral_generate_matches_streaming(tmp_path):
    """BINDING GATE 2 (short context): greedy tokens identical, both the
    no-flush (seq=120) and flush-during-prefill (seq=300) variants — mirrors
    test_packed_generate_matches_streaming / _long_prefill."""
    ...


def test_packed_spectral_two_block_prefill_logits_match_streaming(tmp_path):
    """The cached-two-block-prefill mask-bug class (cf21d06): logit parity on a
    prefill split across two forwards — mirrors
    test_packed_two_block_prefill_logits_match_streaming with the spectral arm."""
    ...
```

(The three `...` bodies clone the named existing tests in `tests/test_packed_streaming.py` with `_k4_specs`; read those tests first and keep their assertion styles/tolerances verbatim — they encode the June lessons.)
- [ ] **Step 2: Verify failure** (spectral hits `dequant_packed`'s NotImplementedError / missing pack threading).
- [ ] **Step 3: Implement** per the interfaces. Keep the `_dequant_block` signature change backward-compatible (`pack=None` default; assert `pack is not None` inside the spectral branch with a clear message).
- [ ] **Step 4: Verify pass** + `uv run pytest tests/ -q -k "packed or chunked or streaming"` (existing k2b/rtn parity and oracle tests must be untouched-green).
- [ ] **Step 5: battery + stage + propose** `feat(packed): spectral read path — chunked dequant-attention branch, fp32 RoPE-at-read, committed-block bitwise + greedy + prefill-logit parity vs streaming`. STOP.

---

### Task 4: Honest accounting on the packed cache — `bits_per_entry` + `memory_report`

**Files:**
- Modify: `src/bmx/cache/packed_streaming.py`, `src/bmx/cache/streaming.py`
- Test: `tests/test_packed_spectral.py` (append)

**Interfaces:**
- Background: the June census note — "chunked's bpe is NaN because PackedStreamingCache has no bits_per_entry accessor" — is still true; the deliverable table needs honest bpe columns next to measured GiB, for every packed arm (k2b included — a bonus fix).
- Produces:
  - `PackedStreamingLayer` accumulates `_quant_bits_k/_quant_bits_v` at flush time from the codec bpe now returned by `_pack_k_block`/`_pack_v_block` (Task 2 signature change; `quantize_packed`'s bpe was discarded before), and maintains blended `bpe_k/bpe_v` in `update()` with streaming's exact blend (quantized pages at codec bpe + fp16 tail at 16.0; tail length is `self.keys.shape[2]` post-prune). Per-page accumulation order matches streaming's per-page float sum, and lowrank `factor_bits` is charged at S=PAGE on both paths (streaming keeps lowrank arms on the per-page loop) — so the accounting is float-identical, not merely close.
  - `streaming.py` extracts two free functions with ZERO numeric change (pure code motion, both caches call them): `cache_bits_per_entry(layers, k_spec) -> (bpe_k, bpe_v)` (the across-layer mean + spectral skeptic charge — the body of `StreamingQuantizedCache.bits_per_entry`, `streaming.py:633-672`; PackedStreamingLayer exposes the same surface: `bpe_k/bpe_v`, `_committed_S_q`, `get_seq_length()`, `_h_kv/_d_head`, `_pack`) and `kv_memory_report(model_config, bpe_k, bpe_v, seq_len) -> dict` (the body of `memory_report`).
  - `PackedStreamingCache.bits_per_entry()` / `.memory_report(seq_len)` — thin wrappers over the shared functions.

- [ ] **Step 1: Failing test:**

```python
def test_packed_bits_per_entry_equals_streaming(tmp_path):
    """Same schedule, same codec calls per page => identical honest accounting,
    including the per-sequence skeptic pack charge. Also un-NaNs the census."""
    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    stream, packed = _run_pair(model, path, seq=300)
    assert packed.bits_per_entry() == stream.bits_per_entry()
    sm = stream.memory_report(seq_len=300)
    pm = packed.memory_report(seq_len=300)
    assert pm == sm  # same bpe in => same honest bytes out
```

- [ ] **Step 2: fail** (AttributeError), **Step 3: implement**, **Step 4: pass** + full streaming/codec test files green (the extraction must not move a single float).
- [ ] **Step 5: battery + stage + propose** `feat(packed): honest bits_per_entry/memory_report on PackedStreamingCache — shared accounting with streaming (census bpe un-NaN'd)`. STOP.

---

### Task 5: Instrument wiring — census `pack_path`, profile_decode_ab spectral 3-way + `--logit-probe`

**Files:**
- Modify: `experiments/k3_kernel_census.py`, `scripts/profile_decode_ab.py`
- Tests: none new (thin tyro scripts per the standing convention; behavior is exercised by Task 6 pre-flight and the VM battery). `k3_niah`/`k3_longbench` need NO changes — `--use-packed` and `--pack-path` already exist and `generate_through_cache` routes spectral specs (pack_path rides on the spec) to `PackedStreamingCache` as of Task 2.

**Interfaces / changes:**
- `k3_kernel_census.Config` gains `pack_path: str = ""`; the `spec_pair(arm)` call becomes `spec_pair(arm, pack_path=cfg.pack_path)` (today a k4 arm raises `ValueError` there). Everything else (per-cell parquet rewrite, OOM sentinel rows, fp16×chunked skip) already serves this plan.
- `profile_decode_ab`:
  - Remove the `PACK_GATED_ARMS` dense+streaming-only early-out — spectral now HAS a packed twin. The short-ctx greedy parity gate (streaming == packed over `n_parity` tokens at `min(ctx_lens)`) now runs for spectral — binding gate 2 at real-model scale.
  - Path probe: for pack-gated arms INVERT the assertion — require `counts["chunked"] > 0` and `fused_total == 0` ("chunked by design" is the licensed Phase-A path; a fused count would mean a kernel was silently wired without its oracle gates).
  - New `--logit-probe N` (default 0): at `max(ctx_lens)`, prefill both caches on the same prompt, then N teacher-forced decode steps (both caches fed streaming's greedy token, so trajectories stay comparable); per step record `max|logits_packed − logits_streaming|` and argmax-flip count. **Gate: 0 flips over N ≥ 32; the max-abs envelope is RECORDED against the documented O(0.25–1.45) 64k class, never gated bitwise** (Global Constraints; 2026-07-15 appendix). Prints a per-step table + summary row.
- [ ] **Step 1:** Implement; `uv run python scripts/profile_decode_ab.py --help` and `uv run python experiments/k3_kernel_census.py --help` parse clean.
- [ ] **Step 2:** battery + stage + propose `feat(instruments): census --pack-path; profile_decode_ab spectral 3-way, inverted path probe, --logit-probe (64k tolerance gate)`. STOP.

---

### Task 6: Local pre-flight gate + push

**Files:** none new — verification + transport.

- [ ] **Step 1:** Full battery clean (expect baseline 451/17/1 + the new tests; record the new count). `uv run ruff format .` / `check` clean.
- [ ] **Step 2:** CPU mechanism pre-flight (scratchpad script, tiny_llama): build the Task-3 `_run_pair` at seq=1200, print (a) `bits_per_entry()` both caches (equal), (b) measured committed-storage bytes on the packed side (`sum(t.numel() * t.element_size())` over all block-dict tensors, K and V separately) vs the streaming prefix's fp16 bytes — the ratio must land near `(bpe + disclosed container overhead)/16`, PRINT the arithmetic. This is the memory MECHANISM check the CPU can do; real resident GiB is CUDA-only (AMD box, no CUDA here).
- [ ] **Step 3:** STOP — propose pushing `feat/triton-decode-kernel` to origin (VM transport prerequisite). User approves the push explicitly.

---

## VM batch (Tasks 7–11) — one rented GH200, ordered, each gated

Transport per `vm-interaction-guide`: push → VM pull (or git bundle — VM has no push creds; results come back by bundle), `scripts/vm_setup.sh`, then the battery. Long runs: `setsid` detached + log under `results/logs/`; NEVER sudo pip; never kill a long run prematurely.

### Task 7: [VM-RUN] GH200 gate battery

```bash
cd ~/bmx && git pull && uv sync
uv run pytest -q                          # record the GH200 count (local 451+/17/1 + Triton extras)
uv run pytest tests/test_packed_spectral.py tests/test_streaming_batched_flush.py tests/test_spectral.py -q
```

The second line re-pins the whole parity ladder ON CUDA — the codec bitwise test, committed-block bitwise, greedy, prefill-logit — because reduction-split/BLAS kernel selection is shape- and device-dependent (the standing reason CPU equality alone doesn't license CUDA, `streaming.py:_flush_batchable` docstring). All green before any spend. If a CUDA-only bitwise failure appears, STOP and diagnose (June precedent: both real prior CUDA bugs surfaced exactly here).

### Task 8: [VM-RUN] Packs + real-model parity ladder (GATES; cheap; before any census/NIAH spend)

Packs: reuse `results/cache/k4_packs_llama31_instruct.safetensors` if still on the VM from the duel batch (check the JSON sidecar's git SHA + corpus provenance); else regenerate deterministically (~minutes):

```bash
for OFF in 2048 4096 6144 8192; do
  uv run python experiments/collect_cache.py --model-name meta-llama/Llama-3.1-8B-Instruct --seq-len 2048 --token-offset $OFF
done
uv run python -m experiments.k4_fit_packs \
  --corpus-cache-paths results/cache/llama-3.1-8b-instruct_2048_off2048.safetensors results/cache/llama-3.1-8b-instruct_2048_off4096.safetensors results/cache/llama-3.1-8b-instruct_2048_off6144.safetensors results/cache/llama-3.1-8b-instruct_2048_off8192.safetensors \
  --out-path results/cache/k4_packs_llama31_instruct.safetensors --model-label llama-3.1-8b-instruct \
  --model-name meta-llama/Llama-3.1-8B-Instruct --w-source corpus
```

Then the ladder at real scale (one command, ~minutes):

```bash
uv run python scripts/profile_decode_ab.py --model-name meta-llama/Llama-3.1-8B-Instruct \
  --arm k4_b2.5 --ctx-lens 4096 16384 65536 --logit-probe 32 \
  --pack-path results/cache/k4_packs_llama31_instruct.safetensors
```

**Gate A (greedy@4k):** streaming == packed greedy tokens — hard fail stops the batch. **Gate B (path probe):** all-chunked, zero fused. **Gate C (logit tolerance @64k):** 0 argmax flips over 32 teacher-forced steps; record the max-abs envelope vs the documented O(0.25–1.45) class. This run's 3-way ms/token table is ALSO the Phase-B gate measurement — keep it (expected: chunked spectral decode well above streaming's 38.7 ms at 64k; the spectral dequant is a per-page (128,C)@(C,C) fp32 matmul × 512 pages × 32 layers per token — record, don't pre-judge).

### Task 9: [VM-RUN] The memory census — 96k FIRST, then the gated 128k

**Stage 1 (32k/64k/96k):**

```bash
mkdir -p results/logs
setsid nohup uv run python -m experiments.k3_kernel_census \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --arms fp16 k2b k4_b2.5 --seq-lens 32768 65536 98304 \
  --pack-path results/cache/k4_packs_llama31_instruct.safetensors \
  > results/logs/census_stage1.log 2>&1 &
```

(`k2b` rides as the June-census continuity anchor — its 32k/64k chunked numbers must reproduce ~27.3/39.5 GiB, drift means an environment change, stop and diagnose. The census writes its parquet after EVERY cell; an OOM is a sentinel row, not a crash.)

**The 128k headroom gate (binding — run BEFORE any 128k cell):** read the stage-1 parquet; for each path × arm, project `resident_128k ≈ resident_96k + (resident_96k − resident_64k)` (growth is linear — June table). Proceed to 128k for a cell only if its projection ≤ **90 GiB** (≥5.6 GiB margin under the 95.6 GiB HBM ceiling; the prefill-mask transient once pushed a packed peak to ~94.7 GiB — that precedent is the reason this gate exists). `dense_stream` rows projected over the ceiling are EXPECTED (that's the point of the table) — run them anyway to get the honest OOM sentinel, but run them LAST in the process order and one arm per invocation so an OOM can't poison earlier cells' allocator state. **This gate is a human read-and-decide step — the GPU is idle while it happens; don't leave Stage 1 running unattended overnight expecting Stage 2 to follow automatically, there is no script here that computes the projection and launches for you:**

```bash
setsid nohup uv run python -m experiments.k3_kernel_census \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --arms fp16 k4_b2.5 --seq-lens 131072 \
  --pack-path results/cache/k4_packs_llama31_instruct.safetensors \
  > results/logs/census_128k_a.log 2>&1 &
wait  # BLOCKS until census_128k_a exits — do not launch the isolated k2b cell until it does
# now the expected-OOM continuity cell, isolated (only after the line above returns):
setsid nohup uv run python -m experiments.k3_kernel_census \
  --model-name meta-llama/Llama-3.1-8B-Instruct --arms k2b --seq-lens 131072 \
  --pack-path results/cache/k4_packs_llama31_instruct.safetensors \
  > results/logs/census_128k_b.log 2>&1 &
```

**Deliverable table** (the plan's headline): resident GiB at {32k, 64k, 96k, 128k} × {packed-spectral (`k4_b2.5` chunked), streaming-spectral (`k4_b2.5` dense_stream), dense fp16}, with honest `bpe_k/bpe_v` columns (Task 4) beside the measured bytes, plus the k2b anchor rows. Reading note for the doc: packed-spectral resident includes the on-device pack matrices (enc+dec fp32, ~268 MB for 32 layers at C=1024) and fp32 scales — both disclosed, both inside the measured number.

### Task 10: [VM-RUN] 128k/64k NIAH through the packed path (delta-parity)

```bash
setsid nohup uv run python -m experiments.k3_niah --model-name meta-llama/Llama-3.1-8B-Instruct \
  --device cuda --arms fp16 k4_b2.5 k2b --lengths 65536 131072 --depths 0.25 0.5 0.75 \
  --use-packed --pack-path results/cache/k4_packs_llama31_instruct.safetensors \
  > results/logs/niah_packed_128k.log 2>&1 &
```

- fp16 routes to the dense cache automatically (`generate_through_cache` is_fp16 rule — the uncompressed baseline, one KV copy, fits at 128k); `k4_b2.5` and `k2b` run packed. Cells are sequential (no co-residency), matching the June 3-arm×3-depth precedent exactly; `partial/` shards resume a killed run (`--resume <run_dir>` — the CLI flag is `--resume`, not `--resume-from`; it takes the crashed run's `run_dir` path directly and is identity-asserted against `config.json` + git SHA).
- **Delta-parity discipline:** report recall as deltas vs the SAME RUN's fp16 rows, never absolute (anchor forensics, `dd84143`). Success bar: k4_b2.5 recall-delta at 64k reproduces the streaming-path result (7.71 vs fp16 6.76 in the duel table — a packed-path number in the same class closes quality), and the 128k row is the FIRST spectral 128k evidence, whatever it reads.
- Known risk, accepted: `compression_for` runs a streaming prefill per (arm, length) cell for the bpe columns — at 128k that's the 83.5-GiB-class transient; it fit in the June 128k process, and a per-cell fresh relaunch via `--resume <run_dir>` is the cheap fallback if fragmentation bites.
- Prefill-rate note: packed spectral flushes per PAGE (512 codec calls/layer at 64k — no batched super-spans on the packed path). k2b's per-page SVD flush at 128k was acceptable in June; if spectral prefill is pathologically slower, record it as a finding (a batched-pack span mirroring `_flush_spans` is the known remedy, NOT to be built mid-run).

### Task 11: [VM-RUN] Results doc + traceability + bundle back

Write `docs/2026-07-XX-packed-spectral-results.md` (kill-or-confirm style, both outcomes pre-drafted): the resident-GiB table with bpe columns; the parity-gate ledger (Gates A/B/C outcomes + the CUDA battery count); NIAH deltas at 64k/128k; the 3-way latency table with the **Phase-B verdict**: fused spectral kernel proceeds ONLY if (a) chunked decode latency is prohibitive for a deployment claim AND (b) a latency/RSS claim is worth making under `publishable-bar` (parity+systems is not enough) — otherwise Phase B is explicitly declined in writing. Cross-reference §6b of `docs/2026-07-15-k4-duel-results.md` (its "packed spectral path (future work)" line is now measured — do not edit the banked doc, the new doc supersedes the scoping note). Disclose: fp32-scale (+0.25 bpe) and tier-container (3/5/6-bit) resident-vs-accounted deltas, with the measured bytes as ground truth. Commit parquets + doc, bundle back, STOP for approval.

---

### Task 12: Phase B — fused spectral decode kernel (SKETCH ONLY; gated on Task 8/11 latency verdict)

NOT planned here; requirements a future plan must meet, recorded so the gate decision is concrete:

- **Layout:** the uniform PAGE=128 paged layout and flat `t{b}_codes`/`t{b}_scale` keys are already kernel-shaped; per-layer tier segment shapes are CONSTANT across pages (bits fixed per layer), so `_PagedStacks`' dict mode can stack them — but any stacking work re-enters the double-buffer/repoint territory (W5-1 history) and must plan the single-storage story up front.
- **Kernel:** in-kernel per-tier unpack (the W5-2/W5-3 constexpr-branch precedent), per-tier dequant, `Y_hat @ dec.mT` as `tl.dot` with a per-layer fp16-resident `dec` (quality delta of fp16-dec vs the fp32 chunked reference must be oracle-diffed — this alone may change parity class), in-kernel RoPE (reuse the k2b machinery). Split-KV single-launch like `fused_decode_attention_k2b`.
- **Gates:** naive-oracle diff AND end-to-end logit parity before any speedup number (`oracle-gated-perf-work`); speedup is KV-fraction-bounded — only report at long context (`triton-decode-win-prediction`).

---

## Self-Review

**Binding-decision coverage:** packed storage + chunked, no Triton → Tasks 1–3 (no `_PagedStacks`, no kernel; Phase B sketched only, Task 12). Per-tier segments, sort-by-tier permutation precomputed once, uniform width per segment, reuse of existing packers → Task 1 (`tier_columns` + container policy; the "what does quantize_packed produce" question answered: int8 containers + fp32 scales, with the true sub-byte utilities being `pack_codes`/`pack_signed_codes`). Per-group scales stored → fp32, deliberately (bitwise-parity constraint), accounted at fp16, disclosed — the one deviation from the letter of "scales stored fp16", flagged here so it isn't "fixed" into a parity break. Dequant per page-chunk → `spectral_dequant_packed` + the chunked branch. RoPE-at-read mirroring the streaming K-branch's table handling → fp32 dtype-keyed `_shared_rope` + rope-then-cast order (Task 3). V `turboquant_mse@2` support checked and stated: functionally already supported; container int16 → packed in Task 2 (T4). Parity gates: bitwise committed-block (Task 3 test), greedy@4k (Task 3 + Task 8 Gate A), logit-tolerance@64k (Task 5 `--logit-probe` + Task 8 Gate C); NO bitwise-64k gate anywhere. Guard move: init-time asserts, `quantize_packed`'s error untouched (Task 2). Memory accounting via the census instrument with 96k-before-128k (Task 9); NIAH 64k/128k delta-parity (Task 10). VM runbook: exact commands, setsid detached, per-cell checkpoints (Tasks 7–11).

**Known-risk coverage:** unpacked containers → Task 2 discipline test + Task 1 codec-level dtype asserts; stacks double-buffering → structurally avoided (no stacks in Phase A; called out in Task 12 for Phase B); 128k mask peak → Task 9 headroom gate with the 90 GiB threshold and isolated expected-OOM cell; pack-gated guard → moved, not deleted; the weakref closure cycle (88 GiB lesson) → Task 2 `_make_layer`; F0 misattribution → Task 3 warn split + Task 5 inverted path probe.

**Placeholder scan:** all code steps carry real code except three test bodies in Task 3 explicitly defined as clones of NAMED existing tests (pattern + tolerance source stated) — deliberate, since those tests' assertion styles are the June lessons and must be copied from source, not paraphrased. VM commands are exact; the one variable (`k4_packs_llama31_instruct.safetensors` reuse-or-refit) has both branches spelled out.

**Type consistency:** `spectral_quantize_packed(M, pack, *, mse_scale, cols_by_tier)` / `spectral_dequant_packed(packed, pack, *, cols_by_tier)` consistent across Tasks 1/2/3; `_dequant_block(..., pack=None)` and `chunked_dequant_attention(..., k_pack=None)` — two names, one object, DELIBERATE (block-level vs call-level scope, mirroring the existing `group`/`v_group` convention); block dict keys `t{b}_codes`/`t{b}_scale` (K) and `indices_packed`/`norms` (V) consistent across Tasks 1/2/3 and the Task 12 sketch. fp32 K scales vs fp16 V norms is NOT an inconsistency: norms are fp16-roundtripped at quantize time (exact to store fp16), scales are not (fp32 required for bitwise). fp16 RoPE for k2b vs fp32 for spectral is NOT an inconsistency: k2b's fp16-rope behavior is pinned by its existing parity tests; spectral's bitwise gate requires streaming's fp32 order.

**Open decisions deferred to run time:** whether `k4_b2.2` joins the census/NIAH arms (add only if the stage-1 census leaves budget headroom; zero code impact — it's an `--arms` entry); the exact 128k projection threshold may tighten from 90 GiB if stage-1 measures a superlinear term (record the rule applied in the run log); the Phase-B go/no-go is Task 11's written verdict, not this plan's.
