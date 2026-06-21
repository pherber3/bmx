# Structured / Streamable Rotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a cheaper / streamable rotation (truncated top-k KLT, block-diagonal per-head KLT, or frozen-prefill full KLT) captures enough of the full-KLT eigenbasis win to beat uniform at HONEST bpe, with an oracle drift control and an eigengap diagnostic.

**Architecture:** Extend the existing `_lowrank_rotwaterfill_channel` core in `codecs.py` with four new rotation modes (`topk`, `blockdiag`, `frozen`, `oracle`), each producing an orthogonal (or block-orthogonal) Q applied to the key residual. The experiment `k2_waterfill.py` gains the corresponding arms plus a frozen/oracle generalization ratio and a per-layer residual-eigengap diagnostic. Offline, on collected caches.

**Tech Stack:** Python, PyTorch (CPU), tyro, pandas/parquet, pytest, uv, ruff.

**Spec:** `docs/superpowers/specs/2026-06-21-structured-rotation-design.md`

## Global Constraints

- Per-task commits PRE-AUTHORIZED (user: "commit as you go"). One clean single-sentence message, conventional prefix, NO AI attribution / Co-Authored-By.
- Before any commit: `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` — all clean.
- Use the Bash tool (git bash), NOT PowerShell. cwd may reset — run `cd "$(git rev-parse --show-toplevel)"` first in each Bash call. Run via `uv`.
- dtype: fp64 in tests, fp32 in codecs. fp16 factor roundtrip UNCONDITIONAL (matches `_lowrank_rtn_channel`; do NOT guard on dtype).
- Orthogonal sampling via `bmx.quant.hadamard.random_orthogonal`; eigenvectors via `torch.linalg.eigh` (ascending → flip to descending). Eigvec SIGN ambiguous — equivalence tests compare LOGIT distortion, never raw tensors.
- Honest bpe, ALL metadata counted: rotation cost is `topk: 16*k/S`, `blockdiag: 16*d/S`, `frozen: 16*C/S`. The verdict is on HONEST bpe.
- NAMING: these are rotation arms, NOT k2b's low-rank subtraction. Truncated arm is `..._topk`, never "lowrank".
- Score on `logit_rope` (headline) + `rel_fro` (MSE), both per arm. Tiny offline synthetics / GPT-2 fixture; never download.

---

## File Structure

- **Modify** `src/bmx/cache/codecs.py` — extend `_lowrank_rotwaterfill_channel` to accept `rotation ∈ {klt, random, topk, blockdiag, frozen, oracle}` plus `topk_k`, `prefill_fit_len` params; register the four new arms; thread params through `quantize_cache`; add dispatch.
- **Modify** `tests/test_cache_codecs.py` — add rotation-mode tests.
- **Modify** `experiments/k2_waterfill.py` — add the new arms, the frozen/oracle ratio, the residual-eigengap diagnostic, honest-bpe verdict for all deployable arms.
- **Modify** `tests/test_k2_waterfill.py` — extend smoke test for the new arms + diagnostics.
- **Create** `docs/2026-06-21-k2-structured-rotation-results.md` — the verdict (Task 3).

---

## Task 1: Four new rotation modes in the core

**Files:**
- Modify: `src/bmx/cache/codecs.py`
- Test: `tests/test_cache_codecs.py`

**Interfaces:**
- Consumes (existing): `_lowrank_rotwaterfill_channel(M, budget_bits, group, rank, tiers, rotation, seed, charge_rotation, svd_factors)` — currently supports `rotation ∈ {klt, random}`; `allocate_channel_bits`; `truncated_svd`; `rtn_quantize`; `random_orthogonal`.
- Produces: the same core extended with `rotation ∈ {klt, random, topk, blockdiag, frozen, oracle}`, plus new kwargs `topk_k: int = 0` (used by topk), `prefill_fit_len: int = 0` (used by frozen), `h_kv: int = 0` (used by blockdiag to reshape channels into heads). Four new arms in `CACHE_ARMS` + `S_DIVISIBILITY_ARMS`: `lowrank_topkwaterfill_channel`, `lowrank_blockdiagwaterfill_channel`, `lowrank_frozenwaterfill_channel`, `lowrank_oraclewaterfill_channel`. Honest charge per mode: topk `16*topk_k/S`, blockdiag `16*d/S` (d=C/h_kv), frozen `16*C/S`, oracle uncharged (control).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cache_codecs.py` (helpers `_seeded_matrix`, `_channel_matrix`, `_qkv_for`, `_logit_distortion`, `from_matrix`, `truncated_svd`, `GROUP`, `quantize_cache`, `allocate_channel_bits`, `random_orthogonal` already imported/defined from prior tasks):

```python
def test_structured_arms_registered():
    from bmx.cache.codecs import CACHE_ARMS, S_DIVISIBILITY_ARMS

    for arm in (
        "lowrank_topkwaterfill_channel",
        "lowrank_blockdiagwaterfill_channel",
        "lowrank_frozenwaterfill_channel",
        "lowrank_oraclewaterfill_channel",
    ):
        assert arm in CACHE_ARMS
        assert arm in S_DIVISIBILITY_ARMS


def test_topk_reduces_to_full_klt_at_k_equals_C():
    # topk with k = C rotates the whole basis => matches full eigwaterfill on logit.
    M = _seeded_matrix(s=64, c=64, seed=4).double()
    h_kv = 2
    q = _qkv_for(M, h_kv=h_kv)
    factors = truncated_svd(M, 4)
    full, _ = quantize_cache(
        "lowrank_eigwaterfill_channel", M, bits=3, group=GROUP, rank=4,
        tiers=(0, 2, 3, 4), svd_factors=factors,
    )
    topk, _ = quantize_cache(
        "lowrank_topkwaterfill_channel", M, bits=3, group=GROUP, rank=4,
        tiers=(0, 2, 3, 4), topk_k=64, svd_factors=factors,
    )
    lg_full = _logit_distortion(from_matrix(M, h_kv).double(), from_matrix(full, h_kv).double(), q)
    lg_topk = _logit_distortion(from_matrix(M, h_kv).double(), from_matrix(topk, h_kv).double(), q)
    assert abs(lg_full - lg_topk) < 0.02, f"topk@k=C diverged from full KLT: {lg_full} vs {lg_topk}"


def test_topk_partial_rotation_lossless_no_quant():
    # With a single high tier (near-lossless), topk reconstruct ~ M (orthogonality of
    # the partial rotation must hold — the stored top-k + recomputed complement).
    M = _seeded_matrix(s=64, c=64, seed=5).double()
    factors = truncated_svd(M, 4)
    topk, _ = quantize_cache(
        "lowrank_topkwaterfill_channel", M, bits=8, group=GROUP, rank=4,
        tiers=(8,), topk_k=16, svd_factors=factors,
    )
    rel = ((topk - M).norm() / M.norm()).item()
    # near-lossless at 8 bits + the rank-4 fp16 low-rank floor; bounded small, not garbage
    assert rel < 0.05, f"topk near-lossless reconstruction too large: {rel}"


def test_topk_honest_charge():
    S_, C_, group_, rank_, k_ = 64, 32, 16, 2, 8
    M = _seeded_matrix(s=S_, c=C_, seed=5).double()
    _, bpe_ideal = quantize_cache(
        "lowrank_topkwaterfill_channel", M, bits=3, group=group_, rank=rank_,
        tiers=(0, 2, 3, 4), topk_k=k_, charge_rotation=False,
    )
    _, bpe_honest = quantize_cache(
        "lowrank_topkwaterfill_channel", M, bits=3, group=group_, rank=rank_,
        tiers=(0, 2, 3, 4), topk_k=k_, charge_rotation=True,
    )
    assert abs((bpe_honest - bpe_ideal) - 16.0 * k_ / S_) < 1e-9


def test_blockdiag_no_cross_head_mixing():
    # The block-diagonal rotation must quantize each head's residual using ONLY that
    # head's own KLT. Test the residual-quantization step directly (bypassing the
    # shared low-rank L, which can couple heads): pass svd_factors with rank that makes
    # L negligible, then perturb head 0's residual and confirm head 1's reconstruction
    # is bit-identical. Cross-head leakage in the rotation would change head 1.
    import torch as _t

    h_kv, S_, d = 2, 64, 16  # C = 32
    C_ = h_kv * d
    M = _seeded_matrix(s=S_, c=C_, seed=6).double()
    factors = truncated_svd(M, 4)
    L = factors[0].half().float() @ factors[1].half().float().mT
    out1, _ = quantize_cache(
        "lowrank_blockdiagwaterfill_channel", M, bits=8, group=16, rank=4,
        tiers=(8,), h_kv=h_kv, svd_factors=factors,
    )
    # Build M2 = M but with head-0 columns replaced; reuse the SAME L (same factors) by
    # constructing M2 so M2 - L differs from M - L only in head 0's residual block.
    delta = _seeded_matrix(s=S_, c=d, seed=99).double()
    M2 = M.clone()
    M2[:, :d] = M[:, :d] + delta  # perturb head-0 residual only
    out2, _ = quantize_cache(
        "lowrank_blockdiagwaterfill_channel", M2, bits=8, group=16, rank=4,
        tiers=(8,), h_kv=h_kv, svd_factors=factors,  # SAME factors => same L
    )
    # head 1 (cols d:) reconstruction must be UNCHANGED by perturbing head 0 (no mixing).
    head1_diff = (out1[:, d:] - out2[:, d:]).abs().max().item()
    assert head1_diff < 1e-9, f"cross-head leakage: head-1 changed by {head1_diff}"


def test_frozen_vs_oracle_detects_drift():
    # Stationary residual: frozen (fit on prefix) ~ oracle (fit on all). Drifting
    # residual: oracle beats frozen. Proves the frozen/oracle ratio detects drift.
    import torch as _t

    h_kv = 2
    S_, C_ = 128, 32
    g = _t.Generator().manual_seed(3)
    # stationary: one fixed channel covariance for the whole sequence
    base = _t.randn(S_, C_, generator=g, dtype=_t.float64)
    stds = _t.tensor([0.2, 1.0, 5.0, 25.0] * 8, dtype=_t.float64)
    M_stat = base * stds
    q = _qkv_for(M_stat, h_kv=h_kv)
    factors_s = truncated_svd(M_stat, 4)
    froz_s, _ = quantize_cache(
        "lowrank_frozenwaterfill_channel", M_stat, bits=3, group=GROUP, rank=4,
        tiers=(0, 2, 3, 4), prefill_fit_len=64, svd_factors=factors_s,
    )
    orac_s, _ = quantize_cache(
        "lowrank_oraclewaterfill_channel", M_stat, bits=3, group=GROUP, rank=4,
        tiers=(0, 2, 3, 4), svd_factors=factors_s,
    )
    lg_froz_s = _logit_distortion(from_matrix(M_stat, h_kv).double(), from_matrix(froz_s, h_kv).double(), q)
    lg_orac_s = _logit_distortion(from_matrix(M_stat, h_kv).double(), from_matrix(orac_s, h_kv).double(), q)
    # stationary: frozen close to oracle
    assert abs(lg_froz_s - lg_orac_s) < 0.02, f"stationary frozen!=oracle: {lg_froz_s} vs {lg_orac_s}"

    # drifting: second half uses a rotated covariance => prefill-fit Q is wrong there
    Qrot = random_orthogonal(C_, seed=7, dtype=_t.float64)
    M_drift = M_stat.clone()
    M_drift[S_ // 2:] = M_stat[S_ // 2:] @ Qrot
    factors_d = truncated_svd(M_drift, 4)
    froz_d, _ = quantize_cache(
        "lowrank_frozenwaterfill_channel", M_drift, bits=3, group=GROUP, rank=4,
        tiers=(0, 2, 3, 4), prefill_fit_len=64, svd_factors=factors_d,
    )
    orac_d, _ = quantize_cache(
        "lowrank_oraclewaterfill_channel", M_drift, bits=3, group=GROUP, rank=4,
        tiers=(0, 2, 3, 4), svd_factors=factors_d,
    )
    lg_froz_d = _logit_distortion(from_matrix(M_drift, h_kv).double(), from_matrix(froz_d, h_kv).double(), q)
    lg_orac_d = _logit_distortion(from_matrix(M_drift, h_kv).double(), from_matrix(orac_d, h_kv).double(), q)
    # drifting: oracle should be at least as good as frozen (refit sees the drift)
    assert lg_orac_d <= lg_froz_d + 1e-9, f"oracle did not beat frozen under drift: {lg_orac_d} vs {lg_froz_d}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "$(git rev-parse --show-toplevel)"
uv run pytest tests/test_cache_codecs.py -k "structured or topk or blockdiag or frozen_vs_oracle" -q
```
Expected: FAIL — new arms not registered.

- [ ] **Step 3: Implement the four modes**

In `src/bmx/cache/codecs.py`:

(a) Add the four arms to `CACHE_ARMS` and `S_DIVISIBILITY_ARMS`:

```python
CACHE_ARMS = (
    "rtn_token", "rtn_channel", "rotate_rtn_token", "turboquant_mse",
    "turboquant_prod", "lowrank_rtn_channel", "lowrank_waterfill_channel",
    "lowrank_eigwaterfill_channel", "lowrank_randwaterfill_channel",
    "lowrank_topkwaterfill_channel", "lowrank_blockdiagwaterfill_channel",
    "lowrank_frozenwaterfill_channel", "lowrank_oraclewaterfill_channel",
)

S_DIVISIBILITY_ARMS = frozenset({
    "rtn_channel", "lowrank_rtn_channel", "lowrank_waterfill_channel",
    "lowrank_eigwaterfill_channel", "lowrank_randwaterfill_channel",
    "lowrank_topkwaterfill_channel", "lowrank_blockdiagwaterfill_channel",
    "lowrank_frozenwaterfill_channel", "lowrank_oraclewaterfill_channel",
})
```

(b) Extend `_lowrank_rotwaterfill_channel`'s signature with `topk_k: int = 0`, `prefill_fit_len: int = 0`, `h_kv: int = 0`, and accept the new `rotation` values. Add a helper that water-fills a rotated residual given a rotation Q (factor the existing rotate→allocate→quantize→unrotate body into a small inner so all modes reuse it), then build Q per mode. Replace the rotation-construction block with:

```python
    S, C = M.shape
    assert rotation in ("klt", "random", "topk", "blockdiag", "frozen", "oracle"), (
        f"unknown rotation {rotation!r}"
    )
    # ... (existing rank asserts, svd, L, R = M - L unchanged) ...

    use_rotation = len(set(tiers)) > 1

    def _waterfill_in_basis(R_in, Q):
        # Q None => identity (no rotation). Returns R_hat in the ORIGINAL basis.
        R_rot = R_in if Q is None else (R_in @ Q)
        bits_pc = allocate_channel_bits(R_rot, budget_bits, tiers=tiers, axis=0)
        R_rot_hat = torch.zeros_like(R_rot)
        for b in sorted(set(int(x) for x in bits_pc.tolist())):
            if b == 0:
                continue
            cols = (bits_pc == b).nonzero(as_tuple=True)[0]
            if cols.numel() == 0:
                continue
            R_rot_hat[:, cols] = rtn_quantize(R_rot[:, cols].mT, b, group).mT
        R_hat_local = R_rot_hat if Q is None else (R_rot_hat @ Q.mT)
        return R_hat_local, float(bits_pc.float().mean().item())

    if not use_rotation:
        R_hat, mean_payload = _waterfill_in_basis(R, None)
        rot_bits = 0.0
    elif rotation in ("klt", "random"):
        if rotation == "klt":
            _, ev = torch.linalg.eigh(R.mT @ R)
            Q = ev.flip(dims=(1,))
        else:
            Q = random_orthogonal(C, seed, dtype=R.dtype, device=R.device)
        R_hat, mean_payload = _waterfill_in_basis(R, Q)
        rot_bits = (16.0 * C / S) if (rotation == "klt" and charge_rotation) else 0.0
    elif rotation == "oracle":
        # full KLT refit on ALL tokens (control; never charged)
        _, ev = torch.linalg.eigh(R.mT @ R)
        Q = ev.flip(dims=(1,))
        R_hat, mean_payload = _waterfill_in_basis(R, Q)
        rot_bits = 0.0
    elif rotation == "frozen":
        # full KLT fit on the first prefill_fit_len tokens, applied to all
        assert prefill_fit_len > 0, "frozen requires prefill_fit_len > 0"
        P = min(prefill_fit_len, S)
        _, ev = torch.linalg.eigh(R[:P].mT @ R[:P])
        Q = ev.flip(dims=(1,))
        R_hat, mean_payload = _waterfill_in_basis(R, Q)
        rot_bits = (16.0 * C / S) if charge_rotation else 0.0
    elif rotation == "topk":
        # Rotate ONLY the stored top-k eigen-directions; the complement stays in the
        # original basis. Honest: only Qk (C×k) is stored => 16*k/S. (Do NOT rotate the
        # full basis and charge top-k — the complement eigvecs are data-dependent and
        # not recomputable by a decoder; that would overstate compression.)
        assert topk_k > 0, "topk requires topk_k > 0"
        kk = min(topk_k, C)
        _, ev = torch.linalg.eigh(R.mT @ R)
        Qk = ev.flip(dims=(1,))[:, :kk]  # (C, k) top-k by eigenvalue
        # Rotate top-k subspace; water-fill those k columns.
        Rk = R @ Qk  # (S, k)
        Rk_hat, p_k = _waterfill_in_basis(Rk, None)  # already in the rotated subspace
        topk_back = Rk_hat @ Qk.mT  # (S, C) contribution of the top-k subspace
        # Complement: residual not explained by the top-k subspace, water-filled in place.
        R_comp = R - (R @ Qk) @ Qk.mT  # project OUT the top-k subspace
        Rcomp_hat, p_c = _waterfill_in_basis(R_comp, None)
        R_hat = topk_back + Rcomp_hat
        # blended payload over all C channels (k rotated + complement)
        mean_payload = (p_k * kk + p_c * (C - kk)) / C
        rot_bits = (16.0 * kk / S) if charge_rotation else 0.0
    else:  # blockdiag
        assert h_kv > 0, "blockdiag requires h_kv > 0"
        assert C % h_kv == 0, f"C={C} not divisible by h_kv={h_kv}"
        d = C // h_kv
        R_hat = torch.zeros_like(R)
        payloads = []
        for hh in range(h_kv):
            sl = slice(hh * d, (hh + 1) * d)
            Rh = R[:, sl]
            _, evh = torch.linalg.eigh(Rh.mT @ Rh)
            Qh = evh.flip(dims=(1,))
            Rh_hat, ph = _waterfill_in_basis(Rh, Qh)
            R_hat[:, sl] = Rh_hat
            payloads.append(ph)
        mean_payload = float(sum(payloads) / len(payloads))
        rot_bits = (16.0 * d / S) if charge_rotation else 0.0

    M_hat = L + R_hat
    scale_term = 16.0 / group
    factor_term = 16.0 * rank * (S + C) / (S * C)
    tier_term = math.ceil(math.log2(len(tiers))) / S
    bpe = mean_payload + scale_term + factor_term + tier_term + rot_bits
    return M_hat, bpe
```

(NOTE: this REPLACES the existing rotation/quantize/unrotate/bpe block for the
function. The existing klt/random behavior is preserved by the `elif rotation in
("klt","random")` branch. Keep the existing docstring, extend it to name the new
modes.

IMPORTANT — topk honesty: the code above rotates the FULL C×C basis but charges only
`16*topk_k/S`. This is honest ONLY IF the complement eigenvectors (`Q[:, topk_k:]`)
are genuinely recomputable at read time without storing them. They are NOT freely
recomputable here — the KLT depends on the data `RᵀR`, which the decoder doesn't have.
So the full-basis-rotate version OVERSTATES what top-k storage buys. The honest topk
must rotate ONLY the stored top-k directions and leave the complement in the original
basis. Implement topk as: `Qk = ev.flip(1)[:, :topk_k]` (C×k); rotate just those
`Rk = R @ Qk` (S×k); water-fill the k rotated columns AND the original-basis residual
`R - (Rk_quant_back)`; reconstruct `R_hat = Rk_hat @ Qk.mT + complement_hat`. Only `Qk`
(C×k) is stored ⇒ `16*topk_k/S` is the true cost. The `test_topk_reduces_to_full_klt_at_k_equals_C`
test pins that at k=C this equals the full KLT. Use this partial-rotation form, not the
full-basis-rotate shortcut shown in the skeleton above.)

(c) Thread the new kwargs through `quantize_cache` (add `topk_k: int = 0`,
`prefill_fit_len: int = 0`, `h_kv: int = 0` to its signature after `charge_rotation`)
and add dispatch branches mapping each new arm to the core with the right `rotation`:

```python
    elif arm == "lowrank_topkwaterfill_channel":
        return _lowrank_rotwaterfill_channel(
            M, float(bits), group, rank, tiers=tiers, rotation="topk",
            topk_k=topk_k, charge_rotation=charge_rotation, svd_factors=svd_factors,
        )
    elif arm == "lowrank_blockdiagwaterfill_channel":
        return _lowrank_rotwaterfill_channel(
            M, float(bits), group, rank, tiers=tiers, rotation="blockdiag",
            h_kv=h_kv, charge_rotation=charge_rotation, svd_factors=svd_factors,
        )
    elif arm == "lowrank_frozenwaterfill_channel":
        return _lowrank_rotwaterfill_channel(
            M, float(bits), group, rank, tiers=tiers, rotation="frozen",
            prefill_fit_len=prefill_fit_len, charge_rotation=charge_rotation,
            svd_factors=svd_factors,
        )
    elif arm == "lowrank_oraclewaterfill_channel":
        return _lowrank_rotwaterfill_channel(
            M, float(bits), group, rank, tiers=tiers, rotation="oracle",
            svd_factors=svd_factors,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cache_codecs.py -k "structured or topk or blockdiag or frozen_vs_oracle" -q
```
Expected: PASS (6 tests).

- [ ] **Step 5: Format, lint, full test**

```bash
uv run ruff format . && uv run ruff check . && uv run pytest -q
```
Expected: all clean (prior 209 + 6 new = 215 passed, 1 xfailed). If the existing klt/random tests from the prior task break, the refactor changed behavior — fix the core, not the tests.

- [ ] **Step 6: Commit (pre-authorized)**

```bash
git add src/bmx/cache/codecs.py tests/test_cache_codecs.py
git commit -m "feat(cache): topk, block-diagonal, frozen, and oracle rotation modes for water-filling"
```
Stage ONLY those two files.

---

## Task 2: Experiment — structured arms, frozen/oracle ratio, eigengap diagnostic

**Files:**
- Modify: `experiments/k2_waterfill.py`
- Modify: `tests/test_k2_waterfill.py`

**Interfaces:**
- Consumes (Task 1): the four new arms via `quantize_cache` with `topk_k`, `prefill_fit_len`, `h_kv`, `charge_rotation`.
- Consumes (existing): `main(cfg)`, `score()`, `_resid_stable_rank`, `_query_eigen_alignment`, `Config`.
- Produces: new `Config` fields `topk_ks: tuple[int,...] = (32, 64, 128)`, `prefill_fit_len: int = 512`; a `_residual_eigengap(R)` helper; new parquet columns for the structured arms incl. `bpe_honest` (honest rotation cost), `frozen_oracle_ratio`, `resid_eigengap`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_k2_waterfill.py`:

```python
def test_experiment_has_structured_arms(tmp_path):
    from safetensors.torch import save_file

    h_kv, S, d = 2, 64, 8  # C=16
    g = torch.Generator().manual_seed(7)
    tensors = {}
    for i in range(2):
        for nm in ("k_pre", "k", "v", "q"):
            tensors[f"layer{i}.{nm}"] = torch.randn(h_kv, S, d, generator=g).half()
    cache_path = tmp_path / "synthetic.safetensors"
    save_file(tensors, str(cache_path))

    cfg = k2_waterfill.Config(
        cache_path=str(cache_path), model_label="synthetic", model_name="",
        budget_bits=3.0, group=16, rank=4, topk_ks=(8,), prefill_fit_len=32,
        out_root=str(tmp_path / "results"),
    )
    df = k2_waterfill.main(cfg)
    arms = set(df["arm"].unique())
    # block-diag, frozen, oracle present; topk present (named with its k)
    assert "lowrank_blockdiagwaterfill_channel" in arms
    assert "lowrank_frozenwaterfill_channel" in arms
    assert "lowrank_oraclewaterfill_channel" in arms
    assert any(a.startswith("lowrank_topkwaterfill_channel") for a in arms)
    # diagnostics
    assert "resid_eigengap" in df.columns
    assert "frozen_oracle_ratio" in df.columns
    fr = df[df.arm == "lowrank_frozenwaterfill_channel"]
    assert fr["frozen_oracle_ratio"].notna().all()
    # honest bpe present for the structured deployable arms
    assert "bpe_honest" in df.columns
    bd = df[df.arm == "lowrank_blockdiagwaterfill_channel"]
    assert bd["bpe_honest"].notna().all()
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_k2_waterfill.py::test_experiment_has_structured_arms -q
```
Expected: FAIL — structured arms / diagnostic columns absent.

- [ ] **Step 3: Implement the experiment extension**

In `experiments/k2_waterfill.py`:

(a) Add Config fields (after the existing ones):

```python
    topk_ks: tuple[int, ...] = (32, 64, 128)
    prefill_fit_len: int = 512
```

(b) Add the eigengap helper after `_resid_stable_rank`:

```python
def _residual_eigengap(R: torch.Tensor) -> float:
    """Max adjacent eigenvalue ratio in the top of the residual channel spectrum.

    A value near 1.0 across the spectrum => no eigengap => frozen rotation likely
    drifts (Davis-Kahan). Returns the largest λ_k/λ_{k+1} among the top-64 to expose
    any elbow; ~1.0 means smooth decay.
    """
    ev = torch.linalg.eigvalsh(R.mT @ R).flip(0).clamp_min(1e-30)
    top = ev[: min(64, len(ev) - 1)]
    ratios = top[:-1] / top[1:]
    return float(ratios.max().item())
```

(c) In the per-layer loop, after the existing arms and before the row-append, add the
structured arms. The honest bpe is computed by re-quantizing with
`charge_rotation=True` (the deployable verdict). The frozen/oracle ratio is
`logit(frozen)/logit(oracle)`. Insert:

```python
        eigengap = _residual_eigengap(R_resid)

        # block-diagonal per-head KLT (honest cost folded in directly via charge_rotation)
        bd, bpe_bd = quantize_cache(
            "lowrank_blockdiagwaterfill_channel", M, bits=cfg.budget_bits,
            group=cfg.group, rank=cfg.rank, tiers=cfg.tiers, h_kv=h_kv,
            charge_rotation=True, svd_factors=factors,
        )
        # frozen-prefill full KLT (honest) + oracle control (uncharged)
        fz, bpe_fz = quantize_cache(
            "lowrank_frozenwaterfill_channel", M, bits=cfg.budget_bits,
            group=cfg.group, rank=cfg.rank, tiers=cfg.tiers,
            prefill_fit_len=cfg.prefill_fit_len, charge_rotation=True, svd_factors=factors,
        )
        orc, bpe_orc = quantize_cache(
            "lowrank_oraclewaterfill_channel", M, bits=cfg.budget_bits,
            group=cfg.group, rank=cfg.rank, tiers=cfg.tiers, svd_factors=factors,
        )
        _, lg_fz = score(fz)
        _, lg_orc = score(orc)
        frozen_oracle_ratio = lg_fz / lg_orc if lg_orc > 0 else float("nan")

        structured = [
            ("lowrank_blockdiagwaterfill_channel", bd, bpe_bd, bpe_bd, float("nan")),
            ("lowrank_frozenwaterfill_channel", fz, bpe_fz, bpe_fz, frozen_oracle_ratio),
            ("lowrank_oraclewaterfill_channel", orc, bpe_orc, float("nan"), float("nan")),
        ]
        # topk k-sweep, honest cost per k
        C_full = M.shape[1]
        for kk in cfg.topk_ks:
            if kk > C_full:  # guard k <= C
                continue
            tk, bpe_tk = quantize_cache(
                "lowrank_topkwaterfill_channel", M, bits=cfg.budget_bits,
                group=cfg.group, rank=cfg.rank, tiers=cfg.tiers, topk_k=kk,
                charge_rotation=True, svd_factors=factors,
            )
            structured.append(
                (f"lowrank_topkwaterfill_channel_k{kk}", tk, bpe_tk, bpe_tk, float("nan"))
            )
```

(d) Append the structured arms into the row loop. After the existing `arm_rows`
loop, add:

```python
        for arm, M_hat, bpe, bpe_h, fo_ratio in structured:
            rf, lg = score(M_hat)
            rows.append(dict(
                model=cfg.model_label or "unknown", layer=layer_i, kind="k_pre",
                arm=arm, rank=cfg.rank, bpe=bpe, bpe_honest=bpe_h,
                rel_fro=rf, logit_rope=lg, resid_stable_rank=sr,
                query_eigen_alignment=float("nan"),
                resid_eigengap=eigengap, frozen_oracle_ratio=fo_ratio,
            ))
            print(
                f"  L{layer_i:2d} {arm:34s} bpe={bpe:.3f} bpe_h={bpe_h if bpe_h==bpe_h else float('nan'):.3f} "
                f"logit={lg:.4f} rel_fro={rf:.4f} gap={eigengap:.3f} fo={fo_ratio if fo_ratio==fo_ratio else float('nan'):.3f}",
                flush=True,
            )
```

Also add `resid_eigengap` and `frozen_oracle_ratio` keys (as `float("nan")`) to the
EXISTING `arm_rows` append dict so all rows share columns.

(e) Extend the SUMMARY block to print the structured arms with their HONEST bpe and
the eigengap/frozen-oracle context (reuse the existing win-rate-vs-uniform logic; the
verdict for structured arms is on `bpe_honest`).

- [ ] **Step 4: Run the smoke test to verify it passes**

```bash
uv run pytest tests/test_k2_waterfill.py -q
```
Expected: PASS.

- [ ] **Step 5: Format, lint, full test**

```bash
uv run ruff format . && uv run ruff check . && uv run pytest -q
```
Expected: all clean.

- [ ] **Step 6: Verify it runs end-to-end on GPT-2 before commit**

```bash
uv run python experiments/k2_waterfill.py \
    --cache-path /d/Projects/bmx/results/cache/gpt2_1024.safetensors \
    --model-label gpt2 --budget-bits 3.0 --rank 16 2>&1 | tail -30
```
Confirm the structured arms print with honest bpe, eigengap, and frozen/oracle ratio.
Confirm no `results/` files are staged (`git status --short`).

- [ ] **Step 7: Commit (pre-authorized)**

```bash
git add experiments/k2_waterfill.py tests/test_k2_waterfill.py
git commit -m "feat(exp): structured-rotation arms with honest bpe, frozen/oracle ratio, and eigengap diagnostic"
```
Stage ONLY those two files.

---

## Task 3: Run the ablation and write the verdict

**Files:**
- Create: `docs/2026-06-21-k2-structured-rotation-results.md`

Execution + science judgment; no test cycle. Depends on Tasks 1-2 committed.

- [ ] **Step 1: Run both caches**

```bash
cd "$(git rev-parse --show-toplevel)"
uv run python experiments/k2_waterfill.py \
    --cache-path /d/Projects/bmx/results/cache/gpt2_1024.safetensors \
    --model-label gpt2 --budget-bits 3.0 --rank 16 2>&1 | tail -40
uv run python experiments/k2_waterfill.py \
    --cache-path /d/Projects/bmx/results/cache/llama-3.1-8b_2048.safetensors \
    --model-label llama-3.1-8b --model-name meta-llama/Llama-3.1-8B \
    --budget-bits 3.0 --rank 16 2>&1 | tail -55
```

- [ ] **Step 2: Pull the honest-bpe decision matrix**

```bash
uv run python -c "
import pandas as pd, glob, os, numpy as np
for d in sorted(glob.glob('results/k2_waterfill/*/')):
    pq = glob.glob(os.path.join(d,'*.parquet'))
    if not pq: continue
    df = pd.read_parquet(pq[0])
    if 'resid_eigengap' not in df.columns: continue  # only structured-arm runs
    lab = df['model'].iloc[0]
    print(f'=== {lab} ({os.path.basename(d.rstrip(chr(47)))}) ===')
    uni = df[df.arm=='lowrank_rtn_channel'].set_index('layer')['logit_rope']
    for arm in sorted(df.arm.unique()):
        sub = df[df.arm==arm]
        lg = sub['logit_rope']; sem = lg.std()/np.sqrt(len(lg))
        piv = sub.set_index('layer')['logit_rope']
        wins = int((piv < uni.reindex(piv.index)).sum())
        bh = sub['bpe_honest'].mean()
        fo = sub['frozen_oracle_ratio'].mean()
        print(f'  {arm:36s} logit={lg.mean():.4f}±{sem:.4f} bpe={sub.bpe.mean():.3f} bpe_honest={bh:.3f} wins={wins}/{sub.layer.nunique()} fo={fo:.3f}')
    print('  mean resid_eigengap:', df['resid_eigengap'].mean())
    print()
"
```

- [ ] **Step 3: Write the verdict doc**

Create `docs/2026-06-21-k2-structured-rotation-results.md` in the terse numbers-first
style of `docs/2026-06-21-k2-eigwaterfill-results.md`. Record faithfully:
- The honest-bpe decision matrix (both models): arm × {logit±sem, bpe, bpe_honest, wins-vs-uniform, frozen/oracle ratio}.
- For each deployable arm (topk per k, blockdiag, frozen) the verdict on HONEST bpe:
  does it beat uniform at matched honest bpe? Use the pass bar: `logit < uniform` by
  >2 sem on a clear majority of layers, at `bpe_honest ≈ bpe(uniform)`.
- The frozen verdict via the frozen/oracle ratio: ratio ≈ 1 ⇒ generalizes; ratio ≫ 1
  (frozen logit much worse) ⇒ drifts. Tie to the measured `resid_eigengap` (≈1.0 ⇒
  no gap ⇒ Davis-Kahan predicts drift). Confirm or refute the pre-registered
  prediction.
- The topk cost/capture frontier: logit vs k at honest bpe — how much of the full-KLT
  2.2× survives per bit.
- Overall gate call: did ANY structured rotation make the eigenbasis win deployable
  (beat uniform at honest bpe)? If block-diag wins → the deployable path is found,
  recommend streaming-cache promotion as the next gate. If all lose → the eigenbasis
  win is confirmed real but fundamentally too expensive to encode at any tested
  structure; the k2b uniform recipe stands; note Hadamard-class fixed rotations as the
  last deferred avenue.
- Carry caveats: honest-bpe is the verdict (idealized is ceiling only); GPT-2 stored-
  basis vs Llama post-RoPE; block-diag needs real GQA (Llama authoritative).

- [ ] **Step 4: Commit (pre-authorized)**

```bash
git add docs/2026-06-21-k2-structured-rotation-results.md
git commit -m "docs: structured-rotation eigenbasis water-filling ablation results and verdict"
```
Stage ONLY the doc. Confirm `results/` run dirs are NOT staged.

---

## Notes for the executor

- **The verdict is on HONEST bpe**, not idealized. A structured arm only "wins" if it beats uniform with its rotation cost included. Idealized bpe is the mechanism ceiling.
- **The frozen/oracle ratio is the load-bearing control for the frozen arm.** A frozen loss is only interpretable against the oracle: ratio≈1 + loss ⇒ structure too expensive; ratio≫1 ⇒ drift. The eigengap (≈1.0, measured) predicts drift — report whether the prediction held.
- **Pre-registered expectation (from the spec's eigenspectrum analysis):** block-diag most likely to win; frozen likely drifts (flat spectrum); topk likely loses at low k. Report honestly which held — a predicted negative is a clean result.
- **fp16 factor roundtrip is UNCONDITIONAL** (the dtype-guard bug bit before). The core mirrors `_lowrank_waterfill_channel`.
- **Eigvec sign ambiguity**: equivalence tests use logit distortion, never raw tensors.
- The `_waterfill_in_basis` inner factored in Task 1 is reused by every mode — if a mode misbehaves, check Q construction, not the shared water-fill body.
- Matched bpe for structured arms is on the HONEST number; the smoke test asserts the columns exist, the real matched-bpe check is in the verdict (topk@low-k and blockdiag should land near uniform's bpe; frozen/full near +8 unless context amortizes).
