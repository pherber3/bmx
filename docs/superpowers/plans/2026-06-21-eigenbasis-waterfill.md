# Eigenbasis (KLT) Water-filling Revival Test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether water-filling per-channel bit allocation in the residual's eigenbasis (KLT) — or any rotation — beats uniform allocation on logit distortion, with a random-rotation control and a dual MSE/logit readout to make the verdict decisive.

**Architecture:** One rotated-waterfill core in `codecs.py` exposing two arms (`lowrank_eigwaterfill_channel` = KLT, `lowrank_randwaterfill_channel` = random control), built directly on the existing `_lowrank_waterfill_channel`. The experiment `k2_waterfill.py` gains both arms, records BOTH `logit_rope` and `rel_fro` for every arm, and logs a `query_eigen_alignment` diagnostic. Offline, on already-collected caches.

**Tech Stack:** Python, PyTorch (CPU), tyro, pandas/parquet, pytest, uv, ruff.

**Spec:** `docs/superpowers/specs/2026-06-21-eigenbasis-waterfill-design.md`

## Global Constraints

- **Never commit without approval — EXCEPT** per-task commits are PRE-AUTHORIZED here (user: "commit as you go"). One clean single-sentence message, conventional prefix, NO AI attribution / Co-Authored-By, ever.
- Before any commit: `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` — all clean.
- Use the Bash tool (git bash), NOT PowerShell. Shell cwd may reset between calls — run `cd "$(git rev-parse --show-toplevel)"` at the start of each Bash call. Run everything via `uv`.
- dtype: **fp64 in tests, fp32 in experiments/codecs.** Codecs fp16-roundtrip low-rank factors UNCONDITIONALLY (matches `_lowrank_rtn_channel`; do NOT guard on dtype — that bug already bit once).
- **Comparisons align on realized bpe with ALL metadata counted.** The rotated arms' idealized bpe charges NO rotation; the honest KLT pass adds `16·C/S`. The random arm's rotation is seeded → 0 stored bits → honest == idealized.
- Metrics: record BOTH `logit_rope` (headline, vs real queries, GQA-aware, RoPE-at-read) AND `rel_fro` (MSE). Never Frobenius as the headline verdict.
- Orthogonal sampling goes through `bmx.quant.hadamard.random_orthogonal` (audited; never hand-roll QR — sign ambiguity).
- Eigenvectors via `torch.linalg.eigh` (symmetric PSD). Eigvec SIGN is ambiguous — equivalence tests must compare logit distortion, never raw tensors.
- Tiny offline synthetics / GPT-2 fixture from existing tests; never download.

---

## File Structure

- **Modify** `src/bmx/cache/codecs.py` — add `_lowrank_rotwaterfill_channel(M, budget_bits, group, rank, tiers, rotation, seed, charge_rotation, svd_factors)` core; register `"lowrank_eigwaterfill_channel"` and `"lowrank_randwaterfill_channel"` in `CACHE_ARMS` + `S_DIVISIBILITY_ARMS`; thread `rotation`/`charge_rotation` through `quantize_cache`; add dispatch branches.
- **Modify** `tests/test_cache_codecs.py` — add rotated-waterfill tests.
- **Modify** `experiments/k2_waterfill.py` — add both arms to the per-layer loop, dual-metric rows, `query_eigen_alignment` + honest-pass logic, richer summary.
- **Modify** `tests/test_k2_waterfill.py` — extend smoke test to assert the new arms, both metrics, and the alignment column.
- **Create** `docs/2026-06-21-k2-eigwaterfill-results.md` — the verdict (Task 3).

---

## Task 1: Rotated-waterfill core + two arms (KLT + random control)

**Files:**
- Modify: `src/bmx/cache/codecs.py`
- Test: `tests/test_cache_codecs.py`

**Interfaces:**
- Consumes (existing): `allocate_channel_bits(R, budget_bits, tiers=(0,2,3,4), *, axis=0) -> (C,) int64`; `truncated_svd(M, rank) -> (Us, V)`; `rtn_quantize(W, bits, group_size) -> Tensor`; `random_orthogonal(d, seed, dtype, device) -> (d,d)` from `bmx.quant.hadamard`.
- Produces:
  ```python
  def _lowrank_rotwaterfill_channel(
      M: torch.Tensor,            # (S, C) fp32
      budget_bits: float,
      group: int,
      rank: int,
      tiers: tuple[int, ...] = (0, 2, 3, 4),
      rotation: str = "klt",      # "klt" | "random"
      seed: int = 0,              # used only when rotation="random"
      charge_rotation: bool = False,  # KLT only: add 16*C/S to bpe
      svd_factors: tuple | None = None,
  ) -> tuple[torch.Tensor, float]  # (M_hat (S,C), bpe)
  ```
  Reachable as `quantize_cache("lowrank_eigwaterfill_channel", M, bits=budget, group=, rank=, tiers=, charge_rotation=, svd_factors=)` and `quantize_cache("lowrank_randwaterfill_channel", M, bits=budget, seed=, group=, rank=, tiers=, svd_factors=)`. Both arms are in `CACHE_ARMS` and `S_DIVISIBILITY_ARMS`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cache_codecs.py` (the helpers `_seeded_matrix`, `_channel_matrix`, `allocate_channel_bits`, `truncated_svd`, `quantize_cache`, `GROUP` already exist in this file from prior tasks):

```python
from bmx.cache.metrics import logit_distortion as _logit_distortion


def _qkv_for(M, h_kv=2, seed=123):
    """A fake query set (h_kv, T, d) matching M's (S, C=h_kv*d) layout for logit scoring."""
    S, C = M.shape
    d = C // h_kv
    g = torch.Generator().manual_seed(seed)
    return torch.randn(h_kv, 8, d, generator=g, dtype=M.dtype)


def test_rotwaterfill_arms_registered():
    from bmx.cache.codecs import CACHE_ARMS, S_DIVISIBILITY_ARMS

    for arm in ("lowrank_eigwaterfill_channel", "lowrank_randwaterfill_channel"):
        assert arm in CACHE_ARMS
        assert arm in S_DIVISIBILITY_ARMS


def test_rotation_is_inner_product_neutral():
    # With a single high-bit uniform tier (near-lossless RTN), rotate+quantize+unrotate
    # must match the unrotated near-lossless arm on LOGIT distortion to tight tol —
    # for BOTH klt and random. Proves Q is orthogonal and rotate/unrotate is exact.
    from bmx.cache.collect import from_matrix

    M = _seeded_matrix(s=64, c=64, seed=4).double()
    h_kv = 2
    q = _qkv_for(M, h_kv=h_kv)
    factors = truncated_svd(M, 4)
    # near-lossless: one tier at 8 bits
    base, _ = quantize_cache(
        "lowrank_waterfill_channel", M, bits=8, group=GROUP, rank=4,
        tiers=(8,), svd_factors=factors,
    )
    lg_base = _logit_distortion(from_matrix(M, h_kv).double(), from_matrix(base, h_kv).double(), q)
    for rotation in ("klt", "random"):
        rot, _ = quantize_cache(
            "lowrank_eigwaterfill_channel" if rotation == "klt" else "lowrank_randwaterfill_channel",
            M, bits=8, group=GROUP, rank=4, tiers=(8,), seed=1, svd_factors=factors,
        )
        lg_rot = _logit_distortion(from_matrix(M, h_kv).double(), from_matrix(rot, h_kv).double(), q)
        assert abs(lg_rot - lg_base) < 1e-6, f"{rotation}: rotation not inner-product-neutral"


def test_klt_reduces_to_raw_waterfill_when_diagonal():
    # Diagonal-covariance residual -> KLT Q is identity (up to sign/perm). KLT arm then
    # matches raw waterfill on LOGIT distortion (not raw tensors — eigvec sign ambiguity).
    from bmx.cache.collect import from_matrix

    # Build M whose residual after rank-r low-rank is independent per-channel:
    # use a matrix with no low-rank structure so L is tiny and R ~= M with diagonal cov.
    stds = [0.3, 1.0, 3.0, 9.0] * 16  # C = 64, varied per-channel, uncorrelated
    R = _channel_matrix(stds, s=64, seed=8)  # (64, 64) fp64, diagonal cov by construction
    h_kv = 2
    q = _qkv_for(R, h_kv=h_kv)
    factors = truncated_svd(R, 4)
    raw, _ = quantize_cache(
        "lowrank_waterfill_channel", R, bits=3, group=GROUP, rank=4,
        tiers=(0, 2, 3, 4), svd_factors=factors,
    )
    klt, _ = quantize_cache(
        "lowrank_eigwaterfill_channel", R, bits=3, group=GROUP, rank=4,
        tiers=(0, 2, 3, 4), svd_factors=factors,
    )
    lg_raw = _logit_distortion(from_matrix(R, h_kv).double(), from_matrix(raw, h_kv).double(), q)
    lg_klt = _logit_distortion(from_matrix(R, h_kv).double(), from_matrix(klt, h_kv).double(), q)
    # diagonal cov => Q ~ I (up to sign) => same allocation, same logit distortion
    assert abs(lg_raw - lg_klt) < 0.05, f"diagonal KLT diverged from raw: {lg_raw} vs {lg_klt}"


def test_random_arm_is_free_and_reproducible():
    M = _seeded_matrix(s=64, c=64, seed=6).double()
    factors = truncated_svd(M, 4)
    a, bpe_a = quantize_cache(
        "lowrank_randwaterfill_channel", M, bits=3, seed=7, group=GROUP, rank=4,
        tiers=(0, 2, 3, 4), svd_factors=factors,
    )
    b, bpe_b = quantize_cache(
        "lowrank_randwaterfill_channel", M, bits=3, seed=7, group=GROUP, rank=4,
        tiers=(0, 2, 3, 4), svd_factors=factors,
    )
    assert torch.allclose(a, b), "random arm not reproducible at fixed seed"
    assert abs(bpe_a - bpe_b) < 1e-12
    # honest == idealized: random rotation costs 0 stored bits. Compare to raw waterfill bpe
    # (same payload+scale+factor+tier terms, no rotation term either way).
    _, bpe_raw = quantize_cache(
        "lowrank_waterfill_channel", M, bits=3, group=GROUP, rank=4,
        tiers=(0, 2, 3, 4), svd_factors=factors,
    )
    assert abs(bpe_a - bpe_raw) < 1e-9, "random arm bpe should match raw waterfill (no rotation charge)"


def test_klt_honest_rotation_charge():
    S_, C_, group_, rank_ = 64, 32, 16, 2
    M = _seeded_matrix(s=S_, c=C_, seed=5).double()
    _, bpe_ideal = quantize_cache(
        "lowrank_eigwaterfill_channel", M, bits=3, group=group_, rank=rank_,
        tiers=(0, 2, 3, 4), charge_rotation=False,
    )
    _, bpe_honest = quantize_cache(
        "lowrank_eigwaterfill_channel", M, bits=3, group=group_, rank=rank_,
        tiers=(0, 2, 3, 4), charge_rotation=True,
    )
    expected = 16.0 * C_ / S_
    assert abs((bpe_honest - bpe_ideal) - expected) < 1e-9, "rotation charge != 16*C/S"


def test_klt_concentrates_random_spreads_variance():
    # KLT increases per-column variance CV (concentration); random decreases it (spreading).
    from bmx.cache.codecs import _round_to_tiers  # noqa: F401  (sanity import path exists)

    stds = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0] * 8  # C=64 anisotropic
    R = _channel_matrix(stds, s=256, seed=2)

    def cv(x):
        v = x.var(dim=0, unbiased=False)
        return (v.std() / v.mean().clamp_min(1e-30)).item()

    from bmx.quant.hadamard import random_orthogonal

    cv_raw = cv(R)
    eigvals, eigvecs = torch.linalg.eigh(R.mT @ R)
    R_klt = R @ eigvecs
    cv_klt = cv(R_klt)
    Qr = random_orthogonal(R.shape[1], seed=3, dtype=R.dtype)
    R_rand = R @ Qr.mT
    cv_rand = cv(R_rand)
    assert cv_klt > cv_raw, f"KLT did not concentrate variance: {cv_klt} <= {cv_raw}"
    assert cv_rand < cv_raw, f"random did not spread variance: {cv_rand} >= {cv_raw}"


def test_rotwaterfill_s_divisibility_assert():
    M = _seeded_matrix(s=63, c=64, seed=9).double()  # 63 % 16 != 0
    with pytest.raises(AssertionError):
        quantize_cache(
            "lowrank_eigwaterfill_channel", M, bits=3, group=16, rank=2, tiers=(0, 2, 3, 4)
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "$(git rev-parse --show-toplevel)"
uv run pytest tests/test_cache_codecs.py -k "rotwaterfill or rotation_is or klt or random_arm" -q
```
Expected: FAIL — arms not in `CACHE_ARMS` (assertion in `quantize_cache`).

- [ ] **Step 3: Implement the core + register both arms**

In `src/bmx/cache/codecs.py`:

(a) Add both arms to `CACHE_ARMS` and `S_DIVISIBILITY_ARMS`:

```python
CACHE_ARMS = (
    "rtn_token",
    "rtn_channel",
    "rotate_rtn_token",
    "turboquant_mse",
    "turboquant_prod",
    "lowrank_rtn_channel",
    "lowrank_waterfill_channel",
    "lowrank_eigwaterfill_channel",
    "lowrank_randwaterfill_channel",
)

S_DIVISIBILITY_ARMS = frozenset(
    {
        "rtn_channel",
        "lowrank_rtn_channel",
        "lowrank_waterfill_channel",
        "lowrank_eigwaterfill_channel",
        "lowrank_randwaterfill_channel",
    }
)
```

(b) Add the core after `_lowrank_waterfill_channel` (this mirrors it, inserting rotate→allocate→quantize→unrotate; reuses `random_orthogonal`):

```python
def _lowrank_rotwaterfill_channel(
    M: torch.Tensor,
    budget_bits: float,
    group: int,
    rank: int,
    tiers: tuple[int, ...] = (0, 2, 3, 4),
    rotation: str = "klt",
    seed: int = 0,
    charge_rotation: bool = False,
    svd_factors: tuple | None = None,
) -> tuple[torch.Tensor, float]:
    """Low-rank + rotated per-channel residual at water-filled mixed bit-widths.

    Same as _lowrank_waterfill_channel, but the residual R = M - L is first rotated
    by an orthogonal Q before per-channel water-filling, then unrotated. Q is either
    the KLT (eigenvectors of R^T R, variance-concentrating) or a seeded random
    orthogonal (variance-spreading control). Q orthogonal => inner products preserved,
    so the rotation is logit-neutral; only the post-rotation quantization distorts.

    bpe: idealized (rotation free) by default; charge_rotation=True (KLT only) adds
    16*C/S for the stored C×C fp16 rotation matrix. The random rotation is seeded
    (0 stored bits), so charge_rotation is a no-op for it.
    """
    from bmx.quant.hadamard import random_orthogonal

    S, C = M.shape
    assert rotation in ("klt", "random"), f"unknown rotation {rotation!r}"
    assert rank > 0, f"rotwaterfill requires rank > 0, got {rank}"
    assert rank <= min(S, C), f"rank {rank} > min(S,C)={min(S, C)}"
    assert S % group == 0, f"S={S} not divisible by group={group}"

    if svd_factors is not None:
        Us, V = svd_factors
    else:
        Us, V = truncated_svd(M, rank)
    Us_stored = Us.half().float()
    V_stored = V.half().float()
    L = Us_stored @ V_stored.mT
    R = M - L  # (S, C)

    # Build the orthogonal rotation Q (C, C).
    if rotation == "klt":
        # eigenvectors of the residual channel covariance, descending eigenvalue.
        _, eigvecs = torch.linalg.eigh(R.mT @ R)  # ascending
        Q = eigvecs.flip(dims=(1,))  # descending eigenvalue order
    else:  # random
        Q = random_orthogonal(C, seed, dtype=R.dtype, device=R.device)

    R_rot = R @ Q  # (S, C) in rotated basis

    bits_per_ch = allocate_channel_bits(R_rot, budget_bits, tiers=tiers, axis=0)

    R_rot_hat = torch.zeros_like(R_rot)
    for b in sorted(set(int(x) for x in bits_per_ch.tolist())):
        if b == 0:
            continue
        cols = (bits_per_ch == b).nonzero(as_tuple=True)[0]
        if cols.numel() == 0:
            continue
        sub = R_rot[:, cols]
        sub_hat = rtn_quantize(sub.mT, b, group).mT
        R_rot_hat[:, cols] = sub_hat

    R_hat = R_rot_hat @ Q.mT  # unrotate (Q orthogonal => Q^{-1} = Q^T)
    M_hat = L + R_hat

    mean_payload = float(bits_per_ch.float().mean().item())
    scale_term = 16.0 / group
    factor_term = 16.0 * rank * (S + C) / (S * C)
    tier_term = math.ceil(math.log2(len(tiers))) / S
    bpe = mean_payload + scale_term + factor_term + tier_term
    if rotation == "klt" and charge_rotation:
        bpe += 16.0 * C / S  # C×C fp16 rotation matrix amortized over S tokens
    return M_hat, bpe
```

(c) Thread `rotation`, `seed`, `charge_rotation` through `quantize_cache` and add dispatch. Update the signature (add `charge_rotation: bool = False`; `seed`, `tiers` already exist), and replace the final dispatch block:

```python
    elif arm == "lowrank_waterfill_channel":
        return _lowrank_waterfill_channel(
            M, float(bits), group, rank, tiers=tiers, svd_factors=svd_factors
        )
    elif arm == "lowrank_eigwaterfill_channel":
        return _lowrank_rotwaterfill_channel(
            M, float(bits), group, rank, tiers=tiers, rotation="klt",
            charge_rotation=charge_rotation, svd_factors=svd_factors,
        )
    else:  # lowrank_randwaterfill_channel — guarded by the CACHE_ARMS assert above
        return _lowrank_rotwaterfill_channel(
            M, float(bits), group, rank, tiers=tiers, rotation="random", seed=seed,
            svd_factors=svd_factors,
        )
```

Add `charge_rotation: bool = False` to the `quantize_cache` signature (after `tiers`) and one docstring line: `charge_rotation : add the KLT rotation-matrix cost (16*C/S) to bpe; lowrank_eigwaterfill_channel only.`

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cache_codecs.py -k "rotwaterfill or rotation_is or klt or random_arm" -q
```
Expected: PASS (7 tests).

- [ ] **Step 5: Format, lint, full test**

```bash
uv run ruff format . && uv run ruff check . && uv run pytest -q
```
Expected: all clean (prior 198 + 7 new = 205 passed, 1 xfailed).

- [ ] **Step 6: Commit (pre-authorized)**

```bash
git add src/bmx/cache/codecs.py tests/test_cache_codecs.py
git commit -m "feat(cache): rotated-waterfill core with KLT and random-rotation arms"
```
Stage ONLY those two files. One sentence, no AI attribution.

---

## Task 2: Experiment — five arms, dual metric, alignment diagnostic, honest pass

**Files:**
- Modify: `experiments/k2_waterfill.py`
- Modify: `tests/test_k2_waterfill.py`

**Interfaces:**
- Consumes (Task 1): `quantize_cache("lowrank_eigwaterfill_channel", ...)` and `quantize_cache("lowrank_randwaterfill_channel", ...)` with `charge_rotation`.
- Consumes (existing): the `main(cfg) -> DataFrame`, `score()` closure, `_resid_stable_rank` already in `k2_waterfill.py`.
- Produces: a `_query_eigen_alignment(M, Q_full, factors, rank, bits_per_ch_rot)`-style helper and new parquet columns `arm`, `bpe`, `logit_rope`, `rel_fro`, `query_eigen_alignment`, `bpe_honest` (KLT only, else NaN).

- [ ] **Step 1: Write the failing test (extend the smoke test)**

Append to `tests/test_k2_waterfill.py`:

```python
def test_experiment_has_rotated_arms_and_dual_metric(tmp_path):
    from safetensors.torch import save_file

    h_kv, S, d = 2, 64, 8
    g = torch.Generator().manual_seed(7)
    tensors = {}
    for i in range(2):
        tensors[f"layer{i}.k_pre"] = torch.randn(h_kv, S, d, generator=g).half()
        tensors[f"layer{i}.k"] = torch.randn(h_kv, S, d, generator=g).half()
        tensors[f"layer{i}.v"] = torch.randn(h_kv, S, d, generator=g).half()
        tensors[f"layer{i}.q"] = torch.randn(h_kv, S, d, generator=g).half()
    cache_path = tmp_path / "synthetic.safetensors"
    save_file(tensors, str(cache_path))

    cfg = k2_waterfill.Config(
        cache_path=str(cache_path), model_label="synthetic", model_name="",
        budget_bits=3.0, group=16, rank=4, out_root=str(tmp_path / "results"),
    )
    df = k2_waterfill.main(cfg)
    arms = set(df["arm"].unique())
    assert {"lowrank_eigwaterfill_channel", "lowrank_randwaterfill_channel"} <= arms
    # dual metric present and populated for every arm
    assert "rel_fro" in df.columns and "logit_rope" in df.columns
    assert df["rel_fro"].notna().all() and df["logit_rope"].notna().all()
    # alignment diagnostic present for the rotated arms
    assert "query_eigen_alignment" in df.columns
    eig = df[df.arm == "lowrank_eigwaterfill_channel"]
    assert eig["query_eigen_alignment"].notna().all()
    assert ((eig["query_eigen_alignment"] >= 0) & (eig["query_eigen_alignment"] <= 1.0001)).all()
    # matched idealized bpe: both rotated arms within tol of uniform, per layer
    for layer in df["layer"].unique():
        sub = df[df["layer"] == layer]
        bpe_uni = sub[sub.arm == "lowrank_rtn_channel"]["bpe"].mean()
        for arm in ("lowrank_eigwaterfill_channel", "lowrank_randwaterfill_channel"):
            bpe_arm = sub[sub.arm == arm]["bpe"].mean()
            assert abs(bpe_uni - bpe_arm) < 0.05, f"{arm} L{layer} bpe {bpe_arm} vs {bpe_uni}"
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_k2_waterfill.py::test_experiment_has_rotated_arms_and_dual_metric -q
```
Expected: FAIL — new arms / `query_eigen_alignment` column absent.

- [ ] **Step 3: Implement the experiment extension**

In `experiments/k2_waterfill.py`:

(a) Add the alignment helper after `_resid_stable_rank` (computes the fraction of real-query energy in the top-k funded KLT eigen-directions; queries are flattened to channel space matching M's `(S, C)` layout via `to_matrix`):

The alignment is computed per kv-head, then averaged — this is faithful to the
`(h_kv, S, d)` channel layout (no GQA proxy hand-wave). `R` is reshaped back to
per-head `(h_kv, S, d)` via `from_matrix`, the KLT is taken on the full `(S, C)`
residual but the query projection is done head-blocked so query head `j` (GQA-mapped
to its kv-head group) projects onto the eigen-directions, and we report the
query-energy fraction landing in the funded directions.

```python
def _query_eigen_alignment(R: torch.Tensor, q_t: torch.Tensor, h_kv: int, k: int) -> float:
    """Fraction of real-query energy in the top-k funded KLT eigen-directions of R.

    R: (S, C) residual (C = h_kv*d). q_t: (h_q, T, d) queries (h_q = query heads,
    a GQA multiple of h_kv). k: number of funded eigencols.

    Faithful to the metric's channel layout: the KLT eigenvectors live in the full
    (S, C) channel space, so each query head is GQA-mapped to its kv-head's d-dim
    block of C and projected there. Returns the mean over query rows of the energy
    fraction ||proj_topk||^2 / ||q||^2, averaged over query heads.

    Diagnostic only (directional, not a metric input): if alignment is high yet the
    KLT arm still loses logit, the funded high-variance directions are read by queries
    but quantizing them still costs more than uniform — a clean Outcome-3 signature.
    """
    if k <= 0:
        return 0.0
    C = R.shape[1]
    d = C // h_kv
    _, eigvecs = torch.linalg.eigh(R.mT @ R)
    Qk = eigvecs.flip(dims=(1,))[:, :k]  # (C, k) top-k eigen-directions
    h_q, T, _ = q_t.shape
    group = h_q // h_kv  # GQA: how many query heads share each kv head
    fracs = []
    for j in range(h_q):
        kv = j // group  # this query head's kv-head index
        # embed this head's d-dim query into the full C-dim channel space at its block
        qf = torch.zeros(T, C, dtype=R.dtype)
        qf[:, kv * d : (kv + 1) * d] = q_t[j].to(R.dtype)
        proj = qf @ Qk  # (T, k)
        num = (proj**2).sum(dim=1)
        den = (qf**2).sum(dim=1).clamp_min(1e-30)
        fracs.append((num / den).mean())
    return float(torch.stack(fracs).mean().item())
```

(b) Replace the per-layer arm block. After computing `factors`, `sr`, and `R = M - (factors[0] @ factors[1].mT)`, add the two rotated arms and the alignment, and record BOTH metrics + `bpe_honest`. Replace lines from the `wf, bpe_wf = ...` call through the row-append loop with:

```python
        R_resid = M - (factors[0] @ factors[1].mT)

        uni, bpe_uni = quantize_cache(
            "lowrank_rtn_channel", M, bits=round(cfg.budget_bits),
            group=cfg.group, rank=cfg.rank, svd_factors=factors,
        )
        wf, bpe_wf = quantize_cache(
            "lowrank_waterfill_channel", M, bits=cfg.budget_bits,
            group=cfg.group, rank=cfg.rank, tiers=cfg.tiers, svd_factors=factors,
        )
        eig, bpe_eig = quantize_cache(
            "lowrank_eigwaterfill_channel", M, bits=cfg.budget_bits,
            group=cfg.group, rank=cfg.rank, tiers=cfg.tiers, svd_factors=factors,
        )
        rnd, bpe_rnd = quantize_cache(
            "lowrank_randwaterfill_channel", M, bits=cfg.budget_bits, seed=cfg.seed,
            group=cfg.group, rank=cfg.rank, tiers=cfg.tiers, svd_factors=factors,
        )
        ot, bpe_ot = _outlier_two_tier(M, cfg.budget_bits, cfg.group, cfg.rank, factors)

        for nm, b in (("eig", bpe_eig), ("rand", bpe_rnd)):
            assert abs(bpe_uni - b) < 0.05, f"L{layer_i}: {nm} bpe {b:.3f} vs uniform {bpe_uni:.3f}"
        assert abs(bpe_uni - bpe_wf) < 0.05, (
            f"L{layer_i}: waterfill bpe {bpe_wf:.3f} not matched to uniform {bpe_uni:.3f}"
        )

        # alignment for the KLT arm: k = funded eigencols of the KLT-rotated residual
        from bmx.cache.codecs import allocate_channel_bits

        _, eigvecs_a = torch.linalg.eigh(R_resid.mT @ R_resid)
        Q_klt = eigvecs_a.flip(dims=(1,))
        bits_rot = allocate_channel_bits(R_resid @ Q_klt, cfg.budget_bits, tiers=cfg.tiers, axis=0)
        k_funded = int((bits_rot > 0).sum().item())
        align = _query_eigen_alignment(R_resid, km["q"].float(), h_kv, k_funded)

        # honest KLT pass: only meaningful if eig beats uniform on logit at this layer
        _, lg_uni = score(uni)
        _, lg_eig = score(eig)
        bpe_eig_honest = float("nan")
        if lg_eig < lg_uni:
            _, bpe_eig_honest = quantize_cache(
                "lowrank_eigwaterfill_channel", M, bits=cfg.budget_bits,
                group=cfg.group, rank=cfg.rank, tiers=cfg.tiers,
                charge_rotation=True, svd_factors=factors,
            )

        arm_rows = {
            "lowrank_rtn_channel": (uni, bpe_uni, float("nan")),
            "lowrank_waterfill_channel": (wf, bpe_wf, float("nan")),
            "lowrank_eigwaterfill_channel": (eig, bpe_eig, bpe_eig_honest),
            "lowrank_randwaterfill_channel": (rnd, bpe_rnd, float("nan")),
            "outlier_two_tier": (ot, bpe_ot, float("nan")),
        }
        for arm, (M_hat, bpe, bpe_honest) in arm_rows.items():
            rf, lg = score(M_hat)
            align_col = align if arm in (
                "lowrank_eigwaterfill_channel", "lowrank_randwaterfill_channel"
            ) else float("nan")
            rows.append(
                dict(
                    model=cfg.model_label or "unknown", layer=layer_i, kind="k_pre",
                    arm=arm, rank=cfg.rank, bpe=bpe, bpe_honest=bpe_honest,
                    rel_fro=rf, logit_rope=lg, resid_stable_rank=sr,
                    query_eigen_alignment=align_col,
                )
            )
            print(
                f"  L{layer_i:2d} {arm:30s} bpe={bpe:.3f} logit={lg:.4f} "
                f"rel_fro={rf:.4f} align={align_col if align_col==align_col else float('nan'):.3f}",
                flush=True,
            )
```

(c) Update the module docstring arm list (lines 4-8) to mention the two rotated arms and the dual metric, and the SUMMARY block to print both metrics + alignment + the per-layer win-rate vs uniform:

```python
    print("\n" + "=" * 70)
    print("SUMMARY — mean logit_rope / rel_fro per arm (lower better); align = query-eigen")
    uni_by_layer = (
        df[df.arm == "lowrank_rtn_channel"].set_index("layer")["logit_rope"]
    )
    for arm in sorted(df.arm.unique()):
        sub = df[df.arm == arm]
        merged = sub.set_index("layer")["logit_rope"]
        wins = int((merged < uni_by_layer.reindex(merged.index)).sum())
        align_mean = sub["query_eigen_alignment"].mean()
        print(
            f"  {arm:30s} logit={sub.logit_rope.mean():.4f} rel_fro={sub.rel_fro.mean():.4f} "
            f"bpe={sub.bpe.mean():.3f} align={align_mean:.3f} beats_uniform={wins}/{sub.layer.nunique()}"
        )
    print(f"\n-> {run}")
    return df
```

- [ ] **Step 4: Run the smoke test to verify it passes**

```bash
uv run pytest tests/test_k2_waterfill.py -q
```
Expected: PASS (all k2_waterfill tests).

- [ ] **Step 5: Format, lint, full test**

```bash
uv run ruff format . && uv run ruff check . && uv run pytest -q
```
Expected: all clean.

- [ ] **Step 6: Commit (pre-authorized)**

```bash
git add experiments/k2_waterfill.py tests/test_k2_waterfill.py
git commit -m "feat(exp): add KLT and random rotated-waterfill arms with dual-metric and alignment diagnostic"
```
Stage ONLY those two files.

---

## Task 3: Run the ablation and write the verdict

**Files:**
- Create: `docs/2026-06-21-k2-eigwaterfill-results.md`

Execution + science judgment; no test cycle. Depends on Tasks 1-2 committed.

- [ ] **Step 1: Run on GPT-2 (fast) then Llama (authoritative)**

```bash
cd "$(git rev-parse --show-toplevel)"
uv run python experiments/k2_waterfill.py \
    --cache-path /d/Projects/bmx/results/cache/gpt2_1024.safetensors \
    --model-label gpt2 --budget-bits 3.0 --rank 16 2>&1 | tail -30
uv run python experiments/k2_waterfill.py \
    --cache-path /d/Projects/bmx/results/cache/llama-3.1-8b_2048.safetensors \
    --model-label llama-3.1-8b --model-name meta-llama/Llama-3.1-8B \
    --budget-bits 3.0 --rank 16 2>&1 | tail -45
```

- [ ] **Step 2: Pull the decision matrix from the parquets**

```bash
uv run python -c "
import pandas as pd, glob, os
for d in sorted(glob.glob('results/k2_waterfill/*/')):
    pq = glob.glob(os.path.join(d,'*.parquet'))
    if not pq: continue
    df = pd.read_parquet(pq[0])
    if 'query_eigen_alignment' not in df.columns: continue  # skip pre-Task2 runs
    lab = df['model'].iloc[0]
    g = df.groupby('arm').agg(logit=('logit_rope','mean'), mse=('rel_fro','mean'),
        bpe=('bpe','mean'), align=('query_eigen_alignment','mean'))
    piv = df.pivot_table(index='layer', columns='arm', values='logit_rope')
    print(f'=== {lab} ({d}) ==='); print(g.to_string())
    uni = piv['lowrank_rtn_channel']
    for arm in ['lowrank_eigwaterfill_channel','lowrank_randwaterfill_channel','lowrank_waterfill_channel']:
        if arm in piv: print(f'{arm}: beats uniform {(piv[arm]<uni).sum()}/{len(piv)} layers')
    print()
"
```

- [ ] **Step 3: Write the verdict doc**

Create `docs/2026-06-21-k2-eigwaterfill-results.md` in the terse numbers-first style of `docs/2026-06-21-k2-waterfill-results.md`. Record, faithfully:
- The decision matrix (both models): arm × {logit_rope, rel_fro, bpe, query_eigen_alignment, beats-uniform/N}.
- The verdict using the spec's decisive logic:
  - **Revival CONFIRMED** iff `lowrank_eigwaterfill_channel` beats uniform on logit by > ~2 sem AND a clear majority of layers AND beats the `lowrank_randwaterfill_channel` control.
  - **Cheap-revival** iff the random arm beats uniform on logit at its (free) bpe — note the KLT matrix is then moot.
  - **Objective-mismatch CONFIRMED** (most likely) iff eig wins `rel_fro` (MSE) but loses `logit_rope` — direct measured proof the kill was an objective mismatch, not a basis mistake.
  - **KILLED-honest** iff eig wins idealized logit but the honest pass (`bpe_honest`, +16·C/S) erases the win — "real but too expensive to encode."
- Tie the result to `query_eigen_alignment`: if alignment is low, queries don't read the high-variance eigen-directions → explains a logit loss causally.
- Gate call, and what (if anything) it changes about the k2b recipe / whether the deferred structured-rotation follow-up is warranted.

- [ ] **Step 4: Commit (pre-authorized)**

```bash
git add docs/2026-06-21-k2-eigwaterfill-results.md
git commit -m "docs: eigenbasis water-filling ablation results and verdict"
```
Stage ONLY the doc. Confirm `results/` run dirs are NOT staged (gitignored).

---

## Notes for the executor

- **The load-bearing control is the random arm.** A KLT win is only meaningful if it beats BOTH uniform and the random-rotation arm. If random ties KLT, "any rotation helps" — not the eigenstructure. Report this explicitly.
- **Dual metric proves the mechanism.** eig-wins-MSE-loses-logit is the measured signature of the objective mismatch; it is a positive scientific finding, not a null. Do not bury it.
- **fp16 factor roundtrip is UNCONDITIONAL** (the dtype-guard bug already bit in the prior cycle). The core mirrors `_lowrank_waterfill_channel` exactly there.
- **Eigvec sign ambiguity**: never assert raw-tensor equality across a KLT; compare logit distortion (the Q=I equivalence test does this).
- `logit_distortion(K_orig, K_hat, Q)` is GQA-aware and takes the `(h_kv, S, d)` layout via `from_matrix` — same call shape as the existing `score()` closure. Do not hand-roll attention.
- Matched bpe is **asserted, not assumed**, for both rotated arms. If it fires on real data, investigate the allocation; do not loosen the tolerance.
