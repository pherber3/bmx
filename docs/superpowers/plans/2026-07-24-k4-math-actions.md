# K4 Math-Review Actions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the three approved math-review actions (charge-aware allocation A-gate, w_rope W-instrument A/B, int8 decoder certificate) plus the folded-in measured-ĝ table — all local (gpt2/qwen3-0.6b mechanism scale + analytic), all default-inert, each gate pre-registered, with the results doc that carries both verdict templates and the user-review sentence for the certificate.

**Architecture:** One allocator upgrade in `codecs.py` (exact Lagrangian tier selection with an optional per-direction fixed charge and an optional measured-ĝ table, beside the untouched midpoint-rounding path), threaded through `spectral.py`'s fitting path as `s_ref`/`g_table` kwargs; one corrected query-moment variant (`w_rope="rotated"`) in `spectral.py` threaded through `_k4_common.py`; one certificate function beside `int8_decoder_roundtrip`. Three thin new tyro experiments (`k4_charge_alloc.py`, `k4_w_rope_ab.py`, `k4_g_table.py`, `k4_int8_certificate.py` — four scripts, three gates) reuse the G1 win machinery (`_layer_ctx`, `_score_tail`, `_tq_layer_curve`, `_log_interp`, `corpus_fit_bases`) verbatim. A results doc closes the loop.

**Tech Stack:** Python 3.12 / torch (CPU) / pandas+parquet / tyro / safetensors / pytest. No new dependencies.

**Binding spec:** `docs/superpowers/specs/2026-07-24-k4-math-actions-design.md`
**Authoritative math (all formulas transcribed from it):** `docs/2026-07-24-k4-math-review.md` (findings #1, #2, #3, #4, #9)

## Global Constraints

- Branch `feat/triton-decode-kernel` (HEAD `5da8faa` at plan time). Working dir `D:\Projects\bmx`; Bash tool (git bash), `cd /d/Projects/bmx` first in every fresh shell.
- **Default-inert everywhere:** `s_ref=None`, `g_table=None`, `w_rope="frozen"`, `selection="round"`, `fixed_charge=0.0` reproduce today's outputs **bit-exactly**, each pinned by a test. The existing pins (`test_k4_fit_packs_default_unchanged`, `test_pack_from_basis_lam_alloc_default_unchanged`, `test_corpus_fit_bases_matches_direct_fit`) must keep passing untouched.
- **NEVER `git commit` without the user's explicit approval.** Every commit step below means: stage, propose the message, STOP and report for approval. No AI attribution ever.
- Before every commit: `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` — all clean, then re-stage.
- Battery baseline: **531 passed / 17 skipped / 1 xfailed** (~50 s). This plan adds 20 tests → final expected **551 passed / 17 skipped / 1 xfailed**. Per-task expected counts are stated in each task.
- Accounting expressions are the scientific record: `skeptic_charge`, `spectral_payload_bpe`, `spectral_payload_v1_bpe` are **not modified**. The charge-aware change is allocation-only (`c_used` shrinks; the bpe expressions stay frozen) — math review #2's discipline note.
- **No recipe/CACHE_ARMS registration** of any `k4_b{budget}_s{S_ref}` arm; no `save_pack_file`/`load_packs` format change; shipped packs (`results/cache/k4_packs_gpt2.safetensors`, `k4_packs_qwen3_06b.safetensors`), duel parquets, and streaming numerics untouched (read-only inputs).
- fp64 for all moment/eig/allocation math; fp32 codec application; tests use tiny in-test fixtures (`_tiny_cache`, `_spiked_keys` idioms), never download; experiments are thin tyro scripts writing `results/<exp>/<run-id>/` via `create_run`/`write_metrics`.
- No eigenvalue shrinkage, no 1-bit tier, no mean-centering lever (future-work lines in the results doc only).
- The int8 certificate's numbers are **REVIEWED BY THE USER before VM Task 8 is released** — the results doc must carry that sentence verbatim (Task 7).

## Conflicts found while planning (executor must know)

1. **w_rope A/B substrate (Task 4) — spec-vs-math tension, needs user sign-off at execution.** The spec says "Fit gpt2 bases both ways on the same caches", but gpt2 has **no RoPE** (`model_name=""` ⇒ `cos=ones, sin=zeros`), so `w_rope="frozen"` and `"rotated"` produce mathematically identical W on gpt2 — the finding-#3 mechanism (sign of `sin·cos` plane terms + triangular offset weights) is provably inert there and the 2% decision rule would be vacuous. The math doc's mechanism requires a RoPE model. Local RoPE-model caches exist: `results/cache/qwen3-0.6b_2048_off{2048,4096}.safetensors` (fit) + `qwen3-0.6b_2048.safetensors` (heldout scored), already used for `k4_packs_qwen3_06b`. **This plan runs the A/B on qwen3-0.6b and keeps gpt2 as the exact-null control pin.** Get the user's explicit OK for this substrate substitution before executing Task 4's experiment run (the library change itself is uncontroversial).
2. **Certificate evaluation measure (Task 6) — reconciled, no action.** Math doc #9 states the added distortion `mean_rows ‖W̃^{1/2}Δŷ‖²` is exactly computable "on calibration data"; the spec says pack-side, "no caches", from "the pack's own lam/moments". These reconcile: `encᵀ Σ_fit enc = diag(lam)` **exactly** (property of the eigendecomposition), so the calibration-row expectation with code second moment `diag(lam)` has the closed form `Σ_i lam_i·‖encᵀ Δdec[:,i]‖²`. What the closed form models away — `E[ŷŷᵀ] ≠ E[yyᵀ]` (payload shift) and the payload×decoder cross-term, both `O(g(b))` relative on a ~7e-5 base — goes in the certificate's honest-limits section, exactly as the spec's deliverable list anticipates ("what it does NOT capture").

## File Structure

- Modify: `src/bmx/cache/codecs.py` — `_tier_g`, `_lagrange_select`, `allocate_bits_from_variance` gains `selection`/`g_table`/`fixed_charge` kwargs (Task 1)
- Modify: `src/bmx/cache/spectral.py` — `pack_from_basis`/`fit_spectral_pack` gain `s_ref`/`g_table` (Task 2); `query_position_moment` gains `w_rope` (Task 4); `int8_decoder_certificate` (Task 6)
- Modify: `experiments/_k4_common.py` — `corpus_query_moment`/`corpus_fit_bases` gain `w_rope` passthrough (Task 4)
- Create: `experiments/k4_charge_alloc.py` — the A-gate (Task 3)
- Create: `experiments/k4_w_rope_ab.py` — the W-instrument A/B (Task 4)
- Create: `experiments/k4_g_table.py` — the measured-ĝ table (Task 5)
- Create: `experiments/k4_int8_certificate.py` — the certificate table (Task 6)
- Create: `docs/2026-07-24-k4-math-actions-results.md` (Task 7)
- Tests: `tests/test_cache_codecs.py` (Task 1), `tests/test_spectral.py` (Tasks 2, 4, 6), `tests/test_k4_experiments.py` (Tasks 3, 4, 5, 6)

---

### Task 1: Lagrangian tier selection + g_table + fixed-charge groundwork in `allocate_bits_from_variance`

**Files:**
- Modify: `src/bmx/cache/codecs.py` (after `_round_to_tiers`, ~line 152; and the body of `allocate_bits_from_variance`, ~line 155)
- Test: `tests/test_cache_codecs.py` (append after `test_allocate_from_variance_rich_tiers`, ~line 400)

**Interfaces:**
- Consumes: existing `_round_to_tiers(b, tiers_t)`, `math`, `torch` (already imported in codecs.py).
- Produces (later tasks rely on these exact signatures):
  - `allocate_bits_from_variance(var, budget_bits, tiers=(0, 2, 3, 4), *, n_search=40, selection="round", g_table=None, fixed_charge=0.0) -> torch.Tensor` — `(C,) int64`, members of `tiers`.
  - `_lagrange_select(var, kappa_l, tiers_t, g_t, fixed_charge) -> torch.Tensor` (fp64 tier values, `(C,)`) — Task 1 tests call it directly with explicit `kappa_l`.
  - `_tier_g(tiers_t, g_table) -> torch.Tensor` — per-tier fp64 g values; validates measured tables.
- Budget semantics: under `selection="lagrange"`, `budget_bits` bounds the mean per-direction **total charge** `(1/C)·Σ_i [b_i + fixed_charge·1[b_i>0]]`; with `fixed_charge=0.0` this is `mean(bits)`, same semantics as `"round"`.

**The math being transcribed (authoritative: math review #1, #2, #4):**
- Lagrangian selection: `b_i(κ_L) = argmin_{b∈T} λ_i·g(b) + κ_L·(b + s·1[b>0])` — exact enumeration over the tier grid (the fixed charge makes the per-direction cost non-convex at 0; enumeration over 7 tiers is exact anyway).
- Everett's theorem: the returned allocation is optimal among all allocations with the same or smaller achieved total charge (finding #1(a) — never used convexity).
- Optimal switch points with `g = 4^{−b}`: `λ*_j = κ_L·Δ_j/(g(t_{j−1}) − g(t_j))`; in `b_cont = ½log₂(λ/κ)` coordinates (`κ_L = κ·ln4`): `b_cont(λ*_j) = t_{j−1} + ½·log₂(Δ_j·ln4 / (1 − 4^{−Δ_j}))` = **t_{j−1} + 0.443** for unit gaps, **t_{j−1} + 0.782** for 2-bit gaps (0↔2, 6↔8) — vs the implemented midpoints +0.5/+1.0. These are the test vectors. The exact enumeration **subsumes** the threshold-offset fix entirely (spec §A(i)); no separate rounding change exists.
- λ-search: bisection on κ_L over the log bracket `[var.min()·ln4·1e−6, var.max()·ln4·1e6]`, keep best feasible, 40 iterations — same discipline as the shipped bisection. Valid because for tiers `b > b'`, `cost(b) − cost(b') = λ(g(b)−g(b')) + κ_L·(b−b'+s·(1[b>0]−1[b'>0]))` has a strictly positive κ_L-coefficient, so the per-direction argmin (hence the charged mean) is non-increasing in κ_L.
- Grid-convexity condition for measured tables (finding #4): per-bit marginals `(g(t_{j−1})−g(t_j))/Δ_j` strictly decreasing; `g(0) = 1` **exact** (a dropped direction's error is the coordinate itself). The doc's measured RTN values `(1, 0.119, 0.0374, 0.0115, 0.0035, 0.0010, 9e−5)` at `b=(0,2,3,4,5,6,8)` have marginals `(0.441, 0.081, 0.026, 0.008, 0.0024, 0.0005)` — decreasing.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_cache_codecs.py`:

```python
# ---------------------------------------------------------------------------
# Lagrangian tier selection (math review 2026-07-24 findings #1/#2/#4)
# ---------------------------------------------------------------------------


def test_allocate_bits_selection_round_default_pin():
    """Defaults reproduce the historical midpoint-rounding path bit-exactly:
    explicit selection='round' == bare call, and a hand-computable case pins
    the round path's exact output (var=(256,16,1), tiers (0,2,4), budget 2.0:
    bisection's left feasible endpoint rounds to (4,2,0))."""
    from bmx.cache.codecs import allocate_bits_from_variance

    var = torch.tensor([256.0, 16.0, 1.0])
    bare = allocate_bits_from_variance(var, 2.0, (0, 2, 4))
    explicit = allocate_bits_from_variance(var, 2.0, (0, 2, 4), selection="round")
    assert torch.equal(bare, explicit)
    assert torch.equal(bare, torch.tensor([4, 2, 0], dtype=torch.int64))
    # New kwargs are rejected on the round path (they only mean anything to
    # the Lagrangian enumeration).
    import pytest

    with pytest.raises(AssertionError):
        allocate_bits_from_variance(var, 2.0, (0, 2, 4), fixed_charge=1.0)
    with pytest.raises(AssertionError):
        allocate_bits_from_variance(var, 2.0, (0, 2, 4), g_table=(1.0, 0.1, 0.01))


def test_lagrange_thresholds_match_math_review():
    """The math doc's worked switch points are test vectors. With g=4^{-b},
    tiers (0,2,3,4,5,6,8), kappa_l=1: the 0->2 switch sits at
    lam* = 2/(1-4^{-2}) = 2.1333; the 2->3 switch at 1/(4^{-2}-4^{-3}) =
    21.3333. In b_cont coordinates these are t + 0.782 (2-bit gap) and
    t + 0.443 (unit gap)."""
    import math

    from bmx.cache.codecs import _lagrange_select, _tier_g

    tiers_t = torch.tensor([0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0], dtype=torch.float64)
    g_t = _tier_g(tiers_t, None)

    def pick(lam):
        return float(
            _lagrange_select(
                torch.tensor([lam], dtype=torch.float64), 1.0, tiers_t, g_t, 0.0
            )[0]
        )

    assert pick(2.13) == 0.0 and pick(2.14) == 2.0  # 0<->2 boundary
    assert pick(21.3) == 2.0 and pick(21.4) == 3.0  # 2<->3 boundary
    # b_cont-coordinate offsets (kappa = kappa_l/ln4):
    off2 = 0.5 * math.log2((2.0 / (1 - 4.0**-2)) * math.log(4.0))
    off1 = 0.5 * math.log2((1.0 / (4.0**-2 - 4.0**-3)) * math.log(4.0)) - 2.0
    assert abs(off2 - 0.782) < 5e-4
    assert abs(off1 - 0.443) < 5e-4
    # The (0.782, 1.0) window above the 0 boundary: b_cont=0.9 rounds to 0
    # under the midpoint rule but the Lagrangian opens it to 2 (the provably
    # dominated-window fix, finding #1(ii)).
    from bmx.cache.codecs import _round_to_tiers

    lam_w = 4.0**0.9  # kappa=1 => b_cont = 0.9 exactly
    assert float(_round_to_tiers(torch.tensor([0.9]), tiers_t)[0]) == 0.0
    assert (
        float(
            _lagrange_select(
                torch.tensor([lam_w], dtype=torch.float64),
                math.log(4.0),
                tiers_t,
                g_t,
                0.0,
            )[0]
        )
        == 2.0
    )


def test_lagrange_select_is_pointwise_argmin():
    """Exact-enumeration property: every chosen tier minimizes
    lam*g(b) + kappa_l*(b + s*[b>0]) over ALL tiers (brute force)."""
    from bmx.cache.codecs import _lagrange_select, _tier_g

    g = torch.Generator().manual_seed(0)
    lam = torch.rand(64, generator=g).double() * 1e4 + 1e-6
    tiers_t = torch.tensor([0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0], dtype=torch.float64)
    g_t = _tier_g(tiers_t, None)
    kappa_l, s = 0.37, 1.7
    chosen = _lagrange_select(lam, kappa_l, tiers_t, g_t, s)
    for i in range(64):
        costs = [
            float(lam[i]) * float(g_t[j])
            + kappa_l * (float(t) + s * (float(t) > 0))
            for j, t in enumerate(tiers_t)
        ]
        chosen_cost = costs[[float(t) for t in tiers_t].index(float(chosen[i]))]
        assert chosen_cost <= min(costs) + 1e-12, f"dir {i} not argmin"


def test_lagrange_fixed_charge_price_of_opening():
    """Math review #2's worked S=4096 row: s = 0.25 + 4.0 = 4.25 makes the
    true price of 0->2 equal 6.25, so the switch moves to
    lam* = kappa_l * 6.25/(1-4^{-2}) = 6.6667 (at kappa_l=1)."""
    from bmx.cache.codecs import _lagrange_select, _tier_g

    tiers_t = torch.tensor([0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0], dtype=torch.float64)
    g_t = _tier_g(tiers_t, None)

    def pick(lam):
        return float(
            _lagrange_select(
                torch.tensor([lam], dtype=torch.float64), 1.0, tiers_t, g_t, 4.25
            )[0]
        )

    assert pick(6.66) == 0.0 and pick(6.67) == 2.0


def test_lagrange_full_allocator_feasible_monotone_deterministic():
    from bmx.cache.codecs import allocate_bits_from_variance

    g = torch.Generator().manual_seed(3)
    var = torch.sort(torch.rand(128, generator=g) * 1e3 + 1e-4, descending=True)[0]
    s = 0.6
    bits = allocate_bits_from_variance(
        var, 2.5, (0, 2, 3, 4, 5, 6, 8), selection="lagrange", fixed_charge=s
    )
    bits2 = allocate_bits_from_variance(
        var, 2.5, (0, 2, 3, 4, 5, 6, 8), selection="lagrange", fixed_charge=s
    )
    assert torch.equal(bits, bits2)  # deterministic
    charged = (bits.double() + s * (bits > 0).double()).mean().item()
    assert charged <= 2.5 + 1e-9  # budget bounds the TOTAL charge
    assert (bits[:-1] >= bits[1:]).all()  # threshold structure in lam
    # fixed_charge=0.0 lagrange must also respect plain mean-bits feasibility
    b0 = allocate_bits_from_variance(
        var, 2.5, (0, 2, 3, 4, 5, 6, 8), selection="lagrange"
    )
    assert b0.double().mean().item() <= 2.5 + 1e-9


def test_g_table_validation_and_equivalence():
    """g_table=None <=> explicit 4^{-b} table (bit-exact); the math doc's
    measured RTN table is accepted (grid-convex); a non-convex table raises;
    and the measured table moves the 2<->3 boundary DOWN (the ~0.9 log2-lam
    misplacement, finding #4): at lam=15, kappa_l=1 the model picks 2, the
    measured table picks 3."""
    import pytest

    from bmx.cache.codecs import _lagrange_select, _tier_g, allocate_bits_from_variance

    tiers = (0, 2, 3, 4, 5, 6, 8)
    tiers_t = torch.tensor([float(t) for t in tiers], dtype=torch.float64)
    measured = (1.0, 0.119, 0.0374, 0.0115, 0.0035, 0.0010, 9e-5)

    g = torch.Generator().manual_seed(4)
    var = torch.rand(64, generator=g).double() * 100 + 1e-3
    explicit = tuple(4.0 ** -float(t) for t in tiers)
    a = allocate_bits_from_variance(var, 2.5, tiers, selection="lagrange")
    b = allocate_bits_from_variance(
        var, 2.5, tiers, selection="lagrange", g_table=explicit
    )
    assert torch.equal(a, b)

    g_meas = _tier_g(tiers_t, measured)
    lam15 = torch.tensor([15.0], dtype=torch.float64)
    assert float(_lagrange_select(lam15, 1.0, tiers_t, _tier_g(tiers_t, None), 0.0)[0]) == 2.0
    assert float(_lagrange_select(lam15, 1.0, tiers_t, g_meas, 0.0)[0]) == 3.0

    with pytest.raises(AssertionError, match="grid-convex"):
        _tier_g(tiers_t, (1.0, 0.119, 0.09, 0.0115, 0.0035, 0.0010, 9e-5))
    with pytest.raises(AssertionError):
        _tier_g(tiers_t, (0.9, 0.119, 0.0374, 0.0115, 0.0035, 0.0010, 9e-5))  # g(0)!=1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_cache_codecs.py -q -k "lagrange or g_table or selection_round"`
Expected: FAIL / ERROR — `ImportError: cannot import name '_lagrange_select'` (and the default-pin test fails on the unexpected-kwarg assertions).

- [ ] **Step 3: Implement** — in `src/bmx/cache/codecs.py`, insert directly after `_round_to_tiers` (keep `_round_to_tiers` and the entire round path byte-identical):

```python
_LN4 = math.log(4.0)


def _tier_g(
    tiers_t: torch.Tensor, g_table: tuple[float, ...] | None
) -> torch.Tensor:
    """Per-tier distortion shape g(b), index-aligned with the ascending tier
    grid. None => the model curve g(b) = 4^{-b}. A measured table (finding #4)
    must satisfy the optimality lemma's conditions: strictly decreasing,
    grid-convex (per-bit marginal gains strictly decreasing), and g(0) = 1
    exactly when tier 0 is present (a dropped direction's error IS the
    coordinate — no quantizer model enters at the drop boundary)."""
    if g_table is None:
        return torch.pow(4.0, -tiers_t)
    g = torch.tensor(g_table, dtype=torch.float64, device=tiers_t.device)
    assert g.shape == tiers_t.shape, (
        f"g_table length {tuple(g.shape)} != n_tiers {tuple(tiers_t.shape)}"
    )
    if float(tiers_t[0]) == 0.0:
        assert float(g[0]) == 1.0, f"g(0) must be exactly 1.0; got {float(g[0])}"
    assert (g[:-1] > g[1:]).all(), "g_table must be strictly decreasing"
    marg = (g[:-1] - g[1:]) / (tiers_t[1:] - tiers_t[:-1])
    assert (marg[:-1] > marg[1:]).all(), (
        "g_table must be grid-convex (strictly decreasing per-bit marginals) — "
        "the optimality lemma (math review 2026-07-24 #1) is false without it"
    )
    return g


def _lagrange_select(
    var: torch.Tensor,
    kappa_l: float,
    tiers_t: torch.Tensor,
    g_t: torch.Tensor,
    fixed_charge: float,
) -> torch.Tensor:
    """Exact per-direction tier choice (math review #1/#2):

        b_i = argmin_{b in T}  var_i * g(b) + kappa_l * (b + s * 1[b>0])

    Enumeration over the tier grid — exact even though the fixed charge s
    makes the per-direction cost non-convex at 0. Ties resolve to the SMALLER
    tier (argmin first occurrence over ascending tiers). Everett's theorem:
    the selection minimizes total distortion among all allocations whose
    total charge sum_i (b_i + s*1[b_i>0]) is <= the achieved one."""
    used = (tiers_t > 0).double()
    cost = var.unsqueeze(-1) * g_t + kappa_l * (tiers_t + fixed_charge * used)
    return tiers_t[cost.argmin(dim=-1)]
```

Then modify `allocate_bits_from_variance`: change the signature line to

```python
def allocate_bits_from_variance(
    var: torch.Tensor,
    budget_bits: float,
    tiers: tuple[int, ...] = (0, 2, 3, 4),
    *,
    n_search: int = 40,
    selection: str = "round",
    g_table: tuple[float, ...] | None = None,
    fixed_charge: float = 0.0,
) -> torch.Tensor:
```

extend the docstring with (append after the existing text):

```
    selection="round" (default): the historical midpoint-rounding bisection,
    bit-exact forever. selection="lagrange": exact Lagrangian tier choice
    b_i = argmin_{b in T} var_i*g(b) + kappa_l*(b + fixed_charge*1[b>0])
    with bisection on kappa_l (Everett-optimal at the achieved total charge;
    math review 2026-07-24 #1/#2). Under "lagrange", budget_bits bounds the
    mean per-direction TOTAL charge (1/C)*sum_i (b_i + fixed_charge*1[b_i>0])
    — with fixed_charge=0.0 that is mean(bits), the same semantics as
    "round". g_table (finding #4) replaces the 4^{-b} model with measured
    per-tier ratios; g_table/fixed_charge are lagrange-only.
```

and insert, immediately after the existing `tiers_t = torch.tensor(...)` line and before the round path's `rounded_mean` definition:

```python
    assert selection in ("round", "lagrange"), f"unknown selection {selection!r}"
    if selection == "round":
        assert g_table is None and fixed_charge == 0.0, (
            "g_table/fixed_charge require selection='lagrange'"
        )
    else:
        g_t = _tier_g(tiers_t, g_table)
        lo = math.log(float(var.min().item()) * _LN4 * 1e-6)
        hi = math.log(float(var.max().item()) * _LN4 * 1e6)

        def charged_mean(b: torch.Tensor) -> float:
            return float((b + fixed_charge * (b > 0).double()).mean().item())

        best = _lagrange_select(var, math.exp(hi), tiers_t, g_t, fixed_charge)
        for _ in range(n_search):
            mid = 0.5 * (lo + hi)
            b = _lagrange_select(var, math.exp(mid), tiers_t, g_t, fixed_charge)
            if charged_mean(b) <= budget_bits + 1e-12:
                best = b  # feasible; try smaller kappa_l (more bits)
                hi = mid
            else:
                lo = mid
        return best.to(torch.int64)
```

The remainder of the function (the round path) stays byte-identical.

- [ ] **Step 4: Run the new tests + the pre-existing allocator tests**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_cache_codecs.py -q`
Expected: PASS (all, including the six new tests and every pre-existing `test_allocate_*`).

- [ ] **Step 5: Full battery**

Run: `cd /d/Projects/bmx && uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: `537 passed, 17 skipped, 1 xfailed`, ruff clean.

- [ ] **Step 6: Stage and propose the commit (STOP for user approval — never commit without it)**

```bash
git add src/bmx/cache/codecs.py tests/test_cache_codecs.py
```

Proposed message: `feat(codecs): exact Lagrangian tier selection + measured-g table + per-direction fixed charge in allocate_bits_from_variance (default-inert, round path byte-identical)`

---

### Task 2: Charge-aware enumeration (`s_ref`) + `g_table` on the pack fitting path

**Files:**
- Modify: `src/bmx/cache/spectral.py` (`pack_from_basis`, ~line 193; `fit_spectral_pack`, ~line 235)
- Test: `tests/test_spectral.py` (append at end)

**Interfaces:**
- Consumes (Task 1): `allocate_bits_from_variance(..., selection="lagrange", g_table=..., fixed_charge=...)`; existing `scale_bits(group)`.
- Produces (Tasks 3/5 rely on these exact signatures):
  - `pack_from_basis(basis, budget, *, tiers=(0, 2, 3, 4, 5, 6, 8), group=64, lam_alloc=None, s_ref=None, g_table=None) -> SpectralPack`
  - `fit_spectral_pack(M_fit, Wh, Wh_inv, budget, *, tiers=(0, 2, 3, 4, 5, 6, 8), group=64, s_ref=None, g_table=None) -> SpectralPack`
- `SpectralPack` dataclass, `save_pack_file`, `load_packs`, recipes, CACHE_ARMS: **untouched** (charge-aware packs exist only in-process in the experiments until the gate passes and the user promotes the arm).

**The math being transcribed (authoritative: math review #2):** the per-direction fixed charge is

    s = 16/group + dec_bits * C / S_ref        (dec_bits = 16: fp16 decoder — the A-gate never invokes the int8 lever)

i.e. in code `s_charge = scale_bits(group) + 16.0 * C / s_ref`, and the selection is `b_i = argmin_{b in T} lam_i*g(b) + kappa_L*(b + s*1[b>0])` from Task 1. Budget semantics with `s_ref` set: `budget` bounds `(1/C)*sum_i (b_i + s*1[b_i>0])` = payload-v2 bpe + `16*c_used/S_ref` = skeptic-v2 bpe@S_ref minus `tier_bits(tiers, S_ref)`. This is an **allocation** change only — the bpe expressions stay frozen; `c_used` simply becomes smaller (math review #2's discipline note, transcribed into the docstring).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_spectral.py`:

```python
def test_pack_from_basis_s_ref_default_inert():
    """s_ref=None / g_table=None reproduce today's allocation bit-exactly
    (the default-inert pin idiom), on both pack_from_basis and
    fit_spectral_pack."""
    from bmx.cache.spectral import fit_spectral_basis, pack_from_basis

    M, _ = _spiked_keys(S=256, C=64, seed=3)
    Wh, Wh_inv = identity_whitener(64)
    basis = fit_spectral_basis(M, Wh, Wh_inv)
    default = pack_from_basis(basis, 2.5)
    explicit = pack_from_basis(basis, 2.5, s_ref=None, g_table=None)
    assert torch.equal(default.bits, explicit.bits)
    p1 = fit_spectral_pack(M, Wh, Wh_inv, 2.5)
    p2 = fit_spectral_pack(M, Wh, Wh_inv, 2.5, s_ref=None, g_table=None)
    assert torch.equal(p1.bits, p2.bits)


def test_pack_from_basis_s_ref_hand_case():
    """Hand-computable charge-aware allocation (math review #2 mechanism).
    lam=(256,16,1,1e-8), tiers (0,2,4), group=16, C=4, s_ref=64 =>
    s = 16/16 + 16*4/64 = 2.0. At budget 3.0 (mean TOTAL charge):
      (4,4,0,0): charge (6+6)/4 = 3.0, D ~ 2.06  <- Lagrangian pick
      (4,4,2,0): charge 4.0 -- infeasible
    Plain at budget 3.0 (mean payload bits) opens THREE directions
    (4,4,4,0). Charge-aware closes the marginal direction: c_used 2 vs 3 --
    the 0<->2 boundary movement the math doc predicts."""
    from bmx.cache.spectral import SpectralBasis, pack_from_basis

    lam64 = torch.tensor([256.0, 16.0, 1.0, 1e-8], dtype=torch.float64)
    eye = torch.eye(4, dtype=torch.float32)
    basis = SpectralBasis(enc=eye, dec=eye.clone(), lam=lam64.float(), lam64=lam64)
    ca = pack_from_basis(basis, 3.0, tiers=(0, 2, 4), group=16, s_ref=64)
    plain = pack_from_basis(basis, 3.0, tiers=(0, 2, 4), group=16)
    assert torch.equal(ca.bits, torch.tensor([4, 4, 0, 0], dtype=torch.int64))
    assert torch.equal(plain.bits, torch.tensor([4, 4, 4, 0], dtype=torch.int64))
    assert ca.c_used == 2 and plain.c_used == 3


def test_s_ref_c_used_monotone():
    """Diagnostic invariant the A-gate pre-registers: c_used decreases as the
    deployment context shortens (bigger per-direction charge)."""
    from bmx.cache.spectral import fit_spectral_basis, pack_from_basis

    M, _ = _spiked_keys(S=512, C=64, seed=6)
    Wh, Wh_inv = identity_whitener(64)
    basis = fit_spectral_basis(M, Wh, Wh_inv)
    c_none = pack_from_basis(basis, 2.5, group=16).c_used
    c_8k = pack_from_basis(basis, 2.5, group=16, s_ref=8192).c_used
    c_1k = pack_from_basis(basis, 2.5, group=16, s_ref=1024).c_used
    assert c_1k <= c_8k <= c_none
    assert c_1k < c_none, "fixture must actually exercise the charge"
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_spectral.py -q -k "s_ref"`
Expected: FAIL — `TypeError: pack_from_basis() got an unexpected keyword argument 's_ref'`.

- [ ] **Step 3: Implement** — in `src/bmx/cache/spectral.py`, change `pack_from_basis`'s signature and allocation call:

```python
def pack_from_basis(
    basis: SpectralBasis,
    budget: float,
    *,
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8),
    group: int = 64,
    lam_alloc: torch.Tensor | None = None,
    s_ref: int | None = None,
    g_table: tuple[float, ...] | None = None,
) -> SpectralPack:
```

Append to its docstring:

```
    s_ref (charge-aware allocation, math review 2026-07-24 #2): when set, the
    allocator prices every used direction's TRUE storage cost under the
    reported accounting -- per-direction fixed charge
    s = 16/group + 16*C/s_ref (fp16 decoder column + group-scale share) --
    via the exact Lagrangian enumeration
    b_i = argmin_{b in T} lam_i*g(b) + kappa_L*(b + s*1[b>0]).
    budget then bounds the mean per-direction TOTAL charge
    (1/C)*sum_i (b_i + s*1[b_i>0]) = payload-v2 bpe + 16*c_used/s_ref
    = skeptic-v2 bpe@s_ref - tier_bits(tiers, s_ref). This is an ALLOCATION
    change only: the bpe accounting expressions stay frozen; c_used simply
    becomes smaller. g_table (finding #4) swaps 4^{-b} for measured per-tier
    ratios (Lagrangian selection, fixed charge 0 unless s_ref is also set).
    Defaults (None, None) reproduce the prior behavior bit-exactly.
```

and replace the single line `bits = allocate_bits_from_variance(alloc_input, budget, tiers)` with:

```python
    if s_ref is not None or g_table is not None:
        fixed = 0.0
        if s_ref is not None:
            assert s_ref > 0, f"s_ref must be positive; got {s_ref}"
            C = basis.enc.shape[0]
            # math review #2: s = 16/group + dec_bits*C/S_ref, dec_bits=16
            # (fp16 decoder; the A-gate never invokes the int8 lever).
            fixed = scale_bits(group) + 16.0 * C / float(s_ref)
        bits = allocate_bits_from_variance(
            alloc_input,
            budget,
            tiers,
            selection="lagrange",
            g_table=g_table,
            fixed_charge=fixed,
        )
    else:
        bits = allocate_bits_from_variance(alloc_input, budget, tiers)
```

Then extend `fit_spectral_pack` to accept and pass through the same two kwargs:

```python
def fit_spectral_pack(
    M_fit: torch.Tensor,
    Wh: torch.Tensor,
    Wh_inv: torch.Tensor,
    budget: float,
    *,
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8),
    group: int = 64,
    s_ref: int | None = None,
    g_table: tuple[float, ...] | None = None,
) -> SpectralPack:
    """Fit basis + allocation on M_fit (the calibration matrix). fp64 internally."""
    return pack_from_basis(
        fit_spectral_basis(M_fit, Wh, Wh_inv),
        budget,
        tiers=tiers,
        group=group,
        s_ref=s_ref,
        g_table=g_table,
    )
```

- [ ] **Step 4: Run the tests**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_spectral.py -q`
Expected: PASS (all, including the three new tests; the existing `test_pack_from_basis_lam_alloc_default_unchanged` still passes — the pin that the default path is untouched).

- [ ] **Step 5: Full battery**

Run: `cd /d/Projects/bmx && uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: `540 passed, 17 skipped, 1 xfailed`, ruff clean.

- [ ] **Step 6: Stage and propose the commit (STOP for user approval)**

```bash
git add src/bmx/cache/spectral.py tests/test_spectral.py
```

Proposed message: `feat(spectral): charge-aware s_ref + measured-g_table params on the pack fitting path (default-inert; allocation-only — accounting expressions frozen)`

---

### Task 3: The A-gate experiment — `experiments/k4_charge_alloc.py`

**Files:**
- Create: `experiments/k4_charge_alloc.py`
- Test: `tests/test_k4_experiments.py` (append)

**Interfaces:**
- Consumes: `pack_from_basis(..., s_ref=, g_table=)` (Task 2); `corpus_fit_bases`, `setup_rope`, `load_layer_keys`, `_layer_ctx`, `_score_tail`, `_tq_layer_curve`, `_log_interp` from `experiments._k4_common`; `quantize_cache` (arm `"turboquant_mse"`) from `bmx.cache.codecs`; `skeptic_charge`, `spectral_quantize` from `bmx.cache.spectral`; `create_run`/`write_metrics`/`git_sha` from `bmx.artifacts`.
- Produces: `experiments.k4_charge_alloc.Config` and `main(cfg) -> run_dir`; artifacts `metrics.parquet` (pack score rows with columns `model, cache, layer, arm, budget, s_ref, C, bpe_model, c_used, rel_fro, logit, logit_rope`), `tq_curve.parquet` (same schema, arm `"turboquant_mse"`, `bpe_model` = tq bpe), `diagnostics.parquet` (`layer, arm, budget, s_ref, c_used, mean_bits, n_t{b}...`), `frontier.parquet` (`arm, budget, s_ref, s_eval, bpe, win`), `charge_alloc_verdict.json`. Task 5 reruns with `--g-table`; Task 7 reads the verdict.

**Pre-registered gate (transcribed verbatim from the spec, binding):** per (ca_budget, s_ref): at MATCHED skeptic-v2 bpe evaluated at S_ref, the charge-aware pack's heldout G1 win must be >= the plain pack's (quality not sacrificed) AND its skeptic-v2 bpe@S_ref at matched win must undercut the plain pack by >= 0.4 blended bits at S_ref=4096 (half the projection — the projection itself is the optimistic bound). Both budgets fail -> honest negative, recorded, allocator stays as-is. Blended = K-side bits / 2 (the duel convention; math review #2's "~3.24 bpe on K ~ 1.62 blended" arithmetic). At S_ref=16384 both quantities are reported, never gated. Diagnostics (reported, never gated): c_used vs S_ref (expect monotone decrease), per-tier direction counts (expect 0<->2 boundary movement), the bpe-vs-S frontier curve at `eval_s` (does optimizing for 4k hurt 64k?).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_k4_experiments.py`:

```python
def test_k4_charge_alloc_smoke(tmp_path):
    import json

    import pandas as pd

    from experiments.k4_charge_alloc import Config, main

    p1, p2 = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    scored = tmp_path / "s.safetensors"
    _tiny_cache(p1, seed=0)
    _tiny_cache(p2, seed=1)
    _tiny_cache(scored, seed=2)
    cfg = Config(
        corpus_cache_paths=(str(p1), str(p2)),
        cache_paths=(str(scored),),
        model_label="tiny",
        plain_budgets=(1.5, 2.0, 2.5, 3.0),
        ca_budgets=(2.0,),
        s_refs=(256, 1024),  # tiny C=16: s = 16/16 + 16*16/256 = 2.0 at 256
        eval_s=(256, 1024, 4096),
        group=16,
        out_root=str(tmp_path / "results"),
    )
    run_dir = main(cfg)
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert set(df.arm) == {"plain", "charge_aware"}
    assert set(df[df.arm == "charge_aware"].s_ref) == {256, 1024}
    assert (df[df.arm == "plain"].s_ref == -1).all()  # sentinel, like bits==-1

    diag = pd.read_parquet(run_dir / "diagnostics.parquet")
    assert {"c_used", "mean_bits"} <= set(diag.columns)
    assert any(c.startswith("n_t") for c in diag.columns)  # tier histogram

    fr = pd.read_parquet(run_dir / "frontier.parquet")
    assert set(fr.s_eval) == {256, 1024, 4096}

    v = json.loads((run_dir / "charge_alloc_verdict.json").read_text())
    assert "a_gate_pass" in v and "honest_negative" in v and "rule" in v
    e = v["per_point"]["b2_s256"]
    for key in (
        "win_ca",
        "win_plain_at_matched_bpe",
        "win_not_worse",
        "bpe_ca",
        "bpe_plain_at_matched_win",
        "bits_saved_k_side",
        "bits_saved_blended",
    ):
        assert key in e, key
    # c_used diagnostic: charge-aware at the harshest s_ref uses no more
    # directions than plain at the same budget (per-layer means).
    cu_ca = diag[(diag.arm == "charge_aware") & (diag.s_ref == 256)].c_used.mean()
    cu_pl = diag[(diag.arm == "plain") & (diag.budget == 2.0)].c_used.mean()
    assert cu_ca <= cu_pl


def test_k4_charge_alloc_verdict_arithmetic(tmp_path):
    """Belt-and-braces regression pin (idiom of test_k4_charge_curve_smoke):
    recompute one verdict entry's bpe_ca from the metrics rows + skeptic_charge
    and assert the JSON carries exactly that number; bits_saved_blended must
    be exactly half of bits_saved_k_side."""
    import json

    import pandas as pd

    from bmx.cache.spectral import skeptic_charge
    from experiments.k4_charge_alloc import Config, main

    p1 = tmp_path / "a.safetensors"
    scored = tmp_path / "s.safetensors"
    _tiny_cache(p1, seed=0)
    _tiny_cache(scored, seed=2)
    cfg = Config(
        corpus_cache_paths=(str(p1),),
        cache_paths=(str(scored),),
        model_label="tiny",
        plain_budgets=(1.5, 2.0, 2.5, 3.0),
        ca_budgets=(2.0,),
        s_refs=(256,),
        eval_s=(256,),
        group=16,
        out_root=str(tmp_path / "results"),
    )
    run_dir = main(cfg)
    df = pd.read_parquet(run_dir / "metrics.parquet")
    v = json.loads((run_dir / "charge_alloc_verdict.json").read_text())
    e = v["per_point"]["b2_s256"]
    sub = df[(df.arm == "charge_aware") & (df.s_ref == 256)]
    expected_bpe = (
        sub.bpe_model
        + sub.apply(
            lambda r: skeptic_charge(
                int(r.C), 256, tuple(cfg.tiers), c_used=float(r.c_used)
            ),
            axis=1,
        )
    ).mean()
    assert abs(e["bpe_ca"] - float(expected_bpe)) < 1e-9
    assert abs(e["bits_saved_blended"] - 0.5 * e["bits_saved_k_side"]) < 1e-12
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_k4_experiments.py -q -k "charge_alloc"`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiments.k4_charge_alloc'`.

- [ ] **Step 3: Implement** — create `experiments/k4_charge_alloc.py` with exactly this content:

```python
"""K4 A-gate: charge-aware (deployment-context-aware) allocation vs the plain
reverse-waterfill — math-review finding #2, pre-registered kill-or-confirm.

Fits plain packs (a budget grid spanning LOW total-charge values, so the plain
frontier can be interpolated at the charge-aware points) and charge-aware
packs (ca_budgets x s_refs) from the SAME corpus bases, scores everything on
heldout caches with the G1 instrument (tail-region distortion vs the
per-(cache, layer) turboquant_mse curve), and emits the pre-registered
verdict. Gate (binding, evaluated at S_ref=4096 points only):

  win_not_worse:  heldout G1 win of the charge-aware pack >= the plain
                  frontier's win interpolated at the SAME skeptic-v2
                  bpe@S_ref (quality not sacrificed), AND
  bits_saved_blended >= 0.4: (plain bpe@S_ref interpolated at the
                  charge-aware win) - (charge-aware bpe@S_ref), halved
                  (K-side only; blended = K/2, the duel convention).

a_gate_pass = the gate passes at ANY ca_budget's S_ref=4096 point; both
budgets fail => honest_negative (recorded; allocator stays as-is). At
S_ref=16384 both quantities are reported, never gated. Diagnostics (reported,
never gated): c_used vs s_ref, per-tier direction counts (the 0<->2 boundary
movement prediction), and every pack's (bpe, win) at each eval_s — the
bpe-vs-S frontier question (does optimizing for 4k hurt 64k?).

Accounting discipline: allocation-only change. payload/skeptic expressions
are the shipped ones (spectral_payload_bpe inside spectral_quantize +
skeptic_charge); c_used simply becomes smaller.
"""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics
from bmx.cache.codecs import quantize_cache
from bmx.cache.spectral import pack_from_basis, skeptic_charge, spectral_quantize
from experiments._k4_common import (
    _layer_ctx,
    _log_interp,
    _score_tail,
    _tq_layer_curve,
    corpus_fit_bases,
    load_layer_keys,
    setup_rope,
)


@dataclasses.dataclass
class Config:
    corpus_cache_paths: tuple[str, ...]
    cache_paths: tuple[str, ...]  # scored (heldout) caches
    model_label: str = ""
    model_name: str = ""  # HF repo id for RoPE; empty => no-RoPE tables (gpt2)
    # Plain grid spans low TOTAL-charge values so the plain frontier brackets
    # the charge-aware points (a CA pack at budget 2.5 sits at ~2.5 total
    # charge; a plain b2.5 pack sits at ~5.9 bpe@4k).
    plain_budgets: tuple[float, ...] = (1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0)
    ca_budgets: tuple[float, ...] = (2.2, 2.5)
    s_refs: tuple[int, ...] = (4096, 16384)
    eval_s: tuple[int, ...] = (4096, 8192, 16384, 32768, 65536)
    tq_bits: tuple[int, ...] = (2, 3, 4)
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8)
    group: int = 64
    ridge: float = 1e-3
    position_stride: int = 8
    g_table: tuple[float, ...] = ()  # empty = None (4^-b); else measured, tier-aligned
    seed: int = 0
    out_root: str = ""

    # Pre-registered gate constants (spec 2026-07-24 §A) — not CLI-tunable in
    # spirit; kept here so the verdict JSON records them.
    gate_s_ref: int = 4096
    gate_blended_bits: float = 0.4


def _arm_list(cfg: Config) -> list[tuple[str, float, int]]:
    """(arm, budget, s_ref) triples; s_ref == -1 is the plain sentinel."""
    arms = [("plain", b, -1) for b in cfg.plain_budgets]
    arms += [("charge_aware", b, s) for b in cfg.ca_budgets for s in cfg.s_refs]
    return arms


def main(cfg: Config):
    assert cfg.corpus_cache_paths and cfg.cache_paths
    gt = tuple(cfg.g_table) if cfg.g_table else None

    run = (
        create_run("k4_charge_alloc", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_charge_alloc", cfg)
    )

    # ---- corpus fit (one basis per layer, w_source="corpus") --------------
    per_cache = [load_layer_keys(p) for p in cfg.corpus_cache_paths]
    layers = sorted(per_cache[0].keys())
    for lk in per_cache[1:]:
        assert sorted(lk.keys()) == layers, "corpus caches disagree on layer set"
    rope_ready = False
    get_cos_sins = []
    for lk in per_cache:
        ready, gcs = setup_rope(cfg.model_name, lk, layers)
        rope_ready = rope_ready or ready
        get_cos_sins.append(gcs)
    fit = corpus_fit_bases(
        per_cache,
        get_cos_sins,
        rope_ready,
        layers,
        w_source="corpus",
        ridge=cfg.ridge,
        position_stride=cfg.position_stride,
    )

    # ---- packs + diagnostics ---------------------------------------------
    arms = _arm_list(cfg)
    packs: dict[tuple[str, float, int], dict[int, object]] = {}
    diag_rows: list[dict] = []
    for arm, budget, s_ref in arms:
        by_layer = {}
        for layer_i in layers:
            pack = pack_from_basis(
                fit.bases[layer_i],
                budget,
                tiers=cfg.tiers,
                group=cfg.group,
                s_ref=None if s_ref < 0 else s_ref,
                g_table=gt,
            )
            by_layer[layer_i] = pack
            row = dict(
                layer=layer_i,
                arm=arm,
                budget=float(budget),
                s_ref=int(s_ref),
                c_used=int(pack.c_used),
                mean_bits=float(pack.bits.double().mean()),
            )
            for t in cfg.tiers:
                row[f"n_t{t}"] = int((pack.bits == t).sum())
            diag_rows.append(row)
        packs[(arm, budget, s_ref)] = by_layer

    # ---- scoring on heldout caches ---------------------------------------
    rows: list[dict] = []
    tq_rows: list[dict] = []
    any_rope = False
    for cache_path in cfg.cache_paths:
        cache_label = cache_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        layer_keys = load_layer_keys(cache_path)
        sc_layers = sorted(layer_keys.keys())
        c_ready, get_cos_sin = setup_rope(cfg.model_name, layer_keys, sc_layers)
        any_rope = any_rope or c_ready
        for layer_i in sc_layers:
            ctx = _layer_ctx(
                layer_keys[layer_i], rope_ready=c_ready, get_cos_sin=get_cos_sin
            )
            base = dict(model=cfg.model_label or "unknown", cache=cache_label)
            for b in cfg.tq_bits:
                M_hat, bpe = quantize_cache(
                    "turboquant_mse", ctx.M_pre, bits=b, seed=cfg.seed
                )
                rf, lg, lg_rope = _score_tail(
                    M_hat, ctx.h_kv, ctx.tail, ctx.K_post_true, ctx.Q_fp32,
                    ctx.cos_l, ctx.sin_l, c_ready, ctx.k_pre_t, ctx.M_pre,
                )
                tq_rows.append(
                    dict(base, layer=layer_i, arm="turboquant_mse", kind="k_pre",
                         budget=float("nan"), s_ref=-1, C=ctx.C, bpe_model=bpe,
                         c_used=float("nan"), rel_fro=rf, logit=lg,
                         logit_rope=lg_rope)
                )
            for (arm, budget, s_ref), by_layer in packs.items():
                if layer_i not in by_layer:
                    continue
                pack = by_layer[layer_i]
                M_hat, bpe_model = spectral_quantize(ctx.M_pre, pack)
                rf, lg, lg_rope = _score_tail(
                    M_hat, ctx.h_kv, ctx.tail, ctx.K_post_true, ctx.Q_fp32,
                    ctx.cos_l, ctx.sin_l, c_ready, ctx.k_pre_t, ctx.M_pre,
                )
                rows.append(
                    dict(base, layer=layer_i, arm=arm, kind="k_pre",
                         budget=float(budget), s_ref=int(s_ref), C=ctx.C,
                         bpe_model=bpe_model, c_used=float(pack.c_used),
                         rel_fro=rf, logit=lg, logit_rope=lg_rope)
                )
                print(
                    f"  cache={cache_label:16s} layer={layer_i:2d} arm={arm:12s} "
                    f"b={budget:g} s_ref={s_ref} bpe_model={bpe_model:.3f} "
                    f"c_used={pack.c_used}",
                    flush=True,
                )

    headline = "logit_rope" if any_rope else "logit"
    df = pd.DataFrame(rows)
    tq_df = pd.DataFrame(tq_rows)
    diag_df = pd.DataFrame(diag_rows)
    write_metrics(run, df)
    write_metrics(run, tq_df, name="tq_curve")
    write_metrics(run, diag_df, name="diagnostics")

    # ---- (bpe, win) point for every (arm, budget, s_ref) at every eval S --
    tq_curves = {
        cache: _tq_layer_curve(g.assign(kind="k_pre"), headline)
        for cache, g in tq_df.groupby("cache")
    }

    def point(arm: str, budget: float, s_ref: int, s_eval: int):
        sub = df[(df.arm == arm) & (df.budget == budget) & (df.s_ref == s_ref)]
        bpes, wins, extrap = [], [], False
        for _, r in sub.iterrows():
            pts = tq_curves.get(r.cache, {}).get(int(r.layer))
            if not pts:
                continue
            bpe = float(r.bpe_model) + skeptic_charge(
                int(r.C), s_eval, tuple(cfg.tiers), c_used=float(r.c_used)
            )
            tq_dist, ex = _log_interp(pts, bpe)
            extrap = extrap or ex
            wins.append(tq_dist / max(float(r[headline]), 1e-300))
            bpes.append(bpe)
        assert bpes, f"no scored rows for ({arm}, {budget}, {s_ref})"
        n = float(len(bpes))
        return sum(bpes) / n, sum(wins) / n, extrap

    frontier_rows = []
    for arm, budget, s_ref in arms:
        for s_eval in cfg.eval_s:
            bpe, win, ex = point(arm, budget, s_ref, s_eval)
            frontier_rows.append(
                dict(arm=arm, budget=float(budget), s_ref=int(s_ref),
                     s_eval=int(s_eval), bpe=bpe, win=win, extrapolated=ex)
            )
    write_metrics(run, pd.DataFrame(frontier_rows), name="frontier")

    # ---- pre-registered verdict ------------------------------------------
    verdict = _verdict(cfg, point, diag_df, headline)
    (run / "charge_alloc_verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\n" + "=" * 88)
    print("A-GATE VERDICT — charge-aware allocation (math review #2)")
    print("=" * 88)
    print(json.dumps(verdict, indent=2))
    print(f"-> {run}")
    return run


def _verdict(cfg: Config, point, diag_df: pd.DataFrame, headline: str) -> dict:
    per_point: dict[str, dict] = {}
    gate_passes: list[bool] = []
    for s_ref in cfg.s_refs:
        # Plain frontier AT this S_ref: (bpe, win) per plain budget.
        plain_pts = sorted(
            point("plain", b, -1, s_ref)[:2] for b in cfg.plain_budgets
        )
        win_by_bpe = plain_pts  # [(bpe, win)] sorted by bpe
        bpe_by_win = sorted((w, b) for b, w in plain_pts)  # [(win, bpe)]
        frontier_monotone = all(
            win_by_bpe[i][1] <= win_by_bpe[i + 1][1]
            for i in range(len(win_by_bpe) - 1)
        )
        for budget in cfg.ca_budgets:
            bpe_ca, win_ca, ex_ca = point("charge_aware", budget, s_ref, s_ref)
            win_plain_at_bpe, ex1 = _log_interp(win_by_bpe, bpe_ca)
            bpe_plain_at_win, ex2 = _log_interp(bpe_by_win, win_ca)
            bits_saved_k = bpe_plain_at_win - bpe_ca
            bits_saved_blended = 0.5 * bits_saved_k
            win_not_worse = bool(win_ca >= win_plain_at_bpe)
            entry = dict(
                s_ref=int(s_ref),
                budget=float(budget),
                bpe_ca=bpe_ca,
                win_ca=win_ca,
                win_plain_at_matched_bpe=win_plain_at_bpe,
                win_not_worse=win_not_worse,
                bpe_plain_at_matched_win=bpe_plain_at_win,
                bits_saved_k_side=bits_saved_k,
                bits_saved_blended=bits_saved_blended,
                extrapolated=bool(ex_ca or ex1 or ex2),
                plain_frontier_monotone=bool(frontier_monotone),
            )
            if s_ref == cfg.gate_s_ref:
                entry["gate_pass"] = bool(
                    win_not_worse and bits_saved_blended >= cfg.gate_blended_bits
                )
                gate_passes.append(entry["gate_pass"])
            per_point[f"b{budget:g}_s{s_ref}"] = entry

    # c_used monotonicity diagnostic (mean over layers), per ca budget.
    c_used_diag = {}
    for budget in cfg.ca_budgets:
        by_s = {
            str(s): float(
                diag_df[
                    (diag_df.arm == "charge_aware")
                    & (diag_df.budget == budget)
                    & (diag_df.s_ref == s)
                ].c_used.mean()
            )
            for s in cfg.s_refs
        }
        by_s["plain"] = float(
            diag_df[
                (diag_df.arm == "plain") & (diag_df.budget == budget)
            ].c_used.mean()
        ) if budget in cfg.plain_budgets else float("nan")
        vals = [by_s[str(s)] for s in sorted(cfg.s_refs)]
        c_used_diag[f"b{budget:g}"] = dict(
            by_s_ref=by_s,
            monotone_decreasing=bool(
                all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))
            ),
        )

    a_gate_pass = bool(gate_passes) and any(gate_passes)
    return dict(
        headline_metric=headline,
        rule=(
            "Pre-registered (spec 2026-07-24 SA): per (budget, s_ref), at "
            "MATCHED skeptic-v2 bpe@S_ref the charge-aware heldout G1 win "
            "must be >= the plain pack's, AND skeptic-v2 bpe@S_ref at "
            "matched win must undercut plain by >= 0.4 blended bits at "
            "S_ref=4096 (blended = K-side/2). Both budgets fail => honest "
            "negative; allocator stays as-is. S_ref=16384 reported, not "
            "gated."
        ),
        gate_s_ref=cfg.gate_s_ref,
        gate_blended_bits=cfg.gate_blended_bits,
        g_table=list(cfg.g_table) if cfg.g_table else None,
        per_point=per_point,
        c_used_diagnostic=c_used_diag,
        a_gate_pass=a_gate_pass,
        honest_negative=not a_gate_pass,
        git_sha=git_sha(),
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
```

Implementation notes for the engineer:
- `_tq_layer_curve` filters on `df.kind == "k_pre"` — the tq rows set `kind="k_pre"` so the shared helper works unmodified; pack rows also carry `kind="k_pre"` for schema uniformity (the `g.assign(kind="k_pre")` in `point`'s curve build is then a no-op kept for safety).
- The module deliberately imports neither `math` nor `torch` — everything tensor-side lives in the helpers; if a later edit adds a direct use, import then.
- The `bpe_by_win` interpolation uses `_log_interp` on (win, bpe) pairs — log-linear bpe in win; `plain_frontier_monotone=False` in the JSON flags when the plain frontier isn't win-monotone (interpolation then still runs on the win-sorted pairs but the entry must be read with that flag).
- gpt2 runs have `model_name=""` → `headline="logit"`; the RoPE branch is exercised by Task 4's experiment, not this one.

- [ ] **Step 4: Run the tests**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_k4_experiments.py -q -k "charge_alloc"`
Expected: PASS (2 tests).

- [ ] **Step 5: Full battery**

Run: `cd /d/Projects/bmx && uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: `542 passed, 17 skipped, 1 xfailed`, ruff clean.

- [ ] **Step 6: Stage and propose the commit (STOP for user approval)**

```bash
git add experiments/k4_charge_alloc.py tests/test_k4_experiments.py
```

Proposed message: `feat(exp): k4_charge_alloc — pre-registered A-gate for deployment-context-aware allocation (win-not-worse + >=0.4 blended bits @ S_ref=4096; c_used/tier-shift/bpe-vs-S diagnostics)`

---

### Task 4: `w_rope="rotated"` corrected query moment + the A/B experiment

**Files:**
- Modify: `src/bmx/cache/spectral.py` (`query_position_moment`, ~line 71)
- Modify: `experiments/_k4_common.py` (`corpus_query_moment`, ~line 121; `corpus_fit_bases`, ~line 157)
- Create: `experiments/k4_w_rope_ab.py`
- Test: `tests/test_spectral.py` (moment-level tests), `tests/test_k4_experiments.py` (experiment smoke)

**Interfaces:**
- Consumes: `_rotate_half` (already imported in spectral.py), `apply_rope` (test ground truth), `corpus_fit_bases`/`_layer_ctx`/`_score_tail`/`_tq_layer_curve`/`_log_interp`, `pack_from_basis`, `spectral_quantize`, `skeptic_charge`, `DEPLOY_S`, `_rank_overlap` from `experiments.k4_corpus_transfer`.
- Produces:
  - `query_position_moment(q, cos, sin, h_kv, *, position_stride=8, w_rope="frozen") -> torch.Tensor`
  - `corpus_query_moment(corpus_layer_keys, corpus_get_cos_sins, rope_ready, layer_i, h_kv, d, position_stride, w_rope="frozen")` (positional args unchanged; `w_rope` keyword-only is NOT possible since callers pass positionally — add it as a trailing default parameter)
  - `corpus_fit_bases(..., *, w_source, ridge, position_stride, w_rope="frozen") -> CorpusFit`
  - `experiments.k4_w_rope_ab.Config` / `main(cfg) -> run_dir`; artifacts `metrics.parquet`, `tq_curve.parquet`, `overlap.parquet`, `w_rope_verdict.json`.

**The math being transcribed (authoritative: math review #3(b)/(c)):** the true causal logit error is `(R_{t-s} q_t)^T e_s` with offset `m = t−s ≥ 0` — **forward** RoPE on the query at relative offsets, aggregate weight per offset triangular (`#pairs at offset m ∝ S − m`). The implemented instrument uses `(R_{−s} q)^T e_s` — opposite-sign offsets, uniform-strided weights. The corrected variant therefore makes exactly two changes inside `query_position_moment` ("one line each" per the doc): (1) forward rotation — `q_rot = q·cos_p + rotate_half(q)·(+sin_p)` (sign of `sin` flipped from the frozen path's `−sp`); (2) triangular weights — position `p` (read as offset `m = p`) contributes with weight `S − p`, normalized by `Σ_p (S − p)`. Per rotary plane the even terms survive either convention; the odd term `sinφcosφ·(JM − MJ)` flips sign — so on a no-RoPE model (sin ≡ 0) the two variants are mathematically identical (the null-control pin below).

**SUBSTRATE NOTE (conflict #1, needs user sign-off before Step 6's real run):** the spec names gpt2, but gpt2 has no RoPE — both variants produce identical W there and the 2% decision rule is vacuous. The A/B runs on **qwen3-0.6b** (`results/cache/qwen3-0.6b_2048_off{2048,4096}.safetensors` fit, `qwen3-0.6b_2048.safetensors` scored — the only local RoPE-model caches); gpt2 serves as the exact-null control via the unit test.

**Pre-registered readout (spec §B, measurement not pass/fail):** heldout G1 win ratio (rotated vs frozen) and per-rank subspace overlap between the two bases, per layer. Decision rule for the PAPER text: `|relative win delta| < 2%` at both budgets (2.2, 2.5) → claim scoped as "the frozen-rotation approximation is measured-negligible at small-RoPE-model scale" (Llama spot-check queued for the rental); `≥ 2%` → the paper uses the rotated form's numbers and the Llama refit rides the rental as a REQUIRED item. Either way the sign-flip footnote enters the methods section.

- [ ] **Step 1: Write the failing moment-level tests** — append to `tests/test_spectral.py`:

```python
def test_query_moment_w_rope_default_inert():
    """w_rope='frozen' default == bare call, bit-exact (the default-inert pin)."""
    g = torch.Generator().manual_seed(11)
    q = torch.randn(4, 16, 8, generator=g)
    S = 32
    theta = torch.linspace(0, 2.0, S).unsqueeze(1) * torch.ones(1, 8)
    cos, sin = theta.cos(), theta.sin()
    bare = query_position_moment(q, cos, sin, h_kv=2, position_stride=8)
    frozen = query_position_moment(
        q, cos, sin, h_kv=2, position_stride=8, w_rope="frozen"
    )
    assert torch.equal(bare, frozen)


def test_query_moment_rotated_matches_explicit_forward_rotation():
    """Value-pin for w_rope='rotated' (math review #3(b)): W must equal
    sum_p (S-p) * R_p (pooled qq^T) R_p^T / sum_p (S-p) with R_p the FORWARD
    rotation (columns apply_rope(e_i)) — no transpose, unlike the frozen
    path's ground-truth test. Duplicated-half cos/sin structure as in real
    RoPE tables."""
    from bmx.cache.rope import apply_rope

    g = torch.Generator().manual_seed(12)
    h, T, d, h_kv, S = 2, 16, 4, 1, 8
    q = torch.randn(h, T, d, generator=g)
    freqs = torch.linspace(0.5, 1.0, d // 2)
    theta = torch.linspace(0.3, 2.0, S).unsqueeze(1) * torch.cat(
        [freqs, freqs]
    ).unsqueeze(0)
    cos, sin = theta.cos(), theta.sin()

    stride = 2
    W = query_position_moment(
        q, cos, sin, h_kv, position_stride=stride, w_rope="rotated"
    )

    q_flat = q.double().reshape(-1, d)
    pooled = q_flat.mT @ q_flat / (h * T)
    positions = list(range(0, S, stride))
    W_expected = torch.zeros(d, d, dtype=torch.float64)
    total = 0.0
    for p in positions:
        basis = torch.eye(d).double().unsqueeze(1)
        Rp_cols = apply_rope(basis, cos[p : p + 1].double(), sin[p : p + 1].double())
        Rp = Rp_cols.squeeze(1).mT  # (d, d), column i = R_p e_i (FORWARD)
        wt = float(S - p)  # triangular: #pairs at offset p ~ S - p
        W_expected += wt * (Rp @ pooled @ Rp.mT)
        total += wt
    W_expected /= total
    assert torch.allclose(W[0], W_expected, atol=1e-10)


def test_query_moment_rotated_null_on_no_rope():
    """gpt2 null control: with sin == 0 the odd (sign-flipping) term vanishes
    and every position contributes the identical qq^T, so frozen == rotated
    up to fp summation order (the reason the A/B must run on a RoPE model)."""
    g = torch.Generator().manual_seed(13)
    q = torch.randn(4, 16, 8, generator=g)
    S = 64
    cos, sin = torch.ones(S, 8), torch.zeros(S, 8)
    frozen = query_position_moment(q, cos, sin, h_kv=2, position_stride=8)
    rotated = query_position_moment(
        q, cos, sin, h_kv=2, position_stride=8, w_rope="rotated"
    )
    assert torch.allclose(frozen, rotated, atol=1e-10)
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_spectral.py -q -k "w_rope"`
Expected: FAIL — `TypeError: query_position_moment() got an unexpected keyword argument 'w_rope'`.

- [ ] **Step 3: Implement the moment variant** — in `src/bmx/cache/spectral.py`, change `query_position_moment`'s signature to

```python
def query_position_moment(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    h_kv: int,
    *,
    position_stride: int = 8,
    w_rope: str = "frozen",
) -> torch.Tensor:
```

append to the docstring:

```
    w_rope="frozen" (default, bit-exact): the shipped instrument — inverse
    rotation R_p^T q at uniform-strided absolute positions (the query's own
    rotation frozen at zero). w_rope="rotated" (math review 2026-07-24 #3):
    the causal-attention-corrected moment — FORWARD rotation R_m q (sign of
    sin flipped) at the same strided positions read as relative offsets
    m = p, each weighted triangularly (#pairs at offset m ~ S - m). Even
    plane terms agree between the two up to the offset distribution; the odd
    sin*cos plane term flips sign — identical when sin == 0 (no-RoPE models).
```

and replace the loop body with (keep everything above `positions = ...` unchanged):

```python
    assert w_rope in ("frozen", "rotated"), f"unknown w_rope {w_rope!r}"
    positions = list(range(0, S, position_stride))
    if w_rope == "frozen":
        for p in positions:
            cp = cos[p].double().view(1, 1, d)
            sp = sin[p].double().view(1, 1, d)
            q_rot = q64 * cp + _rotate_half(q64) * (-sp)  # (h, T, d) = R_p^T q
            for j in range(h_kv):
                qj = q_rot[j * grp : (j + 1) * grp].reshape(-1, d)
                W[j] += qj.mT @ qj
        W /= len(positions) * grp * T
    else:
        total = 0.0
        for p in positions:
            cp = cos[p].double().view(1, 1, d)
            sp = sin[p].double().view(1, 1, d)
            q_rot = q64 * cp + _rotate_half(q64) * sp  # (h, T, d) = R_m q (forward)
            wt = float(S - p)  # triangular offset weight
            for j in range(h_kv):
                qj = q_rot[j * grp : (j + 1) * grp].reshape(-1, d)
                W[j] += wt * (qj.mT @ qj)
            total += wt
        W /= total * grp * T
    return W
```

(The frozen branch is the existing loop moved under `if` — byte-identical arithmetic, same iteration order.)

- [ ] **Step 4: Thread `w_rope` through `_k4_common`** — in `experiments/_k4_common.py`:

`corpus_query_moment` gains a trailing parameter `w_rope="frozen"` and passes it through:

```python
def corpus_query_moment(
    corpus_layer_keys,
    corpus_get_cos_sins,
    rope_ready,
    layer_i,
    h_kv,
    d,
    position_stride,
    w_rope="frozen",
):
```

with the call becoming:

```python
        W_sum += query_position_moment(
            c_q_t.float(), c_cos, c_sin, h_kv,
            position_stride=position_stride, w_rope=w_rope,
        )
```

`corpus_fit_bases` gains keyword-only `w_rope: str = "frozen"` after `position_stride` and passes it as the 8th argument to `corpus_query_moment`. All existing callers (`k4_fit_packs`, `k4_frontier`, `k4_corpus_transfer`) pass neither — default-inert; `test_k4_fit_packs_default_unchanged` and `test_corpus_fit_bases_matches_direct_fit` ARE the pins and must pass unmodified.

- [ ] **Step 5: Run the moment tests + the existing pins**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_spectral.py tests/test_k4_experiments.py -q`
Expected: PASS (including the three new tests and the untouched default pins).

- [ ] **Step 6: Write the failing experiment smoke test** — append to `tests/test_k4_experiments.py`:

```python
def test_k4_w_rope_ab_smoke(tmp_path):
    import json

    import pandas as pd

    from experiments.k4_w_rope_ab import Config, main

    p1, p2 = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    scored = tmp_path / "s.safetensors"
    _tiny_cache(p1, seed=0)
    _tiny_cache(p2, seed=1)
    _tiny_cache(scored, seed=2)
    cfg = Config(
        corpus_cache_paths=(str(p1), str(p2)),
        cache_paths=(str(scored),),
        model_label="tiny",
        budgets=(2.0, 2.5),
        group=16,
        overlap_ranks=(4, 8),
        out_root=str(tmp_path / "results"),
    )
    run_dir = main(cfg)
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert set(df.w_rope) == {"frozen", "rotated"}
    ov = pd.read_parquet(run_dir / "overlap.parquet")
    assert set(ov["rank"]) == {4, 8}
    assert ((ov.value >= -1e-9) & (ov.value <= 1 + 1e-9)).all()
    v = json.loads((run_dir / "w_rope_verdict.json").read_text())
    assert set(v["per_budget"]) == {"2", "2.5"}
    for e in v["per_budget"].values():
        assert {"win_frozen", "win_rotated", "rel_win_delta"} <= set(e)
    assert v["decision"] in ("scoped_negligible", "rotated_form_required")
    # tiny fixture has no RoPE (model_name="") => the null: delta ~ 0.
    for e in v["per_budget"].values():
        assert abs(e["rel_win_delta"]) < 1e-6
```

Run: `cd /d/Projects/bmx && uv run pytest tests/test_k4_experiments.py -q -k "w_rope_ab"`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiments.k4_w_rope_ab'`.

- [ ] **Step 7: Implement the experiment** — create `experiments/k4_w_rope_ab.py`:

```python
"""K4 B-readout: w_rope A/B — frozen (shipped) vs rotated (causal-corrected)
query moment, math review 2026-07-24 finding #3.

Fits corpus bases BOTH ways on the same caches, scores both on heldout caches
with the G1 instrument, and reports (measurement, not a pass/fail gate):
per-budget heldout G1 win ratio rotated/frozen, and per-(layer, rank)
subspace overlap between the two bases' dec columns.

Pre-registered decision rule (spec 2026-07-24 SB), recorded in the verdict:
|rel_win_delta| < 2% at BOTH budgets -> decision "scoped_negligible" (the
paper scopes the claim: frozen-rotation approximation measured-negligible at
this scale; Llama spot-check queued for the rental). Otherwise ->
"rotated_form_required" (the paper uses the rotated form's numbers; Llama
refit rides the rental as REQUIRED). Either way the sign-flip footnote enters
the methods section.

SUBSTRATE: must run on a RoPE model (qwen3-0.6b locally) — on a no-RoPE
model (gpt2, model_name="") the two variants are mathematically identical
and the readout is a null control.
"""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics
from bmx.cache.codecs import quantize_cache
from bmx.cache.spectral import pack_from_basis, skeptic_charge, spectral_quantize
from experiments._k4_common import (
    DEPLOY_S,
    _layer_ctx,
    _log_interp,
    _score_tail,
    _tq_layer_curve,
    corpus_fit_bases,
    load_layer_keys,
    setup_rope,
)
from experiments.k4_corpus_transfer import _rank_overlap

W_ROPE_VARIANTS = ("frozen", "rotated")


@dataclasses.dataclass
class Config:
    corpus_cache_paths: tuple[str, ...]
    cache_paths: tuple[str, ...]  # scored (heldout) caches
    model_label: str = ""
    model_name: str = ""  # HF repo id; MUST be a RoPE model for a live A/B
    budgets: tuple[float, ...] = (2.2, 2.5)
    tq_bits: tuple[int, ...] = (2, 3, 4)
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8)
    group: int = 64
    ridge: float = 1e-3
    position_stride: int = 8
    overlap_ranks: tuple[int, ...] = (8, 16, 32, 64)
    seed: int = 0
    out_root: str = ""


def main(cfg: Config):
    assert cfg.corpus_cache_paths and cfg.cache_paths

    run = (
        create_run("k4_w_rope_ab", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_w_rope_ab", cfg)
    )

    per_cache = [load_layer_keys(p) for p in cfg.corpus_cache_paths]
    layers = sorted(per_cache[0].keys())
    rope_ready = False
    get_cos_sins = []
    for lk in per_cache:
        ready, gcs = setup_rope(cfg.model_name, lk, layers)
        rope_ready = rope_ready or ready
        get_cos_sins.append(gcs)

    fits = {
        variant: corpus_fit_bases(
            per_cache,
            get_cos_sins,
            rope_ready,
            layers,
            w_source="corpus",
            ridge=cfg.ridge,
            position_stride=cfg.position_stride,
            w_rope=variant,
        )
        for variant in W_ROPE_VARIANTS
    }

    # ---- per-(layer, rank) subspace overlap between the two bases --------
    ov_rows = [
        dict(
            layer=layer_i,
            rank=r,
            value=_rank_overlap(
                fits["frozen"].bases[layer_i].dec,
                fits["rotated"].bases[layer_i].dec,
                r,
            ),
        )
        for layer_i in layers
        for r in cfg.overlap_ranks
    ]
    write_metrics(run, pd.DataFrame(ov_rows), name="overlap")

    # ---- score both variants on heldout caches ---------------------------
    rows: list[dict] = []
    tq_rows: list[dict] = []
    any_rope = False
    for cache_path in cfg.cache_paths:
        cache_label = cache_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        layer_keys = load_layer_keys(cache_path)
        sc_layers = sorted(layer_keys.keys())
        c_ready, get_cos_sin = setup_rope(cfg.model_name, layer_keys, sc_layers)
        any_rope = any_rope or c_ready
        for layer_i in sc_layers:
            ctx = _layer_ctx(
                layer_keys[layer_i], rope_ready=c_ready, get_cos_sin=get_cos_sin
            )
            base = dict(model=cfg.model_label or "unknown", cache=cache_label)
            for b in cfg.tq_bits:
                M_hat, bpe = quantize_cache(
                    "turboquant_mse", ctx.M_pre, bits=b, seed=cfg.seed
                )
                rf, lg, lg_rope = _score_tail(
                    M_hat, ctx.h_kv, ctx.tail, ctx.K_post_true, ctx.Q_fp32,
                    ctx.cos_l, ctx.sin_l, c_ready, ctx.k_pre_t, ctx.M_pre,
                )
                tq_rows.append(
                    dict(base, layer=layer_i, kind="k_pre", arm="turboquant_mse",
                         w_rope="", budget=float("nan"), bpe_model=bpe,
                         bpe_skeptic_deploy=bpe, rel_fro=rf, logit=lg,
                         logit_rope=lg_rope)
                )
            for variant in W_ROPE_VARIANTS:
                for budget in cfg.budgets:
                    pack = pack_from_basis(
                        fits[variant].bases[layer_i], budget,
                        tiers=cfg.tiers, group=cfg.group,
                    )
                    M_hat, bpe_model = spectral_quantize(ctx.M_pre, pack)
                    rf, lg, lg_rope = _score_tail(
                        M_hat, ctx.h_kv, ctx.tail, ctx.K_post_true, ctx.Q_fp32,
                        ctx.cos_l, ctx.sin_l, c_ready, ctx.k_pre_t, ctx.M_pre,
                    )
                    bpe_deploy = bpe_model + skeptic_charge(
                        ctx.C, DEPLOY_S, tuple(cfg.tiers), c_used=pack.c_used
                    )
                    rows.append(
                        dict(base, layer=layer_i, kind="k_pre", arm="spectral",
                             w_rope=variant, budget=float(budget),
                             bpe_model=bpe_model, bpe_skeptic_deploy=bpe_deploy,
                             rel_fro=rf, logit=lg, logit_rope=lg_rope)
                    )
                    print(
                        f"  cache={cache_label:16s} layer={layer_i:2d} "
                        f"w_rope={variant:8s} b={budget:g} "
                        f"bpe={bpe_model:.3f} logit={lg:.5f}",
                        flush=True,
                    )

    headline = "logit_rope" if any_rope else "logit"
    df = pd.DataFrame(rows)
    tq_df = pd.DataFrame(tq_rows)
    write_metrics(run, df)
    write_metrics(run, tq_df, name="tq_curve")

    # ---- verdict: win ratio per budget + the 2% decision rule -------------
    tq_curves = {
        cache: _tq_layer_curve(g, headline) for cache, g in tq_df.groupby("cache")
    }
    per_budget: dict[str, dict] = {}
    deltas: list[float] = []
    for budget in cfg.budgets:
        wins = {v: [] for v in W_ROPE_VARIANTS}
        for variant in W_ROPE_VARIANTS:
            sub = df[(df.w_rope == variant) & (df.budget == float(budget))]
            for _, r in sub.iterrows():
                pts = tq_curves.get(r.cache, {}).get(int(r.layer))
                if not pts:
                    continue
                tq_dist, _ = _log_interp(pts, float(r.bpe_skeptic_deploy))
                wins[variant].append(
                    tq_dist / max(float(r[headline]), 1e-300)
                )
        win_f = float(pd.Series(wins["frozen"]).mean())
        win_r = float(pd.Series(wins["rotated"]).mean())
        delta = win_r / win_f - 1.0
        deltas.append(delta)
        per_budget[f"{budget:g}"] = dict(
            win_frozen=win_f, win_rotated=win_r, rel_win_delta=delta,
            n_samples=len(wins["frozen"]),
        )
    scoped = all(abs(d) < 0.02 for d in deltas)
    ov_df = pd.DataFrame(ov_rows)
    verdict = dict(
        headline_metric=headline,
        rule=(
            "Pre-registered (spec 2026-07-24 SB): |rel_win_delta| < 2% at "
            "both budgets -> scoped_negligible (frozen-rotation approximation "
            "measured-negligible at this scale; Llama spot-check queued); "
            ">= 2% -> rotated_form_required (paper uses rotated numbers; "
            "Llama refit REQUIRED on the rental). Sign-flip footnote enters "
            "the methods section either way."
        ),
        per_budget=per_budget,
        decision="scoped_negligible" if scoped else "rotated_form_required",
        llama_refit_required=not scoped,
        overlap_mean_by_rank={
            str(r): float(ov_df[ov_df["rank"] == r].value.mean())
            for r in cfg.overlap_ranks
        },
        git_sha=git_sha(),
    )
    (run / "w_rope_verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\n" + "=" * 88)
    print("W_ROPE A/B VERDICT — frozen vs rotated query moment (math review #3)")
    print("=" * 88)
    print(json.dumps(verdict, indent=2))
    print(f"-> {run}")
    return run


if __name__ == "__main__":
    main(tyro.cli(Config))
```

Note: the module deliberately does not import `torch` (everything tensor-side lives in the helpers). `overlap_ranks` must satisfy `r <= C` of the substrate; the tiny test uses (4, 8) against C=16, the qwen3 run uses the default (8, 16, 32, 64) against C=1024.

- [ ] **Step 8: Run the tests**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_k4_experiments.py tests/test_spectral.py -q`
Expected: PASS (4 new tests total this task).

- [ ] **Step 9: Full battery**

Run: `cd /d/Projects/bmx && uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: `546 passed, 17 skipped, 1 xfailed`, ruff clean.

- [ ] **Step 10: Stage and propose the commit (STOP for user approval)**

```bash
git add src/bmx/cache/spectral.py experiments/_k4_common.py experiments/k4_w_rope_ab.py tests/test_spectral.py tests/test_k4_experiments.py
```

Proposed message: `feat(spectral): w_rope="rotated" causal-corrected query moment (forward RoPE + triangular offsets, default-inert) + k4_w_rope_ab A/B experiment with the pre-registered 2% decision rule`

---

### Task 5: The measured-ĝ table — `experiments/k4_g_table.py` + the A-gate rerun hook

**Files:**
- Create: `experiments/k4_g_table.py`
- Test: `tests/test_k4_experiments.py` (append)

**Interfaces:**
- Consumes: `load_packs(pack_path, budget)` (enc is budget-independent — any stored budget works), `load_layer_keys`, `to_matrix` from `bmx.cache.collect`, `rtn_quantize` from `bmx.quant.rtn`, `_tier_g` from `bmx.cache.codecs` (validation), `create_run`/`write_metrics`/`git_sha`.
- Produces: `experiments.k4_g_table.Config` / `main(cfg) -> run_dir`; artifacts `metrics.parquet` (per (layer, tier, direction-pooled) ratio rows + spread), `g_table.json` — `{"tiers": [...], "g_table": [...], "n_rows": ..., "spread_p10_p90_by_tier": {...}}`. The `g_table` values feed `k4_charge_alloc --g-table ...` (Task 3's Config field, Task 2's `pack_from_basis` hook).

**The measurement being transcribed (authoritative: math review #4):** tabulate ĝ once, measured on calibration codes, `mse_scale=True`, `group=64` — NOT taken from Gaussian tables. Procedure: per layer, `Y = M_fit @ enc` (calibration rows in the eigenbasis, fp32 like the codec path); for each tier `b > 0`: `Y_hat = rtn_quantize(Y.mT, b, group, mse_scale=True).mT`; per-direction relative distortion `r_i(b) = mean((Y−Y_hat)_i^2) / mean(Y_i^2)` over directions with non-negligible energy (`mean(Y_i^2) > 1e-12 · max_i`); `ĝ(b)` = mean of `r_i(b)` pooled over kept directions and layers. `ĝ(0) = 1.0` exact (finding #4: a dropped direction's error is the coordinate itself — no quantizer model at the drop boundary). The per-direction p10/p90 spread per tier is the shared-shape audit (finding #4's fragile leg) — reported, never gated. The resulting table must pass `_tier_g`'s grid-convexity validation (the doc's measured values do).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_k4_experiments.py`:

```python
def test_k4_g_table_smoke(tmp_path):
    import json

    import pandas as pd

    from bmx.cache.codecs import _tier_g
    from experiments.k4_g_table import Config, main

    fit = tmp_path / "f.safetensors"
    _tiny_cache(fit, seed=0)
    packs_path = tmp_path / "packs.safetensors"
    _fit_tiny_pack_file(fit, packs_path, budgets=(2.5,), group=16)
    run_dir = main(
        Config(
            pack_path=str(packs_path),
            corpus_cache_paths=(str(fit),),
            model_label="tiny",
            group=16,
            out_root=str(tmp_path / "results"),
        )
    )
    g = json.loads((run_dir / "g_table.json").read_text())
    tiers = tuple(g["tiers"])
    table = tuple(g["g_table"])
    assert tiers == (0, 2, 3, 4, 5, 6, 8)
    assert table[0] == 1.0  # g(0) exact
    assert all(a > b for a, b in zip(table, table[1:]))  # strictly decreasing
    # The table must be directly consumable by the allocator's validator.
    tiers_t = torch.tensor([float(t) for t in tiers], dtype=torch.float64)
    _tier_g(tiers_t, table)  # raises if not grid-convex
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert {"layer", "tier", "g_hat", "p10", "p90", "n_dirs"} <= set(df.columns)


def test_k4_g_table_deterministic(tmp_path):
    import json

    from experiments.k4_g_table import Config, main

    fit = tmp_path / "f.safetensors"
    _tiny_cache(fit, seed=0)
    packs_path = tmp_path / "packs.safetensors"
    _fit_tiny_pack_file(fit, packs_path, budgets=(2.5,), group=16)
    cfg = dict(
        pack_path=str(packs_path),
        corpus_cache_paths=(str(fit),),
        model_label="tiny",
        group=16,
    )
    r1 = main(Config(**cfg, out_root=str(tmp_path / "r1")))
    r2 = main(Config(**cfg, out_root=str(tmp_path / "r2")))
    g1 = json.loads((r1 / "g_table.json").read_text())["g_table"]
    g2 = json.loads((r2 / "g_table.json").read_text())["g_table"]
    assert g1 == g2
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_k4_experiments.py -q -k "g_table"`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiments.k4_g_table'`.

- [ ] **Step 3: Implement** — create `experiments/k4_g_table.py`:

```python
"""K4 measured-g table (math review 2026-07-24 finding #4): ONE measurement of
the per-tier RTN distortion ratios g_hat(b) on calibration codes, replacing
the 4^{-b} model in the Lagrangian allocator (reported beside the A-gate,
never gated).

Procedure (transcribed from the review): per layer, project the calibration
rows into the shipped eigenbasis (Y = M_fit @ enc, fp32 — the codec path's
own arithmetic); groupwise-RTN every column at each tier (mse_scale=True,
the codec's step policy); per-direction relative distortion
mean((Y-Y_hat)_i^2)/mean(Y_i^2); g_hat(b) = pooled mean over kept directions
(energy > 1e-12 * max) and layers. g_hat(0) = 1 EXACT (a dropped direction's
error is its coordinate). The per-direction p10/p90 spread per tier is the
shared-shape audit (finding #4's fragile leg) — reported only.

The emitted table plugs into `k4_charge_alloc --g-table ...` (and any
`pack_from_basis(g_table=...)` call); `_tier_g` validates grid-convexity at
consumption time — this script also validates at emission time and fails
loudly if the measurement violates the optimality lemma's conditions.
"""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import torch
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics
from bmx.cache.codecs import _tier_g
from bmx.cache.collect import to_matrix
from bmx.cache.spectral import load_packs
from bmx.quant.rtn import rtn_quantize
from experiments._k4_common import load_layer_keys


@dataclasses.dataclass
class Config:
    pack_path: str  # shipped pack file — enc per layer (budget-independent)
    corpus_cache_paths: tuple[str, ...]  # calibration rows (the fit slices)
    model_label: str = ""
    enc_budget: float = 2.5  # any budget stored in the pack file; enc is shared
    tiers: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 8)
    group: int = 64
    energy_floor: float = 1e-12
    out_root: str = ""


def main(cfg: Config):
    assert cfg.corpus_cache_paths
    run = (
        create_run("k4_g_table", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_g_table", cfg)
    )
    packs = load_packs(cfg.pack_path, cfg.enc_budget)
    per_cache = [load_layer_keys(p) for p in cfg.corpus_cache_paths]
    layers = sorted(per_cache[0].keys())

    rows: list[dict] = []
    pooled: dict[int, list[torch.Tensor]] = {t: [] for t in cfg.tiers if t > 0}
    n_rows_total = 0
    for layer_i in layers:
        M_fit = torch.cat(
            [to_matrix(lk[layer_i]["k_pre"]) for lk in per_cache], dim=0
        )
        n_rows_total = max(n_rows_total, M_fit.shape[0])
        S = M_fit.shape[0]
        S_use = (S // cfg.group) * cfg.group  # rtn groups need divisibility
        assert S_use > 0, f"layer {layer_i}: too few rows for group={cfg.group}"
        Y = (M_fit[:S_use] @ packs[layer_i].enc).double()
        energy = (Y**2).mean(dim=0)
        keep = energy > cfg.energy_floor * float(energy.max())
        for t in cfg.tiers:
            if t == 0:
                continue
            Y_hat = rtn_quantize(
                Y[:, keep].float().mT, t, cfg.group, mse_scale=True
            ).mT.double()
            r = ((Y[:, keep] - Y_hat) ** 2).mean(dim=0) / energy[keep]
            pooled[t].append(r)
            rows.append(
                dict(
                    model=cfg.model_label or "unknown",
                    layer=layer_i,
                    tier=t,
                    g_hat=float(r.mean()),
                    p10=float(r.quantile(0.10)),
                    p90=float(r.quantile(0.90)),
                    n_dirs=int(keep.sum()),
                )
            )

    table = []
    spread = {}
    for t in cfg.tiers:
        if t == 0:
            table.append(1.0)  # exact — no quantizer model at the drop boundary
            continue
        all_r = torch.cat(pooled[t])
        table.append(float(all_r.mean()))
        spread[str(t)] = [float(all_r.quantile(0.10)), float(all_r.quantile(0.90))]

    # Fail loudly if the measurement violates the lemma's conditions.
    tiers_t = torch.tensor([float(t) for t in cfg.tiers], dtype=torch.float64)
    _tier_g(tiers_t, tuple(table))

    out = dict(
        tiers=list(cfg.tiers),
        g_table=table,
        n_rows=n_rows_total,
        spread_p10_p90_by_tier=spread,
        git_sha=git_sha(),
    )
    (run / "g_table.json").write_text(json.dumps(out, indent=2))
    write_metrics(run, pd.DataFrame(rows))
    print(json.dumps(out, indent=2))
    print(f"-> {run}")
    return run


if __name__ == "__main__":
    main(tyro.cli(Config))
```

- [ ] **Step 4: Run the tests**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_k4_experiments.py -q -k "g_table"`
Expected: PASS (2 tests).

- [ ] **Step 5: Full battery**

Run: `cd /d/Projects/bmx && uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: `548 passed, 17 skipped, 1 xfailed`, ruff clean.

- [ ] **Step 6: Stage and propose the commit (STOP for user approval)**

```bash
git add experiments/k4_g_table.py tests/test_k4_experiments.py
```

Proposed message: `feat(exp): k4_g_table — measured per-tier RTN distortion ratios on calibration codes (mse_scale, group=64; grid-convexity-validated; feeds k4_charge_alloc --g-table, reported beside the gate)`

---

### Task 6: `int8_decoder_certificate` + analytic identity test + `experiments/k4_int8_certificate.py`

**Files:**
- Modify: `src/bmx/cache/spectral.py` (insert directly after `int8_decoder_roundtrip`, ~line 447)
- Create: `experiments/k4_int8_certificate.py`
- Test: `tests/test_spectral.py` (identity + invariance), `tests/test_k4_experiments.py` (experiment smoke)

**Interfaces:**
- Consumes: `int8_decoder_roundtrip(dec, bits_pc)`, `SpectralPack`, `load_packs`.
- Produces:
  - `int8_decoder_certificate(pack: SpectralPack) -> dict[str, float]` with keys `added`, `payload`, `noise_to_signal`, `implied_rel_degradation`.
  - `experiments.k4_int8_certificate.Config` / `main(cfg) -> run_dir`; artifacts `metrics.parquet` (per (budget, layer) certificate rows), `certificate_verdict.json`.

**The math being transcribed (authoritative: math review #9, closed-formed per conflict note #2):**
- `Δdec = int8_decoder_roundtrip(pack.dec, pack.bits) − pack.dec` is **deterministic per pack**; the added reconstruction error on a row with code vector ŷ is `Δdec ŷ` — not a random variable.
- Weighted added second moment: `‖W̃^{1/2} Δdec ŷ‖² = ŷᵀ Δdecᵀ W̃ Δdec ŷ = ‖encᵀ Δdec ŷ‖²`, using `W̃ = enc encᵀ` exactly (`enc = W̃^{1/2}E`, E orthogonal).
- Taking the expectation with code second moment `diag(lam)` (exact on the fit corpus: `encᵀ Σ_fit enc = diag(lam)` by the eigendecomposition):

      added = Σ_i lam_i · ‖encᵀ Δdec[:, i]‖²        (fp64; the certificate closed form)

- Payload reference (same form as `_distortion_curves`/`_proxy_distortion`): `payload = Σ_i lam_i · 4^{−b_i}` — dropped directions contribute `lam_i` in full (`4^0 = 1`).
- `noise_to_signal = added/payload` (the review's expected scale: per-column int8 noise-to-signal ≈ `crest²/(12·127²) ≈ 2·ln C/193548 ≈ 7e−5` vs payload `g(2.5) ≈ 0.05–0.1` — three orders of margin; fp16 scale rounding at `2^{−11}` relative is noise on noise).
- Mapping onto the VM gate's axis (`rel_degradation_int8 = 1 − win_int8/win_fp16 < 5%`, win ∝ 1/dist): `implied_rel_degradation = added/(payload + added)` — must sit far inside 0.05 for the replacement argument.
- Honest limits (into the docstring AND the results doc): models `E[ŷŷᵀ]` by `diag(lam)` (ignores the payload-error shift of the code moment and the payload×decoder cross-term, both `O(g(b))` relative on a ~7e−5 base); uses the pack's own fit-corpus moments (query-distribution interaction beyond the modeled second moment is not captured); task-level effects are the VM half of the ledger and are NOT certified.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_spectral.py`:

```python
def test_int8_certificate_analytic_identity():
    """Fit-cache-free identity: on synthetic rows whose code second moment is
    EXACTLY diag(lam), the closed form sum_i lam_i*||enc^T ddec[:,i]||^2 must
    equal the brute-force mean weighted row norm mean_s ||enc^T ddec y_s||^2.
    Non-identity diagonal whitener so enc != dec (the mutant-visible regime,
    same rationale as test_rank_overlap_pinned_to_dec_not_enc)."""
    import math as _math

    from bmx.cache.spectral import (
        fit_spectral_pack,
        int8_decoder_certificate,
        int8_decoder_roundtrip,
    )

    C, S = 16, 200
    scales = torch.linspace(0.2, 5.0, C, dtype=torch.float64)
    Wh, Wh_inv = torch.diag(scales), torch.diag(1.0 / scales)
    g = torch.Generator().manual_seed(0)
    M = torch.randn(S, C, generator=g)
    pack = fit_spectral_pack(M, Wh, Wh_inv, 2.5, tiers=(0, 2, 4), group=8)
    assert 0 < pack.c_used < C  # fixture must have both used and dropped dirs

    cert = int8_decoder_certificate(pack)
    for key in ("added", "payload", "noise_to_signal", "implied_rel_degradation"):
        assert key in cert and cert[key] >= 0.0

    # Brute force on rows with exact code moment diag(lam):
    lam = pack.lam.double()
    G = torch.randn(4 * C, C, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(G)  # (4C, C), Q^T Q = I
    Z = _math.sqrt(4 * C) * Q @ torch.diag(lam.clamp_min(0).sqrt())
    ddec = (
        int8_decoder_roundtrip(pack.dec, pack.bits).double() - pack.dec.double()
    )
    brute = float(((Z @ ddec.mT @ pack.enc.double()) ** 2).sum() / (4 * C))
    assert abs(cert["added"] - brute) < 1e-9 * max(brute, 1e-12)

    # Payload closed form pinned against the independent expression.
    payload = float((lam * torch.pow(4.0, -pack.bits.double())).sum())
    assert abs(cert["payload"] - payload) < 1e-12
    assert abs(
        cert["implied_rel_degradation"]
        - cert["added"] / (cert["payload"] + cert["added"])
    ) < 1e-15


def test_int8_certificate_dropped_columns_and_determinism():
    """Mutating dropped dec columns cannot change the certificate (the
    test_dropped_decoder_columns_never_read license extends to it), and the
    certificate is deterministic."""
    import dataclasses as _dc

    from bmx.cache.spectral import fit_spectral_pack, int8_decoder_certificate

    M, _ = _spiked_keys(S=256, C=64, seed=1)
    Wh, Wh_inv = identity_whitener(64)
    pack = fit_spectral_pack(M, Wh, Wh_inv, 1.5, group=16)
    assert pack.c_used < 64
    c1 = int8_decoder_certificate(pack)
    dec_mut = pack.dec.clone()
    dec_mut[:, pack.bits == 0] = torch.randn_like(dec_mut[:, pack.bits == 0])
    c2 = int8_decoder_certificate(_dc.replace(pack, dec=dec_mut))
    assert c1 == c2 == int8_decoder_certificate(pack)
```

and append to `tests/test_k4_experiments.py`:

```python
def test_k4_int8_certificate_smoke(tmp_path):
    import json

    import pandas as pd

    from experiments.k4_int8_certificate import Config, main

    fit = tmp_path / "f.safetensors"
    _tiny_cache(fit, seed=0)
    packs_path = tmp_path / "packs.safetensors"
    _fit_tiny_pack_file(fit, packs_path, budgets=(2.0, 2.5), group=16)
    run_dir = main(
        Config(
            pack_path=str(packs_path),
            budgets=(2.0, 2.5),
            model_label="tiny",
            out_root=str(tmp_path / "results"),
        )
    )
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert {"budget", "layer", "added", "payload", "noise_to_signal",
            "implied_rel_degradation"} <= set(df.columns)
    assert (df.implied_rel_degradation >= 0).all()
    v = json.loads((run_dir / "certificate_verdict.json").read_text())
    assert "max_implied_rel_degradation" in v
    assert v["vm_gate_line"] == 0.05
    assert v["user_review_required_before_vm_task8_release"] is True
    # margin_factor consistency pin
    assert abs(
        v["margin_factor"] - 0.05 / max(v["max_implied_rel_degradation"], 1e-300)
    ) < 1e-6 * v["margin_factor"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_spectral.py tests/test_k4_experiments.py -q -k "certificate"`
Expected: FAIL — `ImportError: cannot import name 'int8_decoder_certificate'` / `ModuleNotFoundError: No module named 'experiments.k4_int8_certificate'`.

- [ ] **Step 3: Implement the certificate** — in `src/bmx/cache/spectral.py`, insert directly after `int8_decoder_roundtrip`:

```python
def int8_decoder_certificate(pack: SpectralPack) -> dict[str, float]:
    """Exact offline certificate for the int8-decoder distortion (math review
    2026-07-24 #9): the roundtrip perturbation ddec = dec_int8 - dec is
    deterministic per pack, so the added weighted reconstruction distortion is
    a computable NUMBER, not a bound and not a VM measurement.

    Closed form (fp64; derivation): the added K-space error on a row with
    code vector y_hat is ddec @ y_hat; its weighted norm is
    ||W^{1/2} ddec y_hat||^2 = ||enc^T ddec y_hat||^2 (W = enc enc^T exactly,
    since enc = W^{1/2} E with E orthogonal). Taking the expectation with the
    code second moment diag(lam) — exact on the fit corpus, where
    enc^T Sigma_fit enc = diag(lam) by the eigendecomposition:

        added   = sum_i lam_i * ||enc^T ddec[:, i]||^2
        payload = sum_i lam_i * 4^{-bits_i}     (dropped dirs: 4^0 = 1)

    noise_to_signal = added/payload; implied_rel_degradation =
    added/(payload + added) — the same axis as the pre-registered VM gate
    rel_degradation_int8 < 5% (win is inverse distortion, so
    1 - win_int8/win_fp16 = added/(payload + added) under matched bpe).

    Honest limits (what this does NOT capture): E[y_hat y_hat^T] is modeled
    by diag(lam) — the payload-error shift of the code moment and the
    payload x decoder cross-term (both O(g(b)) relative on a ~7e-5 base) are
    not represented; query-distribution interaction beyond the modeled second
    moment is not represented; task-level effects are NOT certified (they
    remain the VM half of the ledger). pack.lam is the fp32-stored spectrum
    (1e-7 relative rounding — three orders below the certificate's margin).
    """
    ddec = int8_decoder_roundtrip(pack.dec, pack.bits).double() - pack.dec.double()
    proj = pack.enc.double().mT @ ddec  # (C, C): column i = enc^T ddec[:, i]
    lam = pack.lam.double().clamp_min(0.0)
    added = float((lam * (proj**2).sum(dim=0)).sum())
    payload = float((lam * torch.pow(4.0, -pack.bits.double())).sum())
    return dict(
        added=added,
        payload=payload,
        noise_to_signal=added / max(payload, 1e-300),
        implied_rel_degradation=added / max(payload + added, 1e-300),
    )
```

- [ ] **Step 4: Implement the experiment** — create `experiments/k4_int8_certificate.py`:

```python
"""K4 int8-decoder certificate table (math review 2026-07-24 finding #9): the
exact offline distortion certificate for every layer of an EXISTING pack file
(refits nothing, loads no caches, needs no GPU).

Per (budget, layer): int8_decoder_certificate(pack) -> added / payload /
noise_to_signal / implied_rel_degradation, plus the mapping onto the
pre-registered VM gate axis (rel_degradation_int8 < 5%). The verdict JSON
carries max_implied_rel_degradation, the margin factor to the 5% line, and
the binding review condition: THE USER REVIEWS THESE NUMBERS BEFORE VM TASK 8
IS RELEASED — the VM task stays queued until explicit release.
"""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics
from bmx.cache.spectral import int8_decoder_certificate, load_packs


@dataclasses.dataclass
class Config:
    pack_path: str
    budgets: tuple[float, ...] = (2.2, 2.5)
    model_label: str = ""
    out_root: str = ""


def main(cfg: Config):
    run = (
        create_run("k4_int8_certificate", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_int8_certificate", cfg)
    )
    rows: list[dict] = []
    for budget in cfg.budgets:
        packs = load_packs(cfg.pack_path, budget)
        for layer_i, pack in sorted(packs.items()):
            cert = int8_decoder_certificate(pack)
            rows.append(
                dict(
                    model=cfg.model_label or "unknown",
                    budget=float(budget),
                    layer=layer_i,
                    c_used=int(pack.c_used),
                    **cert,
                )
            )
            print(
                f"  b={budget:g} layer={layer_i:2d} "
                f"noise_to_signal={cert['noise_to_signal']:.3e} "
                f"implied_rel_degradation={cert['implied_rel_degradation']:.3e}",
                flush=True,
            )
    df = pd.DataFrame(rows)
    write_metrics(run, df)

    max_impl = float(df.implied_rel_degradation.max())
    verdict = dict(
        pack_path=cfg.pack_path,
        budgets=list(cfg.budgets),
        max_noise_to_signal=float(df.noise_to_signal.max()),
        max_implied_rel_degradation=max_impl,
        vm_gate_line=0.05,
        margin_factor=0.05 / max(max_impl, 1e-300),
        certificate_far_inside_gate=bool(max_impl < 0.005),  # 10x margin ask
        user_review_required_before_vm_task8_release=True,
        git_sha=git_sha(),
    )
    (run / "certificate_verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\n" + "=" * 88)
    print("INT8 DECODER CERTIFICATE — user reviews before VM Task 8 is released")
    print("=" * 88)
    print(json.dumps(verdict, indent=2))
    print(f"-> {run}")
    return run


if __name__ == "__main__":
    main(tyro.cli(Config))
```

- [ ] **Step 5: Run the tests**

Run: `cd /d/Projects/bmx && uv run pytest tests/test_spectral.py tests/test_k4_experiments.py -q -k "certificate"`
Expected: PASS (3 tests).

- [ ] **Step 6: Full battery**

Run: `cd /d/Projects/bmx && uv run ruff format . && uv run ruff check . && uv run pytest -q`
Expected: `551 passed, 17 skipped, 1 xfailed`, ruff clean.

- [ ] **Step 7: Stage and propose the commit (STOP for user approval)**

```bash
git add src/bmx/cache/spectral.py experiments/k4_int8_certificate.py tests/test_spectral.py tests/test_k4_experiments.py
```

Proposed message: `feat(spectral): int8_decoder_certificate — exact offline distortion certificate (closed form on pack lam/enc) + k4_int8_certificate table; user-review-gated before it can replace VM Task 8's measurement`

---

### Task 7: Run the real measurements + write `docs/2026-07-24-k4-math-actions-results.md`

**Files:**
- Create: `docs/2026-07-24-k4-math-actions-results.md`
- Read-only inputs: `results/cache/gpt2_1024_off{1024,2048,3072,4096}.safetensors`, `results/cache/gpt2_1024.safetensors`, `results/cache/k4_packs_gpt2.safetensors`, `results/cache/qwen3-0.6b_2048_off{2048,4096}.safetensors`, `results/cache/qwen3-0.6b_2048.safetensors`
- Run artifacts produced: one run dir each under `results/k4_charge_alloc/` (x2: plain + g_table rerun), `results/k4_g_table/`, `results/k4_w_rope_ab/`, `results/k4_int8_certificate/`

**Interfaces:**
- Consumes: every experiment from Tasks 3-6 (CLI form; tyro tuples are space-separated).
- Produces: the results doc Task 8 stages; the exact run-ids (printed by each `-> results/...` line) are recorded in the doc header — explicit run selection, never blind-newest.

- [ ] **Step 1: Get user sign-off on the Task-4 substrate (conflict #1).** Message the user: the w_rope A/B runs on qwen3-0.6b because gpt2 has no RoPE (the variants are provably identical there — the unit test `test_query_moment_rotated_null_on_no_rope` pins it). Do not run Step 4 until acknowledged.

- [ ] **Step 2: A-gate run (gpt2, defaults carry the pre-registered grid)**

```bash
cd /d/Projects/bmx
uv run python experiments/k4_charge_alloc.py \
  --corpus-cache-paths results/cache/gpt2_1024_off1024.safetensors results/cache/gpt2_1024_off2048.safetensors results/cache/gpt2_1024_off3072.safetensors results/cache/gpt2_1024_off4096.safetensors \
  --cache-paths results/cache/gpt2_1024.safetensors \
  --model-label gpt2
```

Expected: prints per-(cache, layer, arm) rows, then the A-GATE VERDICT JSON; run dir `results/k4_charge_alloc/<run-id>/`. Record the run-id. Sanity checks before proceeding: `c_used_diagnostic.*.monotone_decreasing` true (if false, that is itself a finding — record it, do not suppress); `per_point.*.extrapolated` — if true at the gate points, widen `--plain-budgets` downward (e.g. add 0.8) and rerun; note the rerun id.

- [ ] **Step 3: measured-ĝ table + A-gate rerun with it**

```bash
uv run python experiments/k4_g_table.py \
  --pack-path results/cache/k4_packs_gpt2.safetensors \
  --corpus-cache-paths results/cache/gpt2_1024_off1024.safetensors results/cache/gpt2_1024_off2048.safetensors results/cache/gpt2_1024_off3072.safetensors results/cache/gpt2_1024_off4096.safetensors \
  --model-label gpt2
```

Read the 7 values from `results/k4_g_table/<run-id>/g_table.json` and rerun the A-gate with them (substitute the actual numbers for `<g0> ... <g8>`; `<g0>` is 1.0 by construction):

```bash
uv run python experiments/k4_charge_alloc.py \
  --corpus-cache-paths results/cache/gpt2_1024_off1024.safetensors results/cache/gpt2_1024_off2048.safetensors results/cache/gpt2_1024_off3072.safetensors results/cache/gpt2_1024_off4096.safetensors \
  --cache-paths results/cache/gpt2_1024.safetensors \
  --model-label gpt2 \
  --g-table <g0> <g2> <g3> <g4> <g5> <g6> <g8>
```

The g_table rerun is REPORTED BESIDE the model-g gate, never gated (spec §A(ii)); the binding A-gate verdict is Step 2's.

- [ ] **Step 4: w_rope A/B run (qwen3-0.6b — after Step 1 sign-off)**

```bash
uv run python experiments/k4_w_rope_ab.py \
  --corpus-cache-paths results/cache/qwen3-0.6b_2048_off2048.safetensors results/cache/qwen3-0.6b_2048_off4096.safetensors \
  --cache-paths results/cache/qwen3-0.6b_2048.safetensors \
  --model-label qwen3-0.6b --model-name Qwen/Qwen3-0.6B
```

Expected: `[rope_validation] rel_fro(...) < 2e-2` lines (RoPE self-validation), then W_ROPE A/B VERDICT. C=1024 eigh x 28 layers x 2 variants — minutes on CPU, not hours.

- [ ] **Step 5: certificate run (existing gpt2 packs)**

```bash
uv run python experiments/k4_int8_certificate.py \
  --pack-path results/cache/k4_packs_gpt2.safetensors \
  --budgets 2.2 2.5 --model-label gpt2
```

Expected: per-layer `noise_to_signal ~ 1e-4..1e-5` scale (the review's ~7e-5 expectation); `margin_factor` large. If `max_implied_rel_degradation >= 0.005` (margin < 10x), that is a red flag against the replacement argument — record it verbatim, do not massage.

- [ ] **Step 6: Write the doc** — create `docs/2026-07-24-k4-math-actions-results.md` with this exact structure, filling every `«...»` from the named verdict JSON field (a `«...»` left in the committed doc is a task failure):

```markdown
# K4 math-review actions — results (2026-07-24)

Provenance: spec `docs/superpowers/specs/2026-07-24-k4-math-actions-design.md`
(math `docs/2026-07-24-k4-math-review.md` #1/#2/#3/#4/#9), plan
`docs/superpowers/plans/2026-07-24-k4-math-actions.md`. All local
(gpt2 mechanism scale + qwen3-0.6b for the RoPE-dependent A/B + analytic);
Llama confirmation of survivors rides the next rental (§D). Run ids:
- A-gate: `results/k4_charge_alloc/«id»/` (model-g), `«id»` (measured-g rerun)
- g-table: `results/k4_g_table/«id»/`
- w_rope A/B: `results/k4_w_rope_ab/«id»/`
- certificate: `results/k4_int8_certificate/«id»/`

## §A Charge-aware allocation (A-gate, finding #2)

Pre-registered rule (binding, from the spec): per (budget, S_ref) at MATCHED
skeptic-v2 bpe@S_ref, charge-aware heldout G1 win >= plain's, AND skeptic-v2
bpe@S_ref at matched win undercuts plain by >= 0.4 blended bits at
S_ref=4096 (blended = K-side/2; half the ~0.8-blended projection). Both
budgets fail => honest negative, allocator stays as-is. S_ref=16384
reported, not gated.

<!-- Keep exactly ONE of the two verdict blocks; delete the other. -->

**VERDICT — GATE PASSED.** At S_ref=4096: b2.2 win_ca «win_ca» vs
win_plain_at_matched_bpe «...» (win_not_worse=«...»), bits_saved_blended
«...»; b2.5 «...»/«...»/«...». a_gate_pass=true. The charge-aware arm is
eligible for promotion (recipe alias k4_b{budget}_s{S_ref}) — promotion is
the USER'S call, not this doc's; nothing is registered yet.

**VERDICT — HONEST NEGATIVE.** At S_ref=4096: b2.2 win_ca «...» vs
«...» (win_not_worse=«...»), bits_saved_blended «...» < 0.4; b2.5 «...».
a_gate_pass=false; the allocator stays as-is. The charge lever is dead at
gpt2 mechanism scale under the pre-registered bar; recorded, not retried
with moved goalposts.

Diagnostics (reported, never gated):
- c_used vs S_ref (mean over layers): plain «...» -> s16384 «...» -> s4096
  «...» (monotone_decreasing=«...» — the math doc's prediction).
- Tier-map shift: n_t0 per layer moves «...» -> «...» at b2.5/s4096 vs
  plain b2.5 (the 0<->2 boundary movement; from diagnostics.parquet).
- bpe-vs-S frontier (frontier.parquet, the "does optimizing for 4k hurt
  64k?" curve): table of (arm, budget, s_ref) x eval_s -> (bpe, win).
  «table» — report the full curve, no gate.

### §A.1 Measured-ĝ rerun (finding #4; reported beside, not gated)

g_table = «7 values» (n=«n_rows» calibration rows; per-tier p10/p90 spread
«...» — the shared-shape audit). A-gate quantities under measured-g vs
model-g: «side-by-side table of per_point entries». Note: the exact
Lagrangian enumeration SUBSUMES the +0.443/+0.782 threshold-offset fix
entirely (spec §A(i)) — there is no separate rounding change to report.

### §A.2 Not included

Eigenvalue shrinkage (finding #7) is NOT included — it needs its own
validation design (future work, §E).

## §B W-instrument RoPE A/B (finding #3)

Substrate: qwen3-0.6b (user-approved deviation from the spec's "gpt2" — gpt2
has no RoPE, the two variants are provably identical there; pinned by
test_query_moment_rotated_null_on_no_rope). Decision rule (pre-registered):
|rel_win_delta| < 2% at both budgets -> scoped claim; >= 2% -> the paper
uses the rotated form and the Llama refit is REQUIRED on the rental.

<!-- Keep exactly ONE block. -->

**VERDICT — SCOPED NEGLIGIBLE.** rel_win_delta b2.2 «...», b2.5 «...»
(both |.| < 2%). Paper claim scoped: "the frozen-rotation approximation is
measured-negligible at small-RoPE-model scale"; Llama spot-check queued for
the rental (§D). Per-rank frozen-vs-rotated basis overlap: «table by rank,
mean over layers».

**VERDICT — ROTATED FORM REQUIRED.** rel_win_delta b2.2 «...», b2.5 «...»
(>= 2% at «which»). The paper uses the rotated form's numbers going forward;
the Llama refit rides the rental as a REQUIRED item (§D). Per-rank overlap:
«table».

Methods-section footnote (enters the paper either way): the instrumented W
freezes the query's own rotation — relative to true causal logits the odd
sin*cos plane components enter sign-flipped and offsets are uniform-strided
rather than triangular; W is exact for the instrument, an approximation to
attention (math review #3(a)/(b)).

## §C int8 decoder certificate (finding #9)

Formula (exact, offline, per pack): added = sum_i lam_i * ||enc^T
ddec[:,i]||^2 with ddec = dec_int8 - dec; payload = sum_i lam_i * 4^{-b_i};
implied_rel_degradation = added/(payload+added) — the same axis as the
pre-registered VM gate rel_degradation_int8 < 5%.

Numbers (gpt2 packs, budgets 2.2/2.5): per-layer table from
`results/k4_int8_certificate/«id»/metrics.parquet`; max noise_to_signal
«...» (review's expectation ~7e-5); max implied_rel_degradation «...»;
margin to the 5% gate line: «margin_factor»x.

Honest limits (what the certificate does NOT capture): the diag(lam) model
of the code moment (payload-shift + payload x decoder cross-term, O(g(b))
relative on a ~7e-5 base); query-distribution interaction beyond the modeled
second moment; task-level effects — the VM half of the ledger is NOT
certified.

**THE USER REVIEWS THESE NUMBERS BEFORE VM TASK 8 IS RELEASED — the VM task
stays queued until explicitly released.** If released, the §3b "accounting
projection only" caveat upgrades to "distortion-certified" and Task 8's gate
measurement is replaced by this certificate; task-level confirmation stays
on the VM checklist per repo discipline.

## §D VM addendum (one line per surviving item)

<!-- Include only lines whose local gate/readout survived; delete the rest. -->
- Charge-aware packs at Llama scale: refit with s_ref in {4096, 16384},
  rerun the NIAH/frontier point at 4k-8k (the region the duel loses).
- w_rope Llama spot-check (scoped claim) OR required Llama refit (rotated
  form) per §B's verdict.
- int8 decoder: IF the user releases Task 8 on the certificate, drop the VM
  distortion gate and keep only the task-level confirmation row.
- Measured-ĝ at Llama scale: one calibration pass, reuse k4_g_table.

## §E Future work (each needs its own validation design; none started)

- Eigenvalue shrinkage before the waterfill (finding #7, MP-edge clipping /
  Ledoit-Wolf).
- Mean-centering sweep row (finding #10a).
- 1-bit sign-quantizer tier (finding #10c; needs a width-1 container).
```

- [ ] **Step 7: Fill every «...» from the run artifacts**, delete the non-applicable verdict blocks, and self-check: `grep -c "«" docs/2026-07-24-k4-math-actions-results.md` must print 0.

- [ ] **Step 8: Stage and propose the commit (STOP for user approval)**

```bash
git add docs/2026-07-24-k4-math-actions-results.md \
  results/k4_charge_alloc results/k4_g_table results/k4_w_rope_ab results/k4_int8_certificate
```

(Stage only the specific run dirs recorded in the doc header, not strays.)
Proposed message: `docs(k4): math-actions results — A-gate + measured-g rerun, w_rope A/B (qwen3 substrate), int8 certificate table; user-review gate on VM Task 8 recorded`

---

### Task 8: Verification gate + push

**Files:** none new — verification only.

- [ ] **Step 1: Full battery from clean state**

```bash
cd /d/Projects/bmx
uv run ruff format . && uv run ruff check . && uv run pytest -q
```

Expected: `551 passed, 17 skipped, 1 xfailed` (baseline 531 + 20 new), ruff clean. If format touched files, re-stage them.

- [ ] **Step 2: Default-inertness spot audit (evidence, not assertion)** — run the pin subset explicitly and paste the output into the report:

```bash
uv run pytest -q tests/test_k4_experiments.py::test_k4_fit_packs_default_unchanged \
  tests/test_spectral.py::test_pack_from_basis_lam_alloc_default_unchanged \
  tests/test_spectral.py::test_pack_from_basis_s_ref_default_inert \
  tests/test_spectral.py::test_query_moment_w_rope_default_inert \
  tests/test_cache_codecs.py::test_allocate_bits_selection_round_default_pin
```

Expected: `5 passed`.

- [ ] **Step 3: Confirm nothing frozen was touched**

```bash
git status --porcelain -- src/bmx/cache/streaming.py src/bmx/cache/recipes.py src/bmx/cache/specs.py results/cache/
git log --oneline 5da8faa..HEAD
```

Expected: first command prints nothing tracked-modified (results/cache is gitignored except the pack sidecars, which must be unmodified); second lists exactly this plan's commits.

- [ ] **Step 4: Propose the push (STOP for user approval — do not push without it)**

```bash
git push origin feat/triton-decode-kernel
```

Report the final state: battery counts, the run-ids, the three verdict outcomes, and the two open user decisions (charge-aware promotion if the gate passed; certificate release of VM Task 8).

---

## Self-Review (performed at plan time)

1. **Spec coverage.** §A charge-aware mechanism -> Tasks 1-2; §A pre-registered gate + diagnostics (c_used, tier shift, bpe-vs-S) -> Task 3; §A(i) threshold offsets -> subsumed by the exact enumeration, tested via the 0.443/0.782 vectors (Task 1) and noted in the doc (Task 7 §A.1); §A(ii) g_table -> Tasks 1, 5; §A(iii) shrinkage excluded -> Task 7 §A.2/§E. §B w_rope flag + A/B + 2% rule + overlap readout -> Task 4 (substrate conflict flagged, gpt2 null pinned). §C certificate + identity test + experiment + presentation with the user-review sentence -> Tasks 6-7. Non-goals respected: no CACHE_ARMS/recipes/save_pack_file changes anywhere; no Llama fitting; VM lines are §D one-liners. Constraints: default-inert pins exist for every new knob (`selection`, `fixed_charge`, `g_table`, `s_ref`, `w_rope`).
2. **Placeholder scan.** No TBD/TODO/"similar to Task N"; every code step carries the full code; the results doc's `«...»` fields are the deliverable's fill-in slots with a grep gate (Task 7 Step 7), not plan placeholders.
3. **Type consistency.** `allocate_bits_from_variance(selection=, g_table=, fixed_charge=)` (Task 1) matches Task 2's call site; `pack_from_basis(s_ref=, g_table=)` (Task 2) matches Tasks 3/4/5 call sites; `query_position_moment(w_rope=)` / `corpus_query_moment(..., w_rope)` / `corpus_fit_bases(w_rope=)` (Task 4) chain positionally-compatibly; `int8_decoder_certificate` dict keys (Task 6) match the experiment columns and the Task 7 doc fields; verdict JSON keys asserted in tests match the emitting code (`per_point`/`b{budget:g}_s{s_ref}`, `per_budget`/`{budget:g}`, `a_gate_pass`, `decision`, `user_review_required_before_vm_task8_release`).
4. **Arithmetic re-checks.** Round-path hand pin (256,16,1)/(0,2,4)/b2.0 -> (4,2,0); Lagrange switch vectors 2.1333/21.3333/6.6667 and offsets 0.443/0.782 recomputed by hand; Task 2 hand case (4,4,0,0) vs (4,4,4,0) traced through both bisections including feasibility monotonicity; certificate identity `W = enc enc^T` and `enc^T Sigma_fit enc = diag(lam)` verified against `fit_spectral_basis`'s construction.
5. **Known judgment calls (recorded, not hidden).** (a) Charge-aware budget semantics = mean total charge (Everett's "achieved total charge" — makes `budget` commensurate with skeptic-v2 bpe@S_ref, which is what the gate matches on). (b) Plain frontier grid extended down to 1.0 so the matched-bpe interpolation brackets the CA points; extrapolation is flagged in the verdict if it still occurs. (c) `blended = K-side/2` transcribed from the duel convention used in math review #2. (d) w_rope substrate: qwen3-0.6b (conflict #1, user sign-off gated).

**Execution:** use superpowers:subagent-driven-development (fresh subagent per task, review between tasks) or superpowers:executing-plans inline. Every commit and the final push stop for explicit user approval per CLAUDE.md.
