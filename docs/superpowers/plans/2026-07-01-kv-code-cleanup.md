# KV Code Cleanup — Parity-Preserving Debt Paydown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Every task is a **behavior-preserving** deletion, motion, or text fix — if you find yourself changing a numeric expression, STOP and escalate.

**Goal:** Remove the accretion debt identified by the 2026-07-01 five-agent design review of the KV-program code (dead code, copy-evolution residue, misplaced registries, stale narration) with **zero change to any computed number**, sequenced so it lands cleanly against the publication plan (`docs/superpowers/plans/2026-07-01-publication-readiness.md`) and the pending GH200 merge-gate re-verify.

**Architecture:** Three tiers. Tier 1 (Tasks 1–3) cleans the Triton module **before** the GH200 re-verify so that run validates final source. Tier 2 (Tasks 4–5) moves the recipe registry and experiment scaffolding into place **before** publication-plan Tasks 5–6 touch the same files (halves publication Task 5). Tier 3 (Tasks 6–15) is the remaining src debt, safe to land any time before the paper writeup. All work happens on branch `feat/triton-decode-kernel` (the Triton file exists only there; the branch merges to main after the GH200 re-verify).

**Tech Stack:** Python 3.12, PyTorch 2.12, transformers 5.11, pytest, ruff, uv. Dev box is AMD (no CUDA) — Triton kernel code cannot execute locally; the module must still *import* cleanly with `TRITON_AVAILABLE=False`, which the local suite exercises.

## Global Constraints

Copied from `CLAUDE.md` — every task implicitly includes these:

- **NEVER `git commit` without the user's explicit approval.** Stage, propose a message, STOP. No "Co-Authored-By" or any AI attribution, ever.
- Before any commit: `uv run ruff format .` → `uv run ruff check .` → `uv run pytest -q` — all clean, then re-stage.
- Use the Bash tool (git bash), not PowerShell. Shell cwd resets between turns — `cd /d/Projects/bmx` first in fresh shells.
- Dependencies only via `uv add` / `uv add --dev` (no task here should need any).
- Local test baseline at plan start: **264 passed, 17 skipped, 1 xfailed** (~50 s). Some tasks legitimately change these counts (tests of deleted functions are removed; new tests added). Each such task states its expected delta; Task 15 updates the recorded baseline in `CLAUDE.md`.

**Parity discipline (the whole point of this plan):**

- Allowed change classes: DELETE verified-dead code; MOVE code verbatim between modules; RENAME symbols; REPLACE dispatch plumbing with provably-equivalent plumbing; EDIT comments/docstrings. Nothing else.
- **Forbidden-touch list** (parity-critical; if a task seems to require touching these, STOP): the honest-bpe arithmetic expressions (`bits + 16.0/group`, `16.0*rank*(S+C)/(S*C)`, tier/norm terms — moving a formula verbatim is fine, re-deriving or reassociating it is not); the two-block prompt prefill split; EOS-set resolution order in `generate_through_cache`; `logits_to_keep=1` placements; LongBench prompt templates; the median-length calibration in `k3_longbench.py`; `LEDGER_COLUMNS` names in `triton_bench.py`; the `arm` parquet column name; the PAGE formula `max(self._g, (128 // self._g) * self._g) if self._g > 1 else 128` (appears twice — both copies stay, both verbatim); RoPE-table dtype handling in either cache (they intentionally differ: streaming slices `.float()`, packed casts fp16 at grow).
- Triton-source caveat: local tests cannot compile Triton kernels. Tier-1 changes are chosen so nothing that *runs* is modified (dead code, host-side Python, text). They are additionally covered by the already-pending GH200 merge-gate re-verify.
- Verification gate for EVERY task: `cd /d/Projects/bmx && uv run ruff format . && uv run ruff check . && uv run pytest -q` → clean, with the task's expected test-count delta and no other change.

---

## Tier 1 — Triton hygiene (land BEFORE the GH200 merge-gate re-verify)

### Task 1: Delete the dead dense kernel `_fused_decode_kernel`

The debloat commit `7b07552` deleted the dense path's launcher (`fused_decode_attention`) and builder (`build_kv_stacked`) but **missed the kernel itself**. It is unlaunchable dead code (~165 lines).

**Files:**
- Modify: `src/bmx/cache/triton_dequant_attention.py` (kernel at ~lines 728–891; banner at ~686–713)
- Modify: `docs/2026-06-24-decode-path-debloat-removal.md` (append a note)

**Interfaces:**
- Consumes: nothing. Produces: nothing (pure deletion). Later tasks in this file anchor on symbol names, not line numbers.

- [ ] **Step 1: Verify the kernel is dead.**

Run: `cd /d/Projects/bmx && grep -rn "_fused_decode_kernel\b" src/ tests/ experiments/ docs/`
Expected: matches ONLY at its own `@triton.autotune(...)` decorator line and `def _fused_decode_kernel(` line inside `triton_dequant_attention.py` (a mention in `docs/` is fine — docs don't call code). If any OTHER Python file references it, STOP and report.

- [ ] **Step 2: Verify what must be KEPT.**

Run: `grep -n "_FUSED_AUTOTUNE_CONFIGS\|_fused_merge_kernel" src/bmx/cache/triton_dequant_attention.py`
Expected: `_FUSED_AUTOTUNE_CONFIGS` defined (~:720) and reused by the LIVE packed kernel's decorator (~:1038); `_fused_merge_kernel` defined (~:894) and launched in `_finalize_decode` (~:1004). **Both stay.**

- [ ] **Step 3: Delete the kernel.** Inside the `if TRITON_AVAILABLE:` block that begins at ~line 715: delete from the line `@triton.autotune(configs=_FUSED_AUTOTUNE_CONFIGS, key=["d", "n_q_groups"])` (the one at ~:728 immediately preceding `def _fused_decode_kernel(`) through the end of `_fused_decode_kernel`'s body (the three `tl.store(...)` lines ending at ~:891, just before the bare `@triton.jit` decorating `_fused_merge_kernel`). Do NOT delete `_FUSED_AUTOTUNE_CONFIGS`, the `if TRITON_AVAILABLE:` line, or `_fused_merge_kernel`.

- [ ] **Step 4: Retitle the orphaned banner.** The comment banner at ~686–713 ("FUSED decode kernel — single launch, internal KV-block loop. ...") documents design points (GQA group fusion, register carry, first-block −inf, LDG.E.128, GEMV-not-tl.dot, split-KV) that apply to the two LIVE kernels. Keep the banner text; change only its first line to:

```
# FUSED decode kernels — shared design notes (_fused_decode_packed_kernel, _fused_decode_k2b_kernel)
```

- [ ] **Step 5: Update the recovery doc.** Append to `docs/2026-06-24-decode-path-debloat-removal.md`:

```markdown
## Addendum (2026-07-01)

The original debloat removed the dense path's launcher (`fused_decode_attention`) and
builder (`build_kv_stacked`) but missed the kernel body itself. `_fused_decode_kernel`
(~165 lines) was deleted in the follow-up cleanup. Recovery: same as above — parent
commit `93751eb` contains the full dense path including this kernel.
```

- [ ] **Step 6: Gate.** `uv run ruff format . && uv run ruff check . && uv run pytest -q` → 264/17/1 unchanged (the kernel was inside `if TRITON_AVAILABLE:`, never imported locally).

- [ ] **Step 7: Stage and propose.** `git add -A`, propose: `refactor(triton): delete dead _fused_decode_kernel missed by 7b07552 debloat (launcher was removed, kernel body was not)`. STOP for approval.

---

### Task 2: Remove the dead `n_blocks_in_split` parameter + fix `pick_num_splits` SM default

`_online_block_kernel_launch` accepts `n_blocks_in_split`, documents it as "passed to kernel as do_not_specialize runtime arg" — but the launch passes only `blk`. Callers compute and thread it for nothing. Separately, `pick_num_splits` hardcodes `n_sms=132` (GH200) as the default.

**Files:**
- Modify: `src/bmx/cache/triton_dequant_attention.py`

**Interfaces:**
- Produces: `_online_block_kernel_launch(q, K_kv, V_kv, acc, m, lse, n_q_groups, scale)` — the `n_blocks_in_split` kwarg is gone. `pick_num_splits(seq_len, blk_size, h_kv, n_sms=None, occupancy_mult=2)` — `n_sms=None` now means "read device props, fallback 132".

- [ ] **Step 1: Verify the parameter never reaches a kernel.**

Run: `grep -n "n_blocks_in_split" src/bmx/cache/triton_dequant_attention.py`
Expected matches: the function signature (`n_blocks_in_split: int = 1`, ~:292), its docstring lines (~:310–313), a comment (~:343), the computation `n_blocks_in_split = len(kb_split) + (...)` (~:607), and two call-site kwargs (~:622, ~:636). Confirm the actual kernel launch `_online_softmax_block_kernel[(n_q_groups,)](...)` (~:344–355) does NOT include it, and the kernel's `@triton.jit(do_not_specialize=["blk"])` (~:182) names `blk`, not `n_blocks_in_split`. If the kernel launch DOES pass it, STOP — the review finding is stale.

- [ ] **Step 2: Remove it everywhere.** Delete: the `n_blocks_in_split: int = 1,` signature line; the docstring entry for it; the comment line `# n_blocks_in_split is a do_not_specialize runtime arg (not constexpr).` (~:343); the computation block at ~:607–609 (`n_blocks_in_split = len(kb_split) + (1 if with_tail and k_tail is not None and k_tail.shape[1] > 0 else 0)`); and the `n_blocks_in_split=n_blocks_in_split,` kwarg at both call sites (~:622, ~:636).

- [ ] **Step 3: Fix the AUTOTUNE NOTE.** At ~:148–150, the block comment claims `do_not_specialize=["n_blocks_in_split"]`. Replace those three lines with:

```
#   do_not_specialize=["blk"] prevents per-block-length recompiles
#   (the AWS 10x TTFT regression class of bug). blk cannot be tl.constexpr
#   because the actual block length varies; it is used only for tile masking.
```

- [ ] **Step 4: `pick_num_splits` device-derived default.** Change the signature from `n_sms: int = 132` to `n_sms: int | None = None` and add as the first lines of the body:

```python
    if n_sms is None:
        n_sms = (
            torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
            if torch.cuda.is_available()
            else 132
        )
```

Append one line to its docstring: `n_sms=None reads the current device's SM count (GH200 = 132, so behavior there is unchanged); the 132 fallback keeps CPU-only test boxes deterministic.` Parity note: on the GH200 (132 SMs) and on the CUDA-less dev box this is bit-identical to today.

- [ ] **Step 5: Verify zero remaining references.** `grep -n "n_blocks_in_split" src/ tests/ experiments/` → no matches.

- [ ] **Step 6: Gate.** Full ruff + pytest → 264/17/1 unchanged.

- [ ] **Step 7: Stage and propose.** Propose: `refactor(triton): drop dead n_blocks_in_split param (never reached the kernel); pick_num_splits reads device SM count (GH200-identical)`. STOP for approval.

---

### Task 3: Fix stale Triton-era docs and deduplicate the retired-kernel eulogies (text only)

The module's front door describes two deleted designs; a helper docstring describes the pre-paged layout; the retired `_k2b_softmax_block_kernel` is eulogized three times across two files.

**Files:**
- Modify: `src/bmx/cache/triton_dequant_attention.py` (module docstring lines 1–12; `_pick_block_n` docstring ~:34–38; retired-kernel comment ~:553–558 region)
- Modify: `src/bmx/cache/packed_streaming.py` (retired-kernel comments ~:611–613 and ~:657–659)

- [ ] **Step 1: Replace the module docstring** (lines 1–12) with:

```python
"""Triton fused dequant-attention DECODE kernels.

Two single-launch split-KV decode kernels that dequantize packed codes IN-KERNEL:
  - fused_decode_attention_packed — RTN arms (int8 codes, post-RoPE K).
  - fused_decode_attention_k2b    — the k2b recipe (lowrank_rtn_channel K
    reconstructed + RoPE'd in-kernel; per-head turboquant V dequanted in-kernel).
Plus the per-block launch path (triton_decode_attention) as the non-fused
fallback for other arms, and _finalize_decode / merge for split-KV combination.

Imports cleanly with TRITON_AVAILABLE=False (AMD/no-CUDA dev box); kernels are
verified on the GH200 VM against the naive oracle + end-to-end logit parity.
Design rationale and staged-build ledger:
  docs/superpowers/specs/2026-06-24-triton-decode-kernel-design.md
"""
```

- [ ] **Step 2: Fix `_pick_block_n`'s stale claim.** Its docstring says "the cache flushes the whole prefill as one stored block of thousands of tokens" — false since the uniform paged layout landed. Replace the docstring with:

```python
    """KV tile size for the per-block decode loop: the largest power of 2 that is
    <= cap AND divides blk_size, so each tile lies within one stored block
    (contiguous load). Blocks are uniform PAGE=128 tokens under the paged layout,
    so in practice this returns 64; kept general for non-uniform test blocks."""
```

- [ ] **Step 3: Deduplicate the retired-kernel eulogies.** `grep -n "_k2b_softmax_block_kernel" src/` → expect three comment sites (triton file ~:553, packed_streaming ~:611–613 and ~:657–659). Replace EACH multi-line eulogy with a single line at the same spot:

```python
        # (A retired _k2b_softmax_block_kernel variant lived here; see docs/2026-06-24-decode-path-debloat-removal.md.)
```

Keep any surrounding lines that describe CURRENT dispatch behavior — only the sentences narrating the deleted kernel's history go.

- [ ] **Step 4: Gate.** Full ruff + pytest → 264/17/1 unchanged (text only).

- [ ] **Step 5: Stage and propose.** Propose: `docs(triton): fix stale module/helper docstrings (deleted designs, pre-paged layout); dedupe retired-kernel eulogies to one pointer each`. STOP for approval.

---

## Tier 2 — Registry + scaffolding moves (land BEFORE publication-plan Tasks 5–6)

### Task 4: Move the recipe registry `_spec_pair` into `src/bmx/cache/recipes.py`

The paper's headline objects (`k2b`, `k2b_ph`, `kivi`, … → (k_spec, v_spec) pairs) live as a private function inside `experiments/k3_live_generation.py`, imported cross-experiment, and are **re-implemented** as `_specs` in `k3_kernel_census.py`. This is the clearest violation of the "experiments are thin" rule, and publication-plan Tasks 5/9/10 all touch its consumers.

**Files:**
- Create: `src/bmx/cache/recipes.py`
- Create: `tests/test_recipes.py`
- Modify: `experiments/k3_live_generation.py` (delete `_spec_pair`, use the import)
- Modify: `experiments/k3_niah.py:26`, `experiments/k3_longbench.py:24` (import path)
- Modify: `experiments/k3_kernel_census.py` (delete `_specs` at :39–49, use `spec_pair`)
- Modify: `tests/test_k3_experiment.py` (the `_spec_pair` import, ~:63)

**Interfaces:**
- Produces: `spec_pair(arm: str, *, rank: int = 16, group: int = 64, seed: int = 0) -> tuple[CacheCodecSpec, CacheCodecSpec]` in `bmx.cache.recipes`. Call sites replace `_spec_pair(arm, cfg)` with `spec_pair(arm, rank=cfg.rank, group=cfg.group, seed=cfg.seed)`. Publication-plan workers import from here.

- [ ] **Step 1: Write the failing test** — `tests/test_recipes.py`:

```python
"""Pin the named-recipe registry: arm string -> (k_spec, v_spec)."""

import pytest

from bmx.cache.recipes import spec_pair
from bmx.cache.specs import CacheCodecSpec


def test_fp16_pair():
    k, v = spec_pair("fp16")
    assert k == CacheCodecSpec(arm="fp16") and v == CacheCodecSpec(arm="fp16")


def test_k2b_canonical():
    k, v = spec_pair("k2b", rank=16, group=64, seed=0)
    assert k == CacheCodecSpec(
        arm="lowrank_rtn_channel", bits=3, rank=16, group=64, seed=0, pre_rope=True
    )
    assert v == CacheCodecSpec(arm="turboquant_mse", bits=2, seed=0)


def test_k2b_parameterized_parsing():
    k, _ = spec_pair("k2b_k2r8", rank=16, group=64, seed=0)
    assert k.bits == 2 and k.rank == 8  # "k2b_k{bits}r{rank}" override


def test_k2b_ph_uses_perhead_v():
    _, v = spec_pair("k2b_ph", seed=0)
    assert v == CacheCodecSpec(arm="turboquant_mse_perhead", bits=2, seed=0)


def test_kivi_pair():
    k, v = spec_pair("kivi", group=64, seed=0)
    assert k.arm == "rtn_channel" and v.arm == "rtn_token"
    assert k.bits == v.bits == 2


def test_turboquant_arms_symmetric():
    for arm in ("turboquant_mse", "turboquant_prod"):
        k, v = spec_pair(arm, seed=0)
        assert k == v and k.arm == arm and k.bits == 2


def test_unknown_arm_raises():
    with pytest.raises(ValueError, match="unknown arm"):
        spec_pair("nope")


def test_census_specs_equivalence():
    # k3_kernel_census previously hand-rolled its own _specs("k2b"); pin that
    # spec_pair with defaults reproduces it exactly so the census swap is a no-op.
    k, v = spec_pair("k2b")
    assert k == CacheCodecSpec(
        arm="lowrank_rtn_channel", bits=3, rank=16, group=64, pre_rope=True
    )
    assert v == CacheCodecSpec(arm="turboquant_mse", bits=2)
```

- [ ] **Step 2: Run it, verify it fails.** `uv run pytest tests/test_recipes.py -v` → FAIL (`ModuleNotFoundError: bmx.cache.recipes`).

NOTE on `test_census_specs_equivalence`: it relies on `CacheCodecSpec`'s `seed` defaulting to 0 (census omits `seed`, `spec_pair` passes `seed=0`). Confirm in `src/bmx/cache/specs.py` that the default is `seed: int = 0` before relying on dataclass equality. If the default differs, STOP and report.

- [ ] **Step 3: Create `src/bmx/cache/recipes.py`.** The bodies are `_spec_pair`'s branches VERBATIM (from `experiments/k3_live_generation.py:60–116`) with `cfg.rank/cfg.group/cfg.seed` → `rank/group/seed`. Carry the docstrings — they hold the bpe-matching rationale:

```python
"""Named end-to-end KV-compression recipes: arm string -> (k_spec, v_spec).

The registry behind every K3 experiment's --arms option (NIAH, LongBench,
live-generation, kernel census). One definition; the parquet `arm` column is
these names.
"""

from __future__ import annotations

from bmx.cache.specs import CacheCodecSpec


def spec_pair(
    arm: str, *, rank: int = 16, group: int = 64, seed: int = 0
) -> tuple[CacheCodecSpec, CacheCodecSpec]:
    """(k_spec, v_spec) for a named arm.

    K2b = lowrank K@3b pre-RoPE + rotate/Lloyd V@2b (the quality-first recipe; spends
    bits on keys, so it lands LOWER on compression than turboquant). For an apples-to-
    apples comparison at turboquant's compression, the ``k2b_kNbM`` arms drop the key
    budget to N bits / rank M: ``k2b_k2r8`` lands at ~7.2x (matched to turboquant_mse's
    7.9x and kivi's 7.1x), so quality differences there are at equal bits, not bought
    with extra storage. See the local bpe table in the session notes.
    """
    if arm == "fp16":
        return CacheCodecSpec(arm="fp16"), CacheCodecSpec(arm="fp16")
    # k2b_ph = canonical k2b but with the PER-HEAD Hadamard V codec
    # (turboquant_mse_perhead). Quality-equivalent to k2b (full-C V) and the arm the
    # fused k2b decode kernel runs — use it with --use-packed on CUDA to exercise +
    # regression-check the fused kernel against the recorded k2b results.
    if arm == "k2b_ph":
        return (
            CacheCodecSpec(
                arm="lowrank_rtn_channel",
                bits=3,
                rank=rank,
                group=group,
                seed=seed,
                pre_rope=True,
            ),
            CacheCodecSpec(arm="turboquant_mse_perhead", bits=2, seed=seed),
        )
    if arm == "k2b" or arm.startswith("k2b_k"):
        # Default canonical k2b: keys@3b, rank as passed. Parameterized variants
        # "k2b_k{bits}r{rank}" override the key budget to match compression.
        bits_k, rank_k = 3, rank
        if arm != "k2b":
            # Parse "k2b_k2r8" -> bits_k=2, rank=8.
            body = arm[len("k2b_k") :]
            bits_str, rank_str = body.split("r")
            bits_k, rank_k = int(bits_str), int(rank_str)
        return (
            CacheCodecSpec(
                arm="lowrank_rtn_channel",
                bits=bits_k,
                rank=rank_k,
                group=group,
                seed=seed,
                pre_rope=True,
            ),
            CacheCodecSpec(arm="turboquant_mse", bits=2, seed=seed),
        )
    if arm in ("turboquant_mse", "turboquant_prod"):
        s = CacheCodecSpec(arm=arm, bits=2, seed=seed)
        return s, s
    if arm == "kivi":
        return (
            CacheCodecSpec(arm="rtn_channel", bits=2, group=group, seed=seed),
            CacheCodecSpec(arm="rtn_token", bits=2, group=group, seed=seed),
        )
    raise ValueError(f"unknown arm {arm!r}")
```

- [ ] **Step 4: Run the new tests, verify pass.** `uv run pytest tests/test_recipes.py -v` → 8 PASS.

- [ ] **Step 5: Rewire the four consumers.**
  - `experiments/k3_live_generation.py`: delete `_spec_pair` (:60–116); add `from bmx.cache.recipes import spec_pair`; replace `k_spec, v_spec = _spec_pair(arm, cfg)` with `k_spec, v_spec = spec_pair(arm, rank=cfg.rank, group=cfg.group, seed=cfg.seed)`.
  - `experiments/k3_niah.py` and `experiments/k3_longbench.py`: replace `from experiments.k3_live_generation import _spec_pair` with `from bmx.cache.recipes import spec_pair`; same call-site rewrite (both Configs have `rank`, `group`, `seed`).
  - `experiments/k3_kernel_census.py`: delete `_specs` (:39–49); add the recipes import; replace `_specs(arm)` call(s) with `spec_pair(arm)` (defaults reproduce it exactly — pinned by `test_census_specs_equivalence`).
  - `tests/test_k3_experiment.py`: update the `_spec_pair` import/usage to `spec_pair` with explicit kwargs (grep the file for `_spec_pair` and mirror the call-site rewrite).
  - Then: `grep -rn "_spec_pair\|def _specs" experiments/ tests/ src/` → zero matches.

- [ ] **Step 6: Gate.** Full ruff + pytest → expected **272 passed** (264 + 8 new), 17 skipped, 1 xfailed.

- [ ] **Step 7: Stage and propose.** Propose: `refactor(cache): move recipe registry to bmx.cache.recipes.spec_pair — kills the census duplicate and the cross-experiment private import`. STOP for approval.

---

### Task 5: Shared experiment model loader (`experiments/_common.py`)

The model/tokenizer dual-mode loading block is copy-adapted in three experiments. Publication Task 5 edits two of them again — dedupe first.

**Files:**
- Create: `experiments/_common.py`
- Modify: `experiments/k3_niah.py` (~:54–71), `experiments/k3_longbench.py` (~:52–62), `experiments/k3_live_generation.py` (~:132–145 — the model/tokenizer lines ONLY, not the wikitext/needle lines)

**Interfaces:**
- Produces: `load_model_and_tokenizer(model_name: str, device: str)` → `(model, tokenizer)`; fp16, `.to(device)`, `.eval()`, lazy transformers import (offline tests never trigger a download).

- [ ] **Step 1: Create `experiments/_common.py`:**

```python
"""Shared scaffolding for the K3 experiment scripts (real-run path only).

The offline test path injects `model=` and never calls this — the transformers
import stays function-local so importing an experiment module downloads nothing.
"""

from __future__ import annotations

import torch


def load_model_and_tokenizer(model_name: str, device: str):
    """fp16 CausalLM + tokenizer, moved to device, eval mode. VM/real-run path."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
```

- [ ] **Step 2: Rewire the three experiments.** In each `run(...)`'s `if model is None:` block, replace the four-statement load sequence (`AutoModelForCausalLM.from_pretrained(...)`, `.to(...)`, `.eval()`, `AutoTokenizer.from_pretrained(...)`) with:

```python
        from experiments._common import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(cfg.model_name, cfg.device)
```

Keep everything else in those blocks untouched (haystack/LongBench/wikitext/needle loading, and each block's other function-local imports). Remove the now-unused `AutoModelForCausalLM, AutoTokenizer` import lines from those blocks; ruff check will confirm no unused imports remain.

- [ ] **Step 3: Confirm no offline-path regression.** This code path only runs when `model is None` (VM). The offline suite must be green and must not have gained any download attempt: `uv run pytest tests/test_k3_experiment.py tests/test_k3_niah_experiment.py tests/test_k3_longbench_experiment.py -v` → all pass (exact file names may differ — `grep -rl "k3_niah\|k3_longbench\|k3_live_generation" tests/` first and run those).

- [ ] **Step 4: Gate.** Full ruff + pytest → 272/17/1 (no delta from Task 4's baseline).

- [ ] **Step 5: Stage and propose.** Propose: `refactor(exp): shared load_model_and_tokenizer in experiments/_common.py (3x copy-adapted block)`. STOP for approval.

---

## Tier 3 — src debt (land any time before the paper writeup; execute in order)

### Task 6: Extract `hf_compat.py`; fix the collect/rope → streaming layering inversion; de-privatize cross-module helpers

Four generic HF-introspection helpers live in `streaming.py`, forcing `collect.py` and `rope.py` (lower-level modules) to import them from `streaming` via deferred function-local imports — the classic cycle-dodge. Separately, `collect._reshape_heads` and `collect._register_hooks` are private names used as public API by 3 other modules.

**Files:**
- Create: `src/bmx/cache/hf_compat.py`
- Modify: `src/bmx/cache/streaming.py` (delete the four defs at ~:497–548; import from hf_compat; internal uses at :101, :420, :428, :476)
- Modify: `src/bmx/cache/collect.py` (function-local imports at :87, :112 → one top-level import; rename `_reshape_heads` → `reshape_heads`, `_register_hooks` → `register_hooks`)
- Modify: `src/bmx/cache/rope.py` (function-local import at :32 → top-level)
- Modify: `src/bmx/cache/packed_streaming.py` (imports at :28, :33–35), `src/bmx/cache/live_eval.py` (:37), `src/bmx/cache/ppl_eval.py` (:41)
- Modify: `experiments/k3_niah.py`, `experiments/k3_longbench.py`, `experiments/k3_kernel_census.py`, `experiments/k3_live_generation.py` (resolve_vocab_size import)
- Modify: `tests/test_streaming_cache.py` (:441, :463, :485 — resolver imports)

**Interfaces:**
- Produces: `bmx.cache.hf_compat` exporting `resolve_text_config`, `resolve_vocab_size`, `resolve_decoder_layers`, `model_config_n_layers` (bodies verbatim from streaming.py). `bmx.cache.collect` exporting `reshape_heads`, `register_hooks` (renamed, bodies untouched). NO re-exports left in `streaming.py`.

- [ ] **Step 1: Create `src/bmx/cache/hf_compat.py`** with a two-line module docstring (`"""HF model/config introspection helpers, model-family agnostic. Layer-0: imports nothing from bmx.cache — collect.py and rope.py depend on this, breaking the old upward import into streaming.py."""`) and MOVE the four function bodies verbatim from `streaming.py:497–548`. They use only builtins/getattr — the module needs no imports.

- [ ] **Step 2: Update every import site.** `grep -rn "resolve_text_config\|resolve_vocab_size\|resolve_decoder_layers\|model_config_n_layers" src/ tests/ experiments/` and point every import at `bmx.cache.hf_compat`. Specifics:
  - `streaming.py`: delete the four defs; add `from bmx.cache.hf_compat import model_config_n_layers, resolve_decoder_layers, resolve_text_config` (it does not use `resolve_vocab_size` itself).
  - `collect.py`: replace BOTH function-local `from bmx.cache.streaming import ...` lines with one top-level `from bmx.cache.hf_compat import resolve_decoder_layers, resolve_text_config` — the cycle those local imports dodged is gone.
  - `rope.py`: same promotion to a top-level `from bmx.cache.hf_compat import resolve_text_config`.
  - `packed_streaming.py`, `live_eval.py`, the four experiments, `tests/test_streaming_cache.py`: mechanical import-path swaps.
  - Then verify NOTHING still imports resolvers from streaming: `grep -rn "from bmx.cache.streaming import" src/ tests/ experiments/ | grep -i "resolve\|n_layers"` → zero.

- [ ] **Step 3: De-privatize the two helpers.** In `collect.py` rename `_reshape_heads` → `reshape_heads` and `_register_hooks` → `register_hooks` (defs + all internal uses). Update the importers: `streaming.py:40`-region, `packed_streaming.py:28`, `ppl_eval.py:41`, plus any test hits from `grep -rn "_reshape_heads\|_register_hooks" src/ tests/ experiments/` → after the rename, zero underscore-form matches.

- [ ] **Step 4: Gate.** Full ruff + pytest → 272/17/1 unchanged. (The three resolver tests in `test_streaming_cache.py` now import from hf_compat but stay where they are.)

- [ ] **Step 5: Stage and propose.** Propose: `refactor(cache): extract hf_compat.py (resolvers out of streaming) — kills the collect/rope upward imports; de-privatize reshape_heads/register_hooks`. STOP for approval.

---

### Task 7: Delete `streaming.py` dead code

Three verified-dead items: `_group_size` (never called — only `self._g` is used), the `new_S_q <= 0` early branch in `update()` (fully subsumed), and the `_quantize_matrix` one-line wrapper.

**Files:**
- Modify: `src/bmx/cache/streaming.py`

- [ ] **Step 1: Verify.** `grep -rn "_group_size\|_quantize_matrix" src/ tests/ experiments/` → `_group_size`: only its own def. `_quantize_matrix`: its def + call sites inside `streaming.py` only (expect 3: the general-K path, the V path, and possibly one more). If outside callers exist, STOP.

- [ ] **Step 2: Delete `_group_size`** (def + docstring, ~:210–217).

- [ ] **Step 3: Delete the `new_S_q <= 0` branch** (~:239–256, from `if new_S_q <= 0:` through its `return self.keys, self.values`). **Equivalence proof (verify you agree before deleting):** `compute_flush_schedule` is monotone in S for fixed (W, page), so `_committed_S_q > 0` implies `new_S_q >= _committed_S_q > 0` — i.e. whenever `new_S_q <= 0`, the prefix is empty (`_q_prefix_k is None`, `_committed_S_q == 0`). Control then falls to the next branch `if new_S_q <= self._committed_S_q:` (0 ≤ 0), whose prefix-None arm sets `self.keys = keys; self.values = values` (identical), and whose blended-bpe arithmetic gives `(0 + S*h*d*16.0) / (S*h*d) = 16.0` exactly (`_quant_bits_* == 0` when nothing is committed) — identical to the deleted branch's hardcoded 16.0. No tensor op differs.

- [ ] **Step 4: Inline `_quantize_matrix`.** Replace each `self._quantize_matrix(X, spec)` call with `quantize_kv_layout(X, spec)` (the wrapper's exact body); delete the method. `quantize_kv_layout` is already imported at the top of the file — confirm, else add it to the existing `from bmx.cache.codecs import ...` line.

- [ ] **Step 5: DO NOT touch** the `_k_pre` prune `elif` (~:345–348). It looks dead but handles the zero-width-tensor edge after exact consumption; the review left it in place deliberately.

- [ ] **Step 6: Gate.** Full ruff + pytest → 272/17/1 unchanged (the streaming schedule/bpe tests pin this path hard).

- [ ] **Step 7: Stage and propose.** Propose: `refactor(cache): delete dead streaming code — _group_size, subsumed new_S_q<=0 branch (monotone-schedule proof in commit), _quantize_matrix wrapper`. STOP for approval.

---

### Task 8: Delete the four dead first-generation arm implementations in `codecs.py`

`_rtn_token`, `_rtn_channel`, `_rotate_rtn_token`, `_lowrank_rtn_channel` are the pre-packed-split implementations. All four arms are in `_SPLIT_ARMS`, so `quantize_cache` routes them through `quantize_packed`/`dequant_packed`, which implement the same math. The only caller of `_rtn_channel` is the dead `_lowrank_rtn_channel` itself.

**Files:**
- Modify: `src/bmx/cache/codecs.py` (defs at ~:214–220, ~:228–234, ~:242–252, ~:308–341 + their `# Arm N:` banner comments)

- [ ] **Step 1: Verify dead.** `grep -rn "_rtn_token\|_rtn_channel\b\|_rotate_rtn_token\|_lowrank_rtn_channel" src/ tests/ experiments/` → function-name matches must be ONLY inside `codecs.py` (their own defs, the internal `_rtn_channel` call at ~:335, docstring mentions). All test/experiment hits are the arm STRINGS (e.g. `"lowrank_rtn_channel"` passed to `quantize_cache`) — those are fine and unaffected. Confirm all four arm strings are in `_SPLIT_ARMS` (~:560–570) so the packed route serves them. If any function-name reference exists outside codecs.py, STOP.

- [ ] **Step 2: Delete** the four functions and their section banners (`# Arm 1: rtn_token`, `# Arm 2: rtn_channel`, `# Arm 3: rotate_rtn_token`, `# Arm 6: lowrank_rtn_channel`). **KEEP** the QJL block between Arms 3 and 6 (`_qjl_sketch`, `qjl_reconstruct`, ~:260–300) — it is live, public API.

- [ ] **Step 3: Check `_rotate` for orphaning.** `grep -n "_rotate(" src/bmx/cache/codecs.py` — `_rotate_rtn_token` was a caller; if the packed implementation (`quantize_packed`'s rotate branch) also calls `_rotate`, it STAYS. Delete `_rotate` only if its remaining callers are zero. (`_unrotate` is live either way — the dequant path uses it.)

- [ ] **Step 4: Gate.** Full ruff + pytest → 272/17/1 unchanged. Specifically confirm `tests/test_cache_codecs.py` (bpe pins for these arm strings) and `tests/test_codec_split.py` are green — they exercise the packed route, which is untouched.

- [ ] **Step 5: Stage and propose.** Propose: `refactor(codecs): delete dead first-generation arm impls (superseded by the packed split; verified zero callers)`. STOP for approval.

---

### Task 9: Merge `_lowrank_waterfill_channel` into the rotated family as `rotation="identity"`; collapse the dispatch ladder

The base waterfill arm is a verbatim clone of `_lowrank_rotwaterfill_channel`'s Q=None path (`_waterfill_in_basis(R, None)` — same SVD+fp16 roundtrip, same tier loop, same four-term bpe sum). Six elif branches in `quantize_cache` differ only by `rotation=` + forwarded kwargs.

**Files:**
- Modify: `src/bmx/cache/codecs.py` (`_lowrank_rotwaterfill_channel` ~:418–553; delete `_lowrank_waterfill_channel` ~:349–403; `quantize_cache` ladder ~:864–935)

**Interfaces:**
- Consumes/Produces: `quantize_cache` signature and behavior are UNCHANGED for every caller. Internal only.

- [ ] **Step 1: Add the `"identity"` mode.** In `_lowrank_rotwaterfill_channel`: add `"identity"` to the allowed-rotation assert tuple; change the no-rotation branch from `if not use_rotation:` to `if rotation == "identity" or not use_rotation:` (body unchanged: `R_hat, mean_payload = _waterfill_in_basis(R, None); rot_bits = 0.0`). Add to the docstring's mode list: `- "identity": no rotation — water-fill in the original basis (the former lowrank_waterfill_channel base arm).`

- [ ] **Step 2: Delete `_lowrank_waterfill_channel`** (~:349–403). Verify zero external callers first: `grep -rn "_lowrank_waterfill_channel" src/ tests/ experiments/` → only its def, the `quantize_cache` elif, and docstring mentions (update those mentions in `_lowrank_rotwaterfill_channel`'s docstring: "Same as the identity mode" replaces "Same as _lowrank_waterfill_channel").

**Equivalence proof (reviewer-verified line-by-line, re-verify):** with Q=None, `_waterfill_in_basis` runs `allocate_channel_bits(R, budget, tiers, axis=0)` + the identical tier loop (`rtn_quantize(R[:, cols].mT, b, group).mT`) as the base arm's :386–393; the bpe sum `mean_payload + 16.0/group + 16.0*rank*(S+C)/(S*C) + ceil(log2(len(tiers)))/S + 0.0` is term-for-term the base arm's :397–402. The SVD/fp16-roundtrip prologue (:466–472 vs :369–377) is verbatim-identical. Only assert-message text differs.

- [ ] **Step 3: Collapse the ladder.** Replace the seven waterfill `elif` branches in `quantize_cache` (~:864–935) with:

```python
    _WATERFILL_ROTATION = {
        "lowrank_waterfill_channel": "identity",
        "lowrank_eigwaterfill_channel": "klt",
        "lowrank_randwaterfill_channel": "random",
        "lowrank_topkwaterfill_channel": "topk",
        "lowrank_blockdiagwaterfill_channel": "blockdiag",
        "lowrank_frozenwaterfill_channel": "frozen",
        "lowrank_oraclewaterfill_channel": "oracle",
    }
    return _lowrank_rotwaterfill_channel(
        M,
        float(bits),
        group,
        rank,
        tiers=tiers,
        rotation=_WATERFILL_ROTATION[arm],
        seed=seed,
        charge_rotation=charge_rotation,
        topk_k=topk_k,
        prefill_fit_len=prefill_fit_len,
        h_kv=h_kv,
        svd_factors=svd_factors,
    )
```

(Hoist the dict to module level next to `_SPLIT_ARMS` if you prefer; either is fine.) **Uniform-forwarding equivalence:** previously-unforwarded kwargs are ignored by construction — `random` sets `rot_bits = 0.0` unconditionally; the klt/oracle/frozen branch computes `charge_rotation and rotation != "oracle"` so oracle stays uncharged; `seed`/`topk_k`/`prefill_fit_len`/`h_kv` are read only by their own modes. No numeric path changes.

- [ ] **Step 4: Gate.** Full ruff + pytest → 272/17/1 unchanged. `tests/test_cache_codecs.py` waterfill pins and `tests/test_k2_waterfill.py` are the authority — if ANY waterfill number moves, revert and report.

- [ ] **Step 5: Stage and propose.** Propose: `refactor(codecs): waterfill base arm = rotation="identity" of the rotated family; dispatch ladder -> mode table (equivalence proofs in plan)`. STOP for approval.

---

### Task 10: Single arm-traits table; fix the stale codecs module docstring

Three hand-synced registries (`CACHE_ARMS` tuple, `S_DIVISIBILITY_ARMS`, `_SPLIT_ARMS`) become derivations of one table. The module docstring says "six compression arms" (there are 14) and documents an outdated signature.

**Files:**
- Modify: `src/bmx/cache/codecs.py` (registries at ~:33–63 and ~:560–570; module docstring lines 1–~20)

- [ ] **Step 1: Replace the three registries** with one table + derivations, **preserving the exact current CACHE_ARMS order** (dict insertion order is the tuple order — copy it from the current tuple):

```python
import dataclasses  # add to existing imports if absent


@dataclasses.dataclass(frozen=True)
class _ArmTraits:
    s_divisible: bool = False  # codec asserts S % group == 0 (streaming alignment)
    packed: bool = False  # has a quantize_packed/dequant_packed split


_ARM_TABLE: dict[str, _ArmTraits] = {
    "rtn_token": _ArmTraits(packed=True),
    "rtn_channel": _ArmTraits(s_divisible=True, packed=True),
    "rotate_rtn_token": _ArmTraits(packed=True),
    "turboquant_mse": _ArmTraits(packed=True),
    "turboquant_mse_perhead": _ArmTraits(packed=True),
    "turboquant_prod": _ArmTraits(packed=True),
    "lowrank_rtn_channel": _ArmTraits(s_divisible=True, packed=True),
    "lowrank_waterfill_channel": _ArmTraits(s_divisible=True),
    "lowrank_eigwaterfill_channel": _ArmTraits(s_divisible=True),
    "lowrank_randwaterfill_channel": _ArmTraits(s_divisible=True),
    "lowrank_topkwaterfill_channel": _ArmTraits(s_divisible=True),
    "lowrank_blockdiagwaterfill_channel": _ArmTraits(s_divisible=True),
    "lowrank_frozenwaterfill_channel": _ArmTraits(s_divisible=True),
    "lowrank_oraclewaterfill_channel": _ArmTraits(s_divisible=True),
}

CACHE_ARMS = tuple(_ARM_TABLE)
S_DIVISIBILITY_ARMS = frozenset(a for a, t in _ARM_TABLE.items() if t.s_divisible)
_SPLIT_ARMS = frozenset(a for a, t in _ARM_TABLE.items() if t.packed)
```

Before writing: diff the derived sets against the current literals (the current `CACHE_ARMS` order and both frozensets are in the file — the table above was transcribed from them; verify element-for-element). `_SPLIT_ARMS` currently sits at ~:560 — move the derivation next to the table and delete the literal there.

- [ ] **Step 2: Fix the module docstring.** Replace the stale "six compression arms" opening and the outdated signature list with an accurate 6–8 line summary: the arm registry (point at `_ARM_TABLE`), the honest-bpe contract (ALL metadata counted — scales, norms, factors, tier maps), the packed split (`quantize_packed`/`dequant_packed` for streaming arms; `quantize_cache` = unpack∘pack for those), and a pointer to `quantize_cache`'s docstring for per-arm parameters. Do not restate the parameter list in the module docstring.

- [ ] **Step 3: Gate.** Full ruff + pytest → 272/17/1 unchanged. The registry-consistency tests (`tests/test_cache_codecs.py` ~:376/:510/:785) now verify a tautology — that's the point; they stay as regression guards.

- [ ] **Step 4: Stage and propose.** Propose: `refactor(codecs): derive the three arm registries from one _ARM_TABLE; fix stale module docstring (six arms -> 14)`. STOP for approval.

---

### Task 11: Delete dead eval-layer code

Four grep-verified items: `niah_recall_generate` (zero callers anywhere), `synthetic_filler`+`_FILLER_SENTENCE` (only its own test; its "used by the offline/CI path" docstring is false), `_quantize_kv` (verbatim pass-through wrapper), and the `CacheCodecSpec` backcompat re-export in `ppl_eval.py`.

**Files:**
- Modify: `src/bmx/cache/niah.py` (delete `niah_recall_generate`, ~:232–250)
- Modify: `src/bmx/cache/haystack.py` (delete `synthetic_filler` + `_FILLER_SENTENCE`; module docstring shrinks to the PG-corpus description)
- Modify: `src/bmx/cache/ppl_eval.py` (delete `_quantize_kv` ~:91–100, inline its call sites; delete the `from bmx.cache.specs import CacheCodecSpec  # re-export; was defined here` line IF nothing else in the file uses the name — if ppl_eval's own signatures use `CacheCodecSpec`, keep the import but delete the `# re-export` comment)
- Modify: `tests/test_haystack.py` (remove the synthetic_filler test(s))
- Modify: any test importing `CacheCodecSpec` FROM `ppl_eval` (`grep -rn "ppl_eval import.*CacheCodecSpec\|from bmx.cache.ppl_eval import" tests/` — expected: `tests/test_cache_specs.py`; point it at `bmx.cache.specs`)

- [ ] **Step 1: Verify each deletion target.**
  - `grep -rn "niah_recall_generate" src/ tests/ experiments/` → only its def. (Do NOT confuse with `niah_recall_argmax` — that one is LIVE, used by `k3_niah.py`'s offline path.)
  - `grep -rn "synthetic_filler\|_FILLER_SENTENCE" src/ tests/ experiments/` → only haystack.py def + `tests/test_haystack.py`.
  - `grep -n "_quantize_kv" src/bmx/cache/ppl_eval.py` → def + its internal call sites; `grep -rn "_quantize_kv" tests/ experiments/` → zero (STOP if not).

- [ ] **Step 2: Delete + inline.** For `_quantize_kv`: replace each internal call `_quantize_kv(X, spec)` with `quantize_kv_layout(X, spec)` (its exact body; the import already exists at ppl_eval.py:40). Delete the four targets. Update `tests/test_haystack.py` (keep any `load_pg_corpus`/`PG_ESSAYS_DATASET` tests). Update the `CacheCodecSpec` import in the affected test file(s).

- [ ] **Step 3: Gate.** Full ruff + pytest → expected count DROPS by the number of deleted synthetic_filler tests (likely 271 or 270 passed — record the actual number for Task 15). Everything else unchanged.

- [ ] **Step 4: Stage and propose.** Propose: `refactor(cache): delete dead eval code — niah_recall_generate, synthetic_filler, _quantize_kv wrapper, ppl_eval re-export`. STOP for approval.

---

### Task 12: Move `generate_through_cache` + `compression_for` to `src/bmx/cache/generate.py`

The task-agnostic generation engine lives inside the NIAH task module (longbench imports its core loop from a sibling task); the compression-accounting calibration lives in `live_eval.py` though live-eval doesn't use it. Both move verbatim to a neutral home. This also updates the publication plan's stale reference (its Task 5 points at `live_eval.py::compression_for`).

**Files:**
- Create: `src/bmx/cache/generate.py`
- Modify: `src/bmx/cache/niah.py` (remove `generate_through_cache` ~:142–229 + now-unused imports), `src/bmx/cache/live_eval.py` (remove `compression_for` :40–59), `src/bmx/cache/longbench.py` (:12 import), `experiments/k3_niah.py` (two imports), `experiments/k3_longbench.py` (two imports)
- Modify: any tests importing either symbol (`grep -rn "generate_through_cache\|compression_for" tests/`)
- Modify: `docs/superpowers/plans/2026-07-01-publication-readiness.md` (its Task 5 references `src/bmx/cache/live_eval.py::compression_for` — update to `src/bmx/cache/generate.py::compression_for`)

**Interfaces:**
- Produces: `bmx.cache.generate.generate_through_cache(...)` and `bmx.cache.generate.compression_for(model, k_spec, v_spec, length)` — signatures and bodies byte-identical to today. NO re-export shims left behind.

- [ ] **Step 1: Map the import surface.** `grep -rn "generate_through_cache\|compression_for" src/ tests/ experiments/` — list every site before touching anything.

- [ ] **Step 2: Create `src/bmx/cache/generate.py`.** Module docstring: `"""Task-agnostic generation + compression accounting through the streaming caches. generate_through_cache is the ONE generation loop shared by NIAH and LongBench (single EOS/packed/fp16-routing logic); compression_for reads honest blended bpe off a calibration prefill."""` Move both function bodies VERBATIM (anchor: `def generate_through_cache(` in niah.py through its final `return text.strip() if strip else text`; `def compression_for(` in live_eval.py through `return bpe_k, bpe_v, mem["compression"]`). Then run `uv run ruff check src/bmx/cache/generate.py` — every F821 undefined-name error names an import to carry over from the source module's header (expect at minimum: `torch`, `StreamingQuantizedCache`, the packed-cache import if `generate_through_cache` references it, `resolve_vocab_size` from `bmx.cache.hf_compat` after Task 6, `CacheCodecSpec` for annotations). Copy those import lines exactly; do not reorder logic.

- [ ] **Step 3: Rewire consumers, no shims.** Update every site from Step 1 to import from `bmx.cache.generate`. In `niah.py` and `live_eval.py` remove imports that became unused (ruff will flag). Then `grep -rn "from bmx.cache.niah import.*generate_through_cache\|from bmx.cache.live_eval import.*compression_for" src/ tests/ experiments/` → zero.

- [ ] **Step 4: Update the publication-plan reference.** In `docs/superpowers/plans/2026-07-01-publication-readiness.md`, Task 5's Interfaces/Context lines mention `src/bmx/cache/live_eval.py::compression_for()` — change the path to `src/bmx/cache/generate.py`, leaving everything else in that doc untouched.

- [ ] **Step 5: Gate.** Full ruff + pytest → same counts as after Task 11.

- [ ] **Step 6: Stage and propose.** Propose: `refactor(cache): move generate_through_cache + compression_for to bmx.cache.generate (neutral home; longbench no longer imports its engine from the niah task module)`. STOP for approval.

---

### Task 13: Replace the `query_abs_start` int-as-flag with `is_prefill: bool`

`chunked_dequant_attention` takes `query_abs_start: int | None` whose **None-ness** is the prefill flag; the integer value is admittedly unused ("kept as a meaningful value in case a future path needs it" — a speculative parameter). Every caller either passes None or strips it.

**Files:**
- Modify: `src/bmx/cache/chunked_attention.py` (signature + docstring + the gate at ~:241)
- Modify: `src/bmx/cache/packed_streaming.py` (`attend()` ~:508–517 flag computation, ~:536 `is_decode`, and the forwarding kwarg)
- Modify: `tests/factories.py` (:105, :144 — drop the kwarg), `tests/test_chunked_gqa_opt.py` (:28 — the strip becomes unnecessary), `experiments/k3_triton_decode.py` (:85, :112–114 — drop the kwarg + the strip)

**Interfaces:**
- Produces: `chunked_dequant_attention(..., is_prefill: bool = False, ...)`. `is_prefill=False` ≡ today's `query_abs_start=None` (decode/online-softmax); `is_prefill=True` ≡ not-None (delegate to `_prefill_dense_attention`). Dispatch is boolean-identical, so every code path is byte-identical.

- [ ] **Step 1:** In `chunked_dequant_attention`: replace the parameter `query_abs_start: int | None = None` with `is_prefill: bool = False`; replace the gate `if query_abs_start is not None:` with `if is_prefill:`; replace the parameter's docstring entry with `is_prefill: True during prefill (n_q > 1) — delegates to the dense flash-SDPA path (the model's attn_mask governs causality). False during decode (n_q == 1) — the online-softmax loop runs, no masking needed.`; trim the long pre-gate comment (~:237–240) to its first sentence (the O(S²) rationale) — the query_abs_start narration goes.

- [ ] **Step 2:** In `packed_streaming.attend()`: replace the block at ~:508–517 with:

```python
        is_prefill = is_causal and n_q > 1
```

(delete the 10-line comment and the `total_seq_len`/`query_abs_start` computation — nothing consumed the integer). Replace `is_decode = query_abs_start is None  # n_q==1` with `is_decode = not is_prefill`. Replace the forwarding kwarg `query_abs_start=query_abs_start` (wherever attend calls chunked_dequant_attention) with `is_prefill=is_prefill`.

- [ ] **Step 3:** Clean the strip sites: `grep -rn "query_abs_start" src/ tests/ experiments/` and at each remaining site — `tests/factories.py` (delete the `query_abs_start=None,` lines), `tests/test_chunked_gqa_opt.py:28` (the dict-comprehension strip can collapse to a plain `**kw`), `experiments/k3_triton_decode.py` (delete the kwarg line and the strip + its comment) — remove the vestige. Final grep → zero matches repo-wide.

- [ ] **Step 4: Gate.** Full ruff + pytest → same counts. `tests/test_packed_dispatch.py` and the chunked-oracle tests pin the dispatch equivalence on CPU.

- [ ] **Step 5: Stage and propose.** Propose: `refactor(cache): query_abs_start int-as-flag -> is_prefill bool (integer value was never used; dispatch boolean-identical)`. STOP for approval.

---

### Task 14: Comment-narration cleanup (explicit list — make ONLY these edits)

The "slop feel" source: commit-message narration fossilized in source. This task is text-only and CLOSED-LIST: make exactly the edits below, nothing else. When in doubt, keep the comment.

**KEEP untouched (load-bearing, all of it):** the correctness-invariant banner in the triton file (~:70–88); the `restore_value` autotune comment; the BLK>blk NaN-repro and masked-tile notes; the GPAD/`tl.dot` register-pressure note; the flat-load LDG.E.128 note; the FAIL-LOUD dispatch rule comment in `packed_streaming.attend`; every honest-bpe rationale in codecs; the `logits_to_keep=1` OOM note; the EOS-set note in `generate_through_cache`; the "template is exact, do not normalize" warning in longbench; the GiB-vs-GB note in kv_memory; the theorem citations (Cover-Thomas, TurboQuant §3.3); the pre-RoPE pitfall comments.

**Files:**
- Modify: `src/bmx/cache/streaming.py`, `src/bmx/cache/packed_streaming.py`, `src/bmx/cache/chunked_attention.py`, `src/bmx/cache/live_eval.py`, `src/bmx/cache/ppl_eval.py`, `src/bmx/bench/kv_memory.py`

- [ ] **Step 1: `streaming.py` module docstring.** Delete the "Lands in two stages: Stage A (prev commit) … Stage B (this commit)" sentences (~:10–15) — commit history belongs to git. Keep the docstring's description of WHAT the cache does (write-once storage, frozen subspace, fp16 window, honest bpe).

- [ ] **Step 2: Ticket-ID sweep.** `grep -n "Task 10\|C1 fix\|I1 fix\|(C3\|C3:\|Fix 3\|Fix 4\|deferred opt #2\|added in 3c\|3a/3b back-compat" src/bmx/cache/*.py` — at each hit, delete the ticket tag and keep the invariant. Examples of the rewrite pattern:
  - `# Special path: frozen subspace across blocks (I1 fix).` → `# Frozen subspace across blocks: fit once at first flush, project thereafter.`
  - `# --- C3: Prune _k_pre to free already-committed positions ---` → `# --- Prune _k_pre to free already-committed positions ---`
  - `# After Fix 3 (slab pruning), self.keys holds only the tail: ...` → `# self.keys holds only the tail: ...` (keep the rest of the sentence).
  - naive_dense_attention's `(added in 3c for k2b oracle tests ...). Default to group / seed for 3a/3b back-compat (...)` → `(k2b oracle tests use K=lowrank_rtn_channel and V=turboquant_mse with different seeds). Default to group / seed.`

- [ ] **Step 3: Prefill-mask war story — one home.** It is told three times. KEEP the version at the `AttentionMaskInterface` registration site in `packed_streaming.py` (~:208–214, where the fix lives). Replace the other two (`packed_streaming.py` ~:186–193; `chunked_attention.py` ~:179–184 inside `_prefill_dense_attention`) with the one-line version each: `# attn_mask (not is_causal) governs masking when provided — see the AttentionMaskInterface registration in packed_streaming.py and docs/2026-06-23-kernel-census-results.md.` Keep chunked's mechanical shape note (`attn_mask is 4D (b,1,q,kv); add the batch dim`) — that line is operational, not narration.

- [ ] **Step 4: `live_eval.py` triplication.** The token-by-token indexing derivation appears in the module docstring (~:14–21), a parameter docstring (~:87–92), and an inline comment (~:128–136). KEEP the inline comment (it sits on the indexing it explains). Trim the module docstring's version to one sentence + `(derivation at the scoring loop below)`; trim the parameter docstring's version to one sentence.

- [ ] **Step 5: `ppl_eval.py` step banners.** Delete banner comments that restate the next call (~:148–150, ~:171–173, ~:206–208 — the `# Step N: <what the next line does>` pattern). Keep any banner that states a non-obvious constraint.

- [ ] **Step 6: `kv_memory.py` compute_bound repetition.** `predict_decode_latency` states "compute_bound_flag is None here / needs peak_flops_per_s" three times (docstring ~:102–105 + comments ~:112–114). Keep the docstring statement; delete the two comment restatements.

- [ ] **Step 7: Gate.** Full ruff + pytest → counts unchanged. `git diff --stat` must show ONLY comment/docstring lines — if any code line changed, revert that hunk.

- [ ] **Step 8: Stage and propose.** Propose: `docs(cache): comment debloat — ticket-ID narration removed, war stories deduped to their one load-bearing home (closed-list edit)`. STOP for approval.

---

### Task 15: Update recorded baselines and close out

**Files:**
- Modify: `CLAUDE.md` (test-count line; Architecture section)
- Create: `docs/2026-07-01-kv-code-cleanup-results.md`

- [ ] **Step 1: Run the full gate one final time** and record the exact counts: `uv run ruff format . && uv run ruff check . && uv run pytest -q`.

- [ ] **Step 2: Update `CLAUDE.md`:** (a) the expected pytest counts in the Commands section (from Step 1's actual output — passed count changed in Tasks 4 and 11); (b) in the Architecture section's `src/bmx/cache/` line, add the three new modules with one-word roles: `recipes` (named arm→spec pairs), `generate` (shared generation loop + compression accounting), `hf_compat` (model introspection). Do not rewrite anything else.

- [ ] **Step 3: Write `docs/2026-07-01-kv-code-cleanup-results.md`:** one page — the review verdict (not slop; accretion debt), the ledger of what was deleted/moved per task with net line counts (`git log --stat` over the cleanup commits), the explicit statement that every change was NONE/LOW parity risk gated by the full suite, and the deferred list (below) with one-line reasons.

- [ ] **Step 4: Stage and propose.** Propose: `docs: KV code-cleanup ledger + baseline updates (CLAUDE.md counts, new module map)`. STOP for approval.

---

## Deferred — explicitly NOT in this plan (do not "helpfully" include)

| Item | Why deferred |
|---|---|
| Unify StreamingQuantizedLayer/PackedStreamingLayer into one flush engine + storage backends | HIGH parity risk near the GH200 merge gate; the RoPE-table dtype divergence (streaming slices fp32, packed casts fp16 at grow) is an intentional per-backend difference a merge must parameterize. Revisit only if the program reopens post-paper. |
| Delete the ~574-line per-block Triton path | Entangled with `k3_triton_decode`'s variant ledger and publication Task 11's latency re-run; do post-paper. |
| Merge the two fused kernels under an `IS_K2B` constexpr | Reviewed and rejected: disjoint dequant bodies + pointer lists; readability regression, zero codegen benefit. |
| Named bpe-term helpers (`scale_bits()` etc.); full-C turboquant = perhead h=1 | LOW-not-NONE (float reassociation risk / touches a headline baseline arm) with authoritative VM runs imminent. Post-paper polish. |
| `predict_peak` retirement; needle/haystack file folds; `_PagedStacks` dict-only normalization; `_assemble_dense_kv` extraction; experiment `arm`→`recipe` rename | Cosmetic or breaks parquet/plot/doc continuity; value below churn cost right now. |

## Self-Review

- **Coverage vs the five-agent review:** Tier 1 = Triton findings 1–3 + minor 8 (dead kernel, stale docs, dead param, SM count). Tier 2 = eval findings 1+3 (registry, scaffolding). Tier 3 = codecs findings 1/2/5/7/8-partial (Tasks 8–10), eval finding 4 + slop markers (Tasks 11–12, 14), streaming findings 3/5/6 + cross-cutting W1/D2 (Tasks 6–7, 13). Deferred items are the review's own HIGH/not-recommended list. ✓
- **Placeholder scan:** every code step shows exact code or names an exact symbol + verbatim-move instruction with a ruff-driven import check; every grep has an expected result and a STOP condition. No TBDs. ✓
- **Name consistency:** `spec_pair` (Tasks 4→5→12 consumers), `hf_compat` import paths (Task 6 feeds Task 12's Step 2 import list), `is_prefill` (Task 13), `reshape_heads`/`register_hooks` (Task 6). Test-count arithmetic: 264 → 272 (Task 4, +8) → −synthetic_filler tests (Task 11, recorded live) → CLAUDE.md updated from actuals (Task 15). ✓
- **Ordering hazards:** Task 12 must follow Task 11 (niah.py edits) and Task 6 (hf_compat import path); Task 9 must follow Task 8 (both edit codecs.py around the same regions); tasks are strictly sequential by number. Line numbers in later tasks are approximate by design — anchors are symbol names. ✓
