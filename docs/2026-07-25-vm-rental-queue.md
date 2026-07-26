# VM rental queue — consolidated (2026-07-25, pre-pause)

> **EXECUTED IN FULL (2026-07-25/26):** every queue item and both riders ran
> green on one GH200 rental, plus a filler/overnight extension —
> see `docs/2026-07-26-gh200-rental-results.md` for the complete record.
> (Suite count below predates the rental-period commits; now 651/17/1.)

One GH200 rental executes everything below, sequentially on one machine
(git-bundle transport; detached setsid launches; module-form invocation;
never sudo pip; don't kill long runs). All runbooks are validated against
the shipped CLIs. Every pre-rental science question is CLOSED — the two
brain-dig rounds plus the Lloyd gate changed no flags.

## Locked refit configuration (decided this week, do not re-litigate)

- `w_rope="rotated"` (REQUIRED — causal-instrument verdict 1.20×, non-circular)
- `lam_alloc=None` (shrinkage rejected), `payload_quant="rtn"` (Lloyd gate
  rejected), `dec_quant="int8_tl"` (promoted; `int8_t5` fallback)
- Record at refit: `n_t0…n_t8` per-tier counts (schema live — collapses the
  §3b band to exact numbers); per-cache `(1/C)·logdet` scalars (Llama Jensen
  r_pred at n/C≈8, bias factor ~0.94); one
  `per_layer_tier_thresholds` CPU call (re-check certificate margins at
  scale — layer-uniform margins run thin, min ~1.1×); the 3-point ridge
  sweep {1e-2, 1e-3, 1e-4} (one config line — confirm gpt2's flatness at
  Llama's n/C).

## Queue (order, runbook, estimate)

| # | Job | Runbook | Est. GPU-h |
|---|---|---|---|
| 1 | Triton decode re-verify (generate-parity + oracle + `scripts/profile_decode_ab.py`) — the branch-merge gate | `docs/2026-06-24-triton-decode-results.md` + kv-code-review ledger | 2–4 |
| 2 | Packed-spectral Tasks 7–11: gate battery → real-model parity ladder → 96k→128k census via `scripts/census_gate_driver.py` → NIAH delta-parity → results | `docs/superpowers/plans/2026-07-23-packed-spectral-path.md` | 8–16 |
| 3 | Rotated-W Llama refit (config above) → G1/NIAH duel point rerun | `docs/2026-07-24-k4-math-actions-results.md` §D + the locked config | 6–12 |
| 4 | Corpus-transfer Llama A1/A2 | `docs/superpowers/plans/2026-07-23-k4-corpus-transfer.md` VM addendum | 2–4 |
| 5 | Qwen3-8B replication Tasks 7–13 | `docs/superpowers/plans/2026-07-23-second-model-replication.md` | ~24 |

Core total ≈ 45–60 GPU-h ≈ 2–2.5 days.

## Open riders (user decides at booking)

1. **Final-recipe LongBench pass** (+15–30 GPU-h): one full pass with the
   final arm (rotated + int8_tl) so the paper's main quality table reflects
   the shipped recipe (baselines already banked). Recommended: yes.
2. **Bigram-synthesis arms on A1/A2** (+2–4 GPU-h): elevates the
   bigram-counts calibration recipe from gpt2-mechanism to
   Llama-demonstrated. Recommended: yes.

## Resume checklist (after the pause)

- Branch `feat/triton-decode-kernel`, all work committed+pushed; suite
  646 passed / 17 skipped / 1 xfailed at pause time; tree clean except the
  user's private notes file (untracked, their call).
- Session ledger `.superpowers/sdd/progress.md` is gitignored scratch — the
  durable record is: this doc, `docs/2026-07-25-k4-paper-shelf.md`, the
  results docs, and the auto-memory.
- Post-rental: paper drafting from the shelf doc; branch-merge decision
  after queue item 1 passes.
