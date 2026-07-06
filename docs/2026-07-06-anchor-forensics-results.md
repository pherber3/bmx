# Anchor forensics — why our LongBench Code ≠ TurboQuant's, settled by experiment

**Date:** 2026-07-06 (overnight session). **Model:** Llama-3.1-8B-Instruct, GH200.
**Question:** our fp16 Code ≈ 62 vs TurboQuant Table-1 Full-Cache Code 46.28 — which
input-policy or harness difference explains the 15.7-point gap, and can the anchor gate
(which licenses transitive baselines) ever pass?

## The three input-policy hypotheses — all tested, all exonerated

| variant | fp16 lcc | fp16 repobench-p | fp16 Code avg |
|---|---|---|---|
| v1 full split, un-truncated (60h run) | — | — | **62.04** |
| v1 full split, truncated 31.5k (`20260706-000742`) | 65.17 | 58.77 | **61.97** |
| **v1_e (LongBench-E), truncated 31.5k** (`20260706-024100`) | 68.01 | 55.40 | **61.71** |
| TurboQuant Table-1 Full Cache | | | **46.28** |

- **Truncation:** Δ = 0.07. Code prompts are short; predicted and confirmed.
- **E-split:** Δ = 0.26 (and lcc moved UP, not toward 46). TurboQuant's stated eval set
  ("we employ LongBench-E") does not explain the gap.
- **Chat-wrap:** official LongBench excludes code tasks from build_chat; our harness
  matches. N/A for the official policy — but see the probe.

## The mechanism probe: code scores are exquisitely prompt-policy sensitive

Paired 100-sample lcc_e probe (`~/probe_chatwrap.log`), same items, same greedy
generation, only the prompt differs:

| prompt policy | code_sim (n=100) |
|---|---|
| raw LongBench template (official for code) | **67.35** |
| force-chat-wrapped (apply_chat_template) | **23.62** |

A 43.7-point swing from prompt policy alone — an order of magnitude larger than any
quantization effect measured in this program (k2b's full-table delta is −1.19).
TurboQuant's 46.28 lies between the two poles, consistent with a pipeline that wraps
and/or post-processes differently (their code is not available to check). Reproducing
someone else's absolute code_sim without their exact prompt/post-processing pipeline is
not achievable, and chasing it further has no scientific value.

## Decision (locks the paper's licensing strategy)

1. **Absolute-anchor gating on Code is retired.** The gate as originally designed
   (reproduce Table-1 absolute rows, then cite their baselines transitively) cannot pass
   across harnesses whose prompt policies differ invisibly.
2. **Transitive baselines are licensed on DELTAS-from-own-full-cache instead:** each
   harness's compression arms measured against its own fp16 row. Theirs: KIVI-3 −1.56,
   PolarQuant −0.28, TurboQuant-2.5 −0.62 (Avg, from Table 1). Ours (same-path,
   same-inputs, full 6-category): k2b −1.19 at 4.07×, turboquant_mse −8.50 at 7.27×.
   Bit-width caveat stays attached: their 2.5b uses an outlier split ours doesn't.
3. **Our absolute numbers stand as the faithful-official-LongBench measurement** (raw
   templates for code per the official exclusion list, official metrics, full sets),
   self-consistent across all three input-policy variants (61.7–62.0).

## Byproduct: real-tokenizer bug in the chat-wrap flag, found and fixed

First real-tokenizer execution of the W3-6 chat-wrap path crashed:
`apply_chat_template(tokenize=True, return_tensors="pt")` returns a list of
`tokenizers.Encoding` on the real Llama tokenizer (the CPU fake returned a tensor, so
tests passed). Fixed via `return_dict=True` + `["input_ids"]`, and the fake tokenizer
now reproduces the trap so the regression test is honest (`bd31499`).

## Cross-references

- Runs: `results/k3_longbench/20260706-000742-84bb1f8` (truncated v1, 3 arms incl.
  k2b_ph), `20260706-024100-84bb1f8` (v1_e fp16 anchor).
- The 60h un-truncated table + systems findings: `docs/2026-07-05-authoritative-vm-results.md`
  (§9 addendum + this doc supersede its §5a framing).
- Memory/latency engineering (separate thread): `docs/superpowers/plans/2026-07-05-resident-memory-realization.md`.
