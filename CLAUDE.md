# CLAUDE.md — bmx

Kill-or-confirm research code for LLM tensor compression (weights program
closed, KV-cache program closed-positive — see README for the findings arc).
Experiments exist to close gates; an honest negative is a valid result. Don't
polish numbers; report them. Theory questions: use the `personal-brain` skill /
`mcp__wiki__*` tools (vault anchors: VQ distortion objectives, Beta-coordinate
rotation, two-stage quantization, BM-decomposition notes).

## Hard rules

- **NEVER `git commit` without the user's explicit approval.** Stage, propose a
  message, stop. No "Co-Authored-By" or any AI attribution, ever.
- Before any commit: `uv run ruff format .` → `uv run ruff check .` →
  `uv run pytest -q` — all clean, then re-stage.
- Dependencies only via `uv add` / `uv add --dev`. Never hand-edit versions in
  `pyproject.toml`.
- Use the Bash tool (git bash), not PowerShell. The shell cwd resets between
  turns — `cd /d/Projects/bmx` first in fresh shells.
- This machine has an AMD 7900 XTX (no CUDA). GPU-authoritative work runs on a
  rented NVIDIA VM via `scripts/` (transport is git: push → pull → run →
  commit parquet back).

## Commands

```bash
uv run pytest -q                  # ≈ 50 s; expected: 220 passed, 1 xfailed
uv run python experiments/<x>.py --help   # tyro CLIs; tuples space-separated
uv run python experiments/collect_cache.py --model-name gpt2 --seq-len 1024
                                  # regenerates results/cache/* (gitignored)
```

The xfail (`test_cold_start_recovery`) is intentional: plain RALS swamps only
on random-dense synthetics; on the BM-ALS paper's own tensors it beats the
paper's solver by 3–10 orders of magnitude (`tests/test_sagemath_agreement.py`;
fixture regenerates via `scripts/export_sagemath_fixture.py`).

## Research state (one line per gate; full record in docs/)

- BM program: all three entries killed/re-scoped with measured reasons —
  `docs/2026-06-10-h100-session-results.md`.
- Avenue 1 (L+S weight residual): honest negative + the break-even inequality
  ε > 1 − 4^(−Δb) — `docs/2026-06-11-lrs-results.md`.
- Frontier law: transform weights hug the break-even line at every scale;
  payers are tables/rogue-channels only — `docs/2026-06-11-frontier-breakeven.md`.
- KV program (K1→K2c, all gates closed positive): recipe = keys pre-RoPE
  lowrank+per-channel @3b, values rotate+Lloyd @2b ⇒ ~3.0 bpe, +0.5% ppl,
  5.3× vs fp16; bits belong to K; unbiased coding dominated; prefill-frozen
  subspaces stream (0.94 of oracle, no drift) — `docs/2026-06-11-k1-*.md`,
  `docs/2026-06-12-k2*-results.md`.
- K3 (quantize-on-append cache, closed positive): `StreamingQuantized{Layer,Cache}`
  (subclass transformers 5.11 `DynamicLayer`/`Cache`) streams token-by-token —
  write-once quantized storage (each token quantized once from pristine source;
  fixes the turboquant_mse V re-quant blowup), frozen pre-RoPE subspace, fp16
  residual window so channel-grouped arms stream. Quality holds (1.001× fp16 tbt),
  honest packed bpe < fp16, all arms on one fair path — `docs/2026-06-19-k3-*.md`.
- NIAH retrieval metric (task metric #0, retrieval half — harness built, awaiting
  VM run): ROUGE-1 needle-recall under compression on the StreamingQuantizedCache
  path (Fu et al. / TurboQuant setup); offline synthetic-argmax mechanism gate +
  real PG-essay generate headline (`experiments/k3_niah.py --model-name`); honest
  per-arm measured compression vs the 4× line. Spec/plan:
  `docs/superpowers/{specs,plans}/2026-06-20-niah-*`; VM handoff:
  `docs/2026-06-20-niah-plan-state.md`.
- LongBench Code eval (task metric #0, coding half — harness built, awaiting VM run):
  TurboQuant Table-1 Code signal (`lcc` + `repobench-p`) via LongBench's exact
  `code_sim` edit-similarity (fuzzywuzzy, range 0–1) on the SAME StreamingQuantizedCache
  path — NIAH + LongBench now share `generate_through_cache`. Offline synthetic
  mechanism gate + real full-set headline (`experiments/k3_longbench.py --model-name`,
  `n_samples=None` = full 500+500). Spec/plan:
  `docs/superpowers/{specs,plans}/2026-06-20-longbench-*`; VM handoff:
  `docs/2026-06-20-longbench-plan-state.md`. (HumanEval pass@1 NOT pursued — the paper
  used LongBench Code; that closes the coding-task question.)
- **Open (engineering, not science):** fused dequant-attention kernel (use the
  Track B byte model in `src/bmx/bench/` to predict before building) for the literal
  process-RSS win; the authoritative SOTA-model VM run (real-text + planted needle,
  `--model-name`); 32k-context re-check.

## Conventions (everything assumes them)

- Cache tensors: `layer{i}.{k,v,q,k_pre}`, shape (h_kv, S, d) fp16; q = last
  n_q_keep positions, pre-RoPE; k_pre = pre-RoPE keys. The (h,S,d) ↔ (S, h·d)
  matrix layout lives ONLY in `bmx.cache.collect.to_matrix/from_matrix` —
  never hand-roll the permute/reshape.
- Stack tensor `T : (n1, n2, h)` — slice axis is mode 3. BM product
  `bmp(A,B,C)[i,j,k] = Σ_t A[i,t,k]·B[i,j,t]·C[t,j,k]`;
  `cyclic_transpose = permute(1,2,0)`.
- **Comparisons align on `param_count()` / total bits (ALL metadata counted:
  scales, norms, factors, indices), never on rank.** Rank is method-interpreted.
- Metrics: rank codecs on inner-product/logit distortion vs real queries, not
  Frobenius — Frobenius inverts rankings under rogue channels. Perplexity is
  the end-to-end verdict, too coarse to attribute component choices.
- dtype: fp64 in tests, fp32 in experiments/codecs (caches stored fp16).
  Fail fast: shape asserts at boundaries, no silent coercion.
- Tiny offline test models come from `tests/factories.py`; never download in
  tests.

## Architecture (one line each)

`src/bmx/decomp/` — registered methods (`@register` → `FitResult`): bmd_rals,
slice_svd, cp, tucker, shared_tucker, lrs. `src/bmx/cache/` — KV program:
`collect` (hooked K/V/Q/k_pre capture + layout helpers), `codecs`
(CACHE_ARMS: rtn token/channel, rotate, turboquant mse/prod, lowrank;
honest bpe), `rope` (cos/sin from config, apply-at-read), `metrics`
(per-head logit/output distortion, GQA-aware), `ppl_eval` (quantized-prefill
perplexity; `run_prefill` state is reusable across arms). `src/bmx/quant/` —
hadamard rotations, groupwise RTN, `breakeven` (the ε > 1−4^(−Δb) instrument),
`arms` (weight-side pipelines), stats. `src/bmx/stacks/`, `bench/` (Track B
factored matvec + byte model), `census.py`, `sweep.py`, `artifacts.py`
(`results/<exp>/<run-id>/` with config + env + SHA). Experiments are thin tyro
scripts; figures read parquet, never refit; commit metrics/figures, never
checkpoints or raw caches.

## Pitfalls already hit (don't rediscover)

- transformers 5.x: `past_key_values.layers[i].keys/.values` (no `.key_cache`);
  DynamicCache layer attrs are directly assignable (ppl_eval relies on it,
  pinned by the identity-invariant test). `LlamaRotaryEmbedding(config=...)`
  inherits rope_scaling — never re-derive RoPE frequencies.
- Quantize keys PRE-RoPE; rotation (RoPE or random) provably smears the
  low-rank/sparse structure you're about to exploit. Post-RoPE subspaces
  drift with position; pre-RoPE ones don't.
- tensorly 0.9 `partial_tucker` returns `(decomposition, errors)` — unpack
  `result[0]`.
- `torch.linalg.lstsq` on CUDA uses the full-rank 'gels' driver → garbage on
  rank-deficient blocks; `fit_bmd_rals` records its solver policy in metrics.
- torch QR has sign ambiguity; all orthogonal sampling goes through
  `quant.hadamard.orthogonalize`.
- A slice-order shuffle alone is a no-op null control; the per-slice rotations
  make the a3 null real. Random-sphere matrices are the codec-theory control.
- Plot scripts must select runs explicitly (`plot_k2.newest_run_with`) —
  blind concat of a results root double-counts reruns and mixes ablation rows
  (bits == -1 sentinel marks asymmetric/ablation rows in k2b parquets).
- `torch.compile` on Windows CPU is unreliable; CP-ALS at high rank is
  overnight-slow on CPU — trim grids for local looks.
- HF datasets v5: wikitext id is `Salesforce/wikitext`; use AutoTokenizer
  (GPT2TokenizerFast silently returns an EMPTY tokenizer on missing files).
