"""CUDA-gated bit-exact test for the Triton k2b decode kernel (stage 3c).

k2b recipe (from experiments/k3_kernel_census.py):
    K = lowrank_rtn_channel, 3-bit, rank from cfg
    V = turboquant_mse, 2-bit

Tests:
  - import_clean: module imports cleanly on AMD/no-CUDA (TRITON_AVAILABLE=False)
  - k2b_oracle: triton_decode_attention (k2b path) vs naive_dense_attention
      max_abs < 2e-2  (real codec + fp16; expect tighter)

Oracle: naive_dense_attention with SEPARATE v_group / v_seed so K and V are
dequanted with their own params — apples-to-apples vs the kernel.

Fixture: quantize_packed("lowrank_rtn_channel", ...) for K (rank>0, S%group==0)
         quantize_packed("turboquant_mse", ...)      for V (bits>=1)

K and V use DIFFERENT seeds (k_seed=0, v_seed=7) so a param-mixup is caught.

Skips loudly on non-CUDA hosts.

VM-verify checklist (PRIORITIZED — run these first):
  1. Codebook gather correctness: M_quant values match Python oracle.
     Check: if max_abs ≫ 0.1, gather has wrong index dtype or stride.
  2. RTN residual in-kernel: verify res_scale broadcast (D, n_groups) → (D, BLK).
     Check: if K drift ≫ fp16 noise (~2e-4), res_scale is mis-indexed.
  3. tl.dot shapes for lowrank: requires BLK ≥ 16, RANK ≥ 16.
     If RANK < 16 → tl.dot errors; replace with elementwise outer-product loop.
  4. _v_dequant_turboquant_mse boundary dtype: M_quant is fp32, _unrotate
     keeps fp32, V_mat cast to fp16. Check fp32 → fp16 cast at the boundary.
  5. Per-head Us/Vfac/res shapes: if quantize_packed returns per-head tensors
     (h_kv, blk, rank), the launcher must index [kv]; if it returns shared
     (blk, rank) tensors, the launcher broadcasts. Both branches are implemented.
"""

import pytest
import torch

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton k2b decode kernel — VM/CUDA only (skipping on non-CUDA host)",
)


# ---------------------------------------------------------------------------
# Helper: move a packed-block list's dict tensors to CUDA
# ---------------------------------------------------------------------------


def _blocks_cuda(blocks: list) -> list:
    """Move each packed dict's tensors to CUDA; keep start/end ints unchanged."""
    out = []
    for packed, start, end in blocks:
        packed_cuda = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in packed.items()
        }
        out.append((packed_cuda, start, end))
    return out


# ---------------------------------------------------------------------------
# k2b fixture: lowrank_rtn_channel K + turboquant_mse V
# ---------------------------------------------------------------------------


def _tiny_k2b_blocks(
    *,
    n_q_heads: int,
    n_q_groups: int,
    d: int,
    blk: int,
    n_blocks: int,
    k_bits: int = 3,
    k_rank: int = 16,
    k_group: int = 32,
    k_seed: int = 0,
    v_bits: int = 2,
    v_seed: int = 7,
    seed_rng: int = 42,
):
    """Build (q, k_blocks, v_blocks, kwargs) for the k2b decode tests.

    K arm: lowrank_rtn_channel (rank>0, S%k_group==0).
    V arm: turboquant_mse (bits>=1).

    K and V use DIFFERENT seeds (k_seed != v_seed) so a seed-mixup is caught.
    v_group is not meaningful for turboquant_mse (it uses C directly), so we
    pass k_group as group for both (oracle will use v_group=k_group, v_seed=v_seed).

    Returns:
        q:        (n_q_heads, 1, d) fp16
        k_blocks: list of (packed_dict, start, end) — lowrank_rtn_channel
        v_blocks: list of (packed_dict, start, end) — turboquant_mse
        kwargs:   dict for triton_decode_attention / naive_dense_attention
    """
    from bmx.cache.codecs import quantize_packed
    from bmx.cache.collect import to_matrix

    assert blk % k_group == 0, f"blk={blk} must be divisible by k_group={k_group}"
    assert k_rank > 0, "lowrank_rtn_channel requires rank > 0"
    assert k_rank <= min(blk, d), f"rank {k_rank} > min(blk={blk}, d={d})"
    assert v_bits >= 1, "turboquant_mse requires bits >= 1"

    h_kv = n_q_heads // n_q_groups
    torch.manual_seed(seed_rng)
    q = torch.randn(n_q_heads, 1, d).to(torch.float16)

    k_blocks = []
    v_blocks = []
    for i in range(n_blocks):
        start, end = i * blk, (i + 1) * blk

        # K: lowrank_rtn_channel
        kM = to_matrix(torch.randn(h_kv, blk, d))  # (blk, h_kv*d) == (S, C)
        kp, _ = quantize_packed(
            "lowrank_rtn_channel",
            kM,
            bits=k_bits,
            group=k_group,
            seed=k_seed,
            rank=k_rank,
        )
        k_blocks.append((kp, start, end))

        # V: turboquant_mse
        vM = to_matrix(torch.randn(h_kv, blk, d))
        vp, _ = quantize_packed(
            "turboquant_mse",
            vM,
            bits=v_bits,
            seed=v_seed,
        )
        v_blocks.append((vp, start, end))

    kwargs = dict(
        k_arm="lowrank_rtn_channel",
        v_arm="turboquant_mse",
        group=k_group,  # K's group (V's group is irrelevant for turboquant_mse)
        seed=k_seed,  # K's seed
        k_pre_rope=False,  # pre-RoPE subspace — RoPE applied before quantize, not here
        rope_cos=None,
        rope_sin=None,
        k_tail=None,
        v_tail=None,
        n_q_groups=n_q_groups,
        scale=d**-0.5,
        v_group=k_group,  # V's group (irrelevant for turboquant_mse but must be set)
        v_seed=v_seed,  # V's OWN seed — different from k_seed
    )
    return q, k_blocks, v_blocks, kwargs


# ---------------------------------------------------------------------------
# Smoke test: import (non-CUDA gate)
# ---------------------------------------------------------------------------


def test_triton_k2b_module_imports():
    """Module must import cleanly on AMD/no-CUDA with TRITON_AVAILABLE=False."""
    from bmx.cache.triton_dequant_attention import TRITON_AVAILABLE  # noqa: F401

    assert isinstance(TRITON_AVAILABLE, bool)


def test_v_dequant_turboquant_mse_matches_oracle():
    """_v_dequant_turboquant_mse must match dequant_packed bit-for-bit on CPU.

    This is the Python V pre-dequant used in the k2b path.  It must match
    _turboquant_mse_dequant (and dequant_packed) exactly — if this diverges,
    the k2b oracle comparison will fail for reasons unrelated to the kernel.

    No CUDA needed — runs entirely on CPU.
    """
    from bmx.cache.codecs import dequant_packed, quantize_packed
    from bmx.cache.collect import from_matrix, to_matrix
    from bmx.cache.triton_dequant_attention import _v_dequant_turboquant_mse

    from tests.factories import tiny_gpt2  # noqa: F401  (just to confirm imports clean)

    torch.manual_seed(13)
    h_kv, blk, d = 2, 64, 32
    vM = to_matrix(torch.randn(h_kv, blk, d))  # (blk*h_kv, d)
    v_seed = 7
    v_bits = 2

    vp, _ = quantize_packed("turboquant_mse", vM, bits=v_bits, seed=v_seed)

    # Oracle: dequant_packed on (S, C) then from_matrix
    oracle_mat = dequant_packed("turboquant_mse", vp, seed=v_seed)  # (S, C)
    oracle = from_matrix(oracle_mat, h_kv).to(torch.float16)  # (h_kv, blk, d)

    # Under test: _v_dequant_turboquant_mse
    result = _v_dequant_turboquant_mse(
        vp, h_kv, v_seed, torch.device("cpu"), torch.float16
    )

    diff = (result.float() - oracle.float()).abs()
    assert diff.max() < 1e-5, (
        f"_v_dequant_turboquant_mse diverged from dequant_packed oracle: "
        f"max_abs={diff.max():.2e}. "
        "This means the Python V pre-dequant is wrong — fix before running on GPU."
    )


# ---------------------------------------------------------------------------
# k2b oracle comparison (CUDA-only)
# ---------------------------------------------------------------------------


@cuda
def test_triton_k2b_decode_matches_oracle():
    """triton_decode_attention (k2b path) must match naive_dense_attention.

    K = lowrank_rtn_channel (3-bit, rank=16), V = turboquant_mse (2-bit).
    K seed = 0, V seed = 7 (DIFFERENT — param-mixup would be caught).

    Oracle: naive_dense_attention with same packed data, using dequant_packed
    for BOTH arms — apples-to-apples (same codes, kernel vs Python unpack).

    Tolerance: max_abs < 2e-2 (real codec + fp16; expect tighter).
    If near 2e-2, investigate — do NOT loosen.
    """
    from bmx.cache.chunked_attention import attention_diff, naive_dense_attention
    from bmx.cache.triton_dequant_attention import (
        TRITON_AVAILABLE,
        triton_decode_attention,
    )

    assert TRITON_AVAILABLE, (
        "TRITON_AVAILABLE=False on a CUDA box — Triton not installed correctly. "
        "Install with: pip install triton"
    )

    torch.manual_seed(99)
    n_q_heads, n_q_groups, d, blk, n_blocks = 8, 4, 64, 64, 4
    k_rank, k_group, k_bits = 16, 32, 3
    v_bits = 2
    k_seed, v_seed = 0, 7

    q, kb_cpu, vb_cpu, kw = _tiny_k2b_blocks(
        n_q_heads=n_q_heads,
        n_q_groups=n_q_groups,
        d=d,
        blk=blk,
        n_blocks=n_blocks,
        k_bits=k_bits,
        k_rank=k_rank,
        k_group=k_group,
        k_seed=k_seed,
        v_bits=v_bits,
        v_seed=v_seed,
    )

    # Oracle: naive_dense_attention on CPU
    # Pass v_seed so V is dequanted with ITS OWN seed (not K's seed).
    # naive_dense_attention now accepts v_group / v_seed (added in 3c).
    oracle_kwargs = {
        k: v
        for k, v in kw.items()
        if k not in ("query_abs_start",)  # naive_dense_attention has no query_abs_start
    }
    ref_cpu = naive_dense_attention(q, kb_cpu, vb_cpu, **oracle_kwargs)

    # Triton k2b kernel on CUDA
    q_cuda = q.cuda()
    kb_cuda = _blocks_cuda(kb_cpu)
    vb_cuda = _blocks_cuda(vb_cpu)

    # triton_decode_attention routes to k2b path because k_arm="lowrank_rtn_channel"
    out_cuda = triton_decode_attention(q_cuda, kb_cuda, vb_cuda, **oracle_kwargs)
    out_cpu = out_cuda.cpu()

    diff = attention_diff(out_cpu, ref_cpu)
    assert diff["max_abs"] < 2e-2, (
        f"Triton k2b decode kernel drifted from oracle: {diff}.\n"
        f"  K arm: lowrank_rtn_channel (rank={k_rank}, bits={k_bits}, group={k_group}, seed={k_seed})\n"
        f"  V arm: turboquant_mse (bits={v_bits}, seed={v_seed})\n"
        "VM-verify checklist:\n"
        "  1. Codebook gather: check tl.load(cb_ptr + idx) index dtype (int16→int32)\n"
        "  2. RTN residual scale broadcast: (D, n_groups) → (D, BLK)\n"
        "  3. tl.dot shapes: RANK ≥ 16 required; if not, replace with loop\n"
        "  4. _unrotate boundary dtype: fp32 M_quant → fp16 V_mat\n"
        "If max_abs >= 2e-2 investigate — do NOT loosen this tolerance."
    )


@cuda
def test_triton_k2b_different_seeds_detected():
    """Verify K and V use DIFFERENT seeds — a seed mixup changes V significantly.

    This test quantizes V with v_seed=7, then dequants with the WRONG seed (0).
    The oracle uses the CORRECT v_seed=7.  The diff must be large (> 0.1),
    proving that a seed-mixup would be caught by test_triton_k2b_decode_matches_oracle.

    If this test fails (diff is small), K and V are using the SAME seed and
    the oracle test would NOT catch a v_seed threading bug.
    """
    from bmx.cache.codecs import dequant_packed, quantize_packed
    from bmx.cache.collect import from_matrix, to_matrix

    torch.manual_seed(55)
    h_kv, blk, d = 2, 64, 64
    v_seed_correct, v_seed_wrong = 7, 0
    v_bits = 2

    vM = to_matrix(torch.randn(h_kv, blk, d))
    vp, _ = quantize_packed("turboquant_mse", vM, bits=v_bits, seed=v_seed_correct)

    oracle = from_matrix(
        dequant_packed("turboquant_mse", vp, seed=v_seed_correct), h_kv
    )
    wrong = from_matrix(dequant_packed("turboquant_mse", vp, seed=v_seed_wrong), h_kv)

    diff = (oracle.float() - wrong.float()).abs().max().item()
    assert diff > 0.1, (
        f"Seed mixup diff is only {diff:.4f} — seeds {v_seed_correct} and "
        f"{v_seed_wrong} produce nearly identical dequant, so a v_seed threading "
        "bug would NOT be caught by the oracle test. Choose more different seeds."
    )
