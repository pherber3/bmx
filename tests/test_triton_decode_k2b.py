"""CUDA-gated bit-exact tests for the fused k2b decode kernel (stage 3c).

k2b recipe (from experiments/k3_kernel_census.py):
    K = lowrank_rtn_channel, 3-bit, rank from cfg
    V = turboquant_mse_perhead, 2-bit (per-head block-diagonal Hadamard)

Tests:
  - import_clean: module imports cleanly on AMD/no-CUDA (TRITON_AVAILABLE=False)
  - k2b_oracle: fused_decode_attention_k2b vs naive_dense_attention
      max_abs < 2e-2  (tf32 kernel + fp16 codec; expect tighter)

Oracle: naive_dense_attention with the SAME per-head packed dicts (same codes,
different compute path) — apples-to-apples vs the fused kernel.

Fixture: quantize_packed("lowrank_rtn_channel", ...) for K (rank>0, S%group==0)
         quantize_packed("turboquant_mse_perhead", ..., h_heads=h_kv) for V

K and V use DIFFERENT seeds (k_seed=0, v_seed=7) so a param-mixup is caught.

Skips loudly on non-CUDA hosts.
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
# k2b fixture: lowrank_rtn_channel K + turboquant_mse_perhead V
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
    V arm: turboquant_mse_perhead (per-head block-diagonal Hadamard, h_heads=h_kv).

    K and V use DIFFERENT seeds (k_seed != v_seed) so a seed-mixup is caught.

    Returns:
        q:        (n_q_heads, 1, d) fp16
        k_blocks: list of (packed_dict, start, end) — lowrank_rtn_channel
        v_blocks: list of (packed_dict, start, end) — turboquant_mse_perhead
        kwargs:   dict for naive_dense_attention / chunked_dequant_attention oracle
    """
    from bmx.cache.codecs import quantize_packed
    from bmx.cache.collect import to_matrix

    assert blk % k_group == 0, f"blk={blk} must be divisible by k_group={k_group}"
    assert k_rank > 0, "lowrank_rtn_channel requires rank > 0"
    assert k_rank <= min(blk, d), f"rank {k_rank} > min(blk={blk}, d={d})"
    assert v_bits >= 1, "turboquant_mse_perhead requires bits >= 1"

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

        # V: turboquant_mse_perhead (per-head block-diagonal Hadamard)
        vM = to_matrix(torch.randn(h_kv, blk, d))
        vp, _ = quantize_packed(
            "turboquant_mse_perhead",
            vM,
            bits=v_bits,
            seed=v_seed,
            h_heads=h_kv,
        )
        v_blocks.append((vp, start, end))

    kwargs = dict(
        k_arm="lowrank_rtn_channel",
        v_arm="turboquant_mse_perhead",
        group=k_group,  # K's group
        seed=k_seed,  # K's seed
        k_pre_rope=False,
        rope_cos=None,
        rope_sin=None,
        k_tail=None,
        v_tail=None,
        n_q_groups=n_q_groups,
        scale=d**-0.5,
        v_group=k_group,  # unused by turboquant_mse_perhead but required by oracle sig
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


# ---------------------------------------------------------------------------
# k2b oracle comparison (CUDA-only)
# ---------------------------------------------------------------------------


@cuda
def test_triton_k2b_decode_matches_oracle():
    """fused_decode_attention_k2b must match naive_dense_attention oracle.

    K = lowrank_rtn_channel (3-bit, rank=16), V = turboquant_mse_perhead (2-bit).
    K seed = 0, V seed = 7 (DIFFERENT — param-mixup would be caught).

    Oracle: naive_dense_attention with the SAME per-head packed dicts — same
    codes, different compute path. The only difference is kernel arithmetic (tf32).

    Tolerance: max_abs < 2e-2 (tf32 kernel + fp16 codec; expect tighter).
    If near 2e-2, investigate — do NOT loosen.
    """
    from bmx.cache.chunked_attention import attention_diff, naive_dense_attention
    from bmx.cache.triton_dequant_attention import (
        TRITON_AVAILABLE,
        build_kv_stacked_k2b,
        fused_decode_attention_k2b,
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

    h_kv = n_q_heads // n_q_groups

    # Oracle: naive_dense_attention on CPU (same per-head packed dicts).
    ref_cpu = naive_dense_attention(q, kb_cpu, vb_cpu, **kw)

    # Fused k2b kernel on CUDA.
    # build_kv_stacked_k2b moves tensors to device internally — pass CPU blocks.
    stacks = build_kv_stacked_k2b(
        kb_cpu,
        vb_cpu,
        max_blocks=n_blocks,
        h_kv=h_kv,
        blk_size=blk,
        d=d,
        device="cuda",
    )
    out_cuda = fused_decode_attention_k2b(
        q.cuda(),
        stacks,
        seq_len=n_blocks * blk,
        n_q_groups=n_q_groups,
        scale=d**-0.5,
        vbits=v_bits,
        v_seed=v_seed,
        rope_cos=None,
        rope_sin=None,
    )
    out_cpu = out_cuda.cpu()

    diff = attention_diff(out_cpu, ref_cpu)
    assert diff["max_abs"] < 2e-2, (
        f"fused_decode_attention_k2b drifted from oracle: {diff}.\n"
        f"  K arm: lowrank_rtn_channel (rank={k_rank}, bits={k_bits}, group={k_group}, seed={k_seed})\n"
        f"  V arm: turboquant_mse_perhead (bits={v_bits}, seed={v_seed})\n"
        "If max_abs >= 2e-2 investigate — do NOT loosen this tolerance."
    )


@cuda
def test_triton_k2b_pre_rope_matches_chunked():
    """k2b with k_pre_rope=True applies RoPE in-kernel.

    Reference is chunked_dequant_attention (PyTorch apply_rope on reconstructed K
    — the verified pre-RoPE path), so this confirms the in-kernel rotate_half matches.
    GQA (n_q_groups=4) + multi-KV-head.
    """
    from bmx.cache.chunked_attention import attention_diff, chunked_dequant_attention
    from bmx.cache.triton_dequant_attention import (
        build_kv_stacked_k2b,
        fused_decode_attention_k2b,
    )

    torch.manual_seed(99)
    n_q_heads, n_q_groups, d, blk, n_blocks = 8, 4, 64, 64, 2
    q, kb_cpu, vb_cpu, kw = _tiny_k2b_blocks(
        n_q_heads=n_q_heads,
        n_q_groups=n_q_groups,
        d=d,
        blk=blk,
        n_blocks=n_blocks,
        k_bits=3,
        k_rank=16,
        k_group=32,
        k_seed=0,
        v_bits=2,
        v_seed=7,
    )

    h_kv = n_q_heads // n_q_groups

    # Fabricate valid RoPE cos/sin covering all positions.
    S = n_blocks * blk
    ang = torch.randn(S, d)
    cos, sin = torch.cos(ang).to(torch.float16), torch.sin(ang).to(torch.float16)

    # Reference: chunked PyTorch path (apply_rope on reconstructed K).
    okw = dict(kw, k_pre_rope=True, rope_cos=cos, rope_sin=sin)
    ref = chunked_dequant_attention(q, kb_cpu, vb_cpu, **okw)

    # Fused k2b kernel with in-kernel RoPE.
    stacks = build_kv_stacked_k2b(
        kb_cpu,
        vb_cpu,
        max_blocks=n_blocks,
        h_kv=h_kv,
        blk_size=blk,
        d=d,
        device="cuda",
    )
    out = fused_decode_attention_k2b(
        q.cuda(),
        stacks,
        seq_len=n_blocks * blk,
        n_q_groups=n_q_groups,
        scale=d**-0.5,
        vbits=2,
        v_seed=7,
        rope_cos=cos.cuda(),
        rope_sin=sin.cuda(),
    ).cpu()

    diff = attention_diff(out, ref)
    assert diff["max_abs"] < 2e-2, (
        f"k2b in-kernel RoPE diverged from chunked reference: {diff}. "
        "If max_abs >= 2e-2 investigate the in-kernel rotate_half — do NOT loosen."
    )


@cuda
def test_triton_k2b_different_seeds_detected():
    """Verify K and V use DIFFERENT seeds — a v_seed mixup changes output significantly.

    This test builds stacks with v_seed=7 (correct), then calls fused_decode_attention_k2b
    with the WRONG v_seed=0. The correct-seed oracle uses v_seed=7. The diff must be
    large (> 0.1), proving that a seed-mixup would be caught by
    test_triton_k2b_decode_matches_oracle.

    If this test fails (diff is small), K and V are using effectively the SAME seed and
    the oracle test would NOT catch a v_seed threading bug.
    """
    from bmx.cache.chunked_attention import attention_diff, naive_dense_attention
    from bmx.cache.triton_dequant_attention import (
        build_kv_stacked_k2b,
        fused_decode_attention_k2b,
    )

    torch.manual_seed(55)
    n_q_heads, n_q_groups, d, blk, n_blocks = 8, 4, 64, 64, 2
    v_seed_correct, v_seed_wrong = 7, 0
    v_bits = 2

    q, kb_cpu, vb_cpu, kw = _tiny_k2b_blocks(
        n_q_heads=n_q_heads,
        n_q_groups=n_q_groups,
        d=d,
        blk=blk,
        n_blocks=n_blocks,
        k_seed=0,
        v_seed=v_seed_correct,
    )

    h_kv = n_q_heads // n_q_groups

    # Oracle: correct v_seed (CPU reference).
    ref_correct = naive_dense_attention(q, kb_cpu, vb_cpu, **kw)

    # Stacks built from blocks quantized with the CORRECT seed.
    stacks = build_kv_stacked_k2b(
        kb_cpu,
        vb_cpu,
        max_blocks=n_blocks,
        h_kv=h_kv,
        blk_size=blk,
        d=d,
        device="cuda",
    )

    # Kernel called with the WRONG v_seed — should diverge materially.
    out_wrong = fused_decode_attention_k2b(
        q.cuda(),
        stacks,
        seq_len=n_blocks * blk,
        n_q_groups=n_q_groups,
        scale=d**-0.5,
        vbits=v_bits,
        v_seed=v_seed_wrong,
        rope_cos=None,
        rope_sin=None,
    ).cpu()

    diff = attention_diff(out_wrong, ref_correct)
    assert diff["max_abs"] > 0.1, (
        f"Seed mixup diff is only {diff['max_abs']:.4f} — v_seeds {v_seed_correct} and "
        f"{v_seed_wrong} produce nearly identical output, so a v_seed threading "
        "bug would NOT be caught by the oracle test. Choose more different seeds."
    )
