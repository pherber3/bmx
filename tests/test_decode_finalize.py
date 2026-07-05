"""CPU-testable pure functions from the decode finalize/merge path.

Desk review: docs/2026-07-04-triton-decode-desk-review.md, fixes F1b + F2.

F2 replaced two per-call `.to(device)` H2D copies (gaussian_codebook, Hadamard
signs) with device-cached getters, mirroring the pre-existing `_hadamard_matrix`
pattern.

F1b replaced the per-decode-step PyTorch tail merge (96 per-split views +
repeat_interleave GQA expansion + a dedicated stack-based online-softmax combine)
with: (1) a GQA-shaped `_tail_partial` (no repeat_interleave), and (2) folding its
(acc, m, lse) into the split partials as one extra "virtual split" so the SAME GPU
`_fused_merge_kernel` used by the no-tail path handles both cases. The kernel body
itself is untouched (Triton code cannot run on this AMD/no-CUDA box) — everything
tested here is the pure-PyTorch math that feeds it, which is exactly what F1b was
scoped to change.
"""

import torch

from bmx.cache.codecs import _hadamard_signs, gaussian_codebook
from bmx.cache.triton_dequant_attention import (
    _codebook_dev,
    _signs_dev,
    _tail_partial,
)


# ---------------------------------------------------------------------------
# F2 — device-cached constants return the SAME object on repeated calls
# ---------------------------------------------------------------------------


def test_codebook_dev_cached_identity():
    """Same (bits, device) args -> the identical tensor object (no re-copy)."""
    a = _codebook_dev(4, "cpu")
    b = _codebook_dev(4, "cpu")
    assert a is b
    assert torch.equal(a, gaussian_codebook(4).to("cpu", torch.float32))


def test_codebook_dev_matches_uncached_construction():
    for bits in (2, 3, 4):
        cached = _codebook_dev(bits, "cpu")
        uncached = gaussian_codebook(bits).to("cpu", torch.float32)
        assert torch.equal(cached, uncached)
        assert cached.dtype == torch.float32


def test_signs_dev_cached_identity():
    a = _signs_dev(32, 7, "cpu")
    b = _signs_dev(32, 7, "cpu")
    assert a is b
    assert torch.equal(a, _hadamard_signs(32, 7).to("cpu", torch.float32))


def test_signs_dev_distinguishes_seed_and_d():
    base = _signs_dev(32, 7, "cpu")
    diff_seed = _signs_dev(32, 8, "cpu")
    diff_d = _signs_dev(16, 7, "cpu")
    assert not torch.equal(base, diff_seed)
    assert base.shape != diff_d.shape


# ---------------------------------------------------------------------------
# F1b(a) — _tail_partial vs the OLD per-step math (repeat_interleave + einsum
# over n_q_heads), lifted from the deleted _finalize_decode tail branch as the
# oracle. GQA-shaped and repeat_interleave-expanded must agree exactly (same
# floating-point operations, just reshaped) at a tight tolerance.
# ---------------------------------------------------------------------------


def _old_tail_partial_expanded(q_kv, k_tail, v_tail, scale, n_q_groups):
    """The deleted _finalize_decode tail branch's math, reshaped to (h_kv, G, d)
    for direct comparison with _tail_partial's return shape.

    q_kv: (h_kv, G, d). k_tail/v_tail: (h_kv, T, d).
    """
    h_kv, G, d = q_kv.shape
    n_q_heads = h_kv * G
    q_flat = q_kv.reshape(n_q_heads, 1, d)
    kt = k_tail.to(torch.float32).repeat_interleave(n_q_groups, dim=0)
    vt = v_tail.to(torch.float32).repeat_interleave(n_q_groups, dim=0)
    qf = q_flat.float()
    st = torch.einsum("hqd,hkd->hqk", qf, kt) * scale  # (n_q_heads, 1, T)
    mt = st.amax(dim=-1, keepdim=True)
    pt = torch.exp(st - mt)
    lse_t = pt.sum(dim=-1, keepdim=True)
    acc_t = torch.einsum("hqk,hkd->hqd", pt, vt)
    return (
        acc_t.reshape(h_kv, G, d),
        mt.reshape(h_kv, G),
        lse_t.reshape(h_kv, G),
    )


def test_tail_partial_matches_old_repeat_interleave_math():
    torch.manual_seed(0)
    h_kv, G, T, d = 4, 3, 17, 32
    scale = 1.0 / (d**0.5)
    q_kv = torch.randn(h_kv, G, d)
    k_tail = torch.randn(h_kv, T, d, dtype=torch.float16)
    v_tail = torch.randn(h_kv, T, d, dtype=torch.float16)

    acc, m, lse = _tail_partial(q_kv, k_tail, v_tail, scale)
    acc_ref, m_ref, lse_ref = _old_tail_partial_expanded(
        q_kv, k_tail, v_tail, scale, n_q_groups=G
    )

    assert acc.dtype == torch.float32
    assert torch.allclose(acc, acc_ref, rtol=1e-6, atol=1e-6)
    assert torch.allclose(m, m_ref, rtol=1e-6, atol=1e-6)
    assert torch.allclose(lse, lse_ref, rtol=1e-6, atol=1e-6)


def test_tail_partial_shapes_and_gqa_grouping():
    """h_kv=1, G=1 sanity: reduces to plain single-head attention over the tail."""
    h_kv, G, T, d = 1, 1, 5, 8
    scale = 1.0 / (d**0.5)
    q_kv = torch.randn(h_kv, G, d)
    k_tail = torch.randn(h_kv, T, d)
    v_tail = torch.randn(h_kv, T, d)

    acc, m, lse = _tail_partial(q_kv, k_tail, v_tail, scale)
    assert acc.shape == (h_kv, G, d)
    assert m.shape == (h_kv, G)
    assert lse.shape == (h_kv, G)

    # Direct dense-softmax reference.
    s = (q_kv[0, 0] @ k_tail[0].T) * scale
    p = torch.softmax(s, dim=-1)
    out_ref = p @ v_tail[0]
    out = (acc / lse.unsqueeze(-1))[0, 0]
    assert torch.allclose(out, out_ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# F1b(b) — slot-append design: combining [split partials + tail-as-extra-slot]
# via the online-softmax merge equals computing attention directly over the
# UNION of the stored (split) tokens and the tail tokens. This is the design
# _finalize_decode now relies on (the merge kernel itself is untouched Triton
# code and can't run here; this pins the surrounding math contract).
# ---------------------------------------------------------------------------


def _online_softmax_merge(accs, ms, lses):
    """Reference merge (kept test-local; this is the deleted _merge_partials'
    invariant — same formula the GPU _fused_merge_kernel implements)."""
    accs_s = torch.stack(accs, dim=0).float()  # (S, ...)
    ms_s = torch.stack(ms, dim=0).float()
    lses_s = torch.stack(lses, dim=0).float()
    m_global = ms_s.amax(dim=0, keepdim=True)
    scale = torch.exp(ms_s - m_global)  # (S, h_kv, G)
    l_merged = (lses_s * scale).sum(dim=0)  # (h_kv, G)
    acc_merged = (accs_s * scale.unsqueeze(-1)).sum(dim=0)  # (h_kv, G, d)
    return acc_merged / l_merged.unsqueeze(-1)


def _split_partial(q_kv, k_split, v_split, scale):
    """Same math as _tail_partial, applied to a stored split's tokens (not
    literally the tail) — used here to build synthetic non-tail split partials."""
    return _tail_partial(q_kv, k_split, v_split, scale)


def test_slot_append_merge_equals_direct_attention_over_union():
    """Pin the design _finalize_decode relies on: splitting the KV sequence into
    N stored chunks + a tail chunk, computing one partial per chunk (the same
    _tail_partial math for every chunk, including the tail), and merging them
    via the online-softmax combine must equal computing dense attention directly
    over the concatenation of all chunks (stored ++ tail) — i.e. slot num_splits
    holding the tail is just "one more split" to the merge.
    """
    torch.manual_seed(1)
    h_kv, G, d = 2, 2, 16
    scale = 1.0 / (d**0.5)
    q_kv = torch.randn(h_kv, G, d)

    chunk_lens = [9, 11, 7]  # N "stored splits" ...
    tail_len = 5  # ... plus the tail, appended as slot N
    chunks_k = [torch.randn(h_kv, ln, d) for ln in chunk_lens]
    chunks_v = [torch.randn(h_kv, ln, d) for ln in chunk_lens]
    k_tail = torch.randn(h_kv, tail_len, d)
    v_tail = torch.randn(h_kv, tail_len, d)

    accs, ms, lses = [], [], []
    for kc, vc in zip(chunks_k, chunks_v):
        a, m, lse = _split_partial(q_kv, kc, vc, scale)
        accs.append(a)
        ms.append(m)
        lses.append(lse)
    # Tail written into the extra slot — same function, same call shape.
    a_t, m_t, lse_t = _tail_partial(q_kv, k_tail, v_tail, scale)
    accs.append(a_t)
    ms.append(m_t)
    lses.append(lse_t)

    merged = _online_softmax_merge(accs, ms, lses)  # (h_kv, G, d)

    # Direct reference: dense attention over the concatenation of every chunk.
    k_all = torch.cat(chunks_k + [k_tail], dim=1)  # (h_kv, sum(len)+tail, d)
    v_all = torch.cat(chunks_v + [v_tail], dim=1)
    s_all = torch.einsum("hgd,htd->hgt", q_kv, k_all) * scale
    p_all = torch.softmax(s_all, dim=-1)
    direct = torch.einsum("hgt,htd->hgd", p_all, v_all)

    assert torch.allclose(merged, direct, rtol=1e-5, atol=1e-5)


def test_slot_append_merge_num_splits_one_reduces_to_serial():
    """num_splits=1 + no tail: merge is exactly acc_0 / lse_0 (documented
    invariant, carried over from the deleted _merge_partials docstring)."""
    torch.manual_seed(2)
    h_kv, G, d, T = 1, 3, 8, 6
    scale = 1.0 / (d**0.5)
    q_kv = torch.randn(h_kv, G, d)
    k = torch.randn(h_kv, T, d)
    v = torch.randn(h_kv, T, d)
    acc, m, lse = _tail_partial(q_kv, k, v, scale)
    merged = _online_softmax_merge([acc], [m], [lse])
    assert torch.allclose(merged, acc / lse.unsqueeze(-1), rtol=1e-7, atol=1e-7)
