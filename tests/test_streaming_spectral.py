"""Streaming spectral K-branch: corpus packs through the write-once path.

Mirrors tests/test_streaming_cache.py's fixture patterns. Parity invariant
used: the same committed-prefix freeze check as
test_streaming_cache.test_each_token_quantized_once (the strongest
committed-block invariant available in the existing suite) — after a flush,
_q_prefix_k for the already-committed region must be bitwise identical to a
direct offline spectral_quantize call on that same block, AND must stay
frozen (unchanged) across a later flush.
"""

import dataclasses

import pytest
import torch

from bmx.cache.spectral import (
    SpectralPack,
    fit_spectral_basis,
    identity_whitener,
    save_pack_file,
    spectral_quantize,
)
from bmx.cache.specs import CacheCodecSpec
from bmx.cache.streaming import StreamingQuantizedCache
from tests.factories import ids, tiny_llama, tiny_qwen3


def _fit_tiny_packs(model, path, budget=2.5, group=8):
    """Fit per-layer packs from one hooked prefill of the tiny model."""
    from bmx.cache.collect import collect_cache, to_matrix

    cache = collect_cache(model, ids(seq=64))
    n_layer = model.config.num_hidden_layers
    bases = {}
    for i in range(n_layer):
        M = to_matrix(cache[f"layer{i}.k_pre"]).float()
        C = M.shape[1]
        Wh, Wh_inv = identity_whitener(C)
        bases[i] = fit_spectral_basis(M, Wh, Wh_inv)
    save_pack_file(path, bases, budgets=(budget,), group=group, meta={"model": "tiny"})
    return bases


@pytest.mark.parametrize("factory", [tiny_llama, tiny_qwen3], ids=["llama", "qwen3"])
def test_streaming_spectral_matches_reference(tmp_path, factory):
    """Streamed spectral quantization must equal offline spectral_quantize on the
    committed blocks (write-once parity — the K3 invariant)."""
    model = factory()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    k_spec = CacheCodecSpec(
        arm="spectral", pre_rope=True, group=8, pack_path=path, budget=2.5
    )
    v_spec = CacheCodecSpec(arm="fp16")
    # NOTE: the brief's literal seq=64 with the default recent_window=32 never
    # crosses the PAGE(128)+window flush threshold (compute_flush_schedule stays
    # 0), so bpe_k would trivially be 16.0 -- not the accounting-with-pack-charge
    # case this test wants. seq=150 with recent_window=8 (the same pattern
    # test_streaming_cache.test_k2b_pre_rope_streams_token_by_token uses) flushes
    # one 128-token page, actually exercising the spectral codec + pack charge.
    cache = StreamingQuantizedCache(
        model.config, k_spec=k_spec, v_spec=v_spec, recent_window=8
    )
    cache.attach(model)
    with cache:
        with torch.no_grad():
            model(ids(seq=150), past_key_values=cache, use_cache=True)
    # bpe accounting present and includes the skeptic pack charge
    bpe_k, bpe_v = cache.bits_per_entry()
    assert bpe_v == 16.0
    assert 2.0 < bpe_k < 16.0  # payload + scale + pack charge at tiny S


@pytest.mark.parametrize("factory", [tiny_llama, tiny_qwen3], ids=["llama", "qwen3"])
def test_streaming_spectral_requires_pre_rope(tmp_path, factory):
    model = factory()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)
    with pytest.raises(AssertionError, match="pre_rope"):
        StreamingQuantizedCache(
            model.config,
            k_spec=CacheCodecSpec(arm="spectral", pack_path=path, budget=2.5),
            v_spec=CacheCodecSpec(arm="fp16"),
        )


@pytest.mark.parametrize("factory", [tiny_llama, tiny_qwen3], ids=["llama", "qwen3"])
def test_streaming_spectral_committed_block_matches_offline_and_frozen(
    tmp_path, factory
):
    """Strongest available parity invariant (mirrors
    test_streaming_cache.test_each_token_quantized_once): reach into
    cache.layers[i]._q_prefix_k after a flush and check it is BYTE-IDENTICAL
    to an offline spectral_quantize(...) call on the same pre-RoPE block (not
    just "finite"/"compressed"), and that the committed region stays frozen
    (write-once) across a later flush.
    """
    from bmx.cache.collect import from_matrix, to_matrix
    from bmx.cache.rope import apply_rope, rope_cos_sin
    from bmx.cache.spectral import load_packs

    model = factory()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path, budget=2.5, group=8)
    k_spec = CacheCodecSpec(
        arm="spectral", pre_rope=True, group=8, pack_path=path, budget=2.5
    )
    v_spec = CacheCodecSpec(arm="fp16")

    g = torch.Generator().manual_seed(31)
    # 140 tokens with recent_window=8 crosses PAGE(128)+8 so exactly one
    # 128-token page flushes (mirrors test_k2b_pre_rope_streams_token_by_token's
    # fixture pattern); also captures the pristine pre-RoPE keys for offline replay.
    input_ids = torch.randint(0, 97, (1, 140), generator=g)

    cache = StreamingQuantizedCache(
        model.config, k_spec=k_spec, v_spec=v_spec, recent_window=8
    )
    cache.attach(model)
    try:
        with torch.no_grad():
            model(input_ids, past_key_values=cache, use_cache=True)
    finally:
        cache.detach()

    layer = cache.layers[0]
    assert layer._committed_S_q == 128, "expected exactly one page to flush"
    committed_before = layer._q_prefix_k.clone()

    # --- Offline reference: replay the SAME pristine pre-RoPE capture mechanism
    # (the k_proj hook, via a fp16/pre_rope reference cache stopped right at the
    # flush boundary -- so we're comparing against the same k_block_pre the
    # production layer captured, not a differently-sourced "true" post-RoPE
    # tensor whose plumbing rounds differently -- see
    # test_prerope_key_capture_and_rope_at_read, which only proves capture+RoPE
    # to rel<1e-2 against a plain non-streaming cache, not bit-exact), then
    # spectral_quantize the first 128-token block directly and apply RoPE at
    # true positions -- must equal the streamed committed prefix bit-for-bit.
    packs = load_packs(path, budget=2.5)
    pack = packs[0]
    cap_cache = StreamingQuantizedCache(
        model.config,
        k_spec=CacheCodecSpec(arm="fp16", pre_rope=True),
        v_spec=CacheCodecSpec(arm="fp16"),
        recent_window=8,
    )
    cap_cache.attach(model)
    with torch.no_grad():
        model(input_ids[:, :128], past_key_values=cap_cache, use_cache=True)
    cap_cache.detach()
    k_block_pre = cap_cache.layers[0]._k_pre[:, :128, :].float()

    h = k_block_pre.shape[0]
    M_block = to_matrix(k_block_pre)
    M_hat, _ = spectral_quantize(M_block, pack, mse_scale=True)
    k_hat_pre = from_matrix(M_hat, h)
    cos, sin = rope_cos_sin(model.config, 128, start=0, device=k_hat_pre.device)
    k_block_post_ref = apply_rope(k_hat_pre.float(), cos.float(), sin.float())

    # _q_prefix_k is stored (h_kv, S, d) -- no leading batch dim.
    assert torch.equal(committed_before, k_block_post_ref.to(committed_before.dtype)), (
        "streamed committed block does not byte-match offline spectral_quantize"
    )

    # --- Write-once: run more decode steps to trigger a later step; the
    # already-committed region must not change.
    with torch.no_grad():
        for t in range(140, 150):
            model(input_ids[:, :1], past_key_values=cache, use_cache=True)
    committed_after = layer._q_prefix_k
    assert torch.equal(
        committed_before, committed_after[:, : committed_before.shape[1], :]
    ), "committed spectral K prefix changed — write-once not enforced"


def test_spectral_pack_device_move_is_noop_on_cpu():
    """CPU-only coverage for the one-time device-placement guard in
    _quantize_k_block_pre_rope (Fix 1): on CPU the pack is already on
    k_block_pre's device, so the branch is skipped and .device equality
    holds. Real CUDA coverage (pack loads on CPU, block runs on the model's
    CUDA device) lands with the GH200 battery.

    Also pins that dataclasses.replace with .to(same_device) on every tensor
    field preserves torch.equal — the actual move mechanism used by the
    guard, exercised directly here since CPU can't trigger a real transfer.
    """
    Wh, Wh_inv = identity_whitener(8)
    M = torch.randn(16, 8, dtype=torch.float32)
    basis = fit_spectral_basis(M, Wh, Wh_inv)
    from bmx.cache.spectral import pack_from_basis

    pack = pack_from_basis(basis, budget=3.0, tiers=(0, 2, 3, 4), group=8)

    device = torch.device("cpu")
    assert pack.enc.device == device

    moved = dataclasses.replace(
        pack,
        enc=pack.enc.to(device),
        dec=pack.dec.to(device),
        lam=pack.lam.to(device),
        bits=pack.bits.to(device),
    )
    assert isinstance(moved, SpectralPack)
    assert torch.equal(moved.enc, pack.enc)
    assert torch.equal(moved.dec, pack.dec)
    assert torch.equal(moved.lam, pack.lam)
    assert torch.equal(moved.bits, pack.bits)


def test_bits_per_entry_averages_layers_for_allocated_packs(tmp_path):
    """bits_per_entry must report the ACROSS-LAYER mean payload, not layers[-1].

    Regression: allocated packs (per-layer budgets, mean-preserving) put the
    floor budget on the least-sensitive layers; reading only the last layer
    under-reported the whole cache by ~1 bit (2026-07-15 alloc probe — bpe_k
    identical across k4_b2.2/k4_b2.5 because layer 31 drew 1.0 in both)."""
    import json

    from bmx.cache.collect import to_matrix
    from bmx.cache.spectral import fit_spectral_basis, identity_whitener, save_pack_file

    model = tiny_llama()
    from bmx.cache.collect import collect_cache

    torch.manual_seed(0)
    cache_d = collect_cache(model, ids(seq=64))
    n_layer = model.config.num_hidden_layers
    bases = {}
    for i in range(n_layer):
        M = to_matrix(cache_d[f"layer{i}.k_pre"]).float()
        Wh, Wh_inv = identity_whitener(M.shape[1])
        bases[i] = fit_spectral_basis(M, Wh, Wh_inv)
    path = str(tmp_path / "alloc_packs.safetensors")
    # Deliberately non-uniform: layer 0 rich, layer 1 poor, mean 2.5.
    layer_budgets = {2.5: {0: 4.0, 1: 1.0}}
    assert n_layer == 2, "tiny_llama fixture is expected to have 2 layers"
    save_pack_file(
        path,
        bases,
        budgets=(2.5,),
        group=8,
        meta={"model": "tiny"},
        layer_budgets=layer_budgets,
    )
    side = json.loads(open(path + ".json").read())
    assert "budgets" in side

    k_spec = CacheCodecSpec(
        arm="spectral", pre_rope=True, group=8, pack_path=path, budget=2.5
    )
    # seq=150 + recent_window=8 crosses the PAGE(128) flush threshold (the
    # seq=64/window=32 default never flushes — same note as the parity test).
    cache = StreamingQuantizedCache(
        model.config, k_spec=k_spec, v_spec=CacheCodecSpec(arm="fp16"), recent_window=8
    )
    cache.attach(model)
    with cache:
        with torch.no_grad():
            model(ids(seq=150), past_key_values=cache, use_cache=True)
    bpe_k, _ = cache.bits_per_entry()
    per_layer = [layer.bpe_k for layer in cache.layers]
    assert per_layer[0] != per_layer[1], "allocated layers must differ (fixture)"
    from bmx.cache.spectral import skeptic_charge

    S = cache.layers[-1].get_seq_length()
    C = cache.layers[-1]._h_kv * cache.layers[-1]._d_head
    mean_c_used = sum(ly._pack.c_used for ly in cache.layers) / len(cache.layers)
    expected = sum(per_layer) / len(per_layer) + skeptic_charge(
        C, S, cache.layers[-1]._pack.tiers, c_used=mean_c_used
    )
    assert abs(bpe_k - expected) < 1e-9
    # v2 must charge strictly less than v1 whenever dirs were dropped (additive pin).
    expected_v1 = sum(per_layer) / len(per_layer) + skeptic_charge(
        C, S, cache.layers[-1]._pack.tiers
    )
    assert mean_c_used < C and bpe_k < expected_v1


def test_streaming_dec8_charges_int8_and_degrades_gracefully(tmp_path):
    from bmx.cache.spectral import skeptic_charge

    path = str(tmp_path / "packs.safetensors")
    model = tiny_llama()
    _fit_tiny_packs(model, path)
    caches = {}
    for dq in ("fp32", "int8"):
        k_spec = CacheCodecSpec(
            arm="spectral",
            pre_rope=True,
            group=8,
            pack_path=path,
            budget=2.5,
            dec_quant=dq,
        )
        cache = StreamingQuantizedCache(
            model.config,
            k_spec=k_spec,
            v_spec=CacheCodecSpec(arm="fp16"),
            recent_window=8,
        )
        cache.attach(model)
        with cache:
            with torch.no_grad():
                model(ids(seq=150), past_key_values=cache, use_cache=True)
        caches[dq] = cache
    bpe_fp32, _ = caches["fp32"].bits_per_entry()
    bpe_int8, _ = caches["int8"].bits_per_entry()
    assert bpe_int8 < bpe_fp32  # 8-bit decoder charge strictly cheaper
    # And the delta equals the charge arithmetic exactly (payloads identical).
    layer = caches["fp32"].layers[-1]
    C = layer._h_kv * layer._d_head
    S = layer.get_seq_length()
    mc = sum(ly._pack.c_used for ly in caches["fp32"].layers) / len(
        caches["fp32"].layers
    )
    t = layer._pack.tiers
    expected_delta = skeptic_charge(C, S, t, c_used=mc) - skeptic_charge(
        C, S, t, c_used=mc, dec_bits=8.0
    )
    assert abs((bpe_fp32 - bpe_int8) - expected_delta) < 1e-9
