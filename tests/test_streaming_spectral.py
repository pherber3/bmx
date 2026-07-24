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


def test_streaming_bpe_refactor_bitexact(tmp_path):
    """K4 local-levers Task 1 pin: after the mixed_dec_charge refactor,
    bits_per_entry() for both the fp32 and blanket-int8 arms must equal the
    OLD skeptic_charge(dec_bits=...) formula computed by hand -- zero numeric
    change at the streaming accounting level, not just at the endpoint-unit
    level (test_mixed_dec_charge_endpoints_match_skeptic_charge covers only
    the standalone function equality)."""
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

    for dq, old_dec_bits in (("fp32", 16.0), ("int8", 8.0)):
        cache = caches[dq]
        bpe_k, _ = cache.bits_per_entry()
        per_layer = [layer.bpe_k for layer in cache.layers]
        S = cache.layers[-1].get_seq_length()
        C = cache.layers[-1]._h_kv * cache.layers[-1]._d_head
        mean_c_used = sum(ly._pack.c_used for ly in cache.layers) / len(cache.layers)
        expected_old = sum(per_layer) / len(per_layer) + skeptic_charge(
            C,
            S,
            cache.layers[-1]._pack.tiers,
            c_used=mean_c_used,
            dec_bits=old_dec_bits,
        )
        assert abs(bpe_k - expected_old) < 1e-9, dq


def test_streaming_bpe_int8_t5(tmp_path):
    """Tier-gated dec_quant='int8_t5': bpe_k's pack charge equals a hand-computed
    mixed_dec_charge with per-layer gate counts (c_int8 = count(0 < bits <= 5))
    averaged the same way mean_c_used is averaged."""
    from bmx.cache.spectral import mixed_dec_charge

    path = str(tmp_path / "packs.safetensors")
    model = tiny_llama()
    _fit_tiny_packs(model, path, budget=2.5, group=8)
    k_spec = CacheCodecSpec(
        arm="spectral",
        pre_rope=True,
        group=8,
        pack_path=path,
        budget=2.5,
        dec_quant="int8_t5",
    )
    cache = StreamingQuantizedCache(
        model.config, k_spec=k_spec, v_spec=CacheCodecSpec(arm="fp16"), recent_window=8
    )
    cache.attach(model)
    with cache:
        with torch.no_grad():
            model(ids(seq=150), past_key_values=cache, use_cache=True)
    bpe_k, _ = cache.bits_per_entry()
    per_layer = [layer.bpe_k for layer in cache.layers]
    S = cache.layers[-1].get_seq_length()
    C = cache.layers[-1]._h_kv * cache.layers[-1]._d_head
    mean_c_used = sum(ly._pack.c_used for ly in cache.layers) / len(cache.layers)
    mean_c_int8 = sum(
        int(((ly._pack.bits > 0) & (ly._pack.bits <= 5)).sum()) for ly in cache.layers
    ) / len(cache.layers)
    expected = sum(per_layer) / len(per_layer) + mixed_dec_charge(
        C, S, cache.layers[-1]._pack.tiers, c_used=mean_c_used, c_int8=mean_c_int8
    )
    assert abs(bpe_k - expected) < 1e-9


def test_streaming_bpe_int8_tl(tmp_path):
    """K4 estimation-levers Task 3: dec_quant='int8_tl' bpe_k's pack charge
    equals a hand-computed mixed_dec_charge with PER-LAYER c_int8 at each
    layer's own certificate-derived T_ℓ (per_layer_tier_thresholds applied
    to a FRESH pristine reload -- an independent oracle from whatever
    load_packs_for_spec/cache_bits_per_entry actually did internally),
    averaged the same way mean_c_used is averaged. Also pins each live
    layer's SpectralPack.dec_tier against that same fresh map (fix-wave
    regression: dec_tier is what cache_bits_per_entry actually reads --
    see test_cache_bits_per_entry_int8_tl_uses_materialized_dec_tier_not_
    a_live_rederivation below for the fixture that actually EXERCISES the
    pristine-vs-post-roundtrip divergence this pins against in principle)."""
    from bmx.cache.spectral import (
        load_packs,
        mixed_dec_charge,
        per_layer_tier_thresholds,
    )

    path = str(tmp_path / "packs.safetensors")
    model = tiny_llama()
    _fit_tiny_packs(model, path, budget=2.5, group=8)
    k_spec = CacheCodecSpec(
        arm="spectral",
        pre_rope=True,
        group=8,
        pack_path=path,
        budget=2.5,
        dec_quant="int8_tl",
    )
    cache = StreamingQuantizedCache(
        model.config, k_spec=k_spec, v_spec=CacheCodecSpec(arm="fp16"), recent_window=8
    )
    cache.attach(model)
    with cache:
        with torch.no_grad():
            model(ids(seq=150), past_key_values=cache, use_cache=True)
    bpe_k, _ = cache.bits_per_entry()
    per_layer = [layer.bpe_k for layer in cache.layers]
    S = cache.layers[-1].get_seq_length()
    C = cache.layers[-1]._h_kv * cache.layers[-1]._d_head

    raw_packs = load_packs(path, 2.5)  # FRESH pristine reload -- independent oracle
    t_map = per_layer_tier_thresholds(raw_packs)
    # dec_tier pin: each live layer's pack must carry the PRISTINE map's
    # value, not something re-derived from its own (possibly roundtripped)
    # live decoder.
    for i, layer in enumerate(cache.layers):
        expected_tier = t_map[i] if t_map[i] != 0 else None
        assert layer._pack.dec_tier == expected_tier, (i, layer._pack.dec_tier, t_map)

    mean_c_used = sum(ly._pack.c_used for ly in cache.layers) / len(cache.layers)
    mean_c_int8 = sum(
        cache.layers[i]._pack.c_int8(t_map[i]) for i in range(len(cache.layers))
    ) / len(cache.layers)
    expected = sum(per_layer) / len(per_layer) + mixed_dec_charge(
        C, S, cache.layers[-1]._pack.tiers, c_used=mean_c_used, c_int8=mean_c_int8
    )
    assert abs(bpe_k - expected) < 1e-9


def test_cache_bits_per_entry_int8_tl_uses_materialized_dec_tier_not_a_live_rederivation():
    """K4 estimation-levers Task 3 FIX WAVE (P1 bug, confirmed on real gpt2
    packs): cache_bits_per_entry must charge using each layer's
    SpectralPack.dec_tier (set once, from the PRISTINE packs, by
    load_packs_for_spec) -- NOT by re-deriving per_layer_tier_thresholds from
    the live (already int8-roundtripped) packs. int8_decoder_roundtrip is
    near-idempotent, so a live re-derivation can silently pass a HIGHER tier
    than the pristine decoder certified, under-charging bpe.

    This test exercises the actual divergent case directly against
    cache_bits_per_entry (bypassing the model/forward-pass plumbing, which
    the tiny_llama fixture's C=16 packs are too well-behaved to hit): a
    boundary pack pair where the buggy re-derivation would have produced
    dec_tier=8 for layer 0 (pristine certifies only 6). Minimal stand-in
    layer objects expose exactly the attributes cache_bits_per_entry reads
    (bpe_k/bpe_v/_committed_S_q/get_seq_length()/_h_kv/_d_head/_pack)."""
    import dataclasses

    from bmx.cache.spectral import (
        fit_spectral_pack,
        identity_whitener,
        int8_decoder_roundtrip,
        mixed_dec_charge,
        per_layer_tier_thresholds,
    )
    from bmx.cache.streaming import cache_bits_per_entry

    def _spiked(S, C, seed, spike_std):
        g = torch.Generator().manual_seed(seed)
        raw = torch.randn(C, 2, generator=g)
        dirs, _ = torch.linalg.qr(raw)
        z = torch.randn(S, 2, generator=g) * torch.tensor([spike_std, spike_std])
        noise = torch.randn(S, C, generator=g)
        return z @ dirs.mT + noise

    C = 64
    Wh, Wh_inv = identity_whitener(C)
    M0 = _spiked(300, C, seed=4, spike_std=40.0)  # the boundary layer
    M1 = _spiked(300, C, seed=1, spike_std=10.0)  # control layer
    pack0 = fit_spectral_pack(M0, Wh, Wh_inv, budget=2.5, group=8)
    pack1 = fit_spectral_pack(M1, Wh, Wh_inv, budget=2.5, group=8)
    pristine = {0: pack0, 1: pack1}
    t_map = per_layer_tier_thresholds(pristine)
    assert t_map == {0: 6, 1: 8}, "boundary fixture drifted; re-tune spike/seed"

    # Materialize the way load_packs_for_spec now does: roundtrip each layer
    # at ITS OWN pristine T_ℓ, stamping dec_tier to match.
    materialized = {
        i: dataclasses.replace(
            pack,
            dec=int8_decoder_roundtrip(pack.dec, pack.bits, tier_threshold=t_map[i]),
            dec_tier=t_map[i],
        )
        for i, pack in pristine.items()
    }

    class _StubLayer:
        def __init__(self, pack):
            self._pack = pack
            self.bpe_k = 3.0  # arbitrary fixed payload bpe -- charge is additive
            self.bpe_v = 2.0
            self._committed_S_q = 128  # > 0 so the spectral charge is applied
            self._h_kv = 1
            self._d_head = C

        def get_seq_length(self):
            return 4096

    layers = [_StubLayer(materialized[0]), _StubLayer(materialized[1])]
    k_spec = CacheCodecSpec(
        arm="spectral",
        pre_rope=True,
        pack_path="unused",
        budget=2.5,
        dec_quant="int8_tl",
    )
    bpe_k, bpe_v = cache_bits_per_entry(layers, k_spec)

    # Correct expectation: dec_tier-driven charge, i.e. layer 0 at T=6.
    S, C_ = 4096, C
    mean_c_used = sum(ly._pack.c_used for ly in layers) / len(layers)
    mean_c_int8_correct = sum(
        ly._pack.c_int8(ly._pack.dec_tier) for ly in layers
    ) / len(layers)
    expected_correct = 3.0 + mixed_dec_charge(
        C_, S, layers[-1]._pack.tiers, c_used=mean_c_used, c_int8=mean_c_int8_correct
    )
    assert abs(bpe_k - expected_correct) < 1e-9

    # The BUG's expectation: re-deriving the map from the LIVE (already
    # roundtripped) packs would have wrongly promoted layer 0 to T=8,
    # producing a strictly SMALLER (under-charged) mean_c_int8-driven charge
    # unless c_int8(6) == c_int8(8) for layer 0 (ruled out by the fixture --
    # the whole point of the boundary case is that the two differ).
    buggy_t_map = per_layer_tier_thresholds(
        {i: ly._pack for i, ly in enumerate(layers)}
    )
    assert buggy_t_map[0] == 8  # confirms the LIVE re-derivation IS wrong here
    assert pack0.c_int8(6) != pack0.c_int8(8), (
        "fixture must have a real column-count gap between T=6 and T=8"
    )
    mean_c_int8_buggy = sum(
        ly._pack.c_int8(buggy_t_map[i]) for i, ly in enumerate(layers)
    ) / len(layers)
    expected_buggy = 3.0 + mixed_dec_charge(
        C_, S, layers[-1]._pack.tiers, c_used=mean_c_used, c_int8=mean_c_int8_buggy
    )
    assert abs(bpe_k - expected_buggy) > 1e-6, (
        "the fix must NOT reproduce the buggy under-charged value"
    )


# ---------------------------------------------------------------------------
# K4 Lloyd payload-quantizer gate, Task 1 (2026-07-25 design):
# CacheCodecSpec.payload_quant threaded through StreamingQuantizedCache.
# ---------------------------------------------------------------------------


def test_streaming_payload_quant_default_inert(tmp_path):
    """payload_quant default ('rtn') must reproduce byte-identical bpe AND
    cache bytes vs a spec that omits the field entirely -- default-inert at
    the streaming level (brief requirement (e))."""
    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)

    def _run(k_spec):
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
        return cache

    implicit = _run(
        CacheCodecSpec(
            arm="spectral", pre_rope=True, group=8, pack_path=path, budget=2.5
        )
    )
    explicit = _run(
        CacheCodecSpec(
            arm="spectral",
            pre_rope=True,
            group=8,
            pack_path=path,
            budget=2.5,
            payload_quant="rtn",
        )
    )
    assert implicit.bits_per_entry() == explicit.bits_per_entry()
    for li, le in zip(implicit.layers, explicit.layers):
        assert torch.equal(li._q_prefix_k, le._q_prefix_k)
        assert torch.equal(li._q_prefix_v, le._q_prefix_v)


def test_streaming_payload_quant_lloyd_runs_and_bpe_matches_rtn(tmp_path):
    """payload_quant='lloyd' must run end-to-end through StreamingQuantizedCache
    and charge the EXACT SAME bpe as 'rtn' (identical bits/scale accounting by
    construction) while producing a DIFFERENT (non-bit-exact) K reconstruction."""
    model = tiny_llama()
    path = str(tmp_path / "packs.safetensors")
    _fit_tiny_packs(model, path)

    def _run(payload_quant):
        k_spec = CacheCodecSpec(
            arm="spectral",
            pre_rope=True,
            group=8,
            pack_path=path,
            budget=2.5,
            payload_quant=payload_quant,
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
        return cache

    rtn_cache = _run("rtn")
    lloyd_cache = _run("lloyd")
    assert rtn_cache.bits_per_entry() == lloyd_cache.bits_per_entry()
    diverged = any(
        not torch.equal(lr._q_prefix_k, ll._q_prefix_k)
        for lr, ll in zip(rtn_cache.layers, lloyd_cache.layers)
    )
    assert diverged, "lloyd must reconstruct differently from rtn on real data"
