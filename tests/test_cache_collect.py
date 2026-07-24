"""Tests for src/bmx/cache/collect.py — offline, tiny random models only.

Test idiom mirrors tests/test_layer_swap.py: build from config, no downloads.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from factories import ids as _ids
from factories import tiny_gpt2 as _tiny_gpt2
from factories import tiny_llama as _tiny_llama
from transformers import GPT2Config

from bmx.cache.collect import (
    collect_cache,
    from_matrix,
    load_cache,
    save_cache,
    to_matrix,
)


# ---------------------------------------------------------------------------
# Test 1: Shapes and keys — GPT-2 and Llama
# ---------------------------------------------------------------------------


def test_shapes_keys_gpt2():
    model = _tiny_gpt2()
    ids = _ids(seq=12)
    n_q_keep = 8

    cache = collect_cache(model, ids, n_q_keep=n_q_keep)

    n_layer = model.config.n_layer
    h = model.config.n_head
    h_kv = model.config.n_head  # GPT-2: h_kv == h
    S = ids.shape[1]
    d = model.config.n_embd // model.config.n_head

    for i in range(n_layer):
        assert f"layer{i}.k" in cache, f"missing layer{i}.k"
        assert f"layer{i}.v" in cache, f"missing layer{i}.v"
        assert f"layer{i}.q" in cache, f"missing layer{i}.q"
        assert f"layer{i}.k_pre" in cache, f"missing layer{i}.k_pre"

        k = cache[f"layer{i}.k"]
        v = cache[f"layer{i}.v"]
        q = cache[f"layer{i}.q"]
        k_pre = cache[f"layer{i}.k_pre"]

        assert k.shape == (h_kv, S, d), f"layer{i}.k shape {k.shape}"
        assert v.shape == (h_kv, S, d), f"layer{i}.v shape {v.shape}"
        assert q.shape == (h, min(n_q_keep, S), d), f"layer{i}.q shape {q.shape}"
        assert k_pre.shape == (h_kv, S, d), f"layer{i}.k_pre shape {k_pre.shape}"

        # All tensors stored in fp16
        assert k.dtype == torch.float16, f"layer{i}.k dtype {k.dtype}"
        assert v.dtype == torch.float16
        assert q.dtype == torch.float16
        assert k_pre.dtype == torch.float16


def test_shapes_keys_llama():
    model = _tiny_llama()
    ids = _ids(seq=12)
    n_q_keep = 5

    cache = collect_cache(model, ids, n_q_keep=n_q_keep)

    n_layer = model.config.num_hidden_layers
    h = model.config.num_attention_heads
    h_kv = model.config.num_key_value_heads
    S = ids.shape[1]
    d = model.config.hidden_size // model.config.num_attention_heads

    for i in range(n_layer):
        k = cache[f"layer{i}.k"]
        v = cache[f"layer{i}.v"]
        q = cache[f"layer{i}.q"]
        k_pre = cache[f"layer{i}.k_pre"]

        assert k.shape == (h_kv, S, d), f"layer{i}.k shape {k.shape}"
        assert v.shape == (h_kv, S, d), f"layer{i}.v shape {v.shape}"
        assert q.shape == (h, min(n_q_keep, S), d), f"layer{i}.q shape {q.shape}"
        assert k_pre.shape == (h_kv, S, d), f"layer{i}.k_pre shape {k_pre.shape}"

        assert k.dtype == torch.float16
        assert v.dtype == torch.float16
        assert q.dtype == torch.float16
        assert k_pre.dtype == torch.float16


def test_q_truncation_respects_n_q_keep():
    """n_q_keep larger than S gives q with S positions (no padding)."""
    model = _tiny_gpt2()
    ids = _ids(seq=6)
    S = 6
    n_q_keep = 100  # larger than S

    cache = collect_cache(model, ids, n_q_keep=n_q_keep)
    h = model.config.n_head
    d = model.config.n_embd // model.config.n_head

    q = cache["layer0.q"]
    assert q.shape == (h, S, d), f"expected (h={h}, S={S}, d={d}), got {q.shape}"


# ---------------------------------------------------------------------------
# Test 2: GPT-2 physics invariant — k_pre ≈ k (no RoPE)
# ---------------------------------------------------------------------------


def test_gpt2_kpre_equals_k():
    """GPT-2 has no RoPE; pre-RoPE key must equal post-RoPE key within fp16 noise."""
    model = _tiny_gpt2()
    ids = _ids(seq=16)
    cache = collect_cache(model, ids, n_q_keep=256)

    for i in range(model.config.n_layer):
        k = cache[f"layer{i}.k"].float()
        k_pre = cache[f"layer{i}.k_pre"].float()
        assert torch.allclose(k_pre, k, atol=1e-2), (
            f"layer{i}: k_pre != k; max abs diff = {(k_pre - k).abs().max():.4f}"
        )


# ---------------------------------------------------------------------------
# Test 3: Llama physics invariant — k_pre ≠ k but per-vector norms preserved
# ---------------------------------------------------------------------------


def test_llama_rope_norm_preserving():
    """RoPE is a rotation per head vector, so ||k_pre[h,t,:]|| == ||k[h,t,:]||."""
    model = _tiny_llama()
    ids = _ids(seq=16)
    cache = collect_cache(model, ids, n_q_keep=256)

    for i in range(model.config.num_hidden_layers):
        k = cache[f"layer{i}.k"].float()  # (h_kv, S, d)
        k_pre = cache[f"layer{i}.k_pre"].float()

        # They must differ (RoPE rotates)
        assert not torch.allclose(k_pre, k, atol=1e-3), (
            f"layer{i}: k_pre == k — RoPE was not applied?"
        )

        # Per-vector (per head, per token) norms must be preserved
        norm_k = k.norm(dim=-1)  # (h_kv, S)
        norm_pre = k_pre.norm(dim=-1)  # (h_kv, S)
        rel_diff = ((norm_k - norm_pre).abs() / norm_k.clamp(min=1e-8)).max()
        assert rel_diff < 1e-2, (
            f"layer{i}: RoPE changed vector norms; rel diff = {rel_diff:.4e}"
        )


# ---------------------------------------------------------------------------
# Test 4: save/load round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory,as_str", [(_tiny_gpt2, False), (_tiny_llama, True)])
def test_save_load_roundtrip(factory, as_str):
    cache = collect_cache(factory(), _ids(seq=10), n_q_keep=4)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.safetensors"
        save_cache(cache, str(path) if as_str else path)  # both path types accepted
        loaded = load_cache(str(path) if as_str else path)

    assert set(loaded.keys()) == set(cache.keys())
    for key in cache:
        assert torch.equal(cache[key], loaded[key]), f"round-trip mismatch on {key}"


def test_unsupported_architecture_raises():
    class Dummy:
        config = GPT2Config()

    with pytest.raises(ValueError, match="unsupported architecture"):
        collect_cache(Dummy(), torch.zeros(1, 4, dtype=torch.long))


# ---------------------------------------------------------------------------
# Test 5: K1 layout convention round-trip
# ---------------------------------------------------------------------------


def test_to_matrix_from_matrix_roundtrip():
    """from_matrix(to_matrix(kv), h) == kv.float() for an odd-shaped tensor."""
    h, S, d = 3, 7, 5  # deliberately odd / non-power-of-2 shape
    g = torch.Generator().manual_seed(123)
    kv = torch.randn(h, S, d, generator=g, dtype=torch.float16)

    M = to_matrix(kv)
    assert M.shape == (S, h * d)
    assert M.dtype == torch.float32

    kv_back = from_matrix(M, h)
    assert kv_back.shape == (h, S, d)
    assert torch.equal(kv_back, kv.float())


# ---------------------------------------------------------------------------
# Test 6: load_eval_tokens token_offset — multi-document calibration corpora
# ---------------------------------------------------------------------------


def test_load_eval_tokens_offset(monkeypatch):
    import bmx.eval.layer_swap as ls

    class _FakeTok:
        def __call__(self, text, return_tensors, truncation, max_length):
            import torch

            ids = torch.arange(max_length).unsqueeze(0)
            return type("E", (), {"input_ids": ids})()

    monkeypatch.setattr(
        "datasets.load_dataset", lambda *a, **k: {"text": ["x"]}, raising=False
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: _FakeTok()
    )
    base = ls.load_eval_tokens("gpt2", n_tokens=16)
    off = ls.load_eval_tokens("gpt2", n_tokens=16, token_offset=8)
    assert base.shape == off.shape == (16,)
    assert off[0].item() == base[0].item() + 8


# ---------------------------------------------------------------------------
# load_eval_tokens generalization — corpus passthrough + shuffled-token null
# ---------------------------------------------------------------------------


def _patch_eval_tokens_io(monkeypatch):
    """Fake tokenizer (arange ids) + fake dataset with 'text' and 'content'
    columns, so no download happens and byte-identity is checkable.

    Returns a recorder list of (args, kwargs) for every load_dataset call, so
    tests can pin WHICH dataset/config/split the defaults fetch (the arange
    tokenizer is blind to text content — without the recorder, a drifted
    default dataset_id/split would pass silently)."""

    calls: list[tuple[tuple, dict]] = []

    class _FakeTok:
        def __call__(self, text, return_tensors, truncation, max_length):
            import torch

            ids = torch.arange(max_length).unsqueeze(0)
            return type("E", (), {"input_ids": ids})()

    def _fake_load_dataset(*a, **k):
        calls.append((a, k))
        return {"text": ["x"], "content": ["y"]}

    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset, raising=False)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: _FakeTok()
    )
    return calls


def test_load_eval_tokens_generalized_defaults(monkeypatch):
    import bmx.eval.layer_swap as ls

    calls = _patch_eval_tokens_io(monkeypatch)
    base = ls.load_eval_tokens("gpt2", n_tokens=16, token_offset=8)
    # Dataset-identity pin: the default path must fetch EXACTLY the pre-change
    # dataset/config/split (the arange tokenizer can't see text, so this
    # recorder is what catches a drifted default dataset_id/split).
    assert calls == [(("Salesforce/wikitext", "wikitext-2-raw-v1"), {"split": "test"})]
    explicit = ls.load_eval_tokens(
        "gpt2",
        "wikitext-2-raw-v1",
        n_tokens=16,
        token_offset=8,
        dataset_id="Salesforce/wikitext",
        data_dir="",
        split="test",
        text_field="text",
        shuffle_seed=-1,
    )
    assert torch.equal(base, explicit)
    # pre-change behavior pin: arange slice starting at token_offset
    assert base.shape == (16,) and base[0].item() == 8


def test_load_eval_tokens_shuffle_after_slice(monkeypatch):
    import bmx.eval.layer_swap as ls

    _patch_eval_tokens_io(monkeypatch)
    nat = ls.load_eval_tokens("gpt2", n_tokens=16, token_offset=8)
    shuf1 = ls.load_eval_tokens(
        "gpt2", n_tokens=16, token_offset=8, shuffle_seed=20260723
    )
    shuf2 = ls.load_eval_tokens(
        "gpt2", n_tokens=16, token_offset=8, shuffle_seed=20260723
    )
    assert torch.equal(shuf1, shuf2)  # deterministic under the recorded seed
    assert not torch.equal(shuf1, nat)  # actually permuted
    # shuffle AFTER slicing: same token multiset as the natural slice
    assert torch.equal(shuf1.sort().values, nat.sort().values)


def test_load_eval_tokens_text_field_assert(monkeypatch):
    import bmx.eval.layer_swap as ls

    _patch_eval_tokens_io(monkeypatch)
    # 'content' exists in the fake dataset -> passes; 'nope' must assert.
    ok = ls.load_eval_tokens("gpt2", n_tokens=8, text_field="content")
    assert ok.shape == (8,)
    with pytest.raises(AssertionError, match="text_field"):
        ls.load_eval_tokens("gpt2", n_tokens=8, text_field="nope")


# ---------------------------------------------------------------------------
# load_eval_tokens capped-accumulation — must not join the ENTIRE text
# column before tokenizing (that allocated 24.5 GB on the 61,373-row
# codeparrot split). Correctness pin: the capped-accumulation result must be
# BYTE-IDENTICAL to the old full-join algorithm, for both a sufficient first
# pass and a case that forces the margin-doubling retry.
# ---------------------------------------------------------------------------


def _char_proportional_tokenizer(chars_per_token: int = 4):
    """A fake tokenizer where token count actually depends on text length
    (1 token per `chars_per_token` chars, deterministic), so
    prefix-sufficiency matters. Unlike the arange fake in
    _patch_eval_tokens_io (blind to text content and thus blind to
    under-accumulation bugs), this fake can distinguish 'tokenized a
    prefix' from 'tokenized the full text'.
    """

    class _CharPropTok:
        def __call__(self, text, return_tensors, truncation, max_length):
            n = min(len(text) // chars_per_token, max_length)
            ids = torch.arange(n).unsqueeze(0)
            return type("E", (), {"input_ids": ids})()

    return _CharPropTok()


def _patch_eval_tokens_rows(monkeypatch, rows, chars_per_token: int = 4):
    """Patch load_dataset to return the given list of text rows, and the
    tokenizer to the char-proportional fake."""
    monkeypatch.setattr(
        "datasets.load_dataset", lambda *a, **k: {"text": rows}, raising=False
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *a, **k: _char_proportional_tokenizer(chars_per_token),
    )


def _old_full_join_tokens(rows, n_tokens, token_offset, tok):
    """The pre-fix algorithm: join the ENTIRE column, then tokenize. Used
    in-test as the reference/expected value — NOT imported from production
    code (that's exactly the code path we're replacing)."""
    text = "\n\n".join(rows)
    ids = tok(
        text, return_tensors="pt", truncation=True, max_length=token_offset + n_tokens
    )
    return ids.input_ids[0][token_offset:]


class _PoisonedTail(list):
    """A list where indexing/iterating past `safe_upto` raises — proves the
    production code only ever touches a PREFIX of the rows, never the whole
    column (the actual defect: joining ALL 61,373 codeparrot rows before
    tokenizing, which allocated 24.5 GB)."""

    def __init__(self, rows, safe_upto):
        super().__init__(rows)
        self.safe_upto = safe_upto

    def __iter__(self):
        for i in range(len(self)):
            if i >= self.safe_upto:
                raise AssertionError(
                    f"row {i} touched but only a prefix < {self.safe_upto} "
                    "should ever be accumulated/joined"
                )
            yield list.__getitem__(self, i)

    def __getitem__(self, key):
        if isinstance(key, slice):
            stop = key.stop if key.stop is not None else len(self)
            if stop > self.safe_upto:
                raise AssertionError(
                    f"slice stop={stop} exceeds safe_upto={self.safe_upto}: "
                    "accumulation read past the needed prefix"
                )
            return _PoisonedTail(list.__getitem__(self, key), self.safe_upto)
        if key >= self.safe_upto:
            raise AssertionError(
                f"row {key} touched but only a prefix < {self.safe_upto} "
                "should ever be accumulated/joined"
            )
        return list.__getitem__(self, key)


def test_load_eval_tokens_does_not_touch_rows_past_needed_prefix(monkeypatch):
    """The core memory-safety property: with many more rows than needed,
    accumulation must stop well short of the full column. A dataset with
    10,000 rows requesting only 8 tokens must never read row 10 (each row
    is 40 chars => margin=16 chars/token covers 8 tokens in ~4 rows; give
    generous headroom at row 10 to avoid a flaky off-by-one, while row 9990
    would blow the old full-join allocation on a real 24.5 GB-sized split)."""
    import bmx.eval.layer_swap as ls

    rows = _PoisonedTail(
        [f"row {i:05d} padded to forty characters!!" for i in range(10_000)],
        safe_upto=10,
    )
    _patch_eval_tokens_rows(monkeypatch, rows)

    got = ls.load_eval_tokens("gpt2", n_tokens=8, token_offset=0)
    assert got.shape == (8,)


class _FakeHFDataset:
    """A fake mimicking the real HF Dataset access pattern: len(ds) is cheap
    metadata, column_names is cheap metadata, and the text column is only
    reachable via .select(range(k))[text_field] — no whole-column getitem.
    Records the max row index any .select call ever requested, so the test
    can assert the production code only ever asks for a row-bounded prefix,
    not len(ds) rows."""

    def __init__(self, rows):
        self._rows = rows
        self.column_names = ["text"]
        self.max_selected = -1

    def __len__(self):
        return len(self._rows)

    def select(self, indices):
        indices = list(indices)
        if indices:
            self.max_selected = max(self.max_selected, max(indices))
        return {"text": [self._rows[i] for i in indices]}


def test_load_eval_tokens_select_bounded_to_needed_prefix(monkeypatch):
    """With a real-Dataset-shaped fake (.select-based access, no whole-column
    getitem), the max row index ever passed to .select must stay far below
    len(ds) for a small n_tokens request — proving the row-bounded .select
    path is actually used, not just tolerated."""
    import bmx.eval.layer_swap as ls

    n_rows = 10_000
    ds = _FakeHFDataset(
        [f"row {i:05d} padded to forty characters!!" for i in range(n_rows)]
    )
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: ds, raising=False)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *a, **k: _char_proportional_tokenizer(),
    )

    got = ls.load_eval_tokens("gpt2", n_tokens=8, token_offset=0)
    assert got.shape == (8,)
    assert ds.max_selected >= 0, "select() was never called"
    # need_chars = 8*16=128 over 40-char rows needs ~4 rows; generous headroom
    # well short of the full 10,000-row column.
    assert ds.max_selected < 50, (
        f"max_selected={ds.max_selected} materialized far more rows than the "
        "needed prefix"
    )


def test_load_eval_tokens_capped_accumulation_matches_full_join(monkeypatch):
    """~10 short rows, sufficient margin on the first pass: capped
    accumulation must equal the old full-join result exactly."""
    import bmx.eval.layer_swap as ls

    rows = [f"row {i} has some words in it" for i in range(10)]
    _patch_eval_tokens_rows(monkeypatch, rows)

    n_tokens, token_offset = 8, 2
    got = ls.load_eval_tokens("gpt2", n_tokens=n_tokens, token_offset=token_offset)
    expected = _old_full_join_tokens(
        rows, n_tokens, token_offset, _char_proportional_tokenizer()
    )
    assert torch.equal(got, expected)
    assert got.shape == (n_tokens,)


def test_load_eval_tokens_capped_accumulation_retries_when_undercovered(monkeypatch):
    """A tokenizer that is DENSER than the margin=16 chars/token assumption
    (here: 1 token per 30 chars, i.e. code-like — real code is denser in
    tokens/char than the margin's generous wikitext-calibrated assumption of
    16): the first accumulation pass under-covers the requested token
    budget, forcing at least one margin-doubling retry. Must still land on
    the full-join-equivalent result — the retry loop is transparent to
    correctness, only to how much text got joined along the way."""
    import bmx.eval.layer_swap as ls

    chars_per_token = 30
    rows = [f"row number {i:04d} of the corpus text body" for i in range(400)]
    _patch_eval_tokens_rows(monkeypatch, rows, chars_per_token=chars_per_token)

    n_tokens, token_offset = 300, 10
    got = ls.load_eval_tokens("gpt2", n_tokens=n_tokens, token_offset=token_offset)
    expected = _old_full_join_tokens(
        rows,
        n_tokens,
        token_offset,
        _char_proportional_tokenizer(chars_per_token),
    )
    assert torch.equal(got, expected)
    assert got.shape == (n_tokens,)


# ---------------------------------------------------------------------------
# collect_cache.py corpus passthrough — output naming + corpus-default guard
# ---------------------------------------------------------------------------


def test_collect_cache_out_path_and_corpus_guard():
    from experiments.collect_cache import Config, _corpus_is_default, _out_path

    default = Config(model_name="gpt2", seq_len=1024)
    assert _corpus_is_default(default)
    assert _out_path(default).name == "gpt2_1024.safetensors"

    off = Config(model_name="gpt2", seq_len=1024, token_offset=1024)
    # pre-change names unchanged: existing wikitext caches stay reusable as-is
    assert _out_path(off).name == "gpt2_1024_off1024.safetensors"

    code = Config(
        model_name="gpt2",
        seq_len=1024,
        token_offset=1024,
        dataset_id="bigcode/the-stack-smol",
        data_dir="data/python",
        split="train",
        text_field="content",
        corpus_label="code",
    )
    assert not _corpus_is_default(code)
    assert _out_path(code).name == "gpt2_1024_code_off1024.safetensors"

    shuf = Config(
        model_name="gpt2",
        seq_len=1024,
        token_offset=1024,
        shuffle_seed=20260723,
        corpus_label="shuf",
    )
    assert not _corpus_is_default(shuf)
    assert _out_path(shuf).name == "gpt2_1024_shuf_off1024.safetensors"

    # --out still overrides everything
    assert _out_path(
        Config(model_name="gpt2", seq_len=1024, out="x/y.safetensors")
    ) == Path("x/y.safetensors")


def test_collect_cache_main_guard_raises_before_model_load(monkeypatch):
    """Non-default corpus knobs without --corpus-label must raise
    AssertionError, and must do so BEFORE any model load. Monkeypatch the
    model loader to fail loudly if called, so the test pins the ordering
    (guard-then-load), not just the guard's existence."""
    from experiments.collect_cache import Config, main

    def _fail_if_called(*a, **k):
        raise AssertionError("model loader was called — guard did not fire first")

    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", _fail_if_called
    )
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", _fail_if_called)
    monkeypatch.setattr("datasets.load_dataset", _fail_if_called, raising=False)

    cfg = Config(
        model_name="gpt2",
        seq_len=1024,
        dataset_id="bigcode/the-stack-smol",
        corpus_label="",  # missing on purpose: dataset_id is non-default
    )
    with pytest.raises(AssertionError, match="corpus-label"):
        main(cfg)
