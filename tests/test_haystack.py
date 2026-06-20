from bmx.cache.haystack import PG_ESSAYS_DATASET, load_pg_corpus, synthetic_filler


def test_synthetic_filler_is_deterministic_and_scales():
    a = synthetic_filler(10)
    b = synthetic_filler(10)
    assert a == b
    assert len(synthetic_filler(20)) > len(a)
    assert isinstance(a, str) and len(a) > 0


def test_pg_essays_dataset_id():
    # The VM headline path pulls the real corpus from this HF dataset (no local clone).
    assert PG_ESSAYS_DATASET == "sgoel9/paul_graham_essays"


def test_load_pg_corpus_lazy_imports_datasets(monkeypatch):
    # load_pg_corpus must lazy-import `datasets` inside the function so importing the
    # module (and the offline/CI path) never triggers a download. Stub load_dataset to
    # prove the wiring without hitting the network.
    import datasets

    class _FakeDS:
        def __getitem__(self, col):
            assert col == "text"
            return ["essay one", "", "essay two"]

    monkeypatch.setattr(
        datasets, "load_dataset", lambda name, split: _FakeDS(), raising=True
    )
    corpus = load_pg_corpus()
    assert "essay one" in corpus and "essay two" in corpus
    # empty entries are dropped, real ones joined.
    assert corpus == "essay one\nessay two"
