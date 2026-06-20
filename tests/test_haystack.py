from bmx.cache.haystack import synthetic_filler, pg_essays_dir, read_pg_corpus


def test_synthetic_filler_is_deterministic_and_scales():
    a = synthetic_filler(10)
    b = synthetic_filler(10)
    assert a == b
    assert len(synthetic_filler(20)) > len(a)
    assert isinstance(a, str) and len(a) > 0


def test_pg_essays_dir_returns_path_or_none():
    d = pg_essays_dir()
    # In this repo the clone is present; if absent (CI elsewhere) None is allowed.
    assert d is None or (d.is_dir() and any(d.glob("*.txt")))


def test_read_pg_corpus_concatenates(tmp_path):
    (tmp_path / "a.txt").write_text("alpha ")
    (tmp_path / "b.txt").write_text("beta")
    corpus = read_pg_corpus(tmp_path)
    assert "alpha" in corpus and "beta" in corpus
