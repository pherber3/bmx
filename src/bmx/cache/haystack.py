"""Haystack filler for the NIAH retrieval metric.

Two regimes (matching the run split):
  - synthetic_filler: deterministic repeated text, no download — the offline/CI path.
  - load_pg_corpus: real Paul Graham essays from the HuggingFace dataset
    ``sgoel9/paul_graham_essays`` — the VM headline path (max comparability to the
    TurboQuant / Fu et al. setup). Self-downloading; no local clone required.
"""

from __future__ import annotations

_FILLER_SENTENCE = "The grass was green and the sky was blue and the day was calm. "

PG_ESSAYS_DATASET = "sgoel9/paul_graham_essays"


def synthetic_filler(n_repeats: int) -> str:
    """Deterministic repeated filler (no download). Used by the offline/CI path."""
    assert n_repeats > 0, "n_repeats must be positive"
    return _FILLER_SENTENCE * n_repeats


def load_pg_corpus() -> str:
    """Concatenate the Paul Graham essays from the HF dataset into one filler string.

    VM/real path only — downloads ``sgoel9/paul_graham_essays`` (215 essays, ``text``
    column, ``train`` split) on first call. Lazy-imports ``datasets`` so importing this
    module never triggers a download (the offline/CI path uses synthetic_filler).
    """
    from datasets import load_dataset

    ds = load_dataset(PG_ESSAYS_DATASET, split="train")
    texts = [t for t in ds["text"] if t]
    assert texts, f"no essay text in {PG_ESSAYS_DATASET}"
    return "\n".join(texts)
