"""Haystack filler for the NIAH retrieval metric.

load_pg_corpus downloads the Paul Graham essays from the HuggingFace dataset for the
real run.
"""

from __future__ import annotations

PG_ESSAYS_DATASET = "sgoel9/paul_graham_essays"


def load_pg_corpus() -> str:
    """Concatenate the Paul Graham essays from the HF dataset into one string.

    ``datasets`` is imported lazily so importing this module triggers no download.
    """
    from datasets import load_dataset

    ds = load_dataset(PG_ESSAYS_DATASET, split="train")
    texts = [t for t in ds["text"] if t]
    assert texts, f"no essay text in {PG_ESSAYS_DATASET}"
    return "\n".join(texts)
