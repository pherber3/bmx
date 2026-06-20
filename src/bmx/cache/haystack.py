"""Haystack filler for the NIAH retrieval metric.

Two regimes (matching the run split):
  - synthetic_filler: deterministic repeated text, no files — the offline/CI path.
  - Paul Graham essays: real filler from the local Fu et al. clone — the VM headline
    path (max comparability to the TurboQuant / Fu et al. setup).
"""

from __future__ import annotations

from pathlib import Path

_FILLER_SENTENCE = "The grass was green and the sky was blue and the day was calm. "


def synthetic_filler(n_repeats: int) -> str:
    """Deterministic repeated filler (no files). Used by the offline/CI path."""
    assert n_repeats > 0, "n_repeats must be positive"
    return _FILLER_SENTENCE * n_repeats


def pg_essays_dir() -> Path | None:
    """Path to the local Paul Graham essays dir, or None if the clone is absent.

    The Fu et al. repo is cloned at the bmx repo root as a local reference (not
    vendored). Resolve relative to this file: src/bmx/cache/haystack.py -> repo root.
    """
    repo_root = Path(__file__).resolve().parents[3]
    d = (
        repo_root
        / "Long-Context-Data-Engineering"
        / "eval"
        / "needle"
        / "PaulGrahamEssays"
    )
    return d if d.is_dir() and any(d.glob("*.txt")) else None


def read_pg_corpus(essays_dir: Path) -> str:
    """Concatenate all *.txt files in essays_dir into one filler string."""
    parts = [
        p.read_text(encoding="utf-8", errors="ignore")
        for p in sorted(essays_dir.glob("*.txt"))
    ]
    assert parts, f"no *.txt files in {essays_dir}"
    return "\n".join(parts)
