"""LongBench Code eval: code_sim scorer, dataset loader, task registry.

Faithful port of LongBench's Code-category eval (lcc, repobench-p). The scorer matches
LongBench's metrics.py::code_sim_score exactly (verified in .git/sdd/longbench-conventions.md
against the local clone). The dataset loader + scoring path are VM-only (real model); code_sim
is a pure CI-testable function.
"""

from __future__ import annotations

from fuzzywuzzy import fuzz


def code_sim(prediction: str, ground_truth: str) -> float:
    """LongBench code edit-similarity, range 0–1 (verbatim port of code_sim_score).

    Post-process: strip leading blank lines, then keep the FIRST line that contains no
    backtick / '#' / '//' (LongBench's rule), then fuzz.ratio normalized to [0, 1].
    """
    all_lines = prediction.lstrip("\n").split("\n")
    pred = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            pred = line
            break
    return fuzz.ratio(pred, ground_truth) / 100.0
