"""Hand-computed fidelity tests for the ported LongBench English scorers.

Each expected value is traced by hand against LongBench/LongBench/metrics.py semantics
(normalize_answer + f1_score for qa_f1; rouge-l f; classification/retrieval/count rules).
"""

from __future__ import annotations

import pytest

from bmx.cache.longbench import (
    DATASET2METRIC,
    classification_score,
    count_score,
    qa_f1_score,
    retrieval_score,
    rouge_score,
)


def test_all_metrics_accept_uniform_kwargs():
    # longbench_score calls EVERY metric as metric(pred, gt, all_classes=...); a metric
    # missing **kwargs (e.g. code_sim) would crash only on a real run, not in schema tests.
    # Guard the uniform signature here. all_classes must be usable by classification_score.
    for dataset, metric in DATASET2METRIC.items():
        score = metric("Paragraph 1", "Paragraph 1", all_classes=["Paragraph 1"])
        assert isinstance(score, (int, float)), dataset


def test_qa_f1_score_exact_match():
    # "the cat sat" -> normalize drops article "the" -> tokens [cat, sat] both sides.
    # common=2, precision=recall=1 -> f1=1.0.
    assert qa_f1_score("the cat sat", "the cat sat") == pytest.approx(1.0)


def test_qa_f1_score_disjoint():
    # tokens [dog] vs [cat]; common=0 -> f1_score returns 0.
    assert qa_f1_score("dog", "cat") == 0


def test_qa_f1_score_partial_overlap():
    # normalize drops "the": [quick, brown, fox] vs [lazy, brown, dog].
    # common={brown:1}=1; p=1/3, r=1/3; f1 = 2*(1/9)/(2/3) = 1/3.
    assert qa_f1_score("the quick brown fox", "the lazy brown dog") == pytest.approx(
        1.0 / 3.0
    )


def test_rouge_score_identical():
    # rouge-l f of a string against itself is 1.0 (allow float slop).
    assert rouge_score("the cat sat on the mat", "the cat sat on the mat") > 0.99


def test_rouge_score_disjoint():
    # No shared tokens -> rouge-l f is 0.0.
    assert rouge_score("alpha beta gamma", "delta epsilon zeta") == pytest.approx(0.0)


def test_classification_score_match():
    all_classes = ["sports", "politics", "science"]
    # prediction contains exactly "sports"; em_match_list=[sports]; gt in list -> 1/1.
    assert classification_score(
        "this is about sports", "sports", all_classes=all_classes
    ) == pytest.approx(1.0)


def test_classification_score_no_match():
    all_classes = ["sports", "politics", "science"]
    # prediction contains none of the classes -> em_match_list=[] -> 0.0.
    assert classification_score(
        "nothing relevant here", "sports", all_classes=all_classes
    ) == pytest.approx(0.0)


def test_retrieval_score_correct():
    # ground_truth_id parsed from "Paragraph 3" -> "3"; pred numbers=[3]; 1/1.
    assert retrieval_score("Paragraph 3", "Paragraph 3") == pytest.approx(1.0)


def test_retrieval_score_wrong_number():
    # gt id "3", pred numbers=[5]; 0/1.
    assert retrieval_score("Paragraph 5", "Paragraph 3") == pytest.approx(0.0)


def test_count_score_correct():
    # pred numbers=[5], gt "5"; 1/1.
    assert count_score("5", "5") == pytest.approx(1.0)


def test_count_score_partial():
    # pred numbers=[5,6,7], gt "5"; right_num=1 -> 1/3.
    assert count_score("5 6 7", "5") == pytest.approx(1.0 / 3.0)
