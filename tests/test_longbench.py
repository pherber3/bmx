import pytest
import torch

from bmx.cache.longbench import LONGBENCH_TASKS, build_longbench_prompt, code_sim


def test_code_sim_identical_is_one():
    # Single clean line (the post-process keeps the first non-comment line); identical => 1.0.
    line = "    return a + b"
    assert code_sim(line, line) == 1.0


def test_code_sim_disjoint_is_low():
    assert code_sim("    return a + b", "xxxxx yyyyy zzzzz") < 0.3


def test_code_sim_partial_is_graded():
    gt = "    return a + b"
    pred = "    return a - b"  # one char off
    s = code_sim(pred, gt)
    assert 0.0 < s < 1.0


def test_code_sim_strips_comment_lines():
    # The post-process skips lines containing `, #, or // and scores the first clean line.
    pred = "# a comment\n    return a + b"
    assert code_sim(pred, "    return a + b") == 1.0


def test_longbench_tasks_registry():
    # The 6 TurboQuant Table-1 categories, English datasets only (16 total).
    expected = {
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "hotpotqa",
        "2wikimqa",
        "musique",
        "gov_report",
        "qmsum",
        "multi_news",
        "trec",
        "triviaqa",
        "samsum",
        "passage_count",
        "passage_retrieval_en",
        "lcc",
        "repobench-p",
    }
    assert set(LONGBENCH_TASKS) == expected
    for t in expected:
        assert "prompt_template" in LONGBENCH_TASKS[t]
        assert isinstance(LONGBENCH_TASKS[t]["max_gen"], int)
        assert "{context}" in LONGBENCH_TASKS[t]["prompt_template"]


def test_dataset2metric_and_categories_consistent():
    from bmx.cache.longbench import CATEGORY2DATASETS, DATASET2METRIC

    # Every registered dataset has a scorer, and vice versa.
    assert set(DATASET2METRIC) == set(LONGBENCH_TASKS)
    # Categories partition exactly the registered datasets (no overlap, full cover).
    flat = [ds for datasets in CATEGORY2DATASETS.values() for ds in datasets]
    assert set(flat) == set(LONGBENCH_TASKS)
    assert len(flat) == len(set(flat))
    assert set(CATEGORY2DATASETS) == {
        "single_qa",
        "multi_qa",
        "summarization",
        "few_shot",
        "synthetic",
        "code",
    }


def test_build_longbench_prompt_shapes():
    class StubTok:
        def __call__(self, text, return_tensors=None):
            import torch

            ids = torch.tensor([[ord(c) % 97 for c in text[:40]]])
            return type("E", (), {"input_ids": ids})()

    item = {
        "context": "def foo():\n    return 1\n",
        "input": "",
        "answers": ["    return 1"],
    }
    ids = build_longbench_prompt(StubTok(), item, "lcc")
    assert ids.shape[0] == 1 and ids.shape[1] > 0


# --- Regression I1: LongBench code_sim indentation fidelity ---


def test_code_sim_indented_prediction_scores_one():
    """An indented prediction identical to an indented ground truth must score 1.0.

    Pins that code_sim is whitespace-preserving for indented lines (which are the common
    case for lcc / repobench-p completions).
    """
    gt = "        return result"
    assert code_sim(gt, gt) == 1.0


def test_code_sim_stripped_indented_scores_less_than_one():
    """A prediction whose leading indent was stripped must score < 1.0 vs the indented gt.

    This documents WHY generate_through_cache must pass strip=False for the LongBench path:
    .strip() on an 8-space-indented completion removes the indent before code_sim sees it,
    producing a score < 1.0 even when the content is otherwise identical.
    Measured regression: indent=8 → LongBench 1.000 vs stripped 0.820.
    """
    gt = "        return result"
    stripped_pred = gt.strip()  # simulates what .strip() in generate_through_cache did
    assert code_sim(stripped_pred, gt) < 1.0


# --- Task 1: middle-truncation (LongBench pred.py parity) ---


class _CountingTok:
    """Deterministic word-level stub tokenizer: one token id per whitespace-split word.

    `decode` is unused by build_longbench_prompt (no chat-wrap step exists in this repo's
    harness — see build_longbench_prompt's docstring), so it is intentionally omitted; a
    call to it would signal an unwanted decode/re-encode round trip.
    """

    def __call__(self, text, return_tensors=None):
        import torch

        words = text.split()
        ids = torch.tensor([[i % 97 for i in range(len(words))]])
        return type("E", (), {"input_ids": ids})()


def test_middle_truncation_keeps_head_and_tail():
    # 100 "words" -> 100 token ids under the counting stub; truncate to 20.
    context = " ".join(f"w{i}" for i in range(100))
    item = {"context": context, "input": "", "answers": [""]}
    tok = _CountingTok()

    full_ids = build_longbench_prompt(tok, item, "lcc").squeeze(0)
    n = 20
    truncated = build_longbench_prompt(tok, item, "lcc", max_prompt_tokens=n).squeeze(0)

    assert truncated.shape[0] == n
    half = n // 2
    assert torch.equal(truncated[:half], full_ids[:half])
    assert torch.equal(truncated[half:], full_ids[-half:])


def test_middle_truncation_short_prompt_unchanged():
    # Prompt tokenizes shorter than the budget -> byte-identical (no-op).
    context = "short context"
    item = {"context": context, "input": "", "answers": [""]}
    tok = _CountingTok()

    full_ids = build_longbench_prompt(tok, item, "lcc")
    truncated = build_longbench_prompt(tok, item, "lcc", max_prompt_tokens=10_000)

    assert torch.equal(truncated, full_ids)


def test_middle_truncation_none_is_noop():
    # max_prompt_tokens=None (the default) must be byte-identical to omitting the arg —
    # the live VM run's no-truncation variant depends on this exact parity.
    context = " ".join(f"w{i}" for i in range(100))
    item = {"context": context, "input": "", "answers": [""]}
    tok = _CountingTok()

    default_ids = build_longbench_prompt(tok, item, "lcc")
    explicit_none_ids = build_longbench_prompt(tok, item, "lcc", max_prompt_tokens=None)

    assert torch.equal(default_ids, explicit_none_ids)


# --- Task 1 Part B: LongBench-E subset loading ---


def _make_fake_longbench_zip(tmp_path, tasks_with_e):
    """Build a tiny local data.zip with data/{task}.jsonl and data/{task}_e.jsonl
    for tasks in tasks_with_e, plus a plain data/{task}.jsonl for 'lcc' (no _e variant)."""
    import json
    import zipfile

    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data/lcc.jsonl", json.dumps({"id": "v1-only"}) + "\n")
        for task in tasks_with_e:
            zf.writestr(f"data/{task}.jsonl", json.dumps({"id": f"{task}-v1"}) + "\n")
            zf.writestr(f"data/{task}_e.jsonl", json.dumps({"id": f"{task}-e"}) + "\n")
    return str(zip_path)


def test_load_longbench_task_v1_e_reads_e_file(tmp_path, monkeypatch):
    import bmx.cache.longbench as lb

    zip_path = _make_fake_longbench_zip(tmp_path, tasks_with_e=["hotpotqa"])
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: zip_path)

    items = lb.load_longbench_task("hotpotqa", None, version="v1_e")
    assert items == [{"id": "hotpotqa-e"}]


def test_load_longbench_task_v1_e_missing_task_fails_loudly(tmp_path, monkeypatch):
    import bmx.cache.longbench as lb

    zip_path = _make_fake_longbench_zip(tmp_path, tasks_with_e=["hotpotqa"])
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: zip_path)

    with pytest.raises(ValueError, match="lcc"):
        lb.load_longbench_task("lcc", None, version="v1_e")


def test_load_longbench_task_rejects_unknown_version():
    import bmx.cache.longbench as lb

    with pytest.raises(ValueError, match="bogus"):
        lb.load_longbench_task("lcc", None, version="bogus")
