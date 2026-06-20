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
    assert set(LONGBENCH_TASKS) == {"lcc", "repobench-p"}
    for t in ("lcc", "repobench-p"):
        assert "prompt_template" in LONGBENCH_TASKS[t]
        assert isinstance(LONGBENCH_TASKS[t]["max_gen"], int)
        assert "{context}" in LONGBENCH_TASKS[t]["prompt_template"]


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
