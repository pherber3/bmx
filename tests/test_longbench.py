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

    `decode` is unused by build_longbench_prompt when chat_wrap=False (the default and the
    only path these truncation tests exercise), so it is intentionally omitted here; a call
    to it would signal an unwanted decode/re-encode round trip on this path. The chat-wrap
    path (chat_wrap=True) does decode, and is covered separately by `_ChatCapableTok` below.
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


# --- Task 6: flag-gated chat-wrap (LongBench pred.py build_chat parity) ---


class _ChatCapableTok:
    """Word-level stub tokenizer that also supports decode + apply_chat_template.

    `__call__` mirrors _CountingTok (one token id per whitespace-split word), but ids are
    offset by 1000 per word-length so decode can invert it deterministically: decode just
    re-joins the words it remembers having assigned those ids to, in order. bos_token_id is
    a reserved id (1) prepended to every apply_chat_template output, mimicking Llama-3's
    template (which embeds the BOS itself) — tests assert build_longbench_prompt never
    double-prepends it.
    """

    bos_token_id = 1

    def __call__(self, text, return_tensors=None):
        import torch

        words = text.split()
        # Deterministic non-trivial ids so decode can invert them (id = index into a
        # per-call word list stashed on the instance).
        self._last_words = words
        ids = torch.tensor([[100 + i for i in range(len(words))]])
        return type("E", (), {"input_ids": ids})()

    def decode(self, ids, skip_special_tokens=True):
        # Inverts __call__: id (100+i) -> self._last_words[i]. Ids outside that range
        # (e.g. a stray BOS) are dropped when skip_special_tokens=True.
        seq = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        words = []
        for tid in seq:
            idx = tid - 100
            if 0 <= idx < len(self._last_words):
                words.append(self._last_words[idx])
            elif not skip_special_tokens:
                words.append(f"<special:{tid}>")
        return " ".join(words)

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors=None,
        return_dict=False,
    ):
        import torch

        content = messages[0]["content"]
        assert messages[0]["role"] == "user"
        # Simulate a template that renders its own BOS + wrapper markers, then re-runs
        # the (deterministic) word tokenizer over the wrapped text — mirrors a real chat
        # template producing fresh ids from wrapped text, not just concatenating old ids.
        wrapped = f"<|begin|><|user|>{content}<|assistant|>"
        words = wrapped.split()
        self._last_words = words
        body = [100 + i for i in range(len(words))]
        ids = torch.tensor([[self.bos_token_id] + body])
        if not tokenize:
            return wrapped
        if return_dict:
            return {"input_ids": ids}
        # Mirror the REAL transformers-5.x trap that bit the first on-VM chat-wrap run
        # (2026-07-06): without return_dict=True, the fast tokenizer hands back a list
        # of tokenizers.Encoding objects, IGNORING return_tensors. Returning a
        # non-tensor here keeps the regression honest — build_longbench_prompt must go
        # through the return_dict interface or these tests fail the way the VM did.
        return [_FakeEncoding(ids[0])]


class _FakeEncoding:
    """Stand-in for tokenizers.Encoding: has .ids, is NOT a tensor, has no .tolist()."""

    def __init__(self, ids_row):
        self.ids = ids_row.tolist()


def _chat_item():
    return {
        "context": "def foo():\n    return 1\n",
        "input": "",
        "answers": ["    return 1"],
    }


def test_chat_wrap_false_is_byte_identical_to_no_flag():
    tok = _ChatCapableTok()
    item = _chat_item()
    old_ids = build_longbench_prompt(tok, item, "narrativeqa")
    new_ids = build_longbench_prompt(tok, item, "narrativeqa", chat_wrap=False)
    assert torch.equal(old_ids, new_ids)


def test_chat_wrap_true_excluded_task_matches_unwrapped():
    # trec is in CHAT_WRAP_EXCLUDED (few-shot) -> chat_wrap=True must be a no-op for it.
    tok = _ChatCapableTok()
    item = {
        "context": "Q: 1 is what?\nA: number\n",
        "input": "Q: 2?\nA:",
        "answers": [""],
    }
    unwrapped = build_longbench_prompt(tok, item, "trec", chat_wrap=False)
    wrapped = build_longbench_prompt(tok, item, "trec", chat_wrap=True)
    assert torch.equal(unwrapped, wrapped)


def test_chat_wrap_true_nonexcluded_task_wraps_with_single_bos():
    tok = _ChatCapableTok()
    item = _chat_item()
    ids = build_longbench_prompt(tok, item, "narrativeqa", chat_wrap=True)
    decoded = tok.decode(ids.squeeze(0), skip_special_tokens=False)
    assert "<|begin|>" in decoded
    assert "<|user|>" in decoded
    assert "<|assistant|>" in decoded
    # Exactly one BOS, at position 0.
    ids_list = ids.squeeze(0).tolist()
    assert ids_list[0] == tok.bos_token_id
    assert ids_list.count(tok.bos_token_id) == 1


def test_chat_wrap_truncation_happens_before_wrap():
    tok = _ChatCapableTok()
    # Long context -> the pre-wrap prompt (narrativeqa's template + 60 "tok<i>" context
    # words) tokenizes to 122 words; a budget of 100 (half=50) keeps template words plus
    # roughly tok0..tok12 and tok35..tok59 (verified: first-half tail is tok12, second-half
    # head is tok35), dropping the interior tok13..tok34 range. Middle-truncation must
    # happen BEFORE the chat markers are applied, so the markers survive and an interior
    # context word (e.g. tok20) is gone from the decoded, wrapped text.
    context = " ".join(f"tok{i}" for i in range(60))
    item = {"context": context, "input": "", "answers": [""]}

    wrapped = build_longbench_prompt(
        tok, item, "narrativeqa", max_prompt_tokens=100, chat_wrap=True
    )
    decoded = tok.decode(wrapped.squeeze(0), skip_special_tokens=False)
    assert "<|begin|>" in decoded
    assert "<|user|>" in decoded
    assert "<|assistant|>" in decoded
    # An interior word must have been dropped by truncation.
    assert "tok20" not in decoded
    # But head and tail context words survive.
    assert "tok5" in decoded
    assert "tok40" in decoded
    # Still exactly one BOS at position 0.
    ids_list = wrapped.squeeze(0).tolist()
    assert ids_list[0] == tok.bos_token_id
    assert ids_list.count(tok.bos_token_id) == 1
