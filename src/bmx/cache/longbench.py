"""LongBench eval over the 6 TurboQuant Table-1 categories (English datasets only).

Scorers (code_sim, qa_f1_score, rouge_score, classification_score, retrieval_score,
count_score) are verbatim ports of LongBench/LongBench/metrics.py — English variants only,
Chinese (_zh) scorers skipped. DATASET2METRIC / CATEGORY2DATASETS mirror LongBench's
eval.py::dataset2metric and the paper's category grouping. The loader and per-item scorer
require a real model and dataset (VM only); the scorers and registry are pure and CI-testable.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from functools import lru_cache

import torch
from fuzzywuzzy import fuzz

from bmx.cache.generate import generate_through_cache
from bmx.cache.specs import CacheCodecSpec

# LongBench's prompt templates (config/dataset2prompt.json) and max_gen
# (config/dataset2maxlen.json). English datasets only. Templates ported verbatim — exact
# whitespace, no normalization (e.g. lcc's trailing space after "below. ").
LONGBENCH_TASKS: dict[str, dict] = {
    # --- single-document QA (qa_f1_score) ---
    "narrativeqa": {
        "prompt_template": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "max_gen": 128,
    },
    "qasper": {
        "prompt_template": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
        "max_gen": 128,
    },
    "multifieldqa_en": {
        "prompt_template": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "max_gen": 64,
    },
    # --- multi-document QA (qa_f1_score) ---
    "hotpotqa": {
        "prompt_template": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "max_gen": 32,
    },
    "2wikimqa": {
        "prompt_template": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "max_gen": 32,
    },
    "musique": {
        "prompt_template": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "max_gen": 32,
    },
    # --- summarization (rouge_score) ---
    "gov_report": {
        "prompt_template": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
        "max_gen": 512,
    },
    "qmsum": {
        "prompt_template": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
        "max_gen": 512,
    },
    "multi_news": {
        "prompt_template": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
        "max_gen": 512,
    },
    # --- few-shot (mixed: trec classification, triviaqa qa_f1, samsum rouge) ---
    "trec": {
        "prompt_template": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
        "max_gen": 64,
    },
    "triviaqa": {
        "prompt_template": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
        "max_gen": 32,
    },
    "samsum": {
        "prompt_template": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
        "max_gen": 128,
    },
    # --- synthetic (passage_count count, passage_retrieval_en retrieval) ---
    "passage_count": {
        "prompt_template": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
        "max_gen": 32,
    },
    "passage_retrieval_en": {
        "prompt_template": 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
        "max_gen": 32,
    },
    # --- code (code_sim) ---
    "lcc": {
        "prompt_template": "Please complete the code given below. \n{context}Next line of code:\n",
        "max_gen": 64,
    },
    "repobench-p": {
        "prompt_template": "Please complete the code given below. \n{context}{input}Next line of code:\n",
        "max_gen": 64,
    },
}


def build_longbench_prompt(tokenizer, item: dict, task: str) -> torch.Tensor:
    """Apply the task's LongBench prompt template to the item; return (1, L) ids.

    LongBench formats dataset2prompt[task].format(**item); for code tasks the context lives in
    item['context']. NO chat/[INST] wrapper: LongBench explicitly skips build_chat for the code
    tasks (lcc, repobench-p are in its exclusion list) — raw template only, even for Instruct
    models.
    """
    template = LONGBENCH_TASKS[task]["prompt_template"]
    prompt = template.format(**item)
    return tokenizer(prompt, return_tensors="pt").input_ids


def load_longbench_task(
    task: str, n_samples: int | None, version: str = "v1"
) -> list[dict]:
    """Load LongBench[task]; return up to n_samples items (all if None). VM-only.

    version 'v1' -> THUDM/LongBench. THUDM/LongBench ships as a loader script + data.zip;
    datasets>=4 no longer runs dataset scripts, so read the task's jsonl out of data.zip
    directly via huggingface_hub.
    """
    import json
    import zipfile

    from huggingface_hub import hf_hub_download

    if version != "v1":
        raise ValueError(f"unsupported longbench version: {version!r} (only 'v1')")

    zip_path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    items: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(f"data/{task}.jsonl") as fh:
            for line in fh:
                items.append(json.loads(line))
                if n_samples is not None and len(items) >= n_samples:
                    break
    return items


def longbench_score(
    model,
    tokenizer,
    item: dict,
    task: str,
    n_prefill: int,
    k_spec: CacheCodecSpec,
    v_spec: CacheCodecSpec,
) -> float:
    """Generate through the compressed cache, score with the task's LongBench metric.

    Follows eval.py::scorer per item: for trec/triviaqa/samsum the prediction is pre-split to
    its first line, the per-item score is max over the item's ground truths (item['answers']),
    and classification passes all_classes=item['all_classes']. Works for every registered
    English task (code tasks route to code_sim via DATASET2METRIC).
    """
    prompt_ids = build_longbench_prompt(tokenizer, item, task)
    max_gen = LONGBENCH_TASKS[task]["max_gen"]
    response = generate_through_cache(
        model, tokenizer, prompt_ids, n_prefill, k_spec, v_spec, max_gen, strip=False
    )
    if task in FIRST_LINE_DATASETS:
        response = response.lstrip("\n").split("\n")[0]
    metric = DATASET2METRIC[task]
    all_classes = item.get("all_classes")
    score = 0.0
    for ground_truth in item["answers"]:
        score = max(score, metric(response, ground_truth, all_classes=all_classes))
    return score


def longbench_code_score(
    model,
    tokenizer,
    item: dict,
    task: str,
    n_prefill: int,
    k_spec: CacheCodecSpec,
    v_spec: CacheCodecSpec,
) -> float:
    """Build the LongBench prompt, generate through the compressed cache, score code_sim.

    Ground truth is item['answers'][0] (LongBench code tasks have a single reference line).
    """
    prompt_ids = build_longbench_prompt(tokenizer, item, task)
    max_gen = LONGBENCH_TASKS[task]["max_gen"]
    response = generate_through_cache(
        model, tokenizer, prompt_ids, n_prefill, k_spec, v_spec, max_gen, strip=False
    )
    ground_truth = item["answers"][0]
    return code_sim(response, ground_truth)


def code_sim(prediction: str, ground_truth: str, **kwargs) -> float:
    """LongBench code edit-similarity, range 0–1 (verbatim port of code_sim_score).

    Accepts (and ignores) **kwargs so it shares the uniform metric signature in
    DATASET2METRIC — `longbench_score` calls every metric with all_classes=...; code
    tasks route here and must not crash on the unused kwarg.

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


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.

    Verbatim port of metrics.py::normalize_answer (English SQuAD-style normalization).
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth, **kwargs):
    """Token-list F1 (verbatim port of metrics.py::f1_score); inputs are token lists."""
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth, **kwargs):
    """SQuAD-style token F1 over normalized answers, range 0–1.

    Verbatim port of metrics.py::qa_f1_score (English). Used by single_qa, multi_qa, and
    the triviaqa few-shot task.
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


@lru_cache(maxsize=1)
def _rouge():
    """One shared Rouge() — it holds only config, so it's safe to reuse across the
    hundreds of per-item calls in a full LongBench sweep (rebuilding it each call
    reconstructs its tokenizers/regexes for nothing)."""
    from rouge import Rouge

    return Rouge()


def rouge_score(prediction, ground_truth, **kwargs):
    """ROUGE-L F, range 0–1 (verbatim port of metrics.py::rouge_score).

    Uses `from rouge import Rouge`; empty/degenerate inputs raise inside get_scores, caught
    to 0.0. Used by summarization and the samsum few-shot task.
    """
    try:
        scores = _rouge().get_scores([prediction], [ground_truth], avg=True)
    except:  # noqa: E722 — verbatim port; LongBench catches any get_scores failure to 0.0
        return 0.0
    return scores["rouge-l"]["f"]


def classification_score(prediction, ground_truth, **kwargs):
    """Exact-membership classification score, range 0–1.

    Verbatim port of metrics.py::classification_score. Needs all_classes=item['all_classes'].
    Used by the trec few-shot task.
    """
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = 1.0 / len(em_match_list)
    else:
        score = 0.0
    return score


def retrieval_score(prediction, ground_truth, **kwargs):
    """Paragraph-index retrieval score, range 0–1 (verbatim port of retrieval_score).

    Parses the target 'Paragraph N' from ground_truth, then fraction of prediction's numbers
    that equal N. Used by passage_retrieval_en.
    """
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def count_score(prediction, ground_truth, **kwargs):
    """Count-matching score, range 0–1 (verbatim port of metrics.py::count_score).

    Fraction of prediction's numbers equal to ground_truth. Used by passage_count.
    """
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


# dataset -> scorer (mirror of eval.py::dataset2metric, English datasets only).
DATASET2METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
    "lcc": code_sim,
    "repobench-p": code_sim,
}

# The 6 TurboQuant Table-1 categories -> their English datasets.
CATEGORY2DATASETS = {
    "single_qa": ["narrativeqa", "qasper", "multifieldqa_en"],
    "multi_qa": ["hotpotqa", "2wikimqa", "musique"],
    "summarization": ["gov_report", "qmsum", "multi_news"],
    "few_shot": ["trec", "triviaqa", "samsum"],
    "synthetic": ["passage_count", "passage_retrieval_en"],
    "code": ["lcc", "repobench-p"],
}

# Datasets whose prediction is pre-split to its first non-empty line before scoring
# (eval.py: `prediction.lstrip('\n').split('\n')[0]`). English subset (skip lsht, Chinese).
FIRST_LINE_DATASETS = {"trec", "triviaqa", "samsum"}
