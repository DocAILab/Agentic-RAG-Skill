"""HotpotQA-specific retrieval and answer metrics."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

METRIC_NAMES = (
    "hit@1",
    "hit@10",
    "recall@10",
    "all_support@10",
    "em",
    "f1",
)


def evaluate_hotpotqa(prediction, gold_answers, retrieved_ids, relevant_ids):
    """Evaluate one HotpotQA answer and its retrieved support documents."""
    retrieved = list(retrieved_ids)
    relevant = set(relevant_ids)
    return {
        "hit@1": _hit_at_k(retrieved, relevant, 1),
        "hit@10": _hit_at_k(retrieved, relevant, 10),
        "recall@10": _recall_at_k(retrieved, relevant, 10),
        "all_support@10": _all_support_at_k(retrieved, relevant, 10),
        "em": _best_answer_score(prediction, gold_answers, _exact_match),
        "f1": _best_answer_score(prediction, gold_answers, _token_f1),
    }


def summarize_hotpotqa(metrics):
    """Average flat HotpotQA metric dictionaries."""
    items = list(metrics)
    if not items:
        return {"count": 0, **dict.fromkeys(METRIC_NAMES, 0.0)}
    return {
        "count": len(items),
        **{
            name: sum(float(item[name]) for item in items) / len(items)
            for name in METRIC_NAMES
        },
    }


def _hit_at_k(retrieved, relevant, k):
    return float(any(document_id in relevant for document_id in retrieved[:k]))


def _recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def _all_support_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    return float(relevant <= set(retrieved[:k]))


def _best_answer_score(prediction, gold_answers, scorer):
    answers = _gold_answers(gold_answers)
    return max(scorer(prediction, answer) for answer in answers)


def _gold_answers(gold_answers):
    if isinstance(gold_answers, str):
        answers = (gold_answers,)
    elif isinstance(gold_answers, Sequence):
        answers = tuple(gold_answers)
    else:
        raise TypeError("gold_answers must be a string or sequence")
    if not answers or not all(isinstance(answer, str) for answer in answers):
        raise ValueError("gold_answers must contain at least one string")
    return answers


def _exact_match(prediction, gold_answer):
    return float(_normalize(prediction) == _normalize(gold_answer))


def _token_f1(prediction, gold_answer):
    prediction_text = _normalize(prediction)
    gold_text = _normalize(gold_answer)
    special_answers = {"yes", "no", "noanswer"}
    if (prediction_text in special_answers or gold_text in special_answers) and (
        prediction_text != gold_text
    ):
        return 0.0
    prediction_tokens = prediction_text.split()
    gold_tokens = gold_text.split()
    overlap = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize(answer):
    lowered = str(answer).lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())
