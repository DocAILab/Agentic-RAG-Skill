"""提供检索命中率与 HotpotQA 风格答案指标。"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Collection, Hashable, Sequence
from itertools import islice


def normalize_answer(answer: str) -> str:
    """按 HotpotQA 官方口径归一化答案文本。"""
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")

    lowered = answer.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def hit_at_k(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
    k: int,
) -> float:
    """判断前 k 个检索结果中是否至少包含一个相关标识符。"""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    _validate_identifier_collection(retrieved_ids, name="retrieved_ids")
    _validate_identifier_collection(relevant_ids, name="relevant_ids")

    relevant = set(relevant_ids)
    return float(any(identifier in relevant for identifier in islice(retrieved_ids, k)))


def hit_at_1(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
) -> float:
    """计算检索结果的 Hit@1。"""
    return hit_at_k(retrieved_ids, relevant_ids, 1)


def hit_at_10(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
) -> float:
    """计算检索结果的 Hit@10。"""
    return hit_at_k(retrieved_ids, relevant_ids, 10)


def recall_at_k(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
    k: int,
) -> float:
    """计算前 k 个结果覆盖的相关文档比例。"""
    _validate_rank_inputs(retrieved_ids, relevant_ids, k)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    retrieved = set(islice(retrieved_ids, k))
    return len(retrieved & relevant) / len(relevant)


def all_support_at_k(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
    k: int,
) -> float:
    """判断前 k 个结果是否覆盖全部多跳支持文档。"""
    _validate_rank_inputs(retrieved_ids, relevant_ids, k)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return float(relevant <= set(islice(retrieved_ids, k)))


def reciprocal_rank(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
) -> float:
    """返回第一个相关文档名次的倒数，未命中时返回零。"""
    _validate_identifier_collection(retrieved_ids, name="retrieved_ids")
    _validate_identifier_collection(relevant_ids, name="relevant_ids")
    relevant = set(relevant_ids)
    for rank, identifier in enumerate(retrieved_ids, start=1):
        if identifier in relevant:
            return 1.0 / rank
    return 0.0


def exact_match_score(
    prediction: str,
    gold_answers: str | Sequence[str],
) -> float:
    """计算预测答案对一个或多个标准答案的最大精确匹配分数。"""
    answers = _normalize_gold_answers(gold_answers)
    return max(_single_exact_match(prediction, answer) for answer in answers)


def f1_score(
    prediction: str,
    gold_answers: str | Sequence[str],
) -> float:
    """计算预测答案对一个或多个标准答案的最大词元 F1。"""
    answers = _normalize_gold_answers(gold_answers)
    return max(_single_f1(prediction, answer) for answer in answers)


def _single_exact_match(prediction: str, gold_answer: str) -> float:
    """计算单个标准答案的归一化精确匹配分数。"""
    return float(normalize_answer(prediction) == normalize_answer(gold_answer))


def _single_f1(prediction: str, gold_answer: str) -> float:
    """计算单个标准答案的 HotpotQA 风格词元 F1。"""
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold_answer)
    special_answers = {"yes", "no", "noanswer"}
    if (
        normalized_prediction in special_answers
        or normalized_gold in special_answers
    ) and normalized_prediction != normalized_gold:
        return 0.0

    prediction_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    matching_tokens = sum(common.values())
    if matching_tokens == 0:
        return 0.0

    precision = matching_tokens / len(prediction_tokens)
    recall = matching_tokens / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize_gold_answers(gold_answers: str | Sequence[str]) -> tuple[str, ...]:
    """把单答案或答案别名序列统一为非空字符串元组。"""
    if isinstance(gold_answers, str):
        return (gold_answers,)
    if isinstance(gold_answers, (bytes, bytearray)) or not isinstance(
        gold_answers, Sequence
    ):
        raise TypeError("gold_answers must be a string or a sequence of strings")
    answers = tuple(gold_answers)
    if not answers:
        raise ValueError("gold_answers must not be empty")
    if not all(isinstance(answer, str) for answer in answers):
        raise TypeError("gold_answers must contain only strings")
    return answers


def _validate_identifier_collection(
    identifiers: Collection[Hashable],
    *,
    name: str,
) -> None:
    """校验检索标识符集合，避免字符串被误当作标识符序列。"""
    if isinstance(identifiers, (str, bytes, bytearray)) or not isinstance(
        identifiers, Collection
    ):
        raise TypeError(f"{name} must be a collection of identifiers")
    if not all(isinstance(identifier, Hashable) for identifier in identifiers):
        raise TypeError(f"{name} must contain only hashable identifiers")


def _validate_rank_inputs(retrieved_ids, relevant_ids, k):
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    _validate_identifier_collection(retrieved_ids, name="retrieved_ids")
    _validate_identifier_collection(relevant_ids, name="relevant_ids")
