"""Retrieval metrics with explicit rank cutoffs for comparable evaluation."""

from __future__ import annotations

import math
from collections.abc import Collection, Hashable, Sequence
from itertools import islice


def retrieval_f1(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
) -> float:
    """Compute set F1 on Top-n, where n is the number of unique gold ids."""
    _validate_identifier_collection(retrieved_ids, name="retrieved_ids")
    _validate_identifier_collection(relevant_ids, name="relevant_ids")
    gold_count = len(set(relevant_ids))
    if gold_count == 0:
        return 0.0
    return retrieval_f1_at_k(retrieved_ids, relevant_ids, gold_count)


def retrieval_f1_at_1(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
) -> float:
    """Compute set F1 using only the first retrieved document."""
    return retrieval_f1_at_k(retrieved_ids, relevant_ids, 1)


def retrieval_f1_at_k(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
    k: int,
) -> float:
    """Compute set precision/recall F1 over the first k retrieved ids."""
    _validate_rank_inputs(retrieved_ids, relevant_ids, k)
    return _set_f1(tuple(islice(retrieved_ids, k)), relevant_ids)


def _set_f1(
    retrieved_ids: Collection[Hashable],
    relevant_ids: Collection[Hashable],
) -> float:
    """Compute F1 between two identifier sets without applying a rank cutoff."""
    _validate_identifier_collection(retrieved_ids, name="retrieved_ids")
    _validate_identifier_collection(relevant_ids, name="relevant_ids")
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)
    true_positive = len(retrieved_set & relevant_set)
    false_positive = len(retrieved_set - relevant_set)
    false_negative = len(relevant_set - retrieved_set)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mean_reciprocal_rank(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
) -> float:
    """返回首个相关文档名次的倒数，对应 XRAG 的 Mrr。"""
    _validate_identifier_collection(retrieved_ids, name="retrieved_ids")
    _validate_identifier_collection(relevant_ids, name="relevant_ids")
    for index, identifier in enumerate(retrieved_ids):
        if identifier in relevant_ids:
            return 1.0 / (index + 1)
    return 0.0


def hit_at_k(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
    k: int,
) -> float:
    """判断前 k 个检索结果中是否至少包含一个相关文档。"""
    _validate_rank_inputs(retrieved_ids, relevant_ids, k)
    return float(
        any(identifier in relevant_ids for identifier in islice(retrieved_ids, k))
    )


def hit_at_1(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
) -> float:
    """计算与 XRAG ``Hit(retrieval_ids[0:1])`` 相同的 Hit@1。"""
    return hit_at_k(retrieved_ids, relevant_ids, 1)


def hit_at_10(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
) -> float:
    """计算与 XRAG ``Hit(retrieval_ids[0:10])`` 相同的 Hit@10。"""
    return hit_at_k(retrieved_ids, relevant_ids, 10)


def recall_at_k(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
    k: int,
) -> float:
    """计算前 k 个结果覆盖的相关文档比例，供检索实验模块继续使用。"""
    _validate_rank_inputs(retrieved_ids, relevant_ids, k)
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    retrieved_set = set(islice(retrieved_ids, k))
    return len(retrieved_set & relevant_set) / len(relevant_set)


def all_support_at_k(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
    k: int,
) -> float:
    """判断前 k 个结果是否覆盖全部相关文档，供多跳检索实验使用。"""
    _validate_rank_inputs(retrieved_ids, relevant_ids, k)
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    return float(relevant_set <= set(islice(retrieved_ids, k)))


def reciprocal_rank(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
) -> float:
    """保留既有检索实验 API，并委托给 XRAG MRR 单样本实现。"""
    return mean_reciprocal_rank(retrieved_ids, relevant_ids)


def mean_average_precision(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Sequence[Hashable],
) -> float:
    """原样实现 XRAG 的 MAP 公式，而非标准逐位置 Average Precision。"""
    _validate_identifier_collection(retrieved_ids, name="retrieved_ids")
    _validate_identifier_sequence(relevant_ids, name="relevant_ids")
    relevant_sequence = tuple(relevant_ids)
    if not retrieved_ids or not relevant_sequence:
        return 0.0
    score = 0.0
    for index, identifier in enumerate(relevant_sequence):
        if identifier in retrieved_ids:
            score += (index + 1) / (retrieved_ids.index(identifier) + 1)
    return score / len(relevant_sequence)


def discounted_cumulative_gain(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
) -> float:
    """按二值相关性和 ``1/log2(rank+1)`` 折扣计算 XRAG DCG。"""
    _validate_identifier_collection(retrieved_ids, name="retrieved_ids")
    _validate_identifier_collection(relevant_ids, name="relevant_ids")
    score = 0.0
    for index, identifier in enumerate(retrieved_ids):
        if identifier in relevant_ids:
            score += 1.0 / math.log2(index + 2)
    return score


def ideal_discounted_cumulative_gain(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
) -> float:
    """按 XRAG 仅前移已检索相关项的特殊规则计算 IDCG。"""
    _validate_identifier_collection(retrieved_ids, name="retrieved_ids")
    _validate_identifier_collection(relevant_ids, name="relevant_ids")
    matched_ids = [
        identifier for identifier in retrieved_ids if identifier in relevant_ids
    ]
    return discounted_cumulative_gain(matched_ids, relevant_ids)


def normalized_discounted_cumulative_gain(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
) -> float:
    """计算 XRAG NDCG，并在 IDCG 为零时返回零。"""
    dcg_score = discounted_cumulative_gain(retrieved_ids, relevant_ids)
    idcg_score = ideal_discounted_cumulative_gain(retrieved_ids, relevant_ids)
    if idcg_score == 0:
        return 0.0
    return dcg_score / idcg_score


def _validate_identifier_collection(
    identifiers: Collection[Hashable],
    *,
    name: str,
) -> None:
    """校验文档标识符集合，避免把单个字符串误当作 ID 序列。"""
    if isinstance(identifiers, (str, bytes, bytearray)) or not isinstance(
        identifiers, Collection
    ):
        raise TypeError(f"{name} must be a collection of identifiers")
    if not all(isinstance(identifier, Hashable) for identifier in identifiers):
        raise TypeError(f"{name} must contain only hashable identifiers")


def _validate_identifier_sequence(
    identifiers: Sequence[Hashable],
    *,
    name: str,
) -> None:
    """校验需要稳定顺序的文档标识符序列。"""
    if isinstance(identifiers, (str, bytes, bytearray)) or not isinstance(
        identifiers, Sequence
    ):
        raise TypeError(f"{name} must be an ordered sequence of identifiers")
    _validate_identifier_collection(identifiers, name=name)


def _validate_rank_inputs(
    retrieved_ids: Sequence[Hashable],
    relevant_ids: Collection[Hashable],
    k: int,
) -> None:
    """校验带截断名次的检索指标输入。"""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    _validate_identifier_collection(retrieved_ids, name="retrieved_ids")
    _validate_identifier_collection(relevant_ids, name="relevant_ids")
