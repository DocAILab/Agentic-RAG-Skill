"""聚合单样本与批量 RAG 测评结果。"""

from __future__ import annotations

from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .metrics import exact_match_score, f1_score, hit_at_1, hit_at_10


class EvaluationError(ValueError):
    """表示待测样本或 RAG 结果不满足测评输入契约。"""


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    """描述一条同时包含检索与答案监督信号的测评样本。"""

    prediction: str
    gold_answers: str | Sequence[str]
    retrieved_ids: Sequence[Hashable]
    relevant_ids: Collection[Hashable]


@dataclass(frozen=True, slots=True)
class ExampleMetrics:
    """保存一条样本的四项测评指标。"""

    hit_at_1: float
    hit_at_10: float
    em: float
    f1: float

    def to_dict(self) -> dict[str, float]:
        """返回适合 JSON 序列化和实验记录的指标字典。"""
        return {
            "hit@1": self.hit_at_1,
            "hit@10": self.hit_at_10,
            "em": self.em,
            "f1": self.f1,
        }


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """保存一个非空样本集合的宏平均指标。"""

    count: int
    hit_at_1: float
    hit_at_10: float
    em: float
    f1: float

    def to_dict(self) -> dict[str, int | float]:
        """返回包含样本数和宏平均指标的字典。"""
        return {
            "count": self.count,
            "hit@1": self.hit_at_1,
            "hit@10": self.hit_at_10,
            "em": self.em,
            "f1": self.f1,
        }


def evaluate_example(example: EvaluationExample) -> ExampleMetrics:
    """计算一条标准测评样本的 Hit@1、Hit@10、EM 和 F1。"""
    if not isinstance(example, EvaluationExample):
        raise TypeError("example must be an EvaluationExample")
    return ExampleMetrics(
        hit_at_1=hit_at_1(example.retrieved_ids, example.relevant_ids),
        hit_at_10=hit_at_10(example.retrieved_ids, example.relevant_ids),
        em=exact_match_score(example.prediction, example.gold_answers),
        f1=f1_score(example.prediction, example.gold_answers),
    )


def evaluate_batch(examples: Iterable[EvaluationExample]) -> EvaluationSummary:
    """对非空样本集合计算四项指标的宏平均值。"""
    metrics = [evaluate_example(example) for example in examples]
    if not metrics:
        raise EvaluationError("examples must not be empty")

    count = len(metrics)
    return EvaluationSummary(
        count=count,
        hit_at_1=sum(item.hit_at_1 for item in metrics) / count,
        hit_at_10=sum(item.hit_at_10 for item in metrics) / count,
        em=sum(item.em for item in metrics) / count,
        f1=sum(item.f1 for item in metrics) / count,
    )


def evaluate_rag_result(
    result: Mapping[str, Any],
    *,
    gold_answers: str | Sequence[str],
    relevant_ids: Collection[Hashable],
    document_id_key: str = "id",
) -> ExampleMetrics:
    """从 run_rag 结果中提取答案和文档标识符并完成单样本测评。"""
    if not isinstance(result, Mapping):
        raise EvaluationError("result must be a mapping")
    prediction = result.get("answer")
    if not isinstance(prediction, str):
        raise EvaluationError("result.answer must be a string")
    if not isinstance(document_id_key, str) or not document_id_key:
        raise EvaluationError("document_id_key must be a non-empty string")

    documents = result.get("documents")
    if isinstance(documents, (str, bytes, bytearray)) or not isinstance(
        documents, Sequence
    ):
        raise EvaluationError("result.documents must be a sequence")

    retrieved_ids = tuple(
        _extract_document_id(document, index, document_id_key)
        for index, document in enumerate(documents)
    )
    return evaluate_example(
        EvaluationExample(
            prediction=prediction,
            gold_answers=gold_answers,
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
        )
    )


def _extract_document_id(
    document: Any,
    index: int,
    document_id_key: str,
) -> Hashable:
    """从单个检索文档中读取并校验可哈希标识符。"""
    if not isinstance(document, Mapping):
        raise EvaluationError(f"result.documents[{index}] must be a mapping")
    if document_id_key not in document:
        raise EvaluationError(
            f"result.documents[{index}] is missing '{document_id_key}'"
        )
    identifier = document[document_id_key]
    if not isinstance(identifier, Hashable):
        raise EvaluationError(
            f"result.documents[{index}].{document_id_key} must be hashable"
        )
    return identifier
