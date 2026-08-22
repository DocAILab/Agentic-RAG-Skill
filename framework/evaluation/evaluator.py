"""聚合与 XRAG 对齐的检索和生成测评结果。"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .generation import (
    get_default_generation_evaluator,
    validate_generation_scores,
)
from .metrics import (
    discounted_cumulative_gain,
    hit_at_1,
    hit_at_10,
    ideal_discounted_cumulative_gain,
    mean_average_precision,
    mean_reciprocal_rank,
    normalized_discounted_cumulative_gain,
    retrieval_f1,
    retrieval_f1_at_1,
)


class EvaluationError(ValueError):
    """表示待测样本或 RAG 结果不满足测评输入契约。"""


class GenerationEvaluator(Protocol):
    """定义可替换生成指标后端必须实现的最小接口。"""

    def evaluate(
        self,
        prediction: str,
        references: str | Sequence[str],
    ) -> Mapping[str, float]:
        """根据预测文本和一个或多个参考答案返回九项生成指标。"""
        ...


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    """描述一条同时包含检索与生成监督信号的测评样本。"""

    prediction: str
    gold_answers: str | Sequence[str]
    retrieved_ids: Sequence[Hashable]
    relevant_ids: Sequence[Hashable]


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Store one example's retrieval metrics, including Top-1 and Top-n F1."""

    f1: float
    f1_at_1: float
    mrr: float
    hit_at_1: float
    hit_at_10: float
    map: float
    ndcg: float
    dcg: float
    idcg: float

    def to_dict(self) -> dict[str, float]:
        """返回使用实验报告指标名的检索指标字典。"""
        return {
            "F1@1": self.f1_at_1,
            "F1": self.f1,
            "MRR": self.mrr,
            "Hit@1": self.hit_at_1,
            "Hit@10": self.hit_at_10,
            "MAP": self.map,
            "NDCG": self.ndcg,
            "DCG": self.dcg,
            "IDCG": self.idcg,
        }


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """保存一条样本的九项 XRAG 生成指标。"""

    chrf: float
    chrf_pp: float
    meteor: float
    r1: float
    r2: float
    rl: float
    ppl: float
    cer: float
    wer: float

    @classmethod
    def from_mapping(cls, scores: Mapping[str, float]) -> GenerationMetrics:
        """从外部评测后端结果创建并校验生成指标对象。"""
        normalized = validate_generation_scores(scores)
        return cls(
            chrf=normalized["chrf"],
            chrf_pp=normalized["chrf++"],
            meteor=normalized["meteor"],
            r1=normalized["r1"],
            r2=normalized["r2"],
            rl=normalized["rl"],
            ppl=normalized["ppl"],
            cer=normalized["cer"],
            wer=normalized["wer"],
        )

    def to_dict(self) -> dict[str, float]:
        """返回使用实验报告指标名的生成指标字典。"""
        return {
            "ChrF": self.chrf,
            "ChrF++": self.chrf_pp,
            "METEOR": self.meteor,
            "R1": self.r1,
            "R2": self.r2,
            "RL": self.rl,
            "PPL": self.ppl,
            "CER": self.cer,
            "WER": self.wer,
        }


@dataclass(frozen=True, slots=True)
class ExampleMetrics:
    """按检索与生成两组保存单条样本指标。"""

    retrieval: RetrievalMetrics
    generation: GenerationMetrics

    def to_dict(self) -> dict[str, dict[str, float]]:
        """返回避免混淆检索 F1 与生成指标的嵌套字典。"""
        return {
            "retrieval": self.retrieval.to_dict(),
            "generation": self.generation.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """保存非空样本集合的指标宏平均。"""

    count: int
    retrieval: RetrievalMetrics
    generation: GenerationMetrics

    def to_dict(self) -> dict[str, int | dict[str, float]]:
        """返回样本数和两组宏平均指标。"""
        return {
            "count": self.count,
            "retrieval": self.retrieval.to_dict(),
            "generation": self.generation.to_dict(),
        }


def evaluate_example(
    example: EvaluationExample,
    *,
    generation_evaluator: GenerationEvaluator | None = None,
) -> ExampleMetrics:
    """Compute one example's retrieval and generation metrics."""
    if not isinstance(example, EvaluationExample):
        raise TypeError("example must be an EvaluationExample")
    evaluator = generation_evaluator or get_default_generation_evaluator()
    retrieval = RetrievalMetrics(
        f1=retrieval_f1(example.retrieved_ids, example.relevant_ids),
        f1_at_1=retrieval_f1_at_1(example.retrieved_ids, example.relevant_ids),
        mrr=mean_reciprocal_rank(example.retrieved_ids, example.relevant_ids),
        hit_at_1=hit_at_1(example.retrieved_ids, example.relevant_ids),
        hit_at_10=hit_at_10(example.retrieved_ids, example.relevant_ids),
        map=mean_average_precision(example.retrieved_ids, example.relevant_ids),
        ndcg=normalized_discounted_cumulative_gain(
            example.retrieved_ids,
            example.relevant_ids,
        ),
        dcg=discounted_cumulative_gain(
            example.retrieved_ids,
            example.relevant_ids,
        ),
        idcg=ideal_discounted_cumulative_gain(
            example.retrieved_ids,
            example.relevant_ids,
        ),
    )
    generation = GenerationMetrics.from_mapping(
        evaluator.evaluate(example.prediction, example.gold_answers)
    )
    return ExampleMetrics(retrieval=retrieval, generation=generation)


def summarize_metrics(metrics: Iterable[ExampleMetrics]) -> EvaluationSummary:
    """对已计算的单题结果取宏平均，避免重复运行昂贵的生成指标。"""
    metric_list = list(metrics)
    if not metric_list:
        raise EvaluationError("metrics must not be empty")
    if not all(isinstance(item, ExampleMetrics) for item in metric_list):
        raise TypeError("metrics must contain only ExampleMetrics")
    count = len(metric_list)
    return EvaluationSummary(
        count=count,
        retrieval=RetrievalMetrics(
            f1=_mean(item.retrieval.f1 for item in metric_list),
            f1_at_1=_mean(item.retrieval.f1_at_1 for item in metric_list),
            mrr=_mean(item.retrieval.mrr for item in metric_list),
            hit_at_1=_mean(item.retrieval.hit_at_1 for item in metric_list),
            hit_at_10=_mean(item.retrieval.hit_at_10 for item in metric_list),
            map=_mean(item.retrieval.map for item in metric_list),
            ndcg=_mean(item.retrieval.ndcg for item in metric_list),
            dcg=_mean(item.retrieval.dcg for item in metric_list),
            idcg=_mean(item.retrieval.idcg for item in metric_list),
        ),
        generation=GenerationMetrics(
            chrf=_mean(item.generation.chrf for item in metric_list),
            chrf_pp=_mean(item.generation.chrf_pp for item in metric_list),
            meteor=_mean(item.generation.meteor for item in metric_list),
            r1=_mean(item.generation.r1 for item in metric_list),
            r2=_mean(item.generation.r2 for item in metric_list),
            rl=_mean(item.generation.rl for item in metric_list),
            ppl=_mean(item.generation.ppl for item in metric_list),
            cer=_mean(item.generation.cer for item in metric_list),
            wer=_mean(item.generation.wer for item in metric_list),
        ),
    )


def evaluate_batch(
    examples: Iterable[EvaluationExample],
    *,
    generation_evaluator: GenerationEvaluator | None = None,
) -> EvaluationSummary:
    """计算非空样本集合的全部指标宏平均。"""
    metrics = [
        evaluate_example(example, generation_evaluator=generation_evaluator)
        for example in examples
    ]
    if not metrics:
        raise EvaluationError("examples must not be empty")
    return summarize_metrics(metrics)


def evaluate_rag_result(
    result: Mapping[str, Any],
    *,
    gold_answers: str | Sequence[str],
    relevant_ids: Sequence[Hashable],
    document_id_key: str = "id",
    generation_evaluator: GenerationEvaluator | None = None,
) -> ExampleMetrics:
    """从 ``run_rag`` 结果提取答案和文档 ID 并完成单样本测评。"""
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
        ),
        generation_evaluator=generation_evaluator,
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


def _mean(values: Iterable[float]) -> float:
    """对已知非空指标序列计算算术平均。"""
    value_list = list(values)
    return sum(value_list) / len(value_list)
