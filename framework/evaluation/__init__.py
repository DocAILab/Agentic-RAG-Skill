"""公开 RAGSkill framework 的测评数据结构与指标入口。"""

from .evaluator import (
    EvaluationError,
    EvaluationExample,
    EvaluationSummary,
    ExampleMetrics,
    evaluate_batch,
    evaluate_example,
    evaluate_rag_result,
)
from .metrics import (
    exact_match_score,
    f1_score,
    hit_at_1,
    hit_at_10,
    hit_at_k,
    normalize_answer,
)

__all__ = [
    "EvaluationError",
    "EvaluationExample",
    "EvaluationSummary",
    "ExampleMetrics",
    "evaluate_batch",
    "evaluate_example",
    "evaluate_rag_result",
    "exact_match_score",
    "f1_score",
    "hit_at_1",
    "hit_at_10",
    "hit_at_k",
    "normalize_answer",
]
