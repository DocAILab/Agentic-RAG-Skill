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
    all_support_at_k,
    exact_match_score,
    f1_score,
    hit_at_1,
    hit_at_10,
    hit_at_k,
    normalize_answer,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "EvaluationError",
    "EvaluationExample",
    "EvaluationSummary",
    "ExampleMetrics",
    "all_support_at_k",
    "evaluate_batch",
    "evaluate_example",
    "evaluate_rag_result",
    "exact_match_score",
    "f1_score",
    "hit_at_1",
    "hit_at_10",
    "hit_at_k",
    "normalize_answer",
    "recall_at_k",
    "reciprocal_rank",
]
