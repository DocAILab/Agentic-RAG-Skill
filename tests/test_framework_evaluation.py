from __future__ import annotations

import math

import pytest

from framework import (
    EvaluationError,
    EvaluationExample,
    XRAGGenerationEvaluator,
    discounted_cumulative_gain,
    evaluate_batch,
    evaluate_example,
    evaluate_rag_result,
    hit_at_1,
    hit_at_10,
    hit_at_k,
    ideal_discounted_cumulative_gain,
    mean_average_precision,
    mean_reciprocal_rank,
    normalized_discounted_cumulative_gain,
    retrieval_f1,
    retrieval_f1_at_1,
    retrieval_f1_at_k,
)


class FixedGenerationEvaluator:
    """提供无需加载 GPT-2 的固定生成指标测试后端。"""

    def __init__(self) -> None:
        """初始化生成评测调用记录。"""
        self.calls: list[tuple[str, str | tuple[str, ...]]] = []

    def evaluate(self, prediction, references):
        """根据预测是否为 ``correct`` 返回全一或全零指标。"""
        recorded_references = (
            references if isinstance(references, str) else tuple(references)
        )
        self.calls.append((prediction, recorded_references))
        score = 1.0 if prediction == "correct" else 0.0
        return {
            "chrf": score,
            "chrf++": score,
            "meteor": score,
            "r1": score,
            "r2": score,
            "rl": score,
            "ppl": score,
            "cer": score,
            "wer": score,
        }


def test_retrieval_metrics_match_xrag_formulas() -> None:
    """验证八项检索指标逐项复现 XRAG 的固定公式。"""
    retrieved = ["noise-a", "gold-b", "noise-b", "gold-a"]
    relevant = ["gold-a", "gold-b"]
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3)

    assert retrieval_f1_at_1(retrieved, relevant) == 0.0
    assert retrieval_f1(retrieved, relevant) == 0.5
    assert retrieval_f1_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)
    assert mean_reciprocal_rank(retrieved, relevant) == 0.5
    assert hit_at_1(retrieved, relevant) == 0.0
    assert hit_at_10(retrieved, relevant) == 1.0
    assert mean_average_precision(retrieved, relevant) == 0.625
    assert discounted_cumulative_gain(retrieved, relevant) == pytest.approx(dcg)
    assert ideal_discounted_cumulative_gain(retrieved, relevant) == pytest.approx(
        idcg
    )
    assert normalized_discounted_cumulative_gain(
        retrieved, relevant
    ) == pytest.approx(dcg / idcg)


def test_xrag_idcg_only_idealizes_relevant_items_that_were_retrieved() -> None:
    """验证 XRAG IDCG 不会把漏检的标准相关文档补入理想排序。"""
    retrieved = ["noise", "gold-a"]
    relevant = ["gold-a", "gold-b"]

    assert ideal_discounted_cumulative_gain(retrieved, relevant) == 1.0
    assert normalized_discounted_cumulative_gain(
        retrieved, relevant
    ) == pytest.approx(1 / math.log2(3))


def test_hit_at_k_rejects_invalid_k_and_identifier_collections() -> None:
    """验证 Hit@K 拒绝非法 K 和被误传为集合的单个字符串。"""
    with pytest.raises(ValueError, match="positive integer"):
        hit_at_k(["doc-1"], {"doc-1"}, 0)
    with pytest.raises(TypeError, match="collection"):
        hit_at_k("doc-1", {"doc-1"}, 1)


def test_generation_metrics_match_xrag_underlying_implementations() -> None:
    """验证 ChrF、METEOR、ROUGE、PPL、CER 与 WER 的 XRAG 数值口径。"""
    evaluator = XRAGGenerationEvaluator(perplexity_calculator=lambda _: 42.0)

    scores = evaluator.evaluate(
        "The cat sat on mat.",
        "The cat is on the mat.",
    )

    assert scores == pytest.approx(
        {
            "chrf": 0.3950649007312187,
            "chrf++": 0.4313252757461113,
            "meteor": 0.3389830508474576,
            "r1": 0.7272727272727272,
            "r2": 0.22222222222222224,
            "rl": 0.7272727272727272,
            "ppl": 42.0,
            "cer": 0.3181818181818182,
            "wer": 0.3333333333333333,
        }
    )


def test_generation_metrics_keep_xrag_multi_reference_and_ppl_rules() -> None:
    """验证多参考 WER/CER 取最大值且 PPL 超过 1600 时置零。"""
    evaluator = XRAGGenerationEvaluator(perplexity_calculator=lambda _: 1601.2)

    scores = evaluator.evaluate("alpha beta", ["alpha beta", "wrong answer"])

    assert scores["chrf"] == 1.0
    assert scores["chrf++"] == 1.0
    assert scores["ppl"] == 0.0
    assert scores["wer"] == 1.0
    assert scores["cer"] == pytest.approx(11 / 12)


def test_evaluate_example_and_batch_return_nested_macro_averages() -> None:
    """验证单样本和批量结果分组保存检索与生成指标。"""
    evaluator = FixedGenerationEvaluator()
    positive = EvaluationExample(
        prediction="correct",
        gold_answers="correct",
        retrieved_ids=("gold", "noise"),
        relevant_ids=("gold",),
    )
    negative = EvaluationExample(
        prediction="wrong",
        gold_answers="correct",
        retrieved_ids=("noise",),
        relevant_ids=("gold",),
    )

    positive_metrics = evaluate_example(
        positive,
        generation_evaluator=evaluator,
    ).to_dict()
    summary = evaluate_batch(
        [positive, negative],
        generation_evaluator=evaluator,
    ).to_dict()

    assert positive_metrics["retrieval"] == {
        "F1@1": 1.0,
        "F1": 1.0,
        "MRR": 1.0,
        "Hit@1": 1.0,
        "Hit@10": 1.0,
        "MAP": 1.0,
        "NDCG": 1.0,
        "DCG": 1.0,
        "IDCG": 1.0,
    }
    assert positive_metrics["generation"] == {
        "ChrF": 1.0,
        "ChrF++": 1.0,
        "METEOR": 1.0,
        "R1": 1.0,
        "R2": 1.0,
        "RL": 1.0,
        "PPL": 1.0,
        "CER": 1.0,
        "WER": 1.0,
    }
    assert summary["count"] == 2
    assert summary["retrieval"]["F1@1"] == 0.5
    assert summary["retrieval"]["F1"] == 0.5
    assert summary["retrieval"]["Hit@1"] == 0.5
    assert summary["generation"] == {
        name: 0.5 for name in positive_metrics["generation"]
    }
    assert "EM" not in positive_metrics["generation"]
    assert "F1" not in positive_metrics["generation"]


def test_evaluate_batch_rejects_empty_input() -> None:
    """验证批量测评不会为无样本输入伪造零分结果。"""
    with pytest.raises(EvaluationError, match="must not be empty"):
        evaluate_batch([], generation_evaluator=FixedGenerationEvaluator())


def test_evaluate_rag_result_consumes_framework_output() -> None:
    """验证便捷入口能直接读取 run_rag 风格答案和文档列表。"""
    metrics = evaluate_rag_result(
        {
            "answer": "correct",
            "documents": [
                {"id": "noise", "text": "Unrelated."},
                {"id": "support", "text": "Supporting evidence."},
            ],
        },
        gold_answers="correct",
        relevant_ids=("support",),
        generation_evaluator=FixedGenerationEvaluator(),
    )

    assert metrics.retrieval.hit_at_1 == 0.0
    assert metrics.retrieval.hit_at_10 == 1.0
    assert metrics.generation.chrf == 1.0


def test_evaluate_rag_result_validates_document_ids() -> None:
    """验证 run_rag 结果缺少文档标识符时会返回明确错误。"""
    with pytest.raises(EvaluationError, match="missing 'id'"):
        evaluate_rag_result(
            {"answer": "correct", "documents": [{"text": "Supporting."}]},
            gold_answers="correct",
            relevant_ids=("support",),
            generation_evaluator=FixedGenerationEvaluator(),
        )
