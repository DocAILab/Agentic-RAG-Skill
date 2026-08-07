from __future__ import annotations

import pytest

from framework import (
    EvaluationError,
    EvaluationExample,
    evaluate_batch,
    evaluate_example,
    evaluate_rag_result,
    exact_match_score,
    f1_score,
    hit_at_1,
    hit_at_10,
    hit_at_k,
    normalize_answer,
)


def test_normalize_answer_matches_hotpotqa_rules() -> None:
    """验证大小写、英文标点、冠词和多余空白会被统一处理。"""
    assert normalize_answer("  The, Quick  Brown Fox! ") == "quick brown fox"


def test_exact_match_uses_normalization_and_best_alias() -> None:
    """验证 EM 会归一化答案并在多个标准答案中取最高分。"""
    assert exact_match_score("The Eiffel Tower.", "eiffel tower") == 1.0
    assert exact_match_score("Paris", ["London", "Paris, France", "Paris"]) == 1.0
    assert exact_match_score("Berlin", ["London", "Paris"]) == 0.0


def test_f1_uses_token_overlap_and_best_alias() -> None:
    """验证 F1 使用词元重叠，并在答案别名中选择最高分。"""
    assert f1_score("blue car", "the blue fast car") == pytest.approx(0.8)
    assert f1_score("New York", ["Boston", "New York City"]) == pytest.approx(0.8)


def test_f1_preserves_hotpotqa_special_answer_rule() -> None:
    """验证 yes、no 和 noanswer 与其他答案不一致时不会获得部分分。"""
    assert f1_score("yes indeed", "yes") == 0.0
    assert f1_score("yes", "no") == 0.0
    assert f1_score("yes", "yes") == 1.0


def test_hit_metrics_use_retrieval_rank() -> None:
    """验证 Hit@1 和 Hit@10 按检索顺序判断任一相关文档是否出现。"""
    retrieved_ids = ["irrelevant", "gold", "another"]
    relevant_ids = {"gold", "other-gold"}

    assert hit_at_1(retrieved_ids, relevant_ids) == 0.0
    assert hit_at_10(retrieved_ids, relevant_ids) == 1.0
    assert hit_at_k(retrieved_ids, relevant_ids, 2) == 1.0


def test_hit_at_k_rejects_invalid_k_and_identifier_collections() -> None:
    """验证 Hit@K 拒绝非法 K 和被误传为集合的单个字符串。"""
    with pytest.raises(ValueError, match="positive integer"):
        hit_at_k(["doc-1"], {"doc-1"}, 0)
    with pytest.raises(TypeError, match="collection"):
        hit_at_k("doc-1", {"doc-1"}, 1)


def test_evaluate_example_and_batch_return_macro_averages() -> None:
    """验证单样本入口与批量入口返回一致的四项指标及宏平均。"""
    positive = EvaluationExample(
        prediction="The Eiffel Tower",
        gold_answers="eiffel tower",
        retrieved_ids=("gold", "noise"),
        relevant_ids={"gold"},
    )
    negative = EvaluationExample(
        prediction="Berlin",
        gold_answers="Paris",
        retrieved_ids=("noise",),
        relevant_ids={"gold"},
    )

    assert evaluate_example(positive).to_dict() == {
        "hit@1": 1.0,
        "hit@10": 1.0,
        "em": 1.0,
        "f1": 1.0,
    }
    assert evaluate_batch([positive, negative]).to_dict() == {
        "count": 2,
        "hit@1": 0.5,
        "hit@10": 0.5,
        "em": 0.5,
        "f1": 0.5,
    }


def test_evaluate_batch_rejects_empty_input() -> None:
    """验证批量测评不会为无样本输入伪造零分结果。"""
    with pytest.raises(EvaluationError, match="must not be empty"):
        evaluate_batch([])


def test_evaluate_rag_result_consumes_framework_output() -> None:
    """验证便捷入口能直接读取 run_rag 风格的答案和文档列表。"""
    metrics = evaluate_rag_result(
        {
            "answer": "Paris",
            "documents": [
                {"id": "noise", "text": "Unrelated."},
                {"id": "support", "text": "Paris is the capital of France."},
            ],
        },
        gold_answers=["Paris", "Paris, France"],
        relevant_ids={"support"},
    )

    assert metrics.to_dict() == {
        "hit@1": 0.0,
        "hit@10": 1.0,
        "em": 1.0,
        "f1": 1.0,
    }


def test_evaluate_rag_result_validates_document_ids() -> None:
    """验证 run_rag 结果缺少文档标识符时会返回明确错误。"""
    with pytest.raises(EvaluationError, match="missing 'id'"):
        evaluate_rag_result(
            {"answer": "Paris", "documents": [{"text": "Paris."}]},
            gold_answers="Paris",
            relevant_ids={"support"},
        )
