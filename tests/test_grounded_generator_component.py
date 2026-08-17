from __future__ import annotations

import re
from pathlib import Path

import pytest

from framework import discover_specs, load_runtime_callable

SAMPLE_ROOT = Path(__file__).parents[1] / "framework" / "skills"

DOCUMENTS = (
    {"id": "apple", "text": "Apple trees grow fruit in orchards."},
    {"id": "banana", "text": "Bananas are long yellow fruit."},
    {"id": "citrus", "text": "Lemons and oranges are citrus fruit."},
)


class FakeContext:
    def __init__(self, answer="Apples grow in orchards."):
        """初始化用于断言模型调用内容的提示词记录。"""
        self.answer = answer
        self.prompts = []

    def call_model(self, prompt, *, temperature=0.0, max_tokens=None):
        """记录生成参数并返回固定测试答案。"""
        self.prompts.append((prompt, temperature, max_tokens))
        return self.answer


def _grounded_generator():
    """加载 Grounded Generator Component 的运行时函数。"""
    specs = {spec.package_name: spec for spec in discover_specs(SAMPLE_ROOT)}
    return load_runtime_callable(specs["component-grounded-generator"])


def test_grounded_generator_formats_evidence_and_model_parameters() -> None:
    """验证 Grounded Generator 保持证据顺序，并固定生成参数。"""
    generator = _grounded_generator()
    context = FakeContext(answer="  Bananas are yellow.  ")

    result = generator(
        {
            "query": "What color are bananas?",
            "documents": [DOCUMENTS[1], DOCUMENTS[2]],
            "max_tokens": 32,
        },
        context,
    )

    assert result == {"answer": "Bananas are yellow."}
    assert context.prompts == [
        (
            "Answer the question using only the supplied evidence. "
            "If the evidence is insufficient, say so.\n\n"
            "Question: What color are bananas?\n\n"
            "Evidence:\n"
            "[1] id=banana\n"
            "Bananas are long yellow fruit.\n\n"
            "[2] id=citrus\n"
            "Lemons and oranges are citrus fruit.",
            0.0,
            32,
        )
    ]


def test_grounded_generator_marks_empty_evidence() -> None:
    """验证无证据时 prompt 使用明确占位文本。"""
    generator = _grounded_generator()
    context = FakeContext(answer="Evidence is insufficient.")

    result = generator({"query": "Where do apples grow?", "documents": []}, context)

    assert result == {"answer": "Evidence is insufficient."}
    assert "Evidence:\n[No evidence supplied]" in context.prompts[0][0]


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({"documents": [DOCUMENTS[0]]}, "query is required"),
        ({"query": "Where?", "documents": [{"text": "missing id"}]}, "documents[0].id"),
        ({"query": "Where?", "documents": [{"id": "missing-text"}]}, "documents[0].text"),
        ({"query": "Where?", "documents": "not documents"}, "documents must be"),
    ],
)
def test_grounded_generator_rejects_invalid_inputs(inputs, message) -> None:
    """验证关键输入缺失或结构错误时给出明确异常。"""
    generator = _grounded_generator()

    with pytest.raises(ValueError, match=re.escape(message)):
        generator(inputs, FakeContext())


def test_grounded_generator_rejects_empty_model_answer() -> None:
    """验证模型返回空文本时不会产出无效答案。"""
    generator = _grounded_generator()

    with pytest.raises(ValueError, match="empty answer"):
        generator(
            {"query": "Where?", "documents": [DOCUMENTS[0]]},
            FakeContext(answer="   "),
        )
