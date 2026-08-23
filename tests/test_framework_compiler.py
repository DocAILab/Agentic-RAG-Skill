from __future__ import annotations

import json
from pathlib import Path

import pytest

from framework import (
    CompilationError,
    ExecutionError,
    RAGSelectionPlan,
    RuntimeComponentContext,
    compile_rag_command,
    run_compiled_rag,
    run_rag,
)

SAMPLE_ROOT = Path(__file__).parents[1] / "framework" / "skills"

DOCUMENTS = (
    {"id": "apple", "text": "Apple trees grow fruit in orchards."},
    {"id": "banana", "text": "Bananas are long yellow fruit."},
    {"id": "citrus", "text": "Lemons and oranges are citrus fruit."},
)


class ScriptedExecutorModel:
    """依次返回选择 JSON 与最终生成文本的冻结 Executor 测试替身。"""

    def __init__(self, responses):
        """保存响应队列并初始化调用记录。"""
        self.responses = list(responses)
        self.calls = []

    def generate(
        self,
        prompt,
        *,
        system=None,
        temperature=0.0,
        max_tokens=None,
    ):
        """记录模型请求并返回下一条预设文本。"""
        self.calls.append((prompt, system, temperature, max_tokens))
        return self.responses.pop(0)


class KeywordEmbeddingModel:
    """按关键词生成确定性向量的测试 Embedding 服务。"""

    def embed(self, texts):
        """将 apple、banana 和 citrus 关键词映射到三个向量维度。"""
        return [
            (
                float("apple" in text.lower()),
                float("banana" in text.lower()),
                float("citrus" in text.lower()),
            )
            for text in texts
        ]


def _vanilla_plan() -> RAGSelectionPlan:
    """构造用于编译测试的 Vanilla RAG 选择计划。"""
    return RAGSelectionPlan(
        manage_skill="manage-rag-default",
        manage_guidance="Use one retrieval route.",
        manage_reason="Simple question.",
        agentic_skill="agentic-vanilla-rag",
        agentic_reason="Sequential retrieval is sufficient.",
        component_bindings={
            "rewriter": (),
            "retriever": ("component-bm25-retriever",),
            "reranker": (),
            "generator": ("component-grounded-generator",),
        },
        component_reason="Use lexical retrieval and grounded generation.",
    )


def test_compile_rag_command_executes_concrete_components() -> None:
    """验证编译命令把 Vanilla workflow 绑定到 BM25 和真实生成组件。"""
    model = ScriptedExecutorModel(["Apples grow in orchards."])
    context = RuntimeComponentContext(executor_model=model)
    command = compile_rag_command(
        _vanilla_plan(),
        skill_root=SAMPLE_ROOT,
        context=context,
    )

    result = command(
        {
            "query": "Where do apple trees grow?",
            "documents": DOCUMENTS,
            "top_k": 1,
            "max_tokens": 64,
        }
    )

    assert result["answer"] == "Apples grow in orchards."
    assert result["documents"][0]["id"] == "apple"
    assert [event["step"] for event in result["trace"]] == [
        "retrieve",
        "generate",
    ]
    assert "component-bm25-retriever" in command.instruction
    assert "Apple trees grow fruit" in model.calls[0][0]


def test_run_compiled_rag_executes_hyde_vector_pipeline() -> None:
    """验证 HyDE 假设文档只驱动向量检索，不会混入生成证据。"""
    model = ScriptedExecutorModel(
        [
            "Apple trees grow in imaginary lunar orchards.",
            "Apples grow in orchards.",
        ]
    )
    context = RuntimeComponentContext(
        executor_model=model,
        embedding_model=KeywordEmbeddingModel(),
    )

    result = run_compiled_rag(
        workflow="agentic-vanilla-rag",
        bindings={
            "rewriter": ["component-hyde-rewriter"],
            "retriever": ["component-vector-retriever"],
            "reranker": [],
            "generator": ["component-grounded-generator"],
        },
        request={
            "query": "Where do apple trees grow?",
            "documents": DOCUMENTS,
            "top_k": 1,
            "max_tokens": 64,
        },
        skill_root=SAMPLE_ROOT,
        context=context,
    )

    assert result["answer"] == "Apples grow in orchards."
    assert result["documents"][0]["id"] == "apple"
    assert [event["step"] for event in result["trace"]] == [
        "rewrite",
        "retrieve",
        "generate",
    ]
    assert "Where do apple trees grow?" in model.calls[0][0]
    assert "Where do apple trees grow?" in model.calls[1][0]
    assert "Apple trees grow fruit in orchards." in model.calls[1][0]
    assert "imaginary lunar orchards" not in model.calls[1][0]


def test_compiler_rejects_hyde_with_bm25_retriever() -> None:
    """验证显式编译也会拒绝 HyDE 与 BM25 的错误组合。"""
    context = RuntimeComponentContext(
        executor_model=ScriptedExecutorModel([]),
    )

    with pytest.raises(
        CompilationError,
        match=(
            "component-hyde-rewriter.*requires capability "
            "'retriever'.*component-vector-retriever"
        ),
    ):
        run_compiled_rag(
            workflow="agentic-vanilla-rag",
            bindings={
                "rewriter": ["component-hyde-rewriter"],
                "retriever": ["component-bm25-retriever"],
                "reranker": [],
                "generator": ["component-grounded-generator"],
            },
            request={"query": "Where?", "documents": DOCUMENTS},
            skill_root=SAMPLE_ROOT,
            context=context,
        )


def test_run_compiled_rag_executes_rrfusion_with_two_retrievers() -> None:
    """验证显式单指令能够绑定 BM25、Vector 和 RRFusion workflow。"""
    model = ScriptedExecutorModel(["Fused answer."])
    context = RuntimeComponentContext(
        executor_model=model,
        embedding_model=KeywordEmbeddingModel(),
    )

    result = run_compiled_rag(
        workflow="agentic-rrfusion",
        bindings={
            "retrievers": [
                "component-bm25-retriever",
                "component-vector-retriever",
            ],
            "reranker": [],
            "generator": ["component-grounded-generator"],
        },
        request={"query": "apple fruit", "documents": DOCUMENTS, "top_k": 2},
        skill_root=SAMPLE_ROOT,
        context=context,
    )

    assert result["answer"] == "Fused answer."
    assert result["documents"][0]["id"] == "apple"
    assert result["trace"][0]["branch_count"] == 2


def test_run_compiled_rag_executes_sim_rag_with_real_components() -> None:
    model = ScriptedExecutorModel(
        [
            "Apples grow in orchards.",
            json.dumps(
                {
                    "approved": True,
                    "score": 0.95,
                    "feedback": "The answer is supported.",
                    "issues": [],
                }
            ),
        ]
    )
    context = RuntimeComponentContext(executor_model=model)

    result = run_compiled_rag(
        workflow="agentic-sim-rag",
        bindings={
            "rewriter": [],
            "retriever": ["component-bm25-retriever"],
            "reranker": [],
            "generator": ["component-grounded-generator"],
            "critic": ["component-critic"],
        },
        request={
            "query": "Where do apple trees grow?",
            "documents": DOCUMENTS,
            "top_k": 1,
            "max_iterations": 3,
            "max_tokens": 64,
        },
        skill_root=SAMPLE_ROOT,
        context=context,
    )

    assert result["answer"] == "Apples grow in orchards."
    assert result["documents"][0]["id"] == "apple"
    assert result["trace"][-1]["reason"] == "critic_approved"
    assert len(model.calls) == 2


def test_compile_rag_command_rejects_incompatible_component_binding() -> None:
    """验证编译器会再次拒绝跨 capability 的恶意或损坏绑定。"""
    plan = _vanilla_plan()
    invalid_plan = RAGSelectionPlan(
        manage_skill=plan.manage_skill,
        manage_guidance=plan.manage_guidance,
        manage_reason=plan.manage_reason,
        agentic_skill=plan.agentic_skill,
        agentic_reason=plan.agentic_reason,
        component_bindings={
            "rewriter": (),
            "retriever": ("component-grounded-generator",),
            "reranker": (),
            "generator": ("component-grounded-generator",),
        },
        component_reason="Invalid test binding.",
    )

    with pytest.raises(CompilationError, match="incompatible"):
        compile_rag_command(
            invalid_plan,
            skill_root=SAMPLE_ROOT,
            context=RuntimeComponentContext(ScriptedExecutorModel([])),
        )


def test_vector_component_requires_embedding_model_at_execution() -> None:
    """验证 Vector Retriever 在缺少 Embedding 服务时给出明确执行错误。"""
    context = RuntimeComponentContext(ScriptedExecutorModel([]))

    with pytest.raises(ExecutionError, match="no embedding_model"):
        run_compiled_rag(
            workflow="agentic-vanilla-rag",
            bindings={
                "rewriter": [],
                "retriever": ["component-vector-retriever"],
                "reranker": [],
                "generator": ["component-grounded-generator"],
            },
            request={"query": "apple", "documents": DOCUMENTS, "top_k": 1},
            skill_root=SAMPLE_ROOT,
            context=context,
        )


def test_run_rag_is_one_call_from_selection_to_generated_answer() -> None:
    """验证 run_rag 一次调用完成三级选择、编译、检索和生成。"""
    model = ScriptedExecutorModel(
        [
            json.dumps(
                {
                    "agentic_selection_guidance": "Use one retrieval route.",
                    "reason": "Simple lexical question.",
                }
            ),
            json.dumps(
                {
                    "selected_agentic_skill": "agentic-vanilla-rag",
                    "reason": "Sequential retrieval is sufficient.",
                }
            ),
            json.dumps(
                {
                    "component_bindings": {
                        "rewriter": [],
                        "retriever": ["component-bm25-retriever"],
                        "reranker": [],
                        "generator": ["component-grounded-generator"],
                    },
                    "reason": "Use BM25 and grounded generation.",
                }
            ),
            "Apples grow in orchards.",
        ]
    )

    result = run_rag(
        {
            "query": "Where do apple trees grow?",
            "documents": DOCUMENTS,
            "top_k": 1,
            "max_tokens": 64,
        },
        model=model,
        skill_root=SAMPLE_ROOT,
    )

    assert result["answer"] == "Apples grow in orchards."
    assert result["documents"][0]["id"] == "apple"
    assert result["selection"]["agentic_skill"] == "agentic-vanilla-rag"
    assert result["selection"]["component_bindings"]["retriever"] == [
        "component-bm25-retriever"
    ]
    assert result["compiled_instruction"].startswith("run_compiled_rag(")
    assert len(model.calls) == 4


def test_run_rag_executes_modified_retrievers_in_original_rrfusion_chain() -> None:
    """验证原框架可选择、绑定并执行修改后的两种 Retriever。"""
    model = ScriptedExecutorModel(
        [
            json.dumps(
                {
                    "agentic_selection_guidance": (
                        "Use complementary lexical and semantic retrieval routes."
                    ),
                    "reason": "The compatibility check requires both retrievers.",
                }
            ),
            json.dumps(
                {
                    "selected_agentic_skill": "agentic-rrfusion",
                    "reason": "RRFusion executes and combines both retrieval routes.",
                }
            ),
            json.dumps(
                {
                    "component_bindings": {
                        "retrievers": [
                            "component-bm25-retriever",
                            "component-vector-retriever",
                        ],
                        "reranker": [],
                        "generator": ["component-grounded-generator"],
                    },
                    "reason": "Bind both modified retrievers to the original workflow.",
                }
            ),
            "Apples grow in orchards.",
        ]
    )

    result = run_rag(
        {
            "query": "Where do apple trees grow fruit?",
            "documents": DOCUMENTS,
            "top_k": 2,
            "max_tokens": 64,
        },
        model=model,
        embedding_model=KeywordEmbeddingModel(),
        skill_root=SAMPLE_ROOT,
    )

    assert result["answer"] == "Apples grow in orchards."
    assert result["documents"][0]["id"] == "apple"
    assert result["selection"]["agentic_skill"] == "agentic-rrfusion"
    assert result["selection"]["component_bindings"]["retrievers"] == [
        "component-bm25-retriever",
        "component-vector-retriever",
    ]
    assert result["trace"] == [
        {
            "step": "parallel_retrieve_and_fuse",
            "branch_count": 2,
            "document_count": 2,
        },
        {"step": "generate"},
    ]
    assert "component-bm25-retriever" in result["compiled_instruction"]
    assert "component-vector-retriever" in result["compiled_instruction"]
    assert "Apple trees grow fruit" in model.calls[-1][0]
    assert len(model.calls) == 4
