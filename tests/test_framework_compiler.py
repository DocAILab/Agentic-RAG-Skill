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

    def __init__(self):
        """初始化底层向量调用记录。"""
        self.calls = []

    def embed(self, texts):
        """将 apple、banana 和 citrus 关键词映射到三个向量维度。"""
        self.calls.append(list(texts))
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


def _vector_plan() -> RAGSelectionPlan:
    """构造使用持久化 Vector Retriever 的 Vanilla RAG 计划。"""
    return RAGSelectionPlan(
        manage_skill="manage-rag-default",
        manage_guidance="Use vector retrieval.",
        manage_reason="Shared corpus evaluation.",
        agentic_skill="agentic-vanilla-rag",
        agentic_reason="One retrieval route is sufficient.",
        component_bindings={
            "rewriter": (),
            "retriever": ("component-vector-retriever",),
            "reranker": (),
            "generator": ("component-grounded-generator",),
        },
        component_reason="Use one cached vector index.",
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
    assert [timing["component"] for timing in result["component_timings"]] == [
        "component-bm25-retriever",
        "component-grounded-generator",
    ]
    assert all(
        timing["duration_seconds"] >= 0
        for timing in result["component_timings"]
    )
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


def test_run_compiled_conditional_rag_executes_hybrid_route() -> None:
    """验证 Conditional workflow 可安全绑定 HyDE、BM25 和 Vector。"""
    hypothetical_document = (
        "Apple trees grow in imaginary lunar orchards."
    )
    model = ScriptedExecutorModel(
        [
            json.dumps(
                {
                    "route": "hybrid",
                    "reason": "Exact and semantic evidence are useful.",
                    "confidence": 0.9,
                }
            ),
            hypothetical_document,
            "Apples grow in orchards.",
        ]
    )
    context = RuntimeComponentContext(
        executor_model=model,
        embedding_model=KeywordEmbeddingModel(),
    )

    result = run_compiled_rag(
        workflow="agentic-conditional-rag",
        bindings={
            "classifier": ["component-classifier"],
            "rewriter": ["component-hyde-rewriter"],
            "lexical_retriever": ["component-bm25-retriever"],
            "semantic_retriever": ["component-vector-retriever"],
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
    assert result["route"] == "hybrid"
    assert result["documents"][0]["id"] == "apple"
    assert [event["step"] for event in result["trace"]] == [
        "classify",
        "rewrite",
        "retrieve_and_fuse",
        "generate",
    ]
    assert [
        timing["component"]
        for timing in result["component_timings"]
    ] == [
        "component-classifier",
        "component-hyde-rewriter",
        "component-bm25-retriever",
        "component-vector-retriever",
        "component-grounded-generator",
    ]
    assert "Where do apple trees grow?" in model.calls[0][0]
    assert "Where do apple trees grow?" in model.calls[1][0]
    assert "Where do apple trees grow?" in model.calls[2][0]
    assert "Apple trees grow fruit in orchards." in model.calls[2][0]
    assert "imaginary lunar orchards" not in model.calls[2][0]


def test_parallel_rag_accepts_hyde_with_mixed_retrievers() -> None:
    """验证 parallel workflow 允许 HyDE 与 BM25+Vector 混合分支。"""
    model = ScriptedExecutorModel(
        [
            "Apple trees grow fruit in orchards.",
            "Fused answer.",
        ]
    )
    context = RuntimeComponentContext(
        executor_model=model,
        embedding_model=KeywordEmbeddingModel(),
    )

    result = run_compiled_rag(
        workflow="agentic-parallel-rag",
        bindings={
            "rewriter": ["component-hyde-rewriter"],
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
    assert result["trace"][0]["step"] == "rewrite"


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


def test_runtime_context_caches_corpus_embeddings_across_queries() -> None:
    """验证共享语料只编码一次，后续问题仅编码新的查询文本。"""
    embedding_model = KeywordEmbeddingModel()
    context = RuntimeComponentContext(
        executor_model=ScriptedExecutorModel(["First answer.", "Second answer."]),
        embedding_model=embedding_model,
    )
    command = compile_rag_command(
        _vector_plan(),
        skill_root=SAMPLE_ROOT,
        context=context,
    )

    command({"query": "apple", "documents": DOCUMENTS, "top_k": 1})
    command({"query": "banana", "documents": DOCUMENTS, "top_k": 1})

    assert embedding_model.calls[0] == [document["text"] for document in DOCUMENTS]
    assert embedding_model.calls[1] == [
        "Represent this sentence for searching relevant passages: apple"
    ]
    assert embedding_model.calls[2] == [
        "Represent this sentence for searching relevant passages: banana"
    ]
    assert context.embedding_cache_info() == {
        "entries": 2,
        "hits": 0,
        "misses": 2,
    }
    index_info = context.vector_index_cache_info()
    assert index_info["builds"] == 1
    assert index_info["memory_hits"] == 1
    assert index_info["queries"] == 2


def test_runtime_context_reloads_persisted_index_without_reencoding_corpus(
    tmp_path,
) -> None:
    """验证新运行可从磁盘加载索引，底层模型只接收新的查询文本。"""
    cache_dir = tmp_path / "vector-indexes"
    first_embedding = KeywordEmbeddingModel()
    first_context = RuntimeComponentContext(
        executor_model=ScriptedExecutorModel(["First answer."]),
        embedding_model=first_embedding,
        vector_index_cache_dir=cache_dir,
        embedding_fingerprint="test-embedding-v1",
    )
    first_command = compile_rag_command(
        _vector_plan(),
        skill_root=SAMPLE_ROOT,
        context=first_context,
    )

    first_command({"query": "apple", "documents": DOCUMENTS, "top_k": 1})

    index_directories = list(cache_dir.iterdir())
    assert len(index_directories) == 1
    assert (index_directories[0] / "manifest.json").is_file()
    assert (index_directories[0] / "vectors.npy").is_file()
    assert first_embedding.calls[0] == [
        document["text"] for document in DOCUMENTS
    ]

    second_embedding = KeywordEmbeddingModel()
    second_context = RuntimeComponentContext(
        executor_model=ScriptedExecutorModel(["Second answer."]),
        embedding_model=second_embedding,
        vector_index_cache_dir=cache_dir,
        embedding_fingerprint="test-embedding-v1",
    )
    second_command = compile_rag_command(
        _vector_plan(),
        skill_root=SAMPLE_ROOT,
        context=second_context,
    )

    second_command({"query": "banana", "documents": DOCUMENTS, "top_k": 1})

    assert second_embedding.calls == [
        ["Represent this sentence for searching relevant passages: banana"]
    ]
    index_info = second_context.vector_index_cache_info()
    assert index_info["disk_loads"] == 1
    assert index_info["builds"] == 0
    assert index_info["last_index"]["source"] == "disk"


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


def test_run_rag_executes_conditional_hybrid_pipeline() -> None:
    """验证一次调用可选择并执行完整 Conditional Hybrid pipeline。"""
    hypothetical_document = (
        "Apple trees grow in imaginary lunar orchards."
    )
    model = ScriptedExecutorModel(
        [
            json.dumps(
                {
                    "agentic_selection_guidance": (
                        "Choose retrieval according to each request."
                    ),
                    "reason": "The retrieval strategy is request dependent.",
                }
            ),
            json.dumps(
                {
                    "selected_agentic_skill": "agentic-conditional-rag",
                    "reason": "Runtime routing is appropriate.",
                }
            ),
            json.dumps(
                {
                    "component_bindings": {
                        "classifier": ["component-classifier"],
                        "rewriter": ["component-hyde-rewriter"],
                        "lexical_retriever": [
                            "component-bm25-retriever"
                        ],
                        "semantic_retriever": [
                            "component-vector-retriever"
                        ],
                        "reranker": [],
                        "generator": [
                            "component-grounded-generator"
                        ],
                    },
                    "reason": "Bind safe route-specific components.",
                }
            ),
            json.dumps(
                {
                    "route": "hybrid",
                    "reason": "Exact and semantic evidence are useful.",
                    "confidence": 0.9,
                }
            ),
            hypothetical_document,
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
        embedding_model=KeywordEmbeddingModel(),
        skill_root=SAMPLE_ROOT,
    )

    assert result["answer"] == "Apples grow in orchards."
    assert result["documents"][0]["id"] == "apple"
    assert result["route"] == "hybrid"
    assert result["selection"]["agentic_skill"] == (
        "agentic-conditional-rag"
    )
    assert result["selection"]["component_bindings"] == {
        "classifier": ["component-classifier"],
        "rewriter": ["component-hyde-rewriter"],
        "lexical_retriever": ["component-bm25-retriever"],
        "semantic_retriever": ["component-vector-retriever"],
        "reranker": [],
        "generator": ["component-grounded-generator"],
    }
    assert [event["step"] for event in result["trace"]] == [
        "classify",
        "rewrite",
        "retrieve_and_fuse",
        "generate",
    ]
    assert "component-classifier" in result["compiled_instruction"]
    assert "component-bm25-retriever" in result["compiled_instruction"]
    assert "component-vector-retriever" in result["compiled_instruction"]
    assert "Where do apple trees grow?" in model.calls[-1][0]
    assert "Apple trees grow fruit" in model.calls[-1][0]
    assert "imaginary lunar orchards" not in model.calls[-1][0]
    assert len(model.calls) == 6


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
