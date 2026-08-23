from __future__ import annotations

import json
from pathlib import Path

import pytest

from framework import (
    AgenticStageResult,
    AnthropicModelClient,
    ManageStageResult,
    OpenAICompatibleEmbeddingClient,
    OpenAICompatibleModelClient,
    SelectionError,
    SentenceTransformerEmbeddingClient,
    create_embedding_client,
    create_model_client,
    discover_specs,
    run_manage_stage,
    select_agentic_skill,
    select_component_skills,
    select_rag_plan,
)

SAMPLE_ROOT = Path(__file__).parents[1] / "framework" / "skills"


class RecordingTransport:
    """记录模型 HTTP 请求并返回预设 JSON 响应。"""

    def __init__(self, response):
        """保存预设响应并初始化请求记录。"""
        self.response = response
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        """记录一次 transport 调用并返回预设响应。"""
        self.calls.append((url, dict(headers), dict(payload), timeout))
        return self.response


class ScriptedModel:
    """按顺序返回 JSON 文本的分级选择测试模型。"""

    def __init__(self, responses):
        """保存响应队列并初始化模型调用记录。"""
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
        """记录提示词和生成参数，并返回下一条预设响应。"""
        self.calls.append((prompt, system, temperature, max_tokens))
        return self.responses.pop(0)


class RecordingSentenceEncoder:
    """记录本地 sentence-transformers 编码参数的测试替身。"""

    def __init__(self):
        """初始化编码调用记录。"""
        self.calls = []

    def encode(self, texts, **options):
        """记录文本和编码参数，并返回确定性二维向量。"""
        self.calls.append((list(texts), dict(options)))
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


def test_openai_compatible_client_uses_chat_completions_shape() -> None:
    """验证 OpenAI-compatible 客户端的地址、鉴权、消息和结果解析。"""
    transport = RecordingTransport(
        {"choices": [{"message": {"content": "openai answer"}}]}
    )
    client = OpenAICompatibleModelClient(
        model="local-executor",
        api_key="openai-secret",
        base_url="http://localhost:8000/v1",
        transport=transport,
    )

    answer = client.generate(
        "question",
        system="system instruction",
        temperature=0.2,
        max_tokens=64,
    )

    assert answer == "openai answer"
    url, headers, payload, timeout = transport.calls[0]
    assert url == "http://localhost:8000/v1/chat/completions"
    assert headers["authorization"] == "Bearer openai-secret"
    assert payload == {
        "model": "local-executor",
        "messages": [
            {"role": "system", "content": "system instruction"},
            {"role": "user", "content": "question"},
        ],
        "temperature": 0.2,
        "max_tokens": 64,
    }
    assert timeout == 120.0


def test_anthropic_client_uses_messages_shape() -> None:
    """验证 Anthropic 客户端的地址、版本头、system 字段和文本块解析。"""
    transport = RecordingTransport(
        {"content": [{"type": "text", "text": "anthropic answer"}]}
    )
    client = create_model_client(
        "anthropic",
        model="claude-test",
        api_key="anthropic-secret",
        transport=transport,
    )
    assert isinstance(client, AnthropicModelClient)

    answer = client.generate(
        "question",
        system="system instruction",
        temperature=0.1,
        max_tokens=96,
    )

    assert answer == "anthropic answer"
    url, headers, payload, _ = transport.calls[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "anthropic-secret"
    assert headers["anthropic-version"] == "2023-06-01"
    assert payload == {
        "model": "claude-test",
        "messages": [{"role": "user", "content": "question"}],
        "temperature": 0.1,
        "max_tokens": 96,
        "system": "system instruction",
    }


def test_openai_compatible_embedding_client_restores_index_order() -> None:
    """验证 OpenAI-compatible 向量客户端按响应 index 恢复输入顺序。"""
    transport = RecordingTransport(
        {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }
    )
    client = OpenAICompatibleEmbeddingClient(
        model="embedding-model",
        api_key="embedding-secret",
        base_url="http://localhost:8000/v1",
        transport=transport,
    )

    embeddings = client.embed(["apple", "banana"])

    assert embeddings == [[1.0, 0.0], [0.0, 1.0]]
    url, headers, payload, _ = transport.calls[0]
    assert url == "http://localhost:8000/v1/embeddings"
    assert headers["authorization"] == "Bearer embedding-secret"
    assert payload == {
        "model": "embedding-model",
        "input": ["apple", "banana"],
    }


def test_sentence_transformer_embedding_client_encodes_locally() -> None:
    """验证本地向量客户端按配置批量编码并返回普通浮点列表。"""
    encoder = RecordingSentenceEncoder()
    client = create_embedding_client(
        "sentence-transformers",
        model="BAAI/bge-large-en-v1.5",
        device="cpu",
        batch_size=8,
        normalize_embeddings=True,
        encoder=encoder,
    )

    embeddings = client.embed(["apple", "banana"])

    assert isinstance(client, SentenceTransformerEmbeddingClient)
    assert embeddings == [[0.0, 1.0], [1.0, 1.0]]
    assert encoder.calls == [
        (
            ["apple", "banana"],
            {
                "batch_size": 8,
                "normalize_embeddings": True,
                "convert_to_numpy": True,
                "show_progress_bar": False,
            },
        )
    ]


def test_manage_stage_exposes_only_manage_skill_body() -> None:
    """验证第一步仅向模型发送 Manage 正文并返回结构化指导。"""
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "agentic_selection_guidance": "Prefer a single retrieval route.",
                    "reason": "The question is direct.",
                }
            )
        ]
    )

    result = run_manage_stage(
        {"query": "Where do apple trees grow?"},
        model=model,
        skill_root=SAMPLE_ROOT,
    )

    assert result.manage_skill == "manage-rag-default"
    assert result.guidance == "Prefer a single retrieval route."
    assert len(model.calls) == 1
    prompt = model.calls[0][0]
    assert "# Default RAG Manager" in prompt
    assert "Arrange a sequential RAG workflow" not in prompt
    assert "# Vanilla RAG Workflow" not in prompt


def test_agentic_stage_advertises_then_loads_only_selected_skill() -> None:
    """验证第二步选择前仅披露广告，选择后才返回所选 Agentic 正文。"""
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "selected_agentic_skill": "agentic-vanilla-rag",
                    "reason": "A sequential route is sufficient.",
                }
            )
        ]
    )
    manage_result = ManageStageResult(
        manage_skill="manage-rag-default",
        guidance="Prefer a single retrieval route.",
        reason="The question is direct.",
    )

    result = select_agentic_skill(
        {"query": "Where do apple trees grow?"},
        manage_result=manage_result,
        model=model,
        skill_root=SAMPLE_ROOT,
    )

    assert result.spec.package_name == "agentic-vanilla-rag"
    assert result.advertised_skills == (
        "agentic-rrfusion",
        "agentic-sim-rag",
        "agentic-vanilla-rag",
    )
    assert "# Vanilla RAG Workflow" in result.instructions
    prompt = model.calls[0][0]
    assert "Prefer a single retrieval route." in prompt
    assert "Arrange a sequential RAG workflow" in prompt
    assert "Arrange parallel retrieval" in prompt
    assert "# Vanilla RAG Workflow" not in prompt
    assert "# RRFusion Workflow" not in prompt


def test_agentic_stage_can_select_and_load_only_sim_rag() -> None:
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "selected_agentic_skill": "agentic-sim-rag",
                    "reason": "The answer needs iterative sufficiency checks.",
                }
            )
        ]
    )
    manage_result = ManageStageResult(
        manage_skill="manage-rag-default",
        guidance="Use bounded iterative retrieval.",
        reason="Evidence may be incomplete.",
    )

    result = select_agentic_skill(
        {"query": "A multi-hop question"},
        manage_result=manage_result,
        model=model,
        skill_root=SAMPLE_ROOT,
    )

    assert result.spec.package_name == "agentic-sim-rag"
    assert result.advertised_skills == (
        "agentic-rrfusion",
        "agentic-sim-rag",
        "agentic-vanilla-rag",
    )
    assert "# SIM-RAG-Inspired Iterative RAG" in result.instructions
    assert "# Vanilla RAG Workflow" not in result.instructions
    assert "# RRFusion Workflow" not in result.instructions
    prompt = model.calls[0][0]
    assert "agentic-sim-rag" in prompt
    assert "# SIM-RAG-Inspired Iterative RAG" not in prompt


def test_component_stage_advertises_then_loads_only_selected_skills() -> None:
    """验证第三步只广告兼容组件，并在选择后加载所选组件正文。"""
    specs = discover_specs(SAMPLE_ROOT, validate_runtime=False)
    agentic = next(
        spec for spec in specs if spec.package_name == "agentic-vanilla-rag"
    )
    agentic_result = AgenticStageResult(
        spec=agentic,
        instructions=(agentic.package_path / "SKILL.md").read_text(encoding="utf-8"),
        reason="Sequential retrieval is sufficient.",
        advertised_skills=("agentic-rrfusion", "agentic-vanilla-rag"),
    )
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "component_bindings": {
                        "rewriter": [],
                        "retriever": ["component-bm25-retriever"],
                        "reranker": [],
                        "generator": ["component-grounded-generator"],
                    },
                    "reason": "Use lexical retrieval and grounded generation.",
                }
            )
        ]
    )

    result = select_component_skills(
        {"query": "Where do apple trees grow?"},
        agentic_result=agentic_result,
        model=model,
        skill_root=SAMPLE_ROOT,
    )

    assert result.bindings["retriever"] == ("component-bm25-retriever",)
    assert set(result.instructions) == {
        "component-bm25-retriever",
        "component-grounded-generator",
    }
    assert "# BM25F Retriever Component" in result.instructions[
        "component-bm25-retriever"
    ]
    prompt = model.calls[0][0]
    assert "# Vanilla RAG Workflow" in prompt
    assert (
        "Retrieve and rank title-and-text documents with a field-aware BM25F"
        in prompt
    )
    assert "# BM25F Retriever Component" not in prompt
    assert "# Grounded Generator Component" not in prompt


def test_component_stage_rejects_hyde_with_bm25_retriever() -> None:
    """验证 Component 选择阶段拒绝 HyDE 与 BM25 的错误组合。"""
    specs = discover_specs(SAMPLE_ROOT, validate_runtime=False)
    agentic = next(
        spec for spec in specs if spec.package_name == "agentic-vanilla-rag"
    )
    agentic_result = AgenticStageResult(
        spec=agentic,
        instructions=(agentic.package_path / "SKILL.md").read_text(
            encoding="utf-8"
        ),
        reason="Sequential retrieval is sufficient.",
        advertised_skills=("agentic-rrfusion", "agentic-vanilla-rag"),
    )
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "component_bindings": {
                        "rewriter": ["component-hyde-rewriter"],
                        "retriever": ["component-bm25-retriever"],
                        "reranker": [],
                        "generator": ["component-grounded-generator"],
                    }
                }
            )
        ]
    )

    with pytest.raises(
        SelectionError,
        match=(
            "component-hyde-rewriter.*requires capability "
            "'retriever'.*component-vector-retriever"
        ),
    ):
        select_component_skills(
            {"query": "Where?"},
            agentic_result=agentic_result,
            model=model,
            skill_root=SAMPLE_ROOT,
        )


def test_sim_rag_component_stage_accepts_semantic_retrieval_stack() -> None:
    specs = discover_specs(SAMPLE_ROOT, validate_runtime=False)
    agentic = next(spec for spec in specs if spec.package_name == "agentic-sim-rag")
    agentic_result = AgenticStageResult(
        spec=agentic,
        instructions=(agentic.package_path / "SKILL.md").read_text(encoding="utf-8"),
        reason="Iterative evidence gathering is required.",
        advertised_skills=("agentic-sim-rag",),
    )
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "component_bindings": {
                        "rewriter": ["component-hyde-rewriter"],
                        "retriever": ["component-vector-retriever"],
                        "reranker": ["component-bge-reranker"],
                        "generator": ["component-grounded-generator"],
                        "critic": ["component-critic"],
                    },
                    "reason": "Bridge a vocabulary gap and rerank multi-hop evidence.",
                }
            )
        ]
    )

    result = select_component_skills(
        {"query": "A paraphrased multi-hop question"},
        agentic_result=agentic_result,
        model=model,
        skill_root=SAMPLE_ROOT,
    )

    assert result.bindings == {
        "rewriter": ("component-hyde-rewriter",),
        "retriever": ("component-vector-retriever",),
        "reranker": ("component-bge-reranker",),
        "generator": ("component-grounded-generator",),
        "critic": ("component-critic",),
    }
    prompt = model.calls[0][0]
    assert "Prefer BM25" in prompt
    assert "Use HyDE only with Vector retrieval" in prompt
    assert "Use BGE Reranker" in prompt


def test_select_rag_plan_calls_model_with_strict_progressive_disclosure() -> None:
    """验证三级选择真实调用模型且每一阶段只披露允许的信息。"""
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "agentic_selection_guidance": "Use one sequential retrieval route.",
                    "reason": "Simple lexical question.",
                }
            ),
            json.dumps(
                {
                    "selected_agentic_skill": "agentic-vanilla-rag",
                    "reason": "One route is sufficient.",
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
                    "reason": "Lexical retrieval followed by grounded generation.",
                }
            ),
        ]
    )

    plan = select_rag_plan(
        {"query": "Where do apple trees grow?", "documents": []},
        model=model,
        skill_root=SAMPLE_ROOT,
    )

    assert plan.agentic_skill == "agentic-vanilla-rag"
    assert plan.component_bindings == {
        "rewriter": (),
        "retriever": ("component-bm25-retriever",),
        "reranker": (),
        "generator": ("component-grounded-generator",),
    }
    assert len(model.calls) == 3
    manage_prompt, agentic_prompt, component_prompt = [call[0] for call in model.calls]
    assert "# Default RAG Manager" in manage_prompt
    assert "# Vanilla RAG Workflow" not in manage_prompt
    assert "Arrange a sequential RAG workflow" in agentic_prompt
    assert "# Vanilla RAG Workflow" not in agentic_prompt
    assert "# Vanilla RAG Workflow" in component_prompt
    assert (
        "Retrieve and rank title-and-text documents with a field-aware BM25F"
        in component_prompt
    )
    assert "# BM25F Retriever Component" not in component_prompt
    assert all(call[2:] == (0.0, 512) for call in model.calls)


def test_select_rag_plan_rejects_incompatible_component_choice() -> None:
    """验证模型不能把未知或接口不兼容的组件绑定到 Agentic 槽位。"""
    model = ScriptedModel(
        [
            '{"agentic_selection_guidance":"Use sequential RAG."}',
            '{"selected_agentic_skill":"agentic-vanilla-rag"}',
            json.dumps(
                {
                    "component_bindings": {
                        "rewriter": [],
                        "retriever": ["component-grounded-generator"],
                        "reranker": [],
                        "generator": ["component-grounded-generator"],
                    }
                }
            ),
        ]
    )

    with pytest.raises(SelectionError, match="incompatible Components"):
        select_rag_plan(
            {"query": "test", "documents": []},
            model=model,
            skill_root=SAMPLE_ROOT,
        )
