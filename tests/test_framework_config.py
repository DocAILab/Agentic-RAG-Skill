from __future__ import annotations

import json
from pathlib import Path

import pytest

import framework.config as config_module
from framework import (
    APIServiceConfig,
    ConfigError,
    OpenAICompatibleModelClient,
    SentenceTransformerEmbeddingClient,
    create_clients_from_config,
    load_framework_config,
    run_rag_from_config,
)

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "framework" / "settings.example.yaml"

DOCUMENTS = (
    {"id": "apple", "text": "Apple trees grow fruit in orchards."},
    {"id": "banana", "text": "Bananas are long yellow fruit."},
)


class ConfiguredScriptedModel:
    """为配置入口依次提供选择 JSON 和生成答案。"""

    def __init__(self, responses):
        """保存预设响应并初始化调用记录。"""
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
        """记录模型调用并返回下一条预设响应。"""
        self.calls.append((prompt, system, temperature, max_tokens))
        return self.responses.pop(0)


def test_example_config_loads_relative_skills_and_clients(monkeypatch) -> None:
    """验证仓库配置模板可解析路径并从环境变量构造两个客户端。"""
    monkeypatch.setenv("VVEAI_API_KEY", "test-secret")
    config = load_framework_config(CONFIG_PATH)

    assert config.skill_root == PROJECT_ROOT / "framework" / "skills"
    assert config.manage_skill == "manage-rag-default"
    assert config.request_defaults == {
        "top_k": 3,
        "max_tokens": 8192,
        "rank_constant": 60,
    }
    assert config.vector_index is not None
    assert config.vector_index.cache_dir == (
        PROJECT_ROOT / "experiments" / "hotpotqa" / "cache" / "vector-indexes"
    )
    assert config.demo is not None
    assert config.demo.corpus_path == (
        PROJECT_ROOT / "experiments" / "hotpotqa" / "data" / "demo" / "corpus.jsonl"
    )
    assert config.demo.test_path == (
        PROJECT_ROOT / "experiments" / "hotpotqa" / "data" / "demo" / "test.jsonl"
    )
    assert config.demo.result_path == (
        PROJECT_ROOT / "experiments" / "hotpotqa" / "outputs" / "demo_results.json"
    )
    assert config.demo.log_path == (
        PROJECT_ROOT / "experiments" / "hotpotqa" / "outputs" / "demo.log.jsonl"
    )
    assert config.demo.max_examples == 100
    assert config.demo.candidate_documents_only is False
    assert config.demo.select_skills_per_example is True
    assert config.demo.batch_selection_query_sample_size == 20
    assert config.demo.request["top_k"] == 10
    assert "constraints" not in config.demo.request
    assert config.executor.model == "deepseek-v4-flash"
    assert config.executor.base_url == "https://api.vveai.com/v1"
    assert config.embedding is not None
    assert config.embedding.model == "BAAI/bge-large-en-v1.5"
    model, embedding = create_clients_from_config(config)
    assert isinstance(model, OpenAICompatibleModelClient)
    assert isinstance(embedding, SentenceTransformerEmbeddingClient)
    assert model.api_key == "test-secret"
    assert model.max_retries == 2
    assert model.retry_backoff_seconds == 2.0


def test_api_service_config_reads_named_environment_variable(monkeypatch) -> None:
    """验证 API key 可以从配置指定的环境变量读取。"""
    monkeypatch.setenv("RAGSKILL_TEST_API_KEY", "test-secret")
    service = APIServiceConfig(
        provider="openai-compatible",
        model="test-model",
        base_url="http://localhost:8000/v1",
        api_key_env="RAGSKILL_TEST_API_KEY",
    )

    assert service.resolve_api_key() == "test-secret"

    monkeypatch.delenv("RAGSKILL_TEST_API_KEY")
    with pytest.raises(ConfigError, match="RAGSKILL_TEST_API_KEY"):
        service.resolve_api_key()


def test_api_service_config_accepts_direct_key_without_repr_leak() -> None:
    """验证本地配置可直写密钥，且对象 repr 不会泄露密钥内容。"""
    service = APIServiceConfig(
        provider="openai-compatible",
        model="test-model",
        base_url="https://example.test/v1",
        api_key="direct-test-secret",
    )

    assert service.resolve_api_key() == "direct-test-secret"
    assert "direct-test-secret" not in repr(service)


def test_api_service_config_rejects_two_key_sources() -> None:
    """验证直写密钥与环境变量密钥不能同时启用。"""
    service = APIServiceConfig(
        provider="openai-compatible",
        model="test-model",
        base_url="https://example.test/v1",
        api_key="direct-test-secret",
        api_key_env="TEST_API_KEY",
    )

    with pytest.raises(ConfigError, match="cannot be used together"):
        service.resolve_api_key()


def test_run_rag_from_config_executes_complete_pipeline(monkeypatch) -> None:
    """验证配置入口能够构造依赖并完成选择、编译、检索和生成。"""
    model = ConfiguredScriptedModel(
        [
            json.dumps(
                {
                    "agentic_selection_guidance": "Use one retrieval route.",
                    "reason": "Simple question.",
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

    def fake_clients(config):
        """替换真实网络客户端，同时确认配置已成功加载。"""
        assert config.config_path == CONFIG_PATH
        return model, None

    monkeypatch.setattr(config_module, "create_clients_from_config", fake_clients)
    result = run_rag_from_config(
        {
            "query": "Where do apple trees grow?",
            "documents": DOCUMENTS,
            "top_k": 1,
        },
        config_path=CONFIG_PATH,
    )

    assert result["answer"] == "Apples grow in orchards."
    assert result["documents"][0]["id"] == "apple"
    assert len(model.calls) == 4
    assert model.calls[-1][3] == 8192
