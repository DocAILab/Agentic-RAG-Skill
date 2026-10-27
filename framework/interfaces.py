"""Agentic Skill 与 Components Skill 共享的运行时协议。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypedDict


# Offline data and evaluation fields must not override an execution request.
# Keep this contract shared by configuration validation and runtime adaptation.
RESERVED_REQUEST_FIELDS = frozenset({
    "documents", "query", "question", "answers", "gold_answers",
    "relevant_document_ids", "relevant_ids", "role", "origin", "source",
    "metadata", "supporting_facts", "label_type", "source_question_ids",
    "reference_count", "distractor_count",
})


class RAGRequest(TypedDict):
    """Required execution fields; workflow-specific options may be added."""

    query: str
    documents: Sequence[Mapping[str, Any]]


def build_rag_request(
    question: str,
    documents: Sequence[Mapping[str, Any]],
    *,
    options: Mapping[str, Any] | None = None,
) -> RAGRequest:
    """Map a QA question and shared runtime corpus to the Skill request contract.

    Pass the annotation-free documents returned by ``load_group`` or
    ``RuntimeComponentContext.bind_corpus``. The sequence is reused unchanged,
    preserving the Group cache identity. This adapter does not deserialize raw
    corpus records; offline fields in request options are discarded.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("RAG question must be a non-empty string")
    if isinstance(documents, (str, bytes, bytearray)) or not isinstance(documents, Sequence):
        raise ValueError("RAG documents must be a runtime document sequence")
    if options is not None and not isinstance(options, Mapping):
        raise ValueError("RAG request options must be a mapping")
    return {
        **{key: value for key, value in (options or {}).items() if key not in RESERVED_REQUEST_FIELDS},
        "query": question,
        "documents": documents,
    }


class ComponentInvoker(Protocol):
    """向 Agentic workflow 暴露已绑定的组件调用能力。"""

    def has(self, slot: str) -> bool:
        """判断指定槽位是否绑定了至少一个 Component Skill。"""
        ...

    def call(
        self,
        slot: str,
        inputs: Mapping[str, Any],
        *,
        index: int = 0,
    ) -> Mapping[str, Any]:
        """调用槽位中指定序号的组件，并返回结构化执行结果。"""
        ...

    def call_all(
        self,
        slot: str,
        inputs: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        """以相同输入调用槽位中绑定的全部组件。"""
        ...


class ComponentContext(Protocol):
    """向具体 Components Skill 注入外部模型服务。"""

    def call_model(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """调用冻结的 Executor Model，并返回生成文本。"""
        ...

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """将一组文本编码成顺序一致的向量。"""
        ...
