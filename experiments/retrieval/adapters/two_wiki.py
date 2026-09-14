"""2WikiMultihopQA 候选文档适配器。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..schema import RetrievalExample
from .common import (
    AdapterError,
    answer_values,
    context_documents,
    relevant_ids,
    required_text,
    sample_id,
    supporting_titles,
)

QUESTION_TYPES = frozenset(
    {"compositional", "inference", "comparison", "bridge_comparison"}
)


def normalize_question_type(value: Any, identity: str) -> str | None:
    """Normalize the optional official 2Wiki reasoning type."""
    if value is None or not str(value).strip():
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in QUESTION_TYPES:
        raise AdapterError(identity, f"unsupported 2Wiki question type: {value!r}")
    return normalized


def adapt_two_wiki(row: Mapping[str, Any]) -> RetrievalExample:
    """把一条 2WikiMultihopQA 样本转换为统一评估结构。"""
    identity = sample_id(row)
    query = required_text(row, "question", identity)
    documents, title_ids = context_documents(row.get("context"), "content", identity)
    titles = supporting_titles(row.get("supporting_facts"), identity)
    relevant = relevant_ids(titles, title_ids)
    metadata = {"dataset": "2wikimultihopqa"}
    question_type = normalize_question_type(row.get("type"), identity)
    if question_type is not None:
        metadata["question_type"] = question_type
    return RetrievalExample(
        id=identity,
        query=query,
        documents=documents,
        relevant_document_ids=relevant,
        label_type="supporting_facts" if relevant else None,
        gold_answers=answer_values(row.get("answer")),
        metadata=metadata,
    )
