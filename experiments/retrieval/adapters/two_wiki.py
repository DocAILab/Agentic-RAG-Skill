"""2WikiMultihopQA 候选文档适配器。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..schema import RetrievalExample
from .common import (
    context_documents,
    relevant_ids,
    required_text,
    sample_id,
    supporting_titles,
)


def adapt_two_wiki(row: Mapping[str, Any]) -> RetrievalExample:
    identity = sample_id(row)
    query = required_text(row, "question", identity)
    documents, title_ids = context_documents(row.get("context"), "content", identity)
    titles = supporting_titles(row.get("supporting_facts"), identity)
    relevant = relevant_ids(titles, title_ids)
    return RetrievalExample(
        id=identity,
        query=query,
        documents=documents,
        relevant_document_ids=relevant,
        label_type="supporting_facts" if relevant else None,
        metadata={"dataset": "2wikimultihopqa"},
    )
