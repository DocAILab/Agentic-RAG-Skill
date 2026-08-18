"""TriviaQA 证据文档与答案别名弱标签适配器。"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from ..schema import RetrievalDocument, RetrievalExample
from .common import AdapterError, join_text, records, required_text, sample_id


def adapt_triviaqa(row: Mapping[str, Any]) -> RetrievalExample:
    identity = sample_id(row)
    query = required_text(row, "question", identity)
    documents = _evidence_documents(row, identity)
    aliases = _answer_aliases(row.get("answer"))
    relevant = tuple(
        document.id for document in documents if _contains_alias(document, aliases)
    )
    label_type = "weak_answer_alias" if aliases else None
    return RetrievalExample(
        id=identity,
        query=query,
        documents=documents,
        relevant_document_ids=relevant,
        label_type=label_type,
        metadata={"dataset": "triviaqa", "weak_labels": True},
    )


def _evidence_documents(row, identity):
    documents = []
    sources = (
        ("entity_pages", "wiki_context", "entity"),
        ("search_results", "search_context", "search"),
    )
    for field, text_field, prefix in sources:
        value = row.get(field, [])
        for index, item in enumerate(records(value, ("title", text_field), identity)):
            title = str(item.get("title", "")).strip()
            text = join_text(item.get(text_field, ""))
            if title or text:
                documents.append(RetrievalDocument(f"{prefix}:{index}", title, text))
    if not documents:
        raise AdapterError(identity, "candidate documents are empty")
    return tuple(documents)


def _answer_aliases(answer):
    if not isinstance(answer, Mapping):
        return ()
    values = [*answer.get("aliases", ()), answer.get("value", "")]
    normalized = (_normalize_match(value) for value in values)
    return tuple(dict.fromkeys(value for value in normalized if value and value != "unk"))


def _contains_alias(document, aliases):
    evidence = f" {_normalize_match(document.title + ' ' + document.text)} "
    return any(f" {alias} " in evidence for alias in aliases)


def _normalize_match(value):
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))
