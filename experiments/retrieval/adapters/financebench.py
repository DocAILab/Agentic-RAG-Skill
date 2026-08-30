"""FinanceBench candidate-document adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..schema import RetrievalDocument, RetrievalExample
from .common import AdapterError, join_text, records, required_text, sample_id


def adapt_financebench(row: Mapping[str, Any]) -> RetrievalExample:
    identity = sample_id(row)
    query = required_text(row, "question", identity)
    documents = _evidence_documents(row, identity)
    relevant = tuple(dict.fromkeys(document.id for document in documents))
    metadata: dict[str, Any] = {"dataset": "financebench"}
    for key in (
        "company",
        "doc_name",
        "question_type",
        "question_reasoning",
        "dataset_subset_label",
        "doc_type",
        "doc_period",
        "gics_sector",
        "doc_link",
    ):
        value = row.get(key)
        if value is not None and str(value).strip():
            metadata[key] = value
    return RetrievalExample(
        id=identity,
        query=query,
        documents=documents,
        relevant_document_ids=relevant,
        label_type="evidence" if relevant else None,
        metadata=metadata,
    )


def _evidence_documents(
    row: Mapping[str, Any],
    identity: str,
) -> tuple[RetrievalDocument, ...]:
    sources = row.get("evidence")
    if sources in (None, [], {}, ""):
        return (_row_document(row, identity, 0),)
    if isinstance(sources, str):
        try:
            sources = json.loads(sources)
        except json.JSONDecodeError:
            sources = [{"evidence_text": sources}]
    elif isinstance(sources, list):
        sources = [
            {"evidence_text": item} if isinstance(item, str) else item
            for item in sources
        ]
    documents = []
    for index, item in enumerate(
        records(
            sources,
            (
                "evidence_doc_name",
                "doc_name",
                "evidence_page_num",
                "evidence_text",
                "evidence_text_full_page",
                "doc_link",
            ),
            identity,
        )
    ):
        documents.append(_document_from_item(row, item, index, identity))
    if not documents:
        raise AdapterError(identity, "evidence documents are empty")
    return tuple(_deduplicate_documents(documents))


def _document_from_item(
    row: Mapping[str, Any],
    item: Mapping[str, Any],
    index: int,
    identity: str,
) -> RetrievalDocument:
    title = _title(item, row, identity)
    page = item.get("evidence_page_num")
    document_id = _document_id(title, page, index)
    text = _document_text(item, row)
    if not text:
        raise AdapterError(identity, f"evidence item {index} has no text")
    return RetrievalDocument(document_id, title, text)


def _row_document(
    row: Mapping[str, Any],
    identity: str,
    index: int,
) -> RetrievalDocument:
    title = _title(row, row, identity)
    text = _document_text(row, row)
    if not text:
        raise AdapterError(identity, "missing evidence text")
    return RetrievalDocument(_document_id(title, None, index), title, text)


def _title(item: Mapping[str, Any], row: Mapping[str, Any], identity: str) -> str:
    for key in ("evidence_doc_name", "doc_name"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    value = row.get("doc_name")
    if value is not None and str(value).strip():
        return str(value).strip()
    raise AdapterError(identity, "missing evidence document name")


def _document_text(item: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    return join_text(
        item.get("evidence_text_full_page")
        or item.get("evidence_text")
        or row.get("evidence_text_full_page")
        or row.get("evidence_text")
        or row.get("doc_text")
        or row.get("text")
        or row.get("content")
        or ""
    )


def _document_id(title: str, page: Any, index: int) -> str:
    if page is not None and str(page).strip():
        return f"{title}#p{str(page).strip()}"
    if index == 0:
        return title
    return f"{title}#{index + 1}"


def _deduplicate_documents(
    documents: list[RetrievalDocument],
) -> list[RetrievalDocument]:
    seen = {}
    for document in documents:
        seen.setdefault(document.id, document)
    return list(seen.values())
