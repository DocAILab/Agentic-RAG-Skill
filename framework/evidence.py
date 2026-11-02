"""Deterministic, auditable evidence excerpts for generation and critique."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import heapq
import re
from typing import Any


# Byte budgets include citation/excerpt formatting. They do not claim to count
# provider-specific tokens, or to include the question, answer, or instructions.
DEFAULT_EVIDENCE_MAX_BYTES = 96 * 1024
DEFAULT_DOCUMENT_MAX_BYTES = 4 * 1024
_WINDOW_CHARACTERS = 1000
_MAX_WINDOWS = 3
_MAX_CANDIDATES = 64
_NO_EVIDENCE = "[No evidence supplied]"
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_STOP_WORDS = frozenset(
    "a an the and or of to in on at is was are were be been what which who where "
    "when how why did does do has have had with for from by that this it its as".split()
)


@dataclass(frozen=True, slots=True)
class EvidencePack:
    """Prompt evidence and text-free source offsets, counts and budget metadata."""

    text: str
    metadata: dict[str, Any]


def prepare_evidence(
    query: str,
    documents: Sequence[Mapping[str, Any]] | None,
    *,
    max_total_bytes: int = DEFAULT_EVIDENCE_MAX_BYTES,
    max_document_bytes: int = DEFAULT_DOCUMENT_MAX_BYTES,
) -> EvidencePack:
    """Keep short evidence intact; select query-matching windows from long text.

    Documents retain retrieval/reranking order. Long-document windows are ranked
    by distinct query-word coverage, capped frequency, then source position. All
    character offsets refer to the unchanged input text and are end-exclusive.
    This function neither mutates retrieval results nor uses relevance labels.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("evidence query must be a non-empty string")
    for name, value in (("max_total_bytes", max_total_bytes), ("max_document_bytes", max_document_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if max_total_bytes < 128:
        raise ValueError("max_total_bytes must be at least 128 for evidence markers")
    records = _documents(documents)
    query_terms = set(
        word for word in dict.fromkeys(match.group().casefold() for match in _WORDS.finditer(query))
        if word not in _STOP_WORDS
    )
    # Bound query-processing work even for a caller-provided oversized question.
    query_terms = set(sorted(query_terms)[:64])
    sizes = [len(record["text"].encode("utf-8")) for record in records]
    full_sizes = [len(_header(index, record["id"]).encode("utf-8")) + size
                  for index, (record, size) in enumerate(zip(records, sizes, strict=True), start=1)]
    all_fit = all(size <= max_document_bytes for size in full_sizes) and (
        sum(full_sizes) + 2 * max(0, len(records) - 1) <= max_total_bytes
    )
    footer_reserve = 0 if all_fit else len(_footer(len(records)).encode("utf-8")) + 2
    remaining = max_total_bytes - footer_reserve
    blocks: list[str] = []
    details: list[dict[str, Any]] = []
    for index, (record, source_bytes) in enumerate(zip(records, sizes, strict=True), start=1):
        separator = 2 if blocks else 0
        allowance = min(max_document_bytes, max(0, remaining - separator))
        block, spans = _document_block(query_terms, record, index, allowance)
        sent_bytes = len(block.encode("utf-8")) if block else 0
        selected_characters = sum(end - start for start, end in spans)
        selected_bytes = sum(len(record["text"][start:end].encode("utf-8")) for start, end in spans)
        details.append({
            "document_id": record["id"], "input_index": index,
            "source_characters": len(record["text"]), "source_utf8_bytes": source_bytes,
            "included": bool(block), "formatted_bytes": sent_bytes,
            "selected_characters": selected_characters, "selected_utf8_bytes": selected_bytes,
            "omitted_characters": len(record["text"]) - selected_characters,
            "omitted_utf8_bytes": source_bytes - selected_bytes,
            "spans": [{"start": start, "end": end} for start, end in spans],
        })
        if block:
            blocks.append(block)
            remaining -= separator + sent_bytes
    omitted = sum(not item["included"] for item in details)
    if not records:
        text = _NO_EVIDENCE
    else:
        if not all_fit:
            blocks.append(_footer(omitted))
        text = "\n\n".join(blocks)
    evidence_bytes = len(text.encode("utf-8"))
    assert evidence_bytes <= max_total_bytes
    return EvidencePack(text, {
        "policy": "query_windows_v1", "budget_unit": "utf8_bytes",
        "max_total_bytes": max_total_bytes, "max_document_bytes": max_document_bytes,
        "evidence_bytes": evidence_bytes, "input_document_count": len(records),
        "included_document_count": len(records) - omitted, "omitted_document_count": omitted,
        "documents": details,
    })


def _documents(documents):
    if documents is None:
        return []
    if isinstance(documents, (str, bytes, bytearray)) or not isinstance(documents, Sequence):
        raise ValueError("documents must be a sequence of mappings")
    records = []
    for index, document in enumerate(documents):
        if not isinstance(document, Mapping):
            raise ValueError(f"documents[{index}] must be a mapping")
        record = {}
        for key in ("id", "text"):
            if key not in document:
                raise ValueError(f"documents[{index}].{key} is required")
            value = str(document[key])
            if not value.strip():
                raise ValueError(f"documents[{index}].{key} must be non-empty")
            record[key] = value.strip() if key == "id" else value
        records.append(record)
    return records


def _header(index, identifier):
    return f"[{index}] id={identifier}\n"


def _footer(omitted):
    return f"[Evidence budget applied: only shown text is supplied; omitted documents: {omitted}.]"


def _format_excerpts(record, index, spans):
    omitted = len(record["text"]) - sum(end - start for start, end in spans)
    return (
        _header(index, record["id"])
        + f"[Selected excerpts; source characters: {len(record['text'])}; omitted characters: {omitted}.]\n"
        + "\n".join(f"[chars {start}:{end}]\n{record['text'][start:end]}" for start, end in spans)
    )


def _document_block(query_terms, record, index, allowance):
    text = record["text"]
    header = _header(index, record["id"])
    if len(header.encode("utf-8")) + len(text.encode("utf-8")) <= allowance:
        return header + text, [(0, len(text))]
    if allowance <= len(header.encode("utf-8")) + 100:
        return "", []
    spans = []
    for start, end, anchor in _rank_windows(text, query_terms):
        if any(start < previous_end and end > previous_start for previous_start, previous_end in spans):
            continue
        candidate = sorted([*spans, (start, end)])
        if len(_format_excerpts(record, index, candidate).encode("utf-8")) > allowance:
            # Keep the matching word near the middle when a window must shrink.
            low, high = 1, end - start
            fitted = None
            while low <= high:
                length = (low + high) // 2
                left = max(start, min(anchor - length // 2, end - length))
                trial = sorted([*spans, (left, left + length)])
                if len(_format_excerpts(record, index, trial).encode("utf-8")) <= allowance:
                    fitted = trial
                    low = length + 1
                else:
                    high = length - 1
            if fitted is None:
                break
            spans = fitted
            break
        spans = candidate
        if len(spans) >= _MAX_WINDOWS:
            break
    return (_format_excerpts(record, index, spans), spans) if spans else ("", [])


def _rank_windows(text, query_terms):
    if not query_terms:
        return [(0, min(_WINDOW_CHARACTERS, len(text)), 0)]
    candidates = []
    for start in range(0, len(text), _WINDOW_CHARACTERS // 2):
        end = min(start + _WINDOW_CHARACTERS, len(text))
        counts = {}
        anchor = start
        for match in _WORDS.finditer(text, start, end):
            word = match.group().casefold()
            if word in query_terms:
                if not counts:
                    anchor = match.start()
                counts[word] = counts.get(word, 0) + 1
        item = (len(counts), sum(min(count, 3) for count in counts.values()), -start, start, end, anchor)
        if len(candidates) < _MAX_CANDIDATES:
            heapq.heappush(candidates, item)
        else:
            heapq.heappushpop(candidates, item)
        if end == len(text):
            break
    return [(start, end, anchor) for _, _, _, start, end, anchor in sorted(candidates, reverse=True)]
