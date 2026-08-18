"""Frozen retrieval baselines used only by comparison experiments."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter


def run_original_bm25(inputs, context):
    """Reproduce the text-only BM25 implementation from baseline 5b9197a."""
    del context
    query = str(inputs["query"])
    documents = [dict(document) for document in inputs.get("documents", ())]
    top_k = int(inputs.get("top_k", 3))
    k1 = float(inputs.get("k1", 1.5))
    b = float(inputs.get("b", 0.75))
    if top_k <= 0 or not documents:
        return {"documents": []}
    if k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
    tokens = [_original_tokenize(document.get("text", "")) for document in documents]
    average_length = sum(map(len, tokens)) / max(1, len(tokens))
    document_frequency = Counter(term for item in tokens for term in set(item))
    scored = [
        dict(
            document,
            score=_original_score(
                _original_tokenize(query),
                item,
                average_length,
                document_frequency,
                len(documents),
                k1,
                b,
            ),
        )
        for document, item in zip(documents, tokens, strict=True)
    ]
    scored.sort(key=lambda document: (-document["score"], str(document["id"])))
    return {"documents": scored[:top_k]}


def _original_score(query, tokens, average_length, document_frequency, count, k1, b):
    frequencies = Counter(tokens)
    score = 0.0
    for term in query:
        frequency = frequencies[term]
        if frequency == 0:
            continue
        found_in = document_frequency[term]
        inverse_frequency = math.log(
            1.0 + (count - found_in + 0.5) / (found_in + 0.5)
        )
        normalization = frequency + k1 * (
            1.0 - b + b * len(tokens) / max(average_length, 1.0)
        )
        score += inverse_frequency * frequency * (k1 + 1.0) / normalization
    return score


def _original_tokenize(text):
    return [token.lower() for token in re.findall(r"\w+", str(text))]


def run_title_weighted_bm25(inputs, context):
    """Reproduce the title-weighted combined-length implementation at b374023."""
    del context
    query = str(inputs.get("query", ""))
    documents = [dict(document) for document in inputs.get("documents", ())]
    top_k = int(inputs.get("top_k", 3))
    k1 = float(inputs.get("k1", 1.5))
    b = float(inputs.get("b", 0.75))
    title_boost = float(inputs.get("title_boost", 1.5))
    if top_k <= 0 or not documents:
        return {"documents": []}
    _validate_title_weighted(k1, b, title_boost)
    query_terms = tuple(dict.fromkeys(_unicode_tokenize(query)))
    if not query_terms:
        return {"documents": []}
    fields = [_field_tokens(document) for document in documents]
    lengths = [len(body) + title_boost * len(title) for title, body in fields]
    average_length = sum(lengths) / len(lengths)
    document_frequency = Counter(
        term for title, body in fields for term in set(title) | set(body)
    )
    scored = []
    for document, field, length in zip(documents, fields, lengths, strict=True):
        score = _title_weighted_score(
            query_terms,
            field,
            length,
            average_length,
            document_frequency,
            len(documents),
            k1,
            b,
            title_boost,
        )
        scored.append(dict(document, score=score))
    scored.sort(key=lambda document: (-document["score"], str(document.get("id", ""))))
    return {"documents": scored[:top_k]}


def _title_weighted_score(query, field, length, average, frequencies, count, k1, b, boost):
    title, body = field
    title_frequency = Counter(title)
    body_frequency = Counter(body)
    score = 0.0
    for term in query:
        frequency = body_frequency[term] + boost * title_frequency[term]
        if frequency == 0:
            continue
        found_in = frequencies[term]
        inverse_frequency = math.log(
            1.0 + (count - found_in + 0.5) / (found_in + 0.5)
        )
        length_ratio = length / max(average, 1.0)
        denominator = frequency + k1 * (1.0 - b + b * length_ratio)
        score += inverse_frequency * frequency * (k1 + 1.0) / denominator
    return score


def _unicode_tokenize(text):
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def _field_tokens(document):
    return _unicode_tokenize(document.get("title", "")), _unicode_tokenize(
        document.get("text", "")
    )


def _validate_title_weighted(k1, b, title_boost):
    if not math.isfinite(k1) or k1 <= 0:
        raise ValueError("BM25 requires a finite k1 > 0")
    if not math.isfinite(b) or not 0 <= b <= 1:
        raise ValueError("BM25 requires a finite 0 <= b <= 1")
    if not math.isfinite(title_boost) or title_boost < 0:
        raise ValueError("BM25 requires a finite title_boost >= 0")
