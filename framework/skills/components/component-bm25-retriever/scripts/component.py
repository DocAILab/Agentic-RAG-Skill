"""不依赖第三方检索库的字段感知 BM25F Component。"""

import math
import re
import unicodedata
from collections import Counter


def run(inputs, context):
    """分别归一化标题与正文字段，计算 BM25F 分数并返回排序结果。"""
    del context
    query = str(inputs.get("query", ""))
    documents = [dict(document) for document in inputs.get("documents", ())]
    top_k = int(inputs.get("top_k", 10))
    k1 = float(inputs.get("k1", 1.2))
    b = float(inputs.get("b", 0.5))
    title_b = float(inputs.get("title_b", 0.75))
    title_boost = float(inputs.get("title_boost", 3.0))
    if top_k <= 0 or not documents:
        return {"documents": []}
    _validate_parameters(k1, b, title_b, title_boost)

    query_terms = tuple(dict.fromkeys(_tokenize(query)))
    if not query_terms:
        return {"documents": []}
    fields = [_tokenize_document(document) for document in documents]
    average_title_length = sum(len(title) for title, _ in fields) / len(fields)
    average_body_length = sum(len(body) for _, body in fields) / len(fields)
    document_frequency = Counter(
        term for title, body in fields for term in set(title) | set(body)
    )
    scored = []
    for document, field in zip(documents, fields, strict=True):
        score = _score_document(
            query_terms,
            field,
            average_title_length=average_title_length,
            average_body_length=average_body_length,
            document_frequency=document_frequency,
            document_count=len(documents),
            k1=k1,
            b=b,
            title_b=title_b,
            title_boost=title_boost,
        )
        scored.append(dict(document, score=score))
    scored.sort(key=lambda document: (-document["score"], str(document.get("id", ""))))
    return {"documents": scored[:top_k]}


def _tokenize(text):
    """执行 NFKC 与大小写折叠后切分 Unicode 字母数字词项。"""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def _tokenize_document(document):
    """分别切分标题与正文，供字段权重计算复用。"""
    return (
        _tokenize(document.get("title", "")),
        _tokenize(document.get("text", "")),
    )


def _score_document(
    query_terms,
    field,
    *,
    average_title_length,
    average_body_length,
    document_frequency,
    document_count,
    k1,
    b,
    title_b,
    title_boost,
):
    title, body = field
    title_frequency = Counter(title)
    body_frequency = Counter(body)
    score = 0.0
    for term in query_terms:
        frequency = _normalized_frequency(
            body_frequency[term], len(body), average_body_length, b
        )
        frequency += title_boost * _normalized_frequency(
            title_frequency[term], len(title), average_title_length, title_b
        )
        if frequency == 0:
            continue
        found_in = document_frequency[term]
        inverse_frequency = math.log(
            1.0 + (document_count - found_in + 0.5) / (found_in + 0.5)
        )
        denominator = frequency + k1
        score += inverse_frequency * frequency * (k1 + 1.0) / denominator
    return score


def _normalized_frequency(frequency, length, average_length, field_b):
    if frequency == 0 or average_length == 0:
        return 0.0
    normalizer = 1.0 - field_b + field_b * length / average_length
    return frequency / normalizer


def _validate_parameters(k1, b, title_b, title_boost):
    if not math.isfinite(k1) or k1 <= 0:
        raise ValueError("BM25 requires a finite k1 > 0")
    if not math.isfinite(b) or not 0 <= b <= 1:
        raise ValueError("BM25 requires a finite 0 <= b <= 1")
    if not math.isfinite(title_b) or not 0 <= title_b <= 1:
        raise ValueError("BM25 requires a finite 0 <= title_b <= 1")
    if not math.isfinite(title_boost) or title_boost < 0:
        raise ValueError("BM25 requires a finite title_boost >= 0")
