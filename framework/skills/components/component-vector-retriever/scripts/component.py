"""基于注入式向量服务的具体 Vector Retriever Component。"""

import math

DEFAULT_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages:"
)


def run(inputs, context):
    """编码查询与文档，按余弦相似度返回前 top_k 个结果。"""
    query = str(inputs.get("query", "")).strip()
    documents = [dict(document) for document in inputs.get("documents", ())]
    top_k = int(inputs.get("top_k", 3))
    if top_k <= 0 or not documents or not query:
        return {"documents": []}

    instruction = str(inputs.get("query_instruction", DEFAULT_QUERY_INSTRUCTION)).strip()
    query_text = f"{instruction} {query}".strip()
    missing = [
        index
        for index, document in enumerate(documents)
        if document.get("embedding") is None
    ]
    texts = [query_text, *(_document_text(documents[index]) for index in missing)]
    encoded = list(context.embed(texts))
    if len(encoded) != len(texts):
        raise ValueError("Embedding service returned an unexpected vector count")
    query_embedding = _normalize_vector(encoded[0], "query embedding")
    generated = dict(zip(missing, encoded[1:], strict=True))
    scored = []
    for index, document in enumerate(documents):
        raw_vector = generated.get(index, document.get("embedding"))
        embedding = _normalize_vector(raw_vector, f"document {document.get('id', index)!r}")
        if len(query_embedding) != len(embedding):
            raise ValueError("Embedding dimensions do not match")
        score = sum(
            left * right for left, right in zip(query_embedding, embedding, strict=True)
        )
        scored.append(dict(document, score=score))
    scored.sort(key=lambda document: (-document["score"], str(document.get("id", ""))))
    return {"documents": scored[:top_k]}


def _document_text(document):
    """把可选标题和正文组合为不带查询指令的 passage。"""
    parts = [str(document.get(field, "")).strip() for field in ("title", "text")]
    return "\n".join(part for part in parts if part)


def _normalize_vector(vector, label):
    """校验并归一化单个向量。"""
    try:
        values = tuple(float(value) for value in vector)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric vector") from exc
    if not values:
        raise ValueError(f"{label} must not be empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain only finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise ValueError(f"{label} is a zero vector")
    return tuple(value / norm for value in values)
