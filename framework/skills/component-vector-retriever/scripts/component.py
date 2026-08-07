"""基于注入式向量服务的具体 Vector Retriever Component。"""

import math


def run(inputs, context):
    """编码查询与文档，按余弦相似度返回前 top_k 个结果。"""
    query = str(inputs["query"])
    documents = [dict(document) for document in inputs.get("documents", ())]
    top_k = int(inputs.get("top_k", 3))
    if top_k <= 0 or not documents:
        return {"documents": []}

    texts = [query, *(str(document.get("text", "")) for document in documents)]
    embeddings = [tuple(map(float, vector)) for vector in context.embed(texts)]
    if len(embeddings) != len(texts):
        raise ValueError("Embedding service returned an unexpected vector count")
    query_embedding = embeddings[0]
    scored = []
    for document, embedding in zip(documents, embeddings[1:], strict=True):
        scored.append(dict(document, score=_cosine(query_embedding, embedding)))
    scored.sort(key=lambda document: (-document["score"], str(document["id"])))
    return {"documents": scored[:top_k]}


def _cosine(left, right):
    """计算两个等长向量的余弦相似度，并处理零向量。"""
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
