"""基于本地 BGE 交叉编码器（Cross-Encoder）的具体 Reranker Component。"""

from framework.reranking import score_cross_encoder_pairs

def run(inputs, context):
    """将查询与候选文档组成 (query, passage) 对，用 BGE 交叉编码器打分并重排。"""
    del context  # 模型由 framework 的进程级资源持有者共享，不依赖 Group 实例。
    query = str(inputs["query"])
    documents = [dict(document) for document in inputs.get("documents", ())]
    top_k = int(inputs.get("top_k", 3))
    if top_k <= 0 or not documents:
        return {"documents": []}

    model = str(inputs.get("model", "BAAI/bge-reranker-large"))
    batch_size = int(inputs.get("batch_size", 32))
    max_length = inputs.get("max_length")
    device = inputs.get("device")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_length is not None and int(max_length) <= 0:
        raise ValueError("max_length must be positive")

    pairs = [(query, str(document.get("text", ""))) for document in documents]
    scores = _score_pairs(
        model,
        pairs,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    scored = [
        dict(document, score=float(score))
        for document, score in zip(documents, scores, strict=True)
    ]
    scored.sort(key=lambda document: (-document["score"], str(document["id"])))
    return {"documents": scored[:top_k]}


def _score_pairs(model, pairs, *, batch_size, max_length, device):
    """复用 framework 持有的 CrossEncoder，返回相关性分数。"""
    return score_cross_encoder_pairs(
        model, pairs,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
