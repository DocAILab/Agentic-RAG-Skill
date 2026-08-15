"""基于本地 BGE 交叉编码器（Cross-Encoder）的具体 Reranker Component。"""

_RERANKER_CACHE = {}

def run(inputs, context):
    """将查询与候选文档组成 (query, passage) 对，用 BGE 交叉编码器打分并重排。"""
    del context  # 本组件加载本地交叉编码器，不依赖运行时注入的 Executor/Embedding 服务
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
    """延迟加载 CrossEncoder，返回相关性分数。"""
    reranker = _get_reranker(model, max_length=max_length, device=device)
    return reranker.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )

def _get_reranker(model, *, max_length=None, device=None):
    """按模型、长度限制和设备缓存 CrossEncoder 实例，避免重复下载权重。"""
    key = (model, max_length, device)
    reranker = _RERANKER_CACHE.get(key)
    if reranker is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "BGE reranker requires the 'sentence-transformers' package. "
                'Install it with: pip install -e ".[rerank]"'
            ) from exc
        reranker = CrossEncoder(model, max_length=max_length, device=device)
        _RERANKER_CACHE[key] = reranker
    return reranker
