"""使用抽象组件调用和本地排名融合的并行检索 workflow。"""


def run(request, components):
    """并行调用多路检索器，融合结果后执行可选重排与生成。"""
    query = str(request["query"])
    top_k = int(request.get("top_k", 3))
    rank_constant = int(request.get("rank_constant", 60))
    retrieval_input = {
        "query": query,
        "documents": request.get("documents", ()),
        "top_k": max(top_k, top_k * 2),
    }
    branch_results = components.call_all("retrievers", retrieval_input)
    documents = _reciprocal_rank_fusion(branch_results, rank_constant, top_k)
    trace = [
        {
            "step": "parallel_retrieve_and_fuse",
            "branch_count": len(branch_results),
            "document_count": len(documents),
        }
    ]

    if components.has("reranker"):
        reranked = components.call(
            "reranker",
            {"query": query, "documents": documents, "top_k": top_k},
        )
        documents = list(reranked.get("documents", ()))
        trace.append({"step": "rerank", "document_count": len(documents)})

    generated = components.call(
        "generator",
        {
            "query": query,
            "documents": documents,
            "max_tokens": request.get("max_tokens"),
        },
    )
    trace.append({"step": "generate"})
    return {
        "answer": str(generated["answer"]),
        "documents": documents,
        "trace": trace,
    }


def _reciprocal_rank_fusion(branch_results, rank_constant, top_k):
    """使用 Reciprocal Rank Fusion 合并多路检索结果并截取前 top_k 项。"""
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    scores = {}
    documents = {}
    for result in branch_results:
        for rank, document in enumerate(result.get("documents", ()), start=1):
            document_id = str(document["id"])
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (
                rank_constant + rank
            )
            documents.setdefault(document_id, dict(document))
    ranked_ids = sorted(scores, key=lambda item: (-scores[item], item))[:top_k]
    return [dict(documents[item], score=scores[item]) for item in ranked_ids]
