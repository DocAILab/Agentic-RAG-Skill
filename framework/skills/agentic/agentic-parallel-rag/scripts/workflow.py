"""可选改写、并行检索分支、逐分支重排与排名融合的 Agentic workflow。"""


def run(request, components):
    """执行可选查询改写、并行检索、逐分支可选重排、RRF 融合与生成。"""
    query = str(request["query"])
    top_k = int(request.get("top_k", 3))
    rank_constant = int(request.get("rank_constant", 60))
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")

    branch_queries, trace = _rewrite_queries(request, components, query)
    branch_top_k = max(top_k, top_k * 2)
    retrieval_input = {
        "documents": request.get("documents", ()),
        "top_k": branch_top_k,
    }

    branch_results = []
    for branch_query in branch_queries:
        for retrieved in components.call_all(
            "retrievers",
            {**retrieval_input, "query": branch_query},
        ):
            documents = list(retrieved.get("documents", ()))
            if components.has("reranker"):
                documents = _rerank(components, query, documents, branch_top_k)
            branch_results.append(documents)
    trace.append(
        {
            "step": "parallel_retrieve_and_rerank",
            "branch_count": len(branch_results),
        }
    )

    documents = _reciprocal_rank_fusion(branch_results, rank_constant, top_k)
    trace.append({"step": "fuse", "document_count": len(documents)})

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


def _rewrite_queries(request, components, query):
    """可选调用改写组件，返回一个或多个分支查询与轨迹。"""
    if not components.has("rewriter"):
        return [query], []
    rewritten = components.call(
        "rewriter",
        {
            "query": query,
            "temperature": request.get("rewrite_temperature", 0.0),
            "max_tokens": request.get("rewrite_max_tokens", 256),
        },
    )
    queries = []
    rewritten_query = rewritten.get("rewritten_query")
    if isinstance(rewritten_query, str) and rewritten_query.strip():
        queries.append(rewritten_query.strip())
    for item in rewritten.get("queries", ()):
        text = str(item).strip()
        if text and text not in queries:
            queries.append(text)
    if not queries:
        raise ValueError(
            "rewriter returned no non-empty rewritten_query or queries"
        )
    return queries, [{"step": "rewrite", "query_count": len(queries)}]


def _rerank(components, query, documents, top_k):
    """对单个检索分支执行可选重排，使用原始问题作为重排查询。"""
    reranked = components.call(
        "reranker",
        {"query": query, "documents": documents, "top_k": top_k},
    )
    return list(reranked.get("documents", ()))


def _reciprocal_rank_fusion(branch_results, rank_constant, top_k):
    """使用 Reciprocal Rank Fusion 合并多路检索结果并截取前 top_k 项。"""
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    scores = {}
    documents = {}
    for ranked_documents in branch_results:
        for rank, document in enumerate(ranked_documents, start=1):
            document_id = str(document["id"])
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (
                rank_constant + rank
            )
            documents.setdefault(document_id, dict(document))
    ranked_ids = sorted(scores, key=lambda item: (-scores[item], item))[:top_k]
    return [dict(documents[item], score=scores[item]) for item in ranked_ids]
