"""只包含抽象组件调用的顺序式 RAG workflow。"""


def run(request, components):
    """依次执行检索、可选重排和生成，并记录流程轨迹。"""
    query = str(request["query"])
    top_k = int(request.get("top_k", 3))
    retrieval = components.call(
        "retriever",
        {
            "query": query,
            "documents": request.get("documents", ()),
            "top_k": top_k,
        },
    )
    documents = list(retrieval.get("documents", ()))
    trace = [{"step": "retrieve", "document_count": len(documents)}]

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
