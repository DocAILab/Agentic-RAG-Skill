"""只包含抽象组件调用的顺序式 RAG workflow。"""


def run(request, components):
    """依次执行可选改写、检索、可选重排和生成。"""
    original_query = str(request["query"])
    retrieval_query = original_query
    top_k = int(request.get("top_k", 3))
    trace = []

    if components.has("rewriter"):
        rewritten = components.call(
            "rewriter",
            {
                "query": original_query,
                "temperature": request.get(
                    "rewrite_temperature",
                    0.0,
                ),
                "max_tokens": request.get(
                    "rewrite_max_tokens",
                    256,
                ),
            },
        )
        rewritten_query = rewritten.get("rewritten_query")
        if (
            not isinstance(rewritten_query, str)
            or not rewritten_query.strip()
        ):
            raise ValueError(
                "Rewriter must return a non-empty rewritten_query"
            )
        retrieval_query = rewritten_query.strip()
        trace.append({"step": "rewrite"})

    retrieval = components.call(
        "retriever",
        {
            "query": retrieval_query,
            "documents": request.get("documents", ()),
            "top_k": top_k,
        },
    )
    documents = list(retrieval.get("documents", ()))
    trace.append(
        {
            "step": "retrieve",
            "document_count": len(documents),
        }
    )

    if components.has("reranker"):
        reranked = components.call(
            "reranker",
            {
                "query": original_query,
                "documents": documents,
                "top_k": top_k,
            },
        )
        documents = list(reranked.get("documents", ()))
        trace.append(
            {
                "step": "rerank",
                "document_count": len(documents),
            }
        )

    generated = components.call(
        "generator",
        {
            "query": original_query,
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