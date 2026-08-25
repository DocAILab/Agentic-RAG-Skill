"""根据分类结果选择词法、语义或混合检索路线的 RAG workflow。"""

_ALLOWED_ROUTES = frozenset({"lexical", "semantic", "hybrid"})


def run(request, components):
    """分类请求、执行对应检索路线，并基于真实文档生成答案。"""
    original_query = str(request["query"]).strip()
    if not original_query:
        raise ValueError("Conditional RAG requires a non-empty query")

    top_k = int(request.get("top_k", 3))
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    source_documents = request.get("documents", ())
    trace = []

    classification = components.call(
        "classifier",
        {
            "query": original_query,
            "documents": source_documents,
            "constraints": request.get("routing_constraints", {}),
            "max_tokens": request.get("classifier_max_tokens"),
        },
    )
    route = classification.get("route")
    if not isinstance(route, str) or route not in _ALLOWED_ROUTES:
        raise ValueError("Classifier route must be lexical, semantic, or hybrid")

    trace.append(
        {
            "step": "classify",
            "route": route,
            "reason": classification.get("reason"),
            "confidence": classification.get("confidence"),
        }
    )

    if route == "lexical":
        documents = _retrieve(
            components,
            "lexical_retriever",
            original_query,
            source_documents,
            top_k,
        )
        trace.append(
            {
                "step": "retrieve",
                "route": "lexical",
                "query_source": "original",
                "document_count": len(documents),
            }
        )

    elif route == "semantic":
        semantic_query, rewritten = _semantic_query(
            original_query,
            request,
            components,
        )
        if rewritten:
            trace.append({"step": "rewrite", "strategy": "hyde"})

        documents = _retrieve(
            components,
            "semantic_retriever",
            semantic_query,
            source_documents,
            top_k,
        )
        trace.append(
            {
                "step": "retrieve",
                "route": "semantic",
                "query_source": ("rewritten" if rewritten else "original"),
                "document_count": len(documents),
            }
        )

    else:
        semantic_query, rewritten = _semantic_query(
            original_query,
            request,
            components,
        )
        if rewritten:
            trace.append({"step": "rewrite", "strategy": "hyde"})

        branch_top_k = max(top_k, top_k * 2)
        lexical_result = {
            "documents": _retrieve(
                components,
                "lexical_retriever",
                original_query,
                source_documents,
                branch_top_k,
            )
        }
        semantic_result = {
            "documents": _retrieve(
                components,
                "semantic_retriever",
                semantic_query,
                source_documents,
                branch_top_k,
            )
        }

        documents = _reciprocal_rank_fusion(
            [lexical_result, semantic_result],
            int(request.get("rank_constant", 60)),
            top_k,
        )
        trace.append(
            {
                "step": "retrieve_and_fuse",
                "route": "hybrid",
                "branch_count": 2,
                "semantic_query_source": ("rewritten" if rewritten else "original"),
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
                "query_source": "original",
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
    trace.append(
        {
            "step": "generate",
            "query_source": "original",
        }
    )

    return {
        "answer": str(generated["answer"]),
        "documents": documents,
        "route": route,
        "trace": trace,
    }


def _semantic_query(original_query, request, components):
    """为语义检索返回原问题或可选 Rewriter 生成的检索文本。"""
    if not components.has("rewriter"):
        return original_query, False

    rewritten = components.call(
        "rewriter",
        {
            "query": original_query,
            "temperature": request.get("rewrite_temperature", 0.0),
            "max_tokens": request.get("rewrite_max_tokens", 256),
        },
    )
    rewritten_query = rewritten.get("rewritten_query")
    if not isinstance(rewritten_query, str) or not rewritten_query.strip():
        raise ValueError("Rewriter must return a non-empty rewritten_query")

    return rewritten_query.strip(), True


def _retrieve(components, slot, query, documents, top_k):
    """调用指定 Retriever 槽位并返回文档列表。"""
    result = components.call(
        slot,
        {
            "query": query,
            "documents": documents,
            "top_k": top_k,
        },
    )
    return list(result.get("documents", ()))


def _reciprocal_rank_fusion(branch_results, rank_constant, top_k):
    """使用 RRF 合并词法和语义检索结果。"""
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")

    scores = {}
    documents = {}

    for result in branch_results:
        for rank, document in enumerate(
            result.get("documents", ()),
            start=1,
        ):
            document_id = str(document["id"])
            scores[document_id] = scores.get(
                document_id,
                0.0,
            ) + 1.0 / (rank_constant + rank)
            documents.setdefault(document_id, dict(document))

    ranked_ids = sorted(
        scores,
        key=lambda document_id: (
            -scores[document_id],
            document_id,
        ),
    )[:top_k]

    return [
        dict(documents[document_id], score=scores[document_id])
        for document_id in ranked_ids
    ]
