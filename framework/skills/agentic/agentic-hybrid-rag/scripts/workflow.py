"""Complexity-routed Agentic workflow."""

SAFE_ANSWER = "Insufficient evidence to answer reliably."
ROUTES = frozenset({"non-retrieval", "single-step", "multi-step"})
ABSTENTION_MARKERS = (
    "insufficient evidence",
    "not enough evidence",
    "cannot answer",
    "unable to answer",
)


def run(request, components):
    """Select the smallest execution route and run it with abstract components."""
    query = request.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Hybrid Agentic requires a non-empty query")
    query = query.strip()
    top_k = _positive_integer(request.get("top_k", 3), "top_k")
    classification = components.call(
        "classifier",
        {
            "query": query,
            "documents": request.get("documents", ()),
            "constraints": request.get("routing_constraints", {}),
            "classification_mode": "complexity",
            "max_tokens": request.get("classifier_max_tokens"),
        },
    )
    route = classification.get("route")
    if route not in ROUTES:
        raise ValueError("Classifier route must be non-retrieval, single-step, or multi-step")
    trace = [{"step": "classify", "route": route, "reason": classification.get("reason"), "confidence": classification.get("confidence")}]
    if route == "non-retrieval":
        answer = _generate(request, components, query, [])
        trace.append({"step": "generate", "document_count": 0})
        return {"answer": answer, "documents": [], "route": route, "trace": trace}
    if route == "single-step":
        documents = _retrieve(request, components, query, top_k, trace)
        documents = _rerank(components, query, documents, trace)
        answer = _generate(request, components, query, documents)
        trace.append({"step": "generate", "document_count": len(documents)})
        return {"answer": answer, "documents": documents, "route": route, "trace": trace}
    return _multi_step(request, components, query, top_k, trace)


def _multi_step(request, components, query, top_k, trace):
    max_iterations = _positive_integer(request.get("max_iterations", 3), "max_iterations")
    evidence = {}
    search_query = query
    for iteration in range(1, max_iterations + 1):
        documents = _retrieve(request, components, search_query, top_k * iteration, trace)
        new_ids = []
        for document in documents:
            item = dict(document)
            identifier = str(item["id"])
            if identifier not in evidence:
                evidence[identifier] = item
                new_ids.append(identifier)
        documents = _rerank(components, query, list(evidence.values()), trace)
        answer = _generate(request, components, query, documents)
        critique = _critique(
            request,
            components,
            query,
            documents,
            answer,
        )
        trace.append({"step": "critique", "iteration": iteration, "approved": critique.get("approved"), "new_document_ids": new_ids})
        if critique.get("approved") is True:
            trace.append({"step": "stop", "iteration": iteration, "reason": "critic_approved"})
            return {"answer": answer, "documents": documents, "route": "multi-step", "trace": trace}
        if not new_ids or iteration == max_iterations:
            trace.append({"step": "stop", "iteration": iteration, "reason": "no_new_evidence" if not new_ids else "max_iterations"})
            return {"answer": SAFE_ANSWER, "documents": documents, "route": "multi-step", "trace": trace}
        search_query = _follow_up_query(query, critique)
    raise AssertionError("multi-step route did not stop")


def _retrieve(request, components, query, top_k, trace):
    retrieval_query = query
    if components.has("rewriter"):
        rewritten = components.call("rewriter", {"query": query, "max_tokens": request.get("rewrite_max_tokens", 256)})
        retrieval_query = rewritten.get("rewritten_query")
        if not isinstance(retrieval_query, str) or not retrieval_query.strip():
            raise ValueError("Rewriter must return a non-empty rewritten_query")
        retrieval_query = retrieval_query.strip()
        trace.append({"step": "rewrite", "query_source": "original"})
    result = components.call("retriever", {"query": retrieval_query, "documents": request.get("documents", ()), "top_k": top_k})
    documents = list(result.get("documents", ()))
    trace.append({"step": "retrieve", "query_source": "rewritten" if retrieval_query != query else "original", "document_count": len(documents)})
    return documents


def _rerank(components, query, documents, trace):
    if not components.has("reranker") or not documents:
        return documents
    result = components.call("reranker", {"query": query, "documents": documents, "top_k": len(documents)})
    ranked = list(result.get("documents", ()))
    trace.append({"step": "rerank", "document_count": len(ranked)})
    return ranked


def _generate(request, components, query, documents):
    result = components.call("generator", {"query": query, "documents": documents, "max_tokens": request.get("max_tokens")})
    return str(result["answer"])


def _critique(request, components, query, documents, answer):
    critique = components.call(
        "critic",
        {
            "query": query,
            "documents": documents,
            "answer": answer,
            "max_tokens": request.get("critic_max_tokens", 4096),
        },
    )
    approved = critique.get("approved")
    score = critique.get("score")
    feedback = critique.get("feedback")
    issues = critique.get("issues")
    if not isinstance(approved, bool):
        raise ValueError("Critic result approved must be a boolean")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("Critic result score must be numeric")
    if not 0.0 <= float(score) <= 1.0:
        raise ValueError("Critic result score must be between 0 and 1")
    if not isinstance(feedback, str) or not isinstance(issues, list):
        raise ValueError("Critic result feedback and issues have invalid types")
    if approved and any(marker in answer.lower() for marker in ABSTENTION_MARKERS):
        critique = dict(critique)
        critique["approved"] = False
        critique["issues"] = [
            "Candidate does not provide a direct answer; retrieve missing evidence.",
            *issues,
        ]
    return critique


def _follow_up_query(query, critique):
    issues = critique.get("issues", ())
    details = "; ".join(str(issue).strip()[:160] for issue in issues[:3] if str(issue).strip())
    feedback = str(critique.get("feedback", "")).strip()[:400]
    return f"{query}\nMissing evidence: {details or feedback}".strip()


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
