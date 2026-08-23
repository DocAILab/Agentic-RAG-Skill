"""SIM-RAG-inspired iterative Agentic workflow."""

SAFE_ANSWER = "Insufficient evidence to answer reliably."
DEFAULT_CRITIC_MAX_TOKENS = 4096
ABSTENTION_MARKERS = (
    "insufficient evidence",
    "evidence is insufficient",
    "not enough evidence",
    "not enough information",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "cannot determine",
    "cannot be determined",
    "not provided in the evidence",
)


def run(request, components):
    """Retrieve, generate, and critique until evidence is sufficient."""
    query, top_k, max_iterations = _validate_request(request)
    evidence = {}
    trace = []
    search_query = query

    for iteration in range(1, max_iterations + 1):
        retrieval_query = _rewrite(request, components, search_query)
        requested_top_k = top_k * iteration
        retrieved = components.call(
            "retriever",
            {
                "query": retrieval_query,
                "documents": request.get("documents", ()),
                "top_k": requested_top_k,
            },
        )
        new_ids = _merge_evidence(evidence, retrieved.get("documents", ()))
        new_count = len(new_ids)
        documents = _rerank(components, query, list(evidence.values()))
        answer = _generate(request, components, query, documents)
        critique = _critique(request, components, query, documents, answer)
        trace.append(
            _iteration_trace(
                iteration,
                query,
                search_query,
                retrieval_query,
                requested_top_k,
                documents,
                new_ids,
                answer,
                critique,
            )
        )
        reason = _stop_reason(critique, iteration, max_iterations, new_count)
        if reason:
            trace.append({"step": "stop", "iteration": iteration, "reason": reason})
            final_answer = answer if critique["approved"] else SAFE_ANSWER
            return {"answer": final_answer, "documents": documents, "trace": trace}
        search_query = _follow_up_query(query, critique)

    raise AssertionError("Iteration loop ended without a stop reason")


def _validate_request(request):
    query = request["query"]
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Iterative request requires a non-empty query")
    top_k = _positive_integer(request.get("top_k", 3), "top_k")
    max_iterations = _positive_integer(
        request.get("max_iterations", 3),
        "max_iterations",
    )
    _positive_integer(
        request.get("critic_max_tokens", DEFAULT_CRITIC_MAX_TOKENS),
        "critic_max_tokens",
    )
    return query.strip(), top_k, max_iterations


def _positive_integer(value, field_name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _rewrite(request, components, search_query):
    if not components.has("rewriter"):
        return search_query
    rewritten = components.call(
        "rewriter",
        {
            "query": search_query,
            "temperature": request.get("rewrite_temperature", 0.0),
            "max_tokens": request.get("rewrite_max_tokens", 256),
        },
    )
    rewritten_query = rewritten.get("rewritten_query")
    if not isinstance(rewritten_query, str) or not rewritten_query.strip():
        raise ValueError("Rewriter must return a non-empty rewritten_query")
    return rewritten_query.strip()


def _rerank(components, query, documents):
    if not components.has("reranker") or not documents:
        return documents
    reranked = components.call(
        "reranker",
        {"query": query, "documents": documents, "top_k": len(documents)},
    )
    return list(reranked.get("documents", ()))


def _merge_evidence(evidence, documents):
    new_ids = []
    for document in documents:
        item = dict(document)
        document_id = str(item["id"])
        if document_id in evidence:
            continue
        evidence[document_id] = item
        new_ids.append(document_id)
    return new_ids


def _generate(request, components, query, documents):
    generated = components.call(
        "generator",
        {
            "query": query,
            "documents": documents,
            "max_tokens": request.get("max_tokens"),
        },
    )
    return str(generated["answer"])


def _critique(request, components, query, documents, answer):
    critique = components.call(
        "critic",
        {
            "query": query,
            "documents": documents,
            "answer": answer,
            "max_tokens": request.get(
                "critic_max_tokens",
                DEFAULT_CRITIC_MAX_TOKENS,
            ),
        },
    )
    _validate_critique(critique)
    return _reject_approved_abstention(critique, answer)


def _reject_approved_abstention(critique, answer):
    normalized = answer.strip().lower()
    if not critique["approved"] or not any(
        marker in normalized for marker in ABSTENTION_MARKERS
    ):
        return critique
    corrected = dict(critique)
    corrected["approved"] = False
    corrected["issues"] = [
        "Candidate does not provide a direct answer; retrieve missing evidence.",
        *critique["issues"],
    ]
    return corrected


def _validate_critique(critique):
    approved = critique.get("approved")
    score = critique.get("score")
    feedback = critique.get("feedback")
    issues = critique.get("issues")
    if not isinstance(approved, bool):
        raise ValueError("Critic result approved must be a boolean")  # noqa: TRY004
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("Critic result score must be numeric")  # noqa: TRY004
    if not 0.0 <= float(score) <= 1.0:
        raise ValueError("Critic result score must be between 0 and 1")
    if not isinstance(feedback, str):
        raise ValueError("Critic result feedback must be a string")  # noqa: TRY004
    if not isinstance(issues, list) or not all(isinstance(issue, str) for issue in issues):
        raise ValueError("Critic result issues must be a list of strings")


def _stop_reason(critique, iteration, max_iterations, new_count):
    if critique["approved"]:
        return "critic_approved"
    if iteration >= max_iterations:
        return "max_iterations"
    if new_count == 0:
        return "no_new_evidence"
    return None


def _follow_up_query(query, critique):
    issues = [
        _clip(str(issue).strip(), 160)
        for issue in critique.get("issues", ())
        if str(issue).strip()
    ][:3]
    if issues:
        return f"{query}\nMissing evidence: {'; '.join(issues)}"
    feedback = _clip(str(critique.get("feedback", "")).strip(), 240)
    return f"{query}\nCritic guidance: {feedback}" if feedback else query


def _clip(text, limit):
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _iteration_trace(
    iteration,
    original_query,
    search_query,
    retrieval_query,
    requested_top_k,
    documents,
    new_ids,
    answer,
    critique,
):
    return {
        "step": "iteration",
        "iteration": iteration,
        "original_query": original_query,
        "search_query": search_query,
        "retrieval_query": retrieval_query,
        "requested_top_k": requested_top_k,
        "document_count": len(documents),
        "new_document_count": len(new_ids),
        "document_ids": [str(document["id"]) for document in documents],
        "new_document_ids": list(new_ids),
        "candidate_answer": answer,
        "critic": dict(critique),
    }
