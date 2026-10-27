"""通过冻结 Executor Model 为 Agentic workflow 选择受限执行路线。"""

import json
from collections.abc import Mapping, Sequence


ROUTES_BY_MODE = {
    "retrieval": frozenset({"lexical", "semantic", "hybrid"}),
    "complexity": frozenset({"non-retrieval", "single-step", "multi-step"}),
}
MAX_SAMPLE_DOCUMENTS = 3
MAX_TITLE_CHARS = 160
MAX_EXCERPT_CHARS = 400
MAX_CONSTRAINT_CHARS = 1500
_ANNOTATION_KEYS = frozenset({
    "role", "origin", "source", "metadata", "source_question_ids", "answers",
    "gold_answers", "relevant_document_ids", "supporting_facts", "label_type",
    "reference_count", "distractor_count",
})


def run(inputs, context):
    """将请求上下文提交给模型，并返回校验后的路由分类。"""
    query = str(inputs["query"])
    corpus = _corpus_summary(inputs.get("documents", ()))
    constraints = _bounded_constraints(inputs.get("constraints", {}))
    mode = inputs.get("classification_mode", "retrieval")
    routes = ROUTES_BY_MODE.get(mode)
    if routes is None:
        raise ValueError("Classifier classification_mode must be retrieval or complexity")
    instruction = (
        "Classify this RAG request by execution complexity. Use non-retrieval when "
        "the answer needs no corpus evidence, single-step when one retrieval and "
        "generation pass is sufficient, and multi-step when iterative evidence "
        "gathering or critique is needed."
        if mode == "complexity"
        else "Classify this RAG request into exactly one execution route. Use lexical "
        "for exact names, identifiers, quotations, or strong word overlap; semantic "
        "for paraphrases or weak lexical overlap; use hybrid when both routes are useful."
    )
    prompt = (
        f"{instruction} Return strict JSON with exactly these fields: route, reason, "
        f"confidence. The route must be one of {', '.join(sorted(routes))}, and "
        "confidence must be a number from 0 to 1.\n\n"
        f"Classification mode: {mode}\n\n"
        f"Query: {query}\n\n"
        "The corpus samples below describe the available material; they are not "
        "retrieval results or relevance labels.\n\n"
        f"Corpus summary:\n{json.dumps(corpus, ensure_ascii=False, indent=2)}\n\n"
        f"Constraints:\n{constraints}"
    )
    response = context.call_model(
        prompt,
        temperature=0.0,
        max_tokens=inputs.get("max_tokens"),
    )
    return _validate_result(_parse_json(response), routes)


def _corpus_summary(documents):
    """Summarize shared read-only documents without serializing the full corpus."""
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise ValueError("Classifier documents must be a sequence of mappings")
    count = len(documents)
    if not count:
        return {"document_count": 0, "average_text_characters": 0, "max_text_characters": 0, "samples": []}
    sample_count = min(MAX_SAMPLE_DOCUMENTS, count)
    indices = (
        {0}
        if sample_count == 1
        else {round(index * (count - 1) / (sample_count - 1)) for index in range(sample_count)}
    )
    total_chars = 0
    maximum_chars = 0
    samples = []
    for index, document in enumerate(documents):
        if not isinstance(document, Mapping):
            raise ValueError("Classifier documents must be mappings")
        text = str(document.get("text", ""))
        total_chars += len(text)
        maximum_chars = max(maximum_chars, len(text))
        if index in indices:
            samples.append({
                "id": str(document.get("id", ""))[:80],
                "title": str(document.get("title", ""))[:MAX_TITLE_CHARS],
                "excerpt": text[:MAX_EXCERPT_CHARS],
            })
    return {
        "document_count": count,
        "average_text_characters": round(total_chars / count, 2),
        "max_text_characters": maximum_chars,
        "samples": samples,
    }


def _bounded_constraints(constraints):
    text = json.dumps(_plain_constraints(constraints), ensure_ascii=False, indent=2)
    if len(text) > MAX_CONSTRAINT_CHARS:
        return text[:MAX_CONSTRAINT_CHARS] + "\n[Remaining constraints omitted]"
    return text


def _plain_constraints(value, depth=0):
    """Preserve bounded routing hints while excluding processing annotations."""
    if isinstance(value, Mapping):
        if depth >= 3:
            return "[Nested constraints omitted]"
        result = {}
        for key, item in value.items():
            key = str(key)
            if key in _ANNOTATION_KEYS:
                continue
            result[key[:80]] = _plain_constraints(item, depth + 1)
            if len(result) == 8:
                break
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if depth >= 3:
            return "[Nested constraints omitted]"
        return [_plain_constraints(item, depth + 1) for item in value[:8]]
    if isinstance(value, str):
        return value[:256]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise ValueError("Classifier routing constraints must be JSON-compatible")


def _parse_json(response):
    """解析模型返回的 JSON，并兼容常见的 markdown fenced 输出。"""
    text = str(response).strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Classifier model must return strict JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Classifier model JSON must be an object")
    return payload


def _validate_result(payload, routes):
    """校验路由枚举、置信度范围和结果字段类型。"""
    required = {"route", "reason", "confidence"}
    if set(payload) != required:
        raise ValueError("Classifier result must contain exactly route, reason, confidence")
    if payload["route"] not in routes:
        raise ValueError(
            "Classifier route must be one of: " + ", ".join(sorted(routes))
        )
    if not isinstance(payload["reason"], str):
        raise ValueError("Classifier reason must be a string")
    if isinstance(payload["confidence"], bool) or not isinstance(
        payload["confidence"], (int, float)
    ) or not 0 <= payload["confidence"] <= 1:
        raise ValueError("Classifier confidence must be a number between 0 and 1")
    return {
        "route": payload["route"],
        "reason": payload["reason"],
        "confidence": float(payload["confidence"]),
    }
