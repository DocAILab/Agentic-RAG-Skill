"""通过冻结 Executor Model 为 Agentic workflow 选择受限执行路线。"""

import json


ROUTES = frozenset({"lexical", "semantic", "hybrid"})


def run(inputs, context):
    """将请求上下文提交给模型，并返回校验后的路由分类。"""
    query = str(inputs["query"])
    documents = list(inputs.get("documents", ()))
    constraints = inputs.get("constraints", {})
    prompt = (
        "Classify this RAG request into exactly one execution route. "
        "Use lexical for exact names, identifiers, quotations, or strong word overlap; "
        "semantic for paraphrases or weak lexical overlap; use hybrid when both routes "
        "are useful. Return strict JSON with exactly these fields: route, reason, "
        "confidence. The route must be one of lexical, semantic, or hybrid, and "
        "confidence must be a number from 0 to 1.\n\n"
        f"Query: {query}\n\n"
        f"Documents:\n{json.dumps(documents, ensure_ascii=False, indent=2)}\n\n"
        f"Constraints:\n{json.dumps(constraints, ensure_ascii=False, indent=2)}"
    )
    response = context.call_model(
        prompt,
        temperature=0.0,
        max_tokens=inputs.get("max_tokens"),
    )
    return _validate_result(_parse_json(response))


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


def _validate_result(payload):
    """校验路由枚举、置信度范围和结果字段类型。"""
    required = {"route", "reason", "confidence"}
    if set(payload) != required:
        raise ValueError("Classifier result must contain exactly route, reason, confidence")
    if payload["route"] not in ROUTES:
        raise ValueError("Classifier route must be lexical, semantic, or hybrid")
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