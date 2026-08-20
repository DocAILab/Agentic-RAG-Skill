"""通过冻结 Executor Model 检查生成答案的证据支持度。"""

import json


def run(inputs, context):
    """将问题、证据和答案提交给模型，并返回校验后的批评结果。"""
    query = str(inputs["query"])
    answer = str(inputs["answer"])
    documents = list(inputs.get("documents", ()))
    evidence = "\n\n".join(
        f"[{index}] id={document['id']}\n{document.get('text', '')}"
        for index, document in enumerate(documents, start=1)
    )
    prompt = (
        "Critique the answer against the question and supplied evidence. "
        "Check factual support, relevance, and completeness. Return strict JSON "
        "with exactly these fields: approved (boolean), score (number from 0 to 1), "
        "feedback (string), and issues (array of strings).\n\n"
        f"Question: {query}\n\n"
        f"Answer: {answer}\n\n"
        f"Evidence:\n{evidence}"
    )
    response = context.call_model(
        prompt,
        temperature=0.0,
        max_tokens=inputs.get("max_tokens"),
    )
    payload = _parse_json(response)
    return _validate_result(payload)


def _parse_json(response):
    """解析模型返回的 JSON，并兼容常见的 markdown fenced 输出。"""
    text = str(response).strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Critic model must return strict JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Critic model JSON must be an object")
    return payload


def _validate_result(payload):
    """校验批评结果的字段类型和分数范围。"""
    required = {"approved", "score", "feedback", "issues"}
    if set(payload) != required:
        raise ValueError("Critic result must contain exactly approved, score, feedback, issues")
    if not isinstance(payload["approved"], bool):
        raise ValueError("Critic approved must be a boolean")
    if isinstance(payload["score"], bool) or not isinstance(
        payload["score"], (int, float)
    ) or not 0 <= payload["score"] <= 1:
        raise ValueError("Critic score must be a number between 0 and 1")
    if not isinstance(payload["feedback"], str):
        raise ValueError("Critic feedback must be a string")
    if not isinstance(payload["issues"], list) or not all(
        isinstance(issue, str) for issue in payload["issues"]
    ):
        raise ValueError("Critic issues must be a list of strings")
    return {
        "approved": payload["approved"],
        "score": float(payload["score"]),
        "feedback": payload["feedback"],
        "issues": list(payload["issues"]),
    }