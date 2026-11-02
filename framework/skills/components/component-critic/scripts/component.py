"""通过冻结 Executor Model 检查生成答案的证据支持度。"""

import json

from framework.evidence import prepare_evidence

MAX_OUTPUT_ATTEMPTS = 3


def run(inputs, context):
    """将问题、证据和答案提交给模型，并返回校验后的批评结果。"""
    query = str(inputs["query"])
    answer = str(inputs["answer"])
    packed = prepare_evidence(inputs.get("evidence_query", query), inputs.get("documents", ()))
    record = getattr(context, "record_evidence_selection", None)
    if callable(record):
        record(packed.metadata)
    evidence = packed.text
    prompt = (
        "Critique the answer against the question and supplied evidence. "
        "Approve only when it gives a direct answer and every required fact or "
        "reasoning hop is supported by the evidence. Abstentions or statements "
        "that evidence is insufficient must use approved=false, even when honest. "
        "Put concrete missing evidence targets in issues. Return strict JSON "
        "with exactly these fields: approved (boolean), score (number from 0 to 1), "
        "feedback (string), and issues (array of strings).\n\n"
        f"Question: {query}\n\n"
        f"Answer: {answer}\n\n"
        f"Evidence:\n{evidence}"
    )
    base_prompt = prompt
    diagnostics = []
    for attempt in range(MAX_OUTPUT_ATTEMPTS):
        response = context.call_model(
            prompt,
            temperature=0.0,
            max_tokens=inputs.get("max_tokens"),
        )
        try:
            return _validate_result(_parse_json(response))
        except ValueError as exc:
            diagnostics.append(str(exc))
            if attempt + 1 >= MAX_OUTPUT_ATTEMPTS:
                raise ValueError(
                    f"Critic output contract failed after {MAX_OUTPUT_ATTEMPTS} attempts: "
                    + "; ".join(dict.fromkeys(diagnostics))
                ) from exc
            prompt = (
                f"{base_prompt}\n\n"
                "OUTPUT CONTRACT CORRECTION:\n"
                f"Validation error: {exc}\n"
                "Return only one JSON object with exactly approved, score, feedback, issues. "
                "approved must be a JSON boolean, score a number from 0 to 1, "
                "feedback a string, and issues an array of strings (use [] when empty). "
                "Keep the original evidence and approval requirements unchanged."
            )
    raise ValueError("Critic output contract validation failed")  # pragma: no cover


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
        raise ValueError("Critic model JSON must be an object")  # noqa: TRY004
    return payload


def _validate_result(payload):
    """校验批评结果的字段类型和分数范围。"""
    required = {"approved", "score", "feedback", "issues"}
    if set(payload) != required:
        raise ValueError("Critic result must contain exactly approved, score, feedback, issues")
    if not isinstance(payload["approved"], bool):
        raise ValueError("Critic approved must be a boolean")  # noqa: TRY004
    if isinstance(payload["score"], bool) or not isinstance(
        payload["score"], (int, float)
    ) or not 0 <= payload["score"] <= 1:
        raise ValueError("Critic score must be a number between 0 and 1")
    if not isinstance(payload["feedback"], str):
        raise ValueError("Critic feedback must be a string")  # noqa: TRY004
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
