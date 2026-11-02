"""通过冻结 Executor Model 执行基于证据的具体生成组件。"""


from framework.evidence import prepare_evidence


def run(inputs, context):
    """将问题和检索证据构造成提示词，并调用 Executor Model 生成答案。"""
    query = _required_text(inputs, "query", "query")
    # evidence_query anchors excerpts to the original question during answer revision.
    packed = prepare_evidence(inputs.get("evidence_query", query), inputs.get("documents", ()))
    record = getattr(context, "record_evidence_selection", None)
    if callable(record):
        record(packed.metadata)
    evidence = packed.text
    prompt = (
        "Answer the question using only the supplied evidence. "
        "Return only the shortest direct answer span, with no explanation. "
        "If the evidence is insufficient, return exactly: "
        "Insufficient evidence to answer reliably.\n\n"
        f"Question: {query}\n\nEvidence:\n{evidence}"
    )
    answer = context.call_model(
        prompt,
        temperature=0.0,
        max_tokens=inputs.get("max_tokens"),
    )
    answer_text = str(answer).strip()
    if not answer_text:
        raise ValueError("Grounded generator returned an empty answer")
    return {"answer": answer_text}


def _required_text(payload, key, label):
    """读取必需文本字段，并提供可定位的错误信息。"""
    try:
        value = payload[key]
    except KeyError as exc:
        raise ValueError(f"{label} is required") from exc
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text
