"""通过冻结 Executor Model 执行基于证据的具体生成组件。"""


_NO_EVIDENCE = "[No evidence supplied]"


def run(inputs, context):
    """将问题和检索证据构造成提示词，并调用 Executor Model 生成答案。"""
    query = _required_text(inputs, "query", "query")
    documents = _normalize_documents(inputs.get("documents", ()))
    evidence = _format_evidence(documents)
    prompt = (
        "Answer the question using only the supplied evidence. "
        "If the evidence is insufficient, say so.\n\n"
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


def _normalize_documents(documents):
    """校验并规范化证据文档，避免生成提示词时隐式吞掉坏输入。"""
    if documents is None:
        return []
    if isinstance(documents, (str, bytes, bytearray)):
        raise ValueError("documents must be a sequence of mappings")
    normalized = []
    for index, document in enumerate(documents):
        try:
            payload = dict(document)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"documents[{index}] must be a mapping") from exc
        normalized.append(
            {
                "id": _required_text(payload, "id", f"documents[{index}].id"),
                "text": _required_text(payload, "text", f"documents[{index}].text"),
            }
        )
    return normalized


def _format_evidence(documents):
    """按稳定顺序把证据格式化为模型可读文本。"""
    if not documents:
        return _NO_EVIDENCE
    return "\n\n".join(
        f"[{index}] id={document['id']}\n{document['text']}"
        for index, document in enumerate(documents, start=1)
    )


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
