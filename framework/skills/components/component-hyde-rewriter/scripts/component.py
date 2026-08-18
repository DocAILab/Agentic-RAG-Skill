"""使用冻结 Executor Model 生成假设文档的 HyDE Rewriter Component。"""


def run(inputs, context):
    """根据原始问题生成只用于语义检索的假设文档。"""
    query = str(inputs["query"]).strip()
    if not query:
        raise ValueError("HyDE requires a non-empty query")

    temperature_value = inputs.get("temperature", 0.0)
    max_tokens_value = inputs.get("max_tokens", 256)

    temperature = (
        0.0 if temperature_value is None else float(temperature_value)
    )
    max_tokens = (
        256 if max_tokens_value is None else int(max_tokens_value)
    )

    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    prompt = (
        "Write a concise passage that would answer the question. "
        "The passage is hypothetical and will only be used to retrieve "
        "real documents. Do not include commentary, citations, or retrieval "
        "instructions.\n\n"
        f"Question: {query}\n\n"
        "Hypothetical passage:"
    )

    rewritten_query = context.call_model(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    ).strip()

    if not rewritten_query:
        raise ValueError(
            "HyDE model returned an empty hypothetical document"
        )

    return {"rewritten_query": rewritten_query}