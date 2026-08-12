"""通过冻结 Executor Model 执行基于证据的具体生成组件。"""


def run(inputs, context):
    """将问题和检索证据构造成提示词，并调用 Executor Model 生成答案。"""
    query = str(inputs["query"])
    documents = list(inputs.get("documents", ()))
    evidence = "\n\n".join(
        f"[{index}] id={document['id']}\n{document.get('text', '')}"
        for index, document in enumerate(documents, start=1)
    )
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
    return {"answer": str(answer)}
