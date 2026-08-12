"""不依赖第三方检索库的具体 Okapi BM25 Component。"""

import math
import re
from collections import Counter


def run(inputs, context):
    """计算查询与候选文档的 BM25 分数并返回排序结果。"""
    del context
    query = str(inputs["query"])
    documents = [dict(document) for document in inputs.get("documents", ())]
    top_k = int(inputs.get("top_k", 3))
    k1 = float(inputs.get("k1", 1.5))
    b = float(inputs.get("b", 0.75))
    if top_k <= 0 or not documents:
        return {"documents": []}
    if k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")

    query_terms = _tokenize(query)
    tokenized = [_tokenize(str(document.get("text", ""))) for document in documents]
    average_length = sum(map(len, tokenized)) / max(1, len(tokenized))
    document_frequency = Counter(
        term for tokens in tokenized for term in set(tokens)
    )
    scored = []
    for document, tokens in zip(documents, tokenized, strict=True):
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if frequency == 0:
                continue
            document_count = len(documents)
            frequency_count = document_frequency[term]
            inverse_document_frequency = math.log(
                1.0 + (document_count - frequency_count + 0.5) / (frequency_count + 0.5)
            )
            normalization = frequency + k1 * (
                1.0 - b + b * len(tokens) / max(average_length, 1.0)
            )
            score += inverse_document_frequency * frequency * (k1 + 1.0) / normalization
        scored.append(dict(document, score=score))
    scored.sort(key=lambda document: (-document["score"], str(document["id"])))
    return {"documents": scored[:top_k]}


def _tokenize(text):
    """将文本切分为小写字母数字词项。"""
    return [token.lower() for token in re.findall(r"\w+", text)]
