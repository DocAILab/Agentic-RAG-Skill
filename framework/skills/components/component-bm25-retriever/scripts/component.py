"""不依赖第三方检索库的字段感知 BM25F Component。"""

import hashlib
import heapq
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from types import CodeType

TOKENIZER_VERSION = "bm25f:nfkc-casefold-unicode-words:v1"
_MODULE_FINGERPRINT = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def run(inputs, context):
    """分别归一化标题与正文字段，计算 BM25F 分数并返回排序结果。"""
    query = str(inputs.get("query", ""))
    documents = inputs.get("documents", ())
    if not isinstance(documents, (list, tuple)):
        documents = tuple(documents)
    top_k = int(inputs.get("top_k", 10))
    k1 = float(inputs.get("k1", 1.2))
    b = float(inputs.get("b", 0.5))
    title_b = float(inputs.get("title_b", 0.75))
    title_boost = float(inputs.get("title_boost", 3.0))
    if top_k <= 0 or not documents:
        return {"documents": []}
    _validate_parameters(k1, b, title_b, title_boost)

    query_terms = tuple(dict.fromkeys(_tokenize(query)))
    if not query_terms:
        return {"documents": []}
    statistics = _statistics(documents, context)
    candidate_indices = set()
    for term in query_terms:
        candidate_indices.update(statistics["postings"].get(term, ()))
    scores = {}
    for index in candidate_indices:
        score = _score_document(
            query_terms,
            statistics["fields"][index],
            average_title_length=statistics["average_title_length"],
            average_body_length=statistics["average_body_length"],
            document_frequency=statistics["document_frequency"],
            document_count=len(documents),
            k1=k1,
            b=b,
            title_b=title_b,
            title_boost=title_boost,
        )
        if score > 0:
            scores[index] = score
    ranked = heapq.nsmallest(
        top_k,
        scores,
        key=lambda index: (-scores[index], str(documents[index].get("id", "")), index),
    )
    if len(ranked) < top_k:
        for index in statistics["id_order"]:
            if index not in scores:
                ranked.append(index)
                if len(ranked) >= top_k:
                    break
    return {
        "documents": [dict(documents[index], score=scores.get(index, 0.0)) for index in ranked]
    }


def _statistics(documents, context):
    """只读 Group 复用统计；没有可选缓存接口时保留即时构建。"""
    cached = getattr(context, "corpus_cache", None)
    if not callable(cached):
        return _build_statistics(documents)
    implementation = hashlib.sha256(json.dumps(
        [_code_signature(function.__code__) for function in
         (_tokenize, _tokenize_document, _build_statistics)],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    version = ":".join(
        (TOKENIZER_VERSION, unicodedata.unidata_version, _MODULE_FINGERPRINT, implementation)
    )
    return cached(
        documents,
        namespace="bm25f-statistics",
        version=version,
        build=lambda: _build_statistics(documents),
    )


def _code_signature(code):
    """只摘要实现内容，不受反复加载时的对象引用计数或工作副本路径影响。"""
    return (
        code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
        code.co_nlocals, code.co_stacksize, code.co_flags,
        code.co_code.hex(), getattr(code, "co_exceptiontable", b"").hex(),
        code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars,
        tuple(_constant_signature(value) for value in code.co_consts),
    )


def _constant_signature(value):
    """以带类型的确定性表示保留代码常量，含嵌套函数和集合常量。"""
    if isinstance(value, CodeType):
        return ("code", _code_signature(value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_constant_signature(item) for item in value))
    if isinstance(value, frozenset):
        items = [_constant_signature(item) for item in value]
        return ("frozenset", sorted(items, key=lambda item: json.dumps(item)))
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, complex):
        return ("complex", value.real.hex(), value.imag.hex())
    if value is Ellipsis:
        return ("ellipsis",)
    if value is None or isinstance(value, (bool, int, str)):
        return (type(value).__name__, value)
    raise TypeError(f"Unsupported Python code constant: {type(value).__name__}")


def _build_statistics(documents):
    """预计算字段词频、长度和倒排表；排序参数在每次查询时应用。"""
    fields = []
    postings = {}
    title_lengths = 0
    body_lengths = 0
    for index, document in enumerate(documents):
        title, body = _tokenize_document(document)
        title_frequency = Counter(title)
        body_frequency = Counter(body)
        fields.append((title_frequency, len(title), body_frequency, len(body)))
        title_lengths += len(title)
        body_lengths += len(body)
        for term in title_frequency.keys() | body_frequency.keys():
            postings.setdefault(term, []).append(index)
    return {
        "fields": tuple(fields),
        "postings": postings,
        "document_frequency": {term: len(indices) for term, indices in postings.items()},
        "average_title_length": title_lengths / len(documents),
        "average_body_length": body_lengths / len(documents),
        "id_order": tuple(sorted(range(len(documents)), key=lambda index: str(documents[index].get("id", "")))),
    }


def _tokenize(text):
    """执行 NFKC 与大小写折叠后切分 Unicode 字母数字词项。"""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def _tokenize_document(document):
    """分别切分标题与正文，供字段权重计算复用。"""
    return (
        _tokenize(document.get("title", "")),
        _tokenize(document.get("text", "")),
    )


def _score_document(
    query_terms,
    field,
    *,
    average_title_length,
    average_body_length,
    document_frequency,
    document_count,
    k1,
    b,
    title_b,
    title_boost,
):
    title_frequency, title_length, body_frequency, body_length = field
    score = 0.0
    for term in query_terms:
        frequency = _normalized_frequency(
            body_frequency[term], body_length, average_body_length, b
        )
        frequency += title_boost * _normalized_frequency(
            title_frequency[term], title_length, average_title_length, title_b
        )
        if frequency == 0:
            continue
        found_in = document_frequency[term]
        inverse_frequency = math.log(
            1.0 + (document_count - found_in + 0.5) / (found_in + 0.5)
        )
        denominator = frequency + k1
        score += inverse_frequency * frequency * (k1 + 1.0) / denominator
    return score


def _normalized_frequency(frequency, length, average_length, field_b):
    if frequency == 0 or average_length == 0:
        return 0.0
    normalizer = 1.0 - field_b + field_b * length / average_length
    return frequency / normalizer


def _validate_parameters(k1, b, title_b, title_boost):
    if not math.isfinite(k1) or k1 <= 0:
        raise ValueError("BM25 requires a finite k1 > 0")
    if not math.isfinite(b) or not 0 <= b <= 1:
        raise ValueError("BM25 requires a finite 0 <= b <= 1")
    if not math.isfinite(title_b) or not 0 <= title_b <= 1:
        raise ValueError("BM25 requires a finite 0 <= title_b <= 1")
    if not math.isfinite(title_boost) or title_boost < 0:
        raise ValueError("BM25 requires a finite title_boost >= 0")
