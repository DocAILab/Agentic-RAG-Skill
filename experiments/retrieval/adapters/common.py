"""数据集适配器共享的结构归一化工具。"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from ..schema import RetrievalDocument


class AdapterError(ValueError):
    """表示单个源样本字段缺失或无法构造候选文档。"""

    def __init__(self, sample_id: str, reason: str):
        self.sample_id = sample_id
        super().__init__(f"sample {sample_id}: {reason}")


def sample_id(row: Mapping[str, Any]) -> str:
    for key in ("id", "_id", "question_id", "financebench_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "<unknown>"


def required_text(row: Mapping[str, Any], key: str, identity: str) -> str:
    """读取数据集样本中的必需非空文本字段。"""
    value = row.get(key)
    if value is None or not str(value).strip():
        raise AdapterError(identity, f"missing non-empty '{key}'")
    return str(value).strip()


def records(value: Any, fields: tuple[str, ...], identity: str) -> list[dict]:
    """把 JSON、列式映射或记录序列统一转换为字典列表。"""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise AdapterError(identity, "contains invalid JSON records") from exc
        return records(decoded, fields, identity)
    if isinstance(value, Mapping):
        return _transpose_mapping(value, fields, identity)
    if not _is_sequence(value):
        raise AdapterError(identity, "expected a sequence or column mapping")
    normalized = []
    for item in value:
        if isinstance(item, Mapping):
            normalized.append(dict(item))
        elif _is_sequence(item):
            normalized.append(dict(zip(fields, item, strict=False)))
        else:
            raise AdapterError(identity, "contains an invalid record")
    return normalized


def context_documents(value: Any, text_field: str, identity: str):
    """把数据集 context 转换为唯一 ID 的候选文档及标题索引。"""
    source = records(value, ("title", text_field), identity)
    documents = []
    title_ids: dict[str, list[str]] = {}
    seen: dict[str, int] = {}
    for index, item in enumerate(source):
        title = str(item.get("title", "")).strip()
        if not title:
            raise AdapterError(identity, f"context item {index} has no title")
        document_id = _unique_id(title, seen)
        text = join_text(item.get(text_field, ""))
        documents.append(RetrievalDocument(document_id, title, text))
        title_ids.setdefault(normalize_title(title), []).append(document_id)
    if not documents:
        raise AdapterError(identity, "candidate documents are empty")
    return tuple(documents), title_ids


def supporting_titles(value: Any, identity: str) -> tuple[str, ...]:
    """从 supporting facts 中按出现顺序提取证据标题。"""
    if value in (None, [], {}):
        return ()
    items = records(value, ("title", "sent_id"), identity)
    return tuple(str(item.get("title", "")).strip() for item in items if item.get("title"))


def relevant_ids(titles, title_ids):
    """把标准化证据标题映射到候选文档 ID。"""
    ids = []
    for title in titles:
        ids.extend(title_ids.get(normalize_title(title), ()))
    return tuple(dict.fromkeys(ids))


def join_text(value: Any) -> str:
    """把句子序列或单段文本归一化为一个文档正文。"""
    if _is_sequence(value):
        return " ".join(str(part).strip() for part in value if str(part).strip())
    return str(value or "").strip()


def normalize_title(value: str) -> str:
    """使用 Unicode 规范化和大小写折叠生成标题匹配键。"""
    return unicodedata.normalize("NFKC", value).casefold().strip()


def answer_values(value: Any) -> tuple[str, ...]:
    """从字符串、序列或 TriviaQA answer 映射中读取答案及别名。"""
    candidates = []
    if isinstance(value, Mapping):
        candidates.append(value.get("value"))
        aliases = value.get("aliases", ())
        if _is_sequence(aliases):
            candidates.extend(aliases)
    elif _is_sequence(value):
        candidates.extend(value)
    else:
        candidates.append(value)
    normalized = [str(item).strip() for item in candidates if item is not None]
    return tuple(dict.fromkeys(item for item in normalized if item))


def _transpose_mapping(value, fields, identity):
    """把字段到列的映射转置为逐条记录。"""
    columns = {field: value.get(field, []) for field in fields}
    lengths = [len(column) for column in columns.values() if _is_sequence(column)]
    if not lengths or len(set(lengths)) != 1:
        raise AdapterError(identity, "column mapping has inconsistent lengths")
    return [
        {field: columns[field][index] for field in fields}
        for index in range(lengths[0])
    ]


def _unique_id(title, seen):
    """为重复标题追加确定性序号，避免文档 ID 冲突。"""
    count = seen.get(title, 0) + 1
    seen[title] = count
    return title if count == 1 else f"{title}#{count}"


def _is_sequence(value):
    """判断对象是否为非字符串序列。"""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
