"""检索基准共享的样本结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievalDocument:
    id: str
    title: str
    text: str

    def to_dict(self) -> dict[str, str]:
        """将候选文档转换为 framework 可直接消费的字典。"""
        return {"id": self.id, "title": self.title, "text": self.text}


@dataclass(frozen=True, slots=True)
class RetrievalExample:
    id: str
    query: str
    documents: tuple[RetrievalDocument, ...]
    relevant_document_ids: tuple[str, ...] = ()
    label_type: str | None = None
    gold_answers: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_labels(self) -> bool:
        """判断样本是否包含可用于检索评估的相关文档标签。"""
        return bool(self.label_type and self.relevant_document_ids)

    def to_request(self, *, top_k: int) -> dict[str, Any]:
        """构造包含查询、候选文档和召回数量的 RAG 请求。"""
        return {
            "query": self.query,
            "documents": [document.to_dict() for document in self.documents],
            "top_k": top_k,
        }
