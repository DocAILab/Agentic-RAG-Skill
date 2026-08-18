"""三数据集共享的检索样本结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievalDocument:
    id: str
    title: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "title": self.title, "text": self.text}


@dataclass(frozen=True, slots=True)
class RetrievalExample:
    id: str
    query: str
    documents: tuple[RetrievalDocument, ...]
    relevant_document_ids: tuple[str, ...] = ()
    label_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_labels(self) -> bool:
        return bool(self.label_type and self.relevant_document_ids)

    def to_request(self, *, top_k: int) -> dict[str, Any]:
        return {
            "query": self.query,
            "documents": [document.to_dict() for document in self.documents],
            "top_k": top_k,
        }
