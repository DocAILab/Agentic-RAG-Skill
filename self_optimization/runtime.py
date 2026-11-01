"""Structured Task events and bounded diagnostic records."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class OptimizationEventLogger:
    """把自优化阶段按顺序追加为 JSON Lines 事件。"""

    path: Path
    run_id: str
    session_id: str = field(default_factory=lambda: uuid4().hex)
    _lock: Any = field(default_factory=RLock, repr=False)

    def write(self, event: str, **payload: Any) -> None:
        """写入一条带时间、运行 ID 和会话 ID 的结构化事件。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "session_id": self.session_id,
            "event": event,
            **payload,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_retrieval_process(
    *,
    question: str,
    result: Mapping[str, Any],
    max_documents: int,
    excerpt_chars: int,
) -> dict[str, Any]:
    """提取选择、检索轨迹、耗时和有界文档片段供优化模型分析。"""
    documents_value = result.get("documents", [])
    documents = []
    if isinstance(documents_value, Sequence) and not isinstance(
        documents_value,
        (str, bytes, bytearray),
    ):
        for document in documents_value[:max_documents]:
            if not isinstance(document, Mapping):
                continue
            text = str(document.get("text", ""))
            documents.append(
                {
                    "id": document.get("id"),
                    "title": document.get("title"),
                    "score": document.get("score"),
                    "text_excerpt": text[:excerpt_chars],
                    "text_truncated": len(text) > excerpt_chars,
                }
            )
    return {
        "question": question,
        "prediction": result.get("answer"),
        "trace": result.get("trace", []),
        "component_timings": result.get("component_timings", []),
        "retrieved_documents": documents,
    }


def task_retrieval_process(
    *, task_name: str, dataset: str, executed_examples: int,
    examples: Sequence[Mapping[str, Any]], selection: Mapping[str, Any],
    task_plan_hash: str, max_chars: int,
) -> dict[str, Any]:
    """Share the Task plan once and fit complete diagnostic examples to budget.

    This samples feedback only; every QA has already executed and its metrics
    remain in evaluation. Skill source files are supplied separately in full.
    """
    available = len(examples)
    for count in range(available, -1, -1):
        if count == 0:
            indices = []
        elif count == 1:
            indices = [available // 2]
        else:
            indices = [round(i * (available - 1) / (count - 1)) for i in range(count)]
        payload = {
            "task_name": task_name, "dataset": dataset,
            "executed_examples": executed_examples,
            "available_process_examples": available,
            "included_process_examples": count,
            "omitted_process_examples": available - count,
            "included_question_ids": [examples[i].get("question_id") for i in indices],
            "selection": dict(selection), "task_plan_hash": task_plan_hash,
            "examples": [examples[i] for i in indices],
        }
        if len(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)) <= max_chars:
            return payload
    # Keep the original structured metadata so the optimizer can report the
    # configured limit explicitly; never send a broken JSON prefix.
    return payload


def _retrieved_ids(result: Mapping[str, Any]) -> list[str]:
    """从执行结果中提取有序检索文档 ID。"""
    documents = result.get("documents")
    if isinstance(documents, (str, bytes, bytearray)) or not isinstance(
        documents,
        Sequence,
    ):
        raise ValueError("RAG result.documents must be a sequence")
    identifiers = []
    for index, document in enumerate(documents):
        if not isinstance(document, Mapping):
            raise ValueError(
                f"RAG result.documents[{index}] must be an object"
            )
        identifiers.append(_required_text(document, "id"))
    return identifiers


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    """读取对象中的必需非空字符串。"""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-empty string")
    return value.strip()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """创建父目录并写入格式化 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _safe_error_message(error: Exception) -> str:
    """在日志前移除可能出现的 OpenAI 风格密钥。"""
    return re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED]", str(error))
