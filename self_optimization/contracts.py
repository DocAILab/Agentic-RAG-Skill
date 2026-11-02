"""Skill 自优化过程使用的不可变数据契约。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_PATCH_REPLACEMENTS = 32


@dataclass(frozen=True, slots=True)
class TextReplacement:
    """One exact replacement against the original LF-normalized file text."""

    old: str
    new: str

    def to_dict(self) -> dict[str, str]:
        return {"old": self.old, "new": self.new}


def normalize_patch_text(content: str) -> str:
    """Match read_text's universal-newline view, including Windows CRLF files."""
    return content.replace("\r\n", "\n").replace("\r", "\n")


def patch_base_sha256(content: str) -> str:
    """Hash the displayed LF-normalized UTF-8 text, not raw on-disk bytes."""
    return hashlib.sha256(normalize_patch_text(content).encode("utf-8")).hexdigest()


def materialize_patch(
    original: str, base_sha256: str, replacements: tuple[TextReplacement, ...],
) -> str:
    """Validate all original spans before replacing any; never chain replacements."""
    if not isinstance(base_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", base_sha256) is None:
        raise ValueError("Patch base_sha256 must contain exactly 64 lowercase hex characters")
    source = normalize_patch_text(original)
    if patch_base_sha256(source) != base_sha256:
        raise ValueError("Patch base_sha256 does not match the current original text (stale patch)")
    if not replacements:
        raise ValueError("Patch replacements must contain at least one item")
    if len(replacements) > MAX_PATCH_REPLACEMENTS:
        raise ValueError(f"Patch replacements exceed {MAX_PATCH_REPLACEMENTS} items per file")
    spans = []
    for replacement in replacements:
        if not isinstance(replacement, TextReplacement):
            raise ValueError("Patch replacements must be TextReplacement values")
        old, new = replacement.old, replacement.new
        if not isinstance(old, str) or not old:
            raise ValueError("Patch old must be a non-empty exact string")
        if not isinstance(new, str):
            raise ValueError("Patch new must be a string; use an empty string for deletion")
        if "\r" in old or "\r" in new:
            raise ValueError("Patch old/new must use LF newlines exactly as in the visible original")
        if old == new:
            raise ValueError("Patch replacement does not make a substantive change: old equals new")
        start = source.find(old)
        if start < 0:
            raise ValueError("Patch old has no exact match in the original text")
        # Count overlapping matches too: e.g. 'aa' is ambiguous in 'aaa'.
        if source.find(old, start + 1) >= 0:
            raise ValueError("Patch old must match exactly once in the original text (ambiguous anchor)")
        spans.append((start, start + len(old), new))
    spans.sort(key=lambda span: span[0])
    for previous, current in zip(spans, spans[1:]):
        if current[0] < previous[1]:
            raise ValueError("Patch replacements overlap in the original text")
    pieces, cursor = [], 0
    for start, end, new in spans:
        pieces.extend((source[cursor:start], new))
        cursor = end
    pieces.append(source[cursor:])
    result = "".join(pieces)
    if result == source:
        raise ValueError("Patch does not make a substantive change to the original text")
    return result


@dataclass(frozen=True, slots=True)
class SkillEdit:
    """Materialized replacement plus optional immutable original patch action."""

    path: str
    content: str
    base_sha256: str | None = None
    replacements: tuple[TextReplacement, ...] = ()

    def patch_action(self) -> dict[str, Any]:
        return {"path": self.path, "base_sha256": self.base_sha256,
                "replacements": [replacement.to_dict() for replacement in self.replacements]}

    def to_dict(self) -> dict[str, Any]:
        """将文件修改转换为可写入日志的 JSON 对象。"""
        result = {"path": self.path, "content": self.content}
        if self.base_sha256 is not None or self.replacements:
            result.update(self.patch_action())
        return result


@dataclass(frozen=True, slots=True)
class OptimizationProposal:
    """保存优化模型选择的唯一 Skill、修改理由和文件修改。"""

    selected_skill: str
    rationale: str
    edits: tuple[SkillEdit, ...]
    edit_mode: str = "full"

    def to_dict(self) -> dict[str, Any]:
        """将优化提案转换为可序列化字典。"""
        result = {
            "selected_skill": self.selected_skill,
            "rationale": self.rationale,
            "edits": [edit.to_dict() for edit in self.edits],
        }
        if self.edit_mode == "patch":
            result.update(edit_mode="patch", raw_action={
                "edits": [edit.patch_action() for edit in self.edits],
            })
        return result


@dataclass(frozen=True, slots=True)
class RevisionResult:
    """记录一次通过结构校验并提交到工作副本的 Skill 修订。"""

    revision_id: str
    selected_skill: str
    changed_files: tuple[str, ...]
    before_hash: str
    after_hash: str
    revision_dir: Path
    diff_path: Path
    edit_mode: str = "full"

    def to_dict(self) -> dict[str, Any]:
        """将修订结果转换为实验报告使用的 JSON 对象。"""
        return {
            "revision_id": self.revision_id,
            "selected_skill": self.selected_skill,
            "changed_files": list(self.changed_files),
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "revision_dir": str(self.revision_dir),
            "diff_path": str(self.diff_path),
            "edit_mode": self.edit_mode,
        }
