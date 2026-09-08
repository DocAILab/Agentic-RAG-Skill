"""Skill 自优化过程使用的不可变数据契约。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SkillEdit:
    """描述优化模型对单个 Skill 内一个文本文件的完整替换。"""

    path: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """将文件修改转换为可写入日志的 JSON 对象。"""
        return {"path": self.path, "content": self.content}


@dataclass(frozen=True, slots=True)
class OptimizationProposal:
    """保存优化模型选择的唯一 Skill、修改理由和文件修改。"""

    selected_skill: str
    rationale: str
    edits: tuple[SkillEdit, ...]

    def to_dict(self) -> dict[str, Any]:
        """将优化提案转换为可序列化字典。"""
        return {
            "selected_skill": self.selected_skill,
            "rationale": self.rationale,
            "edits": [edit.to_dict() for edit in self.edits],
        }


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
        }
