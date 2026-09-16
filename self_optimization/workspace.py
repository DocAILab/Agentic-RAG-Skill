"""创建只修改副本的 Skill 工作区，并以事务方式提交单 Skill 修订。"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from framework import RAGSkillSpec, SkillSpecError, discover_specs

from .contracts import OptimizationProposal, RevisionResult, SkillEdit
from .python_validation import (
    validate_agentic_slot_references,
    validate_component_context_contract,
    validate_component_output_contract,
    validate_python_file,
)

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_TEXT_SUFFIXES = {".json", ".md", ".py", ".txt", ".yaml", ".yml"}
_EDITABLE_DIRECTORIES = {"references", "scripts"}


class WorkspaceError(ValueError):
    """表示工作区创建、路径隔离或源仓库完整性校验失败。"""


class RevisionRejected(WorkspaceError):
    """表示优化提案越界、无变化或破坏了 Skill 契约并已回滚。"""


@dataclass(slots=True)
class SkillWorkspace:
    """维护基础 Skill 的只读来源、运行副本及逐轮修订历史。"""

    source_root: Path
    run_root: Path
    skill_root: Path
    revisions_root: Path
    source_hash: str
    _revision_number: int = field(default=0, init=False, repr=False)

    @classmethod
    def create(
        cls,
        source_root: str | Path,
        runs_root: str | Path,
        *,
        run_id: str | None = None,
    ) -> SkillWorkspace:
        """校验基础 Skill 后创建一次性运行目录和完整工作副本。"""
        source = Path(source_root).resolve()
        runs = Path(runs_root).resolve()
        if not source.is_dir():
            raise WorkspaceError(f"Source Skill root does not exist: {source}")
        _reject_symlinks(source)
        try:
            discover_specs(source)
        except SkillSpecError as exc:
            raise WorkspaceError(f"Source Skill repository is invalid: {exc}") from exc
        if runs == source or runs.is_relative_to(source):
            raise WorkspaceError("runs_root must be outside the source Skill root")

        normalized_run_id = run_id or _default_run_id()
        if not _RUN_ID_PATTERN.fullmatch(normalized_run_id):
            raise WorkspaceError(
                "run_id must use 1-80 letters, digits, dots, underscores, or hyphens"
            )
        run_root = (runs / normalized_run_id).resolve()
        if run_root.parent != runs:
            raise WorkspaceError("run_id escapes runs_root")
        if run_root.exists():
            raise WorkspaceError(f"Self-optimization run already exists: {run_root}")

        runs.mkdir(parents=True, exist_ok=True)
        skill_root = run_root / "skills"
        revisions_root = run_root / "revisions"
        source_hash = _tree_hash(source)
        try:
            shutil.copytree(source, skill_root)
            revisions_root.mkdir(parents=True)
            _write_json(
                run_root / "workspace.json",
                {
                    "schema_version": 1,
                    "run_id": normalized_run_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "source_root": str(source),
                    "source_hash": source_hash,
                    "skill_root": str(skill_root),
                },
            )
        except Exception:
            if run_root.exists():
                shutil.rmtree(run_root)
            raise
        return cls(
            source_root=source,
            run_root=run_root,
            skill_root=skill_root,
            revisions_root=revisions_root,
            source_hash=source_hash,
        )

    @property
    def run_id(self) -> str:
        """返回当前工作区目录所代表的稳定运行标识。"""
        return self.run_root.name

    def working_hash(self) -> str:
        """计算当前 Skill 工作副本的内容哈希，用于记录跨 task 的版本变化。"""
        return _tree_hash(self.skill_root)

    def candidate_snapshots(
        self,
        names: Sequence[str],
        *,
        max_file_chars: int,
    ) -> list[dict[str, Any]]:
        """读取本轮参与的 Skill 文件，并标注优化器可以修改的文件。"""
        if max_file_chars <= 0:
            raise WorkspaceError("max_file_chars must be positive")
        specs = _specs_by_name(self.skill_root)
        snapshots = []
        for name in _unique_names(names):
            spec = specs.get(name)
            if spec is None:
                raise WorkspaceError(f"Unknown candidate Skill: {name}")
            files = []
            for path in sorted(spec.package_path.rglob("*")):
                if not _is_snapshot_file(path, spec.package_path):
                    continue
                relative = path.relative_to(spec.package_path).as_posix()
                try:
                    content = path.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as exc:
                    raise WorkspaceError(f"Cannot read Skill file: {path}") from exc
                truncated = len(content) > max_file_chars
                files.append(
                    {
                        "path": relative,
                        "editable": _is_editable_path(relative),
                        "truncated": truncated,
                        "content": (
                            content[:max_file_chars] + "\n...[truncated]"
                            if truncated
                            else content
                        ),
                    }
                )
            snapshots.append(
                {
                    "name": spec.package_name,
                    "description": spec.description,
                    "kind": spec.kind.value,
                    "version": spec.version,
                    "allowed_component_slots": [slot.name for slot in spec.slots],
                    "provided_capabilities": [
                        {
                            "name": capability.name,
                            "input_type": capability.input_type,
                            "output_type": capability.output_type,
                        }
                        for capability in spec.provides
                    ],
                    "files": files,
                }
            )
        return snapshots

    def apply_proposal(
        self,
        proposal: OptimizationProposal,
        *,
        candidate_names: Sequence[str],
        max_edits: int,
        max_file_chars: int,
        allow_new_files: bool,
    ) -> RevisionResult:
        """只在工作副本中应用一个 Skill 的文件替换，校验失败时完整回滚。"""
        self.assert_source_unchanged()
        specs = _specs_by_name(self.skill_root)
        allowed_candidates = set(_unique_names(candidate_names))
        if proposal.selected_skill not in allowed_candidates:
            raise RevisionRejected(
                f"Optimizer selected a Skill outside this execution: "
                f"{proposal.selected_skill}"
            )
        spec = specs.get(proposal.selected_skill)
        if spec is None:
            raise RevisionRejected(
                f"Optimizer selected an unknown Skill: {proposal.selected_skill}"
            )
        _validate_edits(
            proposal.edits,
            spec.package_path,
            max_edits=max_edits,
            max_file_chars=max_file_chars,
            allow_new_files=allow_new_files,
        )

        self._revision_number += 1
        revision_id = f"revision-{self._revision_number:04d}"
        revision_dir = self.revisions_root / revision_id
        before_dir = revision_dir / "before"
        revision_dir.mkdir(parents=True)
        shutil.copytree(spec.package_path, before_dir)
        _write_json(revision_dir / "proposal.json", proposal.to_dict())
        before_hash = _tree_hash(spec.package_path)

        try:
            for edit in proposal.edits:
                target = _resolve_edit_target(spec.package_path, edit.path)
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write_text(target, edit.content)
                if target.suffix.lower() == ".py":
                    validate_python_file(target)
                    if spec.kind.value == "agentic":
                        validate_agentic_slot_references(
                            target.read_text(encoding="utf-8"),
                            allowed_slots={slot.name for slot in spec.slots},
                            label=str(target),
                        )
                    if spec.kind.value == "component":
                        validate_component_context_contract(
                            target.read_text(encoding="utf-8"),
                            label=str(target),
                        )
                        validate_component_output_contract(
                            target.read_text(encoding="utf-8"),
                            output_types={
                                capability.output_type for capability in spec.provides
                            },
                            label=str(target),
                        )
            discover_specs(self.skill_root)
            after_hash = _tree_hash(spec.package_path)
            if after_hash == before_hash:
                raise RevisionRejected("Optimizer proposal does not change the Skill")
            diff = _directory_diff(before_dir, spec.package_path)
            if not diff.strip():
                raise RevisionRejected("Optimizer proposal produced no textual diff")
        except Exception as exc:
            _restore_package(spec.package_path, before_dir)
            _write_json(
                revision_dir / "result.json",
                {
                    "status": "rejected",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            self.assert_source_unchanged()
            if isinstance(exc, RevisionRejected):
                raise
            raise RevisionRejected(
                f"Skill revision failed validation and was rolled back: {exc}"
            ) from exc

        diff_path = revision_dir / "changes.diff"
        diff_path.write_text(diff, encoding="utf-8")
        result = RevisionResult(
            revision_id=revision_id,
            selected_skill=proposal.selected_skill,
            changed_files=tuple(edit.path for edit in proposal.edits),
            before_hash=before_hash,
            after_hash=after_hash,
            revision_dir=revision_dir,
            diff_path=diff_path,
        )
        _write_json(revision_dir / "result.json", {"status": "accepted", **result.to_dict()})
        self.assert_source_unchanged()
        return result

    def assert_source_unchanged(self) -> None:
        """比较基础 Skill 仓库哈希，保证自优化从未写入原目录。"""
        current_hash = _tree_hash(self.source_root)
        if current_hash != self.source_hash:
            raise WorkspaceError(
                "Source Skill repository changed during self-optimization"
            )

    def rollback_revision(self, revision: RevisionResult, *, reason: str) -> None:
        """把已应用修订恢复为其 before 快照，并记录动态验证失败原因。"""
        specs = _specs_by_name(self.skill_root)
        spec = specs.get(revision.selected_skill)
        if spec is None:
            raise WorkspaceError(
                f"Cannot roll back missing Skill: {revision.selected_skill}"
            )
        before_dir = revision.revision_dir / "before"
        if not before_dir.is_dir():
            raise WorkspaceError(
                f"Revision before snapshot is missing: {before_dir}"
            )
        _restore_package(spec.package_path, before_dir)
        _write_json(
            revision.revision_dir / "result.json",
            {
                "status": "rolled_back",
                "reason": reason,
                **revision.to_dict(),
            },
        )
        self.assert_source_unchanged()


def _validate_edits(
    edits: Sequence[SkillEdit],
    package_path: Path,
    *,
    max_edits: int,
    max_file_chars: int,
    allow_new_files: bool,
) -> None:
    """检查修改数量、路径白名单、文件存在性和内容长度。"""
    if not edits:
        raise RevisionRejected("Optimizer must return at least one file edit")
    if len(edits) > max_edits:
        raise RevisionRejected(f"Optimizer returned more than {max_edits} edits")
    paths = [edit.path for edit in edits]
    if len(paths) != len(set(paths)):
        raise RevisionRejected("Optimizer returned duplicate edit paths")
    for edit in edits:
        if not isinstance(edit.content, str) or not edit.content.strip():
            raise RevisionRejected(f"Edit '{edit.path}' has empty content")
        if len(edit.content) > max_file_chars:
            raise RevisionRejected(
                f"Edit '{edit.path}' exceeds {max_file_chars} characters"
            )
        target = _resolve_edit_target(package_path, edit.path)
        if not target.exists() and not allow_new_files:
            raise RevisionRejected(f"Creating new files is disabled: {edit.path}")
        if target.exists() and not target.is_file():
            raise RevisionRejected(f"Edit target is not a file: {edit.path}")
        if target.exists():
            try:
                existing_content = target.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise RevisionRejected(f"Cannot read edit target: {edit.path}") from exc
            if len(existing_content) > max_file_chars:
                raise RevisionRejected(
                    f"Edit target was truncated in optimizer context: {edit.path}"
                )


def _resolve_edit_target(package_path: Path, relative_path: str) -> Path:
    """把模型返回的 POSIX 相对路径解析到 Skill 包内并阻止目录逃逸。"""
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise RevisionRejected("Edit path must be a non-empty string")
    if "\\" in relative_path:
        raise RevisionRejected("Edit paths must use forward slashes")
    pure_path = PurePosixPath(relative_path)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise RevisionRejected(f"Edit path escapes the Skill package: {relative_path}")
    normalized = pure_path.as_posix()
    if not _is_editable_path(normalized):
        raise RevisionRejected(f"Protected Skill file cannot be edited: {normalized}")
    target = (package_path / Path(*pure_path.parts)).resolve()
    package = package_path.resolve()
    if not target.is_relative_to(package):
        raise RevisionRejected(f"Edit path escapes the Skill package: {relative_path}")
    return target


def _is_editable_path(relative_path: str) -> bool:
    """判断文件是否属于稳定契约之外的 Skill 指令或实现区域。"""
    path = PurePosixPath(relative_path)
    if path.as_posix() == "SKILL.md":
        return True
    return (
        len(path.parts) >= 2
        and path.parts[0] in _EDITABLE_DIRECTORIES
        and path.suffix.lower() in _TEXT_SUFFIXES
    )


def _is_snapshot_file(path: Path, package_path: Path) -> bool:
    """筛选需要交给优化模型阅读的 Skill 文本文件。"""
    if not path.is_file() or path.is_symlink() or path.suffix.lower() not in _TEXT_SUFFIXES:
        return False
    relative = path.relative_to(package_path)
    return not any(part.startswith(".") or part == "__pycache__" for part in relative.parts)


def _specs_by_name(skill_root: Path) -> dict[str, RAGSkillSpec]:
    """发现工作副本中的全部 Skill 并按包名建立索引。"""
    return {spec.package_name: spec for spec in discover_specs(skill_root)}


def _unique_names(names: Sequence[str]) -> tuple[str, ...]:
    """按输入顺序去重并拒绝空 Skill 名称。"""
    unique = []
    seen = set()
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise WorkspaceError("Candidate Skill names must be non-empty strings")
        if name not in seen:
            seen.add(name)
            unique.append(name)
    if not unique:
        raise WorkspaceError("At least one candidate Skill is required")
    return tuple(unique)


def _reject_symlinks(root: Path) -> None:
    """拒绝源 Skill 中的符号链接，避免复制和哈希越过仓库边界。"""
    symlinks = [path for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise WorkspaceError(f"Source Skill root contains symlinks: {symlinks}")


def _tree_hash(root: Path) -> str:
    """根据相对路径和文件字节计算确定性的目录内容哈希。"""
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _directory_diff(before: Path, after: Path) -> str:
    """为修订前后的文本文件生成统一 diff。"""
    before_files = _normalized_diff_tree(before)
    after_files = _normalized_diff_tree(after)
    chunks = []
    for relative in sorted(set(before_files) | set(after_files)):
        chunks.extend(
            difflib.unified_diff(
                before_files.get(relative, "").splitlines(keepends=True),
                after_files.get(relative, "").splitlines(keepends=True),
                fromfile=f"before/{relative}",
                tofile=f"after/{relative}",
            )
        )
    return "".join(chunks)


def _normalized_diff_tree(root: Path) -> dict[str, str]:
    """统一换行风格和文件末尾换行，避免把格式噪声接受为 Skill 优化。"""
    return {
        relative: content.rstrip("\n") + "\n"
        for relative, content in _text_tree(root).items()
    }


def _text_tree(root: Path) -> dict[str, str]:
    """读取目录中参与版本比较的 UTF-8 文本文件。"""
    files = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = path.read_text(encoding="utf-8")
    return files


def _atomic_write_text(path: Path, content: str) -> None:
    """先写同目录临时文件，再原子替换目标文本文件。"""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            handle.write(content)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _restore_package(package_path: Path, backup_path: Path) -> None:
    """删除失败版本并从本轮只读快照恢复 Skill 包。"""
    if package_path.exists():
        shutil.rmtree(package_path)
    shutil.copytree(backup_path, package_path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """创建父目录并写入稳定格式的 UTF-8 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _default_run_id() -> str:
    """使用 UTC 时间和随机后缀生成不会覆盖旧实验的运行标识。"""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"
