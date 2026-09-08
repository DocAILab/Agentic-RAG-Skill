"""将单次 RAG 评估反馈交给 Policy Model，并解析单 Skill 修改提案。"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from framework import ModelClient

from .config import OptimizationSettings
from .contracts import OptimizationProposal, SkillEdit
from .python_validation import (
    PythonStaticValidationError,
    validate_agentic_slot_references,
    validate_component_output_contract,
    validate_python_source,
)
from .workspace import SkillWorkspace

_SELECTION_SYSTEM_PROMPT = """You are the Policy Model selecting one Skill to improve.
Use the evaluation and retrieval process to choose exactly one participating Skill.
Match the intended capability to the Skill kind and description. If the weakness is in
answer generation, select a generator Component, not an Agentic workflow. Return strict
JSON without Markdown fences or additional prose.

The Executor Model's parameters are frozen, but that does not make Generator or other
Skill files frozen. Every Skill in the candidate catalog may be selected, and its listed
editable_paths may be modified in the isolated working copy. Never redirect an edit to
the wrong capability merely because a Component calls the frozen Executor Model.

Schema:
{"selected_skill":"exact-candidate-name","rationale":"short reason grounded in the feedback"}
"""

_EDIT_SYSTEM_PROMPT = """You are editing one already-selected Skill in an isolated copy.
Only that Skill is visible. Improve it using the evaluation, retrieval process, and
selection rationale.

The Executor Model's parameters stay frozen. The selected Skill's files marked
editable=true are still editable, including a Generator Component that calls the
Executor. Change Skill instructions or implementation, never model parameters.

Rules:
1. Return one to three complete text-file replacements for the selected Skill only.
2. Edit only files marked editable=true. Never edit ragskill.yaml, package names,
   runtime IDs, capability contracts, slot contracts, or Python entry signatures.
3. Never edit a file marked truncated=true because its complete content is unavailable.
4. Preserve YAML frontmatter in SKILL.md with exactly name and description.
5. Make a focused, evidence-based change. Do not merely describe a change.
6. Return strict JSON without Markdown fences or additional prose.
7. For Python files, comments or docstring-only edits are not behavioral improvements.
8. Every returned file must differ from the corresponding snapshot. Do not include an
   unchanged file, and do not copy the snapshot back verbatim.
9. Before responding, compare each complete replacement against the visible original
   and verify the intended instruction or executable behavior actually changed.

Schema:
{"edits":[{"path":"SKILL.md or scripts/file.py","content":"complete replacement content"}]}
"""


class OptimizationError(ValueError):
    """表示优化提示超限或 Policy Model 返回了非法提案。"""


class OptimizationNoChange(OptimizationError):
    """表示模型完成选择但连续未提出任何实质性 Skill 修改。"""

    def __init__(self, selected_skill: str, rationale: str, attempts: int) -> None:
        """保存未修改的目标 Skill、选择理由和编辑尝试次数。"""
        self.selected_skill = selected_skill
        self.rationale = rationale
        self.attempts = attempts
        super().__init__(
            f"Policy Model proposed no substantive change to '{selected_skill}' "
            f"after {attempts} edit attempts"
        )


@dataclass(slots=True)
class OptimizationModelCall:
    """保存一次 Policy Model 尝试的完整提示、原始响应、提案和预检错误。"""

    stage: str
    system_prompt: str
    prompt: str
    raw_response: str | None = None
    proposal: OptimizationProposal | None = None
    error: str | None = None


@dataclass(slots=True)
class SkillOptimizer:
    """负责构造有界反馈提示、调用 Qwen 并验证输出结构。"""

    model: ModelClient
    settings: OptimizationSettings
    last_call: OptimizationModelCall | None = field(
        default=None,
        init=False,
        repr=False,
    )
    calls: list[OptimizationModelCall] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def propose(
        self,
        *,
        workspace: SkillWorkspace,
        candidate_names: Sequence[str],
        evaluation: Mapping[str, Any],
        retrieval_process: Mapping[str, Any],
    ) -> OptimizationProposal:
        """向优化模型提交一次评估和检索轨迹，并返回唯一 Skill 提案。"""
        self.last_call = None
        self.calls = []
        normalized_candidates = _unique_names(candidate_names)
        evaluation_json = _json_text(evaluation)
        process_json = _bounded_json(
            retrieval_process,
            self.settings.max_process_chars,
        )
        catalog_snapshots = workspace.candidate_snapshots(
            normalized_candidates,
            max_file_chars=1_000,
        )
        selection_prompt = _build_selection_prompt(
            evaluation_json=evaluation_json,
            process_json=process_json,
            snapshots=catalog_snapshots,
        )
        if len(selection_prompt) > self.settings.max_prompt_chars:
            raise OptimizationError(
                "Optimizer selection prompt exceeds max_prompt_chars"
            )
        selection_call = OptimizationModelCall(
            stage="selection",
            system_prompt=_SELECTION_SYSTEM_PROMPT,
            prompt=selection_prompt,
        )
        self.calls.append(selection_call)
        self.last_call = selection_call
        selection_response = self.model.generate(
            selection_prompt,
            system=_SELECTION_SYSTEM_PROMPT,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )
        selection_call.raw_response = selection_response
        try:
            selected_skill, rationale = _parse_skill_selection(
                selection_response,
                candidate_names=normalized_candidates,
            )
        except OptimizationError as exc:
            selection_call.error = str(exc)
            raise

        file_limit = self.settings.max_skill_file_chars
        edit_prompt = ""
        selected_snapshot: Mapping[str, Any] | None = None
        while file_limit >= 1_000:
            selected_snapshot = workspace.candidate_snapshots(
                (selected_skill,),
                max_file_chars=file_limit,
            )[0]
            edit_prompt = _build_edit_prompt(
                evaluation_json=evaluation_json,
                process_json=process_json,
                selected_skill=selected_skill,
                rationale=rationale,
                snapshot=selected_snapshot,
            )
            if len(edit_prompt) <= self.settings.max_prompt_chars:
                break
            file_limit //= 2
        if (
            selected_snapshot is None
            or not edit_prompt
            or len(edit_prompt) > self.settings.max_prompt_chars
        ):
            raise OptimizationError(
                "Optimizer edit prompt exceeds max_prompt_chars after bounded truncation"
            )

        current_prompt = edit_prompt
        last_error: OptimizationError | None = None
        edit_errors: list[OptimizationError] = []
        for attempt_index in range(self.settings.max_proposal_attempts):
            model_call = OptimizationModelCall(
                stage="editing",
                system_prompt=_EDIT_SYSTEM_PROMPT,
                prompt=current_prompt,
            )
            self.calls.append(model_call)
            self.last_call = model_call
            response = self.model.generate(
                current_prompt,
                system=_EDIT_SYSTEM_PROMPT,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )
            model_call.raw_response = response
            try:
                edits = _parse_skill_edits(
                    response,
                    max_edits=self.settings.max_edits,
                )
                proposal = OptimizationProposal(
                    selected_skill=selected_skill,
                    rationale=rationale,
                    edits=edits,
                )
                model_call.proposal = proposal
                _preflight_proposal(
                    proposal,
                    snapshots=(selected_snapshot,),
                    allow_new_files=self.settings.allow_new_files,
                )
            except OptimizationError as exc:
                model_call.error = str(exc)
                last_error = exc
                edit_errors.append(exc)
                if attempt_index + 1 >= self.settings.max_proposal_attempts:
                    break
                current_prompt = _repair_prompt(
                    current_prompt,
                    error=str(exc),
                    attempt=attempt_index + 1,
                    max_chars=self.settings.max_prompt_chars,
                )
                continue
            return proposal
        if last_error is None:
            raise OptimizationError("Optimizer did not produce a proposal")
        if edit_errors and all(_is_no_change_error(error) for error in edit_errors):
            raise OptimizationNoChange(
                selected_skill,
                rationale,
                len(edit_errors),
            ) from last_error
        raise OptimizationError(
            "Optimizer proposal remained invalid after "
            f"{len(edit_errors)} edit attempts: "
            f"{last_error}"
        ) from last_error


def parse_optimization_proposal(
    response: str,
    *,
    candidate_names: Sequence[str],
    max_edits: int,
) -> OptimizationProposal:
    """解析并严格校验优化模型返回的 JSON 提案。"""
    if not isinstance(response, str) or not response.strip():
        raise OptimizationError("Optimizer response is empty")
    payload = _parse_json_object(response)
    required_fields = {"selected_skill", "rationale", "edits"}
    if set(payload) != required_fields:
        raise OptimizationError(
            "Optimizer response must contain exactly selected_skill, rationale, and edits"
        )
    selected_skill = _required_text(payload, "selected_skill")
    allowed = set(_unique_names(candidate_names))
    if selected_skill not in allowed:
        raise OptimizationError(
            f"Optimizer selected a Skill outside this execution: {selected_skill}"
        )
    rationale = _required_text(payload, "rationale")
    edits_value = payload.get("edits")
    if isinstance(edits_value, (str, bytes, bytearray)) or not isinstance(
        edits_value,
        Sequence,
    ):
        raise OptimizationError("Optimizer edits must be a list")
    if not edits_value or len(edits_value) > max_edits:
        raise OptimizationError(
            f"Optimizer edits must contain between 1 and {max_edits} items"
        )
    edits = []
    for index, value in enumerate(edits_value):
        if not isinstance(value, Mapping) or set(value) != {"path", "content"}:
            raise OptimizationError(
                f"Optimizer edits[{index}] must contain exactly path and content"
            )
        edits.append(
            SkillEdit(
                path=_required_text(value, "path"),
                content=_required_text(value, "content", strip=False),
            )
        )
    return OptimizationProposal(
        selected_skill=selected_skill,
        rationale=rationale,
        edits=tuple(edits),
    )


def _build_selection_prompt(
    *,
    evaluation_json: str,
    process_json: str,
    snapshots: Sequence[Mapping[str, Any]],
) -> str:
    """仅拼接候选摘要供第一阶段选择，不暴露任何 Skill 文件正文。"""
    catalog = [
        {
            "name": snapshot.get("name"),
            "description": snapshot.get("description"),
            "kind": snapshot.get("kind"),
            "allowed_component_slots": snapshot.get("allowed_component_slots"),
            "provided_capabilities": snapshot.get("provided_capabilities"),
            "editable_paths": [
                file.get("path")
                for file in snapshot.get("files", ())
                if file.get("editable") and not file.get("truncated")
            ],
        }
        for snapshot in snapshots
    ]
    return (
        "EVALUATION FEEDBACK:\n"
        f"{evaluation_json}\n\n"
        "RETRIEVAL PROCESS:\n"
        f"{process_json}\n\n"
        "CANDIDATE SKILL CATALOG:\n"
        f"{_json_text(catalog)}\n\n"
        "Choose the Skill whose owned behavior matches the diagnosed weakness. Return "
        "the strict selection JSON now."
    )


def _build_edit_prompt(
    *,
    evaluation_json: str,
    process_json: str,
    selected_skill: str,
    rationale: str,
    snapshot: Mapping[str, Any],
) -> str:
    """只向第二阶段暴露已选 Skill 的完整快照和精确可编辑路径。"""
    editable_paths = [
        file.get("path")
        for file in snapshot.get("files", ())
        if file.get("editable") and not file.get("truncated")
    ]
    return (
        "EVALUATION FEEDBACK:\n"
        f"{evaluation_json}\n\n"
        "RETRIEVAL PROCESS:\n"
        f"{process_json}\n\n"
        "FIXED SELECTED SKILL:\n"
        f"{_json_text({'selected_skill': selected_skill, 'rationale': rationale})}\n\n"
        "SELECTED SKILL SNAPSHOT:\n"
        f"{_json_text(snapshot)}\n\n"
        "EXACT EDITABLE PATHS:\n"
        f"{_json_text(editable_paths)}\n\n"
        "Only edit this fixed selected Skill. Use exact existing paths unless creation "
        "is explicitly allowed, preserve its capability and slot contracts, and make a "
        "substantive change. Return the strict edits JSON now."
    )


def _parse_skill_selection(
    response: str,
    *,
    candidate_names: Sequence[str],
) -> tuple[str, str]:
    """解析第一阶段唯一 Skill 选择并验证其属于本轮候选。"""
    payload = _parse_json_object(response)
    if set(payload) != {"selected_skill", "rationale"}:
        raise OptimizationError(
            "Skill selection must contain exactly selected_skill and rationale"
        )
    selected_skill = _required_text(payload, "selected_skill")
    if selected_skill not in set(candidate_names):
        raise OptimizationError(
            f"Optimizer selected a Skill outside this execution: {selected_skill}"
        )
    return selected_skill, _required_text(payload, "rationale")


def _parse_skill_edits(
    response: str,
    *,
    max_edits: int,
) -> tuple[SkillEdit, ...]:
    """解析第二阶段只包含完整文件替换的严格 JSON。"""
    payload = _parse_json_object(response)
    if set(payload) != {"edits"}:
        raise OptimizationError("Skill edit response must contain exactly edits")
    edits_value = payload.get("edits")
    if isinstance(edits_value, (str, bytes, bytearray)) or not isinstance(
        edits_value,
        Sequence,
    ):
        raise OptimizationError("Optimizer edits must be a list")
    if not edits_value or len(edits_value) > max_edits:
        raise OptimizationError(
            f"Optimizer edits must contain between 1 and {max_edits} items"
        )
    edits = []
    for index, value in enumerate(edits_value):
        if not isinstance(value, Mapping) or set(value) != {"path", "content"}:
            raise OptimizationError(
                f"Optimizer edits[{index}] must contain exactly path and content"
            )
        edits.append(
            SkillEdit(
                path=_required_text(value, "path"),
                content=_required_text(value, "content", strip=False),
            )
        )
    return tuple(edits)


def _preflight_proposal(
    proposal: OptimizationProposal,
    *,
    snapshots: Sequence[Mapping[str, Any]],
    allow_new_files: bool,
) -> None:
    """在写工作副本前校验文件归属、可编辑性和修改是否具有实质差异。"""
    selected = next(
        (
            snapshot
            for snapshot in snapshots
            if snapshot.get("name") == proposal.selected_skill
        ),
        None,
    )
    if selected is None:
        raise OptimizationError(
            f"No snapshot is available for selected Skill: {proposal.selected_skill}"
        )
    files = {
        str(file.get("path")): file
        for file in selected.get("files", ())
        if isinstance(file, Mapping)
    }
    valid_paths = sorted(
        path
        for path, file in files.items()
        if file.get("editable") and not file.get("truncated")
    )
    substantive_change = False
    for edit in proposal.edits:
        current = files.get(edit.path)
        if current is None:
            if allow_new_files:
                substantive_change = True
                continue
            alternative_owners = sorted(
                str(snapshot.get("name"))
                for snapshot in snapshots
                if snapshot.get("name") != proposal.selected_skill
                and any(
                    isinstance(file, Mapping)
                    and file.get("path") == edit.path
                    and file.get("editable")
                    and not file.get("truncated")
                    for file in snapshot.get("files", ())
                )
            )
            alternative_text = (
                f" Other candidate Skills owning this path: {alternative_owners}."
                if alternative_owners
                else ""
            )
            raise OptimizationError(
                f"'{edit.path}' does not exist in selected Skill "
                f"'{proposal.selected_skill}'. Exact editable paths: {valid_paths}."
                f"{alternative_text} Select the Skill that owns the code you intend "
                "to change; do not copy Component code into an Agentic Skill."
            )
        if not current.get("editable"):
            raise OptimizationError(
                f"'{edit.path}' is protected in selected Skill "
                f"'{proposal.selected_skill}'. Exact editable paths: {valid_paths}"
            )
        if current.get("truncated"):
            raise OptimizationError(
                f"'{edit.path}' was truncated and cannot be safely replaced"
            )
        existing_content = str(current.get("content", ""))
        if _has_substantive_change(edit.path, existing_content, edit.content):
            if selected.get("kind") == "agentic" and PurePosixPath(edit.path).suffix == ".py":
                try:
                    validate_agentic_slot_references(
                        edit.content,
                        allowed_slots={
                            str(slot)
                            for slot in selected.get("allowed_component_slots", ())
                        },
                        label=edit.path,
                    )
                except PythonStaticValidationError as exc:
                    raise OptimizationError(str(exc)) from exc
            if selected.get("kind") == "component" and PurePosixPath(edit.path).suffix == ".py":
                try:
                    validate_component_output_contract(
                        edit.content,
                        output_types={
                            str(capability.get("output_type"))
                            for capability in selected.get("provided_capabilities", ())
                            if isinstance(capability, Mapping)
                        },
                        label=edit.path,
                    )
                except PythonStaticValidationError as exc:
                    raise OptimizationError(str(exc)) from exc
            substantive_change = True
    if not substantive_change:
        raise OptimizationError(
            f"Proposal for '{proposal.selected_skill}' does not make a substantive "
            "change. Do not resend the current file or alter only comments/docstrings; "
            "change executable behavior, update SKILL.md instructions, or choose a "
            "different candidate Skill consistent with the rationale."
        )


def _repair_prompt(
    current_prompt: str,
    *,
    error: str,
    attempt: int,
    max_chars: int,
) -> str:
    """累积每次预检错误，阻止模型在多个已知错误之间来回摆动。"""
    repair = (
        f"\n\nPROPOSAL ATTEMPT {attempt} REJECTED BEFORE APPLICATION:\n"
        f"{error}\n"
        "Do not repeat this proposal or any earlier rejected proposal. Return corrected "
        "strict JSON, use an exact editable path, and make a substantive change. Every "
        "returned file must differ from the visible snapshot; omit unchanged files."
    )
    available = max_chars - len(current_prompt)
    if available <= 0:
        return current_prompt
    return current_prompt + repair[:available]


def _is_no_change_error(error: OptimizationError) -> bool:
    """判断一次预检失败是否仅由候选文件没有实质差异导致。"""
    return "does not make a substantive change" in str(error)


def _normalize_comparable_text(content: str) -> str:
    """统一行尾和文件末尾空行，以识别换行噪声造成的伪修改。"""
    return content.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def _has_substantive_change(path: str, before: str, after: str) -> bool:
    """判断文本是否改变 Skill 指令，或在 Python 文件中真正改变可执行 AST。"""
    normalized_before = _normalize_comparable_text(before)
    normalized_after = _normalize_comparable_text(after)
    if normalized_before == normalized_after:
        return False
    if PurePosixPath(path).suffix.lower() != ".py":
        return True
    try:
        before_tree = ast.parse(normalized_before)
        after_tree = ast.parse(normalized_after)
    except SyntaxError as exc:
        raise OptimizationError(
            f"Python replacement for '{path}' is syntactically invalid: {exc.msg}"
        ) from exc
    _remove_docstrings(before_tree)
    _remove_docstrings(after_tree)
    changed = ast.dump(before_tree, include_attributes=False) != ast.dump(
        after_tree,
        include_attributes=False,
    )
    if changed:
        try:
            validate_python_source(after, label=path)
        except PythonStaticValidationError as exc:
            raise OptimizationError(str(exc)) from exc
    return changed


def _remove_docstrings(node: ast.AST) -> None:
    """递归移除模块、类和函数首语句中的 docstring，供行为 AST 比较使用。"""
    for child in ast.walk(node):
        if not isinstance(
            child,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        body = child.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]


def _bounded_json(payload: Mapping[str, Any], max_chars: int) -> str:
    """将检索过程编码为 JSON，超限时保留前缀并显式标记截断。"""
    encoded = _json_text(payload)
    if len(encoded) <= max_chars:
        return encoded
    return _json_text(
        {
            "truncated": True,
            "original_characters": len(encoded),
            "json_prefix": encoded[:max_chars],
        }
    )


def _json_text(payload: Any) -> str:
    """生成便于模型阅读且键顺序稳定的 JSON 文本。"""
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
    except (TypeError, ValueError) as exc:
        raise OptimizationError("Optimizer input must be JSON-compatible") from exc


def _parse_json_object(text: str) -> Mapping[str, Any]:
    """兼容代码围栏或少量前后文本并提取唯一 JSON 对象。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise OptimizationError("Optimizer response does not contain JSON") from None
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise OptimizationError("Optimizer response contains invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise OptimizationError("Optimizer response must be a JSON object")
    return dict(payload)


def _required_text(
    payload: Mapping[str, Any],
    key: str,
    *,
    strip: bool = True,
) -> str:
    """读取优化输出中的必需字符串，并可保留完整文件空白。"""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise OptimizationError(f"Optimizer field '{key}' must be a non-empty string")
    return value.strip() if strip else value


def _unique_names(names: Sequence[str]) -> tuple[str, ...]:
    """按出现顺序规范化候选名称并拒绝空集合。"""
    unique = []
    seen = set()
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise OptimizationError("Candidate Skill names must be non-empty strings")
        if name not in seen:
            seen.add(name)
            unique.append(name)
    if not unique:
        raise OptimizationError("At least one candidate Skill is required")
    return tuple(unique)
