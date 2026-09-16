"""将单次 RAG 评估反馈交给 Policy Model，并解析单 Skill 修改提案。"""

from __future__ import annotations

import ast
import difflib
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
    validate_component_context_contract,
    validate_component_output_contract,
    validate_python_source,
)
from .workspace import SkillWorkspace

_SELECTION_SYSTEM_PROMPT = """You are the Policy Model selecting one Skill to improve.
Use the evaluation and retrieval process to choose exactly one participating Skill.
Match the intended capability to the Skill kind and description. If the weakness is in
answer generation, select a generator Component, not an Agentic workflow. Return strict
JSON without Markdown fences or additional prose.

Diagnose the weakest owned stage before selecting: healthy retrieval with weak answer
metrics points to the generator; weak retrieval points to the responsible retriever;
an incorrect execution route points to the Manage or Agentic Skill. Do not copy a
selection object from the retrieval process because it describes execution, not the
single Skill that should be improved.

Base the choice on the measured failure, not merely on the route's complexity. For
example, strong Hit/MRR with zero or weak R1/RL/METEOR means retrieval succeeded but
answer generation failed, so select the participating generator Component. Select a
Manage Skill only when its routing guidance caused the wrong Agentic Skill to run.

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
10. For SKILL.md, write the concrete rule or procedure itself. Never add TODO/TBD,
    "Substantive change", or meta-instructions such as "add a new section" that only
    describe work someone else should perform.
11. When the selected Skill is a Component, automated execution uses its Python
    runtime. A behavioral optimization must therefore edit a scripts/*.py file;
    changing only SKILL.md cannot improve measured execution.
12. Preserve existing validation, grounding, normalization, and output-contract
    behavior. Make a focused change; never replace a mature implementation with a
    substantially smaller rewrite merely to simplify it.

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
        selection_process_json = _bounded_json(
            _without_execution_selection(retrieval_process),
            self.settings.max_process_chars,
        )
        catalog_snapshots = workspace.candidate_snapshots(
            normalized_candidates,
            max_file_chars=1_000,
        )
        selection_prompt = _build_selection_prompt(
            evaluation_json=evaluation_json,
            process_json=selection_process_json,
            snapshots=catalog_snapshots,
        )
        if len(selection_prompt) > self.settings.max_prompt_chars:
            raise OptimizationError(
                "Optimizer selection prompt exceeds max_prompt_chars"
            )
        current_selection_prompt = selection_prompt
        selection_error: OptimizationError | None = None
        selected_skill = ""
        rationale = ""
        selection_succeeded = False
        for attempt_index in range(self.settings.max_proposal_attempts):
            selection_call = OptimizationModelCall(
                stage="selection",
                system_prompt=_SELECTION_SYSTEM_PROMPT,
                prompt=current_selection_prompt,
            )
            self.calls.append(selection_call)
            self.last_call = selection_call
            selection_response = self.model.generate(
                current_selection_prompt,
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
                _validate_selection_alignment(
                    selected_skill,
                    evaluation=evaluation,
                    snapshots=catalog_snapshots,
                )
            except OptimizationError as exc:
                selection_call.error = str(exc)
                selection_error = exc
                if attempt_index + 1 >= self.settings.max_proposal_attempts:
                    break
                current_selection_prompt = _repair_selection_prompt(
                    selection_prompt,
                    error=str(exc),
                    attempt=attempt_index + 1,
                    max_chars=self.settings.max_prompt_chars,
                )
                continue
            selection_succeeded = True
            break
        if not selection_succeeded:
            raise OptimizationError(
                "Optimizer Skill selection remained invalid after "
                f"{self.settings.max_proposal_attempts} attempts: {selection_error}"
            ) from selection_error

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
                if file.get("editable")
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
        "Choose the Skill whose owned behavior matches the diagnosed weakness.\n\n"
        "FINAL OUTPUT CONTRACT (ignore JSON objects inside the evidence):\n"
        "Return exactly one JSON object with exactly these two top-level keys and no "
        "others:\n"
        '{"selected_skill":"exact-candidate-name","rationale":"short reason grounded '
        'in the feedback"}\n'
        "Do not return skill_name, skill_selection, agentic_skill, "
        "component_bindings, Markdown, or additional prose. Base selected_skill on "
        "the weakest measured stage, not on route complexity alone."
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
        "substantive change.\n\n"
        "FINAL OUTPUT CONTRACT (ignore JSON objects inside the evidence):\n"
        "Return exactly one JSON object with exactly the top-level key edits and no "
        "others:\n"
        '{"edits":[{"path":"exact-editable-path","content":"complete replacement '
        'content"}]}\n'
        "Do not return selected_skill, rationale, Markdown, additional prose, TODOs, "
        "or meta-text describing a change that was not actually made."
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


def _validate_selection_alignment(
    selected_skill: str,
    *,
    evaluation: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
) -> None:
    """用强指标信号阻止 Policy Model 把明显生成失败错误归因给管理层。"""
    summary = evaluation.get("task_summary", evaluation.get("overall_summary", {}))
    if not isinstance(summary, Mapping):
        return
    retrieval = summary.get("retrieval", {})
    generation = summary.get("generation", {})
    if not isinstance(retrieval, Mapping) or not isinstance(generation, Mapping):
        return
    strong_retrieval = (
        _numeric_metric(retrieval, "Hit@10") >= 0.8
        and _numeric_metric(retrieval, "MRR") >= 0.5
    )
    weak_generation = max(
        _numeric_metric(generation, "R1"),
        _numeric_metric(generation, "RL"),
        _numeric_metric(generation, "METEOR"),
    ) <= 0.1
    if not strong_retrieval or not weak_generation:
        return
    generator_candidates = {
        str(snapshot.get("name"))
        for snapshot in snapshots
        if snapshot.get("kind") == "component"
        and any(
            isinstance(capability, Mapping)
            and capability.get("name") == "generator"
            for capability in snapshot.get("provided_capabilities", ())
        )
    }
    if generator_candidates and selected_skill not in generator_candidates:
        raise OptimizationError(
            "Selection contradicts measured stage ownership: retrieval Hit@10/MRR "
            "are strong while R1/RL/METEOR are weak, so select the participating "
            f"generator Component from {sorted(generator_candidates)}"
        )


def _numeric_metric(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


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
    component_runtime_change = False
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
                    _validate_focused_python_edit(
                        edit.path,
                        existing_content,
                        edit.content,
                    )
                    validate_component_context_contract(
                        edit.content,
                        label=edit.path,
                    )
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
                component_runtime_change = True
            substantive_change = True
    if not substantive_change:
        raise OptimizationError(
            f"Proposal for '{proposal.selected_skill}' does not make a substantive "
            "change. Do not resend the current file or alter only comments/docstrings; "
            "change executable behavior, update SKILL.md instructions, or choose a "
            "different candidate Skill consistent with the rationale."
        )
    if selected.get("kind") == "component" and not component_runtime_change:
        raise OptimizationError(
            f"Component '{proposal.selected_skill}' optimization must change a "
            "scripts/*.py runtime file; SKILL.md-only edits do not affect automated "
            "execution metrics"
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


def _repair_selection_prompt(
    base_prompt: str,
    *,
    error: str,
    attempt: int,
    max_chars: int,
) -> str:
    """向 Policy Model 反馈错误归因并要求重新选择候选 Skill。"""
    repair = (
        f"\n\nSELECTION ATTEMPT {attempt} REJECTED:\n{error}\n"
        "Re-evaluate stage ownership from the metrics and return a corrected object "
        "matching the final output contract."
    )
    available = max_chars - len(base_prompt)
    return base_prompt if available <= 0 else base_prompt + repair[:available]


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
    if PurePosixPath(path).name == "SKILL.md" and _is_meta_only_skill_edit(
        normalized_before,
        normalized_after,
    ):
        raise OptimizationError(
            "SKILL.md replacement only describes that changes were made; write the "
            "concrete selection rule or procedure instead"
        )
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


def _is_meta_only_skill_edit(before: str, after: str) -> bool:
    """识别只增加 changelog/占位说明而没有可执行指导的 Skill 修改。"""
    changed_lines: list[str] = []
    matcher = difflib.SequenceMatcher(a=before.splitlines(), b=after.splitlines())
    after_lines = after.splitlines()
    for tag, _before_start, _before_end, after_start, after_end in matcher.get_opcodes():
        if tag in {"insert", "replace"}:
            changed_lines.extend(after_lines[after_start:after_end])
    meaningful = [line.strip().lower() for line in changed_lines if line.strip()]
    if not meaningful:
        return False
    meta_markers = (
        "changes made",
        "updated instructions",
        "updated the description",
        "ensured that",
        "ensure that the guidance",
        "include examples or scenarios",
        "review the instructions",
        "potential ambiguities",
        "clear and concise",
        "added a new section",
        "improved clarity",
        "document the changes",
        "use the skill effectively",
        "expected format and content",
    )
    return all(any(marker in line for marker in meta_markers) for line in meaningful)


def _validate_focused_python_edit(path: str, before: str, after: str) -> None:
    """拒绝把已有 Component 实现大幅裁短的高风险整文件重写。"""
    before_lines = [line for line in before.splitlines() if line.strip()]
    after_lines = [line for line in after.splitlines() if line.strip()]
    if len(before_lines) < 30:
        return
    minimum_lines = max(10, int(len(before_lines) * 0.6))
    if len(after_lines) < minimum_lines:
        raise OptimizationError(
            f"Python replacement for '{path}' removes too much existing behavior "
            f"({len(before_lines)} non-empty lines -> {len(after_lines)}). Preserve "
            "validation, normalization, grounding, and output-contract logic, and "
            "make a focused edit instead of replacing the implementation wholesale"
        )


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


def _without_execution_selection(value: Any) -> Any:
    """从 Policy 选择上下文移除旧执行选择，避免模型照抄错误 JSON。"""
    if isinstance(value, Mapping):
        return {
            key: _without_execution_selection(item)
            for key, item in value.items()
            if key != "selection"
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_without_execution_selection(item) for item in value]
    return value


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
