"""通过大模型执行 Manage、Agentic 和 Component 三级选择。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import ModelClient
from .spec import (
    RAGSkillSpec,
    SkillKind,
    SlotSpec,
    binding_requirement_errors,
    discover_specs,
)

SELECTION_MAX_TOKENS = 8192


class SelectionError(ValueError):
    """表示模型输出或候选绑定不符合分级选择契约。"""


@dataclass(frozen=True, slots=True)
class ManageStageResult:
    """第一步输出，仅包含 Manage Skill 生成的 Agentic 选择指导。"""

    manage_skill: str
    guidance: str
    reason: str


@dataclass(frozen=True, slots=True)
class AgenticStageResult:
    """第二步输出，包含所选 Agentic Skill 及按需加载的完整正文。"""

    spec: RAGSkillSpec
    instructions: str
    reason: str
    advertised_skills: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ComponentStageResult:
    """第三步输出，包含槽位绑定和按需加载的 Component 正文。"""

    agentic_skill: str
    bindings: Mapping[str, tuple[str, ...]]
    instructions: Mapping[str, str]
    reason: str


@dataclass(frozen=True, slots=True)
class RAGSelectionPlan:
    """记录三层选择结果，供后续编译器绑定 workflow。"""

    manage_skill: str
    manage_guidance: str
    manage_reason: str
    agentic_skill: str
    agentic_reason: str
    component_bindings: Mapping[str, tuple[str, ...]]
    component_reason: str

    def to_dict(self) -> dict[str, Any]:
        """将不可变选择计划转换为便于持久化的 JSON-compatible 字典。"""
        return {
            "manage_skill": self.manage_skill,
            "manage_guidance": self.manage_guidance,
            "manage_reason": self.manage_reason,
            "agentic_skill": self.agentic_skill,
            "agentic_reason": self.agentic_reason,
            "component_bindings": {
                slot: list(names) for slot, names in self.component_bindings.items()
            },
            "component_reason": self.component_reason,
        }


def run_manage_stage(
    request: Mapping[str, Any],
    *,
    model: ModelClient,
    skill_root: str | Path,
    manage_skill: str = "manage-rag-default",
) -> ManageStageResult:
    """第一步只加载 Manage Skill，并调用模型生成 Agentic 选择指导。"""
    specs = discover_specs(skill_root, validate_runtime=False)
    manage = _find_spec(specs, manage_skill, expected_kind=SkillKind.MANAGE)
    payload = _call_json_model(
        model,
        system=(
            "You are the frozen Executor Model at the Manage stage. "
            "Follow only the loaded Manage Skill and return strict JSON."
        ),
        prompt=(
            "LOADED MANAGE SKILL:\n"
            f"{_read_skill_document(manage)}\n\n"
            "RAG REQUEST:\n"
            f"{_encode_request(request)}\n\n"
            "Analyze the request. Return "
            '{"agentic_selection_guidance":"...","reason":"..."}.'
        ),
    )
    return ManageStageResult(
        manage_skill=manage.package_name,
        guidance=_required_text(payload, "agentic_selection_guidance"),
        reason=_optional_text(payload, "reason"),
    )


def select_agentic_skill(
    request: Mapping[str, Any],
    *,
    manage_result: ManageStageResult,
    model: ModelClient,
    skill_root: str | Path,
) -> AgenticStageResult:
    """第二步仅广告 Agentic 候选，调用模型选择后再加载所选正文。"""
    specs = discover_specs(skill_root, validate_runtime=False)
    agentic_specs = tuple(spec for spec in specs if spec.kind is SkillKind.AGENTIC)
    if not agentic_specs:
        raise SelectionError("No Agentic Skill candidates are available")
    payload = _call_json_model(
        model,
        system=(
            "You are the frozen Executor Model at the Agentic selection stage. "
            "Choose exactly one advertised Agentic Skill and return strict JSON."
        ),
        prompt=(
            "MANAGE GUIDANCE:\n"
            f"{manage_result.guidance}\n\n"
            "RAG REQUEST:\n"
            f"{_encode_request(request)}\n\n"
            "AGENTIC SKILL ADVERTISEMENTS:\n"
            f"{_advertisements(agentic_specs)}\n\n"
            "Return "
            '{"selected_agentic_skill":"exact-name","reason":"..."}.'
        ),
    )
    selected_name = _required_text(payload, "selected_agentic_skill")
    selected = _find_spec(
        agentic_specs,
        selected_name,
        expected_kind=SkillKind.AGENTIC,
    )
    return AgenticStageResult(
        spec=selected,
        instructions=_read_skill_document(selected),
        reason=_optional_text(payload, "reason"),
        advertised_skills=tuple(spec.package_name for spec in agentic_specs),
    )


def select_component_skills(
    request: Mapping[str, Any],
    *,
    agentic_result: AgenticStageResult,
    model: ModelClient,
    skill_root: str | Path,
) -> ComponentStageResult:
    """第三步按 Agentic 槽位广告兼容组件，选择后再加载对应正文。"""
    specs = discover_specs(skill_root, validate_runtime=False)
    agentic = _find_spec(
        specs,
        agentic_result.spec.package_name,
        expected_kind=SkillKind.AGENTIC,
    )
    component_specs = tuple(spec for spec in specs if spec.kind is SkillKind.COMPONENT)
    slot_candidates = {
        slot.name: _compatible_components(slot, component_specs)
        for slot in agentic.slots
    }
    for slot in agentic.slots:
        if len(slot_candidates[slot.name]) < slot.min_count:
            raise SelectionError(
                f"Slot '{slot.name}' requires {slot.min_count} compatible Components, "
                f"but only {len(slot_candidates[slot.name])} are available"
            )

    payload = _call_json_model(
        model,
        system=(
            "You are the frozen Executor Model at the Component selection stage. "
            "Follow the selected Agentic Skill, respect every slot cardinality, "
            "and return strict JSON."
        ),
        prompt=(
            "SELECTED AGENTIC SKILL:\n"
            f"{agentic_result.instructions}\n\n"
            "RAG REQUEST:\n"
            f"{_encode_request(request)}\n\n"
            "COMPATIBLE COMPONENT ADVERTISEMENTS BY SLOT:\n"
            f"{_slot_advertisements(agentic.slots, slot_candidates)}\n\n"
            "Return "
            '{"component_bindings":{"slot":["exact-component-name"]},'
            '"reason":"..."}. Include every advertised slot; use [] for an '
            "unused optional slot."
        ),
    )
    bindings = _validate_bindings(
        payload.get("component_bindings"),
        agentic.slots,
        slot_candidates,
    )
    component_by_name = {
        spec.package_name: spec for spec in component_specs
    }
    requirement_errors = binding_requirement_errors(
        agentic,
        bindings,
        component_by_name,
    )
    if requirement_errors:
        raise SelectionError(requirement_errors[0])
    selected_names = {name for names in bindings.values() for name in names}
    instructions = {
        name: _read_skill_document(
            _find_spec(component_specs, name, expected_kind=SkillKind.COMPONENT)
        )
        for name in sorted(selected_names)
    }
    return ComponentStageResult(
        agentic_skill=agentic.package_name,
        bindings=bindings,
        instructions=instructions,
        reason=_optional_text(payload, "reason"),
    )


def select_rag_plan(
    request: Mapping[str, Any],
    *,
    model: ModelClient,
    skill_root: str | Path,
    manage_skill: str = "manage-rag-default",
) -> RAGSelectionPlan:
    """调用模型完成三级渐进披露，并返回经过契约校验的 Skill 计划。"""
    manage_result = run_manage_stage(
        request,
        model=model,
        skill_root=skill_root,
        manage_skill=manage_skill,
    )
    agentic_result = select_agentic_skill(
        request,
        manage_result=manage_result,
        model=model,
        skill_root=skill_root,
    )
    component_result = select_component_skills(
        request,
        agentic_result=agentic_result,
        model=model,
        skill_root=skill_root,
    )

    return RAGSelectionPlan(
        manage_skill=manage_result.manage_skill,
        manage_guidance=manage_result.guidance,
        manage_reason=manage_result.reason,
        agentic_skill=agentic_result.spec.package_name,
        agentic_reason=agentic_result.reason,
        component_bindings=component_result.bindings,
        component_reason=component_result.reason,
    )


def _find_spec(
    specs: Sequence[RAGSkillSpec],
    name: str,
    *,
    expected_kind: SkillKind,
) -> RAGSkillSpec:
    """按名称查找指定层级的 Skill，并拒绝未知或跨层选择。"""
    for spec in specs:
        if spec.package_name == name and spec.kind is expected_kind:
            return spec
    raise SelectionError(f"Unknown {expected_kind.value} Skill: {name}")


def _read_skill_document(spec: RAGSkillSpec) -> str:
    """仅在层级选中后读取对应 Skill 的完整指导正文。"""
    path = spec.package_path / "SKILL.md"
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SelectionError(f"Cannot read Skill document: {path}") from exc


def _encode_request(request: Mapping[str, Any]) -> str:
    """将不含完整语料正文的 RAG 选择摘要编码为稳定 JSON。"""
    selection_request = _build_selection_request(request)
    try:
        return json.dumps(
            selection_request,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise SelectionError("RAG request must be JSON-compatible") from exc


def _build_selection_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """保留选择所需任务参数，并把完整 documents 压缩为语料统计。"""
    summary = {key: value for key, value in request.items() if key != "documents"}
    if "documents" not in request:
        return summary

    documents = request["documents"]
    if isinstance(documents, (str, bytes, bytearray)) or not isinstance(
        documents,
        Sequence,
    ):
        raise SelectionError("RAG request documents must be a sequence")

    document_fields: set[str] = set()
    text_lengths: list[int] = []
    for document in documents:
        if not isinstance(document, Mapping):
            raise SelectionError("Every RAG document must be a mapping")
        document_fields.update(str(key) for key in document)
        text_lengths.append(len(str(document.get("text", ""))))

    document_count = len(documents)
    summary["corpus"] = {
        "document_count": document_count,
        "document_fields": sorted(document_fields),
        "average_text_characters": (
            round(sum(text_lengths) / document_count, 2) if document_count else 0.0
        ),
        "max_text_characters": max(text_lengths, default=0),
    }
    return summary


def _call_json_model(
    model: ModelClient,
    *,
    system: str,
    prompt: str,
) -> Mapping[str, Any]:
    """调用统一模型接口，并将严格 JSON 输出解析为映射。"""
    response = model.generate(
        prompt,
        system=system,
        temperature=0.0,
        max_tokens=SELECTION_MAX_TOKENS,
    )
    text = response.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SelectionError(f"Model did not return strict JSON: {response}") from exc
    if not isinstance(payload, Mapping):
        raise SelectionError("Model JSON response must be an object")
    return dict(payload)


def _advertisements(specs: Sequence[RAGSkillSpec]) -> str:
    """仅序列化候选 Skill 的名称和描述，不披露正文与脚本。"""
    payload = [
        {"name": spec.package_name, "description": spec.description}
        for spec in specs
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _compatible_components(
    slot: SlotSpec,
    components: Sequence[RAGSkillSpec],
) -> tuple[RAGSkillSpec, ...]:
    """筛选能力名称和输入输出类型均与槽位一致的组件。"""
    return tuple(
        component
        for component in components
        if any(
            capability.name == slot.capability
            and capability.input_type == slot.input_type
            and capability.output_type == slot.output_type
            for capability in component.provides
        )
    )


def _slot_advertisements(
    slots: Sequence[SlotSpec],
    candidates: Mapping[str, Sequence[RAGSkillSpec]],
) -> str:
    """按 Agentic 槽位组织兼容组件广告和数量约束。"""
    payload = {
        slot.name: {
            "min": slot.min_count,
            "max": slot.max_count,
            "candidates": [
                {"name": spec.package_name, "description": spec.description}
                for spec in candidates[slot.name]
            ],
        }
        for slot in slots
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _validate_bindings(
    payload: Any,
    slots: Sequence[SlotSpec],
    candidates: Mapping[str, Sequence[RAGSkillSpec]],
) -> dict[str, tuple[str, ...]]:
    """校验模型返回的槽位集合、数量、唯一性和候选合法性。"""
    if not isinstance(payload, Mapping):
        raise SelectionError("component_bindings must be a JSON object")
    expected_slots = {slot.name for slot in slots}
    if set(payload) != expected_slots:
        raise SelectionError(
            "component_bindings must contain exactly these slots: "
            f"{sorted(expected_slots)}"
        )
    bindings: dict[str, tuple[str, ...]] = {}
    for slot in slots:
        selected = payload[slot.name]
        if not isinstance(selected, list) or not all(
            isinstance(name, str) for name in selected
        ):
            raise SelectionError(f"Binding for slot '{slot.name}' must be a string list")
        names = tuple(selected)
        if len(names) != len(set(names)):
            raise SelectionError(f"Binding for slot '{slot.name}' contains duplicates")
        if not slot.min_count <= len(names) <= slot.max_count:
            raise SelectionError(
                f"Binding count for slot '{slot.name}' must be between "
                f"{slot.min_count} and {slot.max_count}"
            )
        allowed = {spec.package_name for spec in candidates[slot.name]}
        unknown = set(names) - allowed
        if unknown:
            raise SelectionError(
                f"Slot '{slot.name}' selected incompatible Components: {sorted(unknown)}"
            )
        bindings[slot.name] = names
    return bindings


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    """读取模型 JSON 中必须存在的非空字符串字段。"""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SelectionError(f"Model response requires non-empty '{key}'")
    return value.strip()


def _optional_text(payload: Mapping[str, Any], key: str) -> str:
    """读取可选文本字段，并在缺失时返回空字符串。"""
    value = payload.get(key, "")
    if not isinstance(value, str):
        raise SelectionError(f"Model response field '{key}' must be a string")
    return value.strip()
