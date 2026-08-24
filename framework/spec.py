"""三层 RAG Skill 的结构定义、发现与校验逻辑。"""

from __future__ import annotations

import importlib.util
import inspect
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


class SkillSpecError(ValueError):
    """表示 Skill 包结构、契约或运行入口不符合规范。"""

    pass


class SkillKind(str, Enum):
    MANAGE = "manage"
    AGENTIC = "agentic"
    COMPONENT = "component"


SKILL_KIND_DIRECTORIES = (
    ("manage", SkillKind.MANAGE),
    ("agentic", SkillKind.AGENTIC),
    ("components", SkillKind.COMPONENT),
)


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    type: str
    path: str
    callable: str


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    name: str
    input_type: str
    output_type: str


@dataclass(frozen=True, slots=True)
class CapabilityRequirementSpec:
    capability: str
    components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SlotSpec:
    name: str
    capability: str
    input_type: str
    output_type: str
    min_count: int = 1
    max_count: int = 1

    def __post_init__(self) -> None:
        """校验组件槽位的最小和最大绑定数量。"""
        if self.min_count < 0 or self.max_count < self.min_count:
            raise SkillSpecError(f"Invalid cardinality for slot '{self.name}'.")


@dataclass(frozen=True, slots=True)
class SelectionSpec:
    target_kind: SkillKind
    min_count: int = 1
    max_count: int = 1

    def __post_init__(self) -> None:
        """校验 Manage Skill 选择目标的数量约束。"""
        if self.min_count < 1 or self.max_count < self.min_count:
            raise SkillSpecError("Invalid Manage selection cardinality.")


@dataclass(frozen=True, slots=True)
class RAGSkillSpec:
    package_name: str
    description: str
    runtime_id: str
    kind: SkillKind
    version: str
    mutable: bool
    package_path: Path
    runtime: RuntimeSpec | None = None
    selection: SelectionSpec | None = None
    slots: tuple[SlotSpec, ...] = ()
    provides: tuple[CapabilitySpec, ...] = ()
    requires: tuple[CapabilityRequirementSpec, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)


def discover_specs(
    root: str | Path,
    *,
    validate_runtime: bool = True,
) -> tuple[RAGSkillSpec, ...]:
    """按三级目录发现 Skill 包，并拒绝错层、重复名称和重复运行时 ID。"""
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise SkillSpecError(f"Skill root does not exist: {root_path}")

    specs = []
    discovered_files = set()
    for directory_name, expected_kind in SKILL_KIND_DIRECTORIES:
        kind_root = root_path / directory_name
        if not kind_root.exists():
            continue
        if not kind_root.is_dir():
            raise SkillSpecError(f"Skill kind path is not a directory: {kind_root}")
        for skill_file in sorted(kind_root.glob("*/SKILL.md")):
            spec = load_spec(
                skill_file.parent,
                validate_runtime=validate_runtime,
            )
            if spec.kind is not expected_kind:
                raise SkillSpecError(
                    f"Skill '{spec.package_name}' declares kind '{spec.kind.value}' "
                    f"but is stored under '{directory_name}'."
                )
            specs.append(spec)
            discovered_files.add(skill_file.resolve())

    misplaced_files = sorted(
        skill_file
        for skill_file in root_path.rglob("SKILL.md")
        if skill_file.resolve() not in discovered_files
    )
    if misplaced_files:
        relative_paths = [str(path.relative_to(root_path)) for path in misplaced_files]
        raise SkillSpecError(
            "Skill packages must be stored under manage/, agentic/, or components/: "
            f"{relative_paths}"
        )

    specs = tuple(specs)
    package_names = [spec.package_name for spec in specs]
    runtime_ids = [spec.runtime_id for spec in specs]
    if len(package_names) != len(set(package_names)):
        raise SkillSpecError("Duplicate Skill package names.")
    if len(runtime_ids) != len(set(runtime_ids)):
        raise SkillSpecError("Duplicate Skill runtime IDs.")
    _validate_component_requirements(specs)
    return specs


def load_spec(
    package_path: str | Path,
    *,
    validate_runtime: bool = True,
) -> RAGSkillSpec:
    """读取并完整校验单个 Skill 包的标准元数据与扩展契约。"""
    package = Path(package_path).resolve()
    skill_file = package / "SKILL.md"
    manifest_file = package / "ragskill.yaml"
    package_name, description = _validate_skill_md(skill_file, package.name)
    payload = _read_yaml(manifest_file)
    try:
        if int(payload.get("schema_version", 0)) != 1:
            raise SkillSpecError("Only ragskill schema_version 1 is supported.")
        kind = SkillKind(str(payload["kind"]))
        runtime = _parse_runtime(payload.get("runtime"))
        selection = _parse_selection(payload.get("selection"))
        slots = _parse_slots(payload.get("slots", {}))
        provides = _parse_capabilities(payload.get("provides", {}))
        requires = _parse_requirements(payload.get("requires", {}))
        spec = RAGSkillSpec(
            package_name=package_name,
            description=description,
            runtime_id=str(payload["runtime_id"]),
            kind=kind,
            version=str(payload.get("version", "0.1.0")),
            mutable=bool(payload.get("mutable", True)),
            package_path=package,
            runtime=runtime,
            selection=selection,
            slots=slots,
            provides=provides,
            requires=requires,
            raw=payload,
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, SkillSpecError):
            raise
        raise SkillSpecError(f"Invalid manifest {manifest_file}: {exc}") from exc
    _validate_kind_contract(spec)
    if validate_runtime:
        _validate_runtime(spec)
    return spec


def load_runtime_callable(spec: RAGSkillSpec) -> Any:
    """按 Skill 契约安全加载 Python 运行入口。"""
    if spec.runtime is None:
        raise SkillSpecError(f"Skill '{spec.package_name}' has no runtime.")
    runtime_path = _safe_runtime_path(spec)
    module = _load_module(runtime_path, spec.package_name)
    try:
        function = getattr(module, spec.runtime.callable)
    except AttributeError as exc:
        raise SkillSpecError(
            f"Runtime {runtime_path} has no callable '{spec.runtime.callable}'."
        ) from exc
    if not callable(function):
        raise SkillSpecError(
            f"Runtime entry '{spec.runtime.callable}' is not callable."
        )
    return function


def _validate_kind_contract(spec: RAGSkillSpec) -> None:
    """校验 Manage、Agentic 和 Component 各自允许声明的字段。"""
    if spec.kind is SkillKind.MANAGE:
        if spec.selection is None:
            raise SkillSpecError("Manage Skill must declare selection.")
        if spec.selection.target_kind is not SkillKind.AGENTIC:
            raise SkillSpecError("Manage Skill must select Agentic Skills.")
        if spec.runtime is not None or spec.slots or spec.provides or spec.requires:
            raise SkillSpecError(
                "Manage Skill cannot declare runtime, slots, capabilities, or requirements."
            )
    elif spec.kind is SkillKind.AGENTIC:
        if spec.runtime is None or spec.runtime.type != "python-workflow":
            raise SkillSpecError(
                "Agentic Skill must declare a python-workflow runtime."
            )
        if not spec.slots:
            raise SkillSpecError("Agentic Skill must declare Component slots.")
        if spec.selection is not None or spec.provides or spec.requires:
            raise SkillSpecError(
                "Agentic Skill cannot declare Manage selection, capabilities, or requirements."
            )
    else:
        if spec.runtime is None or spec.runtime.type != "python-component":
            raise SkillSpecError(
                "Component Skill must declare a python-component runtime."
            )
        if not spec.provides:
            raise SkillSpecError("Component Skill must provide a capability.")
        if spec.selection is not None or spec.slots:
            raise SkillSpecError(
                "Component Skill cannot declare Manage selection or Agentic slots."
            )


def _validate_runtime(spec: RAGSkillSpec) -> None:
    """校验 Agentic 与 Component 运行函数是否符合统一签名。"""
    if spec.runtime is None:
        return
    function = load_runtime_callable(spec)
    parameters = tuple(inspect.signature(function).parameters)
    expected = (
        ("request", "components")
        if spec.kind is SkillKind.AGENTIC
        else ("inputs", "context")
    )
    if parameters != expected:
        raise SkillSpecError(
            f"{spec.package_name} runtime signature must be "
            f"{spec.runtime.callable}{expected}, got {parameters}."
        )


def _validate_skill_md(skill_file: Path, directory_name: str) -> tuple[str, str]:
    """校验可移植 SKILL.md 的 frontmatter 和目录命名。"""
    if not skill_file.is_file():
        raise SkillSpecError(f"Missing SKILL.md in {skill_file.parent}.")
    text = skill_file.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.DOTALL)
    if match is None:
        raise SkillSpecError(f"Invalid SKILL.md frontmatter: {skill_file}")
    payload = yaml.safe_load(match.group(1))
    if not isinstance(payload, Mapping):
        raise SkillSpecError(f"SKILL.md frontmatter must be a mapping: {skill_file}")
    if set(payload) != {"name", "description"}:
        raise SkillSpecError(
            f"Portable sample Skills only allow name and description: {skill_file}"
        )
    name = str(payload.get("name", ""))
    if name != directory_name:
        raise SkillSpecError(
            f"Skill name '{name}' does not match directory '{directory_name}'."
        )
    description = str(payload.get("description", "")).strip()
    if not description:
        raise SkillSpecError(f"Skill '{name}' has an empty description.")
    return name, description


def _read_yaml(path: Path) -> Mapping[str, Any]:
    """读取 YAML 映射，并将底层解析错误统一转换为规范错误。"""
    if not path.is_file():
        raise SkillSpecError(f"Missing ragskill.yaml in {path.parent}.")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise SkillSpecError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SkillSpecError(f"Manifest must be a mapping: {path}")
    return dict(payload)


def _parse_runtime(payload: Any) -> RuntimeSpec | None:
    """将 runtime 配置解析为不可变运行入口描述。"""
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise SkillSpecError("runtime must be a mapping.")
    return RuntimeSpec(
        type=str(payload["type"]),
        path=str(payload["path"]),
        callable=str(payload.get("callable", "run")),
    )


def _parse_selection(payload: Any) -> SelectionSpec | None:
    """将 Manage Skill 的选择配置解析为数量受限的契约。"""
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise SkillSpecError("selection must be a mapping.")
    return SelectionSpec(
        target_kind=SkillKind(str(payload["target_kind"])),
        min_count=int(payload.get("min", 1)),
        max_count=int(payload.get("max", 1)),
    )


def _parse_slots(payload: Any) -> tuple[SlotSpec, ...]:
    """将 Agentic Skill 的组件槽位映射解析为有序契约。"""
    if not isinstance(payload, Mapping):
        raise SkillSpecError("slots must be a mapping.")
    slots = []
    for name, value in payload.items():
        if not isinstance(value, Mapping):
            raise SkillSpecError(f"Slot '{name}' must be a mapping.")
        slots.append(
            SlotSpec(
                name=str(name),
                capability=str(value["capability"]),
                input_type=str(value["input"]),
                output_type=str(value["output"]),
                min_count=int(value.get("min", 1)),
                max_count=int(value.get("max", 1)),
            )
        )
    return tuple(slots)


def _parse_capabilities(payload: Any) -> tuple[CapabilitySpec, ...]:
    """解析 Components Skill 对外提供的能力和输入输出类型。"""
    if not isinstance(payload, Mapping):
        raise SkillSpecError("provides must be a mapping.")
    capabilities = []
    for name, value in payload.items():
        if not isinstance(value, Mapping):
            raise SkillSpecError(f"Capability '{name}' must be a mapping.")
        capabilities.append(
            CapabilitySpec(
                name=str(name),
                input_type=str(value["input"]),
                output_type=str(value["output"]),
            )
        )
    return tuple(capabilities)


def _parse_requirements(payload: Any) -> tuple[CapabilityRequirementSpec, ...]:
    """解析 Component 对其他 capability 的允许组件要求。"""
    if not isinstance(payload, Mapping):
        raise SkillSpecError("requires must be a mapping.")
    requirements = []
    for capability, value in payload.items():
        if not isinstance(value, Mapping):
            raise SkillSpecError(
                f"Requirement for capability '{capability}' must be a mapping."
            )
        components = value.get("components")
        if not isinstance(components, list) or not components or not all(
            isinstance(name, str) and name.strip() for name in components
        ):
            raise SkillSpecError(
                f"Requirement for capability '{capability}' must contain "
                "a non-empty components string list."
            )
        normalized = tuple(name.strip() for name in components)
        if len(normalized) != len(set(normalized)):
            raise SkillSpecError(
                f"Requirement for capability '{capability}' contains duplicates."
            )
        requirements.append(
            CapabilityRequirementSpec(
                capability=str(capability),
                components=normalized,
            )
        )
    return tuple(requirements)


def _validate_component_requirements(specs: Sequence[RAGSkillSpec]) -> None:
    """验证 Component requirements 引用已存在且能力匹配的组件。"""
    components = {
        spec.package_name: spec
        for spec in specs
        if spec.kind is SkillKind.COMPONENT
    }
    for component in components.values():
        for requirement in component.requires:
            for required_name in requirement.components:
                required = components.get(required_name)
                if required is None:
                    raise SkillSpecError(
                        f"Component '{component.package_name}' requires unknown "
                        f"Component '{required_name}'."
                    )
                if not any(
                    capability.name == requirement.capability
                    for capability in required.provides
                ):
                    raise SkillSpecError(
                        f"Component '{component.package_name}' requires "
                        f"'{required_name}' to provide capability "
                        f"'{requirement.capability}'."
                    )


def binding_requirement_errors(
    agentic: RAGSkillSpec,
    bindings: Mapping[str, Sequence[str]],
    components: Mapping[str, RAGSkillSpec],
) -> tuple[str, ...]:
    """返回所选 Component 跨 capability 依赖中未满足的错误。

    依赖语义为“至少一个提供该能力的绑定组件位于允许列表内”，
    允许多分支技能（如 parallel）在绑定 HyDE 的同时保留非允许组件。
    """
    selected_by_capability: dict[str, set[str]] = {}
    selected_names = set()
    for slot in agentic.slots:
        names = tuple(bindings.get(slot.name, ()))
        selected_by_capability.setdefault(slot.capability, set()).update(names)
        selected_names.update(names)

    errors = []
    for selected_name in sorted(selected_names):
        component = components.get(selected_name)
        if component is None:
            continue
        for requirement in component.requires:
            selected = selected_by_capability.get(requirement.capability, set())
            allowed = set(requirement.components)
            if not selected or not (selected & allowed):
                errors.append(
                    f"Component '{selected_name}' requires capability "
                    f"'{requirement.capability}' to use one of "
                    f"{sorted(allowed)}, got {sorted(selected)}"
                )
    return tuple(errors)


def _safe_runtime_path(spec: RAGSkillSpec) -> Path:
    """解析运行脚本路径，并阻止路径逃逸出 Skill 包。"""
    assert spec.runtime is not None
    package = spec.package_path.resolve()
    runtime_path = (package / spec.runtime.path).resolve()
    if not runtime_path.is_relative_to(package):
        raise SkillSpecError(f"Runtime path escapes Skill package: {runtime_path}")
    if not runtime_path.is_file():
        raise SkillSpecError(f"Runtime script does not exist: {runtime_path}")
    return runtime_path


def _load_module(path: Path, package_name: str) -> ModuleType:
    """从指定脚本创建并执行隔离的 Python 模块。"""
    module_name = f"ragskill_sample_{package_name.replace('-', '_')}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise SkillSpecError(f"Cannot import runtime script: {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module
