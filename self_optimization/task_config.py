"""读取由多个本地 RAG 数据任务组成的顺序自优化工作流配置。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from framework.data_groups import DataGroup, GroupDataError, load_group
from framework.interfaces import RESERVED_REQUEST_FIELDS

from .config import SelfOptimizationConfig, load_self_optimization_config


class TaskWorkflowConfigError(ValueError):
    """表示任务流配置结构、路径或 task 定义不合法。"""


@dataclass(frozen=True, slots=True)
class RAGTaskConfig:
    """描述一个使用本地规范 JSONL 数据的 RAG 评测任务。"""

    name: str
    dataset: str
    data_path: Path
    max_examples: int | None = None
    request: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskWorkflowConfig:
    """聚合基础自优化配置和按顺序执行的 RAG task 列表。"""

    config_path: Path
    self_optimization: SelfOptimizationConfig
    tasks: tuple[RAGTaskConfig, ...]
    max_process_examples: int = 5
    group: DataGroup | None = None
    validation_examples_per_task: int = 3


def load_task_workflow_config(
    path: str | Path,
    *,
    group_path: str | Path | None = None,
    shuffle_seed: int | None = None,
) -> TaskWorkflowConfig:
    """加载任务流 YAML，并解析其引用的本地自优化配置和数据文件。"""
    config_path = Path(path).resolve()
    payload = _read_yaml(config_path)
    version = payload.get("schema_version")
    if type(version) is not int or version not in (1, 2):
        raise TaskWorkflowConfigError(
            "Only task workflow schema_version 1 or 2 is supported"
        )
    allowed_root = {"schema_version", "self_optimization_config", "workflow"}
    if version == 2:
        allowed_root.update({"group_path", "shuffle_seed"})
    unknown_root = set(payload) - allowed_root
    if unknown_root:
        raise TaskWorkflowConfigError(
            f"Unknown task workflow config keys: {sorted(unknown_root)}"
        )
    self_optimization_path = _resolve_path(
        _required_text(payload, "self_optimization_config"),
        config_path,
    )
    self_optimization = load_self_optimization_config(self_optimization_path)
    workflow = _required_mapping(payload, "workflow")
    allowed_workflow = {"max_process_examples", "validation_examples_per_task"}
    allowed_workflow.update({"request"} if version == 2 else {"tasks"})
    unknown_workflow = set(workflow) - allowed_workflow
    if unknown_workflow:
        raise TaskWorkflowConfigError(
            f"workflow contains unknown keys: {sorted(unknown_workflow)}"
        )
    max_process_examples = _positive_int(
        workflow.get("max_process_examples", 5),
        "workflow.max_process_examples",
    )
    validation_examples = _positive_int(
        workflow.get("validation_examples_per_task", 3),
        "workflow.validation_examples_per_task",
    )
    group = None
    if version == 2 or group_path is not None:
        source = (
            Path(group_path).resolve()
            if group_path is not None
            else _resolve_path(_required_text(payload, "group_path"), config_path)
        )
        seed = payload.get("shuffle_seed", 42) if shuffle_seed is None else shuffle_seed
        try:
            group = load_group(source, shuffle_seed=seed)
        except GroupDataError as exc:
            raise TaskWorkflowConfigError(str(exc)) from exc
        request = _parse_request(workflow.get("request", {}), "workflow.request")
        tasks = tuple(
            RAGTaskConfig(task.task_id, group.dataset, task.path, request=request)
            for task in group.tasks
        )
        return TaskWorkflowConfig(
            config_path=config_path,
            self_optimization=self_optimization,
            tasks=tasks,
            max_process_examples=max_process_examples,
            group=group,
            validation_examples_per_task=validation_examples,
        )
    if shuffle_seed is not None:
        raise TaskWorkflowConfigError("shuffle_seed requires a Group workflow")
    tasks_value = workflow.get("tasks")
    if isinstance(tasks_value, (str, bytes, bytearray)) or not isinstance(
        tasks_value,
        Sequence,
    ):
        raise TaskWorkflowConfigError("workflow.tasks must be a list")
    tasks = tuple(
        _parse_task(value, index=index, config_path=config_path)
        for index, value in enumerate(tasks_value)
    )
    if not tasks:
        raise TaskWorkflowConfigError("workflow.tasks must not be empty")
    names = [task.name for task in tasks]
    if len(names) != len(set(names)):
        raise TaskWorkflowConfigError("workflow task names must be unique")
    return TaskWorkflowConfig(
        config_path=config_path,
        self_optimization=self_optimization,
        tasks=tasks,
        max_process_examples=max_process_examples,
        validation_examples_per_task=validation_examples,
    )


def _parse_task(
    value: Any,
    *,
    index: int,
    config_path: Path,
) -> RAGTaskConfig:
    """解析并校验一个 task 的名称、数据、样本上限和请求覆盖值。"""
    if not isinstance(value, Mapping):
        raise TaskWorkflowConfigError(f"workflow.tasks[{index}] must be a mapping")
    allowed = {"name", "dataset", "data_path", "max_examples", "request"}
    unknown = set(value) - allowed
    if unknown:
        raise TaskWorkflowConfigError(
            f"workflow.tasks[{index}] contains unknown keys: {sorted(unknown)}"
        )
    name = _required_text(value, "name", prefix=f"workflow.tasks[{index}]")
    dataset = _required_text(value, "dataset", prefix=f"workflow.tasks[{index}]")
    data_path = _resolve_path(
        _required_text(value, "data_path", prefix=f"workflow.tasks[{index}]"),
        config_path,
    )
    if not data_path.is_file():
        raise TaskWorkflowConfigError(f"Task data file does not exist: {data_path}")
    max_examples_value = value.get("max_examples")
    if max_examples_value is not None:
        max_examples = _positive_int(
            max_examples_value,
            f"workflow.tasks[{index}].max_examples",
        )
    else:
        max_examples = None
    request = _parse_request(value.get("request", {}), f"workflow.tasks[{index}].request")
    return RAGTaskConfig(
        name=name,
        dataset=dataset,
        data_path=data_path,
        max_examples=max_examples,
        request=dict(request),
    )


def _parse_request(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TaskWorkflowConfigError(f"{name} must be a mapping")
    reserved = RESERVED_REQUEST_FIELDS & set(value)
    if reserved:
        raise TaskWorkflowConfigError(f"{name} cannot override: {sorted(reserved)}")
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise TaskWorkflowConfigError(f"{name} must be JSON-compatible") from exc
    return dict(value)


def _read_yaml(path: Path) -> Mapping[str, Any]:
    """读取 YAML 根映射并统一转换解析异常。"""
    if not path.is_file():
        raise TaskWorkflowConfigError(f"Task workflow config does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise TaskWorkflowConfigError(f"Cannot read config {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TaskWorkflowConfigError("Task workflow config root must be a mapping")
    return dict(payload)


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """读取必需的映射字段。"""
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise TaskWorkflowConfigError(f"'{key}' must be a mapping")
    return value


def _required_text(
    payload: Mapping[str, Any],
    key: str,
    *,
    prefix: str | None = None,
) -> str:
    """读取 task 或根配置中的必需非空字符串。"""
    value = payload.get(key)
    field_name = f"{prefix}.{key}" if prefix else key
    if not isinstance(value, str) or not value.strip():
        raise TaskWorkflowConfigError(f"'{field_name}' must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    """把 task 配置值严格校验为正整数。"""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TaskWorkflowConfigError(f"{name} must be a positive integer")
    return value


def _resolve_path(value: str, config_path: Path) -> Path:
    """相对任务流配置目录解析数据和被引用配置路径。"""
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()
