"""读取由多个本地 RAG 数据任务组成的顺序自优化工作流配置。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from framework.data_groups import DataGroup, GroupDataError, load_group
from framework.interfaces import RESERVED_REQUEST_FIELDS

from .config import SelfOptimizationConfig, load_self_optimization_config
from .scheduling import QAExecutionConfig, parse_qa_execution


class TaskWorkflowConfigError(ValueError):
    """表示任务流配置结构、路径或 task 定义不合法。"""


@dataclass(frozen=True, slots=True)
class RAGTaskConfig:
    """描述一个使用本地规范 JSONL 数据的 RAG 评测任务。"""

    name: str
    dataset: str
    data_path: Path
    request: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskWorkflowConfig:
    """聚合基础自优化配置和按顺序执行的 RAG task 列表。"""

    config_path: Path
    self_optimization: SelfOptimizationConfig
    tasks: tuple[RAGTaskConfig, ...]
    group: DataGroup
    max_process_examples: int = 5
    validation_examples_per_task: int = 3
    qa_execution: QAExecutionConfig = field(default_factory=QAExecutionConfig)


def load_task_workflow_config(
    path: str | Path,
    *,
    group_path: str | Path | None = None,
    shuffle_seed: int | None = None,
    policy_mode: str | None = None,
) -> TaskWorkflowConfig:
    """加载任务流 YAML，并解析其引用的本地自优化配置和数据文件。"""
    config_path = Path(path).resolve()
    payload = _read_yaml(config_path)
    version = payload.get("schema_version")
    if type(version) is not int or version != 2:
        raise TaskWorkflowConfigError(
            "Only Group task workflow schema_version 2 is supported"
        )
    allowed_root = {"schema_version", "self_optimization_config", "workflow", "group_path", "shuffle_seed"}
    unknown_root = set(payload) - allowed_root
    if unknown_root:
        raise TaskWorkflowConfigError(
            f"Unknown task workflow config keys: {sorted(unknown_root)}"
        )
    self_optimization_path = _resolve_path(
        _required_text(payload, "self_optimization_config"),
        config_path,
    )
    self_optimization = load_self_optimization_config(
        self_optimization_path, policy_mode=policy_mode,
    )
    workflow = _required_mapping(payload, "workflow")
    allowed_workflow = {"max_process_examples", "validation_examples_per_task", "request", "qa_execution"}
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
    try:
        qa_execution = parse_qa_execution(workflow.get("qa_execution", {}))
    except ValueError as exc:
        raise TaskWorkflowConfigError(str(exc)) from exc
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
        group=group,
        max_process_examples=max_process_examples,
        validation_examples_per_task=validation_examples,
        qa_execution=qa_execution,
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
) -> str:
    """读取 task 或根配置中的必需非空字符串。"""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TaskWorkflowConfigError(f"'{key}' must be a non-empty string")
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
