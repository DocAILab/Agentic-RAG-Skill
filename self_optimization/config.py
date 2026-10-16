"""读取 Skill 自优化实验配置并创建独立的优化模型客户端。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from framework import (
    APIServiceConfig,
    FrameworkConfig,
    ModelClient,
    create_model_client,
    load_framework_config,
)


class SelfOptimizationConfigError(ValueError):
    """表示自优化配置缺失、类型错误或取值不合法。"""


@dataclass(frozen=True, slots=True)
class OptimizationSettings:
    """控制优化循环、提示长度和修改事务安全边界。"""

    max_examples: int | None = 10
    temperature: float = 0.1
    max_tokens: int = 8192
    max_prompt_chars: int = 120_000
    max_skill_file_chars: int = 24_000
    max_process_chars: int = 30_000
    max_documents: int = 10
    document_excerpt_chars: int = 1_200
    max_edits: int = 3
    max_proposal_attempts: int = 3
    allow_new_files: bool = False
    continue_on_optimization_error: bool = True
    verify_revision_execution: bool = False

    def __post_init__(self) -> None:
        """校验优化轮数、模型参数和上下文边界均可安全执行。"""
        if self.max_examples is not None and self.max_examples <= 0:
            raise SelfOptimizationConfigError(
                "optimization.max_examples must be positive or null"
            )
        if not 0.0 <= self.temperature <= 2.0:
            raise SelfOptimizationConfigError(
                "optimization.temperature must be between 0 and 2"
            )
        positive_fields = {
            "max_tokens": self.max_tokens,
            "max_prompt_chars": self.max_prompt_chars,
            "max_skill_file_chars": self.max_skill_file_chars,
            "max_process_chars": self.max_process_chars,
            "max_documents": self.max_documents,
            "document_excerpt_chars": self.document_excerpt_chars,
            "max_edits": self.max_edits,
            "max_proposal_attempts": self.max_proposal_attempts,
        }
        for name, value in positive_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SelfOptimizationConfigError(
                    f"optimization.{name} must be a positive integer"
                )


@dataclass(frozen=True, slots=True)
class SelfOptimizationConfig:
    """聚合基础 RAG 配置、Qwen 服务、工作副本和优化参数。"""

    config_path: Path
    framework: FrameworkConfig
    optimizer: APIServiceConfig
    runs_root: Path
    run_id: str | None = None
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)


def load_self_optimization_config(path: str | Path) -> SelfOptimizationConfig:
    """从 YAML 加载自优化配置，并相对配置文件解析所有路径。"""
    config_path = Path(path).resolve()
    payload = _read_yaml(config_path)
    if int(payload.get("schema_version", 0)) != 1:
        raise SelfOptimizationConfigError(
            "Only self-optimization config schema_version 1 is supported"
        )
    allowed_root = {"schema_version", "framework_config", "workspace", "optimizer", "optimization"}
    unknown_root = set(payload) - allowed_root
    if unknown_root:
        raise SelfOptimizationConfigError(
            f"Unknown self-optimization config keys: {sorted(unknown_root)}"
        )

    framework_path = _resolve_path(
        _required_text(payload, "framework_config"),
        config_path,
    )
    framework = load_framework_config(framework_path)
    workspace = _required_mapping(payload, "workspace")
    unknown_workspace = set(workspace) - {"runs_root", "run_id"}
    if unknown_workspace:
        raise SelfOptimizationConfigError(
            f"workspace contains unknown keys: {sorted(unknown_workspace)}"
        )
    runs_root = _resolve_path(_required_text(workspace, "runs_root"), config_path)
    run_id_value = workspace.get("run_id")
    run_id = None if run_id_value is None else str(run_id_value).strip()
    if run_id_value is not None and not run_id:
        raise SelfOptimizationConfigError("workspace.run_id cannot be empty")

    optimizer = _parse_service(_required_mapping(payload, "optimizer"))
    optimization_payload = payload.get("optimization", {})
    if not isinstance(optimization_payload, Mapping):
        raise SelfOptimizationConfigError("optimization must be a mapping")
    optimization = _parse_optimization(optimization_payload)
    return SelfOptimizationConfig(
        config_path=config_path,
        framework=framework,
        optimizer=optimizer,
        runs_root=runs_root,
        run_id=run_id,
        optimization=optimization,
    )


def create_optimizer_model(config: SelfOptimizationConfig) -> ModelClient:
    """依据独立 optimizer 配置创建 Qwen 或其他兼容模型客户端。"""
    service = config.optimizer
    options = {
        **service.options,
        "timeout_seconds": service.timeout_seconds,
        "extra_headers": service.extra_headers,
    }
    return create_model_client(
        service.provider,
        model=service.model,
        api_key=service.resolve_api_key(),
        base_url=service.base_url,
        **options,
    )


def _read_yaml(path: Path) -> Mapping[str, Any]:
    """读取 YAML 根对象并统一转换文件与解析错误。"""
    if not path.is_file():
        raise SelfOptimizationConfigError(f"Config file does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise SelfOptimizationConfigError(f"Cannot read config {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SelfOptimizationConfigError("Config root must be a mapping")
    return dict(payload)


def _parse_service(payload: Mapping[str, Any]) -> APIServiceConfig:
    """解析优化模型服务，并保持与 framework 模型配置相同的语义。"""
    allowed = {
        "provider",
        "model",
        "base_url",
        "api_key",
        "api_key_env",
        "timeout_seconds",
        "extra_headers",
        "options",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise SelfOptimizationConfigError(
            f"optimizer contains unknown keys: {sorted(unknown)}"
        )
    provider = _required_text(payload, "provider")
    model = _required_text(payload, "model")
    base_url = _optional_text(payload, "base_url")
    api_key = _optional_text(payload, "api_key")
    api_key_env = _optional_text(payload, "api_key_env")
    if api_key is not None and api_key_env is not None:
        raise SelfOptimizationConfigError(
            "optimizer.api_key and optimizer.api_key_env cannot be used together"
        )
    timeout_seconds = _positive_float(
        payload.get("timeout_seconds", 600.0),
        "optimizer.timeout_seconds",
    )
    extra_headers_value = payload.get("extra_headers", {})
    if not isinstance(extra_headers_value, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in extra_headers_value.items()
    ):
        raise SelfOptimizationConfigError(
            "optimizer.extra_headers must contain string pairs"
        )
    options = payload.get("options", {})
    if not isinstance(options, Mapping):
        raise SelfOptimizationConfigError("optimizer.options must be a mapping")
    return APIServiceConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
        extra_headers=dict(extra_headers_value),
        options=dict(options),
    )


def _parse_optimization(payload: Mapping[str, Any]) -> OptimizationSettings:
    """解析逐样本优化开关、模型调用参数和上下文限制。"""
    allowed = {
        "max_examples",
        "temperature",
        "max_tokens",
        "max_prompt_chars",
        "max_skill_file_chars",
        "max_process_chars",
        "max_documents",
        "document_excerpt_chars",
        "max_edits",
        "max_proposal_attempts",
        "allow_new_files",
        "continue_on_optimization_error",
        "verify_revision_execution",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise SelfOptimizationConfigError(
            f"optimization contains unknown keys: {sorted(unknown)}"
        )
    max_examples_value = payload.get("max_examples", 10)
    if max_examples_value is not None and (
        isinstance(max_examples_value, bool) or not isinstance(max_examples_value, int)
    ):
        raise SelfOptimizationConfigError(
            "optimization.max_examples must be a positive integer or null"
        )
    return OptimizationSettings(
        max_examples=max_examples_value,
        temperature=float(payload.get("temperature", 0.1)),
        max_tokens=_positive_int(payload.get("max_tokens", 8192), "max_tokens"),
        max_prompt_chars=_positive_int(
            payload.get("max_prompt_chars", 120_000), "max_prompt_chars"
        ),
        max_skill_file_chars=_positive_int(
            payload.get("max_skill_file_chars", 24_000), "max_skill_file_chars"
        ),
        max_process_chars=_positive_int(
            payload.get("max_process_chars", 30_000), "max_process_chars"
        ),
        max_documents=_positive_int(
            payload.get("max_documents", 10), "max_documents"
        ),
        document_excerpt_chars=_positive_int(
            payload.get("document_excerpt_chars", 1_200),
            "document_excerpt_chars",
        ),
        max_edits=_positive_int(payload.get("max_edits", 3), "max_edits"),
        max_proposal_attempts=_positive_int(
            payload.get("max_proposal_attempts", 3),
            "max_proposal_attempts",
        ),
        allow_new_files=_boolean(payload, "allow_new_files", False),
        continue_on_optimization_error=_boolean(
            payload,
            "continue_on_optimization_error",
            True,
        ),
        verify_revision_execution=_boolean(
            payload,
            "verify_revision_execution",
            False,
        ),
    )


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """读取必需的映射字段。"""
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise SelfOptimizationConfigError(f"'{key}' must be a mapping")
    return value


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    """读取必需的非空字符串字段。"""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SelfOptimizationConfigError(f"'{key}' must be a non-empty string")
    return value.strip()


def _optional_text(payload: Mapping[str, Any], key: str) -> str | None:
    """读取允许为 null、但不允许为空字符串的可选字段。"""
    value = payload.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        raise SelfOptimizationConfigError(f"optimizer.{key} cannot be empty")
    return normalized


def _positive_int(value: Any, name: str) -> int:
    """把配置值校验为正整数。"""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SelfOptimizationConfigError(
            f"optimization.{name} must be a positive integer"
        )
    return value


def _positive_float(value: Any, name: str) -> float:
    """把配置值校验为正浮点数。"""
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise SelfOptimizationConfigError(f"{name} must be numeric") from exc
    if normalized <= 0:
        raise SelfOptimizationConfigError(f"{name} must be positive")
    return normalized


def _boolean(payload: Mapping[str, Any], key: str, default: bool) -> bool:
    """读取严格布尔配置，避免字符串被误判为真值。"""
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise SelfOptimizationConfigError(f"optimization.{key} must be boolean")
    return value


def _resolve_path(value: str, config_path: Path) -> Path:
    """以自优化配置文件目录为基准解析相对路径。"""
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()
