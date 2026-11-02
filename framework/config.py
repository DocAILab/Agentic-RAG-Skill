"""加载本地或云端 API 配置并构造 framework 运行依赖。"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models import (
    EmbeddingClient,
    ModelClient,
    create_embedding_client,
    create_model_client,
)


class ConfigError(ValueError):
    """表示 framework 配置文件缺失或字段不合法。"""


@dataclass(frozen=True, slots=True)
class APIServiceConfig:
    """描述一个模型或向量 API 服务。"""

    provider: str
    model: str
    base_url: str | None
    api_key_env: str | None = None
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = 600.0
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)

    def resolve_api_key(self) -> str:
        """解析直写或环境变量密钥；未配置时返回空串表示无鉴权。"""
        if self.api_key is not None and self.api_key_env is not None:
            raise ConfigError("api_key and api_key_env cannot be used together")
        if self.api_key is not None:
            return self.api_key
        if self.api_key_env is None:
            return ""
        value = os.getenv(self.api_key_env)
        if value is None:
            raise ConfigError(
                f"Environment variable '{self.api_key_env}' is not set"
            )
        return value


@dataclass(frozen=True, slots=True)
class VectorIndexConfig:
    """描述持久化语料向量索引的缓存目录。"""

    cache_dir: Path


@dataclass(frozen=True, slots=True)
class FrameworkConfig:
    """保存 Skill 路径、Manage 名称及 Executor/Embedding 服务设置。"""

    config_path: Path
    skill_root: Path
    manage_skill: str
    executor: APIServiceConfig
    embedding: APIServiceConfig | None = None
    vector_index: VectorIndexConfig | None = None
    request_defaults: Mapping[str, Any] = field(default_factory=dict)


def load_framework_config(path: str | Path) -> FrameworkConfig:
    """读取 YAML 配置，解析相对路径并校验必需字段。"""
    config_path = Path(path).resolve()
    payload = _read_yaml(config_path)
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ConfigError("Only config schema_version 1 is supported")
    if "demo" in payload:
        raise ConfigError(
            "The legacy demo section is no longer supported; remove it and "
            "select a Group in the task-flow configuration or with --group"
        )
    allowed_root = {"schema_version", "skills", "runtime", "executor", "embedding"}
    unknown_root = set(payload) - allowed_root
    if unknown_root:
        raise ConfigError(f"Unknown framework config keys: {sorted(unknown_root)}")

    skills = _required_mapping(payload, "skills")
    root_value = _required_text(skills, "root")
    skill_root = Path(root_value)
    if not skill_root.is_absolute():
        skill_root = config_path.parent / skill_root
    skill_root = skill_root.resolve()
    if not skill_root.is_dir():
        raise ConfigError(f"Skill root does not exist: {skill_root}")
    manage_skill = str(skills.get("manage_skill", "manage-rag-default")).strip()
    if not manage_skill:
        raise ConfigError("skills.manage_skill cannot be empty")

    executor = _parse_service(_required_mapping(payload, "executor"), "executor")
    runtime_payload = payload.get("runtime", {})
    if not isinstance(runtime_payload, Mapping):
        raise ConfigError("runtime must be a mapping")
    request_defaults = runtime_payload.get("request_defaults", {})
    if not isinstance(request_defaults, Mapping):
        raise ConfigError("runtime.request_defaults must be a mapping")
    vector_index = _parse_vector_index(
        runtime_payload.get("vector_index"),
        config_path,
    )
    embedding_payload = payload.get("embedding")
    embedding = None
    if embedding_payload is not None:
        if not isinstance(embedding_payload, Mapping):
            raise ConfigError("embedding must be a mapping")
        if bool(embedding_payload.get("enabled", True)):
            embedding = _parse_service(embedding_payload, "embedding")
    return FrameworkConfig(
        config_path=config_path,
        skill_root=skill_root,
        manage_skill=manage_skill,
        executor=executor,
        embedding=embedding,
        vector_index=vector_index,
        request_defaults=dict(request_defaults),
    )


def create_clients_from_config(
    config: FrameworkConfig,
) -> tuple[ModelClient, EmbeddingClient | None]:
    """根据配置构造 Executor Model 与可选 Embedding 客户端。"""
    executor = _create_model_from_service(config.executor)
    embedding = (
        _create_embedding_from_service(config.embedding)
        if config.embedding is not None
        else None
    )
    return executor, embedding


def run_rag_from_config(
    request: Mapping[str, Any],
    *,
    config_path: str | Path,
) -> dict[str, Any]:
    """从配置构造全部依赖，并执行一次完整的分级 RAG。"""
    from .compiler import run_rag

    config = load_framework_config(config_path)
    model, embedding_model = create_clients_from_config(config)
    merged_request = {**config.request_defaults, **request}
    return run_rag(
        merged_request,
        model=model,
        embedding_model=embedding_model,
        skill_root=config.skill_root,
        manage_skill=config.manage_skill,
        vector_index_cache_dir=(
            config.vector_index.cache_dir if config.vector_index is not None else None
        ),
        embedding_fingerprint=(
            embedding_service_fingerprint(config.embedding)
            if config.embedding is not None
            else None
        ),
    )


def embedding_service_fingerprint(service: APIServiceConfig) -> str:
    """从不含密钥的 Embedding 服务配置生成稳定缓存指纹。"""
    identity = {
        "provider": service.provider.strip().lower().replace("_", "-"),
        "model": service.model,
        "base_url": service.base_url,
        "options": dict(service.options),
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_yaml(path: Path) -> Mapping[str, Any]:
    """读取 YAML 对象，并统一转换文件与解析异常。"""
    if not path.is_file():
        raise ConfigError(f"Config file does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read config {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("Config root must be a mapping")
    return dict(payload)


def _parse_service(payload: Mapping[str, Any], field_name: str) -> APIServiceConfig:
    """解析并校验一个模型服务配置块。"""
    provider = _required_text(payload, "provider")
    model = _required_text(payload, "model")
    base_url_value = payload.get("base_url")
    base_url = None if base_url_value is None else str(base_url_value).strip()
    if base_url_value is not None and not base_url:
        raise ConfigError(f"{field_name}.base_url cannot be empty")
    api_key_value = payload.get("api_key")
    api_key = None if api_key_value is None else str(api_key_value).strip()
    if api_key_value is not None and not api_key:
        raise ConfigError(f"{field_name}.api_key cannot be empty")
    api_key_env_value = payload.get("api_key_env")
    api_key_env = (
        None if api_key_env_value is None else str(api_key_env_value).strip()
    )
    if api_key_env_value is not None and not api_key_env:
        raise ConfigError(f"{field_name}.api_key_env cannot be empty")
    if api_key is not None and api_key_env is not None:
        raise ConfigError(
            f"{field_name}.api_key and {field_name}.api_key_env cannot be used together"
        )
    try:
        timeout_seconds = float(payload.get("timeout_seconds", 600.0))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name}.timeout_seconds must be numeric") from exc
    if timeout_seconds <= 0:
        raise ConfigError(f"{field_name}.timeout_seconds must be positive")
    extra_headers = _optional_string_mapping(payload, "extra_headers", field_name)
    options = payload.get("options", {})
    if not isinstance(options, Mapping):
        raise ConfigError(f"{field_name}.options must be a mapping")
    reserved = {
        "api_key",
        "api_key_env",
        "base_url",
        "enabled",
        "extra_headers",
        "model",
        "provider",
        "timeout_seconds",
    }
    overlap = set(options) & reserved
    if overlap:
        raise ConfigError(
            f"{field_name}.options contains reserved keys: {sorted(overlap)}"
        )
    return APIServiceConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        extra_headers=extra_headers,
        options=dict(options),
    )


def _parse_vector_index(
    payload: Any,
    config_path: Path,
) -> VectorIndexConfig | None:
    """解析可选持久化向量索引目录，并允许显式禁用。"""
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ConfigError("runtime.vector_index must be a mapping")
    enabled = payload.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("runtime.vector_index.enabled must be boolean")
    allowed = {"enabled", "cache_dir"}
    unknown = set(payload) - allowed
    if unknown:
        raise ConfigError(
            f"runtime.vector_index contains unknown keys: {sorted(unknown)}"
        )
    if not enabled:
        return None
    cache_dir = _resolve_config_path(
        _required_text(payload, "cache_dir"),
        config_path,
    )
    if cache_dir.exists() and not cache_dir.is_dir():
        raise ConfigError(
            f"runtime.vector_index.cache_dir is not a directory: {cache_dir}"
        )
    return VectorIndexConfig(cache_dir=cache_dir)


def _create_model_from_service(service: APIServiceConfig) -> ModelClient:
    """把服务配置转换为统一 Executor Model 客户端。"""
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


def _resolve_config_path(value: str, config_path: Path) -> Path:
    """以配置文件目录为基准解析相对文件路径。"""
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _create_embedding_from_service(service: APIServiceConfig) -> EmbeddingClient:
    """把服务配置转换为统一 Embedding 客户端。"""
    normalized_provider = service.provider.strip().lower().replace("_", "-")
    if normalized_provider in {
        "local-sentence-transformers",
        "sentence-transformer",
        "sentence-transformers",
    }:
        if service.base_url is not None:
            raise ConfigError("Local embedding provider requires base_url: null")
        if service.resolve_api_key():
            raise ConfigError("Local embedding provider does not use an API key")
        return create_embedding_client(
            service.provider,
            model=service.model,
            **service.options,
        )
    options = {
        **service.options,
        "timeout_seconds": service.timeout_seconds,
        "extra_headers": service.extra_headers,
    }
    return create_embedding_client(
        service.provider,
        model=service.model,
        api_key=service.resolve_api_key(),
        base_url=service.base_url,
        **options,
    )


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """读取必需的 YAML 映射字段。"""
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{key}' must be a mapping")
    return value


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    """读取必需的非空字符串字段。"""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{key}' must be a non-empty string")
    return value.strip()


def _optional_string_mapping(
    payload: Mapping[str, Any],
    key: str,
    field_name: str,
) -> dict[str, str]:
    """读取可选的字符串键值映射。"""
    value = payload.get(key, {})
    if not isinstance(value, Mapping) or not all(
        isinstance(header, str) and isinstance(content, str)
        for header, content in value.items()
    ):
        raise ConfigError(f"{field_name}.{key} must contain string pairs")
    return dict(value)
