"""OpenAI-compatible 与 Anthropic 大模型接口适配器。"""

from __future__ import annotations

import codecs
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from http.client import HTTPException
from threading import Lock
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

JsonMapping = Mapping[str, Any]
JsonTransport = Callable[
    [str, Mapping[str, str], Mapping[str, Any], float],
    Mapping[str, Any],
]


class _UnsetAPIKey:
    """Distinguish an omitted factory key from explicitly anonymous config."""


_API_KEY_UNSET = _UnsetAPIKey()


class ModelAPIError(RuntimeError):
    """表示模型 API 请求失败或响应结构不合法。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        error_category: str | None = None,
        provider_error_kind: str | None = None,
        provider_details: Mapping[str, Any] | None = None,
        response_body_sha256: str | None = None,
        response_body_bytes: int | None = None,
    ) -> None:
        """保存错误消息以及供请求层判断的 HTTP 状态与可重试标记。"""
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.error_category = error_category or ("http" if status_code is not None else "api")
        self.provider_error_kind = provider_error_kind
        self.provider_details = dict(provider_details or {})
        self.response_body_sha256 = response_body_sha256
        self.response_body_bytes = response_body_bytes
        self.attempt_count = 0
        self.error_counts: dict[str, int] = {}

    def diagnostics(self) -> dict[str, Any]:
        """Return bounded, JSON-safe request diagnostics without raw response text.

        Built-in transports whitelist provider identifiers and summarize messages;
        neither request headers, credentials, prompts nor error bodies are retained.
        """
        return {
            "error_category": self.error_category,
            "status_code": self.status_code,
            "retryable": self.retryable,
            "provider_error_kind": self.provider_error_kind,
            "provider_details": dict(self.provider_details),
            "attempt_count": self.attempt_count,
            "error_counts": dict(self.error_counts),
            "response_body_sha256": self.response_body_sha256,
            "response_body_bytes": self.response_body_bytes,
        }


_REQUEST_STAT_KEYS = (
    "requests", "succeeded", "failed", "attempts", "retries",
    "response_structure_errors", "provider_errors", "http_errors",
    "transport_errors", "response_content_errors", "other_api_errors",
)
_ERROR_STAT_KEYS = {
    "response_structure": "response_structure_errors",
    "provider": "provider_errors", "http": "http_errors",
    "transport": "transport_errors", "response_content": "response_content_errors",
}


class ModelClient(Protocol):
    """供分级选择器和生成组件共同使用的统一文本生成接口。"""

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """向模型提交系统指令和用户提示，并返回纯文本结果。"""
        ...


class EmbeddingClient(Protocol):
    """供 Vector Retriever 使用的统一文本向量接口。"""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """按输入顺序返回每段文本对应的向量。"""
        ...


@dataclass(slots=True)
class OpenAICompatibleModelClient:
    """通过 Chat Completions 调用 OpenAI 或兼容服务。"""

    model: str
    api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        repr=False,
    )
    base_url: str = "https://api.openai.com/v1"
    default_max_tokens: int = 8192
    timeout_seconds: float = 600.0
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0
    extra_headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    extra_body: Mapping[str, Any] = field(default_factory=dict, repr=False)
    transport: JsonTransport | None = field(default=None, repr=False)
    _request_counts: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(_REQUEST_STAT_KEYS, 0), init=False, repr=False,
    )
    _statistics_lock: Any = field(default_factory=Lock, init=False, repr=False)
    _last_request_counts: ContextVar = field(
        default_factory=lambda: ContextVar("openai_last_request_statistics", default=None),
        init=False, repr=False,
    )

    def __post_init__(self) -> None:
        """补充默认 HTTP transport，并校验基础配置。"""
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if self.default_max_tokens <= 0 or self.timeout_seconds <= 0:
            raise ValueError("token and timeout limits must be positive")
        _validate_retry_options(self.max_retries, self.retry_backoff_seconds)
        if not isinstance(self.extra_body, Mapping):
            raise ValueError("extra_body must be a mapping")
        if "stream" in self.extra_body and not isinstance(self.extra_body["stream"], bool):
            raise ValueError("extra_body.stream must be a boolean")
        reserved_body_keys = {
            "model",
            "messages",
            "temperature",
            "max_tokens",
        }
        overlap = set(self.extra_body) & reserved_body_keys
        if overlap:
            raise ValueError(
                f"extra_body cannot override request keys: {sorted(overlap)}"
            )
        if self.transport is None:
            self.transport = _post_openai

    def request_statistics(self) -> dict[str, int]:
        """Cumulative logical calls, transport attempts and disjoint error counts.

        Retries retain the same request and are counted even when recovery succeeds.
        This snapshot contains no prompts, responses, headers or credentials.
        """
        with self._statistics_lock:
            return dict(self._request_counts)

    def _record_request_stat(self, key: str) -> None:
        current = self._last_request_counts.get()
        if current is not None:
            current[key] += 1
        with self._statistics_lock:
            self._request_counts[key] += 1

    def last_request_statistics(self) -> dict[str, int] | None:
        """Counters for this execution context's last call, never a shared delta.

        A worker that has not made a call has no observation. Concurrent workers
        retain independent dictionaries while request_statistics stays cumulative.
        """
        current = self._last_request_counts.get()
        return dict(current) if current is not None else None

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """调用 OpenAI-compatible Chat Completions 并提取首个文本答案。"""
        self._last_request_counts.set(None)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": (
                max_tokens if max_tokens is not None else self.default_max_tokens
            ),
            **self.extra_body,
        }
        headers = {"content-type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        assert self.transport is not None
        self._last_request_counts.set(dict.fromkeys(_REQUEST_STAT_KEYS, 0))
        return _request_json_with_retries(
            self.transport,
            _endpoint(self.base_url, "chat/completions"),
            headers,
            payload,
            self.timeout_seconds,
            # A single budget covers transport and envelope recovery together.
            # A caller can disable/reduce retries, but generation never exceeds 3 submissions.
            max_retries=min(self.max_retries, 2),
            retry_backoff_seconds=self.retry_backoff_seconds,
            response_parser=_extract_openai_text,
            record_stat=self._record_request_stat,
        )


@dataclass(slots=True)
class OpenAICompatibleEmbeddingClient:
    """通过 `/v1/embeddings` 调用 OpenAI-compatible 向量服务。"""

    model: str
    api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        repr=False,
    )
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 600.0
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0
    extra_headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    transport: JsonTransport | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """补充默认 HTTP transport，并校验向量模型配置。"""
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        _validate_retry_options(self.max_retries, self.retry_backoff_seconds)
        if self.transport is None:
            self.transport = _post_json

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """调用 OpenAI-compatible Embeddings API 并按索引还原向量顺序。"""
        normalized = [str(text) for text in texts]
        if not normalized:
            return []
        headers = {"content-type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        assert self.transport is not None
        response = _request_json_with_retries(
            self.transport,
            _endpoint(self.base_url, "embeddings"),
            headers,
            {"model": self.model, "input": normalized},
            self.timeout_seconds,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
        )
        return _extract_openai_embeddings(response, expected_count=len(normalized))


@dataclass(slots=True)
class SentenceTransformerEmbeddingClient:
    """在本地进程中通过 sentence-transformers 生成文本向量。"""

    model: str
    device: str | None = None
    batch_size: int = 32
    normalize_embeddings: bool = True
    encoder: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """校验本地向量模型名称和批大小。"""
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """按输入顺序批量编码文本，并返回普通浮点数列表。"""
        normalized = [str(text) for text in texts]
        if not normalized:
            return []
        from .reranking import local_model_inference_lock

        # Share the process GPU guard with BGE rather than overlapping multiple
        # local models' peak allocations as independent QA workers arrive.
        with local_model_inference_lock():
            encoder = self._get_encoder()
            try:
                encoded = encoder.encode(
                    normalized,
                    batch_size=self.batch_size,
                    normalize_embeddings=self.normalize_embeddings,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            except Exception as exc:
                raise ModelAPIError(f"Local embedding failed: {exc}") from exc
        rows = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        try:
            embeddings = [[float(value) for value in row] for row in rows]
        except (TypeError, ValueError) as exc:
            raise ModelAPIError("Local embedding returned invalid vectors") from exc
        if len(embeddings) != len(normalized):
            raise ModelAPIError("Local embedding vector count does not match input")
        return embeddings

    def load(self) -> None:
        """显式加载一次模型，使下载或设备错误在评测开始前暴露。"""
        self._get_encoder()

    def _get_encoder(self) -> Any:
        """延迟加载 sentence-transformers 模型，避免未使用向量检索时加载权重。"""
        from .reranking import local_model_inference_lock

        with local_model_inference_lock():
            if self.encoder is not None:
                return self.encoder
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ModelAPIError(
                    "Local embedding requires the 'sentence-transformers' package"
                ) from exc
            try:
                self.encoder = SentenceTransformer(self.model, device=self.device)
            except Exception as exc:
                raise ModelAPIError(f"Cannot load local embedding model: {exc}") from exc
            return self.encoder


@dataclass(slots=True)
class AnthropicModelClient:
    """通过 Anthropic Messages API 调用 Claude 模型。"""

    model: str
    api_key: str | None = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"),
        repr=False,
    )
    base_url: str = "https://api.anthropic.com"
    api_version: str = "2023-06-01"
    default_max_tokens: int = 8192
    timeout_seconds: float = 600.0
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0
    extra_headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    transport: JsonTransport | None = field(default=None, repr=False)
    _request_counts: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(_REQUEST_STAT_KEYS, 0), init=False, repr=False,
    )
    _statistics_lock: Any = field(default_factory=Lock, init=False, repr=False)
    _last_request_counts: ContextVar = field(
        default_factory=lambda: ContextVar("anthropic_last_request_statistics", default=None),
        init=False, repr=False,
    )

    def __post_init__(self) -> None:
        """补充默认 HTTP transport，并校验 Anthropic 必需配置。"""
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if not self.api_key:
            raise ValueError("Anthropic API requires api_key or ANTHROPIC_API_KEY")
        if self.default_max_tokens <= 0 or self.timeout_seconds <= 0:
            raise ValueError("token and timeout limits must be positive")
        _validate_retry_options(self.max_retries, self.retry_backoff_seconds)
        if self.transport is None:
            self.transport = _post_json

    def request_statistics(self) -> dict[str, int]:
        """Return aggregate generate/HTTP counters without request content."""
        with self._statistics_lock:
            return dict(self._request_counts)

    def last_request_statistics(self) -> dict[str, int] | None:
        """Return only the last generate in the calling worker's context."""
        current = self._last_request_counts.get()
        return dict(current) if current is not None else None

    def _record_request_stat(self, key: str) -> None:
        current = self._last_request_counts.get()
        if current is not None:
            current[key] += 1
        with self._statistics_lock:
            self._request_counts[key] += 1

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """调用 Anthropic Messages API 并拼接响应中的文本内容块。"""
        self._last_request_counts.set(None)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": (
                max_tokens if max_tokens is not None else self.default_max_tokens
            ),
        }
        if system:
            payload["system"] = system
        headers = {
            "content-type": "application/json",
            "x-api-key": str(self.api_key),
            "anthropic-version": self.api_version,
            **self.extra_headers,
        }
        assert self.transport is not None
        self._last_request_counts.set(dict.fromkeys(_REQUEST_STAT_KEYS, 0))
        return _request_json_with_retries(
            self.transport,
            _endpoint(self.base_url, "v1/messages"),
            headers,
            payload,
            self.timeout_seconds,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            response_parser=_extract_anthropic_text,
            retry_structure_errors=False,
            record_stat=self._record_request_stat,
        )


def create_model_client(
    provider: str,
    *,
    model: str,
    api_key: str | None | _UnsetAPIKey = _API_KEY_UNSET,
    base_url: str | None = None,
    **options: Any,
) -> ModelClient:
    """根据 provider 创建 OpenAI-compatible 或 Anthropic 模型客户端。"""
    normalized = provider.strip().lower().replace("_", "-")
    if normalized in {"openai", "openai-compatible"}:
        kwargs = dict(options)
        if api_key is not _API_KEY_UNSET:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        return OpenAICompatibleModelClient(model=model, **kwargs)
    if normalized == "anthropic":
        kwargs = dict(options)
        if api_key is not _API_KEY_UNSET:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        return AnthropicModelClient(model=model, **kwargs)
    raise ValueError(f"Unsupported model provider: {provider}")


def create_embedding_client(
    provider: str,
    *,
    model: str,
    api_key: str | None | _UnsetAPIKey = _API_KEY_UNSET,
    base_url: str | None = None,
    **options: Any,
) -> EmbeddingClient:
    """创建 OpenAI-compatible 或本地 sentence-transformers 向量客户端。"""
    normalized = provider.strip().lower().replace("_", "-")
    if normalized in {"openai", "openai-compatible"}:
        kwargs = dict(options)
        if api_key is not _API_KEY_UNSET:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        return OpenAICompatibleEmbeddingClient(model=model, **kwargs)
    if normalized in {
        "local-sentence-transformers",
        "sentence-transformer",
        "sentence-transformers",
    }:
        if api_key is not _API_KEY_UNSET and api_key:
            raise ValueError("Local embedding provider does not accept api_key")
        if base_url is not None:
            raise ValueError("Local embedding provider does not accept base_url")
        return SentenceTransformerEmbeddingClient(model=model, **options)
    raise ValueError(f"Unsupported embedding provider: {provider}")


def _endpoint(base_url: str, path: str) -> str:
    """安全拼接模型服务根地址和相对 API 路径。"""
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _validate_retry_options(max_retries: int, retry_backoff_seconds: float) -> None:
    """校验重试次数和指数退避初始等待时间。"""
    if isinstance(max_retries, bool) or not isinstance(max_retries, int):
        raise ValueError("max_retries must be a non-negative integer")
    if max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    if (
        isinstance(retry_backoff_seconds, bool)
        or not isinstance(retry_backoff_seconds, (int, float))
        or not math.isfinite(retry_backoff_seconds)
        or retry_backoff_seconds < 0
    ):
        raise ValueError("retry_backoff_seconds must be non-negative")


def _request_json_with_retries(
    transport: JsonTransport,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
    *,
    max_retries: int,
    retry_backoff_seconds: float,
    response_parser: Callable[[Mapping[str, Any]], Any] | None = None,
    retry_structure_errors: bool = True,
    record_stat: Callable[[str], None] | None = None,
) -> Any:
    """Retry one shared request budget; optional OpenAI text validation is inside it.

    Explicit provider failures share this budget for every client. Malformed
    normal envelopes for embeddings/Anthropic retain their nonretry boundary.
    """
    def record(key: str) -> None:
        if record_stat is not None:
            record_stat(key)

    record("requests")
    error_counts: dict[str, int] = {}
    for attempt in range(max_retries + 1):
        record("attempts")
        if attempt:
            record("retries")
        try:
            response = transport(url, headers, payload, timeout_seconds)
            _raise_provider_error(response)
            if response_parser is not None:
                result = response_parser(response)
            else:
                result = response
        except ModelAPIError as exc:
            if response_parser is not None and retry_structure_errors:
                if exc.error_category == "response_structure":
                    exc.retryable = True
            if exc.status_code is not None and 400 <= exc.status_code < 500 and exc.status_code not in {408, 425, 429}:
                exc.retryable = False
            # Authentication/billing failures are permanent for every provider,
            # including embeddings and Anthropic, even with HTTP 429 or 5xx.
            if exc.provider_error_kind == "permanent":
                exc.retryable = False
            category = _ERROR_STAT_KEYS.get(exc.error_category, "other_api_errors")
            record(category)
            error_counts[category] = error_counts.get(category, 0) + 1
            exc.attempt_count = attempt + 1
            exc.error_counts = dict(error_counts)
            if not exc.retryable or attempt >= max_retries:
                record("failed")
                raise
            delay = retry_backoff_seconds * (2**attempt)
            if delay > 0:
                time.sleep(delay)
        except Exception:
            record("other_api_errors")
            record("failed")
            raise
        else:
            record("succeeded")
            return result
    raise AssertionError("unreachable retry loop")


def _post_openai(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    """Aggregate opt-in SSE into the same envelope as a normal completion."""
    return _post_json(
        url, headers, payload, timeout_seconds,
        response_reader=_read_openai_stream if payload.get("stream") is True else None,
    )


def _post_json(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
    *,
    response_reader: Callable[[Any], Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """使用标准库发送 JSON POST 请求，并统一处理网络与解析错误。"""
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            if response_reader is not None:
                return response_reader(response)
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        retryable = exc.code in {408, 425, 429} or 500 <= exc.code <= 599
        try:
            detail_bytes = exc.read()
        except (OSError, HTTPException):
            raise ModelAPIError(
                f"Model API returned HTTP {exc.code} with an unreadable error body",
                status_code=exc.code, retryable=retryable, error_category="http",
            ) from None
        try:
            error_response = json.loads(detail_bytes)
            error_kind = _provider_error_kind(error_response)
            provider_details = _safe_provider_details(error_response)
        except (json.JSONDecodeError, UnicodeDecodeError):
            error_kind = None
            provider_details = {}
        raise ModelAPIError(
            f"Model API returned HTTP {exc.code}" + _provider_detail_label(provider_details),
            status_code=exc.code,
            retryable=retryable,
            error_category="http",
            provider_error_kind=error_kind,
            provider_details=provider_details,
            response_body_sha256=hashlib.sha256(detail_bytes).hexdigest(),
            response_body_bytes=len(detail_bytes),
        ) from None
    except (URLError, TimeoutError, OSError, HTTPException) as exc:
        raise ModelAPIError(
            f"Model API request failed ({type(exc).__name__})",
            retryable=True,
            error_category="transport",
        ) from None
    except UnicodeDecodeError as exc:
        raise ModelAPIError(
            "Model API returned invalid UTF-8", error_category="response_structure",
        ) from exc
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ModelAPIError(
            "Model API returned invalid JSON", error_category="response_structure",
        ) from exc
    if not isinstance(decoded, Mapping):
        raise ModelAPIError(
            "Model API response must be a JSON object", error_category="response_structure",
        )
    return dict(decoded)


def _stream_structure_error() -> ModelAPIError:
    return ModelAPIError(
        "Invalid OpenAI-compatible streaming response structure",
        error_category="response_structure",
    )


def _iter_sse_events(response: Any):
    """Read SSE incrementally, preserving UTF-8 and CR/LF across read boundaries."""
    decoder = codecs.getincrementaldecoder("utf-8-sig")()
    reader = getattr(response, "read1", response.read)
    pending = ""
    data: list[str] = []
    event = "message"
    first_content = True
    while True:
        block = reader(8192)
        eof = not block
        pending += decoder.decode(block, final=eof)
        if first_content and pending.strip():
            first_content = False
            # Some compatible services return a JSON error with HTTP 200 even
            # when streaming was requested. Preserve its permanent/error kind.
            if pending.lstrip().startswith("{"):
                body = pending + decoder.decode(response.read(), final=True)
                try:
                    decoded = json.loads(body)
                except json.JSONDecodeError:
                    raise _stream_structure_error() from None
                _raise_provider_error(decoded)
                raise _stream_structure_error()
        while pending:
            positions = [pos for pos in (pending.find("\r"), pending.find("\n")) if pos >= 0]
            if not positions:
                break
            end = min(positions)
            if pending[end] == "\r" and end + 1 == len(pending) and not eof:
                break
            step = 2 if pending[end:end + 2] == "\r\n" else 1
            line, pending = pending[:end], pending[end + step:]
            if not line:
                if data:
                    yield event, "\n".join(data)
                data = []
                event = "message"
            elif not line.startswith(":"):
                name, _, value = line.partition(":")
                if value.startswith(" "):
                    value = value[1:]
                if name == "data":
                    data.append(value)
                elif name == "event":
                    event = value or "message"
        if eof:
            # SSE only dispatches complete events; EOF cannot finalize an answer.
            return


def _read_openai_stream(response: Any) -> Mapping[str, Any]:
    """Require a valid first-choice finish and DONE before returning any text."""
    parts: list[str] = []
    finish: str | None = None
    saw_first_choice = False
    for event, data in _iter_sse_events(response):
        if event == "error":
            try:
                error = json.loads(data)
            except json.JSONDecodeError:
                error = {"message": data}
            if not isinstance(error, Mapping):
                error = {"message": data}
            if _provider_error_fields(error) is None:
                error = {"error": error}
            _raise_provider_error(error)
            raise _stream_structure_error()
        if event != "message":
            raise _stream_structure_error()
        if data.strip() == "[DONE]":
            if not saw_first_choice or finish is None:
                raise _stream_structure_error()
            return {"choices": [{"message": {"content": "".join(parts)}, "finish_reason": finish}]}
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            raise _stream_structure_error() from None
        _raise_provider_error(chunk)
        if not isinstance(chunk, Mapping):
            raise _stream_structure_error()
        usage = chunk.get("usage")
        if usage is not None and not isinstance(usage, Mapping):
            raise _stream_structure_error()
        choices = chunk.get("choices", [])
        if not isinstance(choices, list):
            raise _stream_structure_error()
        if not choices:
            if not isinstance(usage, Mapping):
                raise _stream_structure_error()
            continue
        indexes: set[int] = set()
        for choice in choices:
            if not isinstance(choice, Mapping):
                raise _stream_structure_error()
            index = choice.get("index")
            delta = choice.get("delta")
            reason = choice.get("finish_reason")
            if (type(index) is not int or index < 0 or index in indexes
                    or not isinstance(delta, Mapping)
                    or (reason is not None and (not isinstance(reason, str) or reason not in {
                        "stop", "length", "content_filter", "tool_calls", "function_call",
                    }))):
                raise _stream_structure_error()
            indexes.add(index)
            if any(delta.get(key) is not None and not isinstance(delta[key], str)
                   for key in ("content", "reasoning_content")):
                raise _stream_structure_error()
            if index == 0:
                if finish is not None:
                    raise _stream_structure_error()
                saw_first_choice = True
                parts.append(delta.get("content") or "")
                finish = reason
    raise _stream_structure_error()


def _provider_error_fields(response: Any) -> Mapping[str, Any] | None:
    """Locate an explicit provider error without interpreting normal output."""
    if not isinstance(response, Mapping):
        return None
    error = response.get("error")
    if error is None and "choices" not in response and "message" in response:
        code = response.get("code", response.get("status", response.get("status_code")))
        if code not in (None, 0, 200, "0", "200", "success") or response.get("success") is False:
            error = response
    return None if error is None else error if isinstance(error, Mapping) else {"message": str(error)}


_SAFE_PROVIDER_IDENTIFIERS = frozenset("""
400 401 402 403 404 408 425 429 500 501 502 503 504 524
invalid_api_key authentication_error unauthorized permission_error permission_denied forbidden
insufficient_quota quota_exceeded billing_hard_limit_reached billing_not_active
insufficient_balance balance_not_enough payment_required invalid_request_error
context_length_exceeded prompt_tokens_too_large model_not_found not_found_error unsupported_model
rate_limit_exceeded rate_limit_error too_many_requests server_error internal_server_error
internal_error service_unavailable overloaded_error api_error timeout request_timeout capacity_exceeded
""".split())


def _safe_provider_details(response: Any) -> dict[str, Any]:
    """Keep known codes and a canonical message summary, never arbitrary echoes.

    A gateway may place a credential in any field, including code/type/message.
    Unknown identifiers and raw messages therefore retain only a fingerprint.
    """
    fields = _provider_error_fields(response)
    if fields is None:
        return {}
    result: dict[str, Any] = {}
    for key in ("code", "type", "status", "status_code"):
        value = fields.get(key)
        if value is None:
            continue
        text = str(value).strip()
        normalized = text.lower()
        if normalized in _SAFE_PROVIDER_IDENTIFIERS:
            result[key] = normalized
        else:
            result[key] = "unrecognized"
            result[key + "_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    message = fields.get("message")
    if message is not None:
        text = str(message)
        result["message_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        result["message_characters"] = len(text)
        lower = text.lower()
        summaries = (
            (("insufficient quota", "insufficient_quota", "quota exceeded", "额度不足"), "quota unavailable"),
            (("insufficient balance", "insufficient credit", "billing", "余额不足", "欠费"), "billing or balance rejection"),
            (("invalid api key", "invalid_api_key", "authentication failed", "unauthorized"), "authentication rejection"),
            (("permission denied", "forbidden"), "permission rejection"),
            (("model not found", "model does not exist"), "model unavailable"),
            (("context length", "context_length", "prompt tokens"), "context limit rejection"),
            (("rate limit", "too many requests"), "rate limited"),
            (("timed out", "timeout"), "provider timeout"),
            (("overload", "service unavailable"), "provider unavailable"),
        )
        result["message"] = next((summary for fragments, summary in summaries
                                  if any(fragment in lower for fragment in fragments)), "provider message redacted")
    return result


def _provider_detail_label(details: Mapping[str, Any]) -> str:
    identifiers = [f"{key}={details[key]}" for key in ("code", "type", "status", "status_code") if key in details]
    return " (" + ", ".join(identifiers) + ")" if identifiers else ""


def _provider_error_kind(response: Any) -> str | None:
    """Classify explicit error envelopes without retaining or replaying their text.

    Unknown explicit errors fail closed. Missing normal response fields alone
    are handled separately as malformed envelopes, not claimed provider errors.
    """
    fields = _provider_error_fields(response)
    if fields is None:
        return None
    identifiers = {str(fields.get(key, "")).strip().lower() for key in ("code", "type", "status", "status_code")}
    permanent = {
        "400", "401", "402", "403", "404", "invalid_api_key", "authentication_error",
        "unauthorized", "permission_error", "permission_denied", "forbidden",
        "insufficient_quota", "quota_exceeded", "billing_hard_limit_reached",
        "billing_not_active", "insufficient_balance", "balance_not_enough", "payment_required",
        "invalid_request_error", "context_length_exceeded", "prompt_tokens_too_large",
        "model_not_found", "not_found_error", "unsupported_model",
    }
    # Gateways sometimes use a generic type for billing/authentication failures.
    # Inspect only for classification; messages never enter retry prompts or new diagnostics.
    message = str(fields.get("message", "")).lower()
    permanent_messages = (
        "insufficient quota", "insufficient_quota", "quota exceeded", "insufficient balance",
        "insufficient credit", "billing", "invalid api key", "invalid_api_key",
        "authentication failed", "unauthorized", "permission denied", "model not found",
        "model does not exist", "余额不足", "额度不足", "欠费",
    )
    if identifiers & permanent or any(fragment in message for fragment in permanent_messages):
        return "permanent"
    transient = {
        "408", "425", "429", "500", "502", "503", "504", "rate_limit_exceeded",
        "rate_limit_error", "too_many_requests", "server_error", "internal_server_error",
        "internal_error", "service_unavailable", "overloaded_error", "api_error",
        "timeout", "request_timeout", "capacity_exceeded",
    }
    return "transient" if identifiers & transient else "unknown"


def _raise_provider_error(response: Any) -> None:
    kind = _provider_error_kind(response)
    if kind is not None:
        details = _safe_provider_details(response)
        raise ModelAPIError(
            f"Model API returned an explicit provider error (classification={kind})" + _provider_detail_label(details),
            retryable=kind == "transient", error_category="provider", provider_error_kind=kind,
            provider_details=details,
        )


def _extract_openai_text(response: Mapping[str, Any]) -> str:
    """从 OpenAI Chat Completions 响应中提取文本内容。"""
    try:
        choices = response["choices"]
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            raise TypeError
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise TypeError
        message = choice["message"]
        if not isinstance(message, Mapping):
            raise TypeError
        if "content" not in message and "reasoning_content" not in message and choice.get("finish_reason") != "length":
            raise KeyError("content")
        content = message.get("content")
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelAPIError(
            "Invalid OpenAI-compatible response structure", error_category="response_structure",
        ) from exc
    try:
        return _content_to_text(content, provider="OpenAI-compatible")
    except ModelAPIError as exc:
        reasoning_content = (
            message.get("reasoning_content") if isinstance(message, Mapping) else None
        )
        reasoning_characters = (
            len(reasoning_content) if isinstance(reasoning_content, str) else 0
        )
        finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
        if finish_reason is not None and (
            not isinstance(finish_reason, str)
            or finish_reason not in {"stop", "length", "tool_calls", "function_call", "content_filter"}
        ):
            finish_reason = "unrecognized"
        raise ModelAPIError(
            "OpenAI-compatible response contains no final text "
            f"(finish_reason={finish_reason!r}, "
            f"reasoning_content_characters={reasoning_characters}); "
            "increase max_tokens or disable reasoning for this model",
            error_category="response_content",
        ) from exc


def _extract_openai_embeddings(
    response: Mapping[str, Any],
    *,
    expected_count: int,
) -> list[list[float]]:
    """校验 OpenAI-compatible Embeddings 响应并按 index 排序。"""
    data = response.get("data")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        raise ModelAPIError("Invalid OpenAI-compatible embeddings response")
    indexed: list[tuple[int, list[float]]] = []
    try:
        for item in data:
            if not isinstance(item, Mapping):
                raise TypeError
            embedding = item["embedding"]
            if not isinstance(embedding, Sequence) or isinstance(
                embedding, (str, bytes)
            ):
                raise TypeError
            indexed.append(
                (int(item["index"]), [float(value) for value in embedding])
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelAPIError("Invalid embedding item structure") from exc
    indexed.sort(key=lambda item: item[0])
    if len(indexed) != expected_count or [item[0] for item in indexed] != list(
        range(expected_count)
    ):
        raise ModelAPIError("Embedding response count or indexes do not match input")
    return [embedding for _, embedding in indexed]


def _extract_anthropic_text(response: Mapping[str, Any]) -> str:
    """从 Anthropic Messages 响应中提取全部 text 内容块。"""
    content = response.get("content")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        raise ModelAPIError("Invalid Anthropic response structure")
    parts = [
        str(block["text"])
        for block in content
        if isinstance(block, Mapping) and block.get("type") == "text" and "text" in block
    ]
    text = "".join(parts).strip()
    if not text:
        raise ModelAPIError("Anthropic response contains no text block")
    return text


def _content_to_text(content: Any, *, provider: str) -> str:
    """兼容字符串或文本块数组形式的模型响应内容。"""
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts = [
            str(block["text"])
            for block in content
            if isinstance(block, Mapping) and "text" in block
        ]
        text = "".join(parts).strip()
        if text:
            return text
    raise ModelAPIError(f"{provider} response contains no text")
