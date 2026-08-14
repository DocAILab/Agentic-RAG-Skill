"""OpenAI-compatible 与 Anthropic 大模型接口适配器。"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

JsonMapping = Mapping[str, Any]
JsonTransport = Callable[
    [str, Mapping[str, str], Mapping[str, Any], float],
    Mapping[str, Any],
]


class ModelAPIError(RuntimeError):
    """表示模型 API 请求失败或响应结构不合法。"""


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
    timeout_seconds: float = 120.0
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    transport: JsonTransport | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """补充默认 HTTP transport，并校验基础配置。"""
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.transport is None:
            self.transport = _post_json

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """调用 OpenAI-compatible Chat Completions 并提取首个文本答案。"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        headers = {"content-type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        assert self.transport is not None
        response = self.transport(
            _endpoint(self.base_url, "chat/completions"),
            headers,
            payload,
            self.timeout_seconds,
        )
        return _extract_openai_text(response)


@dataclass(slots=True)
class OpenAICompatibleEmbeddingClient:
    """通过 `/v1/embeddings` 调用 OpenAI-compatible 向量服务。"""

    model: str
    api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        repr=False,
    )
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 120.0
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    transport: JsonTransport | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """补充默认 HTTP transport，并校验向量模型配置。"""
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
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
        response = self.transport(
            _endpoint(self.base_url, "embeddings"),
            headers,
            {"model": self.model, "input": normalized},
            self.timeout_seconds,
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
    default_max_tokens: int = 1024
    timeout_seconds: float = 120.0
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    transport: JsonTransport | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """补充默认 HTTP transport，并校验 Anthropic 必需配置。"""
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if not self.api_key:
            raise ValueError("Anthropic API requires api_key or ANTHROPIC_API_KEY")
        if self.default_max_tokens <= 0 or self.timeout_seconds <= 0:
            raise ValueError("token and timeout limits must be positive")
        if self.transport is None:
            self.transport = _post_json

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """调用 Anthropic Messages API 并拼接响应中的文本内容块。"""
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
        response = self.transport(
            _endpoint(self.base_url, "v1/messages"),
            headers,
            payload,
            self.timeout_seconds,
        )
        return _extract_anthropic_text(response)


def create_model_client(
    provider: str,
    *,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    **options: Any,
) -> ModelClient:
    """根据 provider 创建 OpenAI-compatible 或 Anthropic 模型客户端。"""
    normalized = provider.strip().lower().replace("_", "-")
    if normalized in {"openai", "openai-compatible"}:
        kwargs = dict(options)
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        return OpenAICompatibleModelClient(model=model, **kwargs)
    if normalized == "anthropic":
        kwargs = dict(options)
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        return AnthropicModelClient(model=model, **kwargs)
    raise ValueError(f"Unsupported model provider: {provider}")


def create_embedding_client(
    provider: str,
    *,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    **options: Any,
) -> EmbeddingClient:
    """创建 OpenAI-compatible 或本地 sentence-transformers 向量客户端。"""
    normalized = provider.strip().lower().replace("_", "-")
    if normalized in {"openai", "openai-compatible"}:
        kwargs = dict(options)
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        return OpenAICompatibleEmbeddingClient(model=model, **kwargs)
    if normalized in {
        "local-sentence-transformers",
        "sentence-transformer",
        "sentence-transformers",
    }:
        if api_key:
            raise ValueError("Local embedding provider does not accept api_key")
        if base_url is not None:
            raise ValueError("Local embedding provider does not accept base_url")
        return SentenceTransformerEmbeddingClient(model=model, **options)
    raise ValueError(f"Unsupported embedding provider: {provider}")


def _endpoint(base_url: str, path: str) -> str:
    """安全拼接模型服务根地址和相对 API 路径。"""
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _post_json(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
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
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ModelAPIError(f"Model API returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ModelAPIError(f"Model API request failed: {exc}") from exc
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ModelAPIError("Model API returned invalid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ModelAPIError("Model API response must be a JSON object")
    return dict(decoded)


def _extract_openai_text(response: Mapping[str, Any]) -> str:
    """从 OpenAI Chat Completions 响应中提取文本内容。"""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelAPIError("Invalid OpenAI-compatible response structure") from exc
    return _content_to_text(content, provider="OpenAI-compatible")


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
