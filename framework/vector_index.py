"""构建、持久化并查询可复用的稠密向量索引。"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

VECTOR_INDEX_SCHEMA_VERSION = 1


class VectorIndexError(ValueError):
    """表示向量索引输入、缓存文件或查询向量不合法。"""


@dataclass(frozen=True, slots=True)
class DenseVectorIndex:
    """保存归一化语料矩阵及其可复现缓存身份。"""

    cache_key: str
    document_ids: tuple[str, ...]
    vectors: np.ndarray = field(repr=False)
    source: str = "built"
    cache_path: Path | None = None

    @property
    def dimension(self) -> int:
        """返回索引向量维度。"""
        return int(self.vectors.shape[1])

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """使用余弦相似度查询索引，并按文档 ID 确定性处理同分项。"""
        if top_k <= 0:
            return []
        query = _normalize_query_vector(query_vector, self.dimension)
        scores = self.vectors @ query
        ranked = sorted(
            range(len(self.document_ids)),
            key=lambda index: (-float(scores[index]), self.document_ids[index]),
        )
        return [
            (index, float(scores[index]))
            for index in ranked[: min(top_k, len(ranked))]
        ]


def embedding_model_fingerprint(model: Any) -> str:
    """从运行时 Embedding 客户端的稳定公开属性生成模型指纹。"""
    identity = {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model": getattr(model, "model", None),
        "base_url": getattr(model, "base_url", None),
        "normalize_embeddings": getattr(model, "normalize_embeddings", None),
    }
    return _sha256_json(identity)


def vector_index_cache_key(
    document_ids: Sequence[str],
    document_texts: Sequence[str],
    *,
    embedding_fingerprint: str,
    text_format_version: str,
) -> str:
    """用模型、文本格式以及有序语料内容计算不可碰撞的索引键。"""
    if len(document_ids) != len(document_texts):
        raise VectorIndexError("Document IDs and texts must have the same length")
    if not embedding_fingerprint:
        raise VectorIndexError("Embedding fingerprint cannot be empty")
    if not text_format_version:
        raise VectorIndexError("Text format version cannot be empty")

    digest = hashlib.sha256()
    _update_digest(digest, f"schema:{VECTOR_INDEX_SCHEMA_VERSION}")
    _update_digest(digest, embedding_fingerprint)
    _update_digest(digest, text_format_version)
    for document_id, text in zip(document_ids, document_texts, strict=True):
        _update_digest(digest, str(document_id))
        _update_digest(digest, str(text))
    return digest.hexdigest()


def build_or_load_vector_index(
    document_ids: Sequence[str],
    document_texts: Sequence[str],
    *,
    embed: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    embedding_fingerprint: str,
    text_format_version: str,
    cache_root: Path | None,
) -> DenseVectorIndex:
    """优先加载完整磁盘索引，未命中或损坏时编码语料并原子写盘。"""
    normalized_ids = tuple(str(document_id) for document_id in document_ids)
    normalized_texts = tuple(str(text) for text in document_texts)
    if not normalized_ids:
        raise VectorIndexError("Cannot build an index without documents")
    cache_key = vector_index_cache_key(
        normalized_ids,
        normalized_texts,
        embedding_fingerprint=embedding_fingerprint,
        text_format_version=text_format_version,
    )
    cache_path = cache_root / cache_key if cache_root is not None else None
    if cache_path is not None:
        try:
            return _load_vector_index(
                cache_path,
                cache_key=cache_key,
                document_ids=normalized_ids,
                embedding_fingerprint=embedding_fingerprint,
                text_format_version=text_format_version,
            )
        except (OSError, ValueError, TypeError):
            pass

    encoded = list(embed(normalized_texts))
    vectors = _normalize_document_vectors(encoded, len(normalized_ids))
    if cache_path is not None:
        _persist_vector_index(
            cache_path,
            cache_key=cache_key,
            document_ids=normalized_ids,
            vectors=vectors,
            embedding_fingerprint=embedding_fingerprint,
            text_format_version=text_format_version,
        )
    return DenseVectorIndex(
        cache_key=cache_key,
        document_ids=normalized_ids,
        vectors=vectors,
        source="built",
        cache_path=cache_path,
    )


def _load_vector_index(
    cache_path: Path,
    *,
    cache_key: str,
    document_ids: tuple[str, ...],
    embedding_fingerprint: str,
    text_format_version: str,
) -> DenseVectorIndex:
    """读取并完整校验一个已持久化的向量索引。"""
    manifest_path = cache_path / "manifest.json"
    vectors_path = cache_path / "vectors.npy"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": VECTOR_INDEX_SCHEMA_VERSION,
        "cache_key": cache_key,
        "embedding_fingerprint": embedding_fingerprint,
        "text_format_version": text_format_version,
        "document_ids": list(document_ids),
        "document_count": len(document_ids),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise VectorIndexError(f"Vector index manifest mismatch: {key}")
    vectors = np.load(vectors_path, allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[0] != len(document_ids):
        raise VectorIndexError("Persisted vector matrix shape is invalid")
    if int(manifest.get("dimension", -1)) != vectors.shape[1]:
        raise VectorIndexError("Persisted vector dimension does not match manifest")
    if not np.isfinite(vectors).all():
        raise VectorIndexError("Persisted vector matrix contains non-finite values")
    return DenseVectorIndex(
        cache_key=cache_key,
        document_ids=document_ids,
        vectors=np.asarray(vectors, dtype=np.float32),
        source="disk",
        cache_path=cache_path,
    )


def _persist_vector_index(
    cache_path: Path,
    *,
    cache_key: str,
    document_ids: tuple[str, ...],
    vectors: np.ndarray,
    embedding_fingerprint: str,
    text_format_version: str,
) -> None:
    """先写临时文件再替换正式文件，避免中断留下可见的半成品索引。"""
    cache_path.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary_vectors = cache_path / f"vectors.{token}.tmp"
    temporary_manifest = cache_path / f"manifest.{token}.tmp"
    vectors_path = cache_path / "vectors.npy"
    manifest_path = cache_path / "manifest.json"
    manifest = {
        "schema_version": VECTOR_INDEX_SCHEMA_VERSION,
        "cache_key": cache_key,
        "embedding_fingerprint": embedding_fingerprint,
        "text_format_version": text_format_version,
        "document_ids": list(document_ids),
        "document_count": len(document_ids),
        "dimension": int(vectors.shape[1]),
    }
    try:
        with temporary_vectors.open("wb") as handle:
            np.save(handle, vectors, allow_pickle=False)
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_vectors, vectors_path)
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_vectors.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)


def _normalize_document_vectors(
    vectors: Sequence[Sequence[float]],
    expected_count: int,
) -> np.ndarray:
    """校验语料向量矩阵并按行归一化为 float32。"""
    try:
        matrix = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise VectorIndexError("Embedding model returned invalid document vectors") from exc
    if matrix.ndim != 2 or matrix.shape[0] != expected_count or matrix.shape[1] == 0:
        raise VectorIndexError("Embedding model returned an invalid vector matrix shape")
    if not np.isfinite(matrix).all():
        raise VectorIndexError("Document vectors must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise VectorIndexError("Document vectors must not contain zero vectors")
    return np.asarray(matrix / norms, dtype=np.float32)


def _normalize_query_vector(vector: Sequence[float], dimension: int) -> np.ndarray:
    """校验并归一化单个查询向量。"""
    try:
        query = np.asarray(vector, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise VectorIndexError("Query embedding must be numeric") from exc
    if query.ndim != 1 or query.shape[0] != dimension:
        raise VectorIndexError("Query embedding dimension does not match index")
    if not np.isfinite(query).all():
        raise VectorIndexError("Query embedding must contain only finite values")
    norm = float(np.linalg.norm(query))
    if norm == 0.0:
        raise VectorIndexError("Query embedding must not be a zero vector")
    return np.asarray(query / norm, dtype=np.float32)


def _sha256_json(value: Any) -> str:
    """对稳定 JSON 编码计算 SHA-256。"""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _update_digest(digest: Any, value: str) -> None:
    """以长度前缀写入哈希，避免不同字段拼接产生边界碰撞。"""
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
