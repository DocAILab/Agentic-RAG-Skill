from __future__ import annotations

import json

from framework.vector_index import (
    build_or_load_vector_index,
    vector_index_cache_key,
)


class RecordingEmbedder:
    """记录语料编码调用并返回简单的确定性向量。"""

    def __init__(self) -> None:
        """初始化编码调用列表。"""
        self.calls = []

    def __call__(self, texts):
        """按文本关键词返回非零二维向量。"""
        self.calls.append(list(texts))
        return [
            [1.0, 0.0] if "apple" in text.lower() else [0.0, 1.0]
            for text in texts
        ]


def test_vector_index_key_invalidates_on_corpus_model_and_format_changes() -> None:
    """验证语料、模型配置或文本格式变化都会生成不同索引键。"""
    base = vector_index_cache_key(
        ["a", "b"],
        ["apple", "banana"],
        embedding_fingerprint="model-v1",
        text_format_version="format-v1",
    )

    assert base != vector_index_cache_key(
        ["a", "b"],
        ["changed apple", "banana"],
        embedding_fingerprint="model-v1",
        text_format_version="format-v1",
    )
    assert base != vector_index_cache_key(
        ["a", "b"],
        ["apple", "banana"],
        embedding_fingerprint="model-v2",
        text_format_version="format-v1",
    )
    assert base != vector_index_cache_key(
        ["a", "b"],
        ["apple", "banana"],
        embedding_fingerprint="model-v1",
        text_format_version="format-v2",
    )


def test_vector_index_persists_without_storing_corpus_text(tmp_path) -> None:
    """验证磁盘清单只保存索引身份和文档 ID，不复制语料正文。"""
    embedder = RecordingEmbedder()
    index = build_or_load_vector_index(
        ["a", "b"],
        ["private apple passage", "private banana passage"],
        embed=embedder,
        embedding_fingerprint="model-v1",
        text_format_version="format-v1",
        cache_root=tmp_path,
    )

    manifest_text = (index.cache_path / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["document_ids"] == ["a", "b"]
    assert "private apple passage" not in manifest_text
    assert index.search([1.0, 0.0], 1)[0][0] == 0


def test_vector_index_rebuilds_a_corrupted_disk_cache(tmp_path) -> None:
    """验证向量文件损坏时自动重建，而不是继续使用不可信缓存。"""
    first_embedder = RecordingEmbedder()
    first = build_or_load_vector_index(
        ["a", "b"],
        ["apple", "banana"],
        embed=first_embedder,
        embedding_fingerprint="model-v1",
        text_format_version="format-v1",
        cache_root=tmp_path,
    )
    (first.cache_path / "vectors.npy").write_bytes(b"broken")

    second_embedder = RecordingEmbedder()
    rebuilt = build_or_load_vector_index(
        ["a", "b"],
        ["apple", "banana"],
        embed=second_embedder,
        embedding_fingerprint="model-v1",
        text_format_version="format-v1",
        cache_root=tmp_path,
    )

    assert rebuilt.source == "built"
    assert second_embedder.calls == [["apple", "banana"]]
    assert rebuilt.search([0.0, 1.0], 1)[0][0] == 1
