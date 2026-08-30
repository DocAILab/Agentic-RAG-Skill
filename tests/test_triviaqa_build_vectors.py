from __future__ import annotations

import json

import pytest

from experiments.triviaqa.scripts.build_vectors import (
    chunk_documents,
    collect_documents,
    document_text,
    load_samples,
)


class _FakeTokenizer:
    """按空白分词的最小 tokenizer，token 即单词。"""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._token_id(token) for token in text.split()]

    def convert_ids_to_tokens(self, ids) -> list[str]:
        by_id = {token_id: token for token, token_id in self._ids.items()}
        return [by_id[token_id] for token_id in ids]

    def convert_tokens_to_string(self, tokens) -> str:
        return " ".join(tokens)

    def _token_id(self, token: str) -> int:
        if token not in self._ids:
            self._ids[token] = len(self._ids)
        return self._ids[token]


def test_collect_documents_deduplicates_and_sorts_by_id() -> None:
    samples = [
        {
            "documents": [
                {"id": "wikipedia/Beta", "title": "Beta", "text": "first"},
                {"id": "wikipedia/Alpha", "title": "Alpha", "text": "second"},
                {"id": "wikipedia/Beta", "title": "Beta", "text": "duplicate"},
            ]
        }
    ]

    documents = collect_documents(samples)

    assert [document["id"] for document in documents] == [
        "wikipedia/Alpha",
        "wikipedia/Beta",
    ]
    assert documents[1]["text"] == "first"


def test_chunk_documents_splits_long_documents_with_fake_tokenizer() -> None:
    documents = [
        {
            "id": "wikipedia/Paris",
            "title": "Paris",
            "text": "Paris is the capital of France",
            "source": "wikipedia",
        }
    ]

    chunked = chunk_documents(documents, _FakeTokenizer(), chunk_tokens=2)

    assert [chunk["id"] for chunk in chunked] == [
        "wikipedia/Paris#0000",
        "wikipedia/Paris#0001",
        "wikipedia/Paris#0002",
    ]
    assert [chunk["text"] for chunk in chunked] == [
        "Paris is",
        "the capital",
        "of France",
    ]
    assert all(chunk["parent_id"] == "wikipedia/Paris" for chunk in chunked)


def test_chunk_documents_zero_returns_documents_unchanged() -> None:
    documents = [
        {"id": "web/1", "title": "T", "text": "long text", "source": "web"}
    ]

    chunked = chunk_documents(documents, _FakeTokenizer(), chunk_tokens=0)

    assert chunked == documents


def test_document_text_matches_component_title_text_format() -> None:
    document = {"id": "wikipedia/Paris", "title": "Paris", "text": "Paris is in France."}

    assert document_text(document) == "Paris\nParis is in France."
    assert document_text({**document, "title": ""}) == "Paris is in France."


def test_load_samples_rejects_missing_samples_list(tmp_path) -> None:
    path = tmp_path / "subset.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="samples"):
        load_samples([path])
