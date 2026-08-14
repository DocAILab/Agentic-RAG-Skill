from __future__ import annotations

import math
from pathlib import Path

import pytest

from framework.spec import load_runtime_callable, load_spec

SKILLS = Path(__file__).parents[1] / "framework" / "skills" / "components"
BGE_INSTRUCTION = "Represent this sentence for searching relevant passages:"


def _component(name: str):
    return load_runtime_callable(load_spec(SKILLS / name))


class RecordingEmbeddingContext:
    def __init__(self, vectors_by_text):
        self.vectors_by_text = vectors_by_text
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [self.vectors_by_text[text] for text in texts]


def test_bm25_normalizes_unicode_and_boosts_titles() -> None:
    run = _component("component-bm25-retriever")
    result = run(
        {
            "query": "ＣＡＦÉ",
            "documents": [
                {"id": "body", "title": "Other", "text": "café guide"},
                {"id": "title", "title": "Café", "text": "guide"},
            ],
            "top_k": 2,
        },
        object(),
    )

    assert [item["id"] for item in result["documents"]] == ["title", "body"]
    assert all("score" in item for item in result["documents"])


def test_bm25_duplicate_query_terms_do_not_multiply_scores() -> None:
    run = _component("component-bm25-retriever")
    request = {
        "documents": [
            {"id": "a", "text": "apple apple"},
            {"id": "b", "text": "banana"},
        ],
        "top_k": 2,
    }

    once = run({**request, "query": "apple"}, object())
    repeated = run({**request, "query": "apple apple"}, object())

    assert repeated == once


def test_bm25f_normalizes_title_and_body_lengths_independently() -> None:
    run = _component("component-bm25-retriever")
    result = run(
        {
            "query": "orchid",
            "documents": [
                {"id": "a-short", "title": "orchid", "text": "brief"},
                {
                    "id": "b-long",
                    "title": "orchid",
                    "text": " ".join(["irrelevant"] * 100),
                },
            ],
            "top_k": 2,
        },
        object(),
    )

    first, second = result["documents"]
    assert [first["id"], second["id"]] == ["a-short", "b-long"]
    assert first["score"] == pytest.approx(second["score"])


def test_bm25_handles_empty_query_and_validates_parameters() -> None:
    run = _component("component-bm25-retriever")
    documents = [{"id": "a", "title": "A"}]

    assert run({"query": "", "documents": documents}, object()) == {"documents": []}
    with pytest.raises(ValueError, match="title_boost"):
        run(
            {"query": "a", "documents": documents, "title_boost": -1},
            object(),
        )
    with pytest.raises(ValueError, match="title_b"):
        run(
            {"query": "a", "documents": documents, "title_b": 1.1},
            object(),
        )


def test_vector_adds_instruction_and_reuses_document_embeddings() -> None:
    run = _component("component-vector-retriever")
    query_text = f"{BGE_INSTRUCTION} apple"
    passage_text = "Fresh\nbanana"
    context = RecordingEmbeddingContext(
        {query_text: [1.0, 0.0], passage_text: [0.0, 1.0]}
    )

    result = run(
        {
            "query": "apple",
            "documents": [
                {"id": "cached", "text": "unused", "embedding": [2.0, 0.0]},
                {"id": "fresh", "title": "Fresh", "text": "banana"},
            ],
            "top_k": 2,
        },
        context,
    )

    assert context.calls == [[query_text, passage_text]]
    assert [item["id"] for item in result["documents"]] == ["cached", "fresh"]
    assert result["documents"][0]["score"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("embedding", "message"),
    [
        ([0.0, 0.0], "zero vector"),
        ([math.inf, 0.0], "finite"),
        ([1.0], "dimensions"),
    ],
)
def test_vector_rejects_invalid_document_embeddings(embedding, message) -> None:
    run = _component("component-vector-retriever")
    query_text = f"{BGE_INSTRUCTION} apple"
    context = RecordingEmbeddingContext({query_text: [1.0, 0.0]})

    with pytest.raises(ValueError, match=message):
        run(
            {
                "query": "apple",
                "documents": [{"id": "bad", "embedding": embedding}],
            },
            context,
        )
