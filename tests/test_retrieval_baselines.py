from __future__ import annotations

from experiments.retrieval.retrievers import build_retriever
from experiments.retrieval.schema import RetrievalDocument, RetrievalExample

BGE_INSTRUCTION = "Represent this sentence for searching relevant passages:"


class RecordingEmbeddingModel:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0, float(index)] for index, _ in enumerate(texts)]


def _field_example() -> RetrievalExample:
    return RetrievalExample(
        id="fields",
        query="orchid orchid",
        documents=(
            RetrievalDocument("title", "orchid", "noise"),
            RetrievalDocument("body", "noise", "orchid"),
        ),
    )


def test_b0_reproduces_text_only_original_and_b1_adds_title_signal() -> None:
    example = _field_example()

    b0 = build_retriever("bm25", variant="B0")
    b1 = build_retriever("bm25", variant="B1")

    assert [item["id"] for item in b0.retrieve(example, top_k=2)] == [
        "body",
        "title",
    ]
    assert [item["id"] for item in b1.retrieve(example, top_k=2)] == [
        "title",
        "body",
    ]


def test_vector_variants_change_only_declared_input_representations() -> None:
    example = RetrievalExample(
        id="vector-fields",
        query="orchid",
        documents=(RetrievalDocument("doc", "Flower", "orchid body"),),
    )
    model = RecordingEmbeddingModel()

    for variant in ("V0", "V1", "V2"):
        build_retriever(
            "vector",
            variant=variant,
            embedding_model=model,
        ).retrieve(example, top_k=1)

    assert model.calls == [
        ["orchid", "orchid body"],
        [f"{BGE_INSTRUCTION} orchid", "orchid body"],
        [f"{BGE_INSTRUCTION} orchid", "Flower\norchid body"],
    ]
