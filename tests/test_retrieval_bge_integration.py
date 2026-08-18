from __future__ import annotations

import math
import os

import pytest

from experiments.retrieval.retrievers import build_retriever
from experiments.retrieval.schema import RetrievalDocument, RetrievalExample

RUN_REAL_BGE = os.getenv("RAGSKILL_RUN_BGE_INTEGRATION") == "1"


@pytest.mark.skipif(
    not RUN_REAL_BGE,
    reason="set RAGSKILL_RUN_BGE_INTEGRATION=1 to load the real BGE model",
)
def test_real_bge_retrieves_a_small_fixed_sample() -> None:
    retriever = build_retriever(
        "vector",
        model="BAAI/bge-large-en-v1.5",
        device=os.getenv("RAGSKILL_BGE_DEVICE", "cpu"),
        batch_size=2,
    )
    example = RetrievalExample(
        id="bge-smoke",
        query="Where do apple trees grow fruit?",
        documents=(
            RetrievalDocument(
                "apple",
                "Apple tree",
                "Apple trees grow fruit in orchards.",
            ),
            RetrievalDocument(
                "physics",
                "Quantum mechanics",
                "Quantum mechanics studies physical systems at small scales.",
            ),
        ),
    )

    documents = retriever.retrieve(example, top_k=2)

    assert documents[0]["id"] == "apple"
    assert all(math.isfinite(document["score"]) for document in documents)
