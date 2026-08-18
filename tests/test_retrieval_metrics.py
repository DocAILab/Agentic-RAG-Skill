from __future__ import annotations

import pytest

from experiments.retrieval.schema import RetrievalDocument, RetrievalExample
from experiments.retrieval.scoring import (
    evidence_token_estimate,
    score_example,
    select_smallest_k,
)
from framework import all_support_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k_measures_fraction_of_distinct_supports() -> None:
    retrieved = ["noise", "gold-a", "gold-a", "gold-b"]
    relevant = {"gold-a", "gold-b"}

    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 10) == 1.0


def test_all_support_at_k_requires_every_relevant_document() -> None:
    retrieved = ["gold-a", "noise", "gold-b"]
    relevant = {"gold-a", "gold-b"}

    assert all_support_at_k(retrieved, relevant, 2) == 0.0
    assert all_support_at_k(retrieved, relevant, 3) == 1.0


def test_reciprocal_rank_uses_first_relevant_result() -> None:
    assert reciprocal_rank(["noise", "gold", "other"], {"gold"}) == 0.5
    assert reciprocal_rank(["noise"], {"gold"}) == 0.0


@pytest.mark.parametrize("metric", [recall_at_k, all_support_at_k])
def test_rank_metrics_validate_k(metric) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        metric(["gold"], {"gold"}, 0)


def test_retrieval_scoring_includes_context_sensitive_cutoffs() -> None:
    example = RetrievalExample(
        id="multi-hop",
        query="question",
        documents=(RetrievalDocument("a", "A", "text"),),
        relevant_document_ids=("a", "b"),
        label_type="supporting_facts",
    )

    metrics = score_example(example, ["a", "noise", "b"])

    assert metrics["recall@2"] == 0.5
    assert metrics["all_support@3"] == 1.0


def test_context_estimate_and_smallest_k_selection_are_explicit() -> None:
    documents = [
        {"id": "a", "text": "Two words."},
        {"id": "b", "text": "One"},
    ]
    metrics = {
        "all_support@1": 0.2,
        "all_support@2": 0.8,
        "all_support@3": 0.96,
        "all_support@5": 1.0,
        "all_support@10": 1.0,
        "recall@1": 0.4,
        "recall@2": 0.9,
        "recall@3": 0.95,
        "recall@5": 1.0,
        "recall@10": 1.0,
    }

    assert evidence_token_estimate(documents) == 4
    assert select_smallest_k(metrics) == 3
    assert select_smallest_k(
        {key: 0.0 for key in metrics},
    ) is None
