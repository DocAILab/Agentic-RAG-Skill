from __future__ import annotations

import pytest

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
