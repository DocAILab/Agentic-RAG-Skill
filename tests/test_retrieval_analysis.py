from __future__ import annotations

import pytest

from experiments.retrieval.analysis import compare_paired_metric


def _record(identity, value):
    return {
        "sample_id": identity,
        "status": "ok",
        "metrics": {"all_support@5": value},
    }


def test_paired_bootstrap_is_deterministic_and_zero_for_identical_runs() -> None:
    records = [_record("a", 1.0), _record("b", 0.0)]

    first = compare_paired_metric(records, records, "all_support@5", resamples=100)
    second = compare_paired_metric(records, records, "all_support@5", resamples=100)

    assert first == second
    assert first["difference"] == 0.0
    assert first["confidence_interval_95"] == [0.0, 0.0]
    assert first["practically_superior"] is False


def test_paired_comparison_rejects_missing_or_duplicate_samples() -> None:
    with pytest.raises(ValueError, match="same sample ids"):
        compare_paired_metric(
            [_record("a", 0.0)],
            [_record("b", 1.0)],
            "all_support@5",
            resamples=10,
        )
    with pytest.raises(ValueError, match="duplicate"):
        compare_paired_metric(
            [_record("a", 0.0), _record("a", 1.0)],
            [_record("a", 1.0)],
            "all_support@5",
            resamples=10,
        )
