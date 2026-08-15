from __future__ import annotations

import csv

from experiments.retrieval.reporting import (
    build_final_config,
    directional_failures,
    write_results_table,
)


def _record(identity, value, retrieved):
    return {
        "sample_id": identity,
        "status": "ok",
        "relevant_document_ids": ["gold"],
        "retrieved": [{"id": item, "score": 1.0} for item in retrieved],
        "metrics": {"all_support@5": value},
    }


def test_reporting_writes_flat_utf8_table_and_directional_failures(tmp_path) -> None:
    comparison = {
        "dataset": "hotpotqa",
        "baseline": "B0",
        "candidate": "B3",
        "comparisons": {
            "all_support@5": {
                "baseline_mean": 0.5,
                "candidate_mean": 0.6,
                "difference": 0.1,
                "confidence_interval_95": [0.02, 0.18],
                "practically_superior": True,
            }
        },
    }
    path = tmp_path / "results.csv"
    write_results_table(path, [comparison])

    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    failures = directional_failures(
        [_record("a", 0.0, ["noise"]), _record("b", 1.0, ["gold"])],
        [_record("a", 1.0, ["gold"]), _record("b", 0.0, ["noise"])],
        metric="all_support@5",
    )

    assert rows[0]["difference"] == "0.1"
    assert failures["candidate_wins"][0]["sample_id"] == "a"
    assert failures["baseline_wins"][0]["sample_id"] == "b"


def test_final_config_contains_frozen_choices_and_context_efficient_k() -> None:
    metrics = {
        **{f"all_support@{rank}": value for rank, value in zip((1, 2, 3, 5, 10), (0.2, 0.8, 0.96, 1.0, 1.0), strict=True)},
        **{f"recall@{rank}": value for rank, value in zip((1, 2, 3, 5, 10), (0.4, 0.9, 0.95, 1.0, 1.0), strict=True)},
    }

    config = build_final_config(
        {"selected_parameters": {"k1": 1.5}},
        {"selected_variant": "V1", "model": "fixture"},
        bm25_metrics=metrics,
        vector_metrics=metrics,
        sources={"bm25": "selected_defaults.json", "vector": "selected_bge.json"},
    )

    assert config["bm25f"]["parameters"] == {"k1": 1.5}
    assert config["bm25f"]["top_k"] == 3
    assert config["vector"]["variant"] == "V1"
