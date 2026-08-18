"""Deterministic tables, failure sets, and final retrieval configuration."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

from .analysis import record_identity
from .scoring import select_smallest_k
from .selection import write_frozen_selection

TABLE_FIELDS = (
    "dataset",
    "baseline",
    "candidate",
    "metric",
    "baseline_mean",
    "candidate_mean",
    "difference",
    "ci95_lower",
    "ci95_upper",
    "practically_superior",
)


def write_results_table(path: str | Path, comparisons: Iterable[Mapping]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for comparison in comparisons:
        for metric, result in sorted(comparison["comparisons"].items()):
            interval = result["confidence_interval_95"]
            rows.append(
                {
                    "dataset": comparison["dataset"],
                    "baseline": comparison["baseline"],
                    "candidate": comparison["candidate"],
                    "metric": metric,
                    "baseline_mean": result["baseline_mean"],
                    "candidate_mean": result["candidate_mean"],
                    "difference": result["difference"],
                    "ci95_lower": interval[0],
                    "ci95_upper": interval[1],
                    "practically_superior": result["practically_superior"],
                }
            )
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def directional_failures(
    baseline_records: Iterable[Mapping],
    candidate_records: Iterable[Mapping],
    *,
    metric: str,
    limit: int = 50,
) -> dict[str, list[dict]]:
    if limit < 1:
        raise ValueError("failure limit must be positive")
    baseline = _record_index(baseline_records)
    candidate = _record_index(candidate_records)
    if set(baseline) != set(candidate):
        raise ValueError("failure analysis requires the same sample ids")
    candidate_wins = []
    baseline_wins = []
    for identity in sorted(baseline):
        left = baseline[identity]
        right = candidate[identity]
        difference = float(right["metrics"][metric]) - float(left["metrics"][metric])
        if difference > 0 and len(candidate_wins) < limit:
            candidate_wins.append(_failure_record(identity, left, right, metric))
        elif difference < 0 and len(baseline_wins) < limit:
            baseline_wins.append(_failure_record(identity, left, right, metric))
    return {"candidate_wins": candidate_wins, "baseline_wins": baseline_wins}


def build_final_config(
    bm25_selection: Mapping,
    vector_selection: Mapping,
    *,
    bm25_metrics: Mapping,
    vector_metrics: Mapping,
    sources: Mapping,
) -> dict:
    return {
        "bm25f": {
            "parameters": dict(bm25_selection["selected_parameters"]),
            "top_k": select_smallest_k(bm25_metrics),
        },
        "vector": {
            "variant": vector_selection["selected_variant"],
            "model": vector_selection["model"],
            "top_k": select_smallest_k(vector_metrics),
        },
        "sources": dict(sources),
    }


def write_final_config(path: str | Path, config: Mapping) -> None:
    write_frozen_selection(path, config)


def _record_index(records):
    indexed = {}
    for record in records:
        if record.get("status") != "ok" or record.get("metrics") is None:
            continue
        identity = record_identity(record)
        if identity in indexed:
            raise ValueError(f"duplicate sample identity: {identity}")
        indexed[identity] = record
    return indexed


def _failure_record(identity, baseline, candidate, metric):
    return {
        "sample_id": identity[1],
        "source_index": candidate.get("source_index"),
        "metric": metric,
        "baseline_value": baseline["metrics"][metric],
        "candidate_value": candidate["metrics"][metric],
        "relevant_document_ids": list(candidate.get("relevant_document_ids", ())),
        "baseline_retrieved_ids": [item["id"] for item in baseline["retrieved"]],
        "candidate_retrieved_ids": [item["id"] for item in candidate["retrieved"]],
    }
