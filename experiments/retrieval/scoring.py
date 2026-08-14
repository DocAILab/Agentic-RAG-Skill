"""单样本检索指标计算与分组宏平均。"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from framework.evaluation import (
    all_support_at_k,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)

from .schema import RetrievalExample

RANKS = (1, 5, 10)


def score_example(example: RetrievalExample, retrieved_ids) -> dict | None:
    if not example.has_labels:
        return None
    metrics = {"mrr": reciprocal_rank(retrieved_ids, example.relevant_document_ids)}
    for rank in RANKS:
        metrics[f"hit@{rank}"] = hit_at_k(
            retrieved_ids, example.relevant_document_ids, rank
        )
        metrics[f"recall@{rank}"] = recall_at_k(
            retrieved_ids, example.relevant_document_ids, rank
        )
        metrics[f"all_support@{rank}"] = all_support_at_k(
            retrieved_ids, example.relevant_document_ids, rank
        )
    return metrics


@dataclass(slots=True)
class SummaryAccumulator:
    counts: Counter = field(default_factory=Counter)
    totals: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    labelled_counts: Counter = field(default_factory=Counter)

    def add(self, record: dict) -> None:
        self.counts["processed"] += 1
        status = record["status"]
        self.counts[status] += 1
        metrics = record.get("metrics")
        if status != "ok":
            return
        self.counts["labelled" if metrics is not None else "unlabelled"] += 1
        if metrics is None:
            return
        label = record["label_type"]
        self.labelled_counts[label] += 1
        self.totals[label].update(metrics)

    def to_dict(self, run_metadata) -> dict:
        return {
            "run": dict(run_metadata),
            "counts": dict(self.counts),
            "metrics_by_label_type": {
                label: _averages(self.totals[label], count)
                for label, count in sorted(self.labelled_counts.items())
            },
        }


def _averages(totals, count):
    return {
        "count": count,
        **{key: value / count for key, value in totals.items()},
    }
