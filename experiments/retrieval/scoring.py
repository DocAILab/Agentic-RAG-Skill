"""单样本检索指标计算与分组宏平均。"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from framework.evaluation import (
    all_support_at_k,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)

from .schema import RetrievalExample

RANKS = (1, 2, 3, 5, 10)
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


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


def evidence_token_estimate(documents) -> int:
    """Count Unicode words and punctuation in the evidence text as a stable proxy."""
    return sum(
        len(TOKEN_PATTERN.findall(str(document.get("text", ""))))
        for document in documents
    )


def select_smallest_k(metrics, *, retention: float = 0.95) -> int | None:
    if not 0 < retention <= 1:
        raise ValueError("retention must be in (0, 1]")
    target_all = float(metrics["all_support@10"])
    target_recall = float(metrics["recall@10"])
    if target_all == 0.0 and target_recall == 0.0:
        return None
    for rank in RANKS:
        if (
            float(metrics[f"all_support@{rank}"]) >= retention * target_all
            and float(metrics[f"recall@{rank}"]) >= retention * target_recall
        ):
            return rank
    return None


@dataclass(slots=True)
class SummaryAccumulator:
    counts: Counter = field(default_factory=Counter)
    totals: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    labelled_counts: Counter = field(default_factory=Counter)
    retrieval_totals: Counter = field(default_factory=Counter)

    def add(self, record: dict) -> None:
        self.counts["processed"] += 1
        status = record["status"]
        self.counts[status] += 1
        metrics = record.get("metrics")
        if status != "ok":
            return
        self.retrieval_totals.update(record.get("retrieval_stats", {}))
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
            "retrieval_stats": _averages(
                self.retrieval_totals,
                self.counts["ok"],
            ),
            "metrics_by_label_type": {
                label: _averages(self.totals[label], count)
                for label, count in sorted(self.labelled_counts.items())
            },
        }


def _averages(totals, count):
    if count == 0:
        return {"count": 0}
    return {
        "count": count,
        **{key: value / count for key, value in totals.items()},
    }
