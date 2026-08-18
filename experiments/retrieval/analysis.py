"""Paired retrieval comparisons with deterministic bootstrap intervals."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping

DEFAULT_BOOTSTRAP_SEED = 20260815


def compare_paired_metric(
    baseline_records: Iterable[Mapping],
    candidate_records: Iterable[Mapping],
    metric: str,
    *,
    resamples: int = 10_000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    minimum_effect: float = 0.01,
) -> dict:
    if resamples < 1:
        raise ValueError("resamples must be positive")
    baseline = _metric_index(baseline_records, metric)
    candidate = _metric_index(candidate_records, metric)
    if set(baseline) != set(candidate):
        raise ValueError("paired runs must contain the same sample ids")
    if not baseline:
        raise ValueError("paired runs contain no labelled metric values")
    identities = sorted(baseline)
    baseline_values = [baseline[identity] for identity in identities]
    candidate_values = [candidate[identity] for identity in identities]
    differences = [
        candidate_value - baseline_value
        for baseline_value, candidate_value in zip(
            baseline_values, candidate_values, strict=True
        )
    ]
    interval = _bootstrap_interval(differences, resamples=resamples, seed=seed)
    difference = _mean(differences)
    return {
        "metric": metric,
        "count": len(identities),
        "baseline_mean": _mean(baseline_values),
        "candidate_mean": _mean(candidate_values),
        "difference": difference,
        "confidence_interval_95": interval,
        "resamples": resamples,
        "seed": seed,
        "minimum_effect": minimum_effect,
        "practically_superior": difference >= minimum_effect and interval[0] > 0,
    }


def _metric_index(records, metric):
    indexed = {}
    seen = set()
    for record in records:
        identity = record_identity(record)
        if identity in seen:
            raise ValueError(f"duplicate sample identity: {identity}")
        seen.add(identity)
        metrics = record.get("metrics")
        if record.get("status") != "ok" or metrics is None:
            continue
        if metric not in metrics:
            raise ValueError(f"metric {metric!r} is missing for sample {identity}")
        indexed[identity] = float(metrics[metric])
    return indexed


def record_identity(record: Mapping) -> tuple[str, str]:
    """Return the benchmark's stable row identity.

    Some datasets reuse a question ID for different evidence rows, so the
    source position is part of the identity whenever it is available.
    """
    source_index = record.get("source_index")
    source_key = "" if source_index is None else str(source_index)
    return source_key, str(record["sample_id"])


def _bootstrap_interval(values, *, resamples, seed):
    generator = random.Random(seed)
    count = len(values)
    estimates = [
        sum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    ]
    estimates.sort()
    return [
        estimates[int((resamples - 1) * 0.025)],
        estimates[int((resamples - 1) * 0.975)],
    ]


def _mean(values):
    return sum(values) / len(values)
