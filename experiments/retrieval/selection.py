"""Predeclared parameter grids and frozen retrieval-default selection rules."""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from .persistence import write_json

STRONG_DATASETS = ("hotpotqa", "2wiki")
CURRENT_BM25F_DEFAULTS = {
    "k1": 1.5,
    "b": 0.75,
    "title_b": 0.75,
    "title_boost": 1.5,
}


def bm25f_grid() -> list[dict[str, float]]:
    names = ("k1", "b", "title_b", "title_boost")
    values = (
        (1.2, 1.5, 2.0),
        (0.5, 0.75),
        (0.0, 0.5, 0.75),
        (1.0, 1.5, 2.0, 3.0),
    )
    return [dict(zip(names, combination, strict=True)) for combination in itertools.product(*values)]


def select_bm25f_defaults(records: Iterable[Mapping]) -> dict:
    candidates = [dict(record) for record in records]
    baseline = _find_record(candidates, CURRENT_BM25F_DEFAULTS)
    accepted = []
    rejected = []
    for record in candidates:
        if _recall_regressed(record, baseline):
            rejected.append(dict(record["parameters"]))
        else:
            accepted.append(record)
    if not accepted:
        raise ValueError("all BM25F configurations were rejected")
    best_primary = max(_macro(record, "all_support@5") for record in accepted)
    contenders = [
        record
        for record in accepted
        if _macro(record, "all_support@5") >= best_primary - 0.002
    ]
    contenders.sort(key=_selection_key)
    selected = contenders[0]
    return {
        "selected_parameters": dict(selected["parameters"]),
        "objective": {
            "macro_all_support@5": _macro(selected, "all_support@5"),
            "macro_recall@5": _macro(selected, "recall@5"),
            "macro_mrr": _macro(selected, "mrr"),
        },
        "rejected_for_recall": sorted(rejected, key=_canonical_parameters),
        "rules": {
            "primary_tolerance": 0.002,
            "maximum_recall@10_regression": 0.005,
            "datasets": list(STRONG_DATASETS),
        },
    }


def write_frozen_selection(path: str | Path, payload: Mapping) -> None:
    target = Path(path)
    if target.is_file():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != dict(payload):
            raise ValueError("refusing to overwrite a different frozen selection")
        return
    write_json(target, dict(payload))


def _find_record(records, parameters):
    for record in records:
        if dict(record["parameters"]) == parameters:
            return record
    raise ValueError("current BM25F defaults are missing from tuning records")


def _recall_regressed(record, baseline):
    return any(
        float(record["metrics_by_dataset"][dataset]["recall@10"])
        < float(baseline["metrics_by_dataset"][dataset]["recall@10"]) - 0.005
        for dataset in STRONG_DATASETS
    )


def _macro(record, metric):
    return sum(
        float(record["metrics_by_dataset"][dataset][metric])
        for dataset in STRONG_DATASETS
    ) / len(STRONG_DATASETS)


def _selection_key(record):
    return (
        -_macro(record, "recall@5"),
        -_macro(record, "mrr"),
        _distance_from_defaults(record["parameters"]),
        _canonical_parameters(record["parameters"]),
    )


def _distance_from_defaults(parameters):
    return sum(
        abs(float(parameters[name]) - default)
        for name, default in CURRENT_BM25F_DEFAULTS.items()
    )


def _canonical_parameters(parameters):
    return json.dumps(dict(parameters), sort_keys=True, separators=(",", ":"))
