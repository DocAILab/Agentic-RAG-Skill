"""Run the preregistered BM25F grid and freeze its selected defaults."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from .loading import DatasetItem, iter_huggingface_items
from .persistence import append_jsonl, iter_jsonl
from .retrievers import build_retriever
from .sampling import iter_manifest_items, read_manifest
from .scoring import score_example
from .selection import (
    bm25f_grid,
    select_bm25f_defaults,
    write_frozen_selection,
)


def run_tuning(
    items_by_dataset: Mapping[str, Sequence[DatasetItem]],
    *,
    output_dir: str | Path,
    configurations: Sequence[Mapping[str, float]] | None = None,
    top_k: int = 10,
) -> dict:
    output = Path(output_dir)
    result_path = output / "bm25f_grid.jsonl"
    selected_path = output / "selected_defaults.json"
    grid = [dict(item) for item in (configurations or bm25f_grid())]
    requested = {_parameter_key(item) for item in grid}
    completed = {}
    for record in iter_jsonl(result_path):
        key = _parameter_key(record["parameters"])
        if key in completed:
            raise ValueError("duplicate BM25F tuning configuration")
        if key not in requested:
            raise ValueError("existing tuning results use a different grid")
        completed[key] = record
    for parameters in grid:
        key = _parameter_key(parameters)
        if key in completed:
            continue
        record = {
            "parameters": parameters,
            "metrics_by_dataset": {
                dataset: _evaluate(items, parameters, top_k)
                for dataset, items in sorted(items_by_dataset.items())
            },
        }
        append_jsonl(result_path, record)
        completed[key] = record
    if set(completed) != requested:
        raise ValueError("BM25F tuning did not complete the requested grid")
    selection = select_bm25f_defaults(completed.values())
    write_frozen_selection(selected_path, selection)
    return selection


def _evaluate(items, parameters, top_k):
    retriever = build_retriever("bm25", variant="B3", **parameters)
    totals = Counter()
    count = 0
    for item in items:
        if item.error or item.example is None or not item.example.has_labels:
            continue
        documents = retriever.retrieve(item.example, top_k=top_k)
        metrics = score_example(
            item.example,
            [document["id"] for document in documents],
        )
        totals.update(metrics or {})
        count += 1
    if count == 0:
        raise ValueError("tuning dataset contains no labelled valid examples")
    return {key: value / count for key, value in totals.items()}


def _parameter_key(parameters):
    return json.dumps(dict(parameters), sort_keys=True, separators=(",", ":"))


def _manifest_items(path: Path):
    manifest = read_manifest(path)
    dataset = str(manifest["dataset"])
    split = str(manifest["split"])
    items = iter_huggingface_items(dataset, split)
    return list(
        iter_manifest_items(items, manifest, dataset=dataset, split=split)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune BM25F retrieval parameters")
    parser.add_argument("--hotpot-manifest", required=True, type=Path)
    parser.add_argument("--two-wiki-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    selection = run_tuning(
        {
            "hotpotqa": _manifest_items(args.hotpot_manifest),
            "2wiki": _manifest_items(args.two_wiki_manifest),
        },
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
