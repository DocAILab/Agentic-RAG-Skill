"""Screen the preregistered BGE representation variants and freeze one."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from framework.models import SentenceTransformerEmbeddingClient

from .loading import DatasetItem, iter_huggingface_items
from .persistence import append_jsonl, iter_jsonl
from .retrievers import DEFAULT_BGE_BATCH_SIZE, build_retriever
from .sampling import iter_manifest_items, read_manifest
from .scoring import score_example
from .selection import STRONG_DATASETS, write_frozen_selection

VARIANTS = ("V0", "V1", "V2")


def run_screening(
    items_by_dataset: Mapping[str, Sequence[DatasetItem]],
    *,
    embedding_model,
    output_dir: str | Path,
    model_name: str,
    top_k: int = 10,
) -> dict:
    output = Path(output_dir)
    result_path = output / "bge_variants.jsonl"
    completed = {}
    for record in iter_jsonl(result_path):
        if record.get("model") != model_name or record.get("top_k") != top_k:
            raise ValueError("existing BGE screening results use different settings")
        variant = record["variant"]
        if variant in completed:
            raise ValueError("duplicate BGE screening variant")
        completed[variant] = record
    for variant in VARIANTS:
        if variant in completed:
            continue
        retriever = build_retriever(
            "vector",
            variant=variant,
            embedding_model=embedding_model,
        )
        record = {
            "variant": variant,
            "model": model_name,
            "top_k": top_k,
            "metrics_by_dataset": {
                dataset: _evaluate(items, retriever, top_k)
                for dataset, items in sorted(items_by_dataset.items())
            },
        }
        append_jsonl(result_path, record)
        completed[variant] = record
    selection = _select_improved(completed)
    write_frozen_selection(output / "selected_bge.json", selection)
    return selection


def _evaluate(items, retriever, top_k):
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
        raise ValueError("screening dataset contains no labelled valid examples")
    return {key: value / count for key, value in totals.items()}


def _select_improved(records):
    candidates = [records[variant] for variant in ("V1", "V2")]
    candidates.sort(
        key=lambda record: (
            -_macro(record, "all_support@5"),
            -_macro(record, "recall@5"),
            -_macro(record, "mrr"),
            record["variant"],
        )
    )
    selected = candidates[0]
    return {
        "selected_variant": selected["variant"],
        "baseline_variant": "V0",
        "model": selected["model"],
        "objective": {
            "macro_all_support@5": _macro(selected, "all_support@5"),
            "macro_recall@5": _macro(selected, "recall@5"),
            "macro_mrr": _macro(selected, "mrr"),
        },
    }


def _macro(record, metric):
    return sum(
        float(record["metrics_by_dataset"][dataset][metric])
        for dataset in STRONG_DATASETS
    ) / len(STRONG_DATASETS)


def _manifest_items(path: Path):
    manifest = read_manifest(path)
    dataset = str(manifest["dataset"])
    split = str(manifest["split"])
    items = iter_huggingface_items(dataset, split)
    return list(iter_manifest_items(items, manifest, dataset=dataset, split=split))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Screen BGE retrieval variants")
    parser.add_argument("--hotpot-manifest", required=True, type=Path)
    parser.add_argument("--two-wiki-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BGE_BATCH_SIZE)
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    embedding = SentenceTransformerEmbeddingClient(
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
        normalize_embeddings=True,
    )
    embedding.load()
    selection = run_screening(
        {
            "hotpotqa": _manifest_items(args.hotpot_manifest),
            "2wiki": _manifest_items(args.two_wiki_manifest),
        },
        embedding_model=embedding,
        output_dir=args.output_dir,
        model_name=args.model,
        top_k=args.top_k,
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
