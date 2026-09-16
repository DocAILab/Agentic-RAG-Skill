"""从现有三个数据集 adapter 构建小型本地 RAG task JSONL。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from experiments.retrieval.adapters.two_wiki import normalize_question_type
from experiments.retrieval.loading import DatasetItem, iter_huggingface_items
from experiments.retrieval.sampling import read_manifest


def build_task_data(
    *,
    dataset: str,
    split: str,
    output_path: str | Path,
    max_examples: int,
    dataset_config: str | None = None,
    question_type: str | None = None,
    sample_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """流式读取并保存带答案、候选文档和证据标签的有效样本。"""
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    normalized_question_type = _requested_question_type(question_type, dataset)
    output = Path(output_path).resolve()
    records = []
    source_indices = []
    invalid_count = 0
    filtered_count = 0
    items = iter_huggingface_items(
        dataset,
        split,
        config=dataset_config,
    )
    sample_manifest = None
    if sample_manifest_path is not None:
        sample_manifest = read_manifest(sample_manifest_path)
        items = iter(
            _select_manifest_items(
                items,
                sample_manifest,
                dataset=dataset,
                split=split,
            )
        )
    for item in items:
        if item.error or item.example is None:
            invalid_count += 1
            continue
        example = item.example
        if not example.has_labels or not example.gold_answers:
            invalid_count += 1
            continue
        example_question_type = example.metadata.get("question_type")
        if (
            normalized_question_type is not None
            and example_question_type != normalized_question_type
        ):
            filtered_count += 1
            continue
        record = {
            "id": example.id,
            "dataset": example.metadata.get("dataset", dataset),
            "question": example.query,
            "answers": list(example.gold_answers),
            "documents": [document.to_dict() for document in example.documents],
            "relevant_document_ids": list(example.relevant_document_ids),
            "label_type": example.label_type,
        }
        if normalized_question_type is not None and example_question_type is not None:
            record["question_type"] = example_question_type
        records.append(record)
        source_indices.append(item.source_index)
        if len(records) >= max_examples:
            break
    if len(records) != max_examples:
        raise ValueError(
            f"requested {max_examples} valid examples, found {len(records)}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    output.write_text(content, encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "dataset_config": dataset_config,
        "split": split,
        "examples": len(records),
        "invalid_before_completion": invalid_count,
        "source_indices": source_indices,
        "sample_ids": [record["id"] for record in records],
        "output": output.name,
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
    if normalized_question_type is not None:
        manifest["filtered_before_completion"] = filtered_count
        manifest["question_type"] = normalized_question_type
    if sample_manifest is not None:
        manifest["sample_manifest"] = {
            "dataset": sample_manifest["dataset"],
            "split": sample_manifest["split"],
            "requested_size": sample_manifest["requested_size"],
            "digest": sample_manifest["digest"],
        }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """创建单数据集本地 task 构建命令的参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Build normalized local RAG task data from an existing adapter."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("hotpotqa", "2wiki", "2wikimultihopqa", "triviaqa"),
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--dataset-config")
    parser.add_argument("--examples", type=int, default=1)
    parser.add_argument(
        "--question-type",
        help="Filter 2Wiki records by official reasoning type.",
    )
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        help="Restrict selection to the stable IDs in a retrieval sample manifest.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """解析命令行、构建 task 数据并打印无语料正文的 manifest。"""
    args = build_parser().parse_args()
    manifest = build_task_data(
        dataset=args.dataset,
        split=args.split,
        output_path=args.output,
        max_examples=args.examples,
        dataset_config=args.dataset_config,
        question_type=args.question_type,
        sample_manifest_path=args.sample_manifest,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _requested_question_type(value: str | None, dataset: str) -> str | None:
    """Validate and normalize an optional 2Wiki task filter."""
    if value is None:
        return None
    if _canonical_dataset(dataset) != "2wikimultihopqa":
        raise ValueError("question_type is only supported for 2WikiMultihopQA")
    normalized = normalize_question_type(value, "<question-type-filter>")
    if normalized is None:
        raise ValueError("question_type must be non-empty")
    return normalized


def _select_manifest_items(
    items: Iterable[DatasetItem],
    manifest: Mapping[str, Any],
    *,
    dataset: str,
    split: str,
) -> list[DatasetItem]:
    """Return manifest samples in pinned ID order after validating provenance."""
    if _canonical_dataset(str(manifest.get("dataset", ""))) != _canonical_dataset(
        dataset
    ) or _canonical_split(str(manifest.get("split", ""))) != _canonical_split(split):
        raise ValueError("sample manifest dataset or split does not match the task")
    selected_value = manifest.get("selected_ids")
    if not isinstance(selected_value, list) or not selected_value:
        raise ValueError("sample manifest must contain selected_ids")
    selected_ids = [str(value).strip() for value in selected_value]
    if any(not value for value in selected_ids):
        raise ValueError("sample manifest contains an empty selected id")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("sample manifest contains duplicate selected ids")
    if manifest.get("requested_size") != len(selected_ids):
        raise ValueError("sample manifest requested_size does not match selected_ids")

    selected = set(selected_ids)
    found: dict[str, DatasetItem] = {}
    for item in items:
        if item.sample_id not in selected:
            continue
        if item.sample_id in found:
            raise ValueError(f"duplicate sample id in source: {item.sample_id}")
        found[item.sample_id] = item
        if len(found) == len(selected_ids):
            break
    missing = selected - found.keys()
    if missing:
        raise ValueError(f"sample manifest IDs were not found: {sorted(missing)[:3]}")
    return [found[identity] for identity in selected_ids]


def _canonical_dataset(value: str) -> str:
    """Normalize the aliases used by the loader and stable manifests."""
    normalized = value.strip().lower().replace("-", "")
    if normalized in {"2wiki", "2wikimultihopqa"}:
        return "2wikimultihopqa"
    return normalized


def _canonical_split(value: str) -> str:
    """Normalize the public validation alias used for the 2Wiki dev shard."""
    normalized = value.strip().lower()
    return "validation" if normalized == "dev" else normalized


if __name__ == "__main__":
    raise SystemExit(main())
