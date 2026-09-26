"""从现有三个数据集 adapter 构建小型本地 RAG task JSONL。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.retrieval.loading import iter_huggingface_items


def build_task_data(
    *,
    dataset: str,
    split: str,
    output_path: str | Path,
    max_examples: int,
    dataset_config: str | None = None,
) -> dict[str, Any]:
    """流式读取并保存带答案、候选文档和证据标签的有效样本。"""
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    output = Path(output_path).resolve()
    records = []
    source_indices = []
    invalid_count = 0
    items = iter_huggingface_items(
        dataset,
        split,
        config=dataset_config,
    )
    for item in items:
        if item.error or item.example is None:
            invalid_count += 1
            continue
        example = item.example
        if not example.has_labels or not example.gold_answers:
            invalid_count += 1
            continue
        records.append(
            {
                "id": example.id,
                "dataset": example.metadata.get("dataset", dataset),
                "question": example.query,
                "answers": list(example.gold_answers),
                "documents": [document.to_dict() for document in example.documents],
                "relevant_document_ids": list(example.relevant_document_ids),
                "label_type": example.label_type,
            }
        )
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
        choices=("hotpotqa", "2wiki", "2wikimultihopqa", "triviaqa", "financebench"),
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--dataset-config")
    parser.add_argument("--examples", type=int, default=1)
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
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
