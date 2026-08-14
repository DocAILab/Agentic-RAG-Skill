"""可断点恢复的逐样本检索评测循环。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from framework.models import ModelAPIError

from .loading import DatasetItem
from .persistence import (
    append_jsonl,
    iter_jsonl,
    run_signature,
    validate_resume,
    write_json,
)
from .scoring import SummaryAccumulator, score_example


def run_benchmark(
    items: Iterable[DatasetItem],
    retriever,
    *,
    output_dir: str | Path,
    run_metadata: Mapping,
    top_k: int = 10,
    checkpoint_every: int = 100,
    max_examples: int | None = None,
) -> dict:
    """评测流式样本，逐条落盘并跳过已完成的样本。"""
    _validate_options(top_k, checkpoint_every, max_examples)
    output = Path(output_dir)
    result_path = output / "results.jsonl"
    accumulator = SummaryAccumulator()
    completed = set()
    last_record = None
    for record in iter_jsonl(result_path):
        validate_resume(record, run_metadata)
        accumulator.add(record)
        completed.add(_record_key(record))
        last_record = record
    signature = run_signature(run_metadata)
    processed_now = 0

    for item in items:
        if max_examples is not None and item.source_index >= max_examples:
            break
        if (item.source_index, item.sample_id) in completed:
            continue
        record = _evaluate_item(item, retriever, top_k, signature)
        append_jsonl(result_path, record)
        accumulator.add(record)
        completed.add((item.source_index, item.sample_id))
        last_record = record
        processed_now += 1
        if processed_now % checkpoint_every == 0:
            _write_checkpoint(output, accumulator, item, run_metadata)

    summary = accumulator.to_dict(run_metadata)
    write_json(output / "summary.json", summary)
    write_json(
        output / "checkpoint.json",
        {
            "run": dict(run_metadata),
            "processed": accumulator.counts["processed"],
            "last": last_record,
        },
    )
    return summary


def _evaluate_item(item, retriever, top_k, signature):
    base = {
        "sample_id": item.sample_id,
        "source_index": item.source_index,
        "run_signature": signature,
    }
    if item.error or item.example is None:
        return {**base, "status": "invalid", "error": item.error or "invalid sample"}
    try:
        documents = retriever.retrieve(item.example, top_k=top_k)
    except ModelAPIError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return {**base, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    retrieved_ids = [document["id"] for document in documents]
    return {
        **base,
        "status": "ok",
        "label_type": item.example.label_type,
        "relevant_document_ids": list(item.example.relevant_document_ids),
        "retrieved": [
            {"id": document["id"], "score": document.get("score")}
            for document in documents
        ],
        "metrics": score_example(item.example, retrieved_ids),
    }


def _write_checkpoint(output, accumulator, item, run_metadata):
    write_json(
        output / "checkpoint.json",
        {
            "run": dict(run_metadata),
            "processed": accumulator.counts["processed"],
            "last_source_index": item.source_index,
            "last_sample_id": item.sample_id,
        },
    )


def _validate_options(top_k, checkpoint_every, max_examples):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be positive or None")


def _record_key(record):
    return record.get("source_index"), record["sample_id"]
