from __future__ import annotations

import hashlib
import json

from experiments.retrieval.adapters import adapt_financebench
from experiments.retrieval.loading import DatasetItem
from self_optimization import build_task_data as task_data_module


def test_build_financebench_task_data(tmp_path, monkeypatch) -> None:
    row = {
        "financebench_id": "fb-1",
        "question": "What does Acme make?",
        "answer": "Widgets",
        "doc_name": "ACME_2023_10K",
        "evidence": [{"evidence_page_num": 12, "evidence_text": "Acme makes widgets."}],
    }
    example = adapt_financebench(row)

    def items(dataset, split, *, config):
        assert (dataset, split, config) == ("financebench", "train", None)
        yield DatasetItem(0, "broken", error="missing evidence")
        yield DatasetItem(1, example.id, example=example)

    monkeypatch.setattr(task_data_module, "iter_huggingface_items", items)
    output = tmp_path / "financebench-train.jsonl"
    manifest = task_data_module.build_task_data(
        dataset="financebench",
        split="train",
        output_path=output,
        max_examples=1,
    )

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["answers"] == ["Widgets"]
    assert record["relevant_document_ids"] == ["ACME_2023_10K#p12"]
    assert record["label_type"] == "evidence"
    assert manifest["source_indices"] == [1]
    assert manifest["invalid_before_completion"] == 1
    assert manifest["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
