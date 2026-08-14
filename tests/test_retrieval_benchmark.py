from __future__ import annotations

import json

from experiments.retrieval.benchmark import run_benchmark
from experiments.retrieval.loading import DatasetItem, iter_huggingface_items
from experiments.retrieval.schema import RetrievalDocument, RetrievalExample


class FixedRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, example, *, top_k):
        self.calls.append(example.id)
        return [
            {**document.to_dict(), "score": float(len(example.documents) - index)}
            for index, document in enumerate(example.documents[:top_k])
        ]


def _example(identity="one", label_type="supporting_facts"):
    return RetrievalExample(
        id=identity,
        query="question",
        documents=(
            RetrievalDocument("gold-a", "A", "text"),
            RetrievalDocument("noise", "N", "text"),
            RetrievalDocument("gold-b", "B", "text"),
        ),
        relevant_document_ids=("gold-a", "gold-b"),
        label_type=label_type,
    )


def test_loader_streams_rows_maps_2wiki_validation_and_isolates_errors() -> None:
    captured = {}

    def loader(path, **kwargs):
        captured.update(path=path, **kwargs)
        return iter(
            [
                {
                    "_id": "good",
                    "question": "Q",
                    "context": [["A", ["text"]]],
                    "supporting_facts": [["A", 0]],
                },
                {"_id": "bad", "context": []},
            ]
        )

    items = list(
        iter_huggingface_items("2wiki", "validation", load_dataset_fn=loader)
    )

    assert captured["split"] == "dev"
    assert captured["streaming"] is True
    assert captured["path"] == "parquet"
    assert captured["data_files"]["dev"].endswith("/dev.parquet")
    assert items[0].example.id == "good"
    assert items[1].sample_id == "bad"
    assert "question" in items[1].error


def test_benchmark_writes_metrics_invalid_records_and_resumes(tmp_path) -> None:
    items = [
        DatasetItem(0, "one", example=_example()),
        DatasetItem(1, "bad", error="sample bad: empty candidates"),
    ]
    metadata = {"dataset": "fixture", "retriever": "fixed", "top_k": 10}
    retriever = FixedRetriever()

    summary = run_benchmark(
        items,
        retriever,
        output_dir=tmp_path,
        run_metadata=metadata,
        checkpoint_every=1,
    )
    resumed = run_benchmark(
        items,
        retriever,
        output_dir=tmp_path,
        run_metadata=metadata,
        checkpoint_every=1,
    )

    records = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert retriever.calls == ["one"]
    assert len(records) == 2
    assert records[0]["metrics"]["hit@1"] == 1.0
    assert records[0]["metrics"]["all_support@1"] == 0.0
    assert summary == resumed
    assert summary["counts"] == {
        "processed": 2,
        "ok": 1,
        "labelled": 1,
        "invalid": 1,
    }
    assert summary["metrics_by_label_type"]["supporting_facts"]["count"] == 1


def test_benchmark_keeps_weak_labels_in_separate_summary_group(tmp_path) -> None:
    items = [
        DatasetItem(0, "strong", example=_example("strong")),
        DatasetItem(1, "weak", example=_example("weak", "weak_answer_alias")),
    ]

    summary = run_benchmark(
        items,
        FixedRetriever(),
        output_dir=tmp_path,
        run_metadata={"run": "grouped"},
    )

    assert set(summary["metrics_by_label_type"]) == {
        "supporting_facts",
        "weak_answer_alias",
    }


def test_resume_limit_applies_to_total_source_prefix(tmp_path) -> None:
    items = [
        DatasetItem(0, "one", example=_example("one")),
        DatasetItem(1, "two", example=_example("two")),
    ]
    retriever = FixedRetriever()
    options = {
        "output_dir": tmp_path,
        "run_metadata": {"run": "limited"},
        "max_examples": 1,
    }

    run_benchmark(items, retriever, **options)
    run_benchmark(items, retriever, **options)

    assert retriever.calls == ["one"]
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 1
