from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest
import yaml

from experiments.retrieval.loading import DatasetItem
from experiments.retrieval.sampling import read_manifest
from experiments.retrieval.schema import RetrievalDocument, RetrievalExample

task_data = importlib.import_module("self_optimization.build_task_data")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_DATA_ROOT = PROJECT_ROOT / "data" / "task_flow"
SAMPLE_MANIFEST = (
    PROJECT_ROOT / "data" / "2WikiMultihopQA" / "demo" / "sample_manifest.json"
)
TASK_CONFIG = PROJECT_ROOT / "self_optimization" / "task_flow.2wiki.example.yaml"
TASK_TYPES = (
    "compositional",
    "inference",
    "comparison",
    "bridge_comparison",
)


def _item(identity: str, question_type: str | None) -> DatasetItem:
    metadata = {"dataset": "2wikimultihopqa"}
    if question_type is not None:
        metadata["question_type"] = question_type
    return DatasetItem(
        source_index=int(identity.removeprefix("q")),
        sample_id=identity,
        example=RetrievalExample(
            id=identity,
            query=f"Question {identity}?",
            documents=(
                RetrievalDocument("gold", "Gold", "The answer is Alpha."),
                RetrievalDocument("noise", "Noise", "Unrelated text."),
            ),
            relevant_document_ids=("gold",),
            label_type="supporting_facts",
            gold_answers=("Alpha",),
            metadata=metadata,
        ),
    )


def _write_sample_manifest(path: Path, selected_ids: list[str]) -> dict:
    payload = {
        "dataset": "2wiki",
        "split": "validation",
        "requested_size": len(selected_ids),
        "selected_ids": selected_ids,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest = {**payload, "digest": digest}
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _install_items(monkeypatch, items: list[DatasetItem]) -> None:
    monkeypatch.setattr(
        task_data,
        "iter_huggingface_items",
        lambda dataset, split, *, config=None: iter(items),
    )


def test_builds_type_task_from_manifest_in_pinned_order(
    tmp_path, monkeypatch
) -> None:
    items = [
        _item("q1", "compositional"),
        _item("q2", "inference"),
        _item("q3", "compositional"),
        _item("q4", "comparison"),
    ]
    _install_items(monkeypatch, items)
    sample_manifest_path = tmp_path / "sample.json"
    sample_manifest = _write_sample_manifest(
        sample_manifest_path,
        ["q3", "q2", "q1", "q4"],
    )
    output = tmp_path / "compositional.jsonl"

    manifest = task_data.build_task_data(
        dataset="2wiki",
        split="validation",
        output_path=output,
        max_examples=2,
        question_type="compositional",
        sample_manifest_path=sample_manifest_path,
    )

    records = _records(output)
    assert [record["id"] for record in records] == ["q3", "q1"]
    assert {record["question_type"] for record in records} == {"compositional"}
    assert manifest["question_type"] == "compositional"
    assert manifest["filtered_before_completion"] == 1
    assert manifest["sample_manifest"]["digest"] == sample_manifest["digest"]
    assert manifest["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_normalizes_requested_bridge_type(tmp_path, monkeypatch) -> None:
    _install_items(monkeypatch, [_item("q1", "bridge_comparison")])
    output = tmp_path / "bridge.jsonl"

    manifest = task_data.build_task_data(
        dataset="2wikimultihopqa",
        split="dev",
        output_path=output,
        max_examples=1,
        question_type="bridge-comparison",
    )

    assert manifest["question_type"] == "bridge_comparison"
    assert _records(output)[0]["question_type"] == "bridge_comparison"


def test_repeated_builds_produce_identical_task_content(tmp_path, monkeypatch) -> None:
    items = [_item("q1", "comparison"), _item("q2", "comparison")]
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    _install_items(monkeypatch, items)
    first_manifest = task_data.build_task_data(
        dataset="2wiki",
        split="validation",
        output_path=first,
        max_examples=2,
        question_type="comparison",
    )
    _install_items(monkeypatch, items)
    second_manifest = task_data.build_task_data(
        dataset="2wiki",
        split="validation",
        output_path=second,
        max_examples=2,
        question_type="comparison",
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_manifest["sha256"] == second_manifest["sha256"]


def test_unfiltered_build_keeps_the_existing_record_and_manifest_shape(
    tmp_path, monkeypatch
) -> None:
    _install_items(monkeypatch, [_item("q1", "comparison")])
    output = tmp_path / "task.jsonl"

    manifest = task_data.build_task_data(
        dataset="2wiki",
        split="validation",
        output_path=output,
        max_examples=1,
    )

    assert "question_type" not in _records(output)[0]
    assert "question_type" not in manifest
    assert "filtered_before_completion" not in manifest


def test_type_filter_reports_insufficient_examples(tmp_path, monkeypatch) -> None:
    _install_items(
        monkeypatch,
        [_item("q1", "comparison"), _item("q2", "inference")],
    )

    with pytest.raises(ValueError, match="requested 2 valid examples, found 1"):
        task_data.build_task_data(
            dataset="2wiki",
            split="validation",
            output_path=tmp_path / "comparison.jsonl",
            max_examples=2,
            question_type="comparison",
        )


def test_manifest_provenance_must_match_task(tmp_path, monkeypatch) -> None:
    _install_items(monkeypatch, [_item("q1", "comparison")])
    path = tmp_path / "sample.json"
    manifest = _write_sample_manifest(path, ["q1"])
    manifest["split"] = "train"
    payload = {key: value for key, value in manifest.items() if key != "digest"}
    manifest["digest"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset or split"):
        task_data.build_task_data(
            dataset="2wiki",
            split="validation",
            output_path=tmp_path / "task.jsonl",
            max_examples=1,
            sample_manifest_path=path,
        )


def test_question_type_filter_is_limited_to_2wiki(tmp_path, monkeypatch) -> None:
    _install_items(monkeypatch, [_item("q1", None)])

    with pytest.raises(ValueError, match="only supported for 2Wiki"):
        task_data.build_task_data(
            dataset="hotpotqa",
            split="validation",
            output_path=tmp_path / "task.jsonl",
            max_examples=1,
            question_type="comparison",
        )


def test_parser_accepts_multitask_selection_arguments() -> None:
    parser = task_data.build_parser()

    args = parser.parse_args(
        [
            "--dataset",
            "2wiki",
            "--question-type",
            "inference",
            "--sample-manifest",
            "sample.json",
            "--output",
            "task.jsonl",
        ]
    )

    assert args.question_type == "inference"
    assert args.sample_manifest == Path("sample.json")


def test_tracked_2wiki_tasks_are_disjoint_and_internally_consistent() -> None:
    selected_ids = set(read_manifest(SAMPLE_MANIFEST)["selected_ids"])
    task_ids: set[str] = set()

    for question_type in TASK_TYPES:
        stem = question_type.replace("_", "-")
        data_path = TASK_DATA_ROOT / f"2wiki-{stem}-validation.jsonl"
        manifest_path = data_path.with_suffix(".manifest.json")
        records = _records(data_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ids = [record["id"] for record in records]

        assert len(records) == manifest["examples"] == 10
        assert len(ids) == len(set(ids))
        assert set(ids) <= selected_ids
        assert task_ids.isdisjoint(ids)
        assert {record["question_type"] for record in records} == {question_type}
        assert manifest["question_type"] == question_type
        assert manifest["sample_ids"] == ids
        assert manifest["sha256"] == hashlib.sha256(data_path.read_bytes()).hexdigest()
        for record in records:
            document_ids = [document["id"] for document in record["documents"]]
            assert len(document_ids) == len(set(document_ids))
            assert set(record["relevant_document_ids"]) <= set(document_ids)
            assert record["answers"]
        task_ids.update(ids)

    assert len(task_ids) == 40


def test_tracked_2wiki_task_flow_lists_all_reasoning_types_in_order() -> None:
    payload = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))
    tasks = payload["workflow"]["tasks"]
    expected_stems = [value.replace("_", "-") for value in TASK_TYPES]

    assert payload["schema_version"] == 1
    assert payload["self_optimization_config"] == "settings.yaml"
    assert [task["name"] for task in tasks] == [
        f"2wiki-{stem}-validation" for stem in expected_stems
    ]
    assert all(task["dataset"] == "2wikimultihopqa" for task in tasks)
    assert all(task["max_examples"] == 10 for task in tasks)
    assert all(task["request"] == {"top_k": 10} for task in tasks)
    for task, stem in zip(tasks, expected_stems, strict=True):
        expected = f"../data/task_flow/2wiki-{stem}-validation.jsonl"
        assert task["data_path"] == expected
        assert (TASK_CONFIG.parent / task["data_path"]).resolve().is_file()
