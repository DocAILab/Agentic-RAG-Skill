from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.retrieval.loading import DatasetItem
from experiments.retrieval.sampling import _sample_digest, read_manifest
from experiments.retrieval.schema import RetrievalDocument, RetrievalExample
from self_optimization.split_task_data import (
    SplitTaskDataError,
    build_task_suite,
    create_universe,
    _row_key,
)


def _example(identity: str, *, labelled: bool = True) -> RetrievalExample:
    documents = (RetrievalDocument("d0", "title", "text"),)
    return RetrievalExample(
        id=identity,
        query=f"question {identity}",
        documents=documents,
        relevant_document_ids=("d0",) if labelled else (),
        label_type="weak_answer_alias" if labelled else None,
        gold_answers=("Paris",) if labelled else (),
        metadata={"dataset": "triviaqa"},
    )


def _item(index: int, identity: str, *, labelled: bool = True) -> DatasetItem:
    return DatasetItem(index, identity, example=_example(identity, labelled=labelled))


def _suite_paths(root: Path, task_size: int) -> tuple[Path, Path]:
    tier = root / f"k{task_size}"
    return tier / "suite.manifest.json", tier


def test_universe_matches_retrieval_hash_order(tmp_path: Path) -> None:
    items = [_item(0, "c"), _item(1, "a"), _item(2, "b")]

    manifest = create_universe(
        items,
        tmp_path,
        dataset="triviaqa",
        split="validation",
        dataset_config=None,
        require_labels=True,
    )

    expected = sorted(
        ["a", "b", "c"],
        key=lambda sample_id: _sample_digest("triviaqa", sample_id),
    )
    assert manifest["requested_size"] == 3
    documents = [RetrievalDocument("d0", "title", "text").to_dict()]
    expected = sorted(
        [
            _row_key("triviaqa", sample_id, {"documents": documents})
            for sample_id in ["a", "b", "c"]
        ],
        key=lambda row_key: _sample_digest("triviaqa", row_key),
    )
    assert manifest["selected_keys"] == expected
    assert manifest["source_indices"] == [
        next(item.source_index for item in items if item.sample_id == sample_id)
        for sample_id in manifest["sample_ids"]
    ]
    loaded = read_manifest(tmp_path / "universe.manifest.json")
    assert loaded["digest"] == manifest["digest"]


def test_universe_excludes_unlabelled_and_invalid(tmp_path: Path) -> None:
    items = [
        _item(0, "q1"),
        _item(1, "q2", labelled=False),
        DatasetItem(2, "broken", error="adapter error"),
    ]

    manifest = create_universe(
        items,
        tmp_path,
        dataset="triviaqa",
        split="validation",
        dataset_config=None,
        require_labels=True,
    )

    assert manifest["sample_ids"] == ["q1"]
    assert manifest["invalid_count"] == 2


def test_tier_splits_are_disjoint_and_complete(tmp_path: Path) -> None:
    items = [_item(index, f"q{index:02d}") for index in range(7)]

    suites = build_task_suite(
        items,
        tmp_path,
        dataset="triviaqa",
        split="validation",
        dataset_config=None,
        task_sizes=(3,),
        require_labels=True,
        force=False,
    )

    assert len(suites) == 1
    suite = suites[0]
    assert suite["num_tasks"] == 3
    assert suite["task_size"] == 3
    assert suite["coverage"]["universe_samples"] == 7
    assert suite["coverage"]["assigned_rows"] == 7
    assert suite["coverage"]["duplicated_rows"] == 0
    assert suite["coverage"]["missing_rows"] == 0
    tier = tmp_path / "k3"
    assigned: set[str] = set()
    for index, entry in enumerate(suite["files"]):
        task_path = tier / entry["task"]
        manifest_path = tier / entry["manifest"]
        records = [
            json.loads(line)
            for line in task_path.read_text(encoding="utf-8").splitlines()
        ]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert [record["id"] for record in records] == manifest["sample_ids"]
        assert len(manifest["row_keys"]) == len(records)
        assert len(records) == 3 if index < 2 else len(records) == 1
        assert manifest["examples"] == len(records)
        assert manifest["task_index"] == index
        assert len(manifest["source_indices"]) == len(records)
        assert assigned.isdisjoint(manifest["sample_ids"])
        assigned.update(manifest["sample_ids"])
    assert assigned == {f"q{index:02d}" for index in range(7)}


def test_existing_tier_requires_force(tmp_path: Path) -> None:
    items = [_item(index, f"q{index:02d}") for index in range(5)]
    build_task_suite(
        items,
        tmp_path,
        dataset="triviaqa",
        split="validation",
        dataset_config=None,
        task_sizes=(2,),
        require_labels=True,
        force=False,
    )

    with pytest.raises(SplitTaskDataError, match="already exists"):
        build_task_suite(
            [],
            tmp_path,
            dataset="triviaqa",
            split="validation",
            dataset_config=None,
            task_sizes=(2,),
            require_labels=True,
            force=False,
        )
