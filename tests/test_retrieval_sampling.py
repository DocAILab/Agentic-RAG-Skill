from __future__ import annotations

import pytest

from experiments.retrieval.loading import DatasetItem
from experiments.retrieval.sampling import (
    build_manifest,
    iter_manifest_items,
    read_manifest,
    write_manifest,
)
from experiments.retrieval.schema import RetrievalDocument, RetrievalExample


def _item(index: int, identity: str) -> DatasetItem:
    return DatasetItem(
        index,
        identity,
        example=RetrievalExample(
            id=identity,
            query="question",
            documents=(RetrievalDocument("doc", "title", "text"),),
        ),
    )


def test_manifest_selection_is_stable_across_input_order() -> None:
    items = [_item(0, "a"), _item(1, "b"), _item(2, "c")]

    forward = build_manifest(items, dataset="hotpotqa", split="train", size=2)
    reverse = build_manifest(
        reversed(items), dataset="hotpotqa", split="train", size=2
    )

    assert forward == reverse
    assert forward["requested_size"] == 2
    assert len(forward["selected_ids"]) == 2
    assert len(forward["digest"]) == 64


def test_manifest_round_trip_is_stable_and_rejects_wrong_dataset(tmp_path) -> None:
    items = [_item(0, "a"), _item(1, "b")]
    manifest = build_manifest(items, dataset="hotpotqa", split="train", size=1)
    path = tmp_path / "manifest.json"

    write_manifest(path, manifest)
    first_bytes = path.read_bytes()
    write_manifest(path, manifest)

    assert path.read_bytes() == first_bytes
    assert read_manifest(path) == manifest
    with pytest.raises(ValueError, match="dataset or split"):
        list(
            iter_manifest_items(
                items,
                manifest,
                dataset="2wikimultihopqa",
                split="train",
            )
        )


def test_manifest_cli_defaults_to_training_split(tmp_path) -> None:
    from experiments.retrieval.run_manifest import build_parser

    args = build_parser().parse_args(
        [
            "--dataset",
            "hotpotqa",
            "--size",
            "10000",
            "--output",
            str(tmp_path / "hotpot.json"),
        ]
    )

    assert args.split == "train"
    assert args.size == 10000
